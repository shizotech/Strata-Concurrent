// src/core/handoff.cpp - S4.3.2: the tmpfs handoff arena (see include/strata/core/handoff.hpp).
//
// CPU-ONLY ON PURPOSE.  Nothing here makes a CUDA call, loads a model or starts an engine: the whole
// transport is a file, a header, an advisory lock and a mapping, so all of it is testable against a small
// arena in about a second.  That is deliberate - the bugs this file exists to prevent (a half-written 3 GB
// payload mounted by a reader, a slot stranded forever by a crashed decode instance, an `ftruncate` that
// succeeds on a tmpfs that cannot back it and then SIGBUSes) only show up at GiB scale in production,
// which is exactly why they need a small deterministic test next to them.
//
// The discipline is copied from stage 1's expert arena (`src/core/pinned.cu`): the flock covers a DECISION
// and never the write; publish is a release fence plus one ordered store through the fd rather than
// through a mapping; a claim is only believed when the pid AND its /proc start time still match; and the
// filesystem's free space is checked before the file is sized.
#include "strata/core/handoff.hpp"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <system_error>
#include <thread>

#include <fcntl.h>
#include <sys/file.h>   ///< `flock()` itself - <fcntl.h> only gives the `struct flock` record-lock type
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

namespace strata::core {

namespace {

/// §3.3/§3.4 the two magics.  `STRATA-HARENA-V1` is exactly 16 characters, so it does NOT fit in a
/// `char[16]` with a NUL: the field is compared as 16 raw bytes, never as a C string.
constexpr const char* kArenaMagic = "STRATA-HARENA-V1";
constexpr const char* kSlotMagic = "STRATA-HSLOT-V1";
/// Both magic fields are 16 raw bytes (§3.3/§3.4).
constexpr uint64_t kMagicBytes = 16;

/// §3.5 step 1: the same retry/give-up shape as `arena_lock` (`pinned.cu:111-128`).  The lock is held for
/// microseconds, so failing to get it within a few seconds means a foreign flock - report it rather than
/// act on an arena whose state cannot be read.
constexpr uint64_t kLockRetryMs = 20, kLockGiveUpMs = 10000;

uint64_t env_u64(const char* name, uint64_t fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == 0) return fallback;
    return std::strtoull(v, nullptr, 10);
}

/// Not a CSPRNG and does not need to be: it must simply not repeat within a machine's pid lifetime, so
/// time + pid + a stack address, mixed.  Same reasoning as `make_load_nonce()` (`pinned.cu:200-209`).
uint64_t make_nonce() {
    uint64_t x = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    x ^= ((uint64_t) (uint32_t) getpid() << 32);
    x ^= (uint64_t) (uintptr_t) &x;
    x *= 0x9e3779b97f4a7c15ull;
    x ^= x >> 29;
    return x != 0 ? x : 1;
}

void hash_u64(uint64_t& h, uint64_t v) {
    uint8_t b[8];
    for (int i = 0; i < 8; ++i) b[i] = (uint8_t) (v >> (i * 8));
    h = fnv1a64(b, 8, h);
}

std::string hex16(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return std::string(b);
}

std::string gib(uint64_t bytes) {
    char b[40];
    std::snprintf(b, sizeof b, "%.2f GiB", (double) bytes / (1024.0 * 1024.0 * 1024.0));
    return std::string(b);
}

/// §3.7, the same check as `shared_file_fits` (`pinned.cu:220-239`): `statvfs` on the PARENT directory,
/// `f_bavail x f_frsize`.  True when the check cannot be made (no statvfs, an existing file, an unopenable
/// path - the open below reports that properly); false ONLY when the filesystem says NO.
bool path_fits(const std::string& path, uint64_t file_bytes, std::string& why) {
    std::error_code ec;
    if (std::filesystem::exists(path, ec)) return true;   // already sized; its space is committed
    struct statvfs sv{};
    const std::filesystem::path p(path);
    const std::filesystem::path parent = p.has_parent_path() ? p.parent_path() : std::filesystem::path(".");
    if (statvfs(parent.c_str(), &sv) != 0) {
        why = "cannot read the free space of " + parent.string() + ": " + std::strerror(errno);
        return true;   // not our business to refuse on a check that failed
    }
    const uint64_t block = sv.f_frsize ? sv.f_frsize : sv.f_bsize;
    const uint64_t avail = (uint64_t) sv.f_bavail * block;
    if (avail >= file_bytes) return true;
    char buf[200];
    std::snprintf(buf, sizeof buf, "%s has %llu B free and it needs %llu B", parent.string().c_str(),
                  (unsigned long long) avail, (unsigned long long) file_bytes);
    why = buf;
    return false;
}

}  // namespace

// ================================ IDENTITY (§3.2, §3.10) ================================

std::string handoff_split_spec(const HandoffSplit& split) {
    // `main=<lo>..<hi>|<lo>..<hi>,...|n=<stages>|draft=<last|main>`.  Filename-safe: no space, no slash,
    // so §3.2 can use it directly as a path component.  It covers the main carve, every later carve in
    // engine order, the stage count, and where the draft lives - the tiling rule that makes two splits
    // with the same carves but a different draft placement produce different payloads.
    std::string s = "main=" + std::to_string(split.main_lo) + ".." + std::to_string(split.main_hi);
    s += "|";
    for (size_t i = 0; i < split.stages.size(); ++i) {
        if (i) s += ",";
        s += std::to_string(split.stages[i].layer_lo) + ".." + std::to_string(split.stages[i].layer_hi);
    }
    s += "|n=" + std::to_string((long long) (split.stages.size() + 1));
    s += "|draft=";
    s += (split.draft_on_last && !split.stages.empty()) ? "last" : "main";
    return s;
}

uint64_t handoff_geom_hash(const std::array<int64_t, 18>& geometry, const std::string& split_spec) {
    uint64_t h = 1469598103934665603ull;
    h = fnv1a64((const uint8_t*) "strata-handoff-geom-v1", 22, h);
    for (const int64_t v : geometry) hash_u64(h, (uint64_t) v);
    if (!split_spec.empty()) h = fnv1a64((const uint8_t*) split_spec.data(), (uint64_t) split_spec.size(), h);
    return h == 0 ? 1 : h;
}

std::string handoff_arena_path(const std::string& handoff_dir, uint64_t pack_hash,
                               const std::string& split_spec, const std::string& instance) {
    std::string out = handoff_dir;
    if (!out.empty() && out.back() != '/') out += '/';
    out += hex16(pack_hash) + "-" + split_spec;
    if (!instance.empty()) out += "/" + instance;
    out += "/arena";
    return out;
}

// ================================ SIZING (§3.8) ================================

bool handoff_size_arena(const std::vector<HandoffTierSpec>& tiers, const HandoffPricer& pricer,
                        HandoffSizing& out, std::string& err) {
    std::vector<uint64_t> priced(tiers.size(), 0);
    for (size_t i = 0; i < tiers.size(); ++i) {
        if (!pricer(tiers[i].max_tokens, priced[i], err)) {
            err = "handoff sizing, tier " + std::to_string(tiers[i].max_tokens) + " tokens: " + err;
            return false;
        }
    }
    return handoff_size_arena(tiers, priced, out, err);
}

bool handoff_size_arena(const std::vector<HandoffTierSpec>& tiers,
                        const std::vector<uint64_t>& payload_bytes, HandoffSizing& out, std::string& err) {
    out = HandoffSizing{};
    if (tiers.empty()) { err = "handoff sizing: no tiers"; return false; }
    if (tiers.size() > (size_t) kHandoffMaxTiers) {
        err = "handoff sizing: " + std::to_string(tiers.size()) + " tiers, the arena header holds " +
              std::to_string(kHandoffMaxTiers);
        return false;
    }
    if (payload_bytes.size() != tiers.size()) {
        err = "handoff sizing: " + std::to_string(payload_bytes.size()) + " priced sizes for " +
              std::to_string(tiers.size()) + " tiers";
        return false;
    }
    out.tiers = tiers;
    out.payload_bytes = payload_bytes;
    out.slot_bytes.resize(tiers.size());
    out.slot_off.resize(tiers.size());

    for (size_t i = 0; i < tiers.size(); ++i) {
        if (tiers[i].max_tokens == 0) {
            err = "handoff sizing: tier " + std::to_string(i) + " has 0 tokens";
            return false;
        }
        // Ascending tiers, so "the smallest group that fits" is a forward scan and a reader can trust
        // `tier_slot_off` instead of re-deriving the arithmetic (§3.3).
        if (i && tiers[i].max_tokens <= tiers[i - 1].max_tokens) {
            err = "handoff sizing: tiers must ascend by token count (" +
                  std::to_string(tiers[i - 1].max_tokens) + " then " + std::to_string(tiers[i].max_tokens) + ")";
            return false;
        }
        if (tiers[i].slot_count == 0) {
            err = "handoff sizing: tier " + std::to_string(tiers[i].max_tokens) + " has 0 slots";
            return false;
        }
        // §3.8: round_up_64(priced payload) + 64 MiB slack, which covers the record headers, the blob-pool
        // alignment padding and the segment-directory overhead ConversationBuffer::bytes() counts.
        if (payload_bytes[i] > UINT64_MAX - 63ull - kHandoffSlackBytes) {
            err = "handoff sizing: tier " + std::to_string(tiers[i].max_tokens) + " is too big to size";
            return false;
        }
        out.slot_bytes[i] = handoff_round_up_64(payload_bytes[i]) + kHandoffSlackBytes;
    }

    uint64_t off = kHandoffHeaderBytes;
    for (size_t i = 0; i < tiers.size(); ++i) {
        out.slot_off[i] = off;
        const uint64_t group = tiers[i].slot_count * (kHandoffHeaderBytes + out.slot_bytes[i]);
        if (group > UINT64_MAX - off) { err = "handoff sizing: the arena offset overflows"; return false; }
        off += group;
        out.slot_total += tiers[i].slot_count;
        out.max_tokens = std::max(out.max_tokens, tiers[i].max_tokens);
    }
    out.arena_bytes = off;

    // The figure the engine prints at start-up, spelled like its other numbers.
    std::string slots, tks;
    for (size_t i = 0; i < tiers.size(); ++i) {
        if (i) { slots += ","; tks += ","; }
        slots += std::to_string(tiers[i].slot_count);
        tks += std::to_string(tiers[i].max_tokens);
    }
    out.describe = "slots " + slots + " (tiers " + tks + "), max " + std::to_string(out.max_tokens) +
                   " tokens, arena " + gib(out.arena_bytes) + " (" + std::to_string(out.arena_bytes) + " B)";
    return true;
}

// ================================ /proc, CLOCK, LEASES (§3.6) ================================

/// Is `pid` a running process?  `/proc/<pid>` is the only answer that needs no signal permission
/// (`kill(pid,0)` says EPERM for somebody else's process, which is NOT "dead").  When /proc itself is not
/// mounted the answer is unknown, and unknown must mean ALIVE: guessing "dead" is what lets two processes
/// write the same bytes.  Copied from `pid_alive` (`pinned.cu:141-152`).
bool handoff_pid_alive(uint64_t pid) {
    if (pid == 0 || pid > (uint64_t) INT32_MAX) return false;
    struct stat st{};
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%llu", (unsigned long long) pid);
    if (stat(path, &st) == 0) return true;
    return stat("/proc", &st) != 0;
}

/// Field 22 of `/proc/<pid>/stat`, counted from the LAST ')' because the parenthesised comm field may
/// contain spaces.  0 = unknown, and unknown is treated as a MATCH.  Copied from `proc_start_time`
/// (`pinned.cu:161-185`).  This is what tells a dead owner from a RECYCLED pid.
uint64_t handoff_proc_start_time(uint64_t pid) {
    if (pid == 0 || pid > (uint64_t) INT32_MAX) return 0;
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%llu/stat", (unsigned long long) pid);
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return 0;
    char buf[4096];
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    if (n == 0) return 0;
    buf[n] = 0;
    char* close_paren = std::strrchr(buf, ')');
    if (close_paren == nullptr) return 0;
    const char* p = close_paren + 1;
    for (int field = 3; field < 22 && *p != 0; ++field) {
        while (*p == ' ') ++p;
        while (*p != 0 && *p != ' ') ++p;
    }
    while (*p == ' ') ++p;
    if (*p == 0) return 0;
    return std::strtoull(p, nullptr, 10);
}

bool handoff_claim_alive(uint64_t pid, uint64_t start_time) {
    if (!handoff_pid_alive(pid)) return false;
    if (start_time == 0) return true;                 // the file does not say; do not guess "dead"
    const uint64_t now = handoff_proc_start_time(pid);
    if (now == 0) return true;                        // /proc unreadable; same conservative answer
    return now == start_time;
}

uint64_t handoff_now_ms() {
    struct timespec ts{};
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) return 0;
    return (uint64_t) ts.tv_sec * 1000ull + (uint64_t) (ts.tv_nsec / 1000000);
}

// ================================ small on-disk helpers ================================

HandoffArena::HandoffArena() : lease_ms_(env_u64("STRATA_HANDOFF_LEASE_MS", kHandoffDefaultLeaseMs)) {}

bool HandoffArena::write_slot_fields_(uint64_t base, uint64_t off, const void* p, uint64_t n) const {
    const ssize_t got = ::pwrite(fd_, p, n, (off_t) (base + off));
    return got == (ssize_t) n;
}

bool HandoffArena::slot_base_(int index, uint64_t& base, uint32_t& tier, uint64_t& capacity) const {
    base = 0;
    tier = 0;
    capacity = 0;
    if (index < 0 || (uint64_t) index >= hdr_.slot_total) return false;
    uint64_t seen = 0;
    for (uint32_t t = 0; t < hdr_.tier_count && t < (uint32_t) kHandoffMaxTiers; ++t) {
        const uint64_t n = hdr_.tier_slot_count[t];
        if ((uint64_t) index < seen + n) {
            base = hdr_.tier_slot_off[t] + ((uint64_t) index - seen) *
                   (kHandoffHeaderBytes + hdr_.tier_slot_bytes[t]);
            tier = t;
            capacity = hdr_.tier_slot_bytes[t];
            // A header that claims slots past the end of the file is a corrupt arena, not a slot.
            return base + kHandoffHeaderBytes + capacity <= hdr_.arena_bytes;
        }
        seen += n;
    }
    return false;
}

bool HandoffArena::read_slot_header_(int index, HandoffSlotHeader& h, uint64_t& base,
                                     uint64_t& capacity) const {
    uint32_t tier = 0;
    if (!slot_base_(index, base, tier, capacity)) return false;
    const ssize_t got = ::pread(fd_, &h, sizeof h, (off_t) base);
    return got == (ssize_t) sizeof h;
}

bool HandoffArena::lock_(bool exclusive, std::string& err) const {
    const int op = (exclusive ? LOCK_EX : LOCK_SH) | LOCK_NB;
    for (uint64_t waited = 0;; waited += kLockRetryMs) {
        // `::flock`, not `flock`: <fcntl.h> also declares `struct flock` (POSIX record locks), and in C++
        // that tag is a type name, so an unqualified call can parse as a declaration instead of a call.
        if (::flock(fd_, op) == 0) return true;
        if (errno != EWOULDBLOCK && errno != EAGAIN) {
            err = std::string("cannot lock the handoff arena: ") + std::strerror(errno);
            return false;
        }
        if (waited >= kLockGiveUpMs) {
            err = "the handoff arena is locked by another process for more than " +
                  std::to_string(kLockGiveUpMs / 1000) +
                  " s; refusing to act on an arena whose state cannot be read";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kLockRetryMs));
    }
}

void HandoffArena::unlock_() const { (void) ::flock(fd_, LOCK_UN); }

bool HandoffArena::lease_expired_(uint64_t then_ms) const {
    if (lease_ms_ == 0) return true;
    if (then_ms == 0) return true;                    // no stamp: nothing to age, treat it as stale
    const uint64_t now = handoff_now_ms();
    return now >= then_ms && now - then_ms >= lease_ms_;
}

bool HandoffArena::client_dead_(uint64_t client_id) const {
    if (client_id == 0) return false;
    for (const auto& kv : client_pids_) {
        if (kv.first == client_id) return !handoff_pid_alive(kv.second);
    }
    return false;   // no registration: the time bound is what reclaims it (deviation 2)
}

// ================================ header ================================

bool HandoffArena::read_header_(std::string& err) {
    HandoffArenaHeader h{};
    const ssize_t got = ::pread(fd_, &h, sizeof h, 0);
    if (got != (ssize_t) sizeof h) {
        err = "handoff arena " + path_ + ": cannot read its header";
        return false;
    }
    // magic/version/header_bytes FIRST.  Everything else is meaningless until these match - reading tier
    // offsets out of a file that is not an arena at all is how a wrong file becomes a wrong write.
    if (std::memcmp(h.magic, kArenaMagic, kMagicBytes) != 0) {
        err = "handoff arena " + path_ + " is not a Strata handoff arena (magic)";
        return false;
    }
    if (h.version != kHandoffVersion) {
        err = "handoff arena " + path_ + " is protocol version " + std::to_string(h.version) +
              ", this engine speaks " + std::to_string(kHandoffVersion);
        return false;
    }
    if (h.header_bytes != kHandoffHeaderBytes) {
        err = "handoff arena " + path_ + " has a " + std::to_string(h.header_bytes) +
              " B header_bytes, expected " + std::to_string(kHandoffHeaderBytes);
        return false;
    }
    if (h.tier_count == 0 || h.tier_count > (uint32_t) kHandoffMaxTiers) {
        err = "handoff arena " + path_ + " declares " + std::to_string(h.tier_count) + " tiers";
        return false;
    }
    hdr_ = h;
    return true;
}

void HandoffArena::close_() {
    if (fd_ >= 0) { ::close(fd_); fd_ = -1; }
    writer_ = false;
    path_.clear();
    hdr_ = HandoffArenaHeader{};
    client_pids_.clear();
}

HandoffArena::~HandoffArena() { close_(); }

HandoffArena::HandoffArena(HandoffArena&& o) noexcept
    : fd_(o.fd_), writer_(o.writer_), path_(std::move(o.path_)), hdr_(o.hdr_), lease_ms_(o.lease_ms_),
      client_pids_(std::move(o.client_pids_)) {
    o.fd_ = -1;
    o.writer_ = false;
}

HandoffArena& HandoffArena::operator=(HandoffArena&& o) noexcept {
    if (this != &o) {
        close_();
        fd_ = o.fd_;
        writer_ = o.writer_;
        path_ = std::move(o.path_);
        hdr_ = o.hdr_;
        lease_ms_ = o.lease_ms_;
        client_pids_ = std::move(o.client_pids_);
        o.fd_ = -1;
        o.writer_ = false;
    }
    return *this;
}

uint64_t HandoffArena::tier_max_tokens(uint32_t tier) const {
    return tier < (uint32_t) kHandoffMaxTiers ? hdr_.tier_max_tokens[tier] : 0;
}
uint64_t HandoffArena::tier_slot_bytes(uint32_t tier) const {
    return tier < (uint32_t) kHandoffMaxTiers ? hdr_.tier_slot_bytes[tier] : 0;
}
uint64_t HandoffArena::tier_slot_count(uint32_t tier) const {
    return tier < (uint32_t) kHandoffMaxTiers ? hdr_.tier_slot_count[tier] : 0;
}
uint64_t HandoffArena::tier_slot_off(uint32_t tier) const {
    return tier < (uint32_t) kHandoffMaxTiers ? hdr_.tier_slot_off[tier] : 0;
}

std::string HandoffArena::describe() const {
    std::string slots, tks;
    for (uint32_t t = 0; t < hdr_.tier_count && t < (uint32_t) kHandoffMaxTiers; ++t) {
        if (t) { slots += ","; tks += ","; }
        slots += std::to_string(hdr_.tier_slot_count[t]);
        tks += std::to_string(hdr_.tier_max_tokens[t]);
    }
    return "slots " + slots + " (tiers " + tks + "), max " + std::to_string(max_tokens()) +
           " tokens, arena " + gib(hdr_.arena_bytes) + ", pack hash " + hex16(hdr_.pack_hash) +
           ", geometry hash " + hex16(hdr_.geom_hash);
}

// ================================ open_writer (§3.2, §3.7) ================================

bool HandoffArena::open_writer(const std::string& path, const HandoffSizing& sizing, uint64_t pack_hash,
                               uint64_t geom_hash, std::string& err, bool allow_existing_writer) {
    std::lock_guard<std::mutex> guard(mu_);
    close_();
    err.clear();
    if (pack_hash == 0) { err = "a handoff arena requires a nonzero pack hash"; return false; }
    if (geom_hash == 0) { err = "a handoff arena requires a nonzero geometry hash"; return false; }
    if (sizing.arena_bytes == 0 || sizing.tiers.empty() ||
        sizing.tiers.size() > (size_t) kHandoffMaxTiers) {
        err = "handoff arena: the sizing is empty or has too many tiers";
        return false;
    }
    // The sizing must be self-consistent before it is trusted to size a file.
    uint64_t walk = kHandoffHeaderBytes, total = 0;
    for (size_t i = 0; i < sizing.tiers.size(); ++i) {
        if (sizing.slot_off[i] != walk) {
            err = "handoff arena: tier " + std::to_string(i) + " starts at " +
                  std::to_string(sizing.slot_off[i]) + ", the layout says " + std::to_string(walk);
            return false;
        }
        walk += sizing.tiers[i].slot_count * (kHandoffHeaderBytes + sizing.slot_bytes[i]);
        total += sizing.tiers[i].slot_count;
    }
    if (walk != sizing.arena_bytes || total != sizing.slot_total) {
        err = "handoff arena: the sizing does not add up (" + std::to_string(sizing.arena_bytes) +
              " B claimed, " + std::to_string(walk) + " B laid out)";
        return false;
    }

    // The prefill instance CREATES the directory, the arena and the socket (§3.2).
    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path() && !p.parent_path().empty()) {
        std::filesystem::create_directories(p.parent_path(), ec);
        if (ec) {
            err = "cannot create the handoff directory " + p.parent_path().string() + ": " + ec.message();
            return false;
        }
    }

    // ENOUGH ROOM BEFORE ANYTHING IS CLAIMED.  `ftruncate` on tmpfs does NOT fail when the space is not
    // there - it happily makes a sparse file, and the writer then dies of SIGBUS part-way through a 3 GB
    // payload with no message at all.  Only a FRESH file needs the check: an existing file of the right
    // size already has its space committed, and re-checking it would refuse a working arena because
    // something else filled the filesystem afterwards (`pinned.cu:328-330`).
    //
    // Unlike the expert arena there is NO FALLBACK here: a handoff that cannot be created means the split
    // cannot run, so the refusal is fatal and names both numbers (§3.7).
    struct stat before{};
    const bool maybe_fresh = (stat(path.c_str(), &before) != 0) || before.st_size == 0;
    if (maybe_fresh && !path_fits(path, sizing.arena_bytes, err)) {
        err = "the handoff arena " + path + " needs " + std::to_string(sizing.arena_bytes) + " B (" +
              gib(sizing.arena_bytes) + ") but " + err +
              "; raise --prefill-slots-per-tier, lower --prefill-slot-tiers, or give the filesystem more room";
        return false;
    }

    // O_CLOEXEC matters, not just for tidiness: the flock lives on the OPEN FILE DESCRIPTION, so an exec'd
    // helper that inherited this fd would keep a claim alive forever (`pinned.cu:336-340`).
    const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
    if (fd < 0) {
        err = "cannot open the handoff arena " + path + ": " + std::strerror(errno);
        return false;
    }
    fd_ = fd;
    path_ = path;
    writer_ = true;

    // EVERY decision about this file happens under its exclusive advisory lock.
    if (!lock_(true, err)) { close_(); return false; }

    struct stat st{};
    if (fstat(fd_, &st) != 0) {
        const int e = errno;
        unlock_();
        close_();
        err = "cannot stat the handoff arena " + path + ": " + std::strerror(e);
        return false;
    }

    bool need_init = false;
    if (st.st_size == 0) {
        if (ftruncate(fd_, (off_t) sizing.arena_bytes) != 0) {
            const int e = errno;
            unlock_();
            close_();
            err = "cannot size the handoff arena " + path + ": " + std::strerror(e);
            return false;
        }
        need_init = true;
    } else if ((uint64_t) st.st_size != sizing.arena_bytes) {
        unlock_();
        close_();
        err = "handoff arena " + path + " is " + std::to_string((unsigned long long) st.st_size) +
              " B, this instance needs " + std::to_string(sizing.arena_bytes) +
              " B (its tier table differs)";
        return false;
    } else if (!read_header_(err)) {
        unlock_();
        close_();
        return false;
    } else if (hdr_.pack_hash != pack_hash) {
        err = "handoff arena " + path + " was written for pack hash " + hex16(hdr_.pack_hash) +
              ", expected " + hex16(pack_hash);
        unlock_();
        close_();
        return false;
    } else if (hdr_.geom_hash != geom_hash) {
        err = "handoff arena " + path + " was written for geometry hash " + hex16(hdr_.geom_hash) +
              ", expected " + hex16(geom_hash) + " (this instance runs a different layer split)";
        unlock_();
        close_();
        return false;
    } else if (hdr_.created_pid != (uint64_t) (uint32_t) getpid() &&
               handoff_claim_alive(hdr_.created_pid, hdr_.created_start) && !allow_existing_writer) {
        // ONE arena, ONE writer (deviation 4).  Adopting a live other process's arena is the torn-arena bug
        // stage 1 exists to prevent.  The pid is captured BEFORE close_(), which resets the header - a
        // refusal that names pid 0 is worse than no refusal at all.
        const uint64_t other = hdr_.created_pid;
        unlock_();
        close_();
        err = "handoff arena " + path + " is already being written by pid " +
              std::to_string((unsigned long long) other) + "; one arena per prefill instance";
        return false;
    } else if (hdr_.created_pid != (uint64_t) (uint32_t) getpid()) {
        // The recorded creator is gone (or its pid was recycled): the payloads in this file belong to a
        // process that no longer exists, so the slot table is rebuilt from scratch.  With
        // `allow_existing_writer` the table is deliberately NOT rebuilt - the test uses it to make two
        // processes contend for one slot.
        if (!allow_existing_writer) need_init = true;
    }
    // else: the same live process re-opening its own arena - keep the slots, a job may be in flight.

    if (need_init) {
        HandoffArenaHeader h{};
        std::memcpy(h.magic, kArenaMagic, kMagicBytes);
        h.version = kHandoffVersion;
        h.header_bytes = (uint32_t) kHandoffHeaderBytes;
        h.tier_count = (uint32_t) sizing.tiers.size();
        h.slot_total = sizing.slot_total;
        h.arena_bytes = sizing.arena_bytes;
        h.pack_hash = pack_hash;
        h.geom_hash = geom_hash;
        h.created_pid = (uint64_t) (uint32_t) getpid();
        h.created_start = handoff_proc_start_time(h.created_pid);
        for (size_t i = 0; i < sizing.tiers.size(); ++i) {
            h.tier_max_tokens[i] = sizing.tiers[i].max_tokens;
            h.tier_slot_count[i] = sizing.tiers[i].slot_count;
            h.tier_slot_bytes[i] = sizing.slot_bytes[i];
            h.tier_slot_off[i] = sizing.slot_off[i];
        }
        if (::pwrite(fd_, &h, sizeof h, 0) != (ssize_t) sizeof h) {
            const int e = errno;
            unlock_();
            close_();
            err = "cannot write the handoff arena header" +
                  (e ? std::string(": ") + std::strerror(e) : std::string(": short write"));
            return false;
        }
        hdr_ = h;
        if (!init_slots_(err)) {
            unlock_();
            close_();
            return false;
        }
    }

    unlock_();
    return true;
}

/// Every slot header to FREE.  Done once by the creator, under the arena lock.
bool HandoffArena::init_slots_(std::string& err) {
    HandoffSlotHeader s{};
    std::memcpy(s.magic, kSlotMagic, kMagicBytes);
    s.version = kHandoffVersion;
    s.header_bytes = (uint32_t) kHandoffHeaderBytes;
    s.state = kHandoffFree;
    for (uint32_t t = 0; t < hdr_.tier_count; ++t) {
        for (uint64_t k = 0; k < hdr_.tier_slot_count[t]; ++k) {
            const uint64_t base = hdr_.tier_slot_off[t] + k * (kHandoffHeaderBytes + hdr_.tier_slot_bytes[t]);
            s.tier = t;
            s.pad = 0;
            if (::pwrite(fd_, &s, sizeof s, (off_t) base) != (ssize_t) sizeof s) {
                err = "cannot initialise the handoff slot at " + std::to_string(base);
                return false;
            }
        }
    }
    return true;
}

void HandoffArena::note_client_pid(uint64_t client_id, uint64_t pid) {
    std::lock_guard<std::mutex> guard(mu_);
    for (auto& kv : client_pids_) {
        if (kv.first == client_id) { kv.second = pid; return; }
    }
    client_pids_.emplace_back(client_id, pid);
}

// ================================ reclaim: leases and dead owners (§3.6) ================================
//
// The slot-leak fix.  Without it, a decode instance that dies after `DONE` but before `ACK` strands a
// multi-GB slot until the prefill instance restarts.  A reclaim is a DECISION, so it runs under the arena
// lock, and it sets `flags |= abandoned` BEFORE `state = FREE` so the log can say what it reclaimed and
// why.  `reclaim_locked_` assumes the caller already holds BOTH `mu_` and the arena's fd lock (so
// `claim()` can use it inside its own locked pass without deadlocking on a non-recursive mutex or
// dropping the lock it is relying on).

std::vector<HandoffReclaim> HandoffArena::reclaim_locked_(std::string* err) {
    std::vector<HandoffReclaim> took;
    if (fd_ < 0) { if (err) *err = "the handoff arena is not open"; return took; }
    int index = 0;
    for (uint32_t t = 0; t < hdr_.tier_count; ++t) {
        for (uint64_t k = 0; k < hdr_.tier_slot_count[t]; ++k, ++index) {
            HandoffSlotHeader s{};
            uint64_t base = 0, capacity = 0;
            if (!read_slot_header_(index, s, base, capacity)) continue;
            if (s.state == kHandoffFree) continue;
            HandoffReclaim r;
            r.index = index;
            r.was_state = s.state;
            r.flags = s.flags;
            r.owner_pid = s.owner_pid;
            r.client_id = s.client_id;
            r.nonce = s.nonce;
            r.payload_bytes = s.payload_bytes;
            const uint64_t then = (s.state == kHandoffReady) ? s.published_ms : s.claimed_ms;
            bool take = false;
            if (s.state == kHandoffClaimed) {
                if (!handoff_claim_alive(s.owner_pid, s.owner_start)) {
                    r.why = "the claiming writer (pid " + std::to_string((unsigned long long) s.owner_pid) +
                            ") is gone";
                    take = true;
                } else if (lease_expired_(s.claimed_ms)) {
                    r.why = "its CLAIMED lease expired after " + std::to_string(handoff_now_ms() - then) +
                            " ms";
                    take = true;
                }
            } else if (s.state == kHandoffReady) {
                if (client_dead_(s.client_id)) {
                    r.why = "its reader (client " + hex16(s.client_id) + ") is gone";
                    take = true;
                } else if (lease_expired_(s.published_ms)) {
                    r.why = "nobody read it within the " + std::to_string(lease_ms_) + " ms lease";
                    take = true;
                }
            }
            if (!take) continue;
            const uint32_t flags = s.flags | kHandoffFlagAbandoned;
            const uint32_t free_state = kHandoffFree;
            if (!write_slot_fields_(base, kSlotFlagsOffset, &flags, sizeof flags) ||
                !write_slot_fields_(base, kSlotStateOffset, &free_state, sizeof free_state)) {
                if (err) *err = "cannot reclaim the handoff slot at " + std::to_string(base);
                continue;
            }
            r.flags = flags;
            r.age_ms = then == 0 ? 0 : handoff_now_ms() - then;
            took.push_back(r);
        }
    }
    return took;
}

std::vector<HandoffReclaim> HandoffArena::reclaim(std::string* err) {
    std::lock_guard<std::mutex> guard(mu_);
    if (fd_ < 0) { if (err) *err = "the handoff arena is not open"; return {}; }
    std::string lerr;
    if (!lock_(true, lerr)) { if (err) *err = lerr; return {}; }
    std::vector<HandoffReclaim> took = reclaim_locked_(err);
    unlock_();
    return took;
}

// ================================ claim (§3.5 step 1, §3.8 tier rule) ================================

bool HandoffArena::claim(uint32_t tier, uint64_t client_id, uint64_t request_id, HandoffClaim& out,
                         std::string& err) {
    std::lock_guard<std::mutex> guard(mu_);
    out = HandoffClaim{};
    err.clear();
    if (fd_ < 0) { err = "the handoff arena is not open"; return false; }
    if (!writer_) { err = "a reader never claims a handoff slot"; return false; }
    if (tier >= hdr_.tier_count) {
        err = "handoff tier " + std::to_string(tier) + " is not in this arena (" +
              std::to_string(hdr_.tier_count) + " tiers)";
        return false;
    }
    std::string lerr;
    if (!lock_(true, lerr)) { err = lerr; return false; }

    // Expired leases and dead claims become free slots inside this same locked pass, so a slot stranded by
    // a crashed reader is available here rather than looking permanently full.
    reclaim_locked_(nullptr);

    // §3.8 tier starvation: a job of tier `i` may take a free slot in group `j >= i`, PREFERRING the
    // smallest such `j`, so a 1 024-token job never sits in a 3 GB slot while a 131 072-token job waits.
    bool got = false;
    for (uint32_t j = tier; j < hdr_.tier_count && !got; ++j) {
        int index = 0;
        for (uint32_t q = 0; q < j; ++q) index += (int) hdr_.tier_slot_count[q];
        for (uint64_t k = 0; k < hdr_.tier_slot_count[j] && !got; ++k, ++index) {
            HandoffSlotHeader s{};
            uint64_t base = 0, capacity = 0;
            if (!read_slot_header_(index, s, base, capacity)) {
                err = "cannot read the handoff slot header at index " + std::to_string(index);
                break;
            }
            bool usable = (s.state == kHandoffFree);
            if (!usable && s.state == kHandoffClaimed && !handoff_claim_alive(s.owner_pid, s.owner_start)) {
                usable = true;   // §3.6: a dead or recycled claimer is reclaimable
            }
            if (!usable) continue;

            const uint64_t nonce = make_nonce();
            const uint64_t now = handoff_now_ms();
            const uint64_t pid = (uint64_t) (uint32_t) getpid();
            const uint64_t start = handoff_proc_start_time(pid);
            const uint32_t version = kHandoffVersion;
            const uint32_t header_bytes = (uint32_t) kHandoffHeaderBytes;
            const uint32_t claimed = kHandoffClaimed;
            const uint32_t flags = 0;
            const uint64_t zero64 = 0;
            // One decision, one store: the whole slot header, with `state = CLAIMED` inside it.  Nothing in
            // the slot is readable until the payload is published, so writing the full header here cannot
            // expose a half-claimed slot - and it means a claim is never half-applied.
            uint8_t record[kHandoffHeaderBytes];
            std::memset(record, 0, sizeof record);
            std::memcpy(record + kSlotMagicOffset, kSlotMagic, kMagicBytes);
            std::memcpy(record + kSlotVersionOffset, &version, sizeof version);
            std::memcpy(record + kSlotHeaderBytesOffset, &header_bytes, sizeof header_bytes);
            std::memcpy(record + kSlotStateOffset, &claimed, sizeof claimed);
            std::memcpy(record + kSlotFlagsOffset, &flags, sizeof flags);
            std::memcpy(record + kSlotTierOffset, &j, sizeof j);
            std::memcpy(record + kSlotOwnerOffset, &pid, sizeof pid);
            std::memcpy(record + kSlotOwnerStartOffset, &start, sizeof start);
            std::memcpy(record + kSlotNonceOffset, &nonce, sizeof nonce);
            std::memcpy(record + kSlotClientOffset, &client_id, sizeof client_id);
            std::memcpy(record + kSlotRequestOffset, &request_id, sizeof request_id);
            std::memcpy(record + kSlotPackHashOffset, &hdr_.pack_hash, sizeof hdr_.pack_hash);
            std::memcpy(record + kSlotGeomHashOffset, &hdr_.geom_hash, sizeof hdr_.geom_hash);
            std::memcpy(record + kSlotPayloadBytesOffset, &zero64, sizeof zero64);
            std::memcpy(record + kSlotClaimedMsOffset, &now, sizeof now);
            if (::pwrite(fd_, record, sizeof record, (off_t) base) != (ssize_t) sizeof record) {
                err = "cannot claim the handoff slot at index " + std::to_string(index);
                break;
            }
            out.index = index;
            out.base = base;
            out.payload_off = base + kHandoffHeaderBytes;
            out.capacity = capacity;
            out.tier = j;
            out.nonce = nonce;
            got = true;
        }
    }
    unlock_();
    if (!got) {
        // TEMPORARY, never an ERR on the client's wire (§3.8): the caller answers QUEUED and the decode
        // instance holds the request with reason `prefill`.
        err = "no free handoff slot for tier " + std::to_string(tier) + " or larger (" +
              free_slots_locked_() + " free of " + std::to_string(hdr_.slot_total) + ")";
    }
    return got;
}

// ================================ write / publish / release ================================

bool HandoffArena::write_payload(const HandoffClaim& c, uint64_t offset, const void* data, uint64_t bytes,
                                 std::string& err) {
    if (fd_ < 0) { err = "the handoff arena is not open"; return false; }
    if (!writer_) { err = "a reader never writes a handoff slot"; return false; }
    if (data == nullptr && bytes != 0) { err = "handoff payload write with no data"; return false; }
    if (offset > c.capacity || bytes > c.capacity - offset) {
        err = "handoff slot " + std::to_string(c.index) + " holds " + std::to_string(c.capacity) +
              " B and this write reaches " + std::to_string(offset + bytes) + " B";
        return false;
    }
    // NO LOCK HELD (§3.5 step 2).  The slot's state is still CLAIMED, so a reader that stumbles in sees
    // "not populated", never a half-written body.
    const ssize_t got = ::pwrite(fd_, data, bytes, (off_t) (c.payload_off + offset));
    if (got != (ssize_t) bytes) {
        err = "short handoff payload write in slot " + std::to_string(c.index) + ": wrote " +
              std::to_string((long long) (got < 0 ? 0 : got)) + " of " + std::to_string(bytes) + " B" +
              (got < 0 ? std::string(": ") + std::strerror(errno) : std::string());
        return false;
    }
    return true;
}

bool HandoffArena::publish(const HandoffClaim& c, uint64_t tokens, uint64_t payload_bytes,
                           uint64_t payload_hash, std::string& err) {
    if (fd_ < 0) { err = "the handoff arena is not open"; return false; }
    if (!writer_) { err = "a reader never publishes a handoff slot"; return false; }
    if (payload_bytes > c.capacity) {
        err = "handoff slot " + std::to_string(c.index) + " holds " + std::to_string(c.capacity) +
              " B and the payload is " + std::to_string(payload_bytes) + " B";
        return false;
    }
    std::string lerr;
    if (!lock_(true, lerr)) { err = lerr; return false; }
    HandoffSlotHeader s{};
    uint64_t base = 0, capacity = 0;
    const bool readable = read_slot_header_(c.index, s, base, capacity);
    // The claim must still be OURS: same state, same nonce.  A slot whose lease expired, that was
    // reclaimed, or that a restarted instance re-initialised is not ours to publish - the exact rule
    // `publish_shared_load()` enforces for the expert arena.
    if (!readable || s.state != kHandoffClaimed || s.nonce != c.nonce) {
        unlock_();
        err = "the handoff slot " + std::to_string(c.index) + " claim is gone (state " +
              std::to_string(readable ? (int) s.state : 99) + ", nonce " +
              (readable ? hex16(s.nonce) : std::string("?")) + " != " + hex16(c.nonce) + ")";
        return false;
    }
    if (s.flags & kHandoffFlagCancelled) {
        unlock_();
        err = "the handoff slot " + std::to_string(c.index) + " job was cancelled";
        return false;
    }
    unlock_();

    // Deviation 1, part A: the payload metadata in ONE contiguous store, while the slot is STILL CLAIMED.
    struct PublishBlock {
        uint64_t tokens, payload_bytes, payload_hash, claimed_ms, published_ms, seq;
    };
    PublishBlock blk;
    blk.tokens = tokens;
    blk.payload_bytes = payload_bytes;
    blk.payload_hash = payload_hash;
    blk.claimed_ms = s.claimed_ms;
    blk.published_ms = handoff_now_ms();
    blk.seq = 1;   // §3.9: v1 is one window; a DONE with seq != 1 is ERR geom
    static_assert(sizeof(PublishBlock) == kSlotPublishBlockBytes);
    if (!write_slot_fields_(base, kSlotPublishBlockOffset, &blk, sizeof blk)) {
        err = "cannot write the handoff slot " + std::to_string(c.index) + " payload record";
        return false;
    }
    // Part B: the release fence, then ONE ordered 8-byte store of state+flags.  That store is the publish
    // point: until it lands no reader may copy a byte.  Writing through the fd rather than a mapping is the
    // same reason `publish_shared_load()` does (`pinned.hpp:122-126`).
    std::atomic_thread_fence(std::memory_order_release);
    const uint32_t pair[2] = {kHandoffReady, s.flags & ~kHandoffFlagPartial};
    static_assert(sizeof(pair) == kSlotPublishStateBytes);
    if (!write_slot_fields_(base, kSlotStateOffset, pair, sizeof pair)) {
        err = "cannot publish the handoff slot " + std::to_string(c.index);
        return false;
    }
    return true;
}

bool HandoffArena::release(int index, uint64_t nonce, std::string& err) {
    std::lock_guard<std::mutex> guard(mu_);
    if (fd_ < 0) { err = "the handoff arena is not open"; return false; }
    if (!writer_) { err = "a reader never releases a handoff slot"; return false; }
    std::string lerr;
    if (!lock_(true, lerr)) { err = lerr; return false; }
    HandoffSlotHeader s{};
    uint64_t base = 0, capacity = 0;
    if (!read_slot_header_(index, s, base, capacity) || s.nonce != nonce) {
        unlock_();
        err = "handoff slot " + std::to_string(index) + " is not the one nonce " + hex16(nonce) +
              " refers to";
        return false;
    }
    // §3.5 step 5: FREE, and nothing else.  The body is NOT zeroed - the next writer overwrites it, and
    // zeroing 3 GB to free a slot would make the free path slower than the write path.
    const uint32_t free_state = kHandoffFree;
    const uint32_t zero32 = 0;
    const uint64_t zero64 = 0;
    const bool ok = write_slot_fields_(base, kSlotStateOffset, &free_state, sizeof free_state) &&
                    write_slot_fields_(base, kSlotFlagsOffset, &zero32, sizeof zero32) &&
                    write_slot_fields_(base, kSlotOwnerOffset, &zero64, sizeof zero64) &&
                    write_slot_fields_(base, kSlotOwnerStartOffset, &zero64, sizeof zero64) &&
                    write_slot_fields_(base, kSlotNonceOffset, &zero64, sizeof zero64) &&
                    write_slot_fields_(base, kSlotClientOffset, &zero64, sizeof zero64) &&
                    write_slot_fields_(base, kSlotPayloadBytesOffset, &zero64, sizeof zero64);
    unlock_();
    if (!ok) err = "cannot release the handoff slot " + std::to_string(index);
    return ok;
}

bool HandoffArena::abort_claim(int index, uint64_t nonce, std::string& err) {
    std::lock_guard<std::mutex> guard(mu_);
    if (fd_ < 0) { err = "the handoff arena is not open"; return false; }
    if (!writer_) { err = "a reader never aborts a handoff claim"; return false; }
    std::string lerr;
    if (!lock_(true, lerr)) { err = lerr; return false; }
    HandoffSlotHeader s{};
    uint64_t base = 0, capacity = 0;
    if (!read_slot_header_(index, s, base, capacity) || s.nonce != nonce || s.state != kHandoffClaimed) {
        unlock_();
        err = "handoff slot " + std::to_string(index) + " is not a claim of nonce " + hex16(nonce);
        return false;
    }
    const uint32_t flags = s.flags | kHandoffFlagPartial;
    const uint32_t free_state = kHandoffFree;
    const bool ok = write_slot_fields_(base, kSlotFlagsOffset, &flags, sizeof flags) &&
                    write_slot_fields_(base, kSlotStateOffset, &free_state, sizeof free_state);
    unlock_();
    if (!ok) err = "cannot abort the handoff slot " + std::to_string(index);
    return ok;
}

bool HandoffArena::cancel(int index, uint64_t nonce, std::string& err) {
    std::lock_guard<std::mutex> guard(mu_);
    if (fd_ < 0) { err = "the handoff arena is not open"; return false; }
    std::string lerr;
    if (!lock_(true, lerr)) { err = lerr; return false; }
    HandoffSlotHeader s{};
    uint64_t base = 0, capacity = 0;
    if (!read_slot_header_(index, s, base, capacity) || s.nonce != nonce) {
        unlock_();
        err = "handoff slot " + std::to_string(index) + " is not the one nonce " + hex16(nonce) +
              " refers to";
        return false;
    }
    const uint32_t flags = s.flags | kHandoffFlagCancelled;
    const bool ok = write_slot_fields_(base, kSlotFlagsOffset, &flags, sizeof flags);
    unlock_();
    if (!ok) err = "cannot cancel the handoff slot " + std::to_string(index);
    return ok;
}

// ================================ open_reader / read_slot (§3.5 step 4) ================================

bool HandoffArena::open_reader(const std::string& path, uint64_t pack_hash, uint64_t geom_hash,
                               std::string& err) {
    std::lock_guard<std::mutex> guard(mu_);
    close_();
    err.clear();
    // The decode instance OPENS the arena; it never creates it and never `ftruncate`s it (§3.2).
    const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        err = "cannot open the handoff arena " + path + ": " + std::strerror(errno);
        return false;
    }
    fd_ = fd;
    path_ = path;
    writer_ = false;
    std::string lerr;
    // A SHARED lock for the header read only: several decode instances may look at the table at once, and
    // none of them may do it while the writer is making a decision.
    if (!lock_(false, lerr)) { close_(); err = lerr; return false; }
    struct stat st{};
    bool ok = fstat(fd_, &st) == 0;
    if (ok && !read_header_(err)) ok = false;
    if (ok && (uint64_t) st.st_size != hdr_.arena_bytes) {
        err = "handoff arena " + path + " is " + std::to_string((long long) st.st_size) +
              " B but its header says " + std::to_string(hdr_.arena_bytes) + " B";
        ok = false;
    }
    // Both hashes, before any byte is copied, and the refusal names both values (§3.10).
    if (ok && hdr_.pack_hash != pack_hash) {
        err = "handoff arena " + path + " was written for pack hash " + hex16(hdr_.pack_hash) +
              ", this instance is " + hex16(pack_hash);
        ok = false;
    }
    if (ok && hdr_.geom_hash != geom_hash) {
        err = "handoff arena " + path + " was written for geometry hash " + hex16(hdr_.geom_hash) +
              ", this instance is " + hex16(geom_hash) + " (a different layer split)";
        ok = false;
    }
    unlock_();
    if (!ok) { close_(); return false; }
    return true;
}

bool HandoffArena::slot_info(int index, HandoffSlotInfo& out) const {
    std::lock_guard<std::mutex> guard(mu_);
    out = HandoffSlotInfo{};
    if (fd_ < 0) return false;
    std::string err;
    if (!lock_(false, err)) return false;
    HandoffSlotHeader s{};
    uint64_t base = 0, capacity = 0;
    const bool ok = read_slot_header_(index, s, base, capacity);
    unlock_();
    if (!ok) return false;
    out.index = index;
    out.base = base;
    out.capacity = capacity;
    out.state = s.state;
    out.flags = s.flags;
    out.tier = s.tier;
    out.owner_pid = s.owner_pid;
    out.owner_start = s.owner_start;
    out.nonce = s.nonce;
    out.client_id = s.client_id;
    out.request_id = s.request_id;
    out.pack_hash = s.pack_hash;
    out.geom_hash = s.geom_hash;
    out.tokens = s.tokens;
    out.payload_bytes = s.payload_bytes;
    out.payload_hash = s.payload_hash;
    out.claimed_ms = s.claimed_ms;
    out.published_ms = s.published_ms;
    out.seq = s.seq;
    return true;
}

bool HandoffArena::read_slot(int index, uint64_t expect_nonce, uint64_t client_id, HandoffPayload& out,
                             std::string& err) {
    std::lock_guard<std::mutex> guard(mu_);
    out = HandoffPayload{};
    if (fd_ < 0) { err = "the handoff arena is not open"; return false; }
    if (writer_) { err = "open the arena as a reader to read a handoff slot"; return false; }
    // §3.5 step 4: the shared flock covers the HEADER READ ONLY.  It is never held across the copy.
    if (!lock_(false, err)) return false;
    HandoffSlotHeader s{};
    uint64_t base = 0, capacity = 0;
    const bool readable = read_slot_header_(index, s, base, capacity);
    unlock_();
    if (!readable) {
        err = "cannot read the handoff slot header at index " + std::to_string(index);
        return false;
    }
    if (std::memcmp(s.magic, kSlotMagic, kMagicBytes) != 0 || s.version != kHandoffVersion ||
        s.header_bytes != kHandoffHeaderBytes) {
        err = "handoff slot " + std::to_string(index) + " has an incompatible header";
        return false;
    }
    // A CLAIMED slot is "not populated".  This check is the whole reason a reader can never see a
    // half-written body - the fix stage 1 made for the expert arena (`pinned.cu:381-383`).
    if (s.state != kHandoffReady) {
        err = "handoff slot " + std::to_string(index) + " is " +
              (s.state == kHandoffClaimed ? "CLAIMED (nobody published it)" : "FREE") + ", not READY";
        return false;
    }
    if (s.nonce != expect_nonce) {
        err = "handoff slot " + std::to_string(index) + " nonce " + hex16(s.nonce) +
              " is not the DONE line's nonce " + hex16(expect_nonce);
        return false;
    }
    if (s.client_id != client_id) {
        err = "handoff slot " + std::to_string(index) + " is addressed to client " + hex16(s.client_id) +
              ", not " + hex16(client_id);
        return false;
    }
    if (s.pack_hash != hdr_.pack_hash || s.geom_hash != hdr_.geom_hash) {
        err = "handoff slot " + std::to_string(index) + " was written for pack " + hex16(s.pack_hash) +
              " / geometry " + hex16(s.geom_hash) + ", this arena is " + hex16(hdr_.pack_hash) +
              " / " + hex16(hdr_.geom_hash);
        return false;
    }
    if (s.payload_bytes > capacity) {
        err = "handoff slot " + std::to_string(index) + " claims " + std::to_string(s.payload_bytes) +
              " B in a " + std::to_string(capacity) + " B slot";
        return false;
    }
    if (s.seq != 1) {
        err = "handoff slot " + std::to_string(index) + " has seq " + std::to_string(s.seq) +
              "; v1 speaks seq 1 only";
        return false;
    }
    // Map PROT_READ, MAP_SHARED over the payload region only, page-aligned down so the slot header is not
    // part of the mapping at all.
    const uint64_t off = base + kHandoffHeaderBytes;
    const long ps = sysconf(_SC_PAGESIZE);
    const uint64_t page = ps > 0 ? (uint64_t) ps : 4096ull;
    const uint64_t head = off % page;
    const uint64_t map_off = off - head;
    const uint64_t map_bytes = head + capacity;
    void* map = ::mmap(nullptr, (size_t) map_bytes, PROT_READ, MAP_SHARED, fd_, (off_t) map_off);
    if (map == MAP_FAILED) {
        err = "cannot map handoff slot " + std::to_string(index) + ": " + std::strerror(errno);
        return false;
    }
    out.bind_mapping(map, map_bytes, (const uint8_t*) map + head, s.payload_bytes);
    out.index = index;
    out.tier = s.tier;
    out.nonce = s.nonce;
    out.client_id = s.client_id;
    out.request_id = s.request_id;
    out.tokens = s.tokens;
    out.payload_bytes = s.payload_bytes;
    out.payload_hash = s.payload_hash;
    out.seq = s.seq;
    return true;
}

uint64_t HandoffArena::payload_hash_of(const HandoffPayload& p) {
    // The engine's own FNV-1a (`pinned.hpp:173`), not a second one.
    return fnv1a64(p.data, p.payload_bytes);
}

std::string HandoffArena::free_slots_locked_() const {
    uint64_t free_n = 0;
    int index = 0;
    for (uint32_t t = 0; t < hdr_.tier_count; ++t) {
        for (uint64_t k = 0; k < hdr_.tier_slot_count[t]; ++k, ++index) {
            HandoffSlotHeader s{};
            uint64_t base = 0, capacity = 0;
            if (read_slot_header_(index, s, base, capacity) && s.state == kHandoffFree) ++free_n;
        }
    }
    return std::to_string(free_n) + "/" + std::to_string(hdr_.slot_total);
}

std::string HandoffArena::free_slots(std::string* err) const {
    std::lock_guard<std::mutex> guard(mu_);
    if (fd_ < 0) { if (err) *err = "the handoff arena is not open"; return "0/0"; }
    std::string lerr;
    if (!lock_(false, lerr)) { if (err) *err = lerr; return "0/0"; }
    const std::string s = free_slots_locked_();
    unlock_();
    return s;
}

// ================================ HandoffPayload ================================

void HandoffPayload::bind_mapping(void* map, uint64_t mapping_bytes, const uint8_t* d, uint64_t payload) {
    unbind();
    map_ = map;
    map_bytes_ = mapping_bytes;
    data = d;
    bytes = payload;
}

void HandoffPayload::unbind() {
    if (map_ != nullptr) {
        ::munmap(map_, (size_t) map_bytes_);
        map_ = nullptr;
    }
    map_bytes_ = 0;
    data = nullptr;
    bytes = 0;
}

HandoffPayload::~HandoffPayload() { unbind(); }

HandoffPayload::HandoffPayload(HandoffPayload&& o) noexcept
    : data(o.data), bytes(o.bytes), index(o.index), tier(o.tier), nonce(o.nonce), client_id(o.client_id),
      request_id(o.request_id), tokens(o.tokens), payload_bytes(o.payload_bytes),
      payload_hash(o.payload_hash), seq(o.seq), map_(o.map_), map_bytes_(o.map_bytes_) {
    o.map_ = nullptr;
    o.map_bytes_ = 0;
    o.data = nullptr;
    o.bytes = 0;
}

HandoffPayload& HandoffPayload::operator=(HandoffPayload&& o) noexcept {
    if (this != &o) {
        unbind();
        data = o.data;
        bytes = o.bytes;
        index = o.index;
        tier = o.tier;
        nonce = o.nonce;
        client_id = o.client_id;
        request_id = o.request_id;
        tokens = o.tokens;
        payload_bytes = o.payload_bytes;
        payload_hash = o.payload_hash;
        seq = o.seq;
        map_ = o.map_;
        map_bytes_ = o.map_bytes_;
        o.map_ = nullptr;
        o.map_bytes_ = 0;
        o.data = nullptr;
        o.bytes = 0;
    }
    return *this;
}

}  // namespace strata::core
