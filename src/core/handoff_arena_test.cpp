// src/core/handoff_arena_test.cpp - S4.3.2: the CPU-only test for the tmpfs handoff arena.
//
// CPU-ONLY ON PURPOSE: no model, no GPU, no engine, no server, no request.  Everything the handoff
// protocol decides is a decision about a 4 KiB header, a slot header and a handful of bytes, so all of it
// is testable against a small arena in about a second.  What this file must NOT do is re-derive the
// protocol from the engine's logs - the bugs it covers (a half-written 3 GB payload mounted by a reader, a
// slot stranded forever by a crashed decode instance, an `ftruncate` that succeeds on a tmpfs that cannot
// back it) only show up at GiB scale in production, which is exactly why they need a small deterministic
// test next to them.
//
// It never touches /dev/shm/shared_experts.dat or /dev/shm/strata-core-leases: every arena here lives
// under a fresh /dev/shm/strata-handoff-test-<pid>/ directory that only this process created, and that
// directory is the only thing it removes.
//
// The cross-process cases re-exec THIS binary with `--role`, the pattern `src/core/pinned_shared_test.cpp`
// already uses.  Threads share an fd table, so nothing else in this file can prove the three things the
// protocol actually leans on between processes: that `flock` contends between unrelated address spaces,
// that a slot claim is EXCLUSIVE between two processes, and that a byte one process wrote is visible to
// another through the FILE rather than through a lucky alias.
#include "strata/core/handoff.hpp"

#include <atomic>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>   ///< `flock()` itself - <fcntl.h> only gives the `struct flock` record-lock type
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace strata::core;

namespace {

int g_checks = 0;
int g_fail = 0;

void check(bool ok, const char* what) {
    ++g_checks;
    if (ok) {
        std::printf("  ok   %s\n", what);
        return;
    }
    std::printf("  FAIL %s\n", what);
    ++g_fail;
}

/// `check(!api(..., err), ("... " + err).c_str())` is a BUG: C++ does not fix the evaluation order of
/// arguments, so `err` can be read before the call filled it - the diagnostic prints empty and the real
/// reason is lost.  This form binds `err` by reference and formats inside the function, which runs after
/// every argument expression, so the message is always the one the API actually produced.
void check_err(bool ok, const std::string& err, const char* what) {
    ++g_checks;
    if (ok) {
        std::printf("  ok   %s: %s\n", what, err.c_str());
        return;
    }
    std::printf("  FAIL %s: %s\n", what, err.c_str());
    ++g_fail;
}

/// The §3.3/§3.4 layout, restated INDEPENDENTLY of handoff.hpp.  handoff.hpp `static_assert`s its own
/// constants against its own structs; this table is what catches the two of them agreeing on a WRONG
/// offset, and each row is checked three ways: doc value == the header's constant == `offsetof`.
struct Row { const char* name; uint64_t doc; uint64_t constant; uint64_t actual; };

// ================================ the test arena's home ================================

std::string g_root;

std::string root_dir() {
    if (!g_root.empty()) return g_root;
    char path[256];
    std::snprintf(path, sizeof path, "/dev/shm/strata-handoff-test-%d", (int) getpid());
    std::error_code ec;
    std::filesystem::create_directories(path, ec);
    if (ec) {
        std::fprintf(stderr, "cannot create %s: %s\n", path, ec.message().c_str());
        std::exit(2);
    }
    g_root = std::string(path);
    return g_root;
}

std::string arena_path(const char* tag) { return root_dir() + "/" + tag + "/arena"; }

void remove_arena(const std::string& path) {
    std::error_code ec;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::remove_all(p.parent_path(), ec);
}

/// A pid that is definitely not running: fork, exit, reap.  A made-up pid would be a guess, and pid reuse
/// would make the test flaky in exactly the direction that matters - "dead" must really mean dead.
uint64_t dead_pid() {
    const pid_t p = fork();
    if (p < 0) { std::perror("fork"); std::exit(2); }
    if (p == 0) _exit(0);
    int status = 0;
    waitpid(p, &status, 0);
    return (uint64_t) (uint32_t) p;
}

// ================================ the fake pricer ================================
//
// The engine prices a tier with `conversation_snapshot_bytes`, which is host-side arithmetic over the
// model geometry (`include/strata/core/conversation_snapshot.hpp:125`).  This test must not build a
// SessionState or a QsaState, so it supplies the measured SHAPE that pricing has on this box - a ~236 MiB
// fixed floor plus ~22 KiB per context token (docs/STAGE4-SPLIT-ROLES.md Appendix A.2) - and checks the
// ARENA's arithmetic against the numbers it is handed.  S4.3.5 wires the real pricer in; the contract is
// the callback signature and the formula, not these constants.
bool fake_pricer(uint64_t tokens, uint64_t& bytes, std::string& err) {
    if (tokens == 0) { err = "0 tokens"; return false; }
    bytes = (236ull << 20) + tokens * 22528ull;
    return true;
}

/// The tier table every non-big case uses: 3 groups, 2 slots each, sized per tier.
const std::vector<HandoffTierSpec>& basic_tiers() {
    static const std::vector<HandoffTierSpec> t{{1024, 2}, {4096, 2}, {16384, 2}};
    return t;
}

HandoffSizing size_basic() {
    HandoffSizing s;
    std::string err;
    if (!handoff_size_arena(basic_tiers(), fake_pricer, s, err)) {
        std::fprintf(stderr, "sizing failed: %s\n", err.c_str());
        std::exit(2);
    }
    return s;
}

/// One slot per group: what the cross-process publish/read cases need.
HandoffSizing size_one_per_tier() {
    static const std::vector<HandoffTierSpec> t{{1024, 1}, {4096, 1}, {16384, 1}};
    HandoffSizing s;
    std::string err;
    if (!handoff_size_arena(t, fake_pricer, s, err)) {
        std::fprintf(stderr, "sizing failed: %s\n", err.c_str());
        std::exit(2);
    }
    return s;
}

/// ONE group, ONE slot.  The contention case needs this: §3.8 lets a tier-0 job cascade into a larger
/// group, so with three groups two racing processes could each legitimately get a slot and the test would
/// prove nothing.  With one slot in the only reachable group, exactly one process can win.
HandoffSizing size_single_slot() {
    static const std::vector<HandoffTierSpec> t{{1024, 1}};
    HandoffSizing s;
    std::string err;
    if (!handoff_size_arena(t, fake_pricer, s, err)) {
        std::fprintf(stderr, "sizing failed: %s\n", err.c_str());
        std::exit(2);
    }
    return s;
}

uint64_t slot_base_of(const HandoffSizing& s, int index) {
    int seen = 0;
    for (size_t t = 0; t < s.tiers.size(); ++t) {
        if (index < seen + (int) s.tiers[t].slot_count) {
            return s.slot_off[t] + (uint64_t) (index - seen) * (kHandoffHeaderBytes + s.slot_bytes[t]);
        }
        seen += (int) s.tiers[t].slot_count;
    }
    return UINT64_MAX;
}

// ================================ raw header poking ================================
//
// Some states the API deliberately cannot produce have to be tested anyway, because they are what a CRASH
// leaves behind: a claim by a dead pid, a READY slot with an expired stamp, a header with a future
// version.  The test writes those bytes itself, exactly like `pinned_shared_test.cpp` rewrites the expert
// arena's header to fake a dead loader.
bool patch(const std::string& path, uint64_t off, const void* p, uint64_t n) {
    const int fd = ::open(path.c_str(), O_RDWR);
    if (fd < 0) return false;
    const bool ok = ::pwrite(fd, p, n, (off_t) off) == (ssize_t) n;
    ::close(fd);
    return ok;
}

void forge_claim(const std::string& path, uint64_t base, uint32_t tier, uint64_t pid, uint64_t start,
                 uint64_t nonce, uint64_t client_id, uint64_t claimed_ms) {
    HandoffSlotHeader s{};
    std::memcpy(s.magic, "STRATA-HSLOT-V1", 16);
    s.version = kHandoffVersion;
    s.header_bytes = (uint32_t) kHandoffHeaderBytes;
    s.state = kHandoffClaimed;
    s.tier = tier;
    s.owner_pid = pid;
    s.owner_start = start;
    s.nonce = nonce;
    s.client_id = client_id;
    s.claimed_ms = claimed_ms;
    if (!patch(path, base, &s, sizeof s)) std::fprintf(stderr, "forge_claim could not write\n");
}

/// A READY slot with caller-chosen identity fields.  Used to forge the two states the API will not
/// produce: a slot whose own hashes disagree with the arena's, and a `seq` that is not 1.
void forge_ready(const std::string& path, uint64_t base, uint32_t tier, uint64_t nonce,
                 uint64_t client_id, uint64_t pack_hash, uint64_t geom_hash, uint64_t payload_bytes,
                 uint64_t seq) {
    HandoffSlotHeader s{};
    std::memcpy(s.magic, "STRATA-HSLOT-V1", 16);
    s.version = kHandoffVersion;
    s.header_bytes = (uint32_t) kHandoffHeaderBytes;
    s.state = kHandoffReady;
    s.tier = tier;
    s.owner_pid = (uint64_t) (uint32_t) getpid();
    s.nonce = nonce;
    s.client_id = client_id;
    s.pack_hash = pack_hash;
    s.geom_hash = geom_hash;
    s.tokens = 7;
    s.payload_bytes = payload_bytes;
    s.claimed_ms = handoff_now_ms();
    s.published_ms = handoff_now_ms();
    s.seq = seq;
    if (!patch(path, base, &s, sizeof s)) std::fprintf(stderr, "forge_ready could not write\n");
}

// ================================ the re-exec roles ================================

std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return std::string(b);
}

/// Pull `key=<digits>` out of a child's line.  -1 when absent, so a missing field reads as a failure
/// rather than as "slot 0".
long long field(const std::string& out, const char* key) {
    const std::string k = std::string(key) + "=";
    const size_t at = out.find(k);
    if (at == std::string::npos) return -1;
    return std::strtoll(out.c_str() + at + k.size(), nullptr, 10);
}

std::string hexfield(const std::string& out, const char* key) {
    const std::string k = std::string(key) + "=";
    const size_t at = out.find(k);
    if (at == std::string::npos) return {};
    char buf[32] = {0};
    size_t i = 0;
    const char* p = out.c_str() + at + k.size();
    while (i < sizeof buf - 1 && *p && *p != ' ' && *p != '\n') buf[i++] = *p++;
    return std::string(buf);
}

std::string run(const std::string& cmd) {
    std::string out;
    FILE* p = popen(cmd.c_str(), "r");
    if (p == nullptr) return out;
    char line[2048];
    while (std::fgets(line, sizeof line, p)) out += line;
    pclose(p);
    return out;
}

std::string self_path() {
    char exe[4096];
    const ssize_t n = ::readlink("/proc/self/exe", exe, sizeof exe - 1);
    if (n <= 0) return {};
    exe[n] = 0;
    return std::string(exe);
}

/// The payload a `writer` role child writes, and the pattern the `reader` role child re-checks.
constexpr uint64_t kRolePayload = 4096;
uint8_t role_byte(uint64_t i, uint64_t request_id) { return (uint8_t) (i * 7u + request_id); }

/// `--role <ROLE> <arena> <pack16> <geom16> <tier> <client_id> <request_id> <lease_ms> <sizing> [nonce16]`
///
/// `sizing` names the tier table the child must use, so the parent and the child cannot drift about the
/// arena's layout: `basic` = 2 slots per group, `per-tier` = 1 slot per group, `single` = 1 group 1 slot.
int role_mode(int argc, char** argv) {
    if (argc < 11) return 2;
    const std::string role = argv[2], path = argv[3];
    const uint64_t pack = std::strtoull(argv[4], nullptr, 16);
    const uint64_t geom = std::strtoull(argv[5], nullptr, 16);
    const uint32_t tier = (uint32_t) std::strtoul(argv[6], nullptr, 10);
    const uint64_t client = std::strtoull(argv[7], nullptr, 10);
    const uint64_t request = std::strtoull(argv[8], nullptr, 10);
    const uint64_t lease = std::strtoull(argv[9], nullptr, 10);
    const std::string sizing = argv[10];
    const uint64_t nonce = argc > 11 ? std::strtoull(argv[11], nullptr, 16) : 0;
    HandoffSizing sz;
    if (sizing == "basic") sz = size_basic();
    else if (sizing == "per-tier") sz = size_one_per_tier();
    else if (sizing == "single") sz = size_single_slot();
    else return 2;

    if (role == "flock-hold") {
        // Hold the arena's exclusive lock for a while, so the parent can watch a claim wait for it.
        const int fd = ::open(path.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) { std::printf("HOLD locked=0\n"); return 0; }
        const bool got = ::flock(fd, LOCK_EX) == 0;
        std::printf("HOLD locked=%d\n", (int) got);
        std::fflush(stdout);
        std::this_thread::sleep_for(std::chrono::milliseconds(1500));
        ::flock(fd, LOCK_UN);
        ::close(fd);
        std::printf("HOLD released=1\n");
        return 0;
    }

    if (role == "writer" || role == "writer-hold") {
        // `allow_existing_writer` is the test-only escape from the one-writer rule, which is exactly what
        // the contention case is proving is enforced by the LOCK rather than by luck.
        HandoffArena a;
        a.set_lease_ms(lease);
        std::string err;
        if (!a.open_writer(path, sz, pack, geom, err, true)) {
            std::printf("OPEN ok=0 err=%s\n", err.c_str());
            return 0;
        }
        std::printf("OPEN ok=1\n");
        HandoffClaim c;
        if (!a.claim(tier, client, request, c, err)) {
            std::printf("CLAIM ok=0 slot=-1 nonce=0 err=%s\n", err.c_str());
            return 0;
        }
        std::printf("CLAIM ok=1 slot=%d tier=%u nonce=%s\n", c.index, c.tier, hex(c.nonce).c_str());
        std::fflush(stdout);
        if (role == "writer-hold") {
            // Stay CLAIMED and alive: the parent's reclaim must leave it alone.
            std::this_thread::sleep_for(std::chrono::milliseconds(1200));
            std::printf("HOLD done=1\n");
            return 0;
        }
        std::vector<uint8_t> payload(kRolePayload);
        for (uint64_t i = 0; i < payload.size(); ++i) payload[i] = role_byte(i, request);
        if (!a.write_payload(c, 0, payload.data(), payload.size(), err)) {
            std::printf("WRITE ok=0 err=%s\n", err.c_str());
            return 0;
        }
        const uint64_t h = fnv1a64(payload.data(), payload.size());
        if (!a.publish(c, 1234, payload.size(), h, err)) {
            std::printf("PUBLISH ok=0 err=%s\n", err.c_str());
            return 0;
        }
        std::printf("PUBLISH ok=1 bytes=%llu hash=%s\n", (unsigned long long) payload.size(), hex(h).c_str());
        return 0;
    }

    if (role == "reader" || role == "reader-wrong-nonce" || role == "reader-wrong-client") {
        HandoffArena r;
        std::string err;
        if (!r.open_reader(path, pack, geom, err)) {
            std::printf("OPEN ok=0 err=%s\n", err.c_str());
            return 0;
        }
        std::printf("OPEN ok=1\n");
        const uint64_t want_nonce = role == "reader-wrong-nonce" ? nonce ^ 0x5555ull : nonce;
        const uint64_t want_client = role == "reader-wrong-client" ? client + 1 : client;
        HandoffPayload p;
        if (!r.read_slot((int) tier, want_nonce, want_client, p, err)) {
            std::printf("READ ok=0 err=%s\n", err.c_str());
            return 0;
        }
        const uint64_t h = HandoffArena::payload_hash_of(p);
        std::printf("READ ok=1 bytes=%llu hash=%s tokens=%llu\n", (unsigned long long) p.payload_bytes,
                    hex(h).c_str(), (unsigned long long) p.tokens);
        bool good = p.payload_bytes == kRolePayload;
        for (uint64_t i = 0; i < p.payload_bytes && good; ++i) good = p.data[i] == role_byte(i, p.request_id);
        std::printf("PAYLOAD ok=%d\n", (int) good);
        return 0;
    }

    std::printf("ROLE unknown %s\n", role.c_str());
    return 2;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--role") == 0) return role_mode(argc, argv);

    // ================================ 0. the layout is an ABI, not an implementation detail ============
    std::printf("layout\n");
    {
        const Row arena_rows[] = {
            {"arena.version", 16, kArenaVersionOffset, offsetof(HandoffArenaHeader, version)},
            {"arena.header_bytes", 20, kArenaHeaderBytesOffset, offsetof(HandoffArenaHeader, header_bytes)},
            {"arena.tier_count", 24, kArenaTierCountOffset, offsetof(HandoffArenaHeader, tier_count)},
            {"arena.slot_total", 32, kArenaSlotTotalOffset, offsetof(HandoffArenaHeader, slot_total)},
            {"arena.arena_bytes", 40, kArenaBytesOffset, offsetof(HandoffArenaHeader, arena_bytes)},
            {"arena.pack_hash", 48, kArenaPackHashOffset, offsetof(HandoffArenaHeader, pack_hash)},
            {"arena.geom_hash", 56, kArenaGeomHashOffset, offsetof(HandoffArenaHeader, geom_hash)},
            {"arena.created_pid", 64, kArenaCreatedPidOffset, offsetof(HandoffArenaHeader, created_pid)},
            {"arena.created_start", 72, kArenaCreatedStartOffset, offsetof(HandoffArenaHeader, created_start)},
            {"arena.tier_max_tokens", 80, kArenaTierMaxTokensOffset, offsetof(HandoffArenaHeader, tier_max_tokens)},
            {"arena.tier_slot_count", 144, kArenaTierSlotCountOffset, offsetof(HandoffArenaHeader, tier_slot_count)},
            {"arena.tier_slot_bytes", 208, kArenaTierSlotBytesOffset, offsetof(HandoffArenaHeader, tier_slot_bytes)},
            {"arena.tier_slot_off", 272, kArenaTierSlotOffOffset, offsetof(HandoffArenaHeader, tier_slot_off)},
            {"arena.reserved", 336, kArenaReservedOffset, offsetof(HandoffArenaHeader, reserved)},
        };
        const Row slot_rows[] = {
            {"slot.version", 16, kSlotVersionOffset, offsetof(HandoffSlotHeader, version)},
            {"slot.header_bytes", 20, kSlotHeaderBytesOffset, offsetof(HandoffSlotHeader, header_bytes)},
            {"slot.state", 24, kSlotStateOffset, offsetof(HandoffSlotHeader, state)},
            {"slot.flags", 28, kSlotFlagsOffset, offsetof(HandoffSlotHeader, flags)},
            {"slot.tier", 32, kSlotTierOffset, offsetof(HandoffSlotHeader, tier)},
            {"slot.owner_pid", 40, kSlotOwnerOffset, offsetof(HandoffSlotHeader, owner_pid)},
            {"slot.owner_start", 48, kSlotOwnerStartOffset, offsetof(HandoffSlotHeader, owner_start)},
            {"slot.nonce", 56, kSlotNonceOffset, offsetof(HandoffSlotHeader, nonce)},
            {"slot.client_id", 64, kSlotClientOffset, offsetof(HandoffSlotHeader, client_id)},
            {"slot.request_id", 72, kSlotRequestOffset, offsetof(HandoffSlotHeader, request_id)},
            {"slot.pack_hash", 80, kSlotPackHashOffset, offsetof(HandoffSlotHeader, pack_hash)},
            {"slot.geom_hash", 88, kSlotGeomHashOffset, offsetof(HandoffSlotHeader, geom_hash)},
            {"slot.tokens", 96, kSlotTokensOffset, offsetof(HandoffSlotHeader, tokens)},
            {"slot.payload_bytes", 104, kSlotPayloadBytesOffset, offsetof(HandoffSlotHeader, payload_bytes)},
            {"slot.payload_hash", 112, kSlotPayloadHashOffset, offsetof(HandoffSlotHeader, payload_hash)},
            {"slot.claimed_ms", 120, kSlotClaimedMsOffset, offsetof(HandoffSlotHeader, claimed_ms)},
            {"slot.published_ms", 128, kSlotPublishedMsOffset, offsetof(HandoffSlotHeader, published_ms)},
            {"slot.seq", 136, kSlotSeqOffset, offsetof(HandoffSlotHeader, seq)},
            {"slot.reserved", 144, kSlotReservedOffset, offsetof(HandoffSlotHeader, reserved)},
        };
        bool arena_ok = true;
        for (const auto& r : arena_rows) {
            if (r.doc != r.constant || r.doc != r.actual) {
                std::printf("  ..   %s doc=%llu constant=%llu offsetof=%llu\n", r.name,
                            (unsigned long long) r.doc, (unsigned long long) r.constant,
                            (unsigned long long) r.actual);
                arena_ok = false;
            }
        }
        check(arena_ok, "every arena-header offset: doc == constant == offsetof");
        bool slot_ok = true;
        for (const auto& r : slot_rows) {
            if (r.doc != r.constant || r.doc != r.actual) {
                std::printf("  ..   %s doc=%llu constant=%llu offsetof=%llu\n", r.name,
                            (unsigned long long) r.doc, (unsigned long long) r.constant,
                            (unsigned long long) r.actual);
                slot_ok = false;
            }
        }
        check(slot_ok, "every slot-header offset: doc == constant == offsetof");
        check(kHandoffHeaderBytes == 4096 && sizeof(HandoffArenaHeader) <= 4096 &&
                  sizeof(HandoffSlotHeader) <= 4096,
              "both headers fit their 4096-byte block");
        check(kSlotMagicOffset == 0 && kArenaMagicOffset == 0, "magic is at offset 0 in both");
        check(kSlotPublishBlockOffset == 96 && kSlotPublishBlockBytes == 48 &&
                  kSlotPublishBlockOffset + kSlotPublishBlockBytes == kSlotReservedOffset,
              "the publish block is the contiguous run [96,144) the header documents");
        check(kSlotStateOffset == 24 && kSlotFlagsOffset == 28 && kSlotPublishStateBytes == 8,
              "the publish point is one 8-byte store of state+flags at offset 24");
        check(kHandoffDefaultLeaseMs == 300000ull, "the default lease is 300 s (§3.6)");
        check(kHandoffSlackBytes == (64ull << 20), "the per-slot slack is 64 MiB (§3.8)");
        check(kHandoffFree == 0 && kHandoffClaimed == 1 && kHandoffReady == 2,
              "FREE=0 CLAIMED=1 READY=2, as §3.4 pins them");
        check(kHandoffFlagCancelled == 1 && kHandoffFlagAbandoned == 2 && kHandoffFlagPartial == 4,
              "the three flag bits are as §3.4 pins them");
    }

    // ================================ 1. identity: split spec, geometry hash, path ================================
    std::printf("identity\n");
    {
        HandoffSplit none;
        none.main_lo = 0;
        none.main_hi = 48;
        HandoffSplit split;
        split.main_lo = 0;
        split.main_hi = 16;
        split.stages = {{16, 34}, {34, 48}};
        check(handoff_split_spec(none) != handoff_split_spec(split), "no split and a 3-way split differ");
        check(handoff_split_spec(split).find('/') == std::string::npos &&
                  handoff_split_spec(split).find(' ') == std::string::npos,
              "the split spec is usable as a path component");
        HandoffSplit other;
        other.main_lo = 0;
        other.main_hi = 18;
        other.stages = {{18, 48}};
        std::array<int64_t, 18> g{};
        g[1] = 48;
        const uint64_t h1 = handoff_geom_hash(g, handoff_split_spec(split));
        check(handoff_geom_hash(g, handoff_split_spec(other)) != h1,
              "two different layer splits produce different geometry hashes (§3.10)");
        check(handoff_geom_hash(g, handoff_split_spec(split)) == h1,
              "the geometry hash is a function of the geometry, not of the call");
        std::array<int64_t, 18> g2 = g;
        g2[11] = 128;
        check(handoff_geom_hash(g2, handoff_split_spec(split)) != h1,
              "a head_dim change changes the geometry hash");
        // The tiling rule: same carves, the draft on a different stage, still a different payload.
        HandoffSplit tilted = split;
        tilted.draft_on_last = false;
        check(handoff_geom_hash(g, handoff_split_spec(tilted)) != h1,
              "where the draft lives is part of the geometry hash");
        const std::string p = handoff_arena_path("/dev/shm/strata-handoff", 0x0123456789abcdefull,
                                                 handoff_split_spec(split), "prefill-a");
        check(p.find("0123456789abcdef-") != std::string::npos &&
                  p.find("/prefill-a/arena") != std::string::npos,
              "the arena path is <dir>/<pack16>-<split>/<instance>/arena (§3.2)");
    }

    // ================================ 2. sizing: machine-independent, and it prints its figure =========
    std::printf("sizing\n");
    {
        const std::vector<HandoffTierSpec> tiers{{1024, 2}, {16384, 2}, {131072, 2}};
        HandoffSizing s;
        std::string err;
        check(handoff_size_arena(tiers, fake_pricer, s, err), "the §3.8 default tier table sizes");
        uint64_t expect = kHandoffHeaderBytes;
        for (size_t i = 0; i < tiers.size(); ++i) {
            const uint64_t priced = (236ull << 20) + tiers[i].max_tokens * 22528ull;
            const uint64_t slot = handoff_round_up_64(priced) + kHandoffSlackBytes;
            check(s.payload_bytes[i] == priced, "the tier payload is what the pricer said, never a RAM guess");
            check(s.slot_bytes[i] == slot, "slot_bytes == round_up_64(payload) + 64 MiB slack");
            check(s.slot_off[i] == expect, "slot_off is the running layout the reader must trust");
            expect += tiers[i].slot_count * (kHandoffHeaderBytes + slot);
        }
        check(s.arena_bytes == expect, "arena_bytes == 4096 + sum(slots x (4096 + slot_bytes))");
        check(s.slot_total == 6, "slot_total is the sum of the per-tier counts");
        check(s.max_tokens == 131072, "max_tokens is the largest tier");
        check(s.describe.find("slots 2,2,2") != std::string::npos &&
                  s.describe.find("tiers 1024,16384,131072") != std::string::npos &&
                  s.describe.find("max 131072 tokens") != std::string::npos &&
                  s.describe.find(std::to_string(s.arena_bytes) + " B") != std::string::npos,
              ("the start-up figure names slots, tiers, max tokens and the byte count: " + s.describe).c_str());
        std::printf("  ..   %s\n", s.describe.c_str());
        check(s.slot_bytes[0] < s.slot_bytes[2],
              "groups are sized PER TIER - the difference between an 8 GiB arena and an 18 GiB one");
        HandoffSizing bad;
        check(!handoff_size_arena({}, fake_pricer, bad, err) && err.find("no tiers") != std::string::npos,
              "an empty tier table is refused");
        check(!handoff_size_arena({{16384, 2}, {1024, 2}}, fake_pricer, bad, err) &&
                  err.find("ascend") != std::string::npos, "tiers must ascend by token count");
        check(!handoff_size_arena({{1024, 0}}, fake_pricer, bad, err) && err.find("0 slots") != std::string::npos,
              "a tier with 0 slots is refused");
        std::vector<HandoffTierSpec> many;
        for (uint64_t i = 0; i < (uint64_t) kHandoffMaxTiers + 1; ++i) many.push_back({1024 * (i + 1), 1});
        check(!handoff_size_arena(many, fake_pricer, bad, err) && err.find("tiers") != std::string::npos,
              "more tiers than the header holds is refused");
        check(!handoff_size_arena({{1024, 2}},
                  [](uint64_t, uint64_t&, std::string& e) { e = "no session"; return false; }, bad, err) &&
                  err.find("no session") != std::string::npos,
              "a pricer failure is reported with its own reason");
    }

    // ================================ 3. the writer creates the arena ================================
    const uint64_t pack_hash = 0x0123456789abcdefull;
    std::array<int64_t, 18> geom_key{};
    geom_key[1] = 48;
    geom_key[11] = 64;
    const std::string split_spec = "main=0..48|n=1|draft=main";
    const uint64_t geom_hash = handoff_geom_hash(geom_key, split_spec);
    const std::string path = arena_path("basic");
    std::printf("create (%s)\n", path.c_str());
    {
        const HandoffSizing s = size_basic();
        HandoffArena a;
        std::string err;
        check(a.open_writer(path, s, pack_hash, geom_hash, err), "the prefill instance creates the arena");
        check(a.valid() && a.is_writer(), "the arena is open for writing");
        check(a.tier_count() == 3 && a.slot_total() == 6, "the header records 3 tiers and 6 slots");
        check(a.arena_bytes() == s.arena_bytes, "the header records the arena size");
        check(a.header().pack_hash == pack_hash && a.header().geom_hash == geom_hash,
              "the header records both hashes");
        check(a.header().created_pid == (uint64_t) (uint32_t) getpid() && a.header().created_start != 0,
              "the header records the creator's pid AND its /proc start time");
        bool tables = true;
        for (uint32_t t = 0; t < 3; ++t) {
            tables = tables && a.tier_max_tokens(t) == s.tiers[t].max_tokens &&
                     a.tier_slot_bytes(t) == s.slot_bytes[t] && a.tier_slot_off(t) == s.slot_off[t] &&
                     a.tier_slot_count(t) == s.tiers[t].slot_count;
        }
        check(tables, "the per-tier tables in the header match the sizing");
        check(a.max_tokens() == 16384,
              "max_tokens() is the LARGEST tier (tiers are stored ascending, so not index 0)");
        struct stat st{};
        check(stat(path.c_str(), &st) == 0 && (uint64_t) st.st_size == s.arena_bytes,
              "the file on disk is exactly arena_bytes");
        check(a.describe().find("slots 2,2,2") != std::string::npos &&
                  a.describe().find(hex(pack_hash)) != std::string::npos &&
                  a.describe().find(hex(geom_hash)) != std::string::npos,
              ("describe() names the slots, tiers and both hashes: " + a.describe()).c_str());
        check(a.free_slots() == "6/6", "a fresh arena has every slot free");
        HandoffSlotInfo info;
        check(a.slot_info(0, info) && info.state == kHandoffFree && info.tier == 0,
              "a fresh slot is FREE and carries its tier");
        check(a.slot_info(4, info) && info.state == kHandoffFree && info.tier == 2 &&
                  info.capacity == s.slot_bytes[2],
              "index 4 is tier group 2's first slot, sized by its tier");
        check(a.slot_info(6, info) == false, "an index past the slot table is refused");
        // A second writer over a LIVE other process's arena is the torn-arena bug: refused, by name.
        // Within ONE process `created_pid == getpid()`, so this must be tested with a foreign live pid in
        // the header - here our own parent, whose /proc start time we can read exactly.
        {
            const uint64_t ppid = (uint64_t) (uint32_t) getppid();
            const uint64_t pstart = handoff_proc_start_time(ppid);
            check(pstart != 0 && handoff_claim_alive(ppid, pstart),
                  "the forged foreign creator (our parent) is a live claim by field 22");
            check(patch(path, kArenaCreatedPidOffset, &ppid, sizeof ppid) &&
                      patch(path, kArenaCreatedStartOffset, &pstart, sizeof pstart),
                  "the header's creator is rewritten to a live foreign pid");
            HandoffArena second;
            check_err(!second.open_writer(path, s, pack_hash, geom_hash, err) &&
                          err.find("one arena per prefill instance") != std::string::npos &&
                          err.find(std::to_string((unsigned long long) ppid)) != std::string::npos,
                      err, "a second writer on a live foreign arena is refused, naming the real owner pid");
            // Put our own pid back so the rest of this block still owns the arena.
            const uint64_t me = (uint64_t) (uint32_t) getpid();
            const uint64_t mstart = handoff_proc_start_time(me);
            check(patch(path, kArenaCreatedPidOffset, &me, sizeof me) &&
                      patch(path, kArenaCreatedStartOffset, &mstart, sizeof mstart),
                  "the header's creator is restored to this process");
        }
        // A different tier table on an existing file is refused with both sizes.
        static const std::vector<HandoffTierSpec> resized_tiers{{1024, 3}, {4096, 2}, {16384, 2}};
        HandoffSizing resized_sizing;
        check(handoff_size_arena(resized_tiers, fake_pricer, resized_sizing, err), "the resized table sizes");
        HandoffArena resized;
        check_err(!resized.open_writer(path, resized_sizing, pack_hash, geom_hash, err) &&
                      err.find(std::to_string(resized_sizing.arena_bytes)) != std::string::npos &&
                      err.find(std::to_string(s.arena_bytes)) != std::string::npos,
                  err, "a tier table that changes the size is refused with BOTH sizes");
        check(!a.open_writer(path, s, 0, geom_hash, err) && err.find("pack hash") != std::string::npos,
              "a zero pack hash is refused before anything is opened");
        check(!a.open_writer(path, s, pack_hash, 0, err) && err.find("geometry hash") != std::string::npos,
              "a zero geometry hash is refused before anything is opened");
    }

    // ================================ 4. happy path: claim -> write -> publish -> read -> release ======
    std::printf("lifecycle\n");
    {
        const HandoffSizing s = size_basic();
        HandoffArena a;
        std::string err;
        check(a.open_writer(path, s, pack_hash, geom_hash, err), "the writer re-opens its own arena");
        HandoffClaim c;
        check(a.claim(0, 0x1111, 77, c, err), "claim takes a slot");
        check(c.index == 0 && c.tier == 0, "the first tier-0 job takes tier group 0's first slot");
        check(c.base == s.slot_off[0] && c.payload_off == c.base + kHandoffHeaderBytes,
              "the payload starts 4096 B into the slot");
        check(c.capacity == s.slot_bytes[0], "the claim's capacity is its tier group's size");
        check(c.nonce != 0, "the claim has a nonce");
        HandoffSlotInfo info;
        check(a.slot_info(0, info) && info.state == kHandoffClaimed, "the slot is CLAIMED after the claim");
        check(info.owner_pid == (uint64_t) (uint32_t) getpid() && info.owner_start != 0,
              "the claim records this pid and its /proc start time");
        check(info.client_id == 0x1111 && info.request_id == 77, "the claim names its reader and request");
        check(info.payload_bytes == 0 && info.published_ms == 0, "a fresh claim has written nothing");
        check(info.pack_hash == pack_hash && info.geom_hash == geom_hash,
              "the claim carries both hashes into the slot header too");

        std::vector<uint8_t> payload(1u << 20);
        for (size_t i = 0; i < payload.size(); ++i) payload[i] = (uint8_t) (i * 31u + 7u);
        check(a.write_payload(c, 0, payload.data(), payload.size(), err), "the payload is written");
        check(a.slot_info(0, info) && info.state == kHandoffClaimed,
              "writing does NOT change the state: still CLAIMED");

        // A reader that stumbles in mid-write must never see a half-written body.
        HandoffArena r;
        check(r.open_reader(path, pack_hash, geom_hash, err), "the reader opens the arena");
        check(!r.is_writer(), "the reader handle is not a writer");
        HandoffPayload p;
        check_err(!r.read_slot(0, c.nonce, 0x1111, p, err) && err.find("CLAIMED") != std::string::npos, err,
                  "an unpublished slot is NEVER readable");
        check(!p.valid(), "a refused read maps nothing");

        const uint64_t ph = fnv1a64(payload.data(), payload.size());
        check(a.publish(c, 4321, payload.size(), ph, err), "publish marks the slot READY");
        check(a.slot_info(0, info) && info.state == kHandoffReady && info.tokens == 4321 &&
                  info.payload_bytes == payload.size() && info.payload_hash == ph && info.seq == 1,
              "the published header carries tokens, bytes, hash and seq=1 (§3.9)");
        check(info.published_ms != 0 && info.published_ms >= info.claimed_ms,
              "published_ms is stamped and never precedes claimed_ms");

        check(r.read_slot(0, c.nonce, 0x1111, p, err), "the named reader can now read it");
        check(p.valid() && p.payload_bytes == payload.size(), "the mapped payload has the right size");
        bool same = p.payload_bytes == payload.size();
        for (size_t i = 0; i < payload.size() && same; ++i) same = p.data[i] == payload[i];
        check(same, "the reader sees the writer's bytes through the file");
        check(HandoffArena::payload_hash_of(p) == ph, "the payload hash matches on both sides");
        check(p.nonce == c.nonce && p.client_id == 0x1111 && p.request_id == 77 && p.tokens == 4321,
              "the payload record carries the claim's identity");
        check(p.tier == 0, "the payload knows which tier group it came from");

        HandoffPayload p2;
        check_err(!r.read_slot(0, c.nonce, 0x2222, p2, err) && err.find("addressed to client") != std::string::npos,
                  err, "a slot addressed to another client is refused");
        check_err(!r.read_slot(0, c.nonce ^ 1, 0x1111, p2, err) && err.find("nonce") != std::string::npos, err,
                  "a nonce that is not the DONE line's is refused");
        check(!r.read_slot(1, c.nonce, 0x1111, p2, err), "reading a different slot with this nonce fails");

        check(a.release(0, c.nonce, err), "the ACK releases the slot");
        check(a.slot_info(0, info) && info.state == kHandoffFree && info.owner_pid == 0 && info.nonce == 0 &&
                  info.client_id == 0 && info.payload_bytes == 0 && info.flags == 0,
              "release clears the claim fields (§3.5 step 5)");
        check(!r.read_slot(0, c.nonce, 0x1111, p2, err) && err.find("FREE") != std::string::npos,
              "after release the slot is not readable");
        check(a.free_slots() == "6/6", "the released slot counts as free again");
        check(!a.release(0, c.nonce, err) && err.find("nonce") != std::string::npos,
              "a stale nonce cannot release a slot");
    }

    // ================================ 4b. the publish ORDER, under a racing reader ================================
    //
    // §3.5 step 3's whole point is that the READY transition IS the publish point: a reader must never see
    // READY together with stale payload metadata.  A fence alone is not observable in a functional test, but
    // the ORDER is - so this hammers read_slot from another thread across 100 publishes and requires every
    // successful read to carry the complete record and a matching hash.  An implementation that wrote
    // `state = READY` first and the metadata second fails here.
    std::printf("publish-order\n");
    {
        const std::string opath = arena_path("publish-order");
        remove_arena(opath);
        const HandoffSizing s = size_single_slot();
        HandoffArena a;
        std::string err;
        check(a.open_writer(opath, s, pack_hash, geom_hash, err), "a one-slot arena for the race");
        int bad = 0, ready_seen = 0, rounds_ok = 0;
        constexpr int kRounds = 100;
        constexpr uint64_t kBytes = 4096;
        std::vector<uint8_t> payload(kBytes);
        for (uint64_t round = 0; round < kRounds; ++round) {
            for (uint64_t i = 0; i < kBytes; ++i) payload[i] = (uint8_t) (i + round * 3u);
            const uint64_t want = fnv1a64(payload.data(), payload.size());
            HandoffClaim c;
            if (!a.claim(0, 0x5555, round, c, err)) break;
            if (!a.write_payload(c, 0, payload.data(), kBytes, err)) break;
            std::atomic<bool> stop{false};
            std::atomic<uint64_t> hits{0};
            std::thread reader([&]() {
                HandoffArena r2;
                std::string e2;
                if (!r2.open_reader(opath, pack_hash, geom_hash, e2)) { ++bad; return; }
                while (!stop.load()) {
                    HandoffPayload pp;
                    if (r2.read_slot(c.index, c.nonce, 0x5555, pp, e2)) {
                        ++hits;
                        if (pp.payload_bytes != kBytes || pp.tokens != round + 1 ||
                            pp.payload_hash != want || HandoffArena::payload_hash_of(pp) != want) {
                            ++bad;
                        }
                    }
                }
            });
            const bool pub = a.publish(c, round + 1, kBytes, want, err);
            std::this_thread::sleep_for(std::chrono::microseconds(300));
            stop = true;
            reader.join();
            if (pub) ++rounds_ok;
            ready_seen += (int) hits.load();
            (void) a.release(c.index, c.nonce, err);
        }
        check(rounds_ok == kRounds, "every round published");
        check(ready_seen > kRounds,
              ("the racing reader actually caught READY slots (" + std::to_string(ready_seen) +
               " successful reads over " + std::to_string(kRounds) + " publishes)").c_str());
        check(bad == 0, "no read ever saw READY with stale or partial payload metadata");
        remove_arena(opath);
    }

    // ================================ 5. per-tier claim preference and starvation ================================
    std::printf("tiers\n");
    {
        const HandoffSizing s = size_basic();
        HandoffArena a;
        std::string err;
        check(a.open_writer(path, s, pack_hash, geom_hash, err), "writer open");
        check(a.free_slots() == "6/6", "the arena starts empty");
        // (a) A big job may NOT be squeezed into a smaller group: the tier-2 group is the only one it can
        // use, and once it is full the job waits even though four small slots are free.
        HandoffClaim b1, b2;
        check(a.claim(2, 1, 1, b1, err) && b1.tier == 2 && b1.index == 4,
              "a tier-2 job takes tier group 2's first slot");
        check(a.claim(2, 1, 2, b2, err) && b2.tier == 2 && b2.index == 5,
              "and the second tier-2 job takes its second slot");
        HandoffClaim nobig;
        check_err(!a.claim(2, 1, 3, nobig, err) && err.find("no free handoff slot") != std::string::npos, err,
                  "a tier-2 job is refused when only SMALLER groups have room");
        check(a.free_slots() == "4/6", "the free-slot count reflects the two big claims");
        check(nobig.index == -1 && nobig.nonce == 0, "a refused claim hands back nothing to write into");
        // (b) A small job takes the SMALLEST group that fits, so it never sits in a 3 GB slot while a big
        // job waits (§3.8: "this is a decision, not a knob").
        HandoffClaim c0, c1, c2, c3;
        check(a.claim(0, 1, 4, c0, err) && c0.tier == 0 && c0.index == 0, "a tier-0 job takes group 0");
        check(a.claim(0, 1, 5, c1, err) && c1.tier == 0 && c1.index == 1,
              "the second tier-0 job takes group 0's second slot");
        check(a.claim(0, 1, 6, c2, err) && c2.tier == 1 && c2.index == 2,
              "with group 0 full, a tier-0 job cascades to the SMALLEST group that fits");
        check(a.claim(0, 1, 7, c3, err) && c3.tier == 1 && c3.index == 3, "and then group 1's second slot");
        check(c0.capacity == s.slot_bytes[0] && c2.capacity == s.slot_bytes[1],
              "a claim's capacity comes from the group it actually landed in");
        HandoffClaim none;
        check(!a.claim(0, 1, 8, none, err), "with every group full, even a tier-0 job is refused");
        check(err.find("0/6") != std::string::npos,
              "the full-arena refusal counts the free slots (temporary: QUEUED, never an ERR on the wire)");
        bool released = true;
        for (const HandoffClaim* c : {&c0, &c1, &c2, &c3, &b1, &b2}) {
            released = released && a.release(c->index, c->nonce, err);
        }
        check(released, "every claim releases by its own nonce");
        check(a.free_slots() == "6/6", "releasing every slot empties the arena");
        check(!a.claim(3, 1, 9, none, err) && err.find("not in this arena") != std::string::npos,
              "a tier index past the table is refused by name");
    }

    // ================================ 6. ownership: publish/release refuse what is not theirs ==========
    std::printf("ownership\n");
    {
        const HandoffSizing s = size_basic();
        HandoffArena a;
        std::string err;
        check(a.open_writer(path, s, pack_hash, geom_hash, err), "writer open");
        HandoffClaim c;
        check(a.claim(0, 1, 10, c, err), "claim");
        check(a.cancel(c.index, c.nonce, err), "cancel marks the slot (§4.6)");
        check_err(!a.publish(c, 1, 16, 1, err) && err.find("cancelled") != std::string::npos, err,
                  "a cancelled job cannot publish");
        check(a.abort_claim(c.index, c.nonce, err), "abort_claim frees it and flags it partial");
        HandoffSlotInfo info;
        check(a.slot_info(c.index, info) && info.state == kHandoffFree &&
                  (info.flags & kHandoffFlagPartial) != 0,
              "the aborted slot is FREE with the partial-payload flag");
        HandoffClaim d;
        check(a.claim(0, 1, 11, d, err), "a new claim");
        HandoffClaim fake = d;
        fake.nonce ^= 0xdeadbeef;
        check_err(!a.publish(fake, 1, 16, 1, err) && err.find("claim is gone") != std::string::npos, err,
                  "a stale nonce cannot publish");
        check(!a.cancel(fake.index, fake.nonce, err), "a stale nonce cannot cancel");
        check(!a.release(fake.index, fake.nonce, err), "a stale nonce cannot release");
        std::vector<uint8_t> blob(4096, 0x5a);
        check_err(!a.write_payload(d, d.capacity - 1024, blob.data(), blob.size(), err) &&
                      err.find(std::to_string(d.capacity)) != std::string::npos &&
                      err.find(std::to_string(d.capacity - 1024 + blob.size())) != std::string::npos,
                  err, "a write past the slot capacity is refused with BOTH numbers");
        check(!a.publish(d, 1, d.capacity + 1, 1, err) &&
                  err.find(std::to_string(d.capacity)) != std::string::npos,
              "publishing more than the slot holds is refused with the capacity");
        (void) a.abort_claim(d.index, d.nonce, err);
        // A reader handle never writes.
        HandoffArena r;
        check(r.open_reader(path, pack_hash, geom_hash, err), "reader open");
        HandoffClaim rc;
        check(!r.claim(0, 1, 12, rc, err) && err.find("reader never claims") != std::string::npos,
              "a reader cannot claim");
        check(!r.write_payload(rc, 0, blob.data(), blob.size(), err), "a reader cannot write a payload");
        check(!r.publish(rc, 1, 16, 1, err), "a reader cannot publish");
        check(!r.release(0, 1, err), "a reader cannot release");
        check(!r.abort_claim(0, 1, err), "a reader cannot abort a claim");
        HandoffPayload rp;
        check(!r.read_slot(0, 1, 1, rp, err) && err.find("not READY") != std::string::npos,
              "a reader reading a FREE slot is refused");
        // A writer handle cannot read a slot (that would let the writer skip its own publish rule).
        HandoffPayload wp;
        check(!a.read_slot(0, 1, 1, wp, err) && err.find("as a reader") != std::string::npos,
              "a writer handle cannot read a slot");
    }

    // ================================ 7. a dead owner's CLAIMED slot is taken over ================================
    std::printf("dead-owner\n");
    {
        const HandoffSizing s = size_basic();
        HandoffArena a;
        std::string err;
        check(a.open_writer(path, s, pack_hash, geom_hash, err), "writer open");
        const uint64_t base = slot_base_of(s, 0);
        forge_claim(path, base, 0, dead_pid(), 0, 0x1234, 0x9999, handoff_now_ms());
        check(a.free_slots() == "5/6", "a forged dead claim is not free");
        const auto took = a.reclaim(&err);
        check(took.size() == 1 && took[0].index == 0, "reclaim takes the dead owner's slot");
        check(!took.empty() && took[0].why.find("gone") != std::string::npos,
              ("the reclaim says why: " + (took.empty() ? std::string("?") : took[0].why)).c_str());
        check(!took.empty() && took[0].was_state == kHandoffClaimed, "and says what state it found");
        HandoffSlotInfo info;
        check(a.slot_info(0, info) && info.state == kHandoffFree &&
                  (info.flags & kHandoffFlagAbandoned) != 0,
              "the reclaimed slot is FREE with the abandoned flag (§3.6)");
        check(a.free_slots() == "6/6", "and counts as free");
        // A live claim is neither reclaimed nor stolen.
        HandoffClaim live;
        check(a.claim(0, 1, 20, live, err), "a live claim");
        check(a.reclaim(&err).empty(), "a live owner's claim is not reclaimed");
        HandoffClaim steal;
        check(!a.claim(0, 1, 21, steal, err) || steal.index != live.index,
              "a second claim never steals a live claim");
        // A dead CLAIMED slot is also directly reclaimable through claim(), without an explicit reclaim().
        const uint64_t base1 = slot_base_of(s, 1);
        forge_claim(path, base1, 0, dead_pid(), 0, 0x1235, 0x9998, handoff_now_ms());
        HandoffClaim over;
        check(a.claim(0, 1, 22, over, err) && over.index == 1,
              "claim() itself takes over a dead owner's slot");
        check(over.nonce != 0x1235, "and the take-over gets a fresh nonce");
        (void) a.release(over.index, over.nonce, err);
        (void) a.release(live.index, live.nonce, err);
    }

    // ================================ 8. a pid-RECYCLED owner is detected (field 22) ================================
    std::printf("recycled-pid\n");
    {
        const HandoffSizing s = size_basic();
        HandoffArena a;
        std::string err;
        check(a.open_writer(path, s, pack_hash, geom_hash, err), "writer open");
        const uint64_t base = slot_base_of(s, 1);
        const uint64_t me = (uint64_t) (uint32_t) getpid();
        // Our own pid with a start time that is NOT ours: `/proc/<pid>` exists, so a pid-only check says
        // "alive" and the slot leaks forever.  Field 22 is what makes the take-over exact.
        forge_claim(path, base, 0, me, handoff_proc_start_time(me) ^ 0xdeadbeefull, 0x4321, 0x8888,
                    handoff_now_ms());
        check(handoff_pid_alive(me), "our own pid IS alive, so the pid alone says the claim is fine");
        check(handoff_proc_start_time(me) != 0, "and /proc really does give us field 22");
        check(!handoff_claim_alive(me, 12345),
              "but the recorded start time differs, so the claim is NOT alive (§3.6)");
        const auto took = a.reclaim(&err);
        check(took.size() == 1 && took[0].index == 1, "the recycled-pid claim is reclaimed, not waited for");
        check(!took.empty() && took[0].why.find("gone") != std::string::npos, "and the reason names the owner");
        // A claim with start_time 0 is treated as ALIVE: unknown must never guess "dead".
        forge_claim(path, base, 0, me, 0, 0x4322, 0x8888, handoff_now_ms());
        check(a.reclaim(&err).empty(), "an unknown start time is treated as a match, never as dead");
        check(handoff_claim_alive(me, 0), "and the helper agrees");
        (void) a.abort_claim(1, 0x4322, err);
        check(handoff_pid_alive(0) == false && handoff_proc_start_time(0) == 0,
              "pid 0 is nobody, and nobody is not alive");
    }

    // ================================ 9. leases: a READY slot nobody read is reclaimed ==========================
    std::printf("leases\n");
    {
        const HandoffSizing s = size_basic();
        HandoffArena a;
        std::string err;
        check(a.open_writer(path, s, pack_hash, geom_hash, err), "writer open");
        a.set_lease_ms(120);
        check(a.lease_ms() == 120, "the lease is settable (STRATA_HANDOFF_LEASE_MS in production)");
        std::vector<uint8_t> payload(8192, 0x3c);
        HandoffClaim c;
        check(a.claim(1, 0xAAAA, 30, c, err), "claim tier 1");
        check(a.write_payload(c, 0, payload.data(), payload.size(), err), "write");
        check(a.publish(c, 99, payload.size(), fnv1a64(payload.data(), payload.size()), err), "publish");
        check(a.reclaim(&err).empty(), "a fresh READY slot is NOT reclaimed inside its lease");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const auto took = a.reclaim(&err);
        check(took.size() == 1 && took[0].was_state == kHandoffReady,
              "a READY slot nobody read IS reclaimed when its lease expires (the slot-leak fix)");
        check(!took.empty() && took[0].why.find("lease") != std::string::npos,
              ("the reclaim says the lease expired: " + (took.empty() ? std::string("?") : took[0].why)).c_str());
        HandoffSlotInfo info;
        check(a.slot_info(c.index, info) && info.state == kHandoffFree &&
                  (info.flags & kHandoffFlagAbandoned) != 0,
              "the expired READY slot is FREE and flagged abandoned");
        // A CLAIMED slot whose writer hung is reclaimed by the same bound.
        HandoffClaim stuck;
        check(a.claim(1, 0xAAAA, 31, stuck, err), "claim again");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const auto took2 = a.reclaim(&err);
        check(took2.size() == 1 && took2[0].was_state == kHandoffClaimed,
              "a CLAIMED slot is reclaimed when its lease expires");
        // A READY slot whose READER died is reclaimed at once, without waiting for the lease.
        a.set_lease_ms(600000);
        const uint64_t gone = dead_pid();
        a.note_client_pid(0xBBBB, gone);
        HandoffClaim d;
        check(a.claim(1, 0xBBBB, 32, d, err), "claim for a client whose pid we watch");
        check(a.write_payload(d, 0, payload.data(), payload.size(), err), "write");
        check(a.publish(d, 99, payload.size(), fnv1a64(payload.data(), payload.size()), err), "publish");
        const auto took3 = a.reclaim(&err);
        check(took3.size() == 1 && took3[0].was_state == kHandoffReady &&
                  took3[0].why.find("reader") != std::string::npos,
              ("a READY slot whose reader pid is gone is reclaimed immediately: " +
               (took3.empty() ? std::string("?") : took3[0].why)).c_str());
        // A live reader's READY slot survives.
        a.note_client_pid(0xCCCC, (uint64_t) (uint32_t) getpid());
        HandoffClaim e;
        check(a.claim(1, 0xCCCC, 33, e, err), "claim for a live client");
        check(a.write_payload(e, 0, payload.data(), payload.size(), err), "write");
        check(a.publish(e, 99, payload.size(), fnv1a64(payload.data(), payload.size()), err), "publish");
        check(a.reclaim(&err).empty(), "a READY slot with a live reader and an unexpired lease survives");
        (void) a.release(e.index, e.nonce, err);
        // And reclaim() is safe to call on a closed arena.
        HandoffArena closed;
        std::string cerr;
        check(closed.reclaim(&cerr).empty() && !cerr.empty(), "reclaim on a closed arena reports, not crashes");
    }

    // ================================ 10. hash / version / magic / size refusal ================================
    std::printf("refusal\n");
    {
        const HandoffSizing s = size_basic();
        std::string err;
        HandoffArena r;
        check_err(!r.open_reader(path, pack_hash ^ 0x5555555555555555ull, geom_hash, err) &&
                      err.find("pack hash") != std::string::npos &&
                      err.find(hex(pack_hash)) != std::string::npos &&
                      err.find(hex(pack_hash ^ 0x5555555555555555ull)) != std::string::npos,
                  err, "a reader's pack mismatch names BOTH hashes");
        check_err(!r.open_reader(path, pack_hash, geom_hash ^ 7, err) &&
                      err.find("geometry hash") != std::string::npos && err.find(hex(geom_hash)) != std::string::npos,
                  err, "a reader's geometry mismatch names BOTH hashes");
        check(!r.open_reader(root_dir() + "/nope/arena", pack_hash, geom_hash, err) &&
                  err.find("cannot open") != std::string::npos, "a missing arena is refused by name");
        HandoffArena w;
        check_err(!w.open_writer(path, size_basic(), pack_hash ^ 0x1ull, geom_hash, err) &&
                      err.find("pack hash") != std::string::npos && err.find(hex(pack_hash)) != std::string::npos,
                  err, "the writer refuses a pack mismatch with BOTH hashes too");
        check_err(!w.open_writer(path, size_basic(), pack_hash, geom_hash ^ 0x2ull, err) &&
                      err.find("geometry hash") != std::string::npos, err,
                  "the writer refuses a geometry mismatch too");
        check_err(!r.open_reader(path, pack_hash, geom_hash ^ 0x40ull, err) &&
                      err.find("geometry hash") != std::string::npos, err,
                  "a reader handle re-checks the geometry it was given on every open");
        // A file whose size disagrees with its own header: the reader must refuse rather than map a hole and
        // take a SIGBUS halfway through a copy-out.  (The arena is not truncated here - that would be a
        // write to somebody else's file; the HEADER's arena_bytes is forged instead, which is the same
        // disagreement and is what a corrupt or half-written header produces.)
        {
            const uint64_t wrong = s.arena_bytes - (1ull << 20);
            check(patch(path, kArenaBytesOffset, &wrong, sizeof wrong), "the header's arena_bytes is forged low");
            std::string serr;
            const bool refused = !r.open_reader(path, pack_hash, geom_hash, serr);
            check_err(refused && serr.find("but its header says") != std::string::npos, serr,
                      "an arena whose file size disagrees with its header is refused by both numbers");
            check(patch(path, kArenaBytesOffset, &s.arena_bytes, sizeof s.arena_bytes),
                  "the header's arena_bytes is put back");
            std::string okerr;
            check(r.open_reader(path, pack_hash, geom_hash, okerr), "and the arena reads again once it agrees");
        }
        // A HandoffSizing that does not add up is refused before a single byte is written.
        {
            HandoffSizing broken = s;
            broken.arena_bytes += 4096;
            std::string berr;
            HandoffArena bw;
            check_err(!bw.open_writer(arena_path("broken-sizing"), broken, pack_hash, geom_hash, berr) &&
                          berr.find("does not add up") != std::string::npos,
                      berr, "a sizing whose arena_bytes disagrees with its own layout is refused");
            HandoffSizing shifted = s;
            shifted.slot_off[1] += kHandoffHeaderBytes;
            std::string serr2;
            check_err(!bw.open_writer(arena_path("broken-sizing"), shifted, pack_hash, geom_hash, serr2) &&
                          serr2.find("the layout says") != std::string::npos,
                      serr2, "a sizing whose tier offsets disagree with the layout is refused");
            check(!bw.valid(), "a refused sizing leaves no arena open");
            remove_arena(arena_path("broken-sizing"));
        }
        const std::string vpath = arena_path("version");
        {
            HandoffArena w2;
            check(w2.open_writer(vpath, s, pack_hash, geom_hash, err), "a second arena for the header cases");
            HandoffArena rr;
            const uint32_t v2 = 2;
            check(patch(vpath, kArenaVersionOffset, &v2, sizeof v2), "the version is rewritten to 2");
            std::string verr;
            const bool vrefused = !rr.open_reader(vpath, pack_hash, geom_hash, verr);
            check(vrefused && verr.find("version") != std::string::npos,
                  ("a future protocol version is refused: " + verr).c_str());
            const uint32_t v1 = kHandoffVersion;
            check(patch(vpath, kArenaVersionOffset, &v1, sizeof v1), "the version is put back to 1");
            const uint32_t hb = 2048;
            check(patch(vpath, kArenaHeaderBytesOffset, &hb, sizeof hb), "header_bytes rewritten to 2048");
            std::string herr;
            const bool hrefused = !rr.open_reader(vpath, pack_hash, geom_hash, herr);
            check(hrefused && herr.find("header_bytes") != std::string::npos,
                  ("a header_bytes that is not 4096 is refused: " + herr).c_str());
            const char bad[16] = "NOT-AN-ARENA!!!";
            check(patch(vpath, 0, bad, 16), "magic rewritten");
            std::string merr;
            const bool mrefused = !rr.open_reader(vpath, pack_hash, geom_hash, merr);
            check(mrefused && merr.find("magic") != std::string::npos,
                  ("a foreign file is refused on magic before anything else: " + merr).c_str());
            std::string werr;
            const bool wrefused = !w2.open_writer(vpath, s, pack_hash, geom_hash, werr);
            check(wrefused && werr.find("magic") != std::string::npos,
                  ("the writer refuses a corrupt header too, rather than reusing the file: " + werr).c_str());
            remove_arena(vpath);
        }
    }

    // ================================ 10b. a slot's OWN identity is checked, not just the arena's ====
    std::printf("slot-identity\n");
    {
        const HandoffSizing s = size_basic();
        HandoffArena a;
        std::string err;
        check(a.open_writer(path, s, pack_hash, geom_hash, err), "writer open");
        HandoffClaim c;
        check(a.claim(0, 0x7777, 808, c, err), "claim and publish a real payload");
        std::vector<uint8_t> payload(4096, 0x2b);
        check(a.write_payload(c, 0, payload.data(), payload.size(), err), "write");
        check(a.publish(c, 55, payload.size(), fnv1a64(payload.data(), payload.size()), err), "publish");
        HandoffArena r;
        check(r.open_reader(path, pack_hash, geom_hash, err), "reader open");
        HandoffPayload p;
        check(r.read_slot(c.index, c.nonce, 0x7777, p, err), "the honest slot reads");
        p = HandoffPayload{};
        // A slot header naming a different pack than the arena: the payload was produced by another model.
        forge_ready(path, c.base, 0, c.nonce, 0x7777, pack_hash ^ 0x99ull, geom_hash, payload.size(), 1);
        check_err(!r.read_slot(c.index, c.nonce, 0x7777, p, err) && err.find("was written for pack") != std::string::npos,
                  err, "a slot whose own pack hash differs from the arena's is refused");
        // A slot header from a different layer split.
        forge_ready(path, c.base, 0, c.nonce, 0x7777, pack_hash, geom_hash ^ 0x99ull, payload.size(), 1);
        check_err(!r.read_slot(c.index, c.nonce, 0x7777, p, err) && err.find("geometry") != std::string::npos,
                  err, "a slot whose own geometry hash differs is refused");
        // §3.9: v1 is one window.  A seq of 2 is a second format, and mounting it would read the wrong bytes.
        forge_ready(path, c.base, 0, c.nonce, 0x7777, pack_hash, geom_hash, payload.size(), 2);
        check_err(!r.read_slot(c.index, c.nonce, 0x7777, p, err) && err.find("seq") != std::string::npos, err,
                  "a seq other than 1 is refused (§3.9 windows, not streams)");
        // A payload_bytes past the tier group's capacity would map and hash past the slot.
        forge_ready(path, c.base, 0, c.nonce, 0x7777, pack_hash, geom_hash, c.capacity + 1, 1);
        check_err(!r.read_slot(c.index, c.nonce, 0x7777, p, err) && err.find("in a") != std::string::npos, err,
                  "a payload larger than its own slot is refused before anything is mapped");
        check(!p.valid(), "none of those refusals mapped a byte");
        // A slot header that is not a slot header at all.
        const char junk[16] = "GARBAGE-SLOT!!!";
        check(patch(path, c.base, junk, 16), "the slot magic is overwritten");
        check_err(!r.read_slot(c.index, c.nonce, 0x7777, p, err) && err.find("incompatible header") != std::string::npos,
                  err, "a slot with the wrong magic is refused");
        (void) a.release(c.index, c.nonce, err);
    }

    // ================================ 12. statvfs: a filesystem too small refuses the start =====================
    std::printf("statvfs\n");
    {
        // 400+ GiB: no filesystem here holds it (/dev/shm has ~24 GiB free).  The refusal must come BEFORE
        // ftruncate - a sparse ftruncate on tmpfs succeeds and the writer then dies of SIGBUS with no
        // message - it must name BOTH numbers, and it must leave NO FILE BEHIND.
        const std::string huge_path = arena_path("too-big");
        std::error_code ec;
        remove_arena(huge_path);
        const std::vector<HandoffTierSpec> tiers{{131072, 4}};
        const std::vector<uint64_t> priced(1, 100ull << 30);   // 4 slots x 100 GiB
        HandoffSizing huge;
        std::string err;
        check(handoff_size_arena(tiers, priced, huge, err), "a deliberately enormous arena sizes on paper");
        check(huge.arena_bytes > 400ull << 30, "and really is over 400 GiB");
        HandoffArena a;
        check(!a.open_writer(huge_path, huge, pack_hash, geom_hash, err),
              "an arena larger than the filesystem refuses to start");
        check(err.find(std::to_string(huge.arena_bytes)) != std::string::npos,
              "the refusal names the bytes the arena needs");
        check(err.find("free") != std::string::npos,
              ("the refusal names the filesystem's free space: " + err).c_str());
        check(!std::filesystem::exists(huge_path, ec), "a refused arena left a file behind");
        check(!a.valid(), "the refused arena is not open");
        // The same path at a sane size still works afterwards.
        HandoffArena b;
        check(b.open_writer(huge_path, size_basic(), pack_hash, geom_hash, err),
              "the same path works at a sane size");
        remove_arena(huge_path);
    }

    // ================================ 12. across PROCESSES (the --role re-exec) ================================
    std::printf("processes\n");
    {
        const std::string ppath = arena_path("processes");
        std::error_code ec;
        remove_arena(ppath);
        HandoffArena a;
        std::string err;
        check(a.open_writer(ppath, size_one_per_tier(), pack_hash, geom_hash, err),
              "the parent creates a one-slot-per-group arena");
        const std::string self = self_path();
        check(!self.empty(), "own executable path readable");
        if (!self.empty()) {
            // (a) Does `flock` contend at all here?  It must: the whole single-writer guarantee rests on it.
            {
                const int f1 = ::open(ppath.c_str(), O_RDWR | O_CLOEXEC);
                const int f2 = ::open(ppath.c_str(), O_RDWR | O_CLOEXEC);
                check(f1 >= 0 && f2 >= 0, "two fds on the same arena file");
                check(::flock(f1, LOCK_EX) == 0, "the first fd takes the exclusive lock");
                const int res = ::flock(f2, LOCK_EX | LOCK_NB);
                check(res != 0 && (errno == EWOULDBLOCK || errno == EAGAIN),
                      "a second fd is refused the lock (flock contends, as the protocol assumes)");
                // A SHARED lock is refused too while an exclusive one is held - that is what stops a reader
                // reading a header mid-decision.
                const int sh = ::flock(f2, LOCK_SH | LOCK_NB);
                check(sh != 0 && (errno == EWOULDBLOCK || errno == EAGAIN),
                      "a shared lock is refused while the writer holds the exclusive one");
                ::flock(f1, LOCK_UN);
                ::close(f1);
                ::close(f2);
            }
            // (b) Two processes race for the ONLY slot in an arena that has exactly one.  Exactly one may
            // win.  This arena has ONE group with ONE slot on purpose: §3.8 lets a tier-0 job cascade into
            // a larger group, so with three groups both racers could legitimately get a slot and the case
            // would prove nothing.
            {
                const std::string cpath = arena_path("contention");
                remove_arena(cpath);
                HandoffArena owner;
                check(owner.open_writer(cpath, size_single_slot(), pack_hash, geom_hash, err),
                      "a one-slot arena for the contention case");
                const std::string base = self + " --role writer " + cpath + " " + hex(pack_hash) + " " +
                                         hex(geom_hash) + " 0 4242 ";
                const std::string c1 = base + "101 300000 single";
                const std::string c2 = base + "102 300000 single";
                const std::string o1 = run(c1);
                const std::string o2 = run(c2);
                const bool w1 = o1.find("CLAIM ok=1") != std::string::npos;
                const bool w2 = o2.find("CLAIM ok=1") != std::string::npos;
                check(o1.find("OPEN ok=1") != std::string::npos && o2.find("OPEN ok=1") != std::string::npos,
                      "both processes opened the existing arena as writers");
                check(w1 != w2, ("exactly one of two processes got the only slot: [" + o1 + "] [" + o2 + "]").c_str());
                const std::string loser = w1 ? o2 : o1;
                check(loser.find("no free handoff slot") != std::string::npos,
                      "the loser's answer is the temporary full-arena reason, not a crash or a stolen slot");
                const long long slot = field(w1 ? o1 : o2, "slot");
                check(slot == 0, "the winner took slot 0, the arena's only slot");
                remove_arena(cpath);
            }
            // (c) A separate process publishes; the parent reads it back through the FILE.
            {
                const std::string out = run(self + " --role writer " + ppath + " " + hex(pack_hash) + " " +
                                            hex(geom_hash) + " 1 777 555 300000 per-tier");
                check(out.find("CLAIM ok=1") != std::string::npos && out.find("PUBLISH ok=1") != std::string::npos,
                      "a separate process claimed tier group 1 and published it");
                const long long slot1 = field(out, "slot");
                const uint64_t nonce = std::strtoull(hexfield(out, "nonce").c_str(), nullptr, 16);
                const std::string child_hash = hexfield(out, "hash");
                check(slot1 >= 0 && nonce != 0, "the child reported its slot and nonce");
                HandoffArena r;
                check(r.open_reader(ppath, pack_hash, geom_hash, err), "the parent opens it as a reader");
                HandoffPayload p;
                check(r.read_slot((int) slot1, nonce, 777, p, err),
                      "the parent reads a slot a DIFFERENT process wrote and published");
                check(p.valid() && p.payload_bytes == kRolePayload && p.tokens == 1234,
                      "the payload crossed the process boundary intact");
                check(hex(HandoffArena::payload_hash_of(p)) == child_hash,
                      "and its FNV-1a matches the writer's, computed in another address space");
                bool pattern = p.payload_bytes == kRolePayload;
                for (uint64_t i = 0; i < p.payload_bytes && pattern; ++i) {
                    pattern = p.data[i] == role_byte(i, p.request_id);
                }
                check(pattern, "every byte of the child's payload is what the child wrote");
                // (d) A separate READER process, and its two refusal arms.
                const std::string rcmd = self + " --role reader " + ppath + " " + hex(pack_hash) + " " +
                                         hex(geom_hash) + " " + std::to_string(slot1) + " 777 0 300000 per-tier " +
                                         hex(nonce);
                const std::string rout = run(rcmd);
                check(rout.find("READ ok=1") != std::string::npos &&
                          rout.find("PAYLOAD ok=1") != std::string::npos,
                      ("a separate reader process sees the bytes and the pattern: " + rout).c_str());
                const std::string badn = self + " --role reader-wrong-nonce " + ppath + " " + hex(pack_hash) +
                                         " " + hex(geom_hash) + " " + std::to_string(slot1) + " 777 0 300000 "
                                         "per-tier " + hex(nonce);
                check(run(badn).find("READ ok=0") != std::string::npos,
                      "a separate reader with the wrong nonce is refused");
                const std::string badc = self + " --role reader-wrong-client " + ppath + " " + hex(pack_hash) +
                                         " " + hex(geom_hash) + " " + std::to_string(slot1) + " 777 0 300000 "
                                         "per-tier " + hex(nonce);
                check(run(badc).find("READ ok=0") != std::string::npos,
                      "a separate reader addressed to another client is refused");
                check(a.release((int) slot1, nonce, err), "the writer releases the child's slot on ACK");
                // (e) A live other process's CLAIMED slot is NOT reclaimed by this one.
                const std::string hold = self + " --role writer-hold " + ppath + " " + hex(pack_hash) + " " +
                                         hex(geom_hash) + " 2 4242 103 600000 per-tier";
                FILE* hp = popen(hold.c_str(), "r");
                std::string hout;
                if (hp) {
                    char line[2048];
                    // Read the CLAIM line, then let the child sleep with the slot held.
                    while (std::fgets(line, sizeof line, hp)) {
                        hout += line;
                        if (hout.find("CLAIM ") != std::string::npos) break;
                    }
                    check(hout.find("CLAIM ok=1") != std::string::npos,
                          "a second process holds a slot CLAIMED and stays alive");
                    const long long hs = field(hout, "slot");
                    const auto none = a.reclaim(&err);
                    check(none.empty(), "a live other process's claim is NOT reclaimed by us");
                    HandoffSlotInfo si;
                    check(a.slot_info((int) hs, si) && si.state == kHandoffClaimed,
                          "and it is still CLAIMED after our reclaim pass");
                    int status = 0;
                    pclose(hp);
                    (void) status;
                } else {
                    check(false, "could not start the slot-holding child");
                }
                // (f) A claim waits for a foreign flock instead of barging in.
                {
                    const std::string hk = self + " --role flock-hold " + ppath + " " + hex(pack_hash) + " " +
                                           hex(geom_hash) + " 0 1 1 300000 per-tier";
                    FILE* fp = popen(hk.c_str(), "r");
                    std::string hold_out;
                    if (fp) {
                        char line[2048];
                        while (std::fgets(line, sizeof line, fp)) {
                            hold_out += line;
                            if (hold_out.find("HOLD locked=1") != std::string::npos) break;
                        }
                        check(hold_out.find("HOLD locked=1") != std::string::npos,
                              "a foreign process holds the arena's exclusive lock");
                        const uint64_t t0 = handoff_now_ms();
                        HandoffClaim waited;
                        const bool got = a.claim(0, 1, 104, waited, err);
                        const uint64_t waited_ms = handoff_now_ms() - t0;
                        check(got, "the claim eventually got the slot once the foreign lock was dropped");
                        check(waited_ms >= 200,
                              ("the claim really waited for the foreign flock (" + std::to_string(waited_ms) +
                               " ms) instead of acting on an unreadable table").c_str());
                        (void) a.release(waited.index, waited.nonce, err);
                        int status = 0;
                        pclose(fp);
                        (void) status;
                    } else {
                        check(false, "could not start the flock-holding child");
                    }
                }
            }
        }
        remove_arena(ppath);
    }

    // ================================ 13. a >1 GiB slot, written and read for real ================================
    std::printf("big-slot\n");
    {
        struct statvfs sv{};
        uint64_t avail = 0;
        if (statvfs(root_dir().c_str(), &sv) == 0) {
            const uint64_t block = sv.f_frsize ? sv.f_frsize : sv.f_bsize;
            avail = (uint64_t) sv.f_bavail * block;
        }
        const uint64_t slot_bytes = (1152ull << 20) + kHandoffSlackBytes;
        const uint64_t need = kHandoffHeaderBytes + 2 * (kHandoffHeaderBytes + slot_bytes);
        if (avail < need + (64ull << 20)) {
            std::printf("  skip the >1 GiB case: %s has %llu MiB free and it wants %llu MiB\n",
                        root_dir().c_str(), (unsigned long long) (avail >> 20),
                        (unsigned long long) (need >> 20));
            check(avail > 0, "the >1 GiB case was skipped because statvfs reported the free space");
        } else {
            const std::string bpath = arena_path("big");
            const std::vector<HandoffTierSpec> tiers{{1024, 1}, {1u << 20, 2}};
            // 1 152 MiB priced payload -> a 1216 MiB slot: over 1 GiB, and sized by the TIER TABLE, not by
            // this machine's RAM.  Two of them, because a real arena has more than one slot per group.
            const std::vector<uint64_t> priced{(1u << 20), (1152ull << 20)};
            HandoffSizing s;
            std::string err;
            check(handoff_size_arena(tiers, priced, s, err), "a >1 GiB tier sizes");
            check(s.slot_bytes[1] == slot_bytes, "the big tier's slot is 1216 MiB");
            HandoffArena a;
            check(a.open_writer(bpath, s, pack_hash, geom_hash, err),
                  "the >1 GiB arena is created (statvfs said it fits)");
            HandoffClaim c;
            check(a.claim(1, 0xBEEF, 900, c, err) && c.tier == 1, "the big job claims the big group");
            check(c.capacity == slot_bytes, "and its capacity is the big tier's");
            // A 4 MiB buffer, reused: the write path is exercised at a realistic size without allocating
            // 1 GiB of anonymous RAM.
            const uint64_t chunk = 4ull << 20;
            std::vector<uint8_t> buf(chunk);
            for (uint64_t i = 0; i < chunk; ++i) buf[i] = (uint8_t) (i * 131u + (i >> 12));
            const uint64_t total = (1ull << 30) + 1;   // 1 GiB + 1 B, so the tail is not a round number
            bool wrote = true;
            uint64_t hash = 1469598103934665603ull;
            for (uint64_t off = 0; off < total && wrote; off += chunk) {
                const uint64_t n = std::min(chunk, total - off);
                wrote = a.write_payload(c, off, buf.data(), n, err);
                hash = fnv1a64(buf.data(), n, hash);
            }
            check(wrote, "1 GiB + 1 B written through the pwrite path");
            check(a.publish(c, 1u << 20, total, hash, err), "the >1 GiB payload publishes");
            HandoffArena r;
            check(r.open_reader(bpath, pack_hash, geom_hash, err), "reader open on the big arena");
            HandoffPayload p;
            check(r.read_slot(c.index, c.nonce, 0xBEEF, p, err), "the big slot maps PROT_READ");
            check(p.payload_bytes == total, "and reports the full byte count");
            const bool spot = p.valid() && p.data[0] == buf[0] &&
                              p.data[total - 1] == buf[(total - 1) % chunk] &&
                              p.data[512ull << 20] == buf[(512ull << 20) % chunk];
            check(spot, "the first, middle and last bytes are the writer's");
            const uint64_t got = HandoffArena::payload_hash_of(p);
            check(got == hash, "the FNV-1a over a >1 GiB payload matches on both sides");
            std::printf("  ..   big slot: %llu B written and read back, hash %s\n",
                        (unsigned long long) total, hex(got).c_str());
            check(a.release(c.index, c.nonce, err), "the big slot releases");
            check(a.free_slots() == "3/3", "and the big arena is empty again");
            remove_arena(bpath);
        }
    }

    // ================================ 14. a restarted prefill instance reclaims a dead creator =========
    std::printf("restart\n");
    {
        const std::string rpath = arena_path("restart");
        std::error_code ec;
        remove_arena(rpath);
        const HandoffSizing s = size_basic();
        std::string err;
        {
            // A child creates the arena, claims a slot, publishes it, and dies without ACKing.
            const std::string self = self_path();
            if (!self.empty()) {
                const std::string out = run(self + " --role writer " + rpath + " " + hex(pack_hash) + " " +
                                            hex(geom_hash) + " 2 1234 606 300000 basic");
                check(out.find("PUBLISH ok=1") != std::string::npos,
                      "the child created the arena, wrote a slot and published it, then exited");
                HandoffArena mid;
                check(mid.open_reader(rpath, pack_hash, geom_hash, err), "the arena is readable");
                HandoffSlotInfo si;
                check(mid.slot_info(4, si) && si.state == kHandoffReady,
                      "the child's slot is READY with nobody to read it");
                HandoffPayload before;
                check(mid.read_slot(4, si.nonce, 1234, before, err) && before.payload_bytes == kRolePayload,
                      "and the child's payload is readable while its creator's arena stands");
                before = HandoffPayload{};
                mid = HandoffArena{};
                // A new prefill instance starts: the creator is dead, so the table is rebuilt.
                HandoffArena fresh;
                check(fresh.open_writer(rpath, s, pack_hash, geom_hash, err),
                      "a restarted instance re-opens the same arena path");
                check(fresh.header().created_pid == (uint64_t) (uint32_t) getpid(),
                      "and the header now names the new creator");
                check(fresh.free_slots() == "6/6",
                      "the dead instance's slots are rebuilt to FREE - no payload from a dead process survives");
                HandoffArena after;
                HandoffPayload p;
                check(after.open_reader(rpath, pack_hash, geom_hash, err) &&
                          !after.read_slot(4, si.nonce, 1234, p, err) && err.find("FREE") != std::string::npos,
                      "and the dead instance's payload is no longer readable");
            } else {
                check(false, "could not read own executable path for the restart case");
            }
        }
        remove_arena(rpath);
    }

    if (!g_root.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(g_root, ec);   // only the directory this test created
    }

    std::printf("handoff_arena_test: %d checks, %s\n", g_checks, g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
