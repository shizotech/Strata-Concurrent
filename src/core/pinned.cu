// src/core/pinned.cu - P2.S1: the pinned host arena and the parallel expert load.
#include "strata/core/pinned.hpp"
#include "strata/platform/memory.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <climits>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>

// Loader fix: `fseek`/`ftell` are 32-bit on Windows by default (and the pack is 42.9 GB), and the 64-bit
// spelling is not the same on the two platforms the engine builds for.
#ifdef _WIN32
#define STRATA_FSEEK64(f, o) _fseeki64((f), (long long) (o), SEEK_SET)
#else
#define STRATA_FSEEK64(f, o) fseeko((f), (off_t) (o), SEEK_SET)
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#include <filesystem>
#include <system_error>
#include <linux/mman.h>
#ifndef MAP_HUGE_2MB
#define MAP_HUGE_2MB (21 << 26)
#endif
#endif

namespace strata::core {

namespace {

constexpr char kSharedArenaMagic[16] = "STRATA-ARENA-V1";

/// The on-disk header.  Its layout is the ABI documented in pinned.hpp: the last four fields ARE the header's
/// old `reserved[4]`, which is why they sit here instead of after it - an engine built before this protocol
/// must still read a new file and refuse it correctly, and a new engine must read an old file and see
/// `state == 0` ("not populated"), never garbage.
struct SharedArenaHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t arena_bytes;
    uint64_t pack_hash;
    uint64_t state;        ///< 0 = the body is not populated, kSharedArenaReady = safe to borrow
    uint64_t owner;        ///< pid that owns the load, 0 = nobody does
    uint64_t nonce;        ///< random per load attempt: says WHICH attempt owns it, not just which pid
    uint64_t owner_start;  ///< the owner's process start time (0 = unknown / an old file)
};
static_assert(sizeof(SharedArenaHeader) <= kSharedArenaHeaderBytes);
// The offsets in the header are part of the protocol, so they are pinned to the struct rather than written
// twice: if somebody reorders these fields, this file stops compiling instead of silently publishing an
// arena every other process reads as state 0.
static_assert(kSharedArenaStateOffset == offsetof(SharedArenaHeader, state));
static_assert(kSharedArenaOwnerOffset == offsetof(SharedArenaHeader, owner));
static_assert(kSharedArenaNonceOffset == offsetof(SharedArenaHeader, nonce));
static_assert(kSharedArenaOwnerStartOffset == offsetof(SharedArenaHeader, owner_start));

/// What the shared-backing reservation decided, reported back to the PinnedArena ctor (and from there to the
/// driver).  `role`: 0 = not shared / failed, 1 = this process loads, 2 = the bytes are already there.
struct SharedReservation {
    int role = 0;
    uint64_t nonce = 0;
    uint64_t owner_pid = 0;      ///< the header's owner when the arena was borrowed
    uint64_t waited_ms = 0;
    std::string reason;          ///< why this process got that role, for the note
};

/// Why the working-set lock is skipped for a shared backing.  It is not a shortcut: on Linux the shared arena
/// is a tmpfs file (/dev/shm), and tmpfs pages are ALREADY unevictable - they have no backing store to be
/// written back to, so mlock can only fail.  It does fail here, against `ulimit -l` = 8 MiB, and the old code
/// paid for that failure twice a start with a scary note.  On Windows the shared branch refuses the arena
/// before it ever gets here.
std::string lock_skipped_for_shared_file_note() {
    return "working-set lock skipped: the shared-file arena is tmpfs-backed, whose pages are already "
           "unevictable (mlock would only fail against ulimit -l)";
}

}  // namespace

#ifndef _WIN32
namespace {

/// How long a process may wait for somebody else's load before it decides that load is never coming.  Sized
/// from the measurements, not from a guess: the same 46.84 GiB arena read at 1.4 GiB/s from a warm cache and
/// at 0.09 GiB/s in a throttled launch context, so the slow arm needs ~9 minutes; 20 is the point where
/// waiting longer is worse than doing the read twice.  STRATA_SHARED_ARENA_WAIT_MS overrides it
/// (0 = never wait, take the load over at once) - the same A/B knob style as STRATA_ARENA_LOCK.
uint64_t shared_arena_wait_ms() {
    const char* env = std::getenv("STRATA_SHARED_ARENA_WAIT_MS");
    if (env == nullptr || *env == 0) return 20ull * 60ull * 1000ull;
    return std::strtoull(env, nullptr, 10);
}

/// Take the backing file's advisory lock.  It is held only for a DECISION (microseconds), never across a load,
/// so failing to get it within a few seconds means a foreign flock - report it instead of hanging at startup.
bool arena_lock(int fd, std::string& note) {
    constexpr uint64_t kRetryMs = 20, kGiveUpMs = 10000;
    for (uint64_t waited = 0;; waited += kRetryMs) {
        // `::flock`, not `flock`: <fcntl.h> also declares `struct flock` (POSIX record locks), and in C++ that
        // tag is a type name, so an unqualified call can be parsed as a declaration instead of a call.
        if (::flock(fd, LOCK_EX | LOCK_NB) == 0) return true;
        if (errno != EWOULDBLOCK && errno != EAGAIN) {
            note = std::string("cannot lock shared arena: ") + std::strerror(errno);
            return false;
        }
        if (waited >= kGiveUpMs) {
            note = "shared arena is locked by another process for more than " + std::to_string(kGiveUpMs / 1000) +
                   " s; refusing rather than starting on an arena whose state cannot be read";
            return false;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(kRetryMs));
    }
}

void arena_unlock(int fd) { (void) ::flock(fd, LOCK_UN); }

bool read_arena_header(int fd, SharedArenaHeader& hdr) {
    const ssize_t got = pread(fd, &hdr, sizeof hdr, 0);
    return got == (ssize_t) sizeof hdr;
}

bool write_arena_header(int fd, const SharedArenaHeader& hdr) {
    return pwrite(fd, &hdr, sizeof hdr, 0) == (ssize_t) sizeof hdr;
}

/// Is `pid` a running process?  `/proc/<pid>` is the only portable-enough answer that needs no signal
/// permission (kill(pid, 0) says EPERM for somebody else's process, which is NOT "dead").  When /proc itself
/// is not mounted the answer is unknown, and unknown must mean ALIVE: guessing "dead" would let two processes
/// load the same arena at once, which is the exact bug this protocol exists to fix.
bool pid_alive(uint64_t pid) {
    if (pid == 0 || pid > (uint64_t) INT32_MAX) return false;
    struct stat st{};
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%llu", (unsigned long long) pid);
    if (stat(path, &st) == 0) return true;
    return stat("/proc", &st) != 0;
}

/// The process's start time, from field 22 of /proc/<pid>/stat (clock ticks since boot).
///
/// A pid on its own CANNOT prove a loader is still the loader: if the owner was SIGKILLed and its pid was
/// recycled, `/proc/<pid>` exists, `pid_alive` says yes, and a process that trusts the pid alone stalls for the
/// whole wait budget on a load that will never happen.  The start time distinguishes the dead owner from the
/// recycled pid.  0 means "unknown" (no /proc, or a file written before this field existed), and unknown is
/// treated as a MATCH - the conservative answer, same rule as `pid_alive`.
uint64_t proc_start_time(uint64_t pid) {
    if (pid == 0 || pid > (uint64_t) INT32_MAX) return 0;
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%llu/stat", (unsigned long long) pid);
    FILE* f = std::fopen(path, "rb");
    if (f == nullptr) return 0;
    // The comm field is field 2 and is parenthesised and may contain spaces and parentheses, so the fields
    // after it are counted from the LAST ')' rather than by splitting on ' '.
    char buf[4096];
    const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
    std::fclose(f);
    if (n == 0) return 0;
    buf[n] = 0;
    char* close_paren = std::strrchr(buf, ')');
    if (close_paren == nullptr) return 0;
    // Fields 3.. are space separated after the ')': field 22 is 20 fields further along.
    const char* p = close_paren + 1;
    for (int field = 3; field < 22 && *p != 0; ++field) {
        while (*p == ' ') ++p;
        while (*p != 0 && *p != ' ') ++p;
    }
    while (*p == ' ') ++p;
    if (*p == 0) return 0;
    return std::strtoull(p, nullptr, 10);
}

/// Does the header's claim still belong to a live process?  Both the pid and its start time must match.
bool claim_alive(uint64_t pid, uint64_t start_time) {
    if (!pid_alive(pid)) return false;
    if (start_time == 0) return true;   // the file does not say; do not guess "dead"
    const uint64_t now = proc_start_time(pid);
    if (now == 0) return true;          // /proc unreadable for this pid; same conservative answer
    return now == start_time;
}

/// Tells two load attempts apart, which is what matters after pid reuse: a stale header naming a pid that has
/// since been recycled must not be believed forever.  Not a CSPRNG and does not need to be - it must simply
/// not repeat within a machine's pid lifetime, so time + pid + a stack address, mixed.
uint64_t make_load_nonce() {
    uint64_t x = (uint64_t) std::chrono::steady_clock::now().time_since_epoch().count();
    x ^= ((uint64_t) (uint32_t) getpid() << 32);
    x ^= (uint64_t) (uintptr_t) &x;
    x *= 0x9e3779b97f4a7c15ull;
    x ^= x >> 29;
    return x != 0 ? x : 1;
}

/// Can a fresh backing file of `file_bytes` actually be created at `path`?  True when the check cannot be
/// made (no statvfs, an existing file, an unopenable path - the open below reports that properly), because a
/// guess is better here than refusing a working configuration; false only when the filesystem says NO.
///
/// POSIX-only, like the shared-file branch that calls it: `statvfs` and its header do not exist on Windows,
/// where a shared arena is refused before this is ever reached.
bool shared_file_fits(const std::string& path, uint64_t file_bytes, std::string& why) {
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
    char buf[160];
    std::snprintf(buf, sizeof buf, "%s has %llu MiB free and it needs %llu MiB",
                  parent.string().c_str(), (unsigned long long) (avail >> 20),
                  (unsigned long long) (file_bytes >> 20));
    why = buf;
    return false;
}

}  // namespace
#endif  // !_WIN32

namespace {

// A 2 MB-aligned reservation.  Large pages first, then the largest alignment the OS will give us for free.
// A non-empty shared_file instead maps one file whose first 4 KiB identify the pack and whose remaining bytes
// are the resident arena.  This shared-file layout is intended for tmpfs (/dev/shm); hugetlbfs would need
// hugepage-aligned file size and arena offset rather than the 4 KiB header layout used here.
void* reserve(uint64_t bytes, PageBacking& got, std::string& note, const std::string& shared_file,
              uint64_t shared_pack_hash, void*& mapping_base, uint64_t& mapping_bytes,
              SharedReservation& shared) {
    mapping_base = nullptr;
    mapping_bytes = 0;
#ifdef _WIN32
    // The shared-file protocol below is POSIX-only (flock, /proc, mmap).  Windows still refuses a shared arena,
    // so the role struct stays untouched there rather than reporting a decision nobody made - and this keeps the
    // parameter from reading as dead code under -Wunused-parameter.
    (void) shared;
    if (!shared_file.empty()) {
        note = "shared-file arena backing is not implemented on Windows";
        return nullptr;
    }
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege.  Having it assigned to the account is not enough: the
    // PROCESS must enable it in its own token (AdjustTokenPrivileges) before VirtualAlloc, or the call fails.
    // An account without the assignment, or a failure to enable, leaves the process as it was: VirtualAlloc
    // then refuses and the 4 KB fallback below runs - that is the EXPECTED outcome on a desktop.
    {
        HANDLE tok = nullptr;
        if (OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) {
            TOKEN_PRIVILEGES tp{};
            tp.PrivilegeCount = 1;
            if (LookupPrivilegeValueW(nullptr, L"SeLockMemoryPrivilege", &tp.Privileges[0].Luid)) {
                tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
                if (!AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) && GetLastError() != ERROR_NOT_ALL_ASSIGNED)
                    (void) 0;   // nothing actionable: the large-page attempt below reports the outcome
            }
            CloseHandle(tok);
        }
    }
    // MEM_LARGE_PAGES needs SeLockMemoryPrivilege; a normal account does not have it and VirtualAlloc then
    // fails with ERROR_PRIVILEGE_NOT_HELD.  That is the EXPECTED outcome on a desktop, not an error.
    SIZE_T large = GetLargePageMinimum();
    // A/B switch: STRATA_NO_LARGEPAGES=1 skips the large-page attempt, same run, same boot.
    if (large > 0 && std::getenv("STRATA_NO_LARGEPAGES") == nullptr) {
        // MEM_LARGE_PAGES requires the allocation size to be an exact multiple of the large page size -
        // anything else is ERROR_INVALID_PARAMETER (87), which reads like a privilege problem but is not.
        // Round up: the slack is under 2 MB and the tail stays unused.
        const SIZE_T lbytes = (SIZE_T) (((SIZE_T) bytes + large - 1) / large * large);
        void* p = VirtualAlloc(nullptr, lbytes, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES,
                               PAGE_READWRITE);
        if (p) {
            got = PageBacking::LargePages;
            note = "large pages (" + std::to_string((unsigned long long) large) + " B)";
            return p;
        }
        // 1450 (ERROR_NO_SYSTEM_RESOURCES) is the large-page pool saying no, 87 is a size that is not a
        // multiple of the minimum, 1314 is the privilege: without the byte count the three read as one bug.
        note = "large pages refused for " + std::to_string((unsigned long long) lbytes) + " B (GetLargePageMinimum=" +
               std::to_string((unsigned long long) large) + ", VirtualAlloc error " +
               std::to_string((unsigned long long) GetLastError()) + "); using 4 KB pages";
    } else if (std::getenv("STRATA_NO_LARGEPAGES") != nullptr) {
        note = "large pages skipped (STRATA_NO_LARGEPAGES); using 4 KB pages";
    } else {
        note = "this system has no large-page minimum; using 4 KB pages";
    }
    void* p = VirtualAlloc(nullptr, (SIZE_T) bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE);
    got = PageBacking::NormalPages;
    return p;
#else
    if (!shared_file.empty()) {
        if (shared_pack_hash == 0) {
            note = "shared arena requires a nonzero pack hash";
            return nullptr;
        }
        if (bytes > UINT64_MAX - kSharedArenaHeaderBytes) {
            note = "shared arena size overflows its header";
            return nullptr;
        }
        const uint64_t file_bytes = kSharedArenaHeaderBytes + bytes;
        // ENOUGH ROOM BEFORE ANYTHING IS CLAIMED.  A tmpfs is not infinite: Docker's default /dev/shm is
        // 64 MiB, and a container that gets no --shm-size cannot hold a 47 GiB arena.  `ftruncate` on tmpfs
        // does NOT fail when the space is not there - it happily makes a sparse file, and the process then
        // dies of SIGBUS on the first write past the limit, inside the expert load, with no message.  So the
        // filesystem's own free space is checked here, and a backing that cannot fit is refused the same way
        // any other unusable one is: the caller falls back to a private arena and says why.
        //
        // Only a FRESH file needs the check. An existing file of the right size is already allocated, so its
        // space is committed; re-checking it would refuse a working arena because some other program filled
        // the filesystem afterwards, which is a worse answer than letting a ready arena be borrowed.
        if (!shared_file_fits(shared_file, file_bytes, note)) {
            note = "shared arena " + shared_file + " cannot hold " + std::to_string(file_bytes) +
                   " B: " + note;
            return nullptr;
        }
        // O_CLOEXEC matters here, not just for tidiness: the protocol's flock lives on the OPEN FILE
        // DESCRIPTION, so an exec'd helper that inherited this fd would keep the lock alive after this process
        // has decided the arena's state - and every later process would then sit in `arena_lock` and refuse to
        // start.  (fork alone still shares the description; that is why the lock is only ever held across a
        // decision, never across a load.)
        const int fd = open(shared_file.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0600);
        if (fd < 0) {
            note = "cannot open shared arena " + shared_file + ": " + std::strerror(errno);
            return nullptr;
        }

        // EVERY decision about this file happens under its exclusive advisory lock.  Without it, two processes
        // that start together both see "not populated", both become loaders, and both write the same 46.84 GiB
        // through the same MAP_SHARED pages - which is not merely slow, it is a torn arena.  The lock covers
        // the decision only (read the header, maybe claim the load, maybe map); the load itself runs unlocked,
        // because holding it across a minute-long read would make every other process sit in `arena_lock`.
        if (!arena_lock(fd, note)) {
            close(fd);
            return nullptr;
        }

        struct stat st{};
        if (fstat(fd, &st) != 0) {
            const int e = errno;
            arena_unlock(fd);
            close(fd);
            note = "cannot stat shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }

        const bool fresh = st.st_size == 0;
        if (fresh) {
            if (ftruncate(fd, (off_t) file_bytes) != 0) {
                const int e = errno;
                arena_unlock(fd);
                close(fd);
                note = "cannot size shared arena " + shared_file + ": " + std::strerror(e);
                return nullptr;
            }
            SharedArenaHeader hdr{};
            std::memcpy(hdr.magic, kSharedArenaMagic, sizeof hdr.magic);
            hdr.version = 1;
            hdr.header_bytes = (uint32_t) kSharedArenaHeaderBytes;
            hdr.arena_bytes = bytes;
            hdr.pack_hash = shared_pack_hash;
            // state stays 0: the body is EMPTY until this process writes it.  The old code wrote this header
            // and nothing else, so a second process arriving mid-load saw a valid header, the right size and
            // the right hash and happily read a half-written arena.  `state` is what makes that impossible.
            hdr.owner = (uint64_t) (uint32_t) getpid();
            hdr.nonce = make_load_nonce();
            hdr.owner_start = proc_start_time((uint64_t) (uint32_t) getpid());
            if (!write_arena_header(fd, hdr)) {
                const int e = errno;
                arena_unlock(fd);
                close(fd);
                note = "cannot write shared arena header " + shared_file + ": " +
                       (e ? std::strerror(e) : std::string("short write"));
                return nullptr;
            }
            shared.role = 1;
            shared.nonce = hdr.nonce;
            shared.owner_pid = (uint32_t) hdr.owner;
            shared.reason = "created and claimed by pid " + std::to_string(hdr.owner);
        } else {
            if (st.st_size < 0 || (uint64_t) st.st_size != file_bytes) {
                arena_unlock(fd);
                close(fd);
                note = "shared arena " + shared_file + " is " +
                       std::to_string((unsigned long long) st.st_size) + " B, expected " +
                       std::to_string((unsigned long long) file_bytes) + " B including its header";
                return nullptr;
            }
            SharedArenaHeader hdr{};
            if (!read_arena_header(fd, hdr) ||
                std::memcmp(hdr.magic, kSharedArenaMagic, sizeof hdr.magic) != 0 ||
                hdr.version != 1 || hdr.header_bytes != kSharedArenaHeaderBytes || hdr.arena_bytes != bytes) {
                arena_unlock(fd);
                close(fd);
                note = "shared arena " + shared_file + " has an incompatible or missing header";
                return nullptr;
            }
            if (hdr.pack_hash != shared_pack_hash) {
                arena_unlock(fd);
                close(fd);
                char b[256];
                std::snprintf(b, sizeof b,
                              "shared arena %s was written for pack hash %016llx, expected %016llx",
                              shared_file.c_str(), (unsigned long long) hdr.pack_hash,
                              (unsigned long long) shared_pack_hash);
                note = b;
                return nullptr;
            }

            // ---- the state machine.  `state == ready` is the ONLY state a process may map and read. ----
            //
            // An old v1 file (state/owner/nonce all zero, because the writer had `reserved[4]` there) falls
            // into the not-populated arm and is loaded once, then published.  That is the backward-compatible
            // reading of "reserved == 0": it means NOT READY, never "assume somebody filled it in".
            if (hdr.state == kSharedArenaReady) {
                shared.role = 2;
                shared.owner_pid = (uint32_t) hdr.owner;
                shared.reason = "already populated by pid " + std::to_string(hdr.owner) +
                                " (state ready, pack hash matched)";
            } else if (hdr.owner != 0 && claim_alive(hdr.owner, hdr.owner_start)) {
                // Somebody is loading it RIGHT NOW.  Do not re-load it (that would fight the same pages) and do
                // not read it (it is half-written).  Wait, outside the lock, for the loader to publish.
                //
                // The loop follows the CURRENT claim rather than the one it started with: if the loader we were
                // waiting on died and a third process took the load over, the header's claim changes underneath
                // us and the correct answer is to wait on the NEW claim, not to steal it.  Only a dead or
                // cleared claim, or an expired budget, ends the wait without a published arena.
                arena_unlock(fd);
                const uint64_t budget = shared_arena_wait_ms();
                const auto t0 = std::chrono::steady_clock::now();
                uint64_t state = 0, owner = hdr.owner, owner_start = hdr.owner_start;
                for (;;) {
                    SharedArenaHeader h2{};
                    if (read_arena_header(fd, h2)) {
                        state = h2.state;
                        owner = h2.owner;
                        owner_start = h2.owner_start;
                    }
                    if (state == kSharedArenaReady) break;
                    if (owner == 0 || !claim_alive(owner, owner_start)) { state = 0; break; }
                    const uint64_t waited = (uint64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                                                std::chrono::steady_clock::now() - t0).count();
                    if (waited >= budget) { shared.waited_ms = (uint32_t) waited; break; }
                    std::this_thread::sleep_for(std::chrono::milliseconds(200));
                }
                shared.waited_ms = (uint32_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                                        std::chrono::steady_clock::now() - t0).count();
                if (state == kSharedArenaReady) {
                    shared.role = 2;
                    shared.owner_pid = (uint32_t) owner;
                    shared.reason = "waited " + std::to_string(shared.waited_ms) +
                                    " ms for pid " + std::to_string(owner) + "'s load";
                } else {
                    // Either the loader died, or it is still going and the wait budget ran out.  In both cases
                    // the honest answer is "this process loads it": reading an unpopulated arena is never an
                    // option, and a dead owner must not wedge the file forever.
                    //
                    // On a budget expiry two processes can then load the same arena at once.  That is wasteful
                    // but NOT corrupting: the pack hash matched, so both write byte-identical experts to the
                    // same offsets, and the loser's publish is refused by the nonce check in
                    // `publish_shared_load()`.  Refusing to start instead would trade a slow second server for
                    // no second server at all.
                    shared.role = 1;
                    shared.reason = "owner pid " + std::to_string(hdr.owner) +
                                    (shared.waited_ms >= budget ? " still loading after a " + std::to_string(budget) +
                                                                 " ms wait" : " is gone") +
                                    "; taking the load over";
                }
                if (shared.role == 1) {
                    // Re-claim under the lock, and re-check first: the loader may have published in the window
                    // between the last header read and here.
                    if (!arena_lock(fd, note)) { close(fd); return nullptr; }
                    SharedArenaHeader h3{};
                    if (read_arena_header(fd, h3) && h3.state == kSharedArenaReady) {
                        shared.role = 2;
                        shared.owner_pid = (uint32_t) h3.owner;
                        shared.reason = "already populated by pid " + std::to_string(h3.owner) +
                                        " (state ready, pack hash matched)";
                    } else {
                        SharedArenaHeader mine{};
                        // Rebuilt from scratch rather than copied out of the file: if the re-read above failed we
                        // must not write back a zeroed header and lose the arena's identity (size + pack hash).
                        std::memcpy(mine.magic, kSharedArenaMagic, sizeof mine.magic);
                        mine.version = 1;
                        mine.header_bytes = (uint32_t) kSharedArenaHeaderBytes;
                        mine.arena_bytes = bytes;
                        mine.pack_hash = shared_pack_hash;
                        mine.state = 0;
                        mine.owner = (uint64_t) (uint32_t) getpid();
                        mine.nonce = make_load_nonce();
                        mine.owner_start = proc_start_time((uint64_t) (uint32_t) getpid());
                        if (!write_arena_header(fd, mine)) {
                            arena_unlock(fd);
                            close(fd);
                            note = "cannot claim shared arena load " + shared_file;
                            return nullptr;
                        }
                        shared.nonce = mine.nonce;
                        shared.owner_pid = (uint32_t) mine.owner;
                    }
                    if (shared.role == 1) shared.reason = "took over the load: " + shared.reason;
                }
            } else {
                // Not populated and nobody owns it (fresh-ish file, old-engine header, or an owner that is gone).
                SharedArenaHeader mine{};
                std::memcpy(mine.magic, kSharedArenaMagic, sizeof mine.magic);
                mine.version = 1;
                mine.header_bytes = (uint32_t) kSharedArenaHeaderBytes;
                mine.arena_bytes = bytes;
                mine.pack_hash = shared_pack_hash;
                mine.state = 0;
                mine.owner = (uint64_t) (uint32_t) getpid();
                mine.nonce = make_load_nonce();
                mine.owner_start = proc_start_time((uint64_t) (uint32_t) getpid());
                if (!write_arena_header(fd, mine)) {
                    arena_unlock(fd);
                    close(fd);
                    note = "cannot claim shared arena load " + shared_file;
                    return nullptr;
                }
                shared.role = 1;
                shared.nonce = mine.nonce;
                shared.owner_pid = (uint32_t) mine.owner;
                shared.reason = hdr.state == 0 && hdr.owner == 0
                                    ? "not populated (v1 header with no load state); claimed by pid " +
                                          std::to_string(mine.owner)
                                    : "owner pid " + std::to_string(hdr.owner) + " is gone; claimed by pid " +
                                          std::to_string(mine.owner);
            }
        }

        // The map happens while the lock is still held for the loader/borrower decision paths, so no process
        // can be between "claimed" and "mapped" while another one rewrites the header underneath it.
        void* map = mmap(nullptr, (size_t) file_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        const int e = errno;
        arena_unlock(fd);
        close(fd);
        if (map == MAP_FAILED) {
            note = "cannot map shared arena " + shared_file + ": " + std::strerror(e);
            return nullptr;
        }
        mapping_base = map;
        mapping_bytes = file_bytes;
        got = PageBacking::NormalPages;
        note = "MAP_SHARED file " + shared_file + " (pack hash checked)";
        return (uint8_t*) map + kSharedArenaHeaderBytes;
    }

    // STRATA_NO_LARGEPAGES=1 is the same-run A/B switch the Windows branch documents; honor it
    // here too, so the large-page path can be compared without changing the pool or rebooting.
    if (std::getenv("STRATA_NO_LARGEPAGES") != nullptr) {
        note = "large pages skipped (STRATA_NO_LARGEPAGES); using 4 KB pages";
    } else {
        void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB | MAP_HUGE_2MB, -1, 0);
        if (p != MAP_FAILED) {
            got = PageBacking::LargePages;
            note = "hugetlb 2 MB pages";
            return p;
        }
        // MAP_HUGETLB is all-or-nothing: a pool smaller than the mapping fails exactly like an absent
        // one, and the old message guessed "no hugetlb pool configured" either way. Name the shortfall:
        // how many 2 MiB pages the mapping needs against what vm.nr_hugepages actually holds.
        const unsigned long long need = ((unsigned long long) bytes + (1ull << 21) - 1) / (1ull << 21);
        unsigned long long pool = 0;
        bool have_pool = false;
        if (std::FILE* f = std::fopen("/proc/sys/vm/nr_hugepages", "r")) {
            have_pool = std::fscanf(f, "%llu", &pool) == 1;
            std::fclose(f);
        }
        note = "MAP_HUGETLB unavailable (needed " + std::to_string(need) + " 2 MiB pages, vm.nr_hugepages=" +
               (have_pool ? std::to_string(pool) : std::string("?")) + "); using 4 KB pages";
    }
    // No hugetlb mapping: a 2 MiB-aligned anonymous mapping with MADV_HUGEPAGE, so transparent huge pages back
    // it where THP is "madvise" (the common distro default). The CPU expert pool streams whole experts out of
    // this arena; with 4 KB pages every 3 MB expert costs ~750 TLB misses.
    constexpr uint64_t kAlign = 2ull << 20;
    const uint64_t padded = bytes + kAlign;
    void* raw = mmap(nullptr, padded, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    got = PageBacking::NormalPages;
    if (raw == MAP_FAILED) return nullptr;
    const uintptr_t start = (uintptr_t) raw;
    const uintptr_t aligned = (start + kAlign - 1) & ~(uintptr_t) (kAlign - 1);
    if (aligned > start) munmap(raw, aligned - start);
    const uintptr_t end = aligned + bytes, raw_end = start + padded;
    if (raw_end > end) munmap((void*) end, raw_end - end);
    void* p = (void*) aligned;
    // #771: with THP "always" the kernel already backs this mapping with huge pages where it can; the extra
    // MADV_HUGEPAGE only adds direct reclaim and compaction on every fault (defrag=madvise), which on a fragmented
    // machine stretched a 25 s start to 432 s.  So it is not asked for there.  STRATA_NO_ARENA_THP=1 skips it
    // anywhere (the request alone; STRATA_NO_LARGEPAGES also skips the hugetlb try).
    std::string thp_mode;
    if (std::FILE* f = std::fopen("/sys/kernel/mm/transparent_hugepage/enabled", "r")) {
        char line[128] = {0};
        if (std::fgets(line, sizeof line, f) != nullptr) {
            const char* open = std::strchr(line, '[');
            const char* close = open ? std::strchr(open, ']') : nullptr;
            if (open && close) thp_mode.assign(open + 1, close);
        }
        std::fclose(f);
    }
    const bool skip_thp_request = std::getenv("STRATA_NO_ARENA_THP") != nullptr || thp_mode == "always";
    if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && skip_thp_request) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += std::getenv("STRATA_NO_ARENA_THP") != nullptr
                    ? "; transparent huge pages skipped (STRATA_NO_ARENA_THP)"
                    : "; transparent huge pages are on for every mapping (THP always): MADV_HUGEPAGE not requested";
    } else if (std::getenv("STRATA_NO_LARGEPAGES") == nullptr && madvise(p, bytes, MADV_HUGEPAGE) == 0) {
        const std::string four_k = "; using 4 KB pages";
        if (note.size() >= four_k.size() && note.compare(note.size() - four_k.size(), four_k.size(), four_k) == 0)
            note.resize(note.size() - four_k.size());
        note += "; transparent huge pages requested (MADV_HUGEPAGE)";
    }
    return p;
#endif
}

void release(void* p, uint64_t bytes) {
    if (!p) return;
#ifdef _WIN32
    (void) bytes;
    VirtualFree(p, 0, MEM_RELEASE);
#else
    munmap(p, bytes);
#endif
}

}  // namespace

uint64_t fnv1a64(const uint8_t* p, uint64_t n, uint64_t seed) {
    uint64_t h = seed;
    for (uint64_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

namespace {
bool clear_error() { (void) cudaGetLastError(); return true; }
}  // namespace

int arena_pin_cap_gib() {
    const char* e = std::getenv("STRATA_ARENA_PIN_GIB");
    if (e == nullptr || *e == '\0') return -1;
    if (std::string(e) == "auto") return -2;   // #243: the Windows shared-memory budget sets the sliced pin's cap
    const int v = std::atoi(e);
    return v < 0 ? -1 : v;
}

namespace {
#ifdef _WIN32
// #243: how much of the arena the sliced registration may pin on Windows when the whole arena was refused.
// Page-locked memory the GPU maps is charged to its shared (non-local) WDDM segment; pinned slice by slice until
// the driver refused one (28 GiB of a 63 GB PC), that segment was left full and every later cudaMalloc failed
// "out of memory".  4 GiB of the budget stay free for what the engine allocates after the arena; without the DXGI
// numbers, RAM/2 - 8 GiB (the budget is about half the RAM).  False: no limit could be worked out.
bool sliced_pin_limit(uint64_t& limit, std::string& why) {
    constexpr uint64_t GiB = 1ull << 30;
    char buf[256];
    int dev = 0;
    cudaDeviceProp p{};
    uint64_t budget = 0, usage = 0;
    std::string err = "no CUDA device properties";
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess &&
        strata::platform::gpu_shared_memory_budget(p.luid, budget, usage, err)) {
        limit = budget > usage + 4 * GiB ? budget - usage - 4 * GiB : 0;
        std::snprintf(buf, sizeof buf, "the GPU's shared-memory budget %.1f GiB - %.1f GiB in use - 4 GiB",
                      (double) budget / GiB, (double) usage / GiB);
        why = buf;
        return true;
    }
    (void) cudaGetLastError();
    const uint64_t ram = strata::platform::total_physical_memory();
    if (ram == 0) return false;
    limit = ram / 2 > 8 * GiB ? ram / 2 - 8 * GiB : 0;
    std::snprintf(buf, sizeof buf, "%s: RAM/2 - 8 GiB of %.1f GiB", err.c_str(), (double) ram / GiB);
    why = buf;
    return true;
}
#endif
}  // namespace

namespace {
std::vector<uint64_t> uniform_bounds(uint64_t bytes, uint64_t slice) {
    std::vector<uint64_t> b;
    if (slice == 0) return b;
    for (uint64_t off = 0; off + slice <= bytes; off += slice) b.push_back(off);
    if (!b.empty()) b.push_back(b.back() + slice);
    return b;
}
}  // namespace

PinnedArena::PinnedArena(uint64_t bytes, uint64_t slice) : PinnedArena(bytes, uniform_bounds(bytes, slice)) {
    if (slice_bytes) slice_bytes = slice;   // sliced registration: record the uniform size
}

PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds,
                         uint64_t max_pinned_bytes, const std::string& shared_file,
                         uint64_t shared_pack_hash) : capacity(bytes) {
    if (bytes == 0) return;
    SharedReservation shared;
    base = reserve(bytes, backing, note, shared_file, shared_pack_hash, mapping_base, mapping_bytes, shared);
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }
    const bool is_shared_file = base != nullptr && !shared_file.empty();
    if (is_shared_file) {
        shared_file_ = shared_file;
        this->shared_pack_hash = shared_pack_hash;
        shared_load_nonce = shared.nonce;
        shared_owner_pid = (uint32_t) shared.owner_pid;
        shared_waited_ms = shared.waited_ms;
        shared_borrowed = shared.role == 2;
        shared_load_owner = shared.role == 1;
        // Say WHAT THE ENGINE GOT, not what it asked for: "borrowed" is the whole point of the feature (no
        // second 46.84 GiB read), and "owner" is what tells the operator which process is still loading.
        note = (shared_borrowed ? "SHARED ARENA BORROWED, experts not re-loaded [" + shared.reason + "]"
                                : "SHARED ARENA OWNER, caller must load the arena [" + shared.reason + "]") +
               "; " + note;
    }

    // Register with CUDA BEFORE any page is touched: cudaHostRegister pins what is resident now, and a region
    // that has already been faulted in page by page is far more expensive to register and may fail outright.
    if (base) {
        // #243: STRATA_ARENA_PIN_GIB=N caps the registration from the start where the caller set no cap
        const int env_gib = arena_pin_cap_gib();
        uint64_t cap = max_pinned_bytes;
        std::string cap_why = "by the engine (multi-GPU under WDDM, or remote experts)";
        if (cap == 0 && env_gib > 0) {
            cap = (uint64_t) env_gib << 30;
            cap_why = "by STRATA_ARENA_PIN_GIB";
        }
        const bool capped = cap > 0 && cap < bytes && bounds.size() >= 2;
        const cudaError_t e = capped ? cudaSuccess :
            cudaHostRegister(base, (size_t) bytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
        if (!capped && e == cudaSuccess) {
            note = "cudaHostRegister PORTABLE ok; " + note;
            registered_bytes = bytes;
        } else if (bounds.size() >= 2 && (capped || clear_error())) {
            // Plan v0.3 P5: the whole range is refused, so pin it slice by slice from the start.  The rest stays
            // resident through the working-set lock below.  (P6: slices may differ in size, one per layer.)
            slice_bytes = 1;   // sliced; the uniform constructor records the size
            bool limited = capped;
            uint64_t limit = cap;
            std::string limit_why;
#ifdef _WIN32
            // #243 (opt-in, STRATA_ARENA_PIN_GIB=auto): not up to the driver's refusal but below the shared-memory
            // budget, for a PC where the full sliced pin leaves WDDM refusing later allocations.  Not the default: a
            // 64 GB PC pins 30 GiB past that budget without trouble, and capping it at 26 cost ~20% prompt speed.
            if (!capped && env_gib == -2) limited = sliced_pin_limit(limit, limit_why);
#endif
            for (size_t i = 0; i + 1 < bounds.size(); ++i) {
                const uint64_t off = bounds[i], n = bounds[i + 1] - bounds[i];
                if (limited && (off > limit || n > limit - off)) break;
                if (cudaHostRegister((uint8_t*) base + off, (size_t) n, cudaHostRegisterPortable | cudaHostRegisterMapped) != cudaSuccess) {
                    (void) cudaGetLastError();
                    break;
                }
                slice_starts.push_back(off);
                registered_bytes = off + n;
                ++registered_slices;
            }
            char gib[32];
            std::snprintf(gib, sizeof gib, "%.1f", (double) limit / (double) (1ull << 30));
            note = (capped ? "cudaHostRegister limited to " + std::to_string(cap >> 30) + " GiB " + cap_why + "; " :
                             "cudaHostRegister of the whole arena FAILED (" + std::string(cudaGetErrorString(e)) + "); " +
                             (limited ? "slices capped at " + std::string(gib) + " GiB (" + limit_why +
                                        "; STRATA_ARENA_PIN_GIB=auto; N sets a cap; #243); " : std::string())) +
                   std::to_string(registered_slices) + " slices pinned (" + std::to_string(registered_bytes >> 30) +
                   " GiB); " + note;
            if (registered_bytes < bytes) {
                const char* env = std::getenv("STRATA_ARENA_LOCK");
                // #779 FIRST: large pages cannot be paged out, so there is nothing to lock and the
                // STRATA_ARENA_LOCK arm must not even be consulted.  Then the shared backing: its pages are
                // tmpfs pages, already unevictable, so the lock would only fail against `ulimit -l`.
                if (backing == PageBacking::LargePages) {   // #779: large pages cannot be paged out: nothing to lock
                    note = "large pages are resident without a lock; " + note;
                } else if (env == nullptr || std::string(env) != "0") {
                    if (is_shared_file) {
                        note = lock_skipped_for_shared_file_note() + "; " + note;
                    } else {
                        const strata::platform::LockResult lr =
                            strata::platform::lock_resident((uint8_t*) base + registered_bytes, bytes - registered_bytes);
                        locked_bytes = lr.locked_bytes;
                        note = lr.note + "; " + note;
                    }
                }
            }
        } else {
            note = std::string("cudaHostRegister FAILED (") + cudaGetErrorString(e) +
                   ") - the arena is NOT pinned, so copies will be slow; " + note;
            // **CONSUME THE ERROR, OR IT LIES ABOUT SOMETHING ELSE LATER.**
            //
            // `cudaGetLastError()` returns the last error and CLEARS it; until something reads it, the error
            // state is sticky.  This failure is caught and handled right here - the arena is simply not pinned -
            // but leaving it set meant the next `cudaGetLastError()` in the engine, which is `gr_read`'s launch
            // check, reported "out of memory" for kernels that allocate nothing.  That cost a round: the arena
            // was written off as not fitting the machine when in fact the only thing wrong was a stale error
            // from this line.
            //
            // It is the same trap `gr.cu` warns about for ASYNC faults, in the other direction: a synchronous
            // failure is sticky too, and it lies about where it happened just as convincingly.
            (void) cudaGetLastError();
            // Plan v0.3 P0.1: keep it RESIDENT instead. Unpinned, Windows trims the arena under memory pressure
            // and the CPU pool's rate then depends on the OS; locking it through the working set needs no
            // special privilege. STRATA_ARENA_LOCK=0 is the A/B arm.
            const char* env = std::getenv("STRATA_ARENA_LOCK");
            // Same order as the sliced fallback above: #779 FIRST (large pages are never pageable), then the
            // shared backing (tmpfs pages are already unevictable), then the STRATA_ARENA_LOCK A/B arm.
            if (backing == PageBacking::LargePages) {       // #779: large pages cannot be paged out: nothing to lock
                note = "large pages are resident without a lock; " + note;
            } else if (env == nullptr || std::string(env) != "0") {
                if (is_shared_file) {
                    note = lock_skipped_for_shared_file_note() + "; " + note;
                } else {
                    const strata::platform::LockResult lr = strata::platform::lock_resident(base, bytes);
                    locked_bytes = lr.locked_bytes;
                    note = lr.note + "; " + note;
                }
            } else {
                note = "arena lock disabled (STRATA_ARENA_LOCK=0); " + note;
            }
        }
    }
}

// #285: the deferred-registration constructor.  It reserves the arena and registers NOTHING: the caller runs
// register_slices() on its own thread while the loaders fill the slices already pinned.  It never takes a shared
// backing (empty `shared_file` below), so the flock/owner/nonce protocol is not in play here - and the caller
// must not enable it for a shared arena, because a borrower loads nothing and has no per-layer work to pipeline.
PinnedArena::PinnedArena(uint64_t bytes, const std::vector<uint64_t>& bounds, Deferred)
    : capacity(bytes), bounds_(bounds) {
    if (bytes == 0 || bounds.size() < 2) return;
    SharedReservation shared;
    base = reserve(bytes, backing, note, std::string{}, 0, mapping_base, mapping_bytes, shared);
    if (base != nullptr && mapping_base == nullptr) {
        mapping_base = base;
        mapping_bytes = bytes;
    }
    slice_bytes = 1;   // sliced: one registration per layer
}

void PinnedArena::register_slices(std::atomic<int>& ready) {
    const int n = (int) bounds_.size() - 1;
    int i = 0;
    for (; base != nullptr && i < n; ++i) {
        const uint64_t off = bounds_[(size_t) i], len = bounds_[(size_t) i + 1] - off;
        if (cudaHostRegister((uint8_t*) base + off, (size_t) len, cudaHostRegisterPortable | cudaHostRegisterMapped) !=
            cudaSuccess) {
            (void) cudaGetLastError();
            break;
        }
        slice_starts.push_back(off);
        registered_bytes = off + len;
        ++registered_slices;
        ready.store(i + 1, std::memory_order_release);
    }
    note = "cudaHostRegister per layer, pipelined with the load: " + std::to_string(registered_slices) + " of " +
           std::to_string(n) + " slices pinned; " + note;
    if (i < n && base != nullptr) {   // the rest stays resident through the working-set lock, as the sliced fallback does
        const char* env = std::getenv("STRATA_ARENA_LOCK");
        if (backing == PageBacking::LargePages) {           // #779: large pages cannot be paged out: nothing to lock
            note = "large pages are resident without a lock; " + note;
        } else if (env == nullptr || std::string(env) != "0") {
            const strata::platform::LockResult lr =
                strata::platform::lock_resident((uint8_t*) base + registered_bytes, capacity - registered_bytes);
            locked_bytes = lr.locked_bytes;
            note = lr.note + "; " + note;
        }
    }
    ready.store(INT_MAX, std::memory_order_release);   // every slice done (or given up on)
}

bool PinnedArena::publish_shared_load(std::string* why) {
    const char* bad = nullptr;
#ifndef _WIN32
    if (!base || shared_file_.empty()) bad = "the arena has no shared-file backing";
    else if (!shared_load_owner) bad = "this process does not own the shared arena load";
    else if (shared_load_nonce == 0) bad = "the shared arena load was never claimed";
    else {
        // The body was written through MAP_SHARED pages by other threads; publish the READY flag through a
        // different path (pwrite on the fd), so the ordering is not automatic.  A release fence before the
        // store is what makes "state == ready" mean "every byte of the body is already visible" to the next
        // process - without it, a borrower can see the flag and still read a hole.
        std::atomic_thread_fence(std::memory_order_release);
        const int fd = open(shared_file_.c_str(), O_RDWR | O_CLOEXEC);
        if (fd < 0) {
            bad = "cannot reopen the shared arena to publish it";
        } else {
            // Publish ONLY if this process still owns the claim.  If the owner died and somebody else took the
            // load over, this process's bytes are stale and marking them ready would hand a half-written arena
            // to every later process - the exact bug the protocol exists to fix.
            SharedArenaHeader hdr{};
            uint64_t owner = 0, nonce = 0;
            if (read_arena_header(fd, hdr)) { owner = hdr.owner; nonce = hdr.nonce; }
            if (nonce != shared_load_nonce) {
                close(fd);
                shared_load_owner = false;
                shared_owner_pid = (uint32_t) owner;
                note = "shared arena load NOT published: its header is now claimed by pid " +
                       std::to_string(owner) + " (this process's claim was taken over); " + note;
                bad = "this process no longer owns the shared arena load";
            } else {
                hdr.state = kSharedArenaReady;
                hdr.owner = (uint64_t) (uint32_t) getpid();
                hdr.nonce = shared_load_nonce;
                const bool ok = write_arena_header(fd, hdr);
                close(fd);
                if (!ok) {
                    shared_load_owner = false;
                    note = "shared arena load NOT published (header write failed); " + note;
                    bad = "the shared arena header write failed";
                } else {
                    shared_load_owner = false;   // published: the arena is now borrowable, by us included
                    shared_borrowed = true;
                    shared_owner_pid = (uint32_t) getpid();
                    if (why) why->clear();
                    // The ctor's note said "SHARED ARENA OWNER, caller must load the arena", which was true
                    // then and is stale now: the driver prints `note()` AFTER the caller has loaded and
                    // published, so leaving it reads as an instruction nobody still has to follow.
                    {
                        const std::string must = "SHARED ARENA OWNER, caller must load the arena [";
                        const size_t at = note.find(must);
                        if (at != std::string::npos)
                            note.replace(at, must.size(), "SHARED ARENA PUBLISHED, other processes can borrow it [");
                        else
                            note = "SHARED ARENA PUBLISHED, other processes can borrow it; " + note;
                    }
                    return true;
                }
            }
        }
    }
#else
    bad = "shared-file arena backing is not implemented on Windows";
#endif
    if (why && bad) *why = bad;
    return false;
}

PinnedArena::~PinnedArena() {
    // A load that was claimed but never published must release its claim.  Without this, a process that gave up
    // on the arena (a refused expert load, a `--mmap-experts` fallback, an exception) but keeps RUNNING leaves a
    // header naming a LIVE pid, and every other process then waits out the whole wait budget for a load that is
    // never coming.  Clearing owner makes the next process take the load over immediately.
    // (Windows never reaches this state: the shared-file branch refuses the arena before it can claim a load.)
#ifndef _WIN32
    if (shared_load_owner && !shared_file_.empty() && shared_load_nonce != 0) {
        const int fd = open(shared_file_.c_str(), O_RDWR | O_CLOEXEC);
        if (fd >= 0) {
            SharedArenaHeader hdr{};
            if (read_arena_header(fd, hdr) && hdr.nonce == shared_load_nonce) {
                hdr.state = 0;
                hdr.owner = 0;
                hdr.nonce = 0;
                (void) write_arena_header(fd, hdr);
            }
            close(fd);
        }
    }
#endif
    shared_load_owner = false;
    if (base) {
        if (locked_bytes) strata::platform::unlock_resident((uint8_t*) base + (slice_bytes ? registered_bytes : 0), locked_bytes);
        if (slice_bytes) {
            for (uint64_t off : slice_starts) cudaHostUnregister((uint8_t*) base + off);
        } else {
            cudaHostUnregister(base);
        }
        release(mapping_base ? mapping_base : base, mapping_bytes ? mapping_bytes : capacity);
        base = nullptr;
        mapping_base = nullptr;
        mapping_bytes = 0;
    }
}

LoadStats load_experts(const std::string& path, uint8_t* dst, uint64_t blob_bytes, uint64_t blobs_per_layer,
                       uint64_t layers, int threads, uint64_t chunk) {
    std::vector<uint64_t> off((size_t) layers), n((size_t) layers, blobs_per_layer * blob_bytes);
    for (uint64_t L = 0; L < layers; ++L) off[(size_t) L] = L * blobs_per_layer * blob_bytes;
    return load_experts_ranges(path, dst, off, n, threads, chunk);
}

LoadStats load_experts_direct(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready) {
    LoadStats st;
    st.ok = false;
#ifdef _WIN32
    constexpr uint64_t kAlign = 4096;
    if (((uintptr_t) dst % kAlign) != 0 || chunk == 0 || chunk % kAlign != 0) return st;
    struct Piece { uint64_t off, n; int layer; };
    std::vector<Piece> pieces;
    uint64_t bytes = 0;
    for (size_t L = 0; L < layer_off.size(); ++L) {
        if (layer_off[L] % kAlign != 0 || layer_bytes[L] % kAlign != 0) return st;
        for (uint64_t p = 0; p < layer_bytes[L]; p += chunk)
            pieces.push_back({layer_off[L] + p, std::min<uint64_t>(chunk, layer_bytes[L] - p), (int) L});
        bytes += layer_bytes[L];
    }
    if (threads < 1) threads = 1;
    const auto t0 = std::chrono::steady_clock::now();
    std::atomic<size_t> next{0};
    std::mutex err_mu;
    std::string err;
    const int wide = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
    std::vector<wchar_t> wpath((size_t) (std::max)(wide, 1), L'\0');
    if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, wpath.data(), wide);
    auto worker = [&]() {
        HANDLE h = CreateFileW(wpath.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_FLAG_NO_BUFFERING | FILE_FLAG_SEQUENTIAL_SCAN, nullptr);
        if (h == INVALID_HANDLE_VALUE) {
            std::lock_guard<std::mutex> g(err_mu);
            if (err.empty())
                err = "cannot open " + path + " unbuffered (error " + std::to_string((unsigned long long) GetLastError()) + ")";
            next = pieces.size();
            return;
        }
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= pieces.size()) break;
            if (ready != nullptr)      // the slice must be registered before its pages are touched
                while (ready->load(std::memory_order_acquire) <= pieces[i].layer + 1) std::this_thread::yield();
            OVERLAPPED ov{};
            ov.Offset = (DWORD) pieces[i].off;
            ov.OffsetHigh = (DWORD) (pieces[i].off >> 32);
            DWORD got = 0;
            if (!ReadFile(h, dst + pieces[i].off, (DWORD) pieces[i].n, &got, &ov) || got != pieces[i].n) {
                std::lock_guard<std::mutex> g(err_mu);
                if (err.empty())
                    err = "short unbuffered read at offset " + std::to_string(pieces[i].off) + ": got " + std::to_string(got) +
                          " of " + std::to_string(pieces[i].n) + " B (error " +
                          std::to_string((unsigned long long) GetLastError()) + ")";
                next = pieces.size();
                break;
            }
        }
        CloseHandle(h);
    };
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();
    st.layers = layer_off.size();
    if (!err.empty()) {
        st.seconds = -1.0;
        st.error = err;
        return st;
    }
    st.ok = true;
    st.bytes = bytes;
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
#else
    (void) path; (void) dst; (void) layer_off; (void) layer_bytes; (void) threads; (void) chunk; (void) ready;
#endif
    return st;
}

bool experts_unbuffered(const std::vector<std::string>& files, uint64_t arena_bytes, std::string& why,
                        bool cache_counts, uint64_t read_bytes) {
    const char* env = std::getenv("STRATA_UNBUFFERED_LOAD");
    if (env != nullptr && env[0] != '\0') {
        why = std::string("STRATA_UNBUFFERED_LOAD=") + env;
        return env[0] != '0';
    }
#ifdef _WIN32
    LARGE_INTEGER freq{}, a{}, b{};
    QueryPerformanceFrequency(&freq);
    constexpr DWORD kRead = 64 << 10;
    constexpr int kSamples = 16;
    std::vector<uint8_t> buf(kRead);
    uint64_t total_bytes = 0;
    int fast = 0, n = 0;
    uint64_t seed = (uint64_t) GetTickCount64() * 6364136223846793005ull + 1442695040888963407ull;
    for (const std::string& f : files) {
        const int wide = MultiByteToWideChar(CP_UTF8, 0, f.c_str(), -1, nullptr, 0);
        std::vector<wchar_t> w((size_t) (std::max)(wide, 1), L'\0');
        if (wide > 0) MultiByteToWideChar(CP_UTF8, 0, f.c_str(), -1, w.data(), wide);
        HANDLE h = CreateFileW(w.data(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_FLAG_RANDOM_ACCESS,
                               nullptr);
        if (h == INVALID_HANDLE_VALUE) continue;
        LARGE_INTEGER size{};
        GetFileSizeEx(h, &size);
        total_bytes += (uint64_t) size.QuadPart;
        // random offsets: a probe must not find the blocks an earlier probe put into the cache
        for (int i = 0; i < kSamples && (uint64_t) size.QuadPart > 2ull * kRead; ++i) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            const uint64_t off = ((seed >> 17) % ((uint64_t) size.QuadPart - kRead)) / kRead * kRead;
            OVERLAPPED ov{};
            ov.Offset = (DWORD) off;
            ov.OffsetHigh = (DWORD) (off >> 32);
            DWORD got = 0;
            QueryPerformanceCounter(&a);
            const BOOL ok = ReadFile(h, buf.data(), kRead, &got, &ov);
            QueryPerformanceCounter(&b);
            if (!ok) continue;
            ++n;
            fast += (double) (b.QuadPart - a.QuadPart) * 1e6 / (double) freq.QuadPart < 30.0;
        }
        CloseHandle(h);
    }
    const bool cached = n > 0 && fast * 4 >= n * 3;
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    GlobalMemoryStatusEx(&ms);
    const uint64_t avail = ms.ullAvailPhys;
    // what the cache could keep beside the arena (~4 GiB for everything else)
    const uint64_t need = read_bytes == kAllFileBytes ? total_bytes : read_bytes;
    const bool keepable = strata::platform::file_cache_keeps(avail, arena_bytes, need);
    char msg[256];
    if (read_bytes == kAllFileBytes)
        std::snprintf(msg, sizeof msg, "%d of %d probe reads from the file cache; %.1f GiB available, %.1f GiB of files",
                      fast, n, (double) avail / (1ull << 30), (double) total_bytes / (1ull << 30));
    else
        std::snprintf(msg, sizeof msg, "%.1f GiB available, %.1f GiB of it still to be taken by the RAM copy, %.1f GiB "
                      "of experts read from the files: the file cache %s keep them",
                      (double) avail / (1ull << 30), (double) arena_bytes / (1ull << 30), (double) need / (1ull << 30),
                      keepable ? "can" : "cannot");
    why = msg;
    return (!cached || !cache_counts) && !keepable;
#else
    (void) files; (void) arena_bytes; (void) cache_counts; (void) read_bytes;
    why = "buffered (not Windows)";
    return false;
#endif
}

LoadStats load_experts_ranges(const std::string& path, uint8_t* dst, const std::vector<uint64_t>& layer_off,
                              const std::vector<uint64_t>& layer_bytes, int threads, uint64_t chunk,
                              const std::atomic<int>* ready) {
    LoadStats st;
    const uint64_t layers = (uint64_t) layer_off.size();
    st.layers = layers;
    st.bytes = 0;
    for (uint64_t b : layer_bytes) st.bytes += b;
    if (threads < 1) threads = 1;

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint64_t> layer_hash((size_t) layers, 1469598103934665603ull);
    std::atomic<uint64_t> next_layer{0};
    std::atomic<uint64_t> read_ns_sum{0};   // summed over the threads: see LoadStats::read_seconds
    std::atomic<uint64_t> copy_ns_sum{0};
    std::mutex err_mu;
    std::string err;

    auto worker = [&]() {
        std::vector<uint8_t> buf((size_t) chunk);
        // One handle per thread, seeked once per layer: a shared handle would need a lock around the seek and
        // would serialise the very thing the threads are here to parallelise.  `fread` on a `FILE*` rather than
        // `std::ifstream`: see the header - MSVC's `basic_filebuf::xsgetn` splits any request larger than
        // `_INTERNAL_BUFSIZ - 1` into 4095-byte freads, which turned one 8 MiB chunk into ~2048 4 KiB reads.
        // `fread` sees a request bigger than the stream buffer and passes it to `_read()`/
        // `ReadFile()` unchanged, so the chunk size reaches the disk.  Buffered, not `FILE_FLAG_NO_BUFFERING`:
        // the cache should still hold what it can.
        FILE* f = std::fopen(path.c_str(), "rb");
        if (f == nullptr) {
            std::lock_guard<std::mutex> g(err_mu);
            err = "cannot open " + path;
            return;
        }
        // A `FILE*` has no destructor that closes it, and this function has early returns below (open, seek and
        // short-read failures), so the guard is what keeps the closing correct on every path.
        struct Closer {
            FILE* f;
            ~Closer() { if (f != nullptr) std::fclose(f); }
        } closer{f};
        uint64_t read_ns = 0, copy_ns = 0;
        for (;;) {
            const uint64_t L = next_layer.fetch_add(1);
            if (L >= layers) break;
            if (ready != nullptr)      // the slice must be registered before its pages are touched
                while (ready->load(std::memory_order_acquire) <= (int) L + 1) std::this_thread::yield();
            const uint64_t off = layer_off[(size_t) L];
            uint64_t remaining = layer_bytes[(size_t) L];
            uint64_t pos = 0;
            uint64_t h = 1469598103934665603ull;
            // 64-bit seek: the pack is 42.9 GB, so the 32-bit `fseek` would wrap past 4 GiB
            if (STRATA_FSEEK64(f, off) != 0) {
                std::lock_guard<std::mutex> g(err_mu);
                err = "seek to " + std::to_string(off) + " B failed in layer " + std::to_string(L);
                return;
            }
            while (remaining > 0) {
                const uint64_t n = remaining < chunk ? remaining : chunk;
                const auto t_read = std::chrono::steady_clock::now();
                const size_t got = std::fread(buf.data(), 1, (size_t) n, f);
                read_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_read).count();
                // A short read is EOF or an I/O error, never a silent zero fill: say WHERE and HOW SHORT, and
                // keep going no further - the caller turns this into a refused load, not a wrong answer.
                if (got != (size_t) n) {
                    std::lock_guard<std::mutex> g(err_mu);
                    err = "short read in layer " + std::to_string(L) + ": got " + std::to_string(got) + " of "
                          + std::to_string(n) + " B at offset " + std::to_string(off + pos)
                          + (std::ferror(f) != 0 ? " (ferror set)" : "");
                    return;
                }
                const auto t_copy = std::chrono::steady_clock::now();
                std::memcpy(dst + off + pos, buf.data(), (size_t) n);
                h = fnv1a64(buf.data(), n, h);
                copy_ns += (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
                               std::chrono::steady_clock::now() - t_copy).count();
                pos += n;
                remaining -= n;
            }
            layer_hash[(size_t) L] = h;
        }
        read_ns_sum.fetch_add(read_ns);
        copy_ns_sum.fetch_add(copy_ns);
    };

    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(worker);
    worker();
    for (auto& t : pool) t.join();

    if (!err.empty()) {
        std::fprintf(stderr, "load_experts: %s\n", err.c_str());
        st.seconds = -1.0;
        st.ok = false;
        st.error = err;
        return st;
    }
    st.read_seconds = (double) read_ns_sum.load() / 1e9;
    st.copy_seconds = (double) copy_ns_sum.load() / 1e9;
    st.layer_checksums = std::move(layer_hash);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return st;
}

StreamStats stream_bandwidth(const uint8_t* src, uint64_t bytes, uint64_t chunk, int iters) {
    StreamStats st;
    st.bytes = bytes * (uint64_t) iters;
    st.chunk = chunk;
    uint8_t* dst = nullptr;
    cudaStream_t s{};
    if (cudaMalloc(&dst, (size_t) chunk) != cudaSuccess) {
        std::fprintf(stderr, "stream_bandwidth: cudaMalloc failed for %llu B\n", (unsigned long long) chunk);
        st.seconds = -1.0;
        return st;
    }
    cudaStreamCreate(&s);

    // one untimed pass so the first transfer's page-fault and setup cost is not in the measurement
    for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
        cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
    }
    cudaStreamSynchronize(s);

    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; ++it) {
        for (uint64_t off = 0; off + chunk <= bytes; off += chunk) {
            cudaMemcpyAsync(dst, src + off, (size_t) chunk, cudaMemcpyHostToDevice, s);
        }
    }
    cudaStreamSynchronize(s);
    st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    cudaStreamDestroy(s);
    cudaFree(dst);
    return st;
}

}  // namespace strata::core
