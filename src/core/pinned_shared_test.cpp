// Linux selftest for the optional MAP_SHARED backing of PinnedArena.
//
// CPU-ONLY ON PURPOSE: no model, no GPU work, no server.  Everything the shared-arena protocol decides is a
// decision about a 4 KiB header and a handful of bytes, so all of it is testable against a 64 KiB file in /tmp
// in about a second.  What this file must NOT do is re-derive the protocol from the engine's logs - the bug it
// covers (every process re-loading 46.84 GiB, and a second process reading a half-written arena) only shows up
// in a 47 GiB run, which is exactly why it needs a small deterministic test next to it.
//
// The protocol under test (see include/strata/core/pinned.hpp):
//   * exactly one process claims the load (`shared_load_owner`), writes the body, then `publish_shared_load()`;
//   * every other process sees `state == ready` and BORROWS (`shared_borrowed`) - the caller must load nothing;
//   * `state != ready` is never read: the process either waits for the live owner or takes the load over;
//   * a dead owner does not wedge the file.
#include "strata/core/pinned.hpp"

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
#include <sys/wait.h>
#include <unistd.h>

namespace {

int g_fail = 0;

void check(bool ok, const char* what) {
    if (ok) {
        std::printf("  ok   %s\n", what);
        return;
    }
    std::printf("  FAIL %s\n", what);
    ++g_fail;
}

/// A temp arena path.  NOT /dev/shm: the real backing file may be live under another process, and this test
/// must never touch it.  /tmp is a normal filesystem, which also shows the protocol is not tmpfs-specific.
std::string temp_arena(const char* tag) {
    char path[128];
    std::snprintf(path, sizeof path, "/tmp/strata-shared-arena-%s-XXXXXX", tag);
    const int fd = mkstemp(path);
    if (fd < 0) { std::perror("mkstemp"); std::exit(2); }
    close(fd);
    return std::string(path);
}

void set_wait_ms(const char* ms) { ::setenv("STRATA_SHARED_ARENA_WAIT_MS", ms, 1); }

/// Read the on-disk header the way an OLD engine would: `reserved[4]`, no protocol fields.  If the new fields
/// had been appended after the old struct instead of reusing `reserved[]`, this read would miss them entirely -
/// which is the compatibility requirement, so it is checked rather than asserted in a comment.
struct OldHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint64_t arena_bytes;
    uint64_t pack_hash;
    uint64_t reserved[4];
};

bool read_old_header(const std::string& path, OldHeader& h) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    const ssize_t got = ::pread(fd, &h, sizeof h, 0);
    close(fd);
    return got == (ssize_t) sizeof h;
}

bool write_old_header(const std::string& path, const OldHeader& h) {
    const int fd = ::open(path.c_str(), O_RDWR);
    if (fd < 0) return false;
    const bool ok = ::pwrite(fd, &h, sizeof h, 0) == (ssize_t) sizeof h;
    close(fd);
    return ok;
}

/// A pid that is definitely not running: fork, exit, reap.  (A made-up large pid would be a guess, and pid
/// reuse would make the test flaky in the exact direction that matters - "dead" must really mean dead.)
uint64_t dead_pid() {
    const pid_t p = fork();
    if (p < 0) { std::perror("fork"); std::exit(2); }
    if (p == 0) _exit(0);
    int status = 0;
    waitpid(p, &status, 0);
    return (uint64_t) (uint32_t) p;
}

/// Pull `key=<digits>` out of a child's ROLE line.  -1 when absent, so a missing field reads as a failure
/// rather than as "waited 0 ms".
long long field(const std::string& out, const char* key) {
    const std::string k = std::string(key) + "=";
    const size_t at = out.find(k);
    if (at == std::string::npos) return -1;
    return std::strtoll(out.c_str() + at + k.size(), nullptr, 10);
}

/// Run this binary again as `--role` and return everything it printed.
std::string run_role(const std::string& self, const std::string& path, uint64_t hash, uint64_t bytes,
                     const char* wait_ms) {
    char cmd[512];
    std::snprintf(cmd, sizeof cmd, "%s --role %s %llx %llu %s", self.c_str(), path.c_str(),
                  (unsigned long long) hash, (unsigned long long) bytes, wait_ms);
    FILE* p = popen(cmd, "r");
    std::string out;
    if (p == nullptr) return out;
    char line[512];
    while (std::fgets(line, sizeof line, p)) out += line;
    pclose(p);
    return out;
}

}  // namespace

/// Case 8 runs THIS binary again as a separate process.  Threads share an fd table, so a thread test cannot
/// prove the two things the protocol actually depends on across processes: that `flock` contends between
/// unrelated address spaces, and that a byte written by one process is visible to another through the file
/// rather than through a lucky alias.  `--role` mode prints what that process decided, and the parent reads it
/// back as text.
int role_mode(int argc, char** argv) {
    if (argc < 6) return 2;
    const std::string path = argv[2];
    const uint64_t hash = std::strtoull(argv[3], nullptr, 16);
    const uint64_t bytes = std::strtoull(argv[4], nullptr, 10);
    set_wait_ms(argv[5]);
    const std::vector<uint64_t> no_bounds;
    strata::core::PinnedArena a(bytes, no_bounds, 0, path, hash);
    std::printf("ROLE valid=%d owner=%d borrowed=%d waited=%llu\n", (int) a.valid(), (int) a.shared_load_owner,
                (int) a.shared_borrowed, (unsigned long long) a.shared_waited_ms);
    std::printf("NOTE %s\n", a.note.c_str());
    if (a.valid()) {
        // The sentinel the parent wrote before this process started.  A borrower must see it; a process that
        // (correctly) refused to trust the arena must not be relied on to see anything.
        std::printf("SENTINEL %d\n", (int) a.data()[123]);
        if (a.shared_load_owner) {
            a.data()[123] = 0x5a;
            std::printf("PUBLISHED %d\n", (int) a.publish_shared_load());
        }
    }
    return 0;
}

int main(int argc, char** argv) {
    if (argc > 1 && std::strcmp(argv[1], "--role") == 0) return role_mode(argc, argv);
    const uint64_t bytes = 64u << 10;
    const uint64_t pack_hash = 0x0123456789abcdefull;
    const std::vector<uint64_t> no_bounds;
    set_wait_ms("0");   // most cases below want "do not wait", so the test never sleeps by accident

    // ================================ 1. one loader, everyone else borrows ================================
    {
        const std::string path = temp_arena("load-once");
        std::printf("load-once (%s)\n", path.c_str());
        strata::core::PinnedArena first(bytes, no_bounds, 0, path, pack_hash);
        check(first.valid(), "first arena is valid");
        if (first.valid()) {
            check(first.shared_load_owner && !first.shared_borrowed, "first process OWNS the load");
            check(first.note.find("SHARED ARENA OWNER") != std::string::npos,
                  "note says this process owns the load");
            check(first.shared_load_nonce != 0, "the load attempt has a nonce");
            check(first.locked_bytes == 0, "a shared arena locks nothing (case 6 checks why)");

            // The body is NOT published yet, so the header must still say "not populated".
            OldHeader h0{};
            check(read_old_header(path, h0), "header readable");
            check(std::memcmp(h0.magic, "STRATA-ARENA-V1", 15) == 0 && h0.version == 1 &&
                      h0.header_bytes == 4096, "magic/version/header_bytes unchanged (v1 ABI)");
            check(h0.reserved[0] == 0, "state is 0 before the body is written");

            first.data()[123] = 0x5a;
            check(first.publish_shared_load(), "the owner publishes the load");

            strata::core::PinnedArena second(bytes, no_bounds, 0, path, pack_hash);
            check(second.valid(), "second arena is valid");
            if (second.valid()) {
                check(second.shared_borrowed && !second.shared_load_owner,
                      "second process BORROWS (the caller must load nothing)");
                check(second.note.find("SHARED ARENA BORROWED") != std::string::npos,
                      "note says the arena was borrowed");
                check(second.data()[123] == 0x5a, "the first process's bytes are visible to the second");
                check(!second.publish_shared_load(), "a borrower cannot publish (it owns no load)");
                // Still shared, not copied: a byte written after the borrow is seen through both mappings.
                first.data()[124] = 0xa5;
                check(second.data()[124] == 0xa5, "the second mapping is the same bytes, not a copy");
            }

            // What an old engine would read from the published file: the READY flag lives in the slot it
            // calls `reserved[0]`, and everything it does understand is untouched.
            OldHeader h1{};
            check(read_old_header(path, h1), "published header readable");
            check(h1.reserved[0] == strata::core::kSharedArenaReady, "state == ready after publish");
            check(h1.reserved[1] != 0, "owner pid recorded in the header");
            check(h1.reserved[2] != 0, "load nonce recorded in the header");
            check(h1.reserved[3] != 0, "owner start time recorded in the header");
            check(h1.arena_bytes == bytes && h1.pack_hash == pack_hash, "size and pack hash unchanged");
        }
        unlink(path.c_str());
    }

    // ================================ 2. a not-yet-populated file is never trusted ================================
    {
        const std::string path = temp_arena("not-ready");
        std::printf("not-ready (%s)\n", path.c_str());
        strata::core::PinnedArena a(bytes, no_bounds, 0, path, pack_hash);
        check(a.valid() && a.shared_load_owner, "first process owns the load");
        a.data()[0] = 0x11;   // half-written on purpose: never published

        // A second process must NOT borrow an arena nobody published.  With no wait budget it takes the load
        // over; either way it must not report `borrowed`, because `borrowed` is what tells the caller to skip
        // the expert load - and skipping it here would run the engine on a hole.
        strata::core::PinnedArena b(bytes, no_bounds, 0, path, pack_hash);
        check(b.valid(), "second arena is still valid");
        check(!b.shared_borrowed, "an unpublished arena is NOT borrowed");
        check(b.shared_load_owner, "the second process takes the unpublished load over");
        check(b.note.find("took over") != std::string::npos, "note says the load was taken over");

        // The first process no longer owns the claim, so it must not be able to publish somebody else's arena.
        check(!a.publish_shared_load(), "the displaced owner cannot publish");

        set_wait_ms("400");
        strata::core::PinnedArena c(bytes, no_bounds, 0, path, pack_hash);
        check(!c.shared_borrowed, "waiting does not turn an unpublished arena into a borrowed one");
        check(c.shared_load_owner, "the waiter takes the load over when the budget expires");
        check(c.shared_waited_ms >= 300, "the waiter actually waited for the owner");
        check(c.note.find("SHARED ARENA BORROWED") == std::string::npos, "note does not claim a borrow");
        set_wait_ms("0");

        check(c.publish_shared_load(), "the new owner can publish");
        strata::core::PinnedArena d(bytes, no_bounds, 0, path, pack_hash);
        check(d.shared_borrowed, "after that publish the arena is borrowed");
        unlink(path.c_str());
    }

    // ================================ 3. a dead owner does not wedge the file ================================
    {
        const std::string path = temp_arena("dead-owner");
        std::printf("dead-owner (%s)\n", path.c_str());
        strata::core::PinnedArena a(bytes, no_bounds, 0, path, pack_hash);
        check(a.valid() && a.shared_load_owner, "first process owns the load");
        check(a.publish_shared_load(), "first process publishes");

        // Rewrite the header as "claimed by a pid that no longer exists, body not populated" - what a killed
        // loader leaves behind.
        OldHeader h{};
        check(read_old_header(path, h), "header readable");
        h.reserved[0] = 0;
        h.reserved[1] = dead_pid();
        h.reserved[2] = 0x1234;
        check(write_old_header(path, h), "stale header written");

        set_wait_ms("60000");   // a live owner would be waited for; a dead one must not be
        strata::core::PinnedArena b(bytes, no_bounds, 0, path, pack_hash);
        check(b.valid(), "arena after a dead owner is still valid");
        check(b.shared_load_owner && !b.shared_borrowed, "the dead owner's arena is taken over");
        check(b.shared_waited_ms < 1000, "a dead owner is not waited for");
        check(b.note.find("gone") != std::string::npos, "note says the old owner is gone");
        set_wait_ms("0");
        unlink(path.c_str());

        // Pid REUSE: a header naming a pid that exists but started at a different time is a dead owner wearing
        // somebody else's number.  `/proc/<pid>` alone would say "alive" and stall the wait budget; the recorded
        // start time is what makes the take-over path exact.  This uses OUR OWN pid, so it is deterministic.
        const std::string path2 = temp_arena("pid-reuse");
        std::printf("pid-reuse (%s)\n", path2.c_str());
        strata::core::PinnedArena p(bytes, no_bounds, 0, path2, pack_hash);
        check(p.valid() && p.shared_load_owner, "first process owns the load");
        OldHeader r{};
        check(read_old_header(path2, r), "header readable");
        check(r.reserved[1] == (uint64_t) getpid(), "header names this process");
        check(r.reserved[3] != 0, "header records this process's start time");
        r.reserved[0] = 0;
        r.reserved[3] = r.reserved[3] ^ 0xdeadbeefull;   // "same pid, different process"
        check(write_old_header(path2, r), "recycled-pid header written");
        set_wait_ms("60000");
        strata::core::PinnedArena q(bytes, no_bounds, 0, path2, pack_hash);
        check(q.shared_load_owner && !q.shared_borrowed, "a recycled owner pid is taken over, not waited for");
        check(q.shared_waited_ms < 1000, "the recycled claim is not waited for");
        set_wait_ms("0");
        unlink(path2.c_str());
    }

    // ================================ 4. an old v1 file (reserved all zero) is not trusted ================================
    {
        const std::string path = temp_arena("old-v1");
        std::printf("old-v1 (%s)\n", path.c_str());
        // Hand the file the header the PREVIOUS engine wrote: same magic/version/size/hash, reserved zero.
        const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0600);
        if (fd < 0) { std::perror("open"); return 2; }
        if (ftruncate(fd, (off_t) (strata::core::kSharedArenaHeaderBytes + bytes)) != 0) {
            std::perror("ftruncate");
            return 2;
        }
        OldHeader h{};
        std::memcpy(h.magic, "STRATA-ARENA-V1", 16);
        h.version = 1;
        h.header_bytes = 4096;
        h.arena_bytes = bytes;
        h.pack_hash = pack_hash;
        if (::pwrite(fd, &h, sizeof h, 0) != (ssize_t) sizeof h) { std::perror("pwrite"); return 2; }
        close(fd);

        strata::core::PinnedArena a(bytes, no_bounds, 0, path, pack_hash);
        check(a.valid(), "an old v1 file is still accepted");
        check(a.shared_load_owner && !a.shared_borrowed, "an old v1 file is treated as NOT populated");
        check(a.note.find("not populated") != std::string::npos, "note says the old file was not populated");
        check(a.publish_shared_load(), "the old file is populated once and published");
        strata::core::PinnedArena b(bytes, no_bounds, 0, path, pack_hash);
        check(b.shared_borrowed, "the upgraded file is borrowed afterwards");
        unlink(path.c_str());
    }

    // ================================ 5. a live owner is waited for, then borrowed ================================
    //
    // The race the old code lost: process 2 starting DURING process 1's load.  Here thread 1 claims the load,
    // sleeps, writes the body and publishes; main starts its arena in the middle of that and must end up a
    // borrower with the right bytes - not a second loader, and not a reader of a half-written arena.
    {
        const std::string path = temp_arena("wait-owner");
        std::printf("wait-owner (%s)\n", path.c_str());
        bool loader_ok = false;
        std::thread loader([&]() {
            strata::core::PinnedArena a(bytes, no_bounds, 0, path, pack_hash);
            if (!a.valid() || !a.shared_load_owner) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            std::memset(a.data(), 0x77, (size_t) bytes);
            loader_ok = a.publish_shared_load();
            std::this_thread::sleep_for(std::chrono::milliseconds(400));   // keep the mapping alive
        });
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        set_wait_ms("5000");
        strata::core::PinnedArena b(bytes, no_bounds, 0, path, pack_hash);
        set_wait_ms("0");
        loader.join();
        check(loader_ok, "the loader thread published");
        check(b.valid(), "the second arena is valid");
        check(b.shared_borrowed, "the second process waited for the live owner and borrowed");
        check(b.shared_waited_ms >= 100, "the second process really waited");
        check(b.data()[1000] == 0x77, "the borrowed arena holds the owner's bytes");
        check(b.note.find("waited") != std::string::npos, "note says it waited for another process's load");
        unlink(path.c_str());
    }

    // ================================ 6. no mlock of shared pages, but the pin attempt stays ================================
    //
    // `max_pinned_bytes` with a slice boundary forces the "whole-range registration refused" path without
    // needing CUDA to misbehave, so the lock decision below is deterministic.  For a shared backing the
    // working-set lock must be SKIPPED (tmpfs/file-backed shared pages are not what mlock is for, and it fails
    // against `ulimit -l`), while the same path on an anonymous arena must still go through it.
    {
        const std::string path = temp_arena("no-mlock");
        std::printf("no-mlock (%s)\n", path.c_str());
        const std::vector<uint64_t> two_slices{0, bytes / 2, bytes};
        strata::core::PinnedArena shared_arena(bytes, two_slices, bytes / 2, path, pack_hash);
        check(shared_arena.valid(), "capped shared arena is valid");
        check(shared_arena.locked_bytes == 0, "a shared arena locks nothing");
        check(shared_arena.note.find("working-set lock skipped") != std::string::npos,
              "note says why the shared arena is not mlock'd");
        check(shared_arena.note.find("mlock failed") == std::string::npos, "no mlock failure note for a shared arena");
        unlink(path.c_str());

        strata::core::PinnedArena anon(bytes, two_slices, bytes / 2);
        check(anon.valid(), "capped anonymous arena is valid");
        check(anon.note.find("working-set lock skipped") == std::string::npos,
              "an anonymous arena still goes through the lock path");
    }

    // ================================ 7. the refusals must keep their words ================================
    {
        const std::string path = temp_arena("refuse");
        std::printf("refuse (%s)\n", path.c_str());
        strata::core::PinnedArena first(bytes, no_bounds, 0, path, pack_hash);
        check(first.valid() && first.shared_load_owner, "reference arena owns the load");
        check(first.publish_shared_load(), "reference arena published");

        strata::core::PinnedArena wrong_pack(bytes, no_bounds, 0, path, pack_hash + 1);
        if (wrong_pack.valid() || wrong_pack.note.find("pack hash") == std::string::npos) {
            std::printf("  FAIL pack mismatch was not refused: %s\n", wrong_pack.note.c_str());
            ++g_fail;
        } else {
            std::printf("  ok   pack mismatch refused: %s\n", wrong_pack.note.c_str());
        }

        strata::core::PinnedArena wrong_size(bytes * 2, no_bounds, 0, path, pack_hash);
        if (wrong_size.valid() || wrong_size.note.find("expected") == std::string::npos) {
            std::printf("  FAIL size mismatch was not refused: %s\n", wrong_size.note.c_str());
            ++g_fail;
        } else {
            std::printf("  ok   size mismatch refused: %s\n", wrong_size.note.c_str());
        }

        // A different pack must not be able to steal a load that is in progress either: the hash is checked
        // before the state machine, so it is refused instead of waiting for somebody else's arena.
        strata::core::PinnedArena other(bytes, no_bounds, 0, path, pack_hash ^ 0x5555555555555555ull);
        check(!other.valid() && other.note.find("pack hash") != std::string::npos,
              "a mismatched pack is refused even mid-protocol");
        unlink(path.c_str());
    }

    // ================================ 8. a filesystem too small for the arena ================================
    //
    // The shared backing is ON by default, and the filesystem under it may not be able to hold it: Docker's
    // default /dev/shm is 64 MiB.  `ftruncate` on tmpfs does NOT refuse a size it cannot back - it makes a
    // sparse file, and the process then dies of SIGBUS on the first write past the limit, in the middle of the
    // expert load, with no message at all.  So the free space is checked BEFORE the file is created, and a
    // backing that cannot fit is refused like any other unusable one (the caller falls back to a private
    // arena).  This asks for more than the filesystem has and requires: refused, explained, and NO FILE LEFT
    // BEHIND - a half-made 400 GiB hole would be worse than the refusal.
    {
        // `temp_arena` CREATES the file, and an existing file is exempt from the space check (its space is
        // already committed) - so the path is removed again to test the fresh-file case, which is the one that
        // would otherwise SIGBUS.
        const std::string small = temp_arena("too-big");
        unlink(small.c_str());
        const uint64_t far_too_big = 400ull << 30;   // 400 GiB: no filesystem on a desktop PC holds this
        strata::core::PinnedArena huge(far_too_big, std::vector<uint64_t>{}, 0, small, 0x0123456789abcdefull);
        check(!huge.valid(), "a backing larger than the filesystem was accepted");
        check(huge.note.find("cannot hold") != std::string::npos,
              "the too-small backing says which filesystem was short");
        std::error_code ec;
        check(!std::filesystem::exists(small, ec), "a refused backing left a file behind");
        std::printf("  ok   too-small backing refused: %s\n", huge.note.c_str());
        unlink(small.c_str());
    }

    // ================================ 9. across PROCESSES, not just threads ================================
    //
    // Threads share an fd table, so nothing above proves the two things the protocol actually leans on between
    // processes: that `flock` contends between unrelated address spaces, and that a byte one process wrote is
    // visible to another through the FILE.  This case re-runs the binary with `--role` and reads back what that
    // process decided.
    {
        const std::string path = temp_arena("processes");
        std::printf("processes (%s)\n", path.c_str());

        // Does `flock` even contend here?  It must: the whole single-loader guarantee rests on it.  Two fds on
        // the same file in ONE process already contend (unlike POSIX record locks, which merge per process), so
        // this is a cheap direct check of the mechanism rather than a 10 s wait on a foreign lock.
        {
            const int f1 = ::open(path.c_str(), O_RDWR);
            const int f2 = ::open(path.c_str(), O_RDWR);
            check(f1 >= 0 && f2 >= 0, "two fds on the same arena file");
            // `::flock`, not `flock`: <fcntl.h> also declares `struct flock` (POSIX record locks), which makes
            // the unqualified name a type here.
            check(::flock(f1, LOCK_EX) == 0, "first fd takes the exclusive lock");
            const int r = ::flock(f2, LOCK_EX | LOCK_NB);
            check(r != 0 && (errno == EWOULDBLOCK || errno == EAGAIN),
                  "a second fd is refused the lock (flock contends, as the protocol assumes)");
            close(f1);
            close(f2);
        }

        char exe[4096];
        const ssize_t exlen = ::readlink("/proc/self/exe", exe, sizeof exe - 1);
        check(exlen > 0, "own executable path readable");
        if (exlen > 0) {
            exe[exlen] = 0;
            const std::string self(exe);

            // (a) published arena -> a fresh process borrows and sees the sentinel byte.
            strata::core::PinnedArena owner(bytes, no_bounds, 0, path, pack_hash);
            check(owner.valid() && owner.shared_load_owner, "parent process owns the load");
            owner.data()[123] = 0x5a;
            check(owner.publish_shared_load(), "parent publishes");
            {
                const std::string out = run_role(self, path, pack_hash, bytes, "0");
                check(out.find("owner=0 borrowed=1") != std::string::npos,
                      "a SEPARATE process borrows the published arena");
                check(out.find("SENTINEL 90") != std::string::npos,
                      "the separate process reads the parent's sentinel byte (0x5a) from the file");
            }

            // (b) an arena claimed but never published, owner ALIVE -> the other process must wait, then take
            // over.  It must never come back as a borrower: that is the "half-written arena" bug.
            unlink(path.c_str());
            strata::core::PinnedArena mid(bytes, no_bounds, 0, path, pack_hash);
            check(mid.valid() && mid.shared_load_owner, "parent claims a load it will not publish");
            mid.data()[123] = 0x33;
            {
                const std::string out = run_role(self, path, pack_hash, bytes, "600");
                check(out.find("owner=1 borrowed=0") != std::string::npos,
                      "a separate process does NOT borrow an arena that is still being loaded");
                check(field(out, "waited") >= 500, "it waited for the live owner before taking over");
            }
            // The child took the claim over, so the parent must not be able to publish it any more.
            check(!mid.publish_shared_load(), "the parent's claim was taken over, so it cannot publish");
        }
        unlink(path.c_str());
    }

    std::printf("pinned_shared_test: %s\n", g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}
