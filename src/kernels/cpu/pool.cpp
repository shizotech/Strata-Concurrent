// src/kernels/cpu/pool.cpp - P2.S3: the CPU expert pool.  Read pool.hpp first; it explains the protocol.
#include "strata/kernels/cpu/pool.hpp"
#include "strata/core/progress.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <immintrin.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
// A2: the cross-process core partition.  `flock` for the lease, `dirent` for the /proc scan, `unistd.h`
// for geteuid/access.  All POSIX, all already in libc - nothing new to link and nothing new in CMake.
#include <dirent.h>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strata::kernels::cpu {

namespace {
constexpr uint64_t pack_head(uint32_t epoch, uint32_t n, uint32_t i) {
    return ((uint64_t) epoch << 32) | ((uint64_t) n << 16) | (uint64_t) i;
}
}  // namespace

namespace {

#if !defined(_WIN32)
/// One logical CPU per physical core is the unit `physical_cores()` hands out; the cross-process partition
/// (A2, below) needs the WHOLE SMT family of each core, so the topology is read once into groups and
/// `physical_cores()` is its first column.  Same order, same first-allowed-CPU-of-each-core choice, so a
/// single process cannot tell the difference - which is what "no single-process regression" means in
/// practice, and it is why the two functions share one reader instead of each parsing sysfs their own way.
std::vector<std::vector<int>> core_groups() {
    std::vector<std::vector<int>> groups;
    auto topo = [](int cpu, const char* what) -> long {
        char path[96];
        std::snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, what);
        long v = -1;
        if (std::FILE* f = std::fopen(path, "r")) {
            if (std::fscanf(f, "%ld", &v) != 1) v = -1;
            std::fclose(f);
        }
        return v;
    };
    // The logical CPUs THIS process may run on.  `sched_getaffinity` is respected, so `taskset` remains a
    // working override and a cset/cgroup cpuset is honoured without any extra knob.
    std::vector<int> allowed;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        for (int i = 0; i < CPU_SETSIZE; ++i)
            if (CPU_ISSET(i, &set)) allowed.push_back(i);
    } else {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) allowed.push_back((int) i);
    }
    std::vector<std::pair<long, long>> seen;
    for (int cpu : allowed) {
        const long pkg = topo(cpu, "physical_package_id"), core = topo(cpu, "core_id");
        int at = -1;
        if (pkg >= 0 && core >= 0) {
            const std::pair<long, long> key{pkg, core};
            for (size_t i = 0; i < seen.size(); ++i)
                if (seen[i] == key) { at = (int) i; break; }
            if (at < 0) { at = (int) seen.size(); seen.push_back(key); }
        } else {
            at = (int) seen.size();     // no sysfs: every CPU is its own core, as before
            seen.push_back({-(long) cpu - 1, 0});
        }
        if ((int) groups.size() < at + 1) groups.resize((size_t) at + 1);
        groups[(size_t) at].push_back(cpu);
    }
    return groups;
}
#endif  // !_WIN32

}  // namespace

std::vector<int> physical_cores(bool skip_first) {
    std::vector<int> cores;
#if defined(_WIN32)
    // Ask the OS rather than assuming a layout.  `hardware_concurrency()` returns LOGICAL processors, and on
    // every SMT machine half of them are siblings - pinning one worker to each of the first N would put two
    // workers on each physical core and halve the bandwidth the expert kernel is bound by.
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) {
        for (unsigned i = 0; i < std::thread::hardware_concurrency(); ++i) cores.push_back((int) i);
    } else {
        std::vector<char> buf(len);
        if (GetLogicalProcessorInformationEx(RelationProcessorCore,
                                             (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) buf.data(), &len)) {
            const char* p = buf.data();
            const char* end = p + len;
            while (p < end) {
                const auto* e = (const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*) p;
                if (e->Relationship == RelationProcessorCore) {
                    const GROUP_AFFINITY& g = e->Processor.GroupMask[0];
                    for (int bit = 0; bit < 64; ++bit)
                        if (g.Mask & (1ull << bit)) { cores.push_back((int) (g.Group * 64 + bit)); break; }
                }
                p += e->Size;
            }
        }
    }
#else
    // The logical CPUs this process may run on, ONE PER PHYSICAL CORE (issue #40): SMT siblings share a core's
    // load/store bandwidth, so a worker on each would put two workers on one core, as the Windows branch above
    // explains.  `core_groups()` is the single reader of that topology - the first CPU of each group, in group
    // order, is exactly what this function has always returned, and the cross-process partition below takes the
    // REST of each group.  One reader, so the two can never disagree about which CPU means which core.
    for (const std::vector<int>& g : core_groups())
        if (!g.empty()) cores.push_back(g[0]);
#endif
    if (skip_first && !cores.empty()) cores.erase(cores.begin());
    return cores;
}

namespace {

void pin_this_thread(int core) {
    if (core < 0) return;
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (core & 63));
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

}  // namespace

long long pin_current_thread(int core) {
    if (core < 0) return -1;
#if defined(_WIN32)
    // `SetThreadAffinityMask` RETURNS the previous mask, or 0 on failure - so 0 doubles as the error, which is
    // why the caller must not treat it as a restorable value.
    const DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) 1 << (core & 63));
    return prev == 0 ? -1 : (long long) prev;
#else
    cpu_set_t prev;
    CPU_ZERO(&prev);
    if (pthread_getaffinity_np(pthread_self(), sizeof prev, &prev) != 0) return -1;
    unsigned long mask = 0;
    for (int i = 0; i < CPU_SETSIZE && i < 64; ++i)
        if (CPU_ISSET(i, &prev)) mask |= 1ul << i;
    pin_this_thread(core);
    return (long long) mask;
#endif
}

void restore_thread_affinity(long long previous) {
    if (previous <= 0) return;
#if defined(_WIN32)
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR) previous);
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int i = 0; i < 64; ++i)
        if ((previous >> i) & 1) CPU_SET(i, &set);
    pthread_setaffinity_np(pthread_self(), sizeof set, &set);
#endif
}

// ==================== A2: THE CORES ARE A MACHINE-WIDE RESOURCE ====================
//
// THE BUG, stated as the owner asked it: "is there maybe any locking in place which is unnecessary?
// Experts are read only, so I don't understand how shared experts over shared memory would cause such a
// slowdown."  The answer is NO - there is no lock in the shared-arena read path, and the shared memory is
// not the cause.  The cause is that the pinning is per-process on a machine-wide resource.  Verified on the
// live machine, not inferred: `grep Cpus_allowed_list /proc/<server>/task/*/status` returns
//
//     1 thread  mask 0        <- the host thread (physical_cores(false)[0])
//     5 threads mask 1..5     <- the expert-pool workers (physical_cores(true))
//    21 threads mask 0-11     <- CUDA/runtime threads, unpinned
//
// Every Strata process produces exactly that.  Two processes therefore put 12 HARD-PINNED runnable
// threads on 6 logical CPUs while the 6 SMT siblings (6..11) sit idle, and the pool's park/done waits are
// `_mm_pause` spin loops - so every preemption of a spinner burns a whole timeslice from the thread that
// is doing the real work.  That is the 400 % / 80 % asymmetry and the collapse; neither process can be
// fast because each is timeslicing against the other on the same six CPUs.
//
// THE FIX IS A PARTITION, NOT A LOCK.  A process finds out which logical CPUs the OTHER strata processes
// on this machine already hold and takes a disjoint set, in this order of preference:
//
//   1. a FREE PHYSICAL CORE (one logical CPU each) - what a lone process gets today, so a single process
//      sees exactly the old assignment and exactly the old rate;
//   2. the SMT SIBLING of a core somebody else is using.  Two threads on one physical core is SMT sharing:
//      each has its own logical CPU, neither is preempted, and they share the core's load/store bandwidth.
//      That is strictly better than 12 threads timesliced over 6 CPUs, which is what happens today;
//   3. only if even that is gone, share a CPU - and SHOUT about it in the note, because a silently
//      degraded configuration is the failure mode this whole file exists to prevent.
//
// WHY TWO DISCOVERY SOURCES.  The lease file is exact but only covers processes built with this code; the
// `/proc` scan also covers a server that is ALREADY RUNNING an older binary.  Without the second source,
// rolling this out would leave the first old server and the first new one colliding exactly as before.
// The scan deliberately looks only at processes whose `comm` starts with "strata": kernel worker threads
// (IRQ, kworker) pin themselves to every CPU on the box, so counting everything would mark all 12 CPUs
// claimed on any machine and force the fallback - a nice way to fix nothing.
//
// WHY A LEASE FILE AND NOT A LOCK.  It is a claim, not a mutex: it is held for the microseconds of a
// read-modify-write under `flock`, never across a load (contrast the arena's header claim in
// src/core/pinned.cu, which must NOT be held for twenty minutes), and it needs no cleanup - a line whose
// pid is dead, or whose pid was recycled (start-time mismatch), is dropped by the next writer.  So a
// SIGKILLed server cannot strand its cores, which is the same liveness argument A1 used for the arena.
namespace {

/// Used by every note below, including the Windows one.
std::string join_cpus(const std::vector<int>& v) {
    std::string s;
    for (int c : v) { if (!s.empty()) s += ','; s += std::to_string(c); }
    return s;
}

#if !defined(_WIN32)

/// `/proc/<pid>/stat` field 22 (start time in ticks).  Field 2 is the comm, which may contain spaces and
/// parentheses, so counting tokens from the start is wrong - count from the LAST ')'.  Same argument as
/// A1's `claim_alive()` in src/core/pinned.cu; a pid alone is not a liveness proof, because a recycled pid
/// would strand the partition behind a process that no longer exists.
long proc_start_ticks(int pid) {
    char path[64];
    std::snprintf(path, sizeof path, "/proc/%d/stat", pid);
    std::FILE* f = std::fopen(path, "r");
    if (f == nullptr) return -1;
    char buf[4096];
    const bool got = std::fgets(buf, sizeof buf, f) != nullptr;
    std::fclose(f);
    if (!got) return -1;
    const char* close_paren = std::strrchr(buf, ')');
    if (close_paren == nullptr) return -1;
    int field = 2;   // the first token after the ')' is field 3
    const char* p = close_paren + 1;
    while (*p != '\0') {
        while (*p == ' ') ++p;
        if (*p == '\0') break;
        ++field;
        if (field == 22) return std::strtol(p, nullptr, 10);
        while (*p != '\0' && *p != ' ') ++p;
    }
    return -1;
}

bool lease_alive(int pid, long start) {
    if (pid <= 0) return false;
    const long now = proc_start_ticks(pid);
    if (now < 0) return false;                 // the pid is gone
    return start <= 0 || now == start;         // start unknown: trust the pid (the conservative answer)
}

/// "0-3,6,9-9" -> {0,1,2,3,6}.  Used for both the lease file and /proc's `Cpus_allowed_list`.
void parse_cpu_list(const std::string& s, std::vector<int>& out) {
    size_t i = 0;
    while (i < s.size()) {
        size_t j = i;
        while (j < s.size() && s[j] != ',' && s[j] != '-') ++j;
        const long a = std::strtol(s.c_str() + i, nullptr, 10);
        long b = a;
        if (j < s.size() && s[j] == '-') {
            ++j;
            b = std::strtol(s.c_str() + j, nullptr, 10);
            while (j < s.size() && s[j] != ',') ++j;
        }
        if (a >= 0 && b >= a)
            for (long c = a; c <= b && c < 4096; ++c) out.push_back((int) c);
        i = (j < s.size() && s[j] == ',') ? j + 1 : j;
        if (i == j) ++i;   // never spin on a malformed separator
    }
}

/// Where the leases live.  /dev/shm is the natural home (it is a tmpfs already, and the arena file proves
/// it is writable here); /tmp is the fallback, suffixed with the uid because /tmp is shared between users
/// and a root-owned lease file would make every later process fall back silently.
/// `STRATA_POOL_LEASE=<path>` relocates it, `0`/`none`/`off` turns the file off (the /proc scan still runs).
std::vector<std::string> lease_paths() {
    if (const char* e = std::getenv("STRATA_POOL_LEASE")) {
        const std::string v = e;
        if (v == "0" || v == "none" || v == "off" || v.empty()) return {};
        return {v};
    }
    std::vector<std::string> out;
    if (::access("/dev/shm", W_OK) == 0) out.push_back("/dev/shm/strata-core-leases");
    out.push_back("/tmp/strata-core-leases-" + std::to_string((long) ::geteuid()));
    return out;
}

struct Lease {
    int pid = 0;
    long start = 0;
    std::vector<int> host, workers;
};

/// Open + LOCK the lease file.  The claim is one critical section: read the other leases, choose our cores,
/// write our line, unlock.  Doing the read and the write under the SAME lock is what makes two servers that
/// start at the same instant take disjoint cores - if they were separate lock acquisitions both would read an
/// empty file and both would take CPU 0..5, which is the bug this file exists to fix.
/// The lock is held for microseconds (a few KB of text), never across a load - contrast the arena header claim
/// in src/core/pinned.cu, which must not be held for twenty minutes.
/// Returns the fd with LOCK_EX held, or -1 (no usable lease file: the /proc scan still runs).
int lease_lock(std::string* path_out) {
    for (const std::string& path : lease_paths()) {
        const int fd = ::open(path.c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0666);
        if (fd < 0) continue;
        // `open`'s mode is ANDed with the umask, so a 022 umask would make the file 0644 and lock a second
        // uid out of the /dev/shm lease (it would fall back to its own /tmp file and lose the exact view of
        // the first uid's cores - the /proc scan still covers it, but the primary path should just work).
        ::fchmod(fd, 0666);
        if (::flock(fd, LOCK_EX) != 0) { ::close(fd); continue; }
        if (path_out != nullptr) *path_out = path;
        return fd;
    }
    return -1;
}

std::vector<Lease> leases_from_fd(int fd) {
    std::vector<Lease> out;
    char buf[4096];
    std::string raw;
    ssize_t n;
    while ((n = ::pread(fd, buf, sizeof buf, (off_t) raw.size())) > 0) raw.append(buf, (size_t) n);
    size_t at = 0;
    while (at < raw.size()) {
        const size_t nl = raw.find('\n', at);
        const std::string line = raw.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        at = (nl == std::string::npos) ? raw.size() : nl + 1;
        if (line.empty() || line[0] == '#') continue;
        int pid = 0;
        long start = 0;
        char h[256] = "", w[512] = "";
        // "pid <p> start <s> host <list> workers <list>"
        if (std::sscanf(line.c_str(), "pid %d start %ld host %255s workers %511s", &pid, &start, h, w) < 3)
            continue;
        Lease l;
        l.pid = pid;
        l.start = start;
        parse_cpu_list(h, l.host);
        parse_cpu_list(w, l.workers);
        if (!l.workers.empty() || !l.host.empty()) out.push_back(std::move(l));
    }
    return out;
}

/// Rewrite the file with our line added.  Rewriting rather than appending is what keeps it a few hundred bytes
/// forever, however many servers start and die, and it is what drops the dead ones.
bool lease_write(int fd, const std::vector<int>& host, const std::vector<int>& workers) {
    std::string body = "# strata expert-pool core leases.  flock-guarded, rewritten on every claim.\n"
                       "# A line is valid while its pid is alive with the recorded start time; a dead\n"
                       "# process strands nothing, because the next writer drops it.\n";
    char buf[4096];
    ssize_t n;
    std::string raw;
    while ((n = ::pread(fd, buf, sizeof buf, (off_t) raw.size())) > 0) raw.append(buf, (size_t) n);
    size_t at = 0;
    while (at < raw.size()) {
        const size_t nl = raw.find('\n', at);
        const std::string line = raw.substr(at, nl == std::string::npos ? std::string::npos : nl - at);
        at = (nl == std::string::npos) ? raw.size() : nl + 1;
        if (line.empty() || line[0] == '#') continue;
        int pid = 0;
        long start = 0;
        char h[256] = "", w[512] = "";
        if (std::sscanf(line.c_str(), "pid %d start %ld host %255s workers %511s", &pid, &start, h, w) < 3)
            continue;
        if (pid == (int) ::getpid() || !lease_alive(pid, start)) continue;
        body += line + "\n";
    }
    body += "pid " + std::to_string((int) ::getpid()) + " start " +
            std::to_string(proc_start_ticks((int) ::getpid())) + " host " + join_cpus(host) +
            " workers " + join_cpus(workers) + "\n";
    if (::lseek(fd, 0, SEEK_SET) != 0) return false;
    if (::ftruncate(fd, 0) != 0) return false;
    const ssize_t wrote = ::pwrite(fd, body.data(), body.size(), 0);
    return wrote == (ssize_t) body.size();
}

/// Every logical CPU pinned by another STRATA process, found by reading /proc.  This is what makes the
/// partition work against a server that is already running a binary without the lease file.
/// Only single-CPU masks count as a claim: the pool pins one CPU per thread, and a thread allowed across
/// "0-11" is not claiming anything.  `comm` is the kernel's 15-character name, which is why the prefix test
/// is "strata" and not an exact match (`strata`, `strata-vision` both count - avoiding a CPU the vision
/// server happens to be pinned to costs nothing).
/// `STRATA_POOL_SCAN=0` turns it off (lease file only).  That is an operator knob and a test knob: it is how
/// you prove the lone-process path still returns exactly `physical_cores(false)`, and it is the escape hatch
/// on a host where reading 400 /proc files at startup is not acceptable.
std::vector<int> foreign_pinned_cpus(std::vector<int>* pids) {
    std::vector<int> cpus;
    if (const char* e = std::getenv("STRATA_POOL_SCAN"))
        if (std::strcmp(e, "0") == 0 || std::strcmp(e, "off") == 0 || std::strcmp(e, "none") == 0) return cpus;
    DIR* d = ::opendir("/proc");
    if (d == nullptr) return cpus;
    const pid_t me = ::getpid();
    dirent* de;
    while ((de = ::readdir(d)) != nullptr) {
        const int pid = std::atoi(de->d_name);
        if (pid <= 0 || (pid_t) pid == me) continue;
        char path[96];
        std::snprintf(path, sizeof path, "/proc/%d/comm", pid);
        std::FILE* f = std::fopen(path, "r");
        if (f == nullptr) continue;
        char comm[64] = "";
        const bool got = std::fgets(comm, sizeof comm, f) != nullptr;
        std::fclose(f);
        if (!got || std::strncmp(comm, "strata", 6) != 0) continue;
        // Its threads.  /proc/<pid>/status is only the thread-group leader, and the pool's workers are NOT
        // the leader, so the per-thread files are the only way to see the real claim.
        std::snprintf(path, sizeof path, "/proc/%d/task", pid);
        DIR* t = ::opendir(path);
        if (t == nullptr) continue;
        bool any = false;
        dirent* te;
        while ((te = ::readdir(t)) != nullptr) {
            const int tid = std::atoi(te->d_name);
            if (tid <= 0) continue;
            char sp[128];
            std::snprintf(sp, sizeof sp, "/proc/%d/task/%d/status", pid, tid);
            f = std::fopen(sp, "r");
            if (f == nullptr) continue;
            char line[512];
            while (std::fgets(line, sizeof line, f) != nullptr) {
                // 18, the length of "Cpus_allowed_list:" - and it is spelled out rather than counted by
                // hand twice over.  An earlier version used 16 (the length of the strncmp pattern), which
                // made the parser read "t:\t2" and return CPU 0 for EVERY thread: the scan then reported
                // one foreign core instead of six, and the partition quietly took the wrong set.  A test
                // caught it; the constant is why the test exists.
                static constexpr int kMaskField = 18;
                if (std::strncmp(line, "Cpus_allowed_list:", kMaskField) != 0) continue;
                std::vector<int> v;
                parse_cpu_list(line + kMaskField, v);
                if (v.size() == 1) { cpus.push_back(v[0]); any = true; }
                break;
            }
            std::fclose(f);
        }
        ::closedir(t);
        if (any && pids != nullptr) pids->push_back(pid);
    }
    ::closedir(d);
    return cpus;
}

#endif  // !_WIN32

/// The process's current claim.  One per process: the pool and the session's host pin must agree, and two
/// claims in one process would fight over the same cores.
CorePlan g_plan;
bool g_plan_claimed = false;
int g_plan_want = -1;
std::mutex g_plan_mu;

CorePlan build_plan(int want_workers, const std::vector<int>& keep_host) {
    CorePlan p;
#if defined(_WIN32)
    // No /proc and no flock.  Keep today's assignment and SAY that the partition is unavailable rather
    // than pretending the two-process case is handled.
    const std::vector<int> fixed = physical_cores(false);
    if (!fixed.empty()) p.host.push_back(fixed[0]);
    for (size_t i = 1; i < fixed.size(); ++i) p.workers.push_back(fixed[i]);
    p.note = "cores: host " + join_cpus(p.host) + " + workers " + join_cpus(p.workers) +
             " (cross-process core discovery is not implemented on Windows: two strata processes on this "
             "machine WILL share cores - separate them with a process affinity or --pool-workers)";
    p.overridden = true;
    return p;
#else
    const std::vector<std::vector<int>> groups = core_groups();
    if (groups.empty()) {
        p.note = "cores: no CPU topology was readable, so nothing is pinned";
        return p;
    }

    // ---- the override / opt-out -------------------------------------------------------------
    // `STRATA_POOL_CORES=0,1,2,3`  -> host 0, workers 1,2,3.
    // `STRATA_POOL_CORES=none`     -> today's fixed assignment, with no discovery and no lease.
    if (const char* e = std::getenv("STRATA_POOL_CORES")) {
        const std::string v = e;
        if (v == "none" || v == "off" || v == "0") {
            const std::vector<int> fixed = physical_cores(false);
            if (!fixed.empty()) p.host.push_back(fixed[0]);
            for (size_t i = 1; i < fixed.size(); ++i) p.workers.push_back(fixed[i]);
            p.overridden = true;
            p.note = "cores: host " + join_cpus(p.host) + " + workers " + join_cpus(p.workers) +
                     " (STRATA_POOL_CORES=none: the cross-process partition is OFF, so this process pins the "
                     "same cores every other strata process pins)";
            return p;
        }
        parse_cpu_list(v, p.host);
        if (!p.host.empty()) {
            p.workers.assign(p.host.begin() + 1, p.host.end());
            p.host.resize(1);
            p.overridden = true;
            p.note = "cores: host " + join_cpus(p.host) + " + workers " + join_cpus(p.workers) +
                     " (STRATA_POOL_CORES)";
            return p;
        }
        p.note = "cores: STRATA_POOL_CORES=\"" + v + "\" parsed to nothing - ignoring it and partitioning "
                 "normally";
        // fall through: a typo must not silently disable the partition
    }

    // ---- discovery + claim, under ONE flock --------------------------------------------------
    // The lock spans the read, the decision and the write.  Two servers that start in the same second must not
    // both read an empty lease file and both take CPU 0..5 - that is precisely the bug - so the decision is
    // inside the critical section.  The section is a few KB of text plus a /proc scan (single-digit ms), so
    // holding it cannot wedge startup.
    std::string lease_path;
    const int lfd = lease_lock(&lease_path);
    std::vector<bool> foreign(4096, false);
    std::vector<int> other_pids;
    if (lfd >= 0) {
        for (const Lease& l : leases_from_fd(lfd)) {
            if (l.pid == (int) ::getpid() || !lease_alive(l.pid, l.start)) continue;
            for (int c : l.host) if (c >= 0 && c < 4096) foreign[c] = true;
            for (int c : l.workers) if (c >= 0 && c < 4096) foreign[c] = true;
            other_pids.push_back(l.pid);
        }
    }
    std::vector<int> scan_pids;
    for (int c : foreign_pinned_cpus(&scan_pids)) if (c >= 0 && c < 4096) foreign[c] = true;
    for (int pid : scan_pids)
        if (std::find(other_pids.begin(), other_pids.end(), pid) == other_pids.end()) other_pids.push_back(pid);
    for (size_t i = 0; i < foreign.size(); ++i)
        if (foreign[i]) p.foreign.push_back((int) i);
    p.shared = !other_pids.empty();

    // ---- the assignment ---------------------------------------------------------------------
    // pass 0: a host CPU this process ALREADY pinned stays first.  `SessionLoopScratch::init` pins the host
    // thread for the whole session and restores the old mask only at teardown, so a later re-claim (a second
    // pool, a different worker count) must not hand the host a different core - the thread would be running
    // on the old one while the plan said the new one.
    // pass 1: one logical CPU from every core NO other strata process is on.  On a quiet machine this is
    // exactly physical_cores(false), in the same order - which is the single-process no-regression claim.
    // pass 2: the remaining CPUs of every core (the SMT siblings).  Not preempted, just bandwidth-shared.
    // pass 3: whatever is left, sharing a CPU with another process.  Reported, never silent.
    std::vector<bool> taken(4096, false);
    std::vector<int> cand;
    for (int c : keep_host)
        if (c >= 0 && c < 4096 && !foreign[c]) { cand.push_back(c); taken[c] = true; }
    for (const std::vector<int>& g : groups) {
        bool contended = false;
        for (int c : g) if (c >= 0 && c < 4096 && foreign[c]) contended = true;
        if (!contended && !g.empty() && !taken[(size_t) g[0]]) { cand.push_back(g[0]); taken[(size_t) g[0]] = true; }
    }
    for (const std::vector<int>& g : groups)
        for (int c : g)
            if (c >= 0 && c < 4096 && !foreign[c] && !taken[c]) { cand.push_back(c); taken[c] = true; }
    for (const std::vector<int>& g : groups)
        for (int c : g)
            if (c >= 0 && c < 4096 && !taken[c]) { cand.push_back(c); taken[c] = true; }

    const int need = 1 + (std::max)(0, want_workers);
    if ((int) cand.size() > need) cand.resize((size_t) need);
    if (!cand.empty()) p.host.push_back(cand[0]);
    p.workers.assign(cand.begin() + (p.host.empty() ? 0 : 1), cand.end());
    // What the assignment ACTUALLY costs, counted over the CPUs we took (not over the candidates - counting
    // candidates would report "SMT siblings" on a machine with spare cores, which is a lie).
    //   used_share : our thread sits on a CPU ANOTHER strata process also pinned -> the OS timeslices them
    //   used_smt   : our CPU is free, but another strata process is on the same PHYSICAL core -> SMT sharing
    //   own_cores  : physical cores nobody else touched
    int own_cores = 0, used_smt = 0, used_share = 0;
    for (int c : cand) {
        if (c >= 0 && c < 4096 && foreign[c]) { ++used_share; continue; }
        bool core_contended = false;
        for (const std::vector<int>& g : groups) {
            bool mine = false;
            for (int k : g) if (k == c) mine = true;
            if (!mine) continue;
            for (int k : g) if (k >= 0 && k < 4096 && foreign[k]) core_contended = true;
            break;
        }
        if (core_contended) ++used_smt;
        else ++own_cores;
    }
    p.shared = p.shared || used_share > 0;
    p.oversubscribed = used_share;

    // ---- publish the claim, still holding the lock -------------------------------------------
    if (lfd >= 0) {
        p.leased = lease_write(lfd, p.host, p.workers);
        ::fsync(lfd);
        ::flock(lfd, LOCK_UN);
        ::close(lfd);
        if (!p.leased) lease_path.clear();
    }

    // ---- the report --------------------------------------------------------------------------
    // How many of OUR threads share a physical core with each other (the machine has fewer cores than
    // this process asked for threads - worth saying, because it is a bandwidth halving the caller did not
    // ask for).
    int self_doubled = 0;
    for (const std::vector<int>& g : groups) {
        int ours = 0;
        for (int k : g) for (int t : cand) if (t == k) { ++ours; break; }
        if (ours > 1) self_doubled += ours - 1;
    }
    std::string why;
    if (used_share > 0)
        why = "the machine is OUT of logical CPUs for this many pinned threads, so " +
              std::to_string(used_share) + " of them SHARE a CPU with another strata process and the OS must "
              "timeslice them";
    else if (used_smt > 0 && p.shared)
        why = std::to_string(used_smt) + " thread(s) sit on the SMT siblings of cores another strata "
              "process is already using - a separate logical CPU each (no preemption), sharing only the "
              "core's bandwidth";
    else if (p.shared)
        why = std::to_string(own_cores) + " free physical core(s), " + std::to_string(other_pids.size()) +
              " other strata process(es) on this machine, so this one took a disjoint set";
    else
        why = std::to_string(groups.size()) + " physical cores, no other strata process on this machine";
    if (used_share == 0 && self_doubled > 0)
        why += "; " + std::to_string(self_doubled) + " of this process's own threads share a physical core "
               "with each other";

    std::string how = p.leased ? "lease " + lease_path
                      : (lfd < 0 ? "no lease file (the /proc scan still ran)" : "lease write FAILED");
    p.note = "cores: host " + join_cpus(p.host) + " + workers " + join_cpus(p.workers) + " (" + why +
             "; " + how + ")";
    if (used_share > 0)
        p.note += " *** cores oversubscribed: run one server per " + std::to_string(groups.size()) +
                  " physical cores, or give this one fewer workers (--pool-workers) or an explicit set "
                  "(STRATA_POOL_CORES) ***";
    return p;
#endif
}

}  // namespace

const CorePlan& claim_cores(int want_workers) {
    std::lock_guard<std::mutex> lk(g_plan_mu);
    if (g_plan_claimed && g_plan_want == want_workers) return g_plan;
    // A re-claim keeps the host CPU this process already pinned (see build_plan's pass 0).
    g_plan = build_plan(want_workers, g_plan.host);
    g_plan_claimed = true;
    g_plan_want = want_workers;
    return g_plan;
}

const CorePlan& core_plan() {
    std::lock_guard<std::mutex> lk(g_plan_mu);
    return g_plan;
}

namespace {
std::atomic<const ExpertPool*> g_diag_pool{nullptr};
void diag_active_pool(std::FILE* f) {
    if (const ExpertPool* p = g_diag_pool.load()) p->diag(f);
}
int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
}  // namespace

void ExpertPool::diag(std::FILE* f) const {
    // The core assignment first: a stalled pool on a machine where two processes were forced to share CPUs
    // is a different diagnosis from a stalled pool on a machine it owns, and the watchdog dump is the only
    // place that distinction survives to a log.
    if (!note_.empty()) std::fprintf(f, "  expert pool %s\n", note_.c_str());
    const uint64_t h = head_.load();
    std::fprintf(f, "  expert pool: epoch %u, batch epoch %u: %u of %u jobs claimed, %u done; %u of %d workers parked, "
                    "%u sleeping; mode %d\n", epoch_.load(), (uint32_t) (h >> 32), (uint32_t) h & 0xffffu,
                 (uint32_t) (h >> 16) & 0xffffu, done_.load(), parked_.load(), n_, sleepers_.load(), mode_);
    std::fprintf(f, "  expert pool threads:");
    for (int i = 0; i < n_; ++i) {
        const int32_t s = wstate_[(size_t) i].load();
        if (s == kParked) std::fprintf(f, " w%d=parked", i);
        else if (s == kSleeping) std::fprintf(f, " w%d=sleeping", i);
        else if (s == kBetween) std::fprintf(f, " w%d=draining", i);
        else std::fprintf(f, " w%d=job%d", i, s);
    }
    const int32_t hs = hstate_.load();
    const char* hn = hs == kIdle ? "idle" : hs == kWaitParked ? "waiting for the workers to park"
                   : hs == kWaitDone ? "waiting for the jobs to finish" : "running a job";
    std::fprintf(f, "; host %s", hn);
    if (hs >= 0) std::fprintf(f, " %d", hs);
    std::fprintf(f, " for %lld ms\n", (long long) (now_ms() - hstate_ms_.load()));
}

ExpertPool::ExpertPool(int n_workers, bool pin, bool host_works) : host_works_(host_works) {
    bool spin_from_env = false;
    if (const char* e = std::getenv("STRATA_POOL_SPIN_US")) {   // a test knob; see kSpinBeforeSleep
        spin_before_sleep_ = std::chrono::microseconds((std::max)(0, std::atoi(e)));
        spin_from_env = true;
    }
    // **THE CORES COME FROM THE MACHINE, NOT FROM A CONSTANT (A2).**  `claim_cores` asks what the other
    // strata processes on this box already pinned and takes a disjoint set; on a quiet machine it hands back
    // exactly `physical_cores(false)` in the same order, so a single process is unaffected.  The count is
    // resolved BEFORE the claim, because the claim is what decides how many workers fit.
    const std::vector<int> fallback = physical_cores(true);
    // An UNPINNED pool claims nothing: it would hold a lease on cores it never uses and starve the other
    // tenant for the whole session, which is worse than the bug this fixes.
    const CorePlan empty_plan;
    const CorePlan& plan = pin ? claim_cores((std::max)(1, n_workers > 0 ? n_workers : (int) fallback.size()))
                               : empty_plan;
    std::vector<int> cores = plan.workers.empty() ? fallback : plan.workers;
    host_core_ = pin ? (plan.host.empty() ? -1 : plan.host[0]) : -1;
    n_ = n_workers > 0 ? n_workers : (int) cores.size();
    if (n_ < 1) n_ = 1;
    // The note is the whole point of A2: the engine adapts to the machine it is on and SAYS what it got.
    // A worker count the plan cannot supply is reported, never quietly accepted.
    if (pin) {
        note_ = plan.note;
        if (n_ > (int) cores.size())
            note_ += " *** asked for " + std::to_string(n_) + " workers but only " +
                     std::to_string((int) cores.size()) + " cores are available to this process: the extra "
                     "workers run UNPINNED, so they will land on top of the others ***";
        // "the pool is smaller than this machine could give a lone process" is only news when the caller did
        // NOT ask for that count.  `--pool-workers 2` is a decision; 2 workers on a 6-core box because the
        // other server took four cores is a degradation, and the contract says a degraded configuration is
        // never accepted silently.
        const int lone = (int) fallback.size();
        if (n_workers <= 0 && n_ < lone)
            note_ += " *** " + std::to_string(n_) + " expert-pool workers, where a lone process on this "
                     "machine would get " + std::to_string(lone) + " ***";
    } else {
        note_ = "cores: not pinned (--pin off / unpinned pool); the host thread is wherever the caller left it";
        host_core_ = -1;
    }
    // **A PARK SPIN IS ONLY POLITE WHEN YOU OWN THE CORE.**  `kSpinBeforeSleep` (20 ms of `_mm_pause`) is the
    // right answer on a machine this process owns: the core is ours anyway, and paying an OS wake-up per layer
    // would be worse.  It is the WRONG answer when the plan had to put a worker on a logical CPU another strata
    // process also pinned: the spinner is then not idle, it is RUNNING, and every timeslice it burns is a
    // timeslice the other tenant's real work did not get.  That is the second half of the two-process collapse
    // - the pinning causes the timeslicing, the spin makes the timeslicing expensive.
    //
    // So the spin adapts ONLY in that case (never when the process simply shares a machine: with a disjoint
    // core set the spin costs someone nothing), and `STRATA_POOL_SPIN_US` still wins, because it is the A/B
    // knob.  MEASURED on this 6-core box: process A runs a DRAM barrier on CPUs 6,7,8 while process B pins to
    // the same three CPUs and runs a verbatim copy of this park loop, published every 2 ms with nothing to do
    // (the shape of a request on a card where most experts live in VRAM, where every publish restarts the
    // spin so the workers never sleep).  B computes nothing; only its spin setting changes.  A's batches / 5 s:
    //     no competitor                      168k / 142k
    //     same CPUs, spin 0 (sleeps)         167k / 151k
    //     same CPUs, spin 200 us             150k /  90k
    //     same CPUs, spin 20 ms               17.7k / 58k   <- 3-9x the loss, and the collapse
    //     SMT siblings, spin 0               150k / 145k
    //     SMT siblings, spin 20 ms            47k / 151k    (noisy: SMT sharing costs issue slots, not timeslices)
    // 200 us is not a magic number: it is ~8x the ~25 us a layer's drain measures here, so the fast path
    // (workers parked, next batch microseconds away) still never pays an OS wake-up, while a genuinely idle
    // worker stops stealing after a fifth of a millisecond.
    if (pin && !spin_from_env && plan.oversubscribed > 0 && spin_before_sleep_ > std::chrono::microseconds(200)) {
        spin_before_sleep_ = std::chrono::microseconds(200);
        note_ += " *** park spin cut from 20 ms to 200 us: this process shares CPUs with another strata "
                 "process, and a spinner on a shared CPU is not idle - it is stealing the timeslice ***";
    }
    worker_cores_.assign((size_t) n_, -1);
    scratch_.resize((size_t) n_);
    wstate_.reset(new std::atomic<int32_t>[(size_t) n_]);
    for (int i = 0; i < n_; ++i) wstate_[(size_t) i].store(kParked);
    hstate_ms_.store(now_ms());
    g_diag_pool.store(this);
    strata::core::diag_pool_fn().store(&diag_active_pool);
    split_.resize((size_t) kMaxSplit);
    split_multi_.resize((size_t) kMaxSplitMulti);
    threads_.reserve((size_t) n_);
    for (int i = 0; i < n_; ++i) {
        const int core = pin ? (i < (int) cores.size() ? cores[(size_t) i] : -1) : -1;
        worker_cores_[(size_t) i] = core;
        threads_.emplace_back([this, i, core] {
            pin_this_thread(core);
            worker(i);
        });
    }
}

ExpertPool::~ExpertPool() {
    const ExpertPool* self = this;
    g_diag_pool.compare_exchange_strong(self, nullptr);
    stop_.store(true, std::memory_order_release);
    // Bump the epoch so a PARKED worker notices the stop flag rather than sleeping through it.
    publish();
    for (auto& t : threads_) t.join();
}

void ExpertPool::publish() {
    // Both sides are seq_cst, and that is the whole lost-wakeup argument: a worker going to sleep does
    // `sleepers_++` and then reads `epoch_`, the host does `epoch_++` and then reads `sleepers_`.  In one total
    // order at least one of them sees the other's write - the worker sees the new epoch and does not sleep, or
    // the host sees the sleeper and notifies under the mutex the worker holds until it is inside `wait`.
    // On x86 the fetch_add is a locked xadd either way, so this costs the token path nothing.
    epoch_.fetch_add(1, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) != 0) {
        std::lock_guard<std::mutex> lk(sleep_mu_);
        sleep_cv_.notify_all();
    }
}

void ExpertPool::worker(int id) {
    uint32_t seen = 0;
    // ARRIVE at the park before the first wait, so `parked_ == n_` is true from construction.  Counting only
    // on the RETURN from a drain leaves `parked_` at 0 until each worker has finished one batch, and the first
    // `run()` - which waits for `parked_ == n_` before publishing - then deadlocks.  It deadlocks on the very
    // first call, which is the good case; a version that deadlocked on the second would be far worse.
    parked_.fetch_add(1, std::memory_order_acq_rel);
    for (;;) {
        // Park: wait for work.  `_mm_pause` rather than a bare spin because it yields the pipeline to the
        // sibling hyperthread; `epoch_` is bumped once per LAYER, not once per expert, so most of these
        // iterations are spent here with nothing to do.
        //
        // **AND NOTHING ELSE HAPPENS IN HERE.**  This loop used to do `pauses_.fetch_add(1)` on every iteration
        // - a locked read-modify-write, five workers against one cache line - so the workers spent their wait
        // invalidating each other's caches and the very line the host writes to publish work.  The counter was
        // diagnostic and nothing branched on it.  See the note on the atomics in pool.hpp.
        //
        // After `kSpinBeforeSleep` with no work the worker sleeps instead (issue #4).  The clock is read once
        // every 1024 pauses, so the spin itself is unchanged.
        const auto parked_at = std::chrono::steady_clock::now();
        uint32_t spins = 0;
        while (epoch_.load(std::memory_order_acquire) == seen) {
            if (stop_.load(std::memory_order_relaxed)) return;
            _mm_pause();
            if ((++spins & 1023u) != 0) continue;
            if (std::chrono::steady_clock::now() - parked_at < spin_before_sleep_) continue;
            std::unique_lock<std::mutex> lk(sleep_mu_);
            wstate_[(size_t) id].store(kSleeping, std::memory_order_relaxed);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            sleep_cv_.wait(lk, [&] {
                return epoch_.load(std::memory_order_seq_cst) != seen || stop_.load(std::memory_order_relaxed);
            });
            sleepers_.fetch_sub(1, std::memory_order_relaxed);
            wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        }
        if (stop_.load(std::memory_order_acquire)) return;
        // acquire: the batch this epoch published (`head`, and the description before it) is visible from here
        seen = epoch_.load(std::memory_order_acquire);
        parked_.fetch_sub(1, std::memory_order_acq_rel);   // leaving the park

        // Drain: one claim per iteration, so a slow worker takes fewer experts and a fast one takes more.
        // Every job is the same size (all experts are 1,382,400 bytes), so there is nothing to schedule.  Only
        // this epoch's jobs: if the host has already moved on, the claims fail and the worker parks again.
        wstate_[(size_t) id].store(kBetween, std::memory_order_relaxed);
        drain(id, scratch_[(size_t) id], seen);
        wstate_[(size_t) id].store(kParked, std::memory_order_relaxed);
        parked_.fetch_add(1, std::memory_order_acq_rel);   // back at the park
    }
}

int ExpertPool::claim(uint32_t epoch) {
    uint64_t h = head_.load(std::memory_order_acquire);
    for (;;) {
        if ((uint32_t) (h >> 32) != epoch) return -1;               // not the batch this thread woke for
        const uint32_t n = (uint32_t) (h >> 16) & 0xffffu, i = (uint32_t) h & 0xffffu;
        if (i >= n) return -1;                                       // exhausted
        if (head_.compare_exchange_weak(h, h + 1, std::memory_order_acq_rel, std::memory_order_acquire))
            return (int) i;
    }
}

uint32_t ExpertPool::begin_batch(int n) {
    if (n < 0 || n > 0xffff) {
        std::fprintf(stderr, "strata: expert pool batch of %d jobs is out of range\n", n);
        std::abort();
    }
    // Every job of the previous batch has completed (`wait_done`), and a claim of it can no longer succeed, so
    // nothing adds to `done` until this batch's first claim - which the release below orders after the reset.
    done_.store(0, std::memory_order_relaxed);
    const uint32_t e = epoch_.load(std::memory_order_relaxed) + 1;   // only the host bumps the epoch
    head_.store(pack_head(e, (uint32_t) n, 0), std::memory_order_release);
    publish();
    return e;
}

void ExpertPool::wait_parked(const char* what) {
    hstate_.store(kWaitParked, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0;
    std::chrono::steady_clock::time_point t0{};
    while (parked_.load(std::memory_order_acquire) != (uint32_t) n_) {
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u) t0 = now;
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "strata: the CPU expert pool stalled %s (%u of %d workers parked) - stopping the engine "
                                 "so the server can start it again (issue #29)\n",
                         what, parked_.load(), n_);
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::wait_done(int n) {
    hstate_.store(kWaitDone, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    uint32_t spins = 0, seen = 0;
    std::chrono::steady_clock::time_point t0{};
    for (;;) {
        const uint32_t d = done_.load(std::memory_order_acquire);
        if (d >= (uint32_t) n) return;                                // `>=`: never a wait that an overshoot outlives
        _mm_pause();
        if ((++spins & 1023u) != 0) continue;
        const auto now = std::chrono::steady_clock::now();
        if (spins == 1024u || d != seen) { t0 = now; seen = d; }     // progress restarts the clock
        else if (now - t0 > kStall) {
            std::fprintf(stderr, "strata: the CPU expert pool stalled: %u of %d jobs done, %u of %d workers parked - "
                                 "stopping the engine so the server can start it again (issue #29)\n",
                         d, n, parked_.load(), n_);
            std::fflush(stderr);
            std::abort();
        }
    }
}

void ExpertPool::drain(int id, ExpertScratch& scratch, uint32_t epoch) {
    (void) id;
    for (;;) {
        const int ci = claim(epoch);
        if (ci < 0) break;
        const uint32_t i = (uint32_t) ci;
        if (id >= 0) wstate_[(size_t) id].store(ci, std::memory_order_relaxed);
        else { hstate_.store(ci, std::memory_order_relaxed); hstate_ms_.store(now_ms(), std::memory_order_relaxed); }
        if (mode_ == 0) {
            const ExpertJob& j = jobs_[i];
            s2_expert_vnni_q(j.blob, *j.act, j.out, scratch);
        } else if (mode_ == 1) {
            const int e = (int) i / parts_a_, part = (int) i % parts_a_;
            const int r0 = FF * part / parts_a_, r1 = FF * (part + 1) / parts_a_;
            s2_expert_gu_rows(jobs_[e].blob, *jobs_[e].act, split_[(size_t) e].ff, r0, r1);
        } else if (mode_ == 2) {
            const int e = (int) i / parts_b_, part = (int) i % parts_b_;
            const int r0 = H * part / parts_b_, r1 = H * (part + 1) / parts_b_;
            s2_expert_down_rows(jobs_[e].blob, split_[(size_t) e].a2, jobs_[e].out, r0, r1);
        } else if (mode_ >= 5) {
            // plan v0.3 P6: native layers, 5 = gate/up rows, 6 = down rows
            const int per = mode_ == 5 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 5 && nfmt_->gu_type == 42) {
                    // a native Q2_0 pack: gate and up rows on the Q2_0 kernels, then SwiGLU
                    thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF];
                    float* gp[MAXT];
                    float* up[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) { gp[t] = gbuf[t]; up[t] = ubuf[t]; }
                    const int nbk = (int) (nfmt_->n_embd / 64);
                    q2_rows_any(mjobs_[e].blob, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, gp, r0, r1);
                    q2_rows_any(mjobs_[e].blob + nfmt_->up_off, nfmt_->gu_row, nbk, mjobs_[e].act, mjobs_[e].nt, up, r0, r1);
                    for (int t = 0; t < mjobs_[e].nt; ++t)
                        for (int r = r0; r < r1; ++r)
                            sb.ff[t][r] = (gbuf[t][r] / (1.f + std::exp(-gbuf[t][r]))) * ubuf[t][r];
                } else if (mode_ == 5) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    native_gu_rows(*nfmt_, mjobs_[e].blob, mjobs_[e].nact, mjobs_[e].nt, ff, r0, r1);
                } else if (nfmt_->d_type == 42) {
                    // Q2_0 down (most IQ layers): the AVX-512 kernel, ggml-cpu has only a scalar one on x86
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    q2_rows_any(mjobs_[e].blob + nfmt_->down_off, nfmt_->d_row, (int) (nfmt_->n_ff / 64), a2,
                                mjobs_[e].nt, mjobs_[e].out, r0, r1);
                } else {
                    const void* hq[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) hq[t] = sb.hq[t];
                    native_down_rows(*nfmt_, mjobs_[e].blob, hq, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        } else {
            // plan v0.3 P6: an equal range of the phase's rows across ALL its experts (a range may span two)
            const int per = mode_ == 3 ? FF : H;
            const int64_t g0 = mrows_ * (int64_t) i / mtasks_, g1 = mrows_ * (int64_t) (i + 1) / mtasks_;
            for (int64_t r = g0; r < g1;) {
                const int e = (int) (r / per), r0 = (int) (r % per);
                const int r1 = (int) std::min<int64_t>(per, r0 + (g1 - r));
                SplitBufMulti& sb = split_multi_[(size_t) e];
                if (mode_ == 3) {
                    float* ff[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) ff[t] = sb.ff[t];
                    s2_expert_gu_rows_multi(mjobs_[e].blob, mjobs_[e].act, mjobs_[e].nt, ff, r0, r1);
                } else {
                    const ActQ* a2[MAXT];
                    for (int t = 0; t < mjobs_[e].nt; ++t) a2[t] = &sb.a2[t];
                    s2_expert_down_rows_multi(mjobs_[e].blob, a2, mjobs_[e].nt, mjobs_[e].out, r0, r1);
                }
                r += r1 - r0;
            }
        }
        done_.fetch_add(1, std::memory_order_release);
    }
}

void ExpertPool::run_phase(int mode, int n_tasks) {
    wait_parked("before a phase");
    mode_ = mode;
    njobs_ = n_tasks;
    const uint32_t e = begin_batch(n_tasks);
    if (host_works_) drain(-1, host_scratch_, e);
    wait_done(n_tasks);
    wait_parked("after a phase");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
}

void ExpertPool::run_split(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplit || n_ == 1 || expert_oracle_q8_0_enabled()) { run(jobs, n); return; }
    const auto t0 = std::chrono::steady_clock::now();
    jobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    // about three tasks per thread in each phase, so the tail is short
    parts_a_ = (std::max)(1, (3 * threads + n - 1) / n);
    parts_b_ = parts_a_;
    run_phase(1, n * parts_a_);
    for (int e = 0; e < n; ++e) act_quant_q8_1(split_[(size_t) e].ff, FF, split_[(size_t) e].a2);
    run_phase(2, n * parts_b_);
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi(ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    if (n > kMaxSplitMulti || expert_oracle_q8_0_enabled()) {
        // one token at a time through the single-token path (the oracle contract has no multi kernel)
        std::vector<ExpertJob> single;
        for (int e = 0; e < n; ++e)
            for (int t = 0; t < jobs[e].nt; ++t) {
                ExpertJob j;
                j.blob = jobs[e].blob;
                j.act = jobs[e].act[t];
                j.out = jobs[e].out[t];
                single.push_back(j);
            }
        for (size_t i = 0; i < single.size(); i += kMaxSplit)
            run_split(single.data() + i, (int) (std::min)((size_t) kMaxSplit, single.size() - i));
        return;
    }
    const auto t0 = std::chrono::steady_clock::now();
    mjobs_ = jobs;
    const int threads = n_ + (host_works_ ? 1 : 0);
    mtasks_ = 3 * threads;
    mrows_ = (int64_t) n * FF;
    run_phase(3, mtasks_);
    const auto t1 = std::chrono::steady_clock::now();
    for (int e = 0; e < n; ++e)
        for (int t = 0; t < jobs[e].nt; ++t)
            act_quant_q8_1(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
    const auto t2 = std::chrono::steady_clock::now();
    mrows_ = (int64_t) n * H;
    run_phase(4, mtasks_);
    const auto t3 = std::chrono::steady_clock::now();
    ms_multi_gu += std::chrono::duration<double, std::milli>(t1 - t0).count();
    ms_multi_q += std::chrono::duration<double, std::milli>(t2 - t1).count();
    ms_multi_down += std::chrono::duration<double, std::milli>(t3 - t2).count();
    multi_bytes += (int64_t) n * (int64_t) BLOB;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run_split_multi_native(const NativeFmt& f, ExpertJobMulti* jobs, int n) {
    if (n <= 0) return;
    const auto t0 = std::chrono::steady_clock::now();
    // more distinct experts than buffers: run them in batches
    for (int b0 = 0; b0 < n; b0 += kMaxSplitMulti) {
        const int nb = (std::min)(kMaxSplitMulti, n - b0);
        mjobs_ = jobs + b0;
        nfmt_ = &f;
        const int threads = n_ + (host_works_ ? 1 : 0);
        mtasks_ = 3 * threads;
        mrows_ = (int64_t) nb * FF;
        const auto a = std::chrono::steady_clock::now();
        run_phase(5, mtasks_);
        const auto b = std::chrono::steady_clock::now();
        for (int e = 0; e < nb; ++e)
            for (int t = 0; t < mjobs_[e].nt; ++t)
                if (f.d_type == 42) act_quant_any(split_multi_[(size_t) e].ff[t], FF, split_multi_[(size_t) e].a2[t]);
                else native_quant_h(f, split_multi_[(size_t) e].ff[t], split_multi_[(size_t) e].hq[t]);
        const auto c = std::chrono::steady_clock::now();
        mrows_ = (int64_t) nb * H;
        run_phase(6, mtasks_);
        const auto d = std::chrono::steady_clock::now();
        ms_multi_gu += std::chrono::duration<double, std::milli>(b - a).count();
        ms_multi_q += std::chrono::duration<double, std::milli>(c - b).count();
        ms_multi_down += std::chrono::duration<double, std::milli>(d - c).count();
    }
    multi_bytes += (int64_t) n * (int64_t) f.bytes;
    mode_ = 0;
    ms_drain_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

void ExpertPool::run(ExpertJob* jobs, int n) {
    if (n <= 0) return;
    if (n_ == 1) {   // no workers: run inline, so a single-core machine still produces a token
        for (int i = 0; i < n; ++i) s2_expert_vnni_q(jobs[i].blob, *jobs[i].act, jobs[i].out, scratch_[0]);
        return;
    }
    // Wait for every worker to be parked BEFORE touching the batch, so the publish below is the only thing
    // that can move a worker into the drain loop.
    //
    // THE THREE PHASES ARE TIMED SEPARATELY.  They were one number, which cannot distinguish a pool that is
    // slow at the WORK from one that is slow at the SYNCHRONISATION - and those need opposite fixes.
    const auto t_a = std::chrono::steady_clock::now();
    wait_parked("before a batch");
    const auto t_b = std::chrono::steady_clock::now();
    jobs_ = jobs;
    njobs_ = n;
    mode_ = 0;
    const uint32_t e = begin_batch(n);   // done, then head (release), then the epoch: the batch is described first

    // ---- **THE HOST DRAINS TOO (R2.2), INSTEAD OF SPINNING ON `done_`.**
    //
    // The loop below used to be `while (done_ != n) _mm_pause();`.  The host is pinned to the core the pool
    // kept for it (`host_core()`; `physical_cores(true)` used to imply core 0, and on a quiet machine it still
    // is) - so for the whole drain that core was idle while five cores did six cores' worth of work.  Measured
    // before the change: 33.7 GB/s against 5/6 x 44.14 = 36.8 for five workers and 44.14 for six.
    //
    // The host claims through the SAME `head_` counter, so this is not a second scheduler and nothing about
    // the ordering changes: `head_` is a single `fetch_add`, every job is the same size, and a thread that
    // arrives late simply claims nothing.  `done_` is still the completion signal and the host still waits for
    // it - what changed is only that the host arrives at that wait having done a share of the work.
    //
    // The host's `done_.fetch_add` is a release for the same reason a worker's is: `j.out` is read by the
    // device after `run()` returns, so the write must be published, not merely performed.
    if (host_works_) {
        for (;;) {
            const int ci = claim(e);
            if (ci < 0) break;
            hstate_.store(ci, std::memory_order_relaxed);
            hstate_ms_.store(now_ms(), std::memory_order_relaxed);
            const ExpertJob& j = jobs_[ci];
            s2_expert_vnni_q(j.blob, *j.act, j.out, host_scratch_);
            done_.fetch_add(1, std::memory_order_release);
        }
    }

    wait_done(n);
    // And park again, so the next `run` starts from a known state.  See the header for why `done` alone is
    // not enough.
    const auto t_c = std::chrono::steady_clock::now();
    wait_parked("after a batch");
    hstate_.store(kIdle, std::memory_order_relaxed);
    hstate_ms_.store(now_ms(), std::memory_order_relaxed);
    const auto t_d = std::chrono::steady_clock::now();

    ms_wait_park_ += std::chrono::duration<double, std::milli>(t_b - t_a).count();
    ms_drain_ += std::chrono::duration<double, std::milli>(t_c - t_b).count();
    ms_repark_ += std::chrono::duration<double, std::milli>(t_d - t_c).count();
}

}  // namespace strata::kernels::cpu
