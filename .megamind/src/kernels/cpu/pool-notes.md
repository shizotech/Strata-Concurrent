# CPU expert pool — cores on a shared machine (stage 1b, done)

Files: `src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/pool.hpp`,
`src/core/session.cpp`.

## The answer to "is there an unnecessary lock?"

**No.** The only mutexes in the tree are the pool's sleep CV, the PLE reader's CV
and load-error mutexes — nothing in the arena read path. tmpfs sharing is not the
cause either.

The cause: every Strata process pinned host→CPU 0 and workers→CPUs 1-5
(`physical_cores(true)` + `session.cpp` hardcoding `cores[0]`). Two processes =
**12 hard-pinned runnable threads on 6 logical CPUs** while the SMT siblings 6-11
sat idle. The pool's `wait_parked`/`wait_done` are `_mm_pause` spins, so a
preempted spinner burns a whole timeslice from the thread doing real work.
That is the "one at 400 % CPU, the other at 80 %" collapse.

Measured (synthetic, real `ExpertPool`, 1,382,400-byte DRAM bursts, no GPU):

| | A | B |
|---|---|---|
| disjoint cores | 41 176 / 36 902 / 41 303 ops/s | 40 956 / 42 322 / 42 002 |
| colliding cores (pre-fix) | 16 220 / 12 881 / 16 759 | 16 261 / 13 574 / 17 351 |

2.5-3x per process; p99 barrier latency 1031 µs → 28 µs.

Park spin: 20 ms of `_mm_pause` is right when you own the core and catastrophic
when you share it (168k → 17.7k batches/5 s). Cut to 200 µs **only** when
`plan.oversubscribed > 0`. `STRATA_POOL_SPIN_US` still wins.

## Public surface

* `struct CorePlan { std::vector<int> host, workers, foreign; std::string note;
  bool shared, leased, overridden; int oversubscribed; }`
* `claim_cores(int want_workers)` — machine-wide claim, idempotent per count.
* `core_plan()` — the claim so far.
* `ExpertPool::note()`, `host_core()`, `worker_cores()`.
  The parent prints `pool.note()` at generate.cpp:2829.
* Env: `STRATA_POOL_CORES=list|none`, `STRATA_POOL_LEASE=path|0`, `STRATA_POOL_SCAN=0`.

Discovery = a flock-guarded lease (`/dev/shm/strata-core-leases`, fallback
`/tmp/...-<uid>`, `fchmod 0666`) **plus** a `/proc` scan of other `strata*`
processes' per-thread single-CPU masks. The scan is what makes it work against a
server still running an older binary. Preference: free physical cores → SMT
siblings → share-and-shout. Lease lines validated by pid **and** `/proc/<pid>/stat`
field 22, so a SIGKILLed server strands nothing.

`session.cpp` pins the host from the plan; with no pool it claims **host only**
(0 workers) so it never strands cores it will not use.

## Verified live on this box

The running server (pid 1017331) holds `0` (host) + `1-5` (workers); a fresh
`claim_cores(5)` returned host 6 + workers 7-11 with
"6 free physical core(s), 1 other strata process(es) on this machine, so this one
took a disjoint set".

## Pitfalls found the hard way

* `"Cpus_allowed_list:"` is **18** characters, not 16. Using the `strncmp` length
  as the value offset parsed `"t:\t2"` → 0 → the scan saw one foreign CPU instead
  of six and the partition silently took the wrong set. The constant is named now.
* `pool_test` / `pool_stress` **SKIP on this machine** (Ryzen 5 5600X, Zen 3, no
  AVX-512). Pre-existing, not a regression — do not read a SKIP as a failure.
  They build clean with `-DSTRATA_BUILD_TESTS=ON` (scratch build `/tmp/a2build`).
* `physical_cores()` was refactored to share `core_groups()` with the scan; a
  differential test against the pre-refactor code confirms identical output for
  the full machine, `0-5`, `6-11`, `3,7,9` and a single-CPU taskset.
