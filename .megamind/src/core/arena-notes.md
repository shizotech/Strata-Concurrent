# Shared expert arena — how it works now (stage 1, done)

Files: `src/core/pinned.cu`, `include/strata/core/pinned.hpp`,
`src/core/expert_source.cpp`, `tests/core/file_expert_source_test.cpp`,
`src/core/pinned_shared_test.cpp`.

## The protocol (4 KiB header, in the old `reserved[4]`)

`state` (offset 40, 1 = ready), `owner` (48), `nonce` (56), `owner_start` (64).
`magic` / `version = 1` / `header_bytes = 4096` are unchanged, so an old engine
still reads a new header. An old file (reserved all zero) means **NOT READY** —
never "assume somebody filled it in".

Under an exclusive `flock` on the arena fd:

1. fresh file → `ftruncate` + header with `state = 0`, this pid, its
   `/proc/<pid>/stat` field-22 start time, and a random nonce → **we are the loader**;
2. `state == ready` and the pack hash matches → **borrow**, load nothing;
3. `state == 0` and the claim is alive → **wait** (200 ms polls,
   `STRATA_SHARED_ARENA_WAIT_MS`, default 1200000), following the *current* claim
   in case a third process took it over;
4. claim dead, or the pid was recycled (start time differs), or the budget expired
   → **take the load over** (re-claim under the lock, re-checking for a publish first);
5. the loader writes the body, then a release fence + `pwrite` of `state = 1`.
   A loader whose claim was taken over is **refused** the publish (nonce check).

The lock covers the *decision* only, never the minute-long load. `O_CLOEXEC`
matters: an inherited fd would keep the lock alive across an exec.

## Public surface

`PinnedArena::{shared_borrowed, shared_load_owner, shared_pack_hash,
shared_load_nonce, shared_owner_pid, shared_waited_ms, publish_shared_load()}`.

`ArenaExpertSource::borrowed()` — the driver's "loaded X at Y GiB/s" line must not
print for a borrowed arena.

**Trap (fixed, do not reintroduce):** `publish_shared_load()` sets
`shared_borrowed = true` and clears `shared_load_owner` on success — correct for
the *arena*, wrong for the *caller*. Read the role **before** loading, or the
process that just spent a minute loading reports "nothing loaded".

## Fallbacks (all reported in `note()`, none fatal)

* unwritable / read-only path, another pack's file, **filesystem too small** →
  private anonymous arena. The free-space check runs *before* `ftruncate`: on
  tmpfs `ftruncate` is sparse and succeeds, then the load dies of **SIGBUS** with
  no message. Docker's default `/dev/shm` is 64 MiB.
* `--mmap-experts` has no resident arena to share → the *default* shared path is
  dropped silently; an *explicit* `--shared-expert-arena` + `--mmap-experts` is
  still an error. (Low-RAM installs pass `--resident-experts` ⇒ `--mmap-experts`.)
* Windows: still refuses a shared backing, so the default is `#ifdef _WIN32`-guarded
  to empty. Unconditional default = Windows does not start.
* `mlock`/`lock_resident` is skipped for shared (tmpfs) pages — unevictable
  already, and it fails against `ulimit -l` = 8 MiB here.

## Tests

* `build/pinned_shared_test` — header state machine, dead/recycled owner, old v1
  file, too-small filesystem, pack-hash + size refusals (the strings "pack hash"
  and "expected" are asserted), and a real **cross-process** case (the binary
  re-execs itself with `--role`).
* `build/file_expert_source_test` — `test_shared_arena_borrow` and
  `test_shared_arena_pack_mismatch` at the `ArenaExpertSource` level, synthetic
  6-blob pack, ~0.5 s.
  Gotcha: the mismatch test's marker byte must sit in the **last** layer — the
  first layer's offset is identical between a 2-layer and 4-layer layout, so a
  byte at offset 0 proves nothing.
