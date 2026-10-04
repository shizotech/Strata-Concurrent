# S3.1e-2 — the concurrent driver loop

Files (mine):
```
include/strata/program/serve_driver.hpp   NEW  the driver's decisions (pure, CPU-testable)
src/program/serve_driver_test.cpp         NEW  230 CPU checks, incl. 6 loop simulations
CMakeLists.txt                            serve_driver_test under STRATA_BUILD_TESTS
src/program/generate.cpp                  the driver loop + ReqCtx lifetime + watchdog + startup gates
.shz_cmd/s31e2_serial_proof.py            proves the serial driver text is unchanged
.shz_cmd/s31e2_output_proof.py            proves the serial path's output calls are unchanged
```
`slot.hpp`, `serve_proto.hpp`, `serve_swap.hpp`, `src/core/*`, `src/prefill/*`, `src/kernels/*`,
`serve/server.py`, `serve/test_server.py`: **not touched.**

## The shape

`if (driver_on) { ...two-level loop...; return; }` placed **immediately before** 0.1.30's
`while (next_line(line))`. The gate is an early return, so the serial loop's source was never edited.

`driver_on = slots_reg.concurrent() && swaps_on && parking is enabled`.
`STRATA_NO_SWAP=1` therefore falls back to the serial driver (with the tagged wire still on). That is
S3.1d's promise kept: a slot switch *is* a save/restore, so with the hand-over off, "two slots" would
be two conversations overwriting one session.

## ReqCtx lifetime (the hard part)

`std::deque<ReqCtx> live` owned by the driver, keyed by `R.id`. A deque because a reference into it
must survive the next `push_back` (same reason `conv_slots` is one).

New `ReqCtx` fields: `line`/`has_line`, `phase` (`serve_driver::Phase`), `fail` (0/1/2), `finished`,
`loan_held`, `image_bytes`, `loan_waited`, `pp_total/pp_from/pp_next_check/pp_t0`.

Destruction order vs the serial path (mrope → slot → ReqCtx → busy):
* `MropeScope` + **`StepGuard`** armed per STEP in an inner scope of `run_one`; destroyed when that
  scope closes. `drop_ctx()` runs after it, so `~StepGuard` never reads a destroyed `ReqCtx`.
* `StepGuard` = `SlotGuard` + `ctx->finished`. Arming the serial `SlotGuard` per step would end every
  request after its first step; arming it per request is impossible (the request outlives the step).
  **`SlotGuard` itself is untouched** → the serial path's guard is unchanged.
* `progress().busy` is held for the whole driver and released only while blocked for a line.

## Traps found while writing this (do not repeat)

1. **`cur` is serve-scope and is not a counter.** `checkpoint_at` copies `cur[0, L)` and
   `sp.on_chunk` indexes `cur[p0 + t + 1]`. If a step left `cur` holding another request's ids, the
   checkpoints would be saved under this request's id for the *other* sequence — silent garbage.
   `arm_prompt_state()` re-arms `cur`, `pp_*` and `part_next` before **every** prefill step.
   `part_at` is cleared when the prefilling request changes (a split's parts are keyed by position).
2. **`pick()` cannot see the prompt loan**, so prefill-before-decode livelocks: the blocked slot is
   re-picked forever and the holder (a later row in the fixed array) never runs. Fixed with a
   deferral queue + `pick_unblocked()` tie-break.
3. **A queued slot must never be pre-swapped.** `prep_request` owns the request-line hand-over, which
   takes the incoming image out of the cache *before* the outgoing save so the save counts it as
   `held` RAM. Swapping first gets the budget arithmetic wrong and saves/restores twice.
4. **`unwind` must not jump to `finish_request`.** On a request whose decode loop never ran,
   `consumed` is empty and `finish_request` does `live.swap(consumed)` — it wipes the mounted
   conversation. Unwind runs the *phase tail* instead.
5. **A hand-over refills the loan before it can be refused** (`serve_swap::run`'s `return_loan` step
   runs first), so the driver must clear its loan view on a *refused* swap too.
6. **A refused swap leaves `mount_image` taken out of the cache.** It must go back via
   `return_mount_image()`, or the next swap — to a different slot — would restore the wrong state.
7. **The active cap must count contexts that have not got a registry row yet.** The row is created by
   `prep_request` (S3.1c/d own that), so the driver tracks the pending context and counts it against
   `--serve-slots`; otherwise N queued lines each run with no permit.
8. **`refuse_line` must not release a row a live request owns** (the duplicate-id case). It answers
   untagged instead. `end_slot` is the tagged version, for a slot the driver is running.
9. **`run_one` can erase the context** (a rejected prep), so capture the id before calling it.
10. **An idle pick with a held line must not busy-spin** — sleep a tick.
11. `serve_proto::` is not visible inside `generate.cpp`'s anonymous namespace: use
    `strata::program::serve_proto::` or a `namespace proto =` alias.
12. The detached watchdog thread cannot capture locals — the per-slot state is `static` serve-scope.
13. **The loan must be checked BEFORE the hand-over, not after.** `pick()`'s prefill-before-decode rule
    wants slot 2; if the driver swaps and only then finds slot 1 holds the loan, it has paid a full
    save+restore to do nothing, and next pass it pays another one to come back (risk R10). Ask
    `segment_needs_loan()` + `loan.may_lend()` first and defer.
14. **The driver's loan view must follow what `run_prefill_step` actually did**, not what the driver
    guessed: a segment that goes through the verify windows refills first, a batched one lends, and
    "no segment left" touches neither and returns `finished`. Clearing the view in that last case lets
    a second slot lend rows that are still physically lent.
15. **`swap_to` returns true WITHOUT refilling when the slot is already mounted**
    (`swap_needed(mounted_id, incoming)` is false), so the driver may only clear its loan view on a
    hand-over that actually runs — gate on the same predicate.
16. **A hand-over refused even after the retry-without-restore is not transient.** End that slot
    (`ERR <id>` + release) rather than deferring it forever; the session was not written, so the other
    conversations are unaffected. `do_swap` performs S3.1d's own recovery (drop the image, retry with
    no restore) inside the one call, so the driver does not re-derive it at the call site.
17. **A pass that changes nothing must sleep.** The deferral queue and `pick_unblocked()` are the
    mechanisms; `pass_moved` + a 2 ms sleep is the backstop, because a scheduler with several
    independent reasons to skip a step is exactly where a per-reason fix misses one.
18. `drop_ctx()` must wake the loan waiters if the dying slot held the loan, or a dead holder strands
    every prefill in the process.

## Owner: what only a restart can verify

```
# free enough RAM for at least one extra conversation image first (the budget must be > 0)
engine/strata --pack <pack> --serve --serve-slots 3 --max-context 32768 \
              --conversation-cache-mib 8192 --starve-ms 250 2> s31e2.log
```
Look for, in order:
1. `strata serve: concurrent driver on (3 slots; starve 250 ms; slot swap on; parking budget …)`
   — if instead you see `--serve-slots 3 is NOT running concurrently: …`, the cache is off and you
   are on the serial driver.
2. `READY <ctx> stop slots=3`, and `INFO … slots=3 slots_active=0 concurrency=1 slot_swap=1`.
3. Two clients, different chats, both `max_new` ~200. Expect **interleaved** tagged lines:
   `PP … #1`, `T … #1`, `PP … #2` — and `SLOT 1 prefilling …`, `SLOT 1 decoding …`,
   `SLOT 2 queued …`, `SLOT 2 prefilling …`.
4. `strata serve: swap N: slot 1 -> 2 ok in X ms; saved B, restored B …` — that line is **OQ3**;
   grep it at 32k/128k/512k tokens.
5. `strata serve: slot 2 waits for the prompt loan (held by slot 1)` — R8 working, not a bug.
6. `INFO`-style stderr lines `strata serve: slots=3 slots_active=2 …` as the active set changes.
7. `STOP <id>` for one client: that client gets `DONE … cancel #id`, the other keeps streaming.
8. Bit-exactness: run one request at a time with `--serve-slots 3` and `STRATA_STATE_HASH=1`, then
   the same request with `--serve-slots 0`; the `STATE_HASH` lines must match. Then the S3.1d recipe
   (mount A, mount B, mount A → the two A hashes must match).
9. Watchdog: with `STRATA_WATCHDOG_S=5`, stall one slot (a huge prompt) and confirm the other keeps
   streaming; the log should say `… ending those requests instead of the engine (issue #29, R3)`.
