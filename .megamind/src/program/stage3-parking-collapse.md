# Stage 3 live test: the parking budget collapses and a resumed slot decodes against a foreign session

Observed by the owner on a real run (`--serve-slots 3`, two clients, ~175K-token conversations):
"when a second request comes in, the first one stops for the other's prefill - good. But when it
resumes the older request it just prematurely ends."

## The evidence (strata-iq3_s.log)

```
strata serve: parked-prefix budget 1.6 GiB, not 8.0: only 1.6 GiB of RAM is free above the
             2.5 GiB parking floor and 4.0 GiB of request headroom
strata serve: parked-prefix budget 0.0 GiB, not 8.0: ...          <- collapses over the run
strata serve: conversation cache: skip parking (snapshot 3346 MiB exceeds available budget)
                                                    ... 733 times
strata serve: swap: slot 5 NOT parked - it cannot be resumed and will be re-read from token 0
                (175488 tokens, budget/RAM refused)
strata serve: swap 6: slot 5 -> 6 ok in 3 ms; saved 0 B, restored 0 B, parked=0 bytes=0
strata serve: prompt 176915 tokens = 0 reused + 176915 read in 97107 ms      <- full re-read
```

`slots_active` reached 2 only 5 times; `saved 0 B, restored 0 B` on most swaps.

## Why

1. The 47 GiB expert arena plus the mapped PLE shard leaves ~0-1.6 GiB free, so the *auto* parking
   budget collapses to 0.0-1.6 GiB (`generate.cpp:4000-4006`).
2. A 175K-token conversation's snapshot is ~3.3 GiB. It can therefore **never** be parked here.
3. `swap_to`'s unmount hook fails to park, so it calls `serve_swap::invalidate_unparked(out)`
   (`serve_swap.hpp`) - `out.live_ok = false; out.resumable = false` - and prints the "NOT parked"
   line. The hand-over then **succeeds** with `plan.mount = false`.
4. `h.adopt` (`generate.cpp:5594-5616`) does the destructive part: with `!plan.mount` it clears
   `live`, `live_imgs`, `checks`, `live_ok`, then sets **`in.resumable = true`** for the incoming
   slot. The outgoing slot's conversation is now gone from the session.
5. **The bug.** The outgoing slot's `ReqCtx` is still in `Phase::decode`. Nothing resets it. When
   the scheduler picks it again, `do_swap` (`generate.cpp:7323-7335`) sees `c.resumable == false`
   and skips the mount attempt, `swap_to` runs with no restore, and `dispatch_step` returns
   `Step::decode` because the phase never changed. The slot then runs a verify window against a
   session holding **another conversation's** KV/positional state.

   That is the premature end: the decode loop reads a session that does not describe its tokens, so
   it samples garbage and usually hits EOS immediately. No error is reported, because nothing
   detected it.

## The invariant that is being broken

`docs/STAGE3-CONCURRENCY.md` §5.3 / S3.1d: *a step runs only for the mounted slot, and the mounted
session must describe that slot's conversation.* Step 4 above destroys the outgoing slot's
conversation; step 5 lets it keep decoding anyway.

## CORRECTION — the real root cause (found while fixing, 2026-10-03)

The budget was a **symptom**, not the cause. `park_current`'s first guard is the serve-scope
`live_ok`, and `prep_request` clears `live_ok` for the **whole life of a request**
(`generate.cpp:6540`, "until this request has finished, the session is in between"); only
`finish_request` puts the request's own branch into `live`. So for **every** request in `prefill`
or `decode`, `park_current` returned `skipped` **at any budget whatsoever**. S3.1d's unmount path
then called `invalidate_unparked` and the hand-over reported success.

That is why every swap in the owner's log says `saved 0 B`, why the parking budget was never even
consulted for the case that mattered, and why the resumed slot decoded against a foreign session.

## What landed (S3.6)

1. **`publish_decode_branch(R)`** (`generate.cpp:~6108`) — copies `R.consumed` into `live`/`live_imgs`
   and sets `live_ok` before a hand-over is requested, so the save has something to take. Called at
   both driver swap sites (7796, 8289). `restore_published_branch` puts `consumed` back on the way
   in, or penalties/`finish_request` would park the wrong branch.
2. **The parking guard, asked BEFORE anything is destroyed** — `serve_swap::park_fits` /
   `ParkCheck` / `Outgoing{finished,re_readable,must_park}` and `serve_driver::outgoing_for(phase)`.
   Hook `h.park_guard` runs after the loan return and before `validate`/`unmount`. Two refusal kinds
   handled differently by the driver: `budget_too_small` -> the incoming request gets an `ERR` naming
   both numbers (never parkable, so do not starve or spin); `not_saveable` -> the incoming request
   **waits** (mid-read, resolves itself).
3. **`step_gate(phase, resumable, session_valid)` -> `{run, re_read, end}`** enforced in `run_one`
   (7927). A `decode` slot with no conversation is ended with a named `ERR`; a `prefill` slot is
   genuinely reset by `reset_request_to_token0` — the promise the old log made and never kept.
4. **Admission-time check** (`generate.cpp:~8158`): sizes a worst-case snapshot at `--max_context`
   with the engine's own `conversation_snapshot_bytes` and prints the ceiling before the first
   request, including "up to N such conversations can be swapped".
5. **Logging**: `STRATA_SERVE_TRACE=1` (pick + `Pick::why`, phase transitions, steps, loan defers,
   park waits), unconditional one-line park refusal with both numbers, `resumed from N tokens` /
   `re-reading from token 0` events, and a periodic `activity:` line carrying `peak=` (the old
   `slots_active` line only fired on change, so interleaved and idle looked identical).

## Measured sizing (from the owner's own log)

~**19.5 KiB of snapshot per context token**: 175488 tok -> 3346 MiB, 178336 tok -> 3387 MiB.

| context | one snapshot | 2 parked | 3 parked |
|---|---|---|---|
| 32K | 0.61 GiB | 1.22 | 1.83 |
| 65K | 1.22 GiB | 2.44 | 3.65 |
| 131K | 2.44 GiB | 4.87 | 7.31 |
| 175K | 3.26 GiB | 6.52 | 9.78 |
| 524K | 9.74 GiB | 19.49 | 29.23 |

`--conversation-cache-mib 2048` at a 175K conversation = 3.26 GiB needed: **never parkable**, which
is exactly the configuration that failed.

## Verified

`serve_swap_test` 156 checks, `serve_driver_test` 303 checks, ctest 18/19 (only `expert_multi_test`,
no AVX-512), `serve/test_server.py` 65 passed / 2 skipped, serial driver byte-identical (3652 B),
output proof PASS. Anti-vacuity by mutation: neutering `budget_has_room` fails 14/156 swap checks;
neutering `step_gate`'s decode rule fails exactly "a decode slot whose conversation was destroyed
may not run a window, even while mounted".

## NOT a bug (false alarm, do not chase)

`this CPU cannot run the expert kernel: missing AVX512...` only fires for **non-native** packs
(`generate.cpp:1798`, `if (!native_pack)`). The iq3_s pack is native and runs on AVX2. Both live
servers are fine.

1. **Never let a slot continue in `decode` (or mid-`prefill`) after its state was invalidated.**
   Either reset it to re-read from token 0 (re-derive `read_from`/segments, i.e. re-run the resume
   resolution), or end it with an explicit `ERR`. Continuing is never acceptable.
2. **Refuse to swap out a slot whose state cannot be saved**, rather than swapping and destroying
   it. The session is intact at that point (validate-before-save, S3.1d), so refusing is cheap and
   safe. Then the incoming slot must be answered, not starved: `ERR` it with the reason.
3. **Admission should catch this up front.** With `--serve-slots >= 2`, a conversation whose
   snapshot cannot fit the parking budget cannot be made concurrent at all. Say so at admit time
   with the two numbers (snapshot size vs budget) instead of discovering it 97 seconds later.
4. The owner's box needs RAM: at 175K context a snapshot is ~3.3 GiB, so `--serve-slots 3` needs a
   parking budget of roughly 3x that. `--conversation-cache-mib` is the override.

## Logging the owner asked for

The current log is *almost* enough but not quite. Missing:

- which slot `pick()` chose and **why** (`Pick::why` exists and is dropped on the floor);
- the phase transition of each slot (`queued -> prefill -> prefill_end -> decode -> done`);
- the parking budget vs the snapshot size at the moment of the decision, on one line;
- a per-slot "resumed / re-reading from token 0" line, so a re-read is visible as an event;
- `slots_active` is only printed when it *changes* (`report_active`, `generate.cpp:7600`), so a run
  that never got past 1 looks idle.
