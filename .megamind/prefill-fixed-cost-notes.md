# S0.3 — prefill fixed-cost levers (DONE)

Landed levers 1, 2, 4. Lever 3 deliberately not implemented (see the "why not" in
`bench/prefill/fixed-cost-changes.md`). Stage 3 not touched; the serve loop was not
restructured — every change is local to `lend`/`refill_one`/`res_upload`/`park_current`.

## Files

```
include/strata/program/prefill_loan.hpp   NEW: the three decisions, pure + CPU-testable
src/program/prefill_loan_test.cpp         NEW: 54 checks
bench/prefill/park_backoff_replay.py      NEW: replay lever 4 over a serve log (read-only)
CMakeLists.txt                            prefill_loan_test under STRATA_BUILD_TESTS
src/program/generate.cpp                  the serve loop + switches + --help
bench/prefill/fixed-cost-changes.md       the per-lever report for the owner
```

`src/prefill/*` NOT touched — `relayout`/`carve`/`bytes_needed` already support the
sticky loan, and `Prefill::run()` already rewrites the per-prompt device state
(`mmq::iota`, prefill.cpp:961) and resets `stage_live[]` in `carve()`.

## Switches (read once at startup, like every other STRATA_* knob)

| env | default | reverts to |
| --- | --- | --- |
| `STRATA_PREFILL_STICKY_LOAN` | on | `=0`: refill + re-take the loan even when the layout is unchanged |
| `STRATA_RES_UPLOAD_ALWAYS` | off | `=1`: upload at lend AND at refill, unconditionally (full old behaviour) |
| `STRATA_PREFILL_LOAN_TIMING` | off | `=1`: per-lend / per-refill / per-request loan lines on stderr |
| `STRATA_PARK_REFUSALS` | 3 | `=0`: never back off |
| `STRATA_PARK_QUIET` | 64 requests | window parking stays quiet for |

## Lever 4 semantics — the one that would silently have been a no-op

A request that **starts a conversation** (`resume == 0`) or **mounts a parked one**
(`incoming`) is FORCED: `park_current(held, force=true)` bypasses the quiet window.
It does **not** clear the refusal streak. Only a park that stores does
(`park_backoff.parked()`).

Why: on this box's log ~1 request in 13 starts a new conversation. Clearing the streak
on those means three consecutive refusals essentially never happen and the lever does
nothing. Measured with `bench/prefill/park_backoff_replay.py strata-iq3_s.log`:
**415 parking attempts, 53 % of the estimates skipped** at the default (3/64).

## The two source facts lever 2 rests on (verified by reading, do not re-derive)

1. **The batched prompt path never reads the DEVICE residency table.** `Prefill` holds
   the host pointer and tests `m.host_res[l*n_expert+e] >= 0` at prefill.cpp:1097,
   1586, 1598, 1697, 1729; the CPU pool adapter reads `drive.d.host_res`. The device
   copy is read only by `Verifier::resident_plan` (verify.cpp:621, inside
   `for (l = lb_; l < le_; ++l)`) and by the captured token graph (session.cpp:880,
   serve never captures it — `!multi_gpu` guard, and it is decode-only anyway).
   ⇒ `lend()` does not need an upload; `refill()` does it before any window reads.
2. **A verify window never runs while a loan is outstanding.** The segment loop calls
   `refill()` before every window segment and once after the prompt (also on cancel).

The skip decision is a **content comparison** against a shadow of the last uploaded
table, not a "someone told me" flag — so an unannounced write is still caught. That is
why `apply_pending()` (adaptive swaps, real changes) keeps uploading.

## Lever 1 — what is actually new (be honest with the owner)

The cross-request `relayout` skip **already existed**: `refill_one` never touched
`first_now` or `sp->chunk()`. What S0.3 adds is (a) it is now a named, tested,
switchable invariant, and (b) **the grow path**: a segment needing a wider range no
longer refills the whole loan and re-takes it. `part_slots` is monotone in the chunk,
so the wider range is a superset — the already-lent rows stay lent, the new ones are
added, one relayout.

Grow fires only for two *batched* segments where the later one is wider. It does NOT
fire for the common shape (one batched segment + assistant header through the windows),
because `want <= p.lent_chunk` already covered a narrower second segment. Frequency is
unknown from the log — hence `STRATA_PREFILL_LOAN_TIMING`.

## THE REMAINING FLOOR — what S0.3 does NOT fix

The 4.95 GiB/stage refill at the end of every request. The contract forbids changing
what is refilled or when, and it is already minimal (`p.lent` = only the rows that were
resident, restored into the same slots). Removing it needs either a smaller loan
(lever 3, config, unpriceable from this log) or keeping the loan across requests =
stage 3. **This is the stage-3 argument; `loan refill CUDA*: ... in ... ms` is the line
that proves it.** Expect the fitted `a` to move by hundreds of ms, not by 9 s.

## New log lines (analyze.py does NOT parse these yet)

```
strata serve: loan lend: 768 tokens in 0.4 ms (1 relayout, 0 grown)
strata serve: loan refill CUDA1: 1971 rows (3.85 GiB) in 3120.5 ms
strata serve: loan request: relayout 2, relayout skipped 1, loan grown 1, rows refilled 6481, res upload 0 (2 skipped)
strata serve: conversation cache: 3 consecutive RAM-admission refusals - parking will not re-ask for 64 requests (...)
```
Adding one regex to `analyze.py` per line would make the levers directly comparable
across restarts (S0.2's note asked for exactly this).

## Verified

* `cd build && ninja` clean (the `.sframe` ld lines are pre-existing binutils noise).
* `./build/pinned_shared_test` OK, `./build/file_expert_source_test` PASS.
* Scratch build `/tmp/s03build` (`-G Ninja -DSTRATA_BUILD_TESTS=ON
  -DSTRATA_BUILD_CONVERSATION_TESTS=ON`, CUDA OFF): `ctest` 14/15 pass —
  `prefill_loan_test` 54 checks, `conversation_cache_test` 4225, `conv_cache_test` PASS,
  `conversation_memory_test` 23. The one failure is `expert_multi_test`, which aborts on
  `this CPU cannot run the expert kernel: missing AVX512...` — pre-existing on this
  Zen 3 box, unrelated to these changes.
* `./build/strata --help` renders the new block.
* No engine start, no model load, no request, `/dev/shm` untouched.

## NOT verified (cannot be, on this box)

Any *magnitude* of the loan/refill saving — no run. Those correctness arguments are
source arguments + unit tests. Lever 4's saving IS measured offline (the replay above).

## Gotchas for whoever goes next

* `PfPart` gained 4 counters (`relayouts`, `relayout_skips`, `loan_grows`, `refilled`);
  they are cumulative per process, so the `loan request` line prints the DELTA since the
  request started (`loan_totals()` + the `loan_bill` snapshot). The per-lend and
  per-refill lines are already per-event.
* `include/strata/program/prefill_loan.hpp` needs `<vector>` — `generate.cpp` includes
  it first by luck; a standalone test does not.
* Don't "improve" lever 1 by keeping a bigger layout for a shorter segment. It is
  numerically fine (`Prefill::run` takes `min(T, remaining)`) but it evicts ~30x more
  rows for a short read and refills them over PCIe. The reason is written out in the
  header; don't re-litigate it.
