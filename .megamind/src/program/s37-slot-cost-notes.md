# S3.7 — is a slot a reservation, or does a short conversation cost what it holds?

Owner's question after the `conversation was lost while it was decode … parking budget is too small`
ERR: *"is each slot really reserving max context size for each slot? Might it be that short
conversations really only consume as much of the budget / conversation cache that they actually are?"*

## THE ANSWER (settled, do not re-litigate)

**Two different things, and only one of them was a reservation.**

1. **Parking was never a reservation.** `park_current` sizes a snapshot with
   `conversation_snapshot_bytes(view, …)` where `upto = live.size()` — the tokens the conversation
   actually holds — and `ConversationCache::bytes_` counts `SavedConversation::bytes()` of what was
   stored. `layout()` in `src/core/conversation_snapshot.cpp:40` rounds `upto` up to a page, never up
   to `max_cells`. So a short chat already cost what it held. Log proof (one run, `--max_context
   524288`): 81 tok → 237 413 652 B, 150 tok → 356 515 812 B, 34 015 tok → 1 498 471 712 B,
   95 303 tok → 2 486 870 164 B.
2. **RAM admission WAS a flat per-slot reservation, and that is a real bug.** `generate.cpp` charged
   every request `drv::slot_estimate(budget, --conversation-cache-slots)` = budget/8 (the log's
   "1131 MiB per slot estimate"), independent of the request's length — while `probe.ids.size()` was
   sitting right there in `Request::ids`. A 150-token chat was refused for 1 GiB it would never use.
3. **The startup `ParkCeiling` is a worst case, not a reservation** — it sizes a synthetic
   `max_context`-token view. It is what prints "raise --conversation-cache-mib or lower
   --max-context", which is where the misunderstanding came from.

## THE ACTUAL CAUSE OF THE OWNER'S ERROR (found in the same log, fixed)

**It was not the budget. It was the wrong cache lookup at the hand-over.** `strata-iq3_s.log`
24 986-25 006, budget **10 048 MiB**, one entry parked at **113 MiB**:

```
park: slot 3 parked 80 tokens / 113 MiB; parked=1 entries, 113 MiB of a 10048 MiB budget
swap 65: slot 4 -> 3 ok in 6 ms; saved 0 B, restored 0 B, parked=1 bytes=119295080
slot 3 would step in decode with no conversation to step against - ending it (stage 3 §5.3)
```

`do_swap` looked the outgoing branch up with **`ConversationCache::best()`**, whose prefix rule
requires the query to be **strictly longer** than the entry (`conversation_prefix`:
`n >= prompt.size()` → 0, because a resuming request must still read its last token into the next
verify window). A pre-empted slot's branch is **exactly equal** to what it parked, so the lookup
answered 0, `plan.mount` was false, the adopt hook cleared `live`/`checks` and set
`resumable = true`, `session_established = false`, and `step_gate` correctly ended the request —
with an ERR that blamed the parking budget. **Nothing was ever too small.**

Fix: `ConversationCache::best_exact()` + `serve_swap::seek_mount_index()` / `handover_seeks()`.
`best_exact` searches **live branches only, exact only** — a restore always puts the whole entry
back, so matching a shorter checkpoint would resume the slot past tokens it already generated.
Proved non-vacuous: swapping `best_exact` back for `best` in `seek_mount_index` fails 2 swap checks.

## Measured cost model on this box (from the log, `--max_context 524288`)

`snapshot ≈ fixed floor + rate·tokens + ~110 MiB·checkpoints`

* fixed floor: ~112 MiB (recurrent state + indexer state + draft K/V) — the 81-token park is 226 MiB
  because it also holds one ~113 MiB checkpoint;
* per-token rate: **15.1 KiB/token** from the ceiling (no checkpoints), **~26 KiB/token** from the
  95 303-token park (it carries its checkpoint chain). Use the ceiling rate as the floor of the
  estimate and the measured park rate as the honest one;
* the ceiling at 524288 tokens = **7746 MiB**, budget 9048 MiB → `up to 1 such conversation`.

Do NOT fit a 3-parameter model through the small parks and extrapolate: 81/150/34015-token parks
solve to 29.6 KiB/token and then predict 15 GiB for the ceiling. The parks are not all the same
shape (checkpoint count, and `reuse_kv_bytes` changes what is captured). Quote the two endpoints.

## What landed

* `conversation_cache.hpp`: `conversation_same_branch()` + `ConversationCache::best_exact()`.
* `serve_swap.hpp`: `handover_seeks()`, `seek_mount_index()` — the mount lookup in one testable place.
* `serve_driver.hpp`: `bytes_per_token`, `SlotCost`, `slot_image_bytes(cost, prompt_tokens, max_new)`,
  `ParkCeiling::size_at()/per_token()`, `slot_cost_of()` (rate precedence: **measured park >
  ceiling-implied > budget/slots guess**), `GateEnd` + the two `gate_end_reason` overloads,
  `driver_line(..., per_token_bytes)`.
* `generate.cpp`: `park_per_token` learned from the **largest** snapshot parked so far (a small park
  is dominated by the fixed floor and would under-price a long one); admission prices each request at
  `prompt + max_new` at that rate, capped by the ceiling and by the budget; a refusal now prints the
  prompt length, the price, what is held, free RAM, the floor and which rate was used; `ceiling`
  hoisted above `admit_one` (declared before it, filled just before the loop — the first `admit_one()`
  call is at the loop top, so the reference capture is honest); `do_swap` uses `seek_mount_index` and
  prints a named line when a decode slot's branch is gone; the gate ERR carries
  `SlotConv::parked_bytes` (or the last refused `parkchk.snapshot`) and `conversations.budget()`.
* GENI has no token count on its line → priced at `max_context` (the honest worst case), not 0.
* `docs/DETAILS.md`: new **"Nothing is reserved per slot"** section with the measured table.
* `docs/STAGE3-CONCURRENCY.md`: admission paragraph rewritten + new S3.7 item.
* `--help` for `--conversation-cache-mib` says a parked conversation costs what it holds.

## The second bug found on the way (fixed)

`gate_end_reason(Phase)` hard-coded *"The parking budget is too small for a conversation this long"*
for **every** `Gate::end`. But `step_gate` ends for two unrelated reasons:

* `decode && !resumable` — the image was lost: a parking condition;
* `!session_valid` — the session simply went to another slot: **not** a budget size condition.

The owner's run had a 9048/10048 MiB budget against a 7746 MiB ceiling, so the message sent them to
raise a knob that was already big enough. `GateEnd` now carries the state and the two numbers, and the
text says which cause it is. The 1-arg overload deliberately offers *both* causes rather than guessing.

## Traps

* `ParkCeiling::size_at()` must be called or `per_token()` is 0 and admission silently falls back to
  the guess. `generate.cpp` calls it; the tests call it.
* `slot_image_bytes` returns 0 when nothing is known — that means "no RAM gate", matching
  `admit_fits_ram`'s no-telemetry rule. Do not treat 0 as "refuse".
* Never let the reservation exceed the budget: a price above the budget refuses every request
  forever. That cap is load-bearing and pinned by a test.
* `park_per_token` is engine-thread only (written by `park_current`, read by `admit_one`); both run
  on the one thread that owns the session. Do not read it from the stdin thread.
* `SlotConv::parked_bytes` is the size of the last image that slot parked — the right number for the
  gate message. `parkchk.snapshot` is what a save *would* have cost; the gate prefers the former and
  falls back to the latter.
* **`best()` and `best_exact()` are different questions.** `prep_request` (a NEW request, whose prompt
  is longer than anything parked) keeps `best()`; a HAND-OVER (a slot's own branch, exactly equal)
  must use `seek_mount_index`. Do not "unify" them.

## Verified

`conversation_cache_test` 4232 → **4252**; `serve_swap_test` 156 → **169**; `serve_driver_test`
303 → **340**. ctest 18/19 (only `expert_multi_test`, no AVX-512, pre-existing);
`serve/test_server.py` 65 passed / 2 skipped; serial driver byte-identical (3652 B) and the output
proof PASSes — none of this touches `--serve-slots 0/1`.

Anti-vacuity by mutation: reverting `slot_image_bytes` to the flat `budget/slots` fails **9** checks;
forcing the old blanket budget-blame into the `!session_valid` branch fails **1**; forcing it into the
`!resumable` branch fails **2**; disabling `driver_line`'s rate branch fails **3**; swapping
`best_exact` back for `best` in `seek_mount_index` fails **2** swap checks.
