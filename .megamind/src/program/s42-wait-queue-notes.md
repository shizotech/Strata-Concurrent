# S4.2 — "hold, don't reject" (stage 4, decision D5)

Owner (`prompt.md`): *"Currently if there is no budget to park a conversation or different
circumstances requests get rejected with an error. That is bad, instead, those requests should be put
on hold and be executed as soon as there are ressources free instead of cancelled."*

Two layers, both needed. The **engine** holds a request it cannot run yet, because only it knows when
a slot / a parked conversation's RAM / the prompt loan comes back. The **server** holds a request the
engine still refuses (an older engine, a second server sharing the machine, or `GpuBusy`).

## The classification of every refusal site (the whole point of the task)

| site | reason | wait or error |
|---|---|---|
| `serve_driver::Refuse::slots_full` | every `--serve-slots` busy | **wait** |
| `serve_driver::Refuse::ram` | no free RAM for another conversation | **wait** |
| `serve_driver::Refuse::registry_full` | no slot row free | **wait** |
| `serve_driver::Refuse::no_parking` | cache off with `--serve-slots>=2` | **error** (config, waiting changes nothing) |
| `serve_swap::ParkRefusal::not_saveable` | outgoing branch mid-read, cannot be saved yet | **wait** |
| `serve_swap::ParkRefusal::budget_too_small` | the snapshot can NEVER fit this cache | **error** |
| `SwapResult::wait_park` | hand-over deferred | **wait** |
| `SwapResult::end_incoming` | the outgoing conversation can never be parked here | **error** |
| `SwapResult::fatal` | a failed restore | **error** (and fatal) |
| `Gate::end` | the conversation is gone | **error** |
| prompt + max_new > `--max-context` | the client can only fix it by sending less | **error** |
| price > free RAM **even if every other slot gave everything back** | cannot fit an empty machine | **error** |
| R7 image exclusivity | one position table | **wait** |
| the prompt loan is held by another slot | R8 | **wait** (deferral queue, pre-existing) |
| `serve/server.py` `GpuBusy` | VRAM below `--min-free-vram-mib` | **wait, opt-in** (`hold_gpu_ms`) — it is a deliberate refusal to grab the card |

`admit_decision(const Admit&, Wait&)` is the single tested predicate that orders these: **permanent
reasons are named before temporary ones**, or a request that can never run queues for ten minutes and
then times out blaming the load.

## Engine API (`include/strata/program/serve_driver.hpp`)

`Hold{run_now,wait,error}` · `Wait{none,slots_full,ram,registry_full,handover_not_yet,loan,
image_exclusive,engine_busy}` · `hold_for()` overloads for `Refuse`/`ParkRefusal`/`SwapResult`/`Gate` ·
`hold_for_slots` · `admit_decision` · `WaitQueue` (bounded FIFO, cap 64, `--hold-ms`, one documented
priority: a request that already holds a parked branch outranks a new one) · `wait_line`
(`WAIT n oldest_ms reason`, untagged, tagged-wire only) · `hold_line` / `hold_admitted_line` /
`hold_expired_line` · `watchdog_sees_waiters() == false`.

**The watchdog must never kill a waiter.** A heartbeat only moves inside a step; a waiter is by
definition not in a step. The hold bound, `STOP <id>` and the client's disconnect are the only things
that end a wait.

## Engine wiring (`src/program/generate.cpp`, done by the parent)

`drv::WaitQueue holds(o.hold_ms)` + `hold_lines` (a waiter keeps its request line, so retrying it
cannot reorder the queue) · `ask_admission`/`AdmitInput` (gathers the facts, delegates the decision) ·
`hold_enqueue` / `promote_waiters` / `expire_waiters` / `hold_cancel` / `hold_expire` / `report_waits` ·
**`drop_ctx` is THE wake point** (`holds.wake_all()` + clear the promotion rate limit) · `waiting=N` in
the activity line · `STOP <id>` cancels a waiter both in `admit_one` and via `pending_cancel` at
promotion (a cancelled waiter is answered `DONE … cancel` and never reaches the engine).

New helper: **`ConversationCache::probe()`** — public, `const`, side-effect-free prefix probe. `best()`
advances the LRU stamp ("finding it is using it"); the queue re-asks every waiter on every pass, so a
probing `best()` would keep a cold conversation warm purely because it is queued.

## Server side (`serve/server.py`)

`HoldQueue` (FIFO, counters, `snapshot()`) · `refusal_is_temporary()` — **default is temporary**,
because the failure mode the owner complained about is a request cancelled for a reason that would
have cleared; only a named list of permanent texts is an error · `_gate_hold()` (enter, wait for the
head, cancel-aware, `HoldExpired` -> 503 `hold_expired` naming what it waited for) · the bounded
re-dispatch of a temporary engine refusal (`hold_retries`, only **before the first token** — a request
that already streamed must not emit its tokens twice) · **`_wait_view()`** is the ONE computed source
(server queue + the engine's `WAIT` line) feeding `/status`, `/slots` (an aggregate `waiting` row with
`id < 0` for server waiters, `count` for the engine's), `/metrics`, `/v1/status`.

`unload()` now asks **busy before capability**: a held request is a request, and answering
`"unsupported"` for an engine that happens not to expose `unload()` hid the fact that something was
queued.

## Traps found

1. A raw `ERR` that bypasses `sp_out` is a **client hang**, not a cosmetic miss (see
   `stage3-wiring-review.md`). Every per-request line goes through `sp_out`.
2. A waiter re-sent by an impatient client must not enter the queue twice — two rows for one id make
   the wire unparseable.
3. `pending_cancel` must be consumed at promotion, and a directly-cancelled waiter must NOT also be
   recorded there, or a later request reusing the id is wrongly cancelled.
4. The promotion path re-reads free RAM (`/proc`), so it is rate-limited (20 ms); `drop_ctx` clears
   the limit so a freed slot promotes immediately.
5. `--serve-slots 0/1` never reaches any of this: the whole block is inside `if (driver_on)`, and the
   serial proof is byte-identical.

## Verified

`cd build && ninja` clean · `serve_driver_test` 395 -> **523** · `conversation_cache_test`
4271 -> **4282** · `pytest serve/test_server.py` **81 passed, 2 skipped** (was 75/6 failed) ·
`.shz_cmd/s31e2_serial_proof.py` byte-identical (3652 B, 56 lines) ·
`.shz_cmd/s31e2_output_proof.py` PASS (109 serial-reachable calls, unchanged) ·
**new gate `.shz_cmd/s42_wiring_proof.py`** — structural, not textual: neutering the wait branch or
the permanent-reason branch or the wake point each makes it fail.

The 6 test failures were 4 fixture bugs (a mock script whose first token is a stop id; `max_tokens`
truncating the script; `out.append` vs `extend`; a gate under-held by one permit) and 2 real defects
(`/status` not merging the engine's `WAIT` line; `unload()` answering "unsupported" while requests
were held). No test was deleted, skipped or weakened.

## Still needs the owner's restart

Three clients against `--serve-slots 2`: the third must be held, then
`slot N ran after waiting X ms`; `waiting=` must appear in the activity line; no `ERR` for a RAM or
slot shortage. `--hold-ms` (default 600000, 0 = wait forever) and `STRATA_HOLD_MS` /
`STRATA_HOLD_RETRIES` are the knobs.
