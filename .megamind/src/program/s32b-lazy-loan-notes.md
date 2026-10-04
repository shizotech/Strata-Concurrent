# S3.2b — the prompt loan goes back LAZILY

Files (mine):
```
include/strata/program/prefill_loan.hpp   NEW lever 5: LoanRow, LoanLedger, LazyRefillPolicy,
                                          pump_may_run / pump_batch / lend_must_remark /
                                          narrow_lend_refills_first / may_mark_resident
src/program/prefill_loan_test.cpp         54 -> 131 CPU checks
include/strata/program/serve_driver.hpp   Loan: acquires/releases/handoffs + the "release() is not a
                                          residency event" contract
src/program/serve_driver_test.cpp         230 -> 245 CPU checks
src/program/generate.cpp                  the loan code (lend/refill/refill_one/return_loan_lazy/
                                          settle_pump/pump_loans/drain_loans_idle/reconcile_residency),
                                          PfPart's ledger + loan_live + pump stream/event,
                                          adapt()'s owns() gate, apply_pending()'s drop(),
                                          the switches, --help, the loan lines
bench/prefill/fixed-cost-changes.md       the S3.2b report section
```
`slot.hpp`, `serve_proto.hpp`, `serve_swap.hpp`'s ordering, `src/core/*`, `src/prefill/*`,
`src/kernels/*`, `serve/server.py`, `bench/prefill/analyze.py`: **not touched.**

## The approach: (A) + (C) remove the work, (B) recovers the hit rate

(A) lazy return at `finish_prefill` — nothing copied, rows stay `kNotResident`, a per-cache
**ledger** owns them, the layout (`first_now`, `sp->chunk()`) stays carved.
(C) falls out of (A): the next request's `lend()` finds its range already non-resident, marks
almost nothing, re-lays nothing, and its `finish_prefill` returns nothing. Two slots that both
prefill share one layout for the same reason. **This is where the 4.95 GiB/stage dies.**
(B) the pump: bounded batches (default 16 rows/cache in flight) on each cache's own
non-blocking stream, from the top of every decode step and from the driver's idle pass,
**hottest-row-first** using `drive.d.usage`. A row decode never routes is never copied at all.

Why it is legal at all: **`kNotResident` is a state decode already handles.** A window routes an
expert, finds the row non-resident, sends it to the CPU pool (`expert_source.cpp:1040`,
`:1098-1099`, `Verifier::resident_plan` at `verify.cpp:621`). Decode needs the table TRUE, not
the cache WHOLE. And the loan is the **tail** of a profile-filled cache, so the evicted rows are
the coldest resident experts — the least valuable DMA on the machine.

## The invariant, and the one function per direction

```
(I1) host_res[i] >= 0  <=>  slot host_res[i] on the owning device holds expert i NOW
(I2) at the start of every window, every device's d_res == host_res
```
* out-marking: `lend()` only.
* in-marking: **`settle_pump()` only** (after that cache's event confirms the copies landed),
  plus `refill_one()` (after `sync_queued()`) and `apply_pending()`.
* (I2): `reconcile_residency()` at the top of `run_decode_step()` and `read_windows()`, AFTER
  `pump_loans()` — order matters: uploading before the pump marks would leave the device told a
  row is a CPU miss while the host already sends it to the GPU.

`refill_one()` is **live-loan-only** in both modes: it copies `p.lent` and calls `led.refilled(i)`
per row (never a whole-ledger clear). Under lazy it is reachable only from `lend()` handing a loan
back mid-request when `STRATA_PREFILL_STICKY_LOAN=0`, and that arm must cost what it costs in
0.1.30. There is deliberately **no "drain the whole ledger now" caller in the request path** — the
one place that wants the cache whole is the engine being idle: `drain_loans_idle()`.

## Traps found while writing this (do not repeat)

1. **`lend()`'s "my loan already covers this" short-circuit becomes UNSOUND under lazy.** The
   pump may have returned rows since the last segment; skipping the marking loop carves prompt
   buffers into a slot the devices still believe holds an expert. Lazy mode always re-marks
   (`lend_must_remark`). 24 576 int32 reads per participant per segment — microseconds.
2. **`loan_live` is not `!lent.empty()`.** Under lazy, a lend marks *zero* rows (they were
   already out) but the buffers are still carved there. A pump keyed on `!lent.empty()` would
   copy an expert into a live prompt buffer and then mark it resident. `PfPart::loan_live` is
   set by `lend()` for every participant that can lend, **before any `continue`**.
3. **One batch in flight per cache, enforced.** The confirmation is one event per cache; a second
   batch would be confirmed by the same event and marked resident before its bytes landed.
4. **The adaptive tier and the pump must not both own a row.** `adapt()` treats every
   non-resident row with usage as a promotion candidate — a loan-out row looks exactly like one.
   Promoting it takes some other row's slot as a victim while the pump copies it into its
   ORIGINAL slot; one of the two slots is orphaned forever (`ExpertCache` never re-hands a slot
   out). `adapt()` skips rows `led.owns()`; `apply_pending()` calls `led.drop()`; `pump_loans()`
   drops a row it finds already resident. The reverse hazard (`admit()` stealing a ledger slot)
   cannot happen: `next_free_`/`layer_next_` are always strictly past every slot ever handed out.
5. **A dying holder must clear the ENGINE's loan view, not just the driver's.** `drop_ctx()`
   released `drv::Loan` but left `PfPart::loan_live` true → the pump is refused for every cache
   forever (silent, not fatal). `drop_ctx()` now calls `refill()` first.
6. **`settle_pump` must not mark a row that something else already made resident** — it would
   orphan the newer slot. Guard with `host_res[index] < 0`.
7. **`reconcile_residency()` is lazy-only.** In eager mode nothing marks rows resident except
   `refill_one`/`apply_pending`, which upload themselves; a per-window upload call would be a
   no-op that inflates the `res upload N (M skipped)` bill `analyze.py --compare` reads against
   the S0.3 baseline. Keeping the serial path's counters comparable matters.
8. **`drain_loans_idle` must break when it cannot progress**, not only when the ledger is empty:
   a live loan legitimately refuses the pump, and spinning there burns 250 ms of the thread that
   has to read the next request line. It also bails if a line arrives mid-drain.
9. **`PfPart` gained members** → the two `push_back({...})` aggregate lists silently keep default
   member init but warn; use designated initialisers (`.cache = …, .led = {}`).
10. **GCC 16 false `-Wstringop-overflow`** on `vector<uint8_t>::resize()` growing by a runtime
    amount inside `LoanLedger::mark()`. Use `insert(end, n, 0)`.
11. **`pump(max, usage)` sorts once per loan, not per batch** (`ordered_` flag cleared by
    `take()`). `usage` decays by a constant factor, which cannot reorder anything.
12. **The lazy default is gated to the concurrent driver, not global.** A row still out is
    computed by the CPU instead of the GPU and the two round differently — the same caveat
    `--expert-cache` ships with. The serial path's bar is bit-exactness vs 0.1.30, so it stays
    eager unless `STRATA_PREFILL_LAZY_LOAN=1`.

## Switches

| env | default | effect |
| --- | --- | --- |
| `STRATA_PREFILL_LAZY_LOAN` | unset = on iff `slots_reg.concurrent() && !STRATA_NO_SWAP && conversations.enabled()` | `=0` eager everywhere, `=1` lazy everywhere |
| `STRATA_PREFILL_LOAN_PUMP_ROWS` | 16 | rows per cache in flight; 0 = never pump |
| `STRATA_PREFILL_LOAN_TIMING=1` | off | refill lines say EAGER; each pump batch prints its own; the request line gains `rows out / pumped / still out / loan return lazy|eager`; the driver prints process totals |

## New log lines (analyze.py still does not parse them)

```
strata serve: prompt loan return: LAZY (16 row(s) per cache in flight on the pump; ...)
strata serve: loan lend: 8192 tokens in 0.6 ms (0 relayout, 0 grown, 6481 row(s) now out of the caches)
strata serve: loan request: relayout 0, relayout skipped 3, loan grown 0, rows refilled 0, rows out 0, pumped 0, still out 6481, res upload 1 (2 skipped), loan return lazy
strata serve: loan pump CUDA1: 16 rows (0.03 GiB) queued in 0.1 ms (1955 still out of the cache)
strata serve: loan drain idle: the caches are whole again (1840 ms)
strata serve: prompt loan totals: 14 lend(s), 9 hand-off(s), 1 relayout, 41 relayout skipped, 0 grown, 0 rows refilled eagerly, 6411 pumped home, 0 row(s) still out of the caches (lazy)
```

## Expected effect

Baseline (this file's top): batched path `a = 10 460 ms`. The refill is
4.95 + 3.85 + 3.85 GiB at 2.9 / 1.3 / 24.5 GB/s, **serialised per participant** = **~5.2 s**.
Expect `a` in the **5–6.5 s** region, and `decode tok/s` to move by a few percent (the pump's
DMA shares PCIe; the cold rows run on the CPU pool until they come home). If decode moves more
than a few percent, lower `STRATA_PREFILL_LOAN_PUMP_ROWS` or set it to 0.

## Owner: what only a restart can verify

```sh
STRATA_PREFILL_LOAN_TIMING=1 engine/strata --pack <pack> --serve --serve-slots 3 \
  --max-context 32768 --conversation-cache-mib 8192 --starve-ms 250 2> s32b.log
```
1. `prompt loan return: LAZY …` at startup (if it says EAGER, parking is off or NO_SWAP is set).
2. Two clients, different chats. On the SECOND request's `loan request` line expect
   `rows refilled 0` and `relayout skipped >= 1` — that is the fixed cost gone.
3. `loan pump CUDA*: … queued …` lines interleaved with decode, and `still out` falling.
4. `decode expert cache hit rate` before/after — the honest price of (A).
5. `prompt loan totals:` at shutdown: `hand-offs > 0` with `rows refilled eagerly = 0`.
6. A/B: `STRATA_PREFILL_LAZY_LOAN=0` on the same driver isolates S3.2b from S3.1e-2;
   `--serve-slots 0 STRATA_PREFILL_LAZY_LOAN=1` isolates the loan from the scheduler;
   `STRATA_PREFILL_LOAN_PUMP_ROWS=0` measures (B) separately from (A)+(C).
7. Parity: `STRATA_STATE_HASH=1` with lazy `=0` vs `=1` will DIVERGE (CPU-vs-GPU rounding on the
   rows still out). That is expected and is why the default is gated.

## Verified

`cd build && ninja` clean; `cd /tmp/s3build && ninja` clean; `ctest` 18/19 (only
`expert_multi_test`, pre-existing AVX-512); `prefill_loan_test` 131, `serve_driver_test` 245;
`pytest serve/test_server.py -q` 64 passed / 2 skipped; `--help` renders. No engine start, no
model load, `/dev/shm` untouched.
