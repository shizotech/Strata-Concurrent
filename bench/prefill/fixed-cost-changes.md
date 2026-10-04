# S0.3 — the prefill fixed-cost changes (levers 1, 2, 4)

What landed, why each change is behaviour-preserving, what it should save against the
measured baseline, and how to confirm it after a restart.

Baseline: `bench/prefill/baseline/` (`analyze.py --limit 1600 strata-iq3_s.log`),
this box, IQ3_S, 3-GPU layer split, `--prefill auto` → chunk 8192, loan
2 625 + 1 971 + 1 885 = 6 481 slots ≈ 4.95 GiB per stage.

```
prompt_ms = 9547 + 2.719 * fresh            R2 = 0.902   (all requests)
batched path only: a = 10 460 ms, 2.686 ms/fresh
```

**Nothing here was measured by running the engine.** Starting it is forbidden on this
box (two live servers own the RAM and the GPUs). Every claim below is either a source
fact, an arithmetic consequence of a measured number, or an expectation the owner can
check with `analyze.py` after a restart. The `STRATA_PREFILL_LOAN_TIMING` line exists
precisely so that the next restart can measure what is currently only inferred.

---

## What the fixed cost actually is, from the source

Per request, the prompt path's loan does this:

| step | what it costs |
| --- | --- |
| `lend()` per batched segment | marks every resident row in the borrowed slot range non-resident, then `res_upload()` |
| `res_upload()` | `1 + (n_stages - 1)` **synchronous** `cudaMemcpy` of the whole 24 576-entry table — 3 copies on this box's 3-way split |
| the prompt itself | streams the experts the loan evicted (they are the only misses, since ~100 % of the routed mass is resident) |
| `refill()` at the window boundary and at the end of the request | streams the evicted experts back: 4.95 GiB at 2.9 / 1.3 / 24.5 GB/s, then `res_upload()` again |

A request whose prompt has a turn boundary past `resume` splits into up to 4 segments
(`generate.cpp` segment loop), so `lend`/`refill` can run twice in one request: today
that is **9 residency-table copies and one extra full refill** per such request.

`--serve` prints `prompt_ms` *after* the final `refill()`, so all of this is inside the
measured 9.5 s.

---

## Lever 1 — the sticky loan

**Switch:** `STRATA_PREFILL_STICKY_LOAN=0` reverts to today's behaviour. Default: on.

### What changed

`src/program/generate.cpp`, `lend()`:

* The layout test is now the shared predicate
  `prefill_loan::needs_relayout({sp->chunk(), first_now}, {want, first})` — the same
  condition the code already used, but stated once, unit-tested, and named.
* `refill_one()` clears `p.lent` and `p.lent_chunk` and **deliberately leaves
  `p.first_now` and `sp->chunk()` alone**. The old code already did this, which is why
  a request after another request with the same `want` already skipped `relayout()`;
  it was an accident of the field lifetimes rather than a stated rule. It is now the
  documented invariant the sticky loan rests on: giving the *rows* back to the cache
  does not un-carve the *buffers*.
* **New: a segment that needs a wider range no longer refills its loan and takes it
  back.** `part_slots()` is monotone in the chunk, so a wider range starts no later and
  already contains every row on loan (`prefill_loan::loan_grows`). The loan is grown in
  place: the rows already lent stay lent, the newly covered rows are added, one
  `relayout()`.

So the genuinely new saving in lever 1 is the **intra-request grow**. The cross-request
relayout skip was already there; what lever 1 adds is that it is now explicit, tested,
reversible, and reported.

### Why it is safe

* **What is refilled, and when, is unchanged.** Every row in `p.lent` is still restored
  into the same slot before any verify window reads (`refill()` runs at the window
  boundary and at the end of every request, including a cancelled one).
* **The residency set during any decode window is identical.** Growing a loan replaces
  "refill A, then lend B (A ⊆ B)" with "extend the loan from A to B". During segment 2
  the non-resident rows are exactly B either way, and at the end of the request exactly
  B is refilled either way.
* **A reused layout is not a numerics change.** `Prefill::run()` rewrites the MMQ
  identity table at the start of every prompt (`prefill.cpp:961`), and `carve()` resets
  `stage_live[]` for the ring slots of the new layout. Nothing from a previous prompt
  survives in the buffers that a new prompt reads.
* **A narrower range after a wider one still refills first** — `loan_grows()` returns
  false and the old path runs. That is the case where the old range is *not* a subset.
* Each stage still lends only rows for its own layers (`p.lb..p.le`), from its own
  cache, on its own device.

### Expected saving — and its honest size

* `relayout()` itself is cheap: two stream syncs and a bump-pointer carve. Skipping it
  is worth milliseconds. It is done because it is free, correct and now tested, not
  because it is the win.
* **The intra-request refill avoided by growing is the only large number in lever 1**,
  and it is conditional. It fires when a request has two *batched* segments and the
  later one needs a wider range than the earlier one — then today streams the whole
  earlier loan back (up to 4.95 GiB per stage, ~3.8 s on CUDA1's 1.3 GB/s link) and
  evicts it again moments later. It does **not** fire for the common shape
  (one batched segment, then the assistant header through the verify windows), because
  the existing `want <= p.lent_chunk` rule already covers a *narrower* second segment.
  How often the wider-second-segment case happens is not in the log — the log has no
  lend/refill timer — which is exactly why `STRATA_PREFILL_LOAN_TIMING` was added.
  `loan grown` is the column to read.

### What this does NOT fix

The 4.95 GiB per stage that streams back at the end of every request. That refill is
what the contract puts off limits ("do not change what is refilled or when"), and it is
already minimal: `p.lent` holds only the rows that were resident before the lend, and
they go back into the same slots. It is also genuinely required — decode must see a
whole cache between the prompt and the next request.

So the per-request floor after these changes is still dominated by the refill. The two
ways to attack it are out of this task's scope: a smaller loan (lever 3, config, and
the log cannot price it) or keeping the loan across requests, which needs concurrency
(stage 3) because the cache has to be whole for a single-streaming decode loop. If the
owner wants the 9.5 s floor actually gone rather than trimmed, that is the stage-3
argument, and `loan refill CUDA*: ... in ... ms` is the line that proves it.

### How to confirm

```sh
STRATA_PREFILL_LOAN_TIMING=1   # add to the config's args
```

then per request:

```
strata serve: loan lend: 768 tokens in 0.4 ms (1 relayout, 0 grown)
strata serve: loan refill CUDA1: 1971 rows (3.85 GiB) in 3120.5 ms
strata serve: loan request: relayout 2, relayout skipped 1, loan grown 1, rows refilled 6481, res upload 0 (2 skipped)
```

`loan grown > 0` is lever 1's intra-request win; `relayout skipped > 0` is the
cross-request win. Compare `prompt_ms` for the same `fresh` buckets with
`analyze.py --compare`.

---

## Lever 2 — `res_upload()` only when the table changed

**Switch:** `STRATA_RES_UPLOAD_ALWAYS=1` restores today's unconditional upload.
Default: on.

### What changed

`ResidencyUpload` (`include/strata/program/prefill_loan.hpp`) keeps a shadow of the
table content the devices were last given. `res_upload()` compares `host_res` with it
and returns without copying when they are equal.

`lend()` no longer uploads at all: it records the change and returns. The single upload
happens in `refill()`, before any verify window can read the device copy.

### Why it is safe — the two source facts this rests on

1. **The batched prompt path never reads the device table.** `Prefill` holds the *host*
   pointer (`init`'s `host_res`) and decides residency with
   `m.host_res[l * n_expert + e] >= 0` at `prefill.cpp:1097, 1586, 1598, 1697, 1729`;
   the CPU pool's adapter reads `drive.d.host_res`. The device copy is read by
   `Verifier::resident_plan` (`verify.cpp:621`) and by the captured token graph
   (`session.cpp:880`) — both **decode** paths.
2. **A verify window never runs while a loan is outstanding.** The segment loop calls
   `refill()` before any window segment, and `refill()` again after the prompt, before
   the first decode window. A cancelled prompt also falls through to that refill.

And the skip decision itself is a **content comparison**, not a "someone told me" flag:
it is correct even if a writer never announced its change, which is why the adaptive
tier's `apply_pending()` (which really does change the table between decode rounds)
still uploads. 96 KiB of `operator==` is a few microseconds against a synchronous
96 KiB copy per device.

### Expected saving

Today: 3 copies per `res_upload()` call, 2 calls per unsplit request (lend + refill),
3 for a request that splits at a turn boundary → **6–9 synchronous copies per request**,
on links shared with two other serving processes.

After: **0 for a plain request** — the lend/refill round trip restores the table
byte-for-byte, so the end-of-request upload moves nothing. Requests that also ran
adaptive swaps still upload once.

At 96 KiB and a 1.3 GB/s link a single copy is ~0.1 ms of transfer, but it is a
*synchronous* copy on a device that is also being fed 4.95 GiB of refill, and it is a
driver round trip per device. The honest expectation is tens of milliseconds per
request, not seconds — the value of this lever is that it removes a fixed number of
blocking driver calls from the request's critical path.

### How to confirm

`res upload 0 (2 skipped)` on the `loan request` line. With
`STRATA_RES_UPLOAD_ALWAYS=1` the same request reports `res upload 2 (0 skipped)`.

---

## Lever 4 — parking refusal backoff

**Switches:** `STRATA_PARK_REFUSALS N` (default 3, `0` = always ask, i.e. today) and
`STRATA_PARK_QUIET N` (default 64 requests).

### What changed

`park_current()` is called on every request that does not continue from the live
session — in a chat, nearly all of them, because they resume from the checkpoint at the
turn boundary. Before it can answer "no room" it validates the whole view, walks every
retained checkpoint (`conversation_snapshot_bytes` → `view_validate` →
`conversation_checkpoint_validate` per checkpoint), estimates the snapshot and reads
`/proc/meminfo` twice.

After `STRATA_PARK_REFUSALS` consecutive refusals *caused by the physical-RAM admission
floor*, parking stops asking for `STRATA_PARK_QUIET` requests and says so once:

```
strata serve: conversation cache: 3 consecutive RAM-admission refusals - parking will not re-ask for 64 requests (STRATA_PARK_REFUSALS N to change, 0 = always ask)
```

### Why it is safe — parking is quiet, not off

* The window is counted in **requests** and expires by itself.
* Any successful park clears the streak (`park_backoff.parked()`).
* A request that **starts a new conversation** (`resume == 0`) or **mounts a parked
  conversation** (`incoming`) always gets to park, backoff or not — that is the
  "new conversation, a switch" trigger parking has always had, and it is exactly the
  request parking saves. It passes `force = true`.
* Those requests do **not** clear the refusal streak. That is deliberate and it is the
  difference between a lever and a no-op: on this workload about one request in thirteen
  starts a new conversation, so clearing the streak on them would mean three
  consecutive refusals essentially never happen and nothing is ever saved.
* Only the two `conversation_memory_admit` refusals count. The
  "snapshot exceeds available budget" refusal is a different decision and still runs,
  because it costs nothing (the estimate was already computed for it).
* Nothing about what gets parked, or when it is restored, changes: the backoff only
  skips the *estimate* on requests where the estimate was already known to refuse.

### Expected saving — measured offline, not guessed

The log records every park and every refusal in order, so the state machine can be
replayed against it without starting anything:

```sh
$ python3 bench/prefill/park_backoff_replay.py strata-iq3_s.log
2251 requests, 429 parking attempts: 253 refused, 176 parked, 0 budget-refused; 72 forced (new conversation / mount)
 refusals  quiet   paid   skipped
        3     64    205    224 (52%)  engine default
        0      0    429      0 (0%)  off (today)
```

**About half the parking estimates on this box's own traffic never have to be paid.**
The log is live, so the counts move every time you run it; the ratio is the result.
Each one is a whole-view validation, a walk of up to 6 retained checkpoints, and two
`/proc/meminfo` reads. In absolute time that is still milliseconds per request — this is
the smallest of the three levers, and it is not aimed at the 9.5 s floor.

**Context this lever does not change:** parking only exists in the last two restarts of
the baseline log. It cannot explain the floor in the other 28.

### How to confirm

`analyze.py` already counts `skip parking (physical RAM admission …)` lines
(`RE_REFUSE`). After a restart with the backoff on, expect that count to fall by roughly
half in the parking-enabled restarts, and one
`parking will not re-ask for 64 requests` line per backoff. The new line is not parsed
by `analyze.py`; the refusal count is, and that is the column that moves.

---

## Lever 3 — pricing the chunk: deliberately NOT implemented

The task allows lever 3 only "if you can do it without changing which chunk
`--prefill auto` picks by more than one step". I did not implement it, and the reason
is a source fact that makes the naive version wrong rather than merely speculative:

**The loan's width is a function of the chunk, not of the segment.** `lend()` computes
`first = max(p.first, slots - part_slots(p, want))`, so a request already lends only
what its own segment needs. The *startup* chunk only sets the ceiling and the buffer
size `Prefill::init` carves. Short reads therefore already pay a short loan — the
`want`-bucketed table in `bench/prefill/README.md` shows exactly that (a 256-token loan
evicts on the order of a hundred CUDA0 slots, not 2 625; the exact figure is
`part_slots(p, 256)`, which the log does not print).

What a smaller `--prefill` would change is the **per-token rate**, not the fixed cost:
`ring_slots()` drops from 384 to 96 below `STRATA_PREFILL_STREAM_MIN` (1024), and
`bytes_needed()` shrinks. That is a different lever from the 9.5 s floor, and the log
cannot price it (a request never lends more than it needs, so the log only contains
loans that were taken). Picking 4096 or 2048 by guesswork would move the marginal rate
in a direction nobody has measured, and it changes which kernels run for a chunk —
`prefill.cpp:78` records that below ~1 000 tokens the *output changed* on Q2_0.

The right instrument for lever 3 is `bench/prefill/ab-prefill.sh --run --variant
'p4096:--prefill 4096'`, at a restart, on a machine with the RAM for it. It is a config
decision for the owner, not a code change, and the code should not pre-empt it.

---

## What I could not verify

* **No engine run.** No `--serve`, no model load, no requests. Everything about
  *magnitudes* is expectation, not measurement. The correctness arguments are source
  arguments plus unit tests.
* **The refill is still the dominant fixed cost and this task does not remove it.**
  The contract forbids changing what is refilled or when, and that is the honest limit:
  the 4.95 GiB per stage still streams back at the end of every request. Removing *that*
  needs either a smaller loan (lever 3, config) or concurrency (stage 3), where the
  loan can be taken once and kept across requests.
* **`relayout()`'s real cost is unmeasured.** If it turns out to be much more than the
  two stream syncs and the bump carve it looks like, the sticky-layout skip is worth
  more than stated here.
* **The `grow` path's frequency is unknown** — the log has no lend/refill timer. The
  timing line is the instrument for it.

## Switches added (all read once at startup, like every other `STRATA_*` knob)

| env | default | `=0`/`=1` reverts to |
| --- | --- | --- |
| `STRATA_PREFILL_STICKY_LOAN` | on | `=0`: refill and re-take the loan even when the layout is unchanged |
| `STRATA_RES_UPLOAD_ALWAYS` | off (skip when unchanged) | `=1`: upload at lend and at refill, unconditionally |
| `STRATA_PREFILL_LOAN_TIMING` | off | `=1`: per-lend, per-refill and per-request loan lines on stderr |
| `STRATA_PARK_REFUSALS` | 3 | `=0`: never back off |
| `STRATA_PARK_QUIET` | 64 requests | window parking stays quiet for |

Documented in `--help` under `--prefill`.

## Files

```
include/strata/program/prefill_loan.hpp   new: the three decisions, pure and testable
src/program/prefill_loan_test.cpp         new: 52 CPU-only checks
CMakeLists.txt                            prefill_loan_test under STRATA_BUILD_TESTS
src/program/generate.cpp                  the serve loop: lend/refill/res_upload/park_current,
                                          the switches, the loan lines, the --help text
bench/prefill/park_backoff_replay.py      new: replay lever 4 over a serve log (read-only)
bench/prefill/fixed-cost-changes.md       this file
```

`src/prefill/*` was **not** touched: `relayout`, `carve` and `bytes_needed` already
support everything the sticky loan needs, and `Prefill::run()` already rewrites the
per-prompt device state (`mmq::iota`) that a reused layout might otherwise stale.

---

# S3.2b — the loan goes back LAZILY (stage 3, concurrency makes it possible)

Everything above this line is S0.3. This section is the lever S0.3 explicitly left on the
table: **the 4.95 GiB-per-stage end-of-request refill.**

Baseline for comparison is the same one at the top of this file:

```
all requests:      prompt_ms = 9547 + 2.719 * fresh            R2 = 0.902
batched path only: a = 10 460 ms, 2.686 ms/fresh               R2 = 0.905
```

**Nothing in this section was measured by running the engine either.** Starting it is still
forbidden on this box. The saving below is arithmetic on link speeds that *were* measured,
plus a source argument; `STRATA_PREFILL_LOAN_TIMING=1` is the instrument that turns it into
a measurement at the next restart.

## What the refill costs, from the startup lines in the baseline log

```
strata serve: the prompt path borrows 2625 CUDA0 cache slots (4.95 GiB)
strata serve:   CUDA1 prompt path borrows 1971 of its 9216 slots (3.85 GiB)
strata serve:   CUDA2 prompt path borrows 1885 of its 7168 slots (3.85 GiB)
```

`refill_one()` copies each of those rows back from the arena into the same slot on the
legacy stream and then `sync_queued()`s that stream **per participant**, so the three
stages are serialised, not overlapped. At the link speeds `bench/prefill/README.md`
measures (2.9 / 1.3 / 24.5 GB/s):

| stage | bytes | link | refill |
| --- | --- | --- | --- |
| CUDA0 | 4.95 GiB | 2.9 GB/s | ~1.83 s |
| CUDA1 | 3.85 GiB | 1.3 GB/s | ~3.18 s |
| CUDA2 | 3.85 GiB | 24.5 GB/s | ~0.17 s |
| **total, serialised** | **12.65 GiB** | | **~5.2 s** |

That is roughly **half of the batched path's 10.46 s fixed term**, and it is paid once per
request (twice if the prompt splits at a turn boundary, because the window segment refills
before it reads). It is the single largest remaining fixed cost, and S0.3's honest reason
for not touching it was that a single-streaming request has to hand the cache back before
its own decode loop.

## Why concurrency removes it rather than hiding it

`kNotResident` is a legal state that the decode path already handles. A verify window
routes an expert, looks the row up in the residency table, and if it is not resident sends
that row to the CPU pool (`expert_source.cpp:1040` for the window's GPU/CPU split,
`:1098-1099` for the fallback path, `Verifier::resident_plan` at `verify.cpp:621` for the
device copy). **Nothing in decode requires the cache to be whole. It requires the table to
be true.** So the refill was never a correctness requirement — it is a hit-rate choice, and
a hit-rate choice can be deferred, spread, or skipped.

The second fact that makes this a *removal* and not a *move*: the loan is the **tail** of a
profile-filled cache, and the profile fills hottest-first. The rows the prompt evicts are
therefore the **coldest resident experts**, which is exactly the subset decode is least
likely to route. Leaving them out costs little, and copying them home is the least urgent
DMA the machine can do.

## The lever, in three parts

**(A) Lazy return.** `finish_prefill()` copies nothing back. The rows stay marked
non-resident and go into a per-cache **ledger** (`prefill_loan::LoanLedger`); the layout
(`first_now`, `sp->chunk()`) stays carved.

**(C) The loan survives the request — and across slots.** Because nothing was copied back,
the next request's `lend()` finds its range already non-resident: it marks almost nothing,
re-lays nothing, and its `finish_prefill` returns nothing. Two slots that both want to
prefill share one loan layout for the same reason. This is where the 5.2 s actually dies.

**(B) The pump.** The ledger is drained in bounded batches, on each cache's own non-blocking
stream, from the top of every decode step and from the driver's idle pass, **hottest-row-first**
using the engine's own routing-usage signal. A row decode never routes is never copied at all.

(A) and (C) remove the work. (B) recovers the hit rate off the critical path.

**A second win this unlocks, for free.** `serve_swap::run()`'s step 2 is `return_loan`, and it
calls the same `refill()`. Under the eager rule **every slot hand-over paid the full 5.2 s
refill before it could save the session** — S3.1d's own note flags that a hand-over must not be
taken while the loan is held, and the driver defers slots for exactly that reason. With the lazy
return a hand-over's `return_loan` step is a 96 KiB table upload instead of 12.65 GiB of DMA, so
swaps get cheaper by the same amount prompts do. That is the loop between stage 0 and stage 3 the
task asked to close: the loan stops being a per-request cost *and* stops being a per-swap cost.

## The invariant, and the code path that enforces it

```
(I1)  host_res[i] >= 0   <=>   cache slot host_res[i] on the owning device holds expert i's bytes NOW
(I2)  at the start of every verify window, every device's d_res equals host_res
```

**Where (I1) is enforced, by name:**

* `lend()` — a row is marked `kNotResident` **before** its slot is used by the prompt path.
  Marking out is always the safe direction.
* `settle_pump()` — the **only** place a lazily-returned row becomes resident again, and it
  runs only after `cudaEventSynchronize`/`cudaEventQuery` on that cache's own event has
  confirmed the copies landed. It also refuses to mark a row that something else already
  made resident.
* `refill_one()` — the eager path: `fill_slot_*` then `sync_queued()` **then** the marking.
* `pump_may_run()` (`prefill_loan.hpp`) — the pump never queues a batch for a cache whose
  loan is **live** (`PfPart::loan_live`), never queues a second batch while one is
  unconfirmed (one event per cache), and never runs in eager mode.
* `lend()` calls `settle_pump(p, /*wait=*/true, e)` first, so a batch queued during the
  previous request's decode has landed before the buffers are carved over those slots.
* `adapt()` skips any row the ledger owns (`LoanLedger::owns()`), and `apply_pending()`
  calls `led.drop()` — otherwise the adaptive tier and the pump would both move the same
  expert and one of the two slots would be orphaned forever (`ExpertCache` never re-hands a
  slot out).

**Where (I2) is enforced:** `reconcile_residency()` at the top of `run_decode_step()` and of
`read_windows()`, in that order after `pump_loans()` (which is what may have moved the
table). `res_upload()` decides by **content**, so it cannot be fooled by a writer that
forgot to announce itself, and it costs a 96 KiB compare when nothing moved.

**The adversarial case, pinned by a test:** *a row is marked non-resident and a decode
window asks for it.* The window routes it to the CPU pool — correct, and the ordinary state
for every expert the cache never held. The test also asserts the reverse is detectable: a
row marked resident over a slot holding scratch is exactly what the ordering above makes
unreachable (`prefill_loan_test.cpp::test_decode_window_never_reads_a_slot_that_does_not_hold_its_expert`).

## One soundness change this forces inside `lend()`

Under the lazy rule `lend()` may **no longer** short-circuit with "my current loan already
covers this chunk". The pump may have walked some of those rows home since the last
segment, and a row the pump returned is resident again — skipping the marking loop would
carve prompt buffers into a slot the devices still believe holds an expert. So in lazy mode
the marking loop always runs: 24 576 `int32` reads per participant per segment,
microseconds, against the 3.18 s it protects. Eager mode keeps 0.1.30's short-circuit.
This is `prefill_loan::lend_must_remark()`.

## Switches

| env | default | reverts / does |
| --- | --- | --- |
| `STRATA_PREFILL_LAZY_LOAN` | unset = **on under the concurrent driver**, off on the serial path | `=0` forces eager everywhere; `=1` forces lazy everywhere (the A/B arm for `--serve-slots 0/1`) |
| `STRATA_PREFILL_LOAN_PUMP_ROWS N` | 16 rows per cache in flight | `0` = never pump: a ledger is drained only when something needs the cache whole |
| `STRATA_PREFILL_LOAN_TIMING=1` | off | extended: refill lines say `EAGER`, each pump batch prints its own line, the per-request line splits rows into `refilled / out / pumped / still out`, and the driver prints process totals |

**Why the default is gated and not simply "on".** Lazy return is not only a timing change.
A row still out of the cache is computed by the CPU instead of the GPU, and the two round
differently — the same caveat `--expert-cache` ships with (`expert_cache.hpp`'s parity
note, and the 95-98 % top-1 agreement measured in `bench/results/2026-09-27-cache-parity`).
The serial path's acceptance bar is bit-exactness against 0.1.30, so the default keeps that
bar; `=1` lets the owner measure the lazy path there anyway. Under the concurrent driver the
loan is already a process resource and the swap already moves conversations between slots,
which is where the saving is and where the bar has already moved.

`--serve-slots 0/1` is otherwise untouched: `slots_reg.concurrent()` is false, the driver's
early return is not taken, and `loan_policy.lazy` resolves to false, so the serial path
refills exactly as 0.1.30 did and its `res upload N (M skipped)` bill stays comparable with
the baseline log.

## Expected effect on the fitted fixed term

For the **batched path** (`a = 10 460 ms` today):

* the refill leaves the prompt's critical path: **−5.2 s** of serialised DMA at the
  baseline loan size, so expect `a` in the region of **5–6.5 s**;
* `lend()` also gets cheaper on every request after the first (no new rows to mark, no
  relayout when the chunk matches), worth milliseconds, not seconds;
* the cost reappears as (i) CPU-pool expert work during decode for the rows still out and
  (ii) the pump's DMA beside decode. Neither is on the prompt path.

The honest bound on (i): the evicted rows are the **coldest** resident experts, so the
decode hit rate should fall by much less than 6 481/24 576 of the mass. The honest bound on
(ii): 16 rows per cache per batch is ~30 MiB, ~23 ms of DMA on CUDA1's 1.3 GB/s link —
about one verify window — and the pump keeps at most one batch per cache in flight, so it
self-throttles to roughly the rate the windows run at. It does share PCIe with decode's own
streamed experts and KV reads, so `decode tok/s` is the number to watch, not just
`prompt_ms`.

**Expect `prompt_ms`'s fixed term to roughly halve, and `decode tok/s` to move by a few
percent.** If the decode number moves by more than a few percent, lower
`STRATA_PREFILL_LOAN_PUMP_ROWS` (or set it to 0 and let the next request's `lend()` find the
rows already out for free).

## How the owner confirms it

```sh
STRATA_PREFILL_LOAN_TIMING=1 engine/strata ... --serve --serve-slots 3 ...
python3 bench/prefill/analyze.py --compare <baseline-dir> <new-run.json>
```

`analyze.py` still does not parse the loan lines (S0.3's note said so), so read them
directly and compare the fit:

```
strata serve: prompt loan return: LAZY (16 row(s) per cache in flight on the pump; ...)
strata serve: loan lend: 8192 tokens in 0.6 ms (0 relayout, 0 grown, 6481 row(s) now out of the caches)
strata serve: loan request: relayout 0, relayout skipped 3, loan grown 0, rows refilled 0, rows out 0, pumped 0, still out 6481, res upload 1 (2 skipped), loan return lazy
strata serve: loan pump CUDA1: 16 rows (0.03 GiB) queued in 0.1 ms (1955 still out of the cache)
strata serve: loan drain idle: the caches are whole again (1840 ms)
strata serve: prompt loan totals: 14 lend(s), 9 hand-off(s), 1 relayout, 41 relayout skipped, 0 grown, 0 rows refilled eagerly, 6411 pumped home, 0 row(s) still out of the caches (lazy)
```

What each column proves:

* **`rows refilled 0`** on every `loan request` line — the refill is gone from the prompt
  path. This is the headline. With `STRATA_PREFILL_LAZY_LOAN=0` the same line reads
  `rows refilled 6481 ... loan return eager` and the old `loan refill EAGER CUDA1: 1971
  rows (3.85 GiB) in ~3180 ms` lines come back.
* **`relayout skipped` ≫ `relayout`** across requests — the layout is shared, not rebuilt.
* **`still out`** — how much the caches are missing right now, i.e. the CPU-side decode
  this request inherits.
* **`prompt loan totals: … hand-off(s) … pumped home … still out`** at shutdown — the
  process-level view. `hand-offs > 0` with `rows refilled eagerly = 0` is the lazy loan
  working; `still out = 0` means the pump and the idle drain caught up.
* the fitted `a` from `analyze.py`, which is the number the whole exercise is about.

A/B arms, in the order worth running them:

```sh
# 1. the new default, concurrent driver
--serve-slots 3
# 2. the same driver, loan refilled the old way  -> isolates S3.2b from S3.1e-2
--serve-slots 3  STRATA_PREFILL_LAZY_LOAN=0
# 3. lazy on the serial path                    -> isolates the loan from the scheduler
--serve-slots 0  STRATA_PREFILL_LAZY_LOAN=1
# 4. no overlapped pump at all                  -> measures (B) separately from (A)+(C)
--serve-slots 3  STRATA_PREFILL_LOAN_PUMP_ROWS=0
```

## Files

```
include/strata/program/prefill_loan.hpp   NEW lever 5: LoanRow, LoanLedger, LazyRefillPolicy,
                                          pump_may_run / pump_batch / lend_must_remark /
                                          narrow_lend_refills_first / may_mark_resident
src/program/prefill_loan_test.cpp         54 -> 131 CPU checks (the ledger state machine, the
                                          pump predicate, usage ordering, the adversarial
                                          window case, the cross-request survival)
include/strata/program/serve_driver.hpp   Loan: documents that release() is not a residency
                                          event; acquires / releases / handoffs counters
src/program/serve_driver_test.cpp         230 -> 245 CPU checks
src/program/generate.cpp                  lend / refill / refill_one / return_loan_lazy /
                                          settle_pump / pump_loans / drain_loans_idle /
                                          reconcile_residency, PfPart's ledger + loan_live +
                                          pump stream, adapt()'s owns() gate, apply_pending's
                                          drop(), the switches, --help, the loan lines
bench/prefill/fixed-cost-changes.md       this section
```

Not touched: `slot.hpp`, `serve_proto.hpp`, `serve_swap.hpp`'s ordering,
`src/core/conversation_state.cpp`, `src/prefill/*`, `src/kernels/*`, `serve/server.py`,
`bench/prefill/analyze.py`.

## Verified

* `cd build && ninja` clean (the `.sframe` ld lines and the `argmax` unused-function warning
  are pre-existing on this box).
* `cd /tmp/s3build && ninja` clean; `ctest` 18/19 — the only failure is `expert_multi_test`,
  which aborts on `this CPU cannot run the expert kernel: missing AVX512`, pre-existing on
  this Zen 3 box.
* `prefill_loan_test` 131 checks, `serve_driver_test` 245 checks, `slot_test`,
  `serve_proto_test`, `serve_swap_test` all pass.
* `python3 -m pytest serve/test_server.py -q` → 64 passed, 2 skipped.
* `./build/strata --help` renders the new block.
* No engine start, no model load, no request, `/dev/shm` untouched.

## NOT verified (cannot be, on this box)

* **The magnitude.** No run. The 5.2 s is arithmetic on measured link speeds applied to the
  loan size the baseline log prints, not a measurement of `refill()`.
* **One small cost the lazy rule adds to the prompt path.** `lend()` starts with
  `settle_pump(p, /*wait=*/true, e)`: a pump batch queued during the *previous* request's
  decode must land before the prompt buffers are carved over those slots, so `lend()` can
  block on one event. Bounded by one batch — 16 rows ≈ 30 MiB ≈ 23 ms on CUDA1's link — and
  the pump self-throttles to one batch per window, so in practice it has usually landed.
  It is the one place S3.2b can add milliseconds to a prompt, and `loan lend: ... in X ms`
  is the line that shows it.
* **The decode hit-rate cost.** The argument that the evicted rows are the coldest resident
  experts is a source argument about profile-fill order (`generate.cpp`'s R4.2e block fills
  hottest-first; the loan takes the slot tail). The real number is
  `decode expert cache hit rate` before and after.
* **That the pump's DMA does not slow decode.** It shares PCIe with decode's streamed
  experts and KV reads. `STRATA_PREFILL_LOAN_PUMP_ROWS=0` is the arm that separates them.
* **Token-level parity under the lazy rule.** A row still out is computed by the CPU, so a
  lazy run is not bit-identical to an eager one — the same parity caveat the cache itself
  carries. `STRATA_STATE_HASH` on `--serve-slots 3` with `STRATA_PREFILL_LAZY_LOAN=0`
  versus `=1` will show the divergence; it is expected, and it is why the default is gated
  to the concurrent driver rather than applied to the serial path.
