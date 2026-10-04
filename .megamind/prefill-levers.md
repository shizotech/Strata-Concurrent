# Pre-fill speed levers — measured from the live server log (2026-10-01)

Source: `strata-iq3_s.log`, 1324 requests, engine 0.1.30, 3× RTX 3090 layer split,
IQ3_S, `--prefill auto`, `--kv-resident 32768`, `--max-context 524288`, yarn 2.

**Nothing here was measured by starting anything.** It is all read out of the log the
running server already wrote, plus the source. Do not "verify" it by launching a server.

> **S0.2 update (2026-10-01).** This is now a tool, not a one-off reading:
> [`bench/prefill/analyze.py`](../bench/prefill/analyze.py) parses the log, groups by
> restart, refits the model and writes `--json` / `--csv`;
> [`bench/prefill/README.md`](../bench/prefill/README.md) holds the baseline table and
> the ranked lever list; [`bench/prefill/ab-prefill.sh`](../bench/prefill/ab-prefill.sh)
> is the A/B harness for a restart. Reproduce the snapshot below with
> `analyze.py --limit 1324 strata-iq3_s.log`. Two claims in the original note were wrong;
> they are corrected in place and marked **CORRECTED**.

## The machine as it is configured right now

| fact | value |
| --- | --- |
| arena | 46.84 GiB, **loaded 28× at 0.49–0.54 GiB/s** (~90 s each; two runs at 0.07/0.09). **Only 2 restarts borrowed it** — see "The arena is still loaded 28 times" |
| PCIe probe | CUDA0 **2.9 GB/s**, CUDA1 **1.3 GB/s**, CUDA2 **24.5 GB/s** |
| `pcie_frac` | 0.00 on CUDA0/CUDA1 (auto), 0.55 on CUDA2 |
| expert caches | 7778 + 9216 + 7168 = **24162 of 24576 pairs (~100 % of routed mass)** |
| prompt chunk auto | **8192 tokens** — the same in all 30 restarts; loans 2625 + 1971 + 1885 = **6481 slots (4.95 GiB each stage)** = 33.7 % of CUDA0's cache |
| VRAM free with everything loaded | **457 MiB** (the log calls anything under 256 low; 457 is thin) |
| RAM | 78 GiB total, **~2 GiB available**, swap 3/5 GiB used |
| parking | budget auto-sized to **1.6 GiB** (restart 29) then **0.0 GiB** (restart 30); **CORRECTED** — see the parking section |

## What prefill actually costs (the log, bucketed by *fresh* tokens)

Snapshot = the first 1324 requests, i.e. `analyze.py --limit 1324 strata-iq3_s.log`
(the log is live and keeps growing, so a fixed request count is the only reproducible
cut). The full log is now ~1600 requests and the shape is the same.

| fresh tokens | requests | tokens | wall | effective tok/s |
| --- | ---: | ---: | ---: | ---: |
| 1–64 (verify-window path) | 114 | 744 | 29.8 s | 25 |
| 65–256 | 291 | 49 537 | 2 002 s | **24.7** |
| 257–1024 | 548 | 284 891 | 5 349 s | **53.3** |
| 1k–4k | 248 | 489 211 | 4 041 s | 121 |
| 4k–16k | 95 | 799 471 | 3 346 s | 239 |
| >16k | 28 | 3 066 564 | 6 246 s | **491** |

Least-squares over those 1324 requests:

```
prompt_ms ≈ 9125  +  1.678 * fresh  +  1.62e-6 * fresh * context        R2 = 0.949
            ^^^^^     ^^^^^^^^^^^^    ^^^^^^^^^^^^^^^^^^^^^^
            ~9.1 s     ~596 tok/s      NOT identified — see below
            fixed      marginal
```

**CORRECTED: 54.4 % of all prefill wall time (11 422 s of 21 014 s) is spent on reads
of ≤ 4 000 fresh tokens — not 93 %.** 90.7 % of the *requests* (1 201 of 1 324) are in
those buckets; the original note appears to have mixed the two columns. What survives is
the conclusion that matters: the ≤4 k buckets are still the majority of the wall time,
and that time is dominated by a ~9–10 s per-request fixed cost, not by the per-token
rate. The published 1 443 tok/s IQ3_S figure is the marginal rate of a 128 K-token read;
this workload never gets near it.

**CORRECTED: the fitted `c` (context) term is not a measurement.** Over this workload
fresh and context move together, so `c` is unidentified: +1.6e-6 on the 1324 snapshot,
−2.7e-6 on the 1600 snapshot, and its sign flips restart by restart. Hold `fresh` fixed
and move context instead (from `analyze.py`'s context-control table):

| fresh 257–1024 | ctx <20k | 20–60k | 60–120k | 120–200k | >200k |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean ms | 9 352 | 11 904 | 10 579 | 11 554 | 10 200 |
| mean fresh | 549 | 556 | 535 | 533 | 485 |

**A 500-token read costs ~10 s whether the cached prefix is 5 000 or 250 000 tokens.**
The fixed cost does not scale with context — it is per-request overhead, not attention.
That strengthens lever 1 and removes any "attention over a long prefix" theory for the
short-read buckets.

## The fixed cost, ranked by how much of it the source can explain

1. **Lend + refill churn.** `lend()` / `refill_one()` (`src/program/generate.cpp`
   ~4604–4669) evict the tail slots of *every* stage's cache for each batched
   segment, mark those experts non-resident, stream them back over PCIe afterwards,
   and `res_upload()` re-uploads the whole 24 576-entry residency table per device.
   A request with a turn boundary splits into segments, so this can run twice.
   On a 1.3 GB/s link, one stage's 4.95 GiB loan is 3.8 s of pure refill.
2. **Experts evicted by the loan are streamed during the prompt.** With ~100 % of
   the experts resident, the *only* misses during a prompt are the ones the loan
   created. A 512-token chunk routes ~320 of 512 experts per layer, so the loan's
   135-per-layer evictions are almost all hit.
3. **Checkpoint save/restore + parking.** `checkpoint_at` does a
   `cudaDeviceSynchronize()` and moves ~118 MB; `park_current` moves hundreds of MB.
   8 186 checkpoints over 1 600 requests. Parking only exists in restarts 29–30 — see
   the corrected parking section, so it is not the explanation for the floor in the
   other 28 restarts.
4. **`session_zero` + `mtp.kv_restore`** when `resume == 0`.

## The parking situation (stage 2) — CORRECTED

The original note said the 2 560 MiB floor refuses **every** park. It does not, and
most of the log has no parking at all:

* **Restarts 1–28 (1 162 of 1 600 requests): no parking lines whatsoever** — no parks,
  no refusals, no `parked-prefix budget` line. Those processes were not paying for
  parking, so parking cannot explain their fixed cost.
* **Restarts 29–30: ~120 parks succeeded** (~1.6 M tokens, ~90 s, **~70 GiB of snapshot
  writes**, ~3 200 evictions) against **~120 refusals**. Parking is refused *often*, not
  always — and where it does run, it is not free. (Exact figures for the saved cut:
  `bench/prefill/baseline/report.md`; they drift as the live log grows.)
* The refusals are all in 29–30, where the machine-sized budget had already collapsed to
  1.6 GiB and then 0.0 GiB. The stage-2 budget-sizing code is doing its job; the cost
  being paid is the snapshot write plus the eviction churn, not a wasted estimate.
* **Caveat:** restart 30's parking counters are not trustworthy. A second live process
  is interleaved into that segment — its `parked=` cache totals reach 5.8 GiB against a
  0.0 GiB budget. `analyze.py` flags this with a `!` line instead of hiding it.

`--conversation-cache-mib 0` (or a lower `--conversation-cache-min-free-mib`) is still
the right call *on this box once the budget has collapsed*, for that reason. It is a
config decision for the owner, not a code change.

## New (S0.2): the cost is per *segment*, and the segment is the loan

(The tables from here on are the 1 600-request cut saved in
`bench/prefill/baseline/`, not the 1 324 snapshot above; the shape is the same.)

Bucketing by `want = min(chunk, ceil(fresh/256)*256)` — the number of batched tokens a
request asks `lend()` for buffers (`request_chunk`, generate.cpp:3418) — rather than by
`fresh`:

| want | requests | mean fresh | mean ms | ms per fresh tok | effective tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤256 | 319 | 172 | 7 275 | 42.20 | 23.7 |
| ≤512 | 382 | 369 | 9 762 | 26.46 | 37.8 |
| ≤1024 | 284 | 742 | 12 522 | 16.89 | 59.2 |
| ≤2048 | 213 | 1 443 | 18 647 | 12.92 | 77.4 |
| ≤4096 | 118 | 2 811 | 18 790 | 6.68 | 149.6 |
| ≤8192 | 151 | 31 181 | 87 101 | 2.79 | 358.0 |

A 256-token loan costs ~7.3 s; a 4 096-token loan ~18.8 s. The cost is **not**
proportional to the loan — it is a large floor plus a per-token rate. That floor is
lever 1. Note this table cannot answer "what would a smaller `--prefill` have done": a
request never lends more than its own prompt needs, so the log only contains loans that
were actually taken.

## The arena is still loaded 28 times

`/dev/shm/shared_experts.dat` exists, and stage 1's borrow path works — but across the
log **28 of 30 restarts each logged `loaded 46.84 GiB at 0.49–0.54 GiB/s`** (~90 s, two
runs at 0.07/0.09 GiB/s took ~11 min). Only restarts 29–30 took the shared path: one
`SHARED ARENA PUBLISHED`, one `SHARED ARENA BORROWED, experts not re-loaded`.

So the original "loaded twice" framing understates it: the process that owns the arena
keeps dying and restarting, and every restart pays ~90 s and 47 GiB on a 78 GiB box that
is already at ~2 GiB available. Check `pinned.cu`'s borrow path for the case where the
arena file exists and is published but its owner has gone (the pack-hash / state-ready
check runs before the free-space check?). This is a stage-1 follow-up, not a prefill
lever, but it is 90 s of start time per restart.

## Levers, ranked, with their evidence class

**(a) = measured from the log. (b) = inferred from source. (c) = untested hypothesis.**

| # | lever | evidence |
| --- | --- | --- |
| 1 | **Make the loan sticky across segments and across requests.** `refill()` runs at the end of *every* request (generate.cpp:4620) and `lend()` re-takes it per segment; `want <= p.lent_chunk` already skips a re-lend inside one request. | **(a)** the ~9.5–10.5 s floor, its independence of context, and the 7.3 s cost of a 256-token loan are measured; **(b)** that the loan is most of that floor — the log has no lend/refill timer |
| 2 | **Price the chunk instead of taking the largest that fits.** `--prefill auto` picked 8192 in all 30 restarts; the `fits()` scan (generate.cpp:3560) only asks whether every stage *can* lend it, and the 90 % cap is not binding (the loan is 33.7 % of CUDA0's cache). `Prefill::bytes_needed` is exact and cheap. | **(b)** from source; **(c)** whether 4096/2048 is actually faster — the log cannot show an untried chunk |
| 3 | **`--conversation-cache-mib 0` (or a lower floor) once the budget has collapsed.** | **(a)** the counts (~120 parks / ~70 GiB written / ~3 200 evictions / ~120 refusals); **(b)** that disabling it is net-positive — the log cannot show the prefix re-reads it prevented |
| 4 | **Load the arena once** (stage-1 follow-up, not a prefill lever). | **(a)** 28 loads, 2 borrows |
| 5 | **`--short-read N`** (default 64). The window path borrows no expert-cache slots, so it pays no lend/refill at all. | **(c)** — this log only ever sends ~6-token segments through the windows (mean 262 ms for 6 tokens), so the trade-off the source describes (~16 ms/token windows vs ~300 ms + refill batched) **cannot be priced from this log at all**. Needs the A/B script |
| 6 | **`STRATA_PREFILL_LEND_PCT`, `STRATA_PREFILL_RING`, `STRATA_STAGER_RING`, `STRATA_STAGER_THREADS`.** All read once at startup. | **(c)** — none of them appear in the log, so no run in this file varied them. **The default is not one number on this box**: `kAutoLendPct` is 90 only when `Prefill::pinned_share() >= 0.9`, and `ring_slots()` is 384 under the same test. `analyze.py` now reports the regime per restart from the `expert arena:` line: **18 restarts registered the whole arena (90 % / ring 384), 10 pinned only 7 GiB of 46.84 GiB and 2 pinned nothing (85 % / ring 96)**. Pin both sides explicitly when A/B-ing |
| 7 | **Stop `res_upload()` when nothing changed** — the whole 24 576-entry residency table to every device on each lend/refill. | **(b)** from source only; invisible in the log |

**Config only (no code, needs an owner-approved restart):** levers 2, 3, 5, 6.
**Code:** levers 1 and 7 — that is S0.3.
**Measure them with:** `bench/prefill/ab-prefill.sh` (starts an engine: needs free RAM;
its replay half is untested here) and `bench/prefill/analyze.py --compare`.

Also still worth doing in code, from the original note: stop parking when the budget is
known-refused (remember the last N refusals) — the refusals are per-request, and 116 of
them in two restarts each paid the snapshot estimate and the RAM telemetry.

## Data-quality warnings for anyone reusing this log

* **It is not one process.** 34 startup blocks, 30 reached `session is up`, 4 died while
  loading, and at least two processes wrote into it concurrently. Per-restart grouping is
  by the `session is up` anchor, which cannot separate two live processes.
* **The machine changed configuration mid-log.** The first six restarts report 3 553 MiB
  of free VRAM and an 8 192-slot CUDA0 cache; from restart 7 on it is ~455 MiB and
  7 773–9 570 slots. The per-restart fits move with it: restart 27 (840 requests) fits
  b ≈ 1.9 ms/token at R2 0.997, restart 30 (465 requests) b ≈ 4 ms/token at R2 0.94. Do
  not read a single marginal rate as a property of the model.
* **`prompt_ms` includes the loan** — `refill()` runs before the line prints
  (generate.cpp:4760). Good for pricing the levers, but "effective tok/s" is not a pure
  prompt-path rate.
* **`--short-read` is never printed.** `analyze.py` assumes the default 64; pass
  `--short-read N` if a run used something else.
