# Prefill cost: the tool, the baseline, and how to A/B the levers

Offline analysis of what reading a prompt actually costs on this box, plus the
harness for measuring the levers. Nothing here starts anything: `analyze.py`
reads a `--serve` log that already exists, and `ab-prefill.sh` is a script the
**owner** runs at a restart, on a machine with the RAM for it.

```
bench/prefill/
├── analyze.py        parse a Strata serve log -> buckets, fitted cost model, cache/parking stats
├── ab-prefill.sh     A/B the prefill levers at a restart (starts an engine: needs free RAM)
├── baseline/         the numbers for THIS box as configured today (see below)
│   ├── report.md       full analyze.py output, 1 600 requests
│   ├── aggregate.json  the same as JSON
│   └── requests.csv    one row per request, for diffing a later run against this one
└── README.md         this file
```

## Run it

```sh
python3 bench/prefill/analyze.py strata-iq3_s.log            # the tables, on stdout
python3 bench/prefill/analyze.py --limit 1324 strata-iq3_s.log   # reproduce .megamind/prefill-levers.md
python3 bench/prefill/analyze.py --json strata-iq3_s.log > run.json
python3 bench/prefill/analyze.py --csv bench/prefill/baseline strata-iq3_s.log
python3 bench/prefill/analyze.py --compare run.json bench/prefill/baseline/aggregate.json
```

Stdlib only, Python 3.8+. Options:

| flag | what it does |
| --- | --- |
| `--limit N` | keep only the first N requests in log order. A serve log keeps growing, so this is how you reproduce an earlier snapshot - the numbers in `.megamind/prefill-levers.md` are `--limit 1324`. |
| `--json` | the whole analysis, machine-readable, one object per restart plus an `aggregate`. |
| `--csv DIR` | `DIR/requests.csv`, one row per request. Diff a later run against `baseline/requests.csv`. |
| `--compare A.json B.json …` | the key metrics side by side. This is what `ab-prefill.sh` calls. |
| `--short-read N` | the `--short-read` the run used (default 64). The log does not record it; it decides which requests went through the verify windows rather than the batched prompt path. |

## What each number means

The engine prints one line per request (`src/program/generate.cpp` ~5048):

```
strata serve: prompt 28533 tokens = 28242 reused + 291 read in 9248 ms (31.5 tok/s),
              80 generated in 1203 ms (66.5 tok/s), drafts accepted 51 of 55, 6 checkpoints
```

* **reused** - prompt tokens the engine did not read: a prefix it already had
  (a checkpoint, or a parked conversation it mounted).
* **fresh** (`read in`) - the tokens the prompt path actually processed. **Every
  prefill number in here is about fresh tokens.** A request with 28 242 reused
  and 291 fresh did not read 28 533 tokens.
* **prompt wall** - `prompt_ms` in that line: the clock from the start of the
  request to the end of the prompt read, *including* the lend/refill, the
  checkpoints and the parking that happen inside it. It is not GPU-only time.
* **effective tok/s** - `fresh / prompt wall`. This is the number that looks bad
  (≈200) next to the published 1 443 tok/s, because the published figure is the
  marginal rate of one 128 K-token read and a real workload is mostly short reads.
* **buckets** - requests grouped by *fresh* token count, using the same buckets
  as `.megamind/prefill-levers.md` so the tables line up.
* **fit a / b / c** - least squares over `prompt_ms = a + b*fresh + c*fresh*context`,
  `context` = the prompt length the engine reports (reused + fresh). `a` is the
  per-request fixed cost, `1000/b` the marginal rate.
* **want** - `min(chunk, ceil(fresh/256)*256)`: the number of batched prompt
  tokens a request asks `lend()` for buffers (`request_chunk`, generate.cpp:3418).
  It is the only lever variable the log lets you read per request.
* **path** - `window` (fresh ≤ `--short-read`, read through the verify windows,
  borrows no expert-cache slots) vs `batched` (the prompt path, pays lend/refill).
* **parking** - the `conversation cache:` lines. `parked` counts successful
  snapshots, `refused` counts `skip parking (physical RAM admission …)`.
  `snapshot_bytes` is what each park wrote; `bytes=` is the whole cache's total.

Two counters are **cumulative per process**, so the tool keeps the last line in a
restart's segment rather than averaging them: `decode expert cache hit rate` and
`KV streaming: … block reads hit VRAM`.

## The baseline: this box, as configured today

Ryzen 5 5600X (6c/12t, no AVX-512), 78 GiB RAM, 3× RTX 3090 layer split
(CUDA0 layers 0-15, CUDA1 16-33, CUDA2 34-47), IQ3_S native pack,
`--prefill auto`, `--kv-resident 32768`, `--max-context 524288`, yarn ×2,
`--vram-reserve-mib 700`. Engine 0.1.30.

`baseline/` is `--limit 1600` of `strata-iq3_s.log` (the log is live and keeps
growing, so a fixed request count is the only reproducible cut). 34 startup
blocks, 30 of which reached `session is up`; 4 died during loading.

Re-checking it: every request-derived number in `baseline/report.md` reproduces
exactly. The four **cumulative per-process** rows (expert-cache hit, KV
streaming, parking, suffix drafts) do not, because they are read off the last
matching line in each restart's segment and that segment keeps growing while the
servers run. Diff the tables and expect those rows to move; treat them as
"roughly where the box is", not as a fixed baseline.

### The machine as it resolves at startup

| fact | value |
| --- | --- |
| arena | 46.84 GiB, **loaded 28× at 0.49-0.54 GiB/s** (~90 s each; two runs at 0.07/0.09 GiB/s took ~11 min). Only 2 restarts borrowed it (29, 30), 1 published it |
| arena registration | **18 restarts registered the whole arena** (`cudaHostRegister PORTABLE ok`) → `pinned_share >= 0.9`, so the auto lend cap is 90 % and the prompt ring is 384; **10 pinned only 7 GiB of 46.84 GiB and 2 pinned nothing** → cap 85 %, ring 96. `analyze.py` prints this per restart |
| PCIe probe | CUDA0 **2.9 GB/s**, CUDA1 **1.3 GB/s**, CUDA2 **24.5 GB/s** |
| `pcie_frac` | 0.00 on CUDA0/CUDA1, 0.55 on CUDA2 |
| expert caches | CUDA0 7 778 + CUDA1 9 216 + CUDA2 7 168 = 24 162 of 24 576 pairs (~100 % of routed mass) |
| prompt chunk auto | **8192 tokens** in all 30 restarts |
| loan | 2 625 + 1 971 + 1 885 = **6 481 slots, 4.95 GiB per stage** |
| VRAM free with everything loaded | **457 MiB** (the engine calls anything under 256 LOW) |
| parked-prefix budget | **1.6 GiB** at restart 29, **0.0 GiB** at restart 30 (asked 8.0) |

### Where the prompt time goes

| fresh tokens | requests | fresh tok | wall s | effective tok/s | % of prompt wall | mean ms/req | ms per fresh tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-64 (verify-window path) | 133 | 839 | 34.9 | 24.1 | 0.1 | 262 | 41.56 |
| 65-256 | 319 | 54 999 | 2 320.7 | 23.7 | 8.0 | 7 275 | 42.20 |
| 257-1024 | 666 | 351 539 | 7 285.3 | 48.3 | 25.1 | 10 939 | 20.72 |
| 1k-4k | 331 | 639 087 | 6 189.1 | 103.3 | 21.4 | 18 698 | 9.68 |
| 4k-16k | 113 | 952 216 | 4 105.8 | 231.9 | 14.2 | 36 335 | 4.31 |
| >16k | 38 | 3 756 150 | 9 046.4 | 415.2 | 31.2 | 238 062 | 2.41 |

**54.6 % of all prompt wall time is in reads of ≤ 4 000 fresh tokens** (15 830 s
of 28 982 s). The other 45 % is 151 requests reading more than 4 k tokens each,
38 of them more than 16 k.

Aggregate: 1 600 requests, 153.4 M prompt tokens of which **96.2 % reused**,
5.75 M fresh, 28 982 s of prompt wall → **198.6 effective prompt tok/s**;
decode 53.7 tok/s; expert-cache hit 91.6 %; drafts accepted 66.4 %.

### The fit

```
prompt_ms = 9547 + 2.719 * fresh + -2.652e-06 * fresh * context      R2 = 0.902
            ^9.5 s fixed   ^368 tok/s marginal
batched path only: a = 10 460 ms, 2.686 ms/fresh (372 tok/s)         R2 = 0.905
```

Over the earlier 1 324-request snapshot the same fit gives `a = 9 125 ms`,
`b = 1.678` (596 tok/s), `c = +1.62e-6`, R2 = 0.949 - i.e. the recorded
"~9.3 s fixed, ~590 tok/s marginal" reproduces. **The coefficients are not
stable across the whole log**: `a` stays 9.5-10.5 s, but `b` moves from 1.68 to
2.72 ms/token and `c` changes sign. See the caveats below.

## What this adds to the first reading of the log

Two things the earlier analysis in `.megamind/prefill-levers.md` did not have,
and two corrections to it. (A third addition - the per-restart arena
registration regime - is in the startup table and lever 6 below.)

### New: the cost is per *segment*, and the segment is the loan

Bucketing by `want` (what the request actually lends) instead of by `fresh`:

| want | requests | mean fresh | mean ms | ms per fresh tok | effective tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤256 | 319 | 172 | 7 275 | 42.20 | 23.7 |
| ≤512 | 382 | 369 | 9 762 | 26.46 | 37.8 |
| ≤1024 | 284 | 742 | 12 522 | 16.89 | 59.2 |
| ≤2048 | 213 | 1 443 | 18 647 | 12.92 | 77.4 |
| ≤4096 | 118 | 2 811 | 18 790 | 6.68 | 149.6 |
| ≤8192 | 151 | 31 181 | 87 101 | 2.79 | 358.0 |

A 256-token loan costs ~7.3 s; a 4 096-token loan costs ~18.8 s. The cost is
**not** proportional to the loan - it is a large floor plus a per-token rate.
That floor is what the levers are aimed at.

### New: context is *not* what costs time in the short-read buckets

The fitted `c` term is not identified by this workload (fresh and context move
together, and `c` flips sign between restarts). The honest version is to hold
`fresh` fixed and move context:

| fresh 257-1024 | ctx <20k | 20-60k | 60-120k | 120-200k | >200k |
| --- | ---: | ---: | ---: | ---: | ---: |
| mean ms | 9 352 | 11 904 | 10 579 | 11 554 | 10 200 |
| mean fresh | 549 | 556 | 535 | 533 | 485 |

A 500-token read costs ~10 s whether the context is 5 000 or 250 000 tokens.
**The fixed cost does not scale with the cached prefix.** It is per-request
overhead, not attention. The attention term only becomes visible in the >16 k
bucket (2.47 ms/token at 20-60 k context vs 2.11 ms/token above 200 k - and even
there the direction is confounded by which requests are which).

### Correction 1: "93 % of prefill wall is ≤4 k fresh" is wrong

It is **54-55 %**. Recomputing the same buckets on the same 1 324 requests the
note used gives 54.4 %. (90.7 % of the *requests* are in those buckets - 1 201
of 1 324 - which is probably where the 93 % came from.) The conclusion that
survives is the important one: the ≤4 k buckets are still the majority of the
wall time and they are dominated by a ~9-10 s per-request floor.

### Correction 2: parking is not "always refused", and it is not in the older restarts at all

The note says "every park refused by the 2 560 MiB floor". Across the log:

* Restarts 1-28 (1 162 of the 1 600 requests): **no parking lines at all** - no
  parks, no refusals, no `parked-prefix budget` line. Those processes were not
  paying for parking.
* Restarts 29-30: **124 parks succeeded** (1.62 M tokens, 92.1 s, **71.3 GiB of
  snapshot writes**, 3 211 evictions) against **123 refusals**. So parking is
  refused *often*, not always, and where it runs it is not free. (These are the
  saved baseline's numbers; they grow with the log — see the reproducibility note
  above.)
* The refusals are all in restarts 29-30, where the machine-sized budget had
  already collapsed to 1.6 GiB and then 0.0 GiB.

So `--conversation-cache-mib 0` is still the right call *on this box once the
budget has collapsed*, but the reason is different from the one in the note: the
budget-sizing line is doing its job, and the cost being paid is the snapshot
write plus the eviction churn, not a wasted estimate.

Also: the parking counters for restart 30 are **not trustworthy** - the log has
lines from a second live process interleaved into that segment (the `parked=`
cache totals reach 5.8 GiB against a 0.0 GiB budget). `analyze.py` flags this
with a `!` line rather than hiding it.

## Caveats when reading these numbers

* **The log is not one process.** 34 startup blocks, 30 reached `session is up`,
  and at least two processes wrote into it concurrently. Per-restart grouping is
  by the `session is up` anchor, which cannot separate two processes that are
  both alive. The parking caveat above is how the tool reports when it caught
  this; treat cross-restart aggregates as indicative.
* **The machine changed configuration mid-log.** The first six restarts report
  3 553 MiB of free VRAM and an 8 192-slot CUDA0 cache; from restart 7 on it is
  ~455 MiB and 7 773-9 570 slots. The per-restart fits move with it: restart 27
  (840 requests) fits `b ≈ 1.9 ms/token` at R2 0.997, restart 30 (465 requests)
  `b ≈ 4 ms/token` at R2 0.94. Do not read a single marginal rate as a property
  of the model.
* **`prompt_ms` includes the loan.** `refill()` runs before the line is printed
  (generate.cpp:4760), so the refill is inside the measured window. That is what
  you want when pricing the levers, but it means "effective tok/s" is not a
  pure prompt-path rate.
* **`--short-read` is assumed, not read.** The log never prints it. The tool
  assumes the default 64. In this log the window path only ever carried ~6-token
  segments (the assistant header after a checkpoint), so its per-token cost is
  not separable from its fixed cost, and raising `--short-read` **cannot be
  priced from this log at all**. It needs the A/B script.

## A/B-ing the levers

`ab-prefill.sh` is the owner's tool. It **refuses to run without `--run`** and
prints a warning that it starts a second engine and needs the RAM.

```sh
bench/prefill/ab-prefill.sh                       # prints the warning, does nothing
bench/prefill/ab-prefill.sh --run                 # the default ladder
bench/prefill/ab-prefill.sh --run --variant 'p1024:--prefill 1024' --env 'lp50:STRATA_PREFILL_LEND_PCT=50'
```

Per variant it writes a config from `strata-iq3_s.json` (dropping the base
config's `--prefill` / `--short-read` / `--conversation-cache-*` so the
variant's are the only ones), starts `serve/server.py` on port 8181, replays a
fixed prompt list through `POST /v1/chat/completions`, stops the server, and runs
`analyze.py --json` over that variant's log. At the end it calls
`analyze.py --compare`, which prints the fresh-token totals first: if the two
runs did not see a similar mix, read the fitted `a` and the marginal rate rather
than the wall time.

**Honest status: the replay half was never executed here.** Starting an engine
is forbidden on this box (no free RAM), so `ab-prefill.sh` is written against
`serve/server.py`'s documented endpoints and is expected to need small fixes on
its first real run. The analysis half - `analyze.py`, including `--compare` and
the whole variant→log→JSON→table pipeline - *was* exercised end to end against
the live log with the server start stubbed out.

## Ranked levers, with their evidence class

| # | lever | what the log says | class |
| --- | --- | --- | --- |
| 1 | **Make the loan sticky across segments and requests** (`refill()` runs at the end of *every* request, generate.cpp:4620; `lend()` re-takes it per segment) | The ~9.5-10.5 s per-request floor is where 55 % of the wall time is. A 256-token read costs 7.3 s and a 500-token read costs 10 s regardless of context, so the floor is per-request overhead, not work. On a 1.3 GB/s link one stage's 4.95 GiB refill is ~3.8 s of it. | **measured** (the floor and its context-independence); **inferred** (that the loan is most of it - the log has no lend/refill timer) |
| 2 | **Price the chunk instead of taking the largest that fits** (`--prefill auto` picks 8192 in all 30 restarts; the `fits()` scan at generate.cpp:3560 only checks that every stage *can* lend it) | The auto scan is not binding on the lend cap here: 2 625 of 7 778 slots is 33.7 %, well under the 85/90 % cap. So auto picks the biggest chunk that fits, and the loan's refill is paid on every request. Whether 4096 or 2048 is faster is **not answerable from this log** - a request never lends more than its own prompt needs, so the log only shows loans that were taken. | **inferred from source**; the direction is an **untested hypothesis** |
| 3 | **`--conversation-cache-mib 0` once the budget has collapsed** | Restarts 29-30: ~120 parks succeeded (~70 GiB of snapshot writes, ~3 000 evictions, ~90 s) against ~120 refusals, with the machine-sized budget at 1.6 then 0.0 GiB. Parking is not free and not reliably useful at 2 GiB of free RAM. Exact figures for the saved cut: `baseline/report.md` | **measured** (the counts); **inferred** (that turning it off is net-positive - the log cannot show the prefix re-reads it prevented) |
| 4 | **Load the arena once** (stage-1 follow-up, not a prefill lever) | 28 of 30 restarts logged `loaded 46.84 GiB at 0.49-0.54 GiB/s`; only 2 borrowed. ~90 s of start time and 47 GiB of RAM each time. | **measured** |
| 5 | **`--short-read N`** (default 64) | The window path costs ~262 ms for a ~6-token segment. Nothing in this log has 65-512 fresh tokens going through it, so the trade-off the source describes (~16 ms/token windows vs ~300 ms + refill batched) cannot be checked here. | **untested hypothesis** |
| 6 | **`STRATA_PREFILL_LEND_PCT`, `STRATA_PREFILL_RING`, `STRATA_STAGER_RING`, `STRATA_STAGER_THREADS`** | Read once at startup; none appear in the log, so no run in this file varied them. **The default is not one number on this box**: `kAutoLendPct` is 90 only when `Prefill::pinned_share() >= 0.9`, and `ring_slots()` is 384 under the same test. `analyze.py` reports the regime per restart from the `expert arena:` line — 18 restarts registered the whole arena (90 % / ring 384), 10 pinned only 7 GiB of 46.84 GiB and 2 pinned nothing (85 % / ring 96). Pin both sides explicitly when A/B-ing | **untested hypothesis** |
| 7 | **`res_upload()` on every lend/refill** (the whole 24 576-entry residency table, per device) | Not visible in the log at all. Source says it runs whenever any slot moved. | **inferred from source** |

## Reproducing the recorded numbers

```sh
$ python3 bench/prefill/analyze.py --limit 1324 strata-iq3_s.log
# Strata prefill cost - 1 log file(s), …, 1,324 requests
| requests | 1,324 (5 cancelled) |
| fresh (the prompt path read) | 4,690,418 |
| prompt wall time | 21,013.9 s |
| effective prompt tok/s | 223.2 |
prompt_ms = 9125 + 1.678 * fresh + 1.618e-06 * fresh * context   R2 = 0.949
```

Against `.megamind/prefill-levers.md`: 1 324 requests ✓, 4.65 M fresh → 4.69 M ✓
(the log grew while the note was written), 20.7 ks → 21.0 ks ✓, ~225 tok/s →
223.2 ✓, ~9.3 s fixed → 9.1 s ✓, ~590 tok/s marginal → 596 ✓.

The bucket table agrees closely but not exactly with the **original** note: its
rows summed to **1 309** requests, not 1 324, so it was cut a few requests
earlier. (`prefill-levers.md` has been updated to the reproducible values.) The
1-64 and >16k buckets match the original to the last digit (114 / 744 / 29.8 s /
25.0 and 28 / 3 066 564 / 6 245.8 s / 491.0); the middle four differ by under
2 % (e.g. 65-256: 291 vs the original's 289 requests, 2 002 s vs 1 977 s). The
"93 %" and "every park refused" claims do not reproduce at all - see the
corrections above.
