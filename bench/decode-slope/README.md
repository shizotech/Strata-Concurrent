# `bench/decode-slope/` — what a decode window row actually costs

Stage 4's batching decision rests on one number: **`b`, the marginal cost of one more row in a verify
window** in

```
window_ms = a + b * T
```

If `b` is near the GPU's bandwidth floor, batching concurrent requests scales beautifully. If `b` is
large, batching buys almost nothing and the memory cost is not worth it. `docs/STAGE4-BATCH-DECODE.md`
§3.5 first estimated `b = 8.4 ms` from two hand-picked log lines. This directory measures it.

**Headline result (§9 of `docs/STAGE4-BATCH-DECODE.md`):** the row cost is **not** GPU arithmetic and
**not** launch overhead. It is the **CPU expert pool** — the 5-15 % of routed experts that miss the
7 777-slot VRAM cache and get computed on 5 workers behind a per-layer barrier. A VRAM-resident expert
row costs nothing measurable. So the lever that flattens the slope is **VRAM for the expert cache**,
not a kernel rewrite.

---

## The three files

| file | what it is | safe to run? |
|---|---|---|
| `analyze.py` | parses a server log, derives per-window rows, fits the cost model with bootstrap CIs | **yes** — read-only, no engine, no GPU |
| `selftest.py` | proves `analyze.py`'s derivation recovers a *known* synthetic ground truth | **yes** |
| `slope-ab.sh` | the A/B ladder the **owner** runs to settle what the log cannot | **no** — starts an engine |

---

## Run the analysis (always safe)

```sh
python3 bench/decode-slope/selftest.py
python3 bench/decode-slope/analyze.py strata-iq3_s.log
```

`analyze.py` needs no engine. It reads the log the live server already writes (per-request prompt/
decode lines, expert hit rate, KV-stream hit rate) and reconstructs `T` per request from the identity
`ΣT = drafts_offered + windows`, keeping only requests where `generated == accepted + windows` exactly,
so every window ran to completion.

Useful flags: `--json OUT` / `--csv DIR` (for `--compare`), `--tier`, `--stream`, `--limit`,
`--blob-bytes`, `--hbm-gbps`, `--no-bootstrap`.

### Why the naive fit is wrong, and what the honest one says

`ms/window` in this log is dominated by **context** (30-1660 ms), not by rows (2-5). A naive regression
gives `b = 17 ms/row` at `R² = 0.03` — noise. Controlling for the expert-cache hit rate:

```
a      = 12.4 ms                fixed per window
b_gpu  = -0.016 ms/row          ~0: a VRAM-resident row is free at this resolution
b_cpu  = +0.0601 ms per CPU-pool expert entry   95% CI [+0.0575, +0.0628]
       (median 205 CPU entries/window  =>  12.3 ms/window)
b      = 10.89 ms/row           at this log's 86.4 % hit rate
b      =  5.19 ms/row           at hit = 100 %   <- REACHABLE, and it is a VRAM decision
b      =  0.74 ms/row           HBM bandwidth floor  <- the bound, not a target
```

`selftest.py` recovers a planted ground truth (`a = 12.00`, `b_gpu = 5.60`, `b_cpu = 37.70` per
miss-row), so the decomposition is not an artifact of the derivation.

**Errors-in-variables caveat.** The log gives a per-request *mean* of `T`, not a joint distribution, so
the within-request spread of `T` is measurement error in the regressor and biases `b` toward 0. The
bucketed fit is the honest slope; §3.5's two-point estimate is the upper anchor. Both are reported.

---

## The A/B ladder (owner only — this starts an engine)

The log cannot separate `b_gpu` from `b_cpu`, because in it rows, hit rate and context all move
together. `slope-ab.sh` moves them independently.

```
READ THIS FIRST: THIS SCRIPT STARTS AN ENGINE.
```

It reloads tens of GiB of host RAM and most of a GPU. Run it **only** with the serving processes
stopped and the RAM actually free (`free -g`, `nvidia-smi`). It refuses to do anything unless you pass
`--run`, and it was **never executed** — the box it was written on is forbidden from starting an engine.
Expect small fixes on the first real run.

```sh
bench/decode-slope/slope-ab.sh                      # prints the plan, touches nothing
bench/decode-slope/slope-ab.sh --run                # the default ladder
bench/decode-slope/slope-ab.sh --run --only dt,spec2,spec4,spec6
bench/decode-slope/slope-ab.sh --run --arm 'cache3k:--expert-cache 3000'
bench/decode-slope/slope-ab.sh --run --env 'dt:STRATA_DECODE_TIMING=1'
bench/decode-slope/slope-ab.sh --run --arm-ceiling  # ALSO the T>8 arm (needs a rebuilt engine)
```

Each arm feeds its log into `analyze.py`, and `analyze.py --compare A.json B.json` reports the
difference with CIs.

**What each arm proves:**

* **`dt`** — sets `STRATA_DECODE_TIMING=1`, which prints a real per-window line
  (`windows, avg T, tokens/window, ms/window = verify + commit/emit + draft`). The live log has **zero**
  such lines, so every number above is derived. This arm replaces derivation with direct measurement.
* **`spec2 / spec4 / spec6`** — move `T` while context and hit rate stay put. This is the clean slope.
* **`cache3k`** — shrinks the expert cache, which raises the miss rate at fixed `T`. If `b` rises with
  the miss rate and not with `T`, the slope is the CPU pool. **This is the decisive arm.**
* **`--arm-ceiling`** — needs an engine rebuilt with the row ceilings raised (see §9.1 of the design
  doc: there are **thirteen** hard 8-row limits, not six, and `GMAX` at 24 stops the build outright).

---

## What the numbers imply for batching

Baseline is B=1, D=3 (today's `--spec 4`). `acc` is a **stated parameter** — the log cannot give
acceptance at draft depths this box never ran.

| config | rows | measured slope | full-cache slope (hit = 100 %) |
|---|---|---|---|
| B=2, D=3 | 8 | +12.5 % | **+23.0 %** |
| B=4, D=3 | 16 | +20.0 % | **+39.0 %** |
| B=8, D=3 | 32 | +24.1 % | **+48.7 %** |
| B=16, D=3 | 64 | +26.3 % | **+54.1 %** |

So "+11-13 %" was right **for this box as it runs today** and wrong as a ceiling. With an expert cache
big enough that nothing spills to the CPU pool, B=8 is worth **+49 %**. The bandwidth-floor column
(up to +360 %) is the bound, not a target.

**Cheapest wins, in order:** (1) more VRAM headroom for the expert cache — it attacks the slope
directly and needs no kernel work; (2) `--decode-tokens` (already on) — the cheapest concurrency win
that exists; (3) the S4.3 prefill/decode split, which amortises the ~9.5 s prompt floor across all
decode instances.
