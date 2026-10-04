# S4.5 decode-window cost model

logs: strata-iq3_s.log
requests parsed: 3800   segments (engine processes): 13   usable: 3610   core population for the fits: 3572 (37 trimmed at p99 of ms/window)
dropped: 2 with no closed window count, 188 with < 4 windows, 0 with a mean T above the segment's max_t; 2 cancelled; 3610 W-estimator disagreements (W=G-A used, see the README)

## the segments

| # | lines | engine | max_t | arena MiB | tier | kv-res | pinned GiB | expert slots | expert GiB | slots | requests | usable | cfg inherited |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1-29 | - | - | - | 524288 | 32768 | 6.19 | 7773 | 14.74 | - | 0 | 0 | no |
| 2 | 30-157 | 0.1.30 | 6 | 76.9 | 524288 | 32768 | 6.19 | 7773 | 14.74 | 2 | 9 | 3 | no |
| 3 | 158-297 | 0.1.30 | 6 | 76.9 | 262144 | 32768 | 3.09 | 8014 | 15.19 | 2 | 10 | 5 | no |
| 4 | 298-3343 | 0.1.30 | 6 | 75.4 | 524288 | 32768 | 6.19 | 7773 | 14.74 | 3 | 21 | 12 | no |
| 5 | 3344-3379 | 0.1.30 | 6 | 76.9 | 524288 | 32768 | 6.19 | 7773 | 14.74 | 3 | 0 | 0 | no |
| 6 | 3380-3413 | 0.1.30 | 6 | 76.9 | 262144 | 32768 | 3.09 | 8019 | 15.20 | 3 | 0 | 0 | no |
| 7 | 3414-3482 | 0.1.30 | 6 | 75.4 | 524288 | 32768 | 6.19 | 7773 | 14.74 | 3 | 5 | 2 | no |
| 8 | 3483-3516 | 0.1.30 | 6 | 76.9 | 262144 | 32768 | 3.09 | 8019 | 15.20 | 3 | 0 | 0 | no |
| 9 | 3517-3825 | 0.1.30 | 6 | 75.4 | 262144 | 32768 | 3.09 | 8019 | 15.20 | 3 | 24 | 21 | no |
| 10 | 3826-3836 | 0.1.30 | 6 | 76.9 | 262144 | 32768 | 3.09 | 8019 | 15.20 | 3 | 0 | 0 | no |
| 11 | 3837-5171 | 0.1.30 | 6 | 75.4 | 524288 | 32768 | 6.19 | 7777 | 14.75 | 3 | 136 | 117 | no |
| 12 | 5172-5224 | 0.1.30 | 6 | 76.9 | 262144 | 32768 | 3.09 | 8019 | 15.20 | 3 | 3 | 1 | no |
| 13 | 5225-19760 | 0.1.30 | 6 | 75.4 | 262144 | 32768 | 3.09 | 8019 | 15.20 | 3 | 3592 | 3449 | no |

`cfg inherited` means the segment had no config lines of its own and took the previous
segment's. Two engines write this log concurrently, so their config lines and their
request lines interleave; this is an attribution, not a reading.

## self-consistency checks (do these before trusting any fit)

* `expert_lookups / (48 layers x 10 experts)` vs the line's own `generated`: median 1.140, p10 0.998, p90 1.286 over 3608 requests. A decode token costs exactly 480 lookups, so ~1.0 confirms the counters and the 48x10 geometry.
* expert lookups per ASKED row: median 458, p10 427, p90 484 over 3797 requests. 480 is the floor (every row is routed in every layer); anything above it is the MTP drafter's own layer plus refusals.
* the two window-count estimators `G-A` and `O-A+1` disagree on 3610 of 3800 requests. `G-A` is the one the source supports (see the README); the disagreement is the truncated last window, which is the common case.
* suffix-draft lines seen: 3413, attributed to their own request: 3413

## the population

ms/window: min 30.3  p10 38.5  median 47.9  p90 63.4  max 86.7
rows/window (committed-token lower bound): min 1.62  median 2.79  max 4.61
context mid: p10 36469  median 111786  p90 258996
expert cache hit: p10 78.2%  median 86.5%  p90 91.4%
CPU-pool expert entries per window: median 203

**The whole point of this section:** ms/window varies 30-1660 ms while rows/window varies 2-5. Context and expert-cache hit rate dominate, and any single-variable fit on rows is reading noise.

## the fits

The naive fit reproduces S4.1's ballpark and is wrong for the reason above. It is
printed only so the difference is visible.

```
  ms/win ~ mean_T (naive)      n=3610  R2=0.030  a -5.92 [-16.61, +4.78]  b +17.024 [+13.861, +20.186]

  simple                       n=3572  R2=0.206  a +5.80 [+2.96, +8.64]  b +13.078 [+12.237, +13.920]
  rows_lo_ctx                  n=3572  R2=0.258  const +16.64 [+14.78, +18.51]
      rows_lo = +11.046 [+10.415, +11.678] t34.3
      ctx_mid = +0.000 [+0.000, +0.000] t8.3
  rows_hi_ctx                  n=3572  R2=0.219  const +4.34 [+1.49, +7.18]
      rows_hi = +13.042 [+12.207, +13.877] t30.6
      ctx_mid = +0.000 [+0.000, +0.000] t7.6
  rows_lo_hit                  n=3572  R2=0.645  const +127.83 [+124.23, +131.44]
      rows_lo = +8.294 [+7.849, +8.739] t36.5
      expert_hit_pct = -1.189 [-1.226, -1.152] t-63.6
  rows_hi_hit                  n=3572  R2=0.658  const +119.09 [+115.36, +122.83]
      rows_hi = +11.035 [+10.480, +11.591] t38.9
      expert_hit_pct = -1.245 [-1.280, -1.209] t-68.6
  rows_both_hit                n=3572  R2=0.660  const +119.94 [+116.20, +123.69]
      rows_lo = +2.316 [+1.268, +3.364] t4.3
      rows_hi = +8.358 [+7.025, +9.690] t12.3
      expert_hit_pct = -1.226 [-1.263, -1.190] t-66.0
  rows_lo_hit_ctx              n=3572  R2=0.669  const +110.93 [+106.88, +114.99]
      rows_lo = +8.886 [+8.450, +9.322] t39.9
      expert_hit_pct = -1.213 [-1.249, -1.178] t-66.9
      log_ctx = +1.508 [+1.322, +1.693] t15.9
  expert_split                 n=3572  R2=0.572  const +24.27 [+22.23, +26.31]
      gpu_win = +0.005 [+0.004, +0.007] t7.3
      cpu_win = +0.082 [+0.080, +0.085] t69.0
  expert_split_rows            n=3572  R2=0.643  const +28.23 [+26.34, +30.12]
      gpu_win = -0.015 [-0.017, -0.013] t-14.6
      cpu_win = +0.060 [+0.057, +0.063] t43.5
      rows_lo = +9.634 [+8.921, +10.347] t26.5
  cpu_only                     n=3572  R2=0.566  const +31.55 [+30.99, +32.11]
      cpu_win = +0.081 [+0.079, +0.083] t68.2
```

bootstrap on `ms/win ~ rows_lo` (4000 resamples): a 95% [16.31, 20.34]  b 95% [10.353, 11.754]  median b 11.036

## the two-point anchor (S4.1's method, recomputed)

A request that generated exactly 1 token ran exactly 1 window, and that window is
T=1 by construction (src/program/generate.cpp:7180). n = 187 such windows:
  T=1 window ms: min 14, median 23, max 2888
Longest-mean-T usable request: line 10512, mean T 4.87, 71.1 ms/window
  => b = (71.1 - 23.0) / (4.87 - 1) = 12.44 ms/row, a = 10.6 ms
  S4.1 quoted a = 24 ms, b = 8.4 ms from `log:106` and `log:8578`. The T=1 median
  here confirms a; the 8.4 came from assuming `log:8578` ran T=6 windows, which it
  did not (see the README's correction table).

## rows/window x expert-cache-hit: the form-free identification

Median ms/window. Inside a row the expert hit rate is pinned, so moving along a row
is the effect of rows; inside a column the rows are pinned, so moving down a column
is the effect of the hit rate. n in brackets.

| rows \ hit % | 0-80 | 80-88 | 88-94 | 94-101 |
|---|---|---|---|---|
| 2.0-2.5 | 58.2 [40] | 45.3 [325] | 40.0 [543] | - |
| 2.5-3.0 | 60.0 [237] | 49.5 [716] | 43.8 [437] | - |
| 3.0-3.5 | 65.2 [203] | 55.0 [513] | 45.7 [259] | - |
| 3.5-4.0 | 64.5 [36] | 58.7 [112] | 46.1 [87] | 42.7 [9] |
| 4.0-4.5 | - | 66.2 [14] | 47.0 [11] | - |
| 4.5-5.5 | 74.2 [7] | - | - | - |
| 5.5-6.5 | - | - | - | - |

Read the row-to-row steps: that is the marginal row cost at a fixed cache hit rate.
Read the column-to-column steps: that is what a cache miss costs, which is the term
batching does NOT amortise.

## CPU expert entries x rows: the near-model-free version

Median ms/window. Within a row of this table the number of rows is pinned to
within ~0.1, so moving along the row is the cost of the CPU expert pool alone.
Within a column the CPU entries are pinned, so moving down is the row cost alone.

| CPU entries/win | rows 2.4-3.2 | rows 3.2-4.2 |
|---|---|---|
| 0-100 | - | - |
| 100-150 | 41.4 [323] | 43.3 [93] |
| 150-200 | 45.8 [647] | 47.2 [161] |
| 200-250 | 50.7 [490] | 54.8 [158] |
| 250-300 | 55.3 [312] | 58.3 [136] |
| 300-400 | 59.1 [318] | 62.2 [148] |
| 400-600 | 67.4 [96] | 67.7 [71] |

The along-row steps are ~0.06 ms per CPU expert entry, which is what the
regression's `cpu_win` coefficient says independently. The down-column steps at
fixed CPU entries are the GPU row cost with the pool term removed.

## per-T bucket

| mean T bucket | n | median mean T | median rows_lo | median ms/win | tokens/win | accept | expert hit % | CPU entries/win | KV hit % | rows_lo coef [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.75-2.25 | 1 | 2.15 | 1.77 | 38.4 | 1.77 | 0.667 | 82.9 | 157 | 83.40 | - |
| 2.25-2.75 | 75 | 2.65 | 2.21 | 39.0 | 2.21 | 0.741 | 88.0 | 146 | 98.07 | -2.27 [-10.58, +6.04] |
| 2.75-3.25 | 1410 | 3.08 | 2.46 | 43.5 | 2.46 | 0.705 | 87.7 | 177 | 97.30 | +6.70 [+5.10, +8.30] |
| 3.25-3.75 | 1601 | 3.48 | 2.98 | 50.8 | 2.98 | 0.798 | 85.3 | 227 | 96.47 | +6.25 [+5.09, +7.41] |
| 3.75-4.25 | 452 | 3.86 | 3.49 | 53.4 | 3.49 | 0.869 | 86.6 | 231 | 97.72 | +4.44 [+2.27, +6.62] |
| 4.25-4.75 | 31 | 4.35 | 4.14 | 72.0 | 4.14 | 0.940 | 80.8 | 358 | 98.16 | -2.95 [-14.09, +8.19] |
| 4.75-5.25 | 2 | 4.81 | 4.57 | 67.8 | 4.57 | 0.936 | 84.5 | 314 | 93.39 | - |

## per expert-hit bucket

| expert hit % | n | median mean T | median rows_lo | median ms/win | CPU entries/win | rows_lo coef [95% CI] |
|---|---|---|---|---|---|---|
| 0-60 | 3 | 3.47 | 2.97 | 78.4 | 839 | - |
| 60-70 | 35 | 3.50 | 2.91 | 71.9 | 504 | +4.55 [-3.11, +12.22] |
| 70-80 | 491 | 3.45 | 2.98 | 61.6 | 369 | +8.34 [+6.36, +10.32] |
| 80-85 | 874 | 3.39 | 2.91 | 54.4 | 264 | +10.10 [+8.97, +11.22] |
| 85-90 | 1462 | 3.27 | 2.72 | 46.4 | 185 | +8.93 [+8.37, +9.49] |
| 90-95 | 706 | 3.25 | 2.56 | 41.3 | 128 | +6.12 [+5.64, +6.60] |
| 95-101 | 1 | 3.81 | 3.43 | 35.4 | 73 | - |

## per KV-stream hit bucket

| KV hit % | n | median mean T | median rows_lo | median ms/win | rows_lo coef [95% CI] |
|---|---|---|---|---|---|
| 0-80 | 33 | 3.50 | 2.76 | 54.3 | +11.82 [+7.12, +16.52] |
| 80-90 | 54 | 3.22 | 2.69 | 47.5 | +11.40 [+6.44, +16.36] |
| 90-95 | 849 | 3.38 | 2.82 | 50.1 | +8.86 [+8.16, +9.57] |
| 95-99 | 1790 | 3.32 | 2.79 | 47.9 | +7.49 [+6.83, +8.15] |
| 99-101 | 812 | 3.27 | 2.73 | 45.6 | +8.53 [+7.61, +9.46] |

## per context band

| ctx band | n | median ctx | median mean T | median ms/win | expert hit % | rows_lo coef [95% CI] |
|---|---|---|---|---|---|---|
| 0-2,000 | 35 | 108 | 4.20 | 56.2 | 85.1 | +10.23 [+7.62, +12.84] |
| 8,000-20,000 | 154 | 12,460 | 3.69 | 50.2 | 84.9 | +7.89 [+5.21, +10.57] |
| 20,000-50,000 | 459 | 39,446 | 3.33 | 45.3 | 86.1 | +8.18 [+7.01, +9.34] |
| 50,000-100,000 | 954 | 67,474 | 3.26 | 46.4 | 85.6 | +7.82 [+6.80, +8.85] |
| 100,000-160,000 | 744 | 127,465 | 3.27 | 47.9 | 86.8 | +6.44 [+5.38, +7.50] |
| 160,000-220,000 | 558 | 193,644 | 3.35 | 48.6 | 87.2 | +8.29 [+7.40, +9.18] |
| 220,000-inf | 667 | 263,214 | 3.42 | 50.6 | 87.1 | +9.06 [+8.33, +9.78] |

## per --max-context tier

| tier | n | rows_lo coef [95% CI] | expert-hit coef | R2 |
|---|---|---|---|---|
| 262144 | 3440 | +8.26 [+7.80, +8.71] | -1.159 | 0.632 |
| 524288 | 132 | +3.46 [-13.56, +20.48] | -1.656 | 0.049 |

## per process stream (the engines, separated by the cumulative KV counter)

| stream | requests | usable | start line | tier | max_t | median ctx | median mean T | median ms/win | expert hit % | rows_lo coef [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 645 | 630 | 46 | 262,144 | 6 | 198,817 | 3.43 | 44.9 | 88.9 | +6.30 [+5.60, +7.00] |
| 1 | 271 | 263 | 68 | 262,144 | 6 | 132,478 | 3.35 | 50.0 | 86.6 | +6.58 [+5.51, +7.66] |
| 2 | 447 | 423 | 77 | 262,144 | 6 | 128,088 | 3.35 | 47.0 | 86.9 | +7.66 [+6.54, +8.78] |
| 3 | 454 | 421 | 205 | 262,144 | 6 | 167,178 | 3.30 | 49.0 | 85.9 | +8.54 [+5.86, +11.23] |
| 4 | 320 | 310 | 345 | 262,144 | 6 | 133,734 | 3.35 | 51.5 | 85.8 | +5.54 [+4.07, +7.01] |
| 5 | 261 | 244 | 3428 | 262,144 | 6 | 108,388 | 3.20 | 50.9 | 85.7 | +8.57 [+7.34, +9.79] |
| 6 | 274 | 262 | 3692 | 262,144 | 6 | 74,571 | 3.25 | 46.6 | 84.0 | +12.72 [+10.60, +14.84] |
| 7 | 151 | 146 | 3745 | 262,144 | 6 | 81,250 | 3.25 | 46.4 | 83.4 | +10.76 [+8.39, +13.12] |
| 8 | 186 | 180 | 3851 | 262,144 | 6 | 140,943 | 3.24 | 49.1 | 84.7 | +8.74 [+6.27, +11.21] |
| 9 | 115 | 111 | 3927 | 262,144 | 6 | 102,987 | 3.18 | 44.0 | 85.3 | +11.14 [+9.10, +13.18] |
| 10 | 182 | 176 | 3936 | 262,144 | 6 | 77,232 | 3.42 | 46.5 | 88.0 | +8.00 [+6.16, +9.83] |
| 11 | 86 | 79 | 4053 | 262,144 | 6 | 56,120 | 3.32 | 47.9 | 86.8 | +9.62 [+7.34, +11.90] |
| 12 | 42 | 41 | 4217 | 262,144 | 6 | 79,850 | 3.29 | 46.5 | 88.2 | +40.44 [-48.97, +129.85] |
| 15 | 68 | 60 | 5299 | 262,144 | 6 | 39,134 | 3.44 | 56.5 | 83.6 | +12.72 [+7.70, +17.73] |

## the bandwidth floor (arithmetic, every input labelled)

```
expert bytes / token = 48 layers x top-10 x 1,382,400 B = 663,552,000 B = 633 MiB
  blob size from the log: `largest blob 2.66 MB`; the cache line says 7,773 slots = 14.74 GiB
  => 2036145 B/slot from the cache line (the blob is the LARGEST, not the average)
HBM assumed          = 936 GB/s  (LABELLED INFERENCE: spec sheet)
expert floor         = 0.709 ms/token
KV floor             = 0.028 ms/token  (12 QSA layers x 2048 selected cells x 1056 B)
TOTAL floor          = 0.737 ms/token of rows
measured row cost    = 8.29 ms/row  => 11.3x the floor
```

The floor is a **per-row** cost: one more row is one more token's worth of expert
blobs and one more token's worth of selected KV. If the measured row cost is near the
floor the window is bandwidth-bound and batching cannot scale. If it is far above, the
row is paying for launches, barriers and latency, and batching scales until something
else breaks.

## THE VERDICT

### the marginal row cost

rows that PRODUCED a committed token (the lower bound on rows run):
  b = +8.29 ms/row, 95% CI [+7.85, +8.74], t = 36.5
rows ASKED for (the upper bound on rows run):
  b = +11.04 ms/row, 95% CI [+10.48, +11.59], t = 38.9
The true marginal cost of one more row is between them, because the log
brackets the rows that ran:  tokens/window <= rows run <= asked T.

Measured inside narrow expert-cache-hit bands, where the hit rate cannot move
and context is a covariate (this is the cleanest cut the log supports):

| expert hit band | n | b (ms/row) | 95% CI | fixed cost a (ms) | R2 |
|---|---|---|---|---|---|
| 70-80% | 491 | 8.34 | [6.36, 10.32] | 33.7 | 0.15 |
| 80-85% | 874 | 10.10 | [8.97, 11.22] | 22.0 | 0.32 |
| 85-90% | 1462 | 8.93 | [8.37, 9.49] | 19.0 | 0.51 |
| 90-95% | 706 | 6.12 | [5.64, 6.60] | 21.6 | 0.49 |

b range across bands: 6.12 - 10.10 ms/row.  a range: 19.0 - 33.7 ms.
The slope FALLS as the cache hit rate rises, which is the mechanism: the row
cost is mostly the CPU expert pool, not the GPU.

### is it bandwidth or overhead?  (the decomposition)

Fit:  ms/window = a + b_gpu*rows + b_cpu*rows*(1 - expert hit) + c*context

```
bandwidth floor per row   : 0.74 ms/row
measured marginal row cost: 11.04 ms/row (hit-controlled)
                          = 15.0x the floor
b_gpu  (every row, hit or not) : +5.55 ms/row  95% CI [+5.00, +6.11]  t19.6
b_cpu  (per row that MISSES)   : +37.82 ms/row  95% CI [+36.83, +38.81]  t75.0
at the median hit rate 86.5%: b = 5.55 + 37.82 x 0.135 = 10.67 ms/row
  => 48% of the marginal row cost is the CPU expert pool

the marginal row cost as a function of the expert cache hit rate:
   hit  80.0%  ->  b = 13.12 ms/row  (17.8x the floor)
   hit  85.0%  ->  b = 11.22 ms/row  (15.2x the floor)
   hit  88.7%  ->  b =  9.82 ms/row  (13.3x the floor)
   hit  92.0%  ->  b =  8.58 ms/row  (11.6x the floor)
   hit  95.0%  ->  b =  7.44 ms/row  (10.1x the floor)
   hit  98.0%  ->  b =  6.31 ms/row  (8.6x the floor)
   hit 100.0%  ->  b =  5.55 ms/row  (7.5x the floor)

FALSIFIABLE CHECK - the decomposition predicts the T=1 window it never saw:
  a + b_gpu*1 + b_cpu*1*(1-hit) = 9.8 + 5.55 + 37.82 x 0.135 = 22.8 ms
  measured T=1 windows (n=187): median 23.0 ms, p10 16.0, p90 38.4
  PREDICTION INSIDE the measured p10-p90 band.

cross-check, per CPU-pool expert entry: +0.0601 ms each, 95% CI [+0.0574, +0.0628]
  at the median 203 CPU entries/window that is 12.2 ms/window
cross-check, per VRAM-resident expert entry: -0.01472 ms, 95% CI [-0.01669, -0.01275] (t=-14.6)
  ~0 or negative: a GPU-resident expert row is free at this resolution; the
  row cost is the misses, not the hits.
```

### throughput at T = 8 / 16 / 24 / 32 under each slope hypothesis

`a = 11.5 ms` from the fit; the three slope columns are:

* **measured** `b = 11.04 ms/row` - the fit at this log's median expert-cache hit
  rate of 86.5%.
* **full cache** `b = 5.55 ms/row` - the same fit at hit = 100%, i.e. no expert
  ever sent to the CPU pool. This is the REACHABLE target and its lever is VRAM for
  the cache, not a kernel.
* **bandwidth floor** `b = 0.74 ms/row` - not reachable, the bound.

`acc` is a STATED PARAMETER: the log cannot give acceptance at draft depths the box
never ran. `T = B*(D+1)` rows for B sequences each asking D drafts. Baseline (`+0.0%%`)
is B=1, D=3, the depth this box runs (`--spec 4` -> `S_mtp = 4`, generate.cpp:5227).

| B | D | rows T | acc | tokens/win | measured: ms/win | ms/tok | tok/s | vs today | full cache: ms/win | ms/tok | tok/s | vs today | floor: ms/win | ms/tok | tok/s | vs today |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 4 | 0.72 | 3.16 | 55.6 | 17.60 | 56.8 | +0.0% | 33.7 | 10.66 | 93.8 | +0.0% | 14.4 | 4.56 | 219.2 | +0.0% |
| 1 | 5 | 6 | 0.66 | 4.30 | 77.7 | 18.07 | 55.4 | -2.6% | 44.8 | 10.41 | 96.0 | +2.3% | 15.9 | 3.70 | 270.6 | +23.5% |
| 2 | 3 | 8 | 0.72 | 6.32 | 99.8 | 15.78 | 63.4 | +11.5% | 55.9 | 8.84 | 113.1 | +20.5% | 17.4 | 2.75 | 363.9 | +66.1% |
| 2 | 5 | 12 | 0.66 | 8.60 | 143.9 | 16.73 | 59.8 | +5.2% | 78.1 | 9.08 | 110.1 | +17.4% | 20.3 | 2.36 | 423.4 | +93.2% |
| 2 | 7 | 16 | 0.60 | 10.40 | 188.0 | 18.08 | 55.3 | -2.7% | 100.3 | 9.64 | 103.7 | +10.5% | 23.3 | 2.24 | 447.1 | +104.0% |
| 4 | 3 | 16 | 0.72 | 12.64 | 188.0 | 14.88 | 67.2 | +18.3% | 100.3 | 7.93 | 126.0 | +34.3% | 23.3 | 1.84 | 543.4 | +148.0% |
| 4 | 5 | 24 | 0.66 | 17.20 | 276.3 | 16.07 | 62.2 | +9.6% | 144.7 | 8.41 | 118.9 | +26.7% | 29.2 | 1.69 | 590.0 | +169.2% |
| 4 | 7 | 32 | 0.60 | 20.80 | 364.6 | 17.53 | 57.0 | +0.4% | 189.1 | 9.09 | 110.0 | +17.2% | 35.0 | 1.68 | 593.5 | +170.8% |
| 8 | 3 | 32 | 0.72 | 25.28 | 364.6 | 14.42 | 69.3 | +22.0% | 189.1 | 7.48 | 133.7 | +42.5% | 35.0 | 1.39 | 721.3 | +229.2% |
| 8 | 5 | 48 | 0.66 | 34.40 | 541.2 | 15.73 | 63.6 | +11.9% | 277.9 | 8.08 | 123.8 | +31.9% | 46.8 | 1.36 | 734.5 | +235.2% |
| 8 | 7 | 64 | 0.60 | 41.60 | 717.7 | 17.25 | 58.0 | +2.0% | 366.8 | 8.82 | 113.4 | +20.9% | 58.6 | 1.41 | 709.7 | +223.8% |
| 16 | 3 | 64 | 0.72 | 50.56 | 717.7 | 14.20 | 70.4 | +24.0% | 366.8 | 7.25 | 137.9 | +46.9% | 58.6 | 1.16 | 862.5 | +293.6% |
| 16 | 5 | 96 | 0.66 | 68.80 | 1070.9 | 15.56 | 64.2 | +13.1% | 544.4 | 7.91 | 126.4 | +34.7% | 82.2 | 1.19 | 837.1 | +282.0% |
| 24 | 3 | 96 | 0.72 | 75.84 | 1070.9 | 14.12 | 70.8 | +24.6% | 544.4 | 7.18 | 139.3 | +48.5% | 82.2 | 1.08 | 922.7 | +321.0% |
| 32 | 3 | 128 | 0.72 | 101.12 | 1424.0 | 14.08 | 71.0 | +25.0% | 722.1 | 7.14 | 140.0 | +49.3% | 105.8 | 1.05 | 956.1 | +336.3% |



## STRATA_DECODE_TIMING: 0 lines in this log

The direct per-window breakdown was never recorded. Everything above is derived
from per-request lines. That is the gap `slope-ab.sh` arm 1 closes.

## caveats the numbers carry

* Segments are delimited by `session is up`. Two processes alive at once cannot be
  separated by that anchor alone; the `stream` split (see the README) is the
  best available separation and it is an attribution, not a fact.
* `decode_ms` is the request's whole decode phase (generate.cpp:7343). Under
  `--serve-slots >= 2` a slot is pre-empted between windows, so decode_ms includes
  time the conversation spent parked. The swap lines bound that: 742 swaps in this
  log, median 113 ms each, 0.6 s total against 65177 s of decode - under 0.2 %. Pre-emption is NOT what inflates decode_ms.
* mean T is capped by `--spec`: `o.spec = min(spec + 2, 8)` (generate.cpp:1661) and
  `--spec-min-p` truncates each window (`:7176-7179`). Nothing in this log can have
  mean T above the segment's `max_t`, which is 6 everywhere.

wrote bench/decode-slope/baseline/aggregate.json
wrote bench/decode-slope/baseline/requests.csv
