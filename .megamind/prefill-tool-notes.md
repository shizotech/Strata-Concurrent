# S0.2 — prefill cost tool, baseline, A/B harness (DONE)

Delivered, all new files, no existing file touched except this note and
`.megamind/prefill-levers.md`:

* `bench/prefill/analyze.py` — stdlib-only log analyzer. Groups by restart
  (`strata generate: session is up` anchor), buckets by fresh tokens (same buckets
  as prefill-levers.md), fits `prompt_ms = a + b*fresh + c*fresh*context` in plain
  Python, reports decode/expert-cache/KV-streaming rates, parking park/restore/
  refuse counts and bytes, and the resolved startup settings.
  Flags: `--json`, `--csv DIR`, `--compare A.json B.json`, `--limit N`, `--short-read N`.
* `bench/prefill/README.md` — how to run it, what each number means, the baseline
  table for this box, the corrections, the ranked lever list with evidence classes.
* `bench/prefill/ab-prefill.sh` — owner-run A/B at a restart. Refuses without
  `--run`, warns loudly that it starts a second engine. Replay half **untested**
  (no engine may be started here); the analyze/compare half is tested end to end.
* `bench/prefill/baseline/{report.md,aggregate.json,requests.csv}` — the
  `--limit 1600` cut of `strata-iq3_s.log`.

## Verified

* `analyze.py strata-iq3_s.log` → clean, exit 0, no stderr.
* `analyze.py --json … | python3 -c "json.load"` → OK.
* `bash -n ab-prefill.sh` → OK; running it without `--run` prints the warning and
  exits without touching anything.
* `cd build && ninja` → "no work to do" (no source touched; mtimes confirm).
* Reproduces the recorded snapshot with `--limit 1324`: 1 324 requests, 4 690 418
  fresh, 21 013.9 s wall, 223.2 tok/s, fit a=9 125 ms / b=1.678 (596 tok/s) /
  c=1.62e-6, R2 0.949.
* The whole ab-prefill pipeline (config builder → per-variant log → `--json` →
  `--compare`) was exercised with the server start stubbed out.

## Things S0.3 should know

* **The ~9.5–10.5 s per-request floor is context-independent.** A 500-token read
  costs ~10 s at 5 k context and at 250 k context. So the fixed cost is per-request
  overhead, not attention over the prefix. That is the strongest argument for the
  sticky-loan change.
* **`want = min(chunk, ceil(fresh/256)*256)`** is the loan a request takes
  (`request_chunk`, generate.cpp:3418). A 256-token loan already costs ~7.3 s.
* `refill()` runs at the end of **every** request (generate.cpp:4620) and
  `lend()` re-takes per segment — that is the churn.
* The log has **no lend/refill timer**. If S0.3 touches that code, adding one
  (behind an env flag, printed per request) would make the lever measurable from
  the log instead of inferred. `analyze.py` would pick up a new line shape with a
  one-line regex addition.
* Parking only exists in restarts 29–30 of this log; it cannot explain the floor in
  the other 28.

## Gotchas found while building the tool

* The log is **not one process**: 34 startup blocks, 30 reached `session is up`, and
  two processes write concurrently. `session is up` grouping cannot separate two
  live processes — `analyze.py` flags it when the `parked=` byte totals exceed the
  restart's own budget.
* `decode expert cache hit rate` and `KV streaming: … block reads` are **cumulative
  per process** — take the last line in a segment, never average them.
* `parked-prefix budget` prints `not 8.0:` with **no GiB suffix** after the asked
  value. Easy regex trap.
* `Prefill::pinned_share()` (which picks `kAutoLendPct` 90-vs-85 and `ring_slots()`
  384-vs-96) is **not printed**, but it is derivable from the `expert arena:` line:
  `cudaHostRegister PORTABLE ok` = whole arena registered (>= 0.9), `N slices pinned
  (7 GiB)` or `FAILED` = below 0.9. On this log: 18 restarts whole, 10 at 7 GiB, 2 at
  0 GiB. `analyze.py` reports it per restart. Do not assume the lend cap is 90.
* The `INFO context=…` line (generate.cpp:4158) goes to **stdout**, which
  `serve/server.py` pipes to the engine, so it never reaches this log file. The
  resolved settings have to come from the `strata generate:` / `strata serve:`
  lines instead.
