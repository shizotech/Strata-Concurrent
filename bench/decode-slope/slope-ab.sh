#!/usr/bin/env bash
# slope-ab.sh - S4.5 Part C: measure the marginal cost of one more decode-window row.
#
# ---------------------------------------------------------------------------
# READ THIS FIRST: THIS SCRIPT STARTS AN ENGINE.
#
# It loads the model again (tens of GiB of host RAM and most of a GPU) and sends
# real requests. On a box that is already serving traffic it will not fit and it
# will make that traffic worse. Run it only when the serving processes are
# stopped and the RAM is actually free:  free -g  and  nvidia-smi.
#
# IT DOES NOT RUN BY ITSELF. It refuses to do anything unless you pass --run.
# It was NEVER EXECUTED here: the box it was written on has two live servers and
# starting an engine is forbidden by the owner. The analysis half (analyze.py) IS
# tested, against the live log. Expect small fixes on the first real run.
# ---------------------------------------------------------------------------
#
# WHAT IT MEASURES, AND WHY
# -------------------------
# The Stage 4 batching decision rests on one number: `b` in
#
#     window_ms = a + b * T
#
# S4.1 quoted a = 24 ms, b = 8.4 ms from two hand-picked log lines. S4.5's offline
# analysis (bench/decode-slope/analyze.py, and §9 of docs/STAGE4-BATCH-DECODE.md)
# re-derived it from 3 800 requests and found:
#
#   * the naive fit is garbage: b = 17 ms/row at R^2 = 0.03, because ms/window is
#     dominated by CONTEXT (30-1660 ms across the log) not by rows (2-5).
#   * controlling for the expert-cache hit rate: b = 8.3 ms/row [7.86, 8.75] on the
#     rows that produced a committed token, 11.1 ms/row [10.5, 11.6] on the rows
#     ASKED for, at R^2 = 0.65.
#   * and it DECOMPOSES: b_gpu = 5.6 ms per row that runs at all, plus
#     b_cpu = 37.7 ms per row whose expert MISSED the VRAM cache. At the log's
#     median 86.5 % hit rate that is ~48 % of the slope from the CPU expert pool.
#   * the bandwidth floor is 0.74 ms/row, so the row costs 8-15x the floor.
#
# That last pair is the whole question, and the offline log cannot settle it,
# because in this log rows, hit rate and context all move together. This script
# moves ONE knob at a time.
#
# THE ARMS
# --------
# Every arm is the SAME replay against the SAME config except for the knobs named.
# `analyze.py` prints, per arm: the fitted b with its CI, the b_gpu / b_cpu
# decomposition, the fixed cost a, and the T=1 window it predicts vs the T=1
# window it measures.
#
#  1. `base`      - the box's own config, unchanged. The reference.
#  2. `dt`        - identical to base, but STRATA_DECODE_TIMING=1. This is the
#                   ONLY arm that produces a DIRECT per-window measurement
#                   (`N windows, avg T x, y tokens/window, z ms/window = verify … +
#                   commit/emit … + draft …`, generate.cpp:7344-7358). Every other
#                   number in this programme is derived from per-request lines; this
#                   one is not. It proves or kills the derivation. Run it first.
#  3. `spec2`     - `--spec 2`  -> o.spec = 4, S_mtp = 2 (generate.cpp:1661, :5227).
#  4. `spec4`     - `--spec 4`  -> o.spec = 6, S_mtp = 4. (the box's value)
#  5. `spec6`     - `--spec 6`  -> o.spec = 8, S_mtp = 6.
#     3/4/5 move the ASKED rows per window while leaving the model, the context
#     distribution and the cache alone. The window's asked T is capped by
#     `kVerifyMaxT = 8`, so `--spec 6` is the top of the reachable range WITHOUT
#     touching a ceiling. Three points on one line, one knob: this is the arm that
#     measures `b` without a regression.
#  6. `minp0`     - `--spec-min-p 0`   -> windows stop being truncated, so rows
#                   ASKED == rows RUN. Compare against `minp5` below.
#  7. `minp5`     - `--spec-min-p 0.5` (the box's value).
#     6/7 isolate the rows_lo / rows_hi bracket. The offline analysis can only
#     bracket the rows that ran; this pair closes the bracket exactly, because
#     `--spec-min-p` is the only thing that truncates a window
#     (generate.cpp:7176-7179).
#  8. `cache2k`   - `--expert-cache 2000`  -> a much smaller VRAM expert cache, so
#                   the hit rate falls and the CPU pool does more of the work.
#  9. `cache6k`   - `--expert-cache 6000`  -> a much bigger one.
#     8/9 are the DECISIVE pair. If `b` collapses toward b_gpu = 5.6 ms as the hit
#     rate goes to ~100 % and blows up as it falls, the slope is the CPU expert
#     pool and the fix is VRAM for the cache, not a batched kernel. If `b` does NOT
#     move with the hit rate, the offline decomposition is wrong and the slope is
#     launch/barrier overhead after all.
# 10. `nopool`    - `--no-pool` -> the GPU-only floor: no CPU expert pool at all,
#     so every expert is either resident or refused. The extreme end of 8/9, and
#     it also removes the pool barrier from the window entirely.
# 11. `adapt`     - `--adapt-every 100000` -> residency effectively static, so the
#     hit rate cannot drift during the run and the arms are comparable.
#
# `--spec 8` and above is NOT in the default ladder: o.spec is clamped to 8 at
# generate.cpp:1661 and kVerifyMaxT is 8, so it cannot be reached without the
# ceiling change, and the ceiling change is not proven safe (see §9.6 of the doc).
# If you want to test it, see `--arm-ceiling` below.
#
# WHAT EACH RESULT MEANS
# ----------------------
#   b ~ 0.7-2 ms/row          -> bandwidth-like. Batching scales to hundreds of
#                                tok/s. Fund S4.4 in full.
#   b ~ 5-6 ms/row at high hit, ~10-13 at low hit -> the CPU pool is the slope.
#                                The fix is cache capacity (a VRAM/quality
#                                decision), and B=2 in one window is worth
#                                roughly what the offline table says.
#   b ~ 8-11 ms/row and INVARIANT to the hit rate -> launch/barrier/latency
#                                bound. Then the only lever is fewer launches per
#                                window, and batching cannot fix it: fund S4.4.0
#                                and S4.4.1 (two sessions) and stop there.
#
# Usage:
#   bench/decode-slope/slope-ab.sh                 # prints the plan, does nothing
#   bench/decode-slope/slope-ab.sh --run           # the default ladder
#   bench/decode-slope/slope-ab.sh --run --only dt,spec2,spec4,spec6
#   bench/decode-slope/slope-ab.sh --run --arm 'cache3k:--expert-cache 3000'
#   bench/decode-slope/slope-ab.sh --run --env 'dt:STRATA_DECODE_TIMING=1'
#   bench/decode-slope/slope-ab.sh --run --arm-ceiling   # ALSO the T>8 arm (see below)
#
#   --run                 actually do it. Without this the script only prints the plan.
#   --config FILE         the engine config to base every arm on (default strata-iq3_s.json)
#   --port N              port for the test server (default 8182, NOT 8080)
#   --python EXE          python to run serve/server.py with (default python3)
#   --prompts FILE        replay list, prompts separated by a blank line
#   --max-tokens N        completion length per request (default 200 - decode needs
#                         many windows per request, unlike ab-prefill.sh's 24)
#   --rounds N            repeat the replay N times (default 2: round 1 warms, round 2 measures)
#   --out DIR             where to write logs/JSON (default bench/decode-slope/ab-<date>)
#   --load-timeout SECS   wait for the engine to load (default 1800)
#   --arm NAME:ARGS       add an arm; repeatable. ARGS are engine args differing from base.
#   --env NAME:VAR=VAL    environment for one arm; repeatable.
#   --only a,b,c          run only these arms (by name)
#   --arm-ceiling         add the `t12` arm, which raises the row ceiling to reach
#                         T = 12. DANGEROUS AND NOT YET SAFE - see the note at the bottom.
#
# Everything it measures comes from the engine's own log lines, so the replay only
# has to be IDENTICAL between arms. analyze.py's compare table prints each arm's
# request count, median rows/window and median expert hit first: if those differ
# between two arms, read the fitted coefficients and not the wall time.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/strata-iq3_s.json"
PORT=8182
PYTHON="${PYTHON:-python3}"
PROMPTS=""
MAX_TOKENS=200
ROUNDS=2
OUT=""
LOAD_TIMEOUT=1800
RUN=0
CEILING=0
ONLY=""
ARMS=()
ENVS=()

while [ $# -gt 0 ]; do
  case "$1" in
    --run) RUN=1; shift ;;
    --config) CONFIG="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --prompts) PROMPTS="$2"; shift 2 ;;
    --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --load-timeout) LOAD_TIMEOUT="$2"; shift 2 ;;
    --arm) ARMS+=("$2"); shift 2 ;;
    --env) ENVS+=("$2"); shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --arm-ceiling) CEILING=1; shift ;;
    -h|--help) sed -n '2,140p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "slope-ab.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

warn() { printf '%s\n' "$*" >&2; }

warn ""
warn "=============================================================================="
warn " slope-ab.sh starts ANOTHER Strata engine, on ANOTHER port."
warn ""
warn " The model loads again (tens of GiB of host RAM and most of a GPU). On a"
warn " box already serving traffic it WILL NOT FIT and it will make that traffic"
warn " worse. Stop the serving processes first, or run this where the RAM and"
warn " VRAM are free.  Check with:  free -g  and  nvidia-smi"
warn "=============================================================================="
warn ""

if [ "$RUN" -ne 1 ]; then
  warn "NOT RUN: pass --run to actually start anything. Nothing was touched."
  warn ""
  warn "The default ladder (11 arms, each a full engine start + replay):"
  warn "  base      (the box's own config)"
  warn "  dt        STRATA_DECODE_TIMING=1        <- the direct per-window measurement"
  warn "  spec2 / spec4 / spec6                   <- move asked rows per window"
  warn "  minp0 / minp5                           <- close the rows-ran bracket"
  warn "  cache2k / cache6k / nopool              <- move the expert-cache hit rate"
  warn "  adapt       --adapt-every 100000        <- freeze residency so arms are comparable"
  warn ""
  warn "Run the first four before the rest; the 'dt' arm alone can invalidate everything else."
  exit 1
fi

[ -f "$CONFIG" ] || { warn "slope-ab.sh: no such config: $CONFIG"; exit 2; }
command -v "$PYTHON" >/dev/null || { warn "slope-ab.sh: no python: $PYTHON"; exit 2; }
ANALYZE="$ROOT/bench/decode-slope/analyze.py"
[ -f "$ANALYZE" ] || { warn "slope-ab.sh: missing $ANALYZE"; exit 2; }

if [ ${#ARMS[@]} -eq 0 ]; then
  ARMS=("base:"
        "dt:"
        "spec2:--spec 2"
        "spec4:--spec 4"
        "spec6:--spec 6"
        "minp0:--spec-min-p 0"
        "minp5:--spec-min-p 0.5"
        "cache2k:--expert-cache 2000"
        "cache6k:--expert-cache 6000"
        "nopool:--no-pool"
        "adapt:--adapt-every 100000")
  ENVS=("dt:STRATA_DECODE_TIMING=1")
fi

if [ "$CEILING" -eq 1 ]; then
  # NOT SAFE YET. Reaching T = 12 needs kVerifyMaxT, GMAX, MAX_NCOLS, cpu::MAXT,
  # kFusedGrMaxT, kMaxWindowEntries and the two `n_tok > 8` guards raised TOGETHER
  # (docs/STAGE4-BATCH-DECODE.md §9.1 and §9.7). Raising kVerifyMaxT alone makes
  # s2_expert_grouped.cu:554/:616 `min(..., GMAX)` SILENTLY DROP routed experts:
  # wrong tokens, no error. This arm exists so the owner can see what it would
  # take; it is deliberately not in the default ladder and it does not apply any
  # patch by itself.
  ARMS+=("t12:--spec 10")
  warn "slope-ab.sh: --arm-ceiling adds the 't12' arm, which needs the six ceilings raised"
  warn "  in the SAME build first (§9.1). Without that build it will refuse to start"
  warn "  (verify.cpp:146 'the window must hold 2..8 tokens') or, worse, run and drop"
  warn "  expert entries silently. Only use it with a build that has all six raised."
fi

STAMP="$(date +%Y-%m-%d-%H%M)"
[ -n "$OUT" ] || OUT="$ROOT/bench/decode-slope/ab-$STAMP"
mkdir -p "$OUT" || exit 2

# ---------------------------------------------------------------------------
# The replay. Decode needs MANY windows per request, so the prompts are short
# and the completions long - the opposite of ab-prefill.sh. Prompts are
# separated by a BLANK LINE.
# ---------------------------------------------------------------------------
if [ -z "$PROMPTS" ]; then
  PROMPTS="$OUT/prompts.txt"
  "$PYTHON" - "$PROMPTS" <<'PY'
import sys
out = sys.argv[1]
# A handful of distinct short prompts, each asked to produce a long answer, so
# every request runs dozens of verify windows at a similar context length. That
# is what makes rows/window the thing that varies and context nearly not.
seeds = [
    "Describe, in order, everything that happens inside a computer when you type a URL and press Enter.",
    "Write a detailed history of the development of container shipping, from the 1950s to today.",
    "Explain how a jet engine works, in as much detail as you can, covering every subsystem.",
    "List and describe every major river system in Eurasia, working from west to east.",
    "Give a thorough account of the causes and consequences of the 1907 Panic.",
    "Describe the complete lifecycle of a star from molecular cloud to remnant, in detail.",
]
with open(out, "w") as fh:
    for s in seeds:
        fh.write(s + "\n\n")
PY
fi
[ -f "$PROMPTS" ] || { warn "slope-ab.sh: no such prompts file: $PROMPTS"; exit 2; }

# ---------------------------------------------------------------------------
# the replay driver. Written against serve/server.py's documented endpoints:
#   GET  /health                      -> {"status":"ok","loaded":true,...}
#   POST /v1/chat/completions         -> OpenAI shape
# NOT TESTED HERE (see the header). Kept deliberately small.
# ---------------------------------------------------------------------------
REPLAY_PY="$OUT/replay.py"
cat > "$REPLAY_PY" <<'PY'
"""Replay a fixed prompt list against a Strata server, long completions.

NOT TESTED - see slope-ab.sh's header. temperature 0 and a fixed max_tokens so
the window sequence is the same between arms. Each prompt is a fresh prefix.
Requests are sent ONE AT A TIME: this measures one conversation's windows, and
the concurrency question is a different experiment (see --serve-slots).
"""
import json
import sys
import time
import urllib.request

base, prompts_file, rounds, max_tokens, load_timeout = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                                        int(sys.argv[4]), int(sys.argv[5]))
prompts = [p.strip() for p in open(prompts_file, encoding="utf-8").read().split("\n\n") if p.strip()]

deadline = time.time() + load_timeout
while time.time() < deadline:
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/health", timeout=10) as r:
            h = json.load(r)
        if h.get("status") == "ok" and h.get("loaded"):
            break
    except Exception:
        pass
    time.sleep(2)
else:
    print("replay: the server never reported loaded", file=sys.stderr)
    sys.exit(1)

model = None
try:
    with urllib.request.urlopen(base.rstrip("/") + "/v1/models", timeout=10) as r:
        data = json.load(r).get("data") or []
        model = data[0]["id"] if data else None
except Exception:
    pass

sent = 0
for rnd in range(rounds):
    for i, p in enumerate(prompts):
        body = {"messages": [{"role": "user", "content": p}], "max_tokens": max_tokens,
                "temperature": 0, "stream": False}
        if model:
            body["model"] = model
        req = urllib.request.Request(base.rstrip("/") + "/v1/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=3600) as r:
                json.load(r)
            print("round %d prompt %d: %.1f s" % (rnd, i, time.time() - t0), flush=True)
            sent += 1
        except Exception as e:
            print("round %d prompt %d: FAILED %s" % (rnd, i, e), file=sys.stderr, flush=True)
print("replay: %d requests sent" % sent, flush=True)
PY

# ---------------------------------------------------------------------------
# one config per arm, from the base config
# ---------------------------------------------------------------------------
mk_config() {  # $1 = name, $2 = extra args (space separated)
  "$PYTHON" - "$CONFIG" "$OUT/$1.json" "$PORT" "$OUT/$1.log" "$2" <<'PY'
import json, sys
src, dst, port, log, extra = sys.argv[1:6]
cfg = json.loads(open(src, encoding="utf-8-sig").read())
args = list(cfg.get("args", []))
# drop the base config's own decode knobs so the arm's are the only ones.
# --serve-slots is forced to 1: this experiment measures ONE window's rows, and a
# second slot would put pre-emption inside decode_ms.
drop = {"--spec", "--spec-min-p", "--spec-split", "--mtp-max-t", "--expert-cache",
        "--adapt-every", "--no-pool", "--serve-slots", "--decode-tokens"}
kept, i = [], 0
while i < len(args):
    if args[i] in drop:
        i += 2
        continue
    kept.append(args[i]); i += 1
cfg["args"] = kept + ["--serve-slots", "1"] + ([a for a in extra.split() if a] if extra else [])
cfg["log"] = log
cfg["port"] = int(port)
cfg["host"] = "127.0.0.1"
json.dump(cfg, open(dst, "w"), indent=1)
PY
}

JSONS=()
for A in "${ARMS[@]}"; do
  NAME="${A%%:*}"; ARGS="${A#*:}"
  [ "$NAME" = "$A" ] && ARGS=""
  if [ -n "$ONLY" ] && ! echo ",$ONLY," | grep -q ",$NAME,"; then continue; fi
  LOG="$OUT/$NAME.log"

  ENVSET=()
  for E in "${ENVS[@]:-}"; do
    [ -n "$E" ] || continue
    if [ "${E%%:*}" = "$NAME" ]; then ENVSET+=("${E#*:}"); fi
  done
  # the timing switches every arm needs, and analyze.py's parser expects
  ENVSET+=("STRATA_DECODE_TIMING=1" "STRATA_SERVE_ACTIVITY_S=5")

  if [ ${#ENVSET[@]} -gt 0 ]; then
    ENVCMD=(env "${ENVSET[@]}")
  else
    ENVCMD=()
  fi

  echo "=== arm $NAME: args [$ARGS] env [${ENVSET[*]:-none}]"
  mk_config "$NAME" "$ARGS" || { echo "slope-ab.sh: could not write the config for $NAME" >&2; continue; }

  ( cd "$ROOT" && "${ENVCMD[@]}" "$PYTHON" serve/server.py --engine strata \
      --config "$OUT/$NAME.json" --port "$PORT" --host 127.0.0.1 > "$OUT/$NAME.server.log" 2>&1 ) &
  SRV=$!

  "$PYTHON" "$REPLAY_PY" "http://127.0.0.1:$PORT" "$PROMPTS" "$ROUNDS" "$MAX_TOKENS" "$LOAD_TIMEOUT"
  RC=$?
  [ $RC -ne 0 ] && echo "slope-ab.sh: the replay for $NAME exited $RC - read $OUT/$NAME.server.log" >&2

  # stop the server (SIGTERM takes serve/server.py's clean-shutdown path)
  kill -TERM "$SRV" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -KILL "$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null

  if [ -s "$LOG" ]; then
    "$PYTHON" "$ANALYZE" --json "$OUT/$NAME.json" "$LOG" > "$OUT/$NAME.report.md" \
      || echo "slope-ab.sh: analyze failed for $NAME" >&2
    JSONS+=("$OUT/$NAME.json")
  else
    echo "slope-ab.sh: $LOG is empty - the engine never started, no numbers for $NAME" >&2
  fi
done

if [ ${#JSONS[@]} -gt 0 ]; then
  echo
  "$PYTHON" "$ANALYZE" --compare "${JSONS[@]}" | tee "$OUT/compare.md"
  echo
  echo "per-arm full reports: $OUT/<arm>.report.md   per-request rows:"
  for J in "${JSONS[@]}"; do
    N="$(basename "$J" .json)"
    "$PYTHON" "$ANALYZE" --csv "$OUT/$N" "${OUT}/$N.log" >/dev/null 2>&1
  done
  echo "  $OUT/<arm>/requests.csv"
fi
echo
echo "wrote everything to $OUT"
