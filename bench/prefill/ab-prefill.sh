#!/usr/bin/env bash
# ab-prefill.sh - A/B Strata's prefill levers at a restart.
#
# READ THIS FIRST: this script STARTS A SECOND ENGINE. It needs the model to load
# again (tens of GiB of RAM and most of a GPU), so on a box that is already
# serving traffic it will not fit and it will make that traffic worse. Run it
# only when the serving processes are stopped and the RAM is actually free.
#
# IT DOES NOT RUN BY ITSELF. It refuses to do anything unless you pass --run.
#
# ---------------------------------------------------------------------------
# STATUS: NOT TESTED HERE. It was written against the documented HTTP API
# (serve/server.py: POST /v1/chat/completions, GET /health) and against the log
# format analyze.py parses, but it was never executed: the box it was written on
# has two live servers on it and no free RAM, and starting an engine is forbidden
# by the owner. Expect to fix small things on the first real run. The analysis
# half (analyze.py) IS tested, against the live log.
# ---------------------------------------------------------------------------
#
# Usage:
#   bench/prefill/ab-prefill.sh [--run] [options] [variant ...]
#
#   --run                 actually do it. Without this the script only prints the
#                         plan and exits.
#   --config FILE         the engine config to base every variant on
#                         (default: strata-iq3_s.json next to this repo)
#   --port N              port for the test server (default 8181 - NOT 8080, so it
#                         cannot collide with the serving one)
#   --python EXE          python to run serve/server.py with (default: python3)
#   --prompts FILE        replay list, prompts separated by a blank line
#                         (default: generated, see REPLAY NOTES below)
#   --max-tokens N        completion length for every replayed prompt (default 24)
#   --rounds N            repeat the whole replay N times (default 2: the first
#                         round warms the caches, the second is the one you read)
#   --out DIR             where to write logs/JSON (default: bench/prefill/ab-<date>)
#   --load-timeout SECS   how long to wait for the engine to load (default 1800)
#   --variant ARGS        add a variant; repeatable. ARGS are the engine
#                         arguments that differ from the base config.
#   --env NAME=VALUE      environment for every variant (STRATA_PREFILL_LEND_PCT
#                         and friends); repeatable.
#
# With no --variant arguments it runs the default ladder:
#   auto        --prefill auto                     (what the box runs today)
#   p4096       --prefill 4096
#   p2048       --prefill 2048
#   sr64        --short-read 64                    (the default, pinned explicitly)
#   sr512       --short-read 512
#   nopark      --conversation-cache-mib 0
#   lend90      STRATA_PREFILL_LEND_PCT=90
#   lend50      STRATA_PREFILL_LEND_PCT=50
#
# The "default parking" side of the parking pair is `auto:`: it inherits the base
# config, where no --conversation-cache-mib is given, so the engine sizes the
# budget to the machine (that is the 1.6 GiB / 0.0 GiB line in the log). Do NOT
# add `--conversation-cache-mib 8192` as the other side: an explicit value is
# never second-guessed by the engine, so on a low-RAM box that asks for trouble.
#
# Note on the lend-percentage baseline: the engine's default is 90 only when
# Prefill::pinned_share() >= 0.9, else 85 (generate.cpp ~3437). This box pins
# only ~7 GiB of a 46.84 GiB arena (`cudaHostRegister limited to 8 GiB`,
# `8 slices pinned (7 GiB)`), so pinned_share is likely far below 0.9 and the
# default here is 85 - not 90. Same for STRATA_PREFILL_RING: ring_slots() is 384
# at pinned_share >= 0.9 and 96 otherwise. Neither value is printed by the log,
# so both are inferred; if you want to A/B against 90, say so explicitly:
#   --env 'lp90:STRATA_PREFILL_LEND_PCT=90'
#
# What it does, per variant: write a variant config, start serve/server.py on it,
# wait for /health, replay the same fixed prompts, stop the server, then run
# analyze.py --json over that variant's log. At the end it calls
# `analyze.py --compare` on all the dumps and prints one table.
#
# Everything it measures comes from the engine's own log lines, so the replay only
# has to be *identical* between variants - it does not have to be a known token
# count. analyze.py's compare table prints each run's fresh-token total first for
# exactly that reason: if the mix differs, read the fitted `a` and the marginal
# rate, not the wall time.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="$ROOT/strata-iq3_s.json"
PORT=8181
PYTHON="${PYTHON:-python3}"
PROMPTS=""
MAX_TOKENS=24
ROUNDS=2
OUT=""
LOAD_TIMEOUT=1800
RUN=0
VARIANTS=()
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
    --variant) VARIANTS+=("$2"); shift 2 ;;
    --env) ENVS+=("$2"); shift 2 ;;
    -h|--help) sed -n '2,76p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "ab-prefill.sh: unknown argument: $1" >&2; exit 2 ;;
  esac
done

warn() { printf '%s\n' "$*" >&2; }

warn ""
warn "=============================================================================="
warn " ab-prefill.sh starts ANOTHER Strata engine, on ANOTHER port."
warn ""
warn " That means: the model loads again (tens of GiB of host RAM and most of a"
warn " GPU), and while it loads nothing else on this box can be trusted to stay"
warn " fast. On the box this was written on - 78 GiB RAM with ~2 GiB available"
warn " and three 3090s at ~23.8/24 GiB - it WILL NOT FIT."
warn ""
warn " Stop the serving processes first, or run this on a machine with the RAM"
warn " and VRAM to spare. Check with:  free -g  and  nvidia-smi"
warn "=============================================================================="
warn ""

if [ "$RUN" -ne 1 ]; then
  warn "NOT RUN: pass --run to actually start anything. Nothing was touched."
  exit 1
fi

[ -f "$CONFIG" ] || { warn "ab-prefill.sh: no such config: $CONFIG"; exit 2; }
command -v "$PYTHON" >/dev/null || { warn "ab-prefill.sh: no python: $PYTHON"; exit 2; }
ANALYZE="$ROOT/bench/prefill/analyze.py"
[ -f "$ANALYZE" ] || { warn "ab-prefill.sh: missing $ANALYZE"; exit 2; }

if [ ${#VARIANTS[@]} -eq 0 ]; then
  VARIANTS=("auto:--prefill auto" "p4096:--prefill 4096" "p2048:--prefill 2048"
            "sr64:--short-read 64" "sr512:--short-read 512"
            "nopark:--conversation-cache-mib 0"
            "lend90:" "lend50:")
  ENVS=("lend90:STRATA_PREFILL_LEND_PCT=90" "lend50:STRATA_PREFILL_LEND_PCT=50")
fi

STAMP="$(date +%Y-%m-%d-%H%M)"
[ -n "$OUT" ] || OUT="$ROOT/bench/prefill/ab-$STAMP"
mkdir -p "$OUT" || exit 2

# ---------------------------------------------------------------------------
# REPLAY NOTES - the prompts are synthetic filler, not a benchmark corpus.
# Their exact token count does not matter as long as every variant sees the same
# bytes; the engine's own log records what each prompt actually cost. Sizes are
# chosen to land in the buckets analyze.py reports, because that is where the
# wall time is (see bench/prefill/README.md).
# ---------------------------------------------------------------------------
if [ -z "$PROMPTS" ]; then
  PROMPTS="$OUT/prompts.txt"
  "$PYTHON" - "$PROMPTS" <<'PY'
import sys
out = sys.argv[1]
block = ("You are reviewing a large C++ inference engine. Summarise what this file does, "
         "then list the three changes you would make first. ")
# ~4k, ~1k, ~250, ~64 and ~8k token-ish prompts: the buckets that hold the wall time.
# Prompts are separated by a BLANK LINE (the replay driver splits on blank lines).
sizes = [1000, 250, 64, 1000, 4000, 250, 1000, 8000]
with open(out, "w") as fh:
    for n in sizes:
        fh.write((block * (n // 26 + 1))[:n * 4] + "\nName one colour. Answer in one word.\n\n")
PY
fi
[ -f "$PROMPTS" ] || { warn "ab-prefill.sh: no such prompts file: $PROMPTS"; exit 2; }

# ---------------------------------------------------------------------------
# the replay driver. Written against serve/server.py's documented endpoints:
#   GET  /health                      -> {"status":"ok","loaded":true,...}
#   POST /v1/chat/completions         -> OpenAI shape
# NOT TESTED HERE (see the header). Kept deliberately small.
# ---------------------------------------------------------------------------
REPLAY_PY="$OUT/replay.py"
cat > "$REPLAY_PY" <<'PY'
"""Replay a fixed prompt list against a Strata server. NOT TESTED - see ab-prefill.sh.

Every prompt is sent with temperature 0 and a fixed max_tokens so the two paths
being measured (prompt reading and the first window) are the same between
variants. The conversation is NOT continued between prompts: each request is a
fresh prefix, which is what makes the fresh-token count comparable.
"""
import json
import sys
import time
import urllib.request

base, prompts_file, rounds, max_tokens, load_timeout = (sys.argv[1], sys.argv[2], int(sys.argv[3]),
                                                        int(sys.argv[4]), int(sys.argv[5]))
raw = open(prompts_file, encoding="utf-8").read()
# prompts are separated by a blank line, so a prompt may itself contain newlines
prompts = [p.strip() for p in raw.split("\n\n") if p.strip()]

# wait for the engine to be loaded and idle
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
# build one config per variant from the base config
# ---------------------------------------------------------------------------
mk_config() {  # $1 = name, $2 = extra args (space separated)
  "$PYTHON" - "$CONFIG" "$OUT/$1.json" "$PORT" "$OUT/$1.log" "$2" <<'PY'
import json, sys
src, dst, port, log, extra = sys.argv[1:6]
cfg = json.loads(open(src, encoding="utf-8-sig").read())
args = list(cfg.get("args", []))
# drop the base config's own prefill knobs so the variant's are the only ones
drop = {"--prefill", "--short-read", "--conversation-cache-mib", "--conversation-cache-min-free-mib"}
kept, i = [], 0
while i < len(args):
    if args[i] in drop:
        i += 2
        continue
    kept.append(args[i]); i += 1
cfg["args"] = kept + ([a for a in extra.split() if a] if extra else [])
cfg["log"] = log
cfg["port"] = int(port)
cfg["host"] = "127.0.0.1"
json.dump(cfg, open(dst, "w"), indent=1)
PY
}

JSONS=()
for V in "${VARIANTS[@]}"; do
  NAME="${V%%:*}"; ARGS="${V#*:}"
  [ "$NAME" = "$V" ] && ARGS=""
  LOG="$OUT/$NAME.log"

  ENVSET=()
  for E in "${ENVS[@]:-}"; do
    [ -n "$E" ] || continue
    if [ "${E%%:*}" = "$NAME" ]; then ENVSET+=("${E#*:}"); fi
  done

  if [ ${#ENVSET[@]} -gt 0 ]; then
    ENVCMD=(env "${ENVSET[@]}")
  else
    ENVCMD=()
  fi

  echo "=== variant $NAME: args [$ARGS] env [${ENVSET[*]:-none}]"
  mk_config "$NAME" "$ARGS" || { echo "ab-prefill.sh: could not write the config for $NAME" >&2; continue; }

  ( cd "$ROOT" && "${ENVCMD[@]}" "$PYTHON" serve/server.py --engine strata \
      --config "$OUT/$NAME.json" --port "$PORT" --host 127.0.0.1 > "$OUT/$NAME.server.log" 2>&1 ) &
  SRV=$!

  "$PYTHON" "$REPLAY_PY" "http://127.0.0.1:$PORT" "$PROMPTS" "$ROUNDS" "$MAX_TOKENS" "$LOAD_TIMEOUT"
  RC=$?
  [ $RC -ne 0 ] && echo "ab-prefill.sh: the replay for $NAME exited $RC - read $OUT/$NAME.server.log" >&2

  # stop the server (SIGTERM takes serve/server.py's clean-shutdown path)
  kill -TERM "$SRV" 2>/dev/null
  for _ in $(seq 1 60); do kill -0 "$SRV" 2>/dev/null || break; sleep 1; done
  kill -KILL "$SRV" 2>/dev/null
  wait "$SRV" 2>/dev/null

  if [ -s "$LOG" ]; then
    "$PYTHON" "$ANALYZE" --json "$LOG" > "$OUT/$NAME.json" || echo "ab-prefill.sh: analyze failed for $NAME" >&2
    JSONS+=("$OUT/$NAME.json")
  else
    echo "ab-prefill.sh: $LOG is empty - the engine never started, no numbers for $NAME" >&2
  fi
done

if [ ${#JSONS[@]} -gt 0 ]; then
  echo
  "$PYTHON" "$ANALYZE" --compare "${JSONS[@]}"
  echo
  echo "raw per-request rows:"
  for J in "${JSONS[@]}"; do
    N="$(basename "$J" .json)"
    "$PYTHON" "$ANALYZE" --csv "$OUT/$N" "${OUT}/$N.log" >/dev/null 2>&1
  done
  echo "  $OUT/<variant>/requests.csv"
fi
echo
echo "wrote everything to $OUT"
