#!/usr/bin/env python3
"""Analyze Strata's prefill (prompt-reading) cost from a server log. Offline only.

    python3 bench/prefill/analyze.py strata-iq3_s.log
    python3 bench/prefill/analyze.py --json strata-iq3_s.log > run.json
    python3 bench/prefill/analyze.py --csv bench/prefill/baseline strata-iq3_s.log

It reads the lines `src/program/generate.cpp` already writes to a `--serve` log.
It starts nothing, loads nothing and touches no device: every number is either in
the log or arithmetic over what is in the log.

Per engine restart and in aggregate it reports:

  * requests, prompt tokens split into reused (prefix hit) and fresh (what the
    prompt path actually had to read), and prompt wall time;
  * effective prompt tok/s bucketed by *fresh* token count - the buckets are the
    ones in `.megamind/prefill-levers.md`, so the tables line up with it;
  * the least-squares fit  prompt_ms = a + b*fresh + c*fresh*context, where
    context is the prompt length the engine reports (reused + fresh);
  * decode tok/s, expert-cache hit rate, KV-streaming hit rate;
  * conversation-cache park / restore / refuse counts and bytes;
  * the settings the engine resolved at startup (chunk, per-stage loans, PCIe
    probe per device, expert-cache slots, kv_resident, max-context, ...).

`--json` dumps all of it. `--csv DIR` writes one row per request so a later run
can be diffed against this one.

Line shapes come from generate.cpp: the request line at ~5048, the serve-side
startup lines at ~3505-3615, the generate-side ones in the same block. The
hit-rate and KV-streaming lines are CUMULATIVE per process, so the tool keeps the
LAST one in a restart's segment instead of averaging them.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys

N = r"(\d+(?:\.\d+)?)"

# --- one line per request (generate.cpp ~5048) -----------------------------
RE_REQ = re.compile(
    r"^strata serve: prompt " + N + r" tokens = " + N + r" reused \+ " + N +
    r" read in " + N + r" ms \(" + N + r" tok/s\), " +
    N + r" generated in " + N + r" ms \(" + N + r" tok/s\), " +
    r"drafts accepted " + N + r" of " + N + r", " + N + r" checkpoints( \(cancelled\))?"
)

# --- the restart anchor ----------------------------------------------------
RE_ANCHOR = re.compile(r"^strata generate: session is up \(engine (\S+)\)")

# --- startup settings. "before" lines are emitted before the anchor and are
#     attributed forward to it; "after" lines follow the anchor. -------------
BEFORE = [
    ("pcie",        r"^strata generate: PCIe probe: " + N + r" GB/s host->device -> pcie_frac " + N +
                    r" \(default " + N + r"\)"),
    ("stage_pcie",  r"^strata generate: layer split: CUDA(\d+) PCIe probe " + N + r" GB/s -> pcie_frac " + N),
    ("cache0",      r"^strata generate: expert cache " + N + r" slots, " + N + r" GiB of VRAM"),
    ("stage_cache", r"^strata generate: layer split: CUDA(\d+) runs layers (\d+)-(\d+), expert cache " + N +
                    r" slots \(" + N + r" GiB\), " + N + r" of its " + N + r" profiled pairs"),
    ("split",       r"^strata generate: layer split across (\d+) GPUs: (.*) \(split (\w+)\)"),
    ("split_auto",  r"^strata generate: layer split auto: K=([0-9,]+) - predicted " + N +
                    r" ms per decode window; the caches hold " + N + r" of " + N +
                    r" profiled pairs \(~" + N + r"% of the routed mass\)"),
    ("profile",     r"^strata generate: profile \S+: " + N + r" ranked pairs, built for " + N + r" slots"),
    ("prefilled",   r"^strata generate: pre-filled " + N + r" of " + N + r" slots from the profile"),
    ("max_context", r"--max-context " + N + r" against a trained context of " + N),
    ("rope",        r"^strata generate: rope scaling (\w+), factor " + N),
    ("kv_stream",   r"^strata generate: KV streaming: " + N + r" of " + N +
                    r" cells per QSA layer in VRAM, the K/V in " + N + r" GiB of pinned RAM"),
    ("arena_load",  r"^strata generate: loaded " + N + r" GiB at " + N + r" GiB/s"),
    ("arena_borrow", r"SHARED ARENA BORROWED|expert arena borrowed from another process"),
    ("arena_pub",   r"SHARED ARENA PUBLISHED"),
    ("arena_reg",   r"^strata generate: expert arena: .*"),
    ("cache_auto",  r"^strata generate: expert cache auto: " + N + r" GiB free, " + N +
                    r" MiB reserved \(\+" + N + r" MiB for the draft head\) -> " + N + r" slots"),
    ("max_blob",    r"largest blob " + N + r" MB"),
    ("pool_workers", r"^strata generate: (\d+) expert-pool workers \+ the host thread"),
    ("cores",       r"^strata generate: cores: host (\d+) \+ workers ([0-9,]+)"),
]
AFTER = [
    ("chunk_auto",  r"^strata serve: prompt chunk auto: " + N + r" tokens"),
    ("chunk_forced", r"^strata serve: prompt chunk " + N + r" -> " + N + r" tokens so its buffers fit"),
    ("chunk_none",  r"^strata serve: no stage can lend the prompt path its " + N + r"-token buffers"),
    ("own_buffers", r"^strata serve: the prompt path allocates its own buffers"),
    ("lend0",       r"^strata serve: the prompt path borrows " + N + r" CUDA0 cache slots \(" + N + r" GiB\)"),
    ("lend_stage",  r"^strata serve:   CUDA(\d+) prompt path borrows " + N + r" of its " + N +
                    r" slots \(" + N + r" GiB\)"),
    ("vram_free",   r"^strata serve: " + N + r" MiB of VRAM free with everything loaded"),
    ("park_budget", r"^strata serve: parked-prefix budget " + N + r" GiB, not " + N + r": only " + N +
                    r" GiB of RAM is free above the " + N + r" GiB parking floor and " + N +
                    r" GiB of request headroom \(" + N + r" slots"),
]
RE_PARK = re.compile(r"^strata serve: conversation cache: parked " + N + r" tokens in " + N +
                     r" ms; parked=" + N + r" bytes=" + N + r" evictions=" + N +
                     r" snapshot_bytes=" + N + r" reused_kv_bytes=" + N)
RE_RESTORE = re.compile(r"^strata serve: conversation cache: restored " + N + r" tokens \((checkpoint|live)\) in " +
                        N + r" ms; parked=" + N + r" bytes=" + N)
RE_REFUSE = re.compile(r"^strata serve: conversation cache: skip parking \(physical RAM admission; need " + N +
                       r" MiB plus " + N + r" MiB floor")
RE_HIT = re.compile(r"^strata serve: decode expert cache hit rate: " + N + r"% \(" + N + r" hits / " + N + r" lookups\)")
RE_KV = re.compile(r"^strata serve: KV streaming: " + N + r"% of " + N + r" block reads hit VRAM, " + N +
                   r" MiB read from RAM")
RE_SUFFIX = re.compile(r"^strata serve: suffix drafts: " + N + r" windows, " + N + r" of " + N + r" drafts accepted")

# the buckets from .megamind/prefill-levers.md - keep them, the tables must compare
BUCKETS = [(1, 64), (65, 256), (257, 1024), (1025, 4096), (4097, 16384), (16385, None)]
LABEL = {(1, 64): "1-64 (verify-window path)", (65, 256): "65-256", (257, 1024): "257-1024",
         (1025, 4096): "1k-4k", (4097, 16384): "4k-16k", (16385, None): ">16k"}

# --short-read (generate.cpp:4556): a prompt part of at most this many tokens goes
# through the verify windows, which borrow no expert-cache slots at all. The log
# does not record the value, so it is assumed to be the default; override with
# --short-read if the run used something else.
SHORT_READ = 64

# lines that appear once per startup: a repeat means a new process began
ONE_PER_START = ("pcie", "cache0", "cache_auto", "arena_load", "max_context", "kv_stream", "profile", "prefilled",
                 "split", "split_auto", "rope", "pool_workers", "max_blob", "cores")

# a fresh process start: llama.cpp prints this before anything else. Used only to
# count startup blocks - a block that dies before "session is up" has no anchor.
RE_BLOCK = re.compile(r"^load_hparams: Qwen-VL models require")


def num(s):
    return float(s)


class Restart:
    """Everything the log says between one `session is up` and the next."""

    def __init__(self, index, line):
        self.index = index
        self.line = line
        self.engine = None
        self.s = {}          # scalar settings
        self.lists = {}      # per-device settings
        self.requests = []
        self.park = {"count": 0, "tokens": 0, "ms": 0.0, "snapshot_bytes": 0, "evictions": 0,
                     "reused_kv_bytes": 0, "max_bytes": 0}
        self.restore = {"checkpoint": 0, "live": 0, "tokens": 0, "ms": 0.0}
        self.refusals = {"count": 0, "need_mib_max": None}
        self.hit = None      # cumulative per process: keep the last
        self.kv = None       # cumulative per process: keep the last
        self.suffix = {"windows": 0, "accepted": 0, "offered": 0}

    def apply(self, name, m):
        g = m.groups()
        if name == "pcie":
            self.s["pcie_gibs"], self.s["pcie_frac"] = num(g[0]), num(g[1])
        elif name == "stage_pcie":
            self.lists.setdefault("stage_pcie", {})[g[0]] = (num(g[1]), num(g[2]))
        elif name == "cache0":
            self.s["cache0_slots"], self.s["cache0_gib"] = int(num(g[0])), num(g[1])
        elif name == "stage_cache":
            self.lists.setdefault("stage_cache", {})[g[0]] = {
                "layers": "%s-%s" % (g[1], g[2]), "slots": int(num(g[3])), "gib": num(g[4]),
                "pairs": int(num(g[5])), "of": int(num(g[6]))}
        elif name == "split":
            self.s["split"] = {"gpus": int(num(g[0])), "order": g[1], "mode": g[2]}
        elif name == "split_auto":
            self.s["split_auto"] = {"k": g[0], "ms_per_window": num(g[1]), "held": int(num(g[2])),
                                    "total": int(num(g[3])), "mass_pct": num(g[4])}
        elif name == "profile":
            self.s["profile_pairs"], self.s["profile_slots"] = int(num(g[0])), int(num(g[1]))
        elif name == "prefilled":
            self.s["prefilled"], self.s["prefill_slots"] = int(num(g[0])), int(num(g[1]))
        elif name == "max_context":
            self.s["max_context"], self.s["trained_context"] = int(num(g[0])), int(num(g[1]))
        elif name == "rope":
            self.s["rope"] = "%s x%s" % (g[0], g[1])
        elif name == "kv_stream":
            self.s["kv_resident"], self.s["kv_cells"], self.s["kv_pinned_gib"] = (int(num(g[0])),
                                                                                 int(num(g[1])), num(g[2]))
        elif name == "arena_load":
            self.s["arena_gib"], self.s["arena_rate_gibs"] = num(g[0]), num(g[1])
        elif name == "arena_borrow":
            self.s["arena"] = "borrowed"
        elif name == "arena_pub":
            self.s["arena"] = "published"
        elif name == "arena_reg":
            t = g[0] if g else m.group(0)
            t = t.split("expert arena:", 1)[-1].strip()
            if "PORTABLE ok" in t:
                # the whole arena is host-registered, so pinned() is true for every blob
                self.s["arena_reg"] = "whole arena registered"
                self.s["pinned_share_regime"] = ">= 0.9 (lend cap 90 %, ring 384)"
            elif "slices pinned" in t:
                gib = re.search(r"(\d+) slices pinned \((\d+) GiB\)", t)
                self.s["arena_reg"] = "partial: %s" % (gib.group(0) if gib else t)
                self.s["pinned_share_regime"] = "< 0.9 (lend cap 85 %, ring 96)"
            elif "FAILED" in t:
                self.s["arena_reg"] = "registration failed: %s" % t.split("FAILED")[0].split(": ")[-1]
                self.s["pinned_share_regime"] = "< 0.9 (lend cap 85 %, ring 96)"
        elif name == "cache_auto":
            self.s["cache_auto"] = {"free_gib": num(g[0]), "reserve_mib": int(num(g[1])),
                                    "draft_mib": int(num(g[2])), "slots": int(num(g[3]))}
        elif name == "max_blob":
            self.s["max_blob_mib"] = num(g[0])
        elif name == "pool_workers":
            self.s["pool_workers"] = int(num(g[0]))
        elif name == "cores":
            self.s["cores"] = "host %s + workers %s" % (g[0], g[1])
        elif name == "chunk_auto":
            self.s["chunk"], self.s["chunk_mode"] = int(num(g[0])), "auto"
        elif name == "chunk_forced":
            self.s["chunk_forced"] = "%s -> %s" % (g[0], g[1])
        elif name == "chunk_none":
            self.s["chunk_none"] = int(num(g[0]))
        elif name == "own_buffers":
            self.s["chunk_mode"] = "no loan (the prompt path allocates its own buffers)"
        elif name == "lend0":
            self.s["lend0_slots"], self.s["lend0_gib"] = int(num(g[0])), num(g[1])
        elif name == "lend_stage":
            self.lists.setdefault("lend_stage", {})[g[0]] = {"slots": int(num(g[1])), "of": int(num(g[2])),
                                                             "gib": num(g[3])}
        elif name == "vram_free":
            self.s["vram_free_mib"] = int(num(g[0]))
        elif name == "park_budget":
            self.s["park_budget"] = {"budget_gib": num(g[0]), "asked_gib": num(g[1]), "free_gib": num(g[2]),
                                     "floor_gib": num(g[3]), "headroom_gib": num(g[4]),
                                     "slots": int(num(g[5]))}

    def add_request(self, origin, m):
        fresh = int(num(m.group(3)))
        self.requests.append({
            "restart": self.index, "line": origin,
            "prompt_tokens": int(num(m.group(1))), "reused": int(num(m.group(2))), "fresh": fresh,
            "prompt_ms": num(m.group(4)), "prompt_tps": num(m.group(5)),
            "gen_tokens": int(num(m.group(6))), "gen_ms": num(m.group(7)), "gen_tps": num(m.group(8)),
            "draft_accepted": int(num(m.group(9))), "draft_offered": int(num(m.group(10))),
            "checkpoints": int(num(m.group(11))), "cancelled": bool(m.group(12)),
            # generate.cpp:4556 - a part of at most `short_read` tokens goes through the verify
            # windows instead of the batched prompt path, so it costs no loan at all
            "path": "window" if fresh <= SHORT_READ else "batched",
        })

    def add_park(self, m):
        p = self.park
        p["count"] += 1
        p["tokens"] += int(num(m.group(1)))
        p["ms"] += num(m.group(2))
        p["evictions"] += int(num(m.group(5)))
        p["snapshot_bytes"] += int(num(m.group(6)))
        p["reused_kv_bytes"] += int(num(m.group(7)))
        p["max_bytes"] = max(p["max_bytes"], int(num(m.group(4))))

    def add_restore(self, m):
        self.restore[m.group(2)] += 1
        self.restore["tokens"] += int(num(m.group(1)))
        self.restore["ms"] += num(m.group(3))

    def add_refusal(self, m):
        self.refusals["count"] += 1
        self.refusals["need_mib_max"] = max(self.refusals["need_mib_max"] or 0, int(num(m.group(1))))

    def add_hit(self, m):
        self.hit = {"pct": num(m.group(1)), "hits": int(num(m.group(2))), "lookups": int(num(m.group(3)))}

    def add_kv(self, m):
        self.kv = {"pct": num(m.group(1)), "reads": int(num(m.group(2))), "ram_mib": num(m.group(3))}

    def add_suffix(self, m):
        self.suffix["windows"] += int(num(m.group(1)))
        self.suffix["accepted"] += int(num(m.group(2)))
        self.suffix["offered"] += int(num(m.group(3)))


BEFORE_RX = [(name, re.compile(pat)) for name, pat in BEFORE]
AFTER_RX = [(name, re.compile(pat)) for name, pat in AFTER]
COUNTERS = [(RE_PARK, "add_park"), (RE_RESTORE, "add_restore"), (RE_REFUSE, "add_refusal"),
            (RE_HIT, "add_hit"), (RE_KV, "add_kv"), (RE_SUFFIX, "add_suffix")]


def parse(paths):
    restarts = []
    pending = {}          # generate-side settings waiting for the next anchor
    cur = None            # the restart the lines after an anchor belong to
    seen = set()          # two processes can share one log: dedupe identical request lines
    stats = {"lines": 0, "requests": 0, "duplicates": 0, "anchors": 0, "blocks": 0}

    def new_restart(ln):
        r = Restart(len(restarts) + 1, ln)
        restarts.append(r)
        return r

    for path in paths:
        with open(path, "r", errors="replace") as fh:
            for ln, raw in enumerate(fh, 1):
                line = raw.rstrip("\n")
                stats["lines"] += 1
                origin = "%s:%d" % (path, ln)

                if RE_BLOCK.match(line):
                    stats["blocks"] += 1
                    pending = {}
                    continue

                m = RE_REQ.match(line)
                if m:
                    if line in seen:
                        stats["duplicates"] += 1
                        continue
                    seen.add(line)
                    stats["requests"] += 1
                    if cur is None:                       # a request before any banner
                        cur = new_restart(ln)
                    cur.add_request(origin, m)
                    continue

                m = RE_ANCHOR.match(line)
                if m:
                    stats["anchors"] += 1
                    cur = new_restart(ln)
                    cur.engine = m.group(1)
                    for name, mm in pending.items():      # generate-side settings for THIS restart
                        cur.apply(name, mm)
                    pending = {}
                    continue

                matched = False
                for name, rx in BEFORE_RX:                # before the anchor: buffer, last wins
                    m = rx.search(line) if not rx.pattern.startswith("^") else rx.match(line)
                    if not m:
                        continue
                    if name in ONE_PER_START and name in pending:
                        pending = {}                   # a repeat: a new startup block began
                    pending[name] = m
                    matched = True
                if matched:
                    continue
                for name, rx in AFTER_RX:                 # after the anchor: attribute backward
                    m = rx.match(line)
                    if not m:
                        continue
                    if cur is None:
                        cur = new_restart(ln)
                    cur.apply(name, m)
                    matched = True
                    break
                if matched:
                    continue
                for rx, fn in COUNTERS:
                    m = rx.match(line)
                    if not m:
                        continue
                    if cur is None:
                        cur = new_restart(ln)
                    getattr(cur, fn)(m)
                    break
    return restarts, stats


# ---------------------------------------------------------------------------
# math
# ---------------------------------------------------------------------------

def least_squares(rows, feats, target):
    """Plain-python normal equations with partial pivoting. Returns (coeffs, r2, n)."""
    A = [[ft(r) for ft in feats] for r in rows]
    y = [float(r[target]) for r in rows]
    k = len(feats)
    if len(A) < k:
        return None, None, len(A)
    M = [[sum(A[i][a] * A[i][b] for i in range(len(A))) for b in range(k)] for a in range(k)]
    v = [sum(A[i][a] * y[i] for i in range(len(A))) for a in range(k)]
    for i in range(k):
        p = max(range(i, k), key=lambda q: abs(M[q][i]))
        if abs(M[p][i]) < 1e-9:
            return None, None, len(A)
        M[i], M[p] = M[p], M[i]
        v[i], v[p] = v[p], v[i]
        for q in range(i + 1, k):
            fac = M[q][i] / M[i][i]
            for c in range(i, k):
                M[q][c] -= fac * M[i][c]
            v[q] -= fac * v[i]
    x = [0.0] * k
    for i in range(k - 1, -1, -1):
        x[i] = (v[i] - sum(M[i][j] * x[j] for j in range(i + 1, k))) / M[i][i]
    pred = [sum(A[i][j] * x[j] for j in range(k)) for i in range(len(A))]
    ybar = sum(y) / len(y)
    sst = sum((q - ybar) ** 2 for q in y)
    sse = sum((q - p) ** 2 for q, p in zip(y, pred))
    return x, (1.0 - sse / sst if sst else None), len(A)


def fit_model(rows):
    """prompt_ms = a + b*fresh + c*fresh*context, context = prompt_tokens (reused + fresh)."""
    feats = [lambda r: 1.0, lambda r: float(r["fresh"]),
             lambda r: float(r["fresh"]) * float(r["prompt_tokens"]) / 1e6]
    x, r2, n = least_squares(rows, feats, "prompt_ms")
    if x is None:
        return None
    a, b, c6 = x                       # c6: ms per fresh token per 1e6 of context
    out = {"n": n, "r2": r2, "a_ms": a, "b_ms_per_fresh": b, "c_ms_per_fresh_per_1e6_ctx": c6,
           "fixed_s": a / 1000.0, "marginal_tok_s": (1000.0 / b if b > 0 else None)}
    res = sorted(r["prompt_ms"] - (a + b * r["fresh"] + c6 * r["fresh"] * r["prompt_tokens"] / 1e6)
                 for r in rows)
    out["residual_median_ms"] = res[len(res) // 2]
    out["residual_mean_abs_ms"] = sum(abs(q) for q in res) / len(res)
    out["residual_p90_ms"] = res[int(len(res) * 0.9)]
    x2, r22, _ = least_squares(rows, [lambda r: 1.0, lambda r: float(r["fresh"])], "prompt_ms")
    if x2 is not None:
        out["two_term"] = {"a_ms": x2[0], "b_ms_per_fresh": x2[1], "r2": r22,
                           "marginal_tok_s": (1000.0 / x2[1] if x2[1] > 0 else None)}
    return out


def bucketize(rows):
    total = sum(r["prompt_ms"] for r in rows)
    out = []
    for lo, hi in BUCKETS:
        sel = [r for r in rows if r["fresh"] >= lo and (hi is None or r["fresh"] <= hi)]
        tok = sum(r["fresh"] for r in sel)
        ms = sum(r["prompt_ms"] for r in sel)
        out.append({"label": LABEL[(lo, hi)], "lo": lo, "hi": hi, "requests": len(sel),
                    "fresh_tokens": tok, "wall_s": ms / 1000.0,
                    "tok_s": (1000.0 * tok / ms if ms > 0 else None),
                    "share_of_wall_pct": (100.0 * ms / total if total else 0.0),
                    "mean_ms_per_request": (ms / len(sel)) if sel else None,
                    "mean_fresh": (tok / len(sel)) if sel else None,
                    "mean_reused": (sum(r["reused"] for r in sel) / len(sel)) if sel else None,
                    "mean_ms_per_fresh": (ms / tok) if tok else None})
    return out


def path_split(rows):
    """The two prompt paths, split at --short-read.

    generate.cpp:4556: a prompt part of at most `short_read` tokens is read through
    the verify windows, which borrow no expert-cache slots; anything longer goes
    through the batched prompt path and pays the lend/refill. The log does not say
    which value was in force, so this uses SHORT_READ (the default).
    """
    total = sum(r["prompt_ms"] for r in rows)
    out = {}
    for name in ("window", "batched"):
        sel = [r for r in rows if r["path"] == name]
        tok = sum(r["fresh"] for r in sel)
        ms = sum(r["prompt_ms"] for r in sel)
        out[name] = {"requests": len(sel), "fresh_tokens": tok, "wall_s": ms / 1000.0,
                     "share_of_wall_pct": (100.0 * ms / total if total else 0.0),
                     "mean_fresh": (tok / len(sel)) if sel else None,
                     "mean_ms": (ms / len(sel)) if sel else None,
                     "ms_per_fresh": (ms / tok) if tok else None,
                     "tok_s": (1000.0 * tok / ms if ms else None)}
    return out


def loan_table(rows):
    """prompt_ms bucketed by the chunk a request actually lends.

    generate.cpp:3418 `request_chunk()` rounds `fresh` up to 256 and caps it at the
    configured chunk, and `lend()` takes exactly that many tokens' worth of buffers -
    so this is the only lever variable the log lets us read per request.
    """
    total = sum(r["prompt_ms"] for r in rows)
    out = []
    for lo, hi in [(1, 256), (257, 512), (513, 1024), (1025, 2048), (2049, 4096), (4097, 8192),
                   (8193, None)]:
        sel = [r for r in rows if r["path"] == "batched" and r["want_tokens"] >= lo
               and (hi is None or r["want_tokens"] <= hi)]
        if not sel:
            continue
        tok = sum(r["fresh"] for r in sel)
        ms = sum(r["prompt_ms"] for r in sel)
        mw = sum(r["want_tokens"] for r in sel) / len(sel)
        out.append({"want_label": ("<=%d" % hi) if hi else ">8192", "mean_want": mw,
                    "requests": len(sel), "fresh_tokens": tok,
                    "wall_s": ms / 1000.0, "share_of_wall_pct": (100.0 * ms / total if total else 0.0),
                    "mean_fresh": tok / len(sel), "mean_ms": ms / len(sel),
                    "ms_per_fresh": (ms / tok if tok else None),
                    "ms_per_want_token": (ms / len(sel) / mw) if mw else None,
                    "tok_s": (1000.0 * tok / ms if ms else None)})
    return out


def context_control(rows):
    """Mean prompt_ms inside one fresh-bucket, split by context.

    This is the honest way to see whether context costs anything: inside a bucket
    `fresh` barely moves, so a rise across the context rows is the attention term.
    """
    out = []
    edges = [(0, 20_000), (20_000, 60_000), (60_000, 120_000), (120_000, 200_000), (200_000, None)]
    for lo, hi in BUCKETS[2:]:
        sel0 = [r for r in rows if r["fresh"] >= lo and (hi is None or r["fresh"] <= hi)]
        if len(sel0) < 6:
            continue
        cells = []
        for clo, chi in edges:
            sel = [r for r in sel0 if r["prompt_tokens"] >= clo and (chi is None or r["prompt_tokens"] < chi)]
            if len(sel) < 3:
                continue
            ms = sum(r["prompt_ms"] for r in sel)
            tok = sum(r["fresh"] for r in sel)
            cells.append({"ctx_lo": clo, "ctx_hi": chi, "requests": len(sel),
                          "mean_fresh": tok / len(sel), "mean_ms": ms / len(sel),
                          "ms_per_fresh": ms / tok if tok else None})
        if len(cells) >= 2:
            out.append({"label": LABEL[(lo, hi)], "cells": cells})
    return out


def derive(rows, chunk):
    """Fill in the fields the log does not carry but the source defines."""
    for q in rows:
        q["path"] = "window" if q["fresh"] <= SHORT_READ else "batched"
        q["want_tokens"] = (min(chunk, ((q["fresh"] + 255) // 256) * 256)
                            if chunk and q["path"] == "batched" else 0)
    return rows


def summarize(r):
    rows = r.requests
    chunk = int(r.s.get("chunk") or r.s.get("chunk_none") or 0) or None
    derive(rows, chunk)
    pms = sum(q["prompt_ms"] for q in rows)
    gms = sum(q["gen_ms"] for q in rows)
    fresh = sum(q["fresh"] for q in rows)
    reused = sum(q["reused"] for q in rows)
    ptok = sum(q["prompt_tokens"] for q in rows)
    gen = sum(q["gen_tokens"] for q in rows)
    s = {"restart": r.index, "log_line": r.line, "engine": r.engine, "requests": len(rows),
         "cancelled": sum(1 for q in rows if q["cancelled"]),
         "prompt_tokens": ptok, "reused_tokens": reused, "fresh_tokens": fresh,
         "prompt_wall_s": pms / 1000.0,
         "prompt_tok_s": (1000.0 * fresh / pms if pms else None),
         "reused_pct": (100.0 * reused / ptok if ptok else None),
         "gen_tokens": gen, "decode_wall_s": gms / 1000.0,
         "decode_tok_s": (1000.0 * gen / gms if gms else None),
         "draft_accepted": sum(q["draft_accepted"] for q in rows),
         "draft_offered": sum(q["draft_offered"] for q in rows),
         "checkpoints": sum(q["checkpoints"] for q in rows),
         "buckets": bucketize(rows), "model": fit_model(rows) if len(rows) >= 4 else None,
         "models": {k: fit_model(v) for k, v in
                    (("all", rows), ("batched", [q for q in rows if q["path"] == "batched"]),
                     ("window", [q for q in rows if q["path"] == "window"])) if len(v) >= 4},
         "loan_table": loan_table(rows),
         "context_control": context_control(rows),
         "path_split": path_split(rows),
         "park": dict(r.park), "restore": dict(r.restore), "refusals": dict(r.refusals),
         "expert_cache_hit": r.hit, "kv_streaming": r.kv, "suffix_drafts": dict(r.suffix),
         "settings": dict(r.s), "per_device": {k: dict(v) for k, v in r.lists.items()}}
    caveats = []
    pb = s["settings"].get("park_budget")
    if pb is not None and r.park["max_bytes"] > (pb["budget_gib"] + 0.05) * 2 ** 30:
        caveats.append("parking: the `parked=` byte totals in this segment reach %.1f GiB but this restart's own "
                       "budget is %.1f GiB - the log has lines from ANOTHER live process interleaved here, so the "
                       "parking counters for this restart are not trustworthy"
                       % (r.park["max_bytes"] / 2 ** 30, pb["budget_gib"]))
    elif pb is None and r.park["count"]:
        caveats.append("parking: this restart logged parks but no `parked-prefix budget` line - that line only "
                       "exists once the default budget is machine-sized, so the budget in force is unknown")
    if not r.park["count"] and not r.refusals["count"] and pb is None and r.requests:
        caveats.append("parking: no parking lines at all in this segment (the binary predates it, or it was off)")
    s["caveats"] = caveats
    if "lend0_slots" in s["settings"]:
        slots = int(s["settings"]["lend0_slots"])
        gib = float(s["settings"].get("lend0_gib", 0.0))
        for v in r.lists.get("lend_stage", {}).values():
            slots += v["slots"]
            gib += v["gib"]
        s["loan_slots_total"], s["loan_gib_total"] = slots, gib
    return s


def merge(restarts):
    """A Restart-shaped object holding every request from every restart."""
    m = Restart(0, 0)
    m.engine = restarts[0].engine if restarts else None
    chunks = {int(r.s["chunk"]) for r in restarts if "chunk" in r.s}
    if len(chunks) == 1:
        m.s["chunk"] = chunks.pop()
    # anything every restart resolved the same way is a property of the run, so keep it
    for key in ("chunk", "lend0_slots", "lend0_gib", "cache0_slots", "pcie_gibs", "vram_free_mib",
                "kv_resident", "max_context", "rope", "split", "park_budget", "arena"):
        vals = {repr(r.s[key]) for r in restarts if key in r.s}
        if len(vals) == 1 and len(restarts) > 1:
            for r in restarts:
                if key in r.s:
                    m.s[key] = r.s[key]
                    break
    for r in restarts:
        m.requests += r.requests
        for key, val in r.lists.items():
            if all(dict(x.lists.get(key, {})) == dict(val) for x in restarts if key in x.lists):
                m.lists.setdefault(key, {}).update(val)
        for k in m.park:
            m.park[k] += r.park[k]
        for k in m.restore:
            m.restore[k] += r.restore[k]
        m.refusals["count"] += r.refusals["count"]
        if r.refusals["need_mib_max"] is not None:
            m.refusals["need_mib_max"] = max(m.refusals["need_mib_max"] or 0, r.refusals["need_mib_max"])
        for k in m.suffix:
            m.suffix[k] += r.suffix[k]
        if r.hit:
            h = m.hit or {"hits": 0, "lookups": 0}
            m.hit = {"pct": None, "hits": h["hits"] + r.hit["hits"], "lookups": h["lookups"] + r.hit["lookups"]}
        if r.kv:
            k0 = m.kv or {"reads": 0, "ram_mib": 0.0}
            m.kv = {"pct": None, "reads": k0["reads"] + r.kv["reads"], "ram_mib": k0["ram_mib"] + r.kv["ram_mib"]}
    if m.hit and m.hit["lookups"]:
        m.hit["pct"] = 100.0 * m.hit["hits"] / m.hit["lookups"]
    return m


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def fmt(x, nd=1):
    return "-" if x is None else "{:,.{p}f}".format(x, p=nd)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] + ["---:"] * (len(header) - 1)) + " |"]
    out += ["| " + " | ".join(c for c in row) + " |" for row in rows]
    return out


def report(restarts, stats, paths):
    A = summarize(merge(restarts))
    print("# Strata prefill cost - %d log file(s), %s lines, %d startup block(s), %d engine restart(s), "
          "%s requests" % (len(paths), fmt(stats["lines"], 0), stats["blocks"], stats["anchors"],
                           fmt(A["requests"], 0)))
    if stats["blocks"] and stats["blocks"] != stats["anchors"]:
        print("  (note: %d startup block(s) but %d reached `session is up` - the rest died during loading)"
              % (stats["blocks"], stats["anchors"]))
    if stats["duplicates"]:
        print("  (%d duplicate request lines skipped: more than one process wrote this log)" % stats["duplicates"])
    if stats.get("limited"):
        print("  (--limit: only the first %s requests in log order are counted)" % fmt(stats["requests"], 0))
    print()

    print("## Resolved settings, per restart")
    rows = []
    for r in restarts:
        s = r.s
        rows.append([str(r.index), str(s.get("chunk", "-")),
                     ("%s / %s GiB" % (fmt(s.get("lend0_slots"), 0), fmt(s.get("lend0_gib"), 2)))
                     if s.get("lend0_slots") is not None else "-",
                     fmt(s.get("cache0_slots"), 0), fmt(s.get("pcie_gibs"), 1),
                     fmt(s.get("vram_free_mib"), 0), fmt(s.get("kv_resident"), 0),
                     fmt(s.get("max_context"), 0)])
    print("\n".join(table(["#", "chunk", "CUDA0 loan slots", "CUDA0 cache slots", "PCIe GB/s",
                           "VRAM free MiB", "kv_resident", "max-context"], rows)))
    print()
    for r in restarts:
        notes = []
        if r.lists.get("lend_stage"):
            notes.append("other stage loans: " + ", ".join(
                "CUDA%s %s of %s slots (%s GiB)" % (d, fmt(v["slots"], 0), fmt(v["of"], 0), fmt(v["gib"], 2))
                for d, v in sorted(r.lists["lend_stage"].items())))
        if r.lists.get("stage_pcie"):
            notes.append("stage PCIe probe: " + ", ".join(
                "CUDA%s %s GB/s -> pcie_frac %s" % (d, fmt(v[0], 1), fmt(v[1], 2))
                for d, v in sorted(r.lists["stage_pcie"].items())))
        if r.lists.get("stage_cache"):
            notes.append("stage caches: " + ", ".join(
                "CUDA%s layers %s, %s slots (%s GiB), %s of %s pairs"
                % (d, v["layers"], fmt(v["slots"], 0), fmt(v["gib"], 2), fmt(v["pairs"], 0), fmt(v["of"], 0))
                for d, v in sorted(r.lists["stage_cache"].items())))
        if "split_auto" in r.s:
            sa = r.s["split_auto"]
            notes.append("split auto: K=%s, %s ms/window predicted, caches hold %s of %s pairs (~%s%% of routed mass)"
                         % (sa["k"], fmt(sa["ms_per_window"], 1), fmt(sa["held"], 0), fmt(sa["total"], 0),
                            fmt(sa["mass_pct"], 1)))
        if "park_budget" in r.s:
            pb = r.s["park_budget"]
            notes.append("parked-prefix budget %s GiB (asked %s GiB; %s GiB free above a %s GiB floor and %s GiB "
                         "headroom; %d slots)" % (fmt(pb["budget_gib"], 1), fmt(pb["asked_gib"], 1),
                                                  fmt(pb["free_gib"], 1), fmt(pb["floor_gib"], 1),
                                                  fmt(pb["headroom_gib"], 1), pb["slots"]))
        if "cache_auto" in r.s:
            ca = r.s["cache_auto"]
            notes.append("expert cache auto: %s GiB free, %d MiB reserved (+%d MiB draft head) -> %d slots, "
                         "the cache ended at %s slots" % (fmt(ca["free_gib"], 2), ca["reserve_mib"], ca["draft_mib"],
                                                          ca["slots"], fmt(r.s.get("cache0_slots"), 0)))
        if "arena_gib" in r.s:
            notes.append("arena: %s GiB at %s GiB/s%s" % (fmt(r.s["arena_gib"], 2), fmt(r.s["arena_rate_gibs"], 2),
                                                          " (BORROWED from another process)"
                                                          if r.s.get("arena") == "borrowed" else ""))
        if "arena_reg" in r.s:
            notes.append("arena registration: %s -> pinned_share %s"
                         % (r.s["arena_reg"], r.s.get("pinned_share_regime", "unknown")))
        if "rope" in r.s:
            notes.append("rope %s against a trained context of %s" % (r.s["rope"], fmt(r.s.get("trained_context"), 0)))
        if notes:
            print("restart %d:" % r.index)
            for n in notes:
                print("  " + n)
            for c in summarize(r).get("caveats", []):
                print("  ! " + c)
            print()

    print("## Per restart")
    rows = []
    for r in restarts:
        s = summarize(r)
        md = s["model"]
        rows.append([str(s["restart"]), fmt(s["requests"], 0), fmt(s["fresh_tokens"], 0),
                     fmt(s["reused_tokens"], 0), fmt(s["prompt_wall_s"], 1), fmt(s["prompt_tok_s"], 1),
                     fmt(s["decode_tok_s"], 1), fmt(md["fixed_s"], 2) if md else "-",
                     fmt(md["marginal_tok_s"], 0) if md else "-", fmt(md["r2"], 3) if md else "-"])
    print("\n".join(table(["#", "reqs", "fresh tok", "reused tok", "prompt wall s", "prompt tok/s",
                           "decode tok/s", "fit a (s)", "fit marginal tok/s", "R2"], rows)))
    print()

    print("## Aggregate")
    rows = [
        ["requests", fmt(A["requests"], 0) + (" (%d cancelled)" % A["cancelled"] if A["cancelled"] else "")],
        ["prompt tokens", fmt(A["prompt_tokens"], 0)],
        ["reused (prefix hit)", "%s (%s%%)" % (fmt(A["reused_tokens"], 0), fmt(A["reused_pct"], 1))],
        ["fresh (the prompt path read)", fmt(A["fresh_tokens"], 0)],
        ["prompt wall time", "%s s" % fmt(A["prompt_wall_s"], 1)],
        ["effective prompt tok/s", fmt(A["prompt_tok_s"], 1)],
        ["generated tokens / decode wall", "%s / %s s" % (fmt(A["gen_tokens"], 0), fmt(A["decode_wall_s"], 1))],
        ["decode tok/s", fmt(A["decode_tok_s"], 1)],
        ["drafts accepted", "%s of %s (%s%%)" % (fmt(A["draft_accepted"], 0), fmt(A["draft_offered"], 0),
                                                 fmt(100.0 * A["draft_accepted"] / A["draft_offered"], 1)
                                                 if A["draft_offered"] else "-")],
        ["checkpoints taken", fmt(A["checkpoints"], 0)],
        ["expert-cache hit", "%s hits / %s lookups = %s%%" % (fmt(A["expert_cache_hit"]["hits"], 0),
                                                              fmt(A["expert_cache_hit"]["lookups"], 0),
                                                              fmt(A["expert_cache_hit"]["pct"], 1))
         if A["expert_cache_hit"] else "-"],
        ["KV streaming block reads", "%s reads, %s MiB read from RAM" % (fmt(A["kv_streaming"]["reads"], 0),
                                                                        fmt(A["kv_streaming"]["ram_mib"], 1))
         if A["kv_streaming"] else "-"],
        ["parking: parked", "%d parks, %s tokens, %s s, %s GiB of snapshots, %d evictions"
         % (A["park"]["count"], fmt(A["park"]["tokens"], 0), fmt(A["park"]["ms"] / 1000.0, 1),
            fmt(A["park"]["snapshot_bytes"] / 2 ** 30, 2), A["park"]["evictions"])],
        ["parking: restored", "%d checkpoint + %d live, %s tokens, %s s"
         % (A["restore"]["checkpoint"], A["restore"]["live"], fmt(A["restore"]["tokens"], 0),
            fmt(A["restore"]["ms"] / 1000.0, 1))],
        ["parking: refused", "%d (physical-RAM admission%s)"
         % (A["refusals"]["count"], ", largest need %s MiB" % fmt(A["refusals"]["need_mib_max"], 0)
            if A["refusals"]["need_mib_max"] else "")],
        ["suffix drafts", "%d windows, %s of %s accepted" % (A["suffix_drafts"]["windows"],
                                                             fmt(A["suffix_drafts"]["accepted"], 0),
                                                             fmt(A["suffix_drafts"]["offered"], 0))],
    ]
    print("\n".join(table(["metric", "value"], rows)))
    parking = [r.index for r in restarts if r.park["count"] or r.refusals["count"]]
    if parking and len(parking) < len(restarts):
        print("\nParking counters above come only from restart(s) %s; the other %d restart(s) logged no parking "
              "at all. Aggregate parking numbers are not comparable across restarts - read the per-restart rows."
              % (", ".join(str(i) for i in parking), len(restarts) - len(parking)))
    print()

    print("## Prompt cost by fresh-token bucket")
    rows = []
    for b in A["buckets"]:
        rows.append([b["label"], fmt(b["requests"], 0), fmt(b["fresh_tokens"], 0), fmt(b["wall_s"], 1),
                     fmt(b["tok_s"], 1), fmt(b["share_of_wall_pct"], 1), fmt(b["mean_ms_per_request"], 0),
                     fmt(b["mean_fresh"], 0), fmt(b["mean_reused"], 0), fmt(b["mean_ms_per_fresh"], 2)])
    print("\n".join(table(["fresh tokens", "requests", "fresh tok", "wall s", "effective tok/s",
                           "% of prompt wall", "mean ms/req", "mean fresh", "mean reused", "ms per fresh tok"],
                          rows)))
    small = sum(b["share_of_wall_pct"] for b in A["buckets"] if b["hi"] is not None and b["hi"] <= 4096)
    print("\n**%s%% of all prompt wall time is in reads of <= 4 000 fresh tokens.**" % fmt(small, 1))
    print()

    md = A["model"]
    if md:
        print("## Fitted cost model (least squares over %s requests)" % fmt(md["n"], 0))
        print("```")
        print("prompt_ms = %.0f + %.3f * fresh + %.3e * fresh * context"
              % (md["a_ms"], md["b_ms_per_fresh"], md["c_ms_per_fresh_per_1e6_ctx"] / 1e6))
        print("            ^%.1f s fixed    ^%.0f tok/s marginal  ^attention term        R2 = %.3f"
              % (md["fixed_s"], md["marginal_tok_s"], md["r2"]))
        print("residuals: median %.0f ms, mean |res| %.0f ms, p90 %.0f ms"
              % (md["residual_median_ms"], md["residual_mean_abs_ms"], md["residual_p90_ms"]))
        t = md["two_term"]
        print("without the context term: a = %.0f ms, %.0f tok/s marginal, R2 = %.3f"
              % (t["a_ms"], t["marginal_tok_s"], t["r2"]))
        mb = A["models"].get("batched")
        if mb:
            print("batched path only (%s of the requests): a = %.0f ms, %.3f ms/fresh (%.0f tok/s), "
                  "%.3e*fresh*context, R2 = %.3f"
                  % (fmt(mb["n"], 0), mb["a_ms"], mb["b_ms_per_fresh"], mb["marginal_tok_s"],
                     mb["c_ms_per_fresh_per_1e6_ctx"] / 1e6, mb["r2"]))
        print("```")
        print("`context` is the prompt length the engine reports (reused + fresh). **Do not read the third")
        print("coefficient as a measurement of the attention cost**: over this workload fresh and context")
        print("move together, so the term is not identified (its sign flips between restarts). The")
        print("context-control table below is the honest version of the same question.")
        print()

    ps = A["path_split"]
    if ps["window"]["requests"] and ps["batched"]["requests"]:
        print("## The two prompt paths (split at --short-read = %d)" % SHORT_READ)
        rows = []
        for name in ("window", "batched"):
            p = ps[name]
            rows.append([name, fmt(p["requests"], 0), fmt(p["fresh_tokens"], 0), fmt(p["wall_s"], 1),
                         fmt(p["share_of_wall_pct"], 1), fmt(p["mean_fresh"], 0), fmt(p["mean_ms"], 0),
                         fmt(p["ms_per_fresh"], 2), fmt(p["tok_s"], 1)])
        print("\n".join(table(["path", "requests", "fresh tok", "wall s", "% of prompt wall",
                               "mean fresh/req", "mean ms/req", "ms per fresh tok", "effective tok/s"], rows)))
        w, b = ps["window"], ps["batched"]
        print("\nThe window path borrows no expert-cache slots, so it pays no lend/refill. But this log only"
              % ())
        print("ever sends ~%d-token segments through it (the assistant header after a checkpoint), so its"
              % round(w["mean_fresh"]))
        print("cost per token is not separable from its fixed cost: mean %.0f ms for %.0f tokens."
              % (w["mean_ms"], w["mean_fresh"]))
        print("The batched path costs ~%.0f s per request at a mean of %.0f fresh tokens. Raising"
              % (b["mean_ms"] / 1000.0, b["mean_fresh"]))
        print("`--short-read` above %d cannot be priced from this log - it needs the A/B script." % SHORT_READ)
        print()

    if A["loan_table"]:
        print("## Cost by the chunk the request actually lends")
        print("(`want = min(chunk, ceil(fresh/256)*256)`, generate.cpp:3418 - what `lend()` takes buffers for)")
        rows = []
        for e in A["loan_table"]:
            rows.append([e["want_label"], fmt(e["mean_want"], 0), fmt(e["requests"], 0), fmt(e["mean_fresh"], 0),
                         fmt(e["mean_ms"], 0), fmt(e["ms_per_fresh"], 2), fmt(e["ms_per_want_token"], 2),
                         fmt(e["tok_s"], 1), fmt(e["share_of_wall_pct"], 1)])
        print("\n".join(table(["want", "mean want", "requests", "mean fresh", "mean ms", "ms per fresh tok",
                               "ms per want-token", "effective tok/s", "% of prompt wall"], rows)))
        print("\n`ms per want-token` is the prompt cost divided by the buffers the request asked for. A")
        print("request never lends more than its own prompt needs, so this table cannot show what a")
        print("different `--prefill` cap would have done - it shows how cost scales with the loan a request")
        print("already took. It is the shape the A/B script has to move along.")
        print()

    if A["context_control"]:
        print("## Context, held inside one fresh-token bucket")
        rows = []
        for grp in A["context_control"]:
            for c in grp["cells"]:
                rows.append([grp["label"], "%s-%s" % (fmt(c["ctx_lo"], 0), fmt(c["ctx_hi"], 0) if c["ctx_hi"] else "inf"),
                             fmt(c["requests"], 0), fmt(c["mean_fresh"], 0), fmt(c["mean_ms"], 0),
                             fmt(c["ms_per_fresh"], 2)])
        print("\n".join(table(["fresh tokens", "context", "requests", "mean fresh", "mean ms",
                               "ms per fresh tok"], rows)))
        print("\nInside a bucket `fresh` barely moves, so a rise across the context rows is the attention")
        print("cost over the cached prefix. Where the rows are flat, context is not what costs time.")
        print()
    return A


COMPARE_METRICS = [
    ("requests", lambda a: fmt(a["requests"], 0)),
    ("fresh tokens", lambda a: fmt(a["fresh_tokens"], 0)),
    ("reused tokens", lambda a: fmt(a["reused_tokens"], 0)),
    ("prompt wall s", lambda a: fmt(a["prompt_wall_s"], 1)),
    ("prompt tok/s", lambda a: fmt(a["prompt_tok_s"], 1)),
    ("decode tok/s", lambda a: fmt(a["decode_tok_s"], 1)),
    ("fit a (s)", lambda a: fmt(a["model"]["fixed_s"], 2) if a["model"] else "-"),
    ("fit marginal tok/s", lambda a: fmt(a["model"]["marginal_tok_s"], 0) if a["model"] else "-"),
    ("fit R2", lambda a: fmt(a["model"]["r2"], 3) if a["model"] else "-"),
    ("batched a (s)", lambda a: fmt(a["models"]["batched"]["fixed_s"], 2)
     if a.get("models", {}).get("batched") else "-"),
    ("batched marginal tok/s", lambda a: fmt(a["models"]["batched"]["marginal_tok_s"], 0)
     if a.get("models", {}).get("batched") else "-"),
    ("chunk", lambda a: fmt(a["settings"].get("chunk"), 0)),
    ("CUDA0 loan slots", lambda a: fmt(a["settings"].get("lend0_slots"), 0)),
    ("loan GiB total", lambda a: fmt(a.get("loan_gib_total"), 2)),
    ("expert-cache hit %", lambda a: fmt(a["expert_cache_hit"]["pct"], 1) if a["expert_cache_hit"] else "-"),
    ("parkings / refusals", lambda a: "%d / %d" % (a["park"]["count"], a["refusals"]["count"])),
    ("parking wall s", lambda a: fmt(a["park"]["ms"] / 1000.0, 1)),
]


def compare(paths):
    runs = []
    for p in paths:
        with open(p) as fh:
            d = json.load(fh)
        runs.append((os.path.basename(p), d["aggregate"]))
    rows = []
    for label, fn in COMPARE_METRICS:
        rows.append([label] + [fn(a) for _, a in runs])
    print("## A/B comparison")
    print("\n".join(table(["metric"] + [n for n, _ in runs], rows)))
    print("\nRead a comparison only when the two runs saw a similar mix: compare `fresh tokens` and the")
    print("`requests` row first. If they differ by more than a few percent, the wall-time row is the mix,")
    print("not the lever - use the fitted `a` and `marginal tok/s` instead.")


def main(argv=None):
    global SHORT_READ
    ap = argparse.ArgumentParser(description="Analyze Strata prefill cost from a server log (offline).")
    ap.add_argument("logs", nargs="+", help="one or more Strata --serve logs")
    ap.add_argument("--json", action="store_true", help="dump the whole analysis as JSON on stdout")
    ap.add_argument("--csv", metavar="DIR", help="write per-request rows (requests.csv) into DIR")
    ap.add_argument("--limit", type=int, metavar="N",
                    help="keep only the first N requests - a serve log keeps growing, so this is how you "
                         "reproduce an earlier snapshot (e.g. --limit 1324 for the numbers in "
                         ".megamind/prefill-levers.md)")
    ap.add_argument("--short-read", type=int, metavar="N", default=SHORT_READ,
                    help="the --short-read value the run used (default %d); it decides which requests went "
                         "through the verify windows instead of the batched prompt path" % SHORT_READ)
    ap.add_argument("--compare", action="store_true",
                    help="instead of parsing logs, compare two or more --json dumps from earlier runs "
                         "(this is what ab-prefill.sh calls)")
    args = ap.parse_args(argv)
    SHORT_READ = args.short_read

    for p in args.logs:
        if not os.path.isfile(p):
            sys.stderr.write("analyze.py: no such file: %s\n" % p)
            return 2

    if args.compare:
        compare(args.logs)
        return 0

    restarts, stats = parse(args.logs)
    if not any(r.requests for r in restarts):
        sys.stderr.write("analyze.py: no `strata serve: prompt ...` lines in %s\n" % ", ".join(args.logs))
        return 1

    if args.limit:
        left = args.limit
        for r in restarts:
            if left <= 0:
                r.requests = []
                continue
            r.requests = r.requests[:left]
            left -= len(r.requests)
        stats["requests"] = sum(len(r.requests) for r in restarts)
        stats["limited"] = True

    if args.csv:
        os.makedirs(args.csv, exist_ok=True)
        path = os.path.join(args.csv, "requests.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["restart", "line", "prompt_tokens", "reused", "fresh", "prompt_ms", "prompt_tps",
                        "gen_tokens", "gen_ms", "gen_tps", "draft_accepted", "draft_offered",
                        "checkpoints", "cancelled"])
            for r in restarts:
                for q in r.requests:
                    w.writerow([q["restart"], q["line"], q["prompt_tokens"], q["reused"], q["fresh"],
                                "%.0f" % q["prompt_ms"], "%.1f" % q["prompt_tps"], q["gen_tokens"],
                                "%.0f" % q["gen_ms"], "%.1f" % q["gen_tps"], q["draft_accepted"],
                                q["draft_offered"], q["checkpoints"], int(q["cancelled"])])
        sys.stderr.write("analyze.py: wrote %s (%d rows)\n" % (path, stats["requests"]))

    if args.json:
        json.dump({"logs": [os.path.abspath(p) for p in args.logs], "lines": stats["lines"],
                   "restarts": [summarize(r) for r in restarts],
                   "aggregate": summarize(merge(restarts))}, sys.stdout, indent=1)
        sys.stdout.write("\n")
        return 0

    report(restarts, stats, args.logs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
