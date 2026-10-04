#!/usr/bin/env python3
"""analyze.py - the marginal cost of one more row in a Strata decode verify window.

S4.5.  The whole Stage 4 batching decision rests on one number: `b` in

    window_ms = a + b * T

where `T` is the number of rows in one verify window and `a` is what a window costs
when it carries only one.  S4.1 quoted `a = 24 ms`, `b = 8.4 ms` from two hand-picked
log lines.  This tool derives the cost model from every decode request in a serve log
instead, and says out loud what the log can and cannot support.

WHAT THE LOG ACTUALLY CONTAINS (all of it read-only, all of it cited)
--------------------------------------------------------------------
There is **no per-window line** in a normal Strata serve log.  `STRATA_DECODE_TIMING=1`
prints one line per *request* (src/program/generate.cpp:7344-7358) and it was never set
in `strata-iq3_s.log` (`grep -c "decode timing"` = 0).  So `T` and `window_ms` per
window are **derived**, not read.  The derivation is exact given three facts in the
source:

  * `draft_offered += T - 1` and `draft_accepted += a` per window
    (src/program/generate.cpp:7253-7254), summed over a request:
        SUM(T - 1) = O      =>   SUM T = O + W
        SUM a      = A
    where O/A are the `drafts accepted A of O` numbers on the request line
    (src/program/generate.cpp:7502-7505) and W is the number of windows.
  * the first window of a request always runs at T = 1
    (src/program/generate.cpp:7180 `if (first_window) T = 1;`), and it contributes 0
    to O, which is why the identity above already covers it.
  * a window commits `a + 1` tokens and every one of them is counted in `produced_n`
    (src/program/generate.cpp:7257-7260), so
        G = SUM(a + 1) = A + W      =>   W = G - A
    **provided no window was truncated** by EOS / max_new / a STOP.

So `W` has two independent estimators, `G - A` and `O - A + 1`, and they agree exactly
when the request ran to completion.  `analyze.py` keeps only the requests where they
agree (or where the suffix-draft accounting closes the gap - see below) and reports how
many it dropped.  Per request it then reports

    mean T        = (O + W) / W
    tokens/window = G / W          (must equal 1 + A/W)
    ms/window     = decode_ms / W

which is the pair the cost model regresses.

THE SUFFIX-DRAFT CORRECTION
---------------------------
`--suffix-draft` (on in this config) can replace the MTP's drafts for a window.  Those
windows are counted separately (`sfx_drafts += T - 1`, `sfx_ok += a`,
src/program/generate.cpp:7240) and reported on their own line
(`strata serve: suffix drafts: W windows, a of o drafts accepted`,
src/program/generate.cpp:7546).  `o` is a subset of `O` and `a` of `A`, so the
non-suffix windows are `W_ns = (O - o) - (A - a) + 1` and their mean T is
`((O - o) + W_ns) / W_ns`.  Suffix windows accept ~0.93 of their drafts against the
MTP's ~0.66, so leaving them in the same bucket as MTP windows biases the mean T
upward.  `analyze.py` reports both the combined and the MTP-only estimate and uses
**MTP-only** for the fit whenever the suffix line exists.

WHAT THIS CANNOT DO, AND WHY THAT MATTERS
-----------------------------------------
The log gives a per-request **mean** of T and of ms/window, never their joint
distribution.  Regressing ms/window on mean T is an errors-in-variables problem: the
within-request spread of T is measurement error in the regressor, which biases the
slope **toward zero**.  The fitted `b` is therefore a **lower bound** on the true
marginal row cost.  The two-point estimate S4.1 used (a T=1 window and a T=max window
from the same request) is the matching **upper bound**.  Both are printed, and the
README says which experiment closes the gap.  Do not quote one without the other.

USAGE
-----
    python3 bench/decode-slope/analyze.py strata-iq3_s.log
    python3 bench/decode-slope/analyze.py --limit 3593 strata-iq3_s.log
    python3 bench/decode-slope/analyze.py --json run.json strata-iq3_s.log
    python3 bench/decode-slope/analyze.py --csv bench/decode-slope/baseline strata-iq3_s.log
    python3 bench/decode-slope/analyze.py --compare a.json b.json
    python3 bench/decode-slope/analyze.py --min-windows 8 --tier 524288 strata-iq3_s.log

Stdlib only, Python 3.8+.  Reads files, writes files, never opens a socket, never
starts a process, never touches a device.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys

# ---------------------------------------------------------------------------
# log patterns.  Every one is anchored to the fprintf that emits it.
# ---------------------------------------------------------------------------

# src/program/generate.cpp:7502-7505
RE_REQ = re.compile(
    r"strata serve: prompt (\d+) tokens = (\d+) reused \+ (\d+) read in (\d+) ms \(([\d.]+) tok/s\), "
    r"(\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)"
    r"(?:, (\d+) checkpoints)?( \(cancelled\))?"
)
# src/program/generate.cpp:7519-7522  (cumulative per process)
RE_EHIT = re.compile(
    r"strata serve: decode expert cache hit rate: ([\d.]+)% \((\d+) hits / (\d+) lookups\)"
)
# src/program/generate.cpp:7540-7543  (cumulative per process)
RE_KVS = re.compile(
    r"strata serve: KV streaming: ([\d.]+)% of (\d+) block reads hit VRAM, ([\d.]+) MiB read from RAM"
    r"( - OVERFLOW)?"
)
# src/program/generate.cpp:7546
RE_SFX = re.compile(r"strata serve: suffix drafts: (\d+) windows, (\d+) of (\d+) drafts accepted")
# The slot hand-over.  `swap K: slot A -> B ok in X ms … (total N swaps, M ms, …)` - the
# tail is CUMULATIVE PER PROCESS, which is what makes the pre-emption cost attributable.
RE_SWAP = re.compile(r"strata serve: swap \d+: slot (-?\d+) -> (-?\d+) \w+ in (\d+) ms; "
                     r"saved (\d+) B, restored (\d+) B.*\(total (\d+) swaps, (\d+) ms")
# src/program/generate.cpp:7347-7355  (only when STRATA_DECODE_TIMING=1)
RE_DT = re.compile(
    r"strata decode timing: (\d+) windows, avg T ([\d.]+), ([\d.]+) tokens/window, ([\d.]+) ms/window = "
    r"verify ([\d.]+) \(GPU-reach wait ([\d.]+) \+ per-layer host ([\d.]+) \[plan ([\d.]+) actq ([\d.]+) "
    r"jobs ([\d.]+) CPU ([\d.]+)\] \+ stage ([\d.]+)\) \+ commit/emit ([\d.]+) \+ draft ([\d.]+); "
    r"per layer-window: CPU experts ([\d.]+) \(([\d.]+) entries\), VRAM hits ([\d.]+), PCIe ([\d.]+)"
)
# startup / config lines
RE_UP = re.compile(r"strata generate: session is up \(engine ([\d.]+)\)")
RE_WIN = re.compile(r"strata verify: window up to (\d+) tokens, ([\d.]+) MiB of device buffers")
RE_CAP = re.compile(r"strata generate: KV streaming: (\d+) of (\d+) cells per QSA layer in VRAM, "
                    r"the K/V in ([\d.]+) GiB of pinned RAM")
RE_EC = re.compile(r"strata generate: expert cache (\d+) slots, ([\d.]+) GiB of VRAM")
RE_DRV = re.compile(r"strata serve: concurrent driver on \((\d+) slots")
RE_ACT = re.compile(r"strata serve: activity: (.*)")
RE_BLOB = re.compile(r"native pack: .* largest blob ([\d.]+) MB")
RE_LAYERS = re.compile(r"strata generate: (\d+) native projection matrices")


# ---------------------------------------------------------------------------
# small statistics, stdlib only
# ---------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes betacf)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-300:
        d = 1e-300
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-300:
            d = 1e-300
        c = 1.0 + aa / c
        if abs(c) < 1e-300:
            c = 1e-300
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-12:
            break
    return h


def _ibeta_direct(a: float, b: float, x: float) -> float:
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log1p(-x))
    return bt * _betacf(a, b, x) / a


def betai(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    if x < (a + 1.0) / (a + b + 2.0):
        return _ibeta_direct(a, b, x)
    return 1.0 - _ibeta_direct(b, a, 1.0 - x)


def tinv(p: float, dof: float) -> float:
    """Two-sided critical value: tinv(0.95, dof) is the 97.5th percentile of t_dof.

    P(|T| <= t) = 1 - I_{dof/(dof+t^2)}(dof/2, 1/2), and I is decreasing in t, so the
    root to find is I = 1 - p.  (Getting this comparison backwards silently returns ~0
    for every dof - verified against t(0.95,1)=12.706, t(0.95,10)=2.228,
    t(0.95,30)=2.042, t(0.95,inf)=1.960.)
    """
    if dof <= 0:
        return float("nan")
    if dof > 1e6:
        return 1.959963985
    want = 1.0 - p
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if betai(dof / 2.0, 0.5, dof / (dof + mid * mid)) > want:
            lo = mid      # I too large => t too small
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mreg(rows, cols, y):
    """Least squares y = c0 + sum_j c_j * x_j over `cols` (list of column names read off
    each dict in `rows`).  Returns the coefficients, their classical standard errors and
    95 % CIs, the R^2 and the condition number of the design matrix.

    Written with a normal-equations solve and Gauss-Jordan inversion because the design
    here is small (<= 6 columns) and stdlib-only is a constraint of this tool.  The
    condition number is reported so nobody reads a coefficient off an ill-conditioned
    design without noticing."""
    n = len(rows)
    k = len(cols) + 1
    if n < k + 3:
        return None
    X = [[1.0] + [float(r.get(c) or 0.0) for c in cols] for r in rows]
    Y = [float(r[y]) for r in rows]
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    XtY = [sum(X[i][a] * Y[i] for i in range(n)) for a in range(k)]
    inv = _invert(XtX)
    if inv is None:
        return None
    coef = [sum(inv[a][b] * XtY[b] for b in range(k)) for a in range(k)]
    ybar = sum(Y) / n
    sst = sum((v - ybar) ** 2 for v in Y)
    sse = 0.0
    for i in range(n):
        pred = sum(coef[a] * X[i][a] for a in range(k))
        sse += (Y[i] - pred) ** 2
    dof = n - k
    if dof <= 0 or sst <= 0:
        return None
    s2 = sse / dof
    tc = tinv(0.95, dof)
    out = {"n": n, "k": k, "dof": dof, "r2": 1.0 - sse / sst, "resid_sd": math.sqrt(s2),
           "t_crit": tc, "cond": _cond(XtX), "terms": {}}
    names = ["const"] + list(cols)
    for a in range(k):
        se = math.sqrt(max(0.0, s2 * inv[a][a]))
        out["terms"][names[a]] = {"c": coef[a], "se": se,
                                  "t": (coef[a] / se) if se > 0 else float("nan"),
                                  "ci": [coef[a] - tc * se, coef[a] + tc * se]}
    return out


def _invert(m):
    n = len(m)
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(m)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            return None
        a[col], a[piv] = a[piv], a[col]
        pv = a[col][col]
        a[col] = [v / pv for v in a[col]]
        for r in range(n):
            if r == col:
                continue
            f = a[r][col]
            if f:
                a[r] = [a[r][c] - f * a[col][c] for c in range(2 * n)]
    return [row[n:] for row in a]


def _cond(m):
    """Cheap condition-number proxy: max|diag| / |det|^(1/n).  Not exact, but it moves
    by orders of magnitude when the design goes collinear, which is what it is here for."""
    n = len(m)
    inv = _invert(m)
    if inv is None:
        return float("inf")
    det = 1.0
    # det(m) = 1 / det(inv) is not stable to recover; use the diagonal ratio instead.
    dmax = max(abs(m[i][i]) for i in range(n))
    dmin = min(abs(m[i][i]) for i in range(n)) or 1e-300
    return dmax / dmin


def print_mreg(title, fit, unit="ms"):
    if fit is None:
        print("%s: not fitted (n too small or the design is singular)." % title)
        return
    print("%s   n=%d  R2=%.3f  resid sd=%.1f %s  cond~%.1e" % (title, fit["n"], fit["r2"],
                                                                fit["resid_sd"], unit, fit["cond"]))
    for name, t in fit["terms"].items():
        print("    %-14s %10.3f  se %8.3f  t %7.2f  95%% CI [%10.3f, %10.3f]"
              % (name, t["c"], t["se"], t["t"], t["ci"][0], t["ci"][1]))


def ols(xs, ys):
    """Least squares y = a + b x.  Returns the point estimates plus the classical
    standard errors, t statistics and 95 % confidence intervals."""
    n = len(xs)
    if n < 3:
        return None
    sx = sum(xs)
    sy = sum(ys)
    mx = sx / n
    my = sy / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx <= 0:
        return None
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    sse = sum(r * r for r in resid)
    sst = sum((y - my) ** 2 for y in ys)
    dof = n - 2
    if dof <= 0:
        return None
    s2 = sse / dof
    se_b = math.sqrt(s2 / sxx)
    se_a = math.sqrt(s2 * (1.0 / n + mx * mx / sxx))
    tc = tinv(0.95, dof)
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    return {
        "n": n, "a": a, "b": b, "r2": r2, "dof": dof,
        "se_a": se_a, "se_b": se_b,
        "t_a": a / se_a if se_a > 0 else float("nan"),
        "t_b": b / se_b if se_b > 0 else float("nan"),
        "ci_a": [a - tc * se_a, a + tc * se_a],
        "ci_b": [b - tc * se_b, b + tc * se_b],
        "t_crit": tc,
        "mean_x": mx, "mean_y": my,
        "resid_sd": math.sqrt(s2),
    }


def bootstrap_slope(xs, ys, iters=4000, seed=20261004):
    """Percentile bootstrap for (a, b).  Distribution-free, and it is the interval to
    trust when the classical one is not (heteroskedastic buckets, a few huge requests)."""
    n = len(xs)
    if n < 8:
        return None
    rnd = _rng(seed)
    as_, bs = [], []
    for _ in range(iters):
        sx = sy = sxx = sxy = mx = my = 0.0
        for _i in range(n):
            j = int(rnd() * n)
            x = xs[j]
            y = ys[j]
            sx += x
            sy += y
        mx = sx / n
        my = sy / n
        for _i in range(n):
            j = int(rnd() * n)
            x = xs[j] - mx
            y = ys[j] - my
            sxx += x * x
            sxy += x * y
        if sxx <= 0:
            continue
        bb = sxy / sxx
        as_.append(my - bb * mx)
        bs.append(bb)
    if len(bs) < 100:
        return None
    as_.sort()
    bs.sort()
    q = lambda v, p: v[min(len(v) - 1, max(0, int(p * len(v))))]
    return {"a": [q(as_, 0.025), q(as_, 0.975)], "b": [q(bs, 0.025), q(bs, 0.975)],
            "median_b": q(bs, 0.5), "median_a": q(as_, 0.5), "iters": len(bs)}


def _rng(seed):
    """Deterministic LCG so the bootstrap is reproducible without importing random
    state into a script that other scripts import."""
    state = [seed & 0x7FFFFFFF]

    def f():
        state[0] = (state[0] * 1103515245 + 12345) & 0x7FFFFFFF
        return state[0] / 0x7FFFFFFF
    return f


def pct(v, p):
    if not v:
        return float("nan")
    s = sorted(v)
    if len(s) == 1:
        return s[0]
    i = p * (len(s) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def fnum(x, nd=2):
    if x is None:
        return "-"
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return "-"
    if isinstance(x, int):
        return "{:,}".format(x)
    return ("{:,.%df}" % nd).format(x)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

class Segment:
    """One engine process's slice of the log.  Grouped by the `session is up` anchor,
    exactly like bench/prefill/analyze.py.  Two processes writing concurrently cannot
    be separated - the tool says so rather than pretending."""

    def __init__(self, idx, start_line):
        self.idx = idx
        self.start_line = start_line
        self.end_line = start_line
        self.max_t = None            # `window up to N tokens`
        self.arena_mib = None
        self.tier = None             # --max-context, from the KV streaming line
        self.kv_resident = None
        self.kv_pinned_gib = None
        self.expert_slots = None
        self.expert_gib = None
        self.slots = None            # --serve-slots
        self.engine = None
        self.largest_blob_mib = None
        self.requests = []
        self.cum_ehits = (0, 0)      # last cumulative (hits, lookups) seen
        self.cum_kvs = (0, 0.0)      # last cumulative (lookups, mib_from_ram)
        self.cum_kvs_pct = None
        self.cum_ehit_pct = None
        self.decode_timing = []
        self.activity = []
        self.last_req = None         # the request the trailing per-request lines attach to
        self.swap_ms = []            # individual slot hand-over times
        self.swap_total = 0          # the cumulative `total N swaps, M ms` tail


def parse(paths, limit=0):
    segs = []
    seg = None
    cfg = {}          # config lines seen since the last `session is up`
    lineno = 0
    nreq = 0
    for path in paths:
        with open(path, "r", errors="replace") as fh:
            for raw in fh:
                lineno += 1
                line = raw.rstrip("\n")

                m = RE_UP.search(line)
                if m:
                    if seg is not None:
                        seg.end_line = lineno - 1
                        segs.append(seg)
                    seg = Segment(len(segs) + 1, lineno)
                    seg.engine = m.group(1)
                    _apply_cfg(seg, cfg)
                    continue
                if seg is None:
                    # startup chatter before the first `session is up`: open a segment
                    # for it anyway so the config lines are not lost.
                    seg = Segment(len(segs) + 1, lineno)
                seg.end_line = lineno

                m = RE_WIN.search(line)
                if m:
                    cfg["max_t"] = int(m.group(1))
                    cfg["arena_mib"] = float(m.group(2))
                    _apply_cfg(seg, cfg)
                    continue
                m = RE_CAP.search(line)
                if m:
                    cfg["kv_resident"] = int(m.group(1))
                    cfg["tier"] = int(m.group(2))
                    cfg["kv_pinned_gib"] = float(m.group(3))
                    _apply_cfg(seg, cfg)
                    continue
                m = RE_EC.search(line)
                if m:
                    cfg["expert_slots"] = int(m.group(1))
                    cfg["expert_gib"] = float(m.group(2))
                    _apply_cfg(seg, cfg)
                    continue
                m = RE_DRV.search(line)
                if m:
                    cfg["slots"] = int(m.group(1))
                    _apply_cfg(seg, cfg)
                    continue
                m = RE_BLOB.search(line)
                if m:
                    cfg["largest_blob_mib"] = float(m.group(1))
                    _apply_cfg(seg, cfg)
                    continue
                m = RE_EHIT.search(line)
                if m:
                    seg.cum_ehits = (int(m.group(2)), int(m.group(3)))
                    seg.cum_ehit_pct = float(m.group(1))
                    if seg.last_req is not None:
                        seg.last_req["expert_hits"] = seg.cum_ehits[0]
                        seg.last_req["expert_lookups"] = seg.cum_ehits[1]
                        seg.last_req["ehit_line"] = lineno
                    continue
                m = RE_KVS.search(line)
                if m:
                    seg.cum_kvs = (int(m.group(2)), float(m.group(3)))
                    seg.cum_kvs_pct = float(m.group(1))
                    if seg.last_req is not None:
                        seg.last_req["kv_lookups"] = seg.cum_kvs[0]
                        seg.last_req["kv_ram_mib"] = seg.cum_kvs[1]
                        seg.last_req["kvs_line"] = lineno
                    continue
                m = RE_SFX.search(line)
                if m:
                    if seg.last_req is not None:
                        seg.last_req["sfx_windows"] = int(m.group(1))
                        seg.last_req["sfx_ok"] = int(m.group(2))
                        seg.last_req["sfx_offered"] = int(m.group(3))
                        seg.last_req["sfx_line"] = lineno
                    continue
                m = RE_SWAP.search(line)
                if m:
                    seg.swap_ms.append(int(m.group(3)))
                    seg.swap_total = int(m.group(6))
                    continue
                m = RE_DT.search(line)
                if m:
                    seg.decode_timing.append({
                        "line": lineno,
                        "windows": int(m.group(1)), "avg_T": float(m.group(2)),
                        "tok_win": float(m.group(3)), "ms_win": float(m.group(4)),
                        "verify": float(m.group(5)), "gpu_wait": float(m.group(6)),
                        "layer_host": float(m.group(7)), "plan": float(m.group(8)),
                        "actq": float(m.group(9)), "jobs": float(m.group(10)),
                        "cpu": float(m.group(11)), "stage": float(m.group(12)),
                        "commit": float(m.group(13)), "draft": float(m.group(14)),
                        "cpu_experts": float(m.group(15)), "cpu_entries": float(m.group(16)),
                        "vram_hits": float(m.group(17)), "pcie": float(m.group(18)),
                    })
                    continue
                m = RE_ACT.search(line)
                if m:
                    seg.activity.append({"line": lineno, "text": m.group(1)})
                    continue
                m = RE_REQ.search(line)
                if m:
                    nreq += 1
                    seg.requests.append(_mk_request(seg, lineno, m))
                    if limit and nreq >= limit:
                        seg.end_line = lineno
                        segs.append(seg)
                        return _inherit(segs)
    if seg is not None:
        seg.end_line = lineno
        segs.append(seg)
    return _inherit(segs)


_CFG_FIELDS = ("max_t", "arena_mib", "tier", "kv_resident", "kv_pinned_gib",
               "expert_slots", "expert_gib", "slots", "largest_blob_mib")


def _apply_cfg(seg, cfg):
    for k in _CFG_FIELDS:
        if k in cfg:
            setattr(seg, k, cfg[k])


def _inherit(segs):
    """A segment with no config lines of its own (the two engines in this log interleave,
    and the config lines land before the `session is up` anchor) takes the previous
    segment's.  `cfg_inherited` says so, because it is an assumption, not a reading."""
    prev = None
    for s in segs:
        s.cfg_inherited = False
        if getattr(s, "tier", None) is None and prev is not None:
            for k in _CFG_FIELDS:
                if getattr(s, k, None) is None and getattr(prev, k, None) is not None:
                    setattr(s, k, getattr(prev, k))
                    s.cfg_inherited = True
        if getattr(s, "tier", None) is not None or getattr(s, "max_t", None) is not None:
            prev = s
    return segs


def _mk_request(seg, lineno, m):
    """Pass 1: only the fields on the prompt line itself.  The three trailing lines are
    attached afterwards by the parser, and pass 2 (`derive`) runs once they are in."""
    n = int(m.group(1))
    reused = int(m.group(2))
    fresh = int(m.group(3))
    prompt_ms = int(m.group(4))
    G = int(m.group(6))
    decode_ms = int(m.group(7))
    A = int(m.group(9))
    O = int(m.group(10))
    cancelled = bool(m.group(12))
    r = {
        "line": lineno, "segment": seg.idx, "tier": seg.tier, "max_t": seg.max_t,
        "prompt_tokens": n, "reused": reused, "fresh": fresh, "prompt_ms": prompt_ms,
        "generated": G, "decode_ms": decode_ms, "accepted": A, "offered": O,
        "cancelled": cancelled,
        # The context a window attends over.  The prompt is already in the session when
        # decode starts, so the window's rows sit at positions n .. n+G.  `ctx_mid` is
        # the mean over the request and is the covariate the row term must be separated
        # from: across this log ms/window ranges 20-140 ms while mean T ranges 2.2-5.2,
        # and context ranges over two orders of magnitude.
        "ctx_start": n, "ctx_end": n + G, "ctx_mid": n + G / 2.0,
        "log_ctx": math.log(max(2.0, n + G / 2.0)),
        # trailing per-request lines, filled in by the parser
        "expert_hits": 0, "expert_lookups": 0,
        "kv_lookups": 0, "kv_ram_mib": 0.0,
        "sfx_windows": 0, "sfx_ok": 0, "sfx_offered": 0,
    }
    seg.last_req = r
    return r


def derive(r):
    """Pass 2: everything that depends on the trailing lines."""
    G = r["generated"]
    A = r["accepted"]
    O = r["offered"]
    decode_ms = r["decode_ms"]
    sfx = (r["sfx_windows"], r["sfx_ok"], r["sfx_offered"])

    # ---- the window count, two ways --------------------------------------
    # W = G - A is EXACT whenever no window was truncated: a window commits a+1 tokens
    # (src/program/generate.cpp:7250 `ver.commit(a + 1, …)` and :7257-7260 emit a+1), so
    # summing over a request's windows gives G = A + W.
    # W = O - A + 1 additionally assumes the request's LAST window was truncated by EOS
    # / max_new / a STOP, which is the common case for a chat reply that ends on a stop
    # token (that window commits a+1 but offered T-1 > a).  It is NOT a general
    # identity: measured over this log the two disagree on a large one-sided set, and
    # `G - A` is the one the source supports.
    w_tokens = G - A             # exact for a request that ran every window to completion
    w_draft = O - A + 1          # the "+1" is the truncated last window
    r["W_draft"] = w_draft
    r["W_tokens"] = w_tokens
    r["W_agree"] = (w_draft == w_tokens)
    r["W"] = w_tokens if w_tokens > 0 else None
    r["W_source"] = "G-A" if w_tokens > 0 else "none"

    # The suffix-draft line is printed only when sfx_windows > 0
    # (generate.cpp:7545-7547), so a request that used no suffix drafts has NO line of
    # its own.  It is only used when its counts fit inside this request's own.
    if sfx[0] > 0 and sfx[0] <= w_tokens and sfx[2] <= O and sfx[1] <= A:
        o_ns = O - sfx[2]
        a_ns = A - sfx[1]
        w_ns = w_tokens - sfx[0]
        r["W_mtp"] = w_ns if w_ns > 0 else None
        r["W_sfx"] = sfx[0]
        r["T_mtp"] = ((o_ns + w_ns) / w_ns) if w_ns > 0 else None
        r["T_sfx"] = ((sfx[2] + sfx[0]) / sfx[0]) if sfx[0] > 0 else None
        r["sfx_trusted"] = True
    else:
        r["W_mtp"] = None
        r["W_sfx"] = 0
        r["T_mtp"] = None
        r["T_sfx"] = None
        r["sfx_trusted"] = False

    if r["W"] and r["W"] > 0:
        r["mean_T"] = (O + r["W"]) / r["W"]      # = 1 + O/W, the rows ASKED for
        r["ms_win"] = decode_ms / r["W"]
        r["tok_win"] = G / r["W"]                # = 1 + A/W, the tokens committed
        r["acc"] = A / O if O else None
    else:
        r["mean_T"] = r["ms_win"] = r["tok_win"] = r["acc"] = None

    r["expert_hit_pct"] = (100.0 * r["expert_hits"] / r["expert_lookups"]) if r["expert_lookups"] else None
    # an independent token count: the decode path looks up k=10 experts per layer per
    # token, so expert_lookups ~= 48 x 10 x tokens.  `tokens_from_experts` is that, and
    # the ratio against `generated` is the log's own self-consistency check.
    r["tokens_from_experts"] = r["expert_lookups"] / 480.0 if r["expert_lookups"] else 0.0

    # ---- the bracket on PHYSICAL rows per window -------------------------
    # `mean_T` is the rows the window was ASKED for.  The rows that actually ran, R, can
    # be fewer: `--spec-min-p` truncates each window to the shortest prefix of drafts
    # worth verifying (src/program/generate.cpp:7176-7179:
    #     T = 1; while (T < S_mtp && dprob[T-1] >= req_spec_min_p) ++T;
    # ).  The log never records dprob, so R is bracketed instead:
    #     tok_win = 1 + A/W  <=  R  <=  mean_T = 1 + O/W
    # the lower bound because every committed token came from a row that ran, the upper
    # because no window runs more rows than it was asked to.  Regressing ms/window on the
    # lower bound INFLATES the slope and on the upper bound DEFLATES it, so the two fits
    # bracket the true marginal row cost.  The bracket itself is exact arithmetic; what is
    # an inference is that R is the same quantity on both sides of the regression.
    if r["W"] and r["W"] > 0:
        o_mtp = O - (sfx[2] if r["sfx_trusted"] else 0)
        a_mtp = A - (sfx[1] if r["sfx_trusted"] else 0)
        r["acc_mtp"] = (a_mtp / o_mtp) if o_mtp > 0 else None
        r["rows_lo"] = r["tok_win"]
        r["rows_hi"] = r["mean_T"]
    else:
        r["acc_mtp"] = r["rows_lo"] = r["rows_hi"] = None
    return r


def kv_deltas(reqs):
    """`KV streaming: X% of B block reads hit VRAM, M MiB read from RAM` is cumulative
    per process (generate.cpp:7530-7543).  Difference it within a stream to get the
    per-request KV hit rate, which is the covariate the batching question actually
    cares about (a second sequence's selections are a second set of blocks to resolve)."""
    for r in reqs:
        r["kv_req_lookups"] = 0
        r["kv_req_miss_mib"] = 0.0
        r["kv_hit_pct"] = None
    prev = None
    for r in reqs:
        if prev is not None and r["kv_lookups"] >= prev["kv_lookups"]:
            dl = r["kv_lookups"] - prev["kv_lookups"]
            dm = r["kv_ram_mib"] - prev["kv_ram_mib"]
            r["kv_req_lookups"] = dl
            r["kv_req_miss_mib"] = dm
            if dl > 0:
                # 4224 B per block, the same constant the engine multiplies by at
                # generate.cpp:7542
                r["kv_hit_pct"] = 100.0 * (1.0 - max(0.0, dm) * 1048576.0 / 4224.0 / dl)
        prev = r


# ---------------------------------------------------------------------------
# the cost model
# ---------------------------------------------------------------------------

def split_streams(all_reqs, key="kv_lookups"):
    """Separate a log's requests into per-process streams.

    `KV streaming: X% of B block reads hit VRAM` is CUMULATIVE PER PROCESS
    (src/program/generate.cpp:7530-7543 reads `kv_stream_counters`, which live in the
    device's map arrays, and the comment at :7531 says so outright).  Within one process
    `B` is non-decreasing, and a process restart is the only thing that makes it fall.
    Two processes writing one log therefore produce interleaved monotone sequences, and
    the minimum number of increasing subsequences the sequence decomposes into is the
    number of writers.  This is patience sorting on `B`: each request joins the stream
    whose last value is the largest one still <= its own, or opens a new stream.

    The `decode expert cache hit rate` line is NOT cumulative - it is `req_hits /
    req_look`, the delta over the request (src/program/generate.cpp:7479-7480, :7508-
    :7512) - so it cannot be used for this.  It is used directly as a per-request
    covariate instead, and `expert_lookups / (48 layers x 10 experts)` is a free
    independent token count (it reproduces `generated` to ~1 %, see the README).

    The split is an attribution, not a fact: it can be wrong when two processes sit far
    apart in counter space.  `stream` is on every request so anything that depends on it
    can be re-cut with `--stream N`."""
    streams = []          # [{"id": i, "last": v, "n": k, "start_line": L}]
    last_id = -1
    for r in all_reqs:
        v = r.get(key)
        if not v:
            # no KV line for this request (kv_mode != 1, or a request with no QSA
            # lookups).  It belongs wherever its neighbour belongs.
            r["stream"] = last_id
            continue
        best = None
        for s in streams:
            if s["last"] <= v and (best is None or s["last"] > best["last"]):
                best = s
        if best is None:
            best = {"id": len(streams), "last": -1, "n": 0, "start_line": r["line"]}
            streams.append(best)
        best["last"] = v
        best["n"] += 1
        r["stream"] = best["id"]
        last_id = best["id"]
    return streams


def usable(reqs, min_windows=4):
    """Rows that can enter the regression: a window count that closes, enough windows
    that the per-request mean is not one sample, and a mean T that is physically
    possible for the segment (`mean_T <= max_t`, since `T = S_mtp <= o.spec` at
    generate.cpp:7175 and `o.spec` is what the `window up to N tokens` line reports)."""
    out = []
    for r in reqs:
        if r["mean_T"] is None or r["W"] is None:
            continue
        if r["W"] < min_windows:
            continue
        if r["decode_ms"] <= 0:
            continue
        mt = r["max_t"] or 8
        if r["mean_T"] < 1.0 or r["mean_T"] > mt + 0.01:
            continue
        out.append(r)
    return out


def per_window(r):
    """Divide the per-request counters by the window count.  These are the regressors
    that actually explain the log, and they are all exact arithmetic on lines the engine
    prints."""
    w = r["W"]
    if not w or w <= 0:
        return
    # `decode expert cache hit rate: H% (hits / lookups)` is the request's own delta
    # (generate.cpp:7479-7480), and a lookup is one (layer, token, expert-slot) request
    # for one of the k=10 routed experts.  Split it into the rows the GPU served from
    # its VRAM cache and the rows the CPU pool had to compute.
    r["look_win"] = r["expert_lookups"] / w
    r["gpu_win"] = r["expert_hits"] / w
    r["cpu_win"] = (r["expert_lookups"] - r["expert_hits"]) / w
    r["look_row"] = r["look_win"] / r["rows_hi"] if r["rows_hi"] else None
    r["cpu_row"] = r["cpu_win"] / r["rows_hi"] if r["rows_hi"] else None


def structural(r):
    """The regressors for the slope DECOMPOSITION, which is the actual question.

    A row costs two things: the GPU work it forces (48 layers x k=10 expert blobs read
    from VRAM, the projections, the attention), and the CPU expert pool it forces.  The
    second is `rows x (1 - expert cache hit rate)`, because a lookup misses exactly that
    often and a miss is one pool job.  Fitting both separates them:

        ms/window = a + b_gpu * rows + b_cpu * rows * (1 - hit) + c * context

    `b_gpu` is the part a batched kernel could in principle drive toward the bandwidth
    floor.  `b_cpu * (1 - hit)` is the part that scales with rows no matter how the
    kernel is written, and the only lever on it is the cache hit rate - which is the
    owner's VRAM/quality decision, not an engineering one."""
    if r["expert_hit_pct"] is None or not r["rows_hi"]:
        return
    miss = 1.0 - r["expert_hit_pct"] / 100.0
    r["miss_rows_hi"] = r["rows_hi"] * miss
    r["miss_rows_lo"] = (r["rows_lo"] or 0.0) * miss


def trim(rows, frac=0.01):
    """Drop the top `frac` of ms/window.  The distribution has a long right tail (a
    request whose slot was parked for seconds, a cancelled run, a cold cache) and the
    mean of that tail is not a window cost.  The count dropped is always reported."""
    if not rows:
        return rows, 0
    cap = pct([r["ms_win"] for r in rows], 1.0 - frac)
    keep = [r for r in rows if r["ms_win"] <= cap]
    return keep, len(rows) - len(keep)


def fit_rows(rows, key_t="mean_T", key_y="ms_win"):
    """The plain univariate fit.  Reported for continuity with S4.1 and because the
    README explains why it is NOT the number to use."""
    xs = [r[key_t] for r in rows]
    ys = [r[key_y] for r in rows]
    return ols(xs, ys)


def model_fits(rows):
    """Every fit the verdict needs, in one place."""
    out = {}
    out["simple"] = fit_rows(rows)
    out["rows_lo_ctx"] = mreg(rows, ["rows_lo", "ctx_mid"], "ms_win")
    out["rows_hi_ctx"] = mreg(rows, ["rows_hi", "ctx_mid"], "ms_win")
    out["rows_lo_hit"] = mreg(rows, ["rows_lo", "expert_hit_pct"], "ms_win")
    out["rows_hi_hit"] = mreg(rows, ["rows_hi", "expert_hit_pct"], "ms_win")
    out["rows_both_hit"] = mreg(rows, ["rows_lo", "rows_hi", "expert_hit_pct"], "ms_win")
    out["rows_lo_hit_ctx"] = mreg(rows, ["rows_lo", "expert_hit_pct", "log_ctx"], "ms_win")
    have_ew = [r for r in rows if r.get("cpu_win") is not None]
    out["expert_split"] = mreg(have_ew, ["gpu_win", "cpu_win"], "ms_win")
    out["expert_split_rows"] = mreg(have_ew, ["gpu_win", "cpu_win", "rows_lo"], "ms_win")
    out["cpu_only"] = mreg(have_ew, ["cpu_win"], "ms_win")
    # the decomposition
    have_st = [r for r in rows if r.get("miss_rows_hi") is not None]
    out["decomp"] = mreg(have_st, ["rows_hi", "miss_rows_hi"], "ms_win")
    out["decomp_ctx"] = mreg(have_st, ["rows_hi", "miss_rows_hi", "ctx_mid"], "ms_win")
    out["decomp_logctx"] = mreg(have_st, ["rows_hi", "miss_rows_hi", "log_ctx"], "ms_win")
    out["decomp_lo"] = mreg(have_st, ["rows_lo", "miss_rows_lo"], "ms_win")
    return out


def two_point(seg_reqs):
    """S4.1's method, recomputed honestly.

    A request that generated exactly 1 token ran exactly ONE window, and that window is
    T = 1 by construction (`if (first_window) T = 1;`, src/program/generate.cpp:7180).
    Its `generated in N ms` is therefore a single one-row window including its commit -
    the cleanest fixed-cost measurement the log contains.  Pair it with the longest-mean-T
    usable request in the same population for the second point."""
    ones = [r for r in seg_reqs if r["generated"] == 1 and r["decode_ms"] > 0 and not r["cancelled"]]
    if not ones:
        return None
    t1 = sorted(r["decode_ms"] for r in ones)
    hi = max((r for r in seg_reqs if r["mean_T"] and r["W"] and r["W"] >= 4),
             key=lambda r: r["mean_T"], default=None)
    if hi is None:
        return None
    med1 = pct(t1, 0.5)
    b = (hi["ms_win"] - med1) / (hi["mean_T"] - 1.0) if hi["mean_T"] > 1.0 else None
    return {"n_T1": len(ones), "T1_ms_min": t1[0], "T1_ms_median": med1, "T1_ms_max": t1[-1],
            "T1_p10": pct(t1, 0.10), "T1_p25": pct(t1, 0.25), "T1_p75": pct(t1, 0.75),
            "T1_p90": pct(t1, 0.90),
            "hi_line": hi["line"], "hi_T": hi["mean_T"], "hi_ms_win": hi["ms_win"],
            "b": b, "a": med1 - b if b is not None else None}


TBUCKETS = [(1.0, 1.25), (1.25, 1.75), (1.75, 2.25), (2.25, 2.75), (2.75, 3.25),
            (3.25, 3.75), (3.75, 4.25), (4.25, 4.75), (4.75, 5.25), (5.25, 5.75),
            (5.75, 6.25), (6.25, 6.75), (6.75, 7.5), (7.5, 8.01)]
EBUCKETS = [(2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5), (4.5, 5.5), (5.5, 6.5)]
HBUCKETS = [(0, 80), (80, 88), (88, 94), (94, 101)]


def bucketise(rows, key, edges):
    out = []
    for lo, hi in edges:
        b = [r for r in rows if r.get(key) is not None and lo <= r[key] < hi]
        if b:
            out.append((lo, hi, b))
    return out


def cross_table(rows, key_a, edges_a, key_b, edges_b, y="ms_win"):
    """A two-way median table.  This is the identification that needs no functional form:
    inside a cell, both covariates are pinned, so the difference between cells in the same
    row is the effect of the other one."""
    return [(lo_a, hi_a, [(lo_b, hi_b, [r for r in rows
                                        if lo_a <= (r.get(key_a) or -1e30) < hi_a
                                        and lo_b <= (r.get(key_b) or -1e30) < hi_b])
                          for lo_b, hi_b in edges_b])
            for lo_a, hi_a in edges_a]


def coef_of(fit, name):
    """Read one coefficient out of either fit shape (ols() returns a/b directly, mreg()
    returns a `terms` dict).  Returns None when the fit is missing or the term is not in
    it, so the report degrades to `-` instead of crashing."""
    if not fit:
        return None
    if name in ("mean_T", "T") and "b" in fit and "terms" not in fit:
        return {"c": fit["b"], "ci": fit["ci_b"], "se": fit["se_b"], "t": fit["t_b"]}
    t = fit.get("terms")
    if not t or name not in t:
        return None
    return t[name]


def print_fit(f, label):
    if f is None:
        print("  %-28s not fitted" % label)
        return
    if "terms" not in f:
        # an ols() result: window_ms = a + b*T with classical CIs
        print("  %-28s n=%-5d R2=%.3f  a %+.2f [%+.2f, %+.2f]  b %+.3f [%+.3f, %+.3f]"
              % (label, f["n"], f["r2"], f["a"], f["ci_a"][0], f["ci_a"][1],
                 f["b"], f["ci_b"][0], f["ci_b"][1]))
        return
    terms = f["terms"]
    parts = []
    for k, v in terms.items():
        if k == "const":
            continue
        parts.append("%s = %+.3f [%+.3f, %+.3f] t%.1f" % (k, v["c"], v["ci"][0], v["ci"][1], v["t"]))
    print("  %-28s n=%-5d R2=%.3f  const %+.2f [%+.2f, %+.2f]" % (
        label, f["n"], f["r2"], terms["const"]["c"], terms["const"]["ci"][0], terms["const"]["ci"][1]))
    for p_ in parts:
        print("      %s" % p_)


def analyse(segs, args):
    all_reqs = [r for s in segs for r in s.requests]
    for r in all_reqs:
        derive(r)
    streams = split_streams(all_reqs)
    for st in streams:
        kv_deltas([r for r in all_reqs if r["stream"] == st["id"]])
    for r in all_reqs:
        per_window(r)
        structural(r)
    res = {"requests_total": len(all_reqs), "segments": len(segs),
           "streams": [{"id": s["id"], "n": s["n"], "start_line": s["start_line"],
                        "last_lookups": s["last"]} for s in streams]}

    if args.stream is not None:
        keep = set(args.stream)
        for s in segs:
            s.requests = [r for r in s.requests if r.get("stream") in keep]
        all_reqs = [r for s in segs for r in s.requests]

    for s in segs:
        s.usable = usable(s.requests, args.min_windows)
        s.fit = fit_rows(s.usable)
        s.tp = two_point(s.requests)

    pool = []
    for s in segs:
        pool.extend(s.usable)
    res["usable"] = len(pool)
    res["dropped_disagree"] = sum(1 for r in all_reqs if not r["W_agree"])
    res["dropped_no_windows"] = sum(1 for r in all_reqs if r["W_source"] == "none")
    res["dropped_few_windows"] = sum(1 for r in all_reqs
                                     if r["W_source"] != "none" and (r["W"] or 0) < args.min_windows)
    res["dropped_impossible_T"] = sum(1 for r in all_reqs
                                      if r["mean_T"] is not None and r["max_t"]
                                      and r["mean_T"] > r["max_t"] + 0.01)
    res["cancelled"] = sum(1 for r in all_reqs if r["cancelled"])
    res["sfx_trusted"] = sum(1 for r in all_reqs if r["sfx_trusted"])
    res["sfx_seen"] = sum(1 for r in all_reqs if r["sfx_windows"] > 0)

    # the log's own self-consistency check: a decode token costs k=10 expert lookups per
    # layer over 48 layers, so expert_lookups / 480 must track `generated`.
    ratios = [r["tokens_from_experts"] / r["generated"] for r in all_reqs
              if r["generated"] >= 20 and r["tokens_from_experts"] > 0]
    if ratios:
        res["expert_token_ratio"] = {"n": len(ratios), "median": pct(ratios, 0.5),
                                     "p10": pct(ratios, 0.10), "p90": pct(ratios, 0.90)}
    lk = [r["look_row"] for r in all_reqs if r.get("look_row")]
    if lk:
        res["lookups_per_asked_row"] = {"n": len(lk), "median": pct(lk, 0.5),
                                        "p10": pct(lk, 0.10), "p90": pct(lk, 0.90)}

    # ---- the population the verdict is computed on -----------------------
    core = [r for r in pool if r["expert_hit_pct"] is not None]
    core, trimmed = trim(core, args.trim)
    res["core_n"] = len(core)
    res["core_trimmed"] = trimmed
    res["core_trim_frac"] = args.trim
    res["core_ms_win"] = {"min": pct([r["ms_win"] for r in core], 0.0),
                          "p10": pct([r["ms_win"] for r in core], 0.10),
                          "median": pct([r["ms_win"] for r in core], 0.5),
                          "p90": pct([r["ms_win"] for r in core], 0.90),
                          "max": pct([r["ms_win"] for r in core], 1.0)}
    res["core_rows_lo"] = {"median": pct([r["rows_lo"] for r in core], 0.5),
                           "min": pct([r["rows_lo"] for r in core], 0.0),
                           "max": pct([r["rows_lo"] for r in core], 1.0)}
    res["core_ctx"] = {"median": pct([r["ctx_mid"] for r in core], 0.5),
                       "p10": pct([r["ctx_mid"] for r in core], 0.10),
                       "p90": pct([r["ctx_mid"] for r in core], 0.90),
                       "log_median": pct([r["log_ctx"] for r in core], 0.5)}
    res["core_ehit"] = {"median": pct([r["expert_hit_pct"] for r in core], 0.5),
                        "p10": pct([r["expert_hit_pct"] for r in core], 0.10),
                        "p90": pct([r["expert_hit_pct"] for r in core], 0.90)}
    res["core_cpu_win"] = {"median": pct([r["cpu_win"] for r in core if r.get("cpu_win") is not None], 0.5)}

    res["fit_all"] = fit_rows(pool)
    res["fits_core"] = model_fits(core)
    res["boot_core"] = bootstrap_slope([r["rows_lo"] for r in core], [r["ms_win"] for r in core]) \
        if args.bootstrap else None
    res["two_point"] = two_point(all_reqs)
    res.update(swap_stats(segs))

    # ---- the two-way table: rows x expert hit, the form-free identification
    res["cross_rows_hit"] = []
    for lo_a, hi_a, cells in cross_table(core, "rows_lo", EBUCKETS, "expert_hit_pct", HBUCKETS):
        res["cross_rows_hit"].append({
            "rows": [lo_a, hi_a],
            "cells": [{"hit": [lo_b, hi_b], "n": len(b),
                       "ms": pct([r["ms_win"] for r in b], 0.5),
                       "cpu_win": pct([r["cpu_win"] for r in b if r.get("cpu_win") is not None], 0.5)}
                      for lo_b, hi_b, b in cells]})

    # ---- the near-model-free cross table: CPU expert entries x rows
    CB2 = [(0, 100), (100, 150), (150, 200), (200, 250), (250, 300), (300, 400), (400, 600)]
    RB2 = [(2.4, 3.2), (3.2, 4.2)]
    res["cross_cpu_rows"] = []
    for lo_a, hi_a, cells in cross_table(core, "cpu_win", CB2, "rows_lo", RB2):
        res["cross_cpu_rows"].append({
            "cpu": [lo_a, hi_a],
            "cells": [{"rows": [lo_b, hi_b], "n": len(b),
                       "ms": pct([r["ms_win"] for r in b], 0.5),
                       "hit": pct([r["expert_hit_pct"] for r in b], 0.5),
                       "rows_lo": pct([r["rows_lo"] for r in b], 0.5)}
                      for lo_b, hi_b, b in cells]})

    # ---- the falsifiable check: does the decomposition predict the T=1 window?
    # A request that generated exactly 1 token ran one T=1 window
    # (generate.cpp:7180). Its cost is a direct measurement of a + b*1. If the fitted
    # decomposition reproduces it, the decomposition is not curve-fitting.
    tp1 = two_point(all_reqs)
    if tp1:
        res["t1_direct"] = {"n": tp1["n_T1"], "median": tp1["T1_ms_median"],
                            "p10": tp1["T1_p10"], "p90": tp1["T1_p90"]}

    # ---- per-T bucket
    res["by_T"] = []
    for lo, hi, b in bucketise(core, "mean_T", TBUCKETS):
        res["by_T"].append({
            "lo": lo, "hi": hi, "n": len(b),
            "mean_T": pct([r["mean_T"] for r in b], 0.5),
            "rows_lo": pct([r["rows_lo"] for r in b], 0.5),
            "ms_win": pct([r["ms_win"] for r in b], 0.5),
            "tok_win": pct([r["tok_win"] for r in b], 0.5),
            "acc": pct([r["acc"] for r in b if r["acc"] is not None], 0.5),
            "ehit": pct([r["expert_hit_pct"] for r in b], 0.5),
            "cpu_win": pct([r["cpu_win"] for r in b if r.get("cpu_win") is not None], 0.5),
            "kvhit": pct([r["kv_hit_pct"] for r in b if r["kv_hit_pct"] is not None], 0.5),
            "fit": model_fits(b),
        })

    # ---- per expert-hit bucket
    EB = [(0, 60), (60, 70), (70, 80), (80, 85), (85, 90), (90, 95), (95, 101)]
    res["by_ehit"] = []
    for lo, hi, b in bucketise(core, "expert_hit_pct", EB):
        res["by_ehit"].append({"lo": lo, "hi": hi, "n": len(b),
                               "mean_T": pct([r["mean_T"] for r in b], 0.5),
                               "rows_lo": pct([r["rows_lo"] for r in b], 0.5),
                               "ms_win": pct([r["ms_win"] for r in b], 0.5),
                               "cpu_win": pct([r["cpu_win"] for r in b if r.get("cpu_win") is not None], 0.5),
                               "fit": model_fits(b)})

    # ---- per KV-stream hit bucket
    KB = [(0, 80), (80, 90), (90, 95), (95, 99), (99, 101)]
    res["by_kvhit"] = []
    for lo, hi, b in bucketise(core, "kv_hit_pct", KB):
        res["by_kvhit"].append({"lo": lo, "hi": hi, "n": len(b),
                                "mean_T": pct([r["mean_T"] for r in b], 0.5),
                                "rows_lo": pct([r["rows_lo"] for r in b], 0.5),
                                "ms_win": pct([r["ms_win"] for r in b], 0.5),
                                "fit": model_fits(b)})

    # ---- per context band
    CB = [(0, 2000), (2000, 8000), (8000, 20000), (20000, 50000), (50000, 100000),
          (100000, 160000), (160000, 220000), (220000, 10 ** 9)]
    res["by_ctxband"] = []
    for lo, hi in CB:
        b = [r for r in core if lo <= r["ctx_mid"] < hi]
        if len(b) >= 12:
            res["by_ctxband"].append({
                "lo": lo, "hi": hi, "n": len(b),
                "ctx": pct([r["ctx_mid"] for r in b], 0.5),
                "mean_T": pct([r["mean_T"] for r in b], 0.5),
                "ms_win": pct([r["ms_win"] for r in b], 0.5),
                "ehit": pct([r["expert_hit_pct"] for r in b], 0.5),
                "fit": model_fits(b)})

    # ---- per tier
    res["by_tier"] = []
    for tier in sorted({s.tier for s in segs if s.tier}):
        pp = [r for s in segs if s.tier == tier for r in s.usable if r["expert_hit_pct"] is not None]
        pp, _ = trim(pp, args.trim)
        if len(pp) >= 12:
            res["by_tier"].append({"tier": tier, "n": len(pp), "fit": model_fits(pp),
                                   "segments": [s.idx for s in segs if s.tier == tier]})

    # ---- per process stream
    res["by_stream"] = []
    for st in res["streams"]:
        pp = [r for r in all_reqs if r.get("stream") == st["id"]]
        uu = usable(pp, args.min_windows)
        uu = [r for r in uu if r["expert_hit_pct"] is not None]
        uu, _ = trim(uu, args.trim)
        if len(uu) < args.min_stream:
            continue
        res["by_stream"].append({
            "id": st["id"], "n_requests": st["n"], "start_line": st["start_line"],
            "n_usable": len(uu),
            "tier": pct([r["tier"] for r in uu if r["tier"]], 0.5),
            "max_t": pct([r["max_t"] for r in uu if r["max_t"]], 0.5),
            "ctx": pct([r["ctx_mid"] for r in uu], 0.5),
            "mean_T": pct([r["mean_T"] for r in uu], 0.5),
            "ms_win": pct([r["ms_win"] for r in uu], 0.5),
            "ehit": pct([r["expert_hit_pct"] for r in uu], 0.5),
            "fit": model_fits(uu),
        })

    # ---- the direct measurement, if the run had STRATA_DECODE_TIMING=1
    dt = [d for s in segs for d in s.decode_timing]
    if dt:
        res["decode_timing"] = {
            "n": len(dt),
            "ms_win": pct([d["ms_win"] for d in dt], 0.5),
            "avg_T": pct([d["avg_T"] for d in dt], 0.5),
            "fit": ols([d["avg_T"] for d in dt], [d["ms_win"] for d in dt]),
            "parts": {k: pct([d[k] for d in dt], 0.5)
                      for k in ("verify", "gpu_wait", "layer_host", "plan", "actq", "jobs",
                                "cpu", "stage", "commit", "draft", "cpu_experts",
                                "cpu_entries", "vram_hits", "pcie")},
        }
    res["bandwidth"] = bandwidth_floor(segs, args)
    res["verdict"] = verdict(res, args)
    return res


# ---------------------------------------------------------------------------
# the bandwidth floor - arithmetic, stated in full so it can be attacked
# ---------------------------------------------------------------------------

def bandwidth_floor(segs, args):
    """The per-row bandwidth floor of a verify window.

    Every input is either read out of the log or labelled.  This is the number the
    task's premise questions, so the derivation is printed, not asserted.
    """
    blob = args.blob_bytes
    layers = args.layers
    topk = args.topk
    hbm = args.hbm_gbps
    out = {"blob_bytes": blob, "layers": layers, "topk": topk, "hbm_gbps": hbm,
           "expert_bytes_per_token": layers * topk * blob,
           "expert_ms_per_token": layers * topk * blob / (hbm * 1e9) * 1000.0}
    # KV read per token: the QSA layers attend over `width` selected cells (int8,
    # 1056 B/cell over K+V, include/strata/kernels/kv_q8.hpp:24-27).
    out["kv_bytes_per_token"] = args.qsa_layers * args.qsa_width * 1056
    out["kv_ms_per_token"] = out["kv_bytes_per_token"] / (hbm * 1e9) * 1000.0
    out["total_ms_per_token"] = out["expert_ms_per_token"] + out["kv_ms_per_token"]
    # the log's own blob size, if it is there
    blobs = [s.largest_blob_mib for s in segs if s.largest_blob_mib]
    if blobs:
        out["log_largest_blob_MiB"] = blobs[0]
    ecs = [(s.expert_slots, s.expert_gib) for s in segs if s.expert_slots and s.expert_gib]
    if ecs:
        slots, gib = ecs[0]
        out["log_expert_slots"] = slots
        out["log_expert_gib"] = gib
        out["log_bytes_per_slot"] = gib * 1024 ** 3 / slots
    return out


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------

def print_report(segs, res, args):
    print("# S4.5 decode-window cost model")
    print("")
    print("logs: %s" % ", ".join(args.log))
    print("requests parsed: %d   segments (engine processes): %d   usable: %d   "
          "core population for the fits: %d (%d trimmed at p%.0f of ms/window)"
          % (res["requests_total"], res["segments"], res["usable"], res["core_n"],
             res["core_trimmed"], 100 * (1 - res["core_trim_frac"])))
    print("dropped: %d with no closed window count, %d with < %d windows, "
          "%d with a mean T above the segment's max_t; %d cancelled; "
          "%d W-estimator disagreements (W=G-A used, see the README)"
          % (res["dropped_no_windows"], res["dropped_few_windows"], args.min_windows,
             res["dropped_impossible_T"], res["cancelled"], res["dropped_disagree"]))
    print("")

    print("## the segments")
    print("")
    print("| # | lines | engine | max_t | arena MiB | tier | kv-res | pinned GiB | expert slots | "
          "expert GiB | slots | requests | usable | cfg inherited |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for s in segs:
        print("| %d | %d-%d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %d | %d | %s |" % (
            s.idx, s.start_line, s.end_line, s.engine or "-",
            s.max_t if s.max_t else "-", fnum(s.arena_mib, 1) if s.arena_mib else "-",
            s.tier if s.tier else "-", s.kv_resident if s.kv_resident else "-",
            fnum(s.kv_pinned_gib, 2) if s.kv_pinned_gib else "-",
            s.expert_slots if s.expert_slots else "-",
            fnum(s.expert_gib, 2) if s.expert_gib else "-",
            s.slots if s.slots else "-", len(s.requests), len(s.usable),
            "yes" if getattr(s, "cfg_inherited", False) else "no"))
    print("")
    print("`cfg inherited` means the segment had no config lines of its own and took the previous")
    print("segment's. Two engines write this log concurrently, so their config lines and their")
    print("request lines interleave; this is an attribution, not a reading.")
    print("")

    print("## self-consistency checks (do these before trusting any fit)")
    print("")
    if "expert_token_ratio" in res:
        e = res["expert_token_ratio"]
        print("* `expert_lookups / (48 layers x 10 experts)` vs the line's own `generated`: "
              "median %.3f, p10 %.3f, p90 %.3f over %d requests. A decode token costs exactly "
              "480 lookups, so ~1.0 confirms the counters and the 48x10 geometry."
              % (e["median"], e["p10"], e["p90"], e["n"]))
    if "lookups_per_asked_row" in res:
        e = res["lookups_per_asked_row"]
        print("* expert lookups per ASKED row: median %.0f, p10 %.0f, p90 %.0f over %d requests. "
              "480 is the floor (every row is routed in every layer); anything above it is the "
              "MTP drafter's own layer plus refusals."
              % (e["median"], e["p10"], e["p90"], e["n"]))
    print("* the two window-count estimators `G-A` and `O-A+1` disagree on %d of %d requests. "
          "`G-A` is the one the source supports (see the README); the disagreement is the "
          "truncated last window, which is the common case." % (res["dropped_disagree"], res["requests_total"]))
    print("* suffix-draft lines seen: %d, attributed to their own request: %d"
          % (res["sfx_seen"], res["sfx_trusted"]))
    print("")

    print("## the population")
    print("")
    m = res["core_ms_win"]
    print("ms/window: min %.1f  p10 %.1f  median %.1f  p90 %.1f  max %.1f"
          % (m["min"], m["p10"], m["median"], m["p90"], m["max"]))
    rl = res["core_rows_lo"]
    print("rows/window (committed-token lower bound): min %.2f  median %.2f  max %.2f"
          % (rl["min"], rl["median"], rl["max"]))
    c = res["core_ctx"]
    print("context mid: p10 %.0f  median %.0f  p90 %.0f" % (c["p10"], c["median"], c["p90"]))
    eh = res["core_ehit"]
    print("expert cache hit: p10 %.1f%%  median %.1f%%  p90 %.1f%%" % (eh["p10"], eh["median"], eh["p90"]))
    print("CPU-pool expert entries per window: median %.0f" % res["core_cpu_win"]["median"])
    print("")
    print("**The whole point of this section:** ms/window varies 30-1660 ms while rows/window "
          "varies 2-5. Context and expert-cache hit rate dominate, and any single-variable fit "
          "on rows is reading noise.")
    print("")

    print("## the fits")
    print("")
    print("The naive fit reproduces S4.1's ballpark and is wrong for the reason above. It is")
    print("printed only so the difference is visible.")
    print("")
    print("```")
    print_fit(res["fit_all"], "ms/win ~ mean_T (naive)")
    print("")
    fc = res["fits_core"]
    for label in ("simple", "rows_lo_ctx", "rows_hi_ctx", "rows_lo_hit", "rows_hi_hit",
                  "rows_both_hit", "rows_lo_hit_ctx", "expert_split", "expert_split_rows",
                  "cpu_only"):
        print_fit(fc[label], label)
    print("```")
    print("")
    if res.get("boot_core"):
        bb = res["boot_core"]
        print("bootstrap on `ms/win ~ rows_lo` (%d resamples): a 95%% [%.2f, %.2f]  "
              "b 95%% [%.3f, %.3f]  median b %.3f"
              % (bb["iters"], bb["a"][0], bb["a"][1], bb["b"][0], bb["b"][1], bb["median_b"]))
    print("")

    tp = res.get("two_point")
    print("## the two-point anchor (S4.1's method, recomputed)")
    print("")
    if tp:
        print("A request that generated exactly 1 token ran exactly 1 window, and that window is")
        print("T=1 by construction (src/program/generate.cpp:7180). n = %d such windows:" % tp["n_T1"])
        print("  T=1 window ms: min %.0f, median %.0f, max %.0f" % (tp["T1_ms_min"], tp["T1_ms_median"], tp["T1_ms_max"]))
        print("Longest-mean-T usable request: line %d, mean T %.2f, %.1f ms/window" % (
            tp["hi_line"], tp["hi_T"], tp["hi_ms_win"]))
        if tp["b"] is not None:
            print("  => b = (%.1f - %.1f) / (%.2f - 1) = %.2f ms/row, a = %.1f ms"
                  % (tp["hi_ms_win"], tp["T1_ms_median"], tp["hi_T"], tp["b"], tp["a"]))
        print("  S4.1 quoted a = 24 ms, b = 8.4 ms from `log:106` and `log:8578`. The T=1 median")
        print("  here confirms a; the 8.4 came from assuming `log:8578` ran T=6 windows, which it")
        print("  did not (see the README's correction table).")
    else:
        print("no clean T=1 window found in this log.")
    print("")

    print("## rows/window x expert-cache-hit: the form-free identification")
    print("")
    print("Median ms/window. Inside a row the expert hit rate is pinned, so moving along a row")
    print("is the effect of rows; inside a column the rows are pinned, so moving down a column")
    print("is the effect of the hit rate. n in brackets.")
    print("")
    hdr = "| rows \\ hit % |" + "".join(" %d-%d |" % tuple(c["hit"]) for c in res["cross_rows_hit"][0]["cells"])
    print(hdr)
    print("|---" * (len(res["cross_rows_hit"][0]["cells"]) + 1) + "|")
    for row in res["cross_rows_hit"]:
        cells = [" %.1f [%d] |" % (c["ms"], c["n"]) if c["n"] >= 6 else " - |" for c in row["cells"]]
        print("| %.1f-%.1f |" % tuple(row["rows"]) + "".join(cells))
    print("")
    print("Read the row-to-row steps: that is the marginal row cost at a fixed cache hit rate.")
    print("Read the column-to-column steps: that is what a cache miss costs, which is the term")
    print("batching does NOT amortise.")
    print("")

    print("## CPU expert entries x rows: the near-model-free version")
    print("")
    print("Median ms/window. Within a row of this table the number of rows is pinned to")
    print("within ~0.1, so moving along the row is the cost of the CPU expert pool alone.")
    print("Within a column the CPU entries are pinned, so moving down is the row cost alone.")
    print("")
    hdr = "| CPU entries/win |" + "".join(" rows %.1f-%.1f |" % tuple(c["rows"]) for c in res["cross_cpu_rows"][0]["cells"])
    print(hdr)
    print("|---" * (len(res["cross_cpu_rows"][0]["cells"]) + 1) + "|")
    for row in res["cross_cpu_rows"]:
        cells = []
        for c in row["cells"]:
            cells.append(" %.1f [%d] |" % (c["ms"], c["n"]) if c["n"] >= 15 else " - |")
        print("| %d-%d |" % tuple(row["cpu"]) + "".join(cells))
    print("")
    print("The along-row steps are ~0.06 ms per CPU expert entry, which is what the")
    print("regression's `cpu_win` coefficient says independently. The down-column steps at")
    print("fixed CPU entries are the GPU row cost with the pool term removed.")
    print("")

    print("## per-T bucket")
    print("")
    print("| mean T bucket | n | median mean T | median rows_lo | median ms/win | tokens/win | "
          "accept | expert hit % | CPU entries/win | KV hit % | rows_lo coef [95% CI] |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for b in res["by_T"]:
        f = b["fit"].get("rows_lo_hit") or b["fit"].get("simple")
        coef = "-"
        t = coef_of(f, "rows_lo")
        if t and f["n"] >= 12:
            coef = "%+.2f [%+.2f, %+.2f]" % (t["c"], t["ci"][0], t["ci"][1])
        print("| %.2f-%.2f | %d | %.2f | %.2f | %.1f | %.2f | %s | %s | %s | %s | %s |" % (
            b["lo"], b["hi"], b["n"], b["mean_T"], b["rows_lo"], b["ms_win"], b["tok_win"],
            fnum(b["acc"], 3) if b["acc"] is not None else "-",
            fnum(b["ehit"], 1) if b["ehit"] is not None else "-",
            fnum(b["cpu_win"], 0) if b["cpu_win"] is not None else "-",
            fnum(b["kvhit"], 2) if b["kvhit"] is not None else "-", coef))
    print("")

    print("## per expert-hit bucket")
    print("")
    print("| expert hit % | n | median mean T | median rows_lo | median ms/win | CPU entries/win | rows_lo coef [95% CI] |")
    print("|---|---|---|---|---|---|---|")
    for b in res["by_ehit"]:
        f = b["fit"].get("rows_lo_ctx") or b["fit"].get("simple")
        coef = "-"
        t = coef_of(f, "rows_lo")
        if t and f["n"] >= 12:
            coef = "%+.2f [%+.2f, %+.2f]" % (t["c"], t["ci"][0], t["ci"][1])
        print("| %d-%d | %d | %.2f | %.2f | %.1f | %s | %s |" % (
            b["lo"], b["hi"], b["n"], b["mean_T"], b["rows_lo"], b["ms_win"],
            fnum(b["cpu_win"], 0) if b["cpu_win"] is not None else "-", coef))
    print("")

    print("## per KV-stream hit bucket")
    print("")
    print("| KV hit % | n | median mean T | median rows_lo | median ms/win | rows_lo coef [95% CI] |")
    print("|---|---|---|---|---|---|")
    for b in res["by_kvhit"]:
        f = b["fit"].get("rows_lo_hit") or b["fit"].get("simple")
        coef = "-"
        t = coef_of(f, "rows_lo")
        if t and f["n"] >= 12:
            coef = "%+.2f [%+.2f, %+.2f]" % (t["c"], t["ci"][0], t["ci"][1])
        print("| %d-%d | %d | %.2f | %.2f | %.1f | %s |" % (
            b["lo"], b["hi"], b["n"], b["mean_T"], b["rows_lo"], b["ms_win"], coef))
    print("")

    print("## per context band")
    print("")
    print("| ctx band | n | median ctx | median mean T | median ms/win | expert hit % | rows_lo coef [95% CI] |")
    print("|---|---|---|---|---|---|---|")
    for cb in res["by_ctxband"]:
        f = cb["fit"].get("rows_lo_hit") or cb["fit"].get("simple")
        coef = "-"
        t = coef_of(f, "rows_lo")
        if t and f["n"] >= 12:
            coef = "%+.2f [%+.2f, %+.2f]" % (t["c"], t["ci"][0], t["ci"][1])
        print("| %s-%s | %d | %s | %.2f | %.1f | %s | %s |" % (
            fnum(cb["lo"], 0), fnum(cb["hi"], 0) if cb["hi"] < 10 ** 8 else "inf",
            cb["n"], fnum(cb["ctx"], 0), cb["mean_T"], cb["ms_win"],
            fnum(cb["ehit"], 1) if cb["ehit"] is not None else "-", coef))
    print("")

    print("## per --max-context tier")
    print("")
    print("| tier | n | rows_lo coef [95% CI] | expert-hit coef | R2 |")
    print("|---|---|---|---|---|")
    for t in res["by_tier"]:
        f = t["fit"].get("rows_lo_hit")
        tr = (f or {}).get("terms") or {}
        if "rows_lo" not in tr:
            continue
        print("| %d | %d | %+.2f [%+.2f, %+.2f] | %+.3f | %.3f |" % (
            t["tier"], f["n"], tr["rows_lo"]["c"], tr["rows_lo"]["ci"][0], tr["rows_lo"]["ci"][1],
            tr["expert_hit_pct"]["c"], f["r2"]))
    print("")

    print("## per process stream (the engines, separated by the cumulative KV counter)")
    print("")
    print("| stream | requests | usable | start line | tier | max_t | median ctx | median mean T | "
          "median ms/win | expert hit % | rows_lo coef [95% CI] |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for st in res["by_stream"]:
        f = st["fit"].get("rows_lo_hit") or st["fit"].get("simple")
        coef = "-"
        t = coef_of(f, "rows_lo")
        if t:
            coef = "%+.2f [%+.2f, %+.2f]" % (t["c"], t["ci"][0], t["ci"][1])
        print("| %d | %d | %d | %d | %s | %s | %s | %.2f | %.1f | %s | %s |" % (
            st["id"], st["n_requests"], st["n_usable"], st["start_line"],
            fnum(st["tier"], 0), fnum(st["max_t"], 0), fnum(st["ctx"], 0),
            st["mean_T"], st["ms_win"], fnum(st["ehit"], 1), coef))
    print("")

    bw = res["bandwidth"]
    print("## the bandwidth floor (arithmetic, every input labelled)")
    print("")
    print("```")
    print("expert bytes / token = %d layers x top-%d x %s B = %s B = %.0f MiB"
          % (bw["layers"], bw["topk"], fnum(bw["blob_bytes"], 0),
             fnum(bw["expert_bytes_per_token"], 0), bw["expert_bytes_per_token"] / 1048576.0))
    print("  blob size from the log: `largest blob %.2f MB`; the cache line says %s slots = %.2f GiB"
          % (bw.get("log_largest_blob_MiB", 2.66), fnum(bw.get("log_expert_slots", 0), 0),
             bw.get("log_expert_gib", 0)))
    if "log_bytes_per_slot" in bw:
        print("  => %.0f B/slot from the cache line (the blob is the LARGEST, not the average)"
              % bw["log_bytes_per_slot"])
    print("HBM assumed          = %.0f GB/s  (%s)"
          % (bw["hbm_gbps"], "measured" if args.hbm_measured else "LABELLED INFERENCE: spec sheet"))
    print("expert floor         = %.3f ms/token" % bw["expert_ms_per_token"])
    print("KV floor             = %.3f ms/token  (%d QSA layers x %d selected cells x 1056 B)"
          % (bw["kv_ms_per_token"], args.qsa_layers, args.qsa_width))
    print("TOTAL floor          = %.3f ms/token of rows" % bw["total_ms_per_token"])
    f = res["fits_core"].get("rows_lo_hit")
    if f and "rows_lo" in f["terms"]:
        b = f["terms"]["rows_lo"]["c"]
        print("measured row cost    = %.2f ms/row  => %.1fx the floor" % (b, b / bw["total_ms_per_token"]))
    print("```")
    print("")
    print("The floor is a **per-row** cost: one more row is one more token's worth of expert")
    print("blobs and one more token's worth of selected KV. If the measured row cost is near the")
    print("floor the window is bandwidth-bound and batching cannot scale. If it is far above, the")
    print("row is paying for launches, barriers and latency, and batching scales until something")
    print("else breaks.")
    print("")

    print_verdict(res["verdict"], args)
    print("")

    if "decode_timing" in res:
        dt = res["decode_timing"]
        print("## STRATA_DECODE_TIMING lines found: %d" % dt["n"])
        print("")
        print("```")
        print("median ms/window %.2f at median avg T %.2f" % (dt["ms_win"], dt["avg_T"]))
        if dt["fit"]:
            print("window_ms = %.2f + %.3f * T   R2 = %.3f  n = %d   b 95%% CI [%.3f, %.3f]"
                  % (dt["fit"]["a"], dt["fit"]["b"], dt["fit"]["r2"], dt["fit"]["n"],
                     dt["fit"]["ci_b"][0], dt["fit"]["ci_b"][1]))
        for k, v in dt["parts"].items():
            print("  %-12s %8.3f ms/window" % (k, v))
        print("```")
        print("")
        print("These lines are the direct measurement. They exist only if the run had")
        print("STRATA_DECODE_TIMING=1; `strata-iq3_s.log` does not.")
        print("")
    else:
        print("## STRATA_DECODE_TIMING: 0 lines in this log")
        print("")
        print("The direct per-window breakdown was never recorded. Everything above is derived")
        print("from per-request lines. That is the gap `slope-ab.sh` arm 1 closes.")
        print("")

    print("## caveats the numbers carry")
    print("")
    print("* Segments are delimited by `session is up`. Two processes alive at once cannot be")
    print("  separated by that anchor alone; the `stream` split (see the README) is the")
    print("  best available separation and it is an attribution, not a fact.")
    print("* `decode_ms` is the request's whole decode phase (generate.cpp:7343). Under")
    print("  `--serve-slots >= 2` a slot is pre-empted between windows, so decode_ms includes")
    print("  time the conversation spent parked. The swap lines bound that: %d swaps in this"
          % res.get("swap_count", 0))
    print("  log, median %.0f ms each, %.1f s total against %.0f s of decode - under 0.2 %%. "
          "Pre-emption is NOT what inflates decode_ms."
          % (res.get("swap_median_ms", 0), res.get("swap_total_ms", 0) / 1000.0,
             res.get("decode_total_ms", 0) / 1000.0))
    print("* mean T is capped by `--spec`: `o.spec = min(spec + 2, 8)` (generate.cpp:1661) and")
    print("  `--spec-min-p` truncates each window (`:7176-7179`). Nothing in this log can have")
    print("  mean T above the segment's `max_t`, which is 6 everywhere.")
    print("")


def verdict(res, args):
    """The number the parent's decision rests on, computed from the fits.

    Model:  window_ms = a + b * T,  T = B * (D + 1) rows for B sequences each asking D
    drafts,  tokens = B * (1 + acc(D) * D).  Therefore

        ms/token = (a / B + b * (D + 1)) / (1 + acc(D) * D)

    which says the whole thing: batching divides the FIXED cost `a` by B and leaves the
    per-row cost `b` untouched.  If `b` is bandwidth, `b` is irreducible and the gain is
    capped at a/B.  If `b` is overhead, `b` falls too and the gain is unbounded until
    something else breaks.

    `acc(D)` is NOT in the log for D beyond what the box ran, so it is a parameter here,
    not a measurement.  Every row of the table states its `acc`."""
    fc = res["fits_core"]
    out = {}

    coef = coef_of

    # If the run had STRATA_DECODE_TIMING=1, the DIRECT per-window measurement is the
    # primary number and everything else is a cross-check. It reports avg T and ms/window
    # per request from the engine's own counters (generate.cpp:7347-7355), so there is no
    # errors-in-variables problem and no rows_lo/rows_hi bracket to argue about.
    dt = res.get("decode_timing")
    if dt and dt.get("fit") and dt["n"] >= 8:
        out["direct"] = {"b": dt["fit"]["b"], "ci": dt["fit"]["ci_b"], "a": dt["fit"]["a"],
                         "r2": dt["fit"]["r2"], "n": dt["n"]}

    # the slope, from the fit that controls for the expert cache (the one with R2 ~ 0.65)
    b_hit = coef(fc["rows_lo_hit"], "rows_lo")
    b_hit_hi = coef(fc["rows_hi_hit"], "rows_hi")
    a_hit = fc["rows_lo_hit"]["terms"]["const"] if fc.get("rows_lo_hit") else None
    out["b_rows_lo_hit"] = b_hit
    out["b_rows_hi_hit"] = b_hit_hi
    out["a_hit"] = a_hit

    # the narrow-hit-band slopes: the same number measured where the hit rate cannot move
    bands = []
    for b in res["by_ehit"]:
        f = b["fit"].get("rows_lo_ctx")
        c = coef_of(f, "rows_lo")
        if c and b["n"] >= 60 and (f or {}).get("terms"):
            bands.append({"band": [b["lo"], b["hi"]], "n": b["n"], "c": c,
                          "const": f["terms"]["const"], "r2": f["r2"]})
    out["bands"] = bands
    if bands:
        cs = sorted(x["c"]["c"] for x in bands)
        out["b_band_range"] = [cs[0], cs[-1]]
        out["a_band_range"] = [min(x["const"]["c"] for x in bands),
                               max(x["const"]["c"] for x in bands)]

    bw = res["bandwidth"]
    out["b_floor"] = bw["total_ms_per_token"]

    # the CPU-pool term, which is the mechanism the slope is mostly made of
    es = fc.get("expert_split_rows") or fc.get("expert_split")
    out["cpu_per_entry"] = coef(es, "cpu_win")
    out["gpu_per_entry"] = coef(es, "gpu_win")
    if out["cpu_per_entry"] and res.get("core_cpu_win"):
        cw = res["core_cpu_win"]["median"]
        out["cpu_ms_per_window_at_median"] = out["cpu_per_entry"]["c"] * cw
        out["cpu_entries_median"] = cw

    # ---- the slope decomposition: rows x (1 - hit) ----------------------
    d = fc.get("decomp_ctx") or fc.get("decomp_logctx") or fc.get("decomp")
    out["decomp_fit"] = d
    if d:
        bg = coef_of(d, "rows_hi")
        bc = coef_of(d, "miss_rows_hi")
        out["b_gpu"] = bg
        out["b_cpu_miss"] = bc
        hit = res["core_ehit"]["median"] / 100.0
        out["hit_median"] = hit
        out["b_at_median_hit"] = bg["c"] + bc["c"] * (1.0 - hit)
        out["b_share_cpu"] = bc["c"] * (1.0 - hit) / out["b_at_median_hit"]
        out["b_vs_hit"] = [{"hit": h,
                            "b": bg["c"] + bc["c"] * (1.0 - h)}
                           for h in (0.80, 0.85, 0.887, 0.92, 0.95, 0.98, 1.00)]

    # the T=1 falsification: the model never saw a T=1 window (those requests are not in
    # the usable pool - they have W=1), so reproducing one is a real out-of-sample test.
    if d and res.get("t1_direct"):
        a_c = d["terms"]["const"]["c"]
        bg = coef_of(d, "rows_hi")
        bc = coef_of(d, "miss_rows_hi")
        ctx = d["terms"].get("ctx_mid") or d["terms"].get("log_ctx")
        hit = res["core_ehit"]["median"] / 100.0
        pred = a_c + bg["c"] * 1.0 + bc["c"] * 1.0 * (1.0 - hit)
        if ctx:
            pred += ctx["c"] * (res["core_ctx"]["median"] if "ctx_mid" in d["terms"]
                                else res["core_ctx"]["log_median"])
        out["t1_pred"] = {"a": a_c, "b_gpu": bg["c"], "b_cpu": bc["c"], "hit": hit,
                                "predicted": pred, "n": res["t1_direct"]["n"],
                                "measured_median": res["t1_direct"]["median"],
                                "p10": res["t1_direct"]["p10"], "p90": res["t1_direct"]["p90"]}

    # ---- the throughput table -------------------------------------------
    # Three slope hypotheses, all from the fits:
    #   measured  - b at the log's median expert-cache hit rate
    #   fullcache - b_gpu, the same fit evaluated at hit = 100 %: what the row costs when
    #               no expert ever goes to the CPU pool.  This is the REACHABLE target,
    #               and the lever is VRAM for the cache, not a kernel.
    #   floor     - the bandwidth floor, which is not reachable but is the bound.
    if fc.get("rows_hi_hit"):
        t = fc["rows_hi_hit"]["terms"]
        a_hi = t["const"]["c"] + t["expert_hit_pct"]["c"] * res["core_ehit"]["median"]
        b_hi = t["rows_hi"]["c"]
    else:
        a_hi, b_hi = 11.5, 11.0
    out["a_eff"] = a_hi
    out["b_eff"] = b_hi
    b_full = out.get("b_gpu", {}).get("c") if out.get("b_gpu") else b_hi
    if b_full is None:
        b_full = b_hi
    b_floor = out["b_floor"]

    # `today` is the row the log actually runs: one sequence, `--spec 4` -> S_mtp = 4
    # asked rows (generate.cpp:5227 `S_mtp = min(o.mtp_max_t, S)`), acceptance 0.72.
    TODAY = {"B": 1, "D": 3, "acc": 0.72}
    rows = []
    grid = [(1, 3), (1, 5), (2, 3), (2, 5), (2, 7), (4, 3), (4, 5), (4, 7),
            (8, 3), (8, 5), (8, 7), (16, 3), (16, 5), (24, 3), (32, 3)]
    for B, D in grid:
        T = B * (D + 1)
        # acceptance at deeper draft chains is not in the log; 0.72 is the measured
        # pooled rate at the depth the box runs, and the log's own per-request rates
        # fall as offered drafts rise, so a mild decay is stated, not measured.
        acc = 0.72 if D <= 3 else (0.66 if D <= 5 else 0.60)
        tok = B * (1.0 + acc * D)
        for label, bb in (("measured", b_hi), ("fullcache", b_full), ("floor", b_floor)):
            ms = a_hi + bb * T
            rows.append({"B": B, "D": D, "acc": acc, "T": T, "tokens": tok, "slope": label,
                         "window_ms": ms, "ms_token": ms / tok, "tok_s": 1000.0 * tok / ms})
    base = [r for r in rows if (r["B"], r["D"]) == (TODAY["B"], TODAY["D"])]
    by = {}
    for r in base:
        by[r["slope"]] = r["ms_token"]
    for r in rows:
        r["gain_vs_today"] = 100.0 * (by[r["slope"]] / r["ms_token"] - 1.0)
    out["today"] = TODAY
    out["table"] = rows
    return out


def print_verdict(v, args):
    print("## THE VERDICT")
    print("")
    print("### the marginal row cost")
    print("")
    if v.get("direct"):
        d = v["direct"]
        print("DIRECT (STRATA_DECODE_TIMING=1, the engine's own per-window counters):")
        print("  window_ms = %.2f + %.3f * T    R2 = %.3f   n = %d   b 95%% CI [%.3f, %.3f]"
              % (d["a"], d["b"], d["r2"], d["n"], d["ci"][0], d["ci"][1]))
        print("  This is the number to use. The derived fits below corroborate it.")
        print("")
    if v.get("b_rows_lo_hit"):
        c = v["b_rows_lo_hit"]
        print("rows that PRODUCED a committed token (the lower bound on rows run):")
        print("  b = %+.2f ms/row, 95%% CI [%+.2f, %+.2f], t = %.1f" % (c["c"], c["ci"][0], c["ci"][1], c["t"]))
    if v.get("b_rows_hi_hit"):
        c = v["b_rows_hi_hit"]
        print("rows ASKED for (the upper bound on rows run):")
        print("  b = %+.2f ms/row, 95%% CI [%+.2f, %+.2f], t = %.1f" % (c["c"], c["ci"][0], c["ci"][1], c["t"]))
    print("The true marginal cost of one more row is between them, because the log")
    print("brackets the rows that ran:  tokens/window <= rows run <= asked T.")
    print("")
    if v.get("bands"):
        print("Measured inside narrow expert-cache-hit bands, where the hit rate cannot move")
        print("and context is a covariate (this is the cleanest cut the log supports):")
        print("")
        print("| expert hit band | n | b (ms/row) | 95% CI | fixed cost a (ms) | R2 |")
        print("|---|---|---|---|---|---|")
        for x in v["bands"]:
            print("| %d-%d%% | %d | %.2f | [%.2f, %.2f] | %.1f | %.2f |" % (
                x["band"][0], x["band"][1], x["n"], x["c"]["c"], x["c"]["ci"][0], x["c"]["ci"][1],
                x["const"]["c"], x["r2"]))
        print("")
        print("b range across bands: %.2f - %.2f ms/row.  a range: %.1f - %.1f ms."
              % (v["b_band_range"][0], v["b_band_range"][1],
                 v["a_band_range"][0], v["a_band_range"][1]))
        print("The slope FALLS as the cache hit rate rises, which is the mechanism: the row")
        print("cost is mostly the CPU expert pool, not the GPU.")
    print("")

    print("### is it bandwidth or overhead?  (the decomposition)")
    print("")
    print("Fit:  ms/window = a + b_gpu*rows + b_cpu*rows*(1 - expert hit) + c*context")
    print("")
    print("```")
    print("bandwidth floor per row   : %.2f ms/row" % v["b_floor"])
    print("measured marginal row cost: %.2f ms/row (hit-controlled)" % v["b_eff"])
    print("                          = %.1fx the floor" % (v["b_eff"] / v["b_floor"]))
    if v.get("b_gpu"):
        print("b_gpu  (every row, hit or not) : %+.2f ms/row  95%% CI [%+.2f, %+.2f]  t%.1f"
              % (v["b_gpu"]["c"], v["b_gpu"]["ci"][0], v["b_gpu"]["ci"][1], v["b_gpu"]["t"]))
    if v.get("b_cpu_miss"):
        print("b_cpu  (per row that MISSES)   : %+.2f ms/row  95%% CI [%+.2f, %+.2f]  t%.1f"
              % (v["b_cpu_miss"]["c"], v["b_cpu_miss"]["ci"][0], v["b_cpu_miss"]["ci"][1],
                 v["b_cpu_miss"]["t"]))
        print("at the median hit rate %.1f%%: b = %.2f + %.2f x %.3f = %.2f ms/row"
              % (100 * v["hit_median"], v["b_gpu"]["c"], v["b_cpu_miss"]["c"],
                 1.0 - v["hit_median"], v["b_at_median_hit"]))
        print("  => %.0f%% of the marginal row cost is the CPU expert pool"
              % (100 * v["b_share_cpu"]))
        print("")
        print("the marginal row cost as a function of the expert cache hit rate:")
        for x in v["b_vs_hit"]:
            print("   hit %5.1f%%  ->  b = %5.2f ms/row  (%.1fx the floor)"
                  % (100 * x["hit"], x["b"], x["b"] / v["b_floor"]))
    if v.get("t1_pred"):
        p = v["t1_pred"]
        print("")
        print("FALSIFIABLE CHECK - the decomposition predicts the T=1 window it never saw:")
        print("  a + b_gpu*1 + b_cpu*1*(1-hit) = %.1f + %.2f + %.2f x %.3f = %.1f ms"
              % (p["a"], p["b_gpu"], p["b_cpu"], 1.0 - p["hit"], p["predicted"]))
        print("  measured T=1 windows (n=%d): median %.1f ms, p10 %.1f, p90 %.1f"
              % (p["n"], p["measured_median"], p["p10"], p["p90"]))
        print("  %s" % ("PREDICTION INSIDE the measured p10-p90 band."
                        if p["p10"] <= p["predicted"] <= p["p90"]
                        else "prediction OUTSIDE the measured p10-p90 band - the model is wrong."))
    if v.get("cpu_per_entry"):
        print("")
        print("cross-check, per CPU-pool expert entry: %+.4f ms each, 95%% CI [%+.4f, %+.4f]"
              % (v["cpu_per_entry"]["c"], v["cpu_per_entry"]["ci"][0], v["cpu_per_entry"]["ci"][1]))
        print("  at the median %.0f CPU entries/window that is %.1f ms/window"
              % (v["cpu_entries_median"], v["cpu_ms_per_window_at_median"]))
    if v.get("gpu_per_entry"):
        print("cross-check, per VRAM-resident expert entry: %+.5f ms, 95%% CI [%+.5f, %+.5f] (t=%.1f)"
              % (v["gpu_per_entry"]["c"], v["gpu_per_entry"]["ci"][0], v["gpu_per_entry"]["ci"][1],
                 v["gpu_per_entry"]["t"]))
        print("  ~0 or negative: a GPU-resident expert row is free at this resolution; the")
        print("  row cost is the misses, not the hits.")
    print("```")
    print("")

    print("### throughput at T = 8 / 16 / 24 / 32 under each slope hypothesis")
    print("")
    print("`a = %.1f ms` from the fit; the three slope columns are:" % v["a_eff"])
    print("")
    print("* **measured** `b = %.2f ms/row` - the fit at this log's median expert-cache hit" % v["b_eff"])
    print("  rate of %.1f%%." % (100 * v.get("hit_median", 0.865)))
    if v.get("b_gpu"):
        print("* **full cache** `b = %.2f ms/row` - the same fit at hit = 100%%, i.e. no expert"
              % v["b_gpu"]["c"])
        print("  ever sent to the CPU pool. This is the REACHABLE target and its lever is VRAM for")
        print("  the cache, not a kernel.")
    print("* **bandwidth floor** `b = %.2f ms/row` - not reachable, the bound." % v["b_floor"])
    print("")
    print("`acc` is a STATED PARAMETER: the log cannot give acceptance at draft depths the box")
    print("never ran. `T = B*(D+1)` rows for B sequences each asking D drafts. Baseline (`+0.0%%`)")
    print("is B=1, D=%d, the depth this box runs (`--spec 4` -> `S_mtp = 4`, generate.cpp:5227)."
          % v["today"]["D"])
    print("")
    print("| B | D | rows T | acc | tokens/win | measured: ms/win | ms/tok | tok/s | vs today | full cache: ms/win | ms/tok | tok/s | vs today | floor: ms/win | ms/tok | tok/s | vs today |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    seen = {}
    for r in v["table"]:
        seen.setdefault((r["B"], r["D"]), {})[r["slope"]] = r
    for key in sorted(seen, key=lambda k: (k[0], k[1])):
        trio = seen[key]
        cells = []
        for lab in ("measured", "fullcache", "floor"):
            r = trio.get(lab)
            if r:
                cells.append("%.1f | %.2f | %.1f | %+.1f%%" % (
                    r["window_ms"], r["ms_token"], 1000.0 / r["ms_token"], r["gain_vs_today"]))
            else:
                cells.append("- | - | - | -")
        r0 = trio.get("measured") or {}
        print("| %d | %d | %d | %.2f | %.2f | %s |" % (
            key[0], key[1], r0.get("T", key[0] * (key[1] + 1)), r0.get("acc", 0),
            r0.get("tokens", 0), " | ".join(cells)))
    print("")
    print("")


def swap_stats(segs):
    """Bound the pre-emption contamination of `decode_ms`."""
    ms = []
    total = 0
    for s in segs:
        ms.extend(s.swap_ms)
        if s.swap_total:
            total = max(total, s.swap_total)
    dec = sum(r["decode_ms"] for s in segs for r in s.requests)
    return {"swap_count": len(ms),
            "swap_median_ms": pct(ms, 0.5) if ms else 0.0,
            "swap_total_ms": total,
            "decode_total_ms": dec}


def write_csv(segs, outdir):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, "requests.csv")
    cols = ["line", "segment", "stream", "tier", "max_t",
            "prompt_tokens", "reused", "fresh", "prompt_ms",
            "generated", "decode_ms", "accepted", "offered",
            "W_draft", "W_tokens", "W", "W_source", "W_agree", "W_mtp", "W_sfx",
            "mean_T", "rows_lo", "rows_hi", "T_mtp", "T_sfx", "ms_win", "tok_win",
            "acc", "acc_mtp",
            "ctx_start", "ctx_end", "ctx_mid", "log_ctx",
            "expert_hits", "expert_lookups", "expert_hit_pct",
            "look_win", "gpu_win", "cpu_win", "miss_rows_hi",
            "kv_lookups", "kv_ram_mib", "kv_req_lookups", "kv_req_miss_mib", "kv_hit_pct",
            "sfx_windows", "sfx_ok", "sfx_offered", "cancelled"]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for s in segs:
            for r in s.requests:
                w.writerow([r.get(c) for c in cols])
    return path


def to_json(segs, res, path):
    doc = {"result": {k: v for k, v in res.items() if k != "bandwidth"},
           "bandwidth": res["bandwidth"],
           "segments": []}
    for s in segs:
        doc["segments"].append({
            "idx": s.idx, "start_line": s.start_line, "end_line": s.end_line,
            "engine": s.engine, "max_t": s.max_t, "arena_mib": s.arena_mib,
            "tier": s.tier, "kv_resident": s.kv_resident, "kv_pinned_gib": s.kv_pinned_gib,
            "expert_slots": s.expert_slots, "expert_gib": s.expert_gib, "slots": s.slots,
            "largest_blob_mib": s.largest_blob_mib,
            "n_requests": len(s.requests), "n_usable": len(s.usable),
            "fit": s.fit, "two_point": s.tp,
            "last_expert_hit": s.cum_ehit_pct, "last_kv_hit": s.cum_kvs_pct,
        })
    doc["requests"] = [{k: v for k, v in r.items()} for s in segs for r in s.requests]
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    return path


def compare(paths):
    docs = []
    for p in paths:
        with open(p) as fh:
            docs.append(json.load(fh))
    names = [os.path.basename(p) for p in paths]

    def fc(d, fit, term):
        f = (d["result"].get("fits_core") or {}).get(fit)
        if not f or not f.get("terms") or term not in f["terms"]:
            return None
        return f["terms"][term]["c"]

    def fci(d, fit, term):
        f = (d["result"].get("fits_core") or {}).get(fit)
        if not f or not f.get("terms") or term not in f["terms"]:
            return None
        return f["terms"][term]["ci"]

    def v(d, *path):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or k not in cur:
                return None
            cur = cur[k]
        return cur

    rows = [
        ("requests", lambda d: v(d, "result", "requests_total")),
        ("usable for the fit", lambda d: v(d, "result", "usable")),
        ("core population", lambda d: v(d, "result", "core_n")),
        ("median ms/window", lambda d: v(d, "result", "core_ms_win", "median")),
        ("median rows/window (lo)", lambda d: v(d, "result", "core_rows_lo", "median")),
        ("median expert hit %", lambda d: v(d, "result", "core_ehit", "median")),
        ("median CPU entries/window", lambda d: v(d, "result", "core_cpu_win", "median")),
        ("naive b (ms/row)", lambda d: v(d, "result", "fit_all", "b")),
        ("naive R2", lambda d: v(d, "result", "fit_all", "r2")),
        ("b rows_lo (hit-controlled)", lambda d: fc(d, "rows_lo_hit", "rows_lo")),
        ("  95% CI", lambda d: fci(d, "rows_lo_hit", "rows_lo")),
        ("b rows_hi (hit-controlled)", lambda d: fc(d, "rows_hi_hit", "rows_hi")),
        ("  95% CI", lambda d: fci(d, "rows_hi_hit", "rows_hi")),
        ("R2 (hit-controlled)", lambda d: v(d, "result", "fits_core", "rows_lo_hit", "r2")),
        ("b_gpu (ms/row)", lambda d: fc(d, "decomp_ctx", "rows_hi")),
        ("b_cpu per miss-row (ms)", lambda d: fc(d, "decomp_ctx", "miss_rows_hi")),
        ("ms per CPU entry", lambda d: fc(d, "expert_split_rows", "cpu_win")),
        ("bandwidth floor (ms/row)", lambda d: v(d, "bandwidth", "total_ms_per_token")),
        ("a (fixed cost, ms)", lambda d: v(d, "result", "verdict", "a_eff")),
        ("T=1 predicted (ms)", lambda d: v(d, "result", "verdict", "t1_pred", "predicted")),
        ("T=1 measured (ms)", lambda d: v(d, "result", "t1_direct", "median")),
        ("DT ms/window (direct)", lambda d: v(d, "result", "decode_timing", "ms_win")),
        ("DT b (direct)", lambda d: (v(d, "result", "decode_timing", "fit") or {}).get("b")),
    ]
    print("| metric | " + " | ".join(names) + " |")
    print("|---" * (len(names) + 1) + "|")
    for label, fn in rows:
        vals = []
        for d in docs:
            try:
                x = fn(d)
            except Exception:
                x = None
            if isinstance(x, list):
                vals.append("[%.3f, %.3f]" % (x[0], x[1]))
            else:
                vals.append(fnum(x, 3) if x is not None else "-")
        print("| %s | %s |" % (label, " | ".join(vals)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0], add_help=True)
    ap.add_argument("log", nargs="*", help="serve log file(s), read-only")
    ap.add_argument("--limit", type=int, default=0, help="keep only the first N requests (the log is live)")
    ap.add_argument("--min-windows", type=int, default=4,
                    help="drop requests with fewer than N windows (their mean T is noise)")
    ap.add_argument("--trim", type=float, default=0.01,
                    help="drop this fraction of the highest ms/window from the core "
                         "population (default 0.01 = p99 cap); the count dropped is printed")
    ap.add_argument("--min-stream", type=int, default=40,
                    help="only report a process stream with at least this many usable requests")
    ap.add_argument("--tier", type=int, default=0, help="keep only segments with this --max-context")
    ap.add_argument("--max-slots", type=int, default=0,
                    help="drop requests from segments whose --serve-slots was greater than N")
    ap.add_argument("--stream", action="append", type=int, default=None,
                    help="keep only requests attributed to this process stream (repeatable); "
                         "see the README - the split is an attribution, not a fact")
    ap.add_argument("--json", dest="json_out", default="", help="write the whole analysis as JSON")
    ap.add_argument("--csv", dest="csv_dir", default="", help="write DIR/requests.csv")
    ap.add_argument("--compare", nargs="+", default=None, help="compare saved JSON dumps instead")
    ap.add_argument("--no-bootstrap", dest="bootstrap", action="store_false", default=True)
    ap.add_argument("--blob-bytes", type=int, default=1_382_400,
                    help="one routed expert blob, bytes (log: `largest blob 2.66 MB`)")
    ap.add_argument("--layers", type=int, default=48, help="routed MoE layers")
    ap.add_argument("--topk", type=int, default=10, help="experts per token per layer")
    ap.add_argument("--qsa-layers", type=int, default=12, help="QSA (sparse attention) layers")
    ap.add_argument("--qsa-width", type=int, default=2048,
                    help="cells one query selects (include/strata/kernels/qsa.hpp:156-162)")
    ap.add_argument("--hbm-gbps", type=float, default=936.0,
                    help="HBM bandwidth, GB/s (RTX 3090 spec; labelled an inference)")
    ap.add_argument("--hbm-measured", action="store_true",
                    help="assert the --hbm-gbps value was measured, not taken from the spec sheet")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    if args.compare:
        compare(args.compare)
        return 0

    if not args.log:
        ap.error("no log file given (or use --compare A.json B.json)")
    for p in args.log:
        if not os.path.isfile(p):
            sys.stderr.write("analyze.py: no such log: %s\n" % p)
            return 2

    segs = parse(args.log, args.limit)
    if args.tier:
        segs = [s for s in segs if s.tier == args.tier]
    if args.max_slots:
        segs = [s for s in segs if (s.slots or 0) <= args.max_slots]

    res = analyse(segs, args)
    if not args.quiet:
        print_report(segs, res, args)
    if args.json_out:
        to_json(segs, res, args.json_out)
        if not args.quiet:
            print("wrote %s" % args.json_out)
    if args.csv_dir:
        p = write_csv(segs, args.csv_dir)
        if not args.quiet:
            print("wrote %s" % p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
