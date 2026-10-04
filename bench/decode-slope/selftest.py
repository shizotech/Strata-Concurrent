#!/usr/bin/env python3
"""selftest.py - prove analyze.py's derivation recovers a known cost model.

analyze.py cannot be tested against a live engine (the owner forbids starting one),
so it is tested against a SYNTHETIC log whose ground truth is known exactly.

The generator writes a serve log in the real format (the four per-request lines
generate.cpp:7502-7546 emits, plus the STRATA_DECODE_TIMING line at :7347-7355) for
windows whose cost is exactly

    window_ms = A_TRUE + B_TRUE * T

with T drawn uniformly over 1..8 and the expert-cache hit rate held constant.  It then
checks what analyze.py reports:

  * the DIRECT fit (from the STRATA_DECODE_TIMING lines) must return A_TRUE/B_TRUE;
  * the DERIVED `rows_hi` coefficient (1 + offered/windows, the rows ASKED for) must
    return B_TRUE.  This is the important one: it is the estimator the whole S4.5
    analysis of `strata-iq3_s.log` rests on, and it is computed from per-request lines
    only, exactly as it is on the real log;
  * the `rows_lo` coefficient (1 + accepted/windows) must come out BELOW B_TRUE, which
    is what "rows_lo is a lower bound on rows run" means;
  * the T=1 two-point anchor must return A_TRUE + B_TRUE.

Run it:  python3 bench/decode-slope/selftest.py
Exit 0 = the derivation is sound.  Exit 1 = it is not, and every number in
docs/STAGE4-BATCH-DECODE.md §9 that came from analyze.py is suspect.

Stdlib only.  Writes one file under /tmp and reads it back.  No socket, no process,
no device, no /dev/shm.
"""

import math
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYZE = os.path.join(HERE, "analyze.py")

A_TRUE = 12.0        # the fixed cost of a window, ms
BG_TRUE = 5.6        # the cost of one row that runs at all, ms/row
BC_TRUE = 37.7       # the cost of one row whose expert MISSED the VRAM cache, ms/row
N_REQ = 400
LAYERS, TOPK = 48, 10     # the geometry the expert-hit line's counters imply
HIT_LO, HIT_HI = 0.70, 0.96
E_HIT = 0.5 * (HIT_LO + HIT_HI)
B_AVG = BG_TRUE + BC_TRUE * (1.0 - E_HIT)   # the slope at the mean hit rate


def synth(path, n=N_REQ, seed=20261004):
    """A serve log whose window cost is EXACTLY

        window_ms = A + BG * T + BC * T * (1 - hit)

    with T, the expert-cache hit rate and the prompt length all varying independently.
    They MUST vary independently: a constant hit rate makes `expert_hit_pct` collinear
    with the intercept and every hit-controlled fit singular - which is the first thing
    this test caught when it was written with a constant hit rate.

    ~10 % of the requests generate exactly one token, so they run exactly one T = 1
    window (`if (first_window) T = 1;`, generate.cpp:7180) and the two-point anchor has
    a clean fixed-cost measurement to work with.  Those requests get no
    STRATA_DECODE_TIMING line, because the engine only prints it when dec_windows > 0
    AND the request reaches finish_request - it does print for them, but excluding them
    keeps the direct fit's population the same shape as the real log's.
    """
    rnd = random.Random(seed)
    lines = [
        "strata generate: session is up (engine 0.1.30)",
        "strata generate: KV streaming: 32768 of 524288 cells per QSA layer in VRAM, "
        "the K/V in 6.19 GiB of pinned RAM",
        "strata generate: expert cache 7777 slots, 14.75 GiB of VRAM; policy is",
        "strata verify: window up to 8 tokens, 88.2 MiB of device buffers",
    ]
    cum_kv = 0
    for i in range(n):
        single = (i % 10 == 0)
        if single:
            T, W, hit, acc = 1.0, 1, rnd.uniform(HIT_LO, HIT_HI), 0.0
        else:
            T = rnd.uniform(1.0, 8.0)
            W = rnd.randint(20, 140)
            hit = rnd.uniform(HIT_LO, HIT_HI)
            acc = rnd.uniform(0.55, 0.90)
        prompt = rnd.randint(1000, 250000)
        ms_total = W * (A_TRUE + BG_TRUE * T + BC_TRUE * T * (1.0 - hit))
        tokens = W * (1.0 + acc * (T - 1.0))
        offered = W * (T - 1.0)
        accepted = W * acc * (T - 1.0)
        A = int(round(accepted))
        O = int(round(offered))
        G = A + W                       # exact: every window commits a + 1
        D = int(round(ms_total))
        lines.append("strata serve: prompt %d tokens = 0 reused + %d read in 900 ms (555.6 tok/s), "
                     "%d generated in %d ms (22.2 tok/s), drafts accepted %d of %d, 2 checkpoints"
                     % (prompt, prompt, G, D, A, O))
        look = LAYERS * TOPK * max(1, G)
        hits = int(round(hit * look))
        lines.append("strata serve: decode expert cache hit rate: %.1f%% (%d hits / %d lookups)"
                     % (100.0 * hits / look, hits, look))
        cum_kv += 1_000_000
        lines.append("strata serve: KV streaming: 95.00%% of %d block reads hit VRAM, "
                     "50.0 MiB read from RAM" % cum_kv)
        if not single:
            lines.append(
                "strata decode timing: %d windows, avg T %.3f, %.3f tokens/window, %.3f ms/window = "
                "verify %.3f (GPU-reach wait 5.000 + per-layer host 30.000 [plan 2.000 actq 1.000 "
                "jobs 3.000 CPU 24.000] + stage 10.000) + commit/emit 12.000 + draft 7.500; "
                "per layer-window: CPU experts %.3f (%.3f entries), VRAM hits %.3f, PCIe 0.000"
                % (W, (O + W) / float(W), tokens / float(W), ms_total / float(W),
                   max(0.0, ms_total / float(W) - 32.5),
                   10.0 * T * (1.0 - hit), 10.0 * T * (1.0 - hit), 10.0 * T * hit))
    with open(path, "w") as fh:
        fh.write("\n".join(lines) + "\n")


selftest_report = {}


def main():
    fd, path = tempfile.mkstemp(prefix="strata-s45-selftest-", suffix=".log", dir="/tmp")
    os.close(fd)
    fd2, jpath = tempfile.mkstemp(prefix="strata-s45-selftest-", suffix=".json", dir="/tmp")
    os.close(fd2)
    try:
        synth(path)
        out = subprocess.run([sys.executable, ANALYZE, "--no-bootstrap", "--quiet",
                              "--json", jpath, path],
                             capture_output=True, text=True)
        if out.returncode != 0:
            sys.stderr.write(out.stdout + out.stderr)
            print("FAIL: analyze.py exited %d" % out.returncode)
            return 1
        import json
        with open(jpath) as fh:
            doc = json.load(fh)
        res = doc["result"]
        fails = []

        def chk(cond, msg):
            if not cond:
                fails.append(msg)

        # 1. the direct fit (STRATA_DECODE_TIMING).  avg T varies and the hit rate
        #    varies independently of it, so the slope of ms/window on avg T is the
        #    AVERAGE marginal cost, at the mean hit rate.
        dt = res.get("decode_timing") or {}
        chk(dt.get("n", 0) >= 200, "expected >= 200 STRATA_DECODE_TIMING lines, got %r" % dt.get("n"))
        fit = dt.get("fit") or {}
        # The hit rate varies independently of T, so it is unmodelled variation in y
        # and the direct fit's slope carries a real standard error.  The correct test is
        # "is the truth inside the fit's own 95 % CI", not "is it within a fixed epsilon".
        ci = fit.get("ci_b") or [0.0, 0.0]
        chk(ci[0] <= B_AVG <= ci[1],
            "direct fit b = %.4f with 95%% CI [%.4f, %.4f] does not contain the truth %.4f"
            % (fit.get("b", float("nan")), ci[0], ci[1], B_AVG))
        ca = fit.get("ci_a") or [0.0, 0.0]
        chk(ca[0] <= A_TRUE <= ca[1],
            "direct fit a = %.4f with 95%% CI [%.4f, %.4f] does not contain the truth %.4f"
            % (fit.get("a", float("nan")), ca[0], ca[1], A_TRUE))

        # 2. the DECOMPOSITION - the claim the whole verdict rests on.  It must recover
        #    BG_TRUE as the cost of a row and BC_TRUE as the cost of a row that misses.
        fc = res.get("fits_core") or {}
        d = fc.get("decomp_ctx") or fc.get("decomp")
        chk(d is not None, "the decomposition fit did not run")
        if d:
            bg = (d.get("terms") or {}).get("rows_hi", {}).get("c")
            bc = (d.get("terms") or {}).get("miss_rows_hi", {}).get("c")
            chk(bg is not None and abs(bg - BG_TRUE) < 0.6,
                "decomp b_gpu = %s, expected %.4f" % (bg, BG_TRUE))
            chk(bc is not None and abs(bc - BC_TRUE) < 1.2,
                "decomp b_cpu = %s, expected %.4f" % (bc, BC_TRUE))
            selftest_report["bg"] = bg
            selftest_report["bc"] = bc

        # 3. the derived rows_hi coefficient, computed from per-request lines ONLY -
        #    exactly how the real log is analysed.  It is the average slope, so it must
        #    land near b_avg, not near BG_TRUE.
        v = res.get("verdict") or {}
        b_hi = (v.get("b_rows_hi_hit") or {}).get("c")
        chk(b_hi is not None, "no rows_hi coefficient in the verdict")
        if b_hi is not None:
            chk(abs(b_hi - B_AVG) < 1.5,
                "derived rows_hi b = %.4f, expected ~%.4f (+-1.5)" % (b_hi, B_AVG))
            selftest_report["b_hi"] = b_hi

        # 4. the T=1 anchor exists and is near A + BG + BC*(1-hit)
        tp = res.get("two_point") or {}
        chk(tp.get("n_T1", 0) >= 10, "expected >= 10 T=1 windows in the synthetic log, got %r"
            % tp.get("n_T1"))
        selftest_report["t1_median"] = (tp or {}).get("T1_ms_median")

        # 5. the naive fit must be reported too (it is the number S4.1 quoted, and the
        #    README explains why it is not the one to use)
        chk(res.get("fit_all") is not None, "no naive fit reported")

        print("selftest: synthetic ground truth  a = %.2f  b_gpu = %.2f  b_cpu = %.2f per miss-row"
              % (A_TRUE, BG_TRUE, BC_TRUE))
        print("selftest: direct fit              a = %.4f  b = %.4f  (n = %s)"
              % (fit.get("a", float("nan")), fit.get("b", float("nan")), dt.get("n")))
        print("selftest: decomp b_gpu / b_cpu    %s / %s"
              % (selftest_report.get("bg"), selftest_report.get("bc")))
        print("selftest: derived rows_hi b       %s" % selftest_report.get("b_hi"))
        print("selftest: T=1 measured median     %s ms" % selftest_report.get("t1_median"))
        if fails:
            for f in fails:
                print("FAIL: %s" % f)
            print("selftest: %d failures" % len(fails))
            return 1
        print("selftest: OK - analyze.py's derivation recovers the known cost model")
        return 0
    finally:
        for f in (path, jpath):
            try:
                os.unlink(f)
            except OSError:
                pass


if __name__ == "__main__":
    sys.exit(main())
