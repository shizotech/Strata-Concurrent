#!/usr/bin/env python3
"""S3.1e-1 phase D: extract the verify-window loop into run_decode_step."""
import sys

P = "src/program/generate.cpp"
lines = open(P).read().split("\n")

def L(n): return lines[n - 1]
def chk(n, sub):
    if sub not in L(n):
        raise SystemExit("line %d does not contain %r\n  got: %r" % (n, sub, L(n)))
def ind(n):
    l = L(n); return l[:len(l) - len(l.lstrip())]

R12 = "            "
A, B = 5985, 6096          # the while loop, inclusive of its closing brace
chk(A, "while (!cancelled && produced_n < max_new) {")
chk(B, "}")
chk(B + 1, "const double decode_ms =")

REP = {}
def rep(n, t): REP[n] = t
def dele(n): REP[n] = ""

chk(6006, "if (p + T > o.max_context) break;")
rep(6006, ind(6006) + "if (p + T > o.max_context) return Step::finished;")
for n in [6028, 6041, 6083, 6087]:
    chk(n, "return 1;"); assert L(n).strip() == "return 1;", L(n)
    rep(n, ind(n) + "return Step::error;")
chk(6092, 'if (eos) { finish = "stop"; break; }')
rep(6092, ind(6092) + 'if (eos) { finish = "stop"; return Step::finished; }')
chk(6093, 'if (stopped()) { finish = "cancel"; break; }')
rep(6093, ind(6093) + 'if (stopped()) { finish = "cancel"; return Step::cancelled; }')
dele(A); dele(B)

HEAD = [
    "        // ---- phase 4: ONE verify window (draft, verify, commit, emit, re-draft).  0.1.30 looped here",
    "        // until the request was done; the serial loop calls this until it stops reporting",
    "        // `progressed`, which is the same windows in the same order.  This is what S3.1e-2 interleaves",
    "        // between slots, and it is also the natural pre-emption point (a window is ~16 ms).",
    "        //",
    "        // Nothing here allocates per token: the window/draft/probability buffers live in `ReqCtx` and",
    "        // are sized once by `finish_prefill`.",
    "        auto run_decode_step = [&](ReqCtx& R) -> Step {",
    R12 + "const long long& max_new = R.max_new;",
    R12 + "const double& req_spec_min_p = R.req_spec_min_p;",
    R12 + "const int& hist_n = R.hist_n;",
    R12 + "int64_t& p = R.p;",
    R12 + "int32_t& x = R.x;",
    R12 + "std::vector<int32_t>& drafts = R.drafts;",
    R12 + "std::vector<int32_t>& window = R.window;",
    R12 + "std::vector<int32_t>& outv = R.outv;",
    R12 + "std::vector<int32_t>& sbuf = R.sbuf;",
    R12 + "std::vector<float>& dprob = R.dprob;",
    R12 + "std::vector<int32_t>& consumed = R.consumed;",
    R12 + "bool& first_window = R.first_window;",
    R12 + "bool& cancelled = R.cancelled;",
    R12 + "int64_t& produced_n = R.produced_n;",
    R12 + "int64_t& sfx_windows = R.sfx_windows;",
    R12 + "int64_t& sfx_drafts = R.sfx_drafts;",
    R12 + "int64_t& sfx_ok = R.sfx_ok;",
    R12 + "int64_t& draft_offered = R.draft_offered;",
    R12 + "int64_t& draft_accepted = R.draft_accepted;",
    R12 + "const char*& finish = R.finish;",
    R12 + "double& dt_run = R.dt_run;",
    R12 + "double& dt_commit = R.dt_commit;",
    R12 + "double& dt_draft = R.dt_draft;",
    R12 + "int64_t& dec_windows = R.dec_windows;",
    R12 + "int64_t& dec_T = R.dec_T;",
    R12 + "// 0.1.30's `while (!cancelled && produced_n < max_new)` guard",
    R12 + "if (cancelled || produced_n >= max_new) return Step::finished;",
]
body = []
for i in range(A + 1, B):
    if REP.get(i, None) == "":
        continue
    l = REP.get(i, L(i))
    if l.strip():
        assert l.startswith("                "), (i, repr(l))
        l = l[4:]
    body.append(l)
TAIL = [R12 + "return Step::progressed;", "        };"]

GLUE = [
    "            // ---- S3.1e-1 phase 4: one verify window per step (0.1.30: one straight-line loop)",
    "            Step dst = Step::progressed;",
    "            while ((dst = run_decode_step(R)) == Step::progressed) {}",
    "            if (dst == Step::error) return 1;",
]

out = lines[:A - 1] + GLUE + lines[B + 1 - 1:]
# insert the step right after finish_prefill's closing brace
k = next(i for i, l in enumerate(out) if l.strip() == "R.decode_look0 = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;")
k2 = next(i for i in range(k, len(out)) if out[i] == "        };")
out = out[:k2 + 1] + HEAD + body + TAIL + out[k2 + 1:]

open(P, "w").write("\n".join(out))
print("phase D done")
