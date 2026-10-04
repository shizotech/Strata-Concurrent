#!/usr/bin/env python3
"""S3.1e-1 phase E: extract phase 5 (the request's tail) into finish_request."""
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
A, B = 6136, 6350          # `const double decode_ms = ...` .. the last remote-experts fprintf
chk(A, "const double decode_ms =")
chk(B, "wait_before[(size_t) r]);")
chk(B + 1, "        }")
chk(B + 2, "return 0;")

REP = {}
def rep(n, t): REP[n] = t
for n in [6168, 6263]:
    chk(n, "return 1;"); assert L(n).strip() == "return 1;", L(n)
    rep(n, ind(n) + "return Step::error;")
chk(A, "const double decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();")
rep(A, R12 + "R.decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();")

HEAD = [
    "        // ---- phase 5: the request's tail - the decode-timing report, the conversation state this",
    "        // request leaves behind, the per-request metrics and the DONE line.  0.1.30 ran it inline at",
    "        // the end of the body; S3.1e-2 calls it when a slot's decode step reports it is done.",
    "        auto finish_request = [&](ReqCtx& R) -> Step {",
    R12 + "std::vector<int64_t>& ids = R.ids;",
    R12 + "const int64_t n = R.n;",
    R12 + "const int64_t& resume = R.resume;",
    R12 + "const double& prompt_ms = R.prompt_ms;",
    R12 + "bool& cancelled = R.cancelled;",
    R12 + "std::vector<int32_t>& consumed = R.consumed;",
    R12 + "int64_t& produced_n = R.produced_n;",
    R12 + "int64_t& sfx_windows = R.sfx_windows;",
    R12 + "int64_t& sfx_drafts = R.sfx_drafts;",
    R12 + "int64_t& sfx_ok = R.sfx_ok;",
    R12 + "int64_t& draft_offered = R.draft_offered;",
    R12 + "int64_t& draft_accepted = R.draft_accepted;",
    R12 + "const char*& finish = R.finish;",
    R12 + "const Clock::time_point& d0 = R.d0;",
    R12 + "const DecSnap& ds0 = R.ds0;",
    R12 + "double& dt_run = R.dt_run;",
    R12 + "double& dt_commit = R.dt_commit;",
    R12 + "double& dt_draft = R.dt_draft;",
    R12 + "int64_t& dec_windows = R.dec_windows;",
    R12 + "int64_t& dec_T = R.dec_T;",
    R12 + "const int64_t& decode_hits0 = R.decode_hits0;",
    R12 + "const int64_t& decode_look0 = R.decode_look0;",
    R12 + "const double& decode_ms = R.decode_ms;",
    R12 + "std::array<int64_t, 3>& remote_before = R.remote_before;",
    R12 + "std::array<int64_t, 3>& launches_before = R.launches_before;",
    R12 + "std::array<uint64_t, 3>& compact_before = R.compact_before;",
    R12 + "std::array<uint64_t, 3>& full_before = R.full_before;",
    R12 + "std::array<double, 3>& begin_before = R.begin_before;",
    R12 + "std::array<double, 3>& wait_before = R.wait_before;",
]
body = []
for i in range(A, B + 1):
    if REP.get(i, None) == "":
        continue
    body.append(REP.get(i, L(i)))
TAIL = [R12 + "return Step::finished;", "        };"]

GLUE = [
    "            // ---- S3.1e-1 phase 5: the request's tail (0.1.30: inline at the end of the body)",
    "            if (finish_request(R) == Step::error) return 1;",
]

out = lines[:A - 1] + GLUE + lines[B:]
k = next(i for i, l in enumerate(out) if l.strip() == "if (cancelled || produced_n >= max_new) return Step::finished;")
k2 = next(i for i in range(k, len(out)) if out[i] == "        };")
out = out[:k2 + 1] + HEAD + body + TAIL + out[k2 + 1:]

open(P, "w").write("\n".join(out))
print("phase E done")
