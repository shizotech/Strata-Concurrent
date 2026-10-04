#!/usr/bin/env python3
"""S3.1e-1: prove no std::printf / std::cout / std::fprintf line was added,
removed or REORDERED in the serve block.

Both files are reduced to the sequence of output calls **in execution order**:
the serve-scope declarations, then the request body.  In the old file the body
is one straight-line block; in the new file the step functions appear in the
order the serial driver calls them, so file order == execution order there too.
The hoisted helpers are placed where they execute (inside the prefill step)."""
import re, sys

OUT = re.compile(r"std::(printf|fprintf|cout|puts|write)\b")

def calls(lines, a, b):
    out = []
    for l in lines[a - 1:b]:
        if OUT.search(l):
            out.append(re.sub(r"\s+", " ", l.strip()))
    return out

old = open(sys.argv[1]).read().split("\n")
new = open(sys.argv[2]).read().split("\n")

def fnd(lines, sub, start=1):
    for i in range(start - 1, len(lines)):
        if lines[i].strip() == sub:
            return i + 1
    raise SystemExit("not found: " + sub)

# ---------- OLD: serve block, straight-line body
oa = next(i for i, l in enumerate(old, 1) if l == "    if (o.serve) {")
ob = next(i for i, l in enumerate(old, 1) if i > oa and l == "        return 0;")
OB = calls(old, oa, ob)

# ---------- NEW: serve scope, then the steps in call order
na = next(i for i, l in enumerate(new, 1) if l == "    if (o.serve) {")
nb = next(i for i, l in enumerate(new, 1) if i > na and l == "        return 0;")
def c8(start):
    for i in range(start, nb + 1):
        if new[i - 1] == "        };":
            return i
    raise SystemExit("no close after " + str(start))
def fn(sub):
    return fnd(new, sub, na)
h1 = fn("auto windows_ok = [&](int64_t a, int64_t b) -> bool {")
h2 = fn("auto lend = [&](int64_t tokens, std::string& e) -> bool {"); h2e = c8(h2)
ds = fn("auto dec_snap = [&]() {"); dse = c8(ds)
pp = fn("auto prep_request = [&](ReqCtx& R, const std::string& line) -> Prep {"); ppe = c8(pp)
pl = fn("auto plan_prompt_segments = [&](ReqCtx& R) {"); ple = c8(pl)
rp = fn("auto run_prefill_step = [&](ReqCtx& R) -> Step {"); rpe = c8(rp)
fp = fn("auto finish_prefill = [&](ReqCtx& R) -> Step {"); fpe = c8(fp)
rd = fn("auto run_decode_step = [&](ReqCtx& R) -> Step {"); rde = c8(rd)
fr = fn("auto finish_request = [&](ReqCtx& R) -> Step {"); fre = c8(fr)
lb = fn("while (next_line(line)) {")
NB = []
# execution order: serve scope, prep_request (body part 1), the hoisted prompt
# helpers (which in 0.1.30 sat inside the body, right before the segment loop),
# then the prefill/decode/finish steps, then the loop glue.
for a, b in [(na, h1 - 1), (h2e + 1, pp - 1), (pp + 1, ppe), (h1, h2e), (pl + 1, ple),
             (rp + 1, rpe), (fp + 1, fpe), (rd + 1, rde), (fr + 1, fre), (lb, nb)]:
    NB += calls(new, a, b)

print("old output calls: %d   new: %d" % (len(OB), len(NB)))
if OB == NB:
    print("IDENTICAL, in order.")
    sys.exit(0)
import difflib
for l in difflib.unified_diff(OB, NB, "old", "new", lineterm="", n=0):
    print(l)
sys.exit(1)
