#!/usr/bin/env python3
"""S3.1e-1 phase A: hoist windows_ok / read_windows / lend out of the serve
request body into the serve scope.  Pure move + dedent by 4 spaces."""
import sys

P = "src/program/generate.cpp"
lines = open(P).read().split("\n")          # lines[i] is 1-based line i+1

def find(sub, start=1, end=None):
    for i in range(start - 1, (end or len(lines))):
        if sub in lines[i]:
            return i + 1
    raise SystemExit("not found: " + sub)

# --- locate the three blocks by their unique anchors, not by hard-coded numbers
a_start = find("// A SHORT PART OF THE PROMPT")
lend_start = find("auto lend = [&](int64_t tokens")
# the lambda's closing `};` is the first line at exactly the lambda's own indent
lend_close = None
for i in range(lend_start, len(lines)):
    if lines[i] == "            };":
        lend_close = i + 1
        break
assert lend_close is not None
lend_cmt = lend_start
while lines[lend_cmt - 2].strip().startswith("//"):
    lend_cmt -= 1
assert "refill_one`/`refill` moved up" in lines[lend_cmt - 1], repr(lines[lend_cmt - 1])

rw_end = lend_cmt - 1
assert lines[rw_end - 1].strip() == "};", repr(lines[rw_end - 1])

block = lines[a_start - 1: lend_close]
rest = lines[:a_start - 1] + lines[lend_close:]

ded = []
for l in block:
    if l.startswith("            ") or l.strip() == "":
        ded.append(l[4:] if l.strip() else l)
    else:
        raise SystemExit("unexpected indent: " + repr(l))

ins = None
for i, l in enumerate(rest):
    if l.strip() == "std::vector<const float*> row_ptr;":
        ins = i + 1
assert ins is not None

header = [
    "        // ==================== S3.1e-1: the prompt-path helpers, hoisted out of the request body ====",
    "        // These three only ever captured serve-scope state, so moving them from the request body to the",
    "        // serve scope changes nothing: same captures, same code, same call sites.  They move because",
    "        // S3.1e-2's `run_prefill_step` calls them from outside the body, one segment per step, and a",
    "        // step function cannot reach a lambda declared inside a loop iteration.",
]
out = rest[:ins] + header + ded + rest[ins:]
open(P, "w").write("\n".join(out))
print("moved lines %d..%d -> after %d" % (a_start, lend_close, ins))
