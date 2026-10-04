#!/usr/bin/env python3
"""S3.1e-1: prove an extracted region is the original code.

Normalises (drop blank + comment-only lines, collapse whitespace) and diffs the
original straight-line region against the extracted region.  Every remaining
difference must be an alias declaration or an exit-statement rewrite."""
import re, sys, difflib

def norm(ls):
    out = []
    for l in ls:
        s = l.strip()
        if not s or s.startswith("//"):
            continue
        out.append(re.sub(r"\s+", " ", s))
    return out

old = open(sys.argv[1]).read().split("\n")
new = open(sys.argv[2]).read().split("\n")
OB = norm(old[int(sys.argv[3]) - 1:int(sys.argv[4])])
NB = []
for a, b in zip(sys.argv[5::2], sys.argv[6::2]):
    NB += norm(new[int(a) - 1:int(b)])
for l in difflib.unified_diff(OB, NB, "before", "after", lineterm="", n=0):
    print(l)
print("=== before %d stmt-lines, after %d stmt-lines" % (len(OB), len(NB)))
