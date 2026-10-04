#!/usr/bin/env python3
"""S3.1e-1 verification: prove the extracted step bodies are the original code.

Compares the ORIGINAL request body (a saved copy) against the concatenation of
the extracted step functions + the loop-body glue, after normalising away:
  * leading whitespace
  * blank lines
  * lines that are pure comments
  * the deliberate alias / return-code rewrites (listed explicitly)
Anything left over is a real behaviour difference and gets printed."""
import re, sys

def norm(lines):
    out = []
    for l in lines:
        s = l.strip()
        if not s:
            continue
        if s.startswith("//"):
            continue
        out.append(re.sub(r"\s+", " ", s))
    return out

def region(lines, a, b):
    return norm(lines[a - 1:b])

old = open(sys.argv[1]).read().split("\n")
new = open(sys.argv[2]).read().split("\n")

# the original straight-line request body
OB = region(old, int(sys.argv[3]), int(sys.argv[4]))
# the new: prep_request + the loop glue + the remaining body
NB = region(new, int(sys.argv[5]), int(sys.argv[6]))

import difflib
d = difflib.unified_diff(OB, NB, "before", "after", lineterm="", n=1)
n = 0
for l in d:
    print(l)
    n += 1
print("=== diff lines:", n)
