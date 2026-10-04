cd /ssd/Strata && python3 - <<'PY'
import re
p='docs/STAGE4-BATCH-DECODE.md'
s=open(p).read()
fixes = [
 ("(`strata-iq3_s.log:5165-5168`)", "(`strata-iq3_s.log:5164-5167`)"),
 ("(`strata-iq3_s.log:5191` shows a 2.9 GB/s probe on one card against 24.5 GB/s on another)",
  "(`strata-iq3_s.log:5184` shows a 2.9 GB/s probe on one card against 24.7 GB/s on the other, `:5148`)"),
 ("`strata-iq3_s.log:5167`) is the owner-run parity check", "`strata-iq3_s.log:5166`) is the owner-run parity check"),
 ("(`strata-iq3_s.log:5143`)", "(`strata-iq3_s.log:5142`)"),
]
for a,b in fixes:
    if a not in s: print("MISS:", a[:60])
    s=s.replace(a,b)
open(p,'w').write(s)
PY
grep -n "5143\|5191\|5165-5168\|5167)" docs/STAGE4-BATCH-DECODE.md | head
