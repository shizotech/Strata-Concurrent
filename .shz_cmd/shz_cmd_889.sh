cd /ssd/Strata && python3 - <<'PYEOF'
import re, statistics
from pathlib import Path
lines = Path("strata-iq3_s.log").read_text(errors="replace").split("\n")
pat = re.compile(r"decode expert cache hit rate: ([0-9.]+)% \((\d+) hits / (\d+) lookups\)")
rows=[]
for i,l in enumerate(lines,1):
    m=pat.search(l)
    if m: rows.append((i,float(m.group(1)),int(m.group(2)),int(m.group(3))))
print("total hit-rate lines:", len(rows))
# aggregate hits/lookups over the WHOLE log and over segments
def agg(seg):
    h=sum(r[2] for r in seg); lk=sum(r[3] for r in seg)
    return 100.0*h/lk if lk else 0
# find where the concurrent driver starts (activity: lines)
act=[i for i,l in enumerate(lines,1) if "activity: slots=" in l]
print("first 'activity:' line:", act[0] if act else None, " count:", len(act))
if act:
    cut=act[0]
    before=[r for r in rows if r[0]<cut]
    after=[r for r in rows if r[0]>=cut]
    print(f"\nBEFORE concurrent driver (n={len(before)}): mean {statistics.mean(r[1] for r in before):.1f}%  aggregate {agg(before):.1f}%")
    print(f"AFTER  concurrent driver (n={len(after)}): mean {statistics.mean(r[1] for r in after):.1f}%  aggregate {agg(after):.1f}%")
    print("min/max after:", min(r[1] for r in after), max(r[1] for r in after))
PYEOF
