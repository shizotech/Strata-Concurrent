cd /ssd/Strata && python3 - <<'PYEOF'
import re, statistics
from pathlib import Path
lines = Path("strata-iq3_s.log").read_text(errors="replace").split("\n")
pat = re.compile(r"decode expert cache hit rate: ([0-9.]+)% \((\d+) hits / (\d+) lookups\)")
rows=[(i,float(m.group(1)),int(m.group(2)),int(m.group(3))) for i,l in enumerate(lines,1) if (m:=pat.search(l))]
# swaps and park events
sw=[i for i,l in enumerate(lines,1) if "park: slot" in l or "slot_swap" in l or "swap to slot" in l]
conc=[i for i,l in enumerate(lines,1) if "concurrent driver on" in l]
print("concurrent-driver start-ups:", len(conc), conc[:6])
print("park/swap lines:", len(sw))
# bucket hit rates by: was there a park within 400 lines before this hit-rate line?
swset=set(sw)
def near(i):
    return any((j in swset) for j in range(max(1,i-400), i))
withsw=[r for r in rows if near(r[0])]
without=[r for r in rows if not near(r[0])]
def agg(seg):
    h=sum(r[2] for r in seg); lk=sum(r[3] for r in seg)
    return 100.0*h/lk if lk else 0
print(f"\nnear a park/swap (n={len(withsw)}): mean {statistics.mean(r[1] for r in withsw):.1f}%  aggregate {agg(withsw):.1f}%")
print(f"far from park/swap (n={len(without)}): mean {statistics.mean(r[1] for r in without):.1f}%  aggregate {agg(without):.1f}%")
# also: hit rate vs request length (short requests thrash more)
lens=[]
for i,l in enumerate(lines,1):
    m=re.search(r"prompt (\d+) tokens", l)
    if m: lens.append((i,int(m.group(1))))
print("\nhit-rate percentiles:", [round(x,1) for x in statistics.quantiles((r[1] for r in rows), n=10)[1:9:2]])
PYEOF
