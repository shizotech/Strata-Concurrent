cd /ssd/Strata && python3 - <<'PY'
p='docs/STAGE4-BATCH-DECODE.md'
s=open(p).read()
fixes=[
 ("(`src/program/generate.cpp:7183` `if (first_window) T = 1;`)",
  "(`src/program/generate.cpp:7180` `if (first_window) T = 1;`)"),
 ("(`src/program/generate.cpp:7188-7190`)", "(`src/program/generate.cpp:7196-7197`)"),
 ("today's `window[0..T)` (`generate.cpp:7188-7190`)", "today's `window[0..T)` (`generate.cpp:7196-7197`)"),
 ("(`src/program/generate.cpp:7188`)", "(`src/program/generate.cpp:7196`)"),
]
for a,b in fixes:
    if a not in s: print("MISS:", a[:60])
    s=s.replace(a,b)
open(p,'w').write(s)
PY
grep -n "7183\|7188\|7190\|7196\|7197\|7180" docs/STAGE4-BATCH-DECODE.md
