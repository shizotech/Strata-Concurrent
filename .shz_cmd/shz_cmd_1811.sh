cd /ssd/Strata && python3 - <<'PYEOF'
import io
p='src/program/generate.cpp'
s=io.open(p,encoding='utf-8').read()
old='''                 "                         STRATA_NO_SWAP=1  --serve-slots >= 2: never hand the session to another
"'''
assert s.count(old)==1, "anchor"
new='''                 "                         STRATA_NO_SWAP=1  --serve-slots >= 2: never hand the session to another
"                 "                         STRATA_SERVE_TRACE=1  --serve-slots >= 2: one stderr line per scheduler
"                 "                                 decision - every pick with `Pick::why`, every slot phase
"                 "                                 transition, every swap with the reason it was asked for, and
"                 "                                 every loan/parking deferral.  Off by default so the normal log
"                 "                                 stays readable; the parking numbers, the `resumed from N
"                 "                                 tokens` line and the `re-reading from token 0` line are
"                 "                                 unconditional (one line per decision) because they are the
"                 "                                 ones that explain a request that ended early
"                 "                         STRATA_SERVE_ACTIVITY_S N  --serve-slots >= 2: also print the periodic
"                 "                                 `activity:` line every N seconds (default 30; 0 = only when
"                 "                                 the active set changes).  `slots_active` is printed only on a
"                 "                                 CHANGE, so a run that never got past one conversation looks
"                 "                                 identical to one that was idle; the activity line carries
"                 "                                 `peak=`, which tells them apart
'''
s=s.replace(old,new,1)
io.open(p,'w',encoding='utf-8').write(s)
print('help updated')
PYEOF
cd /ssd/Strata/build && ninja 2>&1 | grep -E "error:" | head -5; echo BUILD_DONE
