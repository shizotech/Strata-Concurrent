cd /ssd/Strata && python3 - <<'PY'
p=".megamind/src/program/s433-prefill-protocol-notes.md"
s=open(p).read()
a='''15. **`HandoffState::begin()` is legal from `H_QUEUED`** — that is the normal admission path after
    S4.2's wait queue lets a request through.'''
assert a in s
s=s.replace(a,'',1)
a2='''* **`HandoffState::begin()` must be legal from `H_QUEUED`** — that is the normal admission path — and
  `startable()` initially forgot it.'''
assert a2 in s
open(p,"w").write(s)
print("ok")
PY
grep -n "deviations from §4" .megamind/roadmap.md; grep -c "^// DEV-" include/strata/program/prefill_svc.hpp
