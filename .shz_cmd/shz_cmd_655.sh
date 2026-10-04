cd /ssd/Strata && python3 - <<'PY'
p=".shz_cmd/s433_gen_tables.py"
s=open(p).read()
old = """        to, act, why = ov.get(e, (dto, dact, dwhy))
        out.append('        {HState::%s, Event::%s, HState::%s, Act::%s,' % (s, e, to, act))"""
new = """        to, act, why = ov.get(e, (dto, dact, dwhy))
        # An illegal (refused) event must never move the state.  Enforced here so a typo in an
        # override cannot quietly make a refusal look like a transition.
        if act == "illegal":
            to = s
        out.append('        {HState::%s, Event::%s, HState::%s, Act::%s,' % (s, e, to, act))"""
assert old in s
s=s.replace(old,new,1)
old2 = """        to, act, why = ov.get(e, (dto, dact, dwhy))
        out.append('        {PJob::%s, PEvent::%s, PJob::%s, PAct::%s,' % (s, e, to, act))"""
new2 = """        to, act, why = ov.get(e, (dto, dact, dwhy))
        if act == "illegal":
            to = s
        out.append('        {PJob::%s, PEvent::%s, PJob::%s, PAct::%s,' % (s, e, to, act))"""
assert old2 in s
s=s.replace(old2,new2,1)
open(p,"w").write(s)
print("ok")
PY
python3 .shz_cmd/s433_gen_tables.py
