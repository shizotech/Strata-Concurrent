cd /ssd/Strata && cp include/strata/program/serve_driver.hpp /tmp/sd.bak && python3 - <<'PYEOF'
import io
p='include/strata/program/serve_driver.hpp'
s=io.open(p,encoding='utf-8').read()
old='''inline Gate step_gate(Phase ph, bool resumable) {
    if (!phase_has_work(ph)) return Gate::run;   // queued/done run no step of this kind
    if (resumable) return Gate::run;'''
assert s.count(old)==1, "no match"
s=s.replace(old,'''inline Gate step_gate(Phase ph, bool resumable) {
    (void) ph; (void) resumable;
    return Gate::run;   // ANTI-VACUITY REVERT: the pre-fix behaviour, nothing checks the invariant''',1)
io.open(p,'w',encoding='utf-8').write(s)
print('reverted step_gate')
PYEOF
cd /tmp/s3build && ninja serve_swap_test serve_driver_test 2>&1 | grep error: | head -3; echo "=== swap ==="; ./serve_swap_test 2>&1 | tail -8; echo "=== driver ==="; ./serve_driver_test 2>&1 | tail -10
