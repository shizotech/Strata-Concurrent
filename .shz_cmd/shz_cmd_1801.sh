cd /ssd/Strata && cp /tmp/ss.bak include/strata/program/serve_swap.hpp && cp include/strata/program/serve_driver.hpp /tmp/sd.bak && python3 - <<'PYEOF'
import io
p='include/strata/program/serve_driver.hpp'
s=io.open(p,encoding='utf-8').read()
# REVERT 2: the step gate always says "run" (the pre-fix behaviour: nothing checks the invariant)
old='''inline Gate step_gate(Phase ph, bool resumable) {
    if (!phase_has_work(ph)) return Gate::run;
    if (resumable) return Gate::run;'''
assert s.count(old)==1
s=s.replace(old,'''inline Gate step_gate(Phase ph, bool resumable) {
    if (true) return Gate::run;
    if (resumable) return Gate::run;''',1)
io.open(p,'w',encoding='utf-8').write(s)
print('reverted step_gate')
PYEOF
cd /tmp/s3build && ninja serve_swap_test serve_driver_test 2>&1 | grep error: | head -3; echo "=== swap ==="; ./serve_swap_test 2>&1 | tail -6; echo "=== driver ==="; ./serve_driver_test 2>&1 | tail -10
