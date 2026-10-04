cd /ssd/Strata && cp include/strata/program/serve_swap.hpp /tmp/ss.bak && python3 - <<'PYEOF'
import io
p='include/strata/program/serve_swap.hpp'
s=io.open(p,encoding='utf-8').read()
# REVERT 1: the guard step is dropped from the order (the pre-fix behaviour: nothing checks parkability)
old='''    if (p.park_guard && !detail::call(h.park_guard, Step::park_guard, r, err, Fault::park)) return r;
    // 4. validate'''
assert s.count(old)==1
s=s.replace(old,'''    // 4. validate''',1)
io.open(p,'w',encoding='utf-8').write(s)
print('reverted park_guard from run()')
PYEOF
cd /tmp/s3build && ninja serve_swap_test >/dev/null 2>&1; ./serve_swap_test 2>&1 | tail -12
