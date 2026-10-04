cd /ssd/Strata; cp /tmp/ssw.bak include/strata/program/serve_swap.hpp; grep -c "MUTANT" include/strata/program/serve_swap.hpp; echo "restored"
# MUTATION 2: make step_gate always allow the step
cp include/strata/program/serve_driver.hpp /tmp/sdrv.bak
python3 - <<'PYEOF'
from pathlib import Path
p = Path("include/strata/program/serve_driver.hpp")
s = p.read_text()
old = "    if (ph == Phase::decode && !resumable) return Gate::end;"
assert s.count(old) == 1
p.write_text(s.replace(old, "    // MUTANT: gate disabled", 1))
PYEOF
cd /tmp/s3build && ninja serve_driver_test >/dev/null 2>&1; ./serve_driver_test 2>&1 | tail -2
