cd /ssd/Strata && python3 - <<'PY'
p="include/strata/program/prefill_svc.hpp"
s=open(p).read()
h=open(".shz_cmd/handoff_table.inc").read().rstrip("\n")
j=open(".shz_cmd/job_table.inc").read().rstrip("\n")
def swap(name, new):
    global s
    i = s.index(name)
    k = s.index("\n}\n", i) + 2
    s = s[:i] + new + "\n" + s[k:]
swap("inline const Transition* handoff_table(int& count) {", h)
swap("inline const JobTransition* prefill_job_table(int& count) {", j)
open(p,"w").write(s)
print("swapped")
PY
g++ -std=c++20 -O0 -Iinclude -o /tmp/prefill_svc_test src/program/prefill_svc_test.cpp 2>&1 | head -20 && /tmp/prefill_svc_test 2>&1 | tail -40
