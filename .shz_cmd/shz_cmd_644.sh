cd /ssd/Strata && python3 - <<'EOF'
p = "include/strata/program/prefill_svc.hpp"
s = open(p).read()
h = open(".shz_cmd/handoff_table.inc").read().rstrip("\n")
j = open(".shz_cmd/job_table.inc").read().rstrip("\n")
assert "PLACEHOLDER_HANDOFF_TABLE" in s and "PLACEHOLDER_JOB_TABLE" in s
s = s.replace("PLACEHOLDER_HANDOFF_TABLE", h).replace("PLACEHOLDER_JOB_TABLE", j)
open(p, "w").write(s)
print("ok", len(s.splitlines()))
EOF
