cd /ssd/Strata && python3 - <<'EOF'
import re
p='src/core/handoff_arena_test.cpp'
s=open(p).read()
# Convert ("label: " + err).c_str()) patterns: we need to restructure check(...) -> check_err(<cond>, err, "label")
# Do it manually per site instead of regex surgery: report the sites.
for m in re.finditer(r'\("([^"]*) \+ err\)\.c_str\(\)\)', s):
    print(m.start(), m.group(1))
EOF
