cd /ssd/Strata && python3 - <<'EOF'
from pathlib import Path
p = Path("src/program/generate.cpp")
lines = p.read_text().split("\n")
# block starts at the blank line after `std::string held_line;` (index of line 7695 is 7694)
start = None
for i,l in enumerate(lines):
    if l.strip() == '// ---- S4.2 "hold, don\'t reject" (stage 4, decision D5) -------------------------------':
        start = i; break
assert start is not None
end = None
for i,l in enumerate(lines):
    if i > start and l.strip() == '};' and 'report_waits' in lines[i-9]:
        end = i; break
assert end is not None
block = lines[start-1:end+1]   # include the preceding blank line
print("moving lines", start-1, end, "len", len(block))
print("first:", block[0][:60], "|", block[1][:60])
print("last:", block[-1][:60], "|", block[-2][:60])
EOF
