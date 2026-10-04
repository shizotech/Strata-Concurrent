cd /ssd/Strata && python3 - <<'PYEOF'
from pathlib import Path
p = Path("src/program/generate.cpp")
src = p.read_text()
lines = src.split("\n")
start = next(i for i,l in enumerate(lines) if l.strip().startswith('auto admit_one = [&]() {'))
# the closing `};` of admit_one is the line before the hand-over comment
end = next(i for i,l in enumerate(lines) if i > start and '// ---- the hand-over (§3.1 step 3)' in l)
print("admit_one:", start+1, "to", end, "(exclusive)")
print("closing line:", repr(lines[end-1]))
assert lines[end-1].strip() == '};'
Path("/tmp/admit_old.txt").write_text("\n".join(lines[start:end-1]))
print("old body lines:", end-1-start)
PYEOF
