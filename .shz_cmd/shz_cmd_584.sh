cd /ssd/Strata && python3 - <<'EOF'
from pathlib import Path
p = Path("src/program/generate.cpp")
lines = p.read_text().split("\n")
start = next(i for i,l in enumerate(lines) if 'S4.2 "hold, don\'t reject" (stage 4' in l)
anchor = next(i for i,l in enumerate(lines) if 'next_wait_line_ms = now + 2000;' in l)
end = anchor + 1   # the `};` line
assert lines[end].strip() == '};', lines[end]
block = lines[start-1:end+1]
print("block lines", start-1, end, "count", len(block))
print("FIRST:", repr(block[0]))
print("LAST:", repr(block[-1]))
# remove the block (plus the blank line before it is block[0]? no, block[0] is the comment)
del lines[start-1:end+1]
# now find insertion point: right after end_slot's closing `};`
es = next(i for i,l in enumerate(lines) if 'auto end_slot = [&](int64_t id, const std::string& why) {' in l)
# find the closing of end_slot: first line == '            };' after es
close = next(i for i in range(es+1, len(lines)) if lines[i] == '            };')
print("end_slot at", es, "closes at", close)
print("after:", repr(lines[close+1]))
lines[close+1:close+1] = [""] + block
p.write_text("\n".join(lines))
print("OK, new total lines", len(lines))
EOF
