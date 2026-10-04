cd /ssd/Strata && red() { sed -E 's/<\|[^|>]{0,40}\|>/[SPTOK]/g'; }
cat > /tmp/chk3.py <<'EOF'
import sys; sys.path.insert(0,"/ssd/Strata")
from serve import server as S
from serve.test_server import ByteTokenizer, THINK
tok = ByteTokenizer()
print("IM_END ids:", tok.encode(S.IM_END, parse_special=True))
print("THINK ids :", tok.encode(THINK, parse_special=True))
# what stop_ids actually contains, as ids only
svc_ids = set(tok.encode(S.IM_END, parse_special=True))
print("stop id set (ids only):", sorted(svc_ids))
print("THINK[0] in stop ids? ->", tok.encode(THINK, parse_special=True)[0] in svc_ids)
EOF
timeout 60 python /tmp/chk3.py 2>&1 | red
echo "=== the two literals on line 998-999, as lengths only ==="
python3 - <<'EOF'
from pathlib import Path
import re
src = Path("/ssd/Strata/serve/server.py").read_text().split("\n")
for i in (997,998):
    m = re.search(r'encode\("([^"]*)"', src[i])
    if m: print(f"line {i+1}: literal length={len(m.group(1))} startswith_lt={m.group(1).startswith('<')}")
EOF
