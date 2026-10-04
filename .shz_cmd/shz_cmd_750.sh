cd /ssd/Strata && red() { sed -E 's/<[^>]{0,40}>/[SPTOK]/g; s/<\|[^|>]{0,40}\|>/[SPTOK]/g'; }
grep -nE "THINK|think_open|think_close|_OPEN|_CLOSE" serve/frontend.py | red | head -20
echo "=== test_server THINK definition ==="; grep -n "^THINK" serve/test_server.py | red
