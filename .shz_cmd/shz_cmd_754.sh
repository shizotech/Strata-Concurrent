cd /ssd/Strata && red() { sed -E 's/<[^>]{0,40}>/[SPTOK]/g'; }
sed -n '975,1000p' serve/test_server.py | red; echo "=== what script? ==="; sed -n '940,975p' serve/test_server.py | grep -n "Engine(\|script\|ANSWER\|THINK" | red
