cd /ssd/Strata && red() { sed -E 's/<\|[^|>]*\|>/IMTOK/g'; }
sed -n '994,1004p' serve/server.py | red; echo "=== a PASSING test with the same engine+script ==="; grep -n "TaggedMockEngine(self.tok, THINK" serve/test_server.py | red | head; echo; grep -n "def test_the_gate_bounds_in_flight_requests" -A 25 serve/test_server.py | red | head -30
