cd /ssd/Strata && python -m pytest "serve/test_server.py::HeldInsteadOfRejected" -q --tb=line -p no:randomly 2>&1 | grep -E "^/ssd|^FAILED|passed|failed" | head -20
