cd /ssd/Strata && python -m pytest "serve/test_server.py::HeldInsteadOfRejected::test_a_full_engine_holds_the_request_instead_of_failing_it" -q 2>&1 | head -60
