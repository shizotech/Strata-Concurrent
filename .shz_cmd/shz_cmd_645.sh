cd /ssd/Strata && g++ -std=c++20 -fsyntax-only -Iinclude -x c++ include/strata/program/prefill_svc.hpp 2>&1 | head -40; echo "rc=$?"
