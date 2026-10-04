cd /ssd/Strata; grep -n "swap_to(" src/program/generate.cpp; echo "=== plan_prompt_segments ==="; sed -n "/auto plan_prompt_segments = /,/^        };/p" src/program/generate.cpp | head -40
