cd /ssd/Strata; grep -n "plan_prompt_segments" src/program/generate.cpp; sed -n "/void plan_prompt_segments/,/^        };/p" src/program/generate.cpp | head -35
