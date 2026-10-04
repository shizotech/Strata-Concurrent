cd /ssd/Strata; grep -n "auto park_current = " src/program/generate.cpp; sed -n "/auto park_current = /,/skip parking/p" src/program/generate.cpp | head -40
