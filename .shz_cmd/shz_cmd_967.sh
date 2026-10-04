cd /ssd/Strata && sed -n '40,60p' strata-iq3_s.log | cat -A | head -5 | cut -c1-120; echo "=== any timestamps? ==="; head -c 300 strata-iq3_s.log
