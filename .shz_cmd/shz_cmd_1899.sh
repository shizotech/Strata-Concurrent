pgrep -af "strata --serve" ; echo "=== the two originals ==="; for p in 1663746 1764902; do printf "pid %s: " $p; [ -d /proc/$p ] && echo ALIVE || echo GONE; done
