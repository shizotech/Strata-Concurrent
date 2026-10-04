cd /ssd/Strata && git status --porcelain | head -30; echo "=== diff stat of tracked files"; git diff --stat | tail -5
