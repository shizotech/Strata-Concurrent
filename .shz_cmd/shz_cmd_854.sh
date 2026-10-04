cd /tmp/s434scratch && CUDA_VISIBLE_DEVICES=-1 ./kv_only_gate 2>&1 | tail -2; ./carve_parity | tail -6
