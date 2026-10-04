nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv 2>/dev/null | head -20; echo ===; nvidia-smi --query-gpu=index,uuid --format=csv 2>/dev/null
