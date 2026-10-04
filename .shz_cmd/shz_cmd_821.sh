cd /ssd/Strata && grep -n "struct FusedGrArgs" -A 25 include/strata/kernels/fused_gr.hpp | head -35; echo "=== apply usage ==="; grep -rn "apply" src/kernels/cuda/fused_gr*.cu | head -20
