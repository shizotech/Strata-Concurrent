cd /ssd/Strata && echo "--- pool.hpp 8-12"; awk 'NR>=8 && NR<=12 {printf "%d: %s\n",NR,$0}' include/strata/kernels/cpu/pool.hpp
echo "--- pool.hpp 301-304"; awk 'NR>=299 && NR<=306 {printf "%d: %s\n",NR,substr($0,1,90)}' include/strata/kernels/cpu/pool.hpp
echo "--- qsa.hpp 96-100"; awk 'NR>=94 && NR<=101 {printf "%d: %s\n",NR,substr($0,1,100)}' include/strata/kernels/qsa.hpp
echo "--- qsa.hpp 112-121"; awk 'NR>=110 && NR<=122 {printf "%d: %s\n",NR,substr($0,1,100)}' include/strata/kernels/qsa.hpp
echo "--- qsa.hpp 149-152"; awk 'NR>=147 && NR<=153 {printf "%d: %s\n",NR,substr($0,1,100)}' include/strata/kernels/qsa.hpp
echo "--- qsa.hpp 176-186"; awk 'NR>=176 && NR<=186 {printf "%d: %s\n",NR,substr($0,1,100)}' include/strata/kernels/qsa.hpp
