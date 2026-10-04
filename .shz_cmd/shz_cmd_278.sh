cd /ssd/Strata && for L in 1659 1661 2042 2055 3128 3607 3936 3944 3962 4014 4015 7183 7188 7190 7369 7440; do printf "%5d: " $L; sed -n "${L}p" src/program/generate.cpp | cut -c1-100; done
