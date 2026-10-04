cd /ssd/Strata && for L in 240 271 272 273 274 275 276 277 278 279 280 281 282 283 284 285 286 287 288 289 290; do printf "%4d: " $L; sed -n "${L}p" src/core/mtp.cpp | cut -c1-95; done
