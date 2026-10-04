### 9. The six row ceilings, priced (S4.5-B)

Everything below is read out of the source or computed from constants cited in the source. Two scratch
programs produced the arithmetic: `/tmp/s45scratch/{sizes,arena2,mtparena,gr}.cpp` (`g++ -std=c++20 -O1
-Iinclude`, host-only, no CUDA call), and `/tmp/s45scratch/{shmtest2,mmvqshm2,mmvqreg,abmulti}.cu`
(`nvcc -c -arch=sm_86 -Xptxas -v`, **compile-only, no device, no launch**). Anything not traceable to a
`file:line` or to one of those is labelled **(inference)** or **(guess)**.

Two facts that change the shape of the problem before any number is read:

* **The six ceilings are not the only ones.** The inventory found **seven more hard 8-row limits** that
  §1.2 did not list: `kFusedGrMaxT` (`include/strata/kernels/fused_gr.hpp:48`),
  `bf16_gemv_fp32_mmvf_multi`'s `n_tok > 8` (`src/kernels/cuda/native_bf16.cu:139`),
  `shared_expert_multi`'s `n_tok > 8` (`src/kernels/cuda/shared_expert.cu:172`),
  `kMaxWindowEntries = 128` with `static_assert(cpu::MAXT * 10 <= 128)`
  (`src/core/expert_source.cpp:974-975`), `CAP = cpu::MAXT * 10` (`src/core/remote_experts.cpp:17`),
  `Verifier::groups_[9]` (`include/strata/core/verify.hpp:206`) and `Verifier::last_tokens_[8]`
  (`include/strata/core/verify.hpp:174`). Raising the six and missing these is exactly risk **B1**
  ("raising one and not the others is silent"), and three of them are silent.
* **The cheapest ceiling is also the only one that hard-fails.** `GMAX` at 24 does not cost memory, it
  **stops the build**: `ptxas error : Entry function … uses too much shared data (0x10e00 bytes, 0xc000
  max)` — measured, see §9.2.

#### 9.1 The table

| # | ceiling | file:line | what it enforces today | what breaks at 16 | what breaks at 24 | what breaks at 32 | VRAM cost of raising it | host RAM cost | is the existing T≤8 path bit-identical after the raise? | the minimal safe change |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `kVerifyMaxT = 8` | `include/strata/kernels/verify_kernels.hpp:21` | the three GDN multi wrappers `std::exit(1)` above it (`src/kernels/cuda/verify_kernels.cu:386,403,416`); `Verifier::init` refuses `max_t > kVerifyMaxT` (`src/core/verify.cpp:146`); `MtpDrafter::load` refuses it (`src/core/mtp.cpp:149`); the PLE row buffer `uint32_t rows[kVerifyMaxT * PLE_N_HEADS]` (`src/core/verify.cpp:972`, `PLE_N_HEADS = 16`, `include/strata/kernels/ngram.hpp:35`); the penalty-history and hand-off buffers (`src/program/generate.cpp:3936,3962`); `static_assert(GMAX >= kVerifyMaxT)` (`src/kernels/cuda/s2_expert_grouped.cu:519`) | `gdn_ab_multi_kernel`'s register array `float acc[kVerifyMaxT]` (`verify_kernels.cu:80`) grows: measured 40 → **48 regs** at 16 (`-Xptxas -v`, sm_86, `__launch_bounds__(256)`), 0 spills. `resident_plan_kernel`'s comment "at most kVerifyMaxT * 10 entries" (`:437`) becomes 160 > the plan cap `T*K` — the cap grows with T, so it is consistent, but `moe_group_resident` refuses `n > 128` (`s2_expert_grouped.cu:703`) and `T*K = 160` **exceeds it** | 63 regs (measured, 0 spills). `moe_group_resident`'s `n ≤ 128` guard is now 240 entries — **hard refusal** (`src/core/mtp.cpp:637` passes `n = T*K`), and `group_resident_kernel`'s `__shared__ int e_s[128], first_s[128], size_s[128], gidx_s[128], gstart_s[129]` (`:661`) is a fixed 128-entry array: **out of bounds**, not a refusal | 64 regs (measured, 0 spills). Same two hard failures, worse | **the arena is the whole price**: `Verifier::init`'s `Bump` carve (`src/core/verify.cpp:229-262`) is linear in T. Replicated: 76.943 MiB @ T=6 (the log prints 76.9, `strata-iq3_s.log:34`), 89.042 @ 8, **95.092 @ 9, 113.240 @ 12, 137.438 @ 16, 185.834 @ 24, 234.229 @ 32** MiB at 524 288 cells. Slope **6.021 MiB/row** (computed: 5.441 MiB of per-row buffers + 0.500 MiB `scores_` + 0.080 MiB `hit_scratch_`); fixed part 40.625 MiB = `kStagingBlobs(16) × max_blob(2 662 400)` (`verify.cpp:251`, `verify.hpp:225`, `native_experts.txt` max blob). Plus `MtpDrafter`'s carve (`src/core/mtp.cpp:264-295`): 22.672 @ 8 → **28.116 @ 16, 33.561 @ 24, 39.006 @ 32** MiB (window 32 768, `generate.cpp:413`). Plus `kHistSlots` device buffer `4096·T·4` (`generate.cpp:3936,3940`): 131 072 B @ 8 → 524 288 B @ 32 | mapped staging (`verify.cpp:190-203`) 985 160 B @ 8 → 1 970 024 @ 16 → 3 939 752 @ 32; the `std::vector<int32_t> hist_stage` mirror of `kHistSlots` (`generate.cpp:3938`) 512 KiB @ 32; the layer-split hand-off `kVerifyMaxT × handoff_floats × 4` (`generate.cpp:3962-3963`) 409 728 B @ 8 → 1 638 912 @ 32 (pinned, one per stage boundary) | **NO.** `verify_kernels.cu:80-100` is a `#pragma unroll` over `kVerifyMaxT` inside a T≤8 `if (t >= T) break` — the loop bound *is* the constant, so the compiled kernel for T≤8 changes (register allocation, unroll depth). `verify.cpp:972` is a stack array whose size is the constant. `generate.cpp:3936/3962` change the size of buffers the T≤8 path reads. Three of the four GDN kernels' launch wrappers also change | one PR that raises `kVerifyMaxT` **and** `GMAX`, `MAX_NCOLS`, `cpu::MAXT`, `kFusedGrMaxT`, `kMaxWindowEntries`, `groups_[]`, `last_tokens_[]`, the two `[9]` array families, the `n_tok > 8` guards in `native_bf16.cu:139` / `shared_expert.cu:172`, and `moe_group_resident`'s `n ≤ 128` — i.e. `kBatchMaxRows` as §6 S4.4.4 proposes. Bit-exactness of T≤8 must then be *proven* by the parity binaries, not assumed |
| 2 | `cpu::MAXT = 8` | `include/strata/kernels/cpu/expert.hpp:131` | `ExpertJobMulti::act[MAXT]/out[MAXT]/nact[MAXT]` (`include/strata/kernels/cpu/pool.hpp:61-64`); `ExpertScratchMulti{ActQ a2[MAXT]; float ff[MAXT][FF]}` (`cpu/expert.hpp:147-151`); `SplitBufMulti{ff[MAXT][FF]; a2[MAXT]; hq[MAXT][kNativeHBytes]}` (`pool.hpp:326-330`); per-worker `thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF]` and the `ff/a2/hq/gp/up` pointer arrays (`src/kernels/cpu/pool.cpp:947-989`); `s2_expert_vnni_multi` refuses `n_tokens > MAXT` (`src/kernels/cpu/expert.cpp:613`); `expert_pool_dispatch_multi` refuses `n_tok > MAXT` (`src/core/expert_source.cpp:982`) and sizes `act_multi`/`nact_multi`/`jobs_multi` from it (`:994-999`); `static_assert(MAXT*10 ≤ 128)` (`:975`); `RemoteMeta` arrays `CAP = MAXT*10` (`src/core/remote_experts.cpp:17-25`) | nothing breaks structurally — `jb.nt` is bounded by `n_tok` (`expert_source.cpp:1156-1159`), so the arrays simply get bigger. **But** `MAXT*10 = 160 > kMaxWindowEntries = 128`, so the `static_assert` at `expert_source.cpp:975` is a **compile error** unless `kMaxWindowEntries` is raised in the same change | `MAXT*10 = 240`: the assert fails harder; `d.job_of` is `int16_t` so 240 is still representable; per-worker stack `gbuf+ubuf` = 120 KiB | `MAXT*10 = 320`; per-worker stack `gbuf+ubuf` = 160 KiB | **zero** — this is a CPU-only ceiling. No device allocation reads it | `split_multi_` is `std::vector<SplitBufMulti>` of `kMaxSplitMulti = 96` (`pool.hpp:209`, `pool.cpp:767`): **5 505 024 B @ 8 → 11 010 048 @ 16 → 16 515 072 @ 24 → 22 020 096 @ 32** (sizeof `SplitBufMulti` = 57 344 → 229 376, computed). Plus per-dispatch `act_multi` 28 672 → 114 688 B (`sizeof(ActQ)` = 3 584, computed), `nact_multi` 32 768 → 131 072 B (`kNativeActBytes` = 4096, `cpu/native_expert.hpp:17`), `jobs_multi` 16 640 → 66 560 B (`sizeof(ExpertJobMulti)` = 208, computed), `RemoteMeta` 1 608 → 6 408 B | **YES, with one exception.** `MAXT` appears in no device kernel and in no device allocation. But `kMaxWindowEntries` (`expert_source.cpp:974`) is a *fixed* array `int32_t kind[kMaxWindowEntries]` (`:1014`) that the T≤8 path uses, and the `static_assert` forces it to move with `MAXT` — so either it is raised (which changes a size the T≤8 path reads, though not the values it holds) or the build fails. Strictly: **YES for the compiled code path, NO for "no allocation size the T≤8 path reads changes"** | raise `MAXT` **and** `kMaxWindowEntries` together, and add `static_assert(kMaxWindowEntries >= MAXT*10)` in the same place. Nothing else |
| 3 | `GMAX = 8` | `src/kernels/cuda/s2_expert_grouped.cu:516` | entries per expert group in the two grouped Q2_0 hit kernels: `ne = min(grp_start[g+1] - e0, GMAX)` at `:554` and `:616`; `static_assert(GMAX >= kVerifyMaxT)` at `:519`; sizes `__shared__ int xs_q[GMAX][H/4]` + `float xs_d[GMAX][H/32]` (`:550-551`) and `int hs_q[GMAX][FF/4]` + `float hs_d[GMAX][FF/32]` (`:612-613`) | **no correctness break, an occupancy break.** `gu_grouped_kernel` static shared 23 040 → **46 080 B** (45.00 KiB) — still under sm_86's 48 KiB static limit (measured with `nvcc -Xptxas -v`), but blocks/SM from shared memory falls 5 → **2** (128 KiB/SM ÷ 46 080). `down_grouped_kernel` 5 760 → 11 520 B, blocks/SM 22 → 11 | **COMPILE ERROR.** `ptxas error : Entry function '_Z8gu_probePi' uses too much shared data (0x10e00 bytes, 0xc000 max)` — 69 120 B requested against 49 152 B allowed (measured). The fix is `extern __shared__` + `cudaFuncSetAttribute` + a launch-time dynamic size, which is a signature change to both kernels and to `moe_grouped_s2` (`:709-736`) | **COMPILE ERROR**, 92 160 B (0x16800) against 0xc000. Even with the dynamic opt-in it is 90 KiB of a 100 KiB budget → **1 block/SM** | shared memory is **not VRAM** — it costs occupancy, not bytes. The device-side plan buffers do grow with T, but through `plan_i32_` (`verify.cpp:206-211`, cap = T·K), already counted in row 1's arena. `moe_hit_grouped_scratch_bytes(T·10, 2560, 640)` (`src/kernels/cuda/s2_expert_grouped.cu`, the function at the head of the file) = 464 000 B @ T=8 → 928 000 @ 16 → 1 392 000 @ 24 → 1 856 000 @ 32; `native_expert_scratch_bytes` (`src/kernels/cuda/iq_kernels.cu:706-709`) is larger and dominates the `std::max` at `verify.cpp:254-256`: 84 224 B per 10 hits | none (device-only) | **NO.** `xs_q/xs_d/hs_q/hs_d` are `__shared__` arrays indexed by `k < ne` inside the T≤8 loop (`:585-591`, `:644-646`); their size is a launch-time property of the very kernel a T≤8 window runs. Occupancy changes, and occupancy changes timing, not arithmetic — but the *allocation* the T≤8 launch touches changes. Also: at GMAX < T the `min(..., GMAX)` at `:554`/`:616` **silently drops entries**, which is a correctness cliff, not a cost | raise `GMAX` only together with converting both kernels to `extern __shared__` + `cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, …)` (the pattern `fused_gr.cu:335-341` already uses), and gate the launch grid on `cudaOccupancyMaxActiveBlocksPerMultiprocessor` as risk B2's mitigation says |
| 4 | `MAX_NCOLS = 8` | `src/kernels/cuda/native_mmvq.cu:783` | `validate_shape` throws when `ncols < 1` or `ncols > MAX_NCOLS` (`:1080-1084`), called by every entry point (`:1104,1131,1168,1195,1208,1234,1247,1273,1286,1312,1325,1351,1364,1390`); `launch_multi`'s `switch (ncols)` has cases 2..8 and `throw`s otherwise (`:1064-1078`); `native_q8_1_bytes` throws for `ncols > 8` (`:1147-1150`) — which is why the Appendix-A harness fails loudly instead of lying | **no compile error, no overflow.** 8 new template instantiations × 7 format families = 56 new kernels (see §9.5). Scratch grows linearly: `native_q8_1_bytes(2560, 16)` = 46 080 B vs 23 040 @ 8. `bf16_gemv_fp32_mmvf_multi` (`native_bf16.cu:139`) and `shared_expert_multi` (`shared_expert.cu:172`) still refuse `n_tok > 8`, so the window **cannot reach** ncols = 9 without changing those two guards too | same, plus `__shared__ float partial[NW-1][NCOLS][ROWS][WARP]` (`:1024`) reaches 36 864 B in the small-K branch (ROWS = WARPS = 4) — still under 48 KiB | same; 49 152 B (48.00 KiB) in the small-K branch — **exactly at** the sm_86 static limit, so any additional static shared in that kernel would overflow. Registers in the ROWS = 4 branch: 62 → 95 → 132 → **177** (measured with a stand-in kernel, `-Xptxas -v`, sm_86, `__launch_bounds__(128,1)`) — 177 regs × 128 threads = 22 656 of 65 536 regs/SM, so 2 blocks/SM; **not** a spill (0 spill stores measured) | `xq_ = b.take<uint8_t>(native_q8_1_bytes(max_in, T))` (`verify.cpp:236`), `max_in = max(N, ZV, NH·HD) = 6144`: 55 296 B @ 8 → 110 592 @ 16 → 165 888 @ 24 → 221 184 @ 32. `nat_xq_ = T·(N/32)·36` (`:253`) 92 160 → 368 640 B. `hit_xq_ = T·(N/32)·34` (`:252`) 87 040 → 348 160 B. All inside the arena already priced in row 1 | `mtp.cpp:275` carves `native_q8_1_bytes(NH·HD, 8)` — **hard-coded 8, not T**, so the drafter's scratch does *not* grow (and its `nc = min(8, T·HC - c0)` at `mtp.cpp:554` is a second hard 8) | **YES for the T≤8 code path.** `MAX_NCOLS` is only a validation bound; `launch_multi`'s switch cases 2..8 are unchanged and the compiler emits the same specialisations. The shared/register arrays are per-instantiation, so `NCOLS = 8`'s kernel is untouched by adding `NCOLS = 9..16`. **Caveat:** the *scratch allocation sizes* the T≤8 path reads (`xq_`, `nat_xq_`, `hit_xq_`) grow with T, so a raise that also raises `max_t` is not zero-effect; a raise of `MAX_NCOLS` alone (with `max_t` still 8) is | add `case 9..16:` to `launch_multi` (`:1068-1077`) and raise `MAX_NCOLS`; **never** touch `g_multi_exact` (`:997`, see §9.5). Then raise the two `n_tok > 8` guards in `native_bf16.cu:139` and `shared_expert.cu:172`, which are the actual blockers on the window reaching ncols = 9 |
| 5 | the `[9]` graph arrays | `include/strata/core/verify.hpp:177-178` (`exec_[9]`, `commit_exec_`), `:206` (`groups_[9]`); `include/strata/core/mtp.hpp:141,145,146,172,173,176` (six `[9]` arrays) | one instantiated window graph per T ∈ 1..8 plus one commit graph; `capture(T)` writes `exec_[T]` (`verify.cpp:821-882`) and `run` launches `exec_[T]` (`:991`); `groups_[T]` is written by `record_window` (`:342`) and read by `run` (`:997`) and `commit` (`:1130`); `last_tokens_[8]` (`:174`) is written `for t < T` (`verify.cpp:988`) and read `for t < n_keep` (`verify.cpp:1187`) | **T = 9 writes `exec_[9]`, one slot past the end** — `exec_[9]` is indices 0..8, so index 9 is out of bounds. Same for `groups_[9]` and `last_tokens_[9]`. This is **undefined behaviour, not a refusal**: nothing checks it. `verify.cpp:954` checks `T > max_t_`, and `max_t_` is capped by `kVerifyMaxT` at `init` (`:146`) — so the OOB is only reachable once ceiling 1 is raised, which is exactly risk B1's "exec_[9] indexes out of bounds at T=9" | same | same | **the graphs themselves are opaque and their VRAM is not enumerable from the source** — `cudaGraphInstantiate` allocates driver-side storage whose size is not documented in this repo and cannot be computed without running on a GPU. What *is* computable is the per-shape arena growth (row 1) and the node count: the window graph is one captured graph per layer per group (`verify.cpp:322-793`), so a T = 9..16 graph is the same node count with different kernel arguments. The log shows **59 capture lines** across **12 process starts** (`grep -c "window up to"` = 12), i.e. **4.9 window graphs per process** — only T = 1..6 are ever captured at `--spec 4` (counting `captured the N-token window` per N in the log → 10/10/10/10/9/10 for T = 1..6, **zero** for 7 and 8). So the *instantiated* graph count today is 6 window + 1 commit + the MTP's round/step graphs, not 9+9 | `exec_[9]`/`groups_[9]`/`last_tokens_[8]` are 9·8 + 9·4 + 8·4 = 140 B; `[17]` versions are 17·8 + 17·4 + 17·4 = 272 B. The MTP's six `[9]` arrays are 54 pointers = 432 B → 816 B at `[17]`. **Negligible** | **YES.** Growing an array and adding new indices never changes the code emitted for `T ≤ 8`, and no allocation the T≤8 path reads changes size (the arrays are inline members, not buffers). This is the one ceiling that is genuinely free | `exec_[9] → exec_[kBatchMaxRows+1]`, `groups_[9] → groups_[kBatchMaxRows+1]`, `last_tokens_[8] → last_tokens_[kBatchMaxRows]`, and the six MTP `[9]` arrays likewise; add a `static_assert` or a bounds check at `verify.cpp:822` so a future raise cannot reintroduce the OOB. Capture cost is per-shape and lazy (`capture(T)` returns early if `exec_[T] != nullptr`), so unused shapes cost nothing at runtime |
| 6 | `--spec`'s clamp | `src/program/generate.cpp:1661` (`o.spec = std::min(o.spec + 2, 8)`, comment `// kVerifyMaxT`) | the window depth the engine will actually ask for. `--spec 4` (`strata-iq3_s.json`) → `o.spec = 6`, `o.mtp_max_t = 4` (`:1660`), and the live line is `strata verify: window up to 6 tokens` (`strata-iq3_s.log:34`) | nothing in the code breaks — the clamp is a policy. But `o.spec` **is** `max_t` for both `ver.init(..., o.spec, ...)` (`:4014`) and `mtp.load(..., o.spec, ...)` (`:2519`), so lifting the clamp without lifting ceilings 1-5 turns `Verifier::init`'s refusal (`verify.cpp:146-148`) into a start-up failure with the message "the window must hold 2..8 tokens" | same | same | zero by itself; it is the switch that *unlocks* the arena cost in row 1 (6.021 MiB per row of `max_t`) | zero by itself | **YES.** It is a runtime `std::min` on a CLI value, not a compiled bound. `--spec 4` still yields 6 whatever the clamp is, so the T≤8 path is byte-identical and every allocation is the same size | change the literal `8` to the new ceiling. It must be the **last** thing merged, because it is the only one that makes the other five reachable |

#### 9.2 Shared memory and occupancy in `s2_expert_grouped.cu`

Declarations, per block, with per-entry bytes computed from the constants in the file
(`H = 2560`, `FF = 640` — `include/strata/kernels/cpu/expert.hpp:34-35`; `int`/`float` = 4 B):

| array | file:line | per-entry bytes | GMAX=8 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|
| `__shared__ int xs_q[GMAX][H/4]` | `:550` | 2560 × 4 = **10 240 B** | 20 480 | 40 960 | 61 440 | 81 920 |
| `__shared__ float xs_d[GMAX][H/32]` | `:551` | 80 × 4 = **320 B** | 2 560 | 5 120 | 7 680 | 10 240 |
| **`gu_grouped_kernel` total** | `:543-601` | 10 560 B/entry | **23 040 B (22.50 KiB)** | **46 080 (45.00)** | **69 120 (67.50)** | **92 160 (90.00)** |
| `__shared__ int hs_q[GMAX][FF/4]` | `:612` | 160 × 4 = **640 B** | 5 120 | 10 240 | 15 360 | 20 480 |
| `__shared__ float hs_d[GMAX][FF/32]` | `:613` | 20 × 4 = **80 B** | 640 | 1 280 | 1 920 | 2 560 |
| **`down_grouped_kernel` total** | `:605-651` | 720 B/entry | **5 760 B (5.62 KiB)** | **11 520 (11.25)** | **17 280 (16.88)** | **23 040 (22.50)** |

The batch-decode note's "2 560 B per entry" (`/ssd/Strata/.megamind/src/kernels/batch-decode-notes.md:80`)
is **wrong by 4×**: it counts `xs_q`'s entries as 2 560 B, but `H/4 = 640` **`int`**s = 2 560 *words* =
10 240 B. The correct per-entry figure for `gu_grouped_kernel` is 10 560 B, and GMAX = 16 is 45 KiB, not
41 KiB. Corrected here so nobody re-uses it.

`__launch_bounds__` / launch config, cited:

* `gu_grouped_kernel` and `down_grouped_kernel`: `__launch_bounds__(256)` at `:543` and `:605`;
  launched `<<<dim3(2*FF/GU_ROWS, cap_groups), 256>>>` = `<<<dim3(40, cap_groups), 256>>>` (`:720`) and
  `<<<dim3(H/D_ROWS, cap_groups), 256>>>` = `<<<dim3(40, cap_groups), 256>>>` (`:732`), with
  `GU_ROWS = 32` (`:520`) and `D_ROWS = 64` (`:521`). Grid.x is **independent of GMAX**.
* `group_resident_kernel`: `<<<1, 128>>>` (`:704`), `__shared__ int e_s[128], first_s[128], size_s[128],
  gidx_s[128], gstart_s[129]` = 2 564 B (`:661`) — **fixed at 128 entries, does not scale with GMAX**,
  and `moe_group_resident` refuses `n > 128` (`:703`). At T = 16, `n = T·K = 160 > 128`: hard refusal.
* `gu_kernel`/`down_kernel` (the non-grouped R4 path) have `__shared__ int warp_count[4]` only (`:400`).

RTX 3090 = sm_86 limits: **48 KiB (49 152 B) static per block without opt-in**, up to **100 KiB dynamic
with `cudaFuncSetAttribute`**, **128 KiB per SM**, max 16 blocks/SM, 1536 threads/SM.

Verified by compiling the exact declarations for `sm_86` (`nvcc -c -arch=sm_86 -Xptxas -v`,
`/tmp/s45scratch/shmtest2.cu`):

| GMAX | `gu_grouped_kernel` smem | ptxas | blocks/SM by shared (128 KiB ÷ smem) | blocks/SM by threads (1536 ÷ 256) | **effective occupancy** |
|---|---|---|---|---|---|
| 8 | 23 040 B | `Used 12 registers, 1 barriers, 23040 bytes smem` | 5 | 6 | **5** |
| 16 | 46 080 B | `Used 12 registers, 1 barriers, 46080 bytes smem` | 2 | 6 | **2** |
| 24 | 69 120 B | **`ptxas error : Entry function '_Z8gu_probePi' uses too much shared data (0x10e00 bytes, 0xc000 max)`** | — | — | **does not compile** |
| 32 | 92 160 B | **`ptxas error : … (0x16800 bytes, 0xc000 max)`** | — | — | **does not compile** |

So the exact overflow numbers are **69 120 B against a 49 152 B static limit at GMAX = 24** (0x10e00 vs
0xc000) and **92 160 B at GMAX = 32** (0x16800). `down_grouped_kernel` compiles at all four values
(5 760 / 11 520 / 17 280 / 23 040 B).

Occupancy consequence at GMAX = 16, stated plainly: the gate/up grouped kernel goes from 5 blocks/SM to 2,
i.e. **60 % fewer concurrent blocks**, on a kernel whose whole purpose is to keep the SM busy while it
streams 1.38 MB blobs. That is risk B2, and it is real — but it is a *speed* cliff, and the parity test
can measure it with `cudaOccupancyMaxActiveBlocksPerMultiprocessor` before shipping.

**The hidden one, not in §1.2 at all:** `kFusedGrMaxT = 8` (`include/strata/kernels/fused_gr.hpp:48`) is a
separate 8-row ceiling on the hyper-connection read, and it is the largest shared-memory consumer in the
window:

| | file:line | GMAX/kFusedGrMaxT = 8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| `gr_up_multi_kernel` static `lo[kFusedGrMaxT][LR]` + `g[kFusedGrMaxT][HC][UPM_COLS]` | `src/kernels/cuda/fused_gr.cu:250-251` (`LR = 320`, `UPM_COLS = 16`, `HC = 4`) | 10 240 + 2 048 = **12 288 B (12.00 KiB)** | 24 576 (24.00) | 36 864 (36.00) | 49 152 (**48.00 — exactly the static limit**) |
| `gr_down_multi_kernel` **dynamic** `extern __shared__ float tile[]`, launched `n_tok * TILE * 4` | `:193`, `:363`; `TILE = 2560` (`:185`) | 80 KiB @ T=8 | **160 KiB** | 240 KiB | 320 KiB |
| `GrMulti` kernel-parameter struct `FusedGrArgs a[kFusedGrMaxT]` | `:135-139`; `sizeof(FusedGrArgs) = 112` (computed) | 912 B | 1 804 B | 2 700 B | 3 596 B |

`fused_gr.cu:335-341` already opts in to `kFusedGrMaxT * TILE * 4 = 81 920 B` of dynamic shared and then
**slices the launch to whatever the card allows** (`chunk[dev] = usable / (TILE*4)`, `:352-354`; on sm_86
that is 102 400/10 240 = 10 > 8, so no slicing on a 3090). Raising `kFusedGrMaxT` past 10 therefore
re-introduces the slicing path (`:365-373`) — which the comment at `:344-347` says is safe because the
down kernel's outputs are strictly per-token. So this ceiling is *softer* than `GMAX`, but its static
shared at 32 lands exactly on the 48 KiB wall.

#### 9.3 Graph arrays

Slots that exist today:

| array | file:line | slots |
|---|---|---|
| `Verifier::exec_[9]` | `include/strata/core/verify.hpp:177` | 9 pointers, indices 0..8; index 0 is never used (`capture(T)` is called with T ≥ 1) |
| `Verifier::commit_exec_` | `:178` | 1, shape-independent (it replays `max_t_` rows, `verify.cpp:890,901,915,923`) |
| `Verifier::groups_[9]` | `:206` | 9 ints — not a graph, but indexed by `T` at `:342,997,1130`, so it has the same OOB at T = 9 |
| `Verifier::last_tokens_[8]` | `:174` | 8 int32, written `t < T` (`verify.cpp:988`), read `t < n_keep` (`verify.cpp:1187`) |
| `MtpDrafter::prefill_exec_[9]`, `prefill_dev_exec_[9]`, `round_exec_[9]`, `step_exec_[9]`, `round_exec_c_[9]`, `step_exec_c_[9]` | `include/strata/core/mtp.hpp:172,173,176,141,145,146` | 6 × 9 = 54 pointers |
| **total `cudaGraphExec_t` pointers per `Verifier` + `MtpDrafter`** | | **10 + 54 = 64** (computed) |

Not window-shaped and therefore unaffected: `SessionGraphs::execs/posts/preA/preB/preP[5]`
(`include/strata/core/session.hpp:134-195`) and `TokenGraph::exec` (`:396`) — both off on this box
(native pack, `docs/STAGE4-BATCH-DECODE.md` §1.6).

Capture/upload cost per extra shape. The code path is `Verifier::capture(T)`
(`src/core/verify.cpp:821-882`): `cudaStreamBeginCapture` → `record_window(T, …)` (the whole 48-layer
window, `:322-793`) → `cudaStreamEndCapture` → `cudaGraphInstantiate` (`:871`) → `cudaGraphDestroy`
(`:872`) → `cudaGraphUpload` (`:877`) → `cudaStreamSynchronize` (`:878`) → the log line at `:879`. The
MTP's equivalent is `finish_capture` (`src/core/mtp.cpp:690-706`), which does the same instantiate +
upload + sync. **The wall-clock cost of one capture is not derivable from the source** — the log carries
no timestamps, so `captured the N-token window` lines cannot be timed. What *is* derivable: each extra
shape costs one full re-record of the window (≈ the same node count as an existing shape) plus one
instantiate + upload round trip, and it is paid **once, lazily, on the first window of that size**
(`capture` returns immediately if `exec_[T] != nullptr`, `:822`).

How many shapes the log actually instantiates:

```
grep -c "captured the" strata-iq3_s.log                 = 59
grep -o "captured the [0-9]*-token window" | sort | uniq -c
   10 captured the 1-token window
   10 captured the 2-token window
   10 captured the 3-token window
   10 captured the 4-token window
    9 captured the 5-token window
   10 captured the 6-token window
```

**Zero** captures at T = 7 or 8, and `grep -c "window graph has" = 0` (`STRATA_VERIFY_NODES` was never
set, so no node counts exist in the log). Twelve `window up to` lines (`strata-iq3_s.log:34` and friends)
mean 12 process starts; 59 captures / 12 starts ≈ 4.9 window graphs per process, plus one commit graph
each. So today's box runs **6 window graphs + 1 commit graph per verifier**, not 8 — the two unused
`exec_` slots are already free.

VRAM delta of instantiating T = 9..16/24/32: **not computable from the source.** `cudaGraphInstantiate`
allocates driver-side exec storage whose size is a function of node count and parameter bytes, and this
repo never queries or prints it. What can be said: the *device buffers* the graphs point at are the arena
already priced in row 1 (one arena per verifier, shared by all its `exec_[T]` — `verify.cpp:265` allocates
`arena_` once, and `capture`/`record_window` only carve pointers into it), so **raising the ceiling does
not add one arena per shape; it adds one graph per shape on top of a single arena that grows 6.021
MiB/row.** The graph-exec storage itself is **(guess) tens of KiB per shape** — the window graph holds
O(48 layers × ~30 kernels) nodes (`verify.cpp:322-793`, `kProfPer = 32` stamps per layer at
`verify.hpp:159`), and each node's parameter blob is a few hundred bytes.

#### 9.4 CPU pool

Constants: `FF = 640`, `H = 2560`, `MAXC = H/QKA = 80` (`include/strata/kernels/cpu/expert.hpp:34-40`);
`kNativeHBytes = 1024`, `kNativeActBytes = 4096` (`include/strata/kernels/cpu/native_expert.hpp:17-19`);
`sizeof(ActQ)` = **3 584 B** (alignas(64) `int8_t q[2560]` + 3 × 80 × 4 + `int`, computed with
`g++ -std=c++20`, `/tmp/s45scratch/sizes.cpp`).

| structure | file:line | MAXT=8 | 16 | 24 | 32 |
|---|---|---|---|---|---|
| `ExpertScratchMulti{a2[MAXT]; ff[MAXT][FF]}` | `cpu/expert.hpp:147-151` | 49 152 B | 98 304 | 147 456 | 196 608 |
| `ExpertJobMulti{blob; nt; act[MAXT]; out[MAXT]; nact[MAXT]}` | `cpu/pool.hpp:58-65` | 208 B | 400 | 592 | 784 |
| `SplitBufMulti{ff[MAXT][FF]; a2[MAXT]; hq[MAXT][kNativeHBytes]}` | `cpu/pool.hpp:326-330` | 57 344 B | 114 688 | 172 032 | 229 376 |
| `split_multi_` = `vector<SplitBufMulti>` × `kMaxSplitMulti = 96` | `cpu/pool.hpp:209`, `cpu/pool.cpp:767` | **5 505 024 B (5.25 MiB)** | 11 010 048 | 16 515 072 | **22 020 096 (21.0 MiB)** |
| per-worker `thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF]` | `cpu/pool.cpp:947` | 40 960 B | 81 920 | 122 880 | **163 840 (160 KiB)** |
| per-worker pointer arrays `gp[MAXT] up[MAXT] ff[MAXT] a2[MAXT] hq[MAXT]` | `cpu/pool.cpp:948-988` | 256 B | 512 | 768 | 1 024 |
| `d.act_multi` (`vector<ActQ>`, sized `MAXT`) | `src/core/expert_source.cpp:994` | 28 672 B | 57 344 | 86 016 | 114 688 |
| `d.nact_multi` (`MAXT × kNativeActBytes`) | `:997` | 32 768 B | 65 536 | 98 304 | 131 072 |
| `d.jobs_multi` (`MAXT·k` × `ExpertJobMulti`) | `:999` | 16 640 B | 33 280 | 49 920 | 66 560 |
| `RemoteMeta` (`CAP = MAXT·10`) | `src/core/remote_experts.cpp:17-25` | 1 608 B | 3 208 | 4 808 | 6 408 |

**Thread-stack implication.** The only per-worker *stack* (as opposed to heap) object is
`thread_local float gbuf[MAXT][FF], ubuf[MAXT][FF]` at `pool.cpp:947` — 40 KiB at MAXT = 8, **160 KiB at
32**. `thread_local` on Linux is `.tbss`, not the growable stack, so it does not blow the 8 MiB stack and
does not need `ulimit -s`; but it is 160 KiB × (5 workers + the host thread) = **960 KiB of always-resident
TLS** at MAXT = 32, and it is touched cold on the native-Q2_0 gate/up path (`pool.cpp:945-956`), which on
a 6-core box with 5 GiB available is a real page-cache cost. The pointer arrays (`ff[MAXT]`, `a2[MAXT]`,
`hq[MAXT]`, `gp[MAXT]`, `up[MAXT]`) are genuinely on the stack and are trivial (≤ 1 KiB).

**Is `ExpertJobMulti` per whole window or per expert group?** **Per expert group** — one job per *distinct
expert that missed the VRAM cache*, carrying every window row routed to it. The construction is
`expert_source.cpp:1139-1160`: `int16_t& jo = d.job_of[e]` looks up the job for expert `e`, creating one
if absent (`:1140-1154`), then appends the row: `jb.act[jb.nt] = &d.act_multi[t]; jb.out[jb.nt] = row;
++jb.nt` (`:1156-1159`). So `nt ≤ n_tok ≤ MAXT` and `njobs ≤ n_tok · k` (`jobs_multi` is sized
`MAXT·k`, `:999`). The whole window's misses are then handed to the pool in **one call**:
`d.pool->run_split_multi_native(lay.fmt[d.layers], d.jobs_multi.data(), njobs)` /
`run_split_multi(..., njobs)` (`:1164-1165`).

**Barrier semantics under a raise.** The quoted note at `include/strata/kernels/cpu/pool.hpp:8-12` —
"the host must SUM all ten outputs before the next layer starts, so a layer is a barrier by construction
and the queue never holds more than one batch" — is **unchanged in kind** by raising `MAXT`, but its cost
changes: `run_split_multi` publishes `mtasks_ = 3 · threads` tasks per phase (`pool.cpp:1046`, `1074`)
where `mrows_ = n · FF` then `n · H` (`:1047,1054`), and `n` is the number of *distinct missed experts*,
not the row count. Raising `MAXT` raises the **worst-case `n`** from `8·10 = 80` to `16·10 = 160` /
`32·10 = 320`, and `n > kMaxSplitMulti = 96` (`pool.hpp:209`) sends `run_split_multi` down the
single-token fallback path (`pool.cpp:1028-1041`) — i.e. **at MAXT ≥ 10, a full-miss layer silently stops
using the multi-token kernel at all** and re-quantizes per token. `run_split_multi_native` batches instead
(`pool.cpp:1069-1070`, `for b0 … b0 += kMaxSplitMulti`), so the native pack on this box is fine; the Q2_0
pack is not. The barrier itself stays one-per-layer; what grows is the work inside it, and the union of
two sequences' miss sets is larger than either's alone — which is the §3.5 "(guess) the slope should be
steeper for cross-sequence rows" point, now with a mechanism.

#### 9.5 MMVQ

The instantiation matrix:

* `constexpr int MAX_NCOLS = 8` — `src/kernels/cuda/native_mmvq.cu:783`.
* `template<typename F, int NCOLS, int NW, int ROWS> __global__
  native_mmvq_multi_kernel` — `:999-1043`, `__launch_bounds__(NW * WARP, 1)` at `:1000`.
* `template<typename F, int NCOLS> void launch_multi_n` — `:1045-1062`. Exact branch:
  `dim3 threads(WARP, WARPS)` with `WARPS = 4` (`:45`), and either `<F, NCOLS, WARPS, WARPS>` (small-K,
  `:1058`) or `<F, NCOLS, WARPS, 1>` (normal, `:1060`). Upstream branch: `<F, NCOLS, NW, 2>` with
  `NW = NCOLS <= 4 ? 4 : 2` (`:1049-1053`).
* `template<typename F> void launch_multi` — the runtime switch `case 2..8` + `throw` at `:1064-1078`.
* Format families that reach `launch_multi`: **7** — `Q5KTraits` (`:790`), `Q4KTraits` (`:831`),
  `Q20Traits` (`:869`), `Q3KTraits` (`:901`), `Q6KTraits` (`:927`), `IQ4XSTraits` (`:954`),
  `SmallTraits<Weight,Qi>` (`:984`, used by `small_mmvq` for Q4_0/Q5_0/Q8_0/IQ4_NL, `:1101-1126`).
  So ncols = 9..16 costs **8 × 7 = 56 new kernel instantiations** (× 2 branches each, since both the
  small-K and normal branch are instantiated per `NCOLS`) — **(inference)** for compile time and
  code size; no runtime cost for shapes never launched.
* **The i-quants are a different kernel with no ceiling at all.** `iq_mmvq`
  (`src/kernels/cuda/iq_kernels.cu:633-651`) dispatches types 16/17/18/20/21/22/23/29/42 to
  `mmvq_kernel<TY>` (`:334-347`), which takes `ncols` as a **runtime loop argument**
  (`for (int c = 0; c < ncols; ++c)`, `:343`) and has **no `validate_shape` and no `MAX_NCOLS` check**.
  On this box the window's projections are Q8_0 (`index.txt:3,9,13` — field 10 = 8 for
  `output.weight`, `attn_qkv.weight`, `ffn_gate_shexp.weight`), which routes through
  `native_mmvq` case 8 → `native_q8_0_mmvq` → `small_mmvq<Q80Block,8>` → `launch_multi<SmallTraits>`
  (`native_mmvq.cu:1474-1489`, `:1388-1391`), so **the ceiling does bite here**. But the grouped-expert
  i-quant path (`iq_mmvq`) is already unbounded.

`g_multi_exact`: declared `bool g_multi_exact = true;` at `native_mmvq.cu:997` with the comment "until the
upstream layout is timed on an idle GPU (plan rule: default only what is measured)". Set only by
`native_mmvq_set_multi_exact(bool)` (`:1144`), read by `native_mmvq_multi_exact()` (`:1145`) and by
`launch_multi_n` (`:1049`). **`grep -rn "native_mmvq_set_multi_exact" src include` finds no caller in the
engine** — only the header declaration (`include/strata/kernels/native_mmvq.hpp:29`) and the parity test.
So the exact layout is the only layout the engine ever uses, and it is a process-global read at capture
time.

**Is ncols = 9..32 bit-exact vs ncols = 1 under the exact layout?** **Yes, by construction, and the
construction is `NCOLS`-agnostic.** `launch_multi_n`'s exact branch keeps `NW = WARPS = 4` and
`BPI = F::BPI * NW / WARPS = F::BPI` (`:1004`), i.e. the identical thread-to-block mapping, blocks per
iteration and reduction order as the ncols = 1 kernel; the comment at `:779-782` states "each (column,
row) value is accumulated over kbx in the same order, summed across warps in the same order and reduced
with the same warp tree as the ncols = 1 kernel, so every column is BITWISE equal to a single-column call
on that column", and `include/strata/core/verify.hpp:3-8` makes the same claim for the window. Nothing in
`:1009-1042` makes the arithmetic depend on `NCOLS` — `tmp[j][i]` is a separate accumulator per column.
**Caveat:** the existing proof only covers the widths it tests. `src/kernels/mmvq_multi_parity.cpp:73` is
`const int WIDTHS[] = {1, 2, 3, 4, 5, 6, 8};` — **7 is missing, and 9..16 are untested**. The negative
control (`g_multi_exact = false`) only has power at `T > COINCIDE_MAX_T = 4` (`:74`, `:25-28`), so a new
width must be added to `WIDTHS` before it is trusted.

Scratch growth, as formulas:

```
native_q8_1_bytes(n_in, ncols) = ncols · (n_in / 32) · sizeof(Q81Block)
                               = ncols · (n_in / 32) · 36          (native_mmvq.cu:1147-1150, :55-58)
   n_in = 6144 (max_in):  ncols · 6 912 B   → 55 296 @ 8, 110 592 @ 16, 221 184 @ 32
   n_in = 2560:           ncols · 2 880 B   →  23 040 @ 8,  46 080 @ 16,  92 160 @ 32

moe_hit_grouped_scratch_bytes(n_hits, n_embd, n_ff)
   = align16(n_hits·2·n_ff·4) + align16(n_hits·(n_ff/32)·34)
     + 2·align16(n_hits·(n_ff/32)·4) + align16((n_embd/32)·4)      (s2_expert_grouped.cu, the
     function named in include/strata/kernels/s2_expert_grouped.hpp:33)
   linear in n_hits = T·10: 464 000 B @ T=8, 522 000 @ 9, 696 000 @ 12, 928 000 @ 16,
                             1 392 000 @ 24, 1 856 000 @ 32

native_expert_scratch_bytes(cap, n_ff)
   = 3·align256(cap·n_ff·4) + align256(cap·(n_ff/32)·36)           (iq_kernels.cu:706-709)
   linear in cap = T·10: 84 224 B per 10 hits → 673 792 @ T=8, 2 695 168 @ T=32
   (this is the larger of the two and is what verify.cpp:254-256 actually reserves)
```

`__shared__ float partial[NW-1>0?NW-1:1][NCOLS][ROWS][WARP]` (`native_mmvq.cu:1024`) measured by
compiling the declaration (`/tmp/s45scratch/mmvqshm2.cu`, sm_86): exact branch, normal path (ROWS = 1)
3 072 / 6 144 / 9 216 / 12 288 B at NCOLS = 8/16/24/32; small-K path (ROWS = 4) 12 288 / 24 576 / 36 864
/ **49 152 B** — the last is exactly sm_86's 48 KiB static ceiling, so NCOLS = 32 in the small-K branch
leaves **zero** headroom for any other static shared in that kernel.

#### 9.6 Which ceiling can be raised with ZERO effect on the existing T≤8 path

Strict test: the compiled code path for T≤8 is byte-identical **and** no allocation size that the T≤8
path reads changes.

| ceiling | verdict | reason |
|---|---|---|
| `--spec` clamp (`generate.cpp:1661`) | **YES** | a runtime `std::min` on a CLI value. `--spec 4` still yields `o.spec = 6`; no compiled bound, no allocation size changes. Raising the literal alone changes nothing until some caller asks for more |
| the `[9]` graph arrays (`verify.hpp:177-178,206`; `mtp.hpp:141,145,146,172,173,176`) | **YES** | they are inline member arrays, never heap allocations, and `T ≤ 8` indexes the same slots with the same values. Growing `[9] → [17]` adds slots that are `nullptr` and never captured (`capture` is lazy, `verify.cpp:822`). `last_tokens_[8] → [17]` likewise. Nothing a T≤8 launch reads changes size |
| `MAX_NCOLS` (`native_mmvq.cu:783`) | **YES, alone** | it is only a validation bound plus the `switch` at `:1068-1077`. Adding `case 9..16` adds new template instantiations; `NCOLS = 8`'s specialisation, its shared array and its registers are untouched. **Conditional:** the moment `max_t` also rises, `xq_`/`nat_xq_`/`hit_xq_` (`verify.cpp:236,252,253`) grow, and that is not zero-effect. `MAX_NCOLS` is free only as a standalone change |
| `cpu::MAXT` (`cpu/expert.hpp:131`) | **NO, narrowly** | the compiled T≤8 code path is identical (no device kernel takes `MAXT`), but `MAXT` sizes `split_multi_` (`pool.cpp:767`, 5.5 → 22 MiB), `d.act_multi`/`d.nact_multi`/`d.jobs_multi` (`expert_source.cpp:994-999`) and the per-worker `thread_local gbuf/ubuf` (`pool.cpp:947`) — all of which the T≤8 path reads — and `static_assert(MAXT*10 ≤ kMaxWindowEntries)` (`:975`) makes the build fail at MAXT ≥ 13 unless `kMaxWindowEntries` moves too. It is the cheapest "no": no device effect at all, host RAM only |
| `kVerifyMaxT` (`verify_kernels.hpp:21`) | **NO** | it is a `#pragma unroll` bound inside `gdn_ab_multi_kernel` (`verify_kernels.cu:80-100`) — measured 40 → 48 → 63 → 64 registers at 8/16/24/32 — the size of `uint32_t rows[kVerifyMaxT * PLE_N_HEADS]` on the host stack (`verify.cpp:972`), and the size of `kHistSlots` (`generate.cpp:3936`) and the hand-off buffers (`:3962`), all read at T≤8. It also forces `GMAX ≥ kVerifyMaxT` (`s2_expert_grouped.cu:519`), so raising it *requires* touching the shared-memory cliff |
| `GMAX` (`s2_expert_grouped.cu:516`) | **NO** | `__shared__` sizes are launch-time properties of the kernels a T≤8 window runs (23 040 → 46 080 B, occupancy 5 → 2 blocks/SM), and at 24/32 the file **does not compile** (`ptxas error … 0x10e00 bytes, 0xc000 max`). This is the one ceiling where "cost" is the wrong word — it is a build failure |

**The answer Part D needs:** exactly **three** of the six can move with zero effect on T≤8 — the `--spec`
clamp, the `[9]` graph arrays, and `MAX_NCOLS` (the last only as a standalone change). Of those, the
graph arrays and `MAX_NCOLS` are the only two that are *load-bearing* for a future raise, and both are
pure "add slots / add instantiations" edits with no arithmetic consequence. Everything else must be
proven by the parity binaries.

#### 9.7 The three answers the parent asked for

**(a) The six verdicts** — table in §9.6: clamp **YES**, graph arrays **YES**, `MAX_NCOLS` **YES
(standalone)**, `cpu::MAXT` **NO (host RAM only, cheapest no)**, `kVerifyMaxT` **NO**, `GMAX` **NO**.

**(b) The single most expensive ceiling: `kVerifyMaxT`, at +145.2 MiB of VRAM for T = 8 → 32.**
The verifier arena is 89.042 MiB at T = 8 and 234.229 MiB at T = 32 (replicated from
`src/core/verify.cpp:229-262`, validated at 76.943 vs the log's 76.9 at T = 6, `strata-iq3_s.log:34`),
i.e. **6.021 MiB per extra row**, of which 5.441 MiB is the per-row activation buffers, 0.500 MiB is
`scores_` (`T × max_blocks × 4`, `max_blocks = max_cells/idx_block + 2`, `verify.cpp:177,243`) and 0.080
MiB is `hit_scratch_`. Add the drafter's +16.3 MiB (`mtp.cpp:264-295`: 22.672 → 39.006 MiB) and the
+1.0 MiB of pinned staging + hand-off, and T = 32 costs **≈ 162 MiB of VRAM and ≈ 4.4 MiB of pinned host
RAM per verifier** — against the 457 MiB free on this box (`strata-iq3_s.log:5178`). At the T = 12 the B=2 case actually needs: **+24.2 MiB verifier arena + 2.7 MiB drafter arena = +26.9 MiB
VRAM, +0.73 MiB pinned RAM**. VRAM is not what stops this; the ceilings are.

**(c) What hard-fails rather than merely costing memory:**

1. **`GMAX ≥ 24` — compile error.** `ptxas error : Entry function … uses too much shared data (0x10e00
   bytes, 0xc000 max)` at 24 (69 120 B) and `(0x16800 bytes, 0xc000 max)` at 32 (92 160 B). Measured,
   sm_86. `gu_grouped_kernel` only; `down_grouped_kernel` compiles at all four.
2. **`T ≥ 9` with the `[9]` arrays unraised — out of bounds, silently.** `exec_[T]` (`verify.hpp:177`),
   `groups_[T]` (`:206`) and `last_tokens_[T]` (`:174`) are written at `verify.cpp:822,342,988` with no
   bounds check; `verify.cpp:954` only checks `T > max_t_`. This is UB, not a refusal.
3. **`T ≥ 13` with `cpu::MAXT` unraised — compile error.** `static_assert(cpu::MAXT * 10 <=
   kMaxWindowEntries)` (`src/core/expert_source.cpp:975`), and `MAXT = 13` gives 130 > 128.
4. **`T ≥ 13` on the Q2_0 pack — silent correctness loss.** `run_split_multi` falls back to one
   single-token job per row when `n > kMaxSplitMulti = 96` (`pool.hpp:209`, `pool.cpp:1028-1041`), and
   `n` can reach `T·10`. The native pack batches instead (`pool.cpp:1069-1070`), so this one is
   pack-dependent.
5. **`T ≥ 13` on the MTP resident-group path — hard refusal.** `moe_group_resident` exits on `n > 128`
   (`src/kernels/cuda/s2_expert_grouped.cu:703`), and `group_resident_kernel`'s shared arrays are fixed
   at 128 entries (`:661`). Its only caller passes `n = T * K` (`src/core/mtp.cpp:637`), so T = 13 is the
   wall.
6. **`GMAX < T` — silent data loss, the worst one.** `ne = min(grp_start[g+1] - e0, GMAX)`
   (`s2_expert_grouped.cu:554,616`) drops routed entries without a word. Raising `kVerifyMaxT` without
   raising `GMAX` in the same commit produces wrong tokens, not an error.
7. **`T > 8` at the two unlisted guards — runtime `throw`.** `bf16_gemv_fp32_mmvf_multi`
   (`src/kernels/cuda/native_bf16.cu:139`) and `shared_expert_multi`
   (`src/kernels/cuda/shared_expert.cu:172`) both reject `n_tok > 8`; `fused_gr_read_multi`
   `std::exit(1)`s on `n_tok > kFusedGrMaxT` (`src/kernels/cuda/fused_gr.cu:305`). These are not in
   §1.2's list and would be discovered only at run time.
