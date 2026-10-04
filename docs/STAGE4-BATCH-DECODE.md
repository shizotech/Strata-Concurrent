# Stage 4 — can the textgen stage really batch concurrent requests on the GPU?

**Status: investigation + design. No source file was changed to produce this document, no engine, server,
model load or GPU work was started, and `/dev/shm` was read-only.** Every byte figure was produced by
running the engine's own cost functions host-side (`session_bytes`, `qsa_state_bytes`,
`qsa_buffers_bytes`, `moe_buffers_bytes`, `block_buffers_bytes`, `gdn_buffers_bytes`) or by replicating
`Verifier::init`'s arena carve arithmetically from the source; every other number is read out of
`strata-iq3_s.log` (read-only) or is labelled **(guess)**.

It answers the owner's Stage 4 requirement (`prompt.md`, and `.megamind/stage4-plan.md` decision **D4**):

> "The textgen stage itself should implement real batching of concurrent requests on the GPU (not just
> simple swap&park)."

and Stage 3's deferral (`docs/STAGE3-CONCURRENCY.md` §10):

> "A shared KV arena with per-sequence page tables … is the only route to true token-level batching, and
> it requires: one pool instead of 12 per stage, a per-row page table, per-row GDN state, and re-capture
> of every window graph per batch shape."

**Short answer.** True token-level batching is *achievable* on this architecture, and the kernels are
much closer to it than Stage 3 assumed — the attention, indexer and selection kernels already take
`n_q` queries with **per-query step records and per-query selections**, and the page table is already an
indirection. What is genuinely single-sequence is narrower and deeper than Stage 3 said: the **GDN
recurrence**, the **indexer key store**, the **conv history**, and the **`n_tok ≤ 8` ceiling that the
MTP drafter's own chain already consumes**. B=2 costs ≈ 180 MiB of VRAM and ≈ 6.2 GiB of pinned RAM
*before* the expert cache pays for it, and the expert cache is the thing that makes decode fast. And
the arithmetic of the log says **B=2 is worth about +11 % to +13 % decode throughput on this box, not
+100 %** (§3.5): the window's fixed cost amortises, but its rows are already spent on drafts. The
recommended first step is a **two sessions, one engine thread** design (option **D2**, §3.4): it is the only route
that is bit-exact by construction, it costs 1.47 GiB of VRAM, and it is the experiment that prices the
rest. The honest verdict on the owner's ask is that **N sessions per GPU + one shared prefill instance
gets most of the concurrency win for a fraction of the risk**, and S4.4 should be sequenced after S4.3
and only taken past step 1 if the two-session experiment shows the rows, not the host, are the limit.

---

## 0. What this document had to establish first: the box is not what Stage 3 described

Stage 3's whole VRAM argument rests on **457 MiB of free VRAM** (`docs/STAGE3-CONCURRENCY.md` §2.3,
risk R1). That number is real, but it is the free VRAM of a **layer-split** configuration, and the box
is not running one any more.

| fact | evidence |
|---|---|
| Two `strata --serve` processes are live, one per GPU: pid 12182 on GPU 2 (`--max-context 524288`, yarn 2) and pid 12236 on GPU 0 (`--max-context 262144`, rope none). | `ps`, `nvidia-smi --query-compute-apps` (22 380 MiB and 22 382 MiB) |
| **Neither runs a layer split.** `grep -c "layer split" strata-iq3_s.log` = **0** over the whole 8 700-line log, and both command lines carry no `--layer-split`. | `ps`, log |
| GPU 1 is nearly idle: 2 396 MiB used, all of it a non-Strata process (`whisper_venv/main.py`, pid 12143, 2 386 MiB). | `nvidia-smi` |
| Each Strata instance therefore owns **one whole GPU**: `expert cache 7777 slots, 14.75 GiB of VRAM` (524288 ctx) / `8019 slots, 15.20 GiB` (262144 ctx), and **457 / 455 MiB free with everything loaded**. | `strata-iq3_s.log:5163`, `:5178`, `:5197`, `:5231` |
| Host: 94 GiB total, **5 GiB available**; `/dev/shm` 70 GiB, 24 GiB free (46.84 GiB expert arena). | `free -g`, `df -h /dev/shm` |

Two consequences that change every number below:

1. **The per-stage arithmetic in Stage 3 §2.2 (0.554/0.560/0.547 GiB) is not what this box pays.** A
   single-GPU session arena at 524288/int8/resident 32768 is **1.390 GiB** (measured, §A.1), and the
   three-stage sum (1.661 GiB) is a layer-split artefact.
2. **There is a whole free GPU.** `docs/MULTI_GPU.md:70-79` already documents what one extra card costs
   ("every card: a copy of the dense weights ~3.4 GB, its own session state, its verify window and its
   prompt-path buffers, and an expert cache for its layers"), and `docs/SECOND_GPU.md` is the older
   helper-card experiment. The cheapest "batching" available on this machine is not a kernel change at
   all — it is putting a second decode session on GPU 1. Priced in §4.3.

Everything in this document is given as a **formula first, this box second**, per the constraint that the
owner will add RAM.

---

## 1. The inventory: what today assumes ONE sequence per GPU pass

### 1.1 The verify window is one sequence at consecutive positions

`include/strata/core/verify.hpp:3-8` states the contract:

> "T tokens at consecutive positions p0 .. p0+T-1 - the last accepted token and T-1 drafts - go through
> all 48 layers in ONE captured graph, and the head's argmax is produced for every one of them."

The consecutive-position assumption is not a comment, it is written into the staging:

| what | file:line | the assumption |
|---|---|---|
| `Verifier::run(int T, const int32_t* tokens, int64_t pos0, …)` | `include/strata/core/verify.hpp:69` | one `pos0` for the whole window |
| per-row step record | `src/core/verify.cpp:964` | `qsa_step_fill(h_step_ + t * kStepCount, pos0 + t, s)` |
| per-row RoPE positions (q / kv / indexer) | `src/core/verify.cpp:965-969` | `h_pos_[…] = pos0 + t` for all three head groups |
| the context bound | `src/core/verify.cpp:957` | `pos0 + T > ss.qsa_states[ss.qsa_primary()].max_cells` → error |
| commit positions | `src/core/verify.cpp:1179` | `h_commit_[2 + t] = t < n_keep ? last_pos0_ + t : -1` |
| the sampler's draw counter | `src/core/verify.cpp:1087` | `sp.counter = (uint64_t) pos0` — one counter for the window |
| penalty rows | `src/core/verify.cpp` + `include/strata/core/verify.hpp:78-88`, `strata/kernels/sampler.hpp:36-38` | row t's history follows *this sequence's* drafts 1..t |

**Important correction to Stage 3.** The per-row machinery already exists. `h_step_` is `T × kStepCount`
and `h_pos_` is `T × (n_head + n_head_kv + idx_q_heads)`
(`include/strata/core/verify.hpp:182-183`, allocated `src/core/verify.cpp:191-192`), and
every QSA kernel reads its row's record from device memory (`include/strata/kernels/qsa.hpp:138-173`).
So "the window carries one position" is a **host-side policy in `run()`**, not a kernel limitation. The
kernels would accept `pos0 + t` replaced by `pos_of_row(t)` today.

### 1.2 `kVerifyMaxT = 8` — where it is actually baked

Stage 3 cited `src/kernels/cuda/verify_kernels.cu:386,403,415`. **Verified, and the line numbers are
right** for the three launch wrappers:

| kernel wrapper | file:line | the bound |
|---|---|---|
| `gdn_conv_l2_multi` | `src/kernels/cuda/verify_kernels.cu:386` | `n_tok < 1 || n_tok > kVerifyMaxT` → `std::exit(1)` |
| `gdn_ab_multi` | `:403` | same |
| `gdn_step_norm_multi` | `:415-416` | same |

The constant itself is `include/strata/kernels/verify_kernels.hpp:21` (`inline constexpr int
kVerifyMaxT = 8`). The **full** set of places the ceiling is baked, which Stage 3's list did not cover:

| place | file:line | why it binds |
|---|---|---|
| `Verifier::init` refuses `max_t > kVerifyMaxT` | `src/core/verify.cpp:146-147` | and also `max_t > cpu::MAXT` |
| `cpu::MAXT = 8` | `include/strata/kernels/cpu/expert.hpp:131` | the CPU pool's multi-token job: `ExpertJobMulti::act[MAXT]`, `out[MAXT]`, `nact[MAXT]` (`cpu/pool.hpp:61-64`), and `alignas(64) float ff[MAXT][FF]` on every worker's stack (`cpu/pool.cpp:947-987`, `pool.hpp:327-329`) |
| `GMAX = 8` in the grouped hit kernel | `src/kernels/cuda/s2_expert_grouped.cu:516-519` | `static_assert(GMAX >= kVerifyMaxT)`; `ne = min(grp_start[g+1] - e0, GMAX)` at `:554`, `:616` — **a longer window silently drops entries** |
| `native_mmvq` `MAX_NCOLS = 8` | `src/kernels/cuda/native_mmvq.cu:783`, dispatch `:1066-1075`, `validate_shape` `:1084` | every dense projection in the window is one multi-column MMVQ over the window's rows; **there is no ncols = 9** |
| the layer-split hand-off buffer | `src/program/generate.cpp:3962` | `kVerifyMaxT * Verifier::handoff_floats(g) * sizeof(float)` |
| the penalty-history device buffer | `src/program/generate.cpp:3936` | `kPenaltyWindowCap * kVerifyMaxT` |
| `Verifier::last_tokens_[8]` | `include/strata/core/verify.hpp:174` | fixed 8 |
| `exec_[9]` / `commit_exec_` | `include/strata/core/verify.hpp:177-178` | one graph per T ∈ 1..8 |
| `MtpDrafter` refuses `max_t > kVerifyMaxT` | `src/core/mtp.cpp:149` | and its six graph arrays are `[9]` (`include/strata/core/mtp.hpp:109,113,114,136,137,140`) |
| `--spec` is clamped to 8 | `src/program/generate.cpp:1659-1661` | `o.spec = std::min(o.spec + 2, 8)` |

**The consequence nobody wrote down before:** the window's row budget is **already spent by one
sequence**. `--spec 4` (`strata-iq3_s.json`) becomes `o.spec = 6` at `generate.cpp:1661`, and the live
line is `strata verify: window up to 6 tokens, 76.9 MiB of device buffers` (`strata-iq3_s.log:34`,
`:5176`). Row 0 is the last accepted token and rows 1..5 are drafts
(`src/program/generate.cpp:7176-7197`). **There are exactly 2 spare rows per window.** B=2 sequences at
`--spec 4` needs T = 12 — over `kVerifyMaxT`, over `MAX_NCOLS`, over `GMAX`, over `cpu::MAXT`, and over
every graph array. That is the real cost of "B=2 in one window", and it is a *fan-out* cost across six
independent 8-limits, not one constant.

### 1.3 `QsaState` — how many pools, per session, per layer

`include/strata/core/layer.hpp:203-261` is one QSA layer's persistent state **for one sequence**:

| field | line | what it is |
|---|---|---|
| `k_pool` / `v_pool` (fp16) or `k_q`/`v_q`/`k_scale`/`v_scale` (int8) or `k_q4`/`v_q4` | `:204-218` | the VRAM KV slots, layout `[page][kv_head][page_size][head_dim]` |
| `page_table` (`n_pages`), `n_pages`, `max_cells` | `:220-222` | logical page → physical page, `-1` = not resident |
| `kv_mode`, `n_slots`, `host`, `map` | `:224-231` | 0 = all in VRAM (identity table), 1 = streamed, 2 = ring |
| `idx_tail`, `idx_dead`, `idx_pooled`, `idx_block_pos` | `:234-237` | the indexer's per-sequence key store |
| `cos_tab`, `sin_tab` | `:239-240` | the RoPE table — **shared across the 12 layers** (`:263-265`) |
| `step`, `attention_status`, `pos_dev` | `:244-248` | the per-token counts in device memory |
| `host_step`, `host_pos` | `:257-260` | pinned staging at fixed addresses, baked into the graphs |

How many: `SessionState::qsa_states` is **one `QsaState` per QSA layer in the session's layer range**
(`include/strata/core/session.hpp:36`, `:55-58`), allocated at `src/core/session.cpp:105-116`:
`first + (qsa_alloc - 1) * rest` from one arena, with state 0 owning the RoPE table the others borrow.
For the full 48-layer range that is **12 states** (layers 3, 7, … 47; `include/strata/core/layout.hpp:56-57`).

**The session carve matters here.** A stage's session does **not** allocate all 12 states:
`include/strata/core/session.hpp:49-61` ("THE LAYER-RANGE CARVE (multi-GPU)") keeps **global ordinal
indexing** with fewer rows, so `qsa_alloc`/`gdn_alloc` are per-range and every consumer subtracts
`gdn_ord0` / reads `qsa_states[qsa_primary()]` for the shared RoPE table. That is why option (a)'s
"one pool per stage instead of 12" is the right unit of work: a stage already *thinks* in ranges, and
`session_bytes(g, cells, k, lo, hi)` already prices one (`src/core/session.cpp:53-72`). Measured for
this box's three-stage split: 0.554 / 0.560 / 0.547 GiB — but **the box is not split** (§0), so today
there is exactly one session with all 12 states. `session_bytes`'s own arithmetic is the same carve
(`src/core/session.cpp:56-66`).

Per-layer cost, measured (§A.1), int8 + `--kv-resident 32768` at `--max-context 524288`:

* the first state (owns RoPE): **225.7 MiB**; the other eleven: **97.7 MiB** each;
* of the 97.7 MiB: KV pool 8.3 MiB, indexer pooled rows 4.0 MiB, page table 0.50 MiB, stream map 0.16 MiB,
  and the rest is the RoPE table (128 MiB, once) plus small fixed pieces.

**The indexer key store is the part that is per-sequence and per-layer and has no page table at all.**
`idx_pooled` is `(max_cells/idx_block + 2, idx_dim)` fp32 (`layer.hpp:236`, `src/core/layer.cpp:539`,
`:604`) — at 524288 that is 131 074 rows × 128 × 4 B = **64.0 MiB per layer**, i.e. **128 B per token per
layer, 1.5 KiB/token over 12 layers** (`include/strata/kernels/qsa.hpp:83-84` says exactly this: "128 B/token/layer =
1.5 KiB/token over the 12 QSA layers, i.e. 50 MiB at 32K"). It is indexed by **cell index / idx_block**, not through `page_table`. A second sequence in
one session therefore needs a second `idx_pooled` per layer, and every kernel that reads it
(`qsa_block_scores`, `native_qsa_indexer_append`) needs to be told which one.

### 1.4 GDN / recurrent state — one state per sequence, and it is the hard one

`include/strata/core/session.hpp:33-34`: `GdnBuffers gdn` is shared scratch, `float* gdn_state` is
`(n_gdn_layers, gdn_state_floats)` — **the state is per layer, and there is one row per sequence**.
Carved at `src/core/session.cpp:98`; `gdn_point_at` (`src/core/session.cpp:158-164`) points
`s.gdn.state`/`conv_state` at layer `l`'s slice, and `verify.cpp:456-457` and `:909-910` do the same
arithmetic inline for the window and the commit.

Size, measured: `gdn_state_floats = ssm_state_size × ssm_v_heads × ssm_state_size +
ssm_conv_channels × (ssm_d_conv - 1)` = 128·48·128 + 10240·3 = **817 152 floats = 3.117 MiB per layer
per sequence**; 36 layers = **112.2 MiB per sequence**. (Stage 3's "117 669 888 B = 112 MiB" matches.)

**What per-row state would cost** (§A.1):

| B (sequences sharing one session) | GDN state, whole model | stage [0,16) | [16,34) | [34,48) |
|---|---|---|---|---|
| 1 | 112.2 MiB | 37.4 | 46.8 | 28.1 |
| 2 | 224.4 MiB | 74.8 | 93.5 | 56.1 |
| 4 | 448.9 MiB | 149.6 | 187.0 | 112.2 |
| 8 | 897.8 MiB | 299.2 | 374.1 | 224.4 |

Formula: `B × 3 268 608 B × n_gdn_layers_in_range`.

But VRAM is the *cheap* half. The expensive half is that **the GDN kernels are the only ones in the
window that carry a serial dependency across rows**:

* `gdn_step_norm_multi_kernel` (`src/kernels/cuda/verify_kernels.cu:115-181`) loads the state into
  registers (`:134-138`), runs tokens `0..n-1` **in order** with `__syncthreads()` barriers between them
  (`:139-176`), and writes the state back only when `n_keep != nullptr` (`:177-180`). One `blockIdx.x`
  = one v-head = one state slice.
* `gdn_conv_l2_multi_kernel` (`:25-51`) reads `hist[c*3 + …]` — the **conv history is per sequence** and
  the window of token `t` is `[hist(3) | qkv_0..qkv_t]`.

Two sequences in one launch means two independent recurrences in one block. That is not a signature
change, it is a **reduction-layout change**: `red[RG][S]`, `sk`, `sq`, `wsum` (`:123-125`) are all
per-block shared state for one sequence. Doubling them halves occupancy on a 3090 (the block is already
`S*RG = 512` threads, `__launch_bounds__(S * RG)`).

**The good news, and it is the most important finding in this document:** the kernel already has the
per-row shape. `gdn_step_norm_multi` takes `n_keep` and `t_out_begin`; the split window
(`--spec-split`) already runs the recurrence over rows `[0, te)` while emitting outputs only for
`[tb, te)` (`src/core/verify.cpp:473-475`, `tb_`/`te_` at `:341`, and the split-window note at `:313-321`). A per-row state pointer array
(`float* state_of[b]`) plus a `blockIdx.z = b` would be a *local* change to this one kernel — the
arithmetic per row is untouched, so it is bit-exact by construction.

### 1.5 The KV buffers and the `--kv-resident` / `kv_stream` path

`include/strata/kernels/kv_stream.hpp:3-16` is the design (note: the path is
`kernels/`, not `core/` — the Stage 4 brief's citation `include/strata/core/kv_stream.hpp` does not exist):

> "A streamed QSA layer keeps its AUTHORITATIVE K/V in pinned, device-mapped host memory, laid out
> exactly as a fully resident pool with the identity page table would be. VRAM holds a pool of `n_slots`
> pages (one page = `page_size` = 4 cells = one indexer block, all KV heads, K and V: 4,224 B in int8),
> and the page table becomes a RESIDENCY MAP: `page_table[block]` is the slot holding that block, or -1."

Sizes and layout: `src/core/layer.cpp:489-521` (`KvPlan`, `kv_pool_bytes`), `:529-545`
(`qsa_state_bytes`), `:547-679` (`qsa_state_init`), the host copy at `:624-654`
(`cudaHostAlloc` + `cudaHostGetDevicePointer`, `g_kv_host_bytes` accounting), and the map arrays at
`:593-600`. `qsa_kv_resident_min() = 20480` at `:526` — "one verify window's selections (8 queries ×
2,051 cells in whole blocks) must fit at once, with room to spare" (`include/strata/core/layer.hpp:273-275`).

Measured (§A.1): one layer's VRAM pool at `resident = 32768` cells is **8.3 MiB** (int8), and its pinned
host copy at 524288 cells is **0.516 GiB** → **6.19 GiB for 12 layers**, which is exactly the live line
`KV streaming: 32768 of 524288 cells per QSA layer in VRAM, the K/V in 6.19 GiB of pinned RAM`
(`strata-iq3_s.log:5156`).

**Why `--kv k8v4` refuses streaming** — `src/core/layer.cpp:557-562`:

```cpp
if (g_kv_hybrid && ring_cells <= 0) {
    if (p.mode == 1) {
        std::fprintf(stderr, "strata: hybrid K8V4 KV does not support --kv-resident streaming\n");
        return 0;
    }
    st.kv_hybrid = true; …
```

The reason is in the struct comment (`include/strata/core/layer.hpp:216-219`): hybrid K8V4 is "Mode 0
only (no KV streaming, no ring)", and `qsa_kv_format` (`layer.hpp:289-297`) **hard-exits** if a hybrid
state ever reaches the block movers, because `kv_stream.cu` moves whole blocks in one of three known
layouts (`kKvF16`/`kKvInt8`/`kKvQ4`, `kv_stream.hpp:46`) and K8V4 is a fourth (INT8 K + rotated Q4_0 V,
816 B/cell, `layer.cpp:513-516`). Under k8v4 the whole K/V is therefore in VRAM: **13.00 GiB fp16 /
7.19 GiB int8 per session at 524288** (§A.1), which is why Stage 3's risk R12 says "refuse
`--serve-slots > 1` when `kv_mode == 0`". **Any batching plan must state what it does under k8v4.**
Answer: nothing that fits — see §4.7.

### 1.6 CUDA graphs and every baked-in pointer

Stage 3 §5.1 is the starting list. Verified and extended:

| object | what it bakes | file:line | live on this box? |
|---|---|---|---|
| `Verifier::exec_[9]`, `commit_exec_` | `ss_` (`SessionState*`) and its own `arena_`; every device pointer carved from it | `include/strata/core/verify.hpp:177-178`; `src/core/verify.cpp:821-882` (capture), `:884-948` (commit capture), arena at `:265` | **yes** — one `Verifier` per process today (`generate.cpp:3944`, `:4014`) |
| the window graph's per-layer state pointers | `ss.gdn_state + (gi - ss.gdn_ord0) * gdn_floats` (`:456`), `ss.qsa_states[qi]` (`:482`), `st.page_table` (`:522-534`), `st.idx_*` (`:536-540`), `st.cos_tab/sin_tab` (`:496`), `st.map` via `qsa_kv_resolve` (`:578`) | `src/core/verify.cpp` | **yes** |
| pinned staging the graph reads *from* | `h_tok_/h_step_/h_pos_/h_commit_/h_ple_/h_out_/h_x_/h_ids_/h_w_/h_ymiss_/h_plan_` | `src/core/verify.cpp:190-203`, `:211` | **yes** |
| the doorbell flags | `m_seq_/m_flag_/m_flagA_/m_flagB_` | `src/core/verify.cpp:199-202`; `include/strata/core/verify.hpp:190-193` | **yes** |
| `VerifyHits::d_res`, `cache_base`, `slot_off` | the residency table and the VRAM expert arena base | `include/strata/core/verify.hpp:46-50`; bound at `generate.cpp:3945-3950`, `:3990-3995` | **yes** |
| the layer-split hand-off buffers | mapped pinned, sized `kVerifyMaxT × handoff_floats` | `generate.cpp:3962-3972` | no (no split running) |
| `MtpDrafter::prefill_exec_[9]`, `prefill_dev_exec_[9]`, `round_exec_[9]`, `step_exec_[9]`, `round_exec_c_[9]`, `step_exec_c_[9]` | `ss_`, `st_` (its own `QsaState`), `window_R_` (the verifier's `final_R_all()`) | `include/strata/core/mtp.hpp:109,113,114,136,137,140`; bound at `generate.cpp:4015` | **yes** |
| `MtpDrafter::ident_` | a **precomputed identity selection** `T × cap_` int32 (`cap_` = the window rounded to 64) | `src/core/mtp.cpp:240`, `:283-288`, `:499` | **yes** — 6 × 32 832 × 4 B = 789 KiB, and it is a *dense* selection over the drafter's ring |
| `SessionGraphs::execs/posts/preA/preB/preP[5]`, `parts_dev` | `ss`, `parts` | `include/strata/core/session.hpp:133-196`, `:229-231`; captured at `generate.cpp:3128` guarded by `!native_pack` | **no** (native pack) |
| `TokenGraph::exec`, `y_src` | `ss`, `parts_dev`, the pinned `y_miss` | `include/strata/core/session.hpp:395-405`; `generate.cpp:3607`, guarded by `!multi_gpu` | **no** |
| `KvStreamMap` | fixed device addresses, read by `kv_stream_resolve` **inside** the graphs | `include/strata/kernels/kv_stream.hpp:49-59`; `layer.cpp:589-600` | **yes** |
| `QsaState::host_step`, `host_pos` | pinned staging per QSA state, captured as the memcpy **source** | `include/strata/core/layer.hpp:257-260`; `layer.cpp:617-623`; `session.cpp:168-180` (`stage_token`) | yes (single-token path) |
| `mrope_table_set(d_mrope)` | a **process-global device table pointer** consulted by every rope kernel | `include/strata/kernels/mrope.hpp:19-22`; `generate.cpp:2042-2055` | yes (vision on) |
| `cvec()` | device tables + a device on/off flag, consulted inside the window | `include/strata/kernels/cvec.hpp:19-35`; `verify.cpp:735-737` | only if `--control-vector-scaled` |
| `Prefill` | **nothing** — no `cudaStreamBeginCapture` anywhere in `src/prefill/prefill.cpp` | grep; `include/strata/prefill/prefill.hpp:110-118` | n/a |

The last row is the one that keeps mattering: **the prompt path is not graphed**, so a prefill and a
decode window can be in flight at the same time on different streams. That is why §4.3 (two sessions)
is cheap and why Stage 3 chose overlap over batching.

### 1.7 `llama_batch` — and the honest statement about it

**Strata does not use `llama_batch` anywhere.** `grep -rn "llama_batch" src include` returns nothing.
The engine links only ggml (`CMakeLists.txt:716-758`, `third_party/ggml/VERSION.txt` = llama.cpp
`3cf03257f219afbe7334045ff7c6a06ac68c627d`), and llama.cpp's own kernels are used in exactly two places:
the MMQ expert kernels in the prompt path (`src/prefill/moe_mmq.cu:1-2`) and the arithmetic of the
native adapters (`src/kernels/cuda/native_qsa_score.cu:1`, `native_mmvq.cu:1`, `native_router.cu:1`,
`native_gdn.cu:1`, `native_flash_attn.cu:1`, `native_ple_postops.cu:1`, `native_gr_*.cu:1`,
`src/kernels/cpu/native_expert.cpp:3`).

The struct, for reference (`third_party/llama.cpp/include/llama.h:263-271`):

```c
typedef struct llama_batch {
    int32_t n_tokens;
    llama_token  * token;
    float        * embd;
    llama_pos    * pos;
    int32_t      * n_seq_id;
    llama_seq_id ** seq_id;
    int8_t       * logits;
} llama_batch;
```

**What would have to change to carry N sequences here.** Strata's equivalent of `llama_batch` is the
window's pinned staging block, and it is already per-row for everything except identity:

| `llama_batch` field | Strata's analogue | per-row already? |
|---|---|---|
| `token[]` | `h_tok_` (`include/strata/core/verify.hpp:181`, allocated `src/core/verify.cpp:190`) | yes |
| `pos[]` | `h_pos_` (`include/strata/core/verify.hpp:183`, allocated `src/core/verify.cpp:192`, filled `:965-969`) | yes in form, `pos0 + t` in policy |
| `n_seq_id[]` / `seq_id[][]` | **nothing** | no — there is no sequence id anywhere in the window |
| `logits[]` (which rows produce output) | every row always produces an argmax (`verify.cpp:789`) | n/a |

So the missing piece is a **row → sequence map**: `int32_t h_seq_of_[T]`, and every place that today
derives "which state / which pool / which page table / which conv history" from *the session* must
derive it from `seq_of[t]`. That is one new pinned array plus the changes in §1.3/§1.4/§1.8. It is not
a `llama_batch` port; llama.cpp's `seq_id` is a *set membership* list for a shared cell pool, and
Strata's need is a *row index into a small array of state sets*.

### 1.8 Attention / indexer kernels: per-token vs per-sequence

This is where the inventory **contradicts** Stage 3's assumption, and it is the reason S4.4 is not
impossible.

**Already multi-query, with per-query state (no change needed to accept N sequences' rows):**

| kernel | file:line | evidence it is per-query |
|---|---|---|
| `qsa_decode_attn_batch(q, pools, ids, steps, cap, s, scratch, attn, n_q, …)` | `include/strata/kernels/qsa_decode_attn.hpp:41-44` | "`n_q` queries at once, **each with its own selection**: q [n_q, n_head, 256], ids [n_q, cap], steps [n_q, kStepCount]" |
| `qsa_block_scores(pooled, dead, q_idx, steps, nq, max_blocks, …)` | `include/strata/kernels/qsa_select.hpp:25-29` | "one warp per **(query, block)**"; `steps [nq, kStepCount]` |
| `qsa_block_topk(scores, steps, nq, max_blocks, cap, …)` | `:37-38` | "one block per query"; `ids [nq, cap]` |
| `kv_stream_resolve(m, slots, host, fmt, ids, steps, n_q, cap, …)` | `include/strata/kernels/kv_stream.hpp:68-73` | "Make every block named by **the selections of `n_q` queries** resident" |
| `kv_append_*_step(…, page_table, step, …)` | `include/strata/kernels/kv_q8.hpp:29-32`, `include/strata/kernels/qsa.hpp:365` | position and page come from `step[]` (device), so per-row |
| `native_mmvq(…, ncols)` | `include/strata/kernels/native_mmvq.hpp:26`, `src/kernels/cuda/native_mmvq.cu:1066-1075` | multi-column, ncols ≤ 8 |
| `bf16_gemv_fp32_mmvf_multi`, `native_router_top10_multi`, `shared_expert_multi`, `native_moe_combine_multi`, `native_quantize_q8_1(…, ncols)` | `src/core/verify.cpp:503`, `:551`, `:610-612`, `:643`, `:724`, `:462` | all take `n` rows |
| `embedding_gather_dev(…, n_tok, …)`, `broadcast_streams(…, n_tok, …)` | `include/strata/kernels/verify_kernels.hpp:61-65`; `verify.cpp:363-364`, `:376-377` | T rows |
| `fused_gr_read_multi` over `FusedGrArgs fa[kFusedGrMaxT]` | `src/core/verify.cpp:426-438` | one arg struct per row |
| `ple_block` per row, `ngram_rows` per row | `src/core/verify.cpp:410-423`, `:971-979` | per row, host-hashed |
| `sample_tokens(logits, n_tokens, …)` | `include/strata/kernels/sampler.hpp:39` | one output per row |

**Bound to one sequence (these are the actual work):**

| kernel / structure | file:line | what binds it |
|---|---|---|
| `gdn_conv_l2_multi_kernel` | `src/kernels/cuda/verify_kernels.cu:25-51` | one `hist` (conv history) per launch; `n_tok ≤ 8` at `:386` |
| `gdn_step_norm_multi_kernel` | `:115-181` | one `state` per launch, serial over tokens, `n_tok ≤ 8` at `:415` |
| `gdn_ab_multi_kernel` | `:70-113` | `acc[kVerifyMaxT]` register array, `n_tok ≤ 8` at `:403` |
| `QsaState.idx_pooled` / `idx_tail` / `idx_dead` | `include/strata/core/layer.hpp:234-237` | indexed by **cell index**, no page table — `qsa.hpp:197` calls it "WHERE THE INDEXER'S **PER-SEQUENCE** STATE LIVES. One set per QSA layer per sequence" |
| `native_qsa_indexer_append(raw, relative_pos_device, pos_base, …)` | `include/strata/kernels/native_qsa_indexer.hpp:18-27`, `:38-42` | "The sequence is **text, contiguous**, with nonnegative `pos_base` divisible by four"; `max_cells` is a fixed capacity |
| `Verifier::record_window`'s `tail_snap_` snapshot/restore | `src/core/verify.cpp:518`, `:921`, `:923-926` | one tail per QSA layer per window |
| `resident_plan_kernel` | `src/kernels/cuda/verify_kernels.cu:433-473` | one thread, `n ≤ max_t_ * k` entries; comment at `:437` "at most kVerifyMaxT * 10" |
| `moe_grouped_s2` / `native_expert_grouped` `GMAX = 8` | `src/kernels/cuda/s2_expert_grouped.cu:516-519`, `:554`, `:616` | entries per group capped at 8 |
| `ExpertJobMulti` | `include/strata/kernels/cpu/pool.hpp:56-64` | `act[MAXT]`, `out[MAXT]`, `nact[MAXT]` |
| `MtpDrafter::ident_` + `window_ids` | `src/core/mtp.cpp:240`, `:499`; `src/kernels/cuda/verify_kernels.cu:361-377` | a dense sliding window over **one** sequence's cells |

**Does any kernel already tolerate non-contiguous positions?** Yes, and it is worth stating precisely
because it is the crux:

* **Attention: yes.** `qsa_decode_attn_batch` reads `ids[q*cap + j]` through `pools.page_table`
  (`qsa_decode_attn.hpp:5-14`, `qsa.hpp:68-74`: "the physical row for logical cell `t` and head `h` is
  `page = table[t / page_size]` … **Nothing outside this header may assume the mapping**"). The
  selection ids are data. A row whose ids come from a different sequence's page table already works.
* **Indexer scores/top-k: yes.** `steps [nq, kStepCount]` — each query's `pos`, `n_kv`, `n_bid`, `width`
  are independent (`qsa_select.hpp:16`, `qsa.hpp:149-151`, `:156-162`).
* **KV append: yes.** position comes from `step[kStepPos]` and the page from `page_table`
  (`kv_q8.hpp:29-32`, `qsa.hpp:365`).
* **Indexer key append: no.** `native_qsa_indexer_append` documents the contiguity requirement
  explicitly (`native_qsa_indexer.hpp:18-27`), and `idx_pooled` has no indirection.
* **GDN: no.** No position at all — its whole per-token input is `x` and what advances is the state
  (`include/strata/core/layer.hpp:507-509`).

So the honest inventory summary is: **attention and selection are already batched; the recurrence and
the indexer key store are not; and the ceiling is 8 rows, all of which one sequence's drafts use.**

---

## 2. The cost model, measured host-side

All numbers in this section came from one scratch binary (`/tmp/s41_scratch/costs.cpp`, recipe in
Appendix A) that calls the engine's own cost functions and replicates `Verifier::init`'s arena carve
(`src/core/verify.cpp:224-263`) arithmetically. **It never calls `session_init` or `qsa_state_init`**
(both `cudaHostAlloc`), and it makes no CUDA call at all.

### 2.1 Sanity: the harness reproduces Stage 3's published numbers

| config | this harness | Stage 3 §2.2 |
|---|---|---|
| 524288, fp16, resident 0 | **13 959 779 840 B = 13.001 GiB** | 13 959 779 840 B = 13.00 GiB ✓ |
| 524288, int8, resident 0 | 7 718 656 256 B = 7.189 GiB | 1.753 GiB ✗ (see below) |
| 524288, int8, resident 32768 | **1 492 082 688 B = 1.390 GiB** | 1 492 082 688 B = 1.390 GiB ✓ |
| stage [0,16) / [16,34) / [34,48) | 0.554 / 0.560 / 0.547 GiB | 0.554 / 0.560 / 0.547 ✓ |
| 131072 / 65536 / 32768 / 8192, int8, r32768 | 0.728 / 0.617 / 0.560 / 0.229 GiB | 0.728 / 0.617 / 0.560 / 0.229 ✓ |

The one disagreement: Stage 3 §2.2 lists "`--max-context 524288`, `--kv int8`, `--kv-resident 0` →
1.753 GiB". That is wrong — with `--kv-resident 0` nothing streams, so the int8 pools are fully
resident: **7.189 GiB** (measured; 12 × 720.5/592.5 MiB states). 1.753 GiB is close to the *resident-32768*
pool total. Flagged so nobody re-uses it.

### 2.2 The verifier arena, and how it grows with the window

Replicating `Verifier::init`'s `Bump` carve (`verify.cpp:60-68`, 256-byte alignment per region):

| `max_t` | device arena | mapped host staging |
|---|---|---|
| 1 | 48.4 MiB | 0.12 MiB |
| 2 | 54.1 MiB | 0.24 |
| 4 | 65.4 MiB | 0.48 |
| **6** | **76.8 MiB** | 0.71 |
| 8 | 88.2 MiB | 0.95 |

The live box prints `strata verify: window up to 6 tokens, 76.9 MiB of device buffers`
(`strata-iq3_s.log:34`, `:5176`; 75.4 MiB at 262144 ctx, `:5229`). **The model is within 0.1 MiB, so it
extrapolates.** The arena is linear in T: `arena(T) ≈ 42.7 + 5.69·T MiB` at 524288 cells (the intercept
is the `nG`/`nQ`-per-layer `qkv_L_`, `h_L_`, `gate_L_`, `beta_L_`, `idx_raw_L_` and `tail_snap_` blocks
plus the fixed `staging_` 16 blobs; the slope is everything else).

**This is the number that prices "B=2 in one window":** T = 12 would be ≈ 111 MiB per verifier — but T
cannot exceed 8 without breaking `MAX_NCOLS`, `GMAX`, `cpu::MAXT` and the `[9]` graph arrays (§1.2).

### 2.3 KV, indexer and page-table bytes

Per QSA layer, int8 (`kv_q8_bytes_per_cell` = 2·2·256 + 2·2·(256/64) = 1 056 B/cell,
`include/strata/kernels/kv_q8.hpp:24-27`):

| cells | VRAM pool (one layer) | ×12 |
|---|---|---|
| 524288 (fully resident) | 132.0 MiB | 1.547 GiB |
| 32768 (`--kv-resident`) | **8.3 MiB** | 99.0 MiB |
| 8192 | 2.1 MiB | 24.4 MiB |

Pinned host copy (the authoritative K/V, `layer.cpp:624-654`): 0.516 GiB/layer at 524288 → **6.19 GiB**
for 12 layers; 3.09 GiB at 262144 (matches `strata-iq3_s.log:5190`).

Per-sequence, per-layer, **not shared** (measured):

| piece | bytes at 524288 |
|---|---|
| page table (`n_pages` × 4) | 0.50 MiB |
| `kv_stream` map arrays (5 ints × `n_slots` + ctl) | 0.16 MiB |
| `idx_pooled` (`resident/idx_block + 2` rows × 128 × 4) | 4.00 MiB (resident 32768) / 64.0 MiB (524288 pooled rows) |
| **total per extra sequence, 12 layers** | **55.9 MiB** (resident 32768) |

RoPE table: 128.0 MiB at 524288, **shared by all 12 layers** (`layer.hpp:263-265`, `session.cpp:105`).

### 2.4 What one extra *decode session* costs (the number §4.3 needs)

A second session is `session_bytes` plus its own verifier plus its own drafter state:

| piece | bytes | source |
|---|---|---|
| session arena (48 layers, 524288, int8, r32768) | **1.390 GiB** | `session_bytes`, §A.1 |
| verifier arena (max_t 6) | 76.8 MiB | §2.2 |
| MTP draft-layer K/V ring | `qsa_state_bytes(g, max_cells, false, window + 4·max_t + 64)` ≈ **35 MiB** | `src/core/mtp.cpp:213-221`; Stage 3 §2.2 measured 35.1 MB |
| MTP buffers + experts + head | 836 MiB + 81 MiB | `strata-iq3_s.log:5158`, `:5177` — **experts (675 MiB) could be shared, buffers/graphs could not** |
| pinned host KV copy | **6.19 GiB of RAM** | `strata-iq3_s.log:5156` |
| **VRAM total, second session, experts shared** | **≈ 1.50 GiB** | |
| **VRAM total, second session, experts not shared** | **≈ 2.36 GiB** | |
| **pinned RAM total** | **≈ 6.2 GiB** | |

---

## 3. The options, priced

Every route below is priced in three columns: **VRAM**, **host RAM**, **engineering risk**. Formulas
first.

### 3.1 (a) Shared KV arena + per-sequence page tables (llama.cpp's model), one pool per stage instead of 12

**Shape.** One KV arena per (stage, layer) holding `n_slots` pages, and a per-sequence page table that
maps *that sequence's* logical block → arena slot. The arena is shared; the tables are not.

**VRAM.** The arena does not grow with B at all if `n_slots` stays the same — today's 12 × 8.3 MiB =
99.0 MiB per session already holds 32 768 cells per layer. What grows is the **per-sequence metadata**:

```
per extra sequence = 12 layers × (page_table + kv_stream map + idx_pooled + tail + dead)
                   = 12 × (0.50 + 0.16 + 4.00 + 0.00 + 0.0005) MiB
                   = 55.9 MiB            (measured, §2.3)
```

So B=2 costs **+55.9 MiB**, B=4 **+168 MiB**. That is the *cheap* option.

But sharing the arena means the **clock-sweep eviction policy becomes cross-sequence**:
`kv_stream_resolve` picks one victim per miss with a second-chance clock that never takes a slot this
call uses (`kv_stream.hpp:11-15`). With two sequences in one call, a block sequence A needs can be
evicted by sequence B's resolve *in the same launch*. That is not wrong (the host copy is authoritative,
`:16-17`: "Writers write the host copy always, and the slot only if the block is resident"), but it
turns the 94.2 % VRAM hit rate (`strata-iq3_s.log:8484`) into a function of B. **Pricing that is a
measurement, not arithmetic** — at 94.19 % hit and 1 002 470 MiB read from RAM over the log's life, a
few points of hit rate is PCIe traffic the 2.9 GB/s link on GPU 0's slot cannot absorb
(`strata-iq3_s.log:5184` shows a 2.9 GB/s probe on one card against 24.7 GB/s on the other, `:5148`).

**Host RAM.** The authoritative copies must be per-sequence too, or per-sequence *ranges* of one shared
host arena. Today it is `12 × 0.516 GiB = 6.19 GiB` for a 524288 session. llama.cpp's model (a shared
cell pool sized by total tokens, not by `max_context`) is the only way this does not multiply:

```
pinned host bytes = Σ over sequences of (tokens_in_that_sequence × 1056 B × 12 layers)
                  ≈ 12.7 KiB/token/12-layers  →  12 672 B/token   (kv_q8.hpp:8-9)
```

At 200 000 tokens per conversation and B=2 that is **4.7 GiB**, versus 12.4 GiB for two full
`max_context` copies. **This is the one part of option (a) that is a genuine win, and it is a RAM win,
not a speed win.**

**Risk: high.** It rewrites `kv_stream.cu`'s resolve (cross-sequence clock), `QsaState` (a table array
instead of one table), every `kv_append_*`/`kv_gather_*`/`qsa_decode_attn_*` call site in
`verify.cpp:519-535`, `:580-582`, and the snapshot machinery (`conversation_snapshot.cpp`'s
host-pool save/restore, `docs/STAGE3-CONCURRENCY.md` §2.4). It does **not** by itself let two sequences
share a window — attention still needs `n_q` rows whose `ids` index different tables, which means
`QsaAttnPools::page_table` (`qsa_decode_attn.hpp:32`) must become per-row.

### 3.2 (b) Per-row GDN state + a window that carries B sequences

**Shape.** `gdn_state` becomes `(B, n_gdn_layers, gdn_state_floats)`; the three GDN multi kernels take a
row index; the window's rows are `[seq0's drafts | seq1's drafts | …]`.

**VRAM.** `B × 3 268 608 B × n_gdn_layers` (§1.4): **+112.2 MiB per extra sequence** on one GPU. Plus
the per-sequence QSA state of (a): **+55.9 MiB**. Plus the verifier arena growth if T grows (§2.2).

**Host RAM.** Every parked conversation image grows by 112.2 MiB of GDN state, and the pinned KV copy
multiplies by B unless (a)'s shared host arena is also done: **+6.19 GiB per extra live sequence** at
524288.

**Risk: very high.** Four independent 8-ceilings must be raised together (§1.2), and each raise is a
kernel change:

* `native_mmvq` `MAX_NCOLS = 8` → new template instantiations for 9..16 (`native_mmvq.cu:1066-1075`).
  The multi-column kernels are **bitwise equal to ncols = 1 only under `g_multi_exact = true`**
  (`native_mmvq.cu:995-997`: the upstream layout "is equal to ncols = 1 only to float rounding"). Raising
  the ceiling past 8 forces a new layout decision, and that decision is a bit-exactness decision.
* `s2_expert_grouped.cu:516-519` `GMAX = 8` with a `static_assert` and a **silent `min(…, GMAX)`** at
  `:554`/`:616`. Raising it grows `__shared__ int xs_q[GMAX][H/4]` (2 560 B per entry) — at GMAX=16 that
  is 41 KB of shared memory per block, over the 3090's 48 KB budget with anything else live. **This one
  can hard-fail occupancy, not just correctness.**
* `cpu::MAXT = 8` (`cpu/expert.hpp:131`) sizes per-worker stack arrays `ff[MAXT][FF]`
  (`pool.cpp:947`, `pool.hpp:327`) — 8 × 640 floats each, so 16 doubles it.
* `Verifier::exec_[9]` and the MTP's six `[9]` arrays (`verify.hpp:177`, `mtp.hpp:109-140`) become `[17]`,
  i.e. **twice as many instantiated graphs per stage**, each with its own `cudaGraphInstantiate` +
  `cudaGraphUpload` stall at first use (`verify.cpp:871-880`).

And the GDN kernel change itself (§1.4): doubling `sk`/`sq`/`red`/`wsum` shared memory in
`gdn_step_norm_multi_kernel` (`:123-125`) at `__launch_bounds__(512)`.

**Payoff.** This is the only option that actually reaches the owner's sentence. Its payoff is bounded by
the acceptance-rate arithmetic in §3.5.

### 3.3 (c) Per-slot CUDA graphs — re-priced

Stage 3 §5.2 (A) refused this "against 457 MiB of free VRAM". Re-priced with the correct box:

* A second `Verifier` costs its arena: **76.8 MiB** (§2.2), plus 0.71 MiB of mapped host staging.
* A second `MtpDrafter` costs its buffers + K/V ring but **not** its experts: 836 − 675 = 161 MiB +
  35 MiB ring + 81 MiB draft head ≈ **277 MiB**.
* A second session arena: **1.390 GiB**.
* Total: **≈ 1.74 GiB** against **457 MiB free** on a running single-GPU instance.

**Still refused on a running box.** But the framing changes: the refusal was never about the graphs, it
was about the session arena. The graphs themselves are 354 MiB — and 354 MiB is **256 expert-cache
slots** (blob 1 382 400 B, `strata-iq3_s.log:5163`), i.e. **3.3 % of a 7 777-slot cache**. The expert
hit rate is 85-94 % (`strata-iq3_s.log:8357`, `:8484`), and the engine warns that residency changes the
rounding (`:5165-5168`). So per-slot graphs are affordable in VRAM if the cache pays; **what kills them
is the 6.19 GiB of pinned RAM per session and the capture stall**, not the graph memory.

### 3.4 (d) B=2 sequences in one verify window vs two serial windows — the smallest increment

Two sub-variants, and they are very different pieces of work.

**D1 — B=2 inside today's T ≤ 8.** `--spec 3` per sequence → T = 2·(3+1) = 8 (row 0 + 3 drafts each,
clamped at 8 by `generate.cpp:1661`). No kernel ceiling changes. Requires: per-row `pos`/`step`
(already there, §1.1), per-row sequence id (new), per-row GDN state (§3.2's kernel change), per-row
page table + indexer store (§3.1), per-row conv history, per-row PLE history, per-row penalty history
(already `kVerifyMaxT` rows, `verify.hpp:78-88`), and a per-row `n_keep` in the commit graph
(`verify.cpp:901`, `:1177-1179` — today one `n_keep` for the window).

**D2 — B=2 sessions, one window graph each, two launches.** Two `SessionState` arenas, two verifier
arenas, and the engine thread issues `ver[0].run(T0, …)` then `ver[1].run(T1, …)` on the same stream (or
two streams). **No kernel changes at all.** Every kernel call is exactly what it is today, on its own
session's pointers.

Be clear about what D2 does and does not buy. It does **not** amortise the window — each sequence still
pays a full 66 ms window for its ~4.3 tokens, so per-request decode speed is unchanged. What it removes
is **the hand-over**: today, two slots means a `conversation_snapshot_save` + `restore` per turn
(237 MB-3.18 GB; the log's own numbers are `parked 153983 tokens in 410.0 ms`, `strata-iq3_s.log:5142`,
and `swap_ms=23498` accumulated over 119 swaps, `:5058`). D2 makes the hand-over **zero**, because both
sessions are already mounted. That is the whole of its win, and it is a latency-and-fairness win rather
than a throughput one — plus it is the experiment that measures the 8.4 ms/row slope for D1 (§3.5).

**Costs.**

| | D1 (B=2 in one window) | D2 (2 sessions, 2 windows) |
|---|---|---|
| VRAM | +112.2 (GDN) +55.9 (QSA per-seq) +11.4 (arena T=6→8) ≈ **+180 MiB**, and the ceilings force T ≤ 8 so `--spec` must drop to 3 | **+1.390 GiB** (session) **+76.8 MiB** (verifier) ≈ **+1.47 GiB** |
| pinned host RAM | +6.19 GiB (second authoritative KV) unless (a) is done | **+6.19 GiB** |
| CPU pool | unchanged (one pool, one window) | unchanged (one pool, two windows in sequence) |
| expert cache | must give back ~130 slots (1.7 %) | must give back ~1 150 slots (14.8 %) |
| kernel changes | GDN ×3, mmvq ncols, GMAX, cpu::MAXT, graph arrays, commit, indexer | **none** |
| bit-exactness | must be *proven* (§5) | **exact by construction** |
| risk | very high | low |

### 3.5 What B=2 is actually worth — the arithmetic that decides everything

Three independent measurements, and they give a *marginal cost per window row*, which is the number
that prices B=2:

1. **A T=1 window costs 24 ms.** `prompt 9691 tokens = … 1 generated in 24 ms (42.2 tok/s)`
   (`strata-iq3_s.log:106`; again at `:233`). The first window of a request is always T=1
   (`src/program/generate.cpp:7180` `if (first_window) T = 1;`), so 24 ms is a clean one-row window
   including its commit. *(This is the figure Stage 3 §3.2 called "~24 ms predicted per decode window"
   and cited to `strata-iq3_s.log:27`; the citation is stale — line 27 of the current log is
   `R4 hit path ON` — but the number is real and it is a measurement, not a prediction.)*
2. **A T=6 window costs ≈ 66 ms.** `3305 generated in 50 085 ms (66.0 tok/s), drafts accepted 1959 of
   2944` (`strata-iq3_s.log:8578`): 0.665 accepted per draft, T = 6 ⇒ `1 + 5·0.665 ≈ 4.3` tokens per
   window ⇒ **≈ 66 ms per window**. Across the log's decode lines the rate is 49.2-73.7 tok/s
   (`:8578-8674`), mean ≈ 61 tok/s ⇒ 16.4 ms/token.
3. **Expert residency.** `decode expert cache hit rate: 85.3 % … 94.2 %` (`:8357-8484`): **5-15 % of
   routed expert rows are computed on the CPU pool**, at 1 382 400 B per blob over the shared arena.

**The window does amortise — measurably.** 24 ms for 1 row, 66 ms for 6 rows ⇒ the marginal cost of a
row is `(66 − 24) / 5 ≈ 8.4 ms`, and a row costs 11 ms on average. So:

| window | rows | predicted ms | tokens at 0.665 acceptance | ms/token | vs today |
|---|---|---|---|---|---|
| today: 1 sequence, `--spec 4` → T=6 | 6 | 66 | 4.3 | 15.3 | — |
| B=2, `--spec 3` each → T=8 | 8 | 66 + 2·8.4 = **83** | 2·(1+3·0.665) = 6.0 | **13.8** | **+11 %** |
| B=2, `--spec 4` each → T=12 (needs the ceilings raised) | 12 | 24 + 11·8.4 = **116** | 2·4.3 = 8.6 | **13.5** | **+13 %** |
| B=4, `--spec 1` each → T=8 | 8 | 83 | 4·1.67 = 6.7 | 12.4 | **+23 %** |

That is the honest shape of the prize: **B=2 is worth roughly +11 % to +13 % decode throughput on this
box**, and it is worth more only if acceptance stays high at the shorter per-sequence draft depth
(`--spec 3` instead of 4), which the log's 0.665 rate does not support. The rows that would have been
drafts become second-sequence rows, and a sequence's tokens per window falls.

**Why the amortisation is real but the win is small.** The window's fixed cost (48 layers of launches,
the doorbell round trips, the commit, the head) is 24 ms and does not grow with rows; the per-row cost
is 8.4 ms and does. Batching converts *fixed* cost into *shared* cost — that is the whole mechanism, and
it works. What it does **not** do is create rows for free: today's 6 rows are 1 accepted token + 5
drafts, and those 5 drafts are what make a window emit 4.3 tokens instead of 1. Give 2 of the 8 rows to
a second sequence and the first sequence drafts less well. The net is +11 %, not +33 %.

Two things could make it better, and both are measurable rather than speculative:

* **The CPU expert term does not amortise.** The pool is a single-batch barrier machine ("the host must
  SUM all ten outputs before the next layer starts, so a layer is a barrier by construction",
  `include/strata/kernels/cpu/pool.hpp:8-12`), it has **5 workers on 6 physical cores**
  (`strata-iq3_s.log:5171`), and `ExpertJobMulti` already carries all `nt ≤ 8` tokens of one group. Two
  sequences' miss *sets* are larger than either's alone, so the 8.4 ms/row slope should be slightly
  **steeper** for cross-sequence rows than for same-sequence draft rows. **(guess)**
* **The PCIe term does not amortise either.** `KV streaming: 94.20 % of 4 287 278 748 block reads hit
  VRAM, 1 002 470.2 MiB read from RAM` (`strata-iq3_s.log:8484`), and one of the two live instances
  probed its link at **2.9 GB/s** (`:5184`) against 24.7 GB/s on the other (`:5148`). A second
  sequence's selections are a second set of blocks to resolve
  (`kv_stream_resolve`'s capacity note, `kv_stream.hpp:69-70`: "`n_slots` must hold the distinct blocks
  of one call (n_q × (cap / page_size + 2))"), which is exactly the term that overflows `ctl[3]`.

### 3.6 (e) Cheaper tricks the inventory reveals

| trick | what the inventory says | price | verdict |
|---|---|---|---|
| **Batch only the MoE router + shared expert** | already done: `bf16_gemv_fp32_mmvf_multi` + `native_router_top10_multi` run the window's rows in 2 launches (`verify.cpp:605-613`), `shared_expert_multi` in one (`:643`), `native_moe_combine_multi` in one (`:724`) | free | **already exploited**; it is why the window amortises at all |
| **Batch only the indexer** | `qsa_block_scores`/`qsa_block_topk` already take `nq` (`qsa_select.hpp:25-38`) and are called with `n` (`verify.cpp:572-575`) | free | **already exploited** |
| **Batch only attention** | `qsa_decode_attn_batch` already takes `n_q` with per-query selections (`qsa_decode_attn.hpp:41-44`) | free | **already exploited** |
| **Batch the CPU expert pool across two windows** | the pool is one barrier machine with 5 workers (`pool.hpp:8-12`, `:301-304`); Stage 3 §4.3 forbids two clients | a real queue | **do not**: it is the stage-1 collapse (`avoid: .megamind/src/kernels/cpu/pool-notes.md`) |
| **`--spec-split` (two token groups per window)** | already exists: `groups_[T]`, `set_split` (`verify.hpp:122`, `:205`), the ordering documented at `verify.cpp:313-321`, `tb_`/`te_` at `:341` | free | **the existing mechanism for overlapping the pool with the GPU** — but it is **opt-in and off by default** (`bool spec_split = false`, `generate.cpp:427`, whose own comment reads "the overlap study: exact, ~7 % slower"; `--spec-split` at `:1308`) |
| **Raise `--decode-tokens` / `--starve-ms` instead** | S3.10 made a turn a token budget (`docs/STAGE3-CONCURRENCY.md` §S3.10); the box runs `--decode-tokens 15` | free | **this is the cheapest concurrency win that exists** and it is already on |
| **One session per GPU, shared prefill** | §1.6: prefill is not graphed; `docs/MULTI_GPU.md:70-79` prices a card | +1 GPU | **§4.3 — the recommended answer to the owner's ask** |

### 3.7 `--kv k8v4`

Under k8v4 the K/V cannot stream (`layer.cpp:557-562`), so a session's KV is fully resident:
**7.19 GiB int8 / 13.00 GiB fp16 at 524288** (§2.1). A second session does not exist on any card.
**Every option in this document is unavailable under k8v4 except (e)'s last row** (a separate GPU per
session, and even then only at a short context). S4.4 must refuse, at start-up, with both numbers —
Stage 3's R12 mitigation, extended.

---

## 4. The verdict, in the order the parent needs it

### 4.1 Is true token-level batching achievable?

**Yes, technically.** The kernels are closer than Stage 3 assumed: attention, selection, the indexer
scoring, the KV append/resolve, the projections, the router, the shared expert and the combine are all
already multi-row with per-row records (§1.8). The blockers are exactly four, and they are named:

1. the GDN recurrence + conv history (§1.4) — one state, one serial chain, per launch;
2. the indexer key store `idx_pooled`/`idx_tail`/`idx_dead` (§1.3) — indexed by cell, no indirection,
   and `native_qsa_indexer_append` documents contiguity (`native_qsa_indexer.hpp:18-27`);
3. the **8-row ceiling**, which is six independent ceilings (§1.2), and which one sequence's drafts
   already consume at `--spec 4`;
4. the per-sequence pinned host KV copy (6.19 GiB at 524288), which is a RAM decision, not a code one.

### 4.2 At what cost

B=2, one window, on a single-GPU instance: **≈ 180 MiB of VRAM** (which the expert cache must give
back: ~130 slots, 1.7 % of 7 777) **+ 6.2 GiB of pinned host RAM** + **changes in six kernel/library
contracts** + **a bit-exactness proof that has to cover the GDN reduction layout, the MMVQ multi-column
layout and the grouped-expert shared-memory ceiling**. B=4 multiplies the RAM and the risk; the payoff
only reaches **+23 %** and only by dropping to `--spec 1`, i.e. by giving up drafting almost entirely
(§3.5's table). The mechanism that caps it is that a window's rows are already spent on drafts.

### 4.3 The cheaper architecture that gets most of the win

**One decode session per GPU, one shared prefill instance, and the prefill/decode split of S4.3.**

Numbers for this box (§2.4, §0):

| | VRAM | pinned RAM | dev effort | what you get |
|---|---|---|---|---|
| **2 decode sessions on 2 GPUs** (GPU 1 is free today) | 1 full engine per card: ~22.4 GiB used each today, so it **fits on an empty card** | 2 × 6.19 GiB = 12.4 GiB | S4.3 only; **zero kernel changes** | 2× decode throughput, 2× tail-latency isolation, and the prefill instance amortises the ~9.5 s prompt floor (`.megamind/prefill-levers.md`) across all of them |
| **B=2 in one window on one GPU** | +180 MiB (−130 expert slots) | +6.2 GiB | S4.4 in full | **+11…13 %** (§3.5, computed from the log) on that one GPU |

The two-GPU route is **strictly better on this machine** and it is the shape the owner already asked for
in `prompt.md` item 1. It is also the only route that works under `--kv k8v4` (each card streams
nothing but holds its own 7.19 GiB).

**The honest caveat:** two sessions on two cards is *not* what the owner's sentence means. "Real
batching of concurrent requests on the GPU" is B sequences in one pass. This document's recommendation
is to **do the split first, measure, and only then decide whether B>1 per GPU is worth six kernel
contracts.**

### 4.4 Recommended order relative to S4.3

1. **S4.3 (the prefill/decode split) first.** It is already designed (`docs/STAGE4-SPLIT-ROLES.md`), it
   needs no kernel work, and it is the thing that makes N decode instances possible at all.
2. **S4.4-step-0 (this document's §5 experiment) in parallel with S4.3.** It is CPU-only and GPU-cheap.
3. **S4.4-step-1 (two sessions, one engine thread, D2)** only after S4.3 lands, because S4.3 already
   puts the second session on a second GPU and D2 is the single-GPU version of the same plumbing.
4. **S4.4-step-2+ (D1, B=2 in one window)** only if step 0's numbers say the GPU is the bottleneck.

---

## 5. The minimal bit-exact step, and the test that proves it

### 5.1 The claim

> **The smallest change that provably produces bit-identical tokens for each batched request versus
> running it alone is: two `SessionState` arenas, two `Verifier` instances, and an engine thread that
> issues one window per session per turn — with no kernel, no graph and no arithmetic touched.**

Why it is exact *by construction* rather than by proof: every kernel call in the window is the same
call, with the same arguments, on the same shapes, as today (`src/core/verify.cpp:322-793` reads only
`ss_` and its own `arena_`). Two verifiers means two independent `arena_` blocks (`verify.cpp:265`) and
two independent graph sets (`exec_[9]`, `verify.hpp:177`). Nothing about sequence A's window can change
a single floating-point operation of sequence B's, because they share no buffer that a window writes.

What they *do* share, and each is a documented, already-solved problem:

| shared resource | who owns it | why it is not a correctness problem |
|---|---|---|
| the VRAM expert cache + `host_res`/`d_res` | one table, process-wide | Stage 3 §4.1-4.2 already decided this and already documented the quality difference (§7.2 row 1): residency changes GPU-vs-CPU rounding. **This is the one legitimate non-bit-exactness, and it exists today with `--serve-slots ≥ 2`.** |
| the CPU expert pool | one pool, one owner at a time | Stage 3 §4.3; the engine thread is the only caller. Two windows in sequence is still one caller at a time. |
| the prompt loan | one loan | Stage 3 §4.4 / S3.2 |
| the CPU cores | `claim_cores`, machine-wide lease | `.megamind/src/kernels/cpu/pool-notes.md` |
| `mrope_table_set` | **one device table per device** (`mrope.hpp:19-22`) | **this one is a real blocker for two sessions on one GPU**: it is a process-global pointer, so two sessions with different image position tables would corrupt each other. Stage 3's R7 admission rule (one non-identity mrope at a time) must survive. |
| `cvec()` | one device table set (`cvec.hpp:19-35`) | same class of problem; per-request switch is already a device flag |

### 5.2 Where batching legitimately changes results, and why

Extending `docs/STAGE3-CONCURRENCY.md` §7.2:

| difference | legitimate? | why |
|---|---|---|
| expert residency rounding (GPU hit vs CPU miss) | **yes, already accepted** | the engine says so: "the GPU computes the experts in the cache; it rounds differently from the CPU" (`strata-iq3_s.log:5164-5167`); suppress with `--adapt-every 100000` |
| `DraftPolicy` statistics | yes | it learns process-wide; a different draft never changes an accepted token, only how many windows it takes (Stage 3 §7.2) |
| KV-streaming warmth after a switch | yes | `kv_stream_reset` on restore; already reported (`strata-iq3_s.log:8484`) |
| **D1: the GDN reduction layout** | **no — this one must be proven, not accepted** | `gdn_step_norm_multi_kernel`'s `red[RG][col]` cross-thread sums (`verify_kernels.cu:148-165`) are a fixed 4-way split. If a per-row variant changes which thread sums what, the recurrence is not bit-exact, and the recurrence feeds every later layer. |
| **D1: MMVQ `ncols > 8`** | **no** | `native_mmvq.cu:995-997`: the upstream multi-column layout "is equal to ncols = 1 only to float rounding". Any new ncols must use the exact layout (`g_multi_exact = true`). |
| **D1: `GMAX` raise** | **no** | `s2_expert_grouped.cu:554`/`:616` `min(…, GMAX)` silently drops entries; a raise must keep the group order and the per-entry arithmetic identical. |
| **D1: two sequences' rows in one `qsa_block_topk`** | **yes** | one block per query, `steps [nq, kStepCount]` (`qsa_select.hpp:16`) — the selection is already per-query |
| **D1: two sequences' rows in one `qsa_decode_attn_batch`** | **yes** | per-query ids and steps (`qsa_decode_attn.hpp:41-44`) |
| **D1: the sampler's `sp.counter = pos0`** | **no, must change** | `verify.cpp:1087` gives every row of the window a draw at `pos0 + t`. Two sequences at unrelated positions need `counter = pos_of_row(t)`, and the Philox stream is counter-based on (seed, position) (`sampler.hpp:28`, `sampler.cu:36-38`) — so per-row counters are *exactly* what makes a sampled request position-stable. Getting this wrong changes sampled output, not greedy. |
| **D1: `penalty_rows`** | yes | already one row per window row (`verify.hpp:78-88`) |

### 5.3 The test that proves it

Three levels, cheapest first. The first two never start the engine.

**(1) CPU-only, no GPU, no model — the shape test.** A new pure header
`include/strata/program/batch_plan.hpp` (owned by S4.4-step-0) that decides, for a set of slots, *which
rows go in which window*: `plan_window(slots, spec, max_t) → {rows[], seq_of[], n_keep[]}`. Test
`src/program/batch_plan_test.cpp` in the `STRATA_BUILD_TESTS` pattern (`CMakeLists.txt:667-693`, the
`slot_test`/`serve_driver_test` shape). Pin:

* with B=1 the plan is byte-identical to today's `window[0..T)` (`generate.cpp:7196-7197`);
* `Σ rows ≤ kVerifyMaxT` and `∀ row: seq_of[row] ∈ active`;
* a slot with no drafts contributes exactly one row (row 0 = the fed-back token, `generate.cpp:7196`);
* the plan is a pure function of its inputs (determinism).

**(2) Host-side cost gate — the regression test for the pricing.** Extend the scratch binary of
Appendix A into a checked test: `session_bytes`, `qsa_state_bytes`, and the replicated arena carve must
reproduce the numbers in §2 exactly. This is the test that stops a later agent from "improving" the
layout and silently changing what a slot costs.

**(3) The GPU gate — the one that actually proves bit-exactness.** Two arms, one binary, one config:

```
arm A (serial):   --serve-slots 1 --decode-tokens 15          # today
arm B (batched):  --serve-slots 2 --decode-tokens 15          # or --decode-batch 2
```

Same prompt, same seed, same `--adapt-every 100000` (static residency, `docs/DETAILS.md:549-551`), and
the acceptance line is: **the token ids of each request are byte-identical between the arms.** The
existing harnesses are the tool: `bench/results/2026-09-27-cache-parity` (referenced at
`strata-iq3_s.log:5166`) is the owner-run parity check, and `STRATA_STATE_HASH=1`
(`src/program/generate.cpp:7369-7440`) already fingerprints gdn/ple/tail/pooled/kv/mtp/stale/dead —
the S3.1d recipe (mount A, mount B, mount A, compare the two A hashes,
`.megamind/src/program/s31d-swap-notes.md`) generalises directly: **run A alone, hash; run A batched
with B, hash; the hashes must match.**

For D1 specifically, add the **kernel-level parity test** the repo already has the pattern for
(`build/qsa_parity`, `build/gdn_parity`, `build/mmvq_multi_parity`, `build/router_top10_parity`, …):
`gdn_step_norm_multi_parity` — one launch over rows `[s0's tokens | s1's tokens]` with a per-row state
array must produce, for each row, the same `y` and the same state as two separate launches. That test
is the gate for the single hardest kernel change in the whole programme, and it can be written before
the change.

---

## 6. A staged plan for S4.4

Risk **R13** (`docs/STAGE3-CONCURRENCY.md` §9): never two agents in one file. Ownership below is
disjoint by construction. `src/program/generate.cpp` (9 704 lines) appears in exactly **one** step
(S4.4.1), and `serve/server.py` + `serve_driver.hpp` appear only in S4.4.6 — after S4.2 has landed them,
so the hand-off is sequential, not concurrent.

| step | workload | owns (exclusive) | CPU-testable against, without ever starting the engine | the gate that proves single-request output did not regress |
|---|---|---|---|---|
| **S4.4.0** | the window planner + the cost gate. Pure decisions: which rows in which window, `seq_of[]`, per-row `n_keep`, the admission predicate "can this batch run at all" (k8v4, T ceiling, RAM) | `include/strata/program/batch_plan.hpp` (new), `src/program/batch_plan_test.cpp` (new), `docs/STAGE4-BATCH-DECODE.md` updates | `batch_plan_test` in the `slot_test` pattern; the Appendix-A scratch binary as a cost assertion | B=1 plan == today's `window[]` construction, asserted field by field; `cd build && ninja` clean |
| **S4.4.1** | the second session, one engine thread (option **D2**). Two `SessionState` + two `Verifier` + two `MtpDrafter` bindings, mounted/unmounted per turn; the mrope/cvec exclusivity rule extended to sessions | `src/core/session.cpp`, `include/strata/core/session.hpp`, and the session-creation call site in `generate.cpp` | `conversation_snapshot_test`, `conversation_cache_test`, `serve_swap_test` — the second session is a save/restore target, and those tests already build synthetic sessions | `--serve-slots 1` path is **text-unchanged** (the S3.1e-2 proof pattern: count the serve block's `printf`/`fprintf` calls before and after, `.shz_cmd/s31e2_output_proof.py`); `STRATA_NO_BATCH=1` falls back |
| **S4.4.2** | per-row GDN state. `gdn_state` → `(B, n_gdn, floats)`; `gdn_point_at` (`session.cpp:158-164`), `verify.cpp:456-457`, `:909-910`; the three GDN multi kernels take a row index | `src/kernels/cuda/verify_kernels.cu`, `include/strata/kernels/verify_kernels.hpp`, `src/kernels/gdn_multi_parity.cpp` (new) | `gdn_multi_parity` — a host-side reference of the recurrence over two rows vs two launches; `gdn_parity` must stay green | the parity test **is** the gate; plus `STRATA_DEC_BATCH=0` (the existing per-token fallback, `verify.cpp:339`), must still produce today's tokens |
| **S4.4.3** | per-sequence QSA state: `idx_pooled`/`idx_tail`/`idx_dead` per row, per-row page table into `QsaAttnPools`, per-row `kv_stream` map | `src/core/layer.cpp`, `include/strata/core/layer.hpp`, `src/kernels/cuda/kv_stream.cu`, `src/kernels/cuda/qsa_select.cu` | `kv_stream_parity`, `qsa_parity`, `kv_q8_parity`, `kv_q4_parity`, `kv_hybrid_parity` (all in `build/`) | the existing parity binaries unchanged; a new `qsa_multi_seq_parity` that runs two sequences' selections through one launch and compares to two launches |
| **S4.4.4** | the ceilings: `kVerifyMaxT`/`MAX_NCOLS`/`GMAX`/`cpu::MAXT`/the `[9]` arrays, behind one new constant `kBatchMaxRows` | `include/strata/kernels/verify_kernels.hpp`, `src/kernels/cuda/native_mmvq.cu`, `src/kernels/cuda/s2_expert_grouped.cu`, `include/strata/kernels/cpu/{expert,pool}.hpp`, `include/strata/core/{verify,mtp}.hpp` | `mmvq_multi_parity`, `native_expert_parity`, `router_top10_parity`, `s2_gemv_parity` | every existing parity binary at T ≤ 8 must be **byte-identical**; the new ones cover T = 9..16 |
| **S4.4.5** | the window: per-row `pos`/`step`/`seq_of`, per-row `n_keep` in the commit graph, per-row sampler counter, per-row PLE history | `src/core/verify.cpp`, `include/strata/core/verify.hpp` | `coupled_draft_test`, `sampler_parity`, `ple_parity` | `STRATA_STATE_HASH` recipe (§5.3 arm A vs arm B); the `--serve-slots 0` wire is untouched |
| **S4.4.6** | the scheduler's batch turn: `pick()` learns about batched turns; `--decode-batch N`; `/slots`, `/metrics`, the activity line | `include/strata/program/slot.hpp`, `include/strata/program/serve_driver.hpp`, `src/program/serve_driver_test.cpp`, `serve/server.py` | `slot_test`, `serve_driver_test`, `python3 -m pytest serve/test_server.py -q` (the `MockEngine` path, `serve/server.py:70-104`) | `serve_driver_test`'s existing end-to-end simulations must still pass unmodified |

**Sequencing.** 0 is free and first. **1 is the decision point**: it is cheap, exact by construction, and
its measurement is what justifies or kills 2-6. 2 and 3 are independent of each other but both depend on
1; 4 and 5 depend on both; 6 depends on 5. Never run 2 and 3 in parallel with 5 (`verify.cpp` is in both).

**What must be true before any of 2-6 starts** (the parent's gate, not a child's):

* S4.3 landed, so the second session has a home;
* the owner has approved shrinking the expert cache by the slots each step costs (§3.4's table), because
  that is a **quality** decision (85-94 % hit rate, `strata-iq3_s.log:8357-8484`), not an engineering one;
* the owner has answered whether pinned-RAM growth is acceptable — 6.19 GiB per extra sequence at
  524288 is not a rounding error on a 94 GiB box that currently has 5 GiB available.

---

## 7. Risks specific to this programme

| # | risk | evidence | mitigation |
|---|---|---|---|
| **B1** | **The 8-row ceiling is six ceilings.** Raising one and not the others is silent: `s2_expert_grouped.cu:554` `min(…, GMAX)` drops entries; `cpu::MAXT` arrays are fixed; `exec_[9]` indexes out of bounds at T=9. | §1.2 | one new constant `kBatchMaxRows`, one PR that raises all six (S4.4.4), and a `static_assert` at each site |
| **B2** | **Raising `GMAX` breaks occupancy, not correctness.** `__shared__ int xs_q[GMAX][H/4]` + `xs_d` (`s2_expert_grouped.cu:550-551`) is 2 560 B per entry; GMAX=16 → 41 KB, at `__launch_bounds__` on a 3090. | `s2_expert_grouped.cu:550-551`, `:612-613` | measure `cudaOccupancyMaxActiveBlocksPerMultiprocessor` in the parity test before shipping the raise |
| **B3** | **The GDN recurrence's cross-thread reduction is the bit-exactness cliff.** | `verify_kernels.cu:123-125`, `:148-175` | S4.4.2's parity test **before** the change; keep the per-row reduction layout byte-identical |
| **B4** | **The MMVQ multi-column layout is only exact under `g_multi_exact = true`.** | `native_mmvq.cu:995-997` | never enable the upstream layout as a side effect of raising ncols |
| **B5** | **`mrope_table_set` is a process-global device pointer.** Two sessions, one with images, corrupt each other's positions. | `mrope.hpp:19-22`; `generate.cpp:2042-2055`; Stage 3 R7 | keep R7's admission rule; a per-session table is more VRAM (R1) |
| **B6** | **The expert cache is the real budget, and spending it changes output quality.** | 7 777 slots = 14.75 GiB, 85-94 % hit (`strata-iq3_s.log:5163`, `:8357-8484`) | the owner decides; `--adapt-every 100000` for reproducibility (`docs/DETAILS.md:549-551`) |
| **B7** | **Two sessions on one GPU double the pinned host KV.** 6.19 GiB each at 524288; the box has 5 GiB available now. | `strata-iq3_s.log:5156`; `free -g` | size by tokens, not `max_context` (option (a)'s host arena), or put the second session on the second card |
| **B8** | **`--kv k8v4` makes all of this impossible.** | `layer.cpp:557-562`; 7.19 GiB/session | refuse at start-up with both numbers |
| **B9** | **The payoff may not be there.** A window costs 24 ms at T=1 and 66 ms at T=6, so only 8.4 ms of a row is amortisable; the rest is fixed host/pool/PCIe work that batching does not remove. | §3.5 | S4.4.0 + S4.4.1 measure before 2-6 are funded |
| **B10** | **Two agents in one file (R13).** `verify.cpp` (1 194 lines), `layer.cpp` (1 312), `generate.cpp` (9 704). | Stage 3 §9 R13 | the ownership table in §6 is disjoint; `generate.cpp` appears in exactly one step (S4.4.1) |

---

## 8. Explicitly out of scope for S4.4

* A radix/prefix KV sharing scheme (llama.cpp's `server_prompt_cache`). Strata's equivalent already
  exists and is the right tool: `ConversationCache` + `conversation_snapshot_*`
  (`docs/STAGE3-CONCURRENCY.md` §2.4), and S4.3's prefill instance owns it
  (`docs/STAGE4-SPLIT-ROLES.md` §5.2).
* Chunked-prefill + decode overlap inside one window (a mixed batch). `Prefill` is not graphed
  (§1.6), so the overlap is already available at the *step* level; mixing them at the *row* level would
  put the 8 192-token GEMM path and the ≤ 8-row MMVQ path in one graph. Not worth it.
* Tensor parallelism. `docs/MULTI_GPU.md:9-11` is explicit: this is layer (pipeline) parallelism.
* Persistent snapshots to disk (`docs/DETAILS.md:543-544`).

---

## Appendix A — how the numbers in this document were produced

### A.1 The scratch binary

`/tmp/s41_scratch/costs.cpp`, built exactly as Stage 3's Appendix A prescribes:

```
g++ -std=c++20 -O1 -Iinclude -I/opt/cuda/include -o /tmp/s41_scratch/costs /tmp/s41_scratch/costs.cpp \
    -L build -lstrata_engine -lstrata_core -lstrata_kernels -lstrata_kernels_cpu -lstrata_spec \
    -lstrata_prefill -lstrata_mmq \
    build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a \
    -L/opt/cuda/lib64 -lcudart -lcublas -lpthread -ldl
```

It calls `session_bytes`, `qsa_state_bytes`, `qsa_buffers_bytes`, `moe_buffers_bytes`,
`block_buffers_bytes`, `gdn_buffers_bytes`, `kv_q8_bytes_per_cell`, `kv_q4_bytes_per_cell`,
`kv_block_bytes`, `kv_stream_map_bytes`, `qsa_selection_width`, `qsa_decode_attn_scratch_floats`,
`native_q8_1_bytes`, `moe_hit_grouped_scratch_bytes`, `native_expert_scratch_bytes`, and it
**replicates `Verifier::init`'s arena carve** (`src/core/verify.cpp:224-263`) as the same `Bump`
arithmetic. It **never calls `session_init` or `qsa_state_init`** (both `cudaHostAlloc`) and makes no
CUDA call. The replication is validated against the live line: predicted 76.8 MiB for `max_t = 6`,
printed 76.9 MiB (`strata-iq3_s.log:34`).

Two notes for whoever reuses it: `native_q8_1_bytes` **throws** for `ncols > 8`
(`native_mmvq.cu:1147-1150` → `validate_shape` `:1084`) — that is the §1.2 ceiling showing up as a
host-side exception, which is a nice confirmation that the limit is real and enforced. And
`qsa_set_kv_int8` / `qsa_set_kv_resident` are process-global (`layer.cpp:524-527`), so the harness must
set them before each `session_bytes` call, exactly as `generate.cpp` does.

### A.2 Log-derived facts (read-only `grep`/`sed` over `strata-iq3_s.log`)

| fact | line |
|---|---|
| `KV streaming: 32768 of 524288 cells per QSA layer in VRAM, the K/V in 6.19 GiB of pinned RAM` | `:5156` |
| `expert cache 7777 slots, 14.75 GiB of VRAM` / `8019 slots, 15.20 GiB` | `:5163` / `:5197` |
| `strata mtp: draft layer loaded, 836 MiB of VRAM (experts 675, dense 111)` | `:5158` |
| `the prompt path borrows 2625 CUDA0 cache slots (4.95 GiB)` | `:5175` |
| `strata verify: window up to 6 tokens, 76.9 MiB of device buffers` | `:5176` |
| `strata mtp: draft head over 40525 tokens (81.2 MiB)` | `:5177` |
| `457 MiB of VRAM free with everything loaded` / `455 MiB` | `:5178` / `:5231` |
| `cores: host 0 + workers 1,2,3,4,5 (6 physical cores …)` | `:5170` |
| decode hit rate 85.3 % … 94.2 % | `:8357`, `:8484` |
| `KV streaming: 94.20 % of 4 287 278 748 block reads hit VRAM, 1 002 470.2 MiB read from RAM` | `:8484` |
| `drafts accepted 1959 of 2944` at 3305 tokens / 50 085 ms = 66.0 tok/s | `:8578` |
| `parked 153983 tokens in 410.0 ms … snapshot_bytes=3180486108` | `:5143` |
| **no `layer split` line anywhere in the log** | `grep -c` = 0 |

### A.3 Host facts (read-only)

`nvidia-smi`: three RTX 3090 (24 576 MiB each); GPU 0 23 829 MiB used, GPU 1 **2 396 MiB**, GPU 2
23 833 MiB. Compute apps: pid 12236 → 22 382 MiB on GPU 0, pid 12182 → 22 380 MiB on GPU 2, pid 12143
(a whisper server, not Strata) → 2 386 MiB on GPU 1. `free -g`: 94 total, **5 available**.
`df -h /dev/shm`: 70 GiB, 24 GiB available.

### A.4 Guesses, labelled

* **"+10 % to +35 % for B=2"** (§3.5). Derived from the 66 ms/window vs ~24 ms GPU-floor split, not
  measured. The whole point of S4.4.0/.1 is to replace it with a number.
* **The arena-carve slope `42.7 + 5.69·T MiB`** (§2.2) is a fit to the harness's own outputs at
  T = 1..8, which are exact; extrapolating past T = 8 is a guess because the ceilings force new
  template instantiations whose shared-memory and register behaviour differs.
* **"MTP experts could be shared, buffers and graphs could not"** (§2.4, §3.3). The experts are one
  contiguous `experts_` allocation (`mtp.cpp:195-201`) read by the grouped kernels, so sharing is
  plausible; it is not implemented and not verified.
* **The 35 MiB MTP ring** is Stage 3 §2.2's measured figure re-derived from `mtp.cpp:213-221`'s
  `ring = window + 4·max_t + 64` formula, not re-measured here.
* **`--spec-split` is off by default and ~7 % slower** (`generate.cpp:427`), so it is not a free lever —
  it is a measured trade the engine already made.
