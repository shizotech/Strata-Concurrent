# S4.1 — batched-decode feasibility (docs/STAGE4-BATCH-DECODE.md)

Deliverable: **`docs/STAGE4-BATCH-DECODE.md`** (937 lines). Design + measurement only: **no source file
was edited, no engine/server/model/GPU work, `/dev/shm` read-only.** Scratch binary lives in
`/tmp/s41_scratch/{costs.cpp,out1.txt}` (rebuild recipe is in the doc's Appendix A).

## The three findings that change the plan

1. **The box is not what Stage 3 described.** `grep -c "layer split" strata-iq3_s.log` = **0**. Both
   live engines (`ps`: pid 12182 on GPU 2 @524288, pid 12236 on GPU 0 @262144) run **one whole GPU
   each**, no split. **GPU 1 is free** (2 396 MiB used, all of it a whisper server, pid 12143). So the
   per-stage 0.554/0.560/0.547 GiB figures are not what this box pays — a single-GPU session arena at
   524288/int8/r32768 is **1.390 GiB**, and the cheapest "batching" on this machine is a second session
   on GPU 1, not a kernel change.
2. **Stage 3's "1.753 GiB for 524288 int8 resident 0" is wrong.** Measured **7.189 GiB** (nothing
   streams, so all 12 int8 pools are resident). Do not reuse that number.
3. **The kernels are already batched where Stage 3 assumed they were not.** `qsa_decode_attn_batch`,
   `qsa_block_scores`, `qsa_block_topk`, `kv_stream_resolve`, `kv_append_*_step` all take `n_q` rows
   with **per-row step records and per-row selections** (`qsa_decode_attn.hpp:41-44`,
   `qsa_select.hpp:25-38`, `kv_stream.hpp:68-73`). The window's `h_step_`/`h_pos_` are already
   `T × …` arrays (`verify.hpp:182-183`); "one sequence per window" is a **host policy in
   `Verifier::run` (`verify.cpp:964-969`)**, not a kernel limit.

## What is genuinely single-sequence (the real work)

* **GDN**: `gdn_step_norm_multi_kernel` (`verify_kernels.cu:115-181`) — one `state`, serial over rows
  with `__syncthreads()`, per-block shared `sk/sq/red/wsum`. +3.117 MiB/layer/sequence = **112.2 MiB**
  for 36 layers. The kernel *already* has the per-row shape (`n_keep`, `t_out_begin`, used by
  `--spec-split` at `verify.cpp:473-475`, `tb_/te_` at `:341`).
* **The indexer key store**: `idx_pooled`/`idx_tail`/`idx_dead` are indexed by **cell index with no page
  table** (`layer.hpp:234-237`; `qsa.hpp:197` literally says "PER-SEQUENCE STATE"). 128 B/token/layer.
  `native_qsa_indexer_append` documents contiguity (`native_qsa_indexer.hpp:18-27`).
* **The 8-row ceiling is SIX ceilings**, and one sequence's drafts already consume all of them at
  `--spec 4` (→ `o.spec = 6`, `generate.cpp:1661`): `kVerifyMaxT` (`verify_kernels.hpp:21`),
  `cpu::MAXT` (`cpu/expert.hpp:131`), `GMAX` (`s2_expert_grouped.cu:516-519`, with a **silent
  `min(…, GMAX)`** at `:554`/`:616`), `MAX_NCOLS` (`native_mmvq.cu:783`, dispatch `:1066-1075`),
  `exec_[9]` + the MTP's six `[9]` arrays, and `--spec`'s own clamp.

## Measured numbers (host-side, scratch binary; validated against the live log)

* `Verifier` arena, replicated from `verify.cpp:224-263`: **76.8 MiB at max_t=6** vs the log's printed
  **76.9 MiB** (`strata-iq3_s.log:34`, `:5176`). Linear: `≈ 42.7 + 5.69·T MiB`. T=8 → 88.2 MiB.
* `session_bytes`: 13.001 GiB (fp16/r0), **7.189 GiB** (int8/r0), **1.390 GiB** (int8/r32768) at 524288.
* GDN per sequence: 112.2 MiB (36 layers). Per extra sequence, 12 QSA layers, r32768: **55.9 MiB**
  (page tables 6.0 + maps 1.9 + `idx_pooled` 48.0).
* Pinned host KV: **6.19 GiB** per session at 524288 (`:5156`), 3.09 GiB at 262144 (`:5190`).
* One extra decode session, all-in: **≈ 1.47 GiB VRAM** (session 1.390 + verifier 76.8 MiB + MTP ring
  ~35 MiB, experts shared) **+ 6.19 GiB pinned RAM**.

## The payoff arithmetic (this is the part the parent must weigh)

Two clean log numbers give a **marginal cost per window row**:

* T=1 window = **24 ms** (`strata-iq3_s.log:106`, `1 generated in 24 ms`; the first window is always
  T=1, `generate.cpp:7180`). Stage 3 §3.2's "~24 ms predicted … `strata-iq3_s.log:27`" cites a stale
  line but the number is real.
* T=6 window ≈ **66 ms** (`:8578`: 3305 tok / 50 085 ms = 66.0 tok/s, `drafts accepted 1959 of 2944`
  ⇒ 0.665/draft ⇒ ≈ 4.3 tokens/window).

⇒ fixed 24 ms + **8.4 ms per extra row**. B=2 at `--spec 3` each (T=8) = 83 ms for 6.0 tokens =
**+11 %**; B=2 at T=12 (ceilings raised) = **+13 %**; B=4 at `--spec 1` (T=8) = **+23 %**.
**Not +100 %.** The rows that would have been drafts become second-sequence rows.

## Recommended shape (the doc's §4.4 / §6)

S4.3 first (no kernel work, already designed). Then **S4.4.1 = two sessions, one engine thread, two
window launches** — **zero kernel changes, bit-exact by construction**, +1.47 GiB VRAM / +6.19 GiB RAM.
That step is the *measurement* that decides whether the six-kernel-contract route (S4.4.2..5) is worth
funding at all. On this box, **a second session on the free GPU 1 beats B=2 in one window** on every
axis except the owner's literal sentence.

## Traps / do-not-redo

* `native_q8_1_bytes` **throws** for `ncols > 8` (`native_mmvq.cu:1147-1150` → `validate_shape:1084`).
  The ceiling is host-enforced, which is a gift: the cost harness fails loudly instead of lying.
* `qsa_set_kv_int8` / `qsa_set_kv_resident` are **process globals** (`layer.cpp:524-527`) — set them
  before every `session_bytes` call in any harness.
* Never call `session_init` / `qsa_state_init` from a scratch binary: both `cudaHostAlloc`
  (`layer.cpp:617-623`, `:631`).
* Raising `GMAX` grows `__shared__ int xs_q[GMAX][H/4]` (`s2_expert_grouped.cu:550-551`) at 2 560 B per
  entry — GMAX=16 is 41 KB, an **occupancy** cliff on a 3090, not a correctness one.
* MMVQ multi-column is bitwise-equal to ncols=1 **only** under `g_multi_exact = true`
  (`native_mmvq.cu:995-997`). Raising ncols must not enable the upstream layout.
* `mrope_table_set` is a **process-global device pointer** (`mrope.hpp:19-22`, `generate.cpp:2055`) —
  two sessions on one GPU with images corrupt each other. Stage 3's R7 admission rule must survive.
* `--spec-split` is **off by default and ~7 % slower** (`generate.cpp:427`), so it is not a free lever.
* `--kv k8v4` refuses streaming (`layer.cpp:557-562`) → 7.19 GiB int8 / 13.00 GiB fp16 per session, so
  **every** batching option is unavailable there. Refuse at start-up with both numbers.

## Best test for the acceptance bar

`STRATA_STATE_HASH=1` (`generate.cpp:7369-7440`) already fingerprints gdn/ple/tail/pooled/kv/mtp/
stale/dead. S3.1d's recipe generalises: **run A alone → hash; run A batched with B → hash; the hashes
must match.** Plus a new `gdn_multi_parity` (host reference of the recurrence over two rows vs two
launches) written **before** the GDN change, in the `build/{gdn,qsa,kv_*}_parity` pattern.
