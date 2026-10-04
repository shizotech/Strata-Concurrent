// src/program/generate.cpp - P2.S6: `strata generate`.
//
// THE DRIVER, and the first program in this project that answers a question.  Everything below it is a
// component; this is the thing that composes them into a token:
//
//     embed_row(token)  ->  48 captured layer graphs (the CPU expert pool behind the doorbell)  ->
//     lm_head(R)        ->  sample        ->  embed_row(next)  ->  ...
//
// WHAT IT IS NOT.  There is no tokenizer here.  `pack/full/tokenizer/` and `tools/strata_tokenizer.py` exist,
// and a C++ BPE is Phase 1's deliverable rather than this program's, so the prompt arrives as IDS via
// `--tokens`.  That is not a placeholder: it is exactly what Gate C1 needs, because C1 compares logits against
// llama.cpp on the SAME ids, and a tokenizer on only one side of that comparison is a second variable.
//
// AND IT IS PHASE 2, so hit rate is `h = 0` and the number it prints is slow on purpose
// (`phase-2-correct-engine.md:5-9`).  What it is FOR is the honest tok/s figure and the logit dump.

#include "strata/core/device.hpp"
#include "strata/core/expert_cache.hpp"
#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/remote_experts.hpp"
#include "strata/core/on_device.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/session.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_indexer.hpp"
#include "strata/kernels/native_rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/qsa.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/verify.hpp"
#include "strata/core/mtp.hpp"
#include "strata/prefill/prefill.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/program/logits_selection.hpp"
#include "strata/program/conv_cache.hpp"
#include "strata/program/prefill_loan.hpp"
#include "strata/program/serve_proto.hpp"   // S3.1c: the --serve wire format, in one place
#include "strata/program/slot.hpp"          // S3.1b: the slot object / state machine / pick()
#include "strata/program/serve_swap.hpp"    // S3.1d: mount/unmount a slot in the one live session
#include "strata/program/serve_driver.hpp"  // S3.1e-2: the concurrent driver's decisions (admission, step dispatch, loan, watchdog)
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"
#include "strata/kernels/cvec.hpp"
#include "strata/core/progress.hpp"
#ifndef NOMINMAX
#define NOMINMAX   // gguf_reader.hpp includes windows.h
#endif
#include "strata/artifact/gguf_reader.hpp"
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#include <io.h>
#else
#include <unistd.h>
#include <cerrno>
#endif

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <algorithm>
#include <iostream>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <map>
#include <mutex>
#include <optional>
#include <new>
#include <charconv>
#include <cmath>
#include <limits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <set>
#include <vector>

namespace {
// perf-review D-4: the lent slots are refilled with queued copies and one wait; STRATA_REFILL_BLOCKING=1 waits on each
bool refill_blocking() {
    static const bool v = std::getenv("STRATA_REFILL_BLOCKING") != nullptr;
    return v;
}

// ---- S0.3: the --serve prefill FIXED COST (see include/strata/program/prefill_loan.hpp and
// bench/prefill/fixed-cost-changes.md).  Measured on this box over 1 600 live requests:
// `prompt_ms ~= 9 500 fixed + 2.7 * fresh`, and the fixed term does not scale with the cached context, so
// it is per-request overhead around the read - the expert-cache loan the prompt path takes and gives back.
// Every switch below defaults to the NEW behaviour and reverts to today's when set; all are read once at
// startup, like every other STRATA_* knob.
//
// STRATA_PREFILL_STICKY_LOAN=0: hand the loan back and re-take it (with a `Prefill::relayout`) even when
// the next segment needs exactly the layout that is still lying in the cache.
bool sticky_loan() {
    static const bool v = [] {
        const char* e = std::getenv("STRATA_PREFILL_STICKY_LOAN");
        return e == nullptr || std::atoi(e) != 0;
    }();
    return v;
}
// STRATA_RES_UPLOAD_ALWAYS=1: re-upload the residency table on every lend/refill whether or not its
// content changed.  The skip is a content comparison against what was last uploaded, so it cannot miss a
// change; this is the A/B arm, not a safety valve.
bool res_upload_always() {
    static const bool v = std::getenv("STRATA_RES_UPLOAD_ALWAYS") != nullptr;
    return v;
}
// STRATA_PREFILL_LOAN_TIMING=1: one stderr line per lend and per refill, so the loan's cost is measurable
// from the serve log instead of inferred from the source (the log has no lend/refill timer today).
// S3.2b extends it: a refill line now says EAGER or LAZY, each pump batch prints its own line, and the
// per-request line splits the rows into eagerly refilled / pumped home / still out of the cache.
bool loan_timing() {
    static const bool v = std::getenv("STRATA_PREFILL_LOAN_TIMING") != nullptr;
    return v;
}

// ---- S3.2b: the loan goes back LAZILY (include/strata/program/prefill_loan.hpp, lever 5). -------------
// The 4.95 GiB per stage that `refill()` streamed back at the end of every prompt is the largest remaining
// prefill fixed cost, and S0.3 named it as the floor that only concurrency could remove.  It is not a
// correctness requirement: `kNotResident` is a legal state the decode path already handles (a verify window
// routes an expert, finds the row non-resident and sends that row to the CPU pool).  So the rows can stay
// out, come back in bounded batches between decode steps, or come back never.
//
// STRATA_PREFILL_LAZY_LOAN: `auto` (unset, the default) = on under the concurrent driver (--serve-slots >= 2
// with the slot hand-over and the conversation cache both live), off on the serial path, which therefore
// stays 0.1.30's byte-for-byte.  `=1` forces it on everywhere (the A/B arm for --serve-slots 0/1), `=0`
// forces it off everywhere.
//
// WHY THE DEFAULT IS GATED AND NOT SIMPLY "on".  Lazy return is not only a timing change: a row that is
// still out of the cache is computed by the CPU instead of the GPU, and the two round differently
// (expert_cache.hpp's parity note - the same caveat the cache itself ships with).  The serial path's
// acceptance bar is bit-exactness against 0.1.30, so the default keeps that bar and the switch lets the
// owner measure the lazy path there.  Under the concurrent driver that bar is already crossed by the swap
// itself, and the loan is a process resource, which is where the saving is.
int lazy_loan_env() {           // 1 = forced on, 0 = forced off, -1 = auto (concurrent driver only)
    static const int v = [] {
        const char* e = std::getenv("STRATA_PREFILL_LAZY_LOAN");
        if (e == nullptr) return -1;
        return (std::atoi(e) != 0) ? 1 : 0;
    }();
    return v;
}
// STRATA_PREFILL_LOAN_PUMP_ROWS N: expert rows per cache that the overlapped pump may have in flight at a
// time (default 16; 0 = never pump, so a ledger is only drained by an eager refill).  Bounded, and one
// batch per cache at a time, on purpose: the copies run on their own non-blocking stream beside the decode
// steps, and an unbounded queue would put a multi-gigabyte DMA in front of the next window.  The bound is
// also what throttles the pump by itself - if a batch has not landed by the next window, no new one is
// queued - so N is a ceiling on queue depth, not a rate: the effective rate is N rows per
// max(window time, the batch's DMA time).
//
// 16 rows is ~30 MiB per cache.  On this box's links that is ~23 ms of DMA on CUDA1 (1.3 GB/s) and ~10 ms
// on CUDA0 (2.9 GB/s) - about one verify window - so a batch is normally in flight continuously without
// ever queueing more than a window's worth behind the GPU.  It is NOT free: that DMA shares PCIe with the
// decode path's own streamed experts and its KV reads.  Raise it to recover the decode hit rate faster;
// set it to 0 to leave the cold rows on the CPU pool for the whole request (the cheapest prompt, and the
// slowest decode if the conversation then runs long).
int64_t loan_pump_rows() {
    static const int64_t v = [] {
        const char* e = std::getenv("STRATA_PREFILL_LOAN_PUMP_ROWS");
        return e != nullptr ? (int64_t) std::atoll(e) : (int64_t) 16;
    }();
    return v < 0 ? 0 : v;
}

using Clock = std::chrono::steady_clock;

// The resident RAM mode and the adaptive tier.  A swap copies `in` (held in RAM) into the slot of `out` (held only
// by that slot).  Before the slot is overwritten, `out`'s bytes are copied back from it into an exchange buffer, so
// the CPU computes `out` from RAM while the swap is in flight; when the swap has landed, `commit_exchanges` moves
// them into `in`'s place in RAM.  The RAM copy then again holds exactly the experts no core slot does, and no swap
// reads the file.  Swaps that need no exchange (`out` in the lend region is held in RAM already; or `in` is not) go
// on as before; ones beyond the buffers' room wait for a later round.  Runs on the adaptive tier's thread while the
// GPU commits and drafts: the copies back are on its stream, and waited for before the refills are queued.
template <class Swap>
bool resident_stage_swaps(strata::core::FileExpertSource& src, strata::core::ExpertCache& cache,
                          const std::vector<int32_t>& host_res, int64_t n_expert, std::vector<Swap>& swaps,
                          cudaStream_t stream) {
    if (!src.complement_ready() || swaps.empty()) return true;
    struct Staged { int32_t layer, in, out; int64_t q; };
    std::vector<Staged> staged;
    std::vector<Swap> kept;
    kept.reserve(swaps.size());
    for (const Swap& s : swaps) {
        if (!src.has_resident(s.layer, s.in) || src.has_resident(s.layer, s.out)) { kept.push_back(s); continue; }
        const int64_t q = (int64_t) staged.size();
        if (q >= src.exchange_capacity()) continue;
        const int32_t slot = host_res[(size_t) s.layer * (size_t) n_expert + (size_t) s.out];
        if (slot < 0) continue;
        if (cudaMemcpyAsync(src.exchange_buffer(q), cache.device_slot(slot),
                            (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer), cudaMemcpyDeviceToHost,
                            stream) != cudaSuccess)
            return false;
        staged.push_back({s.layer, s.in, s.out, q});
        kept.push_back(s);
    }
    if (!staged.empty()) {
        if (cudaStreamSynchronize(stream) != cudaSuccess) return false;
        for (const Staged& x : staged)
            if (!src.stage_exchange(x.layer, x.in, x.out, x.q)) return false;
    }
    swaps.swap(kept);
    return true;
}

struct Options {
    std::string pack = "pack/full";
    std::vector<int64_t> tokens;      // the prompt, PRE-TOKENIZED
    int64_t max_new = 16;
    int64_t max_context = 4096;
    bool greedy = true;
    uint64_t seed = 0;
    int top_k = 20;
    float top_p = 0.95f;
    float temperature = 1.0f;
    std::string dump_logits;          // one line of logits per generated position
    int64_t logits_stride = 1;        // storage selection; all prompt tokens remain conditioned
    /// **THE RESIDUAL, SO THE HEAD CAN BE CHECKED WITHOUT THE LAYERS.**
    ///
    /// C1 fails (LEDGER L116) and the pipeline is `embed -> 48 layers -> head`.  Dumping `R` splits it in half:
    /// the head is one norm, two bf16 projections and one 794 MB GEMV, all of which can be recomputed in Python
    /// from the manifest.  If Python agrees with the engine on the same `R`, the head is right and the layers
    /// are wrong; if it disagrees, the head is wrong.  Nothing else in the engine can be split that cheaply.
    std::string ple_gguf;              // the ORIGINAL second GGUF shard: the PLE table is not in the pack
    bool no_ple = false;              // explicit diagnostic ablation; never a normal inference default
    bool stream_token = false;        // R2.6 experiment: ordered work on the session stream
    bool check_logits = false;        // optional full-vocabulary finite scan
    bool gr_fp32_activations = false;  // pinned CUDA single-token BF16 activation contract
    bool gr_native_mmvf = false;       // pinned projection reduction tree as well as FP32 inputs
    bool native_bf16 = false;          // SSM gates, router and indexer projections only
    bool native_bf16_extra = false;    // PLE value and shared expert scalar gate
    bool native_ple_key = false;       // unchanged Q2_0 key and CUDA Q8_1 activations
    bool native_moe_combine = false;   // pinned fused CUDA weighted reduction
    bool native_gdn = false;          // pinned CUDA recurrence and preprocessing
    bool native_flash_attn_short = false; // diagnostic pinned attention, context <=256
    bool native_qsa_indexer = false;  // pinned F16 key cache and F32 pooling
    bool native_qsa = false;          // pinned F32 QSA norms and gate
    /// THE CONTEXT EXTENSION (rope scaling, rope_scaling.hpp).  These knobs resolve to ONE process config,
    /// set once before `session_init` builds the rope table and the graphs capture the kernels.  There is
    /// deliberately no per-request form: K sits in the cache POST-RoPE, so one cache must never mix two
    /// scalings.  Precedence: an EXPLICIT flag over the model file's rope keys over the struct defaults -
    /// `none` and `1` are explicit values (the opt-outs), the absent flag is not.
    std::string rope_scaling;           ///< --rope-scaling none|linear|yarn (llama.cpp's names); empty = the flag is absent
    double rope_scale = 0;              ///< --rope-scale F: the extension factor; 0 = the flag is absent (the model file's, else 1 = off)
    double rope_freq_base = 0;          ///< --rope-freq-base N: 0 = the model's (1e7)
    double rope_freq_scale = 0;         ///< --rope-freq-scale F: the raw ggml knob; 0 = 1/--rope-scale
    double yarn_orig_ctx = 0;           ///< --yarn-orig-ctx N: 0 = the model's, else 262144
    double yarn_ext_factor = -1.0;      ///< --yarn-ext-factor F: <0 = auto (1 for yarn, 0 otherwise)
    double yarn_attn_factor = 1.0;      ///< --yarn-attn-factor F
    double yarn_beta_fast = 32.0;       ///< --yarn-beta-fast F
    double yarn_beta_slow = 1.0;        ///< --yarn-beta-slow F
    bool native_rope = false;         // pinned text-only CUDA rotary arithmetic
    bool native_ple_postops = false;  // pinned PLE postprojection arithmetic
    bool native_router = false;       // pinned fused 512-expert top-10 router
    bool cpu_oracle_q8_0 = false;      // pinned x86 activation scales/codes at both expert stages
    std::string native_head_gguf;      // native output.weight experiment; same model shard as the pack
    std::vector<std::string> native_dense_gguf; // repeat for native GDN/QSA projection shards
    /// Plan v0.3 P1: the whole native arithmetic set as ONE switch (model shard 1). It enables exactly the
    /// combination recorded in bench/results/2026-09-23-attention-ple plus the native indexer, and never the
    /// <=256-token attention adapter. It becomes the default once P0 shows it is not slower.
    std::string native_preset;
    /// Plan v0.3 P2: how the n-gram table is read. Direct (default) = unbuffered SSD reads, table never in RAM.
    std::string ple_io = "direct";
    int64_t ple_row_cache = 1 << 20;   ///< bounded row cache (rows of 90 B); 0 disables
    int ple_inflight = 64;
    double ple_delay_us = 0;           ///< fault injection: every row read completes no earlier than this
    bool ple_sync_submit = false;      ///< A/B arm: submit reads on the token thread, no I/O worker
    std::string kv = "fp16";           ///< plan v0.3 P7: KV storage, fp16 (default) or int8 (half the VRAM)
    int64_t kv_resident = 0;           ///< KV streaming: resident cells per QSA layer (0: all in VRAM)
    std::string dump_residual;
    /// The head input, `bb.mixed`.  It exists so the head can be SPLIT: steps 1-4 (the per-stream norm, the two
    /// bf16 projections and the stream mean) recompute cheaply in Python, and only the 794 MB GEMV does not.
    std::string dump_mixed;
    /// One residual snapshot per layer per position: `n_layers * hc * n_embd` floats per position, appended in
    /// position order.  This is the C1 BISECTION LADDER - it is what `llama-debug --tensor-filter l_last` prints
    /// for the reference, so the first layer whose `sum` diverges is the layer that holds the bug.  It needs the
    /// captured path (the expert pool only exists there), so it is refused with `--no-capture`.
    std::string dump_layers;
    /// `2 * n_embd + 2 * hc` floats per layer per position: the attention half's block output, the MoE half's,
    /// and the two injection vectors.  It separates `linear_attn_out-<l>` from `ffn_out-<l>`, which the residual
    /// ladder cannot.  **CAPTURED INTO THE LAYER GRAPHS**, so it must be armed before `session_capture`.
    std::string dump_halves;
    /// P0.S8's routing trace, and a prerequisite the Phase 3 plan names explicitly.  One record per layer per
    /// position: `int32 layer, int32 k, k int32 ids, k float weights`.  It is what a hit-rate curve for a
    /// candidate VRAM expert cache is computed from, and it needs no new kernels - the doorbell already
    /// publishes exactly this much to pinned memory.
    std::string dump_routing;
    bool no_capture = false;          // run the layers directly instead of replaying graphs
    bool no_pool = false;             // skip the CPU expert pool: the GPU-only floor
    bool sync_every_layer = false;
    /// Per-stage CUDA-event timings inside the layer halves.  `--no-capture` only: an event recorded inside a
    /// stream capture is silently dropped, so the captured path cannot carry this.
    bool stage_timing = false;
    /// Launch the 48 captured `pre` graphs back to back with no host work between them and report the pure GPU
    /// time per token.  This is the only measurement that separates host-bound from GPU-bound, because the
    /// stage events include every gap where the GPU waited for the host.
    bool graph_only = false;
    bool gpu_only_full = false;   ///< R0.3: pre + post + head, the true per-token GPU floor
    int pool_workers = 0;         ///< R2.2: 0 = "all physical cores minus the host's"; >0 overrides
    /// R2.2's first half, as an A/B arm.  **ON by default**, because the measurement that justifies it is the
    /// pool's own drain: 33.7 GB/s against 5/6 x 44.14 = 36.8 for five workers, on a machine whose sixth core
    /// is reserved for a host thread that has nothing to do while the drain runs.
    bool no_host_worker = false;
    bool mmap_experts = false;    ///< R2.1: opt OUT of the resident arena, back to MapViewOfFile
    /// Linux: file backing for the resident arena, shared by the processes on the machine.  The first process to
    /// claim it loads the experts; the rest map the same bytes (core/pinned.cu).  ON by default where it exists:
    /// two servers on one PC should hold ONE copy of 47 GiB of experts, not two.  Empty on Windows, where the
    /// shared backing is not implemented and a non-empty default would refuse every start.
#ifdef _WIN32
    std::string shared_expert_arena;
#else
    std::string shared_expert_arena = "/dev/shm/shared_experts.dat";
#endif
    bool shared_expert_arena_given = false;   ///< an explicit --shared-expert-arena is obeyed, a default is not
    bool resident_cpu_experts = false; ///< mmap-backed static-cache misses copied into ordinary RAM
    /// `--resident-experts` (the low-RAM PC's resident mode, chosen by setup): `--resident-cpu-experts` with the copy
    /// page-locked when the driver allows (else locked in the working set), 4 GiB of RAM headroom, and plain mmap
    /// (with a warning) when even the experts no slot holds do not fit.
    bool resident_pin = false;
    uint64_t resident_headroom = 8ull << 30;
    bool resident_soft = false;
    /// R4: slots of VRAM-resident experts.  **0 = off, and off is the default.**
    /// **THE COMMENT THAT USED TO BE HERE WAS FALSE AND ROUND 328 MEASURED IT.**  It said "the cache has no
    /// consumer yet - `moe_hit_grouped_s2` does not exist - so switching it on costs the fill traffic and
    /// saves nothing".  The kernel exists (`src/kernels/cuda/s2_expert_grouped.cu`), it is wired at line ~660
    /// via `expert_hit_run`, and switching the cache on **does** move work off the CPU pool: the drain fell
    /// **19.076 -> 10.312 ms/token** at 4096 per-layer slots, for **-2.7 ms/token** end to end.  What was
    /// true is that the ADMISSION POLICY gave every slot to the first position, which is why the earlier
    /// measurement found nothing - see `expert_cache_per_layer`.
    int expert_cache = 0;
    std::array<int, 3> expert_cache_remote{}; ///< CUDA1..3 slots; CUDA0 keeps dense/state/MTP
    std::string expert_cache_remote_placement = "stripe"; ///< stripe experts or assign complete layers to CUDA1..3
    bool expert_cache_cpu_order = false;
    /// **R4.2g.  ROUND 328 MEASURED THAT THE GLOBAL ADMISSION POLICY CANNOT WORK, AND THIS IS THE FIX.**
    /// The default policy hands out slots in arrival order from one counter shared by all 48 layers, so the
    /// first `n_slots` distinct pairs - about 26 LAYERS OF POSITION 0 - take every slot and hits are confined
    /// to them.  Measured at 256 slots: **1781 of 60000 = 2.97%**, against **21.4%** for 8 slots per layer and
    /// **70.4%** for 64, from `Memory/cache_allocation.py` on the same run's routing.  Off by default.
    bool expert_cache_per_layer = false;
    /// The PLE gather's prefetch, as an A/B arm.  The gather measured 2.10-2.61 ms/token because its sixteen
    /// row reads are sixteen SEPARATE page faults into a 26.8 GB mapping; see `ple_prefetch_enable`.
    bool no_ple_prefetch = false;
    /// R4.2e: a `profile.bin` from `tools/make_profile.py`.  **When given, it decides residency instead of the
    /// compulsory-miss policy**, which is the whole point: a profile ranked by routing frequency over a whole
    /// trace is what the plan's `h = 0.6447` refers to, and compulsory-miss measured 0.4864 because it fills
    /// with whatever the prompt touched FIRST.  Empty means no profile.
    std::string expert_profile;
    /// R4.2d: **ON by default**, because the measurement is unambiguous and the alternative is known-broken.
    /// Without it, 17 of 10,562 layers had the hit work done when the pool returned; with it, 9,190.  The
    /// A/B arm is `--no-hit-poke`.
    bool no_hit_poke = false;
    /// R0.9: capture each layer as THREE graphs and time them from outside the capture, which is the only
    /// valid way to get a per-stage table on the real graph.  Prints and exits; it is a measurement, not a run.
    bool gpu_stages = false;
    bool stats = false;
    bool shared_late = false;          ///< plan v0.3 P3 A/B: shared expert inside post[l] (old order)
    bool keep_canonical = false;       ///< plan v0.3 P1 A/B: load canonical copies of natively served tensors
    bool no_token_graph = false;       ///< plan v0.3 P3 A/B: two graphs per layer instead of one per token
    bool no_fused_gr = false;          ///< plan v0.3 P3 A/B: the six-kernel native gr_read + separate gr_write
    bool no_fast_attn = false;         ///< plan v0.3 P3 A/B: gather + one-block-per-head QSA attention
    bool no_publish_kernel = false;    ///< plan v0.3 P3 A/B: memcpy nodes for the doorbell and QSA step
    bool no_fused_gdn = false;         ///< plan v0.3 P3 A/B: llama.cpp-layout GDN step + separate out norm
    bool no_fast_select = false;       ///< plan v0.3 P7 A/B: FP64 row scores + bit-serial cell top-k
    /// Plan v0.3 P4: `--expert-cache auto` sizes the VRAM tier from what is free after the weights, the session
    /// and the KV state, minus this reserve for the graphs, the hit scratch and the head.
    int vram_reserve_mib = 700;
    /// Plan v0.3 P5: batched prompt processing in chunks of this many tokens (0 = the token path).
    int64_t prefill_chunk = 0;
    /// `--prefill auto`: the largest chunk (up to 8192) whose buffers the expert cache can lend.  Every expert a chunk
    /// routes to is streamed once per chunk, so a bigger chunk streams fewer bytes per token (the "ubatch" effect).
    bool prefill_auto = false;
    bool no_split_rows = false;        ///< plan v0.3 P4 A/B: one whole expert per pool thread
    /// Plan v0.3 P5: the prompt path borrows the top expert-cache slots for its buffers and refills them after
    /// the prompt (default); `--no-prefill-borrow` reserves the buffers' VRAM for the whole session instead.
    bool no_prefill_borrow = false;
    /// Plan v0.3 P5 validation: batch only positions [0, P) and run the rest of the prompt through the token path
    /// (teacher-forced), so the logits of positions >= P - which depend on the batched state - can be scored
    /// against the oracle at many positions.  0 = the whole prompt but the last position.
    int64_t prefill_until = 0;
    /// Plan v0.3 P6: after every processed position, append the residual after the last layer (hc x n_embd
    /// floats, the MTP draft head's input) to this file.  Token path only.
    std::string dump_final_r;
    /// Plan v0.3 P6: speculative decoding with a verify window of this many tokens (the last accepted token and
    /// spec-1 drafts); 0 = plain decode.  `spec_oracle` drafts from a token file (the expected continuation, for
    /// the exactness test); `spec_corrupt` N > 0 replaces every Nth draft with a wrong token.
    int spec = 0;
    std::string spec_oracle;
    int spec_corrupt = 0;
    /// Plan v0.3 P6: the MTP draft layer's runtime directory (tools/mtp_rt.py); drafts come from it.
    std::string mtp;
    int64_t mtp_window = 32768;   ///< the draft layer attends to the last N cells (0 = every cell)
    /// Plan v0.3 P6: the share (0..1) of each layer's distinct missed experts the GPU reads over PCIe from the
    /// pinned arena while the CPU computes the rest (verify windows).
    double pcie_frac = -1.0;   ///< < 0: the model's default (0.2 direct for the Q2_0 pack, 0.55 DMA for native packs)
    std::string pcie_mode = "auto";   ///< auto | dma | kernel | direct
    /// Plan v0.3 P6: every `adapt_every` rounds, swap up to `adapt_swaps` of the most-routed missing experts into
    /// the VRAM tier in place of the least-routed resident ones (decayed counts).  0 = static residency.
    int adapt_every = 4;
    /// Plan v0.3 P6: a draft enters the verify window only while every draft before it (and itself) has at least
    /// this probability under the draft layer; 0 = always --spec-1 drafts.
    double spec_min_p = 0.0;
    /// Stop when the model emits an end-of-turn token (<|endoftext|> 248044, <|im_end|> 248046, or --eos-ids).
    bool stop_eos = false;
    std::vector<int64_t> eos_ids = {248044, 248046};
    bool spec_split = false;   ///< opt-in split verify window (the overlap study: exact, ~7% slower)
    /// --serve, multi-GPU layer split: "K" or "K1,K2,.." (the first layer of each later stage) or "auto" (placed
    /// from each GPU's free VRAM); empty = one GPU
    std::string layer_split;
    /// the later stages' devices "D1,D2,.." (default: the next visible GPUs; "0" with one K: both stages on this
    /// GPU, sharing everything - the bit-exact A/B of the hand-off)
    std::string split_device;
    /// Plan v0.3 P8: stay resident and take requests on stdin (see the --serve block in main).
    bool serve = false;
    /// S3.1b/S3.1c/S3.1e-2 (--serve-slots N): how many conversations may be ACTIVE at once inside this
    /// process.  0 (the default) or 1 = today's serial engine, byte-identical on the wire: no `#<id>`
    /// tags, no `SLOT` lines, and `READY` does not advertise `slots=`.  >= 2 turns the stage-3 protocol
    /// on (include/strata/program/serve_proto.hpp, docs/STAGE3-CONCURRENCY.md §6) AND the concurrent
    /// driver: the serve loop admits requests into the slot registry and runs ONE step of ONE slot per
    /// iteration (`run_prefill_step` = a prompt segment, `run_decode_step` = a verify window), handing
    /// the session between conversations through S3.1d's `swap_to`.  It needs a conversation cache:
    /// a slot switch IS a save/restore, so with parking off the engine falls back to the serial driver
    /// (and says so).  The GPU is still serialised - one engine thread issues windows - so what improves
    /// is latency and the fixed cost of prompt reads, not tokens per second.
    int serve_slots = 0;
    /// S3.1b (--starve-ms): the fairness bound of the pick rule (§3.3) - how long a slot may wait for the
    /// session before the scheduler swaps to it.  0 = never force a swap.  Tuning it is a measurement
    /// (OQ4), not a guess; the default is the design's starting point.
    int64_t starve_ms = 250;
    /// S3.10 (--decode-tokens N): how many tokens a slot may GENERATE on its turn before the scheduler
    /// may take the session away.  A decode step used to be exactly one verify window, which with MTP
    /// accepts 1..8 tokens - so a slot emitted ~1-3 tokens and then paid a full save+restore to hand
    /// the session to the next conversation (risk R10: the swap is 237 MB-2.25 GB of memcpy).  N is a
    /// budget in tokens, not windows: the driver runs windows back-to-back for this slot until it has
    /// produced N, EOS, `--max-new`, or a STOP.  0/1 = today's one-window-per-turn behaviour.
    int64_t decode_tokens = 0;
    /// S4.2 (--hold-ms N): how long a request the engine cannot run yet may WAIT before it is answered
    /// with an error.  Stage 4's "hold, don't reject": a request that is blocked by something that
    /// comes back - every slot busy, no free RAM, the mounted conversation mid-read, another slot
    /// holding the prompt loan - is queued and started when the resource frees, instead of getting an
    /// `ERR` on the spot.  Only a request that can NEVER run (a prompt past --max-context, a
    /// conversation whose snapshot exceeds the whole parking budget, --serve-slots >= 2 with parking
    /// off) is an error immediately.  N is the bound on the wait; 0 = wait forever, so only `STOP <id>`
    /// or the client's disconnect ends it.  The default is minutes, not seconds: the worst real queue
    /// on this box is a couple of 36 k-token prompt reads (~70 s each).
    int64_t hold_ms = 600000;
    /// The vision path: keep a per-cell (t, h, w) rotary position table so --serve can take GENI requests.
    bool vision = false;
    int adapt_swaps = 96;
    /// --serve: how many conversation checkpoints to keep between requests (0 = every request reads its whole
    /// prompt again, the v0.1.2 behaviour).  One is the GDN recurrence of the 36 layers, the QSA indexer tails and
    /// the PLE history (~118 MB of host RAM); the KV cache itself is positional and stays where it is.
    int prompt_cache = 6;
    /// --serve: host RAM the PARKED PREFIX cache may hold (see core/conversation_cache.hpp).  Stage 2 of the
    /// concurrency work: a client that switches between prompts - an agent that spawns subagents, each with its
    /// own system prompt - must not rebuild a prefix it will need again, so several prefixes live at once.
    /// Parking only saves the re-read; requests are still served one at a time (stage 3 is not implemented).
    /// 0 = park nothing (the behaviour before parking was on by default).  A cap, not a reservation:
    /// `conversation_cache_min_free_mib`
    /// below refuses to park when the machine cannot afford it.
    int64_t conversation_cache_mib = 8192;
    bool conversation_cache_mib_given = false;   ///< an explicit budget is never second-guessed by the machine-sized default
    /// How many prefixes may be parked at once.  Pruned least-recently-used first, so a prefix a client keeps
    /// mounting survives however many new ones arrive.
    int conversation_cache_slots = (int) strata::core::ConversationCache::default_slots;
    int64_t conversation_cache_min_free_mib = 2560;
    /// --serve: also keep a checkpoint every N freshly read prompt tokens (0 = only at the last turn boundary)
    int64_t prompt_cache_every = 16384;
    /// --serve: a prompt read from token 0 is also checkpointed at its first turn boundary - the end of the system
    /// prompt, which every chat of the same client shares - when that is at least N tokens (0 = never)
    int64_t prompt_cache_root = 2048;
    /// --serve: the token that opens a chat turn (<|im_start|>).  The last one in a prompt is where the chat's
    /// history ends and the new assistant turn begins, which is the checkpoint the next request can reuse.
    int64_t turn_token = 248045;
    /// --serve: a text part of the prompt of at most N tokens (a chat message, the assistant header, a short tool
    /// result) goes through the verify windows, S tokens at a time, instead of the batched prompt path (0 = always
    /// the batched path)
    int64_t short_read = 64;
    /// The suffix drafter (prompt lookup): when the text being written repeats an earlier stretch of the context (code
    /// edits, quoted input, tool-call JSON) by at least this many tokens, the window may be filled with what followed
    /// it there instead of the MTP's drafts, where the MTP's own first guess agrees and the draft policy expects it to
    /// pay (strata/spec/draft_policy.hpp).  On by default; 0 = MTP only.
    int suffix_draft = 3;
    /// The MTP's own window cap (0 = --spec): with --spec 6 --mtp-max-t 4 the long windows come from suffix matches.
    int mtp_max_t = 0;
    /// A control vector on the residual stream (strata/kernels/cvec.hpp), with llama.cpp's flags: the
    /// `experimental-speed-projection` profile passes `--control-vector-scaled FILE:1.0 --control-vector-layer-range
    /// 4 44 --cvec-mode project --cvec-dir per-layer`.  None by default; --serve switches a loaded one per request.
    std::vector<std::pair<std::string, float>> cvec_files;
    int cvec_first = -1, cvec_last = -1;   ///< llama.cpp's defaults: 1 .. the last layer
    int cvec_mode = 1;                     ///< 0 = project, 1 = add (llama.cpp's default)
    int cvec_single = -1;                  ///< --cvec-dir single:L (project mode): layer L's direction everywhere
};

void usage() {
    std::fprintf(stderr,
                 "strata generate --pack DIR --tokens \"1,2,3\" [options]\n"
                 "\n"
                 "  --pack DIR           the pack directory (default pack/full)\n"
                 "  --tokens LIST        the prompt as comma-separated token IDS (required)\n"
                 "  --tokens-file PATH   pretokenized prompt, commas or whitespace (alternative to --tokens)\n"
                 "  --ple-gguf PATH      required PLE table (original second GGUF shard)\n"
                 "  --no-ple             explicit diagnostic ablation of the PLE layer\n"
                 "  --ple-io direct|mmap|ram  n-gram table reads (plan v0.3 P2). direct (default): unbuffered SSD\n"
                 "                       reads, the table never enters RAM or the file cache; mmap: A/B arm;\n"
                 "                       ram: mmap with the whole table locked in RAM at start (Linux/macOS)\n"
                 "  --ple-row-cache N    bounded cache of fetched rows, 90 B each (default 1048576; 0 = off)\n"
                 "  --ple-inflight N     outstanding SSD reads (default 64)\n"
                 "  --ple-delay-us U     fault injection: each row read completes no earlier than U us\n"
                 "  --ple-sync-submit    A/B arm: submit table reads on the token thread (default: an I/O thread)\n"
                 "  --kv fp16|int8       KV storage (plan v0.3 P7): int8 codes + fp16 scale per 64 values, half the\n"
                 "                       VRAM; default fp16 until gate G-C accepts int8\n"
                 "  --kv q4_0            4-bit K/V after a Hadamard rotation (PR #21): half of int8's memory,\n"
                 "                       slightly lower precision (see bench/results/2026-09-27-kv-q4)\n"
                 "  --kv k8v4            hybrid: INT8 K (exact attention scores) + rotated Q4_0 V, 816 B/cell\n"
                 "                       (vs int8's 1,056); not with --kv-resident\n"
                 "  --kv-resident N      KV streaming: keep N cells of each QSA layer in VRAM (min 20480) and the\n"
                 "                       whole K/V in pinned RAM; the freed VRAM goes to expert slots. 0 (default):\n"
                 "                       all of it in VRAM. A context of N cells or fewer is not streamed\n"
                 "  --stream-token       enqueue token work on the session stream (experimental)\n"
                 "  --check-logits       copy and check all logits in the stream-token path\n"
                 "  --gr-fp32-activations  experimental CUDA-oracle GR activation precision\n"
                 "  --gr-native-mmvf      experimental pinned GR norm/projections; implies FP32 activations\n"
                 "  --native-bf16         experimental CUDA-oracle SSM/router/indexer BF16 projections\n"
                 "  --native-bf16-extra   experimental CUDA-oracle PLE/shared gate BF16 projections\n"
                 "  --native-ple-key      experimental native PLE key; requires --native-dense-gguf\n"
                 "  --native-moe-combine  experimental pinned CUDA routed/shared combination\n"
                 "  --native-gdn          experimental pinned CUDA GDN norms/gates/recurrence\n"
                 "  --native-flash-attn-short  diagnostic pinned vector attention; --max-context <=256\n"
                 "  --native-qsa-indexer  experimental pinned indexer key cache and pooling\n"
                 "  --native-qsa          experimental pinned QSA normalization and output gate\n"
                 "  --native-rope         experimental pinned text-only CUDA rotary arithmetic\n"
                 "  --native-ple-postops  experimental pinned PLE postprojection arithmetic\n"
                 "  --native-router       experimental pinned CUDA 512-expert top-10 routing\n"
                 "  --cpu-oracle-q8-0     experimental pinned CPU expert quantization and dot reduction\n"
                 "  --native SHARD1      every full-context native path at once (plan v0.3 P1): stream-token,\n"
                 "                       GR MMVF, BF16, head, dense + PLE key, MoE combine, GDN, router, QSA,\n"
                 "                       indexer, RoPE, PLE postops, and the CPU q8_0 contract unless the\n"
                 "                       expert cache is on. Individual --native-* flags stay for A/B.\n"
                 "  --native-head-gguf PATH  native Q5_K head from model shard 1; requires --stream-token\n"
                 "  --native-dense-gguf PATH native GDN/QSA/shared projections; repeat for each source model shard\n"
                 "  --expert-cache-cpu-order  experimental GPU expert reduction matching CPU order\n"
                 "  --max-new N          tokens to generate (default 16)\n"
                 "  --max-context N      KV/state capacity (default 4096)\n"
                 "  --rope-scaling T     extend the context past the trained one: none, linear\n"
                 "                       (position interpolation) or yarn - llama.cpp's types and names.\n"
                 "                       Default: the model file's rope keys, else none. Fixed at startup:\n"
                 "                       K in the cache is post-RoPE, so one run one scaling\n"
                 "  --rope-scale F       the extension factor for linear/yarn (default: the model file's\n"
                 "                       factor, else 1 = off)\n"
                 "  --rope-freq-base N   the raw ggml knobs: the frequency base (0 = the model's 1e7) and\n"
                 "  --rope-freq-scale F  the angle shrink (0 = 1/--rope-scale)\n"
                 "  --yarn-orig-ctx N    the trained context the correction targets (0 = 262144)\n"
                 "  --yarn-ext-factor F --yarn-attn-factor F --yarn-beta-fast F --yarn-beta-slow F\n"
                 "                       YaRN's knobs; defaults: -1 (auto: 1 for yarn), 1, 32, 1\n"
                 "  --greedy             argmax (the default)\n"
                 "  --seed S             enable sampling with this Philox seed\n"
                 "  --top-k N --top-p F --temperature F\n"
                 "  --dump-logits PATH   write one line of raw logits per position\n"
                 "  --logits-stride N    store every Nth row plus final input (default 1); N>1 requires --max-new 1\n"
                 "  --dump-residual PATH write the final R (hc x n_embd, f32) for head bisection\n"
                 "  --dump-layers PATH   write R after EVERY layer, per position: the C1 bisection ladder\n"
                 "  --dump-halves PATH   write both halves' block_out and inject per layer: the half bisection\n"
                 "  --dump-routing PATH  write the routed expert ids and weights per layer per position (P0.S8)\n"
                 "  --no-capture         run the layers directly instead of replaying graphs\n"
                 "  --shared-late        A/B: shared expert after the CPU pool (default: overlapped with it)\n"
                 "  --keep-canonical     A/B: also load canonical copies of natively served tensors (more VRAM)\n"
                 "  --vision             --serve takes images too (GENI requests; embeddings from strata-vision)\n"
                 "  --serve-slots N      --serve: how many conversations may be ACTIVE at once (default 0 = today's\n"
                 "                       serial engine, byte-identical on the wire).  2..8 turn on the stage-3 wire:\n"
                 "                       READY gains slots=N, every per-request line gains #<id>, STOP can name a\n"
                 "                       request, and the engine emits SLOT transition lines for /slots.  Each active\n"
                 "                       slot past the first costs a parked conversation's worth of host RAM plus a\n"
                 "                       session's worth of KV streaming state, so refuse rather than overcommit\n"
                 "                       (docs/STAGE3-CONCURRENCY.md §2.3, risks R1/R2)\n"
                 "  --starve-ms N        --serve-slots >= 2: how long a conversation may wait for the session before\n"
                 "                       the scheduler swaps to it (default 250; 0 = never force a swap).  A swap is\n"
                 "                       a save+restore of the whole conversation, so this trades fairness against\n"
                 "                       memcpy: 8-15 s per prompt chunk against ~24 ms per decode window\n"
                  "  --decode-tokens N    --serve-slots >= 2: how many tokens a conversation may GENERATE on its turn\n"
                  "                       before the scheduler may hand the session to the next one (default 0 = one\n"
                  "                       verify window per turn, which with MTP is ~1-3 tokens).  N is a token budget,\n"
                  "                       not a window count: the slot keeps running windows until it has produced N,\n"
                  "                       hits EOS, reaches --max-new, or is STOPped.  Raise it to cut swap churn\n"
                  "                       (each swap is a full save+restore, risk R10); lower it for fairer interleaving\n"
                  "  --hold-ms N          --serve-slots >= 2: how long a request the engine cannot run YET may WAIT\n"
                  "                       before it is answered with an error (default 600000 = 10 minutes; 0 = wait\n"
                  "                       forever, so only STOP or the client's disconnect ends it).  Stage 4's\n"
                  "                       `hold, don't reject`: a request blocked by something that comes back - every\n"
                  "                       slot busy, no free RAM for another conversation, the mounted conversation\n"
                  "                       mid-read, another slot holding the prompt loan, an image request running\n"
                  "                       alone - is QUEUED and started when the resource frees, not refused.  Only a\n"
                  "                       request that can NEVER run is an error on the spot: a prompt past\n"
                  "                       --max-context, a conversation whose snapshot would exceed the whole parking\n"
                  "                       budget, or --serve-slots >= 2 with the conversation cache off.  Every wait\n"
                  "                       shows up as `waiting=N` in the activity line, on WAIT lines, and in\n"
                  "                       serve/server.py's /status, /slots and /metrics\n"
                 "  --prompt-cache N     --serve: keep N conversation checkpoints between requests (default 6, ~118 MB\n"
                 "                       of RAM each; 0 = read every prompt from the start)\n"
                 "  --conversation-cache-mib N  --serve: RAM budget for PARKED PREFIXES (default 8192; 0 = park none)\n"
                 "                       a parked conversation costs what it actually holds, never what\n"
                 "                       --max-context allows it to grow to; the startup line prints the rate\n"
                 "  --conversation-cache-slots N  --serve: how many prefixes may be parked at once (default 8);\n"
                 "                       the least recently used one is pruned first, so a prefix a client keeps\n"
                 "                       mounting survives however many new ones arrive\n"
                 "  --conversation-cache-min-free-mib N  --serve: physical RAM floor when parking (default 2560)\n"
                 "  --prompt-cache-every N  --serve: also checkpoint every N fresh prompt tokens (default 16384, 0 = off)\n"
                 "  --turn-token ID      --serve: the token that opens a chat turn (default 248045, <|im_start|>)\n"
                 "  --short-read N       --serve: read at most N fresh text tokens through the decode windows instead\n"
                 "                       of the batched prompt path (default 64, 0 = off)\n"
                 "  --suffix-draft N     prompt lookup: draft from an earlier repeat of the last N+ tokens of context\n"
                 "                       when it pays (default 3; 0 = MTP only)\n"
                 "  --mtp-max-t M        cap the MTP's windows at M tokens (0 = --spec; longer ones come from suffixes)\n"
                 "  --control-vector-scaled FILE:SCALE[,...]  a control vector GGUF on the residual stream (llama.cpp's\n"
                 "                       format; --control-vector FILE = scale 1).  --serve: requests switch it (cvec=0|1)\n"
                 "  --control-vector-layer-range A B  the layers it follows (inclusive; default 1 .. the last)\n"
                 "  --cvec-mode add|project  h += s v (default) or h -= s (h.v) v with v unit\n"
                 "  --cvec-dir per-layer|single:L  each layer's own direction (default) or layer L's everywhere (project)\n"
                 "  --no-token-graph     A/B: two graphs per layer (the host launches each) instead of one per token\n"
                 "  --no-fused-gr        A/B: the six-kernel hyper-connection read and a separate write (native)\n"
                 "  --prefill CHUNK      batched prompt processing in chunks of CHUNK tokens (needs --native); auto =\n"
                 "                       the largest chunk up to 8192 whose buffers the expert cache can lend\n"
                 "                       --serve: the prompt path BORROWS expert-cache slots for its chunk buffers\n"
                 "                       and gives them back after the read.  That loan, not the read, is most of\n"
                 "                       the per-request fixed cost (bench/prefill/README.md).  Its knobs are env,\n"
                 "                       read once at startup; the behaviour ones default to the cheaper form,\n"
                 "                       and each reverts to the old one when set:\n"
                 "                         STRATA_PREFILL_STICKY_LOAN=0  hand the loan back and re-take it even\n"
                 "                                 when the next segment needs the layout still lying in the cache\n"
                 "                         STRATA_RES_UPLOAD_ALWAYS=1  re-upload the residency table on every lend\n"
                 "                                 and refill whether or not its content changed\n"
                 "                         STRATA_PREFILL_LOAN_TIMING=1  one stderr line per lend and per refill,\n"
                 "                                 so the loan's cost is readable from the serve log.  S3.2b\n"
                 "                                 extends it: a refill line says EAGER or LAZY, each pump batch\n"
                 "                                 prints its own, and the per-request line splits the rows into\n"
                 "                                 refilled / out / pumped home / still out of the cache\n"
                 "                         STRATA_PREFILL_LAZY_LOAN  --serve: give the prompt loan back LAZILY.\n"
                 "                                 Unset (the default) = on under the concurrent driver\n"
                 "                                 (--serve-slots >= 2 with the slot hand-over and the parking\n"
                 "                                 cache both live), off on the serial path, which stays 0.1.30's\n"
                 "                                 byte for byte.  =1 forces it on everywhere (the A/B arm), =0\n"
                 "                                 forces it off.  The end-of-request refill - 4.95 GiB per stage\n"
                 "                                 here, ~3.8 s on the slowest link - stops being a prompt cost:\n"
                 "                                 the rows stay marked non-resident (decode runs them on the CPU\n"
                 "                                 pool, which is correct), the layout and the ledger survive into\n"
                 "                                 the NEXT request, and a bounded pump walks the hottest rows\n"
                 "                                 home between decode steps.  A row is marked resident only after\n"
                 "                                 its copy is confirmed landed, so no window ever reads a slot\n"
                 "                                 that does not hold its expert.  Not only a timing change: a row\n"
                 "                                 still out is computed by the CPU instead of the GPU, and the\n"
                 "                                 two round differently (see --expert-cache's parity note)\n"
                 "                         STRATA_PREFILL_LOAN_PUMP_ROWS N  expert rows per cache the pump may\n"
                 "                                 have in flight at a time (default 8; 0 = never pump, so a\n"
                 "                                 ledger is only drained when something needs the cache whole)\n"
                 "                         STRATA_PARK_REFUSALS N / STRATA_PARK_QUIET N  after N consecutive\n"
                 "                                 RAM-admission parking refusals, stop estimating for N requests\n"
                 "                                 (defaults 3 and 64; STRATA_PARK_REFUSALS=0 always asks)\n"
                 "                         STRATA_NO_SWAP=1  --serve-slots >= 2: never hand the session to another\n"
                 "                                 conversation.  The wire stays id-tagged, but every request runs\n"
                 "                                 against the one mounted slot - 0.1.30's serial behaviour, no\n"
                 "                                 recompile.  A swap that does run prints the bytes it moved and\n"
                 "                                 its ms on stderr; INFO reports `slot_swap=0|1`\n"
                 "                         STRATA_SERVE_TRACE=1  --serve-slots >= 2: one stderr line per scheduler\n"
                 "                                 decision - every pick with `Pick::why`, every slot phase\n"
                 "                                 transition, every swap with the reason it was asked for, and every\n"
                 "                                 loan/parking deferral.  Off by default so the normal log stays\n"
                 "                                 readable.  The parking numbers, `resumed from N tokens` and\n"
                 "                                 `re-reading from token 0` are unconditional (one line per decision)\n"
                 "                                 because they are the ones that explain a request that ended early\n"
                 "                         STRATA_SERVE_ACTIVITY_S N  --serve-slots >= 2: print the periodic\n"
                 "                                 `activity:` line every N seconds (default 30; 0 = only when the\n"
                 "                                 active set changes).  `slots_active` prints only on a CHANGE, so a\n"
                 "                                 run that never got past one conversation looks identical to one\n"
                 "                                 that was idle; the activity line carries `peak=`, which tells them\n"
                 "                                 apart\n"
                 "  --no-pool            skip the CPU expert pool (the GPU-only floor)\n"
                 "  --sync-every-layer   debug: synchronise after every layer\n"
                 "  --ple-gguf PATH      the n-gram/PLE shard.  WITHOUT IT LAYER 1's PLE IS SILENTLY SKIPPED,\n"
                 "                       which changes every number downstream - pass it for any real run\n"
                 "  --dump-mixed PATH    write the post-attention residual (n_embd, f32)\n"
                 "  --stage-timing       per-stage KERNEL-COUNT shares.  NOT a time profile: an uncaptured\n"
                 "                       event interval includes host gaps, so run with --gpu-only-full first\n"
                 "  --graph-only         MEASURE: replay the 48 `pre` graphs only.  OMITS the 48 `post` graphs\n"
                 "                       and the LM head, so it is NOT the GPU floor (R0.3, Memory/ERRORS.md A4)\n"
                 "  --gpu-only-full      MEASURE: replay pre+post for all 48 layers plus the LM head, no pool.\n"
                 "                       THE TRUE PER-TOKEN GPU FLOOR.  Quote this one, not --graph-only.\n"
                 "  --stats              print the per-stage breakdown\n"
                 "  --gpu-stages         R0.9: capture the layer as three graphs (mixer / ffn+router / post)\n"
                 "                       and time them from OUTSIDE the capture.  The per-stage table on the\n"
                 "                       real graph that --stage-timing cannot give.  Prints and exits.\n"
                 "  --expert-profile P   R4.2e: pre-load the VRAM tier from a `profile.bin` (see\n"
                 "                       tools/make_profile.py) instead of admitting on first use.\n"
                 "  --no-hit-poke        R4.2d's A/B arm.  The hit path pokes the driver once right after its\n"
                 "                       launch so the GPU starts while the CPU pool runs; without it the work\n"
                 "                       waits for the next driver entry and does not overlap at all.\n"
                 "  --no-ple-prefetch     A/B arm: read the PLE table's sixteen rows one at a time, instead of\n"
                 "                       issuing them in one PrefetchVirtualMemory call.\n"
                 "  --expert-cache N     R4: keep N expert blobs resident in VRAM and compute their rows on the\n"
                 "                       GPU via `moe_hit_grouped_s2`.  DEFAULT 0.  Measured at 4096 slots\n"
                 "                       with --expert-cache-per-layer: 54.4%% hits, CPU pool drain 19.1 -> 10.3\n"
                 "                       ms/token, -2.7 ms/token end to end.\n"
                 "  --expert-cache-device1 N  pre-fill N experts on CUDA1 (experimental)\n"
                 "  --expert-cache-device2 N  pre-fill N more experts on CUDA2\n"
                 "  --expert-cache-device3 N  pre-fill N more experts on CUDA3\n"
                 "  --expert-cache-remote-placement stripe|layer  distribute expert ranks or whole\n"
                 "                       layers across CUDA1..3 (default: stripe)\n"
                 "  --expert-cache-per-layer  R4.2g: give each layer its OWN slots instead of letting the first\n"
                 "                       position take all of them.  The default policy fills in arrival order\n"
                 "                       from one shared counter, so 256 slots went to ~26 layers of position 0\n"
                 "                       and measured **2.97%%**.  Per-layer, the same routing gives 21.4%% at 8\n"
                 "                       slots/layer and 70.4%% at 64.\n"
                 "  --no-host-worker     R2.2: the A/B arm.  By default the HOST THREAD joins the drain, so the\n"
                 "                       pool is six threads on six cores instead of five plus an idle core;\n"
                 "                       this flag restores the five-worker form for comparison on `pool phases`.\n"
                 "  --pool-workers N     R2.2: CPU expert pool worker count.  Default 0 = every physical core\n"
                 "                       except the one the host loop spins on.  A sweep is how the pool's\n"
                 "                       deviation from `cpu_s2` is attributed.\n"
                 "  --mmap-experts       R2.1: opt OUT of the resident expert arena, back to MapViewOfFile.\n"
                 "                       The A/B arm: the mmap's rate depends on the OS page cache holding\n"
                 "                       34 GB, and measured 71.97 vs 34.78 ms/token cold vs warm.\n"
                 "  --shared-expert-arena FILE  Linux: back the resident arena with one MAP_SHARED file.\n"
                 "                       Put this file on /dev/shm, not ordinary SSD storage.\n"
                 "                       A small header binds an existing backing file to the same pack.\n"
                 "  --resident-cpu-experts  with mmap and a static profile, keep the experts the GPU cache does not\n"
                 "                       hold resident in ordinary RAM (and the prompt path's lend region as far as\n"
                 "                       RAM allows); adaptive swaps exchange them, so none is read from the file again.\n"
                 "  --resident-experts   the low-RAM PC's resident mode (setup): --mmap-experts --resident-cpu-experts\n"
                 "                       with the copy page-locked when possible, 4 GiB headroom, plain mmap if it\n"
                 "                       does not fit.  Same answers as --mmap-experts for the same placement.\n");
}

/// All shards of a split GGUF, from shard 1's path ("...-00001-of-00002.gguf"); just the path when it is not split.
std::vector<std::string> model_shards(const std::string& first) {
    const std::string tag = "-00001-of-";
    const size_t at = first.rfind(tag);
    if (at == std::string::npos || first.size() < at + tag.size() + 10) return {first};
    const int total = std::atoi(first.substr(at + tag.size(), 5).c_str());
    std::vector<std::string> out;
    for (int i = 1; i <= total && i <= 99; ++i) {
        char num[8];
        std::snprintf(num, sizeof num, "%05d", i);
        std::string p = first;
        p.replace(at + 1, 5, num);
        if (std::ifstream(p, std::ios::binary)) out.push_back(p);
    }
    return out.empty() ? std::vector<std::string>{first} : out;
}

bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::string text(s);
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

/// The pool's adapter plus the wall-clock it spent, so the report can say how much of the token was the CPU.
struct Drive {
    strata::core::ExpertDispatch d;
    double cpu_ms = 0;
    int64_t calls = 0;
    /// THE ROUTING TRACE, which is P0.S8 and a stated prerequisite of Phase 3.  `drive_pool` is called once
    /// per layer from the main loop - the workers live inside `expert_pool_dispatch` - so a single FILE* here
    /// needs no locking.  `d.layers` is the CURRENT layer on entry (the adapter increments it as it walks the
    /// blob), which is why the layer index comes from there rather than from a counter of our own.
    std::FILE* routing = nullptr;
};

void drive_pool(void* user, const float* x_f, const int32_t* ids, const float* weights, int64_t n_embd, int64_t k,
                float* out) {
    Drive* t = (Drive*) user;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch(&t->d, x_f, ids, weights, n_embd, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // THE ROUTING TRACE.  Written AFTER the dispatch so the layer index is still this layer's: `d.layers` is
    // advanced by the adapter as it consumes the blob, and reading it after the call is the same value the
    // dispatch used.  Record = int32 layer, int32 k, k int32 ids, k float weights.
    if (t->routing != nullptr) {
        // **`d.layers` HAS ALREADY BEEN ADVANCED BY THE TIME THIS RUNS, AND THE FIRST TRACE WAS OFF BY ONE
        // BECAUSE OF IT.**  The adapter walks the blob by incrementing `d.layers` as it consumes each layer's
        // experts, so after the dispatch it holds the NEXT layer's index.  `tools/make_profile.py` caught it
        // with a bounds check when the trace turned out to span 1..48 instead of 0..47.  The hit-rate CURVE was
        // unaffected - it is a per-layer split, and shifting every layer by one preserves both metrics - but
        // anything keyed on the layer index, which is exactly what a cache profile is, would have been wrong.
        const int32_t layer_idx = (int32_t) (t->d.layers - 1);
        if (layer_idx < 0 || layer_idx >= 48) {
            std::fprintf(stderr, "strata generate: the routing trace saw layer %d, outside 0..47\n", layer_idx);
            return;
        }
        const int32_t rec[2] = {layer_idx, (int32_t) k};
        std::fwrite(rec, sizeof rec, 1, t->routing);
        std::fwrite(ids, sizeof(int32_t), (size_t) k, t->routing);
        std::fwrite(weights, sizeof(float), (size_t) k, t->routing);
    }
}

/// Plan v0.3 P6: the pool for a verify window.
void drive_pool_multi(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    Drive* t = (Drive*) user;
    t->d.layers = layer;
    const Clock::time_point a = Clock::now();
    strata::core::expert_pool_dispatch_multi(t->d, x_f, ids, n_tok, k, out);
    t->cpu_ms += std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    ++t->calls;
    // the routing trace for the serve path: the same record format drive_pool writes (layer, k, ids, weights),
    // one record per token.  The multi dispatch fuses the router weights into the kernel and does not surface
    // them, so records carry unit weights: tools/make_profile.py ranks pairs by routed frequency, which is the
    // signal that matters; a one-shot --dump-routing run records true weights if a weighted ranking is wanted.
    if (t->routing != nullptr && layer >= 0 && layer < 48) {
        for (int64_t tok = 0; tok < n_tok; ++tok) {
            const int32_t rec[2] = {(int32_t) layer, (int32_t) k};
            std::fwrite(rec, sizeof rec, 1, t->routing);
            std::fwrite(ids + tok * k, sizeof(int32_t), (size_t) k, t->routing);
            static const float one[64] = {};   // k <= 64 in a verify window; zeros read as unit weights
            std::fwrite(one, sizeof(float), (size_t) k, t->routing);
        }
    }
}

/// Layer split: every verify stage shares one Drive (its counters, usage and failure flags); the GPU plan, the expert
/// cache and the PCIe share the pool uses for a layer are those of the stage that runs it.
struct SplitDrive {
    static constexpr int kMax = 8;
    Drive* base = nullptr;
    int n = 0;                                    ///< stages
    int64_t end[kMax] = {};                       ///< stage i runs the layers from end[i - 1] (0) below end[i]
    strata::core::GpuPlanSink* plan[kMax] = {};
    const uint8_t* cache_base[kMax] = {};
    const uint64_t* cache_slot_off[kMax] = {};
    int pcie_num[kMax] = {};
};
void drive_pool_split(void* user, const float* x_f, const int32_t* ids, int64_t n_tok, int64_t k, float* out,
                      int64_t layer) {
    SplitDrive* s = (SplitDrive*) user;
    int st = 0;
    while (st + 1 < s->n && layer >= s->end[st]) ++st;
    Drive& d = *s->base;
    d.d.plan = s->plan[st];
    d.d.cache_base = s->cache_base[st];
    d.d.cache_slot_off = s->cache_slot_off[st];
    d.d.pcie_num = s->pcie_num[st];
    drive_pool_multi(s->base, x_f, ids, n_tok, k, out, layer);
}

/// Layer split across GPUs: a later stage on its own device, with its own copy of the dense weights, a session, an
/// expert cache for its layers, a verify window and a prompt path; the last one also holds the head (the drafter
/// lives on its device too).
struct GpuStage {
    int dev = 0;
    int64_t lb = 0, le = 0;
    double pcie_frac = 0.0;
    strata::core::WeightTable wt;
    strata::core::NativeDense dense;
    strata::core::NativeHead head;
    strata::core::SessionState ss;
    cudaStream_t stream = nullptr;
    strata::core::ExpertCache cache;
    std::vector<std::pair<int32_t, int32_t>> profile;   ///< its layers' share of the profile, hottest first
    int32_t* d_res = nullptr;                            ///< the residency table on its device
    strata::core::Verifier ver;
    strata::prefill::Prefill sp;
    cudaStream_t adapt_stream = nullptr;
    cudaEvent_t adapt_ev = nullptr;
    bool adapt_live = false;                             ///< swaps of this request are in flight on it
    int32_t* mrope = nullptr;                            ///< --vision: the image-position table on its device
};

// ---- issue #31: what the watchdog prints before it stops a stalled engine
struct MemSample {
    unsigned long long faults = 0, rss_mib = 0, avail_mib = 0, commit_mib = 0;   // faults: hard (Linux) / all (Windows)
};
MemSample mem_sample() {
    MemSample m;
#if defined(_WIN32)
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof pmc)) {
        m.faults = pmc.PageFaultCount;
        m.rss_mib = pmc.WorkingSetSize >> 20;
        m.commit_mib = pmc.PagefileUsage >> 20;
    }
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof ms;
    if (GlobalMemoryStatusEx(&ms)) m.avail_mib = ms.ullAvailPhys >> 20;
#else
    if (std::FILE* f = std::fopen("/proc/self/stat", "r")) {
        char buf[4096];
        const size_t n = std::fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        std::fclose(f);
        const char* s = std::strrchr(buf, ')');   // fields after the command name: 3 state ... 12 majflt
        for (int field = 2; s && field < 12; ++field) s = std::strchr(s + 1, ' ');
        if (s) m.faults = std::strtoull(s + 1, nullptr, 10);
    }
    auto kb = [](const char* path, const char* key) -> unsigned long long {
        unsigned long long v = 0;
        if (std::FILE* f = std::fopen(path, "r")) {
            char line[256];
            const size_t kl = std::strlen(key);
            while (std::fgets(line, sizeof line, f))
                if (std::strncmp(line, key, kl) == 0) { v = std::strtoull(line + kl, nullptr, 10); break; }
            std::fclose(f);
        }
        return v;
    };
    m.rss_mib = kb("/proc/self/status", "VmRSS:") >> 10;
    m.commit_mib = kb("/proc/self/status", "VmSwap:") >> 10;
    m.avail_mib = kb("/proc/meminfo", "MemAvailable:") >> 10;
#endif
    return m;
}

void stall_report(std::FILE* f, uint64_t layers_during) {
    strata::core::Progress& p = strata::core::progress();
    std::fprintf(f, "strata serve: stall report (engine %s): stage \"%s %lld\" for %lld s; %llu layers served since the "
                    "last finished step (0 = stopped, more = slow)\n", STRATA_VERSION, p.where.load(),
                 (long long) p.detail.load(), (long long) ((strata::core::progress_now_ms() - p.since_ms.load()) / 1000),
                 (unsigned long long) layers_during);
    for (int pass = 0; pass < 2; ++pass) {
        if (pass == 1) {
            std::this_thread::sleep_for(std::chrono::seconds(2));
            std::fprintf(f, "  2 s later:\n");
        }
        if (auto fn = strata::core::diag_pool_fn().load()) fn(f);
        if (auto fn = strata::core::diag_verify_fn().load()) fn(f);
        const MemSample m = mem_sample();
        std::fprintf(f, "  memory: %llu MiB resident, %llu MiB %s, %llu MiB RAM available; %llu %s\n", m.rss_mib,
                     m.commit_mib,
#if defined(_WIN32)
                     "committed", m.avail_mib, m.faults, "page faults so far"
#else
                     "in swap", m.avail_mib, m.faults, "major page faults so far"
#endif
        );
        std::fflush(f);
    }
#if defined(_WIN32)
    // every thread's stack, to read against this build's PDB: a few MB beside the engine's working directory
    if (HMODULE dbg = LoadLibraryA("dbghelp.dll")) {
        using Fn = BOOL(WINAPI*)(HANDLE, DWORD, HANDLE, int, void*, void*, void*);
        if (auto write = (Fn) GetProcAddress(dbg, "MiniDumpWriteDump")) {
            char path[64];
            std::snprintf(path, sizeof path, "strata-stall-%lu.dmp", (unsigned long) GetCurrentProcessId());
            HANDLE h = CreateFileA(path, GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (h != INVALID_HANDLE_VALUE) {
                const int kThreadInfo = 0x1000;   // MiniDumpWithThreadInfo; MiniDumpNormal = 0
                const BOOL ok = write(GetCurrentProcess(), GetCurrentProcessId(), h, kThreadInfo, nullptr, nullptr, nullptr);
                CloseHandle(h);
                char full[MAX_PATH];
                if (!GetFullPathNameA(path, MAX_PATH, full, nullptr)) std::snprintf(full, sizeof full, "%s", path);
                std::fprintf(f, "  %s the thread stacks to %s (attach it to the issue)\n", ok ? "wrote" : "could not write",
                             full);
            }
        }
    }
#endif
}

/// STRATA_TRACE=1: the VRAM left at a step of the startup (finds what fills the card after the cache is sized)
void mem_mark(const char* where) {
    static const bool on = std::getenv("STRATA_TRACE") != nullptr;
    if (!on) return;
    size_t free_b = 0, total_b = 0;
    cudaMemGetInfo(&free_b, &total_b);
    std::fprintf(stderr, "strata trace: %lld MiB free after %s\n", (long long) (free_b >> 20), where);
}

int argmax(const std::vector<float>& v) {
    int best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i] > v[best]) best = (int) i;
    return best;
}

// ---- --serve's conversation cache.  A chat or an agent sends the whole conversation again with every request, and
// reading it again is what made a long session wait minutes for every turn.  What a sequence leaves behind splits in
// two, and only one half needs copying:
//   * POSITIONAL state - the KV cache of the 12 QSA layers and their pooled indexer keys, the draft layer's KV.  A
//     cell is written once for its position and read only by later positions (the block scores take `dead` for the
//     block being filled, never its pooled row), so rewinding to a position just means writing from there again.
//   * RUNNING state - the 36 GDN recurrences and conv histories, each QSA layer's indexer tail (the unfinished
//     block's raw keys) and the PLE's normalized history.  Each describes "everything so far" and cannot be
//     rewound, so a checkpoint is a copy of exactly these: ~118 MB, the same set the verifier snapshots to roll
//     back rejected drafts.
// A checkpoint is only valid while the positional cells below it still hold ITS tokens, so the serve loop keeps
// just the checkpoints that are prefixes of the tokens the session holds now.
using ImgKey = strata::core::ConversationImageKey;
using ConvCheckpoint = strata::core::ConversationCheckpoint;

uint64_t fnv1a(const void* data, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* p = (const uint8_t*) data;
    for (size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

using ConvStateSizes = strata::core::ConversationStateSizes;

ConvStateSizes conv_state_sizes(const strata::core::ModelGeometry& g, const strata::core::SessionState& ss) {
    ConvStateSizes z;
    std::string error;
    // a split stage's session owns only its layer range's state (see SessionState's carve note); the geometry
    // and the carve have already passed engine validation
    strata::core::conversation_session_sizes(g, ss, z, error);
    return z;
}

/// Copies the running state out (this session's carve only).  The caller has synchronized the device.
bool checkpoint_save(ConvCheckpoint& c, const strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    std::string error;
    if (strata::core::conversation_checkpoint_save(c, ss, g, error)) return true;
    std::fprintf(stderr, "strata serve: checkpoint save: %s\n", error.c_str());   // the caller's ERR has no reason
    return false;
}

/// Puts a checkpoint's running state back; the positional cells below it are the caller's to guarantee.
bool checkpoint_restore(const ConvCheckpoint& c, strata::core::SessionState& ss, const strata::core::ModelGeometry& g) {
    std::string error;
    if (strata::core::conversation_checkpoint_restore(c, ss, g, error)) return true;
    std::fprintf(stderr, "strata serve: checkpoint restore: %s\n", error.c_str());
    return false;
}

// --control-vector-scaled: llama.cpp's `common_control_vector_load` (every file's `direction.<l>` times its scale,
// summed; layer 0 has none) and `llama_adapter_cvec::apply` with the projection-mode patch (project: the unit
// direction and its norm as the scale), into the tables `cvec_upload` takes.  `summary` is what INFO reports.
bool load_control_vectors(const Options& o, const strata::core::ModelGeometry& g, std::string& summary, std::string& err) {
    const int64_t L = g.n_layers, N = g.n_embd;
    std::vector<float> data((size_t) (L * N), 0.0f);
    std::vector<bool> have((size_t) L, false);
    for (const auto& [path, scale] : o.cvec_files) {
        try {
            strata::GgufFile f(path);
            const strata::MetaValue* arch = f.get("general.architecture");
            if (arch == nullptr || arch->s != "controlvector") {
                err = path + ": not a control vector GGUF (general.architecture is not 'controlvector')";
                return false;
            }
            const strata::MetaValue* hint = f.get("controlvector.model_hint");
            if (hint != nullptr && hint->s != "qwen4exp")
                std::fprintf(stderr, "strata generate: %s was made for '%s', not qwen4exp\n", path.c_str(), hint->s.c_str());
            int found = 0;
            for (const strata::TensorInfo& t : f.tensors()) {
                if (t.name.rfind("direction.", 0) != 0) continue;
                const long l = std::strtol(t.name.c_str() + 10, nullptr, 10);
                if (l < 1 || l >= L) continue;   // layer 0 has no vector; past the model is ignored, as in llama.cpp
                if (t.type != 0 || t.elements() != (uint64_t) N) {
                    err = path + ": " + t.name + " must be " + std::to_string((long long) N) + " f32";
                    return false;
                }
                const float* src = reinterpret_cast<const float*>(f.tensor_data(t));
                for (int64_t j = 0; j < N; ++j) data[(size_t) (l * N + j)] += scale * src[j];
                have[(size_t) l] = true;
                ++found;
            }
            if (found == 0) { err = path + ": no direction.<layer> tensors"; return false; }
        } catch (const std::exception& e) {
            err = e.what();
            return false;
        }
    }
    const int first = o.cvec_first <= 0 ? 1 : o.cvec_first;
    const int last = (o.cvec_last <= 0 || o.cvec_last >= L) ? (int) L - 1 : o.cvec_last;
    const int single = o.cvec_mode == 0 ? o.cvec_single : -1;
    if (single >= 0 && (single >= L || !have[(size_t) single])) {
        err = "--cvec-dir single:" + std::to_string(single) + ": the vector has no direction for that layer";
        return false;
    }
    std::vector<float> dir((size_t) (L * N), 0.0f), s((size_t) L, 0.0f);
    int steered = 0;
    for (int64_t l = first; l <= last; ++l) {
        const int64_t src = single >= 0 ? single : l;
        if (!have[(size_t) src]) continue;
        const float* d = data.data() + (size_t) (src * N);
        if (o.cvec_mode == 0) {
            double nrm = 0.0;
            for (int64_t j = 0; j < N; ++j) nrm += (double) d[j] * d[j];
            nrm = std::sqrt(nrm);
            if (nrm <= 0.0) continue;
            s[(size_t) l] = (float) nrm;
            for (int64_t j = 0; j < N; ++j) dir[(size_t) (l * N + j)] = (float) (d[j] / nrm);
        } else {
            s[(size_t) l] = 1.0f;
            std::copy(d, d + N, dir.begin() + (size_t) (l * N));
        }
        ++steered;
    }
    if (steered == 0) { err = "the control vector has no direction in layers " + std::to_string(first) + ".." + std::to_string(last); return false; }
    if (!strata::kernels::cvec_upload(dir, s, o.cvec_mode, first, last, N, g.hc, err)) return false;
    summary = std::string(o.cvec_mode == 0 ? "project" : "add") + ":" + std::to_string(first) + "-" + std::to_string(last) +
              (single >= 0 ? ":single" + std::to_string(single) : "");
    // the line llama.cpp's patched build prints, so a log shows the same thing
    std::fprintf(stderr, "strata generate: control vector mode = %s, dir = %s, layers %d..%d (%d steered)\n",
                 o.cvec_mode == 0 ? "project" : "add", single >= 0 ? "single" : "per-layer", first, last, steered);
    return true;
}

// The effective host->device bandwidth of the PCIe link: copies from pinned host memory, as the expert arena's
// reads are.  The native default share (0.55) was measured on x16 links (~26-28 GB/s); a x8 card in a x8 slot
// carries about half of that.  Returns < 0 when the probe cannot run (then the caller keeps the default).
double probe_pcie_h2d_gbps() {
    constexpr size_t kBytes = 256ull << 20;
    constexpr int kIters = 4;
    void* h = nullptr;
    void* d = nullptr;
    cudaEvent_t ev0, ev1;
    if (cudaMallocHost(&h, kBytes) != cudaSuccess) return -1.0;
    if (cudaMalloc(&d, kBytes) != cudaSuccess || cudaEventCreate(&ev0) != cudaSuccess ||
        cudaEventCreate(&ev1) != cudaSuccess) {
        if (d != nullptr) cudaFree(d);
        cudaFreeHost(h);
        return -1.0;
    }
    std::memset(h, 0, kBytes);   // fault the pages in before timing
    cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);   // warmup: context up, copy engine primed
    cudaEventRecord(ev0);
    for (int i = 0; i < kIters; ++i) cudaMemcpyAsync(d, h, kBytes, cudaMemcpyHostToDevice);
    cudaEventRecord(ev1);
    const bool ok = cudaEventSynchronize(ev1) == cudaSuccess;
    float ms = 0.f;
    const bool timed = ok && cudaEventElapsedTime(&ms, ev0, ev1) == cudaSuccess && ms > 0.01f;
    const double bw = timed ? ((double) kIters * (double) kBytes / (ms * 1e-3)) / 1e9 : -1.0;
    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaFree(d);
    cudaFreeHost(h);
    return bw;
}

}  // namespace

int main(int argc, char** argv) {
    // **UNBUFFERED, BECAUSE THE INTERESTING OUTPUT IS THE OUTPUT BEFORE A CRASH.**  `stdout` redirected to a
    // pipe or a file is block-buffered, so a program that dies loses every line it had already printed - which
    // turns "it crashed at step 7" into "it crashed somewhere", and the difference is a debugging session.
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // Load every CUDA kernel when the context is created, before the expert cache takes the free VRAM.  With the
    // default lazy loading, a kernel first used mid-prompt (MMQ for IQ3_XXS at 64K+ on a 12 GB card) found no VRAM
    // left for its code and the engine ended ("out of memory: cudaFuncSetAttribute").  Costs ~30 MB of VRAM.
    if (std::getenv("CUDA_MODULE_LOADING") == nullptr) {
#if defined(_WIN32)
        _putenv_s("CUDA_MODULE_LOADING", "EAGER");
#else
        setenv("CUDA_MODULE_LOADING", "EAGER", 0);
#endif
    }
    Options o;
    bool have_tokens = false;
    bool have_logits_stride = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return argv[++i];
        };
        bool parsed = true;
        if (a == "--help" || a == "-h") { usage(); return 0; }
        else if (a == "--pack") o.pack = next("--pack");
        else if (a == "--tokens") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::string e;
            if (!parse_i64_list(next("--tokens"), o.tokens, e)) { std::fprintf(stderr, "%s\n", e.c_str()); return 2; }
            have_tokens = true;
        }
        else if (a == "--tokens-file") {
            if (have_tokens) { std::fprintf(stderr, "supply one token input only\n"); return 2; }
            std::ifstream input(next("--tokens-file"));
            if (!input) { std::fprintf(stderr, "cannot open token file\n"); return 2; }
            std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
            if (input.bad()) { std::fprintf(stderr, "cannot read token file\n"); return 2; }
            std::string error;
            if (text.find('\0') != std::string::npos || !parse_i64_list(text.c_str(), o.tokens, error)) {
                std::fprintf(stderr, "malformed token file: %s\n", error.c_str()); return 2;
            }
            have_tokens = true;
        }
        else if (a == "--max-new") o.max_new = std::atoll(next("--max-new"));
        else if (a == "--max-context") o.max_context = std::atoll(next("--max-context"));
        else if (a == "--rope-scaling") o.rope_scaling = next("--rope-scaling");
        else if (a == "--rope-scale") o.rope_scale = std::atof(next("--rope-scale"));
        else if (a == "--rope-freq-base") o.rope_freq_base = std::atof(next("--rope-freq-base"));
        else if (a == "--rope-freq-scale") o.rope_freq_scale = std::atof(next("--rope-freq-scale"));
        else if (a == "--yarn-orig-ctx") o.yarn_orig_ctx = std::atof(next("--yarn-orig-ctx"));
        else if (a == "--yarn-ext-factor") o.yarn_ext_factor = std::atof(next("--yarn-ext-factor"));
        else if (a == "--yarn-attn-factor") o.yarn_attn_factor = std::atof(next("--yarn-attn-factor"));
        else if (a == "--yarn-beta-fast") o.yarn_beta_fast = std::atof(next("--yarn-beta-fast"));
        else if (a == "--yarn-beta-slow") o.yarn_beta_slow = std::atof(next("--yarn-beta-slow"));
        else if (a == "--greedy") o.greedy = true;
        else if (a == "--seed") { o.seed = (uint64_t) std::atoll(next("--seed")); o.greedy = false; }
        else if (a == "--top-k") o.top_k = std::atoi(next("--top-k"));
        else if (a == "--top-p") o.top_p = (float) std::atof(next("--top-p"));
        else if (a == "--temperature") o.temperature = (float) std::atof(next("--temperature"));
        else if (a == "--dump-logits") o.dump_logits = next("--dump-logits");
        else if (a == "--logits-stride") {
            if (have_logits_stride) { std::fprintf(stderr, "--logits-stride must be supplied only once\n"); return 2; }
            if (!strata::program::logits_selection::parse_stride(next("--logits-stride"), o.logits_stride)) {
                std::fprintf(stderr, "--logits-stride requires a positive decimal int64\n"); return 2;
            }
            have_logits_stride = true;
        }
        else if (a == "--dump-residual") o.dump_residual = next("--dump-residual");
        else if (a == "--dump-mixed") o.dump_mixed = next("--dump-mixed");
        else if (a == "--dump-layers") o.dump_layers = next("--dump-layers");
        else if (a == "--dump-halves") o.dump_halves = next("--dump-halves");
        else if (a == "--dump-routing") o.dump_routing = next("--dump-routing");
        else if (a == "--ple-gguf") o.ple_gguf = next("--ple-gguf");
        else if (a == "--no-ple") o.no_ple = true;
        else if (a == "--ple-io") o.ple_io = next("--ple-io");
        else if (a == "--ple-row-cache") o.ple_row_cache = std::atoll(next("--ple-row-cache"));
        else if (a == "--ple-inflight") o.ple_inflight = std::atoi(next("--ple-inflight"));
        else if (a == "--ple-delay-us") o.ple_delay_us = std::atof(next("--ple-delay-us"));
        else if (a == "--ple-sync-submit") o.ple_sync_submit = true;
        else if (a == "--kv") o.kv = next("--kv");
        else if (a == "--kv-resident") o.kv_resident = std::atoll(next("--kv-resident"));
        else if (a == "--stream-token") o.stream_token = true;
        else if (a == "--check-logits") o.check_logits = true;
        else if (a == "--gr-fp32-activations") o.gr_fp32_activations = true;
        else if (a == "--gr-native-mmvf") o.gr_native_mmvf = true;
        else if (a == "--native-bf16") o.native_bf16 = true;
        else if (a == "--native-bf16-extra") o.native_bf16_extra = true;
        else if (a == "--native-ple-key") o.native_ple_key = true;
        else if (a == "--native-moe-combine") o.native_moe_combine = true;
        else if (a == "--native-gdn") o.native_gdn = true;
        else if (a == "--native-flash-attn-short") o.native_flash_attn_short = true;
        else if (a == "--native-qsa-indexer") o.native_qsa_indexer = true;
        else if (a == "--native-qsa") o.native_qsa = true;
        else if (a == "--native-rope") o.native_rope = true;
        else if (a == "--native-ple-postops") o.native_ple_postops = true;
        else if (a == "--native-router") o.native_router = true;
        else if (a == "--cpu-oracle-q8-0") o.cpu_oracle_q8_0 = true;
        else if (a == "--native") o.native_preset = next("--native");
        else if (a == "--native-head-gguf") o.native_head_gguf = next("--native-head-gguf");
        else if (a == "--native-dense-gguf") o.native_dense_gguf.push_back(next("--native-dense-gguf"));
        else if (a == "--no-capture") o.no_capture = true;
        else if (a == "--no-pool") o.no_pool = true;
        else if (a == "--sync-every-layer") o.sync_every_layer = true;
        else if (a == "--stage-timing") o.stage_timing = true;
        else if (a == "--graph-only") o.graph_only = true;
        else if (a == "--gpu-only-full") o.gpu_only_full = true;
        else if (a == "--pool-workers") o.pool_workers = std::atoi(next("--pool-workers"));
        else if (a == "--no-host-worker") o.no_host_worker = true;
        else if (a == "--no-ple-prefetch") o.no_ple_prefetch = true;
        else parsed = false;
        // The chain continues here in a second statement: one chain of 120+ `else if` passed MSVC's limit of 128
        // nested blocks (C1061).  The order of the tests and what each does are unchanged.
        if (!parsed) {
        if (a == "--expert-cache") {
            const std::string v = next("--expert-cache");
            o.expert_cache = (v == "auto") ? -1 : std::atoi(v.c_str());
        }
        else if (a == "--expert-cache-device1") o.expert_cache_remote[0] = std::atoi(next("--expert-cache-device1"));
        else if (a == "--expert-cache-device2") o.expert_cache_remote[1] = std::atoi(next("--expert-cache-device2"));
        else if (a == "--expert-cache-device3") o.expert_cache_remote[2] = std::atoi(next("--expert-cache-device3"));
        else if (a == "--expert-cache-remote-placement")
            o.expert_cache_remote_placement = next("--expert-cache-remote-placement");
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoi(next("--vram-reserve-mib"));
        else if (a == "--prefill") {
            const std::string v = next("--prefill");
            o.prefill_auto = v == "auto";
            o.prefill_chunk = o.prefill_auto ? 8192 : std::atoll(v.c_str());
        }
        else if (a == "--no-split-rows") o.no_split_rows = true;
        else if (a == "--no-prefill-borrow") o.no_prefill_borrow = true;
        else if (a == "--prefill-until") o.prefill_until = std::atoll(next("--prefill-until"));
        else if (a == "--dump-final-r") o.dump_final_r = next("--dump-final-r");
        else if (a == "--spec") o.spec = std::atoi(next("--spec"));
        else if (a == "--spec-oracle") o.spec_oracle = next("--spec-oracle");
        else if (a == "--spec-corrupt") o.spec_corrupt = std::atoi(next("--spec-corrupt"));
        else if (a == "--mtp") o.mtp = next("--mtp");
        else if (a == "--mtp-window") o.mtp_window = std::atoll(next("--mtp-window"));
        else if (a == "--pcie-frac") o.pcie_frac = std::atof(next("--pcie-frac"));
        else if (a == "--adapt-every") o.adapt_every = std::atoi(next("--adapt-every"));
        else if (a == "--spec-min-p") o.spec_min_p = std::atof(next("--spec-min-p"));
        else if (a == "--stop-eos") o.stop_eos = true;
        else if (a == "--spec-split") o.spec_split = true;
        else if (a == "--layer-split") o.layer_split = next("--layer-split");
        else if (a == "--split-device") o.split_device = next("--split-device");
        else if (a == "--pcie-mode") o.pcie_mode = next("--pcie-mode");
        else if (a == "--serve") o.serve = true;
        else if (a == "--serve-slots") o.serve_slots = std::atoi(next("--serve-slots"));
        else if (a == "--starve-ms") o.starve_ms = std::atoll(next("--starve-ms"));
        else if (a == "--decode-tokens") o.decode_tokens = std::atoll(next("--decode-tokens"));
        else if (a == "--hold-ms") o.hold_ms = std::atoll(next("--hold-ms"));
        else if (a == "--vision") o.vision = true;
        else if (a == "--prompt-cache") o.prompt_cache = std::max(0, std::atoi(next("--prompt-cache")));
        else if (a == "--conversation-cache-mib" || a == "--conversation-cache-slots" ||
                 a == "--conversation-cache-min-free-mib") {
            const std::string value = next(a.c_str());
            int64_t number = 0;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), number);
            const int64_t limit = a == "--conversation-cache-slots" ? INT32_MAX : INT64_MAX / (1024 * 1024);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || number < 0 || number > limit) {
                std::fprintf(stderr, "%s needs a nonnegative integer within range\n", a.c_str());
                return 2;
            }
            if (a == "--conversation-cache-mib") { o.conversation_cache_mib = number; o.conversation_cache_mib_given = true; }
            else if (a == "--conversation-cache-min-free-mib") o.conversation_cache_min_free_mib = number;
            else o.conversation_cache_slots = (int) number;
        }
        else if (a == "--prompt-cache-every") o.prompt_cache_every = std::max(0LL, std::atoll(next("--prompt-cache-every")));
        else if (a == "--prompt-cache-root") o.prompt_cache_root = std::max(0LL, std::atoll(next("--prompt-cache-root")));
        else if (a == "--turn-token") o.turn_token = std::atoll(next("--turn-token"));
        else if (a == "--short-read") o.short_read = std::max(0LL, std::atoll(next("--short-read")));
        else if (a == "--suffix-draft") o.suffix_draft = std::max(0, std::atoi(next("--suffix-draft")));
        else if (a == "--mtp-max-t") o.mtp_max_t = std::max(0, std::atoi(next("--mtp-max-t")));
        else if (a == "--control-vector") o.cvec_files.push_back({next("--control-vector"), 1.0f});
        else if (a == "--control-vector-scaled") {
            // FILE:SCALE, comma-separated; the LAST colon splits, so a Windows path (C:\...) keeps its drive
            std::stringstream list(next("--control-vector-scaled"));
            std::string item;
            while (std::getline(list, item, ',')) {
                const size_t colon = item.rfind(':');
                char* end = nullptr;
                const float sc = colon == std::string::npos ? 0.0f : std::strtof(item.c_str() + colon + 1, &end);
                if (colon == std::string::npos || colon == 0 || end == item.c_str() + colon + 1 || *end != '\0') {
                    std::fprintf(stderr, "--control-vector-scaled: expected FILE:SCALE, got '%s'\n", item.c_str());
                    return 2;
                }
                o.cvec_files.push_back({item.substr(0, colon), sc});
            }
        }
        else if (a == "--control-vector-layer-range") {
            o.cvec_first = std::atoi(next("--control-vector-layer-range"));
            o.cvec_last = std::atoi(next("--control-vector-layer-range"));
        }
        else if (a == "--cvec-mode") {
            const std::string m = next("--cvec-mode");
            if (m == "project") o.cvec_mode = 0;
            else if (m == "add") o.cvec_mode = 1;
            else { std::fprintf(stderr, "--cvec-mode: add or project, got '%s'\n", m.c_str()); return 2; }
        }
        else if (a == "--cvec-dir") {
            const std::string d = next("--cvec-dir");
            if (d == "per-layer") o.cvec_single = -1;
            else if (d.rfind("single:", 0) == 0) o.cvec_single = std::atoi(d.c_str() + 7);
            else { std::fprintf(stderr, "--cvec-dir: per-layer or single:L, got '%s'\n", d.c_str()); return 2; }
        }
        else if (a == "--no-spec-split") o.spec_split = false;
        else if (a == "--eos-ids") {
            std::string e;
            if (!parse_i64_list(next("--eos-ids"), o.eos_ids, e)) { std::fprintf(stderr, "--eos-ids: %s\n", e.c_str()); return 2; }
            o.stop_eos = true;
        }
        else if (a == "--adapt-swaps") o.adapt_swaps = std::atoi(next("--adapt-swaps"));
        else if (a == "--expert-cache-cpu-order") o.expert_cache_cpu_order = true;
        else if (a == "--expert-cache-per-layer") o.expert_cache_per_layer = true;
        else if (a == "--no-hit-poke") o.no_hit_poke = true;
        else if (a == "--expert-profile") o.expert_profile = next("--expert-profile");
        else if (a == "--gpu-stages") o.gpu_stages = true;
        else if (a == "--mmap-experts") o.mmap_experts = true;
        else if (a == "--shared-expert-arena") { o.shared_expert_arena = next("--shared-expert-arena"); o.shared_expert_arena_given = true; }
        else if (a == "--resident-cpu-experts") o.resident_cpu_experts = true;
        else if (a == "--resident-experts") {
            o.mmap_experts = o.resident_cpu_experts = o.resident_pin = o.resident_soft = true;
            o.resident_headroom = 4ull << 30;
            // A/B arms: STRATA_RESIDENT_PIN=0 keeps the copy pageable (the --resident-cpu-experts form);
            // STRATA_RESIDENT_HEADROOM_GIB=N leaves N GiB of the available RAM free instead of 4
            if (const char* v = std::getenv("STRATA_RESIDENT_PIN"); v != nullptr && std::string(v) == "0")
                o.resident_pin = false;
            if (const char* v = std::getenv("STRATA_RESIDENT_HEADROOM_GIB"); v != nullptr && std::atof(v) >= 0.0)
                o.resident_headroom = (uint64_t) (std::atof(v) * 1073741824.0);
        }
        else if (a == "--stats") o.stats = true;
        else if (a == "--shared-late") o.shared_late = true;
        else if (a == "--keep-canonical") o.keep_canonical = true;
        else if (a == "--no-token-graph") o.no_token_graph = true;
        else if (a == "--no-fused-gr") o.no_fused_gr = true;
        else if (a == "--no-fast-attn") o.no_fast_attn = true;
        else if (a == "--no-publish-kernel") o.no_publish_kernel = true;
        else if (a == "--no-fused-gdn") o.no_fused_gdn = true;
        else if (a == "--no-fast-select") o.no_fast_select = true;
        else {
            // An unknown flag is an ERROR and not a warning: a typo'd `--max-neww` that silently generated 16
            // tokens would look like a working run.
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            usage();
            return 2;
        }
        }
    }
    if (o.serve && o.conversation_cache_mib > 0 && (o.prompt_cache == 0 || o.conversation_cache_slots == 0))
        std::fprintf(stderr, "strata serve: warning: conversation caching is disabled by %s\n",
                     o.prompt_cache == 0 ? "--prompt-cache 0" : "--conversation-cache-slots 0");
    // S3.1b/S3.1c: --serve-slots.  0/1 = today's serial engine.  The ceiling is the registry's, and the
    // reason to REFUSE rather than over-commit is risk R1/R2 of docs/STAGE3-CONCURRENCY.md: every active
    // slot past the first costs ~1.4 GiB of VRAM-side session state and ~6.3 GiB of host RAM on the box
    // this is sized against, and under WDDM a full GPU does not fail, it pages - which stalls a verify
    // graph spinning on a host flag forever.
    if (o.serve_slots < 0 || o.serve_slots > (int) strata::program::slot::kMaxActive) {
        std::fprintf(stderr, "--serve-slots needs 0..%d (0 = today's serial engine)\n",
                     (int) strata::program::slot::kMaxActive);
        usage();
        return 2;
    }
    // S3.1e-2, risks R2/R12: a slot switch IS a save/restore (§5.3), and the save is the conversation
    // cache.  With parking off there is nothing to save, so "two slots" would mean two conversations
    // overwriting one session - plausible garbage, not concurrency.  Refuse rather than over-commit,
    // and refuse HERE, before the model load, so the owner is not made to wait a minute for the bad
    // news.  (The other way to end up with no budget - the machine-sized default rounding to zero - is
    // caught below, where that budget is computed.)
    if (o.serve && o.serve_slots >= 2 &&
        (o.prompt_cache == 0 || o.conversation_cache_slots == 0 || o.conversation_cache_mib == 0)) {
        std::fprintf(stderr, "--serve-slots %d needs a conversation cache: a slot switch saves one "
                             "conversation out and restores another (docs/STAGE3-CONCURRENCY.md §5.3), and "
                             "parking is off (--prompt-cache %lld, --conversation-cache-mib %lld or "
                             "--conversation-cache-slots %d). Use --serve-slots 0/1 for today's serial "
                             "engine, or give the cache a budget.\n",
                     o.serve_slots, (long long) o.prompt_cache, (long long) o.conversation_cache_mib,
                     o.conversation_cache_slots);
        usage();
        return 2;
    }
    // The same trap reached a different way: the DEFAULT budget is capped to the machine (stage 2), so
    // on a box with no RAM to spare `--conversation-cache-mib 8192` still resolves to 0 and parking is
    // off anyway.  Refuse that here too rather than after a minute of model loading.  An explicit
    // --conversation-cache-mib is never second-guessed, so it is not re-checked.
    if (o.serve && o.serve_slots >= 2 && !o.conversation_cache_mib_given && o.conversation_cache_mib > 0) {
        const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024;
        const uint64_t headroom = 4ull << 30;   // the same figure the default cap uses (generate.cpp's park budget)
        if (const auto avail = strata::core::conversation_available_memory();
            avail.has_value() && *avail <= floor + headroom) {
            std::fprintf(stderr, "--serve-slots %d needs somewhere to park a conversation: this machine has "
                                 "%.1f GiB free, which is under the %.1f GiB parking floor plus %.1f GiB of "
                                 "request headroom, so the parked-prefix budget resolves to 0 and a slot switch "
                                 "would have nothing to save. Free RAM, set --conversation-cache-mib N, or use "
                                 "--serve-slots 0/1.\n",
                         o.serve_slots, (double) *avail / 1073741824.0, (double) floor / 1073741824.0,
                         (double) headroom / 1073741824.0);
            return 2;
        }
    }
    if (o.starve_ms < 0) { std::fprintf(stderr, "--starve-ms needs a nonnegative integer\n"); return 2; }
    // S3.10: the per-turn decode budget.  0 = today's one window per turn; a negative value is a typo
    // for 0, and an absurd one (larger than any request can ask for) would make a slot hold the session
    // until its own --max-new, which is the serial engine wearing a slot costume.  The cap is the
    // context window: no request can produce more tokens than that.
    if (o.decode_tokens < 0) { std::fprintf(stderr, "--decode-tokens needs a nonnegative integer\n"); return 2; }
    if (o.hold_ms < 0) { std::fprintf(stderr, "--hold-ms needs a nonnegative integer (0 = wait forever)\n"); return 2; }
    if (o.serve && o.serve_slots >= 2 && o.decode_tokens > (int64_t) o.max_context) {
        std::fprintf(stderr, "--decode-tokens %lld is larger than --max-context %lld: a slot would hold the "
                             "session for its whole answer. Use a smaller value (or 0 for one window per turn).\n",
                     (long long) o.decode_tokens, (long long) o.max_context);
        return 2;
    }
    // Parking across a `--layer-split` (S3.1a): a parked conversation now carries every stage's
    // running state and K/V plus the MTP draft state, so a split parks like one GPU does. There is
    // deliberately no guard here any more - the split's stage set is handed to the snapshot calls
    // below, and if it did not tile the model the snapshot core refuses rather than parking a
    // conversation with state missing from it. `--conversation-cache-mib 0` still parks nothing.
    // Layer split (multi-GPU): the later stages run layers [K_i, K_i+1) on their own GPUs (--split-device, default
    // the next visible ones); "auto" places the K from each GPU's free VRAM once the weights are in (below).  Across
    // GPUs, not yet: KV streaming, images, control vectors, the helper caches (--expert-cache-remote), and lending
    // cache slots to the prompt path (each stage's prompt path has its own buffers).
    const bool pcie_given = o.pcie_frac >= 0.0;
    std::vector<int64_t> split_at;
    std::vector<int> split_devs;
    bool split_auto = false, split_same = false;
    if (!o.layer_split.empty()) {
        int n_dev = 1;
        if (cudaGetDeviceCount(&n_dev) != cudaSuccess || n_dev < 1) n_dev = 1;
        cudaGetLastError();
        auto ints = [](const std::string& str, auto& out) -> bool {
            using V = typename std::decay_t<decltype(out)>::value_type;
            size_t a = 0;
            while (a < str.size()) {
                size_t b = str.find(',', a);
                if (b == std::string::npos) b = str.size();
                const std::string t = str.substr(a, b - a);
                if (t.empty() || t.find_first_not_of("0123456789") != std::string::npos) return false;
                out.push_back((V) std::atoll(t.c_str()));
                a = b + 1;
            }
            return !out.empty();
        };
        split_auto = o.layer_split == "auto";
        bool ok = o.serve && (split_auto || ints(o.layer_split, split_at));
        if (ok && !o.split_device.empty()) ok = ints(o.split_device, split_devs);
        else if (ok)
            for (int d = 1; d < n_dev && (split_auto || split_devs.size() < split_at.size()); ++d) split_devs.push_back(d);
        if (ok && !split_auto && split_devs.empty() && split_at.size() == 1) split_devs.push_back(0);   // one GPU
        split_same = ok && split_devs.size() == 1 && split_devs[0] == 0 && !split_auto;
        if (ok && split_auto && split_devs.empty()) {
            std::fprintf(stderr, "strata generate: --layer-split auto: one GPU visible, so no split\n");
            o.layer_split.clear();
            split_auto = false;
        } else if (ok) {
            ok = (split_auto || split_at.size() == split_devs.size()) && split_devs.size() < (size_t) SplitDrive::kMax;
            for (size_t i = 0; ok && i < split_at.size(); ++i) ok = split_at[i] >= 2 && (i == 0 || split_at[i] > split_at[i - 1]);
            for (size_t i = 0; ok && !split_same && i < split_devs.size(); ++i) {
                ok = split_devs[i] > 0 && split_devs[i] < n_dev;
                for (size_t j = 0; ok && j < i; ++j) ok = split_devs[i] != split_devs[j];
            }
        }
        if (!ok) {
            std::fprintf(stderr, "strata generate: --layer-split K[,K2..]|auto needs --serve, rising K from 2, and one "
                                 "distinct GPU per K in --split-device (1..%d; or 0 with one K: the same GPU)\n", n_dev - 1);
            return 2;
        }
    }
    const bool multi_gpu = !split_devs.empty() && !split_same;
    if (o.mmap_experts && !o.shared_expert_arena.empty()) {
        // **THE SHARED ARENA BACKS THE RESIDENT ARENA, AND `--mmap-experts` HAS NONE.**  The mmap path reads the
        // pack's `experts.bin` through the OS file cache instead of holding a copy in RAM - which is what the
        // low-RAM mode does (`setup.py` passes `--resident-experts`, and that implies `--mmap-experts`).  Now
        // that the shared backing is ON by default, treating the two as a contradiction would stop every
        // low-RAM install from starting at all, over an option that simply does not apply to it.  So a DEFAULT
        // shared arena is quietly dropped when there is no resident arena to share; an EXPLICIT one is still a
        // contradiction the user should hear about.
        if (o.shared_expert_arena_given) {
            std::fprintf(stderr, "strata generate: --shared-expert-arena backs the resident arena and cannot be used with --mmap-experts\n");
            return 2;
        }
        o.shared_expert_arena.clear();
    }
    if (o.resident_cpu_experts && (!o.mmap_experts || o.expert_profile.empty())) {
        std::fprintf(stderr, "strata generate: --resident-cpu-experts requires --mmap-experts and a static --expert-profile\n");
        return 2;
    }
    if (o.resident_cpu_experts &&
        (!o.layer_split.empty() || o.expert_cache_remote[0] > 0 || o.expert_cache_remote[1] > 0 ||
         o.expert_cache_remote[2] > 0)) {
        std::fprintf(stderr, "strata generate: --resident-cpu-experts does not support layer splits or remote expert caches\n");
        return 2;
    }
    // the helper-GPU expert caches (--expert-cache-remote, docs/SECOND_GPU.md): CUDA1..3 on one GPU; with a layer
    // split, the visible GPUs no stage runs on, in order
    int remote_dev[3] = {1, 2, 3};
    if (multi_gpu) {
        if (o.expert_profile.empty()) {
            std::fprintf(stderr, "strata generate: a layer split across GPUs needs --expert-profile\n");
            return 2;
        }
        // A STAGE'S PROMPT PATH BORROWS FROM THAT STAGE'S OWN EXPERT CACHE.  This used to set
        // `no_prefill_borrow = true` - "each stage's prompt path has its own buffers" - which is true, but it is
        // a reason to give each stage its own LOAN, not a reason to make every stage withhold a chunk-sized
        // reserve from its cache for the whole session.  Forced on, it also collapsed `--prefill auto` to 2048
        // (below) and skipped the lend arm, so a stage paid for its prompt buffers twice over: once in VRAM it
        // never got back, once in the smaller chunk.  At `--prefill 5524` that reserve is 3.8 GiB per stage,
        // more than either 8 GB card had - which is how adding two GPUs to the two 12 GB ones lost 260K.
        // The loan is the tail of the stage's own cache (see `PfPart` in the serve block); outside the prompt
        // that tail is expert cache, so a large chunk costs a stage nothing permanent.
        int n_vis = 1;
        if (cudaGetDeviceCount(&n_vis) != cudaSuccess || n_vis < 1) n_vis = 1;
        cudaGetLastError();
        int next_free = 1;
        for (int r = 0; r < 3; ++r) {
            if (o.expert_cache_remote[(size_t) r] <= 0) continue;
            while (next_free < n_vis &&
                   std::find(split_devs.begin(), split_devs.end(), next_free) != split_devs.end()) ++next_free;
            if (next_free >= n_vis) {
                std::fprintf(stderr, "strata generate: --expert-cache-remote with a layer split needs a GPU that runs no "
                                     "stage (%d visible, %zu used by the split)\n", n_vis, split_devs.size() + 1);
                return 2;
            }
            remote_dev[r] = next_free++;
        }
        std::string devs;
        for (const int d : split_devs) devs += (devs.empty() ? "" : ",") + std::to_string(d);
        std::fprintf(stderr, "strata generate: layer split across %zu GPUs: CUDA0, then CUDA%s (split %s)\n",
                     split_devs.size() + 1, devs.c_str(), o.layer_split.c_str());
    }
#if defined(STRATA_USE_HIP)
    {
        // every GPU this run uses must be an architecture the binary has code for (a gfx1100 build on a gfx1201
        // card would otherwise fail later with "invalid device function")
        std::vector<int> used{0};
        if (multi_gpu) used.insert(used.end(), split_devs.begin(), split_devs.end());
        for (int r = 0; r < 3; ++r)
            if (o.expert_cache_remote[(size_t) r] > 0) used.push_back(remote_dev[r]);
        for (const int d : used) {
            if (const std::string why = strata::core::gpu_arch_problem(d); !why.empty()) {
                std::fprintf(stderr, "strata generate: %s\n", why.c_str());
                return 1;
            }
        }
    }
#endif
    if (o.prefill_auto && (o.no_prefill_borrow || o.expert_profile.empty())) {
        o.prefill_auto = false;       // nothing to lend from: the buffers are reserved for the session, so keep them small
        o.prefill_chunk = 2048;
    }
    if (!have_tokens && o.serve) {   // plan v0.3 P8: requests bring their own tokens
        o.tokens = {248045};
        o.max_new = 1;
        have_tokens = true;
        o.stop_eos = true;
    }
    if (!have_tokens) {
        std::fprintf(stderr, "strata generate: --tokens is required (this build has no tokenizer; see the "
                             "header of src/program/generate.cpp)\n");
        usage();
        return 2;
    }

    if ((o.ple_io != "direct" && o.ple_io != "mmap" && o.ple_io != "ram") || o.ple_row_cache < 0 || o.ple_inflight < 1 ||
        o.ple_inflight > 1024 || !(o.ple_delay_us >= 0)) {
        std::fprintf(stderr, "strata generate: invalid --ple-io/--ple-row-cache/--ple-inflight/--ple-delay-us\n");
        return 2;
    }
#if defined(_WIN32)
    if (o.ple_io == "ram") {
        std::fprintf(stderr, "strata generate: --ple-io ram is not available on Windows (no mlock); use --ple-io mmap\n");
        return 2;
    }
#endif
    if (o.kv == "q4") o.kv = "q4_0";
    if (o.kv != "fp16" && o.kv != "int8" && o.kv != "q4_0" && o.kv != "k8v4") {
        std::fprintf(stderr, "strata generate: --kv must be fp16, int8, q4_0 or k8v4\n");
        return 2;
    }
    strata::core::qsa_set_kv_int8(o.kv == "int8");
    strata::core::qsa_set_kv_q4(o.kv == "q4_0");   // PR #21: 4-bit codes after a Hadamard rotation (kv_q4.hpp)
    strata::core::qsa_set_kv_hybrid(o.kv == "k8v4");   // K8V4: INT8 K + rotated Q4_0 V, 816 B/cell
    if (o.kv_resident < 0) {
        std::fprintf(stderr, "strata generate: --kv-resident must be >= 0\n");
        return 2;
    }
    if (o.kv == "k8v4" && o.kv_resident > 0) {
        std::fprintf(stderr, "strata generate: --kv k8v4 does not support --kv-resident streaming (yet)\n");
        return 2;
    }
    strata::core::qsa_set_kv_resident(o.kv_resident);
    // Prompt lookup (the suffix drafter, on by default): the MTP keeps its --spec windows and a lookup window may be
    // up to 2 tokens longer; the draft policy (strata/spec/draft_policy.hpp) takes one only where it pays. Code
    // edits +6-11%, ordinary text unchanged (bench/results/2026-09-27-spec). --suffix-draft 0 turns it off.
    if (o.suffix_draft > 0 && o.spec >= 2 && o.mtp_max_t == 0) {
        o.mtp_max_t = o.spec;
        o.spec = std::min(o.spec + 2, 8);   // kVerifyMaxT
    }
    strata::core::layer_set_shared_early(!o.shared_late);
    if (!o.native_preset.empty()) {
        if (o.no_ple || o.ple_gguf.empty()) {
            std::fprintf(stderr, "strata generate: --native requires --ple-gguf (the PLE key is native too)\n");
            return 2;
        }
        o.stream_token = true;
        o.gr_native_mmvf = true;
        o.native_bf16 = o.native_bf16_extra = true;
        o.native_ple_key = o.native_moe_combine = o.native_gdn = o.native_router = true;
        o.native_qsa = o.native_qsa_indexer = o.native_rope = o.native_ple_postops = true;
        if (o.native_head_gguf.empty()) o.native_head_gguf = o.native_preset;
        if (o.native_dense_gguf.empty()) {
            // every shard of the model (<name>-0000N-of-0000M.gguf beside --native), then the PLE shard: a split
            // may put any layer in any shard (Swift's GGUFs: layers 13-47 in shard 2, the PLE table in shard 1)
            o.native_dense_gguf = model_shards(o.native_preset);
            if (std::find(o.native_dense_gguf.begin(), o.native_dense_gguf.end(), o.ple_gguf) == o.native_dense_gguf.end())
                o.native_dense_gguf.push_back(o.ple_gguf);
        }
        // Plan v0.3 (24 Sep): the CPU experts stay on the VNNI kernel.  The llama.cpp-CPU-exact q8_0 contract
        // cost 27.0 vs 17.2 ms/token of pool time and G-C does not need it; `--cpu-oracle-q8-0` still selects it.
    }
    if (o.logits_stride > 1 && (o.max_new != 1 || o.dump_logits.empty())) {
        std::fprintf(stderr, "strata generate: --logits-stride > 1 requires --max-new 1 and --dump-logits\n");
        return 2;
    }
    if (o.no_ple && !o.ple_gguf.empty()) {
        std::fprintf(stderr, "strata generate: --no-ple and --ple-gguf are mutually exclusive\n");
        return 2;
    }
    if (o.native_ple_postops && o.no_ple) {
        std::fprintf(stderr, "strata generate: --native-ple-postops requires PLE enabled\n");
        return 2;
    }
    if (!o.no_ple && o.ple_gguf.empty()) {
        std::fprintf(stderr, "strata generate: --ple-gguf is required; --no-ple explicitly enables a diagnostic ablation\n");
        return 2;
    }
    // P7 audit: positions, cells and pooled-block indices are cast to int32 on the device path.
    if (o.max_context > 2147483647LL - 8) {
        std::fprintf(stderr, "strata generate: --max-context must be below 2^31\n");
        return 2;
    }
    if (o.max_new <= 0 || o.max_context <= 0 || o.max_new > o.max_context ||
        o.tokens.size() > (size_t) (o.max_context - o.max_new)) {
        std::fprintf(stderr, "strata generate: positive --max-new and --max-context must fit the prompt and generation\n");
        return 2;
    }
    // THE ROPE KNOBS (rope_scaling.hpp).  Anything invalid dies here, at second zero, rather than becoming a
    // NaN angle inside one of the twelve QSA layers.  Only the RANGES are checked - the config itself is
    // resolved after the model file has had its say, right before session_init.
    strata::kernels::RopeScaling rope_cfg;   // type filled here; the rest at the resolution below
    {
        using RST = strata::kernels::RopeScalingType;
        // an absent --rope-scaling (the empty default) leaves the type to the model file's rope keys,
        // resolved below; anything present must be one of the three names
        if (o.rope_scaling == "none") rope_cfg.type = RST::None;
        else if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
        else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
        else if (!o.rope_scaling.empty()) {
            std::fprintf(stderr, "strata generate: --rope-scaling must be none, linear or yarn (got '%s')\n",
                         o.rope_scaling.c_str());
            return 2;
        }
        // every knob FINITE first: `atof("nan")` is NaN, and a NaN passes every range comparison below
        for (const double v : {o.rope_scale, o.rope_freq_base, o.rope_freq_scale, o.yarn_orig_ctx, o.yarn_ext_factor,
                               o.yarn_attn_factor, o.yarn_beta_fast, o.yarn_beta_slow})
            if (!std::isfinite(v)) {
                std::fprintf(stderr, "strata generate: a rope scaling knob is not a finite number (%g)\n", v);
                return 2;
            }
        // 0 is the absent default; an explicit factor must extend, not shrink
        if (o.rope_scale != 0 && o.rope_scale < 1.0) {
            std::fprintf(stderr, "strata generate: --rope-scale %g must be >= 1 (it extends the context, not shrinks it)\n",
                         o.rope_scale);
            return 2;
        }
        if (o.rope_freq_base != 0 && o.rope_freq_base <= 1.0) {
            std::fprintf(stderr, "strata generate: --rope-freq-base must be a base above 1 (0 = the model's)\n");
            return 2;
        }
        if (o.rope_freq_scale < 0 || o.yarn_orig_ctx < 0 || o.yarn_ext_factor < -1.0 || o.yarn_attn_factor <= 0 ||
            o.yarn_beta_fast <= 0 || o.yarn_beta_slow <= 0) {
            std::fprintf(stderr, "strata generate: invalid rope scaling knob (see usage: --yarn-ext-factor <0 = auto, "
                                 "--yarn-orig-ctx 0 = default, the rest positive)\n");
            return 2;
        }
    }
    if (!std::isfinite(o.temperature) || o.temperature < 0 || !std::isfinite(o.top_p) ||
        o.top_p <= 0 || o.top_p > 1 || o.top_k < 0 || o.expert_cache < -1 ||
        o.pool_workers < 0 || std::any_of(o.expert_cache_remote.begin(), o.expert_cache_remote.end(),
                                           [](int slots) { return slots < 0; }) ||
        (o.expert_cache_remote[1] > 0 && o.expert_cache_remote[0] == 0) ||
        (o.expert_cache_remote[2] > 0 && o.expert_cache_remote[1] == 0)) {
        std::fprintf(stderr, "strata generate: invalid sampling or resource parameter\n");
        return 2;
    }
    if (o.expert_cache_remote_placement != "stripe" && o.expert_cache_remote_placement != "layer") {
        std::fprintf(stderr, "strata generate: --expert-cache-remote-placement must be stripe or layer\n");
        return 2;
    }

    if (o.native_flash_attn_short && o.max_context > 256) {
        std::fprintf(stderr, "strata generate: --native-flash-attn-short requires --max-context <=256\n");
        return 2;
    }
    if (o.native_flash_attn_short && (o.gpu_only_full || o.graph_only || o.gpu_stages)) {
        std::fprintf(stderr, "strata generate: --native-flash-attn-short requires the normal decode loop for status validation\n");
        return 2;
    }
    // The whole-model graph measurements replay every layer through CUDA0's session, which a layer split carves to
    // CUDA0's own range - the answer would read another stage's state.  Refused here rather than at the call, so
    // the reason is visible before 55 GB is loaded.
    if (multi_gpu && (o.gpu_only_full || o.gpu_stages)) {
        std::fprintf(stderr, "strata generate: --gpu-only-full and --gpu-stages replay the whole model through one "
                             "session, which a layer split does not have; run them without --layer-split\n");
        return 2;
    }
    if (o.native_ple_key && (o.native_dense_gguf.empty() || o.no_ple)) {
        std::fprintf(stderr, "strata generate: --native-ple-key requires PLE and --native-dense-gguf\n");
        return 2;
    }
    if (o.cpu_oracle_q8_0 && (o.expert_cache != 0 || !o.expert_profile.empty())) {
        std::fprintf(stderr, "strata generate: --cpu-oracle-q8-0 cannot be combined with --expert-cache or --expert-profile until the GPU expert contract matches\n");
        return 2;
    }

    // **BEFORE ANYTHING ELSE.**  The CPU expert kernel is AVX-512 (VNNI + VBMI) and its translation unit is
    // compiled `/arch:AVX512`, so on a CPU without those features it does not fail - it executes an illegal
    // instruction at some unpredictable token.  Refusing at second zero is the whole point of P2.S3's check.
    strata::kernels::cpu::expert_set_oracle_q8_0(o.cpu_oracle_q8_0);

    std::string err;
    if (!o.native_head_gguf.empty() && !o.stream_token) {
        std::fprintf(stderr, "--native-head-gguf requires --stream-token\n");
        return 2;
    }
    // Plan v0.3 P6: where the experts live.  A native pack (tools/iq_pack.py: the IQ2_XS / IQ3_XXS files) keeps
    // every quantized tensor in its GGUF form, so it needs --native (the dense projections, head and embedding
    // come from the model file) and runs its experts in verify windows only (--spec).
    {
        const strata::core::ModelGeometry g0;
        if (!strata::kernels::cpu::expert_layout_load(o.pack, g0.n_layers, g0.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
    }
    const bool native_pack = strata::kernels::cpu::expert_layout().native;
    // STRATA_EARLY_REMOTE_CONTEXTS=1: create EVERY secondary context here, like CUDA1's.  Under WSL2 the driver's
    // pinned/mapped host budget (dxg gpadl, ~1 GiB) is spent by CUDA0's weights and MTP before the later loop runs,
    // and a new context then fails with cudaErrorMemoryAllocation (CUDA2: "cudaSetDevice(2) failed: out of memory").
    const char* early_env = std::getenv("STRATA_EARLY_REMOTE_CONTEXTS");
    const bool early_remote = early_env && early_env[0] == '1';
    for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0 && (r == 0 || early_remote)) {
        // Keep CUDA1's proven startup order: initialise its context before
        // allocating GPU0 weights or mapping the large host expert arena.
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(remote_dev[r], free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: CUDA%d context ready, %.2f GiB free before expert arena registration\n",
                     remote_dev[r], free_gib);
    }
    // plan v0.3 P6: the PCIe share of the missed experts, measured per kind of pack (the paper, finding on PCIe).
    // PR #44: a x8 link carries half of what the native default assumes - the GPU's SMs read that share over the
    // link (the copy kernel, since 0.1.14), so on a slower link it must shrink or the window waits for it.  The
    // real H2D bandwidth is probed once; from 20 GB/s up (x16 PCIe 4/5) the measured default stays.  The canonical
    // pack's 0.2 was never measured against the link, so it is left alone.  `--calibrate` measures it outright.
    if (o.pcie_frac < 0.0) {
        const double base = native_pack ? 0.55 : 0.2;
        const double bw = native_pack ? probe_pcie_h2d_gbps() : -1.0;
        if (!native_pack) {
            o.pcie_frac = base;
        } else if (bw > 0.0) {
            // below ~4 GB/s (an x1 link: ~0.9 GB/s) a missed expert's 1.4 MB takes longer to cross than the CPU
            // pool takes to compute it, so none of them go over the link
            o.pcie_frac = bw >= 20.0 ? base : bw < 4.0 ? 0.0 : std::min(base, std::max(0.05, base * (bw / 26.0)));
            std::fprintf(stderr, "strata generate: PCIe probe: %.1f GB/s host->device -> pcie_frac %.2f (default %.2f)\n",
                         bw, o.pcie_frac, base);
        } else {
            o.pcie_frac = base;
            std::fprintf(stderr, "strata generate: PCIe probe failed -> pcie_frac default %.2f\n", base);
        }
    }
    // the canonical Q2_0 pack's CPU kernels are AVX-512 only; a native pack runs on AVX2 CPUs as well
    if (!native_pack) strata::kernels::cpu::cpu_require_expert_support();
    else if (!strata::kernels::cpu::cpu_avx512_ok())
        std::fprintf(stderr, "strata generate: this CPU has no AVX-512: the expert kernels run on %s "
                             "(multi-token for the i-quant gate/up rows)\n",
                     std::getenv("STRATA_NO_IQ256") == nullptr ? "AVX-2" : "ggml-cpu vec_dot (STRATA_NO_IQ256 set)");
    strata::core::ModelGeometry g;   // canonical defaults; the model file overrides the MoE shape below
    int64_t K = 10;
    // THE ROPE CONFIG RESOLVES HERE, BEFORE ANY WEIGHT MOVES - the CLI and the model file have both spoken,
    // and `session_init` below builds the rope table from it and captures the kernels reading its constants
    // (rope_scaling.hpp); the only hard constraint is "set before that", and dying on a bad rope key beats
    // scanning gigabytes of shards first.  Precedence: an EXPLICIT flag over the model file's rope keys over
    // the struct defaults.  The empty --rope-scaling and the 0 --rope-scale mean the flag is absent, so the
    // model file decides; an explicit value - `none` and `1` included, the opt-outs - wins over the model file.
    {
        // The model file's rope keys (llama.cpp's names under the arch prefix), when it carries any - the
        // artifact today ships none, so this is a no-op defaults channel for future fine-tunes.
        std::string gguf_rope_type;
        double gguf_rope_base = 0, gguf_rope_factor = 0, gguf_rope_orig_ctx = 0;
        if (!o.native_preset.empty()) {
            // a pruned variant (GSQ-RCO Coder) ships fewer experts than the canonical 512x10; the model file
            // is the authority on its own MoE shape - everything else in the geometry is unchanged
            try {
                strata::GgufFile model_gguf(o.native_preset);
                if (const strata::MetaValue* v = model_gguf.get("qwen4exp.expert_count")) g.n_expert = (int64_t) v->u;
                if (const strata::MetaValue* v = model_gguf.get("qwen4exp.expert_used_count")) K = (int64_t) v->u;
                if (const strata::MetaValue* v = model_gguf.get("qwen4exp.rope.freq_base")) gguf_rope_base = v->num();
                if (const strata::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.type")) gguf_rope_type = v->s;
                if (const strata::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.factor")) gguf_rope_factor = v->num();
                if (const strata::MetaValue* v = model_gguf.get("qwen4exp.rope.scaling.original_context_length"))
                    gguf_rope_orig_ctx = v->num();
            } catch (const std::exception& e) {
                std::fprintf(stderr, "strata generate: reading the model's expert shape from %s: %s\n",
                             o.native_preset.c_str(), e.what());
                return 1;
            }
        }
        using RST = strata::kernels::RopeScalingType;
        if (!o.rope_scaling.empty()) {
            // the early validation pinned the spelling; `none` here is the CLI opting OUT of the model file's keys
            if (o.rope_scaling == "linear") rope_cfg.type = RST::Linear;
            else if (o.rope_scaling == "yarn") rope_cfg.type = RST::YaRN;
            else rope_cfg.type = RST::None;
        } else if (!gguf_rope_type.empty()) {
            if (gguf_rope_type == "linear") rope_cfg.type = RST::Linear;
            else if (gguf_rope_type == "yarn") rope_cfg.type = RST::YaRN;
            else if (gguf_rope_type != "none") {
                std::fprintf(stderr, "strata generate: %s carries rope.scaling.type '%s' - none, linear or yarn only\n",
                             o.native_preset.c_str(), gguf_rope_type.c_str());
                return 2;
            }
        }
        if (o.rope_scale > 0) rope_cfg.factor = o.rope_scale;            // an explicit factor, 1 included
        else if (gguf_rope_factor > 1.0) rope_cfg.factor = gguf_rope_factor;
        if (o.rope_freq_base > 0) rope_cfg.freq_base = o.rope_freq_base;
        else if (gguf_rope_base > 1.0) rope_cfg.freq_base = gguf_rope_base;
        if (o.yarn_orig_ctx > 0) rope_cfg.orig_ctx = o.yarn_orig_ctx;
        else if (gguf_rope_orig_ctx >= 1) rope_cfg.orig_ctx = gguf_rope_orig_ctx;
        rope_cfg.freq_scale_in = o.rope_freq_scale;
        rope_cfg.ext_factor = o.yarn_ext_factor >= 0 ? o.yarn_ext_factor
                                                     : (rope_cfg.type == RST::YaRN ? 1.0 : 0.0);
        rope_cfg.attn_factor = o.yarn_attn_factor;
        rope_cfg.beta_fast = o.yarn_beta_fast;
        rope_cfg.beta_slow = o.yarn_beta_slow;
        if (rope_cfg.type == RST::None) {
            // none is the trained rotation, exactly: the scaling knobs are inert (the table builder and
            // `kernel_args` ignore them), and resetting them keeps the logged/queried config honest.  Only the
            // frequency base survives - it is the rotation itself, not a scaling knob.
            const bool knobs = o.rope_scale > 1.0 || o.rope_freq_scale > 0 || o.yarn_ext_factor > 0 ||
                               o.yarn_attn_factor != 1.0;
            const double base = rope_cfg.freq_base;
            rope_cfg = strata::kernels::RopeScaling{};
            rope_cfg.freq_base = base;
            if (knobs)
                std::fprintf(stderr, "strata generate: note: no rope scaling is active (none), so --rope-scale, "
                                     "--rope-freq-scale and the --yarn-* knobs have no effect\n");
        }
        // THE RESOLVED CONFIG IS VALIDATED AS A WHOLE, with the one rule every rotation site also applies
        // (rope_scaling.hpp): the CLI ranges above cannot see a model-file value, nor a combination such as a
        // --rope-freq-scale that turns the resolved factor non-finite.
        if (const char* why = strata::kernels::rope_scaling_invalid(rope_cfg)) {
            std::fprintf(stderr, "strata generate: invalid rope scaling configuration: %s (type %s, factor %g, "
                                 "freq_scale %g, base %g, original context %g)\n",
                         why, rope_cfg.type == RST::YaRN ? "yarn" : rope_cfg.type == RST::Linear ? "linear" : "none",
                         rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.orig_ctx);
            return 2;
        }
        strata::kernels::rope_scaling_set(rope_cfg);
        if (rope_cfg.type != RST::None) {
            const char* tn = rope_cfg.type == RST::YaRN ? "yarn" : "linear";
            std::fprintf(stderr,
                         "strata generate: rope scaling %s, factor %.6g (freq_scale %.6g, base %.6g, mscale %.6f), "
                         "--max-context %lld against a trained context of %.0f\n",
                         tn, rope_cfg.factor, rope_cfg.freq_scale(), rope_cfg.freq_base, rope_cfg.mscale(),
                         (long long) o.max_context, rope_cfg.orig_ctx);
            if ((double) o.max_context <= rope_cfg.orig_ctx)
                std::fprintf(stderr,
                             "strata generate: note: the context is within the trained %.0f - no position needs the "
                             "extension, and the resolved scaling still applies to every angle\n",
                             rope_cfg.orig_ctx);
            // only when there IS a magnitude correction: YaRN's log term (ext_factor != 0) or an explicit
            // --yarn-attn-factor; plain linear (mscale 1) has none, and saying otherwise was TODO 22
            if (rope_cfg.mscale() != 1.0)
                std::fprintf(stderr,
                             "strata generate: note: the %s magnitude correction scales cos and sin by %.6f "
                             "at every position\n",
                             rope_cfg.type == RST::YaRN ? "YaRN" : "--yarn-attn-factor", rope_cfg.mscale());
        }
    }
    strata::core::NativeEmbed native_embed;
    if (native_pack) {
        if (o.native_preset.empty() || o.spec < 2 || o.keep_canonical ||
            (o.prefill_chunk <= 0 && o.tokens.size() > 1)) {
            std::fprintf(stderr, "strata generate: %s is a native (IQ) pack: it needs --native SHARD1, --spec T (T >= 2) "
                                 "and --prefill CHUNK\n", o.pack.c_str());
            return 2;
        }
        const strata::core::ModelGeometry g0;
        if (!native_embed.load(o.native_preset, g0.n_embd, 248320, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        strata::core::set_native_embed(&native_embed);
        std::fprintf(stderr, "strata generate: native pack: %s experts (largest blob %.2f MB), token embedding "
                             "type %d in mapped host memory (%.0f MiB)\n",
                     o.pack.c_str(), (double) strata::kernels::cpu::expert_layout().max_blob / 1e6,
                     native_embed.type(), (double) native_embed.bytes() / 1048576.0);
    }
    // Plan v0.3 P1: tensors served in native form are not also loaded in canonical form (~2.7 GB of VRAM back
    // to the expert cache with --native).  `--keep-canonical` loads both, as before.
    std::set<std::string> skip;
    if (!o.keep_canonical) {
        if (!o.native_dense_gguf.empty() &&
            !strata::core::NativeDense::served_names(o.native_dense_gguf, o.native_ple_key, skip, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!o.native_head_gguf.empty()) skip.insert("output.weight");
        // the PLE module validates its canonical key at construction (8 MB); a native pack has none to load
        if (!native_pack) skip.erase("blk.1.ple_key.weight");
        if (native_pack) skip.insert("token_embd.weight");
    }
    uint64_t pool_bytes = 0;
    if (!strata::core::WeightTable::pool_bytes(o.pack, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    void* arena = nullptr;
    if (cudaMalloc(&arena, pool_bytes) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: cudaMalloc(%llu) for the weight arena failed\n",
                     (unsigned long long) pool_bytes);
        return 1;
    }
    strata::core::WeightTable wt;
    if (!wt.load(o.pack, arena, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    std::fprintf(stderr, "strata generate: %llu MiB of weights loaded from %s (%zu canonical tensors skipped: "
                         "served natively)\n",
                 (unsigned long long) (pool_bytes >> 20), o.pack.c_str(), skip.size());

    strata::core::NativeDense native_dense;
    if (!o.native_dense_gguf.empty()) {
        if (!native_dense.load(o.native_dense_gguf, wt, err, o.native_ple_key)) {
            std::fprintf(stderr, "strata generate: native dense projections: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: %zu native projection matrices, %.2f MiB of weights\n",
                     native_dense.tensor_count(), (double) native_dense.weight_bytes() / (1024.0 * 1024.0));
    }

    strata::kernels::gr_set_fp32_activations(o.gr_fp32_activations);
    strata::kernels::gr_set_native_mmvf(o.gr_native_mmvf);
    // Plan v0.3 P3: the fused hyper-connection read rides the native (FP32-activation) contract; the per-stage
    // and dump measurements need the unfused layout of R, so they keep the old kernels.
    strata::core::layer_set_fast_attn(!o.no_fast_attn);
    strata::core::layer_set_publish_kernel(!o.no_publish_kernel);
    strata::core::layer_set_fused_gdn(!o.no_fused_gdn);
    strata::core::layer_set_fast_select(!o.no_fast_select);
    strata::core::layer_set_fused_gr(o.gr_native_mmvf && !o.no_fused_gr && !o.gpu_stages && o.dump_layers.empty() &&
                                     o.dump_halves.empty() && !o.stage_timing);
    strata::core::layer_set_native_bf16(o.native_bf16);
    strata::core::layer_set_native_flash_attn_short(o.native_flash_attn_short);
    strata::kernels::ple_set_native_bf16(o.native_bf16_extra);
    strata::kernels::shared_expert_set_native_bf16(o.native_bf16_extra);
    strata::kernels::native_moe_combine_set_enabled(o.native_moe_combine);
    strata::kernels::native_gdn_set_enabled(o.native_gdn);
    strata::kernels::native_router_set_enabled(o.native_router);
    strata::kernels::native_qsa_set_enabled(o.native_qsa);
    strata::kernels::native_qsa_indexer_set_enabled(o.native_qsa_indexer);
    strata::kernels::native_rope_set_enabled(o.native_rope);
    // The vision path: every rope kernel reads a cell's (t, h, w) from this table (strata/kernels/mrope.hpp).  It is
    // the identity until an image request, and it is set here, before any CUDA graph captures a rope kernel.
    int32_t* d_mrope = nullptr;
    std::vector<int32_t> mrope_host;
    if (o.vision) {
        const int64_t cells = o.max_context + 64;
        mrope_host.resize((size_t) cells * 3);
        for (int64_t c = 0; c < cells; ++c)
            mrope_host[(size_t) c * 3] = mrope_host[(size_t) c * 3 + 1] = mrope_host[(size_t) c * 3 + 2] = (int32_t) c;
        if (cudaMalloc(&d_mrope, mrope_host.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot allocate the image position table\n");
            return 1;
        }
        strata::kernels::mrope_table_set(d_mrope);
    }
    strata::kernels::ple_set_native_postops(o.native_ple_postops);
    // before session_init: every graph captured from here on has the vector's kernels where it applies
    std::string cvec_summary = "0";
    if (!o.cvec_files.empty()) {
        std::string ce;
        if (!load_control_vectors(o, g, cvec_summary, ce)) {
            std::fprintf(stderr, "strata generate: control vector: %s\n", ce.c_str());
            return 2;
        }
    }
    if (o.max_context < (int64_t) o.tokens.size() + o.max_new) {
        std::fprintf(stderr, "strata generate: --max-context %lld cannot hold %zu prompt + %lld new tokens\n",
                     (long long) o.max_context, o.tokens.size(), (long long) o.max_new);
        return 2;
    }

    strata::core::SessionState ss;
    void* sbuf = nullptr;   // allocated after the layer-split search, sized to CUDA0's own layer range (the carve)
    // **THE ENGINE RAN ON THE LEGACY DEFAULT STREAM, WHICH ON WDDM IS THE SLOW PATH.**  All four session
    // calls - `session_capture`, `session_replay`, `session_token` and `session_loop` - were handed `nullptr`,
    // i.e. stream 0.  `bench/micro/kernel_costs.cu` measures what that costs: EVERY kernel it launches through
    // a wrapper comes back at 28-31 us REGARDLESS OF SIZE, `scale_inplace` on 2,048 floats and `silu_inplace`
    // on 10,240 floats being indistinguishable, which is a fixed per-launch cost and not execution.
    // `bench/micro/graph_node_cost.cu` measures the same kernels on a real stream at 3.63 us ungrapped and
    // 0.805 us inside a graph.  **That is an ~8x penalty on every launch in the engine.**
    cudaStream_t main_stream = nullptr;
    if (cudaStreamCreateWithFlags(&main_stream, cudaStreamNonBlocking) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: cannot create the main stream\n");
        return 1;
    }
    void* const main_cs = (void*) main_stream;

    // ---- **THE HALF-LEVEL DUMP HAS TO BE ARMED BEFORE `session_capture`, AND THE LADDER MUST NOT BE.**  The
    // half copies are issued from inside `block_layer_pre`/`block_layer_post`, so they are only ever enqueued
    // while a graph is being CAPTURED - arming `ss.block.dump` afterwards would produce a file of zeros that
    // reads exactly like a wrong answer.  The ladder is the opposite: `session_loop` enqueues it per token on
    // the replay stream, so it must be armed after capture to stay out of the graph.
    const uint64_t half_stride = (uint64_t) 2 * g.n_embd + (uint64_t) 2 * g.hc +
                                 (uint64_t) g.n_head * g.head_dim +
                                 (uint64_t) 5 * g.n_head_kv * g.head_dim + 8;
    std::FILE* half_dump = nullptr;
    float* half_stage = nullptr;
    if (!o.dump_halves.empty()) {
        half_dump = std::fopen(o.dump_halves.c_str(), "wb");
        if (half_dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_halves.c_str());
            return 1;
        }
        const size_t n = (size_t) g.n_layers * (size_t) half_stride;
        if (cudaHostAlloc((void**) &half_stage, n * sizeof(float), cudaHostAllocDefault) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot pin the half-dump staging buffer\n");
            return 1;
        }
        ss.block.dump = half_stage;
    }

    strata::core::Doorbell db;
    if (strata::core::doorbell_init(g, K, db) == 0) {
        std::fprintf(stderr, "strata generate: doorbell_init failed\n");
        return 1;
    }
    ss.db = &db;

    // ================================ THE PLE ================================
    //
    // **ITS ABSENCE IS WHY GATE C1 FAILED** (LEDGER L123): layer 1 carries six `blk.1.ple_*` tensors, the whole
    // module was built and parity-tested, and nothing called it.  Everything below is construction - the table
    // is a mapping of the ORIGINAL second GGUF shard, the six weights are already loaded in the arena, and the
    // three buffers are the only allocation.
    strata::kernels::PleTable ple_table;
    std::vector<float> ple_emb_host((size_t) strata::kernels::NG_N_EMBD);
    float* ple_emb_dev = nullptr;
    float* ple_scratch = nullptr;
    if (!o.ple_gguf.empty()) {
        strata::kernels::PleIoOptions pio;
        pio.mode = o.ple_io == "mmap" || o.ple_io == "ram" ? strata::kernels::PleIo::Mmap : strata::kernels::PleIo::Direct;
        pio.lock = o.ple_io == "ram";
        const auto tpl = Clock::now();
        pio.max_inflight = (uint32_t) o.ple_inflight;
        pio.cache_rows = (uint64_t) o.ple_row_cache;
        pio.io_thread = !o.ple_sync_submit;
        if (!ple_table.open(o.ple_gguf, err, pio)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (pio.lock)
            std::fprintf(stderr, "strata generate: PLE table %s (--ple-io ram) in %.1f s\n",
                         ple_table.locked() ? "locked in RAM" : "loaded (not locked)",
                         std::chrono::duration<double>(Clock::now() - tpl).count());
        const strata::core::WeightRef* wk = wt.find("blk.1.ple_key.weight");
        const strata::core::WeightRef* wv = wt.find("blk.1.ple_value.weight");
        const strata::core::WeightRef* wnk = wt.find("blk.1.ple_norm_key.weight");
        const strata::core::WeightRef* wnq = wt.find("blk.1.ple_norm_query.weight");
        const strata::core::WeightRef* wnc = wt.find("blk.1.ple_norm_conv.weight");
        const strata::core::WeightRef* wc = wt.find("blk.1.ple_conv1d.weight");
        if (!wk || !wv || !wnk || !wnq || !wnc || !wc) {
            std::fprintf(stderr, "strata generate: the pack has no blk.1.ple_* tensors, so the PLE cannot be "
                                 "wired - and running without it is a DIFFERENT MODEL (LEDGER L123)\n");
            return 1;
        }
        // `ple_key` is S2 and the loader has already widened its scales to f32, so the two planes are located
        // by the sizes the `WeightRef` records rather than re-derived - the same rule `plane_ptrs` follows.
        if (!wk->quantized()) {
            // plan v0.3 P6: the IQ model files' BF16 key (the pack's extra.bin, raw BF16)
            ss.ple.w.key_bf16 = (const uint16_t*) wk->data;
        } else if (wk->data != nullptr) {
            ss.ple.w.key_codes = (const uint8_t*) wk->data;
            ss.ple.w.key_scales = (const float*) ((const uint8_t*) wk->data + wk->codes_bytes);
        }
        if (o.native_ple_key && wk->quantized()) {
            if (!wk->native_data || (wk->native_type != 42 && wk->native_type != 18 && wk->native_type != 23) || !wk->native_q8_1) {
                std::fprintf(stderr, "strata generate: native PLE key is absent or incompatible\n");
                return 1;
            }
            ss.ple.w.key_native_data = wk->native_data;
            ss.ple.w.key_native_type = wk->native_type;
            ss.ple.w.key_native_q8_1 = wk->native_q8_1;
        }
        ss.ple.w.value_bf16 = (const uint16_t*) wv->data;
        ss.ple.w.norm_key = (const float*) wnk->data;
        ss.ple.w.norm_query = (const float*) wnq->data;
        ss.ple.w.norm_conv = (const float*) wnc->data;
        ss.ple.w.conv1d_f16 = (const uint16_t*) wc->data;
        ss.ple.consts = strata::kernels::ple_artifact_consts();
        if (o.ple_delay_us > 0) ple_table.set_injected_delay_us(o.ple_delay_us);
        ss.ple.table = &ple_table;
        ss.ple.token = &ss.ple_token;
        ss.ple.prev = ss.ple_prev;
        // `ss.ple.hist` and the ready() check wait for `session_init`, which carves the history - the session is
        // now allocated after the layer-split search (see the carve), and the wiring lands there
        ss.ple.emb_host = ple_emb_host.data();
        if (cudaMalloc((void**) &ple_emb_dev, (size_t) strata::kernels::NG_N_EMBD * 4) != cudaSuccess ||
            cudaMalloc((void**) &ple_scratch, strata::core::ple_run_scratch_bytes()) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the PLE buffers failed\n");
            return 1;
        }
        ss.ple.emb_dev = ple_emb_dev;
        ss.ple.scratch = ple_scratch;
    } else {
        std::fprintf(stderr,
                     "strata generate: PLE OFF by explicit --no-ple diagnostic request.\n"
                     "  The tokens below are NOT this model's; this is only useful for A/B measurement.\n");
    }

    float* d_parts = nullptr;
    if (cudaMalloc(&d_parts, (size_t) K * g.n_embd * 4) != cudaSuccess ||
        cudaMemset(d_parts, 0, (size_t) K * g.n_embd * 4) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: the parts buffer failed\n");
        return 1;
    }

    // ---- layer split across GPUs: each later stage's own copy of the dense weights, its session and (the last) the
    // head, made on its device before the host arena is mapped (as the drafter below, for the same WDDM reason)
    std::vector<std::unique_ptr<GpuStage>> stages;
    for (size_t i = 0; multi_gpu && i < split_devs.size(); ++i) {
        stages.push_back(std::make_unique<GpuStage>());
        GpuStage& st = *stages.back();
        st.dev = split_devs[i];
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(st.dev, free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const strata::core::OnDevice on(st.dev);
        void* arena_s = nullptr;
        if (cudaMalloc(&arena_s, pool_bytes) != cudaSuccess ||
            !st.wt.load(o.pack, arena_s, pool_bytes, err, skip.empty() ? nullptr : &skip)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d weights: %s\n", st.dev,
                         err.empty() ? "the weight arena does not fit" : err.c_str());
            return 1;
        }
        if (!o.native_dense_gguf.empty() && !st.dense.load(o.native_dense_gguf, st.wt, err, o.native_ple_key)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d native dense projections: %s\n", st.dev,
                         err.c_str());
            return 1;
        }
        // THE SESSION AND THE HEAD WAIT FOR THE SPLIT SEARCH.  `session_bytes` prices a stage's session by its
        // LAYER RANGE (the carve - every stage used to hold all 48 layers' state whatever it ran), so the
        // sessions are allocated after the search below has set `st.lb`/`st.le`; the last stage's head follows.
        if (cudaStreamCreateWithFlags(&st.stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaStreamCreateWithFlags(&st.adapt_stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaEventCreateWithFlags(&st.adapt_ev, cudaEventDisableTiming) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: its streams failed\n", st.dev);
            return 1;
        }
        // a control vector (the speed projection): its tables on this device too - the stage's layers apply it here
        if (!strata::kernels::cvec_replicate(err)) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: %s\n", st.dev, err.c_str());
            return 1;
        }
        // --vision: this device's image-position table (the identity until a picture request), read by every rope
        // kernel its stage runs - set before any of its graphs is captured
        if (o.vision) {
            if (cudaMalloc(&st.mrope, mrope_host.size() * sizeof(int32_t)) != cudaSuccess ||
                cudaMemcpy(st.mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
                std::fprintf(stderr, "strata generate: layer split, CUDA%d: the image position table failed\n", st.dev);
                return 1;
            }
            strata::kernels::mrope_table_set(st.mrope);
        }
        // its own PCIe share of the missed experts (the same rule as CUDA0's above: its link is probed)
        st.pcie_frac = o.pcie_frac;
        if (!pcie_given && native_pack) {
            const double bw = probe_pcie_h2d_gbps();
            if (bw > 0.0) st.pcie_frac = bw >= 20.0 ? 0.55 : bw < 4.0 ? 0.0 : std::min(0.55, std::max(0.05, 0.55 * (bw / 26.0)));
            std::fprintf(stderr, "strata generate: layer split: CUDA%d PCIe probe %.1f GB/s -> pcie_frac %.2f\n", st.dev,
                         bw, st.pcie_frac);
        }
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        std::fprintf(stderr, "strata generate: layer split: CUDA%d holds its weights; %.2f GiB free (its session "
                             "follows the split search)\n", st.dev, (double) fb / 1073741824.0);
    }
    GpuStage* const last_st = stages.empty() ? nullptr : stages.back().get();

    std::vector<std::pair<int32_t, int32_t>> profile;
    if (!o.expert_profile.empty()) {
        int64_t pslots = 0;
        if (!strata::core::read_expert_profile(o.expert_profile, g.n_layers, g.n_expert, profile, pslots, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // An explicit number truncates the ranked list ("what would 2,000 slots give" without rebuilding the
        // file).  `--expert-cache 0` used to take the count the profile was built for; the profile now ranks
        // every pair (issue #46: a card that holds more than the old 8,000 used to stop there), so it means auto.
        if (o.expert_cache == 0) o.expert_cache = -1;
        std::fprintf(stderr, "strata generate: profile %s: %zu ranked pairs, built for %lld slots\n",
                     o.expert_profile.c_str(), profile.size(), (long long) pslots);
    }
    // ---- layer split across GPUs: "auto" places the split points by a cost model of one decode window, measured on
    // the 5080 + 3090 rig (bench/results/2026-09-29-layer-split):
    //   - every layer costs its GPU a time inversely proportional to SMs x clock (0.33 ms on an RTX 5080, 0.50 on a
    //     3090: the per-layer round trip and kernels, not the bytes - both cards have ~950 GB/s);
    //   - an expert no cache holds costs ~190 ms per unit of routed mass: the CPU pool in decode and the PCIe stream
    //     in prompts (fitted: the sweep's best K, 26-28, is where one more layer on the faster card stops paying
    //     for the ~0.1% of the mass it pushes out of its cache);
    //   - which experts a cache holds: its layers' profiled pairs, hottest first, until its free VRAM (less the
    //     reserve, the prompt path's buffers and, on a later GPU, 1 GiB for its windows and the drafter) is used;
    //     the routed mass of rank r is taken as (r+1)^-1.2 (fits the sweep's hit rates: K=24/26/28 predicted
    //     99.53/99.34/99.15%, measured 99.5/99.4/99.0%).
    // Up to 3 GPUs every placement is tried; beyond, the layers are shared in proportion to speed.
    // STRATA_SPLIT_MISS_MS tunes the miss cost (a slower CPU: higher).
    // THE PROMPT PATH'S BUFFERS ARE BORROWED FROM THE CACHE, NOT WITHHELD BESIDE IT.  With borrowing the cache
    // is sized first and at full size, and the prompt path is laid out in the tail of it (`Prefill::relayout`),
    // so it withholds no VRAM of its own and this reserve is zero.  Only without borrowing - no profile to fill
    // a cache from, or --no-prefill-borrow - do the buffers take a reserve, and then this estimate stands in
    // for buffers that cannot be priced exactly yet because the sessions do not exist.  `plan_lend` uses the
    // exact `Prefill::bytes_needed` as soon as it can.
    const bool pf_borrow = !o.no_prefill_borrow && !o.expert_profile.empty();
    const int64_t split_pf_mib = (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
    // ---- WHAT A STAGE RESERVES, AND ON WHICH STAGE.  The flat 1 GiB this used to withhold from EVERY stage
    // after the first was booked "for its windows and the drafter", but the windows measure 75 MiB ("window up
    // to 6 tokens, 74.1 MiB of device buffers", on every boot) and the drafter is loaded on ONE stage - the
    // last one, which is also the only one that holds the head.  On the two identical 8 GB cards that GiB was
    // the entire difference between CUDA0's cache and CUDA1's: 814 slots against 188, 2026-09-30.  Both of
    // those allocations are already made before a stage's cache is sized, so what has to be held back here is
    // the windows and - only on the stage that carries them - the drafter and the head.
    const int64_t kWindowMib = 96;       // the verify windows; 75 MiB measured, rounded up
    const int64_t kDrafterMib = 1000;    // the MTP drafter (839 MiB) + the head, on the last stage only
    auto stage_room = [&](int dev, bool later, bool drafter) -> int64_t {
        const strata::core::OnDevice on(dev);
        size_t fb = 0, tb = 0;
        if (const cudaError_t e = cudaMemGetInfo(&fb, &tb); e != cudaSuccess)
            std::fprintf(stderr, "strata generate: layer split: CUDA%d free memory: %s\n", dev < 0 ? 0 : dev,
                         cudaGetErrorString(e));
        const int64_t reserve = ((int64_t) o.vram_reserve_mib + split_pf_mib + (later ? kWindowMib : 0) +
                                 (drafter ? kDrafterMib : 0)) << 20;
        return std::max<int64_t>((int64_t) fb - reserve, 0);
    };
    if (multi_gpu && split_auto) {
        const auto& lay = strata::kernels::cpu::expert_layout();
        const int ns = (int) stages.size() + 1;
        std::vector<int64_t> cap((size_t) ns), used((size_t) ns);
        std::vector<double> layer_ms((size_t) ns);
        for (int i = 0; i < ns; ++i) {
            const int dev = i == 0 ? 0 : stages[(size_t) i - 1]->dev;
            cap[(size_t) i] = stage_room(i == 0 ? -1 : dev, i > 0, i + 1 == ns);
            int sms = 0, khz = 0;
            cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
            if (cudaDeviceGetAttribute(&khz, cudaDevAttrClockRate, dev) != cudaSuccess || khz <= 0) khz = 1800000;
            cudaGetLastError();
            const double speed = std::max(1.0, (double) sms * (double) khz / 1e6);   // SMs x GHz
            layer_ms[(size_t) i] = 0.33 * (84.0 * 2.617) / speed;
            std::fprintf(stderr, "strata generate: layer split auto: CUDA%d %d SMs at %.2f GHz -> %.2f ms per layer, "
                                 "%.2f GiB free before its session carve\n", dev, sms, khz / 1e6, layer_ms[(size_t) i],
                         (double) cap[(size_t) i] / 1073741824.0);
        }
        const double miss_ms = std::getenv("STRATA_SPLIT_MISS_MS") ? std::atof(std::getenv("STRATA_SPLIT_MISS_MS")) : 190.0;
        std::vector<double> mass(profile.size());
        double total_mass = 0;
        for (size_t r = 0; r < profile.size(); ++r) total_mass += (mass[r] = std::pow((double) r + 1.0, -1.2));
        auto cost = [&](int64_t l) -> int64_t {
            return native_pack ? ((int64_t) lay.blob_bytes(l) + 255) / 256 * 256 : (int64_t) lay.max_blob;
        };
        // the predicted window time (ms) of a placement, and the routed mass its caches hold
        auto predict = [&](const std::vector<int64_t>& at, double& held_mass, int64_t& held) -> double {
            // THE CARVE, PRICED: a placement gives stage i the layers [lb, le), and that range's session is a
            // real cost on its device - subtracted here so the search knows what it leaves for experts.  This
            // is why the sessions are allocated after the search: `session_bytes` is pure arithmetic.
            std::vector<int64_t> capr((size_t) ns);
            for (int i = 0; i < ns; ++i) {
                const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1];
                const int64_t le = i + 1 < ns ? at[(size_t) i] : g.n_layers;
                capr[(size_t) i] = cap[(size_t) i] - (int64_t) strata::core::session_bytes(g, o.max_context, K, lb, le);
            }
            std::fill(used.begin(), used.end(), 0);
            held_mass = 0;
            held = 0;
            std::vector<bool> full((size_t) ns, false);
            for (size_t r = 0; r < profile.size(); ++r) {
                const int64_t l = profile[r].first;
                int st = 0;
                while (st + 1 < ns && l >= at[(size_t) st]) ++st;
                if (full[(size_t) st]) continue;
                if (used[(size_t) st] + cost(l) > capr[(size_t) st]) { full[(size_t) st] = true; continue; }   // as the fill
                used[(size_t) st] += cost(l);
                held_mass += mass[r];
                ++held;
            }
            held_mass /= std::max(total_mass, 1e-9);
            double ms = miss_ms * (1.0 - held_mass);
            for (int i = 0; i < ns; ++i) {
                const int64_t lb = i == 0 ? 0 : at[(size_t) i - 1], le = i + 1 < ns ? at[(size_t) i] : g.n_layers;
                ms += (double) (le - lb) * layer_ms[(size_t) i];
            }
            return ms;
        };
        std::vector<int64_t> best, at((size_t) ns - 1);
        double best_ms = 1e30, best_mass = 0;
        int64_t best_held = 0;
        auto consider = [&]() {
            double hm = 0;
            int64_t held = 0;
            const double ms = predict(at, hm, held);
            if (ms < best_ms) { best = at; best_ms = ms; best_mass = hm; best_held = held; }
        };
        const int64_t L = g.n_layers;
        if (ns == 2) {
            for (int64_t k = 2; k < L; ++k) { at[0] = k; consider(); }
        } else if (ns == 3) {
            for (int64_t k1 = 2; k1 + 1 < L; ++k1)
                for (int64_t k2 = k1 + 1; k2 < L; ++k2) { at[0] = k1; at[1] = k2; consider(); }
        } else {
            double total = 0;
            for (const double c : layer_ms) total += 1.0 / c;
            double acc = 0;
            for (int i = 0; i + 1 < ns; ++i) {
                acc += 1.0 / layer_ms[(size_t) i];
                at[(size_t) i] = std::clamp<int64_t>((int64_t) std::llround(acc / total * (double) L),
                                                    i == 0 ? 2 : at[(size_t) i - 1] + 1, L - (ns - 1 - i));
            }
            consider();
        }
        split_at = best;
        std::string ks;
        for (const int64_t k : split_at) ks += (ks.empty() ? "" : ",") + std::to_string(k);
        std::fprintf(stderr, "strata generate: layer split auto: K=%s - predicted %.1f ms per decode window; the caches "
                             "hold %lld of %zu profiled pairs (~%.1f%% of the routed mass)\n", ks.c_str(), best_ms,
                     (long long) best_held, profile.size(), 100.0 * best_mass);
    }
    for (size_t i = 0; i < split_at.size(); ++i)
        if (split_at[i] >= g.n_layers) {
            std::fprintf(stderr, "strata generate: --layer-split: layer %lld is past the last (%lld)\n",
                         (long long) split_at[i], (long long) (g.n_layers - 1));
            return 2;
        }
    // the stage that runs a layer (0: CUDA0's)
    auto stage_of = [&](int64_t l) -> int {
        int st = 0;
        while (st < (int) split_at.size() && l >= split_at[(size_t) st]) ++st;
        return st;
    };
    if (multi_gpu) {
        std::vector<std::pair<int32_t, int32_t>> mine;
        for (const auto& pr : profile) {
            const int st = stage_of(pr.first);
            (st == 0 ? mine : stages[(size_t) st - 1]->profile).push_back(pr);
        }
        profile.swap(mine);
        for (size_t i = 0; i < stages.size(); ++i) {
            stages[i]->lb = split_at[i];
            stages[i]->le = i + 1 < stages.size() ? split_at[i + 1] : g.n_layers;
        }
    }

    // ---- CUDA0's session, and the stages' sessions: sized to each device's own layer range (the carve).  A
    // stage that runs [lb, le) carves only those layers' GDN rows and QSA pools - before the carve every stage
    // held all 48 layers' state whatever layers it ran, which is the same disease the chunked-QSA-prefill PR
    // fixed in llama.cpp: allocation sized by the whole model instead of the device's own work.
    {
        const strata::core::OnDevice on0(0);
        const int64_t hi0 = multi_gpu ? split_at[0] : -1;
        if (cudaMalloc(&sbuf, strata::core::session_bytes(g, o.max_context, K, 0, hi0)) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: session state allocation failed\n");
            return 1;
        }
        if (strata::core::session_init(g, o.max_context, K, sbuf, ss, 0, hi0) == 0) {
            std::fprintf(stderr, "strata generate: session_init failed\n");
            return 1;
        }
        if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1)
            std::fprintf(stderr, "strata generate: KV streaming: %lld of %lld cells per QSA layer in VRAM, the K/V in "
                                 "%.2f GiB of pinned RAM\n",
                         (long long) (ss.qsa_states[ss.qsa_primary()].n_slots * 4),
                         (long long) o.max_context, (double) strata::core::qsa_kv_host_bytes() / 1073741824.0);
        // the PLE block above built everything but the history, which session_init has just carved
        ss.ple.hist = ss.ple_hist;
        // (#167) generate mode starts from an empty sequence, and nothing else zeroes this state before the prompt
        // path or the verifier reads it (--serve zeroes it per request when nothing is reused)
        strata::core::session_zero(ss, g, nullptr, main_cs);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata generate: zeroing the session state failed\n");
            return 1;
        }
        if (!o.ple_gguf.empty()) {
            if (!ss.ple.ready()) {
                std::fprintf(stderr, "strata generate: the PLE run is not ready after construction\n");
                return 1;
            }
            std::fprintf(stderr, "strata generate: PLE on, table %llu rows of %s\n",
                         (unsigned long long) ple_table.rows(), o.ple_gguf.c_str());
        }
    }
    for (size_t i = 0; i < stages.size(); ++i) {
        GpuStage& st = *stages[i];
        const strata::core::OnDevice on(st.dev);
        void* sbuf_s = nullptr;
        if (cudaMalloc(&sbuf_s, strata::core::session_bytes(g, o.max_context, K, st.lb, st.le)) != cudaSuccess ||
            strata::core::session_init(g, o.max_context, K, sbuf_s, st.ss, st.lb, st.le) == 0) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d: the session state failed\n", st.dev);
            return 1;
        }
        const bool last = i + 1 == stages.size();
        const strata::core::WeightRef* wo_s = st.wt.find("output.weight");
        if (wo_s == nullptr ||
            (last && !o.native_head_gguf.empty() && !st.head.load(o.native_head_gguf, g.n_embd, wo_s->ne1, err))) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d head: %s\n", st.dev,
                         wo_s == nullptr ? "output.weight is missing" : err.c_str());
            return 1;
        }
        size_t fb = 0, tb = 0;
        cudaMemGetInfo(&fb, &tb);
        std::fprintf(stderr, "strata generate: layer split: CUDA%d holds its weights, session [%lld, %lld)%s; "
                             "%.2f GiB free\n", st.dev, (long long) st.lb, (long long) st.le,
                     last ? " and the head" : "", (double) fb / 1073741824.0);
    }

    // Secure MTP's CUDA0 allocations before the large host arena is registered with both CUDA contexts.
    // In particular WDDM can refuse the draft weights after mapping tens of GiB of host pages.
    strata::core::MtpDrafter mtp;
    if (!o.mtp.empty()) {
        if (o.spec < 2) {
            std::fprintf(stderr, "strata generate: --mtp is ignored without --spec T (T >= 2)\n");
            o.mtp.clear();
        }
        if (!o.mtp.empty()) mtp.set_prompt_len((int64_t) o.tokens.size());
        // the draft layer is the canonical model's MTP head (512 experts) even when the target is pruned,
        // so it always sees the canonical geometry; `static` because MtpDrafter keeps a reference
        static const strata::core::ModelGeometry draft_geometry{};
        // with a layer split across GPUs the drafter reads the last stage's residual: it lives on that device
        const strata::core::OnDevice on_mtp(last_st ? last_st->dev : -1);
        if (!o.mtp.empty() && !mtp.load(o.mtp, draft_geometry, last_st ? last_st->ss : ss, o.spec, err, o.mtp_window)) { std::fprintf(stderr, "strata generate: %s\n", err.c_str()); return 1; }
    }
    // Create the additional contexts after MTP has secured CUDA0 memory, but
    // before the host arena maps its expert pages into their address spaces.
    for (int r = 1; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0 && !early_remote) {
        double free_gib = 0;
        if (!strata::core::RemoteExperts::preflight(remote_dev[r], free_gib, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: CUDA%d context ready, %.2f GiB free before expert arena registration\n",
                     remote_dev[r], free_gib);
    }

    // ---- the CPU expert pool
    //
    // R2.1: the experts are loaded into a RESIDENT ARENA by default.  The mmap path is kept behind
    // `--mmap-experts` because it is the A/B arm, not because it is competitive.
    //
    // The reasoning is the review's C1 and it is now measured on both sides.  `FileExpertSource` maps the 34 GB
    // file, and mapped file pages are the first thing the OS reclaims; the engine's rate then depends on whether
    // the standby list happens to hold `experts.bin`, which is why two consecutive runs of the SAME BINARY with
    // the SAME FLAGS measured 71.97 and 34.78 ms/token in the pool (7.54 vs 12.18 tok/s).  The arena is
    // anonymous memory the engine owns, and the pool runs at 19.41 ms/token - 1.79x better than the warm mmap
    // and 3.7x better than the cold one.
    //
    // IT IS NOT PINNED, and that is reported rather than hidden: `cudaHostRegister` on 31.64 GiB fails with
    // "out of memory" (you cannot pin 34 of 63 GB) and the arena falls back to 4 KB anonymous pages.  That is
    // fine for the CPU pool - which is all that exists today - and NOT fine for Phase 3, whose cache fills and
    // CPU/PCIe miss split need the GPU to DMA out of this arena.  Read `note()` when that lands.
    //
    // The earlier "the arena does not fit" conclusion was WRONG and is worth recording: the failure was a stale
    // CUDA error left set by the failed `cudaHostRegister` and read later by `gr_read`'s launch check.  See the
    // note in `pinned.cu`.
    strata::core::FileExpertSource src;
    strata::core::ArenaExpertSource arena_src;
    strata::core::ExpertSource* srcp = nullptr;
    if (o.mmap_experts) {
        // FileExpertSource maps the pack's experts.bin: a canonical pack has it; a native (IQ) pack has it when
        // built with `tools/iq_pack.py --experts-bin` (the per-layer blob sizes of its layout, PR #121).  The low-RAM
        // mode: the experts come from the file through the OS cache instead of a pinned copy in RAM, for a PC whose
        // GPU holds most of them but whose RAM cannot hold them all.
        if (native_pack && !std::filesystem::exists(std::filesystem::path(o.pack) / "experts.bin")) {
            std::fprintf(stderr, "strata generate: --mmap-experts needs the pack's experts.bin; %s is a native (IQ) pack "
                                 "built without it: python tools/iq_pack.py --gguf <shard 1> --out %s --experts-bin\n",
                         o.pack.c_str(), o.pack.c_str());
            return 2;
        }
        if (!src.open(o.pack, g.n_layers, g.n_expert, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experts via mmap (--mmap-experts; the A/B arm of R2.1)\n");
        srcp = &src;
    } else {
        arena_src.set_gguf(o.native_preset);   // plan v0.3 P6: a native pack may take its experts from shard 1
        // On the multi-GPU Windows experiment, start with at most 8 GiB of mapped host pages.
        // Unregistered layers remain in the resident arena and use the CPU expert path.
        // (a layer split across GPUs too: pinning all of it into two contexts leaves WDDM refusing every later
        // allocation - measured on the 5080 + 3090 rig: cudaMemGetInfo and the next cudaMalloc fail)
        const uint64_t pin_limit = (o.expert_cache_remote[0] > 0 || multi_gpu) ? (8ull << 30) : 0;
        if (!arena_src.open(o.pack, g.n_layers, g.n_expert, /*threads=*/6, err, pin_limit,
                            o.shared_expert_arena)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: expert arena: %s\n", arena_src.note().c_str());
        // A BORROWED ARENA LOADED NOTHING, and `load_gib_per_second()` is 0.0 for it.  Printing the rate line
        // anyway reads as "46.84 GiB at 0.00 GiB/s" - a catastrophic disk - when in fact the second server did
        // the smart thing and mapped the bytes the first one already wrote.  Say which of the two happened.
        if (arena_src.borrowed())
            std::fprintf(stderr, "strata generate: expert arena borrowed from another process: %.2f GiB mapped, "
                                 "nothing loaded\n",
                         (double) strata::kernels::cpu::expert_layout().total / (1024.0 * 1024 * 1024));
        else
            std::fprintf(stderr, "strata generate: loaded %.2f GiB at %.2f GiB/s\n",
                         (double) strata::kernels::cpu::expert_layout().total / (1024.0 * 1024 * 1024),
                         arena_src.load_gib_per_second());
        // A rate under ~0.2 GiB/s is not the hardware.  Task Scheduler / service contexts throttle this
        // read+fill about 24x (measured 0.05 vs 1.42 GiB/s for the same binary, args and cache state; the
        // scheduler's defaults - Below normal priority and a least-privilege token - were the only
        // difference between the runs).  Say so instead of letting the user blame the disk; see
        // docs/DETAILS.md, "Running it at startup (Task Scheduler)".
#ifdef _WIN32   // a Windows launch context; elsewhere a load this slow is the disk
        if (arena_src.load_gib_per_second() > 0.0 && arena_src.load_gib_per_second() < 0.2) {
            std::fprintf(stderr,
                         "strata generate: hint: ~24x below what this hardware streams from a normal "
                         "launch. If Strata is started by Task Scheduler or a service, register the task "
                         "with Priority 4 (Normal) and 'Run with highest privileges' - the scheduler's "
                         "defaults (Below normal + a least-privilege token) throttle the load. See "
                         "docs/DETAILS.md ('Running it at startup').\n");
        }
#endif
        srcp = &arena_src;
    }
    strata::kernels::cpu::ExpertPool pool(o.pool_workers, /*pin=*/true, /*host_works=*/!o.no_host_worker);
    if (o.no_ple_prefetch) strata::kernels::ple_prefetch_enable(false);
    // ---- R4's slot storage.  Allocated AFTER the weights and the session, so `cudaMemGetInfo` inside `open`
    // sees the memory this process actually has left rather than the card's idle figure - and refuses with both
    // numbers if the slots do not fit, instead of handing back a cache smaller than it was asked for.
    mem_mark("the weights, the session and the drafter");
    strata::core::ExpertCache xcache;
    // THE HEAD BEFORE THE CACHE.  The expert cache takes what is free minus the reserve, so everything allocated
    // after it comes out of the reserve.  The native head (~0.5 GB with IQ3_S) was loaded after it and ate most of
    // the 700 MiB: 128K IQ3_S ended with 30 MiB free, the driver paged, and a request stalled for good at its first
    // verify window.  Loaded first, the cache is sized around it.
    const strata::core::WeightRef* wo = wt.find("output.weight");
    if (wo == nullptr) { std::fprintf(stderr, "strata generate: output.weight is missing\n"); return 1; }
    const int64_t n_vocab = wo->ne1;
    strata::core::NativeHead native_head;
    if (!o.native_head_gguf.empty() && !multi_gpu) {   // a layer split's head is on its last stage
        if (!native_head.load(o.native_head_gguf, g.n_embd, n_vocab, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: experimental native Q5_K head, %llu bytes\n",
                     (unsigned long long) native_head.weight_bytes());
    }
    std::vector<float> logits((size_t) n_vocab);
    float* d_logits = nullptr;
    if (cudaMalloc(&d_logits, (size_t) n_vocab * 4) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: the logits buffer failed\n");
        return 1;
    }
    const bool auto_cache = o.expert_cache < 0;
    if (o.expert_cache < 0) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        // Plan v0.3 P5: the batched prompt path's chunk buffers are allocated later, so they are reserved here -
        // under WDDM an over-subscribed allocation does not fail, it pages to system memory and crawls.
        // (with borrowing - the default with a profile - the prompt path lends cache slots instead; `pf_borrow` is
        // the predicate a local `borrow` was here, hoisted above so both cache-size branches read the same one)
        const int64_t prefill_mib = (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
        // the draft layer's head and logits are allocated when it binds, after this: 0.1.27's CJK subset made them
        // ~110-180 MiB larger, and out of the reserve they left 16 GB cards below the stall line (#199)
        const int64_t mtp_bind = (!o.mtp.empty() && native_head.loaded())
                                     ? (int64_t) mtp.bind_bytes(native_head.row_bytes(), n_vocab) : 0;
        const int64_t reserve = (((int64_t) o.vram_reserve_mib + prefill_mib) << 20) + mtp_bind;
        int64_t slots = ((int64_t) free_b - reserve) / (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        if (!profile.empty()) slots = std::min<int64_t>(slots, (int64_t) profile.size());
        o.expert_cache = (int) std::max<int64_t>(slots, 0);
        std::fprintf(stderr, "strata generate: expert cache auto: %.2f GiB free, %d MiB reserved (+%lld MiB for the "
                             "draft head) -> %d slots\n",
                     (double) free_b / 1073741824.0, o.vram_reserve_mib, (long long) (mtp_bind >> 20), o.expert_cache);
        if (o.expert_cache == 0)   // the verify window cannot start without it (#174): say what makes room
            std::fprintf(stderr, "strata generate: no VRAM is left for the expert cache: lower --max-context, use "
                                 "--kv k8v4, run images on the CPU, or close other programs that use the GPU\n");
    } else if (multi_gpu && o.expert_cache > 0) {
        // an explicit cache size leaves room for the prompt path's buffers and the reserve, or the first prompt
        // fails with "device buffers ... do not fit" (with borrowing - the default with a profile - the path lends
        // slots instead and `prefill_mib` is 0, so only the reserve is checked)
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const int64_t prefill_mib = (o.prefill_chunk > 0 && !pf_borrow) ? 160 + (o.prefill_chunk * 680) / 1024 : 0;
        const int64_t reserve = ((int64_t) o.vram_reserve_mib + prefill_mib) << 20;
        const int64_t fit = std::max<int64_t>(((int64_t) free_b - reserve) / (int64_t) strata::kernels::cpu::expert_layout().max_blob, 0);
        if (o.expert_cache > fit) {
            std::fprintf(stderr, "strata generate: layer split: --expert-cache %d leaves no room for the prompt path's "
                                 "buffers (%lld MiB) on CUDA0: %lld slots\n", o.expert_cache, (long long) prefill_mib,
                         (long long) fit);
            o.expert_cache = (int) fit;
        }
    }
    // plan v0.3 P6: a native pack's blobs differ per layer, so with a profile its slots are sized per pair: the
    // same VRAM holds ~30% more IQ3_XXS experts than slots of the largest blob would
    std::vector<int64_t> sized_slots;
    if (native_pack && o.expert_cache > 0 && !profile.empty()) {
        size_t free_b = 0, total_b = 0;
        cudaMemGetInfo(&free_b, &total_b);
        const auto& lay = strata::kernels::cpu::expert_layout();
        const uint64_t budget = (uint64_t) o.expert_cache * lay.max_blob;   // what the uniform sizing granted
        uint64_t used = 0;
        size_t free_room = free_b > ((size_t) o.vram_reserve_mib << 20) ? free_b - ((size_t) o.vram_reserve_mib << 20) : 0;
        const uint64_t cap = std::min<uint64_t>(budget, (uint64_t) free_room);
        for (const auto& pr : profile) {
            const uint64_t b = (lay.blob_bytes(pr.first) + 255) / 256 * 256;
            if (used + b > cap) break;
            used += b;
            sized_slots.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        o.expert_cache = (int) sized_slots.size();
    }
    if (o.expert_cache > 0) {
        // keep the first `keep_bytes` of the cache (the profile's hottest experts first); false when nothing is left
        auto shrink_to = [&](int64_t keep_bytes) -> bool {
            if (keep_bytes <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            if (!sized_slots.empty()) {
                int64_t used = 0;
                size_t keep = 0;
                while (keep < sized_slots.size() && used + (sized_slots[keep] + 255) / 256 * 256 <= keep_bytes)
                    used += (sized_slots[keep++] + 255) / 256 * 256;
                sized_slots.resize(keep);
                o.expert_cache = (int) keep;
            } else {
                o.expert_cache = (int) (keep_bytes / (int64_t) strata::kernels::cpu::expert_layout().max_blob);
            }
            if (o.expert_cache <= 0) { o.expert_cache = 0; sized_slots.clear(); return false; }
            return true;
        };
        auto cache_bytes = [&]() -> int64_t {
            if (sized_slots.empty()) return (int64_t) o.expert_cache * (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            int64_t b = 0;
            for (const int64_t s : sized_slots) b += (s + 255) / 256 * 256;
            return b;
        };
        // With `--expert-cache auto` the reserve must still be free once the slots are WRITTEN: under WDDM an
        // allocation is not resident until it is touched, and the free figure read before it can be ~1 GB too
        // high.  A cache sized from it filled the card to 0 MiB, the driver then paged, and a request that needed a
        // page back while the verify graph spun on a host flag never finished.  So the slots are zeroed and the
        // free figure read again; while it is short of the reserve the cache is reopened smaller.
        // STRATA_TEST_CACHE_FAIL=N: the first N opens fail as an out-of-commit cudaMalloc does (tests the retry)
        int fake_fails = std::getenv("STRATA_TEST_CACHE_FAIL") ? std::atoi(std::getenv("STRATA_TEST_CACHE_FAIL")) : 0;
        int failed = 0;
        int zero_reads = 0;
        for (int attempt = 0;; ++attempt) {
            bool ok = false;
            if (fake_fails > 0) {
                --fake_fails;
                err = "ExpertCache: cudaMalloc failed: out of memory (STRATA_TEST_CACHE_FAIL)";
            } else {
                ok = sized_slots.empty()
                    ? xcache.open(o.expert_cache, g.n_layers, g.n_expert, (int64_t) strata::kernels::cpu::expert_layout().max_blob, err)
                    : xcache.open_sized(sized_slots, g.n_layers, g.n_expert, err);
            }
            if (!ok) {
                // Issue #60: on Windows a device allocation is also charged to the system commit (RAM + page file),
                // so with a small page file the cache's one big cudaMalloc fails while the VRAM is free.  An auto
                // cache then tries three quarters of the size, a few times, instead of stopping the engine.
                char commit[96] = "";
#if defined(_WIN32)
                MEMORYSTATUSEX ms{};
                ms.dwLength = sizeof ms;
                if (GlobalMemoryStatusEx(&ms))
                    std::snprintf(commit, sizeof commit, " (Windows has %.1f GiB of commit left: RAM + page file)",
                                  (double) ms.ullAvailPageFile / 1073741824.0);
#endif
                if (auto_cache && failed < 8 && shrink_to(cache_bytes() / 4 * 3)) {
                    ++failed;
                    std::fprintf(stderr, "strata generate: %s%s; trying a smaller expert cache: %d slots\n", err.c_str(),
                                 commit, o.expert_cache);
                    continue;
                }
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
#if defined(_WIN32)
                std::fprintf(stderr, "strata generate: on Windows the graphics card's memory also needs room in the page "
                                     "file: set it to \"System managed\" (System > About > Advanced system settings > "
                                     "Performance > Advanced > Virtual memory), or lower --expert-cache\n");
#endif
                return 1;
            }
            if (!auto_cache || attempt - failed >= 6) break;
            cudaMemset(xcache.device_slot(0), 0, (size_t) xcache.bytes());
            cudaDeviceSynchronize();
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            const int64_t want = (int64_t) o.vram_reserve_mib << 20;
            if ((int64_t) free_b >= want - (64ll << 20)) break;
            // short by (want - free); a figure of 0 only says "at least": the first two such reads give back 1 GiB
            // each (under WDDM the free figure read before the allocation runs ~0.7 GiB high), later ones a quarter
            int64_t give = want - (int64_t) free_b + (64ll << 20);
            if (free_b < ((size_t) 16 << 20))
                give = std::max<int64_t>(give, ++zero_reads <= 2 ? 1ll << 30 : xcache.bytes() / 4);
            const int64_t keep_bytes = xcache.bytes() - give;
            std::fprintf(stderr, "strata generate: only %lld MiB free once the slots are written (reserve %d MiB); "
                                 "shrinking the expert cache\n", (long long) (free_b >> 20), o.vram_reserve_mib);
            xcache.close();
            if (!shrink_to(keep_bytes)) break;
        }
        if (failed > 0 && o.expert_cache > 0)
            std::fprintf(stderr, "strata generate: expert cache: %d slots (%.2f GiB) after %d smaller tries - a bigger "
                                 "page file lets it use more of the free VRAM\n",
                         o.expert_cache, (double) xcache.bytes() / 1073741824.0, failed);
    }
    if (o.expert_cache > 0) {
        std::fprintf(stderr, "strata generate: expert cache %lld slots, %.2f GiB of VRAM; policy is\n",
                     (long long) xcache.slots(), xcache.gib());
        mem_mark("opening the expert cache");
        xcache.set_per_layer_admission(o.expert_cache_per_layer);
        // Round 328 warned here that the GPU hit path was wrong (tokens diverged from a cache-off run from
        // token 0). That fault was fixed long since (native_expert_parity, expert_parity, the grouped kernels'
        // tests), and the warning outlived it (issue #23). What remains is rounding: a GPU expert and the CPU's
        // compute the same quantized expert with different float order, so a near-tie can flip. Measured teacher-
        // forced on 2,557 tokens (bench/results/2026-09-27-cache-parity): 95-98% same top-1, and perplexity equal
        // (on - off = -0.005 +- 0.005 nats). Neither output is more correct than the other.
        std::fprintf(stderr,
                     "strata generate: the GPU computes the experts in the cache; it rounds differently from the CPU,\n"
                     "                 so a reply can differ slightly from a run without the cache (same quality:\n"
                     "                 bench/results/2026-09-27-cache-parity).\n");
        if (o.expert_cache_per_layer) {
            int64_t lo = 0, hi = 0;
            xcache.layer_slot_range(0, lo, hi);
            std::fprintf(stderr, "                 R4.2g PER-LAYER: each layer owns %lld slots (%lld..%lld).\n",
                         (long long) (hi - lo), (long long) lo, (long long) (hi - 1));
        } else if (profile.empty()) {
            std::fprintf(stderr, "                 compulsory-miss (fills with whatever the run routes first).\n");
        } else {
            std::fprintf(stderr, "                 PROFILE, ranked by routing frequency, no eviction.\n");
        }
    }
    // ---- R4.2e: fill the tier from the profile.  This is the only place the plan is applied, and it runs
    // ONCE: with `slots` pairs and `slots` slots the cache is full when this returns, so the decode-time
    // admission finds no room and every non-profiled expert stays a CPU miss.  That is what makes the profile
    // the policy rather than a hint.
    int64_t prefilled = 0;
    if (!profile.empty() && srcp != nullptr) {
        const int64_t want = std::min<int64_t>((int64_t) profile.size(), xcache.slots());
        for (int64_t i = 0; i < want; ++i) {
            const int32_t slot = xcache.admit(profile[(size_t) i].first, profile[(size_t) i].second);
            if (slot == strata::core::kNotResident) break;
            const uint8_t* b = srcp->blob(profile[(size_t) i].first, profile[(size_t) i].second);
            if (b == nullptr || !xcache.fill_slot_blocking(slot, b, err,
                    (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[(size_t) i].first))) {
                std::fprintf(stderr, "strata generate: the profile fill failed at pair %lld: %s\n",
                             (long long) i, err.c_str());
                return 1;
            }
            ++prefilled;
        }
        // **AND ONE SLOT IS READ BACK AND COMPARED.**  A residency table that is right about indices and wrong
        // about bytes produces a plausible token, which is this project's most expensive failure mode; the
        // cache's own `verify_slot` is the check and it costs one 1.38 MB D2H at startup.
        if (prefilled > 0 && !xcache.verify_slot(xcache.slot_of(profile[0].first, profile[0].second),
                                srcp->blob(profile[0].first, profile[0].second), err,
                                (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(profile[0].first))) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        mem_mark("the profile fill");
        std::fprintf(stderr, "strata generate: pre-filled %lld of %lld slots from the profile; slot 0 verified\n",
                     (long long) prefilled, (long long) want);
    }

    for (auto& stp : stages) {
        GpuStage& st = *stp;
        const auto& lay = strata::kernels::cpu::expert_layout();
        // the drafter and the head are already allocated by now (they load above, before this), so what is left
        // to hold back is the windows - and `free_b` has already lost the drafter.
        const int64_t room = stage_room(st.dev, true, false);
        const strata::core::OnDevice on(st.dev);
        std::vector<int64_t> sized;
        int64_t used = 0;
        for (const auto& pr : st.profile) {
            const int64_t b = native_pack ? ((int64_t) lay.blob_bytes(pr.first) + 255) / 256 * 256 : (int64_t) lay.max_blob;
            if (used + b > room) break;
            used += b;
            sized.push_back((int64_t) lay.blob_bytes(pr.first));
        }
        if (sized.empty() ||
            !(native_pack ? st.cache.open_sized(sized, g.n_layers, g.n_expert, err)
                          : st.cache.open((int64_t) sized.size(), g.n_layers, g.n_expert, (int64_t) lay.max_blob, err))) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s\n", st.dev,
                         sized.empty() ? "no room" : err.c_str());
            return 1;
        }
        int64_t filled = 0;
        for (const auto& pr : st.profile) {
            if (filled >= st.cache.slots()) break;
            const int32_t slot = st.cache.admit(pr.first, pr.second);
            if (slot == strata::core::kNotResident) break;
            const uint8_t* b = srcp->blob(pr.first, pr.second);
            if (b == nullptr || !st.cache.fill_slot_blocking(slot, b, err, (int64_t) lay.blob_bytes(pr.first))) {
                std::fprintf(stderr, "strata generate: layer split, CUDA%d profile fill failed at pair %lld: %s\n",
                             st.dev, (long long) filled, err.c_str());
                return 1;
            }
            ++filled;
        }
        if (filled == 0 || !st.cache.verify_slot(st.cache.slot_of(st.profile[0].first, st.profile[0].second),
                                                 srcp->blob(st.profile[0].first, st.profile[0].second), err,
                                                 (int64_t) lay.blob_bytes(st.profile[0].first))) {
            std::fprintf(stderr, "strata generate: layer split, CUDA%d expert cache: %s\n", st.dev,
                         filled == 0 ? "nothing filled" : err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: layer split: CUDA%d runs layers %lld-%lld, expert cache %lld slots "
                             "(%.2f GiB), %lld of its %zu profiled pairs; slot 0 verified\n",
                     st.dev, (long long) st.lb, (long long) (st.le - 1), (long long) st.cache.slots(), st.cache.gib(),
                     (long long) filled, st.profile.size());
    }
    if (multi_gpu)
        std::fprintf(stderr, "strata generate: layer split: CUDA0 runs layers 0-%lld\n", (long long) (split_at[0] - 1));

    std::array<strata::core::RemoteExperts, 3> remote_experts;
    const bool multi_remote = o.expert_cache_remote[1] > 0 || o.expert_cache_remote[2] > 0;
    if (o.expert_cache_remote[0] > 0) {
        if (o.expert_cache <= 0 || profile.empty() || o.no_pool) {
            std::fprintf(stderr, "strata generate: remote experts need --expert-profile, "
                                 "a CUDA0 expert cache and the expert pool\n");
            return 2;
        }
        std::vector<std::pair<int32_t, int32_t>> ranked = profile;
        if (!stages.empty()) {   // a layer split: CUDA0's share of the profile, then the later stages' pairs no cache holds
            for (auto& st : stages)
                for (const auto& pr : st->profile)
                    if (st->cache.slot_of(pr.first, pr.second) < 0) ranked.push_back(pr);
        }
        if (multi_remote) {
            // The shipped frequency profile names only 8000 of 24576 experts. Once exhausted,
            // fill remaining VRAM from unranked pairs in expert-then-layer order: this spreads
            // the tail across all layers instead of concentrating it on layer zero.
            std::vector<uint8_t> seen((size_t) g.n_layers * (size_t) g.n_expert, 0);
            for (const auto& pair : ranked)
                if (pair.first >= 0 && pair.first < g.n_layers && pair.second >= 0 && pair.second < g.n_expert)
                    seen[(size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second] = 1;
            for (int64_t e = 0; e < g.n_expert; ++e)
                for (int64_t l = 0; l < g.n_layers; ++l)
                    if (!seen[(size_t) l * (size_t) g.n_expert + (size_t) e])
                        ranked.emplace_back((int32_t) l, (int32_t) e);
            std::fprintf(stderr, "strata generate: remote ranking: %zu profiled pairs, "
                                 "%zu other pairs to fill CUDA1..3\n", profile.size(), ranked.size() - profile.size());
        }
        std::array<std::vector<std::pair<int32_t, int32_t>>, 3> by_device;
        if (multi_remote) {
            // Either stripe experts for parallel GPU work, or give each layer one
            // secondary GPU to reduce switches and transfers over shared USB4.
            std::vector<uint8_t> assigned((size_t) g.n_layers * (size_t) g.n_expert, 0);
            const int devices = 1 + (o.expert_cache_remote[1] > 0) + (o.expert_cache_remote[2] > 0);
            int next = 0;
            for (const auto& pair : ranked) {
                if (pair.first < 0 || pair.first >= g.n_layers || pair.second < 0 || pair.second >= g.n_expert ||
                    xcache.slot_of(pair.first, pair.second) >= 0) continue;
                const size_t index = (size_t) pair.first * (size_t) g.n_expert + (size_t) pair.second;
                if (assigned[index]) continue;
                int target = -1;
                if (o.expert_cache_remote_placement == "layer") {
                    target = pair.first % devices;
                    // Other layers' owners may still have room: keep scanning ranks.
                    if (by_device[(size_t) target].size() >=
                        (size_t) o.expert_cache_remote[(size_t) target]) continue;
                } else {
                    for (int i = 0; i < devices; ++i) {
                        const int r = (next + i) % devices;
                        if (by_device[(size_t) r].size() < (size_t) o.expert_cache_remote[(size_t) r]) {
                            target = r;
                            break;
                        }
                    }
                    if (target < 0) break;
                }
                assigned[index] = 1;
                by_device[(size_t) target].push_back(pair);
                next = (target + 1) % devices;
            }
            std::fprintf(stderr, "strata generate: remote ranks %s across %d CUDA devices\n",
                         o.expert_cache_remote_placement == "layer" ? "grouped by layer" : "striped", devices);
        } else {
            by_device[0] = std::move(ranked);
        }
        std::vector<uint8_t> claimed((size_t) g.n_layers * (size_t) g.n_expert, 0);
        for (auto& st : stages)   // a layer split: what a stage's cache holds is no helper's
            for (const auto& pr : st->profile)
                if (st->cache.slot_of(pr.first, pr.second) >= 0)
                    claimed[(size_t) pr.first * (size_t) g.n_expert + (size_t) pr.second] = 1;
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0) {
            if (!remote_experts[(size_t) r].open(remote_dev[r], o.expert_cache_remote[(size_t) r],
                     g.n_layers, g.n_expert, by_device[(size_t) r], xcache, *srcp, claimed, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "strata generate: CUDA%d: %lld additional experts, %.2f GiB; "
                                 "results return through pinned host rows\n", remote_dev[r],
                         (long long) remote_experts[(size_t) r].resident(), remote_experts[(size_t) r].gib());
        }
    }

    Drive drive;
    for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
        drive.d.remote[drive.d.remote_count++] = &remote_experts[(size_t) r];
    drive.d.hit_cpu_order = o.expert_cache_cpu_order;
    drive.d.split_rows = !o.no_split_rows;
    drive.d.pool = &pool;
    drive.d.src = srcp;
    drive.d.n_expert = g.n_expert;
    drive.d.jobs.resize((size_t) K);
    // ---- R4.2c: THE HIT PATH.  Every one of these is required for `hits_ready()`, which is all-or-nothing on
    // purpose: a half-configured hit path would compute some experts twice and others not at all, and a token
    // built on that is wrong rather than refused.
    void* hit_scratch = nullptr;
    int32_t* d_hit_slot = nullptr;
    int32_t* d_hit_dst = nullptr;
    uint8_t* d_hit_q8 = nullptr;
    float* d_hit_q8_scale = nullptr;   ///< R4.2h: the fp32 activation scales the CPU path also uses
    float* d_hit_out = nullptr;
    if (o.expert_cache > 0 && !o.no_pool) {
        const uint64_t sb = strata::kernels::moe_hit_grouped_scratch_bytes(K, g.n_embd, strata::kernels::cpu::FF);
        if (cudaMalloc(&hit_scratch, (size_t) sb) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_slot, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_dst, (size_t) K * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_q8, (size_t) (g.n_embd / 32) * 34) != cudaSuccess ||
            // R4.2h: the fp32 activation scales.  Without this the GPU's hits use the block's fp16 `d`
            // while the CPU's misses use `ActQ::scale`, which is fp32 - a 4.761e-04 relative disagreement on
            // every chunk, and the reason enabling the cache changed the tokens.
            cudaMalloc((void**) &d_hit_q8_scale, (size_t) (g.n_embd / 32) * sizeof(float)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_out, (size_t) K * g.n_embd * 4) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the R4 hit path could not allocate its device buffers\n");
            return 1;
        }
        drive.d.cache = &xcache;
        drive.d.cache_stream = main_cs;
        drive.d.cache_base = (const uint8_t*) xcache.device_slot(0);
        drive.d.cache_blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        drive.d.cache_slot_off = xcache.slot_offsets();
        drive.d.hit_scratch = hit_scratch;
        drive.d.parts_out = d_parts;
        drive.d.hit_out = d_hit_out;
        drive.d.parts_elems = K * g.n_embd;
        drive.d.mixed = ss.block.mixed;
        drive.d.x_q8_0_hit = d_hit_q8;
        drive.d.x_q8_0_hit_scale = d_hit_q8_scale;
        drive.d.d_slot = d_hit_slot;
        drive.d.d_dst = d_hit_dst;
        drive.d.h_slot.resize((size_t) K);
        cudaEvent_t hit_done = nullptr;
        if (cudaEventCreate(&hit_done) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the hit path could not create its probe event\n");
            return 1;
        }
        drive.d.hit_done = (void*) hit_done;
        drive.d.hit_poke = !o.no_hit_poke;
        drive.d.h_dst.resize((size_t) K);
        mem_mark("the R4 hit path");
        std::fprintf(stderr, "strata generate: R4 hit path ON - resident experts are computed on the GPU\n");
    }
    // ---- P0.S8: the routing trace.  Only meaningful with the pool running, because the ids arrive through
    // the doorbell that the pool consumes - so `--no-pool` is refused rather than silently producing an empty
    // file that would read as "the router selected nothing".
    // ---- PER-STAGE TIMING.  `--no-capture` only: an event recorded inside a stream capture is silently
    // dropped, so a captured graph cannot carry these events and the numbers would be zeros that read as
    // "every stage is free".  Refusing is the fix.
    if (o.stage_timing) {
        if (!o.no_capture) {
            std::fprintf(stderr, "strata generate: --stage-timing records CUDA events inside the layer path, "
                                 "and an event record inside a stream capture is silently dropped. Pass "
                                 "--no-capture as well.\n");
            return 2;
        }
        if (!strata::core::stage_timing_enable()) {
            std::fprintf(stderr, "strata generate: stage_timing_enable failed\n");
            return 1;
        }
        strata::core::stage_timing_name(0, "gr_read (attn)");
        strata::core::stage_timing_name(1, "attention block");
        strata::core::stage_timing_name(2, "gr_write (attn)");
        strata::core::stage_timing_name(3, "gr_read (ffn)");
        strata::core::stage_timing_name(4, "moe_route");
        strata::core::stage_timing_name(5, "moe_finish");
        strata::core::stage_timing_name(6, "gr_write (ffn)");
        // The GDN block's internals.  It is 36 of the 48 layers, 0.96 ms each, and its entire weight traffic
        // is ~26 MB - so ~0.11 ms at the measured read rate.  ~13 tiny latency-bound launches live in it and
        // a single "attention block" number cannot say which one costs anything.
        strata::core::stage_timing_name(8, "  gdn: quantize x");
        strata::core::stage_timing_name(9, "  gdn: qkv gemv");
        strata::core::stage_timing_name(10, "  gdn: conv+silu");
        strata::core::stage_timing_name(11, "  gdn: l2 norms");
        strata::core::stage_timing_name(12, "  gdn: alpha/beta/gate");
        strata::core::stage_timing_name(13, "  gdn: gdn_step");
        strata::core::stage_timing_name(14, "  gdn: z + out_norm");
        strata::core::stage_timing_name(15, "  gdn: out gemv");
    }
    std::FILE* routing = nullptr;
    if (!o.dump_routing.empty()) {
        if (o.no_pool) {
            std::fprintf(stderr, "strata generate: --dump-routing needs the expert pool; the routed ids reach "
                                 "the host through the doorbell the pool reads. Drop --no-pool.\n");
            return 2;
        }
        routing = std::fopen(o.dump_routing.c_str(), "wb");
        if (routing == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_routing.c_str());
            return 1;
        }
        drive.routing = routing;
    }
    strata::core::PoolFn pool_fn = o.no_pool ? nullptr : &drive_pool;
    // The hit hook rides the same switch as the pool: with no pool there is no `parts` staging to
    // write into, and a hit path with nowhere to write is a wrong token rather than an error.
    strata::core::HitFn hit_fn =
        (o.no_pool || o.expert_cache <= 0) ? nullptr : &strata::core::expert_hit_run;
    void* pool_user = o.no_pool ? nullptr : (void*) &drive;
    std::fprintf(stderr, "strata generate: %d expert-pool workers%s%s\n", pool.workers(),
                 pool.host_works() ? " + the host thread" : "",
                 o.no_pool ? " (UNUSED: --no-pool)" : "");
    // **WHICH CORES THIS PROCESS TOOK, AND WHY.**  The pool pins its workers and the host to physical cores, so
    // on a machine running several Strata servers the assignment is a shared resource, not a private choice:
    // two processes pinned onto the same six CPUs measured 2.5-3x slower each than two processes on disjoint
    // cores.  The pool decided that under a machine-wide lease, and this repo's rule is that an adaptation the
    // engine made has to be printed - a user comparing two servers needs to see the cores each one claims.
    if (!pool.note().empty()) std::fprintf(stderr, "strata generate: %s\n", pool.note().c_str());

    // **THE MISALIGNMENT WARNING THAT STOOD HERE IS GONE, BECAUSE THE MISALIGNMENT IS FIXED.**
    //
    // It said the tokens were not the model's, and it was true: `session_loop` handed layer `l`'s expert
    // outputs to layer `l+1`, which multiplied them by layer `l+1`'s router weights (LEDGER L100).  The loop
    // now runs a captured PAIR per layer - `pre[l]` ending with the router and the doorbell, then the CPU
    // pool, then `post[l]` which combines those experts with THAT layer's weights - so layer `l`'s experts meet
    // layer `l`'s routing.  The generated ids changed the moment it landed, which is what a correctness fix
    // looks like from the outside.
    //
    // The cost is real and is recorded rather than hidden: the window for the CPU pool is now whatever GPU
    // work follows the ring inside `pre[l]`, which is the shared expert and nothing else - 0.038 ms against
    // 0.514 ms of CPU work per layer.  A per-layer CPU expert pool cannot be hidden behind a strictly serial
    // residual chain; the CPU term is answered by Phase 3's VRAM expert cache, not by this pipeline.

    // ---- the graphs
    strata::core::SessionGraphs gr;
    if (!o.no_capture && !native_pack) {   // plan v0.3 P6: a native pack runs verify windows only
        // a layer split's CUDA0 session owns only [0, split_at[0]), so its graphs cover that range; the
        // whole-model replay paths (`session_loop`, the plain generate loop) refuse rather than read another
        // stage's state - a split runs its layers on the stages' verifiers (serve) or prefill stage chain
        if (!strata::core::session_capture(wt, g, ss, d_parts, gr, err, /*split=*/o.gpu_stages, 0,
                                           multi_gpu ? split_at[0] : -1)) {
            std::fprintf(stderr, "strata generate: session_capture: %s\n", err.c_str());
            return 1;
        }
    }

    // **`--no-capture` AND THE EXPERTS ARE MUTUALLY EXCLUSIVE, AND SILENTLY SO.**
    //
    // The CPU expert pool is wired into `session_loop` - the host loop around the captured graphs - and
    // `session_token` has no pool hook at all.  So `--no-capture` did not merely change HOW the layers were
    // launched: it ran the whole model with `parts` left at whatever the buffer held, which is ZERO, and the
    // only symptom was `expert blobs 0` in a stats line nobody had to read.  A run that silently omits the
    // routed experts is not a slow measurement of this model, it is a measurement of a different model.
    //
    // Refusing is the fix.  `--no-pool` is the explicit way to say "I want the GPU-only floor".
    if (o.no_capture && !o.no_pool) {
        std::fprintf(stderr,
                     "strata generate: --no-capture runs `session_token`, which has NO CPU expert pool hook, so "
                     "the routed experts would silently contribute nothing. Pass --no-pool as well if the "
                     "GPU-only floor is what you want.\n");
        return 2;
    }
    // The ladder is written by `session_loop`, and `session_token` does not touch the staging buffer at all - so
    // accepting the flag there would produce a file of uninitialised memory, which reads as a wrong answer rather
    // than as a mistake.  `--no-capture` without `--no-pool` is already refused above, so this catches the pair.
    if (o.no_capture && !o.dump_layers.empty()) {
        std::fprintf(stderr,
                     "strata generate: --dump-layers is written by `session_loop`; `--no-capture` runs "
                     "`session_token` instead, which never fills the staging buffer. Drop one of the two.\n");
        return 2;
    }
    if (o.no_capture && !o.dump_halves.empty()) {
        std::fprintf(stderr,
                     "strata generate: --dump-halves is CAPTURED into the layer graphs, so it needs the "
                     "captured path; `--no-capture` never records it. Drop one of the two.\n");
        return 2;
    }

    mem_mark("the expert cache and the graphs");
    std::fprintf(stderr, "strata generate: session is up (engine %s)\n", STRATA_VERSION);
    auto run_head = [&](void* stream) -> bool {
        if (!native_head.loaded())
            return strata::core::lm_head(wt, g, ss.block, d_logits, stream, err);
        return strata::core::lm_head_mix(wt, g, ss.block, stream, err) &&
               native_head.run(ss.block.mixed, d_logits, stream, err);
    };
    float* d_emb = nullptr;
    if (cudaMalloc(&d_emb, (size_t) g.n_embd * 4) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: the embedding buffer failed\n");
        return 1;
    }
    // **`sample_tokens` TAKES DEVICE POINTERS.**  It is a kernel launch; `logits` and `out` are both read and
    // written on the device.  Passing `logits.data()` - the host vector - faults inside the kernel and the
    // error surfaces at the NEXT synchronising call, which here was the next token's `embed_row`, reporting an
    // illegal access on a weight plane.  Nothing in the parameter names said device.
    int* d_next = nullptr;
    if (cudaMalloc(&d_next, sizeof(int)) != cudaSuccess) {
        std::fprintf(stderr, "strata generate: the sampler output buffer failed\n");
        return 1;
    }

    // **`R` IS BOTH THE INPUT AND THE OUTPUT, SO THE NEW TOKEN'S EMBEDDING HAS TO REPLACE THE OLD RESIDUAL.**
    // At `pos == 0` that is `session_zero`, which is the reference's own initial condition - the embedding
    // broadcast to all `hc` streams.  After that `session_zero` would also wipe the recurrence, so the
    // broadcast is done directly.  Getting this wrong is invisible for exactly one token.
    void* token_stream = o.stream_token ? main_cs : nullptr;
    auto put_input = [&](int64_t tok, int64_t pos) -> bool {
        if (!strata::core::embed_row(wt, g, tok, d_emb, token_stream, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return false;
        }
        if (pos == 0) {
            strata::core::session_zero(ss, g, d_emb, token_stream);
        } else {
            for (int64_t c = 0; c < g.hc; ++c)
                if (cudaMemcpyAsync(ss.R + (size_t) c * g.n_embd, d_emb, (size_t) g.n_embd * 4,
                                    cudaMemcpyDeviceToDevice, (cudaStream_t) token_stream) != cudaSuccess) {
                    std::fprintf(stderr, "strata generate: the residual broadcast failed\n");
                    return false;
                }
        }
        return o.stream_token || cudaDeviceSynchronize() == cudaSuccess;
    };

    strata::kernels::SamplerParams sp;
    sp.greedy = o.greedy;
    sp.seed = o.seed;
    sp.top_k = o.top_k;
    sp.top_p = o.top_p;
    sp.temperature = o.temperature;
    // what this run actually samples with (the speculative loop below gets the same parameters); serve samples
    // per request instead
    if (!o.serve) {
        if (sp.greedy || sp.temperature <= 0.0f)
            std::fprintf(stderr, "strata generate: sampling greedy\n");
        else
            std::fprintf(stderr, "strata generate: sampling temperature=%g top_k=%d top_p=%g seed=%llu\n",
                         (double) sp.temperature, sp.top_k > 0 && sp.top_k < 64 ? sp.top_k : 64, (double) sp.top_p,
                         (unsigned long long) sp.seed);
    }

    std::FILE* dump = nullptr;
    const int64_t dump_positions = (int64_t) o.tokens.size() - 1 + o.max_new;
    if (!o.dump_logits.empty()) {
        if (dump_positions > INT32_MAX || n_vocab > INT32_MAX) {
            std::fprintf(stderr, "strata generate: logits dump dimensions exceed int32\n");
            return 2;
        }
        dump = std::fopen(o.dump_logits.c_str(), "wb");
        if (dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_logits.c_str());
            return 1;
        }
        // **THE COUNT IS `n_prompt - 1 + max_new`, NOT `n_prompt + max_new`.**  The loop writes one row per
        // position from 0, and it stops once `produced` holds `max_new` tokens - and `produced` only starts
        // receiving at position `n_prompt - 1`.  So a 5-token prompt with `--max-new 6` writes 10 rows, and the
        // header used to claim 11.  A header that describes a different file from the one written is the same
        // class of defect as a self-check that verifies the wrong invariant: anything reading the count instead
        // of the size gets a wrong answer that looks authoritative.  `tools/logits_identical.py` caught it by
        // parsing the header and refusing the file.
        const int32_t n_rows = (int32_t) strata::program::logits_selection::row_count(dump_positions, o.logits_stride);
        const int32_t hdr[2] = {(int32_t) n_vocab, n_rows};
        if (std::fwrite(hdr, sizeof hdr, 1, dump) != 1) {
            std::fprintf(stderr, "strata generate: cannot write logits header\n");
            std::fclose(dump);
            return 1;
        }
    }

    // ---- THE C1 ORACLE: ONE RESIDUAL SNAPSHOT PER LAYER PER POSITION, so the engine can be bisected against
    // `llama-debug`'s `l_last-<il>` node instead of against a single end-to-end perplexity.  The buffer is
    // PINNED because `session_loop` enqueues a device-to-host copy into it after every layer and the transfer
    // would otherwise be staged through a pageable bounce buffer on the critical path.
    std::FILE* layer_dump = nullptr;
    float* layer_stage = nullptr;
    const size_t layer_floats = (size_t) (g.n_layers + 1) * (size_t) g.hc * (size_t) g.n_embd;
    if (!o.dump_layers.empty()) {
        layer_dump = std::fopen(o.dump_layers.c_str(), "wb");
        if (layer_dump == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_layers.c_str());
            return 1;
        }
        if (cudaHostAlloc((void**) &layer_stage, layer_floats * sizeof(float), cudaHostAllocDefault) !=
            cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot pin the layer-dump staging buffer\n");
            return 1;
        }
    }

    // ---- prefill is the DECODE PATH ONE TOKEN AT A TIME, which `phase-2-correct-engine.md:12-13` says is
    // fine here: "process the prompt through the decode-style graphs in small batches; a 19K-token prompt will
    // take minutes".  A real batched prefill is P2.S6's other half and is not this.
    //
    // **THE LOOP IS `feed -> 48 layers -> head -> sample -> feed`, AND THE FIRST GENERATED TOKEN COMES FROM THE
    // LAST *PROMPT* POSITION.**  The first version sampled only on the decode positions, so `produced` was
    // still empty when the first generated position asked for `produced.back()` - an out-of-bounds read on an
    // empty vector.  Teacher forcing below is what makes the distinction unnecessary to special-case: for every
    // position before the last prompt one, the next input is the PROMPT's next token, and after that it is the
    // sampled one.
    std::vector<int64_t> produced;
    double total_ms = 0;
    double prefill_ms = 0;   // positions 0 .. n_prompt-2: prompt tokens that only condition
    const Clock::time_point t_start = Clock::now();
    double ttft_ms = 0;
    const int64_t n_prompt = (int64_t) o.tokens.size();
    int64_t tok = o.tokens[0];

    // ---- THE PURE-GPU MEASUREMENT.  `session_replay` launches all 48 `pre` graphs back to back on one stream
    // with NO host work between them - no doorbell poll, no pool, no parts copy - so what it times is the GPU
    // executing the layer sequence and nothing else.  It had been declared, defined and never called since the
    // day it was written.
    //
    // **THIS IS THE MEASUREMENT THAT SAYS WHETHER THE ENGINE IS HOST-BOUND OR GPU-BOUND**, and the stage table
    // cannot answer it: those events measure the interval between two marks on a stream, which includes every
    // gap where the GPU sat idle waiting for the host to enqueue the next kernel.  In `--no-capture` those gaps
    // are the host's launch latency and they are proportional to the KERNEL COUNT rather than to any work, so
    // the no-capture stage shares are shares of kernel count - which is why the attention block, with the most
    // kernels, looks like 55% of the token there.
    // ---- R0.9: THE PER-STAGE TABLE ON THE CAPTURED GRAPH.
    if (o.gpu_stages) {
        strata::core::doorbell_reset(db);
        double mix = 0, ffn = 0, post = 0;
        if (!strata::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const int reps = 20;
        double t_mix = 0, t_ffn = 0, t_post = 0;
        for (int r = 0; r < reps; ++r) {
            if (!strata::core::session_replay_stages(g, 0, 0, ss, gr, main_cs, mix, ffn, post, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            t_mix += mix;
            t_ffn += ffn;
            t_post += post;
        }
        const double tot = t_mix + t_ffn + t_post;
        std::printf("\nper-stage GPU time on the CAPTURED graph, one token over %lld layers\n",
                    (long long) g.n_layers);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "mixer (gr_read+attn+gr_write)",
                    t_mix / reps, t_mix / reps / (double) g.n_layers, 100.0 * t_mix / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "ffn front + router",
                    t_ffn / reps, t_ffn / reps / (double) g.n_layers, 100.0 * t_ffn / tot);
        std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  %6.1f%%\n", "post (moe_finish+gr_write)",
                    t_post / reps, t_post / reps / (double) g.n_layers, 100.0 * t_post / tot);
        std::printf("  %-22s %9.3f ms/token\n", "sum of the three", tot / reps);

        // ---- AND THE MIXER BY LAYER KIND, because 36 of the 48 are GDN and 12 are QSA and a total cannot
        // separate them.  Round 309's uncaptured table put GDN at 10.88 ms for 36 layers against QSA's 4.56 for
        // 12, which would make the recurrence the largest single R3 target - and that table had `moe_finish`
        // wrong by 4x, so the ratio is re-derived here from the captured graph rather than inherited.
        {
            // **ACCUMULATED OVER `reps`, NOT MEASURED ONCE AND THEN DIVIDED.**  The first version called the
            // per-layer replay a single time and printed `gdn / reps`, which reported GDN at 0.480 ms/token
            // against a mixer total of 14.039 - a factor of exactly `reps`, and the tell was that
            // 0.480 + 0.220 = 0.700 = 14.039 / 20.
            std::vector<double> acc((size_t) g.n_layers, 0.0), per;
            double f2 = 0, p2 = 0;
            for (int r = 0; r < reps; ++r) {
                if (!strata::core::session_replay_stages_per_layer(g, 0, 0, ss, gr, main_cs, per, f2, p2, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                for (int64_t l = 0; l < g.n_layers; ++l) acc[(size_t) l] += per[(size_t) l];
            }
            double gdn = 0, qsa = 0, worst = 0;
            int64_t ng = 0, nq = 0, worst_l = 0;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                const double v = acc[(size_t) l] / reps;
                if (strata::core::is_qsa_layer(g, l)) { qsa += v; ++nq; }
                else { gdn += v; ++ng; }
                if (v > worst) { worst = v; worst_l = l; }
            }
            std::printf("\n  the mixer by layer kind, averaged over %d runs\n", reps);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "GDN layers",
                        gdn, ng ? gdn / (double) ng : 0.0, (long long) ng);
            std::printf("  %-22s %9.3f ms/token  %8.3f ms/layer  (%lld layers)\n", "QSA layers",
                        qsa, nq ? qsa / (double) nq : 0.0, (long long) nq);
            std::printf("  %-22s layer %lld at %.3f ms\n", "worst mixer layer", (long long) worst_l, worst);
            std::printf("  %-22s %9.3f ms/token (must equal the mixer above)\n", "GDN + QSA", gdn + qsa);
        }

        // ================================ R0.11: THE FIVE STAGES SEPARATELY ================================
        //
        // Prefixes 1..5 are replayed per layer with the residual restored between them, and consecutive
        // differences are the per-stage times.  **THE CHECK IS THAT THE FIVE SUM TO THE THREE-GRAPH TOTAL** -
        // the same independent-restatement test that caught round 320's divide-by-reps bug, and it is the only
        // reason to believe a table built out of differences.
        {
            std::vector<double> acc5(5, 0.0), per, s5;
            for (int r = 0; r < reps; ++r) {
                if (!strata::core::session_replay_stage_prefixes(g, 0, 0, ss, gr, main_cs, s5, per, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
                for (int k = 0; k < 5; ++k) acc5[(size_t) k] += s5[(size_t) k];
            }
            static const char* sn[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                        "3 gr_read (ffn)", "4 moe_route (router)"};
            const char* kind[5] = {"GR", "ATTN", "GR", "GR", "ROUTER"};
            double tot5 = 0, gr_ms = 0;
            std::printf("\n  the five stages separately, by differencing prefixes\n");
            std::printf("  %-24s %-8s %10s %10s %8s\n", "stage", "kind", "ms/token", "ms/layer", "share");
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                tot5 += v;
                if (k != 1 && k != 4) gr_ms += v;
                std::printf("  %-24s %-8s %10.3f %10.4f %7.1f%%\n", sn[k], kind[k], v,
                            v / (double) g.n_layers, 0.0);
            }
            for (int k = 0; k < 5; ++k) {
                const double v = acc5[(size_t) k] / reps;
                (void) v;
            }
            std::printf("  %-24s %-8s %10.3f\n", "sum of the five", "", tot5);
            std::printf("  %-24s %-8s %10.3f   <- R3.3's target is <= 3 ms for all four passes\n",
                        "GR passes (0,2,3)", "GR", gr_ms);
            std::printf("\n  the three-graph total above was %.3f ms/token; the five must account for it.\n",
                        tot / reps);

            // ================================ R3.5c: THE SAME TABLE, A DIFFERENT WAY ================================
            //
            // The differencing table above mixes five graphs per layer, so stage 4's interval carries the launch
            // of the FULL five-stage graph while stage 3's carries a four-stage one.  This sweep launches ONE
            // graph type per layer and nothing else, so that bias cannot exist.  **If the two disagree, the
            // difference IS the bias and this one is right** - and stage 4 is the router, so it is exactly the
            // number that must not be wrong.
            {
                double sweep[6] = {0, 0, 0, 0, 0, 0};
                for (int k = 1; k <= 5; ++k) {
                    double acc = 0, one = 0;
                    for (int r = 0; r < reps; ++r) {
                        if (!strata::core::session_replay_stage_sweep(g, 0, 0, ss, gr, main_cs, k, one, err)) {
                            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                            return 1;
                        }
                        acc += one;
                    }
                    sweep[k] = acc / reps;
                }
                std::printf("\n  the same five stages, by SWEEPING each prefix back to back (no graph switching)\n");
                std::printf("  %-24s %10s %10s %12s\n", "stage", "prefix sweep", "difference", "bias");
                static const char* sn2[5] = {"0 gr_read (attn)", "1 attention", "2 gr_write (attn)",
                                             "3 gr_read (ffn)", "4 moe_route (router)"};
                for (int k = 0; k < 5; ++k) {
                    const double sw = sweep[k + 1] - sweep[k];
                    const double df = acc5[(size_t) k] / reps;
                    std::printf("  %-24s %10.3f %10.3f %11.1f%%\n", sn2[k], sw, df,
                                sw != 0.0 ? 100.0 * (df / sw - 1.0) : 0.0);
                }
                std::printf("  %-24s %10.3f   (full pre, 48 layers)\n", "prefix 5 total", sweep[5]);
            }
        }
        std::printf("\n  compare `--gpu-only-full`, which replays the same work as TWO graphs per layer.  The\n");
        std::printf("  three sum slightly above it because each launch carries the driver's gap.\n");
        strata::core::session_graphs_free(gr);
        strata::core::doorbell_free(db);
        cudaFree(d_next);
        return 0;
    }

    if (o.graph_only) {
        strata::core::doorbell_reset(db);
        // one warm pass so the first launch does not pay for page mapping
        if (!strata::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
            std::fprintf(stderr, "strata generate: session_replay warm: %s\n", err.c_str());
            return 1;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata generate: session_replay warm faulted\n");
            return 1;
        }
        const int reps = 20;
        const Clock::time_point t0 = Clock::now();
        for (int r = 0; r < reps; ++r) {
            if (!strata::core::session_replay(g, 0, 0, ss, gr, main_cs, err)) {
                std::fprintf(stderr, "strata generate: session_replay: %s\n", err.c_str());
                return 1;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata generate: session_replay faulted: %s\n", cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count() / (double) reps;
        std::printf("pre graphs only   %8.2f ms per token over %lld layers  ->  %.2f tok/s of GPU work\n", ms,
                    (long long) g.n_layers, ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms per layer\n", ms / (double) g.n_layers);
        return 0;
    }

    // ---- R0.3: THE TRUE PER-TOKEN GPU FLOOR.
    //
    // `--graph-only` above launches ONLY `gr.execs[l]`, the `pre` graphs.  It omits the 48 `post` graphs - the
    // shared expert, the combine and the second `gr_write` - and the LM head.  Everything this project published
    // as "39.8 ms pure GPU" came from that loop while being described as the whole GPU, which also made the
    // "host's share = 49.4 - 39.8 = 9.6 ms" figure wrong by however much the missing work costs.  See
    // Memory/ERRORS.md A4/A5.
    //
    // This flag is what "pure GPU" has to mean, and it replaces that number everywhere.  No pool runs, so
    // `parts` keeps whatever the buffer holds and the timing is GPU work alone.
    if (o.gpu_only_full) {
        strata::core::doorbell_reset(db);
        const int reps = 20;
        double ms_layers = 0, ms_head = 0;
        for (int r = -1; r < reps; ++r) {          // r == -1 is the warm pass, not counted
            const Clock::time_point t0 = Clock::now();
            if (!strata::core::session_replay_full(g, 0, 0, ss, gr, main_cs, err)) {
                std::fprintf(stderr, "strata generate: session_replay_full: %s\n", err.c_str());
                return 1;
            }
            // The sync is INSIDE the interval on purpose: it is the wait for the GPU, so t1 - t0 is GPU time.
            if (cudaStreamSynchronize((cudaStream_t) main_cs) != cudaSuccess) {
                std::fprintf(stderr, "strata generate: gpu-only-full layers faulted: %s\n",
                             cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            const Clock::time_point t1 = Clock::now();
            if (!run_head(main_cs)) {
                std::fprintf(stderr, "strata generate: gpu-only-full lm_head: %s\n", err.c_str());
                return 1;
            }
            if (cudaStreamSynchronize((cudaStream_t) main_cs) != cudaSuccess) {
                std::fprintf(stderr, "strata generate: gpu-only-full head faulted: %s\n",
                             cudaGetErrorString(cudaGetLastError()));
                return 1;
            }
            const Clock::time_point t2 = Clock::now();
            if (r < 0) continue;
            ms_layers += std::chrono::duration<double, std::milli>(t1 - t0).count();
            ms_head += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        ms_layers /= (double) reps;
        ms_head /= (double) reps;
        const double ms = ms_layers + ms_head;
        std::printf("GPU floor pre+post+head %7.2f ms per token  ->  %.2f tok/s of GPU work\n", ms,
                    ms > 0 ? 1000.0 / ms : 0.0);
        std::printf("                  %8.3f ms layers (%lld x pre+post)\n", ms_layers, (long long) g.n_layers);
        std::printf("                  %8.3f ms per layer\n", ms_layers / (double) g.n_layers);
        std::printf("                  %8.3f ms LM head\n", ms_head);
        return 0;
    }

    // ---- **THE TOKEN PATH ALLOCATES NOTHING (P2.T10, review finding H3).**
    //
    // `session_loop` used to allocate its pinned staging buffer, its probe event and its host pin ON EVERY
    // TOKEN, and `cudaFreeHost` at the end of each call implicitly synchronises the device - so every token
    // finished with a device-wide sync nobody asked for.  The scratch is created once here and reused; it also
    // owns the host pin for the whole session rather than taking and releasing it per token.
    strata::core::SessionLoopScratch loop_scratch;
    struct ScratchFree {
        strata::core::SessionLoopScratch* p;
        ~ScratchFree() { if (p != nullptr) p->free(); }
    } scratch_free{&loop_scratch};
    // Initialised unconditionally, including under --no-pool: the loop validates the scratch it is handed, so
    // passing a default-constructed one is an error rather than a fallback.  (It was, and the guard caught it -
    // which is the point of the guard.)  One allocation at setup either way.
    if (!loop_scratch.init((size_t) K * g.n_embd * 4, err)) {
        std::fprintf(stderr, "strata generate: %s\n", err.c_str());
        return 1;
    }
    // Plan v0.3 P3: the whole token as ONE graph whenever nothing needs a host step between the ring and post[l]
    // (the VRAM expert tier and the per-layer dumps do).  `--no-token-graph` keeps two graphs per layer.
    strata::core::TokenGraph tgraph;
    struct TokenGraphFree {
        strata::core::TokenGraph* p;
        ~TokenGraphFree() { strata::core::token_graph_free(*p); }
    } tgraph_free{&tgraph};
    // Plan v0.3 P4: with a PROFILE-filled cache the residency is static, so the hit decision moves onto the
    // device and the token graph keeps it.  (A cache filled on demand still needs the per-layer host path.)
    std::vector<int32_t> host_res;
    int32_t* d_res = nullptr;
    int32_t* d_hit_count = nullptr;
    strata::core::TokenHits thits;
    const bool graph_hits = hit_fn != nullptr && !profile.empty() && !o.no_pool;
    if (graph_hits && !o.no_capture && !o.no_token_graph && layer_dump == nullptr && half_dump == nullptr) {
        host_res.assign((size_t) (g.n_layers * g.n_expert), strata::core::kNotResident);
        int64_t resident = 0;
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const int st = multi_gpu ? stage_of(l) : 0;
                const int32_t slot = st > 0 ? stages[(size_t) st - 1]->cache.slot_of(l, e) : xcache.slot_of(l, e);
                host_res[(size_t) (l * g.n_expert + e)] = slot;
                if (slot != strata::core::kNotResident) ++resident;
            }
        if (cudaMalloc((void**) &d_res, host_res.size() * sizeof(int32_t)) != cudaSuccess ||
            cudaMalloc((void**) &d_hit_count, sizeof(int32_t)) != cudaSuccess ||
            cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the device residency table could not be staged\n");
            return 1;
        }
        thits.d_res = d_res;
        thits.n_expert = g.n_expert;
        for (auto& st : stages) {   // layer split across GPUs: the same table on every device
            const strata::core::OnDevice on(st->dev);
            if (cudaMalloc((void**) &st->d_res, host_res.size() * sizeof(int32_t)) != cudaSuccess ||
                cudaMemcpy(st->d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice) !=
                    cudaSuccess) {
                std::fprintf(stderr, "strata generate: layer split: CUDA%d residency table failed\n", st->dev);
                return 1;
            }
        }
        thits.cache_base = drive.d.cache_base;
        thits.blob = drive.d.cache_blob;
        thits.d_slot = drive.d.d_slot;
        thits.d_dst = drive.d.d_dst;
        thits.d_count = d_hit_count;
        thits.x_q8 = drive.d.x_q8_0_hit;
        thits.x_scale = drive.d.x_q8_0_hit_scale;
        thits.scratch = drive.d.hit_scratch;
        thits.hit_out = drive.d.hit_out;
        drive.d.host_res = host_res.data();
        std::fprintf(stderr, "strata generate: token graph hit path: %lld resident experts, decided on the device\n",
                     (long long) resident);
    }
    if (!o.no_capture && !o.no_token_graph && layer_dump == nullptr && half_dump == nullptr &&
        (hit_fn == nullptr || thits.on()) && !native_pack && !multi_gpu) {   // a split's token graph cannot span stages
        if (!strata::core::session_capture_token(wt, g, ss, d_parts, loop_scratch.y_miss, loop_scratch.parts_bytes,
                                                 tgraph, err, thits.on() ? &thits : nullptr)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "strata generate: token graph captured (48 layers, one launch per token)\n");
    }

    // ================================ WHERE THE HOST TERM GOES, PER TOKEN ================================
    //
    // **`--gpu-only-full` MEASURES THE 48 LAYER GRAPHS AND THE LM HEAD AND NOTHING ELSE.**  It never enters
    // this loop, so it does not run `ple_stage_token`, `embed_row`, the whole-vocabulary logits readback, the
    // NaN scan or the sampler - and `--no-pool --stats` against that floor was being read as "the per-layer
    // round trip costs 12.4 ms" when an unknown part of it is per TOKEN, not per layer.  That is the same error
    // the review catalogued as A4/A5, one level down: a difference between two measurements attributed to a
    // mechanism that neither of them isolates.
    //
    // Six accumulators, because the six have different fixes.  Reported in `--stats` as ms/token.  **The
    // boundary after the layer loop is the one that matters**: without it the head's interval swallows all 48
    // layers and the report reads as "head = 50 ms", which is not a thing that can happen to 1.4 ms of GPU
    // work.  That is not hypothetical - it is what the first version of this printed.
    double ms_ple = 0, ms_embed = 0, ms_layers = 0, ms_head = 0, ms_readback = 0, ms_sample = 0;
    int64_t phase_tokens = 0;

    // ================================ plan v0.3 P8: THE PERSISTENT ENGINE (--serve) ================================
    //
    // The weights, the expert arena and the VRAM tier load once; then requests arrive on stdin, one per line,
    //
    //     GEN <max_new> <id,id,...>
    //
    // and each generated token is written to stdout as `T <id>` as soon as its verify window is done, followed by
    //
    //     DONE <generated> <prompt_tokens> <prompt_ms> <decode_ms> <stop|length|cancel> <drafts accepted>
    //          <drafts offered> <prompt tokens reused>
    //
    // Before that, `RESUME <n>` (n prompt tokens are not read again), `PP <position> <prompt_tokens> <ms> <tok/s>`
    // after every prompt chunk, and `REUSED <n>` once the prompt is read.  (`ERR <message>` instead when a request
    // cannot run; `STOP` ends the running request at its next step; `QUIT` ends the process.)  A request continues
    // from the live session or the longest conversation checkpoint its prompt starts with (see ConvCheckpoint),
    // otherwise from an empty sequence (`session_zero`); the rest of the prompt goes through the batched prompt path
    // and its last token through the first verify window - the path all three model files share.  Decoding is greedy.
    // The expert-cache slots (from the end of the cache) that hold the prompt path's buffers for a chunk, and the
    // bytes from the first of them to the end.
    // the share of expert bytes the arena could pin (sizes the prompt path's streamed ring and its lend cap)
    if (srcp != nullptr && o.prefill_chunk > 0) {
        uint64_t pinned = 0, total = 0;
        const auto& lay = strata::kernels::cpu::expert_layout();
        for (int64_t l = 0; l < g.n_layers; ++l)
            for (int64_t e = 0; e < g.n_expert; ++e) {
                const uint64_t b = lay.blob_bytes(l);
                total += b;
                if (srcp->pinned(l, e)) pinned += b;
            }
        strata::prefill::Prefill::set_pinned_share(total ? (double) pinned / (double) total : 1.0);
    }
    auto lend_slots = [&](int64_t c) -> int64_t {
        const uint64_t need = strata::prefill::Prefill::bytes_needed(g, ss, c);
        const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
        int64_t k = (int64_t) ((need + (uint64_t) blob - 1) / (uint64_t) blob);
        if (xcache.slot_offsets() != nullptr) {   // sized slots: take slots from the end until they hold `need`
            k = 0;
            while (k < xcache.slots() &&
                   (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[xcache.slots() - k]) < need) ++k;
        }
        return k;
    };
    // `lend_bytes` went with the single-cache serve loan: a participant's loan is priced by `part_bytes` from its
    // OWN cache, and the only other user of the old helper was the serve path's own relayout.
    auto request_chunk = [](int64_t tokens, int64_t max_chunk) -> int64_t {
        if (tokens <= 0 || max_chunk <= 0) return 0;
        const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255
                                    ? tokens
                                    : ((tokens + 255) / 256) * 256;
        return std::min(max_chunk, rounded);
    };
    // The prompt path's chunk and the slots it borrows for its buffers: the requested chunk halved until it fits,
    // or with --prefill auto the largest of kAutoChunks whose buffers take at most kAutoLendPct % of the slots (a
    // lent slot's expert is streamed during the prompt and refilled after it; measured on a 12 GB card, 32K Q2_0
    // prompt: 4096 791 tok/s, 6144 878, 8192 973 with 69% of the slots lent).  A request lends only what its own
    // prompt needs (Prefill::relayout), so a big chunk costs short prompts nothing.  0 = none fits.
    // at 8192-token chunks nearly every expert streams anyway, so a lent slot costs little: 90% when the
    // copies are DMA from pinned RAM (Q2_0 8192 + a 384-slot ring: 1283 tok/s), 85% when host copies are the
    // limit (lending more only streams more through them).  STRATA_PREFILL_LEND_PCT overrides (tuning).
    // Hoisted out of plan_lend: the serve path's per-stage loans obey the same cap, one participant at a time.
    const int64_t kAutoLendPct = [] {
        const char* v = std::getenv("STRATA_PREFILL_LEND_PCT");
        return v ? (int64_t) std::atoi(v)
                 : (int64_t) (strata::prefill::Prefill::pinned_share() >= 0.9 ? 90 : 85);
    }();
    auto plan_lend = [&](int64_t& chunk) -> int64_t {
        static constexpr int64_t kAutoChunks[] = {8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
        auto slots_for = lend_slots;
        if (o.prefill_auto) {
            for (const int64_t c : kAutoChunks) {
                const int64_t k = slots_for(c);
                if (k + 128 <= xcache.slots() && k * 100 <= kAutoLendPct * xcache.slots()) { chunk = c; return k; }
            }
            return 0;
        }
        for (int64_t c = chunk; c >= 256; c /= 2) {
            const int64_t k = slots_for(c);
            if (k + 128 <= xcache.slots()) { chunk = c; return k; }
        }
        return 0;
    };
    // ---- the resident RAM mode (--resident-experts / --resident-cpu-experts): the experts the GPU cache does not
    // hold are copied from experts.bin into RAM once, so no decode or prompt step reads the file (the plain mmap
    // mode reads them through the OS file cache, which a small-RAM PC keeps giving back to the SSD).  Built here,
    // after the prompt path's lend plan is known: the slots it may lend (the cache's last ones) have their experts
    // streamed during a prompt and copied back after it, so those are kept in RAM too as far as RAM allows.  The
    // bytes are the file's bytes and the placement is the same, so the answers are the plain mmap mode's; the
    // share-of-pinned figure above (which sizes the prompt path) is left as the mmap mode's for the same reason.
    if (o.resident_cpu_experts) {
        int64_t lend_from = -1;
        if (o.prefill_chunk > 0 && !o.no_prefill_borrow && d_res != nullptr && xcache.slots() > 0) {
            int64_t chunk = o.prefill_chunk;
            const int64_t k = plan_lend(chunk);
            if (k > 0) lend_from = xcache.slots() - k;
        }
        if (src.pin_cache_complement(xcache, err, o.resident_pin, {}, lend_from, o.resident_headroom)) {
            if (o.adapt_every > 0 && o.adapt_swaps > 0 &&
                !src.reserve_exchanges(std::min<int64_t>(o.adapt_swaps, 96), err)) {
                std::fprintf(stderr, "strata generate: CPU expert residency: %s\n", err.c_str());
                return 1;
            }
            std::fprintf(stderr, "strata generate: resident RAM mode: %.2f GiB of experts in RAM (%s), %lld in the GPU "
                                 "cache; adaptive swaps %s\n",
                         (double) src.resident_bytes() / 1073741824.0,
                         src.complement_pinned() ? "page-locked" : src.locked_bytes() > 0 ? "locked" : "pageable",
                         (long long) xcache.resident(),
                         o.adapt_every > 0 && o.adapt_swaps > 0 ? "exchange them with the GPU cache (no file reads)"
                                                                : "off");
        } else if (o.resident_soft) {
            std::fprintf(stderr, "strata generate: WARNING: the resident RAM mode does not fit (%s); the experts the "
                                 "GPU does not hold are read from the model folder through the OS file cache "
                                 "(--mmap-experts), which is slower when the RAM cannot keep them\n", err.c_str());
        } else {
            std::fprintf(stderr, "strata generate: CPU expert residency: %s\n", err.c_str());
            return 1;
        }
    }
    if (o.serve) {
        if (o.spec < 2 || o.mtp.empty() || o.prefill_chunk <= 0 ||
            (graph_hits && (thits.d_res == nullptr || host_res.empty()))) {
            std::fprintf(stderr, "strata serve: needs --spec T, --mtp DIR and --prefill CHUNK (and a fillable "
                                 "--expert-cache; the graphed hit path additionally needs --expert-profile P)\n");
            return 2;
        }
        strata::prefill::Prefill sp;
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        int32_t lend_first = -1;          // the first slot the prompt path may borrow (its largest chunk)
        // ---- WHO BORROWS, AND FROM WHOSE CACHE.  One entry per prompt path: CUDA0's (layers [0, split_at[0]),
        // which is the whole model without a split) borrowing the tail of CUDA0's cache, then one per stage
        // borrowing the tail of ITS OWN cache.  A loan is sized by the exact `Prefill::bytes_needed` for the
        // chunk, is laid out by `Prefill::relayout`, and is refilled before any window reads - so outside the
        // prompt the whole cache is expert cache.  THIS IS THE POINT OF THE STRUCT: the loan used to exist only
        // for CUDA0, and `no_prefill_borrow` made every stage instead withhold a chunk-sized reserve from its
        // cache for the entire session, which is what cost the 4-way its context (see the note at the top of the
        // layer-split block).  A stage may only lend the rows for ITS OWN layers: `host_res` is one table whose
        // slot values are indices into whichever cache owns the layer, so a loan that marked rows by slot number
        // alone would hand CUDA0 a slot belonging to another stage's cache.
        struct PfPart {
            strata::core::ExpertCache* cache = nullptr;
            const strata::core::SessionState* ses = nullptr;
            strata::prefill::Prefill* sp = nullptr;
            int dev = -1;                  // -1: leave the device alone (CUDA0)
            int64_t lb = 0, le = 0;        // the layers whose rows this cache holds - the only rows it may lend
            int32_t first = -1;            // the first slot it may lend, for the chunk that was chosen
            int32_t first_now = -1;        // where its buffers are laid out now
            int64_t lent_chunk = 0;
            std::vector<std::pair<int32_t, int32_t>> lent;
            // S0.3 lever 1 (sticky loan): what the layout decisions cost.  `first_now` and `sp->chunk()`
            // ARE the layout, and they survive a refill on purpose - giving the rows back to the cache does
            // not un-carve the buffers - so the next lend that wants the same layout re-lays nothing.
            int64_t relayouts = 0;         // Prefill::relayout calls actually run
            int64_t relayout_skips = 0;    // lends that found the layout already right
            int64_t loan_grows = 0;        // lends that widened the loan in place instead of refilling it
            int64_t refilled = 0;          // rows streamed back into the cache, for the loan bill
            // ---- S3.2b lever 5: the rows this cache lent and has NOT got back.  `lent` is the LIVE loan
            // (the rows the prompt buffers are standing in right now); `led` is the ledger (rows that are
            // non-resident because they are out, whether or not a loan is live).  They overlap while a loan
            // is live and are disjoint the moment it is returned lazily.  `led` is what makes the loan a
            // PROCESS resource instead of a per-request one: after a lazy return the layout, the ledger and
            // the non-resident rows all survive into the next request, so that request's `lend()` marks
            // almost nothing and refills nothing at all.
            strata::program::prefill_loan::LoanLedger led;
            /// S3.2b: is a batched prompt segment's loan standing in THIS cache right now?  Set by `lend()`
            /// for every participant that can lend - whether or not it marked a single row, which under the
            /// lazy rule it often does not, because the rows were already out from the previous request.
            /// That distinction is load-bearing: the pump must never copy an expert into a slot the prompt
            /// buffers are standing in, and "did I mark anything" is the wrong question to ask.
            bool loan_live = false;
            // The pump's own stream and completion event (S3.2b B).  One per cache, because a refill must
            // land on the device that owns the layer, and because the confirmation is one event per cache:
            // at most one batch in flight per cache at a time, so `loan_ev` can only ever confirm the copies
            // that were queued since it was last recorded.  Created lazily - a box that never lends never
            // allocates one, and R1's VRAM headroom is not for this.
            cudaStream_t loan_stream = nullptr;
            cudaEvent_t loan_ev = nullptr;
            bool loan_pump_live = false;   ///< a batch is queued on `loan_stream`, not yet confirmed
        };
        auto part_slots = [&](const PfPart& p, int64_t c) -> int64_t {
            const uint64_t need = strata::prefill::Prefill::bytes_needed(g, *p.ses, c);
            strata::core::ExpertCache& xc = *p.cache;
            if (xc.slot_offsets() != nullptr) {   // sized slots: from the end until they hold `need`
                int64_t k = 0;
                while (k < xc.slots() &&
                       (uint64_t) (xc.bytes() - (int64_t) xc.slot_offsets()[xc.slots() - k]) < need) ++k;
                return k;
            }
            const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            return (int64_t) ((need + (uint64_t) blob - 1) / (uint64_t) blob);
        };
        auto part_bytes = [&](const PfPart& p, int32_t first) -> uint64_t {
            strata::core::ExpertCache& xc = *p.cache;
            return xc.slot_offsets() ? (uint64_t) (xc.bytes() - (int64_t) xc.slot_offsets()[first])
                                     : (uint64_t) (xc.slots() - first) *
                                           (uint64_t) strata::kernels::cpu::expert_layout().max_blob;
        };
        std::vector<PfPart> pf_parts;
        // a cache too small to lend the prompt path its buffers would make it allocate them on top - on a card
        // whose cache already filled its reserve, that is the over-subscription the auto sizing avoids - so the
        // chunk is the largest one EVERY participant can lend (a smaller chunk only reads slower)
        if (pf_borrow && d_res != nullptr) {
            // Designated initializers, so the S3.2b members (`led`, `loan_live`, the pump's stream/event)
            // keep their default member initialisers instead of being zero-filled by a short aggregate list.
            pf_parts.push_back(PfPart{.cache = &xcache, .ses = &ss, .sp = &sp, .dev = -1, .lb = 0,
                                      .le = multi_gpu ? split_at[0] : g.n_layers, .lent = {}, .led = {}});
            for (auto& st : stages)
                pf_parts.push_back(PfPart{.cache = &st->cache, .ses = &st->ss, .sp = &st->sp, .dev = st->dev,
                                          .lb = st->lb, .le = st->le, .lent = {}, .led = {}});
            // The two tests plan_lend makes for CUDA0 alone, one participant at a time: a loan must leave the
            // 128-slot floor.  The percentage cap is an AUTO-chunk rule and only the auto scan applies it - an
            // explicit --prefill is the operator's number, and a loan of it only has to fit.  With one participant
            // (no split) this reduces to plan_lend exactly, so the single-GPU loan is unchanged from main.
            auto fits = [&](int64_t c, bool cap) -> bool {
                for (const PfPart& p : pf_parts) {
                    const int64_t k = part_slots(p, c);
                    if (k <= 0 || k + 128 > p.cache->slots()) return false;
                    if (cap && k * 100 > kAutoLendPct * p.cache->slots()) return false;
                }
                return true;
            };
            static constexpr int64_t kAutoChunks[] = {8192, 6144, 4096, 3072, 2048, 1024, 512, 256};
            int64_t chunk = 0;
            if (o.prefill_auto) {
                for (const int64_t c : kAutoChunks)
                    if (fits(c, true)) { chunk = c; break; }
            } else {
                for (int64_t c = o.prefill_chunk; c >= 256; c /= 2)
                    if (fits(c, false)) { chunk = c; break; }
            }
            if (chunk > 0) {
                if (o.prefill_auto)
                    std::fprintf(stderr, "strata serve: prompt chunk auto: %lld tokens\n", (long long) chunk);
                else if (chunk != o.prefill_chunk)
                    std::fprintf(stderr, "strata serve: prompt chunk %lld -> %lld tokens so its buffers fit in "
                                         "every expert cache\n", (long long) o.prefill_chunk, (long long) chunk);
                o.prefill_chunk = chunk;
                for (PfPart& p : pf_parts) {
                    p.first = (int32_t) (p.cache->slots() - part_slots(p, chunk));
                    p.first_now = p.first;
                }
                lend_first = pf_parts[0].first;
                borrow = xcache.device_slot(lend_first);
                borrow_bytes = part_bytes(pf_parts[0], lend_first);
            } else if (o.prefill_auto) {
                o.prefill_chunk = 1024;   // nothing lendable: small buffers of its own
            } else if (pf_parts.size() > 1) {
                // An explicit chunk no stage can lend in full.  main falls back to the prompt path's own buffers
                // here and so do we, rather than refusing to start - but say what every stage has, because a split
                // stage that has to allocate these on top of a cache that already filled its VRAM will not fit,
                // and `init` would otherwise report only that the buffers do not fit.
                std::fprintf(stderr, "strata serve: no stage can lend the prompt path its %lld-token buffers, so "
                                     "each stage allocates its own:\n", (long long) o.prefill_chunk);
                for (const PfPart& p : pf_parts)
                    std::fprintf(stderr, "strata serve:   CUDA%d has %lld slots, and a %lld-token chunk needs "
                                         "the last %lld of them\n", p.dev < 0 ? 0 : p.dev,
                                 (long long) p.cache->slots(), (long long) o.prefill_chunk,
                                 (long long) part_slots(p, o.prefill_chunk));
            }
        } else if (o.prefill_auto && d_res == nullptr) {
            o.prefill_chunk = 1024;       // #85: no expert cache at all (a full 8 GB card): small buffers of its own
        }
        if (borrow != nullptr) {
            std::fprintf(stderr, "strata serve: the prompt path borrows %lld CUDA0 cache slots (%.2f GiB)\n",
                         (long long) (xcache.slots() - lend_first), (double) borrow_bytes / 1073741824.0);
            for (size_t i = 1; i < pf_parts.size(); ++i)   // one loan per stage, from that stage's own cache
                std::fprintf(stderr, "strata serve:   CUDA%d prompt path borrows %lld of its %lld slots (%.2f GiB)\n",
                             pf_parts[i].dev, (long long) (pf_parts[i].cache->slots() - pf_parts[i].first),
                             (long long) pf_parts[i].cache->slots(),
                             (double) part_bytes(pf_parts[i], pf_parts[i].first) / 1073741824.0);
        } else {
            std::fprintf(stderr, "strata serve: the prompt path allocates its own buffers (too few cache slots to borrow)\n");
        }
        // layer split across GPUs: a prompt path per stage, each handing its chunk's rows to the next
        for (size_t i = 0; i < stages.size(); ++i) {
            GpuStage& st = *stages[i];
            st.sp.set_stage(st.lb, i + 1 < stages.size() ? st.le : -1, i + 1 < stages.size() ? &stages[i + 1]->sp : nullptr);
            const strata::core::OnDevice on(st.dev);
            void* sb = nullptr;              // this stage's own loan, out of its own cache
            uint64_t sbb = 0;
            // `first < 0`: no loan was taken (nothing was lendable), so this stage allocates its own buffers
            if (i + 1 < pf_parts.size() && pf_parts[i + 1].first >= 0) {
                sb = st.cache.device_slot(pf_parts[i + 1].first);
                sbb = part_bytes(pf_parts[i + 1], pf_parts[i + 1].first);
            }
            if (!st.sp.init(st.wt, g, st.ss, srcp, &st.cache, host_res.data(), o.prefill_chunk, (void*) st.stream, err,
                            sb, sbb)) {
                std::fprintf(stderr, "strata serve: layer split, CUDA%d prompt path: %s\n", st.dev, err.c_str());
                return 1;
            }
        }
        if (multi_gpu) sp.set_stage(0, split_at[0], &stages[0]->sp);
        if (!sp.init(wt, g, ss, srcp, &xcache, host_res.data(), o.prefill_chunk, main_cs, err, borrow, borrow_bytes)) {
            std::fprintf(stderr, "strata serve: %s\n", err.c_str());
            if (err.find("fit") != std::string::npos)   // #85: say what frees VRAM
                std::fprintf(stderr, "strata serve: the GPU has too little free VRAM for the prompt path: turn images "
                                     "off (setup: --vision no), close other programs using the GPU, use a shorter "
                                     "context, or read prompts in smaller chunks (--prefill 512)\n");
            return 1;
        }
        mem_mark("the head and the prompt path");
        // the penalty-history buffer: one row per verify-window row (`penalty_rows`), each the last
        // `penalty_last_n` tokens that row's pick follows, -1 padded in front.  Allocated once at the cap for
        // the widest window; a request without penalties gets a null buffer and takes the byte-for-byte
        // neutral path (no upload, no buffer handed to the sampler).
        constexpr int kPenaltyWindowCap = 4096;
        constexpr size_t kHistSlots = (size_t) kPenaltyWindowCap * (size_t) strata::kernels::kVerifyMaxT;
        int32_t* d_hist = nullptr;
        std::vector<int32_t> hist_stage(kHistSlots, -1);
        const int hist_dev = last_st ? last_st->dev : -1;   // with the head: the last stage's device
        if (const strata::core::OnDevice on_h(hist_dev); cudaMalloc(&d_hist, kHistSlots * sizeof(int32_t)) != cudaSuccess) {
            std::fprintf(stderr, "strata serve: the penalty-history allocation failed\n");
            return 1;
        }
        strata::core::Verifier ver;
        strata::core::VerifyHits vh;
        vh.d_res = thits.d_res;
        vh.cache_base = thits.cache_base;
        vh.blob = thits.blob;
        vh.slot_off = xcache.slot_offsets();   // E-6: the device plan's pointers
        vh.n_slots = xcache.slots();
        // Layer split: `ver` runs layers [0, K1) and hands its residual to the next stage's verifier, and so on; the
        // last runs the head.  The hand-offs are mapped pinned memory, portable: a stage on another GPU reads it.
        // (--split-device 0: the second stage on this GPU, sharing its weights, session and cache - the A/B.)
        strata::core::Verifier ver_same;
        SplitDrive split_drive;
        auto stage_ver = [&](int st) -> strata::core::Verifier& {
            return st == 0 ? ver : split_same ? ver_same : stages[(size_t) st - 1]->ver;
        };
        auto pcie_num_of = [](double f) { return std::max(0, std::min(256, (int) (f * 256.0 + 0.5))); };
        const int n_stages = split_devs.empty() ? 1 : (int) split_at.size() + 1;
        if (n_stages > 1) {
            const size_t hb = (size_t) strata::kernels::kVerifyMaxT *
                              (size_t) strata::core::Verifier::handoff_floats(g) * sizeof(float);
            std::vector<float*> hand((size_t) n_stages - 1, nullptr);
            for (float*& h : hand) {
                float* hh = nullptr;
                if (cudaHostAlloc((void**) &hh, hb, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess ||
                    cudaHostGetDevicePointer((void**) &h, hh, 0) != cudaSuccess) {
                    std::fprintf(stderr, "strata serve: the layer-split hand-off allocation failed\n");
                    return 1;
                }
                std::memset(hh, 0, hb);
            }
            split_drive.base = &drive;
            split_drive.n = n_stages;
            for (int st = 0; st < n_stages; ++st) {
                stage_ver(st).set_stage(st == 0 ? 0 : split_at[(size_t) st - 1], st + 1 < n_stages ? split_at[(size_t) st] : -1,
                                        st == 0 ? nullptr : hand[(size_t) st - 1], st + 1 < n_stages ? hand[(size_t) st] : nullptr);
                split_drive.end[st] = st + 1 < n_stages ? split_at[(size_t) st] : g.n_layers;
                split_drive.cache_base[st] = drive.d.cache_base;
                split_drive.cache_slot_off[st] = drive.d.cache_slot_off;
                split_drive.pcie_num[st] = pcie_num_of(o.pcie_frac);
            }
            for (int st = 1; st < n_stages; ++st) {
                bool ok_s = false;
                if (split_same) {
                    ok_s = ver_same.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err);
                } else {
                    GpuStage& gs = *stages[(size_t) st - 1];
                    const strata::core::OnDevice on(gs.dev);
                    strata::core::VerifyHits vs;
                    vs.d_res = gs.d_res;
                    vs.cache_base = gs.cache.device_slot(0);
                    vs.blob = thits.blob;
                    vs.slot_off = gs.cache.slot_offsets();
                    vs.n_slots = gs.cache.slots();
                    ok_s = gs.ver.init(gs.wt, g, gs.ss, vs, gs.head.loaded() ? &gs.head : nullptr, o.spec, err);
                    split_drive.cache_base[st] = gs.cache.device_slot(0);
                    split_drive.cache_slot_off[st] = gs.cache.slot_offsets();
                    split_drive.pcie_num[st] = pcie_num_of(gs.pcie_frac);
                }
                if (!ok_s) {
                    std::fprintf(stderr, "strata serve: layer split, stage %d: %s\n", st + 1, err.c_str());
                    return 1;
                }
            }
            for (int st = 0; st + 1 < n_stages; ++st) stage_ver(st).set_next(&stage_ver(st + 1), &split_drive);
            std::string plan_s = "0-" + std::to_string(split_at[0] - 1) + " (CUDA0)";
            for (int st = 1; st < n_stages; ++st)
                plan_s += ", " + std::to_string(split_at[(size_t) st - 1]) + "-" + std::to_string(split_drive.end[st] - 1) +
                          " (CUDA" + std::to_string(split_same ? 0 : stages[(size_t) st - 1]->dev) + ")";
            std::fprintf(stderr, "strata serve: layer split: layers %s, one hand-off per window\n", plan_s.c_str());
        }
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err) ||
            !mtp.bind(last_st ? last_st->wt : wt, last_st ? &last_st->head : &native_head, ver.final_R_all(), err)) {
            std::fprintf(stderr, "strata serve: %s\n", err.c_str());
            return 1;
        }
        for (int st = 0; st < n_stages && n_stages > 1; ++st) {
            split_drive.plan[st] = stage_ver(st).plan_sink();
            if (st > 0) {
                stage_ver(st).set_split(o.spec_split);
                stage_ver(st).set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
            }
        }
        // the pool the verify windows call: with a layer split, the wrapper that routes each layer to its stage
        const strata::core::PoolMultiFn win_pool_fn = n_stages > 1 ? &drive_pool_split : &drive_pool_multi;
        void* const win_pool_user = n_stages > 1 ? (void*) &split_drive : (void*) &drive;
        mem_mark("the verifier and the drafter's binding");
        ver.set_split(o.spec_split);
        // auto: the copy kernel for every pack.  DMA (the native packs' default until 0.1.13) has the host call
        // cudaMemcpyAsync + cudaLaunchHostFunc inside a verify window while the GPU spins on the flag they raise;
        // issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy
        // kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
        ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
        std::vector<int64_t> cur;
        // ---- the conversation cache (see ConvCheckpoint).  `live` is what the session holds right now: the tokens
        // it has consumed, so a request that starts with exactly them continues without any copy.  `checks` are the
        // saved points; every one of them is a prefix of `live` (the loop drops the rest), so they form a chain -
        // the radix cache's tree collapsed onto the one branch of history whose cells the session holds.  The
        // chain's root is the deepest point every request so far shared (the end of the system prompt, in
        // practice); the retention policy pins it and rotates the rest LRU (conv_cache.hpp), so a NEW chat that
        // shares that prefix mounts through it instead of reading it again.
        std::vector<int32_t> live;
        std::vector<ImgKey> live_imgs, req_imgs;
        bool live_ok = false;
        std::vector<ConvCheckpoint> checks;
        uint64_t check_clock = 0;   // the checkpoints' LRU clock; creation and every use advance it
        bool cvec_cached = true;   // the control vector's state the live session and the checkpoints were read with
        // THE BUDGET IS A CAP, AND THE DEFAULT HAS TO FIT THE MACHINE IT RUNS ON.
        //
        // `--conversation-cache-slots 8` is the owner's ask: eight prefixes at a time. The RAM that takes is a
        // different question, and it is the same trap the expert arena already documents (review finding C1,
        // core/expert_source.hpp): a 47 GiB arena plus a 26.8 GB mapped PLE shard is most of a 64 GB machine,
        // and host pages that the engine allocates are paid for by reclaiming the file cache underneath them.
        // Parking 8 GiB of snapshots on a PC that has 3 GiB to spare does not fail - the admission check below
        // allows it, because `MemAvailable` counts reclaimable cache - it quietly drops the pages the prompt
        // path reads and the whole server gets slower than the feature it just gained.
        //
        // So the DEFAULT budget is what this machine can actually hand over right now: the cap, or the free RAM
        // minus the parking floor and a headroom for the next request's own allocations, whichever is smaller.
        // An explicit `--conversation-cache-mib` is never second-guessed - the user set it, the engine obeys it
        // and says what it is.
        size_t park_budget = o.prompt_cache > 0 ? (size_t) o.conversation_cache_mib * 1024 * 1024 : 0;
        if (park_budget > 0 && !o.conversation_cache_mib_given) {
            const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024;
            const uint64_t headroom = 4ull << 30;   // the next request's own allocations, not the cache's
            if (const auto avail = strata::core::conversation_available_memory()) {
                const uint64_t room = *avail > floor + headroom ? *avail - floor - headroom : 0;
                if (room < park_budget) {
                    std::fprintf(stderr, "strata serve: parked-prefix budget %.1f GiB, not %.1f: only %.1f GiB of "
                                         "RAM is free above the %.1f GiB parking floor and %.1f GiB of request "
                                         "headroom (%d slots, pruned least recently used first; "
                                         "--conversation-cache-mib N to override, 0 to park nothing)\n",
                                 (double) room / (1024.0 * 1024.0 * 1024.0),
                                 (double) park_budget / (1024.0 * 1024.0 * 1024.0),
                                 (double) room / (1024.0 * 1024.0 * 1024.0),
                                 (double) floor / (1024.0 * 1024.0 * 1024.0),
                                 (double) headroom / (1024.0 * 1024.0 * 1024.0),
                                 o.conversation_cache_slots);
                    park_budget = (size_t) room;
                }
            }
        }
        strata::core::ConversationCache conversations(park_budget, (size_t) o.conversation_cache_slots);
        // S3.1a: THE STAGE SET A PARKED CONVERSATION COVERS.  A `--layer-split` runs the model across
        // several devices and each stage's `SessionState` owns only its layer carve, so a parked
        // conversation is CUDA0's session (the `ss` argument) PLUS one part per later stage PLUS the
        // MTP draft state, which is bound to the LAST stage's device.  Empty when there is no split,
        // and then every snapshot call is exactly the single-GPU one it always was.
        // Built once: the sessions do not move, and the vector must outlive every snapshot call.
        std::vector<strata::core::ConversationStage> stage_list;
        stage_list.reserve(stages.size());
        for (auto& stp : stages) stage_list.push_back({&stp->ss, stp->dev});
        strata::core::ConversationStageSet conv_stages;
        conv_stages.stages = stage_list.empty() ? nullptr : stage_list.data();
        conv_stages.count = (int64_t) stage_list.size();
        // No split: -1 everywhere, so the snapshot core never asks CUDA to change device and the
        // single-GPU path is exactly what it was.  With a split, CUDA0 owns the main session and
        // the drafter reports the device it bound its own state to (the last stage's).
        conv_stages.main_device = stage_list.empty() ? -1 : 0;
        conv_stages.draft_device = stage_list.empty() ? -1 : mtp.device();
        // STRATA_TRACE=1: one stderr line per step of a request (the log shows where a request stops).
        // S3.1d: declared here rather than in the request body because the slot hand-over traces too.
        const bool trace = std::getenv("STRATA_TRACE") != nullptr;
        auto tr = [&](const char* what, long long a = -1, long long b = -1) {
            if (!trace) return;
            std::fprintf(stderr, "strata trace: %s %lld %lld\n", what, a, b);
            std::fflush(stderr);
        };
        // S0.3 lever 4: PARKING REFUSAL BACKOFF.  `park_current` runs on every request that does not
        // continue from the live session - which in a chat is nearly all of them, because they resume from
        // a checkpoint at the turn boundary - and before it can answer "no room" it validates the whole
        // view, walks every retained checkpoint, estimates the snapshot and reads the RAM telemetry.  On a
        // box whose parked-prefix budget has already collapsed (the log shows 1.6 GiB, then 0.0 GiB) that
        // answer is the same every time and is paid every time: 247 of them in this log, replayable with
        // `bench/prefill/park_backoff_replay.py`.
        //
        // So after STRATA_PARK_REFUSALS consecutive physical-RAM-admission refusals (default 3, 0 = never
        // back off), parking stops asking for STRATA_PARK_QUIET requests (default 64).  It is quiet, not
        // off: the window expires by itself, and any successful park clears it, because that is proof the
        // machine can take snapshots again.  A request that STARTS a conversation or MOUNTS a parked one is
        // always allowed to park regardless of the window - those are the moments the cache is load-bearing,
        // and they are the "new conversation, a switch" triggers parking has always had.  They do not clear
        // the streak: on this box 1 request in 13 starts a new conversation, so clearing on them would make
        // the backoff never engage at all.
        const strata::program::prefill_loan::ParkBackoffPolicy park_policy = [] {
            strata::program::prefill_loan::ParkBackoffPolicy p;
            if (const char* v = std::getenv("STRATA_PARK_REFUSALS")) p.refusals = std::max<int64_t>(0, (int64_t) std::atoi(v));
            if (const char* v = std::getenv("STRATA_PARK_QUIET")) p.quiet_requests = std::max<int64_t>(0, (int64_t) std::atoi(v));
            return p;
        }();
        strata::program::prefill_loan::ParkBackoff park_backoff(park_policy);
        int64_t request_index = 0;   // the backoff's window is counted in requests, not in seconds
        // S3.1d: THE POSITION TABLE, UPLOADED.  `d_mrope` is ONE table for the process (risk R7), so
        // handing the session to another slot means putting THAT slot's positions back in it.  Hoisted
        // out of the request body because the hand-over needs it too; the request body's own
        // `upload_mrope` is now a call to this, byte for byte the same copies.
        auto upload_mrope_table = [&]() -> bool {
            bool ok = cudaMemcpy(d_mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t),
                                 cudaMemcpyHostToDevice) == cudaSuccess;
            for (auto& st : stages) {
                const strata::core::OnDevice on(st->dev);
                cudaDeviceSynchronize();
                ok = ok && cudaMemcpy(st->mrope, mrope_host.data(), mrope_host.size() * sizeof(int32_t),
                                      cudaMemcpyHostToDevice) == cudaSuccess;
            }
            return ok;
        };
        // Save only on a switch/rewind, not on each continuing request. No graph
        // addresses change: all parked images live in ordinary host vectors.
        // S3.1d: the return type is `Saved`, not `bool`.  0.1.30 collapsed "parked", "refused" and
        // "nothing to do" into true and only the snapshot failure into false, and for a request that is
        // the right answer - it just re-reads.  A HAND-OVER cannot collapse them: a branch that was not
        // parked is a branch whose cells are about to belong to somebody else, so it may never be
        // mounted again.  `park_bytes` is what the last call stored (0 = nothing).
        size_t park_bytes = 0;
        // S3.7: WHAT A SNAPSHOT ACTUALLY COSTS ON THIS MODEL, learned from the parks that happened.
        // Admission used to price every slot at `budget / --conversation-cache-slots` - a flat guess
        // that charges a 150-token chat the same gigabyte it charges a 200k-token one, and then refuses
        // the short one because that gigabyte was not free.  Parking has never worked that way:
        // `conversation_snapshot_bytes` measures a branch against `live.size()`, the tokens it really
        // holds, and `ConversationCache::bytes_` counts what was stored.  So the engine keeps the rate
        // from the largest snapshot it has parked and prices each request at its own length.
        // Engine-thread only: `park_current` writes it, the driver's admission and startup lines read
        // it, and both run on the one thread that owns the session.
        uint64_t park_per_token = 0;
        uint64_t park_per_token_tokens = 0;   // the park that set the rate, for the log line
        auto park_current = [&](size_t held, bool force = false,
                                int64_t owner = strata::core::kNoOwner) -> strata::program::serve_swap::Saved {
            park_bytes = 0;
            if (!conversations.enabled() || !live_ok || live.empty()) return strata::program::serve_swap::Saved::skipped;
            // Quiet, not off.  The estimate below is a whole-view validation, a walk of every retained
            // checkpoint and a /proc read, and while a backoff is running its answer is already known.
            // `force` is a request that starts a conversation or mounts a parked one: parking is exactly
            // what saves that request's re-read, so it asks regardless of the window.
            if (!force && park_backoff.quiet(request_index)) {
                if (park_backoff.went_quiet()) {
                    park_backoff.clear_went_quiet();
                    std::fprintf(stderr, "strata serve: conversation cache: %lld consecutive RAM-admission refusals"
                                         " - parking will not re-ask for %lld requests (STRATA_PARK_REFUSALS N to"
                                         " change, 0 = always ask)\n",
                                 (long long) park_backoff.consecutive(),
                                 (long long) park_backoff.policy().quiet_requests);
                }
                return strata::program::serve_swap::Saved::skipped;
            }
            const strata::core::ConversationView view{live, live_imgs, checks, cvec_cached};
            auto reuse = conversations.take_reuse();
            size_t estimate = 0;
            if (!strata::core::conversation_snapshot_bytes(view, ss, conv_stages, g, mtp.kv_state(), estimate, err)) {
                std::fprintf(stderr, "strata serve: conversation cache: skip parking (%s)\n", err.c_str());
                err.clear(); // A recoverable miss must not poison the batched draft prefill's error channel.
                return strata::program::serve_swap::Saved::skipped;
            }
            const size_t fresh_estimate = estimate;
            if (!reuse.kv.empty() && !strata::core::conversation_snapshot_capture_bytes(
                    reuse, view, ss, conv_stages, g, mtp.kv_state(), estimate, err)) {
                reuse = {};
                estimate = fresh_estimate;
                err.clear();
            }
            if (!reuse.kv.empty() && !conversations.can_fit(estimate, held)) {
                // Optional growth capacity must not evict useful conversations.
                reuse = {};
                estimate = fresh_estimate;
            }
            if (!conversations.make_room(estimate, held)) {
                std::fprintf(stderr, "strata serve: conversation cache: skip parking (snapshot %zu MiB exceeds available budget)\n",
                             estimate >> 20);
                return strata::program::serve_swap::Saved::skipped;
            }
            const auto t0 = Clock::now();
            bool parked_ok = false;   // the one answer that makes a later hand-over mountable again
            try {
                const uint64_t floor = (uint64_t) o.conversation_cache_min_free_mib * 1024 * 1024;
                const size_t additional = estimate - reuse.bytes();
                if (!strata::core::conversation_memory_admit(strata::core::conversation_available_memory(),
                        additional, floor)) {
                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM admission; need %zu MiB plus %lld MiB floor, or telemetry unavailable)\n",
                                 additional >> 20, (long long) o.conversation_cache_min_free_mib);
                    park_backoff.refused(request_index);   // S0.3 lever 4: the refusal this backoff exists for
                    return strata::program::serve_swap::Saved::skipped;
                }
                strata::core::SavedConversation image;
                size_t reused_bytes = 0;
                if (!strata::core::conversation_snapshot_save(image, view, ss, conv_stages, g, mtp.kv_state(), err,
                        std::move(reuse), &reused_bytes)) return strata::program::serve_swap::Saved::failed;
                if (!strata::core::conversation_memory_admit(strata::core::conversation_available_memory(), 0, floor)) {
                    std::fprintf(stderr, "strata serve: conversation cache: skip parking (physical RAM floor after capture, or telemetry unavailable)\n");
                    park_backoff.refused(request_index);   // the same floor, checked again after the capture
                    return strata::program::serve_swap::Saved::skipped;
                }
                const size_t snapshot_bytes = image.bytes();
                // S3.9: claim it.  `owner` is the slot whose branch this IS - the one the session
                // reflects at the moment of the park.  Without the claim, the next request whose
                // prompt happens to share a prefix can `take()` this whole entry and destroy a
                // conversation that is still running.
                const bool stored = conversations.put(std::move(image), owner, held);
                parked_ok = stored;
                if (stored) park_bytes = snapshot_bytes;   // 0 unless it really went into the cache
                if (stored) park_backoff.parked();   // the machine can take snapshots: ask every time again
                if (stored && !live.empty()) {
                    // S3.7: learn the cost per context token from the BIGGEST snapshot parked so far.
                    // The largest one is the right sample: a small park is dominated by the fixed
                    // per-layer state (recurrence, indexer tails, the draft ring), so a rate learned
                    // from an 81-token conversation would under-price a 100k one.
                    const uint64_t rate = strata::program::serve_driver::bytes_per_token(
                        snapshot_bytes, (uint64_t) live.size());
                    if (rate > park_per_token) {
                        park_per_token = rate;
                        park_per_token_tokens = (uint64_t) live.size();
                    }
                }
                std::fprintf(stderr, "strata serve: conversation cache: %s %zu tokens in %.1f ms; parked=%zu bytes=%zu evictions=%zu snapshot_bytes=%zu reused_kv_bytes=%zu\n",
                             stored ? "parked" : "skipped", live.size(),
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                             conversations.size(), conversations.bytes(), conversations.evictions(), snapshot_bytes, reused_bytes);
            } catch (const std::bad_alloc&) {
                // The active state has not been touched. Continue with normal
                // prompt processing rather than killing a serving process.
                std::fprintf(stderr, "strata serve: conversation cache: allocation failed; skip parking\n");
            }
            return parked_ok ? strata::program::serve_swap::Saved::stored
                             : strata::program::serve_swap::Saved::skipped;
        };

        // ---- S3.1b/S3.1c: the slot registry and the wire formatter -------------------------------------
        // Declared before the prompt path's callbacks because those capture them by reference.
        //
        // `slots` = --serve-slots.  0/1 keeps EVERYTHING below on 0.1.30's path: `sp_out` is untagged, no
        // SLOT line is ever emitted, the parser never reads a request id, and `stop_req` is the only stop
        // flag that moves.  >= 2 turns the stage-3 wire on: per-request lines carry `#<id>`, STOP can name a
        // request, and each request runs through a registry row so /slots has something real to report.
        // The scheduler that interleaves two conversations is S3.1e - until it lands, the request body still
        // runs to completion, so what lands here is the WIRE and the bookkeeping, not the interleaving.
        const int slots = o.serve_slots;
        const bool tagged = slots >= 2;
        strata::program::serve_proto::Out sp_out(tagged);
        strata::program::slot::Registry slots_reg(slots, o.starve_ms);
        std::mutex slot_mu;                       // guards slots_reg: the stdin thread cancels, the engine runs
        std::atomic<int64_t> running_id{strata::program::serve_proto::kNoId};  // the request the body is running
        // A `STOP <id>` for a request the engine has not read yet (it is still on the pipe, or in in_lines):
        // the stdin thread cannot name a row that does not exist, so it parks the id here and the request loop
        // picks it up when the row is created.  Bounded by the registry, oldest dropped first.
        std::deque<int64_t> pending_cancel;
        // This request's id (kNoId with --serve-slots 0/1, where the wire is 0.1.30's byte for byte).  Every
        // per-request line goes through `sp_out`, so a tagged line and an untagged one are the same call with
        // one flag and cannot drift.
        int64_t req_id = strata::program::serve_proto::kNoId;
        // The request-body ERR sites inside the snapshot / lend / refill code belong to the other stage-3
        // task (S3.1d/e own those call sites); the ones routed here are the request loop's own.  The list is
        // in .megamind/src/program/slot-notes.md so none is missed.
        auto sp_err = [&](const std::string& msg) -> std::string { return sp_out.err(msg, req_id); };
        // The SLOT transition line (§6.2), emitted only on the stage-3 wire.  Takes the registry lock, so the
        // caller must not already hold it.
        auto emit_slot = [&](int64_t id) {
            if (!tagged) return;
            std::string out;
            {
                std::lock_guard<std::mutex> lk(slot_mu);
                const strata::program::slot::Slot* s = slots_reg.find(id);
                if (s == nullptr) return;
                out = sp_out.slot(s->id, s->state_name(), s->ctx_used, s->ctx_cap,
                                  s->prompt_tokens, s->generated, s->parked_bytes);
            }
            std::printf("%s\n", out.c_str());
            std::fflush(stdout);
        };
        // Move a slot along the state machine and say so on the wire.  A refused edge goes to stderr, never to
        // the client: it is a bookkeeping bug, not the request's error, and the request must still get its DONE.
        auto slot_step = [&](int64_t id, strata::program::slot::State to) {
            if (!tagged) return;
            std::string serr;
            {
                std::lock_guard<std::mutex> lk(slot_mu);
                if (!slots_reg.transition(id, to, serr))
                    std::fprintf(stderr, "strata serve: slot %lld -> %s: %s\n", (long long) id,
                                 strata::program::slot::state_name(to), serr.c_str());
            }
            emit_slot(id);
        };
        // A request is over: its row goes to `idle` (one last SLOT line, so /slots can show a finished
        // conversation), then the row is freed and the session is no longer claimed.
        auto slot_finish = [&](int64_t id) {
            if (!tagged) return;
            {
                std::lock_guard<std::mutex> lk(slot_mu);
                std::string serr;
                slots_reg.transition(id, strata::program::slot::State::idle, serr);
                slots_reg.set_active(strata::program::serve_proto::kNoId);
                running_id.store(strata::program::serve_proto::kNoId);
            }
            emit_slot(id);
            std::lock_guard<std::mutex> lk(slot_mu);
            slots_reg.release(id);
        };
        // The running slot's cancel flag is `stopped()` below - it has to be declared after `stop_req`, which
        // the stdin thread owns.

        // ---- S3.1e-2: the per-slot watchdog state (risk R3) --------------------------------------
        // 0.1.30's watchdog aborts the process when the ONE request stops beating.  With N slots that
        // would kill N-1 healthy conversations because one stalled request owns the GPU, so the driver
        // publishes one entry per ACTIVE slot here and the watchdog thread decides between the two
        // cases: every active slot stalled = the engine itself is wedged (abort, as today, and name
        // the slots); only some of them stalled = the engine thread is alive and simply not serving
        // those slots, so they get `ERR <id>` and their slot is destroyed instead.  The engine thread
        // owns every write; the watchdog thread only reads and appends to `watch_kill`, which the
        // engine thread drains between steps (it cannot print to stdout itself - the engine thread
        // owns the wire, §3.1).
        // Static storage on purpose: the watchdog thread is detached, and a capture of a local would
        // dangle the moment the serve block returns.  There is exactly one serve block per process.
        static std::mutex watch_mu;
        static std::vector<strata::program::serve_driver::SlotWatch> watch;
        static std::deque<int64_t> watch_kill;  // slots the watchdog gave up on; the driver ERRs them
        static int watchdog_limit_s = 60;       // STRATA_WATCHDOG_S; set below, read by the driver too

        int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;
        Clock::time_point pp_t0 = Clock::now();
        auto imgs_below = [&](const std::vector<ImgKey>& all, int64_t L) {
            std::vector<ImgKey> v;
            for (const ImgKey& k : all) if (k.start < L) v.push_back(k);
            return v;
        };
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed
        // A layer split's mid-prompt checkpoints: when the last stage reports a chunk, the earlier ones already read
        // the next, so each stage saves its own part when IT reaches a checkpoint position (the same rule as below:
        // every `prompt_cache_every` tokens from where the request resumed), and the last stage puts them together.
        std::mutex part_mu;
        std::map<int64_t, std::vector<ConvCheckpoint>> part_at;   // position -> one part per stage
        std::vector<int64_t> part_next(stages.size() + 1, INT64_MAX);
        // a checkpoint of the state after `cur[0, L)`; false only when the copy itself failed.  `parts`: the stages'
        // states saved at L (a split's mid-prompt checkpoint); without, they are read now (everything is at L)
        auto checkpoint_at = [&](int64_t L, std::vector<ConvCheckpoint>* parts = nullptr) -> bool {
            if (o.prompt_cache <= 0 || L < 1) return true;
            for (ConvCheckpoint& c : checks)
                if ((int64_t) c.ids.size() == L) { c.used = ++check_clock; return true; }
            ConvCheckpoint c;
            c.ids.assign(cur.begin(), cur.begin() + L);
            c.imgs = imgs_below(req_imgs, L);
            if (parts != nullptr) {
                if (parts->size() != stages.size() + 1) return false;
                c.gdn = std::move((*parts)[0].gdn);
                c.ple = std::move((*parts)[0].ple);
                c.tails = std::move((*parts)[0].tails);
                c.dead = std::move((*parts)[0].dead);
                c.block_pos = std::move((*parts)[0].block_pos);
                for (size_t i = 1; i < parts->size(); ++i) c.stage_parts.push_back(std::move((*parts)[i]));
            } else {
                if (cudaDeviceSynchronize() != cudaSuccess || !checkpoint_save(c, ss, g)) return false;
                for (auto& st : stages) {   // a layer split's later stages: their sessions' part
                    const strata::core::OnDevice on(st->dev);
                    ConvCheckpoint part;
                    part.ids = c.ids;
                    if (cudaDeviceSynchronize() != cudaSuccess || !checkpoint_save(part, st->ss, g)) return false;
                    c.stage_parts.push_back(std::move(part));
                }
            }
            c.used = ++check_clock;
            checks.push_back(std::move(c));
            while ((int) checks.size() > o.prompt_cache) {
                std::vector<uint64_t> stamps;
                stamps.reserve(checks.size());
                for (const ConvCheckpoint& k : checks) stamps.push_back(k.used);
                const size_t victim = strata::program::conv_cache::eviction_victim(stamps.data(), stamps.size(),
                                                                                   o.prompt_cache);
                checks.erase(checks.begin() + (std::ptrdiff_t) victim);
            }
            return true;
        };
        sp.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
            std::vector<int32_t> nxt((size_t) T);
            for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) cur[(size_t) (p0 + t + 1)];
            // E-9: batched through the prompt path when it can (one GPU: a layer split's drafter is on the last stage)
            const bool batched = !multi_gpu && sp.draft_kv(mtp, R_rows, nxt.data(), T, p0, e);
            if (!e.empty() || (!batched && !mtp.prefill(R_rows, nxt.data(), T, p0, e))) return false;
            if (std::getenv("STRATA_SNAPSHOT_VERIFY") != nullptr)
                std::fprintf(stderr, "strata serve: DRAFT_PREFILL path=%s mode=%d cells=%lld\n",
                             batched ? "batched" : "token", mtp.kv_state().kv_mode, (long long) T);
            // progress for the server window: PP <position reached> <prompt tokens> <ms> <fresh tokens/s>
            const int64_t done = p0 + T;
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
            std::printf("%s\n", sp_out.pp(done, pp_total, ms,
                                        ms > 0.0 ? 1000.0 * (double) (done - pp_from) / ms : 0.0,
                                        req_id).c_str());
            strata::core::progress_at("reading the prompt (batched), done up to token", done);
            strata::core::progress_beat();
            std::fflush(stdout);
            if (o.prompt_cache_every > 0 && done >= pp_next_check) {
                bool saved = false;
                if (multi_gpu) {   // the stages' parts, saved when each of them read this chunk
                    std::vector<ConvCheckpoint> parts;
                    {
                        std::lock_guard<std::mutex> lk(part_mu);
                        auto it = part_at.find(done);
                        if (it != part_at.end()) parts = std::move(it->second);
                        part_at.erase(part_at.begin(), part_at.upper_bound(done));
                    }
                    bool complete = parts.size() == stages.size() + 1;
                    for (const ConvCheckpoint& k : parts) complete = complete && !k.gdn.empty();
                    saved = !complete || checkpoint_at(done, &parts);   // an incomplete set: no checkpoint here
                } else {
                    saved = checkpoint_at(done);
                }
                if (!saved) { e = "saving a conversation checkpoint failed"; return false; }
                pp_next_check = done + o.prompt_cache_every;
            }
            return true;
        };
        if (multi_gpu) {   // the batched prompt is reported by its last stage (the drafter's rows are there)
            stages.back()->sp.on_chunk = std::move(sp.on_chunk);
            sp.on_chunk = nullptr;
            for (size_t i = 0; i <= stages.size(); ++i) {
                strata::prefill::Prefill& stage_sp = i == 0 ? sp : stages[i - 1]->sp;
                strata::core::SessionState& stage_ss = i == 0 ? ss : stages[i - 1]->ss;
                stage_sp.on_stage_chunk = [&, i](int64_t done, std::string& e) -> bool {
                    if (o.prompt_cache <= 0 || o.prompt_cache_every <= 0 || done < part_next[i]) return true;
                    part_next[i] = done + o.prompt_cache_every;
                    ConvCheckpoint part;   // this stage's state at `done` (its stream is synchronized)
                    part.ids.assign(cur.begin(), cur.begin() + done);
                    if (!checkpoint_save(part, stage_ss, g)) { e = "saving a checkpoint part failed"; return false; }
                    std::lock_guard<std::mutex> lk(part_mu);
                    auto& v = part_at[done];
                    v.resize(stages.size() + 1);
                    v[i] = std::move(part);
                    return true;
                };
            }
        }
        drive.d.plan = ver.plan_sink();
        drive.d.pcie_num = std::max(0, std::min(256, (int) (o.pcie_frac * 256.0 + 0.5)));
        if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        cudaStream_t adapt_stream = nullptr;
        if (cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
            std::fprintf(stderr, "strata serve: cannot create the refill stream\n");
            return 1;
        }
        // plan v0.3 P6: swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        cudaEvent_t adapt_ev = nullptr;
        cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);
        // a layer split's later stages keep a copy of the residency table on their devices, and swap on their own
        //
        // S0.3 lever 2: THE RESIDENCY UPLOAD IS A DECODE-WINDOW COST, NOT A PROMPT COST.
        //
        // The table is `n_layers * n_expert` int32 (24 576 entries = 96 KiB), and today `res_upload()`
        // copies it WHOLE, synchronously, to EVERY device on every lend and every refill: on this box's
        // 3-way split that is 3 blocking copies per call, 6 per request, 9 when the prompt splits at a turn
        // boundary - on links measured at 2.9 / 1.3 / 24.5 GB/s, shared with two other serving processes.
        //
        // Two source facts make most of those copies unnecessary:
        //
        //   * the batched prompt path never reads the DEVICE table.  `Prefill` holds the host pointer
        //     (`init`'s `host_res`) and decides residency with `m.host_res[l * n_expert + e] >= 0`
        //     (prefill.cpp:1097, 1586, 1598, 1697, 1729); the CPU pool's adapter reads `drive.d.host_res`.
        //     The device copy is read by `Verifier::resident_plan` and by the captured token graph - both
        //     DECODE paths.  So a lend does not need an upload: the next thing that reads the device table
        //     is a verify window, and `refill()` always runs before a window reads (the segment loop calls
        //     it at the window boundary and once the prompt is read).
        //   * a request's lend/refill round trip restores the table exactly: `lend` sets
        //     `host_res[i] = kNotResident` for the rows it lends and `refill_one` sets
        //     `host_res[i] = slot` for the same pairs.  So after the refill the host table is byte-identical
        //     to what the devices already hold, and the end-of-request upload moves nothing.
        //
        // `ResidencyUpload` decides by COMPARING the host table with the last content the devices were
        // given, so it is right even if a writer never announced its change - that is what makes skipping
        // safe rather than hopeful, and it is why the adaptive tier's `apply_pending` path (which does
        // change the table for real, between decode rounds) still uploads.
        //
        // STRATA_RES_UPLOAD_ALWAYS=1 restores today's behaviour: upload at lend, and unconditionally.
        strata::program::prefill_loan::ResidencyUpload res_dirty((int64_t) g.n_layers);
        if (d_res != nullptr) res_dirty.synced(host_res);   // startup put this exact table on every device
        auto res_upload = [&]() {
            if (d_res == nullptr) return;
            if (!res_upload_always() && !res_dirty.due(host_res)) {
                res_dirty.note_skipped();
                return;
            }
            cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
            for (auto& st : stages) {
                const strata::core::OnDevice on(st->dev);
                cudaMemcpy(st->d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
            }
            res_dirty.uploaded(host_res);
        };

        // S0.3: the loan's counters, summed over the participants (relayouts run, relayouts skipped,
        // loans grown in place, rows refilled).  Cumulative over the process; a request's own bill is the
        // delta from where it started.  S3.2b adds two cumulative ones: rows the prompt path took out, and
        // rows the pump walked home between decode steps.
        auto loan_totals = [](const std::vector<PfPart>& v) {
            std::array<int64_t, 6> t{0, 0, 0, 0, 0, 0};
            for (const PfPart& p : v) {
                t[0] += p.relayouts; t[1] += p.relayout_skips;
                t[2] += p.loan_grows; t[3] += p.refilled;
                t[4] += p.led.taken(); t[5] += p.led.returned_pumped();
            }
            return t;
        };
        // NOT cumulative: how many rows are out of the caches RIGHT NOW.  Under the lazy rule this is the
        // number that proves the refill was deferred rather than performed - and it is the number that says
        // how much CPU-side decode the next request inherits.
        auto loan_outstanding = [](const std::vector<PfPart>& v) {
            int64_t n = 0;
            for (const PfPart& p : v) n += (int64_t) (p.led.owed() + p.led.inflight());
            return n;
        };
        auto apply_pending = [&](bool wait) {
            if (pending.empty()) return;
            if (wait) cudaEventSynchronize(adapt_ev);
            else if (cudaEventQuery(adapt_ev) != cudaSuccess) return;
            for (auto& st : stages)
                if (st->adapt_live) {
                    if (wait) cudaEventSynchronize(st->adapt_ev);
                    else if (cudaEventQuery(st->adapt_ev) != cudaSuccess) return;
                }
            for (auto& st : stages) st->adapt_live = false;
            src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
            for (const auto& [i, slot] : pending) {
                host_res[(size_t) i] = slot;
                // S3.2b: the adaptive tier just put this row home, in a slot it chose for itself.  If the
                // prompt loan's ledger still owned it, the pump would ALSO copy it into its original slot -
                // two owners for one expert, and the slot the adaptive tier paid a victim for would be
                // orphaned.  `adapt()` already refuses to pick a ledger row as a candidate (see the note
                // there), so this is the belt to that braces: it can only fire for a row that became owed
                // after the swap was queued.
                for (PfPart& p : pf_parts)
                    if (i >= p.lb * g.n_expert && i < p.le * g.n_expert) p.led.drop(i);
            }
            pending.clear();
            res_upload();
        };

        // the batched path's slots are lent just before its first run and given back before a window
        // reads - so the windows always see a cache whose RESIDENCY TABLE is true.  They do not have to see
        // a whole cache: that was S0.3's assumption, and S3.2b's lever 5 is what breaks it.
        //
        // ---- S3.2b: TWO WAYS TO GIVE THE LOAN BACK, and the difference between them is 4.95 GiB. -------
        //
        // EAGER (0.1.30, and the default on the serial path): `refill_one` streams every row it lent back
        // from the arena into the same slot and marks it resident again.  On this box that is 2 625 + 1 971
        // + 1 885 = 6 481 rows, 4.95 GiB per stage, over links measured at 2.9 / 1.3 / 24.5 GB/s - CUDA1's
        // share alone is ~3.8 s - and `--serve` prints `prompt_ms` after it, so all of it is inside the
        // fitted 9.5 s fixed term.
        //
        // LAZY (the default under the concurrent driver): nothing is copied.  The rows stay marked
        // non-resident and go into the cache's LEDGER (`PfPart::led`), and the layout (`first_now`,
        // `sp->chunk()`) stays standing.  Decode runs the rows that are out on the CPU pool, which is what
        // it does for every expert the cache has never held; then the pump walks the ledger home in bounded
        // batches between decode steps.  The NEXT request's `lend()` finds its range already non-resident,
        // marks almost nothing, re-lays nothing, and its `finish_prefill` returns nothing - which is where
        // the fixed cost actually dies, rather than merely being hidden.
        //
        // THE INVARIANT BOTH OBEY, and it is the only thing between this and plausible garbage:
        //     (I1) `host_res[i] >= 0` <=> slot `host_res[i]` on the owning device holds expert i's bytes NOW.
        // A row is marked non-resident BEFORE its slot is handed to the prompt path, and marked resident
        // only AFTER its copy has been confirmed landed (`settle_pump`, one event per cache, at most one
        // batch in flight per cache).  There is no window in which the table claims a slot holds an expert
        // while it holds scratch.  (I2) every device's `d_res` equals `host_res` at the start of every
        // window: `reconcile_residency()` at the top of `run_decode_step`, and `res_upload()` inside every
        // return path.
        //
        // A stage refills through its own cache and its own device, and marks only its own layers' rows - a
        // slot refilled into the wrong cache would leave that stage's cache holding an expert it does not
        // own, which is silent and produces plausible tokens.
        namespace pfl = strata::program::prefill_loan;
        const pfl::LazyRefillPolicy loan_policy = [&] {
            pfl::LazyRefillPolicy p;
            p.rows_per_step = loan_pump_rows();
            const int env = lazy_loan_env();
            // `auto`: only under the concurrent driver.  Those are exactly the three conditions S3.1e-2
            // gates its own loop on, and they are all knowable here (the loop's `driver_on` is computed
            // later, from the same three).
            const bool concurrent_possible = slots_reg.concurrent() &&
                                             !strata::program::serve_swap::swaps_disabled_by_env() &&
                                             conversations.enabled();
            p.lazy = env >= 0 ? (env == 1) : concurrent_possible;
            return p;
        }();
        // Report the mode in force.  Only on the concurrent driver, under the instrument, or when the owner
        // forced the mode: the serial path's stderr stays exactly what 0.1.30 printed otherwise.
        if (loan_timing() || slots_reg.concurrent() || lazy_loan_env() >= 0)
            std::fprintf(stderr, "strata serve: prompt loan return: %s (%lld row(s) per cache in flight on "
                                 "the pump; STRATA_PREFILL_LAZY_LOAN=0 refills the whole loan at the end of "
                                 "every prompt, =1 forces it on)%s\n",
                         loan_policy.lazy ? "LAZY (the rows stay out and walk home between decode steps)"
                                          : "EAGER (streamed back before any window reads)",
                         (long long) loan_policy.rows_per_step,
                         slots_reg.concurrent() ? "" : " - the serial path keeps 0.1.30's behaviour");
        // Confirm the pump batch this cache has in flight, and only then may its rows be resident.  `wait`
        // = block until it lands (an eager refill, or a `lend()` about to overwrite the same slots); false
        // = take it if it already has.  Marking happens here and nowhere else, which is what makes (I1) a
        // property of one function rather than of every caller's discipline.
        auto settle_pump = [&](PfPart& p, bool wait, std::string& e) -> bool {
            if (!p.loan_pump_live) return true;
            {
                const strata::core::OnDevice on(p.dev);
                const cudaError_t q = wait ? cudaEventSynchronize(p.loan_ev) : cudaEventQuery(p.loan_ev);
                if (q == cudaErrorNotReady) return true;      // still in flight: nothing is resident yet
                if (q != cudaSuccess) {
                    e = std::string("confirming a prompt-loan refill copy failed: ") + cudaGetErrorString(q);
                    cudaGetLastError();
                    return false;
                }
            }
            p.loan_pump_live = false;
            // The copies landed.  NOW the rows may be marked resident - in the same order they were queued.
            // This is the only place a lazily-returned row becomes resident again, which is what makes (I1)
            // a property of one function rather than of every caller's discipline.
            //
            // The `host_res < 0` test is not decoration.  If the adaptive tier made this row resident in
            // some OTHER slot while this batch was in flight, `apply_pending()` already dropped it from the
            // ledger and `landed()` will not return it; if anything else did, marking it back into the
            // loan's original slot would orphan the newer one.  So: mark only a row that is still out.
            for (const pfl::LoanRow& r : p.led.landed())
                if (host_res[(size_t) r.index] < 0) host_res[(size_t) r.index] = r.slot;
            return true;
        };
        // Give ONE participant's loan back without copying anything: the rows stay non-resident, the ledger
        // owns them, the layout stays.  The device table must follow the host table, so the caller uploads
        // once afterwards - 96 KiB instead of 4.95 GiB.
        auto return_loan_lazy = [&](PfPart& p) {
            p.loan_live = false;
            if (p.lent.empty()) return;
            p.lent.clear();
            // `lent_chunk` and `first_now` deliberately survive: the buffers are still carved in that slot
            // range, so the next lend that wants the same layout re-lays nothing and re-marks nothing.
        };
        // Put the devices' copy of the table back in step with the host's.  `res_upload()` decides by
        // CONTENT, so when nothing moved this is a 96 KiB compare and no copy.
        //
        // It runs at the top of every window path ONLY in lazy mode, and that is not a performance
        // concession - it is the mode's own requirement.  In lazy mode `settle_pump()` marks rows resident
        // between windows with no lend or refill in between, so the table really can be stale at a window
        // boundary and something must close that gap.  In eager mode nothing marks anything resident except
        // `refill_one()` and `apply_pending()`, and both upload themselves, so the extra call would be a
        // per-window compare for a state that cannot occur - and it would inflate the
        // `res upload N (M skipped)` bill that `bench/prefill/analyze.py --compare` reads against the S0.3
        // baseline.  The serial path's counters therefore stay comparable with the baseline log.
        auto reconcile_residency = [&]() {
            if (!loan_policy.lazy) return;
            res_upload();
        };
        // Stream ONE participant's LIVE loan home and mark it resident.  Eager mode this is the whole
        // end-of-request refill (0.1.30).  Lazy mode it is only reachable from `lend()` handing a loan back
        // mid-request because the operator set `STRATA_PREFILL_STICKY_LOAN=0`, and it deliberately returns
        // the rows of THIS loan and not every row earlier requests left out: turning stickiness off must
        // cost what it costs in 0.1.30, and must not quietly become a full-ledger drain.
        //
        // The full "make the cache whole NOW" primitive is `drain_loans_idle()`, not this: under the lazy
        // rule nothing in the request path needs the cache whole, because a row that is out is marked
        // non-resident and decode runs it on the CPU pool.
        // Stream ONE participant's LIVE loan home and mark it resident.
        //
        // Eager mode this is the end-of-request refill, exactly 0.1.30's.  Lazy mode it is only reachable
        // from `lend()` handing a loan back MID-REQUEST because the operator set
        // `STRATA_PREFILL_STICKY_LOAN=0`, and it deliberately returns the rows of THIS loan and not every
        // row earlier requests left out: turning stickiness off must cost what it costs in 0.1.30, and
        // must not quietly become a full-ledger drain.
        //
        // There is deliberately no "drain the whole ledger now" caller in the request path.  Under the lazy
        // rule nothing in it needs the cache whole - a row that is out is marked non-resident and decode
        // runs it on the CPU pool - and the one place that does want the cache whole is the engine being
        // idle, which is `drain_loans_idle()`.
        auto refill_one = [&](PfPart& p, std::string& e) -> bool {
            if (!settle_pump(p, true, e)) return false;   // no batch may be in flight while we copy
            std::vector<pfl::LoanRow> rows;
            rows.reserve(p.lent.size());
            for (const auto& [i, slot] : p.lent) rows.push_back(pfl::LoanRow{i, slot});
            if (rows.empty()) { p.lent.clear(); p.lent_chunk = 0; p.loan_live = false; return true; }
            tr("refill start", (long long) rows.size());
            const auto tref = Clock::now();
            const strata::core::OnDevice on(p.dev);
            for (const pfl::LoanRow& r : rows) {   // D-4: queued, one wait (STRATA_REFILL_BLOCKING=1: each)
                const uint8_t* b = srcp->blob(r.index / g.n_expert, r.index % g.n_expert);
                const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(r.index / g.n_expert);
                if (b == nullptr || !(refill_blocking() ? p.cache->fill_slot_blocking(r.slot, b, e, nb)
                                                        : p.cache->fill_slot_queued(r.slot, b, e, nb))) {
                    p.led.requeue_inflight();
                    return false;
                }
                // (I1): the copy is queued on the legacy stream and `sync_queued()` below confirms the
                // whole batch before any window can read, so marking here is marking after the bytes are
                // in the slot.
                host_res[(size_t) r.index] = r.slot;
            }
            if (!p.cache->sync_queued(e)) { p.led.requeue_inflight(); return false; }
            p.refilled += (int64_t) rows.size();
            // The rows are home, so the ledger must stop owning them - otherwise the pump would copy the
            // same experts into the same slots a second time.  `refilled()`, never a whole-ledger clear:
            // the rows this call did NOT copy are still out, and still owed.
            for (const pfl::LoanRow& r : rows) p.led.refilled(r.index);
            if (loan_timing()) {
                int64_t bytes = 0;
                for (const pfl::LoanRow& r : rows)
                    bytes += (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(r.index / g.n_expert);
                std::fprintf(stderr, "strata serve: loan refill EAGER CUDA%d: %zu rows (%.2f GiB) in %.1f ms "
                                     "(%zu still out of the cache)\n",
                             p.dev, rows.size(), (double) bytes / 1073741824.0,
                             std::chrono::duration<double, std::milli>(Clock::now() - tref).count(),
                             p.led.owed() + p.led.inflight());
            }
            p.lent.clear();
            p.lent_chunk = 0;   // the rows are the cache's again; the LAYOUT (first_now, sp->chunk()) is not
            p.loan_live = false;
            return true;
        };
        auto refill = [&](std::string& e) -> bool {
            if (loan_policy.lazy) {
                // The lazy return.  Nothing is copied here; the ledger and the pump own the rows from now
                // on, and the table goes out so no window can read a slot the prompt path was standing in.
                // Every participant's `loan_live` clears, including one that marked no rows at all - its
                // buffers were still carved in that range, and the pump must not aim at it until this is off.
                bool any = false;
                for (PfPart& p : pf_parts) {
                    if (p.lent.empty() && !p.loan_live) continue;
                    any = true;
                    return_loan_lazy(p);
                }
                if (any) res_upload();
                return true;
            }
            bool any = false;
            for (PfPart& p : pf_parts)
                if (!p.lent.empty()) { any = true; if (!refill_one(p, e)) return false; }
            // Eager mode: whatever the state, no segment is reading these buffers any more, so the pump
            // (which never runs in this mode) would be allowed to aim at them.  Clear the flag so a mode
            // switch mid-process - or a reader of the report - cannot be misled by a stale one.
            for (PfPart& p : pf_parts) p.loan_live = false;
            if (any) res_upload();
            return true;
        };
        // ---- THE PUMP (lever 5's B): walk the ledgers home in bounded batches, between decode steps. ----
        // One batch per cache at a time, on that cache's own non-blocking stream, so the DMA never sits in
        // front of the next window on the session stream, and so one event per cache can confirm exactly
        // the copies queued since it was last recorded.  A cache with a LIVE loan is skipped entirely: its
        // prompt buffers ARE those slots, and a copy landing inside a live range is scratch by the time the
        // segment finishes - and the table would have said otherwise in between.
        auto pump_loans = [&](std::string& e) -> bool {
            if (!loan_policy.lazy) return true;
            for (PfPart& p : pf_parts) {
                if (!settle_pump(p, false, e)) return false;
                if (!pfl::pump_may_run(loan_policy, p.led, p.loan_live)) continue;
                const size_t n = pfl::pump_batch(loan_policy, p.led);
                if (n == 0) continue;
                if (p.loan_stream == nullptr) {
                    const strata::core::OnDevice on(p.dev);
                    cudaStream_t s = nullptr;
                    cudaEvent_t ev = nullptr;
                    if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess ||
                        cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
                        if (s != nullptr) cudaStreamDestroy(s);
                        e = "cannot create the prompt-loan pump stream";
                        return false;
                    }
                    p.loan_stream = s;
                    p.loan_ev = ev;
                }
                const std::vector<pfl::LoanRow> rows0 = p.led.pump(n, &drive.d.usage);
                // Belt against the one way a ledger row could already be home: the adaptive tier promoted it
                // into some OTHER row's vacated slot (`adapt()` refuses that via `owns()`, and
                // `apply_pending()` drops it, but a row can only be pumped if it was owed when the batch was
                // taken).  Copying it into its original slot anyway would be harmless-but-wasted; marking it
                // resident twice is not, so the row is dropped from the ledger instead.
                //
                // The reverse hazard - `ExpertCache::admit()` handing a LEDGER SLOT to a different expert -
                // cannot happen, and that is worth stating because it is the one thing this design depends
                // on: the cache never evicts and never re-hands-out a slot, so `next_free_` (and per-layer
                // `layer_next_`) is always strictly past every slot ever handed out, and a ledger slot was
                // handed out.  A new admission therefore gets a slot no residency entry has ever pointed at.
                std::vector<pfl::LoanRow> rows;
                rows.reserve(rows0.size());
                for (const pfl::LoanRow& r : rows0) {
                    if (host_res[(size_t) r.index] >= 0) { p.led.drop(r.index); continue; }
                    rows.push_back(r);
                }
                if (rows.empty()) continue;
                const auto tpump = Clock::now();
                int64_t bytes = 0;
                {
                    const strata::core::OnDevice on(p.dev);
                    for (const pfl::LoanRow& r : rows) {
                        const uint8_t* b = srcp->blob(r.index / g.n_expert, r.index % g.n_expert);
                        const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(r.index / g.n_expert);
                        if (b == nullptr || !p.cache->fill_slot(r.slot, b, p.loan_stream, e, nb)) {
                            p.led.requeue_inflight();   // not resident, not home: try again later
                            return false;
                        }
                        bytes += nb;
                    }
                    if (cudaEventRecord(p.loan_ev, p.loan_stream) != cudaSuccess) {
                        p.led.requeue_inflight();
                        e = "cannot record the prompt-loan pump event";
                        return false;
                    }
                }
                p.loan_pump_live = true;
                if (loan_timing())
                    std::fprintf(stderr, "strata serve: loan pump CUDA%d: %zu rows (%.2f GiB) queued in %.1f ms "
                                         "(%zu still out of the cache)\n",
                                 p.dev, rows.size(), (double) bytes / 1073741824.0,
                                 std::chrono::duration<double, std::milli>(Clock::now() - tpump).count(),
                                 p.led.owed() + p.led.inflight());
            }
            return true;
        };
        // the VRAM tier follows the conversation (the same rule as the speculative loop below)
        // ---- S3.2b: the IDLE drain.  The pump between decode steps is bounded so it can never crowd a
        // window; that leaves a question the report has to be able to answer - when does a cache that has
        // been running for a while get WHOLE again?  Here: while the engine has nothing to run.  The GPU is
        // free then, the copies are the only work on the device, and the next request's decode starts from a
        // warmer cache.  Bounded by wall time rather than rows, because the driver thread is the one that
        // reads the next request line: a long drain must not delay an arriving request by more than this.
        //
        // It is NOT required for correctness.  A ledger that never drains is still correct - every row in it
        // is marked non-resident, decode runs it on the CPU pool, and the next `lend()` finds it already out,
        // which is the free case.  This is purely the hit-rate recovery, paid at the cheapest possible moment.
        auto drain_loans_idle = [&](int64_t budget_ms, auto&& line_waiting) {
            if (!loan_policy.lazy) return;
            const auto t0 = Clock::now();
            std::string derr;
            bool bad = false;
            for (;;) {
                const int64_t before = loan_outstanding(pf_parts);
                if (!pump_loans(derr)) { err = derr; err.clear(); bad = true; break; }
                bool any_live = false;
                for (const PfPart& p : pf_parts) if (p.loan_pump_live) any_live = true;
                if (any_live) {
                    // Wait for the batch this round queued, so the next round may queue another.  One batch
                    // per cache at a time is the invariant `settle_pump`'s single event relies on.
                    for (PfPart& p : pf_parts)
                        if (p.loan_pump_live) {
                            if (!settle_pump(p, true, derr)) { err = derr; err.clear(); bad = true; break; }
                            break;
                        }
                    if (bad) break;
                } else if (loan_outstanding(pf_parts) == before) {
                    // Nothing in flight and nothing new queued: either the caches are whole, or a cache's
                    // loan is still physically live and the pump is (correctly) refused.  Either way this
                    // loop cannot make further progress, and spinning here would burn 250 ms of a thread
                    // that is supposed to be waiting for the next request line.
                    break;
                }
                if (std::chrono::duration<double, std::milli>(Clock::now() - t0).count() >= budget_ms) break;
                // A request line that arrived mid-drain wins: this is the same thread that has to notice
                // it, and a bounded cache warming is not worth that much first-token latency.
                if (line_waiting()) break;
            }
            (void) bad;
            // (I2) again, locally: `pump_loans`/`settle_pump` may have marked rows resident, so the devices
            // get the table before this returns rather than at some later caller's window boundary.
            // `res_upload()` compares content, so when nothing moved this is a 96 KiB compare.
            res_upload();
            if (loan_timing()) {
                const int64_t out = loan_outstanding(pf_parts);
                if (out == 0)
                    std::fprintf(stderr, "strata serve: loan drain idle: the caches are whole again (%.0f ms)\n",
                                 std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
            }
        };
        // S3.2b: is this residency index one of the prompt loan's OUT rows?  `adapt()` must not promote one:
        // it would take some other row's slot as a victim, and the pump would then ALSO copy the same expert
        // into the slot it already owns, orphaning the one the adaptive tier paid for.  The pump is the
        // better choice anyway - its slot is already the row's own, so it needs no victim at all.
        auto loan_owns_row = [&](int64_t index) -> bool {
            if (!loan_policy.lazy) return false;   // eager mode never populates a ledger
            for (const PfPart& p : pf_parts)
                if (index >= p.lb * g.n_expert && index < p.le * g.n_expert && p.led.owns((int32_t) index))
                    return true;
            return false;
        };
        auto adapt = [&]() -> bool {
            if (!pending.empty()) return true;   // the previous swaps are still in flight
            struct Swap { float gain; int32_t layer, in, out; };
            std::vector<Swap> swaps;
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) {
                        if (u[e] >= 2.0f && !loan_owns_row(l * g.n_expert + e)) cand.emplace_back(u[e], e);
                    }
                    else vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
                }
            }
            std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            if (!resident_stage_swaps(src, xcache, host_res, g.n_expert, swaps, adapt_stream)) return false;
            bool main_live = false;
            for (const Swap& s : swaps) {
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = host_res[out];
                const uint8_t* b = srcp->blob(s.layer, s.in);
                const int stn = multi_gpu ? stage_of(s.layer) : 0;   // the swap stays in the layer's own cache
                GpuStage* gs = stn > 0 ? stages[(size_t) stn - 1].get() : nullptr;
                const strata::core::OnDevice on(gs ? gs->dev : -1);
                if (slot < 0 || b == nullptr ||
                    cudaMemcpyAsync(gs ? gs->cache.device_slot(slot) : xcache.device_slot(slot), b,
                                    (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer),
                                    cudaMemcpyHostToDevice, gs ? gs->adapt_stream : adapt_stream) != cudaSuccess)
                    return false;
                if (gs) gs->adapt_live = true;
                else main_live = true;
                host_res[out] = strata::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            }
            if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
            (void) main_live;
            for (auto& st : stages)
                if (st->adapt_live) {
                    const strata::core::OnDevice on(st->dev);
                    cudaEventRecord(st->adapt_ev, st->adapt_stream);
                }
            for (float& v : drive.d.usage) v *= 0.7f;
            return true;
        };
        // stdin is read on its own thread, so a STOP line reaches a request that is still running (the client went
        // away, or pressed Esc): the flag is checked between prompt chunks and between verify windows.
        std::atomic<bool> stop_req{false};
        std::mutex in_mu;
        std::condition_variable in_cv;
        std::deque<std::string> in_lines;
        bool in_eof = false;
        std::thread([&] {
            // read(2) on the descriptor, not std::cin: glibc's exit() flushes every stdio stream and waits for
            // stdin's lock, which getline holds while it waits for input - an engine ending on an error (every
            // std::exit) would hang in exit() on Linux, and the server would wait for it forever
            std::string l, buf;
            char chunk[4096];
            auto getline_fd = [&](std::string& out) -> bool {
                for (;;) {
                    const size_t nlpos = buf.find('\n');
                    if (nlpos != std::string::npos) {
                        out.assign(buf, 0, nlpos);
                        buf.erase(0, nlpos + 1);
                        return true;
                    }
#if defined(_WIN32)
                    const int n = _read(0, chunk, (unsigned) sizeof chunk);
#else
                    const ssize_t n = ::read(0, chunk, sizeof chunk);
                    if (n < 0 && errno == EINTR) continue;
#endif
                    if (n <= 0) {
                        if (buf.empty()) return false;
                        out.swap(buf);
                        buf.clear();
                        return true;
                    }
                    buf.append(chunk, (size_t) n);
                }
            };
            while (getline_fd(l)) {
                if (!l.empty() && l.back() == '\r') l.pop_back();
                // S3.1c: STOP, or (with --serve-slots >= 2) STOP <id>.  A bare STOP is 0.1.30's: it stops
                // whatever is running.  STOP <id> stops THAT request - and while the engine still runs one
                // request to completion (S3.1e owns the interleaving), the only request that can be running
                // is the one whose id is in `running_id`, so the legacy `stop_req` is set exactly when the
                // named request is the running one and left alone when it is not.  That is the whole point
                // of naming a request: `STOP 2` while request 1 runs must not cancel request 1.
                if (l == "STOP" || (tagged && l.rfind("STOP ", 0) == 0)) {
                    const bool has_id = l.size() > 5;
                    if (!has_id) {
                        std::lock_guard<std::mutex> lk(in_mu);
                        stop_req.store(true);
                        {
                            std::lock_guard<std::mutex> slk(slot_mu);
                            const int64_t who = slots_reg.newest_id();
                            if (who != strata::program::serve_proto::kNoId) slots_reg.cancel_request(who);
                        }
                        continue;
                    }
                    long long v = 0;
                    bool numeric = true;
                    for (size_t i = 5; i < l.size(); ++i)
                        if (l[i] < '0' || l[i] > '9') { numeric = false; break; }
                    if (numeric) v = std::atoll(l.c_str() + 5);
                    if (!numeric) {   // a malformed STOP <id>: 0.1.30 had no such line, so it is a bad request
                        std::lock_guard<std::mutex> lk(in_mu);
                        in_lines.push_back(l);
                        in_cv.notify_one();
                        continue;
                    }
                    {
                        std::lock_guard<std::mutex> slk(slot_mu);
                        if (!slots_reg.cancel_request(v) && running_id.load() != v) {
                            // The request has not been admitted yet (its line is still on the pipe).  Remember
                            // the cancel so the request loop applies it the moment the row exists - otherwise a
                            // client that cancels a queued request would have it run anyway.
                            if (pending_cancel.size() < 64) pending_cancel.push_back(v);
                        }
                    }
                    if (running_id.load() == v) stop_req.store(true);
                    continue;
                }
                std::lock_guard<std::mutex> lk(in_mu);
                in_lines.push_back(l);
                in_cv.notify_one();
            }
            std::lock_guard<std::mutex> lk(in_mu);
            in_eof = true;
            in_cv.notify_one();
        }).detach();
        auto next_line = [&](std::string& out) -> bool {
            std::unique_lock<std::mutex> lk(in_mu);
            in_cv.wait(lk, [&] { return !in_lines.empty() || in_eof; });
            if (in_lines.empty()) return false;
            out = std::move(in_lines.front());
            in_lines.pop_front();
            return true;
        };
        sp.should_stop = [&] {
            if (!tagged) return stop_req.load();
            if (stop_req.load()) return true;
            std::lock_guard<std::mutex> lk(slot_mu);
            const strata::program::slot::Slot* s = slots_reg.find(req_id);
            return s != nullptr && s->cancel.load();
        };
        // The same question for the request body's own checkpoints (between verify windows, between prompt
        // segments).  With --serve-slots 0/1 this is `stop_req`, exactly today; with >= 2 a `STOP <id>` that
        // named a DIFFERENT request does not stop this one, which is the whole point of naming it.
        auto stopped = [&]() -> bool { return sp.should_stop(); };
        // STRATA_TRACE: `trace`/`tr` are declared above `park_current` (S3.1d needs them in the slot
        // hand-over as well); the request body below uses those same ones.
        {
            // what is left once everything is allocated: under WDDM a GPU filled to the brim does not fail, it pages -
            // and a page-in while the verify graph spins on a host flag stalls the request for good
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            // below ~256 MiB a later allocation (a first-used window's buffers, the desktop, another program) can make
            // the driver page GPU memory, and a verify graph spinning on a host flag then never finishes
            const int64_t free_mib = (int64_t) (free_b >> 20);
            if (free_mib >= 256) {
                std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded\n", (long long) free_mib);
            } else {
                std::fprintf(stderr, "strata serve: %lld MiB of VRAM free with everything loaded - LOW: requests may stall;"
                                     " add --vram-reserve-mib %lld to the config's args (or lower --max-context)\n",
                             (long long) free_mib, (long long) (o.vram_reserve_mib + 512 - free_mib));
            }
        }
        // what the server's Monitor tab shows (servers before 0.1.8 skip unknown lines until READY)
        {
            size_t free_b = 0, total_b = 0;
            cudaMemGetInfo(&free_b, &total_b);
            // "Experts in VRAM" is every tier, not one card's.  This used to report `xcache` alone, so a layer split
            // showed CUDA0's cache as if it were the whole GPU's: a 4-GPU 256K run read 3327 experts when the four
            // cards held 13320, and the Monitor tab was wrong by 4x for every multi-GPU config.  The tiers are
            // disjoint by construction (remote_experts.cpp skips any pair a stage already claimed), so they add.
            const int64_t slots_primary = (int64_t) xcache.slots();
            const int64_t mib_primary = (int64_t) (xcache.bytes() >> 20);
            int64_t slots_all = slots_primary, mib_all = mib_primary;
            for (const auto& st : stages) {
                slots_all += (int64_t) st->cache.slots();
                mib_all += (int64_t) (st->cache.bytes() >> 20);
            }
            for (int r = 0; r < 3; ++r)
                if (o.expert_cache_remote[(size_t) r] > 0) {
                    slots_all += remote_experts[(size_t) r].resident();
                    mib_all += (int64_t) (remote_experts[(size_t) r].gib() * 1024.0);
                }
            std::printf("INFO context=%lld kv=%s kv_resident=%lld expert_slots=%lld expert_cache_mib=%lld "
                        "expert_slots_primary=%lld expert_cache_primary_mib=%lld spec=%d "
                        "mtp_max=%d lookup=%d vram_free_mib=%lld cvec=%s arena_mib=%lld pool_workers=%d pcie_frac=%.2f "
                        "spec_min_p=%.2f conversation_cache_mib=%lld conversation_cache_slots=%d "
                         "conversation_cache_min_free_mib=%lld slots=%d slots_active=0 concurrency=%d "
                         "slot_swap=%d "
                         "engine=" STRATA_VERSION "\n",
                        (long long) o.max_context, o.kv.c_str(),
                        (long long) (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1
                                         ? ss.qsa_states[ss.qsa_primary()].n_slots * 4 : 0),
                        (long long) slots_all, (long long) mib_all,
                        (long long) slots_primary, (long long) mib_primary,
                        o.spec, o.mtp_max_t,
                        o.suffix_draft, (long long) (free_b >> 20), cvec_summary.c_str(),
                        (long long) ((o.mmap_experts ? src.resident_bytes() : strata::kernels::cpu::expert_layout().total) >> 20),
                        pool.workers(), o.pcie_frac,
                        o.spec_min_p, (long long) (conversations.budget() >> 20), o.conversation_cache_slots,
                         (long long) o.conversation_cache_min_free_mib, slots, tagged ? 1 : 0,
                         strata::program::serve_swap::swaps_disabled_by_env() ? 0 : (tagged ? 1 : 0));
        }
        // issue #29: a request whose heartbeat (tokens, prompt chunks, verify windows) stops for this long is stuck on
        // a flag nobody will raise - end the engine with where it was, so the server starts it again instead of the
        // GPU spinning forever.  STRATA_WATCHDOG_S=0 turns it off.  Issue #31: before it does, it reports what every
        // part was doing (stall_report), so one occurrence says where the wait is.
        {
             const char* ws = std::getenv("STRATA_WATCHDOG_S");
             watchdog_limit_s = ws ? std::atoi(ws) : 60;   // one step (a prompt layer, a verify window) takes seconds
             if (watchdog_limit_s > 0)
                 std::thread([limit = watchdog_limit_s, tagged] {
                     strata::core::Progress& p = strata::core::progress();
                     uint64_t last = p.beats.load(), ticks_at = p.ticks.load();
                     auto since = std::chrono::steady_clock::now();
                     for (;;) {
                         std::this_thread::sleep_for(std::chrono::seconds(1));
                         const auto now = std::chrono::steady_clock::now();
                         const uint64_t b = p.beats.load();
                         if (!p.busy.load() || b != last) { last = b; ticks_at = p.ticks.load(); since = now; continue; }
                         if (now - since < std::chrono::seconds(limit)) continue;
                         // ---- S3.1e-2, risk R3: with the concurrent driver, WHICH slots stalled? ----
                         // `beats` is process-wide, so a frozen counter means NO slot moved - which with
                         // one engine thread means the thread is stuck inside one step.  The per-slot
                         // view still matters: a slot the driver has not run for many times the limit is
                         // the one that is actually stuck, and the others are merely starved behind it.
                         // Aborting is right only when there is nothing else to save; naming the slots is
                         // right always.
                         if (tagged) {
                             const int64_t now_ms = strata::core::progress_now_ms();
                             std::vector<strata::program::serve_driver::SlotWatch> snap;
                             std::vector<int64_t> kill_left;
                             {
                                 std::lock_guard<std::mutex> lk(watch_mu);
                                 snap = watch;
                                 // The driver drains `watch_kill` between steps.  Anything still here on
                                 // the next pass means the engine thread never came back - it is stuck
                                 // INSIDE a step, which is exactly issue #29 - and no request-level
                                 // unwind is going to happen.  Escalate.
                                 for (const int64_t id : watch_kill) kill_left.push_back(id);
                             }
                             const std::vector<int64_t> stalled = strata::program::serve_driver::stalled_slots(
                                 snap, now_ms, last, (int64_t) limit * 1000);
                             if (!kill_left.empty()) {
                                 std::fprintf(stderr, "strata serve: no progress for %d s and the driver did not act on "
                                                      "the %zu stalled slot(s) it was told about (%s) - the engine "
                                                      "thread itself is stuck; stopping it (issue #29)\n",
                                              limit, kill_left.size(),
                                              strata::program::serve_driver::stalled_list(kill_left).c_str());
                                 stall_report(stderr, p.ticks.load() - ticks_at);
                                 std::fflush(stderr);
                                 std::abort();
                             }
                             if (!stalled.empty() &&
                                 !strata::program::serve_driver::watchdog_aborts(stalled, snap.size())) {
                                 // Not every watched slot is responsible: give up on THOSE requests and
                                 // keep the engine and the other conversations alive.  The driver turns
                                 // these into `ERR <id>` + slot destruction between steps; if it cannot
                                 // (because it is the thing that is stuck), the pass above aborts.  `since`
                                 // is NOT reset, so that escalation is one second away, not another limit.
                                 std::fprintf(stderr, "strata serve: no progress for %d s; %s stalled while the "
                                                      "other slots are current - ending those requests instead of "
                                                      "the engine (issue #29, stage 3 R3)\n",
                                              limit, strata::program::serve_driver::stalled_list(stalled).c_str());
                                 std::fflush(stderr);
                                 std::lock_guard<std::mutex> lk(watch_mu);
                                 for (const int64_t id : stalled) watch_kill.push_back(id);
                                 continue;
                             }
                             std::string who;
                             if (!stalled.empty())
                                 who = " every slot the driver is watching stalled: " +
                                       strata::program::serve_driver::stalled_list(stalled) +
                                       " (" + std::to_string((long long) snap.size()) + " watched)";
                             std::fprintf(stderr, "strata serve: no progress for %d s during a request (%s %lld)%s - "
                                                  "stopping the engine so the server starts it again (issue #29)\n",
                                          limit, p.where.load(), (long long) p.detail.load(), who.c_str());
                             stall_report(stderr, p.ticks.load() - ticks_at);
                             std::fflush(stderr);
                             std::abort();
                         }
                         std::fprintf(stderr, "strata serve: no progress for %d s during a request (%s %lld) - stopping "
                                              "the engine so the server starts it again (issue #29)\n",
                                      limit, p.where.load(), (long long) p.detail.load());
                         stall_report(stderr, p.ticks.load() - ticks_at);
                         std::fflush(stderr);
                         std::abort();
                     }
                 }).detach();
         }
        std::printf("%s\n", sp_out.ready(o.max_context, slots).c_str());   // "stop": this engine honours STOP;
        // "slots=N": it also names requests (S3.1c).  Absent with --serve-slots 0/1, which is what makes an
        // old server read a new engine exactly as it read 0.1.30 (docs/STAGE3-CONCURRENCY.md §6.4).
        std::fflush(stdout);
        std::string line;
        int64_t rounds = 0;
        const int S = o.spec;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, S) : S;   // the MTP's windows; suffixes go up to S
        if (S_mtp < S) mtp.set_max_drafts(S_mtp - 1);
        strata::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        // S3.1d: the drafter's history IS the sequence (spec/suffix_drafter.hpp:47-48), and `hist_` is
        // private, so the serve loop keeps a mirror of what it appended.  The mirror is what travels
        // with a slot; `apply_slot` rebuilds the drafter from it on a mount.  Reserved once so the
        // decode path allocates nothing new.
        std::vector<int32_t> sfx_hist;
        if (tagged) sfx_hist.reserve((size_t) o.max_context + 4096);   // no new allocation at all when off
        strata::spec::DraftPolicy policy(S);   // MTP or lookup window, learned over the whole process
        // The vision path (--vision): GENI <max_new> <embeddings file> <id,id,...> carries images.  The file is one
        // or more strata-vision records (int32 'SVE1', n, nx, ny, n_embd, then n x n_embd floats) in prompt order;
        // each image's rows go to its run of <|image_pad|> tokens, whose M-RoPE positions are mtmd's: t = p,
        // h = p + y, w = p + x, and the text after the image continues at p + max(nx, ny).
        constexpr int64_t kImagePad = 248056;   // qwen4exp.ple.image_token_id: the PLE hash reads it for image cells
        bool mrope_identity = true;
        std::vector<float> img_rows;
        std::vector<const float*> row_ptr;
        // ==================== S3.1e-1: the prompt-path helpers, hoisted out of the request body ====
        // These three only ever captured serve-scope state, so moving them from the request body to the
        // serve scope changes nothing: same captures, same code, same call sites.  They move because
        // S3.1e-2's `run_prefill_step` calls them from outside the body, one segment per step, and a
        // step function cannot reach a lambda declared inside a loop iteration.
        // A SHORT PART OF THE PROMPT - the new message of a chat that continues from a checkpoint, the assistant
        // header - goes through the verify windows, S tokens at a time, as decode reads them.  The batched path
        // costs ~300 ms per run however few tokens it has (it streams every expert the chunk routes to that is
        // not in VRAM over PCIe), and it borrows slots it must refill after (~180 ms); a window costs ~16 ms a
        // token, with the misses on the CPU.  Each part below is decided on its own, so a long first message is
        // read batched and its header still goes through the windows.  Picture rows need the batched path.
        // STRATA_CKPT_REREAD compares a restored checkpoint with a batched re-read, so it keeps every read batched.
        static const bool no_short = std::getenv("STRATA_CKPT_REREAD") != nullptr;
        auto windows_ok = [&](int64_t a, int64_t b) -> bool {
            if (no_short || b - a > o.short_read) return false;
            if (sp.embd_rows != nullptr)
                for (int64_t i = a; i < b; ++i)
                    if (sp.embd_rows[i] != nullptr) return false;
            return true;
        };
        // tokens [a, b) through the windows: commit all of them, then give the draft layer their residuals
        auto read_windows = [&](int64_t a, int64_t b, std::string& e) -> bool {
            strata::core::progress_at("reading the prompt (verify windows), from token", a);   // #217: not "batched"
            // S3.2b: these ARE verify windows, so they read the DEVICE table (`Verifier::resident_plan`).
            // Under the lazy loan the host table can have moved since the last upload without any lend or
            // refill in between - the pump marks rows resident as their copies land - so the table goes out
            // here too.  `res_upload()` compares content, so when nothing moved this costs a 96 KiB compare.
            // Order matters: pump first (which is what may have moved it), then upload.
            if (!pump_loans(e)) return false;
            reconcile_residency();
            // every token is committed and the picks are discarded: no head sampling (see set_head_sampling)
            struct NoHeadSampling {
                strata::core::Verifier& v;
                explicit NoHeadSampling(strata::core::Verifier& x) : v(x) { v.set_head_sampling(false); }
                ~NoHeadSampling() { v.set_head_sampling(true); }
            } no_head_sampling(ver);
            std::vector<int32_t> win((size_t) S), outw((size_t) S), nxt((size_t) S);
            for (int64_t q = a; q < b;) {
                if (stopped()) { e = "cancelled"; return false; }
                const int T = (int) std::min<int64_t>(S, b - q);
                for (int t = 0; t < T; ++t) {
                    win[(size_t) t] = (int32_t) cur[(size_t) (q + t)];
                    nxt[(size_t) t] = (int32_t) cur[(size_t) (q + t + 1)];
                }
                drive.d.layers = 0;
                drive.d.experts = 0;
                drive.d.failed = false;
                if (!ver.run(T, win.data(), q, win_pool_fn, win_pool_user, outw.data(), e) || drive.d.failed) {
                    if (drive.d.failed && drive.d.fail) e = drive.d.fail;
                    return false;
                }
                if (!ver.commit(T, e) || !mtp.prefill(ver.final_R_all(), nxt.data(), T, q, e)) return false;
                q += T;
            }
            const double ms = std::chrono::duration<double, std::milli>(Clock::now() - pp_t0).count();
            std::printf("%s\n", sp_out.pp(b, pp_total, ms,
                                        ms > 0.0 ? 1000.0 * (double) (b - pp_from) / ms : 0.0,
                                        req_id).c_str());
            strata::core::progress_beat();
            std::fflush(stdout);
            return true;
        };
        // `refill_one`/`refill` moved up next to `apply_pending` (S3.1d): a slot hand-over has to
        // give the prompt loan back BEFORE it saves the session, so the swap needs them too.  Same
        // code, declared earlier; every call site below is unchanged.
        // lend the slots `tokens` batched prompt tokens need: the prompt path's buffers for min(chunk, tokens
        // rounded up to 256), laid out in the last of the slots it may borrow - per participant, out of that
        // participant's own cache, and marking only that participant's own layers
        auto lend = [&](int64_t tokens, std::string& e) -> bool {
            if (pf_parts.empty()) return true;                     // its own buffers: nothing to lend
            // what this segment needs, capped by the configured chunk: a request lends only what its own
            // prompt needs, so a large chunk costs a short prompt nothing
            const int64_t want = request_chunk(tokens, o.prefill_chunk);
            if (want <= 0) {
                e = "prefill: cannot lend buffers for an empty request segment";
                return false;
            }
            const auto tlend = Clock::now();
            bool any = false;
            int64_t did_relayout = 0, did_grow = 0;
            for (PfPart& p : pf_parts) {
                if (p.first < 0) continue;
                // S3.2b: a pump batch aimed at THIS cache's slots must have landed (or been abandoned)
                // before the prompt buffers are carved over them.  `pump_loans()` never queues one while a
                // loan is live, so the only batch that can still be in flight here was queued during the
                // previous request's decode - and waiting for it is one event, not a stream sync.
                if (!settle_pump(p, true, e)) return false;
                // the range this segment's chunk needs, and the layout that would serve it
                const int32_t first = std::max<int32_t>(p.first, (int32_t) (p.cache->slots() - part_slots(p, want)));
                const strata::program::prefill_loan::LoanLayout have{p.sp->chunk(), p.first_now};
                const strata::program::prefill_loan::LoanLayout need{want, first};
                bool grow = false;
                // Does the loan already in place cover this segment?  Under the lazy rule the answer is
                // irrelevant to whether the marking loop has to run (see below), and under the eager rule
                // "yes" means the whole lend is a no-op - 0.1.30's short-circuit.
                const bool covers = !p.lent.empty() && want <= p.lent_chunk;
                // From here this participant's cache is the one the segment's buffers sit in.  Set before
                // any `continue`: the pump reads this flag, and a stale `false` would let it copy an expert
                // into a slot the prompt buffers occupy.
                p.loan_live = true;
                if (covers && !strata::program::prefill_loan::lend_must_remark(loan_policy))
                    continue;                                  // its current loan already covers this
                if (!p.lent.empty() && !covers) {
                    // S0.3: a bigger segment does not need its loan handed back first.  `part_slots` is
                    // monotone in the chunk, so the wider range starts no later and already contains every
                    // row on loan: keep those, add the new ones, and re-lay once.  Handing the loan back
                    // here would stream the whole range back over PCIe and take it again a moment later.
                    grow = sticky_loan() && strata::program::prefill_loan::loan_grows(p.first_now, first);
                    // Handing it back: eager mode does, for a range that is not a superset (0.1.30).  Lazy
                    // mode does not need to - the rows outside the new range are already non-resident and
                    // already in the ledger, so they simply stay there and the pump takes them home later.
                    // `STRATA_PREFILL_STICKY_LOAN=0` still means "give it back first", lazy or not.
                    if (!grow && (!sticky_loan() ||
                                  strata::program::prefill_loan::narrow_lend_refills_first(loan_policy)) &&
                        !refill_one(p, e))
                        return false;   // ONLY this participant's loan goes back: `refill` would return the
                                        // other participants' loans too, and their buffers are still laid out
                                        // in their caches - marking those slots resident again would hand the
                                        // next window a prompt buffer in place of an expert.  `live_only`
                                        // because this is a MID-REQUEST hand-back: it returns the rows of
                                        // this loan, not every row earlier requests left out.
                }
                const strata::core::OnDevice on(p.dev);
                if (strata::program::prefill_loan::needs_relayout(have, need)) {
                    if (!p.sp->relayout(want, p.cache->device_slot(first), part_bytes(p, first), e)) return false;
                    p.first_now = first;
                    ++p.relayouts;
                    ++did_relayout;
                } else {
                    ++p.relayout_skips;   // the layout already in place is the one this segment needs
                }
                if (grow) { ++p.loan_grows; ++did_grow; }
                // MARK THE ROWS OUT, which is the point at which their slots stop holding them.  Nothing
                // reads the table between the relayout above and here (one engine thread, and no window runs
                // while a loan is being taken), and nothing reads it after here until the segment's own
                // residency check - which reads the HOST table, not the device copy.  Under the lazy rule
                // this loop is also what re-takes the rows the pump may have walked home since the previous
                // segment, which is why the "my loan already covers this" short-circuit above is skipped in
                // lazy mode.
                for (int64_t l = p.lb; l < p.le; ++l) {             // THIS participant's layers only
                    for (int64_t ex = 0; ex < g.n_expert; ++ex) {
                        const size_t i = (size_t) (l * g.n_expert + ex);
                        if (host_res[i] >= first) {
                            p.lent.emplace_back((int32_t) i, host_res[i]);
                            // S3.2b: the ledger owns the row from the moment its slot stops holding it.
                            // `take()` is idempotent, so a row that has been out since the previous request
                            // is not counted twice - which is the whole cross-request saving: a range that is
                            // already non-resident marks nothing new.  Eager mode keeps the ledger empty:
                            // `refill_one` works from `p.lent` there, and populating a 6 481-entry ledger it
                            // would immediately throw away is a cost the revert arm should not pay.
                            if (loan_policy.lazy) p.led.take((int32_t) i, host_res[i]);
                            host_res[i] = strata::core::kNotResident;
                            any = true;
                        }
                    }
                }
                p.lent_chunk = want;
                // The prompt buffers for this segment ARE in this cache's slot range until the loan is
                // returned.  That is true whether or not the marking loop above found a single row to mark
                // (under the lazy rule it usually finds none, because the previous request left them out),
                // and the pump has to know the difference: copying an expert into a slot the buffers occupy
                // and then marking it resident is invariant (I1) broken in the dangerous direction.
                p.loan_live = true;
            }
            if (loan_timing())
                std::fprintf(stderr, "strata serve: loan lend: %lld tokens in %.1f ms (%lld relayout, %lld grown, "
                                     "%zu row(s) now out of the caches)\n",
                             (long long) want,
                             std::chrono::duration<double, std::milli>(Clock::now() - tlend).count(),
                             (long long) did_relayout, (long long) did_grow, loan_outstanding(pf_parts));
            // S0.3 lever 2: no upload here.  The prompt path decides residency from the HOST table
            // (`Prefill` holds `host_res`; the CPU pool's adapter reads `drive.d.host_res`), so the
            // devices' copy is not read until a verify window runs - and `reconcile_residency()` runs at the
            // top of every window path, every time.  `change()` is bookkeeping for the report only: the skip
            // decision is a content comparison, so it cannot be fooled by an unannounced write.
            // STRATA_RES_UPLOAD_ALWAYS=1 reverts this too: upload at lend, as before.
            if (any) {
                if (res_upload_always()) res_upload();
                else res_dirty.change();
            }
            return true;
        };

        // ============================ S3.1d: MOUNT / UNMOUNT A SLOT ============================
        //
        // docs/STAGE3-CONCURRENCY.md §5.3: "There is exactly one active slot per layer-split stage.
        // The captured graphs always point at it.  Changing which conversation is active is a
        // save/restore, never a re-capture."  Everything the scheduler (S3.1e) will need in order to
        // put a second conversation into this one session is here, and the pure part of it - the
        // order, the failure classes, the budget question, the mrope exclusivity - lives in
        // include/strata/program/serve_swap.hpp where a CPU test can pin it
        // (src/program/serve_swap_test.cpp, 105 checks).
        //
        // WHAT IS AND IS NOT PER-SLOT.  The ConversationCache image is the vehicle for the session's
        // sequence state: `SavedConversation` already carries `live`, `live_imgs`, `checks` and the
        // cvec flag, and `conversation_snapshot_restore` puts them back.  So the checkpoint chain is
        // NOT copied into the slot record - it is a SHARED prefix chain (PR #65's pinned system-prompt
        // root), and per-slot copies would both break that reuse and cost ~118 MB each.  What the
        // image cannot carry is what `SlotConv` owns: the suffix drafter's history, the sampling
        // params and penalty window, the M-RoPE table (R7), the chain's LRU clock and the drafter's
        // prompt length.
        //
        // GATE.  `swaps_on` is false for --serve-slots 0/1 (byte-identical to 0.1.30, §7.1) and for
        // STRATA_NO_SWAP=1 (the owner's escape hatch: today's serial behaviour, no recompile).  With
        // the gate off, the request body below runs 0.1.30's park/mount sequence verbatim.
        const bool swap_env_off = strata::program::serve_swap::swaps_disabled_by_env();
        const bool swaps_on = tagged && !swap_env_off;
        if (tagged)   // with --serve-slots 0/1 there is nothing to report: the path is 0.1.30's
            std::fprintf(stderr, "strata serve: slot swap %s (--serve-slots %d; STRATA_NO_SWAP=1 falls back to "
                                 "today's serial path)\n",
                         swaps_on ? "on" : "off", slots);
        // The per-slot conversation records, keyed by request id.  The engine thread owns every write.
        // A deque, not a vector: `swap_to` holds references across calls that may add a record, and a
        // vector reallocation would leave them dangling - the kind of bug that shows up as a wrong
        // token, not a crash.
        std::deque<strata::program::serve_swap::SlotConv> conv_slots;
        auto conv_of = [&](int64_t id) -> strata::program::serve_swap::SlotConv& {
            for (strata::program::serve_swap::SlotConv& c : conv_slots)
                if (c.id == id) return c;
            conv_slots.push_back(strata::program::serve_swap::SlotConv{});
            conv_slots.back().id = id;
            return conv_slots.back();
        };
        // Which slot the live session reflects right now (§5.3), and who owns the one position table.
        int64_t mounted_id = strata::program::serve_proto::kNoId;
        int64_t mrope_owner = strata::program::serve_proto::kNoId;
        int64_t mtp_prompt_len = 0;   // the mirror of mtp.set_prompt_len(), which has no getter
        int64_t swap_count = 0, swap_ms_total = 0;
        int64_t swap_bytes_out = 0, swap_bytes_in = 0;
        // S3.6: the slots the concurrent driver is holding, and what losing the session would cost
        // each of them.  The driver adds an entry when it creates a context and refreshes it after
        // every step from `ReqCtx::phase`; `drop_ctx` removes it.  Empty on the serial and
        // tagged-serial paths, which is the right answer there: one request at a time means the
        // outgoing slot is always `finished` and needs no save, exactly as 0.1.30 assumed.
        //
        // WHY THE PHASE MAPS THE WAY IT DOES.  `park_current` can only save a branch whose token list
        // matches the session's positional cells, and it asks that with `live_ok` - which
        // `prep_request` clears for the WHOLE life of a request ("until this request has finished, the
        // session is in between") and `finish_request` sets again.  So a mid-request conversation
        // cannot be saved at all today, whatever the budget.  That is what `not_saveable` means, and
        // it is why the honest answer for a mid-decode slot is "this hand-over waits", not "this
        // conversation is destroyed and its slot keeps stepping".
        std::deque<std::pair<int64_t, strata::program::serve_swap::Outgoing>> working_ids;
        auto working_of = [&](int64_t id) -> strata::program::serve_swap::Outgoing {
            for (const auto& w : working_ids) if (w.first == id) return w.second;
            return strata::program::serve_swap::Outgoing::finished;
        };
        auto working_set = [&](int64_t id, strata::program::serve_swap::Outgoing o) {
            for (auto& w : working_ids) if (w.first == id) { w.second = o; return; }
            working_ids.push_back({id, o});
        };
        auto working_drop = [&](int64_t id) {
            for (size_t i = 0; i < working_ids.size(); ++i)
                if (working_ids[i].first == id) { working_ids.erase(working_ids.begin() + (std::ptrdiff_t) i); return; }
        };
        // S3.6: what the last `swap_to` did to the two slots it moved between, so the driver can put
        // each request's `session_valid` back in step.  `swap_to` cannot reach `ReqCtx` (the driver's
        // `live` deque is declared after it), so the hand-over reports facts and the driver applies
        // them - which is also what makes the pair testable.
        //   swap_restored         - the incoming slot's parked image WAS restored: the session now
        //                           describes it.
        //   swap_invalidated_out  - the outgoing slot's branch was dropped without a save.  Only
        //                           reachable for a slot with no work left (the parking guard refuses
        //                           it otherwise), but the driver still marks the context so the step
        //                           gate cannot be fooled by a row that outlives its conversation.
        bool swap_restored = false, swap_invalidated_out = false;
        // ...and the one question the step gate asks: does the session hold the sequence of whoever
        // is mounted right now?  A hand-over that restored an image, or a `prep_request` that read or
        // zeroed the session for its own request, answers yes.  A hand-over that moved the session
        // without restoring anything answers no - and then the slot the session was taken FROM may
        // not step, which is the §5.3 invariant `serve_driver::step_gate` enforces.
        bool session_established = false;
        // The image the current request will mount, taken out of the cache but not yet restored.
        std::optional<strata::core::SavedConversation> mount_image;
        // What the last hand-over did, so the request body neither re-parks the outgoing branch nor
        // re-mounts an image the swap already restored.
        bool parked_by_swap = false, mounted_by_swap = false;
        // Set by the last `swap_to` that failed: the session was already written, so the request must
        // not retry the hand-over and must not run blind against it.
        bool swap_wrote_session = false;
        // S3.6: set by the last `swap_to` that refused BECAUSE the outgoing conversation could not be
        // saved.  The caller must NOT recover from that the way it recovers from a bad snapshot - the
        // retry-without-a-restore is exactly what destroys a running conversation.  The incoming
        // request is answered with an ERR naming both numbers instead.
        bool swap_park_refused = false;
        // ...and WHICH kind, so the driver can tell "never parkable here" (answer the incoming
        // request) from "not saveable yet" (the incoming request waits).  Same enum the guard
        // computed, so the two cannot drift apart.
        strata::program::serve_swap::ParkRefusal swap_park_kind =
             strata::program::serve_swap::ParkRefusal::none;
         // S3.7: the two numbers from the last refused hand-over, so a later `step_gate` ERR can name
         // what was actually true instead of blaming the budget by default.  0 = never measured.
         uint64_t swap_park_snapshot = 0, swap_park_budget = 0;
        // A hand-over that did not use the image gives it BACK to the cache.  Dropping it would throw
        // away a parked conversation the next request of that chat would otherwise re-read.
        auto return_mount_image = [&]() {
            if (mount_image == std::nullopt) return;
            // `put` may prune the least recently used conversation to make room, and it can refuse if
            // the budget closed in the meantime.  Both are the cache's normal policy; only the second
            // loses a conversation, so it is the one worth saying out loud.
            // S3.9: and it goes back with the CLAIM it came with.  `mount_image` was taken out of the
            // cache, not un-owned; re-parking it unclaimed would let the next prefix-matching request
            // steal the branch of the slot that is still waiting for it.
            const int64_t back_owner = mount_image->owner;
            if (!conversations.put(std::move(*mount_image), back_owner))
                std::fprintf(stderr, "strata serve: swap: the image taken for request %lld would not fit back "
                                     "in the cache - it is dropped and that conversation re-reads\n",
                             (long long) req_id);
            mount_image.reset();
        };
        // S3.9: a parked entry is CLAIMED by the request whose branch it is, for exactly as long as
        // that request is running.  The claim is what stops another request - one whose prompt happens
        // to share a prefix - from `take()`ing the whole entry and destroying a live conversation.
        // When the request goes away its branch must become reusable again, or stage 2's prefix
        // mechanism (the next request of the same chat resumes from it) would silently stop working and
        // the cache would fill with entries nobody may mount.  `mount_image` is the one entry that is
        // out of the cache at the time, so its claim has to be cleared by hand.
        auto release_conv_claims = [&](int64_t id) {
            conversations.release_owner(id);
            if (mount_image != std::nullopt && mount_image->owner == id)
                mount_image->owner = strata::core::kNoOwner;
        };
        // The claim a park should attach to this branch, or `kNoOwner`.  A slot that still owes a step
        // (`working_of` != finished) is one the driver will come back to, and its parked entry is the
        // only copy of its conversation - so it must not be handed to another request.  A slot with no
        // work left is finished: its branch is warm storage for the NEXT request of that chat, which
        // is stage 2's whole point, and claiming it then would freeze the cache.
        auto park_owner_for = [&](int64_t id) -> int64_t {
            if (!swaps_on) return strata::core::kNoOwner;
            return working_of(id) == strata::program::serve_swap::Outgoing::finished
                       ? strata::core::kNoOwner : id;
        };
        // A record exists only while its registry row does, or while it is the mounted one.  A finished
        // conversation's parked image is found by `ConversationCache::best()` on its token prefix -
        // that is stage 2's mechanism and it needs no record - so keeping rows for requests that are
        // gone would grow without bound (request ids are the client's to choose).
        auto prune_conv = [&]() {
            for (size_t i = 0; i < conv_slots.size();) {
                const int64_t id = conv_slots[i].id;
                bool keep = id == mounted_id;
                {
                    std::lock_guard<std::mutex> lk(slot_mu);
                    keep = keep || slots_reg.find(id) != nullptr;
                }
                if (keep) { ++i; continue; }
                conv_slots.erase(conv_slots.begin() + (std::ptrdiff_t) i);
            }
        };
        // Snapshot of the process-wide conversation variables for whichever slot is mounted.  Called at
        // the moments 0.1.30 already updates them, so the mirror can never disagree with the session.
        auto sync_conversation = [&]() {
            if (mounted_id == strata::program::serve_proto::kNoId) return;
            strata::program::serve_swap::SlotConv& c = conv_of(mounted_id);
            c.live = live;
            c.live_imgs = live_imgs;
            c.checks = checks;
            c.cvec_cached = cvec_cached;
            c.live_ok = live_ok;
            c.check_clock = check_clock;
            if (o.suffix_draft > 0) c.sfx_hist.assign(sfx_hist.begin(), sfx_hist.end());
            c.prompt_len = mtp_prompt_len;
        };
        // R7, and it has to happen BEFORE the request body rebuilds the table: `mrope_host` is ONE
        // buffer for the process, so the outgoing slot's positions must be copied out of it before a
        // new request overwrites them.  There is no way to recover them afterwards.
        auto sync_positions = [&]() {
            if (mounted_id == strata::program::serve_proto::kNoId) return;
            strata::program::serve_swap::SlotConv& c = conv_of(mounted_id);
            c.mrope_image = !mrope_identity;
            c.mrope = mrope_identity ? std::vector<int32_t>{} : mrope_host;
        };
        // Put a slot's non-snapshot state back into the process variables.  The snapshot itself is
        // restored by the mount hook; this is the part the snapshot does not carry.
        // R7, on its own: put a slot's positions back into the ONE host table.  Split out because the
        // request body's bail-out guard needs the positions and nothing else - it must not rebuild the
        // suffix drafter or re-dispatch sampling for a request that never started.
        auto apply_positions = [&](const strata::program::serve_swap::SlotConv& c) {
            mrope_identity = !c.mrope_image;
            if (c.mrope_image && !c.mrope.empty()) mrope_host = c.mrope;
            else if (!c.mrope_image && !mrope_host.empty()) {
                const int64_t cells = (int64_t) mrope_host.size() / 3;
                for (int64_t i = 0; i < cells; ++i)
                    mrope_host[(size_t) i * 3] = mrope_host[(size_t) i * 3 + 1] =
                        mrope_host[(size_t) i * 3 + 2] = (int32_t) i;
            }
        };
        // `restore_positions` is false when the caller is a REQUEST LINE, because the request body has
        // already rebuilt the table for this request (0.1.30's rule: a text request resets it to the
        // identity, an image request writes its own).  It is true for a scheduler-driven hand-over
        // (S3.1e), where nothing else can put the slot's positions back.
        auto apply_slot = [&](const strata::program::serve_swap::SlotConv& c, bool restore_positions) {
            // A request line rebuilds the drafter from its own prompt below (0.1.30's `sfx.reset()` +
            // append of every prompt id), so rebuilding it here too would be pure waste.  Only a
            // scheduler-driven hand-over - which has no prompt to rebuild from - needs this.
            if (restore_positions) {
                sfx_hist = c.sfx_hist;
                sfx.reset();
                if (!sfx_hist.empty()) sfx.append(sfx_hist.data(), sfx_hist.size());
            }
            // The checkpoint chain is SHARED by design (the pinned system-prompt root, PR #65), so its
            // LRU clock is process-wide: take the slot's clock only to move it FORWARD, never to rewind
            // it.  A restored chain carries its own `used` stamps, and rewinding the clock would make
            // a fresh checkpoint older than the ones already in the chain.
            if (c.check_clock > check_clock) check_clock = c.check_clock;
            // §3.5: sampling is applied at DISPATCH, never at parse time.  A request line dispatches
            // its own params a few lines below (0.1.30's `ver.set_sampling`), so only a
            // scheduler-driven hand-over has to put the slot's back here.
            if (c.smpl_set) {
                ver.set_sampling(c.smpl);
                mtp.set_draft_sampling(c.smpl);
                const int ph = std::min(c.penalty_last_n, kPenaltyWindowCap);
                ver.set_history(ph > 0 ? d_hist : nullptr, ph);
                drive.d.pcie_num = c.pcie_num;
            }
            if (!restore_positions) return;
            apply_positions(c);
            if (c.prompt_len > 0) mtp.set_prompt_len(c.prompt_len);
        };
        // THE HAND-OVER.  `incoming_id` becomes the mounted slot; its parked image, if any, is
        // restored into the session.  Returns false when the swap could not complete; `poisoned`
        // reports the one case where the session may not be used afterwards (a failed restore -
        // 0.1.30's rule, kept fatal).
        //
        // S3.6 - THE PARKING GUARD, and what losing the session would cost the slot it reflects.
        // `working_of(mounted_id)` is the driver's answer, derived from that slot's `ReqCtx::phase`;
        // the serial path has no entries at all, so it reads `finished` and the guard is inert there -
        // which is what keeps --serve-slots 0/1 behaving exactly as 0.1.30 does.
        //
        //   * finished    - the request is over.  It needs no save: the client has its answer and a
        //                   later request of the same chat finds the branch through
        //                   `ConversationCache::best()`, exactly as stage 2 always did.  The hand-over
        //                   always proceeds, budget or no budget; losing the in-session copy costs
        //                   reuse, not a conversation.
        //   * re_readable - mid-prompt.  Its whole prompt is still in `ReqCtx::ids`, so the hand-over
        //                   may proceed WITHOUT a save - but only because the driver then sends that
        //                   request back to token 0 (`reset_request_to_token0`).  Continuing the read
        //                   where it left off, against a session whose cells now hold somebody else's
        //                   prefix, is the same silent-garbage failure one segment later.
        //   * must_park   - mid-decode.  The tokens it generated are already on the wire, so its
        //                   prompt cannot be re-read without emitting them twice: its state MUST be
        //                   saved.  If it cannot be, the hand-over is REFUSED before the unmount hook
        //                   runs - the session is left exactly as it was, because the guard runs
        //                   before `validate` and `unmount` and both only read - and the caller
        //                   answers the INCOMING request instead of starving it or spinning on it.
        //
        // Destroying a running conversation and letting its slot keep stepping against the next one's
        // session is the bug this exists to make impossible.
        auto swap_to = [&](int64_t incoming_id, std::string& serr, bool& poisoned,
                           bool restore_positions = true) -> bool {
            poisoned = false;
            swap_wrote_session = false;
            swap_park_refused = false;
            swap_park_kind = strata::program::serve_swap::ParkRefusal::none;
            swap_invalidated_out = false;
            swap_restored = false;
            // The tested predicate (`serve_swap::swap_needed`), so the decision the CPU test pins is
            // the one production makes.  Staying on the mounted slot is the default and the cheap
            // answer: one swap is a full save+restore of a 237 MB-2.25 GB image (risk R10).
            if (!swaps_on || !strata::program::serve_swap::swap_needed(mounted_id, incoming_id)) return true;
            // `mounted_id == kNoId` means nothing is mounted yet (the first request of the process):
            // there is no outgoing record, and creating one keyed kNoId would strand it forever.
            strata::program::serve_swap::SlotConv& in = conv_of(incoming_id);
            const bool have_out = mounted_id != strata::program::serve_proto::kNoId;
            strata::program::serve_swap::SlotConv& out = have_out ? conv_of(mounted_id) : in;
            // S3.6: what losing the session would cost the slot the session reflects.  `finished` on
            // the serial path (nothing is in `working_ids`), so the guard is inert there and 0.1.30's
            // behaviour - park if you can, re-read if you cannot - is untouched.
            const strata::program::serve_swap::Outgoing out_state =
                have_out ? working_of(mounted_id) : strata::program::serve_swap::Outgoing::finished;
            // R7, as an acquired resource and not an assumption: the one position table belongs to the
            // mounted slot.  A hand-over that would leave a non-identity holder mounted is refused.
            const bool saving_out = have_out &&
                                    strata::program::serve_swap::save_for(out, conversations.enabled()) ==
                                        strata::program::serve_swap::Save::park;
            if (!strata::program::serve_swap::mrope_exclusive_ok(out, incoming_id, saving_out)) {
                serr = "the image position table is held by slot " + std::to_string((long long) mounted_id);
                return false;
            }
            // S3.6 — THE PARKING GUARD, and it runs BEFORE anything can destroy the outgoing branch.
            //
            // What used to happen: the unmount hook let `park_current` fail, `invalidate_unparked`
            // cleared the outgoing branch, the hand-over reported SUCCESS, and the outgoing slot's
            // conversation existed nowhere.  Its request was still running, so the next pick stepped it
            // against a session holding ANOTHER conversation's K/V and positions: garbage tokens,
            // usually an immediate EOS, and no error anywhere.  The log promised "will be re-read from
            // token 0" and nothing ever kept that promise.
            //
            // The rule now, in the order the two facts arrive:
            //
            //   * a slot with NO work left needs no save.  Its client already has the answer and a
            //     later request of the same chat finds the branch through `ConversationCache::best()`,
            //     exactly as stage 2 always did.  Dropping its in-session image costs reuse, not a
            //     conversation, so the hand-over proceeds and says so on one line.
            //   * a slot WITH work left may only be swapped out if its state can actually be saved.
            //     "Can be saved" has two halves, and BOTH are checked here rather than discovered by
            //     a failed save: the session must hold a whole branch the cache could take
            //     (`save_for == Save::park`, which is `park_current`'s own first guard), and the
            //     snapshot must fit the parking budget (`budget_of`/`budget_has_room`, which mirror
            //     `ConversationCache::make_room`).  A refusal leaves the session exactly as it was -
            //     the guard runs before `validate` and `unmount`, and both only read.
            //
            // The two refusals are different and the driver treats them differently:
            //   * `budget_too_small` - never parkable here.  The incoming request is ANSWERED with an
            //     ERR naming both numbers, not starved and not spun on.
            //   * `not_saveable`     - the session is mid-read, so there is no whole branch to take
            //     yet.  That clears itself when the running request ends, so the incoming request
            //     WAITS.  ERRing it would punish a client for a race that resolves on its own;
            //     swapping anyway is what used to destroy a running conversation.
            strata::program::serve_swap::ParkCheck parkchk;
            const bool park_guard_on = have_out &&
                strata::program::serve_swap::save_is_mandatory(out_state);
            // A slot whose conversation was NOT parked on its way out has no image to come back to.
            // That is not an error and it must not refuse the hand-over: the request simply re-reads
            // from token 0, which is the request body's job.  What must never happen is a restore for
            // such a slot, so the image is dropped here rather than trusted.
            if (!strata::program::serve_swap::can_mount(in) && mount_image != std::nullopt) {
                std::fprintf(stderr, "strata serve: swap: slot %lld is not resumable - its parked image was "
                                     "pruned or never stored; re-reading from token 0\n",
                             (long long) incoming_id);
                return_mount_image();
            }
            strata::program::serve_swap::Plan plan;
            plan.save = saving_out;
            plan.mount = mount_image != std::nullopt;
            plan.draft_kv = plan.mount;
            plan.adopt = true;
            plan.device_state = true;
            plan.park_guard = park_guard_on;
            strata::program::serve_swap::Hooks h;
            // What this hand-over actually moved, for the report below.  The hooks fill them.
            int64_t rep_saved = 0, rep_restored = 0;
            // 1. drain the in-flight expert-cache swaps (risk R9).  A swap resets the KV page tables,
            //    and a half-applied residency table is the silent failure expert_cache.hpp:135-138 warns
            //    about.  `apply_pending(true)` also re-uploads `host_res` if it moved.
            h.drain_residency = [&](std::string&) { apply_pending(true); return true; };
            // 2. return the prompt loan (risk R8).  The loan is the tail of ONE cache; a snapshot taken
            //    while it is lent describes a cache that does not exist.
            h.return_loan = [&](std::string& e) { return refill(e); };
            // 3. S3.6: can the outgoing conversation be saved at all?  Asked here, after the loan is
            //    back and before the validate/save/restore, because everything from `unmount` on can
            //    destroy it.  A refusal costs nothing: the session has only been read.
            h.park_guard = [&](std::string& e) -> bool {
                const size_t held = mount_image != std::nullopt ? mount_image->bytes() : 0;
                // `saveable` is `park_current`'s own first guard, so the two cannot disagree: the
                // cache takes a branch whose token list matches the session's positional cells
                // (`live_ok && !live.empty()`), and nothing else.  The driver publishes that for a
                // mid-decode slot before it asks for a hand-over (`publish_decode_branch`), which is
                // what makes pre-empting a decoder parkable at all - without it `park_current` has
                // always refused a mid-request slot, budget or no budget, and every pre-emption has
                // been destroying a live conversation since S3.1d landed.
                // Read the SERVE-scope truth, not the slot record's mirror.  `park_current` asks
                // `live_ok && !live.empty()` on the serve-scope variables, and `SlotConv::live_ok` is
                // only refreshed by `sync_conversation()` at a few points, so the mirror can be stale
                // true for a slot whose request has since started reading.  A guard that trusted it
                // would let a hand-over through and then have `park_current` refuse it - which is the
                // exact disagreement this whole check exists to remove.
                const bool saveable = conversations.enabled() && live_ok && !live.empty();
                size_t est = 0;
                std::string berr;
                const strata::core::ConversationView view{live, live_imgs, checks, cvec_cached};
                if (!strata::core::conversation_snapshot_bytes(view, ss, conv_stages, g,
                                                               mtp.kv_state(), est, berr)) {
                    // The snapshot cannot even be sized (a checkpoint whose buffers do not match the
                    // session, a token count past the session's cells).  `park_current` would refuse it
                    // for the same reason, so treat that as "does not fit", not as "no problem".
                    std::fprintf(stderr, "strata serve: park: slot %lld snapshot cannot be sized (%s)\n",
                                 (long long) out.id, berr.c_str());
                    est = std::numeric_limits<size_t>::max() / 4;
                }
                parkchk = strata::program::serve_swap::park_fits(
                    strata::program::serve_swap::budget_of(conversations), est, held, out_state, saveable);
                // S3.7: keep the two numbers at serve scope so a later `step_gate` ERR can quote them.
                swap_park_snapshot = parkchk.snapshot;
                swap_park_budget = parkchk.budget;
                if (parkchk.ok()) return true;
                swap_park_refused = true;
                swap_park_kind = parkchk.refusal;
                // ONE line, unconditionally: the two numbers the owner asked for, at the moment the
                // decision is made rather than 97 seconds later when a request ends early.
                std::fprintf(stderr, "%s\n",
                             strata::program::serve_swap::park_refusal_line(out.id, incoming_id, parkchk).c_str());
                std::fflush(stderr);
                e = strata::program::serve_swap::park_refusal_line(out.id, incoming_id, parkchk) +
                    ".  A slot in the middle of a request cannot be swapped out unless its state can "
                    "be saved: raise --conversation-cache-mib, lower --max-context, or run this box "
                    "with --serve-slots 0/1.";
                serr = e;
                return false;
            };
            // 3. validate BEFORE anything is written - this is what makes "a failed validation leaves
            //    the outgoing state intact" a property of the order rather than a hope.
            h.validate = [&](std::string& e) {
                if (strata::core::conversation_snapshot_validate(*mount_image, ss, conv_stages, g,
                                                                 mtp.kv_state(), e))
                    return true;
                std::fprintf(stderr, "strata serve: swap: discard invalid snapshot (%s)\n", e.c_str());
                return false;
            };
            // 4. unmount: the snapshot save.  It only READS the session, so a failure here leaves
            //    everything usable and the outgoing slot stays mounted.
            h.unmount = [&](std::string& e) {
                // S3.9: the branch being saved here is the OUTGOING slot's, and that slot is still
                // running - it is about to wait for its session back.  Claim the entry for it, or the
                // next request whose prompt shares a prefix can mount (and thereby remove) it.
                const auto saved = park_current(mount_image ? mount_image->bytes() : 0, true,
                                                park_owner_for(mounted_id));
                if (saved == strata::program::serve_swap::Saved::failed) { e = err; err.clear(); return false; }
                parked_by_swap = true;   // the request body must not park this branch again
                if (saved == strata::program::serve_swap::Saved::stored) {
                    rep_saved = (int64_t) park_bytes;
                    out.parked_bytes = (int64_t) park_bytes;
                    swap_bytes_out += (int64_t) park_bytes;
                    out.resumable = true;
                    std::fprintf(stderr, "strata serve: park: slot %lld parked %zu tokens / %lld MiB; "
                                         "parked=%zu entries, %zu MiB of a %zu MiB budget\n",
                                 (long long) out.id, out.live.size(), (long long) (park_bytes >> 20),
                                 conversations.size(), conversations.bytes() >> 20,
                                 conversations.budget() >> 20);
                } else if (strata::program::serve_swap::save_is_mandatory(out_state)) {
                    // S3.6 THE SECOND HALF OF THE GUARD.  The budget pre-check passed, but the save
                    // still did not store: physical-RAM admission refused it, `put` refused it, or the
                    // capture was skipped for a reason the estimate could not see.  A slot that still
                    // owes a step may NOT be destroyed by that, so the hand-over fails HERE -
                    // `unmount` only reads the session, so `serve_swap::run` reports the session intact
                    // and the caller answers the incoming request with an ERR instead of starving it.
                    serr = "slot " + std::to_string((long long) mounted_id) + " could not be parked (" +
                           std::to_string(out.live.size()) + " tokens, " +
                           std::to_string((long long) (parkchk.snapshot >> 20)) + " MiB snapshot vs a " +
                           std::to_string((long long) (parkchk.budget >> 20)) +
                           " MiB parking budget) and it still has work to do";
                    std::fprintf(stderr, "strata serve: park: slot %lld NOT parked although the budget looked "
                                         "enough (%zu tokens, %lld MiB snapshot vs a %lld MiB budget) - "
                                         "hand-over aborted, the session is untouched\n",
                                 (long long) out.id, out.live.size(),
                                 (long long) (parkchk.snapshot >> 20),
                                 (long long) (parkchk.budget >> 20));
                    swap_park_refused = true;   // same handling as the guard's refusal: never retry blind
                    // The budget looked enough but the save did not store - physical-RAM admission,
                    // or `put` refusing.  That clears itself when RAM frees, so it is the WAIT kind,
                    // not the NEVER kind.
                    swap_park_kind = strata::program::serve_swap::ParkRefusal::not_saveable;
                    return false;
                } else {
                    // S3.6: the outgoing slot's branch is dropped without a save, and that is allowed
                    // for exactly TWO reasons, both decided by `out_state` before anything was
                    // destroyed:
                    //   * `finished`    - the request is over, so no step can ever run against a
                    //     stale session.  Dropping the in-session copy costs reuse, not a
                    //     conversation: a later request of the same chat finds the branch through
                    //     `ConversationCache::best()`, exactly as stage 2 always did.
                    //   * `re_readable` - the request is mid-prompt and its whole prompt is still in
                    //     `ReqCtx::ids`.  `serve_driver::step_gate` sees `resumable == false` and
                    //     sends it back to token 0 (`reset_request_to_token0`), which is the promise
                    //     this log line has always made and never kept.
                    // A slot in `decode` reaching this line is the bug; the guard above refuses it.
                    strata::program::serve_swap::invalidate_unparked(out);
                    swap_invalidated_out = true;
                    const bool over = out_state == strata::program::serve_swap::Outgoing::finished;
                    std::fprintf(stderr, "strata serve: swap: slot %lld NOT parked (%zu tokens, %lld MiB snapshot "
                                         "vs a %lld MiB budget) - %s\n",
                                 (long long) out.id, out.live.size(),
                                 (long long) (parkchk.snapshot >> 20),
                                 (long long) (parkchk.budget >> 20),
                                 over ? "its request is over, so nothing will step against it and a later "
                                        "request of this chat re-reads from the cache"
                                      : "it is mid-prompt, so it will be RE-READ from token 0");
                    std::fflush(stderr);
                }
                return true;
            };
            // 5. mount: the snapshot restore.  Pre-validated, so a failure here is a transfer failure
            //    and the session is finished for this process.
            h.mount = [&](std::string& e) {
                const auto t0 = Clock::now();
                const auto res = strata::core::conversation_snapshot_restore(*mount_image, ss, conv_stages, g,
                                                                            mtp.kv_state(), e);
                if (res != strata::core::ConversationRestore::restored) { poisoned = true; return false; }
                rep_restored = (int64_t) mount_image->bytes();
                swap_bytes_in += rep_restored;
                live = std::move(mount_image->live.ids);
                live_imgs = std::move(mount_image->live.imgs);
                checks = std::move(mount_image->checkpoints);
                cvec_cached = mount_image->cvec;
                live_ok = !live.empty();
                if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr)
                    conversations.retain(std::move(mount_image->kv), int64_t(live.size()));
                mount_image.reset();
                mounted_by_swap = true;   // the request body must not re-run the parked-prefix search
                swap_restored = true;    // S3.6: the session now describes the incoming slot
                std::fprintf(stderr, "strata serve: swap: mounted slot %lld, %zu tokens in %.1f ms; "
                                     "parked=%zu bytes=%zu\n",
                             (long long) incoming_id, live.size(),
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                             conversations.size(), conversations.bytes());
                return true;
            };
            // 6. the drafter's ring may hold cells past the resume point from a longer turn.
            h.draft_kv = [&](std::string&) { if (!live.empty()) mtp.kv_restore((int64_t) live.size()); return true; };
            // 7. the per-slot state the snapshot does not carry.
            h.adopt = [&](std::string&) {
                // No restore happened, so the session's POSITIONAL cells still hold the OUTGOING
                // conversation's tokens.  Nothing the incoming slot inherits from them may be used:
                // a checkpoint is only valid while the cells below it still hold its tokens.  Clear
                // the branch and the chain, and the request body reads from token 0 - the honest
                // answer, and the one that cannot produce plausible garbage.
                if (!plan.mount) {
                    live.clear();
                    live_imgs.clear();
                    checks.clear();
                    live_ok = false;
                }
                in.live = live; in.live_imgs = live_imgs; in.checks = checks;
                in.cvec_cached = cvec_cached; in.live_ok = live_ok;
                in.resumable = true;   // whatever it was, it now describes the session again
                apply_slot(in, restore_positions);
                return true;
            };
            // 8. the per-slot DEVICE state the session depends on: the one position table (R7) and the
            //    control-vector mode.  Sampling is re-dispatched by the request body at dispatch time
            //    (§3.5), never here.
            h.device_state = [&](std::string&) {
                if (strata::kernels::cvec().loaded()) strata::kernels::cvec_set_enabled(in.cvec_cached);
                // The table only exists with --vision; without it every slot holds the identity and
                // there is nothing to hand over - but ownership still moves, so the predicate stays
                // honest and a later --vision session cannot inherit a stale owner.
                if (strata::program::serve_swap::mrope_upload_needed(mrope_owner, incoming_id)) {
                    if (!mrope_host.empty() && !upload_mrope_table()) { serr = "the position table upload failed"; return false; }
                    mrope_owner = incoming_id;
                }
                return true;
            };
            const auto tswap = Clock::now();
            const strata::program::serve_swap::Report rep = strata::program::serve_swap::run(plan, h, serr);
            const int64_t ms = (int64_t) std::chrono::duration<double, std::milli>(Clock::now() - tswap).count();
            // Report every swap on stderr with the bytes moved and the milliseconds, and count them, so
            // the owner can price OQ3 (the real swap cost per token count) at a restart.
            std::fprintf(stderr, "strata serve: swap %lld: slot %lld -> %lld %s in %lld ms; saved %lld B, "
                                 "restored %lld B, parked=%zu bytes=%zu (total %lld swaps, %lld ms, "
                                 "%lld B out / %lld B in)\n",
                         (long long) (swap_count + 1), (long long) mounted_id, (long long) incoming_id,
                         rep.ok ? "ok" : strata::program::serve_swap::fault_name(rep.fault),
                         (long long) ms, (long long) rep_saved, (long long) rep_restored,
                         conversations.size(), conversations.bytes(),
                         (long long) (swap_count + 1), (long long) (swap_ms_total + ms),
                         (long long) swap_bytes_out, (long long) swap_bytes_in);
            if (trace) {
                std::string steps;
                for (strata::program::serve_swap::Step st : rep.ran) steps += std::string(" ") +
                                                                            strata::program::serve_swap::step_name(st);
                std::fprintf(stderr, "strata trace: swap steps:%s\n", steps.c_str());
            }
            swap_count++;
            swap_ms_total += ms;
            {
                std::lock_guard<std::mutex> lk(slot_mu);
                slots_reg.note_swap(ms);
                // /slots and the SLOT line report what each slot costs in the cache (§6.2's
                // `parked_bytes`), so the owner can see a swap's price without grepping stderr.
                if (strata::program::slot::Slot* sl = slots_reg.find(incoming_id))
                    sl->parked_bytes = in.parked_bytes;
                if (have_out)
                    if (strata::program::slot::Slot* sl = slots_reg.find(mounted_id))
                        sl->parked_bytes = out.parked_bytes;
            }
            if (!rep.ok) {
                // A failure AT OR AFTER the restore means the session already holds the incoming
                // conversation, whether or not the hand-over finished.  Say so, and say so to the
                // caller: `mounted_id` must describe the session or the next save would write the
                // wrong branch out of it, and a request may NOT retry a hand-over that already
                // restored - the second attempt would save the incoming branch as if it were the
                // outgoing one.  (A failure BEFORE it - drain, validation, save - left the session
                // alone, and the caller may retry without an image.)
                if (!rep.session_intact) {
                    swap_wrote_session = true;
                    mounted_id = incoming_id;
                    sync_conversation();
                }
                return false;
            }
            mounted_id = incoming_id;
            // S3.6: the session describes the incoming slot only if its own state was put into it.
            // A hand-over with no restore cleared `live`/`checks` (adopt), so the answer is no.
            session_established = plan.mount;
            sync_conversation();
            return true;
        };
        // ==================== S3.1e-1: the resumable request ====================
        //
        // Stage 3's scheduler (S3.1e-2) cannot run a request as one straight-line block: with two
        // conversations in one process it must give each of them ONE step at a time and swap the
        // session between them.  So the request body below is cut into the named steps
        // (`prep_request`, `run_prefill_step`, `finish_prefill`, `run_decode_step`,
        // `finish_request`), and the serial loop calls them in exactly the order the old straight-line
        // code ran.  This is a pure refactor: same order, same captures, same error paths, same
        // stdout bytes.
        //
        // WHY A STRUCT AND NOT A CLASS: `ReqCtx` is only the per-request state the old body kept in
        // its own locals.  It is constructed INSIDE `while (next_line(line))`, so its lifetime and its
        // construction/destruction order are exactly those locals' - nothing about when a conversation
        // image, a slot row or a CUDA buffer exists changes.  The steps are lambdas in the serve scope
        // because that is where every capture already lives (`ver`, `mtp`, `sp`, `conversations`,
        // `slots_reg`, ...); a class would have to be handed all of them.
        //
        // The moved code keeps its ORIGINAL local names as references into `ReqCtx`, so each step's
        // body is the old text verbatim apart from the alias declarations and the exit statements.
        enum class Prep { ok, rejected, fatal };   // rejected: the ERR line is out, take the next line
                                                   // fatal:    the ERR line is out, the engine exits 1
        enum class Step { progressed, finished, cancelled, needs_swap, error, fatal_exit };
        //   progressed  the step did work and there is more of the same step to do
        //   finished    the step is done; move on to the next phase
        //   cancelled   the request was STOPped; fall through to the phase's tail, not an error
        //   needs_swap  RESERVED for S3.1e-2: this step wants the session handed to another slot.
        //               The serial path never returns it and nothing handles it yet.
        //   error       the ERR line is already printed; the caller must `return 1` exactly as today
        //   fatal_exit  #224: a CUDA fault, already flushed; the caller must `std::_Exit(1)`

        // S3.1e-1: the decode-timing snapshot moved out of the request body with the decode step, so
        // `ReqCtx` can hold one.  Same captures (`ver`, `drive`), same static flag, same fields.
        static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
        struct DecSnap {
            double wait, pool, host, plan, actq, jobs, run;
            int64_t misses, entries, hits, pcie;
        };
        auto dec_snap = [&]() {
            return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,
                           drive.d.ms_run, drive.d.multi_misses, drive.d.multi_entries, drive.d.cache_hits,
                           drive.d.pcie_experts};
        };
        struct ReqCtx {
            // ---- S3.1e-2: what makes the context OUTLIVE a step -------------------------------
            // The serial path hands this struct one line and runs it to completion inside the same
            // iteration, so it never needs to remember the line.  The concurrent driver admits a
            // request in one iteration and steps it in later ones, so the line has to live here:
            // `prep_request` is called exactly once per request, with the line it was admitted from.
            std::string line;
            bool has_line = false;      // `line` is set and has not been consumed yet
            // Where the resumable body sits.  This is the driver's phase machine
            // (serve_driver::Phase); the serial path never reads it, it just calls the steps in
            // 0.1.30's order.
            strata::program::serve_driver::Phase phase = strata::program::serve_driver::Phase::queued;
            // A failure the driver could not act on INSIDE a step (returning from a step lambda
            // would skip the slot's unwind and leak its active-slot permit).  The driver acts on it
            // as soon as it is back in the loop: 0 = none, 1 = the ERR line is out and the engine
            // exits 1 (0.1.30's `return 1`), 2 = #224, a CUDA fault already flushed: `_Exit(1)`.
            int fail = 0;
            // Has this request reached its end (DONE/ERR out, its image handed to the cache)?  The
            // per-step `SlotGuard` releases the row only when this is set, which is what makes a
            // guard armed per STEP instead of per REQUEST possible at all.
            bool finished = false;
            // Does THIS request currently hold the one prompt loan (risk R8)?  Set when a batched
            // segment lends, cleared when `finish_prefill` refills.  The driver keeps
            // `serve_driver::Loan` in step with it, and this is what makes a slot that dies mid-read
            // give the loan back rather than strand it.
            bool loan_held = false;
            // The RAM this slot was admitted against (the driver's admission estimate).
            int64_t image_bytes = 0;
            // S3.6: does the ONE live session still hold THIS request's sequence?  Set true by
            // `prep_request` (which either restored an image, resumed a checkpoint or zeroed the
            // session) and by a hand-over that restored this slot's image; cleared by any hand-over
            // that moved the session elsewhere without restoring this slot's state.  It is the one
            // fact `serve_driver::step_gate` needs, and it cannot come from `SlotConv`: `live_ok` is
            // false for every mid-flight request by design, and `resumable` is reset by the adopt
            // hook.  Only the request knows whether the session is still its own.
            // S3.6 (§5.3): does the ONE live session still hold THIS request's sequence?
            //   true  - `prep_request` established it (its own read, or a hand-over restored this
            //           slot's parked image into the session);
            //   false - a hand-over moved the session to another slot and did not restore this one's
            //           image, so the session's positional cells and running state describe somebody
            //           else's conversation.
            // It cannot be derived from `SlotConv`: `live_ok` is false for the whole life of a request
            // by design ("until this request has finished, the session is in between"), and
            // `resumable` is set back to true by the adopt hook for whatever was just mounted.  Only
            // the request knows.  `serve_driver::step_gate` reads it before every step.
            bool session_valid = false;
            // S3.8: how many times THIS request has been sent back to token 0.  A re-read is an honest
            // recovery when a branch was lost once; unbounded it is the two-prefill livelock the owner
            // hit (slot 3 reset 101 times, neither prefill ever finishing), and it looks like progress
            // to the watchdog because every pass really does read tokens.  So the count is bounded and
            // the request is ended with a named ERR instead of restarting forever.
            int rereads = 0;
            // S3.6: this request's own picture keys.  The serve-scope `req_imgs` names whichever
            // request was prepped LAST, so a driver that publishes another slot's branch must not
            // read it - it would attach a different conversation's images to this one's token list,
            // and the ConversationCache compares them when it looks a branch up.
            std::vector<ImgKey> own_imgs;
            // ---- phase 1: the parsed request line
            strata::program::serve_proto::Request rq;
            bool geni = false;
            long long max_new = 0;
            float req_temperature = 0.0f, req_top_p = 1.0f, req_min_p = 0.0f;
            float req_penalty_repeat = 1.0f, req_penalty_freq = 0.0f, req_penalty_present = 0.0f;
            int req_top_k = 20, req_penalty_last_n = 0, req_cvec = 1;
            unsigned long long req_seed = 0;
            double req_pcie_frac = 0.0, req_spec_min_p = 0.0;
            std::string emb_path;
            std::vector<int64_t> ids;
            int64_t id = strata::program::serve_proto::kNoId;   // this request's id (== `req_id` while it runs)
            int64_t n = 0;
            bool slot_open = false;        // the registry row exists -> SlotGuard must finish it
            bool mrope_touched = false;    // this request rewrote the one position table
            // per-request metric baselines (phase 5 prints the deltas)
            std::array<int64_t, 3> remote_before{}, launches_before{};
            std::array<uint64_t, 3> compact_before{}, full_before{};
            std::array<double, 3> begin_before{}, wait_before{};
            // ---- phase 2: the resume point, the mount, the sampling dispatch
            Clock::time_point r0{};
            std::array<int64_t, 6> loan_bill{};
            int64_t res_uploads0 = 0, res_skips0 = 0;
            bool want_cvec = true;
            int64_t resume = 0;
            bool from_live = false;
            int64_t reread_to = -1;
            int64_t read_from = 0;
            strata::kernels::SamplerParams req_sp;
            int hist_n = 0;
            // ---- phase 3: the prompt segments and how far the read has got
            int64_t turn_at = -1, root_at = -1;
            std::array<int64_t, 4> seg{{-1, -1, -1, -1}};   // the four segment ends 0.1.30 iterated
            size_t seg_i = 0;
            int64_t at = 0;
            bool cancelled = false;
            double prompt_ms = 0.0;
            // S3.1e-2: the prompt-progress state the PP line and the mid-prompt checkpoints read.  It
            // lives at serve scope because 0.1.30's body wrote it there; the driver mirrors it here so
            // it can put the RIGHT request's values back before each prefill step.
            int64_t pp_total = 0, pp_from = 0, pp_next_check = 0;
            Clock::time_point pp_t0{};
            // ---- phase 4: the decode loop's state (`finish_prefill` sets it up, `run_decode_step`
            // advances it one verify window at a time).  The vectors are sized once per request, never
            // per token, so the decode hot path allocates nothing new.
            int64_t p = 0;
            int32_t x = 0;
            std::vector<int32_t> drafts, window, outv, sbuf, consumed;
            std::vector<float> dprob;
            bool first_window = true;
            int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
            int64_t draft_offered = 0, draft_accepted = 0;
            const char* finish = "length";
            Clock::time_point d0{};
            DecSnap ds0{};
            double dt_run = 0, dt_commit = 0, dt_draft = 0;
            int64_t dec_windows = 0, dec_T = 0;
            int64_t decode_hits0 = 0, decode_look0 = 0;
            double decode_ms = 0.0;
        };
        // The two iteration guards.  They stay INSTANTIATED in the loop body: their destructors must
        // run at the end of the whole iteration, exactly as the old locals' did.  Only their definitions
        // move out, so a body-local instance can be armed from inside `prep_request` through `ReqCtx`.
        // The values they read at destruction (`slot_open`, `mrope_touched`) are never written after
        // setup, so reading them from the context is the same answer as the old constructor argument.
        struct SlotGuard {
            ReqCtx* ctx = nullptr;
            std::function<void(int64_t)> finish;
            ~SlotGuard() { if (ctx && ctx->slot_open && finish) finish(ctx->id); }
        };
        struct MropeScope {
            std::function<bool()> skip;     // this request IS the mounted slot: nothing to undo
            std::function<void()> restore;
            ~MropeScope() { if (skip && skip()) return; if (restore) restore(); }
        };
        // ---- S3.6: publish a running request's branch so a hand-over CAN save it ----------------
        //
        // THE REAL ROOT CAUSE, and it is not the budget.  `park_current`'s first guard is the
        // serve-scope `live_ok`, and `live`/`live_ok` describe the branch the SESSION holds as a
        // resumable conversation.  `prep_request` clears `live_ok` for the WHOLE life of a request
        // ("until this request has finished, the session is in between") and `finish_request` is the
        // only thing that puts the request's own branch into `live` (`live.swap(consumed)`).  So for
        // every request in `prefill` or `decode`, `park_current` returned `skipped` - at ANY budget -
        // S3.1d's unmount path called `invalidate_unparked`, and the hand-over SUCCEEDED anyway.
        // That is why every swap in the owner's log says `saved 0 B`, why the parking budget was
        // never consulted for the case that mattered, and why the resumed slot decoded against a
        // foreign session and ended early in silence.
        //
        // The fix is to publish the running request's own branch before asking for a hand-over, so
        // the save has something to take and the budget question becomes the real question.
        // `ReqCtx::consumed` IS that list: `finish_prefill` seeds it with the prompt ids [0, n-1) and
        // every committed window token joins it - exactly what `finish_request` later swaps into
        // `live`.  COPY it, never swap it: the decode loop keeps appending and `finish_request` still
        // owns it.
        //
        // These two live at SERVE scope on purpose: inside the driver block the name `live` is the
        // driver's own `std::deque<ReqCtx>`, and a helper defined there would silently write the
        // wrong container.  `own_imgs` rather than the serve-scope `req_imgs` for the same reason -
        // `req_imgs` names whichever request was prepped LAST, and attaching another conversation's
        // picture keys to this token list would corrupt the cache's prefix comparison.
        auto publish_decode_branch = [&](ReqCtx& R) {
            if (R.phase != strata::program::serve_driver::Phase::decode) return;
            if (R.consumed.empty()) return;   // nothing read yet: nothing to publish
            live = R.consumed;
            live_imgs = imgs_below(R.own_imgs, (int64_t) live.size());
            live_ok = o.prompt_cache > 0;
            sync_conversation();              // the slot's mirror must match what we are about to save
        };
        // S3.8 — THE SAME PUBLISH FOR A MID-PROMPT READ, and this one is what makes two concurrent
        // prefills finish at all.
        //
        // THE LIVELOCK.  A slot in `prefill` had no publish at all, so `live_ok` was still false and
        // `park_current` refused its save at any budget: every swap out reported `saved 0 B`,
        // `invalidate_unparked` cleared the branch, and the next step sent the request back to token 0.
        // The owner's log has slot 3 `reset to token 0: 81 prompt tokens will be read again` 101 times
        // while slot 2 parked and resumed normally - two prefills erasing each other's progress, so
        // neither ever finished.  `outgoing_for` used to call that slot `re_readable`, which licenses
        // exactly this: re-reading is survivable ONCE, and on every swap it is a spin.
        //
        // What the session actually holds for such a slot is the prompt read so far: `run_prefill_step`
        // reads [at, to) and stops at `n - 1`, because the last prompt token always starts the first
        // verify window.  So the resumable branch is `ids[0, at)` - which is precisely what
        // `finish_request` would swap in at the end of the read (`consumed` = ids[0, n-1)).
        //
        // `checks` is deliberately NOT touched.  The adopt hook maintains the invariant that serve-scope
        // `checks` belongs to the mounted slot (a restoring hand-over loads the image's chain, a
        // non-restoring one clears it, `prep_request` rebuilds it), and every checkpoint of this read was
        // taken at a position <= `at`, so they are all prefixes of the branch being published here.
        auto publish_prefill_branch = [&](ReqCtx& R) {
            if (R.phase != strata::program::serve_driver::Phase::prefill &&
                R.phase != strata::program::serve_driver::Phase::prefill_end) return;
            if (R.at <= 0) return;   // nothing read yet: the prompt is entirely in hand, re-read is cheap
            // A cancelled or failed read leaves the session somewhere BETWEEN two chunks, so its
            // running state is ahead of `at` and there is nothing clean to continue from - the same
            // reason `finish_request` guards `live.swap(consumed)` with `if (!cancelled)`.  Such a
            // request is unwinding anyway; it will be finished on a later step, and a hand-over away
            // from it is refused as `not_saveable` (which the incoming slot waits out) rather than
            // parking a branch the session does not describe.
            if (R.cancelled) return;
            // `R.at` is a segment end, and every segment end is at most `n - 1`, so this stays inside
            // the prompt.  The cast is 0.1.30's own (`finish_prefill` seeds `consumed` the same way).
            const int64_t upto = std::min<int64_t>(R.at, (int64_t) R.ids.size());
            live.assign(R.ids.begin(), R.ids.begin() + (size_t) upto);
            live_imgs = imgs_below(R.own_imgs, (int64_t) live.size());
            live_ok = o.prompt_cache > 0;
            sync_conversation();
        };
        // Publish whichever branch the mounted slot owes the cache.  One call site, so a hand-over can
        // never see a running slot with nothing to save - which is what made every pre-emption of a
        // prefill both un-parkable and (wrongly) allowed.
        auto publish_working_branch = [&](ReqCtx& R) {
            publish_decode_branch(R);
            publish_prefill_branch(R);
        };
        // On the way back in, `consumed` must name the sequence the restored session holds.  Without
        // this pair a resumed decode would count sampling penalties over the wrong tokens and
        // `finish_request` would park the wrong branch - a wrong-but-plausible answer, which is the
        // class of bug this whole file exists to prevent.
        auto restore_published_branch = [&](ReqCtx& R) {
            if (R.phase == strata::program::serve_driver::Phase::decode) {
                if (!live_ok || live.empty()) return;
                R.consumed.assign(live.begin(), live.end());
                return;
            }
            // S3.8: the prefill half of the pair.  A restored session holds `ids[0, at)` of this
            // request, so the read must continue from there and NOT from wherever the cursor was
            // left.  `seek_mount_index` mounts the EXACT branch, so `live.size() == R.at` by
            // construction; re-deriving it here is what makes that an invariant of the code rather
            // than an accident of the lookup, and re-planning the segments keeps `seg_i` in step with
            // the cursor (`run_prefill_step` skips any segment end at or below `at`).
            if (R.phase != strata::program::serve_driver::Phase::prefill &&
                R.phase != strata::program::serve_driver::Phase::prefill_end) return;
            if (!live_ok || live.empty()) return;
            R.at = (int64_t) live.size();
            R.seg_i = 0;
            while (R.seg_i < R.seg.size() && R.seg[R.seg_i] <= R.at) ++R.seg_i;
            // `pp_next_check` is NOT re-derived: it travels with the request (`arm_prompt_state` puts
            // the serve-scope copy back from `R.pp_next_check` and the step copies it out again), so
            // the mid-prompt checkpoint schedule keeps its original positions across a swap.
            // No log line here either: `do_swap` already prints `slot N resumed from T tokens`, and
            // for a prefill `T` IS this cursor.
        };
        // ---- phases 1 + 2: parse the line, resolve the resume point, mount, dispatch sampling.
        // `Prep::rejected` means the request was refused and its ERR line is already printed - the
        // old `continue` sites, in the same places.
        auto prep_request = [&](ReqCtx& R, const std::string& line) -> Prep {

            // ---- S3.1c: the request line, parsed by the one parser that knows both wire forms ----------
            const strata::program::serve_proto::Defaults proto_def{o.pcie_frac, o.spec_min_p};
            R.rq = strata::program::serve_proto::parse_request(line, proto_def, tagged);
            const strata::program::serve_proto::Request& rq = R.rq;

            if (rq.kind == strata::program::serve_proto::Kind::stop) {
                // A STOP that reached the queue rather than the stdin thread: only possible for a malformed
                // `STOP <junk>`, which 0.1.30 had no word for at all.
                std::printf("%s\n", sp_err(rq.error).c_str());
                return Prep::rejected;
            }
            if (rq.kind != strata::program::serve_proto::Kind::gen &&
                rq.kind != strata::program::serve_proto::Kind::geni) {
                std::printf("%s\n", sp_out.err(strata::program::serve_proto::err_expected()).c_str());
                return Prep::rejected;
            }
            if (!rq.error.empty()) {
                std::printf("%s\n", sp_out.err(rq.error).c_str());
                return Prep::rejected;
            }
            R.geni = rq.kind == strata::program::serve_proto::Kind::geni;
            const bool& geni = R.geni;
            R.max_new = (long long) rq.max_new;
            const long long& max_new = R.max_new;
            // optional sampling keys between max_new and the ids: temperature=F, top_p=F, top_k=N, min_p=F,
            // penalty_last_n=N, penalty_repeat=F, penalty_freq=F, penalty_present=F, seed=N (text requests
            // only).  Absent keys keep today's behavior: greedy, no penalties.
            R.req_temperature = rq.temperature;
            const float& req_temperature = R.req_temperature;
            R.req_top_p = rq.top_p;
            const float& req_top_p = R.req_top_p;
            R.req_top_k = rq.top_k;
            const int& req_top_k = R.req_top_k;   // the sampler's own default; the sampled path REQUIRES top_k in 1..64
            R.req_seed = rq.seed;
            const unsigned long long& req_seed = R.req_seed;
            R.req_min_p = rq.min_p; R.req_penalty_repeat = rq.penalty_repeat;
            const float& req_min_p = R.req_min_p; const float& req_penalty_repeat = R.req_penalty_repeat;
            R.req_penalty_freq = rq.penalty_freq; R.req_penalty_present = rq.penalty_present;
            const float& req_penalty_freq = R.req_penalty_freq; const float& req_penalty_present = R.req_penalty_present;
            R.req_penalty_last_n = rq.penalty_last_n;
            const int& req_penalty_last_n = R.req_penalty_last_n;
            R.req_cvec = rq.cvec;
            const int& req_cvec = R.req_cvec;   // cvec=0|1: a loaded control vector for this request (on when absent)
            // tuning keys (setup's calibration measures settings without restarting the engine): the PCIe share of
            // the missed experts and the draft-probability floor, for this request only
            R.req_pcie_frac = rq.pcie_frac; R.req_spec_min_p = rq.spec_min_p;
            const double& req_pcie_frac = R.req_pcie_frac;
            R.emb_path = rq.emb_path;
            const std::string& emb_path = R.emb_path;
            R.ids = rq.ids;
            std::vector<int64_t>& ids = R.ids;
            // ---- S3.1b: the request's registry row.  With --serve-slots 0/1 there is none: the serial path
            // is 0.1.30's path, and inventing a row for it would change nothing except the risk.
            req_id = rq.id;
            R.id = rq.id;
            bool& slot_open = R.slot_open;
            if (tagged) {
                if (req_id == strata::program::serve_proto::kNoId) {
                    // A tagged session needs an id to route its answer by.  An old client against a
                    // --serve-slots >= 2 engine is the one pair the compatibility matrix does NOT support
                    // (§6.4: the server enables multi-slot only when it sees slots=), so say so plainly
                    // instead of guessing an id and mis-routing the answer.
                    std::printf("%s\n", sp_out.err("this engine was started with --serve-slots " +
                                                   std::to_string(slots) + ": requests need a request id "
                                                   "(GEN <id> <max_new> ...)").c_str());
                    return Prep::rejected;
                }
                std::string serr;
                {
                    std::lock_guard<std::mutex> lk(slot_mu);
                    const int64_t now_ms = (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                        Clock::now().time_since_epoch()).count();
                    strata::program::slot::Slot* s = slots_reg.add(req_id, now_ms);
                    if (s == nullptr) serr = "no slot free for request id";
                    else {
                        s->req = rq;
                        s->ctx_cap = o.max_context;
                        s->prompt_tokens = (int64_t) ids.size();
                        s->ctx_used = (int64_t) ids.size();
                        // A `STOP <id>` that arrived before this line did (serve/server.py can send both back
                        // to back): apply it now, so the request is cancelled rather than run.
                        for (size_t k = 0; k < pending_cancel.size(); ++k)
                            if (pending_cancel[k] == req_id) {
                                pending_cancel.erase(pending_cancel.begin() + (std::ptrdiff_t) k);
                                slots_reg.cancel_request(req_id);
                                break;
                            }
                    }
                    running_id.store(req_id);
                }
            if (!serr.empty()) { std::printf("%s\n", sp_err(serr).c_str()); return Prep::rejected; }
                emit_slot(req_id);
                slot_open = true;
            }
            R.n = (int64_t) ids.size();
            const int64_t n = R.n;
            // S3.1d R7: copy the mounted slot's position table out BEFORE this request may overwrite
            // it.  One table, one owner - the save has to come first or the outgoing slot loses it.
            if (swaps_on) sync_positions();
            req_imgs.clear();
            if (geni && !o.vision) {
                std::printf("%s\n", sp_err("this engine was started without --vision").c_str());
                return Prep::rejected;
            }
            if (geni || !mrope_identity) {
                R.mrope_touched = true;
                // positions for every cell this request can reach; the identity again for a text request
                std::string ve;
                row_ptr.assign((size_t) n, nullptr);
                const int64_t cells = (int64_t) mrope_host.size() / 3;
                auto put = [&](int64_t c, int64_t t, int64_t h, int64_t w) {
                    mrope_host[(size_t) c * 3] = (int32_t) t;
                    mrope_host[(size_t) c * 3 + 1] = (int32_t) h;
                    mrope_host[(size_t) c * 3 + 2] = (int32_t) w;
                };
                if (!geni) {
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                } else {
                    struct Img { int64_t n, nx, ny; size_t off; };
                    std::vector<Img> imgs;
                    img_rows.clear();
                    std::FILE* f = std::fopen(emb_path.c_str(), "rb");
                    if (!f) ve = "cannot open " + emb_path;
                    while (f && ve.empty()) {
                        int32_t hdr[5];
                        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
                        if (got == 0) break;
                        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[2] < 1 || hdr[3] < 1 ||
                            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != (int32_t) g.n_embd) {
                            ve = "bad embeddings file (expected strata-vision records of width " +
                                 std::to_string((long long) g.n_embd) + ")";
                            break;
                        }
                        const size_t off = img_rows.size(), cnt = (size_t) hdr[1] * (size_t) hdr[4];
                        img_rows.resize(off + cnt);
                        if (std::fread(img_rows.data() + off, sizeof(float), cnt, f) != cnt) { ve = "short embeddings file"; break; }
                        imgs.push_back({hdr[1], hdr[2], hdr[3], off});
                    }
                    if (f) std::fclose(f);
                    int64_t p = 0, i = 0;
                    size_t k = 0;
                    while (ve.empty() && i < n) {
                        if (ids[(size_t) i] != kImagePad) { put(i, p, p, p); ++p; ++i; continue; }
                        if (k >= imgs.size()) { ve = "the prompt has more images than the embeddings file"; break; }
                        const Img& im = imgs[k++];
                        {   // what the conversation cache compares: a picture is its grid and its embeddings
                            const int64_t grid[3] = {im.n, im.nx, im.ny};
                            uint64_t h = fnv1a(grid, sizeof grid);
                            h = fnv1a(img_rows.data() + im.off, (size_t) im.n * (size_t) g.n_embd * sizeof(float), h);
                            req_imgs.push_back({i, h});
                        }
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j)
                            if (i + j >= n || ids[(size_t) (i + j)] != kImagePad)
                                ve = "image " + std::to_string(k) + " has " + std::to_string((long long) im.n) +
                                     " rows but fewer <|image_pad|> tokens";
                        for (int64_t j = 0; j < im.n && ve.empty(); ++j) {
                            const int64_t y = j / im.nx, x = j % im.nx;
                            put(i + j, p, p + y, p + x);
                            row_ptr[(size_t) (i + j)] = img_rows.data() + im.off + (size_t) j * (size_t) g.n_embd;
                        }
                        i += im.n;
                        p += std::max(im.nx, im.ny);
                    }
                    if (ve.empty() && k != imgs.size()) ve = "the embeddings file has more images than the prompt";
                    if (ve.empty() && n > 0 && ids[(size_t) (n - 1)] == kImagePad) ve = "the prompt cannot end in an image";
                    for (int64_t c = n; ve.empty() && c < cells; ++c) put(c, p + (c - n), p + (c - n), p + (c - n));
                }
                tr("positions built", (long long) img_rows.size());
                cudaDeviceSynchronize();
                tr("device idle");
                // CUDA0's table and, with a layer split, every later stage's (each device reads its own)
                auto upload_mrope = [&]() -> bool { return upload_mrope_table(); };
                if (ve.empty() && !upload_mrope()) ve = "the image position upload failed";
                if (!ve.empty()) {
                    // leave the table as the identity so the next text request is untouched
                    for (int64_t c = 0; c < cells; ++c) put(c, c, c, c);
                    upload_mrope();
                    mrope_identity = true;
                    std::printf("%s\n", sp_err(ve).c_str());
                    std::fflush(stdout);
                    return Prep::rejected;
                }
                mrope_identity = !geni;
            }
            // S3.1d R7: the table in the host AND in the device now describes THIS request, and this
            // request's slot is the one about to be mounted.  Record that - both who owns the one
            // table and a copy of it in the slot's record, so a later hand-over that does NOT come
            // from a request line (S3.1e pre-empting a decode) can put it back.  This runs whether or
            // not the block above executed: when it did not, the table is still the identity and the
            // device still matches it, so the record must say "identity" too.
            if (swaps_on) {
                mrope_owner = req_id;
                strata::program::serve_swap::SlotConv& c = conv_of(req_id);
                c.mrope_image = !mrope_identity;
                c.mrope = mrope_identity ? std::vector<int32_t>{} : mrope_host;
            }
            sp.embd_rows = geni ? row_ptr.data() : nullptr;
            if (n + max_new + 8 > o.max_context) {
                std::printf("%s\n", sp_err("prompt (" + std::to_string(n) + " tokens) + max_new (" +
                                           std::to_string(max_new) + ") exceeds the context (" +
                                           std::to_string(o.max_context) + ")").c_str());
                return Prep::rejected;
            }
            bool bad = false;
            for (int64_t t : ids) bad = bad || t < 0 || t >= n_vocab;
            if (bad) { std::printf("%s\n", sp_err("a token id is outside the vocabulary").c_str()); return Prep::rejected; }
            std::array<int64_t, 3>& remote_before = R.remote_before;
            std::array<int64_t, 3>& launches_before = R.launches_before;
            std::array<uint64_t, 3>& compact_before = R.compact_before;
            std::array<uint64_t, 3>& full_before = R.full_before;
            // ms_begin/ms_wait are cumulative since boot; the log line used to print them next to per-request deltas,
            // so the host time read as if it belonged to this request.  Take deltas here like every other column.
            std::array<double, 3>& begin_before = R.begin_before;
            std::array<double, 3>& wait_before = R.wait_before;
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            {
                remote_before[(size_t) r] = remote_experts[(size_t) r].computed();
                launches_before[(size_t) r] = remote_experts[(size_t) r].launched_layers();
                compact_before[(size_t) r] = remote_experts[(size_t) r].returned_bytes();
                full_before[(size_t) r] = remote_experts[(size_t) r].full_row_bytes();
                begin_before[(size_t) r] = remote_experts[(size_t) r].ms_begin();
                wait_before[(size_t) r] = remote_experts[(size_t) r].ms_wait();
            }
            cur = ids;
            // ---- S3.1d: the hand-over, when this request belongs to a DIFFERENT slot ----------------
            //
            // This is the whole of S3.1d's call site, and it is deliberately the ONLY new thing a
            // request can hit: with --serve-slots 0/1 or STRATA_NO_SWAP=1 it never runs, and the
            // park/mount sequence below is 0.1.30's verbatim (§7.1's bit-exactness bar).
            //
            // It runs BEFORE the resume search and before `r0`, because everything after it reads
            // `live`/`checks`/`cvec_cached`, and those must already be the incoming slot's.  The swap's
            // own time is not charged to this request's `prompt_ms`; it is reported on its own stderr
            // line and counted in `slots_reg.note_swap()` so /metrics can price OQ3.
            parked_by_swap = false;
            mounted_by_swap = false;
            if (swaps_on) prune_conv();
            if (swaps_on && strata::program::serve_swap::swap_needed(mounted_id, req_id)) {
                sync_conversation();   // the outgoing slot's mirror must describe the session we are about to save
                const bool want_cvec_now = strata::kernels::cvec().loaded() ? req_cvec != 0 : true;
                // Take this request's parked image out of the cache FIRST: the outgoing save then
                // counts it as `held` RAM, exactly as 0.1.30's `park_current(incoming->bytes())` does.
                // S3.9: `req_id` is the requester, so an entry CLAIMED by another running slot is not
                // offered.  This lookup matches on any checkpoint prefix, and the mount that follows
                // takes the WHOLE entry - which is how a 9 849-token prefix reuse destroyed another
                // slot's 10 394-token branch and got that slot ended.
                const auto pre = conversations.best(ids, req_imgs, want_cvec_now,
                                                    swaps_on ? req_id : strata::core::kNoOwner);
                if (pre.tokens > 0) mount_image.emplace(conversations.take(pre.index));
                bool poisoned = false;
                std::string serr;
                bool swapped = swap_to(req_id, serr, poisoned, /*restore_positions=*/false);
                if (swapped) {
                    // The hand-over already dealt with the outgoing branch - it saved it, or it
                    // refused to and invalidated it.  Either way the session now holds the INCOMING
                    // conversation, so the request body must not park: that would snapshot the branch
                    // it has just mounted, a gigabyte of memcpy and a duplicate cache entry.
                    parked_by_swap = true;
                }
                if (!swapped && poisoned) {
                    // A restore that failed mid-write: fatal to the session, exactly as today.
                    std::printf("%s\n", sp_out.err("restoring parked conversation: " + serr, req_id).c_str());
                    std::fflush(stdout);
                    return Prep::fatal;
                }
                if (!swapped && swap_wrote_session) {
                    // The restore happened and something after it failed (or the restore itself
                    // failed).  The request cannot run against a half-established slot, and it must
                    // not retry: a second hand-over would save THIS branch out as the outgoing one.
                    return_mount_image();
                    std::printf("%s\n", sp_out.err("slot hand-over failed after the restore: " + serr, req_id).c_str());
                    std::fflush(stdout);
                    slot_step(req_id, strata::program::slot::State::error);
                    return Prep::rejected;
                }
                if (!swapped && swap_park_refused) {
                    // S3.6: the slot the session reflects still has work to do and its state cannot be
                    // saved, so the session stays with it and the retry-without-a-restore below must
                    // NOT run - that retry is what used to destroy a running conversation.
                    //
                    // The INCOMING request is answered, not starved and not spun on: an ERR naming
                    // both numbers, its row closed, its permit back.  A request line is a one-shot
                    // hand-over - the client asked for this conversation now - so unlike the driver's
                    // `do_swap` there is nothing to defer it behind.
                    return_mount_image();
                    std::fprintf(stderr, "strata serve: swap to slot %lld refused at the request line (%s) - "
                                         "that request is ended, the mounted slot keeps the session\n",
                                 (long long) req_id, serr.c_str());
                    std::printf("%s\n", sp_out.err(serr, req_id).c_str());
                    std::fflush(stdout);
                    slot_step(req_id, strata::program::slot::State::error);
                    return Prep::rejected;
                }
                if (!swapped) {
                    // The image was rejected (or the save refused) BEFORE anything was written.  The
                    // outgoing branch is intact and still mounted; drop the image and hand over
                    // without a restore, which is the same answer 0.1.30 gives for an invalid
                    // snapshot: fall back to re-reading.
                    std::fprintf(stderr, "strata serve: swap: slot %lld -> %lld retried without a restore (%s)\n",
                                 (long long) mounted_id, (long long) req_id, serr.c_str());
                    return_mount_image();
                    poisoned = false;
                    serr.clear();
                    if (!swap_to(req_id, serr, poisoned, /*restore_positions=*/false)) {
                        std::printf("%s\n", sp_out.err("slot hand-over failed: " + serr, req_id).c_str());
                        std::fflush(stdout);
                        if (poisoned) return Prep::fatal;
                        slot_step(req_id, strata::program::slot::State::error);
                        return Prep::rejected;
                    }
                }
            }
            R.r0 = Clock::now();
            const Clock::time_point& r0 = R.r0;
            // S0.3: the request's loan bill is a delta over these (STRATA_PREFILL_LOAN_TIMING)
            R.loan_bill = loan_totals(pf_parts);
            R.res_uploads0 = res_dirty.uploads(); R.res_skips0 = res_dirty.skipped();
            // ---- where this request starts reading: the live session, or a checkpoint, whose tokens AND pictures are
            // exactly the start of this prompt - at most n - 1 of them, the last token is always the first window
            auto starts_with = [&](const std::vector<int32_t>& pre, const std::vector<ImgKey>& pre_imgs) -> bool {
                const int64_t L = (int64_t) pre.size();
                if (L < 1 || L > n - 1) return false;
                for (int64_t i = 0; i < L; ++i)
                    if ((int32_t) ids[(size_t) i] != pre[(size_t) i]) return false;
                return imgs_below(req_imgs, L) == pre_imgs;
            };
            R.want_cvec = strata::kernels::cvec().loaded() ? req_cvec != 0 : true;
            const bool& want_cvec = R.want_cvec;
            int64_t& resume = R.resume;
            bool& from_live = R.from_live;
            if (o.prompt_cache > 0 && want_cvec == cvec_cached) {
                if (live_ok && starts_with(live, live_imgs)) { resume = (int64_t) live.size(); from_live = true; }
                for (const ConvCheckpoint& c : checks)
                    if ((int64_t) c.ids.size() > resume && starts_with(c.ids, c.imgs)) {
                        resume = (int64_t) c.ids.size();
                        from_live = false;
                    }
            }
            // S3.1d: the swap already restored this slot's image, so the session IS its longest
            // prefix; searching the cache again would mount a second image over it.
            // S3.9: and the same claim rule as the request-line lookup - this is the other prefix
            // `take()`, and it can destroy a different slot's parked branch just as easily.
            const auto parked = mounted_by_swap
                                    ? strata::core::ConversationCache::Match{}
                                    : conversations.best(ids, req_imgs, want_cvec,
                                                         swaps_on ? req_id : strata::core::kNoOwner);
            std::optional<strata::core::SavedConversation> incoming;
            if (parked.tokens > resume) incoming.emplace(conversations.take(parked.index));
            // Reject the entire image before parking/overwriting the outgoing
            // state. Invalid entries can safely fall back to its existing prefix.
            if (incoming && !strata::core::conversation_snapshot_validate(*incoming, ss, conv_stages, g, mtp.kv_state(), err)) {
                std::fprintf(stderr, "strata serve: conversation cache: discard invalid snapshot (%s)\n", err.c_str());
                incoming.reset();
                err.clear();
            }
            // Preserve the outgoing branch before any checkpoint rewind, reset,
            // or incoming restore overwrites the positional state it requires.
            // S0.3 lever 4: a request that mounts a parked conversation or starts one from zero is exactly
            // where parking pays, so it asks even inside a quiet window.  It does NOT clear the refusal
            // streak: on this box roughly one request in thirteen starts a new conversation, and clearing on
            // those would mean the backoff never engages at all.
            const bool park_forced = incoming != std::nullopt || resume == 0;
            // S3.1d: after a hand-over the outgoing branch was already saved by the swap, so the
            // request body must not park it a second time (that would duplicate the image the swap has
            // just restored).  Only a snapshot FAILURE is fatal here, exactly as in 0.1.30.
            const bool need_park = (!from_live || incoming) && !parked_by_swap;
            // S3.9: what is parked here is the branch the SESSION holds, which belongs to the slot
            // still mounted - not to the request being prepped.  Claiming it for `req_id` would be
            // worse than not claiming it: it would hide another slot's conversation behind this one's
            // id, and release it when this request finished.
            if (need_park && park_current(incoming ? incoming->bytes() : 0, park_forced,
                                          park_owner_for(mounted_id)) ==
                            strata::program::serve_swap::Saved::failed) {
                std::printf("%s\n", sp_out.err(err, R.id).c_str());
                return Prep::fatal;
            }
            if (incoming) {
                const auto t0 = Clock::now();
                if (strata::core::conversation_snapshot_restore(*incoming, ss, conv_stages, g, mtp.kv_state(), err) !=
                    strata::core::ConversationRestore::restored) {
                    // Already prevalidated above: a failure here is fatal, never
                    // permission to decode from a partially restored session.
                    std::printf("%s\n", sp_out.err("restoring parked conversation: " + err, R.id).c_str());
                    return Prep::fatal;
                }
                if (std::getenv("STRATA_SNAPSHOT_VERIFY") != nullptr) {
                    uint64_t draft_hash = 0;
                    if (!strata::core::conversation_kv_verify(incoming->kv.back(), mtp.kv_state(), g,
                            int64_t(incoming->live.ids.size()), false, draft_hash, err)) {
                        std::printf("%s\n", sp_out.err("verifying restored draft KV: " + err, R.id).c_str());
                        return Prep::fatal;
                    }
                    std::fprintf(stderr, "strata serve: SNAPSHOT_VERIFY draft=%016llx cells=%lld mode=%d source=%s resident=%lld\n",
                                 (unsigned long long) draft_hash, (long long) incoming->kv.back().cells,
                                 mtp.kv_state().kv_mode, "ram",
                                 (long long) (mtp.kv_state().n_slots * strata::kernels::qsa_real_shapes().page_size));
                }
                live = std::move(incoming->live.ids);
                live_imgs = std::move(incoming->live.imgs);
                checks = std::move(incoming->checkpoints);
                cvec_cached = incoming->cvec;
                resume = parked.tokens;
                from_live = parked.live;
                if (std::getenv("STRATA_SNAPSHOT_FULL_CAPTURE") == nullptr)
                    conversations.retain(std::move(incoming->kv), int64_t(live.size()));
                incoming.reset(); // Running-state/checkpoint copies are no longer needed.
                std::fprintf(stderr, "strata serve: conversation cache: restored %lld tokens (%s) in %.1f ms; parked=%zu bytes=%zu\n",
                             (long long) resume, from_live ? "live" : "checkpoint",
                             std::chrono::duration<double, std::milli>(Clock::now() - t0).count(),
                             conversations.size(), conversations.bytes());
            }
            if (want_cvec != cvec_cached) {
                live_ok = false;
                checks.clear();
                cvec_cached = want_cvec;
            }
            if (strata::kernels::cvec().loaded()) strata::kernels::cvec_set_enabled(want_cvec);
            // this request rewrites every cell from `resume` on, so a checkpoint past it (or not on this prompt's
            // path) no longer has its cells; the ones kept are prefixes of both the old tokens and the new
            checks.erase(std::remove_if(checks.begin(), checks.end(), [&](const ConvCheckpoint& c) {
                             return (int64_t) c.ids.size() > resume || !starts_with(c.ids, c.imgs);
                         }), checks.end());
            live_ok = false;   // until this request has finished, the session is in between
            int64_t& reread_to = R.reread_to;   // STRATA_CKPT_REREAD only: read [0, reread_to) again instead of restoring
            if (resume == 0) {
                strata::core::session_zero(ss, g, nullptr, main_cs);
                cudaStreamSynchronize(main_stream);
                for (auto& st : stages) {
                    const strata::core::OnDevice on(st->dev);
                    strata::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                    cudaStreamSynchronize(st->stream);
                }
                checks.clear();
            } else if (!from_live) {
                ConvCheckpoint* c = nullptr;
                for (ConvCheckpoint& k : checks) if ((int64_t) k.ids.size() == resume) c = &k;
                if (c != nullptr) c->used = ++check_clock;   // mounting through it is the use LRU counts
                static const bool reread = std::getenv("STRATA_CKPT_REREAD") != nullptr;
                if (reread && c != nullptr) {
                    // THE CHECK OF THE CHECKPOINT: instead of restoring it, read its tokens again from position 0 in
                    // one run (below, with the prompt path's slots lent like any read) - the same chunks the request
                    // that saved it read them in, when that request started at 0.  With the VRAM expert set fixed
                    // (--adapt-swaps 0) the answer must match the restored one token for token; anything the
                    // checkpoint missed shows up as a difference.
                    strata::core::session_zero(ss, g, nullptr, main_cs);
                    cudaStreamSynchronize(main_stream);
                    for (auto& st : stages) {
                        const strata::core::OnDevice on(st->dev);
                        strata::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                        cudaStreamSynchronize(st->stream);
                    }
                    reread_to = resume;
                    std::fprintf(stderr, "strata serve: STRATA_CKPT_REREAD: reading %lld tokens again instead of "
                                         "restoring\n", (long long) resume);
                } else if (c == nullptr || !checkpoint_restore(*c, ss, g) || c->stage_parts.size() != stages.size() ||
                           [&] {
                               for (size_t i = 0; i < stages.size(); ++i) {
                                   const strata::core::OnDevice on(stages[i]->dev);
                                   if (!checkpoint_restore(c->stage_parts[i], stages[i]->ss, g)) return true;
                               }
                               return false;
                           }()) {
                    std::printf("%s\n", sp_out.err("restoring a conversation checkpoint failed", R.id).c_str());
                    return Prep::fatal;
                }
            }
            // KV streaming: the drafter's ring may hold cells past `resume` from a longer turn; the main layers'
            // host copies and slots are always current (every writer writes both), so they need nothing
            if (resume > 0 && reread_to <= 0) mtp.kv_restore(resume);
            tr("request", n, geni ? 1 : 0);
            mtp.set_prompt_len(n);
            if (swaps_on) mtp_prompt_len = n;   // travels with the slot (serve_swap::SlotConv::prompt_len)
            R.read_from = reread_to > 0 ? 0 : resume;
            const int64_t& read_from = R.read_from;
            conversations.limit_reuse(read_from);
            pp_total = n;
            pp_from = read_from;
            pp_t0 = r0;
            pp_next_check = reread_to > 0 ? INT64_MAX : resume + o.prompt_cache_every;
            {
                std::lock_guard<std::mutex> lk(part_mu);
                part_at.clear();
                std::fill(part_next.begin(), part_next.end(), pp_next_check);
            }
            std::printf("%s\n", sp_out.resume(resume, req_id).c_str());   // before reading: this many prompt tokens are reused
            slot_step(req_id, strata::program::slot::State::prefilling);
            strata::core::progress_at("reading the prompt, from token", read_from);
            std::fflush(stdout);
            apply_pending(true);
            apply_pending(true);
            // per-request sampling for the verify window's head (greedy when temperature is absent)
            strata::kernels::SamplerParams& req_sp = R.req_sp;
            req_sp.greedy = req_temperature <= 0.0f;
            req_sp.temperature = req_temperature;
            req_sp.top_p = req_top_p;
            req_sp.top_k = req_top_k;
            req_sp.seed = req_seed ? req_seed
                                   : (unsigned long long) std::chrono::steady_clock::now().time_since_epoch().count();
            req_sp.min_p = std::clamp(req_min_p, 0.0f, 1.0f);
            req_sp.penalty_last_n = std::max(req_penalty_last_n, 0);
            req_sp.penalty_repeat = req_penalty_repeat;
            req_sp.penalty_freq = req_penalty_freq;
            req_sp.penalty_present = req_penalty_present;
            req_sp.counter = 0;
            ver.set_sampling(req_sp);
            mtp.set_draft_sampling(req_sp);   // STRATA_SPEC_COUPLED=1: sampled drafts (a no-op otherwise)
            drive.d.pcie_num = std::max(0, std::min(256, (int) (req_pcie_frac * 256.0 + 0.5)));
            // a layer split: CUDA0's share as asked; a later GPU keeps its own (its link) unless the request sets one
            for (int st = 0; st < split_drive.n; ++st)
                split_drive.pcie_num[st] = (st == 0 || split_same || req_pcie_frac != o.pcie_frac)
                                               ? drive.d.pcie_num : pcie_num_of(stages[(size_t) st - 1]->pcie_frac);
            R.hist_n = std::min(req_sp.penalty_last_n, kPenaltyWindowCap);
            const int& hist_n = R.hist_n;
            ver.set_history(hist_n > 0 ? d_hist : nullptr, hist_n);
            // S3.1d: the same state, remembered for the slot, so a hand-over that is NOT driven by a
            // request line (S3.1e pre-empting a decode) can put it back.  `smpl.counter` is not
            // mirrored: the verifier derives the draw counter from the window position
            // (verify.cpp:1087), which is per-slot sequence state, not sampler state.
            // S3.6: from here to the next hand-over, the session describes THIS request - `prep_request`
            // either restored its parked image, resumed one of its checkpoints, or zeroed the session
            // and is about to read the whole prompt into it.
            R.session_valid = true;
            session_established = true;
            R.own_imgs = req_imgs;
            if (swaps_on) {
                strata::program::serve_swap::SlotConv& c = conv_of(req_id);
                c.smpl = req_sp;
                c.smpl_set = true;
                c.penalty_last_n = req_sp.penalty_last_n;
                c.pcie_num = drive.d.pcie_num;
            }
            return Prep::ok;
        };
        // ---- phase 3 prologue: decide which prompt segments this request reads.
        // S3.1e-1: split out of the straight-line body so S3.1e-2 can plan a request's reads when it
        // admits it and then drive them one step at a time.  Same four segment ends, same order.
        auto plan_prompt_segments = [&](ReqCtx& R) {
            std::vector<int64_t>& ids = R.ids;
            const int64_t n = R.n;
            const int64_t& resume = R.resume;
            const int64_t& read_from = R.read_from;
            const int64_t& reread_to = R.reread_to;
            int64_t& turn_at = R.turn_at;
            int64_t& root_at = R.root_at;
            tr("prompt start", n - 1);
            // The prompt is read in two parts when it has a turn boundary past `resume`: up to the last <|im_start|>
            // (the conversation so far), a checkpoint there, then the new turn's header.  The next request of the same
            // chat renders the same history - but not always the same header or the thinking of this reply - so that
            // checkpoint is the one it reuses.
            if (o.prompt_cache > 0 && o.turn_token >= 0)
                for (int64_t i = n - 1; i > resume; --i)
                    if (ids[(size_t) i] == o.turn_token) { turn_at = i; break; }
            // A prompt read from token 0 also stops at its FIRST turn boundary: the end of the system prompt (with
            // the tools), which every new chat of the same client shares.  That checkpoint becomes the chain's root,
            // which the retention policy pins (conv_cache.hpp), so the next new chat reads only what comes after it.
            // (PR #65, code-martin.)  Only for a system prompt of --prompt-cache-root tokens or more: a small one
            // is cheaper to read again than the extra part costs (~0.3 s).
            if (o.prompt_cache > 0 && o.turn_token >= 0 && o.prompt_cache_root > 0 && read_from == 0)
                for (int64_t i = 1; i < turn_at; ++i)
                    if (ids[(size_t) i] == o.turn_token) {
                        if (i >= o.prompt_cache_root) root_at = i;
                        break;
                    }
            R.at = read_from;
            R.seg = {reread_to, root_at, turn_at, n - 1};
            R.seg_i = 0;
        };
        // ---- S3.6: send a request back to token 0, for real.
        //
        // `swap_to`'s unmount path has always promised that a slot which lost its parked image
        // "will be re-read from token 0".  Nothing ever honoured it: the request's `ReqCtx` kept its
        // phase, its cursor and its segment list, so the next step resumed exactly where it left off
        // - against a session that now held somebody else's tokens.  This is the part that makes the
        // promise true, and it is only ever reached for a request still in `prefill`/`prefill_end`
        // (`serve_driver::step_gate` ends a `decode` request instead, because re-reading its prompt
        // would emit the tokens it already sent).
        //
        // What has to change, and why each piece is load-bearing:
        //   * the SESSION: zeroed, because its positional cells hold another conversation's tokens.
        //     `prep_request` does exactly this when `resume == 0`; skipping it is the bug.
        //   * the process-wide branch/chain (`live`, `live_imgs`, `checks`, `live_ok`): they describe
        //     the conversation that was destroyed, and a checkpoint is only valid while the cells
        //     below it hold its tokens.
        //   * the request's own resume point: `resume = 0`, `read_from = 0`, `reread_to = -1`, so the
        //     segments re-plan from the start and the mid-prompt checkpoint schedule restarts.
        //   * `R.session_valid = true` at the end: the session is now THIS request's again, so the
        //     gate lets it step.
        auto reset_request_to_token0 = [&](ReqCtx& R) {
            if (R.phase != strata::program::serve_driver::Phase::prefill &&
                R.phase != strata::program::serve_driver::Phase::prefill_end) return;
            // The loan must be back before the session is torn down: a snapshot or a zeroing taken
            // while the prompt path holds cache slots describes a cache that does not exist (R8).
            if (!refill(err)) {
                std::fprintf(stderr, "strata serve: slot %lld: returning its loan before re-reading failed "
                                     "(%s)\n", (long long) R.id, err.c_str());
                err.clear();
            }
            strata::program::serve_swap::SlotConv& c = conv_of(R.id);
            const int64_t dropped = R.n;
            const size_t dropped_checks = checks.size();
            // 1. the session, from token 0.  Same calls `prep_request` makes on that path.
            strata::core::session_zero(ss, g, nullptr, main_cs);
            cudaStreamSynchronize(main_stream);
            for (auto& st : stages) {
                const strata::core::OnDevice on(st->dev);
                strata::core::session_zero(st->ss, g, nullptr, (void*) st->stream);
                cudaStreamSynchronize(st->stream);
            }
            // 2. the branch and the chain the destroyed conversation owned.
            live.clear();
            live_imgs.clear();
            checks.clear();
            live_ok = false;
            cvec_cached = R.want_cvec;
            if (strata::kernels::cvec().loaded()) strata::kernels::cvec_set_enabled(R.want_cvec);
            // 3. the slot's own record: it no longer has an image, and its mirror is empty.
            c.live.clear();
            c.live_imgs.clear();
            c.checks.clear();
            c.live_ok = false;
            c.resumable = false;      // nothing was parked; a later hand-over must not restore it
            c.parked_index = -1;
            c.parked_bytes = 0;
            // 4. the request's resume point and its segment plan.
            R.resume = 0;
            R.from_live = false;
            R.reread_to = -1;
            R.read_from = 0;
            R.turn_at = R.root_at = -1;
            R.at = 0;
            R.seg = {-1, -1, -1, R.n - 1};
            R.seg_i = 0;
            R.pp_total = R.n; R.pp_from = 0; R.pp_t0 = R.r0;
            R.pp_next_check = R.resume + o.prompt_cache_every;
            conversations.limit_reuse(0);
            {
                std::lock_guard<std::mutex> lk(part_mu);
                part_at.clear();
                std::fill(part_next.begin(), part_next.end(), R.pp_next_check);
            }
            // 5. the phase, back to the start of the read, and the session is ours again.  BOTH
            // flags: `R.session_valid` is recomputed from `session_established` at the top of every
            // step, so setting only the request's copy would be undone on the next pass.
            R.phase = strata::program::serve_driver::Phase::prefill;
            R.session_valid = true;
            session_established = true;
            // The prompt path's per-stage mid-prompt parts belong to the read that was just thrown
            // away; the next checkpoint of this request must not splice onto them.  `arm_prompt_state`
            // clears `part_at` when the prefilling request CHANGES, but a re-read is the same request,
            // so it has to be cleared here explicitly.
            {
                std::lock_guard<std::mutex> lk(part_mu);
                part_at.clear();
            }
            // The wire: this request now reuses nothing.  `RESUME` is printed once per request by
            // `prep_request`, so the client is told again here rather than left with a stale number.
            std::printf("%s\n", sp_out.resume(0, R.id).c_str());
            std::fflush(stdout);
            std::fprintf(stderr, "strata serve: slot %lld reset to token 0: %lld prompt tokens will be read "
                                 "again (%zu checkpoints dropped)\n",
                         (long long) R.id, (long long) dropped, dropped_checks);
            std::fflush(stderr);
        };
        // ---- phase 3: read ONE prompt segment.  0.1.30 ran the four segments back to back in one
        // `for`; the serial loop now calls this until it stops reporting `progressed`, which is the
        // same lend / read / refill / checkpoint calls in the same order.  S3.1e-2 interleaves them.
        auto run_prefill_step = [&](ReqCtx& R) -> Step {
            std::vector<int64_t>& ids = R.ids;
            const int64_t& turn_at = R.turn_at;
            const int64_t& root_at = R.root_at;
            int64_t& at = R.at;
            bool& cancelled = R.cancelled;
            // the next segment with something to read - 0.1.30's `if (to <= at) continue;`
            int64_t to = -1;
            bool have = false;
            while (R.seg_i < R.seg.size()) {
                const int64_t t = R.seg[R.seg_i++];
                if (t > at) { to = t; have = true; break; }
            }
            if (!have) return Step::finished;
            err.clear();
            const bool win = windows_ok(at, to);
            if (win && !refill(err)) {
                std::printf("%s\n", sp_out.err("refilling a lent slot failed: " + err, R.id).c_str());
                return Step::error;
            }
            if (!win && !lend(to - at, err)) {
                std::printf("%s\n", sp_out.err("lending the prompt path its slots failed: " + err, R.id).c_str());
                return Step::error;
            }
            const auto tsp = Clock::now();
            const bool sp_ok = win ? read_windows(at, to, err) : sp.run(ids.data() + at, to - at, at, err);
            if (trace) {
                std::fprintf(stderr, "strata trace: read %lld tokens (%s) in %.1f ms\n", (long long) (to - at),
                             win ? "windows" : "batched",
                             std::chrono::duration<double, std::milli>(Clock::now() - tsp).count());
                std::fflush(stderr);
            }
            if (!sp_ok) {
                if (!stopped()) {
                    std::fprintf(stderr, "strata serve: %s\n", err.c_str());
                    std::printf("%s\n", sp_out.err(err, R.id).c_str());
                    // #224: a CUDA fault (an illegal address) poisons the context for the whole process, and
                    // unwinding the destructors on it could hang until the 60 s watchdog: leave at once
                    if (cudaPeekAtLastError() != cudaSuccess) {
                        std::fflush(stdout);
                        std::fflush(stderr);
                        return Step::fatal_exit;
                    }
                    return Step::error;
                }
                cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel
                return Step::cancelled;
            }
            at = to;
            if ((to == turn_at || to == root_at) && !checkpoint_at(to)) {
                std::printf("%s\n", sp_out.err("saving a conversation checkpoint failed", R.id).c_str());
                return Step::error;
            }
            return Step::progressed;
        };
        // ---- phase 3 tail + phase 4 prologue: give the prompt loan back, report the prompt, and
        // set the decode loop up.  Runs for a finished prompt AND for a cancelled one, exactly as
        // 0.1.30's straight-line code did (its `break` fell through to here).
        auto finish_prefill = [&](ReqCtx& R) -> Step {
            std::vector<int64_t>& ids = R.ids;
            const int64_t n = R.n;
            const long long& max_new = R.max_new;
            const Clock::time_point& r0 = R.r0;
            const std::array<int64_t, 6>& loan_bill = R.loan_bill;
            const int64_t& res_uploads0 = R.res_uploads0;
            const int64_t& res_skips0 = R.res_skips0;
            const int64_t& resume = R.resume;
            bool& cancelled = R.cancelled;
            const char*& finish = R.finish;
            if (!refill(err)) {
                std::printf("%s\n", sp_out.err("refilling a lent slot failed: " + err, R.id).c_str());
                return Step::error;
            }
            tr(loan_policy.lazy ? "prompt done (loan returned lazily)" : "prompt done (slots refilled)");
            // S0.3: the loan's bill for THIS request (the PfPart counters are cumulative over the process, so
            // this reads the deltas taken when the request started).  The serve log has no lend/refill timer,
            // which is why the fixed cost could only be inferred from the source; these lines make it
            // measurable, and `bench/prefill/fixed-cost-changes.md` says which column each lever moves.
            // S3.2b adds the two that matter for the floor: `rows out` is what this request took out and did
            // NOT copy back, `pumped` is what the overlapped pump walked home, and `still out` is what the
            // caches are missing right now - the number that says the refill was deferred rather than paid.
            if (loan_timing() && !pf_parts.empty()) {
                const std::array<int64_t, 6> now = loan_totals(pf_parts);
                std::fprintf(stderr, "strata serve: loan request: relayout %lld, relayout skipped %lld, "
                                     "loan grown %lld, rows refilled %lld, rows out %lld, pumped %lld, "
                                     "still out %lld, res upload %lld (%lld skipped), loan return %s\n",
                             (long long) (now[0] - loan_bill[0]), (long long) (now[1] - loan_bill[1]),
                             (long long) (now[2] - loan_bill[2]), (long long) (now[3] - loan_bill[3]),
                             (long long) (now[4] - loan_bill[4]), (long long) (now[5] - loan_bill[5]),
                             (long long) loan_outstanding(pf_parts),
                             (long long) (res_dirty.uploads() - res_uploads0),
                             (long long) (res_dirty.skipped() - res_skips0),
                             loan_policy.lazy ? "lazy" : "eager");
            }
            R.prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();
            std::printf("%s\n", sp_out.reused(resume, req_id).c_str());   // the prompt is read; the first window comes next
            slot_step(req_id, strata::program::slot::State::decoding);
            if (tagged) {
                std::lock_guard<std::mutex> lk(slot_mu);
                if (strata::program::slot::Slot* sl = slots_reg.find(req_id)) {
                    sl->reused = resume;
                    sl->ctx_used = n;
                    slots_reg.set_active(req_id);      // §5.3: this slot is what the live session reflects
                }
            }
            std::fflush(stdout);
            // the verify windows: the first holds the last prompt token alone
            R.p = n - 1;
            R.x = (int32_t) ids[(size_t) (n - 1)];
            R.drafts.assign((size_t) S, 0); R.window.assign((size_t) S, 0); R.outv.assign((size_t) S, 0);
            R.dprob.assign((size_t) S, 0.0f);
            R.sbuf.assign((size_t) S, 0);
            if (o.suffix_draft > 0) {
                sfx.reset();
                sfx_hist.clear();
                for (int64_t t : ids) { sfx.append((int32_t) t); if (tagged) sfx_hist.push_back((int32_t) t); }
            }
            R.first_window = true;
            R.produced_n = 0; R.sfx_windows = 0; R.sfx_drafts = 0; R.sfx_ok = 0;
            R.draft_offered = 0; R.draft_accepted = 0;
            // what the session holds once this request is done: the prompt read so far, then every committed token
            std::vector<int32_t>& consumed = R.consumed;
            consumed.clear(); consumed.reserve((size_t) (n + max_new + S));
            for (int64_t i = 0; i < n - 1; ++i) consumed.push_back((int32_t) ids[(size_t) i]);
            R.finish = "length";
            R.d0 = Clock::now();
            // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
            R.ds0 = dec_snap();
            R.dt_run = 0; R.dt_commit = 0; R.dt_draft = 0;
            R.dec_windows = 0; R.dec_T = 0;
            R.decode_hits0 = drive.d.cache_hits;
            R.decode_look0 = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            if (cancelled) finish = "cancel";
            return Step::finished;
        };
        // ---- phase 4: ONE verify window (draft, verify, commit, emit, re-draft).  0.1.30 looped here
        // until the request was done; the serial loop calls this until it stops reporting
        // `progressed`, which is the same windows in the same order.  This is what S3.1e-2 interleaves
        // between slots, and it is also the natural pre-emption point (a window is ~16 ms).
        //
        // Nothing here allocates per token: the window/draft/probability buffers live in `ReqCtx` and
        // are sized once by `finish_prefill`.
        auto run_decode_step = [&](ReqCtx& R) -> Step {
            const long long& max_new = R.max_new;
            const double& req_spec_min_p = R.req_spec_min_p;
            const int& hist_n = R.hist_n;
            int64_t& p = R.p;
            int32_t& x = R.x;
            std::vector<int32_t>& drafts = R.drafts;
            std::vector<int32_t>& window = R.window;
            std::vector<int32_t>& outv = R.outv;
            std::vector<int32_t>& sbuf = R.sbuf;
            std::vector<float>& dprob = R.dprob;
            std::vector<int32_t>& consumed = R.consumed;
            bool& first_window = R.first_window;
            bool& cancelled = R.cancelled;
            int64_t& produced_n = R.produced_n;
            int64_t& sfx_windows = R.sfx_windows;
            int64_t& sfx_drafts = R.sfx_drafts;
            int64_t& sfx_ok = R.sfx_ok;
            int64_t& draft_offered = R.draft_offered;
            int64_t& draft_accepted = R.draft_accepted;
            const char*& finish = R.finish;
            double& dt_run = R.dt_run;
            double& dt_commit = R.dt_commit;
            double& dt_draft = R.dt_draft;
            int64_t& dec_windows = R.dec_windows;
            int64_t& dec_T = R.dec_T;
            // ---- S3.10: ONE TURN = UP TO --decode-tokens TOKENS, not one window. ------------------
            // A turn used to be a single verify window: with MTP that accepts 1..8 tokens, so a slot
            // emitted ~1-3 tokens and the scheduler paid a full save+restore (risk R10: 237 MB-2.25 GB
            // of memcpy) to move to the next conversation.  `--decode-tokens N` makes the turn a token
            // budget instead, which is the unit the owner asked for.  The loop below is 0.1.30's decode
            // body verbatim; the only additions are the guard, the per-turn budget check and the
            // `turn_windows` counter.  With N = 0/1 it runs exactly ONE iteration, so the default path is
            // today's path by construction.
            const int64_t budget = o.decode_tokens;
            const int64_t produced_at_turn_start = produced_n;
            int64_t turn_windows = 0;
            for (;;) {
            // 0.1.30's `while (!cancelled && produced_n < max_new)` guard, plus the turn's budget.
            // `decode_step_tokens` is that same cap stated once: 0 means this request owes no more
            // tokens, so no window may run.
            if (cancelled ||
                strata::program::serve_driver::decode_step_tokens(budget, produced_n, max_new) == 0)
                return Step::finished;
            // The FIRST window of a turn always runs (a turn that runs nothing is a scheduler stall).
            // After that, the turn ends as soon as it has produced `--decode-tokens` tokens.
            if (turn_windows > 0 &&
                strata::program::serve_driver::decode_turn_done(budget, produced_at_turn_start, produced_n))
                return Step::progressed;   // the turn bought its tokens: hand the session back
            ++turn_windows;
            int T = S_mtp;
            if (req_spec_min_p > 0.0) {
                T = 1;
                while (T < S_mtp && dprob[(size_t) T - 1] >= (float) req_spec_min_p) ++T;
            }
            if (first_window) T = 1;
            // a repeat of earlier context (prompt lookup) where the MTP's own first guess agrees: the policy takes it
            // when its expected tokens per ms, from the measured acceptance and window costs, beat the MTP window's
            bool from_sfx = false;
            int sfx_match = 0;
            if (o.suffix_draft > 0 && !first_window) {
                const int k = sfx.propose(S - 1, sbuf.data());
                sfx_match = sfx.last_match();
                if (k > 0 && sbuf[0] == drafts[0]) {
                    const strata::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                    if (pk.lookup) { T = pk.t; from_sfx = true; }
                }
            }
            const bool timed_round = !first_window;
            const Clock::time_point round0 = Clock::now();
            if (p + T > o.max_context) return Step::finished;
            window[0] = x;
            for (int i = 1; i < T; ++i) window[(size_t) i] = from_sfx ? sbuf[(size_t) i - 1] : drafts[(size_t) i - 1];
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            // ---- S3.2b: the two residency gates every window path must pass, IN THIS ORDER. ----------
            // 1. the pump first: `settle_pump()` marks the rows whose copies have landed RESIDENT in
            //    `host_res`, and queues the next bounded batch on the loan's own stream.  Both are off the
            //    window's critical path (a non-blocking stream, one batch per cache at a time, and a cache
            //    whose loan is live is skipped entirely).
            if (!pump_loans(err)) {
                std::printf("%s\n", sp_out.err("walking the prompt loan home failed: " + err, R.id).c_str());
                return Step::error;
            }
            // 2. then the table.  (I2): every device's `d_res` equals `host_res` at the start of every
            // window.  Unconditional, and deliberately AFTER the pump marked whatever landed - an upload
            // before it would leave the device told a row is a CPU miss while the host already sends it to
            // the GPU, and the two halves of the window would disagree about the same row.  `res_upload()`
            // decides by CONTENT, so when nothing moved this is a 96 KiB compare and no copy, and it cannot
            // be skipped because some writer forgot to announce itself.  This is the code path that enforces
            // "a decode window never reads a slot that does not hold its expert": the rows still out of the
            // cache are marked non-resident in the table the window reads, so the window routes them to the
            // CPU pool instead of to a prompt buffer.
            reconcile_residency();
            apply_pending(false);
            if (hist_n > 0) {
                // the tails the penalties count over, ONE PER ROW: the tokens the state has consumed, the
                // fed-back head `x` (it joins `consumed` only after this window commits), then the drafts
                // before that row - what plain decode would have counted there.  (Until 0.1.19 only row 0
                // was staged, and the drafted rows read unwritten slots.)
                strata::kernels::penalty_rows(consumed.data(), (int64_t) consumed.size(), window.data(), T,
                                              hist_n, hist_stage.data());
                const strata::core::OnDevice on_h(hist_dev);
                cudaMemcpy(d_hist, hist_stage.data(), (size_t) T * (size_t) hist_n * sizeof(int32_t),
                           cudaMemcpyHostToDevice);
            }
            tr("window", p, T);
            const Clock::time_point tw0 = Clock::now();
            if (!ver.run(T, window.data(), p, win_pool_fn, win_pool_user, outv.data(), err) || drive.d.failed) {
                std::printf("%s\n", sp_out.err(drive.d.failed && drive.d.fail ? drive.d.fail : err, R.id).c_str());
                return Step::error;
            }
            int a = 0;
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
            const Clock::time_point tw1 = Clock::now();
            std::thread adapt_thr;   // the adaptive tier beside the commit and the draft (as in generate)
            bool adapt_ok = true;
            if (!drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0)
                adapt_thr = std::thread([&] { adapt_ok = adapt(); });
            if (!ver.commit(a + 1, err)) {
                if (adapt_thr.joinable()) adapt_thr.join();
                std::printf("%s\n", sp_out.err(err, R.id).c_str());
                return Step::error;
            }
            // the window's first a + 1 tokens are in the session now (the last output is not: it is next x)
            for (int i = 0; i <= a; ++i) consumed.push_back(window[(size_t) i]);
            draft_offered += T - 1;
            draft_accepted += a;
            first_window = false;
            bool eos = false;
            for (int i = 0; i <= a && produced_n < max_new && !eos; ++i) {
                std::printf("%s\n", sp_out.token(outv[(size_t) i], req_id).c_str());
                strata::core::progress_beat();
                ++produced_n;
                if (o.suffix_draft > 0) { sfx.append(outv[(size_t) i]); if (tagged) sfx_hist.push_back(outv[(size_t) i]); }
                eos = std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
            }
            std::fflush(stdout);
            // The slot's counters move once per WINDOW, not once per token: the decode path takes no lock
            // and allocates nothing (the contract on include/strata/program/slot.hpp).
            if (tagged) {
                std::lock_guard<std::mutex> lk(slot_mu);
                if (strata::program::slot::Slot* sl = slots_reg.find(req_id)) {
                    sl->generated = produced_n;
                    sl->ctx_used = p + a + 1;
                }
            }
            ++rounds;
            const Clock::time_point tw2 = Clock::now();
            // coupled drafts with penalties: the next window's row-0 history (`consumed` holds this window's
            // commit, outv[a] is its row 0) - the drafts extend it on the device as the verify rows will
            if (hist_n > 0 && mtp.coupled() && !eos && produced_n < max_new)
                mtp.set_draft_history(consumed.data(), (int64_t) consumed.size(), outv[(size_t) a]);
            const bool drafted = eos || produced_n >= max_new ||
                                 mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(), (float) req_spec_min_p);
            {
                const Clock::time_point tw3 = Clock::now();
                auto msd = [](Clock::time_point a0, Clock::time_point b0) { return std::chrono::duration<double, std::milli>(b0 - a0).count(); };
                dt_run += msd(tw0, tw1); dt_commit += msd(tw1, tw2); dt_draft += msd(tw2, tw3);
                ++dec_windows; dec_T += T;
            }
            if (adapt_thr.joinable()) adapt_thr.join();
            if (!adapt_ok) {
                std::printf("%s\n", sp_out.err("an adaptive refill failed", R.id).c_str());
                return Step::error;
            }
            if (!drafted) {
                std::printf("%s\n", sp_out.err(err, R.id).c_str());
                return Step::error;
            }
            if (timed_round && !eos)
                policy.observe(from_sfx, T, a, sfx_match,
                               std::chrono::duration<double, std::milli>(Clock::now() - round0).count());
            if (eos) { finish = "stop"; return Step::finished; }
            if (stopped()) { finish = "cancel"; return Step::cancelled; }
            x = outv[(size_t) a];
            p += a + 1;
            // S3.10: the window is done.  With a budget this loops and runs the NEXT window for the
            // SAME slot (the drafts it just made are already in `drafts`, `x`/`p` are the next row), so
            // the slot keeps the session until it has produced `--decode-tokens` tokens.  With the
            // default budget of 0/1 the guard at the top of the loop returns here instead, which is
            // 0.1.30's single window per step.
            }   // S3.10: end of one decode TURN (one or more verify windows)
        };
        // ---- phase 5: the request's tail - the decode-timing report, the conversation state this
        // request leaves behind, the per-request metrics and the DONE line.  0.1.30 ran it inline at
        // the end of the body; S3.1e-2 calls it when a slot's decode step reports it is done.
        auto finish_request = [&](ReqCtx& R) -> Step {
            const int64_t n = R.n;
            const int64_t& resume = R.resume;
            const double& prompt_ms = R.prompt_ms;
            bool& cancelled = R.cancelled;
            std::vector<int32_t>& consumed = R.consumed;
            int64_t& produced_n = R.produced_n;
            int64_t& sfx_windows = R.sfx_windows;
            int64_t& sfx_drafts = R.sfx_drafts;
            int64_t& sfx_ok = R.sfx_ok;
            int64_t& draft_offered = R.draft_offered;
            int64_t& draft_accepted = R.draft_accepted;
            const char*& finish = R.finish;
            const Clock::time_point& d0 = R.d0;
            const DecSnap& ds0 = R.ds0;
            double& dt_run = R.dt_run;
            double& dt_commit = R.dt_commit;
            double& dt_draft = R.dt_draft;
            int64_t& dec_windows = R.dec_windows;
            int64_t& dec_T = R.dec_T;
            const int64_t& decode_hits0 = R.decode_hits0;
            const int64_t& decode_look0 = R.decode_look0;
            const double& decode_ms = R.decode_ms;
            std::array<int64_t, 3>& remote_before = R.remote_before;
            std::array<int64_t, 3>& launches_before = R.launches_before;
            std::array<uint64_t, 3>& compact_before = R.compact_before;
            std::array<uint64_t, 3>& full_before = R.full_before;
            std::array<double, 3>& begin_before = R.begin_before;
            std::array<double, 3>& wait_before = R.wait_before;
            R.decode_ms = std::chrono::duration<double, std::milli>(Clock::now() - d0).count();
            if (dec_timing && dec_windows > 0) {
                const DecSnap d1 = dec_snap();
                const double w = (double) dec_windows, L = (double) g.n_layers;
                std::fprintf(stderr, "strata decode timing: %lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = "
                                     "verify %.2f (GPU-reach wait %.2f + per-layer host %.2f [plan %.2f actq %.2f jobs %.2f "
                                     "CPU %.2f] + stage %.2f) + commit/emit %.2f + draft %.2f; per layer-window: CPU experts "
                                     "%.2f (%.2f entries), VRAM hits %.2f, PCIe %.2f\n",
                             (long long) dec_windows, dec_T / w, produced_n / w, decode_ms / w, dt_run / w,
                             (d1.wait - ds0.wait) / w, (d1.pool - ds0.pool) / w, (d1.plan - ds0.plan) / w,
                             (d1.actq - ds0.actq) / w, (d1.jobs - ds0.jobs) / w, (d1.run - ds0.run) / w,
                             (d1.host - ds0.host) / w, dt_commit / w, dt_draft / w, (d1.misses - ds0.misses) / (w * L),
                             (d1.entries - ds0.entries) / (w * L), (d1.hits - ds0.hits) / (w * L), (d1.pcie - ds0.pcie) / (w * L));
                const std::string pr = ver.profile_report();
                if (!pr.empty()) std::fprintf(stderr, "strata decode GPU stages (ms/window):%s\n", pr.c_str());
            }
            if (!cancelled) {
                // a prompt stopped halfway leaves the session somewhere between two chunks: nothing to continue from
                // (the checkpoints taken while reading it are still good)
                live.swap(consumed);
                live_imgs = imgs_below(req_imgs, (int64_t) live.size());
                live_ok = o.prompt_cache > 0;
            }
            // S3.1d: the mounted slot's mirror follows the session, so the NEXT hand-over saves what
            // this request actually left behind rather than what it started with.
            if (swaps_on) sync_conversation();
            static const bool state_hash = std::getenv("STRATA_STATE_HASH") != nullptr;
            if (state_hash && live_ok) {
                // DEBUG: a fingerprint of every part of the session over the positions it holds ([0, L)), and
                // separately of what lies past them in the last KV page (stale cells, fine unless something reads them)
                if (cudaDeviceSynchronize() != cudaSuccess) {
                    std::printf("%s\n", sp_out.err("synchronizing state fingerprint", R.id).c_str());
                    return Step::error;
                }
                const int64_t L = (int64_t) live.size();
                const strata::kernels::QsaShapes qs = [&] {
                    strata::kernels::QsaShapes s = strata::kernels::qsa_real_shapes();
                    s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_dim = g.idx_key_dim;
                    return s;
                }();
                bool hash_ok = true;
                std::array<uint8_t, 65536> hash_buffer;
                auto hash_dev = [&](const void* p, size_t bytes, uint64_t h) {
                    for (size_t offset = 0; hash_ok && offset < bytes;) {
                        const size_t n = std::min(hash_buffer.size(), bytes - offset);
                        // VRAM or a streamed host copy, with fixed diagnostic workspace.
                        if (cudaMemcpy(hash_buffer.data(), static_cast<const uint8_t*>(p) + offset, n, cudaMemcpyDefault) != cudaSuccess) {
                            hash_ok = false;
                            break;
                        }
                        h = fnv1a(hash_buffer.data(), n, h);
                        offset += n;
                    }
                    return h;
                };
                // the cells [c0, c1) of one int8 K or V pool ([page][kv_head][page_size][head_dim]), bytes per value `w`
                auto hash_cells = [&](const void* pool, int64_t per_cell, int64_t c0, int64_t c1, uint64_t h) {
                    const int64_t ps = qs.page_size;
                    for (int64_t pg = c0 / ps; pg * ps < c1; ++pg)
                        for (int64_t hd = 0; hd < qs.n_head_kv; ++hd) {
                            const int64_t a = std::max(c0, pg * ps) - pg * ps, e = std::min(c1, (pg + 1) * ps) - pg * ps;
                            const size_t off = (size_t) (((pg * qs.n_head_kv + hd) * ps + a) * per_cell);
                            h = hash_dev((const uint8_t*) pool + off, (size_t) ((e - a) * per_cell), h);
                        }
                    return h;
                };
                const ConvStateSizes z = conv_state_sizes(g, ss);
                uint64_t h_gdn = hash_dev(ss.gdn_state, z.gdn, 1469598103934665603ull);
                if (std::getenv("STRATA_STATE_HASH_GDN") != nullptr && ss.gdn_alloc > 0) {   // per GDN layer: which one differs first
                    const size_t per = z.gdn / (size_t) ss.gdn_alloc;
                    std::string s;
                    char b[8];
                    for (int64_t i = 0; i < ss.gdn_alloc; ++i) {
                        std::snprintf(b, sizeof(b), "%04llx ", (unsigned long long) (hash_dev((const uint8_t*) ss.gdn_state + i * per, per, 1469598103934665603ull) & 0xffff));
                        s += b;
                    }
                    std::fprintf(stderr, "strata serve: STATE_HASH_GDN %s\n", s.c_str());
                }
                uint64_t h_ple = hash_dev(ss.ple_hist, ss.ple_hist ? z.ple : 0, 1469598103934665603ull);
                uint64_t h_tail = 1469598103934665603ull, h_pool = h_tail, h_kv = h_tail, h_stale = h_tail;
                // pooled= keeps its 0.1.29 meaning: the completed rows [0, L / idx_block) only.  pooled_full= adds the
                // spare row at L / idx_block (the `dead` key the next block completion overwrites), which a
                // conversation restore writes back; dead= is the spare key itself
                uint64_t h_dead = h_tail, h_pool_full = h_tail;
                const int64_t kvb = qs.head_dim, scb = (qs.head_dim / 64) * 2;
                // a state's K/V arrays and their bytes per (cell, head) row: the host copy when it has one
                auto kv_arrays = [&](const strata::core::QsaState& st) {
                    const bool h = st.kv_mode != 0;
                    std::vector<std::pair<const void*, int64_t>> a;
                    if (st.kv_q4) {
                        const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                        a = {{h ? st.host.k_q4 : st.k_q4, q4b}, {h ? st.host.v_q4 : st.v_q4, q4b}};
                    } else if (st.kv_hybrid) {
                        const int64_t q4b = (int64_t) strata::kernels::kv_q4_bytes_per_head((int) qs.head_dim);
                        a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q4 : st.v_q4, q4b},
                             {h ? st.host.k_scale : st.k_scale, scb}};
                    } else if (st.kv_int8) {
                        a = {{h ? st.host.k_q : st.k_q, kvb}, {h ? st.host.v_q : st.v_q, kvb},
                             {h ? st.host.k_scale : st.k_scale, scb}, {h ? st.host.v_scale : st.v_scale, scb}};
                    } else {
                        a = {{h ? st.host.k_pool : st.k_pool, qs.head_dim * 2},
                             {h ? st.host.v_pool : st.v_pool, qs.head_dim * 2}};
                    }
                    return a;
                };
                const int64_t end_cell = std::min<int64_t>(((L + qs.page_size - 1) / qs.page_size) * qs.page_size,
                                                           ss.max_cells);   // = the primary state's max_cells
                for (int64_t j = 0; j < ss.qsa_alloc; ++j) {   // this session's owned QSA ordinals only
                    const strata::core::QsaState& st = ss.qsa_states[ss.qsa_ord0 + j];
                    h_tail = hash_dev(st.idx_tail, z.tail, h_tail);
                    h_dead = hash_dev(st.idx_dead, z.dead, h_dead);
                    h_pool = hash_dev(st.idx_pooled, (size_t) (L / qs.idx_block) * qs.idx_dim * 4, h_pool);
                    h_pool_full = hash_dev(st.idx_pooled, (size_t) (L > 0 ? L / qs.idx_block + 1 : 0) * qs.idx_dim * 4,
                                           h_pool_full);
                    // KV streaming: the host copy is the identity layout and holds every cell
                    for (const auto& [pool, w] : kv_arrays(st)) {
                        h_kv = hash_cells(pool, w, 0, L, h_kv);
                        h_stale = hash_cells(pool, w, L, end_cell, h_stale);
                    }
                }
                const strata::core::QsaState& ms = mtp.kv_state();
                uint64_t h_mtp = 1469598103934665603ull;
                const int64_t mL = std::min<int64_t>(L, ms.max_cells);
                for (const auto& [pool, w] : kv_arrays(ms))
                    if (pool != nullptr) h_mtp = hash_cells(pool, w, 0, mL, h_mtp);
                if (!hash_ok) {
                    std::printf("%s\n", sp_out.err("reading state fingerprint", R.id).c_str());
                    return Step::error;
                }
                std::fprintf(stderr, "strata serve: STATE_HASH L=%lld gdn=%016llx ple=%016llx tail=%016llx pooled=%016llx "
                                     "kv=%016llx mtp=%016llx stale=%016llx dead=%016llx pooled_full=%016llx ple_prev=%d,%d\n", (long long) L,
                             (unsigned long long) h_gdn, (unsigned long long) h_ple, (unsigned long long) h_tail,
                             (unsigned long long) h_pool, (unsigned long long) h_kv, (unsigned long long) h_mtp,
                             (unsigned long long) h_stale, (unsigned long long) h_dead,
                             (unsigned long long) h_pool_full, ss.ple_prev[0], ss.ple_prev[1]);
            }
            const int64_t req_hits = drive.d.cache_hits - decode_hits0;
            const int64_t req_look = (drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused) - decode_look0;
            if (tagged) {
                std::lock_guard<std::mutex> lk(slot_mu);
                if (strata::program::slot::Slot* sl = slots_reg.find(req_id)) {
                    sl->prompt_ms = (int64_t) prompt_ms;
                    sl->decode_ms = (int64_t) decode_ms;
                    sl->generated = produced_n;
                    sl->reused = resume;
                    sl->ctx_used = n + produced_n;
                }
                slots_reg.note_ran(req_id, (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                    Clock::now().time_since_epoch()).count(), (int64_t) (prompt_ms + decode_ms));
            }
            // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused> [hits] [lookups]
            std::printf("%s\n", sp_out.done(produced_n, n, prompt_ms, decode_ms, finish,
                                            draft_accepted, draft_offered, resume, req_hits, req_look,
                                            req_id).c_str());
            std::fflush(stdout);
            if (drive.routing != nullptr) std::fflush(drive.routing);   // the routing trace survives a crash and is watchable mid-session
            const int64_t fresh = n - resume;
            std::fprintf(stderr, "strata serve: prompt %lld tokens = %lld reused + %lld read in %.0f ms (%.1f tok/s), "
                                 "%lld generated in %.0f ms (%.1f tok/s), drafts accepted %lld of %lld, %zu checkpoints%s\n",
                         (long long) n, (long long) resume, (long long) fresh, prompt_ms,
                         prompt_ms > 0 ? 1000.0 * fresh / prompt_ms : 0.0, (long long) produced_n, decode_ms,
                         decode_ms > 0 ? 1000.0 * produced_n / decode_ms : 0.0, (long long) draft_accepted,
                         (long long) draft_offered, checks.size(), cancelled ? " (cancelled)" : "");
            // the VRAM share of the experts the pool looked up while decoding; experts it sent over PCIe for the GPU
            // to read (--pcie-frac) are in neither count
            if (req_look > 0) {
                std::fprintf(stderr, "strata serve: decode expert cache hit rate: %.1f%% (%lld hits / %lld lookups)\n",
                             100.0 * (double) req_hits / (double) req_look,
                             (long long) req_hits, (long long) req_look);
            }
            // the resident RAM mode, cumulative: experts read from experts.bin since the copy was made (what the plain
            // mmap mode reads through the OS file cache, from the SSD when the RAM could not keep it)
            if (src.complement_ready())
                std::fprintf(stderr, "strata serve: resident RAM: %.2f GiB of experts in RAM, %lld exchanged with the "
                                     "VRAM tier, %lld blob reads from the file\n",
                             (double) src.resident_bytes() / 1073741824.0, (long long) src.exchanges(),
                             (long long) src.file_reads());
            // STRATA_SPLIT_TIMING: where each verify stage's host time went, cumulative per window since the start
            // (waiting for its GPU to ring a layer, the CPU pool and plan per layer, staging the window)
            if (static const bool st_timing = std::getenv("STRATA_SPLIT_TIMING") != nullptr; st_timing)
                for (int st = 0; st < n_stages; ++st) {
                    const strata::core::Verifier& v = stage_ver(st);
                    const double w = v.windows > 0 ? (double) v.windows : 1.0;
                    std::fprintf(stderr, "strata serve: stage %d: %lld windows; per window: wait for the GPU %.3f ms, "
                                         "pool + plan %.3f ms, host staging %.3f ms, commit %.3f ms\n", st,
                                 (long long) v.windows, v.ms_wait / w, v.ms_pool / w, v.ms_host / w, v.ms_commit / w);
                }
            if (g.n_qsa_layers() > 0 && ss.qsa_states[ss.qsa_primary()].kv_mode == 1) {
                // KV streaming, cumulative over the process: blocks the selections named vs blocks read from RAM
                // (this device's owned ordinals; a split's other stages hold theirs)
                uint64_t miss = 0, look = 0;
                bool over = false;
                for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
                    const strata::kernels::KvStreamCounters c =
                        strata::kernels::kv_stream_counters(ss.qsa_states[ss.qsa_ord0 + j].map);
                    miss += c.misses; look += c.lookups; over = over || c.overflow;
                }
                std::fprintf(stderr, "strata serve: KV streaming: %.2f%% of %llu block reads hit VRAM, %.1f MiB read "
                                     "from RAM%s\n", look ? 100.0 * (double) (look - miss) / (double) look : 100.0,
                             (unsigned long long) look, (double) miss * 4224.0 / 1048576.0,
                             over ? " - OVERFLOW (too few resident cells)" : "");
            }
            if (sfx_windows > 0)
                std::fprintf(stderr, "strata serve: suffix drafts: %lld windows, %lld of %lld drafts accepted\n",
                             (long long) sfx_windows, (long long) sfx_ok, (long long) sfx_drafts);
            for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
                std::fprintf(stderr, "strata serve: CUDA%d: %lld expert entries, %lld active layer launches, %.1f MiB returned "
                                     "(%.1f MiB with full rows) in this request; host %.0f ms staging+launching, %.0f ms "
                                     "waiting for it in this request\n", r + 1,
                             (long long) (remote_experts[(size_t) r].computed() - remote_before[(size_t) r]),
                             (long long) (remote_experts[(size_t) r].launched_layers() - launches_before[(size_t) r]),
                             (double) (remote_experts[(size_t) r].returned_bytes() - compact_before[(size_t) r]) / 1048576.0,
                             (double) (remote_experts[(size_t) r].full_row_bytes() - full_before[(size_t) r]) / 1048576.0,
                             remote_experts[(size_t) r].ms_begin() - begin_before[(size_t) r],
                             remote_experts[(size_t) r].ms_wait() - wait_before[(size_t) r]);
            return Step::finished;
        };

        // ==================== S3.1e-2: THE CONCURRENT DRIVER ====================
        //
        // docs/STAGE3-CONCURRENCY.md §3.1: one engine thread, N slots, ONE step at a time.  This block
        // runs only when the registry says concurrency is on AND the slot hand-over is available, and it
        // RETURNS before the serial driver below.  The gate is an early return rather than an `if`
        // wrapped around the old loop, so the serial driver's text is byte-identical to 0.1.30's:
        // §7.1's bit-exactness bar becomes a fact about the source, not a claim about behaviour.
        //
        // WHY `&& swaps_on`.  A slot switch IS a save/restore (§5.3).  With STRATA_NO_SWAP=1 the
        // hand-over is disabled, so two conversations would share one session with nothing moving
        // between them - plausible garbage, not concurrency.  S3.1d's promise is that the env switch
        // "falls back to today's serial behaviour without recompiling", and that is what this does: the
        // tagged wire stays on (S3.1c), the interleaving does not.
        //
        // WHAT A STEP IS (§3.2).  `run_prefill_step` = one prompt segment (8-15 s at 8 192 tokens),
        // `run_decode_step` = one verify window (~16-24 ms).  Both pre-emption points already existed in
        // 0.1.30's code; the scheduler's whole trick is to stop calling them in a `while`.
        //
        // ReqCtx LIFETIME - THE HARD PART.  The serial path builds one `ReqCtx` on the loop's stack and
        // destroys it at the end of the iteration.  A step cannot: the state must outlive the step, so
        // the driver owns `std::deque<ReqCtx> live`, keyed by request id.  A deque and not a vector
        // because a reference into it must survive the next `push_back` - the same reason `conv_slots`
        // is a deque.  Destruction order, checked against the serial path's (mrope scope -> slot guard
        // -> ReqCtx -> busy scope):
        //   * `MropeScope` and `StepGuard` are armed per STEP inside an inner scope of `run_one` and
        //     destroyed when that scope closes - strictly EARLIER than the serial path destroyed them,
        //     and never while a `ReqCtx` is missing (the guard's `ctx` always points at the live
        //     context).  `drop_ctx()` - the only thing that destroys a context - runs after that scope.
        //   * `StepGuard` is 0.1.30's `SlotGuard` plus one condition: it releases the row only once
        //     `R.finished` is set.  Arming the serial guard per step would end every request after its
        //     first step; arming it per request is impossible, because the request outlives the step.
        //     `SlotGuard` itself is NOT touched, so the serial path's guard is unchanged.
        //   * `MropeScope` keeps the serial path's exact `skip` predicate (`!swaps_on ||
        //     !R.mrope_touched || mounted_id == req_id`), so on a normal step it does nothing and on a
        //     bail-out it restores the mounted slot's positions - which is what R7 needs MORE with N
        //     slots, not less.
        //   * `cur`/`live`/`checks`/`row_ptr`/`img_rows`/`mrope_host` stay serve-scope, so nothing about
        //     when a conversation image or a CUDA buffer exists changes.  `sp.embd_rows` is re-pointed
        //     before every step, and R7's exclusivity is enforced at ADMISSION (below) rather than
        //     discovered in the middle of a read.
        //   * `progress().busy` is held for the whole driver, as `BusyScope` held it for a serial
        //     iteration, and released while the driver blocks for a line, so an idle engine is never
        //     reported as stalled.
        //
        // THE INVARIANTS THIS LOOP ENFORCES.  Each is a predicate in
        // include/strata/program/serve_driver.hpp and each is pinned by src/program/serve_driver_test.cpp:
        //   1. a step runs only for the slot the live session reflects (§5.3) - `dispatch_step` answers
        //      `swap`, never a step, for anything else;
        //   2. the prompt loan has ONE owner (R8): `loan`, mirrored by `R.loan_held`, and a slot waiting
        //      for it is DEFERRED rather than re-picked forever;
        //   3. at most one `swap_to` per iteration, so a refused hand-over cannot spin;
        //   4. an admitted slot holds an active-slot permit and every exit path releases it - a stranded
        //      permit is the one bug that would deadlock admission, because the permit bounds everything;
        //   5. the watchdog sees one heartbeat per ACTIVE slot (R3) and the driver drains its verdicts;
        //   6. at most one live request holds the one M-RoPE table and the one image-row pointer set
        //      (R7), enforced at admission.
        // The runtime form of the startup check: if the parking budget resolved to 0 after all (the
        // machine-sized default, or a budget that only closed once the model was loaded), the swap path
        // has nothing to save into, so interleaving would be two conversations over one session.  Fall
        // back to the serial driver - which is exactly S3.1c's state: the tagged wire, one request at a
        // time - and say why, on stderr, where the owner will see it.
        const bool driver_on = slots_reg.concurrent() && swaps_on &&
                               !strata::program::serve_driver::parking_off_refuses_slots(
                                   slots, conversations.enabled());
        if (slots_reg.concurrent() && !driver_on) {
            // S3.1e-2-FALLBACK-MSG-BEGIN (runtime-gated on slots_reg.concurrent(), i.e. slots >= 2)
            std::fprintf(stderr, "strata serve: --serve-slots %d is NOT running concurrently: %s - the serial "
                                 "driver is serving these requests one at a time on the tagged wire "
                                 "(S3.1c's behaviour)\n",
                         slots, swaps_on ? "the conversation cache is off, so a slot switch would have nothing "
                                           "to save"
                                         : "STRATA_NO_SWAP=1 disabled the slot hand-over");
            // S3.1e-2-FALLBACK-MSG-END
        }
        if (driver_on) {
            namespace drv = strata::program::serve_driver;
            namespace proto = strata::program::serve_proto;
            using DStep = strata::program::serve_driver::Step;
            using PAct = strata::program::slot::Pick::Action;
            using SState = strata::program::slot::State;
            // S3.6: ONE switch for every scheduler decision the owner asked to see - the pick and its
            // `why`, each phase transition, each swap's reason, and each loan/park deferral.  Off by
            // default so the normal log stays readable; the parking numbers and the re-read/resume
            // events are unconditional because they are one line per decision and they are the ones
            // that explain a request that ended early.
            const bool serve_trace = strata::program::serve_driver::serve_trace_on();
            if (serve_trace)
                std::fprintf(stderr, "strata serve: STRATA_SERVE_TRACE on - one line per pick, phase "
                                     "transition, swap and deferral\n");
            // ---- the driver's own state --------------------------------------------------------
            std::deque<ReqCtx> live;                       // one per admitted request, keyed by R.id
            auto ctx_of = [&](int64_t id) -> ReqCtx* {
                for (ReqCtx& R : live) if (R.id == id) return &R;
                return nullptr;
            };
            drv::Loan loan;                                // R8: the one prompt loan, and who holds it
            // A line that has been read and passed admission but has not been brought into the
            // registry yet.  The ROW is created by `prep_request` (S3.1c/S3.1d own that: it is where
            // the id is checked against the registry, the pending cancels are applied and the SLOT
            // line goes out), so the driver cannot create it earlier without restructuring the step
            // functions - which is not this phase's file to change.  It is tracked here instead, and
            // it counts against the active cap: admitting ten contexts on a two-slot engine and then
            // failing their transition to `prefilling` would leave ten requests running with no
            // permit, which is the one accounting error that makes --serve-slots meaningless.
            auto pending_ctx = [&]() -> ReqCtx* {
                for (ReqCtx& Q : live) if (Q.has_line && Q.phase == drv::Phase::queued) return &Q;
                return nullptr;
            };
            auto pending_count = [&]() {
                int n = 0;
                for (const ReqCtx& Q : live) if (Q.has_line && Q.phase == drv::Phase::queued) ++n;
                return n;
            };
            // Slots whose next step needs the loan and cannot have it.  They must not be picked again
            // until the loan is free, or the loop spins on a slot that can never advance while the
            // holder - a different row, later in the registry's array - never gets the session.
            std::deque<int64_t> deferred;
            auto is_deferred = [&](int64_t id) {
                for (const int64_t d : deferred) if (d == id) return true;
                return false;
            };
            auto un_defer = [&](int64_t id) {
                for (size_t i = 0; i < deferred.size(); ++i)
                    if (deferred[i] == id) { deferred.erase(deferred.begin() + (std::ptrdiff_t) i); break; }
            };
            auto clear_deferred = [&]() { deferred.clear(); };
            int64_t admitted = 0, steps_run = 0, swaps_done = 0, refused = 0, deferred_hits = 0;
            bool driver_quit = false, driver_fatal = false;
            // Did THIS pass change anything?  Declared out here because `admit_one` (defined below)
            // sets it.  A pass that changes nothing must not become a CPU spin - see the bottom of
            // the loop.  Named `pass_moved` because `progress()` is the heartbeat accessor.
            bool pass_moved = true;   // true to start: the first pass must not sleep
            // A line read from stdin that could not be admitted yet.  It is kept here rather than put
            // back on the queue so a refusal cannot reorder the client's requests.
            std::string held_line;
            // Non-blocking: the driver may never wait for a line while another slot has work, or
            // concurrency would be one request deep again.
            auto try_next_line = [&](std::string& out) -> bool {
                std::lock_guard<std::mutex> lk(in_mu);
                if (in_lines.empty()) return false;
                out = std::move(in_lines.front());
                in_lines.pop_front();
                return true;
            };
            // ---- the watchdog's per-slot record (R3) -------------------------------------------
            // `progress().beats` is process-wide, so a slot is "current" when the counter moved while
            // that slot was the one running.  A slot whose step began long ago and has not moved the
            // counter is the one that is actually stuck.
            auto watch_begin = [&](int64_t id) {
                const uint64_t b = strata::core::progress().beats.load();
                const int64_t t = drv::now_ms();
                std::lock_guard<std::mutex> lk(watch_mu);
                for (strata::program::serve_driver::SlotWatch& w : watch)
                    if (w.id == id) { w.step_started_ms = t; w.beats_at_start = b; return; }
                watch.push_back(strata::program::serve_driver::SlotWatch{id, t, b});
            };
            auto watch_idle = [&](int64_t id) {
                std::lock_guard<std::mutex> lk(watch_mu);
                for (strata::program::serve_driver::SlotWatch& w : watch)
                    if (w.id == id) { w.step_started_ms = 0; return; }
            };
            auto watch_forget = [&](int64_t id) {
                std::lock_guard<std::mutex> lk(watch_mu);
                for (size_t i = 0; i < watch.size(); ++i)
                    if (watch[i].id == id) { watch.erase(watch.begin() + (std::ptrdiff_t) i); return; }
            };
            // ---- the per-request prompt-path view ----------------------------------------------
            // `sp.embd_rows` is one process-wide pointer into the serve-scope `row_ptr`, and
            // `prep_request` set it for whichever request arrived last.  With N slots the driver has to
            // re-point it before every step, or a text request's step would read an image request's
            // rows.  Safe because admission keeps at most one GENI request live (R7).
            auto arm_prompt_view = [&](ReqCtx& R) {
                sp.embd_rows = R.geni ? row_ptr.data() : nullptr;
            };
            // The segment end the NEXT prefill step will read to, so `windows_ok` can be asked with the
            // same arguments 0.1.30's `for` used.  `run_prefill_step` re-derives it identically.
            auto next_segment_end = [&](const ReqCtx& R) -> int64_t {
                for (size_t i = R.seg_i; i < R.seg.size(); ++i)
                    if (R.seg[i] > R.at) return R.seg[i];
                return -1;
            };
            // Does this slot's NEXT prefill segment need the prompt loan?  The same test
            // `run_prefill_step` applies to itself: a segment that goes through the verify windows
            // refills first, a batched one lends, and "no segment left" touches neither.  Asked
            // before the step so the slot can be DEFERRED rather than run and refused - and asked
            // about the step the slot would actually run, so a cancelled slot (whose dispatch is the
            // phase tail, which never lends) is never parked behind a loan it does not need.
            auto segment_needs_loan = [&](const ReqCtx& R) -> bool {
                for (size_t i = R.seg_i; i < R.seg.size(); ++i) {
                    if (R.seg[i] <= R.at) continue;
                    return !windows_ok(R.at, R.seg[i]);
                }
                return false;
            };
            // Put THIS request's prompt state back where 0.1.30's body left it for the whole request.
            // Four of these are serve-scope counters the PP line and the mid-prompt checkpoints read,
            // and one is not a counter at all: `cur` is the token array `checkpoint_at` copies
            // (`cur[0, L)`) and the prompt path's chunk callback indexes (`cur[p0 + t + 1]`).  A step
            // that left `cur` holding another request's tokens would checkpoint THAT sequence under
            // this request's id - silent, and exactly the kind of plausible-garbage bug the design
            // refuses.  0.1.30 set it once per request (`cur = ids` in `prep_request`); with N slots
            // the driver re-arms it before every prefill step.
            int64_t prompt_owner = proto::kNoId;
            auto arm_prompt_state = [&](ReqCtx& R) {
                pp_total = R.pp_total; pp_from = R.pp_from;
                pp_t0 = R.pp_t0; pp_next_check = R.pp_next_check;
                cur = R.ids;
                // S3.8: and the pictures.  `checkpoint_at` stamps `c.imgs = imgs_below(req_imgs, L)`,
                // and `req_imgs` names whichever request was prepped LAST - so a prefill resumed after
                // a hand-over would stamp ANOTHER conversation's image keys onto this branch's
                // checkpoints.  `ReqCtx::own_imgs` exists precisely to stop that; re-arming `cur`
                // without it left the hole open.  Silent, and it corrupts the cache's prefix
                // comparison for every later request of this chat.
                req_imgs = R.own_imgs;
                {
                    // A layer split's per-stage mid-prompt parts are keyed by POSITION, and two
                    // requests reach the same positions.  When the request being prefilling changes,
                    // whatever the stages had accumulated for the previous one is now a partial set
                    // for a sequence that is not running: drop it.  (An incomplete set is already
                    // handled - `sp.on_chunk` skips the checkpoint rather than saving a mixed one -
                    // but dropping it is what lets the NEXT checkpoint of the new request be real.)
                    std::lock_guard<std::mutex> lk(part_mu);
                    if (prompt_owner != R.id) { part_at.clear(); prompt_owner = R.id; }
                    std::fill(part_next.begin(), part_next.end(), pp_next_check);
                }
            };
            // ---- §3.4's refuse edge: ERR, release the row, free the permit, keep the process -----
            // This is also risk R4's fix: 0.1.30's parking failure was a `return 1`, which took the
            // whole server down over one conversation.
            auto refuse_line = [&](int64_t id, const std::string& why) {
                // A row that belongs to a request the driver is ALREADY running must not be released
                // by a refusal aimed at a new line: the duplicate-id case below is the one way that
                // happens, and killing the healthy request would be worse than the bad line.  Its ERR
                // goes out untagged, so it lands on the server's control queue instead of that
                // request's stream.
                const bool owns_live = ctx_of(id) != nullptr;
                std::printf("%s\n", sp_out.err(why, owns_live ? proto::kNoId : id).c_str());
                std::fflush(stdout);
                if (owns_live) { ++refused; return; }
                std::lock_guard<std::mutex> lk(slot_mu);
                if (slots_reg.find(id) != nullptr) {
                    std::string serr;
                    slots_reg.transition(id, SState::error, serr);
                    slots_reg.release(id);
                }
                ++refused;
            };
            // A slot the driver IS running, ended abnormally: ERR tagged to it, its row destroyed,
            // its permit freed.  Distinct from `refuse_line`, which answers a line that never became a
            // slot (and must therefore NOT touch a row another request owns).
            auto end_slot = [&](int64_t id, const std::string& why) {
                std::printf("%s\n", sp_out.err(why, id).c_str());
                std::fflush(stdout);
                std::lock_guard<std::mutex> lk(slot_mu);
                if (slots_reg.find(id) != nullptr) {
                    std::string serr;
                    slots_reg.transition(id, SState::error, serr);
                    slots_reg.release(id);
                }
                if (mounted_id == id) mounted_id = proto::kNoId;
                ++refused;
            };

            // ---- S4.2 "hold, don't reject" (stage 4, decision D5) -------------------------------
            // The owner's complaint: "if there is no budget to park a conversation or different
            // circumstances requests get rejected with an error. That is bad, instead, those requests
            // should be put on hold and be executed as soon as there are ressources free instead of
            // cancelled."  So a request that cannot run NOW is queued, not ERRed.  Only a request that
            // can NEVER run on this engine is an error, and `serve_driver::admit_decision` is the one
            // place that says which is which.
            //
            // THE RULES, all of them predicates in serve_driver.hpp and pinned by serve_driver_test:
            //   * the queue is FIFO, with one documented exception (a request that already holds a
            //     conversation outranks one that has never run - it has tokens on the wire);
            //   * it is BOUNDED twice: `--hold-ms` (0 = wait forever) and `WaitQueue::cap`, because an
            //     unbounded queue is a memory leak with a nicer name;
            //   * it is VISIBLE: `waiting=N` in the activity line, `WAIT n oldest_ms reason` lines, and
            //     the same numbers on the server's /status and /slots;
            //   * an expired wait is an honest ERR that names what it waited for, never a silent drop;
            //   * `STOP <id>` cancels a waiter and answers it `DONE ... cancel` - it is never run and
            //     then cancelled, which would emit tokens to a client that asked us to stop.
            // The waiter's request LINE lives in `hold_lines` and is NOT removed until the request is
            // actually brought in, so retrying a waiter every pass cannot reorder the queue.
            drv::WaitQueue holds(o.hold_ms);
            std::deque<std::pair<int64_t, std::string>> hold_lines;   // waiter id -> its request line
            auto hold_find_line = [&](int64_t id) -> std::string* {
                for (std::pair<int64_t, std::string>& e : hold_lines)
                    if (e.first == id) return &e.second;
                return nullptr;
            };
            auto hold_forget = [&](int64_t id) {
                for (size_t i = 0; i < hold_lines.size(); ++i)
                    if (hold_lines[i].first == id) { hold_lines.erase(hold_lines.begin() + (std::ptrdiff_t) i); return; }
            };
            // Record why a waiter waits.  `set_reason` returns true only when the reason CHANGED, which
            // is the cue to print `hold_line` once - the driver's pass loop runs thousands of times a
            // second, and a per-pass line would bury every other line in the log.
            auto note_wait = [&](int64_t id, drv::Wait why) {
                if (holds.set_reason(id, why))
                    std::fprintf(stderr, "%s\n", drv::hold_line(id, why, holds.size()).c_str());
            };
            // Put a parsed request into the hold queue (or just refresh its reason if it is already
            // there, which is the case for a waiter retried on a later pass).
            auto hold_enqueue = [&](const proto::Request& probe, const std::string& line,
                                    drv::Wait why, uint64_t price_bytes) {
                const int64_t now = drv::now_ms();
                if (holds.find(probe.id) == nullptr) {
                    // A request that already has a parked branch of its own (the same chat ran before)
                    // outranks a brand-new one: ending it costs a user an answer they are half-reading.
                    const bool resumed = conversations.probe(probe.ids, std::vector<
                                          strata::core::ConversationImageKey>{}, probe.cvec != 0,
                                          probe.id).tokens > 0;
                    if (!holds.enqueue(probe.id, why, now,
                                       resumed ? drv::kPriorityResumed : drv::kPriorityNew,
                                       (int64_t) probe.ids.size(), probe.max_new, price_bytes,
                                       probe.kind == proto::Kind::geni)) {
                        // The queue is full.  That is a refusal, and it says so with the queue's own
                        // bound rather than pretending the request is being held.
                        refuse_line(probe.id, std::string("the hold queue is full (") +
                                              std::to_string((long long) holds.cap()) +
                                              " requests can wait at once; --hold-ms " +
                                              std::to_string((long long) holds.hold_ms()) +
                                              " ms bounds each wait)");
                        holds.note_refused();
                        return false;
                    }
                    hold_lines.push_back(std::make_pair(probe.id, line));
                    std::fprintf(stderr, "%s\n", drv::hold_line(probe.id, why, holds.size()).c_str());
                } else {
                    note_wait(probe.id, why);
                }
                return true;
            };
            // Answer a waiter that gave up or was cancelled, and forget it.  Both numbers, always:
            // "it is queued" and "it is lost" must never look the same in the log.
            auto hold_expire = [&](int64_t id) {
                const int64_t now = drv::now_ms();
                const int64_t waited = holds.waited_ms(id, now);
                const drv::Wait why = holds.reason_of(id);
                std::printf("%s\n", sp_out.err(drv::hold_expired_line(id, waited, why, holds.hold_ms()),
                                               id).c_str());
                std::fflush(stdout);
                holds.pop(id);
                holds.note_expired();
                hold_forget(id);
                ++refused;
            };
            auto hold_cancel = [&](int64_t id) {
                drv::Waiter w;
                if (!holds.cancel(id, w)) return;
                // The client is still waiting on this request, so it gets an answer: 0 tokens, finish
                // `cancel` - the same shape a request cancelled at admission gets.
                std::printf("%s\n", sp_out.done(0, w.prompt_tokens, 0.0, 0.0, "cancel", 0, 0, 0, 0, 0,
                                                id).c_str());
                std::fflush(stdout);
                hold_forget(id);
                std::fprintf(stderr, "strata serve: slot %lld cancelled after waiting %lld ms - it never "
                                     "reached the engine\n", (long long) id,
                             (long long) (drv::now_ms() - w.since_ms));
            };
            // The engine's queue, on the wire, so `serve/server.py` can show waits it cannot see from
            // the outside.  Untagged (it is about the process, like SLOT), and only when there is
            // something to say or the situation changed - never on the serial path, which never gets
            // here at all.
            int64_t next_wait_line_ms = 0;
            int64_t next_promote_ms = 0;
            size_t last_wait_reported = 0;
            auto report_waits = [&]() {
                const int64_t now = drv::now_ms();
                if (holds.empty() && last_wait_reported == 0) return;
                if (holds.size() != last_wait_reported || now >= next_wait_line_ms) {
                    std::printf("%s\n", drv::wait_line(holds.size(), holds.oldest_wait_ms(now),
                                                       holds.head_reason()).c_str());
                    std::fflush(stdout);
                    last_wait_reported = holds.size();
                    next_wait_line_ms = now + 2000;
                }
            };
            // Drop a finished or refused request's context.  Called only after the step's guards have
            // run, so ~StepGuard never sees a destroyed ReqCtx.
            auto drop_ctx = [&](int64_t id) {
                // S4.2: this is THE wake point.  A context going away is the only thing that returns an
                // active-slot permit, the RAM it was priced against, and the prompt loan - so it is
                // where a waiter becomes runnable.  Clear every recorded reason (a waiter parked behind
                // RAM may now be blocked by the cap instead, and a reason that is never re-derived is
                // how a stale "waiting for RAM" line outlives the RAM problem) and let the next pass
                // re-ask the head immediately instead of waiting for the rate limit.
                if (!holds.empty()) { holds.wake_all(); next_promote_ms = 0; }
                for (size_t i = 0; i < live.size(); ++i) {
                    if (live[i].id != id) continue;
                    if (live[i].loan_held) {
                        // S3.2b: the ENGINE's view of the loan (`PfPart::loan_live`) has to be cleared too,
                        // not just the driver's.  A stranded `loan_live` is not a wrong token - the rows are
                        // all correctly marked non-resident - but it refuses the pump for every cache
                        // forever, so the experts the prompt evicted would never come back.  `refill()` is a
                        // no-op when nothing is lent, and under the lazy rule it copies nothing.
                        if (!refill(err)) {
                            std::fprintf(stderr, "strata serve: slot %lld: returning its loan failed (%s)\n",
                                         (long long) id, err.c_str());
                            err.clear();
                        }
                        loan.release(id);
                        live[i].loan_held = false;
                        // A slot that dies mid-read must wake the slots parked behind its loan, or a
                        // dead holder strands every prefill in the process.
                        if (!loan.held()) clear_deferred();
                    }
                    live.erase(live.begin() + (std::ptrdiff_t) i);
                    break;
                }
                // S3.6: the slot no longer owes itself a step, so a later hand-over may drop its
                // in-session branch without a guard refusing it.
                working_drop(id);
                // S3.9: and its parked entry stops being claimed.  The branch itself stays in the
                // cache - that is stage 2's mechanism, the next request of the same chat resumes from
                // it - but it is no longer a live conversation that another mount would destroy.
                release_conv_claims(id);
                watch_forget(id);
                for (size_t i = 0; i < deferred.size(); ++i)
                    if (deferred[i] == id) { deferred.erase(deferred.begin() + (std::ptrdiff_t) i); break; }
            };
            // The watchdog's verdict: ERR + destroy THAT slot, keep the engine and the other slots.
            auto kill_slot = [&](int64_t id) {
                ReqCtx* R = ctx_of(id);
                if (R == nullptr) return;
                std::fprintf(stderr, "strata serve: slot %lld made no progress within %d s - ending it and "
                                     "keeping the engine (stage 3 R3)\n", (long long) id, watchdog_limit_s);
                if (R->loan_held) {
                    if (!refill(err))
                        std::fprintf(stderr, "strata serve: slot %lld: refilling its loan failed (%s)\n",
                                     (long long) id, err.c_str());
                    err.clear();
                    loan.release(id);
                    R->loan_held = false;
                    clear_deferred();
                }
                R->finished = true;
                std::printf("%s\n", sp_out.err("no progress within the watchdog limit - this request was ended;"
                                               " the engine stayed up", id).c_str());
                std::fflush(stdout);
                slot_step(id, SState::error);
                slot_finish(id);
                drop_ctx(id);
            };
            // S3.7: the startup worst-case snapshot, and the per-token rate it implies.  Declared here
            // because `admit_one` below reads it; FILLED just before the loop runs, which is before any
            // call site of this lambda, so the reference capture is honest rather than lucky.
            drv::ParkCeiling ceiling;
            // ---- admit: take one queued line and bring it in (§3.1 step 1) ----------------------
            // S4.2 turns this from "admit or ERR" into three questions, and only the third may create a
            // context:
            //   1. is the line a request at all (QUIT / STOP / a parse error)?
            //   2. `admit_decision` - can it NEVER run here (an ERR), can it run NOW, or must it WAIT
            //      (and for which resource)?  A temporary answer queues it, it does not cancel it.
            //   3. `promote_waiters` - the head of the queue that now fits becomes the context.
            // Still at most ONE admission per pass, so a burst of lines cannot starve running slots.

            // Everything the admission decision needs for one parsed request, computed once.  The
            // decision itself is `serve_driver::admit_decision` - a tested predicate - so the ORDER of
            // these checks, and therefore the reason a request waits, is pinned by the CPU test rather
            // than being an accident of this call site.
            struct AdmitInput {
                drv::Admit a;
                drv::SlotCost cost;
                uint64_t est = 0;
            };
            // "priced at N B per context token (...)" - the same wording S3.7 introduced, now shared by
            // the wait line and the never-fits error so both name the two real numbers.
            auto price_note = [&](const AdmitInput& in) {
                if (in.cost.per_token) {
                    return std::string("priced at ") + std::to_string((long long) in.cost.per_token) +
                           " B per context token (" +
                           (park_per_token_tokens
                                ? "the rate of the largest park so far, " +
                                  std::to_string((long long) park_per_token_tokens) + " tokens"
                                : "the rate this --max_context implies") +
                           ")";
                }
                return std::string("priced at the budget/slots estimate: nothing has been parked and a "
                                   "snapshot could not be sized at --max_context");
            };
            // The one line that explains a RAM wait or a RAM refusal, with BOTH numbers.  S3.7's lesson:
            // an admission answer that names no number is an answer nobody can act on.
            auto admit_diag = [&](int64_t id, const AdmitInput& in) {
                std::fprintf(stderr, "strata serve: admit slot %lld held back: %lld prompt tokens priced at "
                                     "%lld MiB of snapshot (%lld MiB already held by other slots, %lld MiB "
                                     "free, floor %lld MiB) - %s\n",
                             (long long) id, (long long) in.a.prompt_tokens, (long long) (in.est >> 20),
                             (long long) (in.a.held_bytes >> 20),
                             in.a.have_telemetry ? (long long) (in.a.avail_bytes >> 20) : -1LL,
                             (long long) (in.a.floor_bytes >> 20), price_note(in).c_str());
            };
            auto ask_admission = [&](const proto::Request& probe, AdmitInput& in, drv::Wait& why) {
                const auto avail = strata::core::conversation_available_memory();
                // A GENI request's prompt lives in the embeddings file, so its token count is not on the
                // request line. Price it at the ceiling - the honest worst case - rather than at zero.
                const uint64_t prompt_tokens = probe.kind == proto::Kind::geni
                                                   ? (uint64_t) o.max_context
                                                   : (uint64_t) probe.ids.size();
                const uint64_t max_new = (uint64_t) (probe.max_new > 0 ? probe.max_new : 0);
                in.cost = drv::slot_cost_of((uint64_t) conversations.budget(),
                                            conversations.slots(), ceiling, park_per_token);
                in.est = drv::slot_image_bytes(in.cost, prompt_tokens, max_new);
                uint64_t held = 0;
                bool any_live = false, image_live = false;
                for (const ReqCtx& Q : live) {
                    held += (uint64_t) (Q.image_bytes > 0 ? Q.image_bytes : 0);
                    any_live = true;
                    image_live = image_live || Q.geni;
                }
                // R7, as an admission rule rather than a mid-read discovery: `mrope_host`/`d_mrope` and
                // the one `row_ptr`/`img_rows` set describe ONE sequence, so an image request runs alone.
                const bool want_image = probe.kind == proto::Kind::geni;
                bool cap_free = false, row_free = false;
                {
                    std::lock_guard<std::mutex> lk(slot_mu);
                    // The cap, counting the contexts already waiting to be brought in.
                    cap_free = slots_reg.can_admit() &&
                               slots_reg.active_count() + pending_count() < slots;
                    row_free = slots_reg.count() < strata::program::slot::kMaxSlots;
                }
                in.a = drv::Admit{};
                in.a.prompt_tokens = (int64_t) prompt_tokens;
                in.a.max_new = probe.max_new;
                in.a.max_context = o.max_context;
                in.a.prompt_known = probe.kind != proto::Kind::geni;
                in.a.cache_enabled = conversations.enabled();
                in.a.have_telemetry = avail.has_value();
                in.a.avail_bytes = avail.value_or(0);
                in.a.floor_bytes = (uint64_t) o.conversation_cache_min_free_mib * 1024ull * 1024ull;
                in.a.price_bytes = in.est;
                in.a.uncapped_price = drv::slot_price_uncapped(in.cost, prompt_tokens, max_new);
                in.a.held_bytes = held;
                in.a.budget_bytes = (uint64_t) conversations.budget();
                in.a.cap_free = cap_free;
                in.a.row_free = row_free;
                in.a.image_ok = !(image_live || (want_image && any_live));
                return drv::admit_decision(in.a, why);
            };
            // The ERR for a request that can NEVER run here.  Named per reason, because "the server is
            // full" and "your prompt does not fit" are different messages to different readers, and the
            // whole point of the hold queue is that the first one is no longer an error at all.
            auto permanent_admit_reason = [&](const AdmitInput& in) {
                if (!in.a.cache_enabled)
                    return std::string(drv::refuse_reason(drv::Refuse::no_parking));
                if (in.a.prompt_known &&
                    drv::prompt_never_fits(in.a.prompt_tokens, in.a.max_new, in.a.max_context))
                    return std::string("the prompt is ") + std::to_string((long long) in.a.prompt_tokens) +
                           " tokens and the answer may add " + std::to_string((long long) in.a.max_new) +
                           ", which does not fit --max-context " +
                           std::to_string((long long) in.a.max_context) +
                           " - send a shorter prompt or raise --max-context";
                return std::string("this request's conversation (" +
                                   std::to_string((long long) (in.est >> 20)) + " MiB) does not fit the free "
                                   "RAM on this machine even if every other request finished (" +
                                   std::to_string((long long) (in.a.avail_bytes >> 20)) + " MiB free, floor " +
                                   std::to_string((long long) (in.a.floor_bytes >> 20)) + " MiB) - " +
                                   price_note(in) + ". Give the machine more RAM or lower --max-context");
            };
            // Bring one request in: its context, its working-branch claim, its place in the accounting.
            auto open_ctx = [&](const proto::Request& probe, const std::string& line, uint64_t est) {
                live.push_back(ReqCtx{});
                ReqCtx& R = live.back();
                R.line = line;
                R.has_line = true;
                R.id = probe.id;
                R.image_bytes = (int64_t) est;
                // S3.6: this slot now owes itself a step, so `swap_to` must be able to save it before
                // it hands the session to anybody else.  `working_state` refreshes it from the phase
                // after every step and `drop_ctx` removes it.
                working_set(probe.id, strata::program::serve_swap::Outgoing::re_readable);
                pass_moved = true;
                ++request_index;      // S0.3 lever 4: the parking backoff's window is in requests
            };
            auto admit_one = [&]() {
                // A line already read but not admitted: retry it before reading another, so the
                // client's order is kept.
                if (held_line.empty()) {
                    if (!try_next_line(held_line)) return;
                }
                // `pass_moved` is set only where a line is CONSUMED or a context is CREATED.  A
                // deferral (the cap is full, an image request holds the position table) must leave it
                // false, or the loop's anti-spin backstop at the bottom never fires.
                if (held_line == "QUIT") { driver_quit = true; held_line.clear(); pass_moved = true; return; }
                const proto::Request probe = proto::parse_request(
                    held_line, proto::Defaults{o.pcie_frac, o.spec_min_p}, tagged);
                if (probe.kind == proto::Kind::stop) {
                    // A STOP that reached the queue instead of the stdin thread.  `STOP <id>` names a
                    // slot; a bare STOP keeps 0.1.30's meaning ("the running one"), which
                    // `bare_stop_target` resolves through the registry.
                    std::lock_guard<std::mutex> lk(slot_mu);
                    if (probe.id != proto::kNoId) {
                        // S4.2: a waiter is cancelled directly - it has no registry row to cancel, and
                        // recording it in `pending_cancel` as well would leave a stale entry that could
                        // wrongly cancel a later request reusing the same id.
                        if (holds.contains(probe.id)) hold_cancel(probe.id);
                        else if (!slots_reg.cancel_request(probe.id) && pending_cancel.size() < 64)
                            pending_cancel.push_back(probe.id);
                    } else {
                        const int64_t who = drv::bare_stop_target(slots_reg);
                        if (who != proto::kNoId) slots_reg.cancel_request(who);
                    }
                    held_line.clear();
                    pass_moved = true;
                    return;
                }
                if (probe.kind != proto::Kind::gen && probe.kind != proto::Kind::geni) {
                    // 0.1.30's untagged ERR for a line it could not attribute to anything.
                    std::printf("%s\n", sp_out.err(probe.error.empty() ? proto::err_expected()
                                                                       : probe.error).c_str());
                    std::fflush(stdout);
                    held_line.clear();
                    pass_moved = true;
                    return;
                }
                if (probe.id == proto::kNoId) {   // a tagged session needs an id to route by
                    refuse_line(probe.id, "this engine was started with --serve-slots " + std::to_string(slots) +
                                          ": requests need a request id (GEN <id> <max_new> ...)");
                    held_line.clear();
                    pass_moved = true;
                    return;
                }
                if (ctx_of(probe.id) != nullptr) {
                    refuse_line(probe.id, "request id " + std::to_string((long long) probe.id) +
                                          " is already running");
                    held_line.clear();
                    pass_moved = true;
                    return;
                }
                if (holds.contains(probe.id)) {
                    // An impatient client re-sent a request that is ALREADY waiting.  It is not an error
                    // and it must not enter the queue twice (two rows for one id would make the wire
                    // unparseable).  Refresh its place in the log and move on.
                    std::fprintf(stderr, "strata serve: slot %lld is already waiting (%s, %lld ms so far)\n",
                                 (long long) probe.id, drv::wait_reason(holds.reason_of(probe.id)),
                                 (long long) holds.waited_ms(probe.id, drv::now_ms()));
                    held_line.clear();
                    pass_moved = true;
                    return;
                }
                AdmitInput in;
                drv::Wait why = drv::Wait::none;
                const drv::Hold h = ask_admission(probe, in, why);
                if (h == drv::Hold::error) {
                    // The one case that is still an ERR: this request can never run on this engine.
                    refuse_line(probe.id, permanent_admit_reason(in));
                    held_line.clear();
                    pass_moved = true;
                    return;
                }
                if (h == drv::Hold::wait) {
                    // HOLD, DON'T REJECT.  The request is queued and will run when the resource it names
                    // frees; the client sees a slow first token, not an error.
                    if (why == drv::Wait::ram) admit_diag(probe.id, in);
                    hold_enqueue(probe, held_line, why, in.est);
                    held_line.clear();
                    pass_moved = true;   // the line WAS consumed
                    return;
                }
                open_ctx(probe, held_line, in.est);
                held_line.clear();
            };
            // ---- S4.2: let the queue through -----------------------------------------------------
            // A resource came back (a slot finished, a park landed, the loan was released, a branch was
            // pruned, a waiter expired), so re-ask the head of the queue.  One promotion per pass,
            // exactly as a fresh line gets one admission per pass, and the head is the only candidate:
            // letting a later waiter past an earlier one is how a queue stops being a queue.
            auto promote_waiters = [&]() {
                for (;;) {
                    const drv::Waiter* w = holds.head();
                    if (w == nullptr) return;
                    const int64_t id = w->id;
                    std::string* line = hold_find_line(id);
                    if (line == nullptr) {
                        // Bookkeeping net: a waiter with no line cannot be dispatched.  Drop it rather
                        // than wedge the queue forever behind it.
                        holds.pop(id);
                        continue;
                    }
                    const proto::Request probe = proto::parse_request(
                        *line, proto::Defaults{o.pcie_frac, o.spec_min_p}, tagged);
                    if (probe.kind != proto::Kind::gen && probe.kind != proto::Kind::geni) {
                        refuse_line(probe.id, probe.error.empty() ? proto::err_expected() : probe.error);
                        holds.pop(id); hold_forget(id);
                        continue;
                    }
                    // A `STOP <id>` that arrived while this request waited.  The stdin thread cannot see
                    // the hold queue (it is created before it), so it recorded the cancel in
                    // `pending_cancel`.  Answer it HERE, before the request becomes a context: a waiter
                    // that was cancelled is never run and then cancelled, which would emit tokens to a
                    // client that asked us to stop.
                    for (size_t k = 0; k < pending_cancel.size(); ++k) {
                        if (pending_cancel[k] != id) continue;
                        pending_cancel.erase(pending_cancel.begin() + (std::ptrdiff_t) k);
                        const int64_t waited = holds.waited_ms(id, drv::now_ms());
                        drv::Waiter cw;
                        if (holds.cancel(id, cw)) {
                            hold_forget(id);
                            std::printf("%s\n", sp_out.done(0, (int64_t) probe.ids.size(), 0.0, 0.0,
                                                            "cancel", 0, 0, 0, 0, 0, id).c_str());
                            std::fflush(stdout);
                            std::fprintf(stderr, "strata serve: slot %lld cancelled after waiting %lld ms "
                                                 "- it never reached the engine\n", (long long) id,
                                         (long long) waited);
                        }
                        pass_moved = true;
                        break;
                    }
                    if (holds.find(id) == nullptr) continue;   // it was cancelled just now
                    AdmitInput in;
                    drv::Wait why = drv::Wait::none;
                    const drv::Hold h = ask_admission(probe, in, why);
                    if (h == drv::Hold::error) {
                        // It waited, and it turns out it can never run.  Say so with the real reason -
                        // a hold timeout blaming the load would send the owner to the wrong knob.
                        end_slot(id, permanent_admit_reason(in));
                        holds.pop(id); holds.note_expired(); hold_forget(id);
                        continue;
                    }
                    if (h == drv::Hold::wait) {
                        // Still blocked.  Report the CURRENT reason (it may have moved from RAM to the
                        // cap) and stop: the head is the head.
                        if (holds.set_reason(id, why) && why == drv::Wait::ram) admit_diag(id, in);
                        return;
                    }
                    const int64_t waited = holds.waited_ms(id, drv::now_ms());
                    const std::string line_copy = *line;
                    holds.pop(id);
                    holds.note_admitted();
                    hold_forget(id);
                    // The wait is REPORTED, because §7.2's rule is that a request which waited must say
                    // so - otherwise its `prompt_ms` reads like a slowdown in the engine.
                    std::fprintf(stderr, "%s\n", drv::hold_admitted_line(id, waited).c_str());
                    open_ctx(probe, line_copy, in.est);
                    return;
                }
            };
            // ---- S4.2: the queue's own deadlines --------------------------------------------------
            // A wait with no bound is a hung client.  Expired waiters are answered where they are found,
            // and each one names what it waited for.  `STOP <id>` is handled in `admit_one`, by the
            // stdin thread and here; the watchdog deliberately never sees a waiter
            // (`watchdog_sees_waiters`), because a waiter is not inside a step and killing one would be
            // R3 in a new costume.  `--hold-ms 0` means the bound is off and only a cancel ends a wait.
            auto expire_waiters = [&]() {
                for (const int64_t id : holds.expired(drv::now_ms())) {
                    hold_expire(id);
                    pass_moved = true;
                }
            };
            // ---- the hand-over (§3.1 step 3) ---------------------------------------------------
            // `swap_to` owns the order, the residency drain and the loan return (S3.1d) - the driver
            // only decides WHEN, and it never retries a hand-over that already wrote the session.
            //
            // S3.6: the return type is the driver's own `SwapResult`, not a bool, because a refused
            // hand-over now has TWO answers and they are opposite:
            //   * `wait_park`     - the session is mid-read and cannot be saved YET.  The incoming slot
            //                       waits; the outgoing one keeps the session.  This is the ordinary
            //                       two-client race, and it must not be an error.
            //   * `end_incoming`  - the outgoing conversation can NEVER be parked here.  The incoming
            //                       request is answered with an ERR naming both numbers, its permit
            //                       comes back, and the session is untouched.  Refused, not starved and
            //                       not spun on.
            // Both stop the retry-without-a-restore path, because that retry is exactly what used to
            // destroy a running conversation.
            auto do_swap = [&](ReqCtx& R, std::string& serr, const char* why) -> drv::SwapResult {
                // The image to mount: this slot's parked branch, if the cache still holds one.  A slot
                // being admitted for the first time has no branch yet and `prep_request` does 0.1.30's
                // own mount search for it, so the driver only mounts a slot it has already run.
                //
                // S3.7: the lookup is `serve_swap::seek_mount_index`, and it is NOT the prefix search
                // `prep_request` uses.  A hand-over wants the branch that IS this slot's sequence;
                // `ConversationCache::best()` requires a strictly shorter token list and answers 0 for
                // the one conversation the cache parked for it.  That is what ended the owner's
                // requests mid-decode with an ERR blaming the parking budget, on a run whose budget
                // was 10 048 MiB and whose branch 113 MiB.
                if (R.phase != drv::Phase::queued && mount_image == std::nullopt) {
                    const strata::program::serve_swap::SlotConv& c = conv_of(R.id);
                    const int64_t idx = strata::program::serve_swap::seek_mount_index(conversations, c);
                    if (idx >= 0) mount_image.emplace(conversations.take((size_t) idx));
                    else if (strata::program::serve_swap::handover_seeks(c) &&
                             R.phase == drv::Phase::decode)
                        // S3.8: DECODE only, and that is the whole point.  A decoder's prompt tokens
                        // are already on the wire, so losing its branch is unrecoverable and this is
                        // the one case that ends a running request.  A PREFILL slot that finds nothing
                        // parked can still re-read - bounded by `kMaxRereads`, and `step_gate` reports
                        // it - so it is not named here.  (The old one-argument `outgoing_for` would
                        // now answer `re_readable` for a prefill regardless of its cursor, which is
                        // exactly the ambiguity this avoids.)
                        std::fprintf(stderr, "strata serve: swap: slot %lld's parked branch "
                                             "(%zu tokens) is no longer in the cache - pruned or replaced "
                                             "while the slot waited, and a decoder cannot re-read what it "
                                             "has already sent (parked=%zu entries, %zu MiB of a %zu MiB "
                                             "budget)\n",
                                     (long long) R.id, c.live.size(),
                                     conversations.size(), conversations.bytes() >> 20,
                                     conversations.budget() >> 20);
                }
                // R8, and the invariant the whole loan rule rests on: a hand-over refills
                // (`serve_swap::run`'s `return_loan` step, which always runs when the hand-over runs),
                // so it ends the current holder's loan whether the hand-over succeeds or fails - and
                // the driver's view has to say the same thing the caches now do.  `swap_to` returns
                // true WITHOUT refilling when the slot is already mounted, so gate on the same
                // predicate it gates on.
                if (loan.held() && strata::program::serve_swap::swap_needed(mounted_id, R.id)) {
                    if (ReqCtx* H = ctx_of(loan.owner)) H->loan_held = false;
                    loan.release(loan.owner);
                    clear_deferred();
                }
                // S3.6: the slot about to be saved is the one the session reflects, NOT the one being
                // picked.  A decoder has a branch the cache can take - its own consumed token list,
                // the same list `finish_request` swaps into `live` - but until now nothing published
                // it mid-request, so `park_current`'s `live_ok` guard refused every pre-emption for a
                // reason that had nothing to do with the budget.  Publish it before asking.
                // S3.8: `publish_working_branch` covers the mid-PROMPT case too, which is what makes
                // two concurrent prefills parkable instead of restarting each other forever.
                if (ReqCtx* OUT = ctx_of(mounted_id)) {
                    publish_working_branch(*OUT);
                    // And re-classify what losing the session would cost it, from the cursor it has
                    // NOW.  `working_set` was last armed before its previous step, so a read that has
                    // since consumed tokens would still be classified `re_readable` - and that
                    // classification is what licenses throwing the read away.
                    working_set(OUT->id, drv::outgoing_for(OUT->phase, OUT->at));
                }
                bool poisoned = false;
                if (serve_trace)
                    std::fprintf(stderr, "strata serve: trace: swap slot %lld -> %lld why=%s\n",
                                 (long long) mounted_id, (long long) R.id, why);
                const bool had_image = mount_image != std::nullopt;
                if (swap_to(R.id, serr, poisoned, /*restore_positions=*/true)) {
                    ++swaps_done;
                    // S3.6: a re-mount is an EVENT.  Until now the only trace of one was a
                    // suspiciously large `N reused` figure on the request's summary line, which is
                    // exactly how the premature-end bug hid for a whole run.
                    if (had_image && swap_restored)
                        std::fprintf(stderr, "%s\n", drv::resumed_line(R.id, (int64_t) live.size()).c_str());
                    // S3.6: the session now holds this slot's branch.  Put `consumed` back in step
                    // with it (the penalty history and `finish_request` both read it), and mark the
                    // outgoing slot's context as no longer described by the session unless its state
                    // was saved.
                    restore_published_branch(R);
                    std::lock_guard<std::mutex> lk(slot_mu);
                    slots_reg.set_active(R.id);
                    return drv::SwapResult::mounted;
                }
                // S3.6: the outgoing slot could not be saved and it still has work to do.  That is NOT
                // the case the "retry without a restore" recovery below is for - a retry would destroy
                // the outgoing conversation, which is the bug.  The session is untouched either way.
                if (swap_park_refused) {
                    return_mount_image();
                    if (swap_park_kind == strata::program::serve_swap::ParkRefusal::not_saveable) {
                        // Transient: the running request has not finished its read, so there is no
                        // whole branch to save.  It will have one the moment its read ends, so this
                        // request waits rather than being punished for a race that resolves itself.
                        std::fprintf(stderr, "%s\n", drv::trace_park_wait(R.id, mounted_id).c_str());
                        return drv::SwapResult::wait_park;
                    }
                    std::fprintf(stderr, "strata serve: swap to slot %lld refused for good (%s) - that request "
                                         "is ended, the mounted slot keeps the session\n",
                                 (long long) R.id, serr.c_str());
                    return drv::SwapResult::end_incoming;
                }
                if (!poisoned && !swap_wrote_session && mount_image != std::nullopt) {
                    // The incoming image was rejected (or the save refused) BEFORE anything was
                    // written.  S3.1d's own request-line call site handles that by dropping the image
                    // and handing over WITHOUT a restore - the same answer 0.1.30 gives for an invalid
                    // snapshot: fall back to re-reading.  Doing it here, inside the one call the driver
                    // makes, keeps the driver from having to re-derive S3.1d's recovery at the call
                    // site (which the phase contract forbids).
                    std::fprintf(stderr, "strata serve: swap: slot %lld -> %lld retried without a restore (%s)\n",
                                 (long long) mounted_id, (long long) R.id, serr.c_str());
                    return_mount_image();
                    poisoned = false;
                    serr.clear();
                    if (swap_to(R.id, serr, poisoned, /*restore_positions=*/true)) {
                        ++swaps_done;
                        restore_published_branch(R);
                        std::lock_guard<std::mutex> lk(slot_mu);
                        slots_reg.set_active(R.id);
                        return drv::SwapResult::mounted;
                    }
                    // S3.6: the retry can hit the same parking guard.  Same two answers, same rule:
                    // never destroy the outgoing conversation to let a second request in.
                    if (swap_park_refused) {
                        return_mount_image();
                        return swap_park_kind == strata::program::serve_swap::ParkRefusal::not_saveable
                                   ? drv::SwapResult::wait_park
                                   : drv::SwapResult::end_incoming;
                    }
                }
                if (poisoned || swap_wrote_session) {
                    // 0.1.30's rule, kept verbatim: a restore that failed mid-write is fatal to the
                    // session, and a hand-over that already wrote it must not be retried.
                    std::printf("%s\n", sp_out.err("slot hand-over failed after the restore: " + serr, R.id).c_str());
                    std::fflush(stdout);
                    std::fflush(stderr);
                    return drv::SwapResult::fatal;
                }
                // Refused BEFORE anything was written (R7 exclusivity, a failed drain, a failed save):
                // the session is intact.  Do not run this slot's step; let the pick try something else.
                // And give the image back: `mount_image` is a hole in the cache, and leaving it out
                // would (a) lose the conversation if the process then exited and (b) let the NEXT
                // hand-over - to a DIFFERENT slot - restore this slot's state into the session.
                return_mount_image();
                std::fprintf(stderr, "strata serve: swap to slot %lld refused (%s) - trying other work\n",
                             (long long) R.id, serr.c_str());
                return drv::SwapResult::wait_park;
            };
            // A driver-level tie-break over `pick()`: the longest-waiting runnable slot that is NOT
            // deferred.  `pick()` cannot know about the loan (it is engine state, not registry state),
            // so without this the loop would re-pick the blocked slot forever and the holder - a later
            // row in the registry's fixed array - would never get the session.  That is a livelock, and
            // it is the one thing a one-thread scheduler must never do.
            auto pick_unblocked = [&](int64_t now) -> strata::program::slot::Pick {
                strata::program::slot::Pick p;      // idle
                std::lock_guard<std::mutex> lk(slot_mu);
                const int64_t act = slots_reg.active_id();
                int64_t best = proto::kNoId, bw = -1;
                slots_reg.each([&](const strata::program::slot::Slot& s) {
                    if (s.state != SState::prefilling && s.state != SState::decoding &&
                        s.state != SState::cancelling) return;
                    if (is_deferred(s.id)) return;
                    const int64_t w = slots_reg.waited_ms(now, s);
                    if (w > bw) { bw = w; best = s.id; }
                });
                if (best == proto::kNoId) return p;
                p.action = (best == act) ? PAct::run : PAct::swap;
                p.id = best;
                if (const strata::program::slot::Slot* s = slots_reg.find(best)) p.state = s->state;
                p.waited_ms = bw > 0 ? bw : 0;
                p.why = "driver-loan-tiebreak";
                if (serve_trace) std::fprintf(stderr, "%s\n", drv::trace_pick("tiebreak", p).c_str());
                return p;
            };
            // ---- run ONE step of ONE slot ------------------------------------------------------
            // `why` is `Pick::why` (or "pending-admit"), kept only for the trace line: the scheduler's
            // reason for choosing this slot was already computed by `pick()` and thrown away, which is
            // why a live log could not answer "why THAT slot and not this one".
            auto run_one = [&](ReqCtx& R, DStep s, const char* why) {
                req_id = R.id;
                arm_prompt_view(R);
                // S3.6 — THE STEP GATE (§5.3, the assertion the contract demands).  A slot may only
                // step when the mounted session still describes it.  `R.session_valid` is that fact;
                // `swap_to` clears it for a slot whose session went to somebody else without a
                // restore, and `prep_request` / a restoring hand-over set it back.  After the parking
                // guard this should be unreachable for a decode slot, so it is the net under the
                // guard rather than the guard itself - but the failure it catches is silent garbage,
                // so it stays armed.
                {
                    // `resumable` is the contract's predicate (`invalidate_unparked` is the only thing
                    // that clears it); `session_valid` catches the other half - a slot the session was
                    // moved away from without a restore, whose record the adopt hook still calls
                    // resumable.  Either one being false means the session does not describe this slot.
                    R.session_valid = (R.id == mounted_id) && session_established;
                    const drv::Gate g = drv::step_gate(R.phase, conv_of(R.id).resumable,
                                                       R.session_valid);
                    if (g == drv::Gate::end) {
                        std::fprintf(stderr, "strata serve: slot %lld would step in %s with no conversation "
                                             "to step against - ending it (stage 3 §5.3)\n",
                                     (long long) R.id, drv::phase_name(R.phase));
                        std::fflush(stderr);
                        if (R.loan_held) { loan.release(R.id); R.loan_held = false; clear_deferred(); }
                        // S3.7: the reason names the state it found, and for the decode case it quotes
                        // the two numbers that decided it - the size this slot's last parked image
                        // actually was (or the last refused save would have needed) and the budget in
                        // force.  The old text blamed "the parking budget is too small" for every
                        // variant of this, including the ones with a 9 GiB budget, which sent the owner
                        // to raise a knob that was not the problem.
                        drv::GateEnd ge;
                        ge.ph = R.phase;
                        ge.resumable = conv_of(R.id).resumable;
                        ge.session_valid = R.session_valid;
                        const int64_t pb = conv_of(R.id).parked_bytes;
                        ge.snapshot_bytes = (uint64_t) (pb > 0 ? pb : (int64_t) swap_park_snapshot);
                        ge.budget_bytes = (uint64_t) conversations.budget();
                        end_slot(R.id, drv::gate_end_reason(ge));
                        drop_ctx(R.id);
                        return;
                    }
                    if (g == drv::Gate::re_read) {
                        // S3.8 — THE PROMISE, KEPT, AND BOUNDED.  A re-read is an honest recovery the
                        // first time a branch is lost.  Repeated, it is the two-prefill livelock the
                        // owner hit: slot 3 was reset to token 0 101 times and never finished, because
                        // a mid-prompt read was classified `re_readable` and never published, so every
                        // swap erased it.  `outgoing_for` now calls such a slot `must_park`, which
                        // removes the mechanism; this bound is the net under it, and it turns any
                        // remaining spin into a named ERR instead of an infinite re-read.
                        ++R.rereads;
                        if (!drv::reread_allowed(R.rereads)) {
                            std::fprintf(stderr, "strata serve: %s\n",
                                         drv::reread_limit_line(R.id, R.rereads, drv::kMaxRereads).c_str());
                            std::fflush(stderr);
                            if (R.loan_held) { loan.release(R.id); R.loan_held = false; clear_deferred(); }
                            end_slot(R.id, drv::reread_limit_line(R.id, R.rereads, drv::kMaxRereads));
                            drop_ctx(R.id);
                            return;
                        }
                        // THE PROMISE, KEPT.  This slot lost its session state and survives, so it
                        // really does go back to token 0: phase, cursor, segments, checkpoints and the
                        // resume resolution all reset.  `prep_request` already ran, so its sampling
                        // dispatch and its position table are still in `R`; what has to be rebuilt is
                        // the session and everything that reads where the prompt starts.
                        std::fprintf(stderr, "%s\n", drv::reread_line(R.id, R.n).c_str());
                        std::fflush(stderr);
                        reset_request_to_token0(R);
                    }
                }
                // S3.6: the driver's view of what losing the session would cost this slot.  Refreshed
                // from the phase before every step, which is what `swap_to`'s parking guard reads.
                // S3.8: and from the CURSOR, because a prefill that has already read tokens has
                // progress the cache can save - classifying it `re_readable` is what let every swap
                // erase it and send it back to token 0 forever.
                working_set(R.id, drv::outgoing_for(R.phase, R.at));
                if (serve_trace)
                    std::fprintf(stderr, "%s\n", drv::trace_step(R.id, s, why).c_str());
                const drv::Phase phase_before = R.phase;
                // A STOP that arrived between requests is stale, exactly as the serial path cleared it
                // at the top of its iteration.  The per-slot `cancel` flag is what carries the intent
                // now: the stdin thread sets both, and `stopped()` reads the slot's.
                stop_req.store(false);
                {
                    std::lock_guard<std::mutex> lk(slot_mu);
                    running_id.store(R.id);
                }
                const int64_t t0 = drv::now_ms();
                bool ran = true;
                // Was THIS slot STOPped?  `dispatch_step` already turned that into "run the phase's
                // tail, never a new step"; the two tail branches need the same answer to record why
                // the request ended (`cancel`) and whether the conversation state may be updated.
                bool unwinding = false;
                {
                    std::lock_guard<std::mutex> lk(slot_mu);
                    if (const strata::program::slot::Slot* sl = slots_reg.find(R.id))
                        unwinding = sl->cancel.load();
                }
                {
                    // 0.1.30's SlotGuard, plus `finished`: armed per step, it releases the row only
                    // when the request actually ended.  Same destructor order (mrope, then slot).
                    struct StepGuard {
                        ReqCtx* ctx = nullptr;
                        std::function<void(int64_t)> finish;
                        ~StepGuard() { if (ctx && ctx->slot_open && ctx->finished && finish) finish(ctx->id); }
                    } step_guard{&R, std::function<void(int64_t)>(slot_finish)};
                    MropeScope mrope_scope;
                    mrope_scope.skip = [&]() { return !swaps_on || !R.mrope_touched || mounted_id == req_id; };
                    mrope_scope.restore = [&]() {
                        if (mounted_id == strata::program::serve_proto::kNoId) return;
                        apply_positions(conv_of(mounted_id));
                        if (!mrope_host.empty()) upload_mrope_table();
                        mrope_owner = mounted_id;
                    };
                    watch_begin(R.id);
                    if (s == DStep::admit) {
                        const Prep pr = prep_request(R, R.line);
                        R.has_line = false;
                        if (pr == Prep::ok) {
                            plan_prompt_segments(R);
                            R.phase = drv::Phase::prefill;
                            R.pp_total = pp_total; R.pp_from = pp_from;
                            R.pp_t0 = pp_t0; R.pp_next_check = pp_next_check;
                            bool cancel_now = false;
                            {
                                std::lock_guard<std::mutex> lk(slot_mu);
                                if (const strata::program::slot::Slot* sl = slots_reg.find(R.id))
                                    cancel_now = sl->cancel.load();
                                if (strata::program::slot::Slot* sl = slots_reg.find(R.id)) sl->started_ms = t0;
                                slots_reg.set_active(R.id);   // §5.3: the session now reflects this slot
                            }
                            if (cancel_now) {
                                // `STOP <id>` arrived before this line did, so the row was cancelled at
                                // admission and `slot_step` already moved it to `cancelling`.  Skip the
                                // read entirely and go to the prefill tail: `finish_prefill` reports
                                // `cancel`, and `finish_request` leaves `live` alone because
                                // `R.cancelled` is set.  Running the read anyway would leave the
                                // conversation state claiming a prompt the session never consumed.
                                R.cancelled = true;
                                R.phase = drv::Phase::prefill_end;
                            }
                            ++admitted;
                        } else if (pr == Prep::rejected) {
                            R.finished = true;                // the guard releases the row
                        } else {
                            R.fail = 1;
                        }
                    } else if (s == DStep::prefill) {
                        arm_prompt_state(R);
                        // What THIS segment will do, decided exactly as `run_prefill_step` decides it:
                        //   `through_windows` -> it calls `refill` first (the loan goes back);
                        //   `batched`         -> it calls `lend` (the loan is taken);
                        //   neither (no segment left) -> it touches nothing and returns `finished`.
                        // The driver's loan view has to follow that, or a slot that still physically
                        // holds the loan would let a second slot lend the same cache rows (risk R8).
                        const int64_t to = next_segment_end(R);
                        const bool has_seg = to > R.at;
                        const bool through_windows = has_seg && windows_ok(R.at, to);
                        const bool batched = has_seg && !through_windows;
                        if (batched && !loan.acquire(R.id)) {
                            // R8's hard guard.  The loop defers a slot before it gets here, so this
                            // should be unreachable; it stays because "two slots lent the same cache
                            // rows" is silent and produces plausible tokens, and a guard that cannot
                            // be bypassed is worth more than one that relies on the caller.
                            std::fprintf(stderr, "strata serve: slot %lld could not take the prompt loan "
                                                 "(held by slot %lld) - step skipped\n", (long long) R.id,
                                         (long long) loan.owner);
                            ran = false;
                        } else {
                            if (batched) R.loan_held = true;   // taken by the guard above
                            const Step r = run_prefill_step(R);
                            R.pp_next_check = pp_next_check;
                            if (through_windows && R.loan_held) {
                                // The segment refilled before it read (0.1.30's `if (win &&
                                // !refill(err))`), so a loan this slot held from an earlier batched
                                // segment is gone.
                                loan.release(R.id); R.loan_held = false;
                                if (!loan.held()) clear_deferred();
                            } else if (batched && R.loan_held) {
                                loan.acquire(R.id);   // still ours: the sticky loan keeps the layout
                            }
                            if (r == Step::progressed) R.phase = drv::Phase::prefill;
                            else if (r == Step::finished || r == Step::cancelled) R.phase = drv::Phase::prefill_end;
                            else if (r == Step::needs_swap) R.phase = drv::Phase::prefill;
                            else if (r == Step::error) R.fail = 1;
                            else if (r == Step::fatal_exit) R.fail = 2;
                        }
                    } else if (s == DStep::prefill_end) {
                        arm_prompt_state(R);
                        // A between-steps cancel that skipped the remaining segments is the same
                        // situation as a cancel caught inside `run_prefill_step`: the prompt was not
                        // read to the end, so the session sits between two chunks and `live` must not
                        // be swapped.  `finish_prefill` turns `cancelled` into `DONE ... cancel`.
                        if (unwinding && R.phase == drv::Phase::prefill) R.cancelled = true;
                        const Step r = finish_prefill(R);   // this is the call that refills the loan
                        if (R.loan_held) { loan.release(R.id); R.loan_held = false; }
                        if (r == Step::error) R.fail = 1;
                        else R.phase = drv::Phase::decode;
                        if (!loan.held()) clear_deferred();   // the waiters can go now
                    } else if (s == DStep::decode) {
                        if (loan.held()) {
                            // A verify window may not read over a lent cache: `refill` puts the expert
                            // rows back.  Normally the hand-over's return_loan step already did this;
                            // it is belt-and-braces so a window can never see a lent slot.
                            if (!refill(err)) {
                                std::printf("%s\n", sp_out.err("refilling a lent slot failed: " + err, R.id).c_str());
                                std::fflush(stdout);
                                err.clear();
                                R.fail = 1;
                            } else {
                                for (ReqCtx& Q : live) Q.loan_held = false;
                                loan.release(loan.owner);
                                clear_deferred();
                            }
                        }
                        if (ran) {
                            const Step r = run_decode_step(R);
                            if (r == Step::progressed) R.phase = drv::Phase::decode;
                            else if (r == Step::finished || r == Step::cancelled) R.phase = drv::Phase::done;
                            else if (r == Step::error) R.fail = 1;
                            else if (r == Step::fatal_exit) R.fail = 2;
                        }
                    } else if (s == DStep::finish || s == DStep::unwind) {
                        // A slot cancelled BETWEEN steps never ran the window that would have set
                        // `finish = "cancel"` (`run_decode_step`'s `if (stopped())`).  Say the same
                        // thing, but only when the slot still had work left: a request that reached
                        // max_new or EOS keeps 0.1.30's "length"/"stop".  `R.cancelled` stays as it is
                        // on purpose - a decode-time cancel still records the tokens it produced
                        // (0.1.30's rule: only a prefill-time cancel sets it, and only then is `live`
                        // left alone).
                        if (unwinding && R.phase == drv::Phase::decode) R.finish = "cancel";
                        const Step r = finish_request(R);
                        if (r == Step::error) R.fail = 1;
                        else { R.phase = drv::Phase::done; R.finished = true; }
                    }
                    watch_idle(R.id);
                }   // ~MropeScope, then ~StepGuard - the serial path's order, one step at a time
                // `note_ran` is the fairness clock AND the watchdog's per-slot heartbeat.  A step that
                // did not run must NOT reset it, or a deferred slot would look served forever.
                if (ran) {
                    const int64_t t1 = drv::now_ms();
                    std::lock_guard<std::mutex> lk(slot_mu);
                    slots_reg.note_ran(R.id, t1, t1 > t0 ? t1 - t0 : 0);
                    ++steps_run;
                }
                // S3.6: the phase transition, as its own event.  The registry's SLOT line reports the
                // state machine's shape; this reports the DRIVER's, which is the one that decides what
                // step runs next, and it is the line that shows a slot going backwards to `prefill`
                // because it lost its state and is re-reading.  Read BEFORE `drop_ctx`, which destroys
                // the context this reference points into.
                const drv::Phase phase_after = R.phase;
                if (serve_trace) {
                    const std::string tl = drv::trace_phase(R.id, phase_before, phase_after,
                                                            R.finished ? "request ended" : "step result");
                    if (!tl.empty()) std::fprintf(stderr, "%s\n", tl.c_str());
                }
                if (R.finished) {
                    {
                        // `slot_finish` cleared the registry's active id; the SESSION still reflects
                        // this conversation until the next hand-over, so put the two back in step.
                        std::lock_guard<std::mutex> lk(slot_mu);
                        slots_reg.set_active(mounted_id);
                    }
                    // S3.6: a request that is over no longer needs its state saved, so a later
                    // hand-over away from it is never refused for its sake.  `drop_ctx` removes the
                    // entry too; this makes the answer correct in the window between the step that
                    // ended the request and the one that frees the row.
                    working_set(R.id, strata::program::serve_swap::Outgoing::finished);
                    drop_ctx(R.id);
                }
            };
            // ---- the loop ----------------------------------------------------------------------
            // S3.6 — CATCH THE PARKING BUDGET AT ADMISSION, not 97 seconds later.
            //
            // `parking_off_refuses_slots` already refuses `--serve-slots >= 2` when the cache is off.
            // The owner's box was the next case, and the one that actually bit: a budget of 2 GiB is
            // not zero, so it passed every existing check - but a conversation at this
            // --max_context needs a snapshot of ~3.4 GiB, so it could NEVER be parked, and the
            // engine discovered that mid-decode by destroying the conversation.  Size the worst case
            // here, with the engine's own sizing call, and say what concurrency is actually possible
            // on this machine before the first request arrives.
            //
            // S3.7: `ceiling` (declared above, where `admit_one` can read it) is filled here.  It is
            // also the FALLBACK price of one slot for RAM admission - the per-token rate it implies is
            // what a request is charged against until a real park has been measured - so it has to be
            // computed before the driver line that reports it and before the first request is admitted.
            {
                ceiling.budget_bytes = conversations.budget();
                ceiling.slots = conversations.slots();
                ceiling.size_at((uint64_t) o.max_context);
                // The largest branch the session could ever be asked to save: every cell of the
                // context, no checkpoints (the chain only adds to the figure, so this is a floor of
                // the worst case, which is the honest number to quote).
                std::vector<int32_t> worst((size_t) o.max_context, 0);
                std::vector<ImgKey> no_imgs;
                std::vector<ConvCheckpoint> no_checks;
                const strata::core::ConversationView view{worst, no_imgs, no_checks, cvec_cached};
                std::string cerr;
                if (strata::core::conversation_snapshot_bytes(view, ss, conv_stages, g, mtp.kv_state(),
                                                              ceiling.snapshot_bytes, cerr))
                    ceiling.sized = true;
                else
                    std::fprintf(stderr, "strata serve: could not size a worst-case snapshot at "
                                         "--max_context %lld (%s) - the parking-budget admission check "
                                         "is skipped\n", (long long) o.max_context, cerr.c_str());
            }
            std::fprintf(stderr, "%s\n", drv::driver_line(slots, o.starve_ms, swaps_on,
                                                          conversations.budget(), conversations.slots(),
                                                          ceiling.per_token(), o.decode_tokens).c_str());
            {
                const drv::ParkCeiling& ceil = ceiling;
                if (ceil.sized && ceil.useless_budget())
                    std::fprintf(stderr, "%s\n", drv::park_ceiling_line(slots, ceil, driver_on).c_str());
                else if (ceil.sized)
                    // S3.7: quote the RATE as well as the worst case. `budget / max_context` is the
                    // only per-token figure the engine has before anything has been parked, and it is
                    // what makes the answer to "does a short conversation cost less?" concrete: yes,
                    // proportionally, and this is the proportion.
                    std::fprintf(stderr, "strata serve: a conversation at this --max_context parks in "
                                         "%lld MiB, and the budget is %lld MiB - up to %d such "
                                         "conversation(s) can be swapped. A parked conversation costs "
                                         "about %lld B per context token, so a shorter one costs "
                                         "proportionally less; nothing is reserved per slot.\n",
                                 (long long) (ceil.snapshot_bytes >> 20),
                                 (long long) (ceil.budget_bytes >> 20), ceil.capacity(),
                                 (long long) ceil.per_token());
                std::fflush(stderr);
            }
            // S3.4's `/status` and `/metrics` need the number the serial INFO line can only state once
            // (it is printed before any request exists).  The driver reports it again, live, on stderr
            // every time the active set changes, so the Monitor tab can show N conversations without a
            // new wire line.  stdout is untouched: §6.1's rule is that the SLOT line carries state.
            std::fflush(stderr);
            int last_active = -1, peak_active = 0;
            auto report_active = [&]() {
                std::lock_guard<std::mutex> lk(slot_mu);
                const int a = slots_reg.active_count();
                if (a > peak_active) peak_active = a;
                if (a == last_active) return;
                last_active = a;
                std::fprintf(stderr, "strata serve: slots=%d slots_active=%d swaps=%lld swap_ms=%lld\n",
                             slots, a, (long long) slots_reg.swaps(), (long long) slots_reg.swap_ms());
                std::fflush(stderr);
            };
            // S3.6: the PERIODIC activity line.  `report_active` above only fires when the active set
            // CHANGES, so a run where two conversations were genuinely interleaved and a run that
            // never got past one look identical in the log - the only difference is the ABSENCE of a
            // line, which is the least greppable fact there is.  This one is on a time bound and
            // carries the peak, so `peak=1` vs `peak=2` is a positive statement about the run.
            const int64_t activity_every_ms = [] {
                if (const char* v = std::getenv("STRATA_SERVE_ACTIVITY_S")) {
                    const int n = std::atoi(v);
                    return n <= 0 ? (int64_t) 0 : (int64_t) n * 1000;
                }
                return (int64_t) 30000;   // 30 s: cheap, and far below any interesting request
            }();
            int64_t next_activity_ms = drv::now_ms() + activity_every_ms;
            auto report_activity = [&]() {
                if (activity_every_ms <= 0) return;
                const int64_t now = drv::now_ms();
                if (now < next_activity_ms) return;
                next_activity_ms = now + activity_every_ms;
                int a = 0;
                {
                    std::lock_guard<std::mutex> lk(slot_mu);
                    a = slots_reg.active_count();
                    if (a > peak_active) peak_active = a;
                }
                std::fprintf(stderr, "%s\n",
                             drv::activity_line(slots, a, peak_active, (long long) slots_reg.swaps(),
                                                (long long) slots_reg.swap_ms(), steps_run, admitted,
                                                refused, conversations.size(), conversations.bytes(),
                                                conversations.budget(), holds.line(now)).c_str());
                std::fflush(stderr);
            };
            struct DriverBusy {
                DriverBusy() { strata::core::progress().busy.store(true); strata::core::progress_at("request"); }
                ~DriverBusy() { strata::core::progress().busy.store(false); strata::core::progress_at("idle"); }
                void off() { strata::core::progress().busy.store(false); strata::core::progress_at("idle"); }
                void on() { strata::core::progress().busy.store(true); strata::core::progress_at("request"); }
            } busy_scope;
            while (!driver_quit && !driver_fatal) {
                // Progress for THIS pass.  A pass that changes nothing (a refused hand-over, a cap
                // that has not freed, a pick that keeps choosing a slot that cannot run) must never
                // become a CPU spin, so the pass that follows a still one sleeps first.  It is checked
                // at the TOP because every no-progress path `continue`s: a check at the bottom of the
                // body would be unreachable.  This is a backstop, not the mechanism - the mechanisms
                // are the deferral queue and `pick_unblocked` - because a scheduler with several
                // independent reasons to skip a step is exactly where a per-reason fix misses one.
                if (!pass_moved) std::this_thread::sleep_for(std::chrono::milliseconds(2));
                pass_moved = false;
                // 1. the watchdog's verdicts first (R3): they free permits and change what is pickable.
                for (;;) {
                    int64_t kid = proto::kNoId;
                    {
                        std::lock_guard<std::mutex> lk(watch_mu);
                        if (!watch_kill.empty()) { kid = watch_kill.front(); watch_kill.pop_front(); }
                    }
                    if (kid == proto::kNoId) break;
                    kill_slot(kid);
                    pass_moved = true;
                }
                // 2. admit, bounded by the cap and by RAM.  One admission per pass, so a burst of lines
                //    cannot starve the slots already running.
                admit_one();
                if (driver_quit) break;
                // 2a. S4.2 "hold, don't reject": the queue's deadlines first (an expired wait is an
                //     honest ERR that names what it waited for), then let the head through if the
                //     resource it waited for has come back.  Rate-limited, because the pass loop runs
                //     thousands of times a second and re-asking means reading /proc's free RAM.
                if (!holds.empty()) {
                    expire_waiters();
                    const int64_t nownow = drv::now_ms();
                    if (nownow >= next_promote_ms) {
                        next_promote_ms = nownow + 20;
                        promote_waiters();
                    }
                    report_waits();
                }
                report_active();
                report_activity();
                // 2b. bring a pending request in, as its own step.  This is §3.1's "drain the request
                // queue; admit/refuse", and it is a STEP: `prep_request` mounts, dispatches sampling
                // and can take tens of milliseconds, so it must not be smuggled into another slot's
                // step.  It only runs when a permit is actually free, so the transition to
                // `prefilling` inside `prep_request` cannot be refused.
                if (ReqCtx* P = pending_ctx()) {
                    bool free_permit = false;
                    {
                        std::lock_guard<std::mutex> lk(slot_mu);
                        free_permit = slots_reg.can_admit();
                    }
                    if (free_permit) {
                        pass_moved = true;
                        // S3.6: `prep_request` owns the REQUEST-LINE hand-over, and it cannot publish
                        // the outgoing slot's branch itself - at the point its body was written the
                        // driver's `live` deque does not exist yet, so the name `live` inside it is the
                        // serve-scope token vector.  Publish here, before the admit step, or the
                        // parking guard sees a mid-decode slot with nothing to save and refuses a
                        // hand-over that would otherwise have been perfectly parkable - turning a new
                        // client's first request into an ERR.  S3.8: a mid-PROMPT slot needs the same
                        // publish, or the admit step's hand-over erases its read.
                        if (ReqCtx* OUT = ctx_of(mounted_id)) {
                            publish_working_branch(*OUT);
                            working_set(OUT->id, drv::outgoing_for(OUT->phase, OUT->at));
                        }
                        const int64_t pid = P->id;      // `run_one` may drop the context (a rejected
                        run_one(*P, drv::Step::admit, "pending-admit");  // prep), so the id is
                        ReqCtx* A = ctx_of(pid);
                        if (A != nullptr) {
                            if (A->fail == 2) { std::fflush(stdout); std::fflush(stderr); std::_Exit(1); }
                            if (A->fail == 1) { driver_fatal = true; break; }
                        }
                        continue;
                    }
                }
                // 3. pick.
                const int64_t now = drv::now_ms();
                strata::program::slot::Pick p;
                {
                    std::lock_guard<std::mutex> lk(slot_mu);
                    p = slots_reg.pick(now);
                }
                if (serve_trace)
                    std::fprintf(stderr, "%s\n", drv::trace_pick("registry", p).c_str());
                if (p.action == PAct::idle) {
                    // Nothing runnable.  A line already in hand that admission deferred (the cap, an
                    // image request holding the position table) is handled by the pass-level backstop
                    // above, so just wait for the next line here.  Release the watchdog's "busy" while
                    // blocked, exactly as 0.1.30's BusyScope did between iterations.
                    busy_scope.off();
                    // S3.2b: the engine is idle, the GPU is free, and the caches may be missing rows the
                    // last prompt left out.  Walk them home for a bounded moment before blocking on stdin,
                    // so the next request's decode starts from a warmer cache.  Bounded because this same
                    // thread is the one that has to notice an arriving request - and `line_waiting()` lets
                    // the drain stop early when one has.
                    drain_loans_idle(250, [&]() {
                        std::lock_guard<std::mutex> lk(in_mu);
                        return !in_lines.empty() || in_eof;
                    });
                    const bool got = !held_line.empty() || next_line(held_line);
                    busy_scope.on();
                    if (!got) break;
                    pass_moved = true;   // a real line arrived: admit it on the next pass, no sleep
                    continue;
                }
                if ((p.action == PAct::run || p.action == PAct::swap) && is_deferred(p.id)) {
                    const strata::program::slot::Pick q = pick_unblocked(now);
                    if (q.action != PAct::idle) p = q;
                    else {
                        // Everything runnable is waiting for the loan and its holder is not runnable.
                        // Yield rather than spin; the holder becomes runnable on the next pass.
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                        continue;
                    }
                }
                ReqCtx* R = ctx_of(p.id);
                if (R == nullptr) {
                    // A row with no context: its request was refused or finished and its release was
                    // missed.  Say so loudly and free the permit - a stranded permit is the one bug that
                    // would deadlock admission, because the permit IS the bound on everything else.
                    std::fprintf(stderr, "strata serve: slot %lld picked with no request context - releasing it\n",
                                 (long long) p.id);
                    end_slot(p.id, "internal: this request's slot lost its state and was released");
                    watch_forget(p.id);
                    pass_moved = true;
                    continue;
                }
                // 4. fairness (OQ4).  `pick()` bounds a SWAP by --starve-ms; what only the driver can
                //    see is a slot that stays pickable but never gets served (the loan, a refused
                //    hand-over, a cap that never frees).  A real wedge is caught by the watchdog (the
                //    heartbeat freezes and every watched slot stalls with it), so here the bound only
                //    reports: killing a slot that is fairly waiting behind a 36 k-token read would be
                //    worse than the bug it guards against.
                if (drv::slot_starved_to_death(p.waited_ms, (int64_t) watchdog_limit_s * 1000)) {
                    static int64_t last_warn_ms = 0;
                    if (now - last_warn_ms >= 10000) {
                        last_warn_ms = now;
                        std::fprintf(stderr, "strata serve: slot %lld has waited %lld ms for the GPU "
                                             "(starve bound %lld ms, %lld slot(s) deferred on the prompt "
                                             "loan)\n", (long long) p.id, (long long) p.waited_ms,
                                     (long long) o.starve_ms, (long long) deferred.size());
                    }
                }
                // 5. dispatch.  R8 is asked FIRST, before the hand-over: a swap is a full
                //    save+restore (R10), so moving the session to a slot only to discover it cannot
                //    lend, then moving it back next pass, is exactly the swap churn the design refuses.
                //    The question is about the step the slot would ACTUALLY run, so a cancelled slot
                //    (whose dispatch is the phase tail, which never lends) is never parked behind a
                //    loan it does not need.
                arm_prompt_view(*R);
                DStep s = drv::dispatch_step(p, R->phase);
                const bool next_lends = (s == DStep::prefill || s == DStep::swap) &&
                                        R->phase == drv::Phase::prefill && segment_needs_loan(*R);
                if (next_lends && !loan.may_lend(R->id)) {
                    if (!is_deferred(R->id)) {
                        deferred.push_back(R->id);
                        ++deferred_hits;
                        std::fprintf(stderr, "strata serve: slot %lld waits for the prompt loan "
                                             "(held by slot %lld)\n", (long long) R->id,
                                     (long long) loan.owner);
                        if (serve_trace)
                            std::fprintf(stderr, "%s\n", drv::trace_loan_defer(R->id, loan.owner).c_str());
                    }
                    continue;
                }
                if (is_deferred(R->id)) un_defer(R->id);
                if (s == DStep::swap) {
                    req_id = R->id;      // `swap_to`'s own messages name the request it is moving to
                    std::string serr;
                    const drv::SwapResult sr = do_swap(*R, serr, p.why);
                    if (sr == drv::SwapResult::fatal) { driver_fatal = true; break; }
                    if (sr == drv::SwapResult::end_incoming) {
                        // S3.6: the outgoing conversation can NEVER be parked here.  The incoming
                        // request gets an ERR naming both numbers and its permit comes back; the
                        // session was not written, so every other conversation is untouched.
                        if (R->loan_held) { loan.release(R->id); R->loan_held = false; clear_deferred(); }
                        end_slot(R->id, serr);
                        drop_ctx(R->id);
                        pass_moved = true;
                        continue;
                    }
                    if (sr != drv::SwapResult::mounted) {
                        // Refused and recoverable later (the running read has not finished, R7's
                        // exclusivity, a residency drain that failed).  Do NOT end the request and do
                        // NOT re-pick it this pass: park it behind the mounted slot, exactly like a
                        // slot parked behind the prompt loan, or the loop spins on the same refusal.
                        if (!is_deferred(R->id)) { deferred.push_back(R->id); ++deferred_hits; }
                        continue;
                    }
                    if (mounted_id != R->id) {
                        // The hand-over was refused and the retry without a restore was refused too
                        // (R7's exclusivity, a residency drain that failed, a snapshot save that
                        // failed).  That is not transient, and re-picking the slot would spin on the
                        // same refusal, so this request ends and its permit comes back.  The session
                        // was not written, so the other conversations are untouched - which is the
                        // whole point of validating before saving (S3.1d).
                        std::fprintf(stderr, "strata serve: swap to slot %lld failed (%s) - ending that "
                                             "request; the session was not written\n",
                                     (long long) R->id, serr.c_str());
                        if (R->loan_held) { loan.release(R->id); R->loan_held = false; clear_deferred(); }
                        end_slot(R->id, "slot hand-over failed: " + serr);
                        drop_ctx(R->id);
                        continue;
                    }
                    strata::program::slot::Pick q;
                    q.action = PAct::run;
                    q.id = p.id;
                    q.state = p.state;
                    q.waited_ms = p.waited_ms;
                    q.why = p.why;
                    s = drv::dispatch_step(q, R->phase);
                }
                if (s == DStep::none) continue;
                pass_moved = true;
                run_one(*R, s, p.why);
                // 6. a failure the step could not act on itself (returning from inside a step lambda
                //    would have skipped the slot's unwind and leaked its permit).
                ReqCtx* A = ctx_of(p.id);
                if (A != nullptr) {
                    if (A->fail == 2) { std::fflush(stdout); std::fflush(stderr); std::_Exit(1); }
                    if (A->fail == 1) { driver_fatal = true; break; }
                }
            }
            // The engine is ending: give every live slot its answer rather than closing the pipe under
            // the server, then exit normally.
            for (ReqCtx& R : live)
                if (R.slot_open) std::printf("%s\n", sp_out.err("engine shutting down", R.id).c_str());
            std::fflush(stdout);
            std::fprintf(stderr, "strata serve: concurrent driver finished: %lld admitted, %lld steps, "
                                 "%lld swaps, %lld refused, %lld deferred step(s), %lld slot(s) still live\n",
                         (long long) admitted, (long long) steps_run, (long long) swaps_done,
                         (long long) refused, (long long) deferred_hits, (long long) live.size());
            // S3.2b: the loan's process totals.  `handoffs` over `refilled` is the whole point of the lazy
            // rule - a process that handed the loan between slots many times while refilling few rows is one
            // that stopped paying the end-of-request refill.  `still out` is what the caches are missing at
            // the moment the driver stops, which is the honest cost of never having copied them home.
            if (!pf_parts.empty()) {
                const std::array<int64_t, 6> t = loan_totals(pf_parts);
                std::fprintf(stderr, "strata serve: prompt loan totals: %lld lend(s), %lld hand-off(s), "
                                     "%lld relayout, %lld relayout skipped, %lld grown, %lld rows refilled "
                                     "eagerly, %lld pumped home, %lld row(s) still out of the caches (%s)\n",
                             (long long) loan.acquires, (long long) loan.handoffs,
                             (long long) t[0], (long long) t[1], (long long) t[2], (long long) t[3],
                             (long long) t[5], (long long) loan_outstanding(pf_parts),
                             loan_policy.lazy ? "lazy" : "eager");
            }
            std::fflush(stderr);
            return driver_fatal ? 1 : 0;
        }
        // ---- S3.1e-2: everything below is 0.1.30's serial driver, unchanged.  The gate above is an
        // early return, so this loop's text is byte-identical to the pre-S3.1e-2 source.
        while (next_line(line)) {
            if (line == "QUIT") break;
            // the watchdog watches a request from here until this iteration ends, whichever way it ends
            struct BusyScope {
                BusyScope() { strata::core::progress().busy.store(true); strata::core::progress_at("request"); }
                ~BusyScope() { strata::core::progress().busy.store(false); strata::core::progress_at("idle"); }
            } busy_scope;
            stop_req.store(false);   // a STOP that arrived between requests is stale
            ++request_index;         // S0.3 lever 4: the parking backoff's window is in requests
            err.clear();
            // S3.1e-1: everything the old body kept as per-request locals.  Constructed here, once per
            // iteration, so its lifetime is those locals' lifetime.
            ReqCtx R;
            // Whatever happens to this request from here - a validation ERR, a body ERR, a normal DONE - its
            // row is finished at the end of the iteration.  A guard rather than a call at every exit: the loop
            // has a dozen `continue`s, and a leaked row would strand an active-slot permit, which under
            // --serve-slots is the resource that bounds everything (§2.3).
            SlotGuard slot_guard{&R, tagged ? std::function<void(int64_t)>(slot_finish) : nullptr};
            // S3.1d R7, the guard.  The block below REWRITES the one host table and uploads it, for a
            // request that may still bail out (a bad embeddings file, a prompt too long for the
            // context, a token outside the vocabulary).  If it does, the slot that is actually mounted
            // is left with somebody else's positions in the device - and it is still mounted, so no
            // hand-over will put them back.  This guard restores the mounted slot's table on the way
            // out whenever the request did not become the mounted one.  On the normal path the swap
            // sets `mounted_id == req_id` and the guard does nothing.
            MropeScope mrope_scope;
            mrope_scope.skip = [&]() { return !swaps_on || !R.mrope_touched || mounted_id == req_id; };
            mrope_scope.restore = [&]() {
                if (mounted_id == strata::program::serve_proto::kNoId) return;
                apply_positions(conv_of(mounted_id));
                if (!mrope_host.empty()) upload_mrope_table();
                mrope_owner = mounted_id;
            };
            const Prep prep = prep_request(R, line);
            if (prep == Prep::rejected) continue;
            if (prep == Prep::fatal) return 1;

            // ---- S3.1e-1 phase 3: the prompt, one segment per step (0.1.30: one straight-line loop)
            plan_prompt_segments(R);
            Step st = Step::progressed;
            while ((st = run_prefill_step(R)) == Step::progressed) {}
            if (st == Step::error) return 1;
            if (st == Step::fatal_exit) {   // #224: a CUDA fault poisons the context for the whole process
                std::fflush(stdout);
                std::fflush(stderr);
                std::_Exit(1);
            }
            if (finish_prefill(R) == Step::error) return 1;
            // ---- S3.1e-1 phase 4: one verify window per step (0.1.30: one straight-line loop)
            Step dst = Step::progressed;
            while ((dst = run_decode_step(R)) == Step::progressed) {}
            if (dst == Step::error) return 1;
            // ---- S3.1e-1 phase 5: the request's tail (0.1.30: inline at the end of the body)
            if (finish_request(R) == Step::error) return 1;
        }
        return 0;
    }

    // ---- plan v0.3 P5: the prompt's conditioning positions [0, n_prompt - 1) in batched chunks.  The token loop
    // then starts at the last prompt position, whose prediction is the first generated token.
    int64_t pos_start = 0;
    int64_t spec_pos = 0;   // plan v0.3 P6: where the speculative loop starts (0 = not used)
    strata::prefill::Prefill prefill;
    double prefill_batched_ms = 0;
    std::FILE* final_r = o.dump_final_r.empty() ? nullptr : std::fopen(o.dump_final_r.c_str(), "wb");
    std::vector<float> final_r_host(final_r ? (size_t) (g.hc * g.n_embd) : 0);
    std::vector<std::pair<int32_t, int32_t>> lent;     // (residency index, slot) lent to the prompt path
    const int64_t n_batched = (o.prefill_until > 0 && o.prefill_until < n_prompt - 1) ? o.prefill_until : n_prompt - 1;
    if (o.prefill_chunk > 0 && n_prompt > 1) {
        void* borrow = nullptr;
        uint64_t borrow_bytes = 0;
        if (!o.no_prefill_borrow && !host_res.empty() && d_res != nullptr) {
            int64_t chunk = o.prefill_chunk;
            int64_t k = plan_lend(chunk);             // auto: the largest chunk that fits; fixed: halved to fit
            const int64_t request_sized = request_chunk(n_batched, chunk);
            if (k > 0 && request_sized < chunk) {                     // no bigger than this prompt segment needs
                chunk = request_sized;
                k = lend_slots(chunk);
                if (k + 128 > xcache.slots()) k = 0;
                if (!o.prefill_auto) o.prefill_chunk = chunk;
            }
            if (o.prefill_auto) {
                o.prefill_chunk = k > 0 ? chunk : request_chunk(n_batched, 1024);
                std::fprintf(stderr, "strata generate: prompt chunk auto: %lld tokens\n", (long long) o.prefill_chunk);
            } else if (chunk != o.prefill_chunk) {
                k = 0;                                 // a fixed chunk that does not fit: its own buffers, as before
            }
            const int64_t blob = (int64_t) strata::kernels::cpu::expert_layout().max_blob;
            if (k > 0) {   // the lent slots are refilled after the prompt
                const int32_t first = (int32_t) (xcache.slots() - k);
                for (size_t i = 0; i < host_res.size(); ++i)
                    if (host_res[i] >= first) {
                        lent.emplace_back((int32_t) i, host_res[i]);
                        host_res[i] = strata::core::kNotResident;
                    }
                cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
                borrow = xcache.device_slot(first);
                borrow_bytes = xcache.slot_offsets() ? (uint64_t) (xcache.bytes() - (int64_t) xcache.slot_offsets()[first])
                                                     : (uint64_t) k * (uint64_t) blob;
                std::fprintf(stderr, "strata generate: prompt path borrows %lld cache slots (%.2f GiB)\n", (long long) k,
                             (double) borrow_bytes / 1073741824.0);
            }
        }
        if (borrow == nullptr)
            std::fprintf(stderr, "strata generate: prompt path allocates its own buffers (no cache slots to borrow)\n");
        if (!prefill.init(wt, g, ss, srcp, o.expert_cache > 0 ? &xcache : nullptr,
                          host_res.empty() ? nullptr : host_res.data(), o.prefill_chunk, main_cs, err, borrow,
                          borrow_bytes)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        if (!o.mtp.empty()) {
            if (!mtp.bind(wt, &native_head, nullptr, err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            prefill.on_chunk = [&](const float* R_rows, int64_t T, int64_t p0, std::string& e) -> bool {
                // cell i pairs R_i with the token at i + 1 (every such token is in the prompt)
                std::vector<int32_t> nxt((size_t) T);
                for (int64_t t = 0; t < T; ++t) nxt[(size_t) t] = (int32_t) o.tokens[(size_t) (p0 + t + 1)];
                if (prefill.draft_kv(mtp, R_rows, nxt.data(), T, p0, e)) return true;   // E-9
                return e.empty() && mtp.prefill(R_rows, nxt.data(), T, p0, e);
            };
        }
        const Clock::time_point tp0 = Clock::now();
        if (!prefill.run(o.tokens.data(), n_batched, 0, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        // refill the lent slots from the arena and give them back to the decode tier
        if (!lent.empty()) {
            const Clock::time_point tr = Clock::now();
            for (const auto& [i, slot] : lent) {   // D-4: queued, one wait (STRATA_REFILL_BLOCKING=1: each)
                const uint8_t* b = srcp->blob(i / g.n_expert, i % g.n_expert);
                const int64_t nb = (int64_t) strata::kernels::cpu::expert_layout().blob_bytes(i / g.n_expert);
                if (b == nullptr || !(refill_blocking() ? xcache.fill_slot_blocking(slot, b, err, nb)
                                                        : xcache.fill_slot_queued(slot, b, err, nb))) {
                    std::fprintf(stderr, "strata generate: refilling a lent slot failed: %s\n", err.c_str());
                    return 1;
                }
                host_res[(size_t) i] = slot;
            }
            if (!xcache.sync_queued(err)) {
                std::fprintf(stderr, "strata generate: refilling the lent slots failed: %s\n", err.c_str());
                return 1;
            }
            cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
            std::fprintf(stderr, "strata generate: %zu lent slots refilled in %.1f ms\n", lent.size(),
                         std::chrono::duration<double, std::milli>(Clock::now() - tr).count());
        }
        prefill_batched_ms = std::chrono::duration<double, std::milli>(Clock::now() - tp0).count();
        prefill_ms += prefill_batched_ms;
        pos_start = n_batched;
        tok = o.tokens[(size_t) pos_start];
        // the PLE window of the token path: the two tokens before `pos_start`
        ss.ple_prev[0] = pos_start >= 2 ? (int32_t) o.tokens[(size_t) (pos_start - 2)] : -1;
        ss.ple_prev[1] = pos_start >= 1 ? (int32_t) o.tokens[(size_t) (pos_start - 1)] : -1;
        const strata::prefill::PrefillStats& ps = prefill.stats();
        std::fprintf(stderr, "strata generate: prefill %lld tokens in %lld chunks, %.1f ms (%.1f tok/s); experts "
                             "streamed %lld (%lld by DMA, host %.1f ms), resident %lld; PLE %.1f ms\n",
                     (long long) ps.tokens, (long long) ps.chunks, ps.ms_total,
                     ps.ms_total > 0 ? 1000.0 * (double) ps.tokens / ps.ms_total : 0.0, (long long) ps.experts_streamed,
                     (long long) ps.experts_dma, ps.ms_experts_host, (long long) ps.experts_resident, ps.ms_ple);
    }

    for (int64_t pos = pos_start;; ++pos) {
        // plan v0.3 P6: a native pack's last prompt token is the first verify window (T = 1)
        if (native_pack) { spec_pos = pos; break; }
        if (pos >= o.max_context) {
            std::fprintf(stderr, "strata generate: ran out of context at position %lld\n", (long long) pos);
            return 2;
        }
        // **THE TOKEN TIMER STARTS HERE, BEFORE ANY OF THE TOKEN'S WORK (A7).**  It used to start after
        // `put_input`/`embed_row`, which excluded the embedding and the PLE window advance from the reported
        // rate, and it stopped before the NaN scan, the logits dump and the sampler.  The published tok/s
        // figure is a WALL-CLOCK rate: everything one token costs, PLE advance to sampled id.  A rate that
        // excludes real per-token work is not a rate anyone can plan against.
        const Clock::time_point t0 = Clock::now();
        // **THE PLE'S TOKEN WINDOW ADVANCES HERE, ONCE PER TOKEN, AND `ple_stage_token` RUNS OUTSIDE THE
        // CAPTURE.**  Both are the driver's job: `ngram_rows` is a host hash over the last three tokens and the
        // table gather is a host read, so either one inside a captured graph would run once at capture time and
        // replay forever.  `ple_prev` is OLDEST FIRST and `-1` means "no predecessor", which `ngram_rows`
        // treats as the EOS cut - a sequence boundary.
        ss.ple_token = (int32_t) tok;
        Clock::time_point tp = Clock::now();
        // Plan v0.3 P2: the 16 SSD reads start here and complete while the embedding is staged; `ms_ple` is
        // the issue plus the time still spent WAITING afterwards, i.e. the part the embedding did not hide.
        if (ss.ple.ready() && !strata::core::ple_issue_token(ss.ple, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_ple += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        if (!put_input(tok, pos)) return 1;
        {
            const Clock::time_point n = Clock::now();
            ms_embed += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        if (ss.ple.ready() && !strata::core::ple_finish_token(ss.ple, token_stream, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_ple += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }

        if (pos % 256 == 0 || pos + 1 >= n_prompt - 1)
            std::fprintf(stderr, "strata generate: position %lld, token %lld%s\n", (long long) pos, (long long) tok,
                         pos < n_prompt ? " (prompt)" : "");
        // **`d.layers` IS THE BLOB'S LAYER AXIS, NOT A COUNTER.**  The adapter uses it to index
        // `experts.bin` as `layer * n_expert + expert`, so it MUST restart at 0 for every token.  Leaving it
        // running across tokens asks for layer 48 of a 48-layer file on the second token - which
        // `FileExpertSource` REFUSES rather than wrapping into layer 0's experts, and that refusal is the only
        // reason this was a clean error instead of a silently wrong second token.
        drive.d.layers = 0;
        drive.d.experts = 0;
        drive.d.failed = false;
        err.clear();
        if (o.no_capture) {
            if (!strata::core::session_token(wt, g, pos, /*pos_base=*/0, ss, d_parts, main_cs,
                                             o.sync_every_layer, err)) {
                std::fprintf(stderr, "strata generate: session_token: %s\n", err.c_str());
                return 1;
            }
        } else {
            strata::core::doorbell_reset(db);
            if (tgraph.captured) {
                if (!strata::core::session_run_token(g, pos, /*pos_base=*/0, ss, tgraph, pool_fn, pool_user,
                                                     loop_scratch.y_miss, main_cs, err)) {
                    std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                    return 1;
                }
            } else if (!strata::core::session_loop(g, pos, /*pos_base=*/0, ss, gr, pool_fn, hit_fn, pool_user, /*overlap=*/true, main_cs,
                                        err, layer_stage, &loop_scratch)) {
                std::fprintf(stderr, "strata generate: session_loop: %s\n", err.c_str());
                return 1;
            }
        }
        if (final_r != nullptr) {
            cudaMemcpy(final_r_host.data(), ss.R, final_r_host.size() * sizeof(float), cudaMemcpyDeviceToHost);
            const int64_t posrec[2] = {pos, tok};
            std::fwrite(posrec, sizeof posrec, 1, final_r);
            std::fwrite(final_r_host.data(), sizeof(float), final_r_host.size(), final_r);
        }
        if (drive.d.failed) {
            std::fprintf(stderr, "strata generate: the expert pool failed at layer %lld expert %lld: %s\n",
                         (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                         drive.d.fail ? drive.d.fail : "(no message)");
            return 1;
        }
        {
            // **THE LAYER LOOP ITSELF, WHICH IS WHAT `--gpu-only-full` HAS TO BE COMPARED AGAINST.**
            const Clock::time_point n = Clock::now();
            ms_layers += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        // ---- the ladder for THIS position, in the order `session_loop` filled it: layer 0 first.
        if (layer_dump != nullptr) std::fwrite(layer_stage, sizeof(float), layer_floats, layer_dump);
        if (half_dump != nullptr) {
            std::fwrite(half_stage, sizeof(float), (size_t) g.n_layers * (size_t) half_stride, half_dump);
        }
        if (!run_head(token_stream)) {
            std::fprintf(stderr, "strata generate: lm_head: %s\n", err.c_str());
            return 1;
        }
        // **A CHECKPOINT AFTER THE HEAD, BECAUSE AN ASYNC FAULT IS STICKY AND LIES ABOUT WHERE IT HAPPENED.**
        // Measured, and it cost an hour: without this, `embed_row`'s D2H on the NEXT token reported "an illegal
        // memory access" at a plane offset that has nothing to do with the fault, and the layer that actually
        // faulted had completed its own error checks successfully - because its kernels had not run yet.  A
        // sticky error surfaces at the next SYNCHRONISING call, which is whatever happens to come next.
        if (!o.stream_token && cudaDeviceSynchronize() != cudaSuccess) {
            std::fprintf(stderr, "strata generate: the device faulted in lm_head at position %lld: %s\n",
                         (long long) pos, cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        {
            const Clock::time_point n = Clock::now();
            ms_head += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }
        const bool emit_logits = dump != nullptr &&
            strata::program::logits_selection::selected(pos, dump_positions, o.logits_stride);
        const bool read_logits = !o.stream_token || o.check_logits || emit_logits;
        if (read_logits && (cudaMemcpyAsync(logits.data(), d_logits, (size_t) n_vocab * 4,
                                           cudaMemcpyDeviceToHost, (cudaStream_t) token_stream) != cudaSuccess ||
                            cudaStreamSynchronize((cudaStream_t) token_stream) != cudaSuccess)) {
            std::fprintf(stderr, "strata generate: reading the logits back failed\n");
            return 1;
        }
        int bad = 0;
        if (read_logits) for (float v : logits) if (!std::isfinite(v)) ++bad;
        if (bad != 0) {
            std::fprintf(stderr, "strata generate: %d of %lld logits are not finite at position %lld\n", bad,
                         (long long) n_vocab, (long long) pos);
            return 1;
        }
        if (emit_logits && std::fwrite(logits.data(), sizeof(float), (size_t) n_vocab, dump) != (size_t) n_vocab) {
            std::fprintf(stderr, "strata generate: cannot write logits at position %lld\n", (long long) pos);
            std::fclose(dump);
            return 1;
        }
        {
            // **993 KB OF SYNCHRONOUS D2H AND A 248,320-FLOAT HOST SCAN, EVERY TOKEN.**  (The review's notes
            // say 151,936 floats; the artifact's `output.weight` is 248,320 rows, so the real figure is 1.6x
            // that - a number nobody had checked because nothing measured this term.)  R2.6 asks for the dump
            // and the scan to be behind flags; round 36 did exactly that and measured it SLOWER, because on
            // this driver a large blocking readback is also what flushes the pipeline for the sampler that
            // follows.  Timed so the claim can be re-checked rather than remembered.
            const Clock::time_point n = Clock::now();
            ms_readback += std::chrono::duration<double, std::milli>(n - tp).count();
            tp = n;
        }

        int next = 0;
        // The draw is Philox(seed, position), as in a verify window (row t at pos0 draws pos0 + t): a seed gives
        // the same text whether a token comes from this path or from the speculative loop below.
        sp.counter = (uint64_t) pos;
        strata::kernels::sample_tokens(d_logits, 1, (int) n_vocab, nullptr, 0, sp, d_next, token_stream);
        if (cudaMemcpyAsync(&next, d_next, sizeof(int), cudaMemcpyDeviceToHost,
                            (cudaStream_t) token_stream) != cudaSuccess ||
            cudaStreamSynchronize((cudaStream_t) token_stream) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: reading the sampled token back failed: %s\n",
                         cudaGetErrorString(cudaGetLastError()));
            return 1;
        }
        // The sampled-token synchronization also completes every captured QSA
        // status readback. Retain one status per layer so a later layer cannot
        // hide an earlier failure; no extra synchronization or token allocation.
        if (o.native_flash_attn_short) for (int64_t j = 0; j < ss.qsa_alloc; ++j) {
            const int64_t i = ss.qsa_ord0 + j;   // the global ordinal, as the session's carve names it
            const int32_t status = ss.qsa_states[i].host_step[strata::kernels::kStepCount];
            if (status != 0) {
                std::fprintf(stderr, "strata generate: native attention status %d at QSA layer %lld, position %lld\n",
                             status, (long long) i, (long long) pos);
                return 1;
            }
        }
        if (next < 0 || next >= n_vocab) {
            std::fprintf(stderr, "strata generate: the sampler returned %d, outside 0..%lld\n", next,
                         (long long) (n_vocab - 1));
            return 1;
        }
        {
            // **TWO DEVICE-WIDE SYNCS FOR FOUR BYTES.**  `sample_tokens(nullptr)` ends in
            // `cudaDeviceSynchronize()` (`sampler.cu:245`) and the blocking 4-byte read below is the second.
            const Clock::time_point n = Clock::now();
            ms_sample += std::chrono::duration<double, std::milli>(n - tp).count();
            ++phase_tokens;
        }
        // CHARGED HERE, AFTER THE SAMPLER, so the wall-clock rate covers the whole token including the embedding,
        // the NaN scan, the logits readback and the sample (A7).  Only DECODE positions count; prefill is
        // measured separately.
        if (pos >= n_prompt - 1) total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        else prefill_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        if (pos == n_prompt - 1) ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
        // **THE PREDICTION AT THE LAST PROMPT POSITION *IS* THE FIRST GENERATED TOKEN.**  Sampling on every
        // position and recording only from `n_prompt - 1` onward is what keeps the two cases from needing
        // separate handling - and the version that "obviously" only samples after the prompt loses exactly one
        // token's worth of conditioning.
        if (pos >= n_prompt - 1) produced.push_back(next);
        if ((int64_t) produced.size() >= o.max_new) break;
        if (o.stop_eos && pos >= n_prompt - 1 &&
            std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) next) != o.eos_ids.end()) break;
        // TEACHER FORCING while the prompt lasts: the next input is the prompt's own next token, not the
        // model's guess.  Feeding the guess would make the run depend on the model's own errors from position
        // 1, which is a different (and worse) measurement of the same prompt.
        // and the window advances: the token just decoded becomes the newest predecessor.
        ss.ple_prev[0] = ss.ple_prev[1];
        ss.ple_prev[1] = (int32_t) tok;
        tok = (pos + 1 < n_prompt) ? o.tokens[(size_t) (pos + 1)] : next;
        // Plan v0.3 P6: from the first generated token on, the speculative loop below takes over.
        if (o.spec > 0 && pos >= n_prompt - 1) { spec_pos = pos + 1; break; }
    }

    // ================================ plan v0.3 P6: SPECULATIVE DECODING ================================
    //
    // Each round verifies [the last emitted token, drafts...] in one window; the window's argmax after token t
    // is exactly what greedy decode would emit there, so the first draft that differs ends the round and the
    // round emits (accepted drafts + 1) tokens.  `commit` keeps the state of the tokens that were emitted.
    const bool ended = o.stop_eos && !produced.empty() &&
                       std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) produced.back()) != o.eos_ids.end();
    if (spec_pos > 0 && (int64_t) produced.size() < o.max_new && !ended) {
        std::vector<int64_t> oracle;
        if (!o.spec_oracle.empty()) {
            std::ifstream in(o.spec_oracle);
            std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            std::string e;
            if (!in || !parse_i64_list(text.c_str(), oracle, e)) {
                std::fprintf(stderr, "strata generate: cannot read --spec-oracle %s\n", o.spec_oracle.c_str());
                return 2;
            }
        }
        if (thits.d_res == nullptr) {
            std::fprintf(stderr, "strata generate: --spec needs the device residency table (--expert-profile, "
                                 "--expert-cache and the token graph)\n");
            return 2;
        }
        mem_mark("the head and the prompt path");
        strata::core::Verifier ver;
        strata::core::VerifyHits vh;
        vh.d_res = thits.d_res;
        vh.cache_base = thits.cache_base;
        vh.blob = thits.blob;
        vh.slot_off = xcache.slot_offsets();   // E-6: the device plan's pointers
        vh.n_slots = xcache.slots();
        if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        const bool use_mtp = !o.mtp.empty();
        if (use_mtp && !mtp.bind(wt, &native_head, ver.final_R_all(), err)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        mem_mark("the verifier and the drafter's binding");
        ver.set_sampling(sp);   // the CLI's own sampling (until 0.1.19 this loop was always greedy); no penalties here
        if (use_mtp) mtp.set_draft_sampling(sp);   // STRATA_SPEC_COUPLED=1: sampled drafts (a no-op otherwise)
        ver.set_split(o.spec_split);
        // auto: the copy kernel for every pack.  DMA (the native packs' default until 0.1.13) has the host call
        // cudaMemcpyAsync + cudaLaunchHostFunc inside a verify window while the GPU spins on the flag they raise;
        // issue #31's thread dumps show the host stuck in that cudaMemcpyAsync on a driver lock for good.  The copy
        // kernel needs no host CUDA call there, and costs ~1-3% decode on IQ3_S (45.3 -> 44.8 tok/s, 8 requests).
        ver.set_pcie_mode(o.pcie_mode == "dma" ? 0 : o.pcie_mode == "direct" ? 1 : 2);
        drive.d.plan = ver.plan_sink();
        drive.d.pcie_num = (int) (o.pcie_frac * 256.0 + 0.5);
        if (drive.d.pcie_num < 0) drive.d.pcie_num = 0;
        if (drive.d.pcie_num > 256) drive.d.pcie_num = 256;
        const int64_t pcie0 = drive.d.pcie_experts;
        if (o.adapt_every > 0 && o.adapt_swaps > 0) drive.d.usage.assign((size_t) (g.n_layers * g.n_expert), 0.0f);
        int64_t swaps_total = 0;
        double ms_adapt = 0;
        cudaStream_t adapt_stream = nullptr;
        if (!drive.d.usage.empty() && cudaStreamCreateWithFlags(&adapt_stream, cudaStreamNonBlocking) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: cannot create the refill stream\n");
            return 1;
        }
        // plan v0.3 P6: swaps in flight - (residency index, slot) admitted when adapt_ev has completed
        std::vector<std::pair<int32_t, int32_t>> pending;
        cudaEvent_t adapt_ev = nullptr;
        cudaEventCreateWithFlags(&adapt_ev, cudaEventDisableTiming);
        auto apply_pending = [&](bool wait) {
            if (pending.empty()) return;
            if (wait) cudaEventSynchronize(adapt_ev);
            else if (cudaEventQuery(adapt_ev) != cudaSuccess) return;
            src.commit_exchanges();   // the resident RAM mode: the evicted experts take their places in RAM
            for (const auto& [i, slot] : pending) host_res[(size_t) i] = slot;
            pending.clear();
            if (d_res != nullptr)
                cudaMemcpy(d_res, host_res.data(), host_res.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        };
        // Plan v0.3 P6: the VRAM tier follows the conversation.  Candidates are missing experts routed at least
        // twice (decayed); each is paired with its layer's least-routed resident expert and swapped when it was
        // routed clearly more often.  Copies run between rounds, when the GPU is idle.
        auto adapt = [&]() -> bool {
            const Clock::time_point ta = Clock::now();
            if (!pending.empty()) return true;   // the previous swaps are still in flight
            struct Swap { float gain; int32_t layer, in, out; };
            std::vector<Swap> swaps;
            std::vector<std::pair<float, int32_t>> cand, vict;
            for (int64_t l = 0; l < g.n_layers; ++l) {
                cand.clear();
                vict.clear();
                const float* u = drive.d.usage.data() + l * g.n_expert;
                const int32_t* r = host_res.data() + l * g.n_expert;
                for (int32_t e = 0; e < (int32_t) g.n_expert; ++e) {
                    if (r[e] < 0) { if (u[e] >= 2.0f) cand.emplace_back(u[e], e); }
                    else vict.emplace_back(u[e], e);
                }
                if (cand.empty() || vict.empty()) continue;
                std::sort(cand.begin(), cand.end(), [](auto& a, auto& b) { return a.first > b.first; });
                const size_t nc = std::min(cand.size(), vict.size());
                std::partial_sort(vict.begin(), vict.begin() + (ptrdiff_t) nc, vict.end(),
                                  [](auto& a, auto& b) { return a.first < b.first; });
                for (size_t i = 0; i < nc; ++i) {
                    if (cand[i].first < vict[i].first + 1.5f) break;
                    swaps.push_back({cand[i].first - vict[i].first, (int32_t) l, cand[i].second, vict[i].second});
                }
            }
            std::sort(swaps.begin(), swaps.end(), [](const Swap& a, const Swap& b) { return a.gain > b.gain; });
            if ((int) swaps.size() > o.adapt_swaps) swaps.resize((size_t) o.adapt_swaps);
            if (!resident_stage_swaps(src, xcache, host_res, g.n_expert, swaps, adapt_stream)) {
                std::fprintf(stderr, "strata generate: an adaptive refill failed (copying evicted experts back)\n");
                return false;
            }
            for (const Swap& s : swaps) {
                const size_t in = (size_t) s.layer * g.n_expert + s.in, out = (size_t) s.layer * g.n_expert + s.out;
                const int32_t slot = host_res[out];
                const uint8_t* b = srcp->blob(s.layer, s.in);
                // asynchronous: the copies run while the MTP drafts; the next window waits for them
                if (slot < 0 || b == nullptr ||
                    cudaMemcpyAsync(xcache.device_slot(slot), b, (size_t) strata::kernels::cpu::expert_layout().blob_bytes(s.layer),
                                    cudaMemcpyHostToDevice, adapt_stream) != cudaSuccess) {
                    std::fprintf(stderr, "strata generate: an adaptive refill failed\n");
                    return false;
                }
                host_res[out] = strata::core::kNotResident;   // evicted now: the CPU computes it meanwhile
                pending.emplace_back((int32_t) in, slot);      // resident once the copy has landed
            }
            if (!swaps.empty()) cudaEventRecord(adapt_ev, adapt_stream);
            for (float& v : drive.d.usage) v *= 0.7f;
            swaps_total += (int64_t) swaps.size();
            ms_adapt += std::chrono::duration<double, std::milli>(Clock::now() - ta).count();
            return true;
        };
        int64_t p = spec_pos;
        int32_t x = (int32_t) tok;
        std::vector<int32_t> drafts((size_t) o.spec, 0);
        std::vector<float> dprob((size_t) o.spec, 1.0f);
        std::vector<int64_t> window_hist((size_t) o.spec + 1, 0);
        // plan v0.3 P6: with a native pack the first window is the last prompt token alone (it produces the first
        // generated token and the MTP's first cell); otherwise the token loop already did that.
        bool first_window = native_pack;
        if (use_mtp && !first_window &&
            !mtp.draft_first(o.spec, ss.R, x, p - 1, drafts.data(), err, dprob.data(), (float) o.spec_min_p)) {
            std::fprintf(stderr, "strata generate: %s\n", err.c_str());
            return 1;
        }
        std::vector<int32_t> window((size_t) o.spec), outv((size_t) o.spec);
        std::vector<int64_t> accepted_hist((size_t) o.spec, 0);
        int64_t rounds = 0, drafts_total = 0, drafts_ok = 0, corrupt_counter = 0;
        const int S_mtp = o.mtp_max_t > 0 ? std::min(o.mtp_max_t, o.spec) : o.spec;
        if (use_mtp && S_mtp < o.spec) mtp.set_max_drafts(S_mtp - 1);
        strata::spec::SuffixDrafter sfx(std::max(1, o.suffix_draft), 64, (size_t) o.max_context + 4096);
        strata::spec::DraftPolicy policy(o.spec);   // MTP or lookup window (see draft_policy.hpp)
        std::vector<int32_t> sbuf((size_t) o.spec, 0);
        int64_t sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
        if (o.suffix_draft > 0) {
            for (int64_t t : o.tokens) sfx.append((int32_t) t);
            for (int64_t t : produced) sfx.append((int32_t) t);
        }
        const double pool_ms0 = drive.cpu_ms;
        const int64_t misses0 = drive.d.multi_misses, entries0 = drive.d.multi_entries;
        while ((int64_t) produced.size() < o.max_new) {
            const Clock::time_point t0 = Clock::now();
            int T = S_mtp;
            if (use_mtp && o.spec_min_p > 0.0) {
                T = 1;
                while (T < S_mtp && dprob[(size_t) T - 1] >= (float) o.spec_min_p) ++T;
            }
            if (first_window) T = 1;
            bool from_sfx = false;
            int sfx_match = 0;
            if (o.suffix_draft > 0 && !first_window) {
                const int k = sfx.propose(o.spec - 1, sbuf.data());
                sfx_match = sfx.last_match();
                if (k > 0 && (!use_mtp || sbuf[0] == drafts[0])) {
                    const strata::spec::DraftPolicy::Pick pk = policy.choose(T, k, sfx_match);
                    if (pk.lookup) { T = pk.t; from_sfx = true; }
                }
            }
            const bool timed_round = !first_window;
            ++window_hist[(size_t) T];
            if (p + T > o.max_context) {
                std::fprintf(stderr, "strata generate: ran out of context at position %lld\n", (long long) p);
                return 2;
            }
            window[0] = x;
            for (int i = 1; i < T; ++i) {
                const size_t at = produced.size() - 1 + (size_t) i;
                int32_t d = from_sfx ? sbuf[(size_t) i - 1] : use_mtp ? drafts[(size_t) i - 1]
                                                    : at < oracle.size() ? (int32_t) oracle[at] : 0;
                if (o.spec_corrupt > 0 && (++corrupt_counter % o.spec_corrupt) == 0) d = (d + 1) % (int32_t) n_vocab;
                window[(size_t) i] = d;
            }
            drive.d.layers = 0;
            drive.d.experts = 0;
            drive.d.failed = false;
            apply_pending(false);
            if (!ver.run(T, window.data(), p, &drive_pool_multi, &drive, outv.data(), err)) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            if (drive.d.failed) {
                std::fprintf(stderr, "strata generate: the expert pool failed at layer %lld expert %lld: %s\n",
                             (long long) drive.d.fail_layer, (long long) drive.d.fail_expert,
                             drive.d.fail ? drive.d.fail : "(no message)");
                return 1;
            }
            int a = 0;
            while (a < T - 1 && window[(size_t) a + 1] == outv[(size_t) a]) ++a;
            if (first_window) {
                first_window = false;
                ttft_ms = std::chrono::duration<double, std::milli>(Clock::now() - t_start).count();
            }
            // plan v0.3 P6: the adaptive tier's host work (ranking, copy submission) runs on its own thread while the
            // GPU commits and drafts; it touches only the residency tables, which nothing reads until the next window
            std::thread adapt_thr;
            bool adapt_ok = true;
            if (!drive.d.usage.empty() && ((rounds + 1) % o.adapt_every) == 0)
                adapt_thr = std::thread([&] { adapt_ok = adapt(); });
            if (!ver.commit(a + 1, err)) {
                if (adapt_thr.joinable()) adapt_thr.join();
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            ++rounds;
            drafts_total += T - 1;
            drafts_ok += a;
            ++accepted_hist[(size_t) a];
            if (from_sfx) { ++sfx_windows; sfx_drafts += T - 1; sfx_ok += a; }
            bool eos = false;
            for (int i = 0; i <= a && (int64_t) produced.size() < o.max_new && !eos; ++i) {
                produced.push_back(outv[(size_t) i]);
                if (o.suffix_draft > 0) sfx.append(outv[(size_t) i]);
                eos = o.stop_eos && std::find(o.eos_ids.begin(), o.eos_ids.end(), (int64_t) outv[(size_t) i]) != o.eos_ids.end();
            }
            if (eos) {
                if (adapt_thr.joinable()) adapt_thr.join();
                total_ms += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
                break;
            }
            const bool drafted = !use_mtp || (int64_t) produced.size() >= o.max_new ||
                                 mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(), (float) o.spec_min_p);
            if (adapt_thr.joinable()) adapt_thr.join();
            if (!adapt_ok) return 1;
            if (!drafted) {
                std::fprintf(stderr, "strata generate: %s\n", err.c_str());
                return 1;
            }
            x = outv[(size_t) a];
            p += a + 1;
            const double round_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
            total_ms += round_ms;
            if (timed_round) policy.observe(from_sfx, T, a, sfx_match, round_ms);
            if (rounds % 64 == 0)
                std::fprintf(stderr, "strata generate: position %lld, %lld tokens, %lld rounds\n", (long long) p,
                             (long long) produced.size(), (long long) rounds);
        }
        std::printf("%-24s %lld rounds of %d, drafts accepted %lld of %lld (%.3f), %.2f tokens per round\n",
                    "speculation", (long long) rounds, o.spec, (long long) drafts_ok, (long long) drafts_total,
                    drafts_total > 0 ? (double) drafts_ok / (double) drafts_total : 0.0,
                    rounds > 0 ? (double) (drafts_ok + rounds) / (double) rounds : 0.0);
        if (o.spec_min_p > 0.0) {
            std::printf("%-24s", "window sizes");
            for (size_t i = 1; i < window_hist.size(); ++i) std::printf(" T%zu:%lld", i, (long long) window_hist[i]);
            std::printf("  (min draft probability %.2f)\n", o.spec_min_p);
        }
        if (o.suffix_draft > 0)
            std::printf("%-24s %lld windows, drafts accepted %lld of %lld\n", "suffix drafts", (long long) sfx_windows,
                        (long long) sfx_ok, (long long) sfx_drafts);
        std::printf("%-24s", "accepted per round");
        for (size_t i = 0; i < accepted_hist.size(); ++i) std::printf(" %zu:%lld", i, (long long) accepted_hist[i]);
        std::printf("\n");
        if (rounds > 0)
            std::printf("%-24s wait for rings %.3f  pool %.3f  host %.3f  commit %.3f ms/round; CPU experts %.2f "
                        "distinct / %.2f routed per layer\n",
                        "verify window", ver.ms_wait / rounds, ver.ms_pool / rounds, ver.ms_host / rounds,
                        ver.ms_commit / rounds,
                        (double) (drive.d.multi_misses - misses0) / (double) (rounds * g.n_layers),
                        (double) (drive.d.multi_entries - entries0) / (double) (rounds * g.n_layers));
        if (rounds > 0)
            std::printf("%-24s gate/up %.3f  quantize %.3f  down %.3f ms/round; %.1f GB/s over the rows phases; "
                        "CPU pool call %.3f ms/round\n", "pool multi", pool.ms_multi_gu / rounds,
                        pool.ms_multi_q / rounds, pool.ms_multi_down / rounds,
                        (double) pool.multi_bytes / 1e6 / std::max(1e-9, pool.ms_multi_gu + pool.ms_multi_down),
                        (drive.cpu_ms - pool_ms0) / rounds);
        if (rounds > 0)
            std::printf("%-24s plan %.3f  activation quantize %.3f  jobs %.3f  run %.3f ms/round\n", "dispatch",
                        drive.d.ms_plan / rounds, drive.d.ms_actq / rounds, drive.d.ms_jobs / rounds,
                        drive.d.ms_run / rounds);
        if (rounds > 0 && !drive.d.usage.empty())
            std::printf("%-24s %lld experts swapped into the VRAM tier (every %d rounds, %.3f ms/round)\n", "adaptive tier",
                        (long long) swaps_total, o.adapt_every, ms_adapt / rounds);
        if (src.complement_ready())
            std::printf("%-24s %.2f GiB of experts in RAM, %lld exchanged with the VRAM tier, %lld blob reads from "
                        "the file\n", "resident RAM", (double) src.resident_bytes() / 1073741824.0,
                        (long long) src.exchanges(), (long long) src.file_reads());
        if (rounds > 0 && drive.d.pcie_num > 0)
            std::printf("%-24s %.2f distinct experts per layer read over PCIe (share %d/256 of the misses)\n",
                        "pcie experts", (double) (drive.d.pcie_experts - pcie0) / (double) (rounds * g.n_layers),
                        drive.d.pcie_num);
        (void) pool_ms0;
        if (use_mtp && rounds > 0)
            std::printf("%-24s %.3f ms/round drafting (%lld rounds), MTP prompt %.1f ms, %.0f MiB of VRAM\n", "mtp",
                        mtp.ms_draft / (double) mtp.rounds, (long long) mtp.rounds, mtp.ms_prefill,
                        (double) mtp.vram_bytes() / 1048576.0);
    }

    if (dump != nullptr && std::fclose(dump) != 0) {
        std::fprintf(stderr, "strata generate: cannot finish logits dump\n");
        return 1;
    }
    if (layer_dump != nullptr) {
        std::fclose(layer_dump);
        cudaFreeHost(layer_stage);
        std::printf("%-24s %s (%lld layers + the input x %d streams x %lld per position)\n", "layers dumped",
                    o.dump_layers.c_str(), (long long) g.n_layers, (int) g.hc, (long long) g.n_embd);
    }
    if (half_dump != nullptr) {
        std::fclose(half_dump);
        cudaFreeHost(half_stage);
        std::printf("%-24s %s (%lld layers x %llu per position)\n", "halves dumped", o.dump_halves.c_str(),
                    (long long) g.n_layers, (unsigned long long) half_stride);
    }
    if (routing != nullptr) {
        std::fclose(routing);
        drive.routing = nullptr;
        std::printf("%-24s %s (%lld records of layer, k, ids, weights)\n", "routing dumped",
                    o.dump_routing.c_str(), (long long) drive.calls);
    }
    if (o.stage_timing) strata::core::stage_timing_report(g.n_layers);

    const int64_t decoded = (int64_t) produced.size();
    std::printf("prompt  :");
    for (int64_t t : o.tokens) std::printf(" %lld", (long long) t);
    std::printf("\noutput  :");
    for (int64_t t : produced) std::printf(" %lld", (long long) t);
    std::printf("\n");
    const double decode_ms = decoded > 0 ? total_ms / (double) decoded : 0.0;
    std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s\n", "decode", (long long) decoded, total_ms,
                decode_ms > 0.0 ? 1000.0 / decode_ms : 0.0);
    if (n_prompt > 1)
        std::printf("%-24s %lld tokens in %.1f ms  ->  %.2f tok/s  (time to first token %.1f ms)\n", "prefill",
                    (long long) (n_prompt - 1), prefill_ms,
                    prefill_ms > 0 ? 1000.0 * (double) (n_prompt - 1) / prefill_ms : 0.0, ttft_ms);
    if (!o.dump_mixed.empty()) {
        std::vector<float> mx((size_t) g.n_embd);
        if (cudaMemcpy(mx.data(), ss.block.mixed, mx.size() * sizeof(float), cudaMemcpyDeviceToHost) !=
            cudaSuccess) {
            std::fprintf(stderr, "strata generate: reading mixed back failed\n");
            return 1;
        }
        std::FILE* mf = std::fopen(o.dump_mixed.c_str(), "wb");
        if (mf == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_mixed.c_str());
            return 1;
        }
        std::fwrite(mx.data(), sizeof(float), mx.size(), mf);
        std::fclose(mf);
        double s2 = 0, mag = 0;
        for (float v : mx) { s2 += (double) v * (double) v; mag += std::fabs((double) v); }
        std::printf("%-24s %s (n_embd %lld, rms %.5g, mean|.| %.5g)\n", "mixed dumped", o.dump_mixed.c_str(),
                    (long long) g.n_embd, std::sqrt(s2 / (double) mx.size()), mag / (double) mx.size());
    }

    // ---- the residual, for bisecting the head against the layers (see `dump_residual`'s note)
    if (!o.dump_residual.empty()) {
        std::vector<float> R((size_t) g.hc * g.n_embd);
        if (cudaMemcpy(R.data(), ss.R, R.size() * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            std::fprintf(stderr, "strata generate: reading R back failed\n");
            return 1;
        }
        std::FILE* rf = std::fopen(o.dump_residual.c_str(), "wb");
        if (rf == nullptr) {
            std::fprintf(stderr, "strata generate: cannot write %s\n", o.dump_residual.c_str());
            return 1;
        }
        const int32_t hdr[2] = {(int32_t) g.hc, (int32_t) g.n_embd};
        std::fwrite(hdr, sizeof hdr, 1, rf);
        std::fwrite(R.data(), sizeof(float), R.size(), rf);
        std::fclose(rf);
        double mag = 0, mx = 0;
        int bad = 0;
        for (float v : R) {
            if (!std::isfinite(v)) ++bad;
            else { mag += std::fabs((double) v); mx = std::max(mx, (double) std::fabs((double) v)); }
        }
        std::printf("%-24s %s (%d x %lld, nonfinite %d, mean|.| %.4g, max|.| %.4g)\n", "residual dumped",
                    o.dump_residual.c_str(), (int) g.hc, (long long) g.n_embd, bad, mag / (double) R.size(), mx);
    }

    if (o.stats) {
        std::printf("%-24s %.3f ms/token (WALL CLOCK: embed, layers, head, sample)\n", "  per token", decode_ms);
        // **THE PER-TOKEN HOST TERM, WHICH `--gpu-only-full` CANNOT SEE.**  That measurement never enters the
        // token loop, so it excludes all six of these.  On the 78-token fixture + 200 generated tokens the six
        // sum to ~9 ms of non-layer work against ~1.5 ms of actual head GPU work - 16% of the token, and it is
        // not the pool.
        if (phase_tokens > 0) {
            const double pt = (double) phase_tokens;
            std::printf("%-24s PLE %.3f  embed %.3f  LAYERS %.3f  head %.3f  readback %.3f  sample %.3f  "
                        "(sum %.3f of %.3f ms)\n",
                        "  token host phases", ms_ple / pt, ms_embed / pt, ms_layers / pt, ms_head / pt,
                        ms_readback / pt, ms_sample / pt,
                        (ms_ple + ms_embed + ms_layers + ms_head + ms_readback + ms_sample) / pt, decode_ms);
        }
        if (const std::string io = ple_table.io_report(); !io.empty()) std::printf("  %s\n", io.c_str());
        // **THE DENOMINATOR IS THE POSITIONS THE POOL ACTUALLY RAN ON, NOT THE DECODED TOKENS (A6).**
        // `drive_pool` is called once per layer per position and PREFILL runs the loop too, so accumulating
        // `cpu_ms` over prefill and then dividing by `decoded` inflates this figure.  `drive.calls / n_layers`
        // is the number of positions - the same correction the ring counters below already received, which is
        // why they print "of 192" rather than "240 of 192".
        const double pool_positions = g.n_layers > 0 ? (double) drive.calls / (double) g.n_layers : 0.0;
        std::printf("%-24s %.3f ms/token over %lld layers (%.0f positions, %lld dispatches)\n",
                    "  the CPU expert pool", pool_positions > 0.0 ? drive.cpu_ms / pool_positions : 0.0,
                    (long long) g.n_layers, pool_positions, (long long) drive.calls);
        // **AND WHERE INSIDE `run()` IT WENT.**  Three phases per layer and they were one number, which cannot
        // tell a pool that is slow at the WORK from one that is slow at the SYNCHRONISATION - opposite fixes.
        // Wait-for-park is expected to be ~0 (the workers re-parked at the end of the previous layer); the
        // question is whether the time is in the drain or in the re-park barrier.
        if (pool_positions > 0.0) {
            double wp = 0, dr = 0, rp = 0;
            pool.phase_ms(wp, dr, rp);
            const double per = pool_positions;
            std::printf("%-24s   wait-park %.3f  drain %.3f  re-park %.3f  ms/token\n",
                        "  pool phases", wp / per, dr / per, rp / per);
        }
        std::printf("%-24s %lld blobs read\n", "  expert blobs", (long long) srcp->reads());
        for (int r = 0; r < 3; ++r) if (o.expert_cache_remote[(size_t) r] > 0)
            std::printf("  CUDA%d experts           %lld routed entries computed\n",
                        r + 1, (long long) remote_experts[(size_t) r].computed());
        // ---- **R4's DISPATCH MEASUREMENT: h, ON THE ENGINE'S OWN ROUTING.**  No offline trace, no corpus
        // question, no k-fold - these are the ids the router actually produced on this run.  Reported as
        // hits/lookups so it can be read directly as the h the cache would deliver, and alongside `refused`
        // so a full cache is visible rather than silently capping the rate.
        if (o.expert_cache > 0) {
            const int64_t look = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;
            const int64_t hl = drive.d.hit_ready + drive.d.hit_late;
            std::printf("%-24s %lld of %lld layers the hit work was DONE when the pool returned\n",
                        "  R4 overlap", (long long) drive.d.hit_ready, (long long) hl);
            std::printf("%-24s %lld of %lld = %.4f      (%lld admitted, %lld refused, cache %.4f%% full)\n",
                        "  R4 expert-cache hits", (long long) drive.d.cache_hits, (long long) look,
                        look > 0 ? (double) drive.d.cache_hits / (double) look : 0.0,
                        (long long) drive.d.cache_admitted, (long long) drive.d.cache_refused,
                        100.0 * (double) (drive.d.cache_admitted + drive.d.cache_hits > 0
                                              ? (double) xcache.resident() / (double) xcache.slots()
                                              : 0.0));
        }
        if (tgraph.captured && tgraph.calls > 0) {
            const double per = (double) tgraph.calls;
            std::printf("%-24s wait for rings %.3f  pool %.3f ms/token  (%lld flushes over %lld positions)\n",
                        "  token graph", tgraph.ms_wait / per, tgraph.ms_pool / per, (long long) tgraph.flushes,
                        (long long) tgraph.calls);
        }
        if (gr.captured && gr.calls_total > 0) {
            // The counters are CUMULATIVE over every `session_loop` call, and PREFILL runs the loop too - so
            // the denominator is the number of positions, not the number of generated tokens.  Dividing by
            // `n_layers * decoded` printed "240 of 192", which is a reporting bug that looks like a ring
            // firing more often than it should.
            const int64_t positions = gr.calls_total;
            std::printf("%-24s %lld of %lld over %lld positions\n", "  rings seen MID-GRAPH",
                        (long long) gr.rings_mid_graph, (long long) (g.n_layers * positions),
                        (long long) positions);
            std::printf("%-24s %.3f ms of a %.3f ms layer\n", "  ring latency",
                        gr.ms_to_ring / (double) (g.n_layers * positions), decode_ms / (double) g.n_layers);
            // **THE ROUND TRIP, SPLIT AT THE RING.**  `ring latency` is the first half and stops when the ring
            // is seen; this is the second half - the driver calls after it, during which the GPU is IDLE
            // because `post[l]` has not been launched yet.  `--no-pool` is the arm that isolates it: 38.73
            // ms/token against a 26.32 ms pure-GPU floor is 12.4 ms of round trip with no expert work at all.
            //
            // Same denominator as the pool line above (the positions the loop actually ran on), so the two can
            // be added without one of them being inflated by prefill.
            const double perlap = (double) (g.n_layers * positions);
            std::printf("%-24s %.3f ms/token over %.0f positions (%.3f ms/layer, after the ring)\n",
                        "  host after ring", pool_positions > 0.0 ? gr.ms_host / pool_positions : 0.0,
                        pool_positions, gr.ms_host / perlap);
        }
    }

    if (dump != nullptr) std::printf("%-24s %s\n", "logits dumped", o.dump_logits.c_str());

    strata::core::session_graphs_free(gr);
    strata::core::doorbell_free(db);
    cudaFree(d_next);
    cudaFree(d_logits);
    cudaFree(d_emb);
    cudaFree(d_parts);
    cudaFree(sbuf);
    cudaFree(arena);
    return 0;
}
