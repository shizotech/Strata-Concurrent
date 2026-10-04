# Stage 3 — concurrent requests inside one Strata process

**Status: design only. No source file was changed to produce this document.**
Written for the implementation agent (taskboard `S3.1`). Every claim is cited to a file:line that was
read while writing it, and every byte figure was produced by running the engine's own cost functions
(`session_bytes`, `qsa_state_bytes`, `strata-plan`) or read out of `strata-iq3_s.log`. Where a number
could not be obtained without starting the engine it is marked **(OQ)** with the open-question id.

The machine everything is sized against is the owner's box: 3× RTX 3090, `--layer-split auto` (K=16,34),
IQ3_S native pack, `--max-context 524288`, `--kv int8`, `--kv-resident 32768`, yarn 2, `--spec 4`,
`--prefill auto`, `--expert-cache auto`, `--vram-reserve-mib 700` (`strata-iq3_s.json`).

---

## 0. What is true today, in one page

* One process, one `while (next_line(line))` loop: `src/program/generate.cpp:4361`. Requests are
  `GEN <max_new> [k=v …] <ids>` / `GENI <max_new> <embfile> [k=v …] <ids>` (`:4371`), answers are
  `RESUME` (`:4697`), `PP` (`:4743`, `:4016`), `REUSED` (`:4979`), `T <id>` per token (`:5082`),
  `DONE …` (`:5256`), `ERR …` (many), `READY <ctx> stop` (`:4344`), `INFO k=v …` (`:4300`).
  `STOP` is read on the stdin thread and sets one process-wide flag (`:4203`, `:4239`); `QUIT` ends
  the process (`:4362`).
* `serve/server.py` serialises with one lock — `self.fifo = threading.Lock()` (`serve/server.py:624`),
  taken in `Service.run` (`:995`) and `Service.prepare` (`:906`) — and `StrataEngine.generate`
  (`:342`) treats the next `DONE` line as the end of the request. `/slots` returns one hard-coded slot
  (`:1515-1521`).
* One of everything that holds sequence state: `SessionState ss` (`generate.cpp:1861`, allocated at
  `:2238`), one `ExpertCache xcache` (`:2408`), one `Verifier ver` (`:3706`), one
  `strata::prefill::Prefill sp` (`:3543`), one `MtpDrafter mtp` (`:2295`), one `ExpertPool pool`
  (`:2402`), one `SuffixDrafter sfx` (`:4351`), one `ConversationCache conversations` (`:3847`), one
  `live`/`checks` conversation branch (`:3806-3811`).
* The docs already say this is not implemented: `docs/DETAILS.md:378-379` ("Serving two requests at the
  same time from ONE server is a different thing and is not implemented yet") and `docs/DETAILS.md:546`
  ("Current limits (v1): one request at a time").

---

## 1. Goal and non-goals

### Goal

A single `strata --serve` process accepts **N requests in flight at once**, each with its own
conversation state, and makes progress on all of them without any client seeing another client's
tokens. Concretely, "concurrent requests" means:

1. **Request-level concurrency is mandatory.** While request A is decoding, request B may be
   prefilled, decoded, cancelled, parked, or finished. A's `T`/`DONE` and B's `T`/`DONE` are
   distinguishable on the wire (§6).
2. **Step-level interleaving is the mechanism.** The scheduler runs one *step* at a time — a verify
   window, a prefill chunk, a state swap — and chooses which slot's step runs next. Between two steps
   of A the engine may run three steps of B.
3. **Prefill and decode overlap on the GPU.** The batched prompt path is *not* CUDA-graph captured
   (`src/prefill/prefill.cpp` contains no `cudaStreamBeginCapture`; it launches kernels directly), so
   B's prefill can run on its own stream while A's captured window graph is in flight. This is the
   single largest real win available on this box and it is free of the graph problem (§5).
4. **The existing per-request machinery keeps working per slot**: conversation checkpoints, parking,
   the prompt loan, adaptive residency, STOP, the watchdog, `/status`, `/slots`, `/metrics`.

### Non-goals (explicit, so nobody "improves" the design mid-implementation)

* **Not a shared decode batch.** Two slots do **not** share one verify window. `Verifier::run(T, …)`
  puts T tokens at *consecutive* positions `pos0 … pos0+T-1` through all 48 layers
  (`include/strata/core/verify.hpp:3-8`), and the GDN kernels carry one recurrence per row
  (`src/kernels/cuda/verify_kernels.cu:386,403` bound `n_tok` to `kVerifyMaxT = 8`,
  `include/strata/kernels/verify_kernels.hpp:21`). Two conversations at unrelated positions in one
  window would need a per-row GDN state, a per-row KV page table, and a per-row draft chain. That is a
  different engine. **Strata's continuous batching is slot-level, not token-level.**
* **Not a shared KV arena with per-sequence cell lists.** llama.cpp can do this because its KV cache is
  one pool with `cell` slots handed out per sequence and its attention reads a page table. Strata's
  QSA state is 12 separate pools plus an indexer plus a page table, all *baked into captured graphs*
  (§5). Making them one arena is a kernel rewrite, not a scheduler change. It is listed as a
  post-stage-3 direction (§10), not part of this stage.
* **Not multi-process.** Stage 1 (shared arena, core leases) already handles several *processes*. This
  stage is about one process.
* **Not a new sampler, drafter, or expert kernel.** Everything downstream of "which slot runs now"
  stays exactly as it is.
* **Not persistent state across restarts.** Snapshots stay host-RAM only (`docs/DETAILS.md:543-544`).

### What "as good as llama.cpp" means here

llama.cpp's shape, from this repo's copy:

| llama.cpp | file:line | Strata's equivalent in this design |
|---|---|---|
| `struct server_slot` with per-slot `n_ctx`, `state`, `generated_tokens`, `smpl` | `third_party/llama.cpp/tools/server/server-context.cpp:239`, `:270` | `Slot` (§2) |
| `enum slot_state { IDLE, WAIT_OTHER, STARTED, PROCESSING_PROMPT, DONE_PROMPT, GENERATING }` | `server-context.cpp:100-107` | §3.4 state machine (same six shapes, plus parked/cancelled/error) |
| `server_queue` (`q_result`, `q_prompt`, `yield_to_queue`) | `server-queue.h:104` | the scheduler's ready queue; `yield_to_queue` has **no Strata analogue** — see §3.6 |
| one `llama_batch` assembled from many slots, then `llama_decode(ctx_tgt, batch_view)` | `server-context.cpp:3682` | **not followed** — one window per slot (§1 non-goals) |
| `completion_token_output` per slot | `server-task.h:299` | per-slot `T`/`DONE` with a request id (§6) |
| shared KV pool + `server_prompt_cache` (radix) | `server-context.cpp` `prompt_save`/`prompt_cache` | the existing `ConversationCache` + `conversation_snapshot_*` (§2.4), which is Strata's radix equivalent and already exists |

**Where Strata can follow:** the slot object, the state machine, the queue, per-slot sampling and
per-slot token accounting, `/slots`, and the "a slot joins or leaves every step" scheduler discipline.

**Where Strata cannot follow (and must not pretend to):** the shared cell arena, the multi-sequence
`llama_batch`, and `yield_to_queue`. Strata's per-token work is one captured graph per window length
plus a CPU expert pool that is *inside* that graph's timing loop; there is no place to "yield to the
queue" mid-`llama_decode` without breaking the doorbell protocol (§4.3).

---

## 2. The slot model

### 2.1 What is per-slot, what is shared

| state | owner | why |
|---|---|---|
| QSA KV pools + page table + indexer (`idx_tail`/`idx_dead`/`idx_pooled`) | **per-slot** (in the slot's `SessionState`) | sequence state; `include/strata/core/session.hpp:36-39` |
| GDN recurrence + conv history (`ss.gdn_state`, 36 rows) | **per-slot** | `session.hpp:33-34`, `src/core/session.cpp:40-43` |
| PLE history (`ss.ple_hist`, `ss.ple_prev`, `ss.ple_token`) | **per-slot** | `session.hpp:66-87`; `session_zero` resets it because a stale row is a real contribution (`src/core/session.cpp:145-153`) |
| gated-residual stack `R` (`ss.block.R`) | **per-slot** | `session.hpp:46`; it is inside the session arena |
| MoE / block / QSA scratch (`moe`, `block`, `qsa_bufs`) | **per-slot** (cheap: 34 KB + 169 KB + 7.3 MB) | they are inside the same arena and the graphs point at them |
| MTP draft-layer K/V ring (`mtp.st_`) | **per-slot** | `include/strata/core/mtp.hpp:147`; the drafter's attention is dense over *its* cells (`mtp.hpp:16-18`) |
| the conversation branch: `live`, `live_imgs`, `checks`, `cvec_cached` | **per-slot** | `generate.cpp:3805-3810` |
| sampling params, penalty history, seed | **per-slot** | `verify.hpp:73-92` are per-request setters on a shared object — §3.5 |
| suffix drafter history | **per-slot** | `include/strata/spec/suffix_drafter.hpp:47-48` (`hist_` + `table_` are the sequence) |
| `DraftPolicy` acceptance statistics | **shared** | `strata::spec::DraftPolicy policy(S)` (`generate.cpp:4352`) learns process-wide; sharing it is a *quality* difference, §7 |
| **the captured graphs** (`Verifier::exec_[9]`, `MtpDrafter::*_exec_[9]`) | **shared, one set** | §5 |
| **the VRAM expert cache** (`xcache`, `stages[i].cache`) | **shared, one residency table** | §4.1 |
| **the CPU expert pool** (`pool`, 5 workers) | **shared** | §4.3 |
| **the prompt loan** (`pf_parts`, `PfPart::first_now`) | **shared, one loan at a time** | §4.4 |
| weights, `WeightTable`, `NativeHead`, PLE table, expert arena, `d_mrope` | **shared** | read-only |
| `ConversationCache` (the parked images) | **shared** | it is already the swap store (§2.4) |

### 2.2 What one slot costs, measured

`session_bytes()` (`src/core/session.cpp:53-72`) is pure arithmetic and was run against the real
`ModelGeometry` (`include/strata/core/layout.hpp:27-56`) with `k = 10`:

| configuration | one full 48-layer session |
|---|---|
| `--max-context 524288`, `--kv fp16`, `--kv-resident 0` | **13 959 779 840 B = 13.00 GiB** |
| `--max-context 524288`, `--kv int8`, `--kv-resident 0` | 1.753 GiB |
| **`--max-context 524288`, `--kv int8`, `--kv-resident 32768` (this box)** | **1 492 082 688 B = 1.390 GiB** |
| same, layer-split stage `[0,16)` / `[16,34)` / `[34,48)` | 0.554 / 0.560 / 0.547 GiB |
| `--max-context 131072`, int8, resident 32768 | 0.728 GiB |
| `--max-context 65536`, int8, resident 32768 | 0.617 GiB |
| `--max-context 32768`, int8, resident 32768 | 0.560 GiB |
| `--max-context 8192`, int8, resident 32768 | 0.229 GiB |

Breakdown at 524288 / int8 / resident 32768 (`qsa_state_bytes` `src/core/layer.cpp:529-545`,
`kv_pool_bytes` `:512-521`):

* 12 QSA states: 1 with RoPE 269 127 280 B + 11 without 134 909 552 B = **1.74 GiB → 0.25 + 11×0.126**
  (the RoPE table alone is `max_cells × (n_rot/2) × 4 × 2` = 134 217 728 B = 128 MiB, `layer.cpp:541`);
* GDN running state, 36 layers × 3 268 608 B = **117 669 888 B = 112 MiB** (`session.cpp:40-43`);
* `qsa_buffers_bytes` 7 283 280 B, `block_buffers_bytes` 169 024 B, `moe_buffers_bytes` 34 704 B,
  `gdn_buffers_bytes` 3 490 096 B, PLE history a few hundred KB.

The 6.8 GB figure in the task brief is `strata-plan`'s `kv_bytes` at 524288
(`./build/strata-plan --max-context 524288 --vram 17000000000` → `KV + indexer keys 6.845104 GB`),
which is **per QSA layer** — `include/strata/plan/plan.hpp:91-100` multiplies by `n_qsa_layers` only in
the plan's own arithmetic, and 12 × 0.57 GiB ≈ 6.8 GiB is exactly the fully-resident fp16/int8 pool
cost. With `--kv-resident 32768` the VRAM half collapses to 32 768 cells/layer and the authoritative
K/V moves to pinned host RAM: `strata-iq3_s.log:6056`
`KV streaming: 32768 of 524288 cells per QSA layer in VRAM, the K/V in 6.19 GiB of pinned RAM`
(`src/core/layer.cpp:625-640`, `include/strata/kernels/kv_stream.hpp:3-16`).

**Per-slot totals on this box:**

| | VRAM | host RAM |
|---|---|---|
| session arena (`session_bytes`) | 1.390 GiB | — |
| KV-streaming authoritative host copy | — | 6.19 GiB |
| MTP draft ring state (`qsa_state_bytes(g, 524288, false, 32768+…)`) | 35.1 MB | (shares the host-copy budget) |
| suffix drafter (`SuffixDrafter(3, 64, max_context+4096)`, `generate.cpp:4351`; `Slot` = 32 B, table ≥ 2× capacity) | — | ≈ 66 MiB |
| **one extra slot, all-in** | **≈ 1.43 GiB** | **≈ 6.3 GiB** |

### 2.3 How many slots fit — and the answer is "not by allocating them up front"

VRAM headroom on this box with everything loaded is **457 MiB** (`strata-iq3_s.log:6079`, and the
engine's own warning threshold is 256 MiB, `generate.cpp:4265-4277`). 457 MiB is **346 expert-cache
slots** (blob 1 382 400 B) — it is not even one session arena, let alone two.

Host RAM is worse: `MemAvailable` is ~2 GiB and the parking budget has already collapsed to
**0.0 GiB** on this box (`strata-iq3_s.log:6078`: `parked-prefix budget 0.0 GiB, not 8.0: only 0.0 GiB
of RAM is free above the 2.5 GiB parking floor and 4.0 GiB of request headroom`).

Therefore:

> **Slots are not N pre-allocated sessions. There is exactly ONE live session (the one the captured
> graphs point at), plus a bounded set of parked slot images in host RAM, plus a queue.**

This is the single most important decision in the document, and it is forced by arithmetic, not taste.

The design therefore has **one VRAM session per layer-split stage** (unchanged from today:
`generate.cpp:2238` and `:2273`) and treats "how many conversations" as a **host-RAM** question, which
is the question `ConversationCache` already answers (`include/strata/core/conversation_cache.hpp:139`,
`default_slots = 8`, budget-capped, LRU-pruned).

**Sizing policy, in priority order:**

1. **Per-slot context cap.** A slot's *logical* context may be up to `--max-context`, but its
   *resident* KV window is `--kv-resident` cells (32 768 today). That is already how KV streaming
   works: the VRAM pool is a clock-swept cache over the authoritative host copy
   (`kv_stream.hpp:5-14`). Nothing new is needed — a slot's cost scales with its **token count**, not
   with `--max-context`, for the host copy (`conversation_kv_save` copies `[0, upto)`,
   `src/core/conversation_snapshot.cpp:139-171`).
2. **Admission, not eviction.** A new request is admitted only if its snapshot fits the parking
   budget (`ConversationCache::can_fit` / `make_room`, `conversation_cache.hpp:169-227`) *and* the
   physical-RAM gate (`conversation_memory_admit`, used at `generate.cpp:3920`). If it does not fit,
   the engine says so on the wire (`ERR …`, §6) rather than thrashing. This is exactly the stage-2
   lesson from `.megamind/src/core/conversation-cache-notes.md:37-48`.
3. **`--slots N`** caps the number of *simultaneously active* slots (default 2, max 4 on this class of
   box) independently of the parked count. Active is expensive (it costs a host image plus queue
   residency); parked is what the budget already governs.
4. **Swap, never duplicate.** Mounting a slot means `conversation_snapshot_restore` into the one live
   session; unmounting means `conversation_snapshot_save` out of it. Both already exist and are
   validated (§2.4).

### 2.4 The swap machinery already exists — use it, do not write a new one

`include/strata/core/conversation_snapshot.hpp`:

* `conversation_snapshot_save(SavedConversation&, view, session, g, draft, …)` — `:66`
* `conversation_snapshot_validate(...)` — `:70` (all layers and checkpoints validated **before** any
  CUDA call or write)
* `conversation_snapshot_restore(...) -> {restored, invalid, transfer_failed}` — `:75`
* `conversation_kv_save/restore` — `:17`, `:25`; for a streamed layer the source is the **host** pool
  (`src/core/conversation_snapshot.cpp:70-77`), and restore ends with `kv_stream_reset(st.map)` so the
  VRAM slots are refilled lazily by `kv_stream_resolve` inside the next window
  (`src/core/conversation_snapshot.cpp:207`, `kv_stream.hpp:8-12`).

That last point is why swapping is cheap on this box: **a slot switch is a host→host memcpy plus a
page-table reset, not a VRAM copy.** The VRAM KV slots stay where the graphs expect them.

Measured snapshot sizes on this box, from the live log (`grep -o 'snapshot_bytes=[0-9]*'
strata-iq3_s.log`, 186 parks): **237 MB (smallest) … 2.25 GB (largest)**. A 36 k-token conversation
parks at roughly 1.0-1.9 GB.

**Two gaps the implementation must close, both already known:**

* ~~`SavedConversation::layer_lo/layer_hi` records the carve the image came from and restore requires the
  same one (`conversation_cache.hpp:96-97`); `ConversationCheckpoint::stage_parts` exists but
  "whole-session parking is currently single-GPU and rejects these parts"
  (`conversation_cache.hpp:58-60`), and `docs/DETAILS.md:509-512` says a layer split disables parking.
  **On this box (3-way split) parking is off, so the swap path does not exist yet.** Closing it is
  stage-3 prerequisite S3.1a, not an optional follow-up.~~
  **CLOSED by S3.1a.** A parked conversation now carries one `ConversationStageSnapshot` per later
  stage plus the MTP draft state on the drafter's own device, and the split's parking guard is gone.
  The scheduler calls the stage-set overloads:
  `conversation_snapshot_{bytes,capture_bytes,save,validate,restore}(…, const SessionState&,
  const ConversationStageSet&, const ModelGeometry&, const QsaState& draft, …)`; build the set once
  from the engine's `stages[]` (`{&st->ss, st->dev}`), with `main_device = 0` and
  `draft_device = mtp.device()`. See `include/strata/core/conversation_snapshot.hpp`.
* `conversation_snapshot_save` returning false makes `park_current` return false and the serve loop
  does `return 1` — it **kills the process** (`generate.cpp:4587-4589`). Under concurrency that must
  become a per-slot error, not a process exit (§3.5).

---

## 3. The scheduler

### 3.1 Shape

One **engine thread** owns everything that touches the GPU. It is today's `while (next_line(line))`
body, refactored from "run one request to completion" into "run one step of one slot". A **reader
thread** (today's stdin thread, `generate.cpp:4208-4246`) parses lines into a request queue. A
**writer** is not needed: the engine thread writes stdout, so `T`/`DONE` ordering is total and no
locking of `stdout` is required.

```
                     ┌──────────────────────────────────────────┐
  stdin thread ───►  │ request queue (parsed, id-tagged)        │
  (parse GEN/GENI/   └──────────────────────────────────────────┘
   STOP/QUIT)                     │ admit / refuse
                                  ▼
  ┌────────────────────────────────────────────────────────────────────┐
  │ ENGINE THREAD  — the only thread that touches the GPU              │
  │                                                                    │
  │  loop:                                                             │
  │    1. drain the request queue; admit/refuse                        │
  │    2. pick a step (fairness, §3.3)                                 │
  │    3. if the step's slot != active slot → swap (§3.5)              │
  │    4. run the step: window / prefill chunk / checkpoint / DONE     │
  │    5. emit protocol lines tagged with the request id               │
  └────────────────────────────────────────────────────────────────────┘
                                  │
                     (optional, phase 4) prefill worker thread
                     running Prefill::run on its own stream
```

**Why one GPU thread.** The doorbell protocol between the CPU expert pool and the captured window graph
is a spin handshake on fixed pinned flags (`Verifier::h_flag_`/`h_seq_`, `verify.hpp:190-193`;
`Doorbell`, `include/strata/core/layer.hpp:396-413`). Two threads issuing windows would race on those
flags exactly like issue #29 raced on the pool's `head` word (`include/strata/kernels/cpu/pool.hpp:23-27`).
One thread, many slots, is the safe shape.

### 3.2 The step

A **step** is the unit the scheduler picks. Its size is what bounds a slot's latency:

| step | cost on this box | interruptible? |
|---|---|---|
| one verify window (`Verifier::run` + `commit`) | ~24 ms predicted per decode window (`strata-iq3_s.log:27`); 1 window = 1-6 tokens | **yes** — the loop already checks `stop_req` between windows (`generate.cpp:5116`) |
| one prompt chunk (`Prefill::run`, ≤ 8192 tokens) | 8 192 tokens ≈ 8-15 s at the measured 500-1 100 tok/s (`prefill-levers.md`) | **yes** — `Prefill::should_stop` is checked before every chunk (`include/strata/prefill/prefill.hpp:92-93`, wired at `generate.cpp:4256`) |
| one slot swap (save + restore) | 237 MB-2.25 GB of host memcpy; the log's own park/restore lines put a restore at tens of ms per few hundred MB (`generate.cpp:4623-4626` prints the ms) | no — but it is bounded and it is the price of fairness |
| one checkpoint save (`checkpoint_at`) | ~118 MB + a `cudaDeviceSynchronize()` (`prefill-levers.md:90`) | no |

**The scheduler must not pick a whole request as a step.** Today's loop is exactly that, and it is why
one 36 k-token read blocks everything (`strata-iq3_s.log:6080`: a 35 957-token prompt in 33 080 ms).
Chunk boundaries and window boundaries are the preemption points, and both already exist.

### 3.3 The pick rule

Deterministic, and simple enough to unit-test on the CPU:

```
pick():
  if a slot is CANCELLING:            finish its unwind first   (must not leak the loan)
  if the active slot is DECODING and
     no other slot has been waiting > T_starve:
        → the active slot's next window          # zero swaps: today's behaviour
  if some slot is QUEUED and its image is parked:
        → admit it (restore)                     # only if active_slots < --slots
  if some slot is PREFILLING:
        → its next chunk                         # prefill before decode: it is the long pole
  if some slot is DECODING and != active:
        → swap to it, then its next window
  else: idle (wait for a line)
```

`T_starve` (default 250 ms, `--starve-ms`) is the fairness bound: a decode slot that has not run for
that long forces a swap. Without it, a long prompt's chunk stream starves decodes; with it set too low
the engine spends its time swapping. **The tuning is a measurement, not a guess (OQ4).**

The rule deliberately prefers **prefill over decode**. Prefill is the long pole (54.6 % of prompt wall
time is in reads of ≤ 4 k fresh tokens with a ~9.5 s fixed floor — `.megamind/prefill-levers.md:55-61`),
and a request that cannot prefill cannot decode at all.

### 3.4 Slot state machine

```
                       admit (fits budget + slots)
  QUEUED ───────────────────────────────────────────────► PREFILLING
    │  refuse (budget, slots, context)                        │ chunk loop
    │        │                                                │ (short parts go
    │        ▼                                                │  through windows,
    ERR <id> …  → IDLE                                        │  --short-read)
    │                                                         ▼
    │ (error)                                            DECODING ◄────┐
    ▼                                                      │  │  │     │ swap in
  IDLE ◄──── DONE ──── PARKED ◄──── swap out ──────────────┘  │  │     │
    ▲            │                                            │  │     │
    │            └────────────────────────────────────────────┘  │     │
    │                        (EOS / length / max_new)            │     │
    │                                                            ▼     │
    └──────── ERR <id> …  ◄──── ERROR ◄──── (any step failure) ────────┘
                                  ▲
                        CANCELLING ◄──── STOP <id> (from DECODING or PREFILLING)
                                  │
                                  └──► DONE <id> … cancel → IDLE / PARKED
```

| state | meaning | invariants |
|---|---|---|
| `IDLE` | no task | not in the active set |
| `QUEUED` | parsed, not admitted | holds only ids/max_new/sampling; **no state allocated** |
| `PREFILLING` | reading its prompt | holds the prompt loan (§4.4) while a chunk is running; `PP` lines per chunk |
| `DECODING` | windows | is the active slot, or parked in host RAM |
| `PARKED` | a finished or paused conversation whose image is in `ConversationCache` | **this is exactly stage 2's parked conversation** — no new store |
| `CANCELLING` | `STOP <id>` seen; running until the current step's unwind finishes | must refill the loan and free its active slot before `DONE` |
| `ERROR` | a step failed | `ERR <id> …`, then the slot is destroyed; **the process stays up** |

`PARKED` is not a new store: it *is* `ConversationCache`'s entry. A slot in `PARKED` costs host RAM and
nothing else, and `conversations.best(ids, …)` (`generate.cpp:4570`) is how a new request finds it.

### 3.5 What has to move from "process" to "slot"

| today | file:line | becomes |
|---|---|---|
| `std::atomic<bool> stop_req` | `generate.cpp:4203` | `std::atomic<bool> stop_req[MAX_SLOTS]`; `STOP` → `STOP <id>` (§6); a bare `STOP` means "the newest uncancelled slot" for compatibility |
| `sp.should_stop = [&]{ return stop_req.load(); }` | `:4256` | set per prefill step to the running slot's flag |
| `ver.set_sampling / set_history / set_head_sampling` | `verify.hpp:73,84,92` | applied at **dispatch**, immediately before `ver.run`, never at parse time |
| `live`, `live_imgs`, `checks`, `cvec_cached` | `generate.cpp:3806-3811` | per-slot; the active slot's are the ones the session currently reflects |
| `park_current` → `return 1` on failure | `:4587-4589` | per-slot `ERROR`; the process survives |
| `progress()` — one global `busy`/`where`/`beats` | `include/strata/core/progress.hpp:17-24,36` | per-slot heartbeat; the watchdog checks **each active slot** and only aborts the one that stalled (§9 R3) |
| `sfx` (one `SuffixDrafter`) | `:4351` | per-slot (its `hist_`/`table_` are the sequence) |
| `policy` (`DraftPolicy`) | `:4352` | shared, documented as a quality difference (§7) |
| `drive.d.usage` (the residency usage vector) | `:4064` | shared, combined (§4.2) |
| `mrope_host` / `d_mrope` upload | `:1830-1843`, `:4436-4513` | per-prefill-step, guarded by a "whose positions are in the table" tag (§9 R7) |
| `pp_total/pp_from/pp_next_check/part_at/part_next` | `:3951`, `:4691-4695` | per-slot |

### 3.6 What llama.cpp does that we deliberately do not copy

`server_context_impl::decode()` wraps `llama_decode` in `queue_tasks.yield_to_queue(...)`
(`server-context.cpp:3682-3687`, `server-queue.h:104`) so the HTTP thread can take work while the GPU
is busy. Strata cannot: the window graph's host-side partner (the pool drain and the doorbell flags)
lives on the engine thread and is single-threaded by construction (§3.1). The equivalent capability is
provided at a coarser grain — the stdin thread already parses while the engine runs
(`generate.cpp:4208-4246`) — and admission happens between steps.

---

## 4. The expert cache and the CPU pool under two conversations

### 4.1 Residency: one table, one owner, one view per slot

Today: `host_res` is a `n_layers × n_expert` int32 table (24 576 entries, 96 KiB) built once from the
profile (`generate.cpp:3348-3362`), uploaded to `d_res` on every device (`:3363-3380`), and re-uploaded
by `res_upload()` (`:4105-4119`) whenever `ResidencyUpload::due()` says the host table differs from
what the devices hold (`include/strata/program/prefill_loan.hpp`, `ResidencyUpload`).

The invariant that must survive stage 3, stated in the code's own words: *"`host_res` is one table whose
slot values are indices into whichever cache owns the layer"* (`generate.cpp:3554-3556`).

**Decision: the physical residency stays exactly one table, process-wide.** Reasons:

* The captured window graph reads `d_res` at a baked-in address (`VerifyHits::d_res`,
  `verify.hpp:46`; `TokenHits::d_res`, `session.hpp:411`). A per-slot table means a per-slot graph set
  (§5 refuses that).
* The VRAM slots are one arena (`ExpertCache::base_`, `include/strata/core/expert_cache.hpp:149`).
  Two residency tables over one arena is two lies about the same bytes.
* The cache **never evicts** (`expert_cache.hpp:88-91`) — eviction policy is `adapt()`'s job
  (`generate.cpp:4146-4199`).

So: **`host_res` is the single source of truth; a slot has no residency of its own.** What a slot has is
a *usage vector contribution*.

### 4.2 How the usage vectors combine

`adapt()` reads one `drive.d.usage` array (`:4154`), swaps the hottest missing pairs in and the
coldest resident ones out, decays everything by 0.7 (`:4198`), and runs every `--adapt-every` rounds
(default 4, `:366`).

Under concurrency:

* `usage` stays **one shared array**. Every window, for any slot, adds to it — the pool adapter already
  increments it per layer from inside `expert_pool_dispatch`, so this needs no new plumbing.
* `adapt()` runs **between steps, on the engine thread**, and only when *no* slot is mid-window (it
  already runs beside `commit` on a separate thread today, `:5066-5069` — that thread must not race a
  swap).
* The decay is per **adapt round**, not per slot, so a busy slot dominates. That is a quality change
  (§7) and it is the honest one: the alternative (per-slot usage + a merged table) means uploading a
  different `d_res` per slot, which is a 96 KiB blocking copy to every device per switch
  (`:4111-4115`) — 3 copies on this box, on links measured at 2.9 / 1.3 / 24.5 GB/s
  (`generate.cpp:4078-4081`). **Do not do it in stage 3.** Re-evaluate as OQ5.
* `pending` swaps in flight (`:4071`) must be drained (`apply_pending(true)`) **before** a slot swap,
  because a swap resets the KV page tables and a half-applied residency table is exactly the "silent
  plausible tokens" failure `expert_cache.hpp:135-138` warns about.

### 4.3 The CPU pool: one pipeline, one slot at a time

`ExpertPool` is a single-batch barrier machine: `run()` publishes N jobs and waits for `done == n`
**and** every worker parked, with the epoch-tagged claim word fixing issue #29
(`pool.hpp:19-27`, `:301-304`). There is no queue depth to share — "the host must SUM all ten outputs
before the next layer starts, so a layer is a barrier by construction" (`pool.hpp:8-12`).

**Decision: the pool is not shared concurrently. It is owned by whichever window is running.** That is
automatic if the engine thread is the only one that calls `Verifier::run` (§3.1). The rules:

1. Only the engine thread may call `ver.run` / `ver.commit` / `mtp.draft`.
2. A layer-split stage's verifier is chained (`set_next`, `:3769`) and shares one `SplitDrive`
   (`:701-722`) — that chain is per-window, not per-slot, and stays as it is.
3. **The pool's cores are already a machine-wide resource** (`claim_cores`, `pool.hpp:126`). Two slots
   in one process do not change that; they share the same 5 workers. Do not spawn per-slot pools —
   that is the stage-1 collapse all over again (`.megamind/src/kernels/cpu/pool-notes.md:11-17`).
4. If a second *prefill* thread is added (phase 4), it must not use the pool: `Prefill` streams experts
   through its own `Stager` threads (`src/prefill/prefill.cpp:137-200`), which have the same
   generation-counter discipline for the same reason (`:149-151`). Two pool clients would need a real
   queue; two `Stager` clients would need two `Stager`s. Neither is required for phases 1-3.

### 4.4 The prompt loan is one loan

`pf_parts` (`generate.cpp:3557-3575`) lends the tail slots of each stage's cache to that stage's
`Prefill` (`:3598-3600`), sized by `Prefill::bytes_needed` (`include/strata/prefill/prefill.hpp:73`),
and `refill_one` streams the evicted experts back (`:4764`). On this box the loan is **2 625 CUDA0
slots = 4.95 GiB** (`strata-iq3_s.log:6075`).

Two concurrent prefills cannot both hold it: the loan is the tail of one cache. Options:

* **(chosen) One loan, held across requests.** `lend()` already skips a re-lend when
  `want <= p.lent_chunk` (`:4821`, and the sticky-loan note at `:4756-4762`). Make the loan a
  process-level resource with an owner and a wait queue: a second prefill either waits for it or runs
  through the verify windows (`--short-read`, `:413`) if its part is small. This also finally removes
  the 4.95 GiB/stage end-of-request refill that S0.3 explicitly left as the floor
  (`.megamind/taskboard.md:52-53`).
* Two loans (splitting the tail) is possible but halves the chunk size for both, and the log shows the
  cost is a floor plus a rate, not proportional to the loan
  (`.megamind/prefill-levers.md:125-141`) — two half-loans cost more than one full loan. Not chosen.

---

## 5. The graph problem

### 5.1 The inventory of baked-in pointers

| object | what it bakes | file:line |
|---|---|---|
| `Verifier::exec_[9]` (one per T ≤ `kVerifyMaxT`) + `commit_exec_` | `ss_` (the one `SessionState`) and its own `arena_` | `verify.hpp:168,177-178`; `verify.cpp:821-881`, `arena_` at `verify.cpp:265` |
| `MtpDrafter::prefill_exec_[9]`, `prefill_dev_exec_[9]`, `round_exec_[9]`, `step_exec_[9]`, `round_exec_c_[9]`, `step_exec_c_[9]` | `ss_`, `st_` (its K/V), `window_R_` (the verifier's `final_R_all()`) | `mtp.hpp:109,113,114,136,137,140`; bound at `generate.cpp:3777` |
| `SessionGraphs::execs/posts/preA/preB/preP[5]`, `parts_dev` | `ss`, `parts` | `session.hpp:133-196`, `:229-231`; captured at `generate.cpp:2916`, guarded by `!native_pack` |
| `TokenGraph` (`exec`, `y_src`) | `ss`, `parts_dev`, the pinned `y_miss` | `session.hpp:395-405`; captured at `generate.cpp:3395`, guarded by `!multi_gpu` |
| `KvStreamMap` | fixed device addresses, read by `kv_stream_resolve` **inside** the graphs | `kv_stream.hpp:49-61` |
| `Prefill` | **nothing** — no capture anywhere in `src/prefill/prefill.cpp` | grep: only `mtp.idle()` at `prefill.cpp:750` |

On this box (native pack + 3-way layer split) the *live* graph sets are the per-stage `Verifier`s
(`generate.cpp:3706-3769`) and the `MtpDrafter`'s. `SessionGraphs` and `TokenGraph` are not captured at
all. That matters: the thing to duplicate is ~6 arrays of 9 `cudaGraphExec_t`, not everything.

### 5.2 The three options, priced

**(A) Re-capture per slot.**
Cost: a second `Verifier` costs its own arena — **76.9 MiB** on this box
(`strata-iq3_s.log:6076`: `strata verify: window up to 6 tokens, 76.9 MiB of device buffers`) — plus a
second `MtpDrafter` state and graphs (the drafter itself is **836 MiB**, `:6058`, of which its K/V ring
is 35 MB and its experts 675 MB; the experts could be shared, the graphs could not). Plus a capture
pause per new slot: the log shows each window capture printing a line
(`verify.cpp:878-880`), and `cudaGraphInstantiate` + `cudaGraphUpload` are synchronous
(`verify.cpp:871-878`). **Against 457 MiB of free VRAM, a second verifier arena plus a second MTP
buffer set does not fit.** Verdict: **refused on this box.** (It might fit on a 24 GB card with a short
context; that is a config decision, not a design one.)

**(B) Move the state behind indirection** (device pointer tables the kernels dereference).
Cost: every QSA/GDN/MoE kernel that takes a `SessionState`-derived pointer would change signature, and
the *page tables* would have to be per-slot too. It is a kernel rewrite across
`src/kernels/cuda/{qsa,gdn,gr,elementwise,verify_kernels}.cu` plus `src/core/layer.cpp`, and it breaks
the bit-exactness bar in §7 unless every change is provably arithmetic-preserving. Verdict: **refused
as a mechanism.** It is the right long-term shape for a real shared-KV engine (§10), not for stage 3.

**(C) One graph set, one active slot, swap the state through the snapshot machinery; overlap prefill
and decode.**
Cost per switch: `conversation_snapshot_save` + `conversation_snapshot_restore` over the slot's token
count. On this box, with `--kv-resident`, the K/V half is **host→host** (`conversation_snapshot.cpp:70-77`)
and the VRAM slots are simply reset (`:207`). The running-state half is 112 MiB of GDN + a few MB of
indexer/PLE. Measured park/restore times exist in the log's own lines
(`generate.cpp:3940-3943`, `:4623-4626`) — order tens of ms for a few hundred MB, **(OQ3)** for the
exact figure because parking is currently disabled by the layer split.
Verdict: **chosen.** It reuses code that is already validated end-to-end
(`conversation_snapshot.cpp:173-194` validates before touching anything; `:196-232` restores and
resets residency), it needs no new VRAM, and it changes no kernel.

### 5.3 The chosen design, stated as a rule

> **There is exactly one "active slot" per layer-split stage. The captured graphs always point at it.
> Changing which conversation is active is a save/restore, never a re-capture.**

Consequences the implementation must honour:

* A step for slot B while A is active is illegal. The scheduler must swap first (§3.3).
* The swap must be **atomic with respect to the loan and the residency table**: `apply_pending(true)`
  (`generate.cpp:4130`), `refill()` if a loan is held (`:4791`), then save, then restore, then
  `res_upload()` if the table changed.
* A swap is not free, so the scheduler's default is to **stay on the active slot** until fairness or
  work forces a change (§3.3).
* **A slot may only be swapped out if its state can be saved, and it may only step while the session
  describes it.** (S3.6, §8) Both halves are now predicates, not prose:
  `serve_swap::save_is_mandatory` / `park_fits` decide the first *before* the unmount hook runs, and
  `serve_driver::step_gate` decides the second before every step. The failure mode this closes is
  silent: a slot whose conversation was destroyed keeps decoding against another conversation's K/V
  and positions, samples garbage, and usually hits EOS at once.
* Prefill does not need the active slot to be the prefilling slot **only if** prefill runs on its own
  state. It does not: `Prefill::init` takes `SessionState& ss` (`include/strata/prefill/prefill.hpp:58`)
  and writes the same KV pools. So in phases 1-3 **prefill also requires the slot to be active**. The
  phase-4 overlap (§8.5) is what lifts that, and it requires a second session arena — which is the
  1.39 GiB VRAM problem again. **(OQ6): is a second session arena affordable at any useful
  `--max-context` on this box?** At `--max-context 32768` a session is 0.560 GiB, so two extra stages
  would fit only if the expert caches shrink by ~1.2 GiB (≈ 900 slots each). That is a real trade the
  owner must approve, not something the implementation agent should decide alone.

---

## 6. The protocol change

### 6.1 The rule: every line that belongs to a request carries its id

Today's protocol has no request identity at all: `T <id>` and `DONE …` mean "the one request running".
With N slots that is unparseable. The change is additive and versioned.

**Handshake.** `READY` gains a feature token, exactly as it already gained `stop`
(`generate.cpp:4344`, consumed at `serve/server.py:195-200`):

```
READY <max_context> stop slots=<N>
```

`slots=<N>` is present **only** in a concurrency-capable engine. `serve/server.py` already ignores
unknown `READY` fields (`self.can_stop = "stop" in f[2:]`, `:200`), so an old server reads a new engine
unchanged. A new server enables multi-slot **only** when it sees `slots=`.

**Requests (engine ← server).**

```
GEN  <req_id> <max_new> [k=v ...] <id,id,...>
GENI <req_id> <max_new> <embfile> [k=v ...] <id,id,...>
STOP <req_id>
QUIT
```

Backward compatibility: the id is **optional in the parser**. A token immediately after `GEN`/`GENI`
that is followed by `=` or by a comma-list is `max_new`, not an id — the existing parse
(`strtoll(line.c_str() + (geni ? 5 : 4), &endp, 10)`, `generate.cpp:4377`) already stops at the first non-digit, so
the rule is: *if the token after `GEN` is a bare integer and the token after that also is, the first is
the id*. A bare `STOP` (no id) cancels the most recently started uncancelled slot, which is what today's
`STOP` means (`:4239`).

**Responses (engine → server).** Every per-request line is suffixed with `#<req_id>`:

```
RESUME <n> #7
PP <pos> <prompt_tokens> <ms> <tok/s> #7
REUSED <n> #7
T <token_id> #7
DONE <generated> <prompt> <prompt_ms> <decode_ms> <finish> <drafts_acc> <drafts_off> <reused> <hits> <lookups> #7
ERR <message> #7
```

The `DONE` field list is **unchanged** (`:5256`, parsed by `_parse_done`, `serve/server.py:279-287`,
which indexes `f[1..10]`); the tag is appended after it, so `_parse_done` keeps working if it ignores
trailing tokens — it does (`line.split()` then positional indexing).

`INFO k=v …` (`:4300`) stays untagged and process-wide, plus new keys:
`slots=<N> slots_active=<n> concurrency=1`.

Untagged lines (`INFO`, and the `strata …` stderr lines) stay process-wide. **Anything a client must
route to a request is tagged; anything describing the process is not.**

### 6.2 New engine-side lines (optional, for `/slots`)

```
SLOT <req_id> <state> <ctx_used> <ctx_cap> <prompt_tokens> <generated> <parked_bytes>
```

emitted on every state transition (§3.4), so `serve/server.py` can maintain `/slots` without polling
the engine. `state` ∈ `queued prefilling decoding parked cancelling error idle`.

### 6.3 `serve/server.py` — the exact changes

| what | where | change |
|---|---|---|
| the serialising lock | `self.fifo = threading.Lock()` `:624`; taken at `:708`, `:906`, `:995` | becomes a **slot semaphore**: `self.slots = threading.Semaphore(n)` where `n = engine.slots` (1 when the engine reports no `slots=`, which reproduces today exactly). `ensure_loaded` (`:667`) and `load`/`unload` (`:702`, `:711`) keep a **separate exclusive lock**, because loading/unloading the process is not a slot operation. |
| one pipe, one line queue | `StrataEngine.lines: queue.Queue` `:208`, `_pump` `:211` | unchanged (one pipe is still correct — the engine multiplexes). Add `self.pending: dict[int, queue.Queue]`: `_pump` reads the `#<id>` tag and routes the line, or puts it on a `self.control` queue when untagged. |
| "the next DONE is mine" | `generate()` `:342-403`, `finally` drain `:389-403`; `Service.run`'s identity-token trick `:991`, `:1005`, `:1040-1043`, `:1052-1053` | replaced by the id: `generate(req_id, …)` waits only on lines tagged `#req_id`. The `engine_last0` identity dance (`:991`, `:1052-1053`) becomes unnecessary and should be deleted, not kept alongside. |
| `self.last` (one dict) | `:182`, written by `_parse_done` `:279` | `self.last: dict[int, dict]`, keyed by request id. Every reader (`Service.run` `:995`, `:1052`; `request_timings` `:1097`; `metrics` `:818`) takes the id. |
| `self.status` (one request) | `:627`, updated at `:1001-1003`, read by `/status` `:1487-1505`, `_tok_s` `:790`, `_note` `:942`, `_progress` `:960` | becomes `self.status: dict[int, dict]` plus a process-level summary. `/status` returns `{"busy": <n_active>, "requests": [per-slot…]}`; keep the old flat keys for one release, filled from the *oldest* active slot, so the existing Monitor tab does not break. |
| `/slots` | `:1515-1521` — one hard-coded `{"id": 0, "n_ctx": …, "is_processing": busy}` | returns the real per-slot list from the `SLOT` lines, in llama.cpp's shape (`id`, `n_ctx`, `n_prompt_tokens`, `n_generated_tokens`, `state`, `is_processing`). |
| `/metrics` | `:818-852`, handler `:1462` | add `slots_active`, `swaps`, `swap_ms`, `parked_bytes`, and per-slot rows. `self.totals` (`:631`) stays process-wide. |
| `STOP` on cancel | `:392` (`self.proc.stdin.write("STOP\n")`) | `STOP <req_id>\n`. |
| `Service.run`'s `with self.fifo:` | `:995` | `with self.slot_lock_for(req_id)` — acquire a semaphore permit, choose/restore the slot, run, release. |
| `self.embeddings = threading.local()` | `:625` | **must be reviewed**: the GENI embedding path (`:988`, `:1048`) is per-request state on a thread-local; with several HTTP threads feeding one engine it is still per-request, so it survives, but the *engine-side* embeddings file (`emb_path`, `generate.cpp:4415-4422`) is read at parse time and must be read before the request is queued, not at dispatch. |
| the module docstring | `:7` ("One sequence at a time behind a FIFO") | update. |

### 6.4 Compatibility matrix

| server | engine | behaviour |
|---|---|---|
| old | old | unchanged |
| old | new | the new engine must accept **id-less** `GEN`/`STOP` and emit **untagged** lines. Achieved by a `--serve-slots` default of **0** (concurrency off) and a `tag_lines` flag that is off until a `slots=` handshake. **An engine started with `--serve-slots 0` is byte-identical on the wire to 0.1.30.** |
| new | old | the server sees no `slots=` → semaphore of 1, id-less requests, untagged parse. Exactly today. |
| new | new | full concurrency. |

That is the whole compatibility story: **the wire protocol is versioned by a `READY` token, and the
default is off.**

---

## 7. Bit-exactness / quality policy

This is the acceptance bar. The taskboard states it: *"Stage 3 must not change the tokens a single
request produces (bit-exactness against today's serial path is the acceptance bar)"*
(`.megamind/taskboard.md:86-87`).

### 7.1 Must be identical to today

For a **single request in flight**, with the same engine binary, the same flags and the same seed, the
token sequence must be byte-identical to 0.1.30's. That includes:

* the prompt path's chunking and the `--short-read` window path (`:413`, `windows_ok` at `:4708`);
* the checkpoint chain and its LRU (`include/strata/program/conv_cache.hpp:56-59`);
* the prompt loan's size and layout (`part_slots` `:3575`, `lend` `:4801`) — the loan must be the same
  slots for the same chunk, because a lent expert is computed on the CPU and "rounds differently"
  (`:3554`, `expert_cache.hpp` notes);
* the residency table at every window (`host_res`/`d_res`) — the same experts resident at the same
  round means the same GPU-vs-CPU split, which is what the existing parity note is about
  (`strata-iq3_s.log:36-40`: "the GPU computes the experts in the cache; it rounds differently from the
  CPU");
* the sampling call order and the penalty rows (`verify.hpp:78-88`, `penalty_rows` at `:5050`);
* the MTP drafts, the suffix drafter's history, and `DraftPolicy`'s state.

**The single-request path must therefore be reachable with `--serve-slots 0`, and the implementation's
first test is that this path is unchanged.**

### 7.2 Where two concurrent requests may legitimately differ

These are consequences of the design, not bugs, and each must be documented in `docs/DETAILS.md`:

| difference | why it is legitimate | how to suppress it |
|---|---|---|
| **expert residency** | `adapt()` merges usage across conversations (§4.2), so a conversation that runs alongside another sees a different resident set than one that runs alone. The engine already warns that residency changes the rounding (`strata-iq3_s.log:36-40`). | `--adapt-every 100000` (static residency) — already the documented answer for reproducibility (`docs/DETAILS.md:549-551`). |
| **draft acceptance / `DraftPolicy`** | the policy learns process-wide (`:4352`); another conversation's acceptance statistics change which drafter is picked. Only draft *quality* changes: the verify window decides every emitted token (`verify.hpp:3-8`), so a different draft never changes an accepted token, only how many windows it takes. | none needed; it is not a correctness difference. |
| **suffix drafter** | per-slot (§3.5), so it is *more* correct than sharing it. | n/a |
| **sampling** | per-request already (`:4378-4410`); unchanged. | n/a |
| **wall-clock timings** (`prompt_ms`, `decode_ms`) | a slot that waited for the GPU or for a swap reports larger numbers. | report them as-is; add `wait_ms` per slot. |
| **`--kv-resident` cache warmth** | a swap calls `kv_stream_reset` (`conversation_snapshot.cpp:207`), so the first windows after a switch miss more VRAM pages. The engine already reports this (`strata-iq3_s.log:6041`: "KV streaming: 99.84 % of 80 679 216 block reads hit VRAM"). | raise `--kv-resident`, or reduce swapping (`--starve-ms`). |

### 7.3 The verification the implementation must run (no engine start)

* **CPU-only determinism tests** for the pure decisions: the scheduler's `pick()`, the admission test,
  the slot state machine, the protocol parser (id present/absent, `STOP` with/without id, tag
  round-trip). Follow the existing pattern: `include/strata/program/prefill_loan.hpp` +
  `src/program/prefill_loan_test.cpp` (54 CPU checks, `.megamind/taskboard.md:43-51`).
* **A replay test against the log**: `bench/prefill/analyze.py` already parses `strata-iq3_s.log`;
  add a mode that replays the request sequence through the new scheduler and asserts the *admission and
  swap decisions* are what the design says, without a GPU.
* **The existing parity harnesses** (`bench/results/2026-09-27-cache-parity`, referenced at
  `strata-iq3_s.log:38`) are the owner-run check that tokens did not change; they need a model, so they
  are an owner action at a restart, not an agent action.

---

## 8. Staged implementation plan

Each phase is one agent run, has a contract, and is verifiable **without starting the engine** unless
stated. Phases 1-3 deliver request-level concurrency with the GPU serialised; phase 4 is where the
real overlap begins.

### S3.1a — parking across a layer split (prerequisite) — **LANDED**

**Build.** Make `conversation_snapshot_save/restore` cover a split: `SavedConversation` already carries
`layer_lo/layer_hi` (`conversation_cache.hpp:96-97`) and `ConversationCheckpoint::stage_parts` already
exists (`:60`). Extend `conversation_snapshot_*` to walk every stage's `SessionState` and the MTP
drafter on the last stage, and drop the "a split disables parking" branch
(`docs/DETAILS.md:509-512`, `generate.cpp`'s parking/split guard).
**Test.** `conversation_cache_test` / `conv_cache_test` extended with a two-carve synthetic session
(no GPU: the snapshot code paths take `SessionState` by reference and the tests already build synthetic
ones). **Owner action:** a live park/restore on the 3-way split at a restart.
**Contract.** `--layer-split` + `--conversation-cache-mib N` works and reports bytes; the existing
single-GPU snapshot format is unchanged (bump nothing).

### S3.1b — the slot object and the state machine (no behaviour change)

**Build.** `include/strata/program/slot.hpp`: a pure `Slot` struct (id, state, ids, sampling, counters,
parked index) + `pick()` + the transition table, with **no CUDA and no I/O**. A `SlotRegistry` that owns
them. Wire it into `generate.cpp`'s serve loop so that the existing single-request path runs *through*
it with exactly one slot.
**Test.** `src/program/slot_test.cpp`, CPU-only, wired into `STRATA_BUILD_TESTS` the way
`prefill_loan_test.cpp` is (`CMakeLists.txt`). Cases: every transition, refusal paths, fairness bound,
starvation, cancel unwind ordering.
**Contract.** With one slot the engine's wire output is byte-identical (verified by a golden-output
diff of the parser against the old formatter, no GPU).

### S3.1c — the protocol

**Build.** id-tagged `GEN`/`GENI`/`STOP`; `#<id>` suffixes; `READY … slots=N`; the `SLOT` line; the
optional-id parser. Keep the formatter in one place so a tagged and an untagged line cannot drift.
**Test.** `src/program/serve_proto_test.cpp` (CPU-only): round-trip, old-form requests, old-form
output when `slots=0`, malformed input, `STOP` without an id.
**Contract.** `serve/server.py` gains `slots=` detection and per-id routing (§6.3) in the same phase,
so the pair is testable end-to-end with `MockEngine` (`serve/server.py:70-104`, `MockEngine`) — **that is the only
way to test the server side without a model, and it already exists.**

### S3.1d — the swap: mount/unmount a slot — **LANDED**

**Build.** `mount(slot)` = `conversation_snapshot_restore` into the live session + `mtp.kv_restore`
(`:4683`) + `checks`/`live` swap-in; `unmount(slot)` = `park_current`-equivalent save. Both must
pass the `ConversationStageSet` S3.1a added (see §2.4) — on this box there is no single-GPU
session to mount into. Both must be called only from the engine thread, only after
`apply_pending(true)` and `refill()`.
**What landed.** `include/strata/program/serve_swap.hpp` owns the order and the decisions
(`serve_swap::run(plan, hooks, err)` walks `kOrder` through the engine's own hooks, so the ordering
is testable without a GPU); `src/program/generate.cpp` owns the state and exposes
`swap_to(incoming_id, serr, poisoned, restore_positions)` — the one call S3.1e makes. The order is
`apply_pending(true)` → `refill()` → `conversation_snapshot_validate` → save → restore →
`mtp.kv_restore` → per-slot adopt → per-slot device state (cvec, M-RoPE). Validation runs **before**
the save, which is what makes "a failed validation leaves the outgoing state intact" a property of
the order rather than a hope; a failed restore stays fatal. A conversation the cache refused to take
is marked un-resumable and re-read from token 0 instead of being mounted over stale K/V. R7 is an
acquired resource: `mrope_owner` names who the one position table belongs to, the outgoing holder's
table is copied out before any request may overwrite it, and a hand-over that would leave a
non-identity holder mounted is refused. `STRATA_NO_SWAP=1` falls back to today's serial behaviour
without recompiling; every swap prints the bytes it moved and its ms on stderr and is counted
(`Registry::note_swap`, INFO `slot_swap=0|1`) so **OQ3** can be priced at a restart.
**Test.** CPU-only: `src/program/serve_swap_test.cpp` (105 checks — the order, each failure's effect
on the session, the save/mount decisions, the budget predicate against `ConversationCache::can_fit`,
the mrope exclusivity). **Owner action:** measure the real swap cost at 32 k / 128 k / 512 k tokens
— this closes **OQ3**.
**Contract.** A slot switch leaves the session exactly as a fresh mount would: assert with
`STRATA_STATE_HASH` (`:5143-5247`), which already fingerprints gdn/ple/tail/pooled/kv/mtp/stale/dead.
**That is the single best test this design has, and it already exists.** The recipe (mount A, mount
B, mount A, compare the two A hashes) is in `.megamind/src/program/s31d-swap-notes.md`.

### S3.1e-1 — the request body as resumable steps (landed, pure refactor)

**What landed.** The serve request body in `src/program/generate.cpp` is no longer one
straight-line block. It is named steps over a per-iteration `ReqCtx`, all in the serve scope:

```cpp
enum class Prep { ok, rejected, fatal };
enum class Step { progressed, finished, cancelled, needs_swap, error, fatal_exit };

Prep prep_request(ReqCtx& R, const std::string& line);  // parse + resume + mount + sampling dispatch
void plan_prompt_segments(ReqCtx& R);                   // the four segment ends + the cursor
Step run_prefill_step(ReqCtx& R);                       // ONE prompt segment
Step finish_prefill(ReqCtx& R);                         // loan back, REUSED, decode-loop setup
Step run_decode_step(ReqCtx& R);                        // ONE verify window
Step finish_request(ReqCtx& R);                         // DONE, conversation state, metrics
```

`progressed` means "call me again"; `finished`/`cancelled` end the phase; `error` means the ERR
line is already printed and the caller must `return 1`; `fatal_exit` is #224 (already flushed,
`_Exit(1)`). **`needs_swap` is the channel S3.1e-2 uses to say "this step wants the session handed
to another slot" — nothing returns it yet.**

The serial loop calls them in 0.1.30's order, so the serial path is unchanged by construction.
`windows_ok`, `read_windows`, `lend`, `DecSnap` and `dec_snap` moved from the body to the serve
scope (they only ever captured serve-scope state). The `SlotGuard` and `MropeScope` **instances**
stay in the loop body — their destructors must run at the end of the iteration, not at the end of
`prep_request`; only their definitions moved, and they are armed through `ReqCtx`.

`run_decode_step` allocates nothing per token: `drafts`, `window`, `outv`, `dprob`, `sbuf` and
`consumed` live in `ReqCtx` and are sized once by `finish_prefill`. It is also the natural
pre-emption point (one window ≈ 16 ms).

**Test.** No new test — this is a refactor. `cd build && ninja`; `slot_test`, `serve_proto_test`,
`serve_swap_test`, `prefill_loan_test`, `conversation_cache_test`, `conv_cache_test`,
`conversation_memory_test`; `python3 -m pytest serve/test_server.py -q` (64 passed, 2 skipped).
Behaviour preservation was proven offline: the serve block has 96 `std::printf`/`std::fprintf`
calls before and 96 after, in the same execution order, and per-phase statement diffs whose only
changes are `ReqCtx` aliases and exit rewrites.

### S3.1e-2 — the concurrent driver loop — **LANDED**

**What landed.** `include/strata/program/serve_driver.hpp` (new, header-only, no CUDA/no I/O) owns the
driver's **decisions**, and `generate.cpp`'s serve block gains the loop that calls them:

```
if (driver_on) {                       // Registry::concurrent() && swaps_on && the cache is enabled
  while (!quit && !fatal) {
    drain the watchdog's per-slot kill list;             // R3
    admit_one();                                         // cap + RAM + R7 image exclusivity
    bring one pending context in  -> prep_request + plan_prompt_segments   (its own step)
    Pick p = slots_reg.pick(now);
    if p is idle -> release `busy`, block for a line, continue
    if the picked slot is deferred on the loan -> pick the longest-waiting un-deferred slot
    Step s = serve_driver::dispatch_step(p, R.phase);
    if s == swap -> swap_to(p.id, serr, poisoned, /*restore_positions=*/true), then re-dispatch
    run_one(R, s);                                     // exactly ONE step
    act on R.fail (error -> exit 1, fatal_exit -> _Exit(1));
  }
  return;
}
// 0.1.30's serial driver, text unchanged
while (next_line(line)) { ... }
```

**The gate is an early return, not an `if` around the old loop.** The serial driver's 56 lines are
byte-identical to S3.1e-1's (`while (next_line(line)) {` … `return 0;`), and the serve block's
serial-reachable output calls are still **96** — the number S3.1e-1's proof established (24 new calls,
all inside the gated regions). Every output
call the change added is inside a region that cannot run with `--serve-slots 0/1` (the driver block,
or the watchdog's `if (tagged)` branch). `STRATA_NO_SWAP=1` also falls back to the serial driver,
because a slot switch *is* a save/restore and with the hand-over off, "two slots" would mean two
conversations overwriting one session.

**`ReqCtx` lifetime — the hard part.** The driver owns `std::deque<ReqCtx> live`, keyed by request id
(a deque, not a vector: a reference into it must survive the next `push_back`, the same reason
`conv_slots` is one). `ReqCtx` gained `line`/`has_line` (the request is no longer run inside the
iteration that read it), `phase` (the driver's phase machine), `fail`, `finished`, `loan_held`,
`image_bytes` and the per-request `pp_total/pp_from/pp_t0/pp_next_check`.
Destruction order, checked against the serial path's (mrope → slot → ReqCtx → busy):
`MropeScope` and a new **`StepGuard`** are armed per *step* inside an inner scope of `run_one` and
destroyed when that scope closes — strictly *earlier* than the serial path destroyed them, and
`drop_ctx()` (the only thing that destroys a context) runs after that scope, so `~StepGuard` can never
read a destroyed `ReqCtx`. `StepGuard` is `SlotGuard` plus one condition (`ctx->finished`): arming the
serial guard per step would end every request after its first step, and arming it per request is
impossible because the request outlives the step. **`SlotGuard` itself is untouched**, so the serial
path's guard is unchanged.

**Per-step re-arming of serve-scope state.** `cur`, `pp_total/pp_from/pp_t0/pp_next_check`,
`part_next`/`part_at` and `sp.embd_rows` are process-wide, and 0.1.30 set them once per request. With
N slots each prefill step re-arms them for the slot it is about to run (`arm_prompt_state`,
`arm_prompt_view`). This is not cosmetic: `checkpoint_at` copies `cur[0, L)` and the prompt path's
chunk callback indexes `cur[p0 + t + 1]`, so a step that left `cur` holding another request's tokens
would checkpoint *that* sequence under this request's id — silent, and exactly the plausible-garbage
failure the design refuses. `part_at` is cleared when the prefilling request changes, because a
layer-split's mid-prompt parts are keyed by position and two requests reach the same positions.

**Dispatch rules that are not in §3.3's pseudocode** (both are in `dispatch_step`, both are tested):

* A **`queued` slot always admits**, never pre-swaps. `prep_request` owns the *request-line* hand-over
  (S3.1d's call site), and that is the form which takes the incoming image out of the cache *before*
  saving the outgoing conversation, so the save counts it as `held` RAM. A driver that swapped first
  would get the budget arithmetic wrong and would save and restore the same session twice.
* **`unwind` is a reason, not a step.** A cancelling slot runs its *phase tail* — `prefill_end`
  returns the loan, `decode` goes to `finish` — and never starts a new segment or window. Jumping
  straight to `finish_request` would be a real bug: on a request whose decode loop never ran,
  `consumed` is empty and `finish_request` does `live.swap(consumed)`, wiping the mounted
  conversation. A cancelling slot that is *not* mounted still needs the hand-over first (§5.3).

**The prompt loan (R8).** `serve_driver::Loan` names one owner, mirrored by `R.loan_held`. A batched
segment acquires it; a verify window and a hand-over may not run while it is held (`serve_swap::run`'s
`return_loan` step refills it, so the driver clears its own view on every hand-over, success *or*
refusal — a refusal happens after the refill). A slot that wants the loan and cannot have it is
**deferred**, and because `pick()` cannot see engine state the driver has a tie-break
(`pick_unblocked`: longest-waiting runnable slot that is not deferred). Without that the loop
livelocks: `pick()`'s prefill-before-decode rule keeps handing the blocked slot the session, the
holder is a later row in the registry's fixed array, and nothing ever advances.

**Per-slot cancellation.** `STOP <id>` sets that slot's `cancel` (S3.1c's stdin routing already did);
a bare `STOP` keeps 0.1.30's meaning and resolves through `serve_driver::bare_stop_target` =
`Registry::newest_id()` — the newest *running* slot, falling back to the newest queued one. A cancel
that arrives between steps is recorded as `DONE … cancel`, and one that arrives *before* admission
skips the read entirely (`R.cancelled = true`), so `finish_request` leaves `live` alone instead of
recording a prompt the session never consumed.

**The watchdog (R3).** The thread now reads a per-active-slot table the driver publishes
(`serve_driver::SlotWatch`, one entry per slot inside a step, stamped with the process-wide beat
count). If **every** watched slot stalled it aborts, as 0.1.30 did, and names them. If only *some*
stalled it queues them in `watch_kill` and the driver answers `ERR <id>` and destroys those slots —
the engine and the other conversations survive. If the driver never drained the list, the next pass
aborts: that means the engine thread itself is stuck inside a step, which is issue #29 and the
watchdog's whole reason for existing. `!tagged` keeps 0.1.30's body verbatim.

**Admission.** Bounded by `can_admit()` **counting the contexts already waiting to be admitted** (the
row itself is created by `prep_request`, so the driver tracks the pending context and counts it —
otherwise ten queued lines could each run with no permit and `--serve-slots` would mean nothing), and
by RAM (`admit_fits_ram`: free RAM ≥ floor + this slot's estimate + what the other live slots hold).
**S3.7 changed what that estimate is.** It used to be `parking budget / cache entries` — a flat
per-slot reservation that charged a 150-token chat the same gigabyte as a 200k-token one and refused
it for RAM it would never use. Parking itself has always been demand-sized
(`conversation_snapshot_bytes` measures `live.size()`, `ConversationCache::bytes_` counts what was
stored), so admission now prices each request at `prompt + max_new` tokens times a per-token rate:
the rate of the largest snapshot this process has actually parked, else the rate the startup
`ParkCeiling` implies at `--max_context`, else the old `budget / entries` guess — capped by the
ceiling and by the budget. `serve_driver::slot_cost_of` + `slot_image_bytes` own that precedence.
Startup refuses `--serve-slots >= 2` when parking
is off, and when the machine-sized default budget resolves to 0 (R2/R12). A refusal is `ERR <id>` and
the process survives — which is also risk R4's fix, since 0.1.30's parking failure was a `return 1`.

**R7 in phases 1-3.** `mrope_host`/`d_mrope` and the one `row_ptr`/`img_rows` set describe **one**
sequence, and a text request's prep rewrites them to the identity. So admission keeps an image request
alone: no second slot while a `GENI` runs, and no `GENI` while anything else runs. That is an
admission-time rule rather than a mid-read discovery, and it is what makes `sp.embd_rows` safe to
re-point per step.

**Test.** `src/program/serve_driver_test.cpp` — 230 CPU checks: the dispatch table, the phase machine,
admission arithmetic, the active cap, the pick rule's scenarios, the loan state machine, the N-slot
watchdog, cancel routing, and **six end-to-end simulations of the loop itself** against a fake engine
(two requests finish; the loan never has two owners; a slot blocked on the loan does not cause swap
churn; a refused hand-over does not spin the loop; a cancelled slot is unwound and answered; a row
with no context is not picked forever; an empty registry terminates). Plus the two source proofs
(`.shz_cmd/s31e2_serial_proof.py`, `.shz_cmd/s31e2_output_proof.py`).
**Owner action, at a restart:** two clients against `--serve-slots 3` — see
`.megamind/src/program/s31e2-driver-notes.md` for the exact command sequence and the lines to look for.
**Contract.** With `--serve-slots 1` the behaviour is today's; with 2+, two `GEN` lines produce two
interleaved tagged streams and each `STOP <id>` cancels only its request.

### S3.1e — the scheduler in the serve loop  *(split into S3.1e-1 and S3.1e-2, both landed)*

**Build.** Replace the request-to-completion body with the step loop (§3.1). The steps themselves
already exist (S3.1e-1 above) — this phase writes the loop that interleaves them, not the extraction.
Per-slot `stop_req`, per-slot sampling dispatch, per-slot `sfx`, per-slot `progress()` heartbeat,
per-slot watchdog.
**What S3.1d already gives you.** The hand-over is one call:
`swap_to(incoming_id, serr, poisoned, /*restore_positions=*/true)` in `generate.cpp`'s serve loop,
engine thread only. Set `mount_image` (an `optional<SavedConversation>`) to the image taken from the
`ConversationCache` first, or leave it empty for a hand-over with no restore. Call it with
`restore_positions = true` — that is the form only a scheduler can use, and it is what re-establishes
the per-slot suffix-drafter history, sampling params, penalty window, PCIe share, M-RoPE table and
`mtp.set_prompt_len` (§3.5's "applied at dispatch"). `mounted_id` is the §5.3 active slot;
`conv_of(id)` is the per-slot `serve_swap::SlotConv`; `prune_conv()` drops records whose registry row
is gone. If `swap_to` fails with `swap_wrote_session` set, the session was already written: do not
retry the hand-over and do not run the request. The drains, the ordering, the loan and the residency
rules are inside `serve_swap::run` — do not re-implement them at the call site.
**Test.** CPU-only scheduler tests; then the owner runs two clients.
**Contract.** With `--serve-slots 1` the behaviour is today's; with 2+, two `GEN` lines produce two
interleaved tagged streams and each `STOP <id>` cancels only its request.

### S3.2 — the prompt loan as a process resource

**Build.** One loan, an owner, a wait queue; the sticky loan (`:4756-4762`, `:4821`) becomes the
default across requests, which removes the 4.95 GiB/stage end-of-request refill
(`.megamind/taskboard.md:52-53`).
**Test.** `STRATA_PREFILL_LOAN_TIMING=1` (`prefill_loan.hpp`) prints the per-stage refill bill per
request; the owner's A/B is `bench/prefill/ab-prefill.sh` + `analyze.py --compare`.
**Contract.** `prompt_ms` for a *single* request must not increase (bit-exactness: the loan must be the
same slots).

### S3.3 — residency under two conversations

**Build.** Keep one `host_res`; make `adapt()` run only between steps and never during a swap; drain
`pending` before a swap.
**Test.** CPU-only: a synthetic usage vector from two conversations, assert the swap set is
deterministic and that `ResidencyUpload::due()` fires exactly when the table changed.
**Contract.** `d_res` on every device always equals `host_res` at the start of every window. Assert it
with a `STRATA_RES_VERIFY` read-back (cheap: 96 KiB).

### S3.4 — `/slots`, `/status`, `/metrics`, the Monitor tab

**Build.** §6.3. **Test.** `tools/conversation_cache_http_smoke.py` is the existing pattern for
HTTP-level tests without a model (`tools/test_conversation_cache_http_smoke.py`).

### S3.5 — prefill/decode overlap (the real win, and the risky one)

Only after S3.1a-e are green. Two sub-options, and the choice needs **OQ6** answered:

* **S3.5a — one session, prefill and decode never overlap, but the loan is shared.** Cheap, no new
  VRAM, and it already removes most of the fixed cost.
* **S3.5b — a second session arena for the prefilling slot, run on its own stream.** Needs
  `session_bytes` worth of VRAM per extra slot (1.39 GiB at 524288, 0.56 GiB at 32768) and a second
  `Prefill` instance with its own loan. **Requires the owner to accept shrinking the expert caches**,
  which costs hit rate (85.1 % today, `strata-iq3_s.log:6040`).

---

### S3.6 — the parking-collapse fix and the driver's log — **LANDED**

**The symptom** (owner, `--serve-slots 3`, two ~175K-token conversations): the first request stops
while the second prefills — correct — but when the scheduler switches back to the first one it
**ends prematurely, silently**. Diagnosis and the exact log lines:
`.megamind/src/program/stage3-parking-collapse.md`.

**The cause was one step deeper than the budget.** The parking budget collapsing to 0.0–1.6 GiB is
real and it is the reason a *finished* 175K conversation cannot be re-mounted. But it was never the
reason a *running* one was destroyed. `park_current`'s first guard is the serve-scope `live_ok`, and
`prep_request` clears `live_ok` for the **whole life of a request** ("until this request has
finished, the session is in between"); only `finish_request` puts a request's own branch into `live`
(`live.swap(consumed)`, `generate.cpp`). So for every slot in `prefill` or `decode`, `park_current`
returned `skipped` **at any budget**, S3.1d's unmount hook called `invalidate_unparked`, and the
hand-over reported `ok`. That is why every swap line in the owner's log says `saved 0 B`, and why the
resumed slot decoded against a foreign session.

**The fix, in four parts.**

1. **Publish the running branch so it can be saved.** `publish_decode_branch(ReqCtx&)` copies
   `ReqCtx::consumed` — the prompt ids plus every committed window token, i.e. exactly what
   `finish_request` later swaps into `live` — into `live`/`live_imgs`/`live_ok` and re-syncs the slot
   record, before any hand-over that would save it. `restore_published_branch` puts `consumed` back
   from the restored `live` on the way in, so the penalty history and `finish_request`'s swap see the
   sequence the slot left behind. Both live at **serve scope**: inside the driver block the name
   `live` is the driver's own `std::deque<ReqCtx>`, and a helper defined there writes the wrong
   container.
2. **Refuse the hand-over when the save is impossible** — asked *before* the unmount hook, as a new
   step in the canonical order (`serve_swap::kOrder`: `drain-residency → return-loan → park-guard →
   validate → unmount → mount → …`), so the session is left exactly as it was. `Outgoing` classifies
   what losing the session would cost the mounted slot: `finished` (no save needed, always allowed),
   `re_readable` (mid-prompt with **nothing read yet**: allowed, but the caller owes the request a
   reset to token 0),
   `must_park` (mid-decode, **or mid-prompt with tokens already read**: the state has to be saved).
   Two refusal kinds, treated oppositely by the
   driver: `budget_too_small` (never parkable here → the incoming request is **ERR'd** with both
   numbers, not starved and not spun on) and `not_saveable` (nothing whole to save *yet* → the
   incoming request **waits**, because the running read ends shortly).
3. **Keep the "re-read from token 0" promise, or end the slot.** `serve_driver::step_gate(phase,
   resumable, session_valid)` runs before every step: `run`, `re_read` (a prefill slot:
   `reset_request_to_token0` zeroes the session, clears the branch and the chain, resets
   `resume`/`read_from`/segments/checkpoints and re-issues `RESUME 0`), or `end` (a decode slot: an
   explicit `ERR`, because its generated tokens are already on the wire and re-reading the prompt
   would emit them twice). `decode && !resumable` is checked **first**, because a slot can be mounted
   and session-valid and still have lost its parked image — that is exactly the owner's case.
4. **Say it at admission.** `serve_driver::ParkCeiling` sizes a worst-case snapshot at
   `--max_context` with the engine's own `conversation_snapshot_bytes` and compares it to the budget
   in force, at driver start-up, with both numbers and what concurrency is actually possible here.
   `parking_off_refuses_slots` still catches a zero budget; this catches the **nonzero-but-useless**
   one that actually bit.
5. **S3.7 — the hand-over mount lookup, and two things that misled the owner.** From the owner's
   follow-up question ("is each slot really reserving max context size?").
   * **The error was not a budget error.** `do_swap` looked a pre-empted slot's branch up with
     `ConversationCache::best()`, whose prefix rule requires the query to be **strictly longer** than
     the entry (a resuming request must still read its last token into the next verify window). A
     pre-empted slot's branch is **exactly equal** to what it parked, so the lookup answered 0, nothing
     was mounted, the adopt hook cleared the branch, and `step_gate` ended a live decoder — on a run
     with a 10 048 MiB budget and a 113 MiB branch. `ConversationCache::best_exact()` +
     `serve_swap::seek_mount_index()` are the hand-over's own lookup: live branches only, exact only
     (a restore puts the whole entry back, so a shorter checkpoint would resume the slot past tokens it
     already generated). `prep_request` keeps `best()` — a new request's prompt really is longer.
   * *Nothing is reserved per slot* — parking sizes at `live.size()` — but `admit_fits_ram`'s estimate
     **was** a flat `budget / cache entries`, which refused short conversations for RAM they would
     never use. It is now `(prompt + max_new) x a per-token rate`: the rate of the largest snapshot this
     process has actually parked, else the rate the startup `ParkCeiling` implies at `--max_context`,
     else the old guess (`slot_cost_of` / `slot_image_bytes` / `bytes_per_token`), capped by the ceiling
     and by the budget.
   * And `gate_end_reason` used to assert "the parking budget is too small for a conversation this
     long" for **every** `Gate::end`, including `!session_valid` (not a budget condition) and runs
     whose budget was already ample. `GateEnd` now carries the state and the two real numbers and says
     which cause it is.
6. **S3.8 — two concurrent prefills erased each other and never finished.** The owner: *"when two
   prefills happen at the same time and they swap, they seem to erase the other's previous progress,
   as after each swap they always start at 0 tokens again which makes them never finish."* The log
   agrees: slot 2 (decode) parks 777 MiB and resumes, while slot 3 (prefill) reports `saved 0 B,
   restored 0 B` on every swap and `reset to token 0` **101 times**. Five links, and the last is the
   licence:
   * `publish_decode_branch` early-returns unless `phase == decode`, so a mid-**prefill** slot was
     never published — `live_ok` stayed false (`prep_request` clears it for the whole request);
   * `park_current`'s first guard is `!live_ok` → `Saved::skipped` → `saved 0 B`;
   * `invalidate_unparked` → `resumable = false`;
   * `step_gate(prefill, false, false)` → `re_read` → `reset_request_to_token0`;
   * `outgoing_for(prefill)` was **always** `re_readable` — "losing this is survivable, the caller
     owes it a reset" — so the guard *allowed* the hand-over without a save. Re-reading is survivable
     once; on every swap it is a spin, and the watchdog cannot see it because every pass really does
     read tokens.

   Fixed on both sides: `outgoing_for(Phase, read_tokens)` makes a prefill that has consumed tokens
   **`must_park`** (only a slot that has read *nothing* stays `re_readable` — re-reading that is free),
   and `publish_prefill_branch` puts `ids[0, at)` into `live`/`live_ok` before the hand-over so
   `park_current` has something to take (guarded on `R.cancelled`: a read stopped between chunks has
   nothing clean to continue from, the same reason `finish_request` guards `live.swap(consumed)`).
   `restore_published_branch` gained its prefill half — `R.at` re-derived from `live.size()`, `seg_i`
   re-planned — so the read resumes where it stopped. `kMaxRereads` + `reread_allowed` are the net
   under it: a fifth reset ends the request with a named error instead of restarting forever.
   **Both halves are needed** — publishing without the reclassification still lets the driver treat the
   slot as throwaway, and the reclassification without the publish leaves `park_current` with nothing
   to save. The end-to-end simulation in `serve_driver_test` drives off the real predicates, so
   neutering either one breaks it.
7. **S3.9 — a parked branch was being stolen by another request.** The owner, right after S3.8: *"the
   parking doesnt seem to work now, as when a new request arrives it always kicks the currently running
   request out with an 'it was decode but not parked' error."* **The park was succeeding.** The log:
   `park: slot 14 parked 10394 tokens / 994 MiB`, then `restored 9849 tokens (checkpoint) … parked=4`,
   then `slot 14's parked branch (10394 tokens) is no longer in the cache`, then the gate ends it — with
   **`evictions=0` for the whole run**, so nothing was ever pruned.
   * A parked entry is **one object**: live branch, checkpoint chain, K/V pages. A new request looks a
     branch up by **prefix** (`best()`/`find()`), which matches any *checkpoint* of any entry, and the
     mount then `take()`s the **whole** entry. Two chats sharing a beginning (9 849 of 10 394 tokens)
     meant request 15 mounted — and destroyed — the branch slot 14 had parked for itself.
   * The cache had **no concept of ownership**: nothing distinguished "warm storage for the next request
     of this chat" from "the only copy of a conversation that is still running".
   * **Why it surfaced after S3.8:** S3.8 made mid-prefill slots park, so far more entries were owned by
     live slots. The bug was always there; S3.8 moved it onto the common path.
   * **The fix is a claim, not a pin.** `SavedConversation::owner` (`kNoOwner` = −1) is the request whose
     branch this is *while that request runs*. `find`/`best`/`best_exact` take a `requester` and skip
     entries claimed by somebody else; `seek_mount_index` passes `s.id`, so a slot always gets its own
     branch back. `put(image, owner, held)` records it — one signature with defaults, because an overload
     pair would make `put(img, 5)` ambiguous between `owner` and `held`. `release_owner` runs from
     `drop_ctx`, the only thing that destroys a context, so every end path (finish, error, watchdog,
     refusal) releases it and a claim cannot leak. `return_mount_image` re-parks with the claim it came
     with. `victim_to_prune()` prefers **unclaimed** entries — pruning a claimed one does not cost a
     re-read, it ends a request — and falls back to plain LRU when everything is claimed, because the
     byte budget is still the hard limit.
   * `park_owner_for(id)` decides the claim from `working_of(id)`: `finished` → `kNoOwner`. That is
     exactly "does this slot still owe a step", and it is what keeps **stage 2** working — a finished
     conversation's branch is parked unclaimed, so the next request of that chat still resumes from its
     prefix. A claim is *not* a pin: LRU may still evict it, and the owner gets the honest "pruned while
     it waited" line. Pinning would be a guess about which client comes back.
   * Do **not** claim the `prep_request` park for `req_id`: that park saves the branch the *session*
     holds, which belongs to the still-mounted slot, not to the request being prepped.

**The log the owner asked for.** One switch, `STRATA_SERVE_TRACE=1`: every `pick()` with
`Pick::why` (computed and discarded since S3.1b), every phase transition, every swap with the reason
it was asked for, and every loan/parking deferral. Unconditional and cheap, one line per decision:
the park refusal (`park: slot N snapshot X MiB vs parking budget Y MiB …`), the park success
(`park: slot N parked T tokens / X MiB; parked=E entries, B MiB of a C MiB budget`),
`slot N resumed from T tokens`, and `slot N re-reading from token 0`. Plus a periodic
`activity:` line (`STRATA_SERVE_ACTIVITY_S`, default 30 s) carrying `peak=`, because
`slots_active` prints only on a *change* and a run that never got past one conversation otherwise
looks identical to one that was idle.

**Verified on a CPU.** `serve_swap_test` 173 checks (the guard step in the order; a mid-decode slot
that cannot be saved refuses the hand-over with the session intact and nothing after the guard run;
a finished slot swaps with no save even at a zero budget; `budget_has_room` mirrors
`ConversationCache::make_room` and *not* `can_fit` — a full cache is an eviction, not a refusal;
**S3.7:** `seek_mount_index` finds a slot's own branch where the prefix rule finds nothing, and refuses
a shorter checkpoint, a longer branch and a disabled cache; **S3.9:** a branch claimed by another
running slot is not mountable by a hand-over, and the owner still gets its own);
`conversation_cache_test` 4271 (**S3.9:** the theft replayed — live `{1,2,3,4,5}` claimed by slot 14
with a `{1,2,3}` checkpoint, request 15 asks for `{1,2,3,99}` and must get 0 with the entry still
present; releasing the claim restores stage 2's prefix reuse; `victim_to_prune` prefers unclaimed
entries and falls back to LRU when all are claimed);
`serve_driver_test` 377 (the gate over every phase × both flags; `outgoing_for` totals
`phase_has_work` **at every cursor**; the admission line's numbers and wording; a NEVER refusal ends the **incoming**
request and leaves the session with the running one; a NOT-YET refusal waits and both finish; a
parkable decoder is still pre-empted normally; **S3.7:** a slot is priced at its own length, and the
gate blames the cause it actually found; **S3.8:** a prefill with progress is `must_park`, and an
end-to-end two-prefill simulation that finishes when the read is parked and livelocks when it is not).
Anti-vacuity: dropping `park_guard` from
`serve_swap::run` fails 16 checks; making `step_gate` always `run` fails 3+3; making
`save_is_mandatory` always true fails 6; **S3.7** — reverting `slot_image_bytes` to the flat estimate
fails 9, forcing the blanket budget-blame fails 1+2, swapping `best_exact` back for `best` fails 2;
**S3.8** — `outgoing_for` back to phase-only fails 8 (including the simulation), neutering
`reread_allowed` fails 3, `default: re_readable` fails 11, neutering `save_is_mandatory` fails 13+5;
**S3.9** — removing the claim filter from `find`/`best_exact` fails 2+1 (cache+swap), neutering
`release_owner` fails 3 (the leak shows up as stage 2's prefix reuse breaking), `victim_to_prune`
back to plain LRU fails 1.

**Still needs the owner's box.** That a real pre-empted decoder's snapshot lands byte-identical
(`STRATA_STATE_HASH=1`: mount A, mount B, mount A — the two A hashes must match) and that the
`saved 0 B` lines become real byte counts. See §9 R2/R10 and
`.megamind/src/program/s36-parking-fix-notes.md`.

### S3.10 — how many tokens a slot writes per turn (`--decode-tokens N`) — **LANDED**

**The symptom** (owner): *"Currently it seems like a request generates 1-3 tokens or so and then
switches to the next request, which is too low. I need to be able to set a value here, for example 10
tokens are generated when its a requests turn."*

**The mechanism, not a bug.** A decode step was exactly **one verify window**, and a window commits
`a + 1` tokens where `a` is how many MTP drafts the model accepted — 1..8, typically 1-3. Every step
boundary is a hand-over candidate, so the scheduler was paying (or at least evaluating) a full
save+restore of a 237 MB-2.25 GB image every ~2 tokens. That is R10 in its purest form.

**The fix.** `--decode-tokens N` turns a slot's decode **turn** into a token budget instead of a
window count: `run_decode_step` loops its existing window body until the slot has produced N tokens
since the turn began, then returns `progressed` and the scheduler may hand the session over. N is in
tokens because that is the unit the owner asked for and the unit the wire, `--max-new` and the slot
counters already count in; a window count would be a different amount of work every round depending on
acceptance. A window may overshoot the budget — a window is the smallest unit of work that exists.

What is deliberately unchanged: the first window of a turn always runs (a turn that runs nothing is a
scheduler stall, and the watchdog's heartbeat would freeze); EOS, `--max-new`, `cancelled` and
`STOP <id>` still end the request from inside the window, so a long turn never delays a cancel by more
than one window; the pre-emption point is still between windows, so `swap_to` never interrupts a
window; and the default (`0`/`1`) runs exactly one iteration, which is 0.1.30's step.

**Contract.** `serve_driver::decode_step_tokens(budget, produced, max_new)` — how many more tokens
this turn may emit, 0 meaning "no window may run" — and `serve_driver::decode_turn_done(budget,
produced_before, produced_now)`. Both are pure and pinned by `serve_driver_test` (395 checks, 18 new).
The driver's start-up line prints the setting (`decode 10 tokens/turn` / `decode 1 window/turn`) so a
log can be read without guessing.

**Interaction with `--starve-ms`.** They are the two ends of the same trade. `--decode-tokens` sets
how much work a slot does before it *can* be pre-empted; `--starve-ms` sets how long another slot will
wait before the scheduler *forces* the pre-emption. Raising the first without the second is what
removes the swap churn; raising the second alone just delays the same 2-token hand-over.

**Still needs the owner's box:** OQ3 (the swap cost) and OQ4 (the fairness bound) were never measured,
and this knob is exactly where they meet. Start with `--decode-tokens 10` and compare `swaps=` /
`swap_ms=` in the activity line against the default.

## 9. Risk list

| # | risk | evidence | mitigation |
|---|---|---|---|
| **R1** | **457 MiB of VRAM headroom.** Any new per-slot VRAM allocation is a hard no on this box; under WDDM a full GPU does not fail, it *pages*, and a page-in while a verify graph spins on a host flag stalls forever. | `strata-iq3_s.log:6079`; the engine's own warning `generate.cpp:4265-4277` | No per-slot graphs, no per-slot `d_res`, no per-slot verifier arena. Budget check at startup: refuse `--serve-slots N` rather than overcommit. |
| **R2** | **~2 GiB of host RAM, parking budget already 0.0 GiB.** Slots are host-RAM objects. | `strata-iq3_s.log:6078` | Admission control (§2.3) using the existing `conversation_memory_admit` (`:3920`). Default `--serve-slots` must be **0/1** until the owner frees RAM. **S3.6:** a *nonzero-but-useless* budget is now named at driver start-up with both numbers (`serve_driver::ParkCeiling`), and a hand-over that cannot save is refused instead of destroying the conversation. |
| **R14** | **A slot stepped against a session that does not describe it.** A pre-emption that cannot save its slot used to destroy the conversation and let the slot keep decoding — plausible tokens, immediate EOS, no error. | `strata-iq3_s.log` (`swap: slot 5 NOT parked …`, `saved 0 B` on every swap); `.megamind/src/program/stage3-parking-collapse.md` | **S3.6.** `serve_swap::park_guard` refuses the hand-over before anything is destroyed; `serve_driver::step_gate` ends (decode) or re-reads (prefill) any slot whose session went away. `publish_decode_branch` makes a decoder parkable at all — without it `park_current`'s `live_ok` guard refused every pre-emption regardless of budget. |
| **R3** | **The watchdog `std::abort()`s the process** after 60 s without a beat. With two slots, one stalled request kills the other's work and the server restarts the engine mid-answer. | `generate.cpp:4325-4341`, `progress.hpp:17-24` | Per-slot heartbeat; the watchdog aborts only if **every** active slot stalled, and reports which. A slot that stalls alone gets `ERR <id>` + slot destruction. |
| **R15** | **Two concurrent prefills erase each other and never finish.** A mid-prompt slot was classified `re_readable` and never published, so every swap saved nothing and sent it back to token 0 — invisible to the watchdog, because every pass really does read tokens. | `strata-iq3_s.log` (`slot 3 reset to token 0: 81 prompt tokens will be read again`, ×101, while slot 2 parked/resumed normally) | **S3.8.** `outgoing_for(phase, read_tokens)` makes a prefill with progress `must_park`; `publish_prefill_branch` gives `park_current` something to save; `restore_published_branch` resumes the read at `live.size()`. `kMaxRereads` ends a request with a named error rather than restarting forever. |
| **R16** | **A parked branch mounted (and thereby destroyed) by a different request.** A parked entry is one object; a prefix lookup matches any checkpoint in it, and mounting `take()`s the whole entry — so a new request sharing a beginning with a running slot's conversation deleted the only copy of that slot's state, and the slot was ended. The park itself succeeded; `evictions=0` throughout. | `strata-iq3_s.log` (`park: slot 14 parked 10394 tokens` → `restored 9849 tokens (checkpoint) … parked=4` → `slot 14's parked branch … is no longer in the cache` → `would step in decode with no conversation`) | **S3.9.** `SavedConversation::owner` claims an entry for the request that parked it while that request runs; `best`/`best_exact` refuse it to anybody else, `seek_mount_index` passes the slot's own id, `release_owner` runs from `drop_ctx`, and `victim_to_prune` prefers unclaimed entries. |
| **R4** | **`park_current` failure kills the process** (`return 1`). | `generate.cpp:4587-4589` | Per-slot error path (§3.5). |
| **R5** | **The doorbell/pool epoch protocol.** Two window issuers would reproduce issue #29 (a late worker claiming from the next batch → `done != n` forever → the GPU waits forever). | `pool.hpp:19-27`, `prefill.cpp:149-151` | One engine thread issues windows (§3.1). No second `Verifier::run` caller, ever. |
| **R6** | **CUDA context / device switching.** The layer split already switches devices per call (`OnDevice`, `generate.cpp:4180`, `verify.hpp:147`, `mtp.hpp:131`). A prefill thread on a second stream multiplies the switching. | `verify.cpp` device handling; `st->dev` | Keep one thread until S3.5; when adding a thread, pin it to a device and never let it touch another stage's. |
| **R7** | **`d_mrope` is one device table for the process.** A GENI request rewrites it for the whole cell range and resets it to the identity for the next text request (`generate.cpp:4436-4513`, table allocated at `:1830-1843` and per stage at `:2038-2039`). Two concurrent requests, one with images, would corrupt each other's positions. | `:4459-4508` | In phases 1-3, **only one slot may hold a non-identity mrope at a time**: treat the mrope table as a resource the prefill step acquires, and force a swap+re-upload on every slot change. Longer term, a per-slot table — which is more VRAM (R1). |
| **R8** | **The prompt loan is one resource.** Two prefills both lending the cache tail would hand the same slots to two `Prefill` instances. | `:3557-3600`, `:4801` | Loan ownership + wait queue (§4.4). |
| **R9** | **Residency is one table but conversations have different hot sets.** A swap that leaves the previous conversation's experts resident is not wrong, just slower; a swap that leaves a *half-applied* `pending` swap is wrong. | `:4071`, `:4130-4144` | `apply_pending(true)` before every swap (§4.2). |
| **R10** | **Swapping too much.** Each swap is a full save+restore of a 237 MB-2.25 GB image. A naive round-robin would spend the request's whole budget on memcpy. | snapshot sizes from `strata-iq3_s.log` | Stay-on-active-slot default; `--starve-ms` fairness bound; **S3.10: `--decode-tokens N` makes a turn worth N generated tokens instead of one 1-3 token window**, so the swap is amortised over real work; measure (OQ3). |
| **R11** | **Windows / HIP.** `--shared-expert-arena` is `#ifdef _WIN32`-guarded off; the cross-process core lease is Linux-only; HIP has its own blocking-staging path in `ExpertCache`. | `.megamind/roadmap.md:100-102`, `144-146`; `expert_cache.hpp:144-148` | Every new file must compile under `_WIN32` and `STRATA_USE_HIP`; the core-lease and arena code is untouched by stage 3, so the risk is only in new syscall use (none allowed). |
| **R12** | **`--kv k8v4` cannot stream** (`qsa_state_init` refuses `--kv-resident` with hybrid), so on that config the K/V is fully resident in VRAM and a slot costs 13 GB. | `src/core/layer.cpp:557-562`; `docs/DETAILS.md:70` ("It does not stream its KV cache") | Refuse `--serve-slots > 1` when `kv_mode == 0`. Detect it from `ss.qsa_states[ss.qsa_primary()].kv_mode`, which `INFO` already reports (`:4306-4307`). |
| **R13** | **Two agents in one file.** `generate.cpp` is 6 131 lines and S0.3 already edited its serve loop. | `.megamind/taskboard.md:63-71` | S0.3 must land before S3.1 (already sequenced). Keep new logic in `include/strata/program/slot.hpp` + `src/program/serve_*.cpp`, so `generate.cpp` only gains a call site. |

### Open questions the implementation agent must resolve **first**

* **OQ1** — Does `--kv k8v4` (no streaming) mean the swap path is impossible, or should k8v4 gain
  `--kv-resident` support? (`src/core/layer.cpp:557-562`)
* **OQ2** — What is the real per-slot VRAM cost on the *owner's* split config, per stage?
  `session_bytes` says 0.554/0.560/0.547 GiB, but the split search already spends the free VRAM
  (`strata-iq3_s.log:24-26`); the answer is "whatever the cache gives back", which needs a live
  `cudaMemGetInfo`.
* **OQ3** — Measured save+restore time per token count, on the split, once S3.1a lands. Everything
  about `--starve-ms` and the swap policy depends on it.
* **OQ4** — The real `T_starve`. Predicted from a 24 ms window (`strata-iq3_s.log:27`) and an 8-15 s
  chunk; must be measured.
* **OQ5** — Is a per-slot `d_res` (96 KiB × 3 devices per swap) ever worth it? Measure the swap cost
  before answering.
* **OQ6** — Is a second session arena affordable at any useful `--max-context`? Needs a live
  `cudaMemGetInfo` after shrinking the caches.
* **OQ7** — Does `Verifier::set_head_sampling(false)` (`:4719-4722`) interact with a per-slot dispatch
  correctly when a prompt read through windows is interrupted by a swap? The flag is restored by an RAII
  guard scoped to the read; a swap between chunks must not leave it false.

---

## 10. Deliberately out of scope, recorded so it is not lost

* **A shared KV arena with per-sequence page tables** (llama.cpp's model). It is the only route to
  true token-level batching, and it requires: one pool instead of 12 per stage, a per-row page table,
  per-row GDN state, and re-capture of every window graph per batch shape. That is a stage-4 engine,
  not a stage-3 scheduler.
* **Token-level batching of two decode slots into one window.** Rejected in §1; the kernels bound
  `n_tok ≤ kVerifyMaxT = 8` at *consecutive* positions.
* **Per-slot CUDA graphs.** Rejected in §5.2 (A).
* **Persisting snapshots to disk.** `docs/DETAILS.md:543-544` — still no.
* **Multi-process scheduling.** Stage 1's job (shared arena + core leases).

---

## Appendix A — how the numbers in this document were produced

No engine, no model, no GPU work was started. The figures come from:

1. **Running the cost functions.** `session_bytes`, `qsa_state_bytes`, `qsa_buffers_bytes`,
   `moe_buffers_bytes`, `block_buffers_bytes`, `gdn_buffers_bytes` are host-side arithmetic
   (`src/core/session.cpp:53-72`, `src/core/layer.cpp:529-545`). A scratch binary linked against
   `build/libstrata_engine.a` + `libstrata_core.a` + `libstrata_kernels*.a` +
   `libstrata_spec.a` + `build/ggml/src/libggml-{cpu,base}.a` and called them with the real
   `ModelGeometry`. It never calls `session_init` (that one does `cudaHostAlloc`).
2. **`./build/strata-plan --max-context 524288 --vram 17000000000`** → `KV + indexer keys 6.845104 GB`,
   `recurrent state 0.117670 GB`, `expert cache 10.037226 GB (7260 slots of 1382400 B)`.
3. **The live server's own log**, `strata-iq3_s.log` (read-only), for everything only a run knows:
   cache slot counts, the prompt loan, the verifier arena, the MTP, the free VRAM, the parking budget,
   the snapshot sizes, the KV-streaming hit rate.

Reproduce (1) with:

```
g++ -std=c++20 -O1 -Iinclude -I/opt/cuda/include -o /tmp/sb scratch.cpp \
    -L build -lstrata_engine -lstrata_core -lstrata_kernels -lstrata_kernels_cpu -lstrata_spec \
    build/ggml/src/libggml-cpu.a build/ggml/src/libggml-base.a \
    -L/opt/cuda/lib64 -lcudart -lcublas -lpthread -ldl
```
