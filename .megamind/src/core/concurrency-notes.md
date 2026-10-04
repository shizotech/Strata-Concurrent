# Stage 3 — concurrency design notes (S3.0, design only)

Deliverable: `docs/STAGE3-CONCURRENCY.md`. No source was modified.

## Measured numbers (all from the real code, not guessed)

`session_bytes()` was **run** (host-only arithmetic; `session_init` is the only CUDA-touching half):

| config | one full 48-layer session |
|---|---|
| `--max-context 524288`, `--kv fp16`, `--kv-resident 0` | **13.96 GB (13.00 GiB)** — matches the task's "~6.8 GB KV+indexer" × 12 QSA layers |
| `--max-context 524288`, `--kv int8`, `--kv-resident 32768` (this box, `strata-iq3_s.json`) | **1.49 GB (1.39 GiB)** |
| same, per layer-split stage [0,16) / [16,34) / [34,48) | 0.594 / 0.601 / 0.588 GB |
| `--max-context 131072`, int8, resident 32768 | 0.78 GB |
| `--max-context 32768`, int8, resident 32768 | 0.60 GB |

Plus, per slot, NOT inside `session_bytes`: the KV-streaming **host** copy (6.19 GiB at 524288 — log line
"KV streaming: 32768 of 524288 cells per QSA layer in VRAM, the K/V in 6.19 GiB of pinned RAM"),
the MTP draft ring state 35.1 MB, and the RoPE table 134 MB (128 MiB) at 524288.

`./build/strata-plan --max-context 524288 --vram 17000000000` → `kv_bytes = 6.845 GB` (that is **per QSA
layer**; `plan.hpp:91` multiplies by `n_qsa_layers`).

Live box (`strata-iq3_s.log`, restart at line 6046): CUDA0 cache **7778 slots = 14.75 GB**, prompt loan
**2625 slots = 4.95 GB**, verifier arena **76.9 MiB**, MTP **836 MiB**, **457 MiB free with everything
loaded**. 457 MiB = **346 expert slots**.

## The pointer-baking inventory (why "just re-capture per slot" is not the answer)

* `Verifier::exec_[9]` (`include/strata/core/verify.hpp:177`) — one graph per window length, captured
  against `ss_` and its own `arena_` (`src/core/verify.cpp:265`).
* `MtpDrafter::prefill_exec_[9]`, `prefill_dev_exec_[9]`, `round_exec_[9]`, `step_exec_[9]`,
  `step_exec_c_[9]` (`include/strata/core/mtp.hpp:109,136,137,140,113,114`).
* `SessionGraphs::execs/posts/preA/preB/preP[5]` (`session.hpp:133-196`) — serve never captures them
  (`generate.cpp:2916` is guarded by `!native_pack`, and this box runs a native pack).
* `TokenGraph` (`session.hpp:395`) — only when `!multi_gpu` (`generate.cpp:3395`), so not on this box.
* **`Prefill` is NOT captured** (`src/prefill/prefill.cpp` has no `cudaStreamBeginCapture`) — it launches
  kernels directly. That is what makes the chosen design possible.

## The decision

**One captured graph set, bound to one "active slot"; switch the slot between windows; swap the rest of
the slot's state through the existing snapshot machinery.** Re-capture per slot is refused (a second
Verifier + MTP set is ~160 MiB against 457 MiB free, plus a capture stall per switch). Indirection is
refused as the mechanism (it rewrites every kernel signature and breaks the bit-exactness bar).

The three things that make this cheap on this box:

1. `--kv-resident 32768` already keeps the **authoritative K/V in host RAM** (`src/core/layer.cpp:625-640`),
   so a slot's positional state is a host memcpy, not a VRAM copy.
2. `conversation_snapshot_save/restore` (`include/strata/core/conversation_snapshot.hpp:66,75`) already
   moves exactly that set (running state + checkpoints + used K/V pages + draft K/V).
3. The prompt path is not graphed, so a prefill for slot B can overlap a decode for slot A.

## Pitfalls found while reading (do not rediscover them)

* `park_current` returns **false** on a snapshot-save failure and the caller does `return 1` — it kills the
  process (`generate.cpp:4587-4589`). A concurrency scheduler must not inherit that.
* The watchdog `std::abort()`s the whole engine on 60 s without a beat (`generate.cpp:4326-4341`).
  With two slots, one stalled request must not kill the other.
* `stop_req` is one `std::atomic<bool>` for the process (`generate.cpp:4203`); `STOP` cannot name a request.
* `Verifier::set_sampling/set_history/set_head_sampling` mutate shared state before `run`
  (`verify.hpp:73,84,92`) — they must be applied at dispatch, not at parse.
* `host_res` is ONE table whose slot values index whichever cache owns the layer (`generate.cpp:3554`);
  a per-conversation residency view must be a derived copy, never a second source of truth.
* `session_bytes` at `--kv k8v4` **refuses** `--kv-resident` (`src/core/layer.cpp:557-562`) — so the
  host-KV trick that makes swapping cheap does not exist for k8v4. Open question O1.
