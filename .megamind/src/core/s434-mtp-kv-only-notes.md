# S4.3.4 — `MtpDrafter::bind_kv_only` (the KV-only drafter)

Files touched: **`include/strata/core/mtp.hpp`, `src/core/mtp.cpp` only.** No engine, no model, no GPU work,
`/dev/shm` untouched. `generate.cpp`, `CMakeLists.txt`, `serve/*`, `setup.py`, `src/kernels/*`,
`docs/STAGE4-*` all untouched.

## The API (this is what S4.3.5 calls)

```cpp
bool bind_kv_only(const WeightTable& wt, std::string& err);   // the whole new surface
bool kv_only() const;                                        // the mode flag
static const char* draft_refusal_reason();                   // the exact refusal string
int64_t draft_refusals() const;                              // must stay 0 in a prefill instance
uint64_t draft_only_bytes() const;                           // what load() skipped, computed from the code
```

`bind_kv_only(wt, err)` mirrors `bind`'s table handling and nothing else:

* `OnDevice(device_)` (so it is legal on a layer split, where the drafter lives on the last stage);
* requires `wt.find("output.weight")` — the **same** refusal `bind()` gives (`"mtp: output.weight is missing"`)
  — and takes `n_vocab_` from it;
* sets `wt_ = &wt` (required: `record_forward` reads `wt_->find("token_embd.weight")`);
* sets `kv_only_ = true`, and pins `head_ = window_R_ = nullptr`, `coupled_ok_/active_/rec_ = false`;
* allocates **nothing**.

**Order with `load()` does not matter.** `bind_kv_only` never reads `max_t_`; `load()` reads `kv_only_` when it
carves. S4.3.5's natural order is `load()` at `generate.cpp:2519` (before the expert cache is sized) and
`bind_kv_only()` where `bind()` is today (`generate.cpp:4015`) — that works, but then the buffer saving is NOT
taken (see "the ordering decision" below). To take it, call `bind_kv_only` before `load`.

Guards, in both directions:

* `bind()` on a KV-only drafter **fails**: `"mtp: this drafter is bound KV-only (bind_kv_only); it cannot be
  bound for drafting"`. The mode can never be silently upgraded.
* `bind_kv_only()` on a drafter already bound for drafting (`head_ != nullptr`) **fails**:
  `"… already bound for drafting (bind); it cannot be downgraded to KV-only"`. A failed `bind_kv_only` leaves
  the object exactly as it was.

## The one-line change S4.3.5 must make in `generate.cpp` (the parent owns that file)

`src/program/generate.cpp:4015` is the serve path's bind:

```cpp
if (!ver.init(wt, g, ss, vh, native_head.loaded() ? &native_head : nullptr, o.spec, err) ||
    !mtp.bind(last_st ? last_st->wt : wt, last_st ? &last_st->head : &native_head, ver.final_R_all(), err)) {
```

For `--role prefill` that becomes (the drafter's own guards make the rest of the line irrelevant):

```cpp
!(last_st ? mtp.bind_kv_only(last_st->wt, err) : mtp.bind_kv_only(wt, err))
```

i.e. replace the `mtp.bind(<wt>, <head>, <window_R>, err)` call with `mtp.bind_kv_only(<same wt>, err)`.
Nothing else at that site is needed: `ver.init` is skipped for the role anyway (§1.3 of the design doc — the
prefill role allocates no `Verifier`), and `ver.final_R_all()` is exactly the pointer the mode must never have.

**To also collect the buffer saving**, hoist the mode flag above `mtp.load()` at `generate.cpp:2519`, e.g.
call `mtp.bind_kv_only(...)` right before `mtp.load(...)` in the `if (!o.mtp.empty())` block at
`generate.cpp:2508-2520`, and keep the `bind_kv_only` at 4015 as the authoritative bind (it is idempotent for
the same table). If the mode is only set at 4015, `load()` has already allocated the draft-only buffers and
`draft_only_bytes()` reports the number it did **not** skip — the VRAM is still there. This is a real choice,
not a detail: see the numbers below.

## Preserved (byte-identical) — what a prompt read does

The prompt path is **not touched at all**. `prefill()` (`src/core/mtp.cpp:791`, plus one new null guard at
`:796`), `capture_prefill` (`:710`), `capture_prefill_dev` (`:721`) and `record_forward`'s K/V-only branch are unchanged, statement for statement.
The draft K/V a prefill instance writes is the same bytes today's prefill writes, so the payload format and the
K/V contents do not move:

* `conversation_snapshot_test` **2089 checks passed** (unchanged count, on a device);
* `conversation_validation_test` **1323 host-only checks passed** (unchanged count);
* `Prefill::draft_kv` (`src/prefill/prefill.cpp:686`) is untouched and still applies in this mode — it only
  reads `kv_state_rw()`, `tensor_f32/bf16/q8`, `first_needed()`, `device()` and `idle()`, all of which the mode
  keeps. It is still the fast path, and it still writes the same K/V.
* `kv_restore(upto)` is kept (the ring refill a resume needs) — it touches only `st_`.
* `set_prompt_len()` / `first_needed()` are kept: that skip is part of what makes the draft K/V identical.

## Skipped in kv-only mode (file:line after the change)

| what | where | note |
|---|---|---|
| `head_logits_` (draft logits) | `bind()` `mtp.cpp:426-429` never runs | 5.7 MiB at this box |
| `dhead_` + `dvocab_` (the draft head over `draft_vocab.bin`) | `bind()` `mtp.cpp:432-449` never runs | 81.2 MiB, `strata-iq3_s.log:35` |
| `setup_coupled` (the coupled draft sampler: `cparams_`, `cring_`, `cscratch_`, `dinv_`, `h_cparams_`, `h_chist_`) | `bind()` `mtp.cpp:450` never runs | 0 here (`STRATA_SPEC_COUPLED` unset) |
| `window_R_` (the verifier's `final_R_all()`) | never set | the prefill role has no `Verifier` |
| `head_` → the whole `NativeHead` in the prefill role | `bind()`'s `--native` requirement (`mtp.cpp:425`) gone | **497 MiB** (`strata-iq3_s.log:19`) — this is OQ-S4-9 |
| `round_exec_[]`, `step_exec_[]`, `round_exec_c_[]`, `step_exec_c_[]` | `capture_round` `mtp.cpp:729`, `capture_step` `mtp.cpp:764` refuse | 36 graph execs never captured |
| the draft-only device buffers | `load()`'s carve, `mtp.cpp:264-300` (`if (!kv)` groups), priced at `:301-306` | see the numbers |
| `h_row_/m_row_`, `h_out_/m_out_`, `h_prob_/m_prob_` mapped staging | `load()` `mtp.cpp:255-258` | 240 B of pinned host |
| `ident_` upload (`T × cap_` int32) | `load()` `mtp.cpp:313` (`if (ident_ != nullptr)`) | skipped with the buffer |
| `bind_bytes()` | `mtp.cpp:338` returns 0 | the expert cache keeps the reserve (`generate.cpp:2654`) |
| `set_draft_sampling` | `mtp.cpp:400` forces `coupled_active_ = false` | can never write the null `*h_cparams_` |

## The bytes saved — computed from the code, not guessed

`load()` now prices both carves and stores the difference in `draft_only_bytes_` (printed at start-up).
Measured by replaying the exact `Bump` sequence of `load()` before and after the change
(`.shz_cmd/s434_carve_parity.cpp`, host-only):

```
live box: --spec 4 (+2 suffix bump, generate.cpp:1659-1661 -> max_t 6), --mtp-window 32768, max_cells 524288
          arena 17 033 984 B  ->  1 564 416 B   saved 15 469 568 B = 14.8 MiB
--spec 4, max_t 4, window 32768                     saved 14 545 408 B = 13.9 MiB
--spec 8, window 0 (whole context in VRAM)          saved 112 503 296 B = 107.3 MiB   (ident_ dominates)
--spec 2, window 2048                               saved  1 486 848 B =  1.4 MiB
```

The same program asserts that in **full** mode the arena size and **every buffer's offset** are identical to
before the change (4 configs, name-by-name). So `--role full` is untouched down to the byte layout.

Total VRAM a prefill instance does not pay for, at the live config:

```
draft-only device buffers (load)      15 469 568 B   14.8 MiB   <- only if bind_kv_only runs BEFORE load
head_logits_ (bind)                    5 959 680 B    5.7 MiB   = max_t 6 x n_vocab 248320 x 4
dhead_ + dvocab_ (bind)               85 264 600 B   81.3 MiB   = 40525 tokens (log:35) + the id table
                                      -----------------------
inside the drafter                   106 693 848 B  101.7 MiB
plus NativeHead, which the role no longer needs  521 472 000 B  497 MiB  (generate.cpp's decision, unlocked here)
```

`bind_bytes() == 0` means the `mtp_bind` reserve at `generate.cpp:2654` disappears, so `--expert-cache auto`
gives those bytes back as cache slots (≈ 77 slots at 1 382 400 B/blob for the 87 MiB the drafter itself stops
allocating).

## How a draft pass is prevented (four layers, all tested)

1. `draft()` `mtp.cpp:893` → `refuse_draft(...)`;
2. `draft_first()` `mtp.cpp:951` → the same, **before** it stages into the null `window_R_` (and `:954` refuses a
   null `window_R_` even when the mode is off — a new guard, not a rewrite);
3. `capture_round()` `mtp.cpp:729` / `capture_step()` `mtp.cpp:764` refuse, so no draft graph is ever captured;
4. `record_forward()` `mtp.cpp:526` refuses the `full` branch (`step_row0 >= 0`) **before dereferencing
   `g_`/`ss_`**, so even a path that skipped 1-3 cannot run a draft forward. The `!full` branch is untouched.

`refuse_draft` (`mtp.cpp:507`) zeroes `drafts[0 .. max_t_-2]` and `probs[0 .. max_t_-2]` (the same span
`draft()`'s own tail-clearing loop writes, and always inside the caller's buffer, which the engine sizes to
`o.spec` == `max_t_`), sets `*n_drafts = 0`, names the reason in `err`, bumps `draft_refusals_`, and returns
**false**. Deliberate: every caller in `generate.cpp` checks the return
(`if (!drafted) { sp_out.err(err, R.id); return Step::error; }` at `:7293`, `return 1` at `:9649`/`:9752`), so
a role mistake is a loud, named request error — not a window of zero tokens, and not a stale token. A caller
that *ignores* the return still reads zeros, never garbage.

## Caller audit (the task asked for `verify.cpp` + `generate.cpp`)

* **`src/core/verify.cpp` has no drafter caller at all.** `grep -n "Mtp\|mtp" src/core/verify.cpp
  include/strata/core/verify.hpp` → zero hits. The drafter and the verifier meet only through
  `ver.final_R_all()`, which `bind()` passes as `window_R_`; the mode never has that pointer, and
  `draft_first`'s guard is before the first use of it.
* `src/program/generate.cpp` (read-only for me) — every drafter call site and its answer in kv-only mode:

| line | call | kv-only behaviour |
|---|---|---|
| 2519 | `mtp.load(...)` | unchanged (plus the smaller carve if the mode is set first) |
| 2654 | `mtp.bind_bytes(...)` | 0 → the reserve goes to the expert cache |
| 4015 | `mtp.bind(wt, head, ver.final_R_all(), err)` | **the one site S4.3.5 replaces** with `bind_kv_only` |
| 4102 | `conv_stages.draft_device = mtp.device()` | unchanged |
| 4191/4198/4227 | `conversation_snapshot_{bytes,capture_bytes,save}(…, mtp.kv_state(), …)` | unchanged — this is the payload |
| 4415/4416 | `sp.draft_kv(mtp, …)` / `mtp.prefill(…)` | unchanged — the K/V append, the whole point |
| 5296 | `mtp.prefill(ver.final_R_all(), …)` (the non-batched prompt path) | unchanged |
| 5227-5228 | `mtp.set_max_drafts(S_mtp - 1)` | harmless (a setter; `max_drafts_` is only read by `draft()`) |
| 5654 | `mtp.set_draft_sampling(c.smpl)` | forces `coupled_active_ = false` |
| 5661 / 6787 | `mtp.set_prompt_len(...)` | unchanged (needed for the identical K/V) |
| 5942 | `mtp.kv_restore(live.size())` | unchanged (ring refill) |
| 6667/6695/6704/6711 | snapshot validate/restore/verify on `mtp.kv_state()` | unchanged |
| 7278-7279 | `mtp.coupled()` / `mtp.set_draft_history(...)` | `coupled()` is false, so `set_draft_history` is never reached; and it early-returns on `!coupled_active_` anyway |
| **7281** | **`mtp.draft(T, outv.data(), p, a, drafts.data(), err, dprob.data(), …)`** | refuses: 0 candidates, outputs zeroed, `err` named, returns false → `:7293` ends the request with the reason. **No edit needed** — the drafter's API makes the site correct as written. |
| 7463-7475 | `STRATA_STATE_HASH` over `mtp.kv_state()` | unchanged — this is the live proof hook |
| 9240-9250 | the non-serve prompt path: `mtp.bind(wt, &native_head, nullptr, err)` then `prefill.draft_kv`/`mtp.prefill` | a prefill-only *batch* run would use `bind_kv_only` here too; today's `--role full` batch path is unchanged |
| 9545/9648/9656/9670/9747 | the non-serve decode path: `bind`, `draft_first`, `set_max_drafts`, `draft` | `draft_first`/`draft` refuse; `bind` refuses if the mode was set. A prefill instance never reaches this block (it does not serve and does not decode). |
| 9807-9810 | `mtp.ms_draft / rounds` reporting | guarded by `rounds > 0`; `rounds` never increments in this mode |

* `src/prefill/prefill.cpp:686-827` (`Prefill::draft_kv`) — no change needed, see above.

## Verification actually run

* `cd /ssd/Strata/build && ninja` → **clean** (exit 0; the only warning is the pre-existing unused `argmax` in
  `generate.cpp`; the `ld: … no .sframe` lines are this toolchain's noise on every link, present before the change).
* `conversation_snapshot_test` **2089 passed**, `conversation_validation_test` **1323 host-only passed** —
  built and run UNCHANGED in the scratch tree `/tmp/s434build`
  (`-G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
  -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON`).
* No-drift on the rest of the CPU suite, same tree, all green: `saved_conv_wire_test` 483 ·
  `handoff_arena_test` 268 · `prefill_svc_test` 2640 · `serve_driver_test` 523 · `serve_swap_test` 173 ·
  `serve_proto_test` 140 · `slot_test` 176 · `prefill_loan_test` 128 · `conv_cache_test` PASS ·
  `coupled_draft_test` PASS · `conversation_cache_test` 4282 · `conversation_memory_test` 23 ·
  `file_expert_source_test` PASS.
* **There is no existing mtp test target** (`grep -n mtp CMakeLists.txt` → only the `strata_engine` source
  list), so per the constraints I did **not** add one. The CPU-only check lives at
  `.shz_cmd/s434_kv_only_gate.cpp` (33 checks, links the real `libstrata_engine.a`, no GPU, no model, no
  `session_init`, no `/dev/shm`). It is scratch, not a repo target — if the parent wants it permanent it must
  add the target itself.
* **Non-vacuity, three ways:**
  1. *Arm A of the scratch check*: with the mode **off**, `draft()` gets past the guard and fails only at
     `"mtp: begin capture"` (no device) — so what stops it in arm B is the guard, not the absence of a GPU.
  2. *Arm B*: with the mode **on**, `draft()`/`draft_first()` return false, `*n_drafts == 0`, every
     `drafts`/`probs` slot is zeroed, `draft_refusals()` moves, all 36 `round_/step_(c_)` exec slots stay
     null, `capture_round`/`capture_step`/`record_forward(full)` refuse, `coupled()` stays false, `bind()`
     refuses, `bind_bytes() == 0`.
  3. *Mutations*: neutering `draft()`'s guard → **4 checks fail**; neutering the `record_forward`/
     `capture_round`/`capture_step` gates → the scratch binary **segfaults (exit 139)**, which is precisely the
     "read uninitialised state" failure the guards exist to prevent. Both mutations reverted; the tree is clean.

## What still needs the owner's live `STRATA_STATE_HASH` run

The design doc says it plainly (`docs/STAGE4-SPLIT-ROLES.md:1746-1748`): "`bind_kv_only` being sufficient … is
**not** proven until S4.3.4 builds and the owner runs the `STRATA_STATE_HASH` comparison." It builds. The
comparison is the owner's, and it is the only thing that can prove the draft K/V a prefill instance writes is
the same K/V a full instance writes. Recipe (`STRATA_STATE_HASH=1`, `generate.cpp:7463-7475` prints
`… kv=%016llx mtp=%016llx …` per request; the `mtp=` field is a FNV-1a over the drafter's K/V pools):

1. Run A: `--role full` (today), one request, note `mtp=` and `kv=` for the same `L`.
2. Run B: `--role prefill` + `--role decode` with the handoff, the same prompt and the same ids.
3. **The `mtp=` hash at the end of the prompt must be identical between A and B** at the same context length.
   Compare at the first hash line after the prompt (before any decode), because decode windows also append
   draft K/V in run A.
4. Also compare the `kv=` (main layers) hash — that one proves the payload mount, not this step, but a
   mismatch there means the handoff, not the drafter.
5. `mtp.draft_refusals()` must be **0** in the prefill instance for the whole run. If it is not, something
   reached a decode path; it is still safe (no draft ran), but the role plumbing is wrong. S4.3.5 should print
   it in the prefill instance's `INFO`/`--help`-adjacent start-up line together with `draft_only_bytes()`, so
   the 14.8 MiB (or 107 MiB at `--spec 8` with no window) is visible in the log.
6. The start-up lines to expect from this step:
   `strata mtp: KV-ONLY mode (S4.3.4): prompt K/V only, no drafting; 15469568 B of draft-only buffers not allocated (14.8 MiB)`
   and `strata mtp: bound KV-ONLY: prompt draft K/V only, no draft head, no draft logits, no coupled sampler, no drafting`,
   and **no** `strata mtp: draft head over 40525 tokens (81.2 MiB)` line (that is `bind()`'s).

## Traps

* **Call `bind_kv_only` BEFORE `load()` if you want the buffer saving.** After `load()` the arena is already
  allocated; the mode still blocks drafting, but `draft_only_bytes()` reports what was *not* skipped.
* **Do not "fix" `draft()` returning false into returning true.** The false is the point: it is what makes a
  role mistake loud. The zeroed outputs are what makes it safe for a caller that does not check.
* `refuse_draft`'s zeroing span is `max_t_ - 1`, not `T`. `T` is the caller's window length and can be 1;
  `max_t_ - 1` is the span `draft()` itself clears and the span the engine allocates (`o.spec`).
* `Prefill::draft_kv` is still the fast path in this mode and is still **not bit-identical** to `mtp.prefill`
  (FP16/Q8_1 GEMM difference — pre-existing, documented at `include/strata/prefill/prefill.hpp:46-48`). For the
  handoff that is fine: the decode instance mounts what the prefill instance wrote, it never re-derives it. But
  it means the `STRATA_STATE_HASH` comparison above must be run with the **same** `STRATA_MTP_BATCH` setting on
  both sides, or the `mtp=` hashes will differ for an unrelated reason.
* `kv_only_` is read by `load()`, `bind()`, `bind_bytes()`, `set_draft_sampling()`, `record_forward()`,
  `capture_round()`, `capture_step()`, `draft()`, `draft_first()`. If a future path adds a new draft entry
  point, it must go through `refuse_draft`.
