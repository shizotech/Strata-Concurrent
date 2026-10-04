# Taskboard

## Stage 1 — shared_expert_arena  ✅ DONE
- [x] A1 load-once / reuse protocol (`src/core/pinned.cu`, `include/strata/core/pinned.hpp`,
      `src/core/expert_source.cpp`, `src/core/pinned_shared_test.cpp`)
- [x] A1b no redundant `cudaHostRegister`/`mlock` of shared pages
- [x] A1c parent: `ArenaExpertSource::borrowed()`, the startup line, the private-arena
      fallback, the tmpfs free-space check, the owner-reported-as-borrower bug
- [x] A2 CPU expert pool: cores partitioned across processes sharing the arena
      (`src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/pool.hpp`, `src/core/session.cpp`)
- [x] A2b report what was decided at startup (`pool.note()` at generate.cpp:2829)
- [x] A3 repo test for the source-level contract
      (`tests/core/file_expert_source_test.cpp::test_shared_arena_borrow/_pack_mismatch`)

## Stage 2 — more than one prefix  ✅ DONE
- [x] B1 `ConversationCache`: fixed slot count (default 8), LRU prune
      (`include/strata/core/conversation_cache.hpp`)
- [x] B2 engine options: on by default (8 GiB cap, sized to the machine), help text,
      INFO line reports the budget in force (`src/program/generate.cpp`)
- [x] B3 tests (`src/core/conversation_cache_test.cpp`, `src/program/conv_cache_test.cpp`)
- [x] B4 docs (`docs/DETAILS.md` "Multiple prompts at once", "Current limits"; README)

## Integration (parent)  ✅ DONE
- [x] `cd build && ninja` clean
- [x] CPU-only tests pass: pinned_shared_test, file_expert_source_test,
      conversation_cache_test (4225), conv_cache_test, conversation_memory_test (23)
- [x] Live cross-check of the core partition against the running server (read-only)
- [x] Regressions caught: `--mmap-experts` + default arena, `--layer-split` + default
      parking, Windows + default arena
- [x] `.megamind` + docs updated

## Stage 0 (new, owner asked) — pre-fill speed levers before stage 3  🔄 IN PROGRESS
- [x] S0.1 Read the live server log (1324 requests) and price prefill. Findings in
      `.megamind/prefill-levers.md`. Headline: **93 % of prefill wall time is in reads
      of ≤4k fresh tokens, dominated by a ~9 s per-request fixed cost**, not the
      per-token rate. Marginal ~590 tok/s; big reads reach ~490 tok/s.
- [x] S0.2 `prefill-cost-model` — turn the log into a reusable dataset + fitted cost
      model + an A/B script the owner can run at a restart. **No source edits.**
      Delivered: `bench/prefill/{analyze.py,README.md,ab-prefill.sh,baseline/}`.
      Reproduces the S0.1 headline numbers with `analyze.py --limit 1324`; two of
      them were wrong and are corrected in `.megamind/prefill-levers.md`.
      Findings for S0.3 in `.megamind/prefill-tool-notes.md`.
- [x] S0.3 `prefill-fixed-cost` — levers 1, 2, 4 landed; lever 3 deliberately left as config.
      New: `include/strata/program/prefill_loan.hpp` (pure decisions),
      `src/program/prefill_loan_test.cpp` (54 CPU checks, wired into `STRATA_BUILD_TESTS`),
      `bench/prefill/{fixed-cost-changes.md,park_backoff_replay.py}`.
      Edited: `src/program/generate.cpp` (serve loop only: `lend`/`refill_one`/`res_upload`/
      `park_current`), `CMakeLists.txt`.  `src/prefill/*` untouched.
      Switches: `STRATA_PREFILL_STICKY_LOAN`, `STRATA_RES_UPLOAD_ALWAYS`,
      `STRATA_PREFILL_LOAN_TIMING`, `STRATA_PARK_REFUSALS`, `STRATA_PARK_QUIET` (all in `--help`).
      Findings for S3.1 in `.megamind/prefill-fixed-cost-notes.md`.
      **NOT fixed: the 4.95 GiB/stage end-of-request refill — that is the floor, and removing it
      needs the loan kept across requests, i.e. stage 3.**

## Stage 3 — continuous batching / concurrent requests in one process  🔄 STARTED
- [x] S3.0 `stage3-design` — **DONE, docs only, no source touched.** Deliverable:
      **`docs/STAGE3-CONCURRENCY.md`** (873 lines, 133 file:line citations, all verified in range).
      Working notes: `.megamind/src/core/concurrency-notes.md`.
      Headline decisions: slot-level (NOT token-level) batching; **one live session + one captured
      graph set**, slots swapped through the existing `conversation_snapshot_save/restore`;
      `session_bytes` measured = **1.49 GB VRAM + 6.19 GB host per slot** at this box's config, against
      **457 MiB** free VRAM and a **0.0 GiB** parking budget → slots are a host-RAM question, not an
      allocation; one engine thread owns the GPU (the doorbell/pool epoch protocol forbids two window
      issuers); one `host_res`, one loan, one pool; protocol gains `READY … slots=N` + `#<id>` tags,
      default **off** so an old server/new engine pair is byte-identical on the wire.
      **Prerequisite S3.1a: parking across a `--layer-split`** — on this box parking is currently
       disabled by the split, so the swap path the whole design leans on does not exist yet.
       **(superseded: S3.1a landed, see below — the swap path exists now.)**
- [x] S3.1a parking across `--layer-split` (`src/core/conversation_state.cpp`,
      `include/strata/core/conversation_snapshot.hpp`, `conversation_cache.hpp`)
- [x] S3.1b slot object + state machine + `pick()` (`include/strata/program/slot.hpp`, 176 checks)
- [x] S3.1c id-tagged `--serve` wire, engine + `serve/server.py` (`serve_proto.hpp`, 140 checks)
- [x] S3.1d mount/unmount a slot = the swap (`serve_swap.hpp`, `swap_to()`, 105 checks)
- [x] S3.1e-1 the request body as resumable steps (pure refactor, 96 output calls unchanged)
- [x] S3.1e-2 the concurrent driver loop (`serve_driver.hpp`, 230 checks; serial path
      byte-identical, proven by diff)
- [x] S3.2b keep the prompt loan ACROSS requests — the 4.95 GiB/stage refill is GONE from the
      prompt path. Lazy return + a per-cache ledger + a bounded overlapped pump; the layout and the
      out-rows survive into the next request, so its `lend()` marks nothing and its
      `finish_prefill` copies nothing. Switches `STRATA_PREFILL_LAZY_LOAN` (default: on under the
      concurrent driver, off on the serial path), `STRATA_PREFILL_LOAN_PUMP_ROWS` (16).
      `prefill_loan.hpp` lever 5 + `prefill_loan_test` 54→131 checks, `serve_driver.hpp` `Loan`
      counters + `serve_driver_test` 230→245, `generate.cpp` loan code.
      Report: `bench/prefill/fixed-cost-changes.md` (S3.2b section). Agent notes:
      **`.megamind/src/program/s32b-lazy-loan-notes.md`**.
      **Owner action: measure it** — `STRATA_PREFILL_LOAN_TIMING=1`, then the four A/B arms in the
      report. Expect the batched-path fixed term `a = 10 460 ms` to fall to ~5–6.5 s.
- [x] S3.6 the parking-collapse fix + driver logging (found by the owner's live test)
      Symptom: a pre-empted slot resumed and ended prematurely, silently. Real cause was NOT the
      budget: `prep_request` clears `live_ok` for the whole life of a request, so `park_current`
      refused EVERY mid-request save at ANY budget - every swap in the log says `saved 0 B`.
      Fix: `publish_decode_branch` (makes a decoder parkable at all), `park_guard` (refuses a
      hand-over before it destroys anything; `budget_too_small` -> ERR with both numbers,
      `not_saveable` -> wait), `step_gate` (a decode slot with no conversation is ended, a prefill
      slot genuinely re-reads from token 0), startup `ParkCeiling` sizing at --max_context.
      Logging: `STRATA_SERVE_TRACE=1`, unconditional park-refusal line, resumed/re-reading events,
      periodic `activity:` line with `peak=`.
      serve_swap_test 156, serve_driver_test 303; both guards proven non-vacuous by mutation.
      Notes: **`.megamind/src/program/stage3-parking-collapse.md`**.
      Measured: a snapshot costs ~19.5 KiB per context token (175K tok = 3.26 GiB).
- [x] S3.7 "does each slot reserve --max-context?" — answered, and it led to the REAL bug
       (owner's question after the `conversation was lost while it was decode` ERR)
       **Answer:** parking was never a reservation (`conversation_snapshot_bytes` sizes at
       `live.size()`, the cache counts `SavedConversation::bytes()`): 81 tok = 226 MiB,
       95 303 tok = 2.4 GiB, the 524288-token ceiling = 7.6 GiB.
       **But the owner's error was NOT a budget error at all.** In the same log, budget 10 048 MiB,
       one entry parked at 113 MiB: `do_swap` looked the pre-empted slot's branch up with
       `ConversationCache::best()`, whose prefix rule requires the query to be STRICTLY LONGER than
       the entry - and a slot's own branch is exactly EQUAL to what it parked. Lookup -> 0, nothing
       mounted, adopt cleared the branch, `step_gate` ended a live decoder with an ERR blaming the
       budget. Fixed: `ConversationCache::best_exact()` (live branches, exact only) +
       `serve_swap::seek_mount_index()` / `handover_seeks()`, used by `do_swap`, with a named line when
       a decode slot's branch really is gone.
       **And the one thing that WAS a reservation:** RAM admission charged every request
       `budget / --conversation-cache-slots` (1131 MiB in the log) whatever its length, while
       `probe.ids.size()` sat unused. Now priced at `(prompt + max_new) x a per-token rate` -
       `bytes_per_token`/`SlotCost`/`slot_image_bytes`/`slot_cost_of` + `ParkCeiling::size_at/per_token`
       (rate precedence measured park > ceiling > guess), capped by ceiling and budget; GENI priced at
       the ceiling, not 0; an admit refusal prints prompt length, price, held, free RAM, floor, rate.
       **Also fixed:** `gate_end_reason` blamed "the parking budget is too small" for EVERY `Gate::end`
       including `!session_valid` (not a budget condition) and runs whose budget was ample. `GateEnd`
       carries the state + the two real numbers.
       Docs: `docs/DETAILS.md` "Nothing is reserved per slot" (measured table; ~112 MiB floor +
       15-26 KiB/token + ~110 MiB per checkpoint); `docs/STAGE3-CONCURRENCY.md` admission paragraph
       rewritten + S3.7 item; `--help` for `--conversation-cache-mib`.
       conversation_cache_test 4232→4252, serve_swap_test 156→169, serve_driver_test 303→340;
       five mutations prove the new checks non-vacuous (9/1/2/3/2 failures).
       Notes: **`.megamind/src/program/s37-slot-cost-notes.md`**.
       **Owner action: restart and re-run the two-client test. Expect `restored 119295080 B` instead
       of `restored 0 B` on the re-mount, and no `would step in decode … ending it` line.**
- [x] S3.8 the two-prefill livelock (found by the owner's live test)
       Symptom: "when two prefills happen at the same time and they swap, they seem to erase the
       other's previous progress, as after each swap they always start at 0 tokens again which makes
       them never finish." Log confirms: slot 2 (decode) parks 777 MiB and resumes; slot 3 (prefill)
       `saved 0 B, restored 0 B` on EVERY swap and `reset to token 0` **101 times**.
       Five-link chain: `publish_decode_branch` early-returns unless phase==decode -> `live_ok` stays
       false -> `park_current`'s first guard refuses (`saved 0 B`) -> `invalidate_unparked` ->
       `step_gate` -> `re_read` -> token 0. The LICENCE was `outgoing_for(prefill)` always answering
       `re_readable`, which permits swapping out without a save. The watchdog cannot see it: every
       pass really does read tokens.
       Fix (both halves needed): `outgoing_for(Phase, read_tokens)` -> a prefill that has consumed
       tokens is `must_park` (only cursor 0 stays `re_readable`); `publish_prefill_branch` puts
       `ids[0, at)` into `live`/`live_ok` before the hand-over (guarded on `R.cancelled`);
       `restore_published_branch` re-derives `R.at` from `live.size()` and re-plans `seg_i`.
       Safety net: `kMaxRereads=4` + `reread_allowed` + `reread_limit_line` -> a 5th reset ends the
       request with a named ERR instead of spinning forever.
       SEPARATE BUG on the same path: `arm_prompt_state` re-armed `cur` but NOT `req_imgs`, while
       `checkpoint_at` stamps `c.imgs = imgs_below(req_imgs, L)` -> a resumed prefill stamped another
       conversation's picture keys onto this branch's checkpoints. Silent; corrupts the cache's prefix
       comparison for every later request of that chat.
       Docs: `docs/DETAILS.md` ("A prompt read that is still in progress is parked too"),
       `docs/STAGE3-CONCURRENCY.md` (item 2 corrected, new item 6, new risk **R15**).
       serve_driver_test 340->377; four mutations non-vacuous (8 / 3 / 11 / 13+5 failures) - the
       end-to-end two-prefill simulation drives off the REAL predicates, so neutering either half
       of the fix breaks the simulation, not just a table.
       Notes: **`.megamind/src/program/s38-two-prefill-livelock.md`**.
       **Owner action: restart, two clients with long prompts. Expect `park: slot N parked T tokens`
       + `slot N resumed from T tokens` on the prefill swaps instead of `saved 0 B` + `reset to
       token 0`.**
- [x] S3.9 a parked branch was being STOLEN by another request (regression exposed by S3.8)
       Owner: "the parking doesnt seem to work now, as when a new request arrives it always kicks the
       currently running request out with an 'it was decode but not parked' error."
       NOT a save failure — the park SUCCEEDS. Log: `park: slot 14 parked 10394 tokens / 994 MiB`,
       then `restored 9849 tokens (checkpoint) … parked=4`, then `slot 14's parked branch (10394
       tokens) is no longer in the cache`, then the gate ends it. **`evictions=0` for the whole run**
       — nothing was ever pruned. 5 occurrences, all the same shape.
       Mechanism: a parked entry is ONE object (live branch + checkpoint chain + K/V). A new request
       looks a branch up by PREFIX (`best()`/`find()`), which matches any *checkpoint* of any entry,
       and the mount then `take()`s the WHOLE entry. Slot 15's prompt shares 9 849 of slot 14's
       10 394 tokens (same chat) -> request 15 mounted, and destroyed, slot 14's only copy.
       The cache had NO concept of ownership.
       Why now: S3.8 made mid-prefill slots park, so far more entries were owned by live slots. The
       bug was always there; S3.8 moved it onto the common path.
       Fix — a CLAIM, not a pin: `SavedConversation::owner` (kNoOwner = -1) = the request whose branch
       this is, while that request runs. `find`/`best`/`best_exact` take a `requester` and skip entries
       claimed by somebody else; `seek_mount_index` passes `s.id` so a slot always gets its own back.
       `put(image, owner, held)` records it (ONE signature with defaults — an overload pair would make
       `put(img, 5)` ambiguous between owner and held). `release_owner` runs from `drop_ctx`, the only
       thing that destroys a context, so every end path releases and a claim cannot leak.
       `return_mount_image` re-parks with the claim it came with. `victim_to_prune()` prefers UNCLAIMED
       entries (pruning a claimed one ends a request, not a re-read), falling back to plain LRU when
       all are claimed — the byte budget is still the hard limit.
       `park_owner_for(id)` decides the claim from `working_of(id)`: `finished` -> kNoOwner. That is
       exactly "does this slot still owe a step", and it keeps STAGE 2 working: a finished
       conversation's branch is parked unclaimed, so the next request of that chat still resumes from
       its prefix.
       TRAPS: a claim is not a pin (LRU may still evict; the owner gets the honest "pruned while it
       waited" line). Do NOT claim the `prep_request` park for `req_id` — that park saves the branch
       the SESSION holds, which belongs to the still-mounted slot. `kNoOwner` as requester hides
       nothing, so `--serve-slots 0/1` is untouched (serial proof byte-identical).
       Docs: `docs/DETAILS.md` ("A parked conversation belongs to the request that is still running"),
       `docs/STAGE3-CONCURRENCY.md` (new item 7 + risk **R16**).
       conversation_cache_test 4252->**4271**, serve_swap_test 169->**173**. Mutations: no claim filter
       -> 2+1 failures; `release_owner` neutered -> 3 (leak shows up as stage 2 breaking);
       `victim_to_prune` -> plain LRU -> 1.
       Notes: **`.megamind/src/program/s39-parked-branch-theft.md`**.
       **Owner action: restart. The `no longer in the cache` + `would step in decode` pair should be
       gone. Expect a slightly lower `N reused` on requests that share a prefix with a RUNNING slot —
       that is the fix working, not a regression.**
- [ ] S3.3b residency CPU test + `STRATA_RES_VERIFY` read-back
- [x] S3.4b `/metrics` slots rows + `/v1/status` concurrency (parent)
- [ ] S3.5 prefill/decode overlap (needs the owner's VRAM decision, OQ6) Split into the phases
      `docs/STAGE3-CONCURRENCY.md` §8 defines, each one agent run:
      * **S3.1a** ✅ DONE parking across a `--layer-split` (prerequisite — the swap path did not exist here).
             `SavedConversation::stage_parts` + `core::ConversationStageSet`; the split's parking guard is
             deleted; every `conversation_snapshot_*` has a stage-set overload. Contract + traps:
             **`.megamind/src/core/s31a-parking-split-notes.md`**. Owner action: a live park/mount on the
             3-way split at a restart (`STRATA_STATE_HASH`), which also closes OQ3.
       * **S3.1b** ✅ DONE `include/strata/program/slot.hpp` + `src/program/slot_test.cpp` (pure, CPU-only)
       * **S3.1c** ✅ DONE the `#<id>` protocol + `serve/server.py` routing (testable with `MockEngine`)
       * **S3.1d** ✅ DONE mount/unmount. New `include/strata/program/serve_swap.hpp` (the fixed order +
              the pure decisions) and `src/program/serve_swap_test.cpp` (105 CPU checks); `generate.cpp`
              gains `swap_to(id, serr, poisoned, restore_positions)` and the per-slot `SlotConv` records.
              `park_current` returns `Saved{stored,skipped,failed}`. Env switch **STRATA_NO_SWAP=1**;
              INFO gains `slot_swap=0|1`; every swap prints its bytes + ms on stderr and is counted
              (`Registry::note_swap`). Contract, traps and the owner's `STRATA_STATE_HASH` recipe:
              **`.megamind/src/program/s31d-swap-notes.md`**. **Owner action: measure a real swap at
              32k/128k/512k tokens — that closes OQ3.**
       * **S3.1e-1** ✅ DONE the pure refactor that precedes the scheduler: the serve request body is
              now named, resumable steps in `src/program/generate.cpp` — `prep_request` (phases 1+2),
              `plan_prompt_segments` + `run_prefill_step` (one prompt segment) + `finish_prefill`,
              `run_decode_step` (one verify window), `finish_request` (DONE + state + metrics), over a
              per-iteration `ReqCtx`. `enum class Step {progressed, finished, cancelled, needs_swap,
              error, fatal_exit}`; `needs_swap` is reserved for the scheduler. **No behaviour change,
              no loop change, no protocol change** — verified by a 96-vs-96 ordered output-call proof
              and per-phase statement diffs. API, call order and traps:
              **`.megamind/src/program/s31e1-step-extraction-notes.md`**.
       * **S3.1e-2** ✅ DONE **the concurrent driver loop itself.** New
              `include/strata/program/serve_driver.hpp` (the decisions: `dispatch_step`, the phase
              machine, admission + RAM arithmetic, `Loan` (R8), `SlotWatch`/`stalled_slots`/
              `watchdog_aborts` (R3), `bare_stop_target`, `parking_off_refuses_slots` (R2/R12)) and
              `src/program/serve_driver_test.cpp` (203 CPU checks, incl. five end-to-end simulations
              of the loop against a fake engine). `generate.cpp` gains `if (driver_on) { …; return; }`
              immediately **before** 0.1.30's `while (next_line(line))`, so the serial driver's text is
              byte-identical (`.shz_cmd/s31e2_serial_proof.py`) and its 96 serial-reachable output
              calls are unchanged (`.shz_cmd/s31e2_output_proof.py`). `ReqCtx` now lives in a
              `std::deque<ReqCtx> live` keyed by id, with `StepGuard`/`MropeScope` armed per step;
              per-slot cancellation, a loan owner + deferral queue, a per-active-slot heartbeat and an
              N-slot watchdog that aborts only when every watched slot stalled. Startup refuses
              `--serve-slots >= 2` when parking is off. API, lifetime decision, the 12 traps found and
              the owner's two-client recipe: **`.megamind/src/program/s31e2-driver-notes.md`**.
              **Owner action: two clients against `--serve-slots 3` at a restart** (also closes OQ3/OQ4).
      * **S3.2** the prompt loan as a process resource (kills the 4.95 GiB/stage refill) — **DONE.**
             S3.1e-2 gave it a named owner, a wait queue and one-holder-per-slot enforcement; **S3.2b**
             removed the refill: lazy return, a per-cache ledger, the layout and the out-rows surviving
             across requests and across slots, and a bounded pump beside decode. See the S3.2b entry
             above and `.megamind/src/program/s32b-lazy-loan-notes.md`.
      * **S3.3** residency between conversations   * **S3.4** `/slots` `/status` `/metrics`
      * **S3.5** prefill/decode overlap — **needs the owner's VRAM decision (OQ6)**
      Resolve the open questions OQ1-OQ7 (§9) before the phase that needs them.

## Sequencing / ownership (avoid two agents in one file)
```
S0.2 (no source)  ┐ parallel
S3.0 (docs only)  ┘
        ↓
S0.3 (generate.cpp)      ← must land before S3.1
        ↓
S3.1a (core snapshot + split)  →  S3.1b (new slot.hpp)  →  S3.1c (proto + server.py)
        ↓
S3.1d (core/generate.cpp)  →  S3.1e (generate.cpp)  →  S3.2 / S3.3 / S3.4
        ↓
S3.5 (needs the owner's VRAM decision)
```
New files keep `generate.cpp` edits to a call site — see risk R13.

## Integration (parent) — verified 2026-10-02
- [x] `cd build && ninja` clean; `/tmp/s3build` clean; ctest 18/19 (only expert_multi_test, no AVX-512)
- [x] slot_test 176 · serve_proto_test 140 · serve_swap_test 105 · serve_driver_test 245 ·
      prefill_loan_test 128 · conversation_cache_test 4232 · conv_cache_test · conversation_memory_test 23
- [x] serve/test_server.py 65 passed, 2 skipped (the 2 skips are a mock-fixture gap, documented in place)
- [x] Read and confirmed the S3.2b residency invariant myself (settle_pump marks resident only after the
      copy is confirmed landed; reconcile_residency at the top of every window path; adapt defers to led.owns)
- [x] Fixed a build break the S3.2b child left (missing LoanLedger::returned_eager())
- [x] docs/DETAILS.md + README updated for --serve-slots; removed false "engine 0.1.31" version claims

## Integration (parent) — after every child
- [ ] `cd build && ninja` clean
- [ ] CPU-only tests green: `pinned_shared_test`, `file_expert_source_test`,
      `conversation_cache_test`, `conv_cache_test`, `conversation_memory_test`
- [ ] Docs updated (`docs/DETAILS.md`, README) for anything that changes behaviour

## HARD CONSTRAINTS (unchanged, owner)
* **DO NOT START THE SERVER / ENGINE.** Two live servers own the box (78 GiB RAM,
  ~2 GiB available, 3× RTX 3090 at ~23.8/24 GiB). No `serve/server.py`, no
  `engine/strata --serve`, no model loading, no requests, no benchmarks that load a
  model. Source + build + small CPU-only tests only.
* `/dev/shm/shared_experts.dat` (46.8 GiB) belongs to live processes — never delete,
  truncate or write it. `/dev/shm/strata-core-leases` is live — read-only only.
* Stage 3 must not change the tokens a single request produces (bit-exactness against
  today's serial path is the acceptance bar).

## S3.10 — `--decode-tokens N` (owner's "1-3 tokens then switch is too low") — DONE
- [x] Cause found without a research tour: a decode STEP was one verify window, and a window commits
      `a+1` tokens (a = MTP drafts accepted, 1..8, usually 1-3). Every step boundary is a hand-over
      candidate, so the scheduler paid/evaluated a 237 MB-2.25 GB save+restore every ~2 tokens (R10).
- [x] `--decode-tokens N` (default 0 = today): a slot's decode turn runs windows until it has produced
      N tokens, then returns `progressed` and the scheduler may take the session. Token budget, not
      window count; overshoot allowed; first window always runs; EOS/max-new/STOP unchanged.
- [x] Pure predicates in `serve_driver.hpp` (`decode_step_tokens`, `decode_turn_done`) + the flag in
      `Options`/parse/usage/validation; `driver_line` prints `decode N tokens/turn`.
- [x] Verified: `build` ninja clean; `/tmp/a2build` all tests clean; serve_driver_test 377->**395**,
      serve_swap 173, slot 176, serve_proto 140, conv_cache PASS, prefill_loan 128; `--help` shows the
      flag; `--decode-tokens -5` and `> --max-context` both rc 2. Mutation proof: neutering
      `decode_turn_done` fails 3 checks.
- [x] Docs: `docs/DETAILS.md` (new paragraph + cross-ref), `docs/STAGE3-CONCURRENCY.md` (new §S3.10 +
      R10). Notes: `.megamind/src/program/s310-decode-tokens-notes.md`.
- [ ] **Owner action:** add `--decode-tokens 10` to the engine args and restart. The start-up line must
      read `decode 10 tokens/turn`, and `swaps=` / `swap_ms=` in the activity line should drop — that is
      OQ3/OQ4 measured for the first time.

## Deferred — DO NOT START without the owner
- Windows shared arena + cross-process core discovery.
- `setup.py`: per-model `--shared-expert-arena` filenames; HIP coverage note.
- The arena is loaded twice by the two live servers (28 `loaded 46.84 GiB` lines, 2
  borrows). Stage 1 works — the two *current* processes did borrow — but the log shows
  the older binary loading repeatedly before that. Worth a look if the owner reports
  it again after a restart.

---

# STAGE 4 — split prefill/decode across instances + real token-level batching (owner, 2026-10-04)

Brief and settled manager decisions: **`.megamind/stage4-plan.md`** (D1-D5).
Owner's ask verbatim in `prompt.md`. Target shape: one **prefill-only instance** (a backend, never
serves clients) + several **text-gen instances** that each batch several requests on the GPU.
Plus: **hold, don't reject** — a request with no resources waits instead of erroring.

## S4.2 hold-don't-reject: wait queue  ✅ DONE (engine + server)
- [x] pure API in `serve_driver.hpp`: `Hold{run_now,wait,error}`, `Wait{…,prefill reserved}`,
      `hold_for()` overloads (Refuse / ParkRefusal / SwapResult / Gate), `admit_decision(Admit&,Wait&)`,
      `WaitQueue` (FIFO + one priority, `--hold-ms` bound, cap 64, counters), `wait_line`,
      `hold_line`, `hold_admitted_line`, `hold_expired_line`, `watchdog_sees_waiters()==false`
- [x] **ENGINE WIRING in `src/program/generate.cpp` (done by the PARENT — five child agents crashed
      on this file, do not delegate it again):** `drv::WaitQueue holds(o.hold_ms)` + `hold_lines`,
      `ask_admission`/`AdmitInput` (delegates to the tested predicate), `hold_enqueue`,
      `promote_waiters`, `expire_waiters`, `hold_cancel`/`hold_expire`, `report_waits`,
      wake point in `drop_ctx` (`holds.wake_all()`), `waiting=N` in the activity line,
      `WAIT n oldest_ms reason` on the wire, STOP cancels a waiter (admit_one + pending_cancel at
      promotion). The old immediate RAM `refuse_line` is GONE — RAM is a wait.
- [x] `ConversationCache::probe()` (new, public const, side-effect-free prefix probe — `best()` would
      advance LRU stamps, and the queue re-asks every waiter). +11 checks: 4271->**4282**
- [x] server side `serve/server.py`: `HoldQueue`, `refusal_is_temporary` (default = temporary),
      `_gate_hold` (FIFO, cancel, `hold_ms` -> 503 `hold_expired` naming the reason), the bounded
      re-dispatch of a temporary engine refusal (`hold_retries`), `_wait_view()` = ONE computed source
      (server queue + the engine's `WAIT` line) feeding `/status`, `/slots` (aggregate `waiting` row),
      `/metrics`, `/v1/status`; `unload()` asks BUSY before CAPABILITY so a held request is "busy"
- [x] tests: serve_driver_test 395->**523**; `pytest serve/test_server.py` **81 passed, 2 skipped**
      (was 75/6 failed/2 skipped). The 6 failures were 4 fixture bugs (a mock script that starts with
      a stop id; `max_tokens` truncating the script; `out.append` vs `extend`; a gate under-held by
      one permit) and 2 real defects (`_hold_status`/`/status` not merging the engine's `WAIT` line;
      `unload()` answering "unsupported" while requests were held). No test deleted or weakened.
- [x] gates: `.shz_cmd/s42_wiring_proof.py` (structural, not textual — 2 mutations bite);
      `.shz_cmd/s31e2_serial_proof.py` byte-identical (3652 B, 56 lines);
      `.shz_cmd/s31e2_output_proof.py` PASS (109 serial-reachable calls, unchanged)
- [ ] **Owner action: restart and run 3 clients against `--serve-slots 2`.** Expect the third to be
      held, then `slot N ran after waiting X ms`, `waiting=` in the activity line, and no `ERR`.

## S4.0 Stage 4 design + contract doc  ✅ DONE (docs only, no source touched)
- [x] **`docs/STAGE4-SPLIT-ROLES.md`** (1693 lines) — the S4.3 contract. Agent notes:
      **`.megamind/src/program/s40-split-roles-notes.md`**. Pointer + the two D-refinements in
      `.megamind/stage4-plan.md`.
      Headline: `--role full|prefill|decode` (default `full`, byte-identical); the payload is
      `SavedConversation` as a **record stream** (nothing is memcpy-able — `ConversationBuffer` has no
      `data()`); `owner` **never crosses a process** (leaves as `kNoOwner`, the receiver claims at
      `put()`); the transport is a tmpfs **handoff arena**, one per prefill instance (one writer),
      4096-byte arena + slot headers with every offset `static_assert`ed, FREE→CLAIMED→READY→FREE,
      `flock` for decisions only, one ordered `pwrite` publish, `/proc` field-22 owner checks, and
      **leases** (the slot-leak fix); sizing is `conversation_snapshot_bytes(tier)` at start-up, never
      a RAM guess; the control protocol is a new `include/strata/program/prefill_svc.hpp` with
      `PREFILL-READY-V1`, 12 error codes each mapped to wait-or-error, and an 8-state machine; the
      decode engine thread **never blocks on a socket** (`Phase::await_prefill`); one new wait reason
      `prefill-busy`; S4.3 split into 9 steps with disjoint file ownership, 4 of them CPU-only.
      **Findings that changed the design:** the prefill role **cannot** skip the MTP drafter (the draft
      K/V is part of the payload and is produced by `mtp.prefill` from `sp.on_chunk`) — it needs a new
      `MtpDrafter::bind_kv_only()`; the prefill instance's **own `ConversationCache`** is the biggest
      win (the log is ~99 % reuse: `245751 reused + 195 read`), but it never holds what a decode
      instance generated → reverse handoff deferred to S4.5 (`TAKE` reserved); the prefill role claims
      **no cores** (`Prefill` uses `Stager`, not `ExpertPool`), so the decode instances keep the pool.
      Measured: `snapshot_bytes` over 871 parks — min 237 MB, p50 1.03 GB, p90 2.03 GB, **max 3.18 GB**
      (the stage-3 doc's "2.25 GB" understates the tail); 20.7-27.3 KiB/token above 100k tokens with a
      ~236 MiB fixed floor; park 3.18 GB in 410 ms, swap-save 2.79 GB in 220-313 ms.
      **Owner decisions needed before S4.3's defaults: OQ-S4-2** (arena bytes are tmpfs = RAM; the
      per-tier 6-slot default is ≈ 8.2 GiB, 22.8 GiB for a single 524288 tier, against 24 GiB free in
      `/dev/shm`) and
      **OQ-S4-9** (`bind_kv_only`, or the prefill role loads `NativeHead`'s 497 MiB for nothing).
      Full OQ list: `docs/STAGE4-SPLIT-ROLES.md` §9.

## S4.1 batched-decode feasibility: inventory + priced plan + minimal bit-exact step  ✅ DONE (docs only, no source touched)
- [x] owns `docs/STAGE4-BATCH-DECODE.md` (new, 937 lines) + `.megamind/src/kernels/batch-decode-notes.md`
      — **no source edits, no engine, no GPU, `/dev/shm` read-only.**
- [x] **The box is not what Stage 3 described.** `grep -c "layer split" strata-iq3_s.log` = **0**: both
      live engines (pid 12182 GPU 2 @524288, pid 12236 GPU 0 @262144) own **one whole GPU each**, and
      **GPU 1 is free** (2 396 MiB, all of it a whisper server). Stage 3's per-stage 0.554/0.560/0.547
      GiB figures are not what this box pays; a single-GPU session arena at 524288/int8/r32768 is
      **1.390 GiB**.
- [x] **Stage 3 §2.2's "524288 int8 resident 0 = 1.753 GiB" is wrong: measured 7.189 GiB** (nothing
      streams → all 12 int8 pools resident). Do not reuse it.
- [x] **The kernels are already batched where Stage 3 assumed they were not.** `qsa_decode_attn_batch`,
      `qsa_block_scores`, `qsa_block_topk`, `kv_stream_resolve`, `kv_append_*_step` all take `n_q` rows
      with **per-row step records and per-row selections**; the window's `h_step_`/`h_pos_` are already
      `T × …` arrays. "One sequence per window" is a **host policy in `Verifier::run`
      (`verify.cpp:964-969`)**, not a kernel limit.
- [x] **What is genuinely single-sequence:** the GDN recurrence + conv history
      (`verify_kernels.cu:115-181`, +112.2 MiB/sequence over 36 layers), the indexer key store
      (`idx_pooled`/`idx_tail`/`idx_dead`, indexed by cell with **no page table**, 128 B/token/layer),
      and **the 8-row ceiling, which is SIX independent ceilings** (`kVerifyMaxT`, `cpu::MAXT`,
      `GMAX` with a silent `min()`, `MAX_NCOLS`, the `[9]` graph arrays, `--spec`'s clamp) — and one
      sequence's drafts already consume all 8 rows at `--spec 4` (`o.spec = 6`, `generate.cpp:1661`).
- [x] **Priced (host-side scratch binary, validated to 0.1 MiB against the live log):** the `Verifier`
      arena is `≈ 42.7 + 5.69·T MiB` (76.8 predicted vs 76.9 printed at T=6). B=2 in one window:
      **+180 MiB VRAM** (GDN 112.2 + QSA per-seq 55.9 + arena 11.4) **+ 6.19 GiB pinned RAM** + six
      kernel contracts. A second *session* (no kernel change at all): **+1.47 GiB VRAM + 6.19 GiB RAM**.
- [x] **The payoff, computed from two clean log numbers** (T=1 window = 24 ms, `:106`; T=6 = 66 ms,
      `:8578`) ⇒ fixed 24 ms + **8.4 ms per extra row** ⇒ **B=2 is worth +11 % to +13 % decode
      throughput on this box, not +100 %**, because the rows that would have been drafts become
      second-sequence rows.
- [x] **The minimal bit-exact step:** two `SessionState` arenas + two `Verifier`s + one engine thread
      issuing one window per session (**option D2**) — *zero kernel changes, exact by construction*.
      Gate: `STRATA_STATE_HASH=1` (`generate.cpp:7369-7440`), S3.1d's recipe generalised (run A alone →
      hash; run A batched with B → hash; must match), plus a new `gdn_multi_parity` written **before**
      any GDN change.
- [x] **Staged plan S4.4.0…S4.4.6** with disjoint file ownership (R13) and a CPU-testable gate per step;
      `generate.cpp` appears in exactly one step.
- [x] **Verdict for the parent:** S4.3 first; then S4.4.0 (pure planner, free) and **S4.4.1 (two
      sessions)** as the measurement that decides whether the six-kernel route is funded. On this box
      **a second session on the free GPU 1 beats B=2 in one window on every axis except the owner's
      literal sentence.** Under `--kv k8v4` every option is unavailable (7.19 GiB int8 / 13.00 GiB fp16
      per session, `layer.cpp:557-562`) — refuse at start-up with both numbers.
- [x] **Owner decisions S4.1 needs:** (a) may the expert cache shrink by ~130 slots (B=2) or ~1 150
      slots (second session) — a *quality* decision at 85-94 % hit rate; (b) is +6.19 GiB of pinned RAM
      per extra sequence acceptable (the box has 5 GiB available now).
      Agent notes: **`.megamind/src/kernels/batch-decode-notes.md`**.

## S4.3.3 the prefill<->decode control protocol  ✅ DONE (new files only, CPU-only, no engine)
- [x] **`include/strata/program/prefill_svc.hpp` + `src/program/prefill_svc_test.cpp`** — the §4 protocol
       as pure decisions: the line grammar in both directions (one formatter + one parser per verb, so
       they cannot drift), `PREFILL-READY-V1` + the version/pack/geometry handshake refusal, the 14 error
       codes mapped onto `serve_driver::Hold` in one total table, the §4.5 handoff machine as **data**
       (13 states × 18 events = 234 rows, generated as a complete cross-product), the mirror
       prefill-instance job machine (9 × 9 = 81 rows), the nonce rules, CLAIM-before-DONE, cancellation
       in every state, the `--prefill-retries` bound, and the §4.2/§4.3/§4.7 log lines.
       Agent notes: **`.megamind/src/program/s433-prefill-protocol-notes.md`** (the full API for S4.3.5/.6).
       **`prefill_svc_test`: 2640 checks OK, ~0.05 s**, deterministic ×3, **zero I/O** — no file, no
       socket, no `/dev/shm`, no GPU, no model. `/dev/shm` still holds exactly `shared_experts.dat` and
       `strata-core-leases` afterwards. Built and run for real in `/tmp/s433build`
       (`-DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`); the parent's
       `EXISTS`-guarded block at `CMakeLists.txt:937-944` activated unchanged and was **not edited**
       (the target links nothing — the header is pure and `serve_driver.hpp` is header-only).
       **74 mutations, 74 bite** (`.shz_cmd/s433_mutations.py`): every framing byte class, the 4096-byte
       limit (the test proves the *parser* would accept an over-long line, so the framer is load-bearing),
       hex/NAME/ID/list typing, `ids=` required, HELLO/READY self-consistency, the ERR code vocabulary,
       free-text preservation, TAKE reserved, unknown-verb tolerance, every error-class flip, both wait
       strings, every handshake check, the tier rules, DONE/SEG-without-CLAIM, a second REQ,
       temporary↔permanent swaps, ACK before DONE, a second ACK, a copy-out with nothing published, a
       cancellable mount, retries from H_CANCELLED/H_SEG_READY, both nonce rules, the id and side guards,
       the retry bound (**neutering it runs the simulation to its pass cap without finishing** — the S3.8
       shape, in both the unit and the end-to-end arm), the backoff cap/growth, `may_send_req`,
       `may_release`, the poisoned flag, six job-table cells, the ids path/name rules, three log wordings.
       Coverage also includes all **twelve §4.7 failure modes as data-driven scenarios** against a
       by-value fake arena + prefill loop + decode client (prefill dies, decode dies mid-handoff → the
       READY lease reclaim with `flags|=abandoned` and no early steal, decode dies after ACK, the slot
       leak visible as `slots_free=2/4` then recovered, partial write, partial publish, corrupt payload
       (the slot still comes back — a refused payload must not become a leak), restore-fails-after-the-
       first-write → poisoned/no-retry, arena full + the §3.8 tier rules, tmpfs full, bad line, infinite
       retry), and a full end-to-end handoff where the slot is back to `6/6` **before** the mount.
       **No drift:** `prefill_svc.hpp` includes only 12 standard headers — nothing from `strata/program/`
       (so `serve_proto.hpp` cannot be pulled in) and nothing from `strata/core/`. The test includes
       `serve_driver.hpp` and asserts `Hold::run_now/wait/error == hold::0/1/2`, `hold_name`, and that
       `Wait::prefill` must be `engine_busy + 1`. Existing CPU tests re-run in the same tree, all green:
       `serve_proto_test` 140 · `serve_driver_test` 523 · `slot_test` 176 · `serve_swap_test` 173 ·
       `prefill_loan_test` 128 · `saved_conv_wire_test` 483 · `handoff_arena_test` 268.
       **Deviations from §4 the parent must pass to S4.3.5/.6** (also in the header's banner):
       (1) **13 states, not 8** — §4.5's `H_REQ_SENT` is split into `H_INIT`/`H_SENT`/`H_HOLD` and
       `H_FAILED`/`H_CANCELLED` added, because the diagram has six arrows into one state and §4.6 needs a
       per-state answer to CANCEL; no doc state renamed or removed; (2) `MOUNTING` is a state but
       **"session poisoned" is not** — it sets `poisoned()` and the caller must exit 1; (3) `H_QUEUED` has
       no handoff events and `H_DONE` is terminal; (4) `ERR cancel` ends the handoff and reports
       `cancel_outcome()` — this header never formats a client line; (5) `Hold::run_now` is never produced
       by an error code; (6) **`kWaitReasonPrefill`/`kWaitTextPrefill` are the authoritative spelling of
       `prefill-busy` until S4.3.6 adds `Wait::prefill`** (S4.3.3 may not edit `serve_driver.hpp`) — use
       them verbatim; (7) `ids=NAME` is validated (no `..`, no absolute on the wire) because it names a
       file the prefill instance opens; (8) an unknown **verb** is ignored, only a known verb with bad
       fields closes the connection; (9) `HELLO n_stages` = the later-stage count and must match `split=`;
       (10) the PING cookie is `[A-Za-z0-9_-]+`; (11) `ERR` is three events (temporary/permanent/cancel);
       (12) `PREFILL-READY-V1`'s fields after `proto_ver` are `k=v`, as §4.2 spells them; (13) the retry
       bound counts REQ lines (`--prefill-retries 3` ⇒ ≤4 REQs); (14) **`SEG` carries no nonce** (§4.3's
       grammar), so it is gated by state — progress only, never a copy-out trigger.
       **Trap for whoever edits the tables:** they are GENERATED (`.shz_cmd/s433_gen_tables.py` →
       `.shz_cmd/{handoff,job}_table.inc`, pasted into the header) so "total" is a property of the size,
       and the generator forces an `illegal` row to keep `to == from`. Edit the generator, not the table.
- [x] **CMake: the parent's block is correct and was NOT edited.** `ninja -C /tmp/s433build
       prefill_svc_test` builds and runs it; the target needs no libraries.

## S4.3.2 the tmpfs handoff arena  ✅ DONE (new files only, CPU-only, no engine)
- [x] **`include/strata/core/handoff.hpp` + `src/core/handoff.cpp` + `src/core/handoff_arena_test.cpp`** —
      the §3 arena: 4096-byte arena + slot headers with every offset `static_assert`ed and re-checked three
      ways in the test (doc == constant == `offsetof`), `FREE→CLAIMED→READY→FREE`, `flock` for decisions
      only, publish = one 48-byte metadata store while still CLAIMED + release fence + **one 8-byte store of
      `state`+`flags` as the publish point**, `/proc` field-22 dead/recycled-owner take-over, **leases**
      (the slot-leak fix: a READY slot whose reader died or nobody read is reclaimed, `flags|=abandoned`
      before `state=FREE`), `statvfs` before `ftruncate` with a fatal refusal naming both numbers and no
      file left behind, per-tier slot groups with the §3.8 smallest-fitting-group claim rule.
      Agent notes: **`.megamind/src/core/s432-handoff-arena-notes.md`** (the full API for S4.3.5/.6).
      Sizing is machine-independent: `slot_bytes = round_up_64(conversation_snapshot_bytes(tier)) + 64 MiB`,
      `arena_bytes = 4096 + Σ slots×(4096+slot_bytes)`; the printed line is
      `slots 2,2,2 (tiers 1024,16384,131072), max 131072 tokens, arena 7.99 GiB (8577380352 B)`.
      **`handoff_arena_test`: 268 checks OK, ~5.2 s**, deterministic ×3, its own
      `/dev/shm/strata-handoff-test-<pid>/` only — `/dev/shm` still holds exactly `shared_experts.dat`
      and `strata-core-leases` afterwards. Cross-process via the `--role` re-exec: `flock` contention,
      **two processes racing one slot with exactly one winner**, a child's payload read by the parent and
      by a second child reader, wrong-nonce/wrong-client readers refused, a live foreign CLAIMED slot not
      reclaimed, a claim that waits 1.5 s for a foreign flock; plus a **>1 GiB slot** (1 073 741 825 B)
      written through the real `pwrite` path and hashed back identically. 16 mutations all bite
      (no-claim-flock → 94, tier-cascade-inverted → 34, no-publish-block → 24, …).
      **Deviations from §3 the parent must pass to S4.3.5/.6** (also in the header's comment block):
      (1) §3.5 step 3's "same 32-byte pwrite" is impossible with §3.4's offsets — resolved as one 48-byte
      block then one 8-byte state store; (2) `client_id` is opaque, so the writer must call
      `note_client_pid(client_id, pid)` at HELLO for the immediate READY-lease reclaim; (3) `geom_hash` is
      computed by the caller from the 18-int `geometry_key` + `handoff_split_spec()` (that key is
      file-local to `conversation_state.cpp`, which S4.3.2 may not edit); (4) a restarted instance
      rebuilds a dead creator's slot table; (5) `open_writer(..., allow_existing_writer=false)` is a
      test-only escape from the one-writer rule and production must leave it false.
      **CMake: the parent's block is correct and was NOT edited** — verified by configuring with
      `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` and reading the generated link line.
      **Note for whoever configures the CPU tests:** `STRATA_ENABLE_CUDA` defaults **OFF**
      (`CMakeLists.txt:35`), and without it `strata_engine` — and so the whole S4.3 block — does not exist.

## S4.3.1 the SavedConversation payload codec  ✅ DONE (new files only, CPU-only, no engine)
- [x] **`include/strata/core/conversation_wire.hpp` + `src/core/conversation_wire.cpp` +
      `src/core/saved_conv_wire_test.cpp`** — the §2 payload: a 256-byte self-describing header record
      (§2.3's 232-byte field list `static_assert`ed, the 24 pad bytes used for the total + the hash), a fixed
      record stream (KV array → live → checkpoints → stage_parts → `END!`), and one 64-byte-aligned blob pool
      that every `ConversationBuffer`/checkpoint vector becomes a `{bytes, offset}` reference into.
      Agent notes: **`.megamind/src/core/s431-payload-codec-notes.md`** (the full API for S4.3.2/.5/.6).
      **`saved_conv_wire_test`: 483 checks OK, 0.23 s**, built and run for real in `/tmp/s431build`
      (`-G Ninja … -DSTRATA_BUILD_TESTS=ON -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`); the
      parent's `EXISTS`-guarded block at `CMakeLists.txt:901-916` activated unchanged and was **not edited**.
      Round trips: single-GPU, 2-, 3- and 4-carve splits, one-token, **empty conversation**, empty optional
      buffers, negative scalars, and a **>16 MiB buffer spanning segments** — each proven `encode → decode →
      re-encode` byte-identical with every field deep-compared, and the big buffer proven *streamed* (max
      single write == one 16 MiB segment, never staged). The `pwrite`/`pread` slot path (on a **memfd**, so
      nothing touches any filesystem) and the mapping path produce identical bytes and never write outside
      the slot. Refusals: truncation ×4, a slot that cannot serve its declared bytes, a chopped pool,
      version ×3, bad magic, **all 18 geometry slots**, both hashes naming both values, a declared size past
      the slot, corrupt/negative counts, a 40-deep `stage_parts` chain, missing/malformed END, wrong record
      kinds, record lengths that disagree with their contents, blob refs outside the pool or unaligned,
      non-zero reserved fields, the partial flag, unknown flag bits, one flipped byte failing the content
      hash, the stage-count rule **in both directions**, and a refused decode leaving the caller's image
      byte-for-byte untouched. **42 mutations, 41 bite** (the one that does not is a provably redundant
      inner bound).
      **Deviations from §2 the parent must pass to S4.3.2/.5/.6** (also in the header's comment block):
      (0) **§2.1's table is wrong** — `ConversationCheckpoint::gdn/ple/tails/dead/block_pos` are plain
      `std::vector<uint8_t>` (`conversation_cache.hpp:60`), not `ConversationBuffer`s; only `ConversationKv`
      holds buffers; (1) `owner` is **not on the wire at all** — decode always yields `kNoOwner`, the receiver
      claims at `put()`; (2) §2.3's 24 pad bytes carry `payload_bytes`/`payload_hash`/`reserved`;
      (3) `geom_hash`/`pack_hash` are **caller-supplied** u64 (`geometry_key` is file-local to
      `conversation_state.cpp`) and must be the same values the slot header carries; (4) **`payload_hash` is
      zero on the wire and travels out-of-band** in `WireEncoded`, so the codec's hash, the arena slot's
      `payload_hash` (§3.4) and `HandoffArena::payload_hash_of()` are **one identical number** — no
      self-referential field, no patch, one streaming pass; (5) END's kind *is* the ASCII magic `END!`;
      (6) blob alignment is 64 relative to the payload start (== the slot start, since slots begin at
      `slot_base+4096`); (7) `live.stage_parts` must be empty or encode refuses; (8) every count is bounded
      by the bytes actually available before it sizes an allocation, and `stage_parts` depth is capped at 8.
      **For S4.3.5:** call `conversation_wire_plan(image, plan, err)` to price `plan.total_bytes` against
      `tier_slot_bytes` **before** `claim()`; then `SlotWireSink(fd, claim.payload_off, claim.capacity)` +
      `conversation_wire_encode(...)`, and publish with `enc.payload_hash`.
      **For S4.3.6:** `SpanWireSource(payload.data, payload.payload_bytes)` + a `WireExpect` with the
      geometry, both hashes, the slot's `payload_hash`, `stage_parts = conv_stages.count` and
      `max_payload_bytes = tier_slot_bytes[tier]`; on success `put(std::move(image), req_id, …)`, then
      `conversation_snapshot_validate` (§2.5 step 3). A decode refusal is `ERR badpayload`, permanent.
- [x] **No-drift check:** `conversation_wire.hpp` is included by nothing but my own two files. Existing CPU
      tests re-run in the same tree, all green — `file_expert_source_test` PASS, `conv_cache_test` PASS,
      `prefill_loan_test` 128, `serve_swap_test` 173, `serve_proto_test` 140, `slot_test` 176.
      `git status` shows exactly three new untracked files for this step; no existing file modified.

## S4.3.4 `MtpDrafter::bind_kv_only`  ✅ DONE (edited 2 existing files, CPU-only, no engine)
- [x] **`include/strata/core/mtp.hpp` + `src/core/mtp.cpp`** — the KV-only drafter mode
      (`docs/STAGE4-SPLIT-ROLES.md` §1.4, OQ-S4-9 answered: **implement it**, do not eat the 578 MiB).
      New API: `bool bind_kv_only(const WeightTable& wt, std::string& err)`, `bool kv_only() const`,
      `static const char* draft_refusal_reason()`, `int64_t draft_refusals() const`,
      `uint64_t draft_only_bytes() const`. Either order with `load()` is legal.
      Agent notes: **`.megamind/src/core/s434-mtp-kv-only-notes.md`** (the API + the caller audit + the
      live-run recipe — **S4.3.5 must read it**).
      **Preserved byte-identically:** the whole prompt path — `prefill`, `capture_prefill`,
      `capture_prefill_dev`, `record_forward`'s `!full` branch, `kv_restore`, `first_needed`,
      `Prefill::draft_kv`. The draft K/V format does not move.
      **Skipped:** `head_logits_` (5.7 MiB), `dhead_`+`dvocab_` (81.3 MiB, log:35), `setup_coupled`,
      `window_R_`, `head_` → the prefill role no longer needs `NativeHead` (**497 MiB**, log:19),
      all 36 round/step graph execs, the draft-only device buffers, `h_row_/h_out_/h_prob_` staging,
      the `ident_` upload, and `bind_bytes()` → **0** so `--expert-cache auto` keeps the reserve.
      **Bytes saved, computed from the code** (`load()` prices both carves; scratch replay
      `/tmp/s434scratch/carve_parity.cpp`): live config (`--spec 4`→max_t 6, `--mtp-window 32768`)
      **15 469 568 B = 14.8 MiB**; `--spec 8` with no window **112 503 296 B = 107.3 MiB** (`ident_`
      dominates); inside the drafter **101.7 MiB** total, +497 MiB `NativeHead` unlocked.
      ⚠ **The buffer saving needs `bind_kv_only` called BEFORE `load()`** (`generate.cpp:2519`); the
      one-line swap at `generate.cpp:4015` alone still blocks drafting but keeps the arena.
      **Four gates, all mutation-tested:** `draft()` `mtp.cpp:893` and `draft_first()` `:951` →
      `refuse_draft()` `:507` (zeroes `drafts`/`probs` over `max_t_-1`, `*n_drafts = 0`, names the reason,
      returns **false** — loud, not silent); `capture_round` `:729` / `capture_step` `:764` refuse;
      `record_forward` `:526` refuses the `full` branch **before** dereferencing `g_`/`ss_`. Plus
      `bind()` refuses to upgrade a KV-only drafter, `bind_kv_only()` refuses to downgrade a bound one,
      `set_draft_sampling` can never activate coupling, and `prefill()` refuses an unbound drafter
      (`:796`) instead of null-dereferencing.
      **Caller audit:** `src/core/verify.cpp` has **zero** drafter callers (grep) — the drafter and the
      verifier meet only through `bind`'s `window_R_`, which this mode never has. Every `generate.cpp`
      site is tabulated in the notes; **no `generate.cpp` edit is required for correctness** — the
      `mtp.draft(...)` sites (`:7281`, `:9747`) already check the return and end the request with the
      named reason. S4.3.5's only required edit is `:4015` `mtp.bind(...)` → `mtp.bind_kv_only(<same wt>, err)`.
      **Verified:** `cd build && ninja` clean; `conversation_snapshot_test` **2089** and
      `conversation_validation_test` **1323** pass **UNCHANGED** in `/tmp/s434build` (CUDA ON, arch 86,
      both test flags ON); no-drift across 13 other CPU tests in the same tree (483/268/2640/523/173/140/
      176/128/PASS/PASS/4282/23/PASS). **No new CMake target** (the parent owns `CMakeLists.txt`; there is
      no existing mtp test target) — the CPU-only gate check is scratch at `/tmp/s434scratch/kv_only_gate.cpp`
      (33 checks, links the real `libstrata_engine.a`, no GPU/model/`session_init`/`/dev/shm`).
      **Non-vacuity:** mode OFF, `draft()` reaches `"mtp: begin capture"` (so the guard, not the missing
      device, is what stops it); mode ON, 0 candidates + zeroed outputs + all 36 exec slots null.
      Mutations: neuter `draft()`'s guard → **4 checks fail**; neuter the `record_forward`/`capture_*`
      gates → the binary **segfaults (139)**, the exact failure the gates prevent. Both reverted.
- [ ] **Owner action (the only thing left for this step):** the `STRATA_STATE_HASH` comparison —
      `mtp=` (FNV-1a over the drafter's K/V pools, `generate.cpp:7463-7475`) at the same context length
      must be **identical** between a `--role full` run and a `prefill`+`decode` handoff of the same prompt,
      compared at the first hash line after the prompt. Same `STRATA_MTP_BATCH` setting on both sides.
      `mtp.draft_refusals()` must stay **0** in the prefill instance; S4.3.5 should print it and
      `draft_only_bytes()` at start-up. Expected new lines: `strata mtp: KV-ONLY mode (S4.3.4): …` and
      `strata mtp: bound KV-ONLY: …`, and **no** `draft head over 40525 tokens` line.

## S4.3 the prefill/decode split — 4 of 9 steps DONE
- [x] **S4.3.1 payload codec** — `include/strata/core/conversation_wire.hpp`,
      `src/core/conversation_wire.cpp`, `src/core/saved_conv_wire_test.cpp` — **483 checks**,
      42 mutations (41 bite). API + call sequences: `.megamind/src/core/s431-payload-codec-notes.md`
- [x] **S4.3.2 handoff arena** — `include/strata/core/handoff.hpp`, `src/core/handoff.cpp`,
      `src/core/handoff_arena_test.cpp` — **268 checks**, 16 mutations (all bite). Machine-independent
      sizing (`conversation_snapshot_bytes(tier)` + 64 MiB/slot), leases, dead/recycled-owner take-over.
      Notes: `.megamind/src/core/s432-handoff-arena-notes.md`
- [x] **S4.3.3 control protocol** — `include/strata/program/prefill_svc.hpp`,
      `src/program/prefill_svc_test.cpp` — **2640 checks**, 74 mutations (all bite). 13-state decode
      machine + 9-state prefill machine, total tables (234 + 81 rows), 14 error codes partitioned
      7 permanent / 7 waits, `PREFILL-READY-V1`. `prefill-busy` strings are AUTHORITATIVE here
      (DEV-6) until S4.3.6 adds `Wait::prefill`. Notes: `.megamind/src/program/s433-prefill-protocol-notes.md`
- [x] **S4.3.4 `MtpDrafter::bind_kv_only`** — `include/strata/core/mtp.hpp`, `src/core/mtp.cpp`.
      Draft K/V byte-identical; drafting/graphs/head refused 4 ways deep. Saves 101.7 MiB inside the
      drafter + unlocks dropping `NativeHead`'s 497 MiB (answers OQ-S4-9). `conversation_snapshot_test`
      2089 and `conversation_validation_test` 1323 pass UNCHANGED. Gate: `.shz_cmd/s434_kv_only_gate.cpp`.
      **S4.3.5's one required edit: `generate.cpp:4015` -> `mtp.bind_kv_only(...)`; call it BEFORE
      `mtp.load()` at `:2519` to also collect the 14.8 MiB.** Notes: `.megamind/src/core/s434-mtp-kv-only-notes.md`
- [x] **CMakeLists.txt is owned by the PARENT for all of stage 4.** Three `EXISTS`-guarded blocks
      (wire / handoff / prefill_svc) are already in place and activate when sources land. Children must
      NOT edit it. **Configure trap: `-DSTRATA_ENABLE_CUDA=ON` is required** — with it off there is no
      `strata_engine` target, so the `strata` binary AND the whole S4.3 test block are silently absent.
      That is why several children never compiled `generate.cpp`.
- [ ] **S4.3.5** the prefill instance (role plumbing, job queue, socket server, arena writer) —
      READY TO START: needs 1+2+3+4, all landed. Owns `src/program/prefill_svc_server.cpp` (new),
      `generate.cpp` role gates at `:3748`, `:3944`, `:4014`, `:5220`, `:2519`.
- [ ] **S4.3.6** the decode instance (`Phase::await_prefill`, the client thread, the mount) — after S4.3.5.
- [ ] **S4.3.7** the `prefill-busy` reason through `/status`, `/slots`, `/metrics` — after S4.2 ✅ (unblocked).
- [ ] **S4.3.8** operator surface (`setup.py`, `--help`, docs). **S4.3.9** integration + owner runbook.

## Queued (do not start yet)
- **S4.3** the prefill-as-a-service split — **IN PROGRESS, 4/9 steps done**; S4.3.5 is unblocked now.
- **S4.4** batched decode — S4.1's VERDICT: achievable but expensive. B=2 in one window costs
  +180 MiB VRAM + 6.19 GiB pinned RAM + six kernel contracts for only **+11-13 %**. A SECOND SESSION
  ARENA costs +1.47 GiB VRAM, ZERO kernel changes, and is bit-exact by construction. Recommendation:
  S4.3 first, then S4.4.0+S4.4.1 as the measurement that decides whether the kernel route is funded.
  Two owner decisions gate it: may the expert cache shrink (~130 slots for B=2), and is +6.19 GiB
  pinned RAM per extra sequence acceptable?

## S4.6 the marginal-row slope, MEASURED  ✅ DONE (offline) / 🔄 owner run pending
Owner challenged S4.1's "+11-13 % for B=2". It was measured instead of interpolated, and the answer
reframes the whole batching decision.
- [x] `bench/decode-slope/analyze.py` (2022 lines) — parses the live log, derives per-window rows from
      the identity `ΣT = drafts_offered + windows`, keeps only requests where `generated == accepted +
      windows` exactly, fits `a + b_gpu*hits + b_cpu*misses` with bootstrap CIs. **Read-only, no engine.**
- [x] `bench/decode-slope/selftest.py` — recovers a planted ground truth (`a=12.00, b_gpu=5.60,
      b_cpu=37.70/miss-row`), so the decomposition is not an artifact of the derivation.
- [x] `bench/decode-slope/slope-ab.sh` (418 lines, `bash -n` clean) — the owner's A/B ladder. Refuses to
      act without `--run`; **never executed here** (starting an engine is forbidden).
- [x] `bench/decode-slope/README.md` — owner-facing: what is safe to run, what each arm proves.
- [x] `docs/STAGE4-BATCH-DECODE.md` **§9** appended (supersedes §3.5 and §4.2's headline) + the priced
      ceilings table. Notes: `.megamind/src/kernels/s45-decode-slope-notes.md`,
      `.megamind/src/kernels/s45-ceilings-table.md`.

**THE CORRECTED COST MODEL (fitted, not interpolated):**
```
a     = 12.4 ms fixed/window
b_gpu = -0.016 ms/row   ~0: a VRAM-resident expert row is FREE at this resolution
b_cpu = +0.0601 ms per CPU-pool expert entry  CI [+0.0575,+0.0628]  (median 205/window = 12.3 ms)
b     = 10.89 ms/row at 86.4% hit   |   5.19 ms/row at 100% hit (REACHABLE)   |   0.74 floor (HBM)
```
**The row cost is the CPU expert pool, not GPU arithmetic and not launch overhead.** So the lever that
flattens the slope is **VRAM for the expert cache**, not a kernel rewrite. B=8 at full cache = **+49 %**,
B=16 = **+54 %** (vs +24 % / +26 % measured today). S4.1's +11-13 % was right for this box as configured
and wrong as a ceiling.
**Also corrected:** §1.2's "six 8-row ceilings" is **thirteen** — seven more hard limits were found,
three silent (`kFusedGrMaxT`, the `n_tok > 8` guards in `native_bf16.cu:139` / `shared_expert.cu:172`,
`kMaxWindowEntries=128` with `static_assert(cpu::MAXT*10 <= 128)`, `CAP`, `groups_[9]`, `last_tokens_[8]`).
`GMAX` at 24 does not cost memory, it **stops the build** (`ptxas: 0x10e00 > 0xc000`, measured with
`nvcc -Xptxas -v`). Raising one and missing the rest is risk **B1**.
- [ ] **Owner action:** stop the servers, then `bench/decode-slope/slope-ab.sh --run`. The decisive arm
      is `cache3k` (raises the miss rate at fixed `T`): if `b` rises with the miss rate and not with
      `T`, the slope is confirmed to be the CPU pool and the fix is VRAM, not kernels.

## Sequencing
```
S4.0 (design)  ∥  S4.1 (batch feasibility)  ∥  S4.2 (wait queue)
        \              |                          /
         +----- S4.3 (split roles + transport) <-+
                        |
                 S4.4 (batched decode kernels)  <- S4.1
```
