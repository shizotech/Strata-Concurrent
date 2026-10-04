# Roadmap — Strata: shared expert arena + multi-prefix cache (stages 1 & 2)

**STATUS: stages 1, 2 and 3 implemented, built and CPU-tested. Stage 0 (pre-fill levers, the owner's
extra ask) also done. Stage 3 needs the owner's restart to be measured and accepted — see
"Stage 3 — what is implemented" below. Nothing here was ever run against a live model.**

User objective (`prompt.md`): the engine served one request at a time and
`shared_expert_arena` re-loaded the experts per process and collapsed when two
servers ran concurrently. Stage 1 = fix `shared_expert_arena`; stage 2 = prefix
caching for more than one prefix (~8 caches, prune the least recently used first).
**Stop before stage 3** (continuous batching / concurrent requests).

## HARD CONSTRAINTS (owner, 2026-10-01)

* **DO NOT START THE SERVER / ENGINE.** Not enough free RAM. No `serve/server.py`,
  no `engine/strata --serve`, no model loading, no requests. Source + build +
  small CPU-only tests only.
* Build check: `cd build && ninja` (Release, CUDA arch 86, tests OFF).
* `/dev/shm/shared_experts.dat` (46.8 GiB) belongs to a live process — never
  delete or truncate it.

## Environment

* Repo `/ssd/Strata`, engine 0.1.30, C++20, Linux, Ryzen 5 5600X (6c/12t),
  `ulimit -l` = 8 MiB, IQ3_S native pack, arena total 46.84 GiB.
* `build/` configured + clean. `/tmp/a2build` is a scratch build with
  `STRATA_BUILD_TESTS=ON` (A2 used it; `pool_test`/`pool_stress` SKIP on this box
  because Zen 3 has no AVX-512 — pre-existing, not a regression).
* A live server (pid 1017331) pins host→CPU 0 and workers→CPUs 1-5.

## What each workstream delivered

### A1 — the arena is loaded once (pinned.cu / pinned.hpp / expert_source.cpp)
* 4 KiB header protocol in the existing `reserved[4]`: `state`, `owner`,
  `owner_start`, `nonce`. `magic`/`version=1`/`header_bytes=4096` unchanged, so an
  old engine still reads a new header; an old file (reserved all zero) = NOT READY.
* `flock`-guarded decision: claim → load → `publish_shared_load()` (release fence +
  `state=1`). A live owner's in-progress load is WAITED for
  (`STRATA_SHARED_ARENA_WAIT_MS`, default 1200000); a dead or pid-recycled owner
  (checked via `/proc/<pid>/stat` field 22) is taken over. A displaced owner's
  publish is refused by the nonce check.
* `ArenaExpertSource::open()` loads nothing when `shared_borrowed`, and publishes
  after an owner load. `mlock`/`lock_resident` skipped for shared (tmpfs) pages.
* New public: `PinnedArena::{shared_borrowed, shared_load_owner, shared_pack_hash,
  shared_load_nonce, shared_owner_pid, shared_waited_ms, publish_shared_load()}` and
  the header offset constants.
* Parent added: `ArenaExpertSource::borrowed()`, the "borrowed … nothing loaded"
  startup line, and the **fallback to a private arena** when the shared backing is
  unusable (Docker's 64 MiB /dev/shm, unwritable path, another pack's file).
* Parent added: **free-space check before `ftruncate`** — tmpfs `ftruncate` is
  sparse, so a too-small `/dev/shm` would otherwise SIGBUS mid-load with no message.
* Parent fixed a real bug in A1's reporting: `publish_shared_load()` flips
  `shared_borrowed=true`/`shared_load_owner=false`, so reading those flags after the
  load reported the OWNER as a borrower ("nothing loaded" for the process that just
  spent a minute loading). The role is now snapshotted before the load, and the note
  becomes "SHARED ARENA PUBLISHED, other processes can borrow it".

### A2 — the concurrency collapse was CPU affinity, not a lock (pool.cpp / pool.hpp / session.cpp)
* **Answer to the owner's question: there is no unnecessary lock, and tmpfs sharing
  is not the cause.** Every Strata process pinned host→CPU 0 and workers→CPUs 1-5,
  so two processes put 12 hard-pinned runnable threads on 6 logical CPUs while the
  SMT siblings 6-11 sat idle; the pool's park/`wait_done` loops are `_mm_pause`
  spins, so a preempted spinner burns a whole timeslice. That is the 400 %/80 % split.
* Measured (synthetic, real `ExpertPool`, no GPU): disjoint cores 41k/41k ops/s vs
  colliding cores 16k/16k — **2.5-3x per process**, p99 barrier latency 1031 µs → 28 µs.
* `claim_cores()` / `core_plan()` / `CorePlan` / `ExpertPool::note()`: a machine-wide
  lease (`/dev/shm/strata-core-leases`, flock-guarded, validated by pid + start time)
  **plus** a `/proc` scan of other `strata*` processes' per-thread CPU masks — the
  scan is what makes it work against a server running an older binary.
* Park spin drops 20 ms → 200 µs **only** when cores are genuinely oversubscribed.
* Env: `STRATA_POOL_CORES=list|none`, `STRATA_POOL_LEASE=path|0`, `STRATA_POOL_SCAN=0`.
* Verified live on this box: the running server holds 0-5, a new process claimed
  host 6 + workers 7-11 and said why.
* Bug A2 found and fixed: `"Cpus_allowed_list:"` is **18** chars, not 16.

### B — 8 prefixes, pruned least-recently-used (conversation_cache.hpp + tests + docs)
* `ConversationCache::default_slots = 8`, per-entry use stamps from a monotonic
  clock; `best()` (a hit), `take()` (a mount) and `put()` advance them;
  `make_room()` evicts `lru_victim()` instead of `pop_front()`.
* New: `slots()`, `budget()`, `clock()`, `use(i)`, `touch(i)`, `lru_victim()`.
  **`best()` is no longer `const`** — do not introduce a `const ConversationCache&`.
* Deliberately no pinned entry at this level (parked conversations are independent
  branches); the pinned root stays in `program/conv_cache.hpp`, which now shares the
  same `lru_victim()` helper, and a test asserts the two agree.
* Parent: `Options::conversation_cache_slots` = `default_slots` (was 4),
  `conversation_cache_mib` default 8192 (was 0 = off), and the **default budget is
  capped to the machine** (free RAM − 2.5 GiB floor − 4 GiB request headroom) so
  parking cannot evict the expert/PLE file cache — the review's C1 trap. An explicit
  `--conversation-cache-mib` is never second-guessed. The INFO line reports the
  budget actually in force via `conversations.budget()`.

## Parent integration fixes (regressions caught before they shipped)

1. `--mmap-experts` + the now-default shared arena was a **hard `return 2`**, which
   would have stopped every low-RAM install (`setup.py` passes `--resident-experts`,
   which implies `--mmap-experts`). A default is now dropped when there is no
   resident arena to share; only an explicit `--shared-expert-arena` +
   `--mmap-experts` is still an error.
2. Parking + `--layer-split` was a hard `return 2`; with parking on by default that
   would have broken every multi-GPU install. A split now disables parking and says
   so; an explicit budget + split is still an error.
3. The `shared_expert_arena` default is `#ifdef _WIN32`-guarded — the Windows branch
   of `reserve()` refuses a shared backing, so an unconditional default would have
   made Windows unstartable.

## Verification (all run, all green)

* `cd build && ninja` → clean (the `/usr/local/bin/ld: error in ...sframe` lines are
  pre-existing binutils noise on this box, present before any change).
* `./build/pinned_shared_test` → OK (owner/borrower sentinel, unpublished never
  borrowed, dead + recycled owner take-over, old-v1 file, too-small filesystem,
  pack-hash and size refusals, and a real **cross-process** case via `--role`).
* `./build/file_expert_source_test` → PASS (extended with `test_shared_arena_borrow`
  and `test_shared_arena_pack_mismatch`).
* `conversation_cache_test` 4225 checks, `conv_cache_test` PASS,
  `conversation_memory_test` 23 checks.
* Live cross-check of the core partition against the running server (read-only).
* No server, no model, no `/dev/shm/shared_experts.dat` touched.

## Stage 0 — pre-fill fixed cost before stage 3 (owner asked)

* **S0.1/S0.2 (offline, no source):** the cost model. `prompt_ms ≈ 9 547 + 2.72 * fresh`
  over 1 600 live requests; **54.6 % of prompt wall time is in reads of ≤4 k fresh**, and
  the fixed term is **context-independent** (a 500-token read costs ~10 s at 5 k and at
  250 k context). Tool: `bench/prefill/analyze.py`, baseline in `bench/prefill/baseline/`.
  Two claims from the first reading were wrong and are corrected in `prefill-levers.md`.
* **S0.3 (code, landed):** levers 1, 2, 4 in the serve loop. Pure decisions in
  `include/strata/program/prefill_loan.hpp` + `src/program/prefill_loan_test.cpp` (54
  CPU checks); engine edits confined to `src/program/generate.cpp`'s
  `lend`/`refill_one`/`res_upload`/`park_current`. `src/prefill/*` untouched.
  Report: `bench/prefill/fixed-cost-changes.md`; agent notes:
  `.megamind/prefill-fixed-cost-notes.md`.
  Lever 3 (price the chunk) deliberately **not** implemented: `lend()` already sizes the
  loan to the segment, so a smaller `--prefill` moves the *marginal rate*, not the floor,
  and the log cannot price it. That is an owner config call via `ab-prefill.sh`.
* **The honest limit, and the stage-3 argument:** the 4.95 GiB/stage refill at the end of
  every request is still there. The contract forbids changing what is refilled or when,
  and it is already minimal. Removing it means keeping the loan across requests, which
  needs concurrency. `STRATA_PREFILL_LOAN_TIMING=1` prints the refill cost per stage per
  request so stage 3 can be priced before and after.

## Stage 3 — implemented, CPU-verified, NOT yet run against a live model (S3.0-S3.2b, 2026-10-02)

`docs/STAGE3-CONCURRENCY.md` is the design and the contract. Landed: S3.1a (parking across
`--layer-split`), S3.1b (`slot.hpp`), S3.1c (`serve_proto.hpp` + `serve/server.py` routing),
S3.1d (`serve_swap.hpp`, `swap_to()`), S3.1e-1 (the request body as resumable steps), S3.1e-2
(`serve_driver.hpp`, the concurrent driver), S3.2b (lazy/overlapped prompt-loan return), S3.4b
(`/metrics` slot rows, `/v1/status` concurrency). Still open: S3.3b residency test +
`STRATA_RES_VERIFY`, S3.5 prefill/decode overlap (needs the owner's VRAM call), and every
measurement (OQ3/OQ4) — the engine was never started here.

The decisions that are already settled and must not be re-litigated:

* **Slot-level batching, not token-level.** Two conversations never share one verify window: the
  window is T tokens at *consecutive* positions (`verify.hpp:3-8`) and the kernels bound
  `n_tok ≤ kVerifyMaxT = 8`. llama.cpp's shared `llama_batch` is a stage-4 engine here, not stage 3.
* **One live session, one captured graph set, slots swapped.** `Verifier::exec_[9]` + the
  `MtpDrafter` graph arrays bake in one `SessionState`. Re-capturing per slot needs a second
  verifier arena (76.9 MiB) + a second MTP set against **457 MiB** of free VRAM: refused. Indirection
  is a kernel rewrite: refused. The swap is `conversation_snapshot_save/restore`, which already
  exists and already validates before it writes.
* **Why the swap is cheap on this box:** `--kv-resident` keeps the authoritative K/V in **host** RAM
  (`kv_stream.hpp:3-16`), so a slot switch is host→host memcpy + `kv_stream_reset`, not a VRAM copy.
  That trick does **not** exist for `--kv k8v4` (it refuses streaming, `layer.cpp:557-562`) — OQ1.
* **Measured, not guessed:** `session_bytes` at this box's config = **1.49 GB VRAM** per full session
  (13.00 GiB if KV is fully resident at 524288/fp16), + **6.19 GiB host** KV copy, + 35 MB draft ring,
  + ~66 MB suffix drafter. Snapshots on this box run 237 MB … 2.25 GB.
* **One engine thread owns the GPU.** The doorbell/pool epoch handshake forbids two `Verifier::run`
  callers (issue #29 in a new costume). Prefill/decode overlap is phase S3.5 and needs a second
  session arena — the owner's VRAM call (OQ6).
* **Prerequisite S3.1a — DONE.** Parking now covers a `--layer-split`: `SavedConversation::stage_parts`
  carries every later stage's running state and K/V, the MTP draft state is saved on the drafter's own
  device, and the "a split disables parking" guard is gone. The swap path S3.1d/e lean on exists.
  Contract and traps: `.megamind/src/core/s31a-parking-split-notes.md`.
* **S3.1d — DONE: the swap itself.** `serve_swap::run()` executes the fixed order
  (drain residency → return the loan → validate → save → restore → draft ring → adopt → device state)
  through engine hooks, so the order is CPU-testable; `generate.cpp`'s `swap_to()` is what S3.1e calls.
  A failed validation costs nothing because validation runs before the save; a failed restore stays
  fatal. A conversation that could not be parked is marked un-resumable rather than mounted on stale
  K/V. `STRATA_NO_SWAP=1` falls back to serial; every swap reports bytes + ms and is counted (OQ3).
  API, decisions and traps: `.megamind/src/program/s31d-swap-notes.md`.
* **S3.1e-1 — DONE: the request body is now resumable steps.** A pure, behaviour-preserving refactor
  of `generate.cpp`'s serve body into `prep_request` / `plan_prompt_segments` + `run_prefill_step` +
  `finish_prefill` / `run_decode_step` / `finish_request` over a per-iteration `ReqCtx`, with
  `enum class Step {progressed, finished, cancelled, needs_swap, error, fatal_exit}`. The serial loop
  calls them in 0.1.30's order, so the serial path is unchanged by construction. Proven offline: 96
  output calls before and after in the same order, and per-phase statement diffs whose only changes
  are aliases and exit rewrites. API + traps: `.megamind/src/program/s31e1-step-extraction-notes.md`.
* **S3.1e-2 — DONE: the concurrent driver loop.** `--serve-slots >= 2` (with a conversation cache and
  without `STRATA_NO_SWAP=1`) now runs N conversations inside one process: the serve loop admits
  requests into the slot registry, calls `Registry::pick()`, hands the session over with S3.1d's
  `swap_to(..., restore_positions=true)`, and runs **one** step (`run_prefill_step` = a prompt
  segment, `run_decode_step` = a verify window) per iteration. New
  `include/strata/program/serve_driver.hpp` owns the decisions (step dispatch, the phase machine,
  admission + RAM arithmetic, the prompt loan's owner, the N-slot watchdog, bare-`STOP` routing);
  `src/program/serve_driver_test.cpp` pins them with 230 CPU checks including six end-to-end
  simulations of the loop against a fake engine. `ReqCtx` moved from the loop's stack into a
  `std::deque<ReqCtx> live` keyed by slot id, with a per-step `StepGuard` (= `SlotGuard` +
  `finished`) and a per-step `MropeScope`; `cur`/`pp_*`/`part_next`/`sp.embd_rows` are re-armed per
  step because they are serve-scope. The watchdog now aborts only when **every** active slot stalled
  and names them; a lone stalled slot gets `ERR <id>` and dies (R3). The prompt loan has one owner
  and a deferral queue (R8); an image request runs alone (R7). Startup refuses `--serve-slots >= 2`
  when parking is off (R2/R12). **The serial driver's text is byte-identical to S3.1e-1's** (the gate
  is an early return placed before it) and its 96 serial-reachable output calls are unchanged —
  `.shz_cmd/s31e2_serial_proof.py`, `.shz_cmd/s31e2_output_proof.py`. Shape, lifetime decision, 18
  traps and the owner's two-client recipe: **`.megamind/src/program/s31e2-driver-notes.md`**.
  **Still unverified without a restart:** that two real conversations interleave, that a swap leaves
  the session bit-identical (`STRATA_STATE_HASH`), and the swap cost (OQ3) / starve bound (OQ4).
* **Wire protocol is versioned by a `READY` token and defaults off**: `--serve-slots 0` must be
  byte-identical to 0.1.30, which is also the bit-exactness harness (§7.1).

## S3.7 — the owner's "does each slot reserve --max-context?" question, and the real bug behind it

Short answer: **no, nothing is reserved per slot** — a parked conversation is sized by the tokens it
actually holds (81 tok = 226 MiB, 95 303 tok = 2.4 GiB, the 524288-token ceiling = 7.6 GiB). But the
investigation found three real defects, all fixed:

1. **The owner's ERR was not a budget error.** `do_swap` looked a pre-empted slot's branch up with
   `ConversationCache::best()`, whose prefix rule needs the query **strictly longer** than the entry. A
   slot's own branch is **exactly equal** to what it parked, so the lookup returned 0, nothing was
   mounted, and `step_gate` ended a live decoder — with a 10 048 MiB budget and a 113 MiB branch. Fixed
   with `best_exact()` + `serve_swap::seek_mount_index()` (live branches, exact only).
2. **RAM admission *was* a flat reservation** — `budget / --conversation-cache-slots` (1131 MiB) per
   request regardless of length. Now `(prompt + max_new) x a per-token rate`, learned from the largest
   real park, else from the startup ceiling, capped by the ceiling and the budget.
3. **`gate_end_reason` blamed the budget for every `Gate::end`**, including `!session_valid` and runs
   with an ample budget. `GateEnd` now carries the state and the two real numbers.

Full notes, the measured cost model and the five mutation proofs:
**`.megamind/src/program/s37-slot-cost-notes.md`**. Docs: `docs/DETAILS.md` "Nothing is reserved per
slot", `docs/STAGE3-CONCURRENCY.md` §S3.7. Tests: conversation_cache 4232→4252, serve_swap 156→169,
serve_driver 303→340. **Owner action: restart and re-run two clients — expect `restored …B` instead of
`restored 0 B` on the re-mount.**

## S3.8 — two concurrent prefills erased each other and never finished

Owner's report: *"when two prefills happen at the same time and they swap, they seem to erase the
other's previous progress, as after each swap they always start at 0 tokens again which makes them
never finish."* Confirmed in the log: slot 2 (decode) parks 777 MiB and resumes normally, while slot 3
(prefill) reports `saved 0 B, restored 0 B` on **every** swap and `reset to token 0` **101 times**.

Five links, and the last one is the licence: `publish_decode_branch` early-returns unless
`phase == decode`, so a mid-prefill slot was never published and `live_ok` stayed false;
`park_current`'s first guard is `!live_ok` → `Saved::skipped`; `invalidate_unparked` clears
`resumable`; `step_gate` answers `re_read`; and `outgoing_for(prefill)` was **always** `re_readable`,
which is exactly what permits swapping out without a save. The watchdog cannot see the spin, because
every pass really does read tokens.

Fixed on both sides — **both halves are needed**: `outgoing_for(Phase, read_tokens)` makes a prefill
that has consumed tokens `must_park` (only cursor 0 stays `re_readable`), and `publish_prefill_branch`
gives `park_current` something to save. `restore_published_branch` re-derives the cursor from
`live.size()` so the read resumes where it stopped. `kMaxRereads` is the net underneath: a fifth reset
ends the request with a named ERR instead of restarting forever.

A **separate silent bug** on the same path: `arm_prompt_state` re-armed `cur` but not `req_imgs`, while
`checkpoint_at` stamps `c.imgs = imgs_below(req_imgs, L)` — a resumed prefill stamped another
conversation's picture keys onto this branch's checkpoints, corrupting the cache's prefix comparison
for every later request of that chat.

Full chain, traps and the four mutation proofs (8 / 3 / 11 / 13+5 failures):
**`.megamind/src/program/s38-two-prefill-livelock.md`**. Docs: `docs/DETAILS.md`,
`docs/STAGE3-CONCURRENCY.md` (new item 6 + risk **R15**). Tests: serve_driver 340→377.
**Owner action: restart and run two long-prompt clients — the prefill swaps must show
`park: slot N parked T tokens` and `slot N resumed from T tokens`.**

## S3.9 — a parked branch was being stolen by another request

Owner, right after S3.8: *"the parking doesnt seem to work now, as when a new request arrives it always
kicks the currently running request out with an 'it was decode but not parked' error."*

**The park was succeeding.** The log: `park: slot 14 parked 10394 tokens / 994 MiB`, then
`restored 9849 tokens (checkpoint) … parked=4`, then `slot 14's parked branch (10394 tokens) is no
longer in the cache`, then the gate ends it — with **`evictions=0` for the whole run**. Nothing was
ever pruned.

A parked entry is **one object** (live branch + checkpoint chain + K/V). A new request looks a branch up
by **prefix**, which matches any *checkpoint* of any entry, and the mount then `take()`s the whole
entry. Slot 15's prompt shared 9 849 of slot 14's 10 394 tokens (same chat), so request 15 mounted —
and destroyed — the only copy of slot 14's conversation. **The cache had no concept of ownership.**

It surfaced after S3.8 because S3.8 made mid-prefill slots park, so far more entries were owned by live
slots. The bug was always there; S3.8 moved it onto the common path.

Fixed with a **claim, not a pin**: `SavedConversation::owner` is the request whose branch this is while
that request runs. `find`/`best`/`best_exact` take a `requester` and skip entries claimed by somebody
else; `seek_mount_index` passes `s.id`, so a slot always gets its own branch back. `release_owner` runs
from `drop_ctx` — the only thing that destroys a context — so every end path releases and a claim
cannot leak. `victim_to_prune()` prefers unclaimed entries, falling back to LRU when all are claimed:
the byte budget is still the hard limit. `park_owner_for(id)` decides the claim from `working_of(id)`,
which is exactly "does this slot still owe a step" — and that is what keeps **stage 2** working, since a
finished conversation's branch is parked unclaimed and the next request of that chat still resumes from
its prefix.

Full chain, traps and the three mutation proofs (2+1 / 3 / 1):
**`.megamind/src/program/s39-parked-branch-theft.md`**. Docs: `docs/DETAILS.md`,
`docs/STAGE3-CONCURRENCY.md` (new item 7 + risk **R16**). Tests: conversation_cache 4252→**4271**,
serve_swap 169→**173**.
**Owner action: restart. The `no longer in the cache` + `would step in decode` pair should be gone. A
slightly lower `N reused` on requests sharing a prefix with a *running* slot is the fix working.**

## S3.10 — `--decode-tokens N`: how many tokens a slot writes on its turn

Owner: *"a request generates 1-3 tokens or so and then switches to the next request, which is too low.
… for example 10 tokens are generated when its a requests turn (for the decode)."*

Not a bug — a **decode step was one verify window**, and a window commits `a + 1` tokens (`a` = MTP
drafts accepted, 1..8, usually 1-3). Every step boundary is a hand-over candidate, so the scheduler
paid or evaluated a 237 MB–2.25 GB save+restore every ~2 tokens: R10 in its purest form.

`--decode-tokens N` (default 0 = today) makes a slot's decode **turn** a token budget: `run_decode_step`
loops its existing window body until the slot has produced N tokens since the turn began, then returns
`progressed` and the scheduler may take the session. Tokens, not windows, because a window's yield
varies with acceptance; a window may overshoot (it is the smallest unit of work); the first window of a
turn always runs (a turn that runs nothing is a stall and freezes the watchdog heartbeat); EOS,
`--max-new` and `STOP <id>` still end the request from inside a window, so a cancel is delayed by at
most ~24 ms. The pre-emption point did not move, so none of S3.1d/S3.6/S3.8/S3.9's parking logic had
to change, and the serial driver is untouched.

Pure predicates `decode_step_tokens` / `decode_turn_done` in `serve_driver.hpp`; the driver's start-up
line now prints `decode 10 tokens/turn` vs `decode 1 window/turn`. `--starve-ms` and `--decode-tokens`
are the two ends of one trade: the second sets how much work a slot does before it *can* be
pre-empted, the first how long another slot waits before it is *forced*.

Full notes and the mutation proof: **`.megamind/src/program/s310-decode-tokens-notes.md`**. Docs:
`docs/DETAILS.md`, `docs/STAGE3-CONCURRENCY.md` (new §S3.10 + R10). Tests: serve_driver 377→**395**.
**Owner action: restart with `--decode-tokens 10` and compare `swaps=` / `swap_ms=` — that is OQ3/OQ4
measured for the first time.**

## Stage 4 — split prefill/decode across instances (in progress)

**Design contract: `docs/STAGE4-SPLIT-ROLES.md`. Plan: `.megamind/stage4-plan.md`. Per-step status and
the full agent notes index: `.megamind/taskboard.md`.**

CPU-only steps landed so far, each new-files-only and mutation-tested, none of them having started an
engine, loaded a model, touched a GPU or opened a socket:

* **S4.3.1** the payload codec — `include/strata/core/conversation_wire.hpp` (483 checks, 41/42
  mutations). Notes: `.megamind/src/core/s431-payload-codec-notes.md`.
* **S4.3.2** the tmpfs handoff arena — `include/strata/core/handoff.hpp` (268 checks, 16 mutations).
  Notes: `.megamind/src/core/s432-handoff-arena-notes.md`.
* **S4.3.3** the prefill↔decode control protocol — `include/strata/program/prefill_svc.hpp`
  (2640 checks, **74/74 mutations**, all twelve §4.7 failure modes as data-driven scenarios).
  Notes: `.megamind/src/program/s433-prefill-protocol-notes.md`. **S4.3.5 and S4.3.6 must read that
  notes file: it is the single source for the API and for the 14 deviations from §4** (the header's
  banner numbers them `DEV-1 … DEV-14`, deliberately *not* `D1…D5`, which are the manager's
  decisions in `stage4-plan.md`).
* **S4.3.4** the KV-only drafter — `MtpDrafter::bind_kv_only` in `include/strata/core/mtp.hpp` +
  `src/core/mtp.cpp` (the only two files edited). The prompt K/V path is byte-identical; drafting, the draft
  graphs, the draft head/logits, the coupled sampler and the draft-only buffers are gated off
  (14.8 MiB at the live config, 107.3 MiB at `--spec 8` with no window; `bind_bytes()` → 0; the prefill role
  no longer needs `NativeHead`'s 497 MiB). **OQ-S4-9 answered: implement it.** Notes:
  **`.megamind/src/core/s434-mtp-kv-only-notes.md`** — S4.3.5 needs the one-line swap at
  `generate.cpp:4015` and the caller audit. Owner action: the `STRATA_STATE_HASH` `mtp=` comparison.
* **S4.2** landed the wait-queue vocabulary (`Hold`/`Wait`/`WaitQueue`/`--hold-ms`); S4.3.6 adds
  `Wait::prefill` with the strings `prefill_svc::kWaitReasonPrefill` / `kWaitTextPrefill` verbatim.

**Configure note for every S4.3 CPU test:** `-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
-DSTRATA_BUILD_TESTS=ON`. `STRATA_ENABLE_CUDA` defaults OFF (`CMakeLists.txt:35`), and without it
`strata_engine` — and with it the whole S4.3 block — silently does not exist.

## Not done / deferred

* **Stage 3 is implemented but never run.** It needs the owner's restart to be measured and
  accepted: OQ3 (swap cost), OQ4 (`--starve-ms`), OQ6 (VRAM for prefill/decode overlap, S3.5).
  On this box `--serve-slots 3` refuses to start (rc 2) because `MemAvailable` ~2.9 GiB is under
  the parking floor + request headroom — free RAM first. `--serve-slots 0` and
  `STRATA_NO_SWAP=1` are the escape hatches.
* Stage 3 still open: S3.3b (residency CPU test + `STRATA_RES_VERIFY`), S3.5 (prefill/decode
  overlap). Two agents crashed on oversized stage-3 tasks; keep them split.
* Windows: shared arena and cross-process core discovery are still not implemented
  (each server keeps its own arena and today's fixed pinning; the note says so).
* `setup.py` still carries the owner's uncommitted local edit (the removed
  IQ3_XXS/IQ3_S 128K context clamp) — left untouched, not ours.
* Suggested follow-ups (not started): per-model arena filenames in `setup.py`;
  HIP shared-arena coverage.

## Stage 4 — split prefill/decode across instances + real batching + hold-don't-reject (2026-10-04)

Owner's ask in `prompt.md`. Program brief and the settled manager decisions **D1-D5** are in
**`.megamind/stage4-plan.md`**; the taskboard's Stage 4 section is the live checklist.
**Nothing in stage 4 has been run against a live model** — the owner's restart is still required for
every measurement in stage 3 and stage 4.

**Done and CPU-verified:** S4.0 (the design contract `docs/STAGE4-SPLIT-ROLES.md`, 1748 lines),
S4.1 (the batching feasibility verdict `docs/STAGE4-BATCH-DECODE.md`, 956 lines), S4.2 (hold, don't
reject — end to end), and S4.3.1-.4 (payload codec 483 checks, handoff arena 268, control protocol
2640, `MtpDrafter::bind_kv_only`). **S4.3.5 is unblocked and is the next thing to start.**

The three things a new agent must not re-learn the hard way:

1. **`-DSTRATA_ENABLE_CUDA=ON` is mandatory when configuring a test build.** With it off there is no
   `strata_engine` target, so the `strata` binary and the whole stage-4 test block are silently
   absent — a build that "passes" without ever compiling `generate.cpp`. Five child agents crashed on
   the S4.2 engine wiring; it is now done, by the parent, and **should not be delegated again as one
   task** — `generate.cpp` is 10 000 lines.
2. **`CMakeLists.txt` is the parent's for all of stage 4.** Three `EXISTS`-guarded blocks are already
   in place and activate when a child's sources land. Children report link-library problems instead of
   editing the file (risk R13).
3. **S4.1's verdict changes the plan's shape.** Token-level batching (B=2 in one verify window) is
   real but costs +180 MiB VRAM, +6.19 GiB pinned RAM and six kernel contracts for **+11-13 %**. A
   second session arena is +1.47 GiB VRAM, needs **zero kernel changes**, and is bit-exact by
   construction. Do not start S4.4's kernel work without the owner's two decisions (may the expert
   cache shrink; is +6.19 GiB pinned RAM per extra sequence acceptable).

Verification snapshot (all green, no engine started, `/dev/shm` untouched):
`cd build && ninja` clean; serve_driver 523 · saved_conv_wire 483 · handoff_arena 268 ·
prefill_svc 2640 · serve_swap 173 · slot 176 · serve_proto 140 · prefill_loan 128 · conv_cache PASS ·
conversation_cache 4282 · conversation_memory 23 · conversation_snapshot 2089 ·
conversation_validation 1323; `pytest serve/test_server.py` 81 passed / 2 skipped;
`.shz_cmd/s31e2_serial_proof.py` byte-identical and `.shz_cmd/s42_wiring_proof.py` PASS.
