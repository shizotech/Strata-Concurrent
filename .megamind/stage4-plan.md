# STAGE 4 — program brief (manager)

Owner's ask (`prompt.md`, 2026-10-04):

1. **Split the two stages across processes/instances.** One **prefill-only instance** does prompt
   processing for everybody; **several text-gen instances** do decode. The prefill instance is a
   *backend*: it never accepts client requests, only text-gen instances talk to it.
2. **The text-gen stage must do real batching of concurrent requests on the GPU** — "not just
   simple swap&park". This is exactly the thing `docs/STAGE3-CONCURRENCY.md` §10 deferred as
   "a stage-4 engine": a shared KV arena with per-sequence page tables, per-row GDN state, and a
   verify window that carries several sequences at once.
3. **API change: hold, don't reject.** Today a request with no parking budget / no free slot gets
   `ERR …`. It must instead wait and run when resources free up.

Target shape:

```
PREFILL INSTANCE  (one, its own GPU(s))
   ^   ^   ^
   |   |   +-- TEXTGEN INSTANCE (2 requests batched)
   |   +------ TEXTGEN INSTANCE (3 requests batched)
   +---------- TEXTGEN INSTANCE (...)
```

## Manager decisions (settled — children must not re-litigate these)

**D1 — roles, not new binaries.** The engine gains `--role full|prefill|decode` (default `full` =
today, byte-identical). A `prefill` role is a `--serve`-style process that speaks a *new internal*
protocol and never loads a decode session's drafter/MTP/suffix machinery; a `decode` role is today's
serve process with the prompt path replaced by "ask the prefill instance". `setup.py` and
`serve/server.py` stay the only client-facing surfaces.

**D2 — the hand-off payload is a `SavedConversation`, and it moves through shared memory, not a
socket.** `strata::core::SavedConversation` (`include/strata/core/conversation_cache.hpp`) already is
"the complete state of a conversation after its prompt was read": live ids, image keys, the
checkpoint chain, per-layer `ConversationKv`, and the `--layer-split` stage parts. That is the
payload. It is 237 MB … 2.4 GiB on this box, so the transport is a **tmpfs handoff arena**
(`/dev/shm/strata-handoff/…`) with a slot table, and the socket only carries a small text control
line naming the slot — the same flock + header-state + nonce discipline stage 1's
`src/core/pinned.cu` already uses for the expert arena. Never stream GiB through a pipe.

**D3 — the control protocol mirrors the existing wire, versioned by a `READY` token, default off.**
`include/strata/program/serve_proto.hpp` is the only place wire lines are formatted/parsed; the new
prefill↔decode protocol is a **separate** header (`include/strata/program/prefill_svc.hpp`) so the
client wire cannot drift. Nothing about `--serve-slots 0` changes.

**D4 — batching is a kernel/session-layout problem, not a scheduler problem, so it gets a
feasibility gate before any rewrite.** Stage 3's non-goals stand until measured: `Verifier::run`
puts T tokens at *consecutive* positions, `kVerifyMaxT = 8` is baked into the GDN kernels
(`src/kernels/cuda/verify_kernels.cu:386,403,415`), `QsaState` is one pool set per session, and the
graphs bake in pointers. The first deliverable is an inventory + a priced plan + the smallest
provably bit-exact step (e.g. B=2 sequences in one window vs two serial windows), **not** a big-bang
rewrite. Bit-exactness against the serial path remains the acceptance bar.

**D5 — "hold, don't reject" is a queue, with a bound and visibility.** Rejection sites are
`serve_driver::Refuse{slots_full,ram,no_parking,registry_full}`, `serve_swap::park_guard`'s
`end_incoming`, `step_gate`'s `end`, and `serve/server.py`'s `GpuBusy`/503. The rule: a request that
cannot run *now* is parked in a **wait queue** and started when resources free; only a request that
can *never* run (prompt longer than `--max-context`, parking off with `--serve-slots>=2`) is an
error. Every wait must be visible on `/status`, `/slots` and in the engine's `activity:` line.

## Workstreams and file ownership (avoid risk R13: two agents in one file)

| id | workload | owns | must not touch |
|---|---|---|---|
| **S4.0** | Stage 4 design + contract doc | `docs/STAGE4-SPLIT-ROLES.md` (new) | any source |
| **S4.1** | batched-decode feasibility: kernel/session inventory, priced plan, the minimal bit-exact step | `docs/STAGE4-BATCH-DECODE.md` (new), `.megamind/src/kernels/` | any source, `docs/STAGE4-SPLIT-ROLES.md` |
| **S4.2** | hold-don't-reject: wait queue in `serve/server.py` + engine admission predicates | `serve/server.py`, `serve/test_server.py`, `include/strata/program/serve_driver.hpp`, `src/program/serve_driver_test.cpp`, the admission call sites in `generate.cpp` | `src/core/*`, `src/kernels/*`, `setup.py` |
| **S4.3** | the prefill-as-a-service split (roles + transport + handoff) | new `include/strata/program/prefill_svc.hpp`, `src/program/prefill_svc*`, `src/core/*` snapshot serialization, `generate.cpp` role plumbing, `setup.py` | `serve/server.py` until S4.2 lands |
| **S4.4** | batched decode implementation | `src/kernels/cuda/*`, `src/core/session.cpp`, `src/core/verify.cpp`, `src/core/layer.cpp` | the serve loop |

S4.0 ∥ S4.1 ∥ S4.2 now. S4.3 after S4.0 + S4.2. S4.4 after S4.1.

## Pointer — S4.0 landed (2026-10-04)

**`docs/STAGE4-SPLIT-ROLES.md`** is the S4.3 contract (design only, no source touched). Agent notes:
**`.megamind/src/program/s40-split-roles-notes.md`**. D1-D5 stand; two refinements, both recorded in the
document and neither a re-litigation:

* **D1 refined.** "A `prefill` role … never loads a decode session's drafter/MTP/suffix machinery" is
  **not implementable as written**: `SavedConversation::kv` is "main layers followed by the draft
  layer" (`conversation_cache.hpp:131`), and the draft K/V is produced *during the prompt read* by
  `mtp.prefill` from `sp.on_chunk` (`generate.cpp:4413-4416`). A payload without it is incomplete and
  the decode instance would re-read the prompt. What the prompt path needs is only the drafter's
  forward-to-K/V half (`mtp.cpp:486` returns right after the K/V append), so the role skips
  `MtpDrafter::bind`'s head/logits/coupled allocations via a new `bind_kv_only()`, and skips the
  `Verifier`, `NativeHead`, `SuffixDrafter`, `DraftPolicy` and the CPU pool entirely. See §1.4.
* **D3 refined.** "Mirrors the existing wire" means the same *style*, not the same vocabulary: the
  prefill↔talk protocol has no `GEN`/`T`/`DONE`, and its version token is `PREFILL-READY-V1`. The
  `READY`-token-versioning mechanism is copied; the line set is not shared. See §1.6 rule 4.

New decisions the document makes (all inside D1-D5): the payload is a **record stream**, not a struct
dump (`ConversationBuffer` has no `data()`); `SavedConversation::owner` **never crosses a process**;
the arena is **one per prefill instance** (one writer); sizing is by `conversation_snapshot_bytes(tier)`
at start-up, **never by this machine's RAM**; the prompt ids travel in a **file**, not on the wire; the
decode engine thread **never blocks on a socket**; and a handoff retry needs a bound
(`--prefill-retries`) because a handoff loop is as invisible to the watchdog as S3.8's re-read loop was.

## Pointer — S4.1 landed (2026-10-04)

**`docs/STAGE4-BATCH-DECODE.md`** is the S4.4 feasibility gate D4 asked for (design + host-side
measurement only; no source touched). Agent notes: **`.megamind/src/kernels/batch-decode-notes.md`**.
Four things the parent must know before deciding whether S4.4 happens:

* **The box changed.** `grep -c "layer split" strata-iq3_s.log` = **0** — both live engines own one
  whole GPU each, and **GPU 1 is free** (2 396 MiB, all of it a whisper server). Stage 3's per-stage
  0.554/0.560/0.547 GiB numbers are not what this box pays; a single-GPU session arena at
  524288/int8/r32768 is **1.390 GiB**. Stage 3 §2.2's "524288 int8 resident 0 = 1.753 GiB" is
  **wrong** — measured **7.189 GiB**.
* **The kernels are already batched** where Stage 3 §1 said they were not: attention, indexer scores,
  top-k, `kv_stream_resolve` and `kv_append_*_step` all take `n_q` rows with per-row step records and
  per-row selections. "One sequence per window" is a host policy in `Verifier::run`
  (`verify.cpp:964-969`), not a kernel limit. The genuinely single-sequence pieces are the **GDN
  recurrence + conv history**, the **indexer key store** (no page table), and the **8-row ceiling —
  which is six independent ceilings** that one sequence's drafts already consume at `--spec 4`.
* **The prize is small and it is computed, not guessed.** T=1 window = 24 ms (`strata-iq3_s.log:106`),
  T=6 = 66 ms (`:8578`) ⇒ 24 ms fixed + **8.4 ms per extra row** ⇒ **B=2 buys +11 % to +13 %**, not
  +100 %, because the rows that would have been drafts become second-sequence rows. B=2 costs
  **+180 MiB VRAM + 6.19 GiB pinned RAM + six kernel contracts**. A **second session** (zero kernel
  changes, bit-exact by construction) costs **+1.47 GiB VRAM + 6.19 GiB RAM**.
* **Recommendation.** S4.3 first; then S4.4.0 (a pure window planner, free) and **S4.4.1 = two
  sessions, one engine thread** as the measurement that decides whether the six-kernel route is funded.
  On this box **a second session on the free GPU 1 beats B=2 in one window on every axis except the
  owner's literal sentence.** Under `--kv k8v4` every option is unavailable — refuse at start-up.
  **Owner decisions S4.4 needs:** may the expert cache shrink (~130 slots for B=2, ~1 150 for a second
  session, at an 85-94 % hit rate), and is +6.19 GiB of pinned RAM per extra sequence acceptable.

**Needs the owner (OQ-S4-1…10, §9).** The two that block S4.3's defaults: **OQ-S4-2** — the arena is
tmpfs, so it is RAM next to the 46.8 GiB expert arena; the proposed per-tier 6-slot default is ≈ 8.2 GiB
and a single 524288 tier with 2 slots is 22.8 GiB, against 24 GiB currently free in `/dev/shm`.
**OQ-S4-9** —
implement `MtpDrafter::bind_kv_only`, or the prefill role loads `NativeHead` (497 MiB) for nothing.

Measured for the sizing (host-side only, 871 parks): `snapshot_bytes` min 237 MB, p50 1.03 GB,
p90 2.03 GB, **max 3.18 GB** — the stage-3 doc's "237 MB … 2.25 GB" understates the tail. Rate
20.7-27.3 KiB/token above 100k tokens, with a ~236 MiB fixed floor.

## HARD CONSTRAINTS (unchanged, owner)

* **DO NOT START THE SERVER / ENGINE.** Two live `strata --serve` processes own the box
  (pids 1000, 1051; GPU 0 and 2 at ~23.8/24 GiB; `MemAvailable` ~10 GiB of 94). No
  `serve/server.py`, no `engine/strata --serve`, no model load, no requests, no benchmarks that
  load a model. Source + build + small CPU-only tests only.
* `/dev/shm/shared_experts.dat` (46.8 GiB) and `/dev/shm/strata-core-leases` belong to live
  processes — read-only, never delete/truncate/write.
* Build check: `cd build && ninja` (Release, CUDA arch 86, tests OFF). CPU tests build in
  `/tmp/a2build` (`STRATA_BUILD_TESTS=ON`).
* `--serve-slots 0` must stay byte-identical to engine 0.1.30 (the bit-exactness harness,
  `docs/STAGE3-CONCURRENCY.md` §7.1).
* Do not "pin" the design to this machine's RAM/VRAM — the owner will add RAM.

## Pointer — S4.3.2 landed (the tmpfs handoff arena)

New files only: `include/strata/core/handoff.hpp`, `src/core/handoff.cpp`,
`src/core/handoff_arena_test.cpp`. No existing source touched, no `CMakeLists.txt` edit — the parent's
`EXISTS`-guarded block activated and its link libraries are correct.
Notes + the full API for S4.3.5 (writer) and S4.3.6 (reader): **`.megamind/src/core/s432-handoff-arena-notes.md`**.

`handoff_arena_test` = **268 checks OK, ~5.2 s, CPU-only** (no engine, no model, no GPU). It only ever
creates `/dev/shm/strata-handoff-test-<pid>/` and cleans that up; `/dev/shm` still holds exactly
`shared_experts.dat` + `strata-core-leases` afterwards. 16 mutations of the arena all bite.

Three things the parent must carry into S4.3.5/.6:
* **§3.5 step 3 as written is impossible** (`payload_bytes`/`payload_hash`/`published_ms` are not adjacent
  to `state`). Publish is now one 48-byte metadata store while the slot is still CLAIMED, a release fence,
  then **one 8-byte store of `state`+`flags` — that store is the publish point.** Same invariant.
* **`client_id` is opaque, so §3.6's "the READY slot's client pid is dead" needs a hint**: the writer must
  call `note_client_pid(client_id, pid)` at HELLO (it has the pid from `SO_PEERCRED`). Without it the
  300 s lease still reclaims the slot, so the leak stays bounded — the registration only makes it instant.
* **`geom_hash` is the caller's job**: `handoff_geom_hash(geometry_key_18_ints, handoff_split_spec(split))`.
  `geometry_key` is file-local to `conversation_state.cpp`, so S4.3.5 must pass the same 18 values
  `SavedConversation::geometry` already carries, and both sides must spell the split with
  `handoff_split_spec()` or the hashes will not agree.

Sizing is machine-independent, as §3.8 demands:
`slot_bytes = round_up_64(conversation_snapshot_bytes(tier)) + 64 MiB`,
`arena_bytes = 4096 + Σ tier_slots × (4096 + slot_bytes)`. At the §3.8 default tiers that prints
**`slots 2,2,2 (tiers 1024,16384,131072), max 131072 tokens, arena 7.99 GiB (8577380352 B)`** — the doc's
"≈ 8.2 GiB". RAM only decides whether it *fits*: the `statvfs` check runs before `ftruncate` and the
refusal is fatal with both numbers (no fallback for a handoff).

**Config trap for the CPU tests:** `STRATA_ENABLE_CUDA` defaults **OFF** (`CMakeLists.txt:35`), and with it
off there is no `strata_engine` target, so the whole `if(STRATA_BUILD_TESTS AND TARGET strata_engine)`
S4.3 block — including `handoff_arena_test` — is silently absent. Configure with
`-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86`.

## Pointer — S4.3.1 landed (the SavedConversation payload codec)

New files only: `include/strata/core/conversation_wire.hpp`, `src/core/conversation_wire.cpp`,
`src/core/saved_conv_wire_test.cpp`. No existing source touched, no `CMakeLists.txt` edit — the parent's
`EXISTS`-guarded block (`CMakeLists.txt:901-916`) activated and its link libraries are correct.
Notes + the full API for S4.3.2/.5/.6: **`.megamind/src/core/s431-payload-codec-notes.md`**.

`saved_conv_wire_test` = **483 checks OK, 0.23 s, CPU-only** (no engine, no model, no GPU, no `/dev/shm` —
the only file it makes is a `memfd`). Round trips for single-GPU, 2-, 3- and 4-carve splits, an empty
conversation, empty optional buffers and a **>16 MiB buffer spanning `ConversationBuffer` segments**, each
proven `encode → decode → re-encode` byte-identical. 42 mutations, 41 bite.

Four things the parent must carry into S4.3.5/.6:
* **One hash, three places.** The codec writes `payload_hash = 0` inside the payload and returns the real
  `fnv1a64` in `WireEncoded::payload_hash`. That value **is** the arena slot's `payload_hash` (§3.4) and
  `HandoffArena::payload_hash_of()` — identical bytes, identical number, no patch and no second pass.
  Publish with it, and pass it back as `WireExpect::payload_hash` on the read side.
* **§2.1's table is wrong about the checkpoint.** `ConversationCheckpoint::gdn/ple/tails/dead/block_pos` are
  plain `std::vector<uint8_t>` (`conversation_cache.hpp:60`), not `ConversationBuffer`s. Only
  `ConversationKv` holds buffers. The wire format is unaffected; do not go looking for `visit()` on a
  checkpoint.
* **`owner` never crosses the boundary.** `encode` does not read it and `decode` always yields `kNoOwner`;
  the receiver claims at `put(std::move(image), req_id, …)`. That is §2.4's rule enforced by construction.
* **Price the tier before the claim.** `conversation_wire_plan(image, plan, err)` allocates nothing and
  writes nothing; `plan.total_bytes` is what must fit `tier_slot_bytes`. `WireExpect::stage_parts` must be
  the receiver's later-stage count (`conv_stages.count`) — §2.5 step 2 in both directions.
