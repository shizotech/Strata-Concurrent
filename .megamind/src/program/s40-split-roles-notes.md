# S4.0 — the stage-4 split-roles design contract (docs only, no source touched)

Deliverable: **`docs/STAGE4-SPLIT-ROLES.md`** (1693 lines). Design only: **no source file was edited,
no engine or server was started, `/dev/shm` was read-only.**

## What the document settles (the parts an implementer would otherwise have to invent)

1. **Roles.** `--role full|prefill|decode`, default `full`. A per-subsystem table of what each role
   initialises, with the allocation site for every row. `--role` requires/forbids `--serve` per role.
   New flags: `--prefill-endpoint`, `--prefill-listen`, `--prefill-instance`, `--prefill-handoff-dir`,
   `--prefill-slots`, `--prefill-slot-tiers`, `--prefill-max-tokens`, `--prefill-queue`,
   `--prefill-fallback none|local`, `--prefill-retries`, `--prefill-client-id`.
   New env: `STRATA_HANDOFF_LEASE_MS`, `STRATA_HANDOFF_DIR`, `STRATA_PREFILL_TRACE`,
   `STRATA_PREFILL_DISABLE`, **`STRATA_NO_HANDOFF`** (the bit-exactness arm, the `STRATA_NO_SWAP` twin).
2. **The payload.** `SavedConversation` verbatim, plus an explicit **record-stream wire format**
   (nothing is memcpy-able: `ConversationBuffer` has no `data()`, `std::vector` everywhere). Exact
   field lists, `BLOBREF {bytes, offset}`, 64-byte blob alignment, an END marker.
   **`owner` never crosses a process**: the payload leaves with `kNoOwner`, the receiver assigns the
   claim at `put()`. Request ids are per-instance; the slot header carries `client_id` + `request_id`.
3. **The transport.** `/dev/shm/strata-handoff/<pack16>-<split>/<instance>/arena`, 4096-byte arena
   header + N × (4096-byte slot header + slot bytes), every offset pinned by `static_assert` the way
   `pinned.hpp:51-56` does. FREE→CLAIMED→READY→FREE, `flock` for decisions only, one ordered `pwrite`
   publish, `/proc/<pid>/stat` field 22 for dead/recycled owners, **leases** (the slot-leak fix),
   `statvfs` before `ftruncate`. Sizing is machine-independent: `conversation_snapshot_bytes(tier)`
   priced at start-up, never a RAM guess.
4. **The protocol.** `include/strata/program/prefill_svc.hpp`, separate from `serve_proto.hpp`.
   `HELLO` / `PREFILL-READY-V1` / `PING` / `REQ` / `CANCEL` / `ACK` / `BYE` out; `PONG` / `QUEUED` /
   `CLAIM` / `SEG` / `DONE` / `ERR` / `INFO` in. Exact grammars, 14 error codes each mapped to
   wait-or-error, and an 8-state handoff state machine (`H_*`, deliberately **not** the engine's
   `Phase` names). **Deliberately absent:** `prefix=` (the prefill instance owns the reuse decision —
   two lookups over one store is S3.9), a `RELEASE` alias of `ACK`, and an `ACK-ED` echo.
   **The prompt ids go in a file, not on the wire** (a 245 946-token prompt is ~1.7 MB of ASCII).
5. **Topology.** One arena per prefill instance (one writer, always). The prefill instance owns its own
   `Registry` and its own `ConversationCache`. One job at a time (the prompt loan is one loan).
   Parallelism is in the copy-out, not the compute.
6. **The wait queue.** One new reason, `Wait::prefill` / `prefill-busy`. Three queues, each with one
   job. **The decode engine thread never blocks on a socket** — `Phase::await_prefill` +
   `Step::await_prefill` are non-blocking polls.
7. **S4.3 plan.** Nine steps, disjoint file ownership, `generate.cpp` appears in only three of them
   and those are strictly sequential (R13). Steps 1-4 are CPU-only and retire most of the risk before
   any GPU cycle.

## The three findings that changed the design (put these in front of the implementer)

* **The prefill role CANNOT skip the MTP drafter.** `SavedConversation::kv` is
  "main layers followed by the draft layer" (`conversation_cache.hpp:131`), and the draft K/V is
  produced *during the prompt read* by `mtp.prefill` from `sp.on_chunk` (`generate.cpp:4412-4416`). Skipping it makes the payload incomplete, which means the decode
  instance re-reads the prompt, which defeats the split. What the prompt path needs is only
  `record_forward(T, -1, …)`, which returns right after the K/V append (`mtp.cpp:486`).
  **Fix: a new `MtpDrafter::bind_kv_only()`** — `record_forward` needs `wt_` for `token_embd.weight`
  (`mtp.cpp:428-430`) but not `head_`, `window_R_`, `head_logits_`, `dhead_` or the coupled sampler.
  Without it the prefill role must load `NativeHead` (497 MiB) purely because `bind()` demands it
  (`mtp.cpp:383`). OQ-S4-9.
* **The prefill instance's own `ConversationCache` is the biggest win in the design.** The owner's log
  is almost entirely reuse, not reads: `prompt 245946 tokens = 245751 reused + 195 read in 1116 ms`
  (`strata-iq3_s.log:6649`). A continuing chat can be a handoff with **zero GPU work**: `best()` →
  write the parked image into a slot → `DONE`.
  **The catch:** that cache only ever holds what the prefill instance itself read. The tokens a decode
  instance *generated* live only in that decode instance's cache, so a chat that moves between decode
  instances re-reads. Closing it needs a reverse handoff — deliberately deferred (S4.5, OQ-S4-4); the
  verb `TAKE` is reserved for it.
* **The prefill role claims no cores.** `Prefill` never uses `ExpertPool` — it stages experts on its
  own `Stager` threads (`prefill.cpp:137-250`). So the prefill instance skips
  `generate.cpp:2614` and takes nothing from `/dev/shm/strata-core-leases`. The decode instances keep
  the whole pool. The stage-1 collapse is avoided by construction, not by negotiation.

## Measurements taken for this document (host-side only)

`grep -o 'snapshot_bytes=[0-9]*' strata-iq3_s.log` over **871** parks:

```
min 237 413 652   p50 1 033 605 768   p90 2 029 940 496   p99 2 939 786 284   max 3 180 486 108
```

(the stage-3 doc's "237 MB … 2.25 GB" understates the tail — the real max is **3.18 GB**, and the
config now asks for a 20 000 MiB parking budget, `strata-iq3_s.json:37-38`.)

Rate over the 30 parks above 100 000 tokens: **20 655 … 27 342 B/token, mean 22 532**. The apparent
"3.1 MB/token" outlier is an 81-token park at 237 MB — that is the **fixed floor** (~236 MiB of
per-layer running state), not a rate.

Cost of moving one, from the engine's own lines: park 3.18 GB in **410 ms**; swap-save 2.79 GB in
**220-313 ms**; restore 97 934 tokens in **310 ms**.

Host: `/dev/shm` 70 GiB total / **24 GiB free**; `Mem` 94 GiB / **3 GiB available**. The arena is tmpfs,
so its size is a RAM decision sitting next to the 46.8 GiB expert arena → OQ-S4-2.

## Traps this document exists to prevent

* **Never stream GiB through a pipe.** The socket carries a slot name.
* **`ftruncate` on tmpfs is sparse** — it succeeds when there is no room and the writer dies of SIGBUS
  with no message. `statvfs` first (`pinned.cu:220-239`, `:320-335`).
* **A `READY` slot whose client died is a leak** unless there is a lease. Without it one crashed decode
  instance strands a 3 GB slot until the prefill instance restarts.
* **A handoff retry loop is invisible to the watchdog**, exactly like S3.8's re-read loop — every pass
  really does move gigabytes. `--prefill-retries` is not optional, and the test must fail when it is
  neutered.
* **`arm_prompt_state`'s lesson travels to the prefill instance**: re-arm `cur`, `req_imgs`, `pp_*` and
  `part_at` per job, or `checkpoint_at` saves another chat's tokens under this job's id.
* **The prefill instance uses `best()`, never `best_exact()`** — it never hands a session over.
* **The payload is written from the image in place**; do not `take()` the entry to write a slot, or the
  reuse win dies with it.
* **`--role prefill|decode` must be refused on `_WIN32`** — the whole transport is POSIX.

## What the parent must decide (the OQ list, §9)

OQ-S4-1 the prefill instance's GPU · **OQ-S4-2 how much tmpfs/RAM the arena gets** (the default 6-slot
proposal is ≈ 8.2 GiB (slots grouped and sized per tier); a single 524 288 tier with 2 slots is
22.8 GiB) · OQ-S4-3 a third parking budget ·
OQ-S4-4 whether the reverse handoff is in scope · OQ-S4-5 one or several prefill instances ·
OQ-S4-6 whether decode ever reads locally · OQ-S4-7 the `--max-context` mismatch rule ·
OQ-S4-8 Windows/HIP · **OQ-S4-9 `bind_kv_only` or eat 578 MiB** · OQ-S4-10 one model per prefill instance.

## For S4.3

Read §1.4 (the MTP problem) and §3 (the arena) before writing any code. Steps S4.3.1/.2/.3 are
independent, CPU-only, and touch no shared file — start those three in parallel.
