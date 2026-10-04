# Stage 4 — splitting prompt processing from token generation across processes

**Status: design only. No source file was changed to produce this document.**

This is the contract for **S4.3** (`.megamind/stage4-plan.md`, workstream table). It is written so that the
implementation agent can build from it without re-deciding anything: every protocol line has an exact
grammar, every shared-memory field has an exact type and offset, every subsystem is listed with the
file:line that allocates it, and every number is either cited to a file:line that was read or measured
by a host-side-only method (Appendix A). Where a decision genuinely needs the owner it is an **OQ**
(§9), never a shrug.

It refines the manager's settled decisions **D1–D5** (`.megamind/stage4-plan.md:27-60`). It does not
re-litigate them. The one place it *narrows* a decision is called out explicitly (§1.6).

The machine everything is sized against is the owner's box: 3× RTX 3090, `--layer-split auto`
(K = 16, 34 → three stages), IQ3_S native pack, `--max-context 524288` in the stage-3 log and
`262144` in the live config (`strata-iq3_s.json:22-23`), `--kv int8`, `--kv-resident 32768`,
`--spec 4`, `--prefill auto`, `--expert-cache auto`, `--vram-reserve-mib 700`,
`--conversation-cache-mib 20000`, `--decode-tokens 15` (`strata-iq3_s.json:3-41`).

---

## 0. The one-page design

```
                     ┌──────────────────────────────────────────────────────────┐
   client ──HTTP──►  │ DECODE INSTANCE  (--role decode --serve)                 │
                     │   serve/server.py  ->  engine stdin/stdout  (UNCHANGED)  │
                     │   slot registry, wait queue, swap, verify windows        │
                     └───────────────┬──────────────────────────────────────────┘
                                     │  AF_UNIX control socket  (text lines, §4)
                                     │  "PREFILL-READY-V1 …" / "REQ …" / "DONE …"
                                     ▼
                     ┌──────────────────────────────────────────────────────────┐
                     │ PREFILL INSTANCE  (--role prefill)                       │
                     │   one engine thread, one job queue, its own slot claims  │
                     │   Prefill::run chunks + MTP draft-KV + checkpoints       │
                     └───────────────┬──────────────────────────────────────────┘
                                     │  writes a slot, publishes with a release fence
                                     ▼
        /dev/shm/strata-handoff/<pack>-<split>/<instance>/arena     (tmpfs, GiB-scale slots)
        /dev/shm/strata-handoff/<pack>-<split>/<instance>/prefill.sock
```

* **The payload is a `SavedConversation`** — the same object stage 2/3 already park in host RAM
  (`include/strata/core/conversation_cache.hpp:124-162`). It is *already* "the complete state of a
  conversation after its prompt was read". Nothing new is invented.
* **The transport is a tmpfs handoff arena with a slot table**, and the socket carries only a small
  text line naming the slot. Measured payload size on this box: **237 MB … 3.18 GB** per conversation
  (Appendix A.2). Never stream GiB through a pipe.
* **The discipline is stage 1's, verbatim in shape**: a fixed 4 KiB header (magic / version /
  header_bytes / state / owner pid / owner_start / nonce), `flock`-guarded *decisions* only, publish
  with a release fence, dead-or-recycled-owner take-over through `/proc/<pid>/stat` field 22, and a
  free-space check **before** `ftruncate` because a tmpfs `ftruncate` is sparse and the writer then
  dies of SIGBUS mid-write with no message (`src/core/pinned.cu:320-335`, `.megamind/src/core/arena-notes.md:43-48`).
* **The client-facing surface does not change.** `setup.py` and `serve/server.py` stay the only
  client-facing surfaces (D1); the engine's stdin/stdout wire (`include/strata/program/serve_proto.hpp`)
  is untouched by this stage except for the wait-queue reason S4.2 already added.
* **`--role full` (the default) is today's engine, byte-identical.** Every new code path is behind a
  role test, the same way stage 3's driver is behind an early-return gate
  (`docs/STAGE3-CONCURRENCY.md:846-853`).

---

## 1. Roles

### 1.1 The flag

```
--role full|prefill|decode          (default: full)
```

* **`full`** — today. Byte-identical to engine 0.1.30 at `--serve-slots 0`
  (`docs/STAGE3-CONCURRENCY.md:648`, `.megamind/stage4-plan.md:84-85`). Nothing in this document runs.
* **`prefill`** — a prompt-reading backend. It is **not** a `--serve` process: it never reads
  `GEN`/`GENI` from stdin, never emits `T`/`DONE`, never samples, and never accepts a client. It
  listens on its own AF_UNIX socket (§4.2) and speaks only `include/strata/program/prefill_svc.hpp`.
  It owns its own slot registry and its own claims (§5.3).
* **`decode`** — today's `--serve` process with the *batched prompt path* replaced by "ask the prefill
  instance". Everything else — the slot registry, the wait queue, the swap, the verify windows, the
  drafter, the conversation cache, `/slots` — is unchanged.

`--role` is parsed in the same place as every other engine flag (`src/program/generate.cpp:1200-1400`,
the `else if (a == "--…")` chain; `Options` at `src/program/generate.cpp:228`). `--role` **requires**
`--serve` for `decode` and **forbids** it for `prefill`:

```
--role prefill  +  --serve      ->  startup error 2: "--role prefill does not serve clients"
--role decode   without --serve ->  the engine starts --serve anyway (it is a serve process)
--role decode   without --prefill-endpoint -> startup error 2 (see §1.2)
```

### 1.2 The new CLI / env surface

Engine flags (all in `Options`, all in `--help`, all refused by `setup.py` for an engine that does not
know them — the existing probe mechanism `setup.py:1165-1202` and `ENGINE_FLAGS` at `setup.py:1162`):

| flag | default | meaning |
|---|---|---|
| `--role full\|prefill\|decode` | `full` | §1.1 |
| `--prefill-endpoint PATH` | none | the AF_UNIX path a **decode** instance connects to. Required with `--role decode`. |
| `--prefill-listen PATH` | none | the AF_UNIX path a **prefill** instance binds. Required with `--role prefill`. |
| `--prefill-handoff-dir PATH` | `/dev/shm/strata-handoff` | where the arena and the socket live (§3.2). |
| `--prefill-instance NAME` | the basename of `--prefill-listen` | the arena/socket namespace, so two prefill instances never share one slot table (§3.2). |
| `--prefill-slots N` | `2` | slots **per tier** in the arena (§3.8). |
| `--prefill-slot-tiers LIST` | `1024,16384,131072` | token counts defining the size classes (§3.8). |
| `--prefill-max-tokens N` | `--max-context` | the largest prompt the prefill instance will accept; a bigger one is `ERR toobig` (§4.4). |
| `--prefill-queue N` | `32` | how many jobs the prefill instance will hold; beyond it, `ERR full` (§4.4, §5.2). |
| `--prefill-fallback none\|local` | `none` | may a decode instance read a prompt locally when the prefill instance cannot serve it? §1.5. |
| `--prefill-retries N` | `3` | the prefill analogue of `kMaxRereads` (§4.7, `.megamind/src/program/s38-two-prefill-livelock.md:48`). |
| `--prefill-client-id HEX` | random u64 | the decode instance's identity (§5.1). |

Env switches (the engine's A/B style, read once at startup, documented in `--help` next to
`STRATA_NO_SWAP` at `src/program/generate.cpp:679-696`):

| env | meaning |
|---|---|
| `STRATA_HANDOFF_LEASE_MS N` | how long a `CLAIMED`/`READY` slot survives without its owner (default 300000). §3.6. |
| `STRATA_HANDOFF_DIR PATH` | same as `--prefill-handoff-dir`, for a container that cannot pass flags. |
| `STRATA_PREFILL_TRACE=1` | one stderr line per handoff decision (the `STRATA_SERVE_TRACE` pattern, `src/program/generate.cpp:684`). |
| `STRATA_PREFILL_DISABLE=1` | the decode instance behaves as if the endpoint were down: every new prompt is held, not read. The A/B arm for "is the split the problem?". |
| `STRATA_NO_HANDOFF=1` | **`--role full` behaviour with the new binary compiled in**: the role plumbing exists but the handoff client/server never starts. This is the bit-exactness arm (§7.1), the analogue of `STRATA_NO_SWAP=1` (`include/strata/program/serve_swap.hpp:77-80`). |

`setup.py` surface: `--role`, `--prefill-endpoint`, `--prefill-slots`, `--prefill-slot-tiers`,
`--prefill-max-tokens`, `--prefill-fallback` join `ENGINE_FLAGS` (`setup.py:1162`) and the
`engine_flags` dict (`setup.py:1914-1916`), so they land in the config's `"args"` list and
`serve/server.py` passes them through `engine_args(cfg)` (`serve/server.py:868-874`) without knowing
they exist. **`serve/server.py` needs no new flag for the split.** That is the whole point of D1.

### 1.3 What each role initialises

The engine's start-up order is fixed and the roles skip whole blocks of it. Every row below names the
allocation site that is skipped or kept.

| subsystem | allocated at | `full` | `prefill` | `decode` |
|---|---|---|---|---|
| weights, `WeightTable` | `src/program/generate.cpp` (the `wt` load; 1466 MiB on this box, `strata-iq3_s.log:11`) | yes | yes | yes |
| expert arena (shared tmpfs) | `src/core/pinned.cu:311-565`; 46.84 GiB mapped, `strata-iq3_s.log:18` | yes | yes (borrowed) | yes (borrowed) |
| CPU expert pool | `src/program/generate.cpp:2614` | yes | **no** — see §1.3.1 | yes |
| `PleTable` (26.8 GB mapped shard) | `src/program/generate.cpp:2126-2144`, 320 001 536 rows (`strata-iq3_s.log:15`) | yes | yes | yes |
| CUDA0 session arena | `cudaMalloc(session_bytes(...))` `src/program/generate.cpp:2450`, `session_init` `:2454` | yes | yes | yes |
| per-stage session arenas | `src/program/generate.cpp:2485-2486` | yes | yes | yes |
| KV-streaming authoritative host pools (pinned) | `src/core/layer.cpp:626-654` (`cudaHostAlloc`), reported 6.19 GiB `strata-iq3_s.log:14` | yes | yes | yes |
| `d_mrope` + `mrope_host` | `src/program/generate.cpp:2042-2051`, per stage `:2250-2251` | yes | yes (GENI needs it) | yes (GENI forwards, §1.5) |
| VRAM expert cache `xcache` + per-stage caches | `src/program/generate.cpp:2620`, sized `:2645-2670`; 7773 slots / 14.74 GiB `strata-iq3_s.log:21` | yes | yes, **bigger** (§1.3.2) | yes |
| `NativeHead` (the output head) | `src/program/generate.cpp:2630`; 521 472 000 B `strata-iq3_s.log:19` | yes | **no** (§1.4) | yes |
| `MtpDrafter::load` (draft weights + its K/V) | `src/program/generate.cpp:2507-2519` → `src/core/mtp.cpp:142`; 836 MiB (experts 675, dense 111) `strata-iq3_s.log:16` | yes | **yes** (§1.4 — it is not skippable) | yes |
| `MtpDrafter::bind` (draft head, draft logits, coupled sampling) | `src/program/generate.cpp:4015` → `src/core/mtp.cpp:375-410`; draft head 81.2 MiB `strata-iq3_s.log:35` | yes | **no** — `bind_kv_only` (§1.4) | yes |
| `Verifier` + its captured graphs | `src/program/generate.cpp:4014`, per stage `:3987`/`:3997`; 76.9 MiB of device buffers `strata-iq3_s.log:34` | yes | **no** | yes |
| penalty-history device buffer | `src/program/generate.cpp:3940` (4096 × `kVerifyMaxT` × 4 B) | yes | **no** | yes |
| `Prefill` + the prompt loan (`pf_parts`) | `src/program/generate.cpp:3755`, `sp.init` `:3922`, per stage `:3915`; the loan is 2626 CUDA0 slots = 4.95 GiB `strata-iq3_s.log:33` | yes | **yes** | **no** unless `--prefill-fallback local` (§1.5) |
| `ConversationCache` (parked conversations) | `src/program/generate.cpp:4085` | yes | yes, sized separately (§1.3.3) | yes |
| `SuffixDrafter` | `src/program/generate.cpp:5229` | yes | **no** | yes |
| `DraftPolicy` | `src/program/generate.cpp:5236` | yes | **no** | yes |
| the serve wire (`READY`, stdin thread, watchdog, driver) | `src/program/generate.cpp:5220`, `:4268-4356` | yes | **no** — it speaks `prefill_svc` instead | yes |
| the handoff arena + the control socket | new (§3, §4) | no | yes | yes |

#### 1.3.1 Why the prefill instance still needs the CPU pool — or does it

`Prefill` does **not** use `ExpertPool`. It stages non-resident experts on its own `Stager` threads
(`src/prefill/prefill.cpp:137-250`, `:510`, `:1093`, `:1583`), which is exactly what stage 3's §4.3
rule 4 says (`docs/STAGE3-CONCURRENCY.md:458-461`). So the prefill role skips
`strata::kernels::cpu::ExpertPool pool(...)` (`src/program/generate.cpp:2614`) and the window pool
wrapper (`:4027-4028`).

Consequence, and it is a good one: the prefill instance claims **no cores** from the machine-wide core
lease (`/dev/shm/strata-core-leases`, `strata-iq3_s.log:28`, `docs/DETAILS.md:358-366`), so the decode
instances keep the whole pool. That is the stage-1 collapse (`.megamind/src/kernels/cpu/pool-notes.md:11-17`)
avoided by construction rather than by negotiation.

#### 1.3.2 VRAM consequence of skipping the verifier and the head in the prefill role

On this box, per instance:

| skipped in `--role prefill` | bytes | source |
|---|---|---|
| `Verifier` device buffers, stage 0 | 76.9 MiB | `strata-iq3_s.log:34` (`ver.init` `src/program/generate.cpp:4014`) |
| `Verifier` device buffers, later stages | 75.4 MiB each | `strata-iq3_s.log:302` (`gs.ver.init` `:3997`) |
| `NativeHead` | 521 472 000 B = 497 MiB | `strata-iq3_s.log:19` (`:2630`) |
| MTP draft head (`dhead_` + `dvocab_`) | 81.2 MiB | `strata-iq3_s.log:35` (`src/core/mtp.cpp:390-405`) |
| MTP `head_logits_` (`max_t × n_vocab × 4`) | ≈ 2.4 MiB | `src/core/mtp.cpp:384-388` |
| penalty history | 128 KiB | `src/program/generate.cpp:3935-3940` |
| **total freed** | **≈ 1.2 GiB** | |

`--expert-cache auto` sizes the cache from what is free minus the reserve
(`src/program/generate.cpp:2645-2658`), so that ≈ 1.2 GiB becomes **≈ 900 more expert slots** at
1 382 400 B/blob (`strata-iq3_s.log:20-21`). The prefill instance's hit rate matters: a chunk streams
every expert it routes to that is not resident, so a bigger cache is directly a faster prompt.

Host RAM consequence of the prefill role: it keeps the 46.84 GiB **shared** arena (mapped, not
duplicated — `src/core/pinned.cu:553`, `ArenaExpertSource::borrowed()`), the 26.8 GB PLE mapping, and
6.19 GiB of pinned KV per instance (`src/core/layer.cpp:631`, `strata-iq3_s.log:14`). It does **not**
keep a suffix drafter (~66 MiB, `docs/STAGE3-CONCURRENCY.md:166`) and it does not keep the parked
conversations of the decode instances.

#### 1.3.3 The prefill instance's own conversation cache — the biggest win in the design

Look at the owner's log: prompts are almost entirely **reused**, not read.
`prompt 245946 tokens = 245751 reused + 195 read in 1116 ms` (`strata-iq3_s.log:6649`);
`prompt 222112 tokens = 221911 reused + 201 read in 1144 ms` (`:6449`). The reuse lookup is
`ConversationCache::best()` (`include/strata/core/conversation_cache.hpp:272-278`), called at
`src/program/generate.cpp:6563` and `:6661`.

So the prefill instance **must** have its own `ConversationCache` (the same object, the same budget
flags, `src/program/generate.cpp:4085`), because then a handoff for a continuing chat is:

```
best() hit -> take() the image -> write it straight into a slot -> DONE
```

with **zero GPU work**. The transfer is a tmpfs read (a memcpy out of page cache), which is orders of
magnitude cheaper than the read it replaces. This is the case that dominates the log, and it is why the
split can be a net win rather than a tax.

Its parking budget is a separate `--conversation-cache-mib` on that process. It is *not* the same RAM
as the decode instances' budgets — this is OQ-S4-3.

**The limitation this exposes, stated honestly.** The prefill instance's cache only ever contains what
the prefill instance itself read. The tokens a decode instance *generated* are in the decode
instance's cache and nowhere else. So the next request of the same chat, if it lands on a different
decode instance, is a full re-read by the prefill instance. Closing that needs a **reverse handoff**
(the decode instance hands its completed `SavedConversation` back into the prefill instance's cache,
using the identical slot machinery in the other direction). That is deliberately **out of scope for
S4.3** and is reserved as S4.5 (§9 OQ-S4-4, §10); the protocol line name `TAKE` is reserved for it
(§4.3).

### 1.4 The prefill role's MTP problem — read this before implementing

The obvious plan — "the prefill role skips the drafter" — is **wrong**, and the reason is in the code.

The MTP draft layer's K/V is part of `SavedConversation`: `SavedConversation::kv` is
"main layers followed by the draft layer" (`include/strata/core/conversation_cache.hpp:131`), and
`conversation_snapshot_save` writes and validates it (`src/core/conversation_snapshot.cpp`,
`include/strata/core/conversation_snapshot.hpp:97-100`). If the prefill instance does not produce that
draft K/V, the payload is incomplete and the decode instance must re-derive it — which means reading
the whole prompt again, which defeats the split.

And the draft K/V **is** produced during a prompt read, by the drafter itself:

* `sp.on_chunk` calls `mtp.prefill(R_rows, nxt, T, p0, err)` (or `sp.draft_kv`, the batched form) at
  `src/program/generate.cpp:4412-4416`;
* `MtpDrafter::prefill` (`src/core/mtp.cpp:679`) runs `capture_prefill` / `capture_prefill_dev`
  (`:600-616`), which call `record_forward(T, -1, …)` — `full = false` — and
  `record_forward` **returns immediately after the K/V append** (`src/core/mtp.cpp:470-486`, `if (!full) return true;` at `:486`).

So the prompt path needs only the drafter's *forward-to-K/V* half. What it does **not** need:
`draft`, `draft_first`, `round_exec_[]`, `step_exec_[]`, the coupled-draft sampler, and the draft head.

**Contract for S4.3.** Add `MtpDrafter::bind_kv_only(const WeightTable& wt, std::string& err)` in
`include/strata/core/mtp.hpp` / `src/core/mtp.cpp`. It sets `wt_` (required: `record_forward` reads
`wt_->find("token_embd.weight")` at `src/core/mtp.cpp:428-430`) and nothing else — no `head_`, no
`window_R_`, no `head_logits_`, no `dhead_`/`dvocab_`, no `setup_coupled`. `capture_prefill_dev`
touches only `pf_dev_`, `tok_`, `step_`, `pos_`, `Rin_`, `st_` and the Q8_0 MMVQ scratch
(`src/core/mtp.cpp:611-616`, `:431-484`), all of which `load()` already allocated.
`prefill_exec_[T]` / `prefill_dev_exec_[T]` are captured lazily and are the only graphs the prefill
role ever builds.

The prefill role therefore **keeps** `MtpDrafter::load` (836 MiB, `strata-iq3_s.log:16`) and **skips**
`MtpDrafter::bind`'s head allocations. `MtpDrafter::prefill` also needs `set_prompt_len()`
(`include/strata/core/mtp.hpp:44`) to be called with the prompt length, because `first_needed`
(`include/strata/core/mtp.hpp:96`, `src/core/mtp.cpp:684`) skips cells the attention window can never
reach again — and that skip is part of what makes the draft K/V byte-identical to today's. Call it at
the same point the serve path does (`src/program/generate.cpp:6787`).

If `bind_kv_only` is not implemented, the fallback is to call the existing `bind()` in the prefill
role, which **requires** `NativeHead` to be loaded (`src/core/mtp.cpp:383`: "mtp: the draft layer
needs the native head (--native)") and costs 497 MiB + 81 MiB of VRAM for nothing. OQ-S4-9.

### 1.5 The decode role's prompt path

`--role decode` replaces one thing: the batched prompt read inside `run_prefill_step`
(`src/program/generate.cpp:7008`, `sp.run(ids.data() + at, to - at, at, err)`).

The decode instance's request machine gains one phase, in
`include/strata/program/serve_driver.hpp`'s `Phase` enum (`:83-89`):

```
Phase::await_prefill   // between queued and prefill: the payload is coming from outside
```

and one step, in `Step` (`:53-63`):

```
Step::await_prefill    // non-blocking: poll the handoff client, never block the engine thread
```

The rule that keeps the engine thread honest: **the GB-scale copy-out of a slot never happens on the
engine thread.** A dedicated handoff client thread (the decode instance's only new thread) does
`map → read into a SavedConversation → ACK`, and hands the finished object to the engine thread as
`mount_image` — the exact `std::optional<SavedConversation>` the swap already takes
(`.megamind/src/program/s31d-swap-notes.md:21-23`). The engine thread then does what it already does:
validate, mount, `mtp.kv_restore`, adopt, device state (`include/strata/program/serve_swap.hpp:402-419`).

What the decode role keeps, and why:

* **its own `ConversationCache`** — a continuing chat on the *same* decode instance must stay a local
  mount with no handoff at all. `prep_request`'s existing `conversations.best(...)` lookup
  (`src/program/generate.cpp:6563`) runs **first**; only a miss (or a hit shorter than the client's
  prompt by more than `--short-read`) sends a `REQ`. This keeps stage 2's behaviour and stage 3's
  swap behaviour byte-identical for the common single-instance case.
* **the verify-window short-read path** (`windows_ok` / `read_windows`,
  `src/program/generate.cpp:5258-5266`) — it is the decode path, not the prompt path.
* **`Prefill` and the prompt loan only if `--prefill-fallback local`** (`src/program/generate.cpp:3755`,
  `:3915`, `:3922`). With the default `none`, the decode instance never lends cache slots, which means
  its expert cache stays whole for the whole session — the 4.95 GiB loan (`strata-iq3_s.log:33`) and
  the lazy-return pump (`src/program/generate.cpp:4590-4640`) simply do not exist there. That is a
  decode-side win worth having.
* **GENI (images) still works**: the decode instance forwards the embeddings file path in the `REQ`
  line (§4.3) and the prefill instance reads it. The prefill instance must be started with `--vision`
  for this; a `GENI` to a prefill instance without it is `ERR novision` (permanent, §4.4). The
  M-RoPE table (risk R7, `docs/STAGE3-CONCURRENCY.md:1231`) is **not** part of the payload — the
  decode instance rebuilds it from the payload's `live.imgs` on mount, exactly as `swap_to`'s
  `device_state` step does today (`include/strata/program/serve_swap.hpp:411`,
  `src/program/generate.cpp:5607-5628`).

`--prefill-fallback local` semantics: when the prefill instance answers `ERR full`, the socket is
down, or the payload is `toobig`, the decode instance runs the prompt itself through the path it kept.
With `none` (the default) the request instead enters the S4.2 wait queue with reason `prefill` (§6).

### 1.6 Invariants (and the one place this narrows D3)

1. **`--role full` + `--serve-slots 0` is byte-identical to 0.1.30 on the wire and in the tokens.**
   Proven the way stage 3 proved it: the serial driver's text is unchanged and the serial-reachable
   output-call count is unchanged (`.megamind/src/program/s31e2-driver-notes.md:17-23`,
   `.shz_cmd/s31e2_serial_proof.py`, `.shz_cmd/s31e2_output_proof.py`).
2. **`include/strata/program/serve_proto.hpp` is not edited by S4.3.** The client wire cannot drift
   (D3, `.megamind/stage4-plan.md:42-45`). The only client-visible addition in stage 4 is the `WAIT`
   line's reason vocabulary, which S4.2 already owns (`include/strata/program/serve_driver.hpp:1147-1159`).
3. **`setup.py` and `serve/server.py` stay the only client-facing surfaces** (D1). The split adds
   engine flags, not HTTP surface.
4. **Narrowing of D3, stated explicitly:** D3 says the control protocol "mirrors the existing wire".
   It does **not** reuse it, and it must not: the prefill↔decode protocol has no `GEN`, no `T`, no
   `DONE`-with-ten-fields, and its `READY` token is a different token (`PREFILL-READY-V1`, §4.2).
   "Mirrors" here means *the same style* — one line per message, `KEYWORD` first, space-separated
   tokens, unknown trailing tokens ignored, versioned by a `READY` token, default off. The
   `READY`-token-versioning mechanism is copied; the vocabulary is not shared. This is deliberate: a
   shared vocabulary would mean one parser serving two protocols, which is exactly the drift D3
   forbids.
5. **One engine thread per process, always.** The prefill instance's engine thread is the only thing
   that calls `Prefill::run`; the decode instance's is the only thing that calls `Verifier::run`
   (`docs/STAGE3-CONCURRENCY.md:283-287`, risk R5). The handoff client thread on the decode side does
   host memcpy and socket I/O only — it never touches a device.

### 1.7 The compatibility matrix

The stage-3 matrix (`docs/STAGE3-CONCURRENCY.md:643-653`) still holds and is not re-derived here. This
is the stage-4 extension of it.

| server (`serve/server.py`) | engine | role | behaviour |
|---|---|---|---|
| old | 0.1.30 | n/a | unchanged |
| old | new | `full` | unchanged. `--serve-slots 0` ⇒ no `slots=` in `READY`, no tags, no `WAIT` line — byte-identical to 0.1.30 (`serve_proto.hpp:20-23`) |
| old | new | `full`, `--serve-slots >= 2` | stage 3's matrix row: the server sees no `slots=`, uses a semaphore of 1 and id-less requests (`serve/server.py:404-412`) |
| new | new | `full` | stage 3's concurrency + S4.2's hold queue. **No handoff, no arena, no socket.** This is today |
| new | new | `decode` + `--prefill-endpoint` | the split. The server sees **nothing new on its wire**: the same `READY … slots=N`, the same tagged lines, one extra `WAIT … prefill-busy` reason it already treats as temporary (`serve/server.py:183-189`) |
| new | new | `decode`, endpoint down, `--prefill-fallback none` | every **new** prompt is held (visible on `/status`, `/slots`, `/metrics`); conversations already running keep running. No client-visible `ERR` until `--hold-ms` |
| new | new | `decode`, endpoint down, `--prefill-fallback local` | degrades to today's behaviour, with a startup warning |
| new | new | `prefill` | **no `serve/server.py` at all.** It is a backend (D1, `.megamind/stage4-plan.md:6-7`). Starting the HTTP server against it is an operator error, and it will fail at the `READY` handshake because a prefill instance never prints `READY` |
| any | new | `full` with `STRATA_NO_HANDOFF=1` | the bit-exactness arm: the new binary, the role plumbing compiled in, the handoff machinery never started (§1.2) |
| any | old | `--role …` | `setup.py` refuses the flag before writing a config the engine would reject, through the existing `engine_supports` probe (`setup.py:1165-1202`); a hand-written config gets the engine's own `unknown argument: --role` and exit 2 (`src/program/generate.cpp:1406-1410`) |

Two rules fall out of the matrix and are worth stating because they are the ones an implementer can
get wrong:

* **`--serve-slots 0` changes nothing in this stage either.** The handoff is orthogonal to the slot
  count: a `--role decode --serve-slots 0` engine is legal (one conversation, prompt read elsewhere),
  and a `--role full --serve-slots 0` engine is 0.1.30.
* **The client wire gains exactly one token in stage 4**: the `WAIT` line's `prefill-busy` reason
  (§6.1). Nothing else a client parses changes.

---

## 2. The hand-off payload

### 2.1 The payload is `SavedConversation`, field by field

`include/strata/core/conversation_cache.hpp:124-162`:

| field | line | in the payload? | notes |
|---|---|---|---|
| `std::array<int64_t,18> geometry` | `:126` | yes | written by `geometry_key(g)` (`src/core/conversation_state.cpp:14-19`), compared at `:433` |
| `int64_t layer_lo, layer_hi` | `:128` | yes | the carve the image came from; restore requires the same (`conversation_cache.hpp:127`) |
| `ConversationCheckpoint live` | `:129` | yes | the live branch: ids, imgs, gdn/ple/tails/dead/block_pos, `used` |
| `std::vector<ConversationCheckpoint> checkpoints` | `:130` | yes | the prefix chain; each carries its own `stage_parts` (`:67`) |
| `std::vector<ConversationKv> kv` | `:131` | yes | **main layers, then the draft layer** — the order is part of the format |
| `std::vector<ConversationStageSnapshot> stage_parts` | `:133` | yes when `--layer-split` | one per **later** stage (`:100-111`) |
| `bool cvec` | `:134` | yes | the control-vector state the branch was read with; the cache keys on it (`:301`, `:418`) |
| `int64_t owner` | `:152` | **yes, but remapped** | §2.4 |

`ConversationCheckpoint` (`:57-76`): `ids` (`vector<int32_t>`), `imgs`
(`vector<ConversationImageKey>{int64_t start; uint64_t hash}`, `:51-55`), five
`ConversationBuffer`s (`gdn`, `ple`, `tails`, `dead`, `block_pos`), `used` (the chain's LRU stamp), and
`stage_parts` (recursive, same type).

`ConversationKv` (`:80-87`): `int format`, `int64_t cells, heads, head_dim, page_size, pooled_rows,
idx_dim`, and five `ConversationBuffer`s (`k`, `v`, `k_scale`, `v_scale`, `pooled`). Its layout is
computed by `layout()` (`src/core/conversation_snapshot.cpp:27-67`) and checked field-by-field by
`conversation_kv_validate` (`:175-194`) — that is the validate-before-write path the decode instance
reuses (§2.5).

`ConversationStageSnapshot` (`:112-122`): `layer_lo`, `layer_hi`, `ConversationCheckpoint state`,
`std::vector<ConversationKv> kv` — exactly `qsa_alloc` entries, never a draft entry (`:105-107`).

### 2.2 What is flat/POD-safe and what is not

**Not safe to memcpy as-is:**

* `std::vector` and `std::deque` members everywhere (`ConversationCheckpoint`, `SavedConversation`,
  `ConversationStageSnapshot`) — pointers into the writer's heap.
* `ConversationBuffer` (`include/strata/core/conversation_buffer.hpp:16-152`) — a
  `std::vector<std::vector<uint8_t>>` segment directory with a 16 MiB segment cap (`:18`) and
  geometric growth (`:126-129`). It has **no `data()`** and no contiguous payload
  (`.megamind/src/core/s31a-parking-split-notes.md:63`). The only accessors are `size()`, `resize()`,
  `visit(offset,count,fn)`, `read(target,offset,count)`, `operator==`.
* `std::array<int64_t,18>` is POD — fine.
* `bool` — do **not** put a `bool` in a wire struct; use `int8_t` (size and sign of `bool` are
  implementation-defined).

**Therefore the wire format is a record stream, not a struct dump.** Every scalar is written at a fixed
width and fixed alignment; every `ConversationBuffer` becomes a `{bytes, offset}` reference into one
blob pool appended after the headers. The reader rebuilds a real `SavedConversation` by
`resize()`+`visit()`-copying each blob into freshly-owned segments. That is the same shape as the
snapshot's own save path (`conversation_snapshot.cpp:160-171` walks buffers with `visit`), so no new
copy primitive is needed.

### 2.3 The serialization

Inside one slot, after the 4096-byte slot header (§3.3), the payload is:

```
payload := header_record body_records blob_pool
```

**`header_record`** — **232 bytes**, padded to 256, little-endian, no padding holes (the field list
below is the layout; a `static_assert(sizeof(...) == 232)` pins it):

```
u32  magic          'S','R','4','H'                       (0x53 52 34 48)
u16  fmt_version    = 1
u16  flags          bit0 = cvec, bit1 = has_stage_parts, bit2 = partial (§3.5)
u64  geometry[18]   the array from SavedConversation::geometry
u64  layer_lo
u64  layer_hi
i64  n_kv           SavedConversation::kv.size()          (main layers + draft)
i64  n_checkpoints  checkpoints.size()
i64  n_stage_parts  stage_parts.size()                    (0 = single-GPU image)
i64  live_tokens    live.ids.size()
u64  blob_pool_off  offset (from payload start) of the blob pool
u64  blob_pool_bytes
u64  geom_hash       the FNV-1a over the geometry array + the split spec (§3.10)
u64  pack_hash      the arena's pack identity (§3.10)
```

**`body_records`** — a fixed order, so a reader never has to search:

```
1. KV array            n_kv entries, in SavedConversation::kv order (main layers, then the draft layer)
2. live                one CHECKPOINT record
3. checkpoints         n_checkpoints CHECKPOINT records, in vector order
4. stage_parts         n_stage_parts STAGE records, in engine order
5. END                 u32 magic 'E','N','D','!'
```

A record is `{u32 kind; u32 flags; u64 bytes}` followed by its body:

```
kind 1  KV
   i32 format; i64 cells; i64 heads; i64 head_dim; i64 page_size; i64 pooled_rows; i64 idx_dim
   BLOBREF k, v, k_scale, v_scale, pooled
kind 2  CHECKPOINT
   i64 n_ids; i32 ids[n_ids] (4-byte aligned)
   i64 n_imgs; {i64 start; u64 hash}[n_imgs]
   BLOBREF gdn, ple, tails, dead, block_pos
   u64 used
   i64 n_stage_parts; CHECKPOINT[n_stage_parts]     (recursive, same kind; imgs empty in a stage part,
                                                      conversation_cache.hpp:64-66)
kind 3  STAGE
   i64 layer_lo; i64 layer_hi
   CHECKPOINT state                                  (its ids = the live tokens, its imgs empty,
                                                      conversation_cache.hpp:105-106)
   i64 n_kv; KV[n_kv]                                (exactly qsa_alloc entries, never a draft entry,
                                                      conversation_cache.hpp:107)
kind 4  END
```

`BLOBREF` = `{u64 bytes; u64 offset}`, `offset` relative to the payload start; a zero-length buffer is
`{0, 0}` and has no pool entry. **Every blob in the pool starts at a 64-byte boundary** relative to the
slot start (so the reader's `memcpy` is aligned and the writer can `pwrite` in whole cache lines).

The writer computes the pool offsets in a first pass (every size is already known:
`ConversationBuffer::size()`), writes the headers, then writes the blobs with `pwrite` from the slot's
mapping or the fd. The reader does the reverse: `resize()` each `ConversationBuffer` and copy through
`visit()` (`conversation_buffer.hpp:94-105`).

**What is deliberately not in the payload:** the suffix-drafter history, the sampling parameters, the
M-RoPE table, the checkpoint LRU clock, and `mtp.set_prompt_len` — the same list
`serve_swap::SlotConv` documents as "what the image CANNOT carry"
(`include/strata/program/serve_swap.hpp:91-95`). The decode instance re-establishes all of them at
dispatch, exactly as a scheduler-driven `swap_to(..., restore_positions = true)` does
(`.megamind/src/program/s31d-swap-notes.md:24-28`).

### 2.4 `owner` across processes

`SavedConversation::owner` is a **process-local claim**, not a global identity: it is the request id of
the slot that parked this branch, and `ConversationCache::claimed_by_other()` compares it against a
`requester` id from the same process (`conversation_cache.hpp:320-323`, and the whole S3.9 story in
`.megamind/src/program/s39-parked-branch-theft.md`).

Rule, stated exactly:

* **the prefill instance writes `owner = kNoOwner` (−1) into the payload, always.** The payload is a
  copy leaving the process; a claim that crossed processes would be a lie about who can release it
  (`release_owner` runs from `drop_ctx`, `conversation_cache.hpp:328-330`, and no `drop_ctx` exists in
  the other process).
* **the decode instance assigns the claim itself, at `put()` time**, with its own request id:
  `conversations.put(std::move(image), /*owner=*/req_id, /*held=*/…)`
  (`conversation_cache.hpp:388-396`). This is what makes S3.9's theft protection work unchanged for a
  received image: two decode requests that share a prefix cannot both mount the one received branch.
* **request ids are per-decode-instance and must not be assumed globally unique.** The control
  protocol therefore always pairs a request id with the client id: the slot header carries both
  `client_id` and `request_id` (§3.3), and every protocol line names the slot (§4.3). The decode
  instance's own `Registry` ids are its business (`include/strata/program/slot.hpp:131-133`).
* The `GEOM` header's `owner` field is written as `kNoOwner` and **must be ignored by the reader** —
  reading it would be the S3.9 bug in a new costume.

### 2.5 Validate before mounting

The decode instance never mounts an unvalidated image. The existing machinery already gives a
validate-before-write path, and the handoff uses it rather than inventing one:

1. **slot-level, before copying anything out** — the slot header's `state == READY`, `nonce` matches
   the `DONE` line, `pack_hash` and `geom_hash` match this process's (§3.10). A mismatch is
   `ERR geom` / `ERR pack` and the slot is released.
2. **payload-level, after copy-out, before the cache** — the `header_record`'s `geometry[18]` must
   equal `geometry_key(g)` (`src/core/conversation_state.cpp:433`: "incompatible runtime geometry"),
   and `stage_parts.size()` must equal this process's stage count in **both** directions
   (`.megamind/src/core/s31a-parking-split-notes.md:36-38`: a split image mounted without a stage set
   is refused; a single-GPU image mounted with one is refused).
3. **session-level, on the engine thread, before the first write** —
   `conversation_snapshot_validate(image, ss, conv_stages, g, mtp.kv_state(), err)`
   (`include/strata/core/conversation_snapshot.hpp:136`), which is exactly the `validate` step of
   the canonical hand-over order and runs *before* `unmount` so a bad image costs nothing
   (`include/strata/program/serve_swap.hpp:402-419`, `:543-575`).
4. If any of these fail, the request is **not** mounted: with `--prefill-fallback local` it re-reads
   locally, otherwise it is answered `ERR badpayload` (permanent — the same payload will fail again).

A restore that fails *after* its first write is `transfer_failed` and is fatal to the session; that is
0.1.30's rule and this stage does not soften it
(`include/strata/core/conversation_snapshot.hpp:104-108`,
`include/strata/program/serve_swap.hpp:470-471`).

---

## 3. The transport — the tmpfs handoff arena

### 3.1 Why this shape

Measured payload sizes on this box (Appendix A.2, 871 parks in `strata-iq3_s.log`):

| | bytes | GiB |
|---|---|---|
| min | 237 413 652 | 0.22 |
| p50 | 1 033 605 768 | 0.96 |
| p90 | 2 029 940 496 | 1.89 |
| p99 | 2 939 786 284 | 2.74 |
| max | 3 180 486 108 | 2.96 |

and the cost of moving one, from the engine's own log lines:

* a park of 3 180 486 108 B took **410 ms** (`strata-iq3_s.log`, `parked 153983 tokens in 410.0 ms`);
* a swap that saved 2 785 539 216 B took **220-313 ms** (`swap N: … ok in 220 ms; saved 2785504484 B`);
* a restore of 97 934 tokens took **310 ms** (`conversation cache: restored 97934 tokens (checkpoint) in 310.1 ms`).

So the payload is 0.2-3 GB and moves in a few hundred ms **inside one process**. Across processes the
only sane carrier is a shared page-cache-backed file: `MAP_SHARED` on tmpfs, which is what stage 1
already does for 46.84 GiB (`src/core/pinned.cu:553`, `strata-iq3_s.log:18`). A pipe would serialise,
copy twice, and have a 64 KiB buffer. **Never stream GiB through a pipe**
(`.megamind/stage4-plan.md:37-40`).

### 3.2 Files

```
--prefill-handoff-dir = /dev/shm/strata-handoff            (default)

/dev/shm/strata-handoff/
    <pack16>-<split-spec>/
        <instance>/
            arena            the slot table + the slots (§3.3)
            prefill.sock     the AF_UNIX control socket (§4.2)
            req-<client>-<nonce>.ids     a request's prompt ids (§4.3)
```

`<pack16>` is the arena's `pack_hash` as 16 lowercase hex digits — the same identity stage 1 computes
(`shared_arena_pack_hash`, `src/core/expert_source.cpp:1363-1380+`; currently internal to that TU, so
S4.3 must expose it or wrap it). `<split-spec>` is the layer-split description as a canonical string
(§3.10). One directory per (pack, split) pair, because the payload's *shape* depends on both:
`SavedConversation::stage_parts` is per later stage (`conversation_cache.hpp:100-111`) and the K/V
array is main-layers-then-draft (`:131`).

`<instance>` is `--prefill-instance NAME`, defaulting to the basename of `--prefill-listen`. **One
arena per prefill instance, always.** The slot table has exactly one writer (§3.5), and two prefill
instances sharing one arena would be two writers over the same bytes — the torn-arena bug stage 1
exists to prevent (`src/core/pinned.cu:347-351`). Running two prefill instances to get parallelism
therefore means two directories, two sockets, two arenas; the decode instance picks one endpoint
(`--prefill-endpoint`) and that is the whole of the multi-prefill story (OQ-S4-5).

The prefill instance **creates** the directory, the arena and the socket at start-up. The decode
instance **opens** them (it never creates, never `ftruncate`s). Mode `0600`, like the expert arena
(`src/core/pinned.cu:341`).

### 3.3 The arena file header — exact layout

4096 bytes at offset 0. `inline constexpr` offsets in the header, `static_assert`ed against the struct
in the .cpp — the exact pattern of `include/strata/core/pinned.hpp:51-56` and
`src/core/pinned.cu:64-71`.

```
offset   0  char[16]  magic      "STRATA-HARENA-V1"
offset  16  u32       version    = 1
offset  20  u32       header_bytes = 4096
offset  24  u32       tier_count                     (1..8)
offset  28  u32       pad
offset  32  u64       slot_total                     sum of tier_slot_count[i]
offset  40  u64       arena_bytes                    the whole file
offset  48  u64       pack_hash
offset  56  u64       geom_hash            (§3.10)
offset  64  u64       created_pid
offset  72  u64       created_start        /proc start time of created_pid
offset  80  u64       tier_max_tokens[8]        0 beyond tier_count
offset 144  u64       tier_slot_count[8]
offset 208  u64       tier_slot_bytes[8]        payload capacity after a slot header
offset 272  u64       tier_slot_off[8]          byte offset of tier i's first slot group
offset 336  u64       reserved[8]
```

and the file is

```
arena := arena_header(4096)
         + for each tier i, tier_slot_count[i] * ( slot_header(4096) + tier_slot_bytes[i] )
```

**Slots are grouped by tier and sized per tier**, not all sized to the largest. That is the difference
between an 8 GiB arena and an 18 GiB one at the proposed default (§3.8), and it costs the reader
nothing: a slot's size is `tier_slot_bytes[header.tier]`, and `tier` is in the slot header (§3.4).
`tier_slot_off[i]` is written once by the creator and is the only way to find a group; a reader that
computes it independently has to agree with the creator about ordering, and agreeing about arithmetic
is how protocol bugs are born.

### 3.4 The slot header — exact layout

4096 bytes at the start of each slot.

```
offset   0  char[16]  magic         "STRATA-HSLOT-V1"
offset  16  u32       version       = 1
offset  20  u32       header_bytes  = 4096
offset  24  u32       state         0 FREE, 1 CLAIMED, 2 READY
offset  28  u32       flags         bit0 cancelled, bit1 abandoned-by-client, bit2 partial-payload
offset  32  u32       tier          index into tier_max_tokens
offset  36  u32       pad
offset  40  u64       owner_pid     the prefill pid that wrote it, 0 = nobody
offset  48  u64       owner_start   its /proc start time, 0 = unknown
offset  56  u64       nonce         identifies THIS handoff attempt (§3.5)
offset  64  u64       client_id     the decode instance allowed to read it, 0 = none
offset  72  u64       request_id    the decode instance's request id
offset  80  u64       pack_hash
offset  88  u64       geom_hash
offset  96  u64       tokens        live branch tokens
offset 104  u64       payload_bytes bytes written after this header
offset 112  u64       payload_hash  FNV-1a over [4096, 4096+payload_bytes)
offset 120  u64       claimed_ms    monotonic ms since boot
offset 128  u64       published_ms  0 until published
offset 136  u64       seq           segment sequence (v1: always 1)
offset 144  u64       reserved[7]
```

```cpp
static_assert(sizeof(HandoffSlotHeader) <= 4096);
static_assert(kSlotStateOffset   == offsetof(HandoffSlotHeader, state));
static_assert(kSlotOwnerOffset   == offsetof(HandoffSlotHeader, owner_pid));
static_assert(kSlotNonceOffset   == offsetof(HandoffSlotHeader, nonce));
static_assert(kSlotOwnerStartOffset == offsetof(HandoffSlotHeader, owner_start));
static_assert(kSlotClientOffset  == offsetof(HandoffSlotHeader, client_id));
```

`fnv1a64` is the engine's existing FNV-1a (`include/strata/core/pinned.hpp:173`,
`src/core/pinned.cu:594-601`) — reuse it, do not write a second one.

### 3.5 The lifecycle

One writer (the prefill instance), one named reader (one decode instance), and the slot table is
**owned by the prefill instance**. The decode instance never writes a slot. This is the same
single-writer rule that makes the expert arena work (`src/core/pinned.cu:347-351`).

```
FREE --(claim under flock)--> CLAIMED --(write body, release fence, pwrite state)--> READY --(ACK)--> FREE
```

1. **claim.** The prefill instance takes the arena's `flock(LOCK_EX|LOCK_NB)` — the same retry/give-up
   shape as `arena_lock` (`src/core/pinned.cu:111-128`: retry every 20 ms, refuse after 10 s rather
   than start on an arena whose state cannot be read). Under the lock it reads the slot header,
   requires `state == FREE` **or** a dead/abandoned owner (§3.6), then `pwrite`s
   `{state=CLAIMED, owner_pid, owner_start, nonce, client_id, request_id, tier, claimed_ms,
   pack_hash, geom_hash, payload_bytes=0, flags=0}`. Then it drops the lock. **The lock covers the
   decision only, never the write** — the same rule and the same reason as
   `src/core/pinned.cu:347-351`.
2. **write.** `pwrite` the payload (headers then the blob pool) at `slot_base + 4096`. No lock held.
   The slot's `state` is still `CLAIMED`, so a reader that stumbles in sees "not populated", never a
   half-written body — the exact fix stage 1 made for the expert arena
   (`src/core/pinned.cu:381-383`).
3. **publish.** `std::atomic_thread_fence(std::memory_order_release)`, then a **single `pwrite` of the
   8-byte `state` field** at `slot_base + kSlotStateOffset`, with `payload_bytes`, `payload_hash` and
   `published_ms` written in the same 32-byte `pwrite` as `state` (one ordered store, not a page write
   another process can see half-done — the reason `publish_shared_load` writes through the fd rather
   than the mapping, `include/strata/core/pinned.hpp:122-126`).
4. **read.** The decode instance `open`s the arena `O_RDONLY|O_CLOEXEC`, takes a shared `flock`
   (`LOCK_SH`) only to read the header, checks `state == READY && client_id == me && nonce == the
   DONE line's nonce`, then `mmap(PROT_READ, MAP_SHARED)` the slot and copies out. It never holds the
   lock across the copy.
5. **release.** The decode instance sends `ACK` (§4.3). The prefill instance, under the arena lock,
   `pwrite`s `{state=FREE, owner_pid=0, client_id=0, nonce=0, payload_bytes=0, flags=0}`. It does
   **not** zero the body: the next writer overwrites it, and zeroing 3 GB to free a slot would make
   the free path slower than the write path.

`O_CLOEXEC` matters for the same reason it does in stage 1: the flock lives on the open file
description, and an inherited fd across an `exec` would keep a claim alive forever
(`src/core/pinned.cu:336-340`).

### 3.6 Dead owners and recycled pids

Copy stage 1's answer exactly (`src/core/pinned.cu:141-194`):

* `pid_alive(pid)`: `stat("/proc/<pid>")`; when `/proc` itself is not mounted the answer is **unknown,
  and unknown means ALIVE** (`src/core/pinned.cu:141-152`). Guessing "dead" is what lets two processes
  write the same bytes.
* `proc_start_time(pid)`: field 22 of `/proc/<pid>/stat`, counted from the **last `)`** because the
  comm field is parenthesised and may contain spaces (`src/core/pinned.cu:161-185`).
* `claim_alive(pid, start)`: both must match; `start == 0` or an unreadable `/proc` is treated as a
  match (`src/core/pinned.cu:188-194`).

A slot whose owner is dead or whose pid was recycled may be taken over by the prefill instance. A slot
whose **reader** (the decode instance) is dead is reclaimed by the lease rule below.

**Leases.** Every state has a bound, measured with a monotonic clock written into the header:

| state | bound | who reclaims |
|---|---|---|
| `CLAIMED` | `STRATA_HANDOFF_LEASE_MS` (default 300 000 ms) from `claimed_ms`, **or** the owner is dead | the prefill instance, at the top of every claim attempt and on its own idle tick |
| `READY` | same, from `published_ms`, **or** `client_id`'s pid is dead | the prefill instance |

Reclaiming a `READY` slot whose client is gone is the **slot-leak fix**: without it, a decode instance
that dies after `DONE` but before `ACK` strands a multi-GB slot until the prefill instance restarts.
The reclaim is a decision, so it happens under the arena lock, and it sets `flags |= abandoned`
before `state = FREE` so the log can say what it reclaimed and why.

### 3.7 Free space before `ftruncate`

Non-negotiable, and the reason is a dead process with no message:

> `ftruncate` on tmpfs does **not** fail when the space is not there — it happily makes a sparse file,
> and the process then dies of SIGBUS on the first write past the limit, with no message.
> (`src/core/pinned.cu:320-326`)

So the prefill instance runs the `shared_file_fits` check (`src/core/pinned.cu:220-239`, `statvfs` on
the parent directory, `f_bavail × f_frsize`) **before** `ftruncate`, and only for a fresh file (an
existing file of the right size has its space committed — re-checking would refuse a working arena
because something else filled the filesystem afterwards, `src/core/pinned.cu:328-330`).

If the arena does not fit, the prefill instance **refuses to start** with both numbers on one line
(unlike the expert arena, which falls back to a private mapping — there is no fallback for a handoff):

```
strata prefill: the handoff arena /dev/shm/strata-handoff/<…>/arena needs 8804655104 B
  (tiers 1024x2, 16384x2, 131072x2) and /dev/shm has 25769803776 B free; raise
  --prefill-slots-per-tier, lower --prefill-slot-tiers, or give /dev/shm more room
```

Measured on this box: `/dev/shm` is 70 GiB total, 47 GiB used by `shared_experts.dat`, **24 GiB
available** (`df -h /dev/shm`, read-only). `Mem: 94 GiB total, 3 GiB available` (`free -g`). tmpfs
pages are RAM, so the arena's size is a RAM decision — OQ-S4-2.

### 3.8 Sizing — machine-independent

**The sizing rule, in one sentence: the arena is sized from the model's geometry and the tier token
counts, never from this machine's RAM.** RAM only decides whether the arena *fits*, and that is the
§3.7 check.

The per-payload size is not guessed. The prefill instance computes it with the engine's own function:

```
conversation_snapshot_bytes(view, session, conv_stages, g, mtp.kv_state(), bytes, err)
    include/strata/core/conversation_snapshot.hpp:125-127
```

with a synthetic `ConversationView` whose `ids` has exactly `tier_max_tokens[i]` entries. That is
host-side arithmetic — `layout()` (`src/core/conversation_snapshot.cpp:27-67`),
`conversation_session_sizes` (`src/core/conversation_state.cpp`), and
`ConversationBuffer::allocation_peak` (`conversation_buffer.hpp:47-61`) — with no CUDA call, so it can
run at start-up and in a CPU test. It is the same function the parking budget already uses
(`docs/STAGE3-CONCURRENCY.md:1067-1072`).

```
tier_slot_bytes[i] = round_up_64( conversation_snapshot_bytes(tier_max_tokens[i]) ) + 64 MiB slack
arena_bytes        = 4096 + sum over i of tier_slot_count[i] * (4096 + tier_slot_bytes[i])
```

The 64 MiB slack covers the record headers, the blob-pool alignment padding, and the segment-directory
overhead `ConversationBuffer::bytes()` counts (`conversation_buffer.hpp:40-44`).

**Proposed default tiers** (`--prefill-slot-tiers 1024,16384,131072`, `--prefill-slots 2`), priced
with the measured rate of ~20.7-22.5 KiB per context token plus the ~236 MiB fixed floor
(Appendix A.2 — the floor is real: an 81-token conversation parks at 237 MB):

| tier tokens | priced payload | + 64 MiB slack | × 2 slots |
|---|---|---|---|
| 1 024 | ≈ 258 MiB | 322 MiB | 644 MiB |
| 16 384 | ≈ 604 MiB | 668 MiB | 1.31 GiB |
| 131 072 | ≈ 3.05 GiB | 3.11 GiB | 6.22 GiB |
| **arena total** | | | **≈ 8.2 GiB** |

For reference, the same arithmetic at the config's real limits:

| max tokens | priced payload | + slack, × 2 slots |
|---|---|---|
| 262 144 | ≈ 5.6 GiB | 11.4 GiB |
| 524 288 | ≈ 11.3 GiB | 22.8 GiB |

so a single 524 288-token tier with 2 slots needs **22.8 GiB** of tmpfs. That fits the 24 GiB currently
free on this box with nothing to spare, and it is the number the owner must decide on (OQ-S4-2).
**Do not "pin" the design to this machine's RAM** (`.megamind/stage4-plan.md:86`).

**When the arena is full.** A full arena is a *temporary* condition (a slot frees when its client
ACKs), so it is never an `ERR` on the client's wire (D5, `.megamind/stage4-plan.md:55-60`). The
prefill instance answers the decode instance with `QUEUED` (§4.3), the decode instance puts the request
in its own hold queue with reason `prefill` (§6), and it re-asks. It becomes an error only through the
hold bound (`--hold-ms`, default 600 000 ms, `include/strata/program/serve_driver.hpp:1109`) or
`STOP <id>`.

**Tier starvation.** A big job must not be starved by small ones forever, and a small job must not be
able to eat the only big slot. Rule: a job of tier `i` may claim a free slot in group `j` for any
`j >= i`, **preferring the smallest such `j`** (so a 1 024-token job does not sit in a 3 GB slot while
a 131 072-token job waits), and claims are served in `REQ` arrival order (FIFO, the same reason as
`include/strata/program/serve_driver.hpp:1163-1177`). If the head of the queue cannot get a slot but a
later job of a smaller tier can, the smaller job **waits** rather than jumping the queue — jumping is
how a big prompt never runs on a busy box. This is a decision, not a knob.

### 3.9 Windows, not streams

A slot is written once and read once. There is no incremental "append more K/V" in v1: `seq` is
reserved (`§3.4`) and must be `1`; a `DONE` with `seq != 1` is `ERR geom`. A prompt bigger than the
largest tier is refused up front (`ERR toobig`, priced before any GPU work, §4.4) rather than split
across slots — splitting a `SavedConversation` across slots means splitting each layer's K/V by cell
range and re-joining it on the other side, which is a second format and a second set of bugs.

### 3.10 The hashes, and refusal

Two hashes, both u64 FNV-1a (`include/strata/core/pinned.hpp:173`):

* **`pack_hash`** — the pack identity, exactly the value the expert arena already stores
  (`shared_arena_pack_hash`, `src/core/expert_source.cpp:1363`). A mismatch is refused with both
  values printed, the way `src/core/pinned.cu:417-427` does.
* **`geom_hash`** — over `geometry_key(g)` (`src/core/conversation_state.cpp:14-19`) **plus the
  canonical split spec**: `layer_lo/layer_hi` of the main session and of every later stage, in engine
  order, plus `n_stages`, plus `draft_device == last stage` (the tiling rule
  `.megamind/src/core/s31a-parking-split-notes.md:39-40`). Two decode instances with different
  `--layer-split` produce different payloads for the same conversation and must never mount each
  other's.

Both are in the arena header, the slot header, the `READY` line and the `HELLO` line. Any mismatch is
refused **before** any byte is copied, and the refusal names both hashes:

```
strata serve: handoff refused: geometry hash 3f2a… != 91c7… (this instance runs the layer split
  [0,16) [16,34) [34,48); the prefill instance runs [0,18) [18,48))
```

This is the payload-level analogue of "a parked image is tied to the split it came from: change
`--layer-split` and the images are refused, never half-mounted"
(`.megamind/src/core/s31a-parking-split-notes.md:74-75`).

---

## 4. The control protocol

### 4.1 The new header

`include/strata/program/prefill_svc.hpp` — **new**, deliberately separate from
`include/strata/program/serve_proto.hpp` so the client wire cannot drift (D3,
`.megamind/stage4-plan.md:42-45`). Same rules as `serve_proto.hpp`
(`include/strata/program/serve_proto.hpp:11-15`):

* header-only, no CUDA, no I/O, no session — every function **returns** the line without the `\n`, and
  the caller writes it. That is what makes a golden CPU test possible.
* one place formats, one place parses, so a line and its parser cannot drift.
* `detail::f()` (the `snprintf` helper, `serve_proto.hpp:54-67`) is duplicated here rather than shared,
  so neither header can pull the other in. `serve_swap.hpp` already includes `serve_proto.hpp`
  (`include/strata/program/serve_swap.hpp:58`); `prefill_svc.hpp` must include **nothing** from
  `program/` except `<cstdint>`/`<string>`/`<vector>`.
* the test is `src/program/prefill_svc_test.cpp`, wired into `STRATA_BUILD_TESTS` exactly like
  `serve_proto_test` (`CMakeLists.txt:667-673`).

### 4.2 The socket and the handshake

`AF_UNIX`, `SOCK_STREAM`, bound by the prefill instance at
`<handoff-dir>/<pack>-<split>/<instance>/prefill.sock`. One connection per decode instance, held for
the process's lifetime. The stream is line-oriented: `\n`-terminated, ASCII, no CR, no NUL, no
embedded space inside a token. A line longer than 4096 bytes closes the connection.

**Handshake, in order:**

```
decode   ->  HELLO <proto_ver> <client_id> <pack_hash> <geom_hash> <max_ctx> <n_stages> <split> <cvec> <vision>
prefill  ->  PREFILL-READY-V1 <proto_ver> <slots> <max_tokens> <tiers> <pack_hash> <geom_hash>
```

`PREFILL-READY-V1` is the version token (D3, `.megamind/stage4-plan.md:42-45`). A prefill instance that
does not know `HELLO` closes the connection without a line; the decode instance then reports
`strata serve: the prefill endpoint does not speak PREFILL-READY-V1` and, per `--prefill-fallback`,
either reads locally or holds every new prompt.

Grammar:

```
HELLO proto_ver=u64 client_id=u64 pack_hash=x64 geom_hash=x64 max_ctx=i64 n_stages=i64
      split=LIST cvec=0|1 vision=0|1
      LIST  := N | N,N,...        (the layer_lo of every later stage, engine order; empty for no split)
```

`cvec` and `vision` are the decode instance's requirements: a prefill instance started without
`--vision` refuses every `GENI` (`ERR novision`, permanent), and `cvec` is echoed so a mismatch is
caught at `HELLO` rather than after a 3 GB transfer.

```
PREFILL-READY-V1 proto_ver=u64 slots=LIST max_tokens=i64 tiers=LIST pack_hash=x64 geom_hash=x64
```

`slots=LIST` and `tiers=LIST` are the same length and index-aligned: `slots=2,2,2` with
`tiers=1024,16384,131072`. `max_tokens` is the largest tier. The decode instance checks `proto_ver`,
`pack_hash` and `geom_hash` here and refuses to start a request until they match (§3.10); it caches
`slots`/`tiers` for the sizing log line and for `--prefill-max-tokens` validation.

Startup lines (both sides, stderr, so the owner can read a log without guessing):

```
strata prefill: serving on /dev/shm/strata-handoff/<…>/prefill.sock; slots 2,2,2 (tiers
  1024,16384,131072), max 131072 tokens, arena 8.2 GiB, geometry hash 3f2a…
strata serve: prefill endpoint /dev/shm/strata-handoff/<…>/prefill.sock: PREFILL-READY-V1
  (slots 2,2,2, max 131072 tokens); handoff on
```

and with `STRATA_NO_HANDOFF=1` / `--role full`, neither line exists — the serial path is untouched.

### 4.3 Every line, both directions

Legend: `u64` unsigned decimal, `i64` signed decimal, `x64` 16 lowercase hex digits, `NAME` a token
matching `[A-Za-z0-9._/-]+`, `ID` a token matching `[A-Za-z0-9_-]+`. **Unknown trailing tokens on any
line are ignored** (the compatibility rule `serve/server.py:407` already applies to `READY`). A line
whose *known* tokens do not parse closes the connection and logs the offending line.

#### decode → prefill

```
HELLO …                                   §4.2, once per connection
PING <cookie>                             liveness; answered PONG <cookie>
REQ <id> <tokens> <tier> ids=NAME [k=v …] ask for a prompt read
      <tokens> = the prompt length in tokens (u64); <tier> = the tier index (u64) the decode
                 instance is asking for, which the prefill instance may honour or override.
      ids=NAME is REQUIRED (§ "the prompt ids do not go on the wire", below).
      optional keys, space separated, `=`-joined:
        geni=NAME        the strata-vision embeddings file (GENI); absent = text
        cvec=0|1         the control-vector state (default 1)
        resume=i64       v1 requires 0; anything else is `ERR resume` (see below)
      <id> is the decode instance's request id (u64); it is echoed unchanged on every line about
      this request.
CANCEL <id>                               stop that request; see §4.6
ACK <id> <nonce>                          the payload was copied out and validated; the slot may be
                                          reused. This is the ONLY release verb.
BYE <reason>                              orderly shutdown of this connection
```

**There is deliberately no `prefix=` key.** The prefill instance owns the reuse decision: it looks the
prompt up in **its own** `ConversationCache` with `best()` (§1.3.3), which is the only lookup that can
know what it already holds. A decode instance hinting at a prefix would be a second, weaker reuse
mechanism competing with the first — and stage 3's S3.9 showed what two lookups over one store do.

**There is deliberately no `RELEASE` alias.** One action, one verb; an alias is a second thing to test
and a second thing to get subtly different from the first.

**The prompt ids do not go on the wire.** A 245 946-token prompt is ~1.7 MB of ASCII
(`strata-iq3_s.log:6649`), and the wire is for control. `ids=NAME` names a file the decode instance
wrote. The file format is one `i64` per line, `\n`-terminated, no other content, mode `0600`, under the
instance directory (§3.2). The decode instance deletes it after `ACK`. A missing or unreadable file is
`ERR ids` (permanent for that request).

`resume` is pinned to 0 in v1: the prefill instance always reads the whole prompt from token 0,
because its own `ConversationCache` is what decides how much of it is actually *computed*
(`ConversationCache::best()`, §1.3.3). A `resume != 0` is `ERR resume` (permanent, and it names the
reason: "the prefill instance owns the reuse decision").

#### prefill → decode

```
PREFILL-READY-V1 …                        §4.2, once
PONG <cookie>
QUEUED <id> <ahead>                       accepted, no slot claimed yet; <ahead> = jobs ahead of it
CLAIM <id> <slot> <nonce>                 a slot was claimed; the read is about to start
SEG <id> <slot> <seq> <tokens_done> <tokens_total> <ms>
                                          the prompt READ is finished and the payload write has
                                          started. v1 emits exactly one SEG, `seq=1`, between `CLAIM`
                                          and `DONE`: `<ms>` is the read time, and `DONE`'s `<ms>` is
                                          the read + publish time. It exists because the write of a
                                          3 GB payload is a few hundred ms of silence on a connection
                                          that a client is watching, and because a future
                                          multi-segment protocol needs no new verb. A reader that
                                          ignores SEG entirely is still correct.
DONE <id> <slot> <nonce> <bytes> <tokens> <ms>
                                          the payload is published (state=READY); copy it out
ERR <id> <code> <message>                 this request failed at the prefill instance; see §4.4
INFO k=v …                                process-wide, untagged, ignored by an old reader
```

**There is no `ACK-ED` echo.** The decode instance sent `ACK`; hearing it back tells it nothing, and a
line a reader must deliberately ignore is surface area for no gain. The prefill instance records the
release on **its** stderr instead (`strata prefill: slot 3 released by client 7f3a…, 3.05 GiB, held
412 ms`), which is where the leak diagnostics belong.

`INFO` keys the prefill instance emits (the same shape as the serve `INFO` line,
`src/program/generate.cpp:5113-5120`):

```
INFO role=prefill slots=<n> slots_free=<n> jobs=<n> jobs_done=<n> jobs_err=<n>
     reused_tokens=<n> read_tokens=<n> arena_mib=<n> max_ctx=<n> pack=<x64> geom=<x64>
```

and the periodic activity line (the `STRATA_SERVE_ACTIVITY_S` pattern,
`src/program/generate.cpp:691-696`):

```
strata prefill: activity: jobs=42 done=40 err=2 queued=0 slots_free=4/6
     reused=1820512 tokens read=8192 tokens handoff_ms=1240 arena=3120MiB/8396MiB
```

### 4.4 Error codes, and which are permanent

`ERR <id> <code> <message>`. The code is a stable single token; the message is free text for the log.
The decode instance maps each code onto the S4.2 hold vocabulary — **`Wait::prefill` is a WAIT, and
`Hold::error` is an error** (`include/strata/program/serve_driver.hpp:885-941`). The mapping lives in
one table in `prefill_svc.hpp` so the two sides cannot drift, and `serve/test_server.py` pins the
strings the way it already pins the engine's refusal texts (`serve/server.py:159-180`).

| code | permanent? | meaning | decode-side action |
|---|---|---|---|
| `full` | **no** | no free slot and no queue room | hold with reason `prefill`; re-ask |
| `queue` | **no** | `--prefill-queue` is full | hold with reason `prefill`; re-ask |
| `toobig` | **yes** | `tokens > --prefill-max-tokens` or > the largest tier | `ERR` to the client, naming both numbers |
| `ids` | **yes** | the ids file is missing/unparsable | `ERR` |
| `novision` | **yes** | a `GENI` against a prefill instance without `--vision` | `ERR` |
| `resume` | **yes** | `resume != 0` in v1 | `ERR` (a bug in the caller) |
| `geom` | **yes** | geometry / split mismatch | `ERR`; also drop the connection |
| `pack` | **yes** | pack hash mismatch | `ERR`; also drop the connection |
| `cancel` | n/a | the request was cancelled before it ran | answer the client `DONE … cancel` |
| `read` | **no** | the prompt read itself failed (a CUDA error, a lend failure) | retry up to `--prefill-retries`, then `ERR` |
| `slotlost` | **no** | the slot was reclaimed mid-write (arena full, owner died) | retry once, then hold |
| `shutting` | **no** | the prefill instance is exiting | hold; the client's `--hold-ms` bounds it |
| `badpayload` | **yes** | the decode instance's own §2.5 checks 1-2 rejected the payload it copied out | `ERR`, or a local re-read with `--prefill-fallback local`. Emitted by the **decode** instance, not the prefill one — it is listed here so the code vocabulary is one table |
| `endpoint` | **no** | the socket is down, closed, or a line did not parse | hold; reconnect with backoff. Also decode-emitted |

**D5's rule, restated for this protocol (`.megamind/stage4-plan.md:55-60`): a temporary refusal must
never become an `ERR` on the client's wire.** `full`, `queue`, `read`, `slotlost`, `shutting` and a
dropped connection are all *internal* answers. The client sees either a slow first token or, after
`--hold-ms`, one honest timeout line. Only `toobig`, `ids`, `novision`, `resume`, `geom`, `pack` reach
the client as `ERR`, and every one of them is a fact the client (or the operator) can fix.

### 4.5 The handoff state machine

Per request, on the **decode** instance. The handoff states are **not** the engine's `Phase` values —
they live in the handoff client, and the engine's `Phase::await_prefill` (§1.5) is the one `Phase` that
means "one of these is in progress". `H_*` states run on the **handoff thread** unless marked `[engine]`.

```
                                       admit (S4.2's WaitQueue head, §6)
  QUEUED ───────────────────────────────────────────────────────────────┐
    │  local ConversationCache::best() hit long enough                  │
    │  -> NO handoff at all: today's mount path, unchanged              │
    ▼                                                                   │
  H_REQ_SENT ── ERR full/queue ──► HOLD(prefill) ──(wake)──► H_REQ_SENT  │
    │         (bounded backoff 50 ms -> 2 s, one outstanding REQ)        │
    ├── ERR toobig/ids/geom/pack/novision/resume ──► ERR <id> … ─► IDLE  │
    │                                                                   │
    ├── QUEUED <id> <ahead>   (accepted, no slot yet) ─┐                 │
    ▼                                                   │                 │
  H_ACCEPTED ◄──────────────────────────────────────────┘                 │
    │  CLAIM <id> <slot> <nonce>                                          │
    ▼                                                                     │
  H_PREFILLING  (the prefill instance is reading; SEG lines are progress)  │
    │  DONE <id> <slot> <nonce> <bytes> <tokens> <ms>                     │
    ▼                                                                     │
  H_SEG_READY   (the slot is state=READY; the copy-out starts HERE)        │
    │  map -> read -> rebuild SavedConversation -> §2.5 checks 1-2        │
    ▼                                                                     │
  H_HANDED_OVER (ACK sent; the slot is FREE again)                        │
    │  the image is posted to the engine thread as `mount_image` [engine]  │
    ▼                                                                     │
  MOUNTING [engine]  serve_swap::run: validate -> unmount -> mount ->      │
    │        draft_kv -> adopt -> device_state                            │
    │        (include/strata/program/serve_swap.hpp:402-419)              │
    ├── validate fails ──► ERR badpayload, or a local re-read ──────► IDLE│
    ├── mount fails ─────► session poisoned ──► return 1 (0.1.30's rule)  │
    ▼                                                                     │
  Phase::prefill [engine]  (only the short-read / window path may run here,│
    │                      §1.5; a retry of a FAILED handoff re-enters    │
    │                      H_REQ_SENT, bounded by --prefill-retries) ─────┘
    ▼
  Phase::decode … (today's path, unchanged)
```

Ordering rules the implementation must honour:

1. **`ACK` is sent as soon as the copy-out and checks 1-2 pass**, before the mount. Holding a 3 GB
   slot across a mount is a slot leak with extra steps. The rebuilt `SavedConversation` is owned by
   the decode instance from that moment.
2. **The copy-out never runs on the engine thread** (§1.5). The engine thread must keep stepping other
   slots while a payload is being read.
3. **A `DONE` for a request the decode instance already cancelled is answered `CANCEL <id>`
   immediately**, and the slot is released without being read. The prefill instance must treat
   `CANCEL` after `DONE` as a normal release, not a protocol error.
4. **`CLAIM` before `DONE` is the only order.** A `DONE` with no preceding `CLAIM` is a protocol
   violation: close the connection and log it.
5. **`nonce` ties everything together.** The `CLAIM`, `SEG`, `DONE` and `ACK` lines for one handoff all
   carry the same nonce, and the slot header's nonce must equal it. A nonce mismatch means a stale line
   about a slot that has already been reused — the same "says WHICH attempt owns it, not just which
   pid" role the nonce plays in the expert arena (`include/strata/core/pinned.hpp:41`,
   `src/core/pinned.cu:196-206`).
6. **`Phase::prefill` on a decode instance with `--prefill-fallback none` is reachable only through
   `windows_ok`/`read_windows`** (`src/program/generate.cpp:5258-5266`, `:6998`) — the short-read path,
   which is a decode path, not the batched prompt path. The batched branch at `:7003-7008`
   (`lend(...)` + `sp.run(...)`) is replaced by the handoff (§1.5). A `--role decode` engine with
   `--prefill-fallback none` must never call `sp.run`.

### 4.6 Cancellation

```
decode -> CANCEL <id>
```

* Cancelled while `QUEUED` (no slot claimed): the prefill instance drops the job and answers
  `ERR <id> cancel …`. No slot was touched.
* Cancelled while reading: the prefill instance sets the job's stop flag, which is wired to
  `Prefill::should_stop` (`include/strata/prefill/prefill.hpp:92-93`, checked before every chunk —
  the same hook the serve path wires at `src/program/generate.cpp:5064`). The read stops at the next
  chunk boundary, the slot is released **without publishing**, and the answer is `ERR <id> cancel …`.
* Cancelled after `DONE`, before `ACK`: the decode instance sends `ACK` and never mounts. The client
  gets `DONE … cancel`, which is what a request cancelled before its first token already produces
  (`src/program/generate.cpp`, `R.cancelled = true` before admission, and
  `include/strata/program/serve_driver.hpp:1311-1322` for a waiter).
* Cancelled after `ACK`: that is an ordinary `STOP <id>` on the client wire and is stage 3's business,
  unchanged.

`STOP` (bare, no id) is **not** forwarded to the prefill instance. It keeps 0.1.30's meaning — the
newest running slot (`include/strata/program/serve_proto.hpp:223-224`,
`serve_driver::bare_stop_target`) — and cancelling a decode slot whose prompt is still being read does
cancel the pending `REQ` for it.

### 4.7 Failure semantics

| failure | detected by | how | consequence |
|---|---|---|---|
| **prefill instance dies** | the socket closes; and its `owner_pid` in every `CLAIMED`/`READY` slot goes dead (`/proc`, §3.6) | the decode instance logs `strata serve: the prefill instance died; holding new prompts`, marks every in-flight handoff as failed, and either holds (default) or reads locally (`--prefill-fallback local`) | no client sees an `ERR` for a temporary reason (D5). Slots the dead process owned are reclaimed by the lease rule. |
| **decode instance dies mid-handoff** | the prefill instance sees the socket close; and the slot's `client_id` pid goes dead | the slot is reclaimed after `STRATA_HANDOFF_LEASE_MS` or immediately on a dead-pid check, with `flags |= abandoned` logged | **this is the slot-leak fix.** Without it a 3 GB slot is stranded until the prefill instance restarts. |
| **decode instance dies after `ACK`** | nothing to do | the slot is already FREE | clean |
| **handoff slot leak** | the prefill instance's idle tick | every `CLAIMED`/`READY` slot is checked against its lease and its owner/client pid; the activity line prints `slots_free=4/6`, so a leak is visible before it is fatal | slots come back; the leak is reported, not hidden |
| **partial write** (prefill killed mid-`pwrite`) | `state` is still `CLAIMED`, so no reader ever maps the body (§3.5 step 2) | the slot is reclaimed and rewritten from scratch | impossible to read a half-written payload — the same guarantee `state` gives the expert arena (`src/core/pinned.cu:381-383`) |
| **partial publish** | the publish is one ordered `pwrite` of `state` (§3.5 step 3) | a reader either sees `CLAIMED` or `READY` | no torn state |
| **corrupt payload** (bit flip, wrong model) | `payload_hash`, `pack_hash`, `geom_hash`, then §2.5 checks 1-2, then `conversation_snapshot_validate` | refused before any write; `ERR badpayload` or a local re-read | the session is never written from a bad image |
| **restore fails after the first write** | `conversation_snapshot_restore` → `transfer_failed` | fatal to the session, exactly as today (`include/strata/core/conversation_snapshot.hpp:104-108`, `serve_swap.hpp:470-471`) | the process exits 1 and `serve/server.py` restarts it (`serve/server.py:549-558`) |
| **arena full** | claim fails | `QUEUED`/`ERR full` internally → the decode instance holds (§3.8, §6) | never a client-visible `ERR` until `--hold-ms` |
| **tmpfs full** | the §3.7 `statvfs` check, before `ftruncate` | the prefill instance refuses to start with both numbers | no SIGBUS, no mystery death |
| **socket line too long / unparseable** | the reader | close the connection, log the line, treat it as "prefill instance down" | bounded; the decode instance retries the connection |
| **a request re-handoffs forever** | a per-request counter | `--prefill-retries` (default 3), the direct analogue of `kMaxRereads = 4` (`include/strata/program/serve_driver.hpp:557-563`, `.megamind/src/program/s38-two-prefill-livelock.md:48`) | a fourth attempt ends the request with a named `ERR` instead of spinning |

The last row is the one stage 3's history says will be missed. S3.8's livelock was invisible to the
watchdog because every pass really did read tokens (`.megamind/src/program/s38-two-prefill-livelock.md:19-20`).
A handoff loop has the same shape: every pass really does transfer gigabytes. So the retry counter is
**not** optional, and the test must show that removing it makes the end-to-end simulation run to its
pass cap without finishing.

---

## 5. Multi-instance topology: one prefill instance, N decode instances

### 5.1 Discovery and registration

There is no discovery service and there is no broadcast. The endpoint is configuration, because that
is the only thing that survives a container restart and a `--host 0.0.0.0` box.

```
prefill instance:   --role prefill --prefill-listen /dev/shm/strata-handoff/<…>/prefill.sock
decode instance:    --role decode  --prefill-endpoint /dev/shm/strata-handoff/<…>/prefill.sock
                                     --prefill-client-id 7f3a91c0beef0001
```

* The **prefill instance binds and owns** the socket, the arena and the directory. It creates them at
  start-up (§3.7) and removes the socket on orderly exit (never the arena: another instance may be
  mid-read, and a stale arena is caught by the pid/lease rule, not by deletion).
* The **decode instance connects**. It retries the connect with a bounded backoff (20 ms → 2 s, the
  same retry/give-up shape as `arena_lock`, `src/core/pinned.cu:111-128`), and it reconnects
  automatically after a dropped socket. A decode instance that cannot connect starts anyway: it is
  then a serve process that holds every new prompt (or reads locally, `--prefill-fallback local`).
  Refusing to start would take the operator's running conversations away because a *backend* is down.
* `--prefill-client-id` is a u64 the decode instance sends in `HELLO` and the prefill instance stamps
  into every slot it writes for it (`§3.4`). It is random by default. It is what makes "this slot
  belongs to that client" checkable without trusting a pid, and it is what the lease rule uses to
  decide that a `READY` slot's reader is gone.
* **`setup.py` writes both configs.** One `strata-<model>.json` per instance, exactly as it already
  writes one per model (`setup.py:1914-1916`, `serve/server.py:2576`, `strata-iq3_s.json`). The
  prefill instance's config has no `"port"` and no `"tokenizer"` need, because nothing runs
  `serve/server.py` against it — it is started directly:
  `engine/strata --role prefill <the same model flags> --prefill-listen …`.
  **`serve/server.py` is only ever run for the decode instances** (D1).

The model flags must match on both sides: same `--pack`, same `--native`, same `--ple-gguf`, same
`--max-context`, same `--kv`, same `--kv-resident`, same rope scaling, same `--layer-split`. The
`geom_hash`/`pack_hash` check (§3.10) is what enforces it, and it fires before any byte moves. The
rope scaling is not optional to match: "one cache must never mix two scalings"
(`src/program/generate.cpp:260-264`).

### 5.2 How the prefill instance serialises and parallelises concurrent jobs

**One engine thread, one job queue, one job at a time.** The reasons are the same ones stage 3 found
and the same ones the prompt loan enforces:

* The prompt loan is one loan: `pf_parts` lends the tail slots of each stage's cache to that stage's
  `Prefill` (`src/program/generate.cpp:3759-3800`, `docs/STAGE3-CONCURRENCY.md:463-480`). Two
  concurrent prefills would hand the same buffers to two readers.
* The layer-split prompt path chains stages through one pair of pinned hand-off buffers per stage
  boundary (`src/program/generate.cpp:3961-4012`), and `Prefill::set_stage` wires one `Prefill` chain
  (`include/strata/prefill/prefill.hpp:104-106`). A second concurrent read would need a second chain
  and a second set of pinned buffers.
* The captured MTP prefill graphs bake in `st_`, `pf_dev_`, `tok_`, `step_`, `pos_`, `Rin_`
  (`src/core/mtp.cpp:600-616`), i.e. one draft-KV stream per drafter.

So the prefill instance's scheduler is:

```
loop:
  drain the socket reader thread's job queue (FIFO by REQ arrival)
  if a job is CANCELLING: finish its unwind first (return the loan, release the slot)
  if the queue is empty: block on the socket
  take the head job
  claim a slot for its tier (FIFO; no queue-jumping, §3.8)
     -> if no slot: QUEUED/ERR full to the client, and the job goes back on the queue
  run it: ConversationCache::best() -> maybe zero GPU work (§1.3.3)
          else Prefill::run chunks + on_chunk (draft KV) + checkpoint_at
  publish (release fence + one ordered pwrite)
  DONE; keep the job's image parked in the prefill instance's own ConversationCache
```

**Where parallelism does exist, and is free:**

1. **The copy-out is parallel by construction.** The prefill instance publishes and moves on; each
   decode instance copies its own slot out on its own thread. N decode instances read N slots at once
   at page-cache speed. This is the only place GiB moves, and it is not serialised.
2. **The socket reader thread** parses lines into the job queue while the engine thread reads a prompt
   — the exact stdin-thread/engine-thread split stage 3 kept (`docs/STAGE3-CONCURRENCY.md:256-260`).
3. **Multiple prefill instances** (OQ-S4-5) is how you get parallel prompt *computation*: N instances,
   N arenas, N sockets, and the decode instances pick one each. Nothing in this design prevents it;
   nothing in S4.3 schedules across them.

**Explicitly refused:** a second `Prefill` in one prefill instance, or a second thread calling
`Prefill::run`. That is risk R8 with the labels off, and stage 3's §4.3 rule 4 already says two pool
clients would need a real queue and two `Stager` clients would need two `Stager`s
(`docs/STAGE3-CONCURRENCY.md:458-461`).

### 5.3 Not repeating stage 3's livelock and parking bugs

Both bugs are instructive here, so the design rules are stated as rules.

**S3.8 — two concurrent prefills erased each other and never finished**
(`.megamind/src/program/s38-two-prefill-livelock.md`). The chain was: a mid-prefill slot was never
published → `park_current`'s `live_ok` guard refused the save → `saved 0 B` → `resumable = false` →
`step_gate` → `re_read` → token 0, and the *licence* was `outgoing_for(prefill)` always answering
`re_readable`. The watchdog could not see it because every pass really did read tokens.

The prefill instance's equivalent failure would be: a job is preempted, its partial read is not
parkable, and the next pick restarts it from token 0 forever. Rules that prevent it:

1. **The prefill instance owns its own slot registry and its own claims.** It is a serve-shaped process
   with a `Registry` (`include/strata/program/slot.hpp`) and a `ConversationCache`
   (`src/program/generate.cpp:4085`), and its request ids are its own. It never consults, mirrors or
   trusts a decode instance's registry. A decode instance's `REQ` carries only ids, tokens and keys.
2. **A job is never preempted mid-read in v1.** One job runs to completion or is cancelled. That makes
   the whole S3.8 class unreachable by construction, and it is the honest choice while the prompt loan
   is one loan. (`publish_prefill_branch` and `outgoing_for(phase, read_tokens)` therefore have no
   analogue to re-implement — but if a later phase adds preemption, both must be re-derived here, not
   assumed.)
3. **The retry bound is real** (`--prefill-retries`, §4.7), and the end-to-end test must fail when it
   is neutered — the exact anti-vacuity discipline S3.8 used
   (`.megamind/src/program/s38-two-prefill-livelock.md:85-93`).
4. **`arm_prompt_state`'s lesson travels.** `cur`, `pp_total/pp_from/pp_next_check`, `part_at` and
   `sp.embd_rows` are process-wide and 0.1.30 set them once per request; the driver re-arms them
   before **every** prefill step, because `checkpoint_at` copies `cur[0, L)` and `sp.on_chunk` indexes
   `cur[p0 + t + 1]` (`src/program/generate.cpp:4412-4414`). A prefill instance that runs job A after
   job B must re-arm them per job, or it checkpoints B's tokens under A's id — silent, and it corrupts
   the cache's prefix comparison for every later request of that chat
   (`.megamind/src/program/s38-two-prefill-livelock.md:52-56`). **`req_imgs` must be re-armed with
   `cur`**, which is the separate bug S3.8 fixed on the same path.

**S3.9 — a parked branch was being stolen by another request**
(`.megamind/src/program/s39-parked-branch-theft.md`). The mechanism: a parked entry is ONE object, a
prefix lookup matches any *checkpoint* of any entry, and mounting `take()`s the whole entry.

The prefill instance has exactly this hazard, one level up: two decode instances send the same chat's
prompt (9 849 of 10 394 tokens shared), `best()` matches one entry's checkpoint, `take()` removes the
entry, and the *other* decode instance's `DONE` describes a payload that no longer exists in the cache.
Rules:

1. **`owner` is remapped at the boundary (§2.4).** The payload leaves with `owner = kNoOwner`; the
   receiving process assigns the claim. A claim never crosses a process, so `claimed_by_other()`
   (`conversation_cache.hpp:320-323`) stays a within-process predicate and cannot be fooled by an id
   collision between two decode instances.
2. **The prefill instance's own claims are its own.** While a job is running, its entry is claimed for
   that job's id, and `victim_to_prune()` prefers unclaimed entries
   (`conversation_cache.hpp:373-381`). A job that loses its entry to pruning re-reads; it does not
   publish a payload built on a half-taken branch.
3. **The payload is a copy, not a loan.** `take()`-ing an entry to write a slot does not remove it from
   the prefill instance's cache: the write path uses the image **in place** (`conversation_kv_save`
   and the snapshot save only *read* the session/cache, `src/core/conversation_snapshot.cpp:139-173`).
   The entry stays parked for the next request of that chat, which is what makes §1.3.3's zero-GPU
   handoff repeatable.
4. **`best()` vs `best_exact()` stays exactly as stage 3 split them.** The prefill instance looks a
   prompt up with `best()` (a new request's prompt really is longer,
   `docs/STAGE3-CONCURRENCY.md:1058-1066`). It never uses `best_exact()` — that is a hand-over lookup,
   and the prefill instance never hands a session over.

### 5.4 N decode instances: what actually changes

Nothing in the protocol is per-client-count. The socket accepts N connections; each is identified by
`client_id`; each slot names exactly one `client_id`; the job queue is one FIFO across all of them.
The only per-instance numbers are the arena size (§3.8) and `--prefill-queue`, and both are
configuration, not derived from N.

What the operator must size: **the prefill instance's VRAM.** One prefill instance serving N decode
instances needs its own expert cache, its own session arenas, its own MTP, and its own prompt loan
(4.95 GiB of cache slots, `strata-iq3_s.log:33`). On a 24 GiB card that is a second full engine's
worth of VRAM, so in practice the prefill instance gets its own GPU — OQ-S4-1.

---

## 6. Interaction with the wait queue (S4.2)

S4.2 landed the vocabulary: `Hold{run_now, wait, error}`, `Wait{…}`, `WaitQueue`, `--hold-ms`
(`include/strata/program/serve_driver.hpp:852-1359`; server side `serve/server.py:192-295`,
`_gate_hold` `:1455-1504`). Stage 4's split adds **one reason** and moves **one wait**.

### 6.1 The new reason

```cpp
enum class Wait : uint8_t {
    …,
    prefill,   ///< the prefill instance cannot run this prompt yet (no slot, no queue room, or it is down)
};
```

with `wait_reason(Wait::prefill) == "prefill-busy"` and
`wait_text(Wait::prefill) == "the prefill instance has no free handoff slot for a prompt this long"`.
Both strings live next to the existing ones (`include/strata/program/serve_driver.hpp:913-941`) so the
log, the wire and the test cannot drift, and `serve/test_server.py` pins them the way it pins the
others (`serve/server.py:159-180`).

`serve/server.py`'s `PERMANENT_REFUSALS` (`serve/server.py:163-180`) must **not** gain
`prefill-busy`: an unrecognised "no" is already treated as temporary there
(`refusal_is_temporary`, `serve/server.py:183-189`), and the whole point of D5 is that a temporary
refusal never becomes an `ERR`.

### 6.2 Who waits where

Three queues, and each has exactly one job:

| queue | owner | what it waits for | visible as |
|---|---|---|---|
| `serve/server.py`'s `HoldQueue` | the HTTP server | a free engine slot permit (`slot_gate`) | `/status`, `/slots` (`_waiting_slots`, `serve/server.py:1520-1538`), `/metrics` |
| the decode engine's `WaitQueue` | the decode instance | a slot, RAM, the loan, **or the prefill instance** | the `WAIT n oldest_ms reason` line (`include/strata/program/serve_driver.hpp:1156-1159`), the activity line's `waiting=` tail (`:636-651`) |
| the prefill instance's job queue | the prefill instance | a free arena slot, and its one engine thread | its `INFO`/activity line (`slots_free=4/6`, `queued=`) |

**The rule: the decode instance waits, not the client, and the prefill instance never blocks a decode
engine thread.**

* The decode instance's engine thread **never blocks on the socket.** `Step::await_prefill` is
  non-blocking: it polls the handoff client's state and returns `Step::wait`
  (`include/strata/program/serve_driver.hpp:61`) if the payload has not arrived. The scheduler then
  steps another slot. This is the same reason stage 3 defers a slot blocked on the loan rather than
  spinning on it (`include/strata/program/serve_driver.hpp:692-704` — `Loan::may_lend` /
  `may_read_cache`, and `.megamind/src/program/s31e2-driver-notes.md:48-50`, `:71-74`): a `pick()`
  that cannot see the resource needs an explicit deferral, or the loop livelocks.
* **Wake points.** A waiter is woken when (a) a `DONE` arrives for it, (b) the prefill instance sends
  `INFO`/activity showing a free slot, (c) any local slot/RAM/loan reason clears
  (`WaitQueue::wake_all`, `include/strata/program/serve_driver.hpp:1263-1270`), or (d) the poll bound
  expires. The re-ask is bounded: at most one outstanding `REQ` per request, and a `QUEUED`/`ERR full`
  answer sets a backoff (start 50 ms, cap 2 s) so a full arena is not hammered.
* **The bound is `--hold-ms`, and it is the only thing that ends a wait on its own.**
  `hold_expired` (`include/strata/program/serve_driver.hpp:1111-1113`), `hold_expired_line` (`:1125-1131`).
  The watchdog must not kill a waiter: `watchdog_sees_waiters() == false`
  (`include/strata/program/serve_driver.hpp:1115-1120`), and a request sitting in
  `Phase::await_prefill` is not inside a step, so its heartbeat is not expected to move. **This is
  risk R3 in a new costume and the implementation must not regress it.**
* **Priority.** A request that already holds a conversation outranks a new one
  (`kPriorityResumed`, `include/strata/program/serve_driver.hpp:1175-1176`). A handoff waiter keeps
  whatever priority it had; the split does not add a second ordering rule.
* **Admission vs handoff ordering.** Admission (S4.2's `admit_decision`,
  `include/strata/program/serve_driver.hpp:1082-1100`) runs **before** a `REQ` is sent. A request that
  can never run locally must not occupy a handoff slot. The permanent checks gain one more:
  `tokens > --prefill-max-tokens` is `ERR toobig` immediately, because waiting will not make the
  prefill instance able to read it.
* **RAM pricing.** A handoff payload is priced with the same per-token rate the parking budget uses
  (`bytes_per_token` / `SlotCost` / `slot_image_bytes`,
  `include/strata/program/serve_driver.hpp:219-283`), because the decode instance must allocate a real
  `SavedConversation` to receive it. `conversation_snapshot_bytes` at the request's own length is the
  right figure, and `slot_price_uncapped` (`:1024-1029`) is the "cannot fit even an empty machine"
  question S4.2 exported for exactly this use (`:1080-1081`).

### 6.3 The one sentence

> A request that cannot run because the **prefill instance** is busy is a **wait** with reason
> `prefill`; a request that cannot run because its **prompt is too long for the arena** is an **error**
> naming both numbers; and no decode engine thread ever blocks on a socket.

---

## 7. Bit-exactness and verification

### 7.1 What must be identical

The acceptance bar is stage 3's, unchanged (`docs/STAGE3-CONCURRENCY.md:657-681`):

1. **`--role full` + `--serve-slots 0` is byte-identical to 0.1.30** on the wire and in the tokens.
   Proven the way stage 3 proved it: the serial driver's text is unchanged
   (`.shz_cmd/s31e2_serial_proof.py`) and the serial-reachable output-call count is unchanged
   (`.shz_cmd/s31e2_output_proof.py`, 109 calls, `.megamind/src/program/s38-two-prefill-livelock.md:82-83`).
2. **`--role decode` + `--prefill-fallback local` + `STRATA_NO_HANDOFF=1` produces the same tokens as
   `--role full`.** This is the arm that proves the role plumbing itself is inert.
3. **`--role decode` with a handoff produces the same tokens as the same request read locally**, for
   the same prompt, the same seed and the same `--adapt-every`. This is the interesting one, and it is
   checkable: the payload is the *same object* the local path would have parked, produced by the same
   `Prefill::run` + `checkpoint_at` + `conversation_snapshot_save` code. The things that legitimately
   differ are the same ones stage 3 §7.2 lists — expert residency (`adapt()` merges usage across
   conversations, so the prefill instance's resident set is its own) and wall-clock timings.
   **`--adapt-every 100000` is the documented reproducibility switch**
   (`docs/STAGE3-CONCURRENCY.md:689`, `docs/DETAILS.md:728`) and it is what the owner should use
   for the comparison.
4. **`STRATA_STATE_HASH=1` is the single best test this design has, and it already exists**
   (`docs/STAGE3-CONCURRENCY.md:777-780`, `.megamind/src/program/s31d-swap-notes.md:97-120`). The
   recipe extends directly: read a prompt **through the handoff**, hash the decode session; read the
   same prompt **locally**, hash again; the two hashes must match on
   `gdn= ple= tail= pooled= kv= mtp= stale= dead=`. `stale=`/`pooled_full=` may differ (cells past L).

### 7.2 What may legitimately differ

| difference | why it is legitimate | how to suppress it |
|---|---|---|
| expert residency in the prefill instance | its `adapt()` learns from other decode instances' prompts. The engine already warns that residency changes the rounding (`strata-iq3_s.log:22-24`). | `--adapt-every 100000` on both sides |
| draft acceptance / `DraftPolicy` | the policy is per-process and only affects draft *quality*; the verify window decides every emitted token (`docs/STAGE3-CONCURRENCY.md:690`) | none needed |
| `prompt_ms` | it now includes the handoff (read + publish + copy-out + mount) | report it as-is; add `handoff_ms` to the `DONE` tail and to `/metrics` |
| `--kv-resident` warmth | a mount calls `kv_stream_reset` (`src/core/conversation_snapshot.cpp:207`), and a handoff is always a mount | raise `--kv-resident`; the prefill instance's reuse path (§1.3.3) avoids the read, not the mount |
| which instance holds a conversation | the prefill instance's cache does not contain the tokens a decode instance generated (§1.3.3) | pin a chat to one decode instance, or land S4.5's reverse handoff |

### 7.3 What can be CPU-tested without ever starting the engine

Everything in §3, §4 and §2 is host-side. The tests, and what each pins:

| test | file | what it proves, with no GPU and no model |
|---|---|---|
| `prefill_svc_test` | `src/program/prefill_svc_test.cpp` | every line's grammar and round-trip; unknown trailing tokens ignored; a >4096-byte line rejected; the error-code → `Hold` table is total (every code maps to exactly one of wait/error); `HELLO`/`PREFILL-READY-V1` version and hash refusal |
| `handoff_arena_test` | `src/core/handoff_arena_test.cpp` | the header/slot layouts (`static_assert` + explicit offset checks, the `pinned.hpp:51-56` pattern); the per-tier group layout (`tier_slot_off`/`tier_slot_bytes` agree with the file size, and a slot's size comes from its `tier`); the FREE→CLAIMED→READY→FREE lifecycle; **cross-process** via the binary re-execing itself with `--role` (the existing pattern, `src/core/pinned_shared_test.cpp:106-152`); dead-owner take-over; **pid-recycled** take-over via field 22; lease expiry of a `READY` slot whose client pid is gone (the slot-leak fix); the `statvfs` free-space refusal before `ftruncate`; `pack_hash`/`geom_hash` refusal with both values on the line; a `CLAIMED` slot is never readable; the smallest-fitting-tier claim preference never strands a big job |
| `saved_conv_wire_test` | `src/core/saved_conv_wire_test.cpp` | `SavedConversation` → payload → `SavedConversation` round-trip on a **synthetic** session (the pattern `conversation_validation_test.cpp` already uses, `.megamind/src/core/s31a-parking-split-notes.md:56-63`): single-GPU and 2- and 3-carve splits; empty buffers; a 16 MiB+ buffer spanning several `ConversationBuffer` segments; `owner` forced to `kNoOwner` on write and assigned on receive; the record stream's END marker; a truncated payload refused before any write |
| `prefill_driver_test` | `src/program/prefill_svc_test.cpp` (same binary, a second suite) | the §4.5 state machine over every edge including `CANCEL` in each state; the retry bound (neutering it makes the end-to-end simulation run to its pass cap without finishing — the S3.8 anti-vacuity shape); a `DONE` with no `CLAIM` closes the connection; nonce mismatch refuses a slot |
| `serve_driver_test` (extend) | `src/program/serve_driver_test.cpp` | `Phase::await_prefill` and `Step::await_prefill` in the dispatch table; `Wait::prefill` in the classification; an end-to-end simulation where the prefill instance answers `ERR full` twice and then `DONE`, and both requests finish; **no engine thread blocks**; a waiter is not killed by the watchdog |
| `serve/test_server.py` (extend) | existing | the new `WAIT … prefill-busy` reason is parsed and shown on `/status`, `/slots`, `/metrics`; `refusal_is_temporary("…prefill…")` is True |

**The build check stays `cd build && ninja`, and the CPU tests build in `/tmp/a2build` with
`STRATA_BUILD_TESTS=ON`** (`.megamind/stage4-plan.md:82-83`). `pool_test`/`expert_multi_test` skip on
this box for want of AVX-512 — pre-existing, not a regression (`.megamind/src/kernels/cpu/pool-notes.md:64-65`).

### 7.4 What only the owner's box can verify

Everything that needs a model. The owner's runbook, at a restart:

```
# 1. the prefill instance (its own GPU, or a card with room for a second full engine)
engine/strata --role prefill --pack … --native … --ple-gguf … --expert-profile … \
  --expert-cache auto --prefill auto --max-context 262144 --kv int8 --kv-resident 32768 \
  --layer-split auto --conversation-cache-mib 20000 \
  --prefill-listen /dev/shm/strata-handoff/iq3_s-16,34/a/prefill.sock \
  --prefill-slots 2 --prefill-slot-tiers 1024,16384,131072 2> prefill.log &

# 2. two decode instances, each with its own serve/server.py + config
engine/strata --role decode --serve --serve-slots 2 … \
  --prefill-endpoint /dev/shm/strata-handoff/iq3_s-16,34/a/prefill.sock 2> decode1.log &
```

Look for, in order:

1. `strata prefill: serving on …; slots 2,2,2 (tiers 1024,16384,131072), max 131072 tokens, arena 8.2 GiB`
2. `strata serve: prefill endpoint …: PREFILL-READY-V1 (slots 2,2,2, max 131072 tokens); handoff on`
3. a first request: `CLAIM`, `SEG`, `DONE … 1033605768 B …`, then the decode instance's
   `conversation cache: restored N tokens (checkpoint) in X ms`
4. a second request of the **same chat**: the prefill instance's activity line shows
   `reused=… read=0`, and the handoff costs only the copy — that is §1.3.3 working
5. `handoff_ms=` on the `DONE` tail, and `slots_free=` never stuck at 0 after the load stops
   (the leak check)
6. kill the prefill instance mid-request: the decode instance logs the endpoint-down line, holds, and
   the client sees no `ERR`; restart it and the held request runs
7. kill a decode instance after `DONE` and before `ACK`: the prefill instance's `slots_free=` recovers
   within `STRATA_HANDOFF_LEASE_MS` — **this is the slot-leak fix, measured**
8. `STRATA_STATE_HASH=1`: the same prompt through the handoff and read locally must hash identically
9. `--prefill-retries 1` against a deliberately tiny arena: the request ends with the named retry ERR,
   not forever

---

## 8. Staged plan for S4.3

Each step is one agent run, owns a disjoint set of files (risk R13,
`docs/STAGE3-CONCURRENCY.md:1237`), and has a test that proves it. Steps 1-4 are CPU-only and can run
**before** any engine is started. Steps 5, 6 and 8 touch `generate.cpp` and must be strictly sequential.

**`CMakeLists.txt` is owned by exactly one step: S4.3.1.** It adds all four new targets
(`saved_conv_wire_test`, `handoff_arena_test`, `prefill_svc_test`, `prefill_svc_server` if it becomes a
target) in one commit, following the existing `STRATA_BUILD_TESTS` blocks
(`CMakeLists.txt:654-695`). S4.3.2/.3/.5 must **not** edit it — they hand their target lines to
S4.3.1's owner, or S4.3.1 adds them up front from this document and they are inert until the sources
land. Two agents in `CMakeLists.txt` is the R13 violation most likely to happen here, because it is
one line each and therefore feels free.

| # | step | owns (create/edit) | must not touch | the test that proves it |
|---|---|---|---|---|
| **S4.3.1** | the payload codec: `SavedConversation` ⇄ flat payload | new `include/strata/core/conversation_wire.hpp`, new `src/core/conversation_wire.cpp`, new `src/core/saved_conv_wire_test.cpp`, `CMakeLists.txt` | `generate.cpp`, `serve/*`, `setup.py`, any kernel | `saved_conv_wire_test`: round-trip on synthetic single-GPU, 2-carve and 3-carve sessions; empty buffers; a >16 MiB buffer spanning segments; truncation and hash refusal. **CPU-only, no engine.** |
| **S4.3.2** | the handoff arena: header, slot table, lifecycle, leases, sizing | new `include/strata/core/handoff.hpp`, new `src/core/handoff.cpp`, new `src/core/handoff_arena_test.cpp`, `CMakeLists.txt` | everything in S4.3.1's files, `generate.cpp`, `serve/*` | `handoff_arena_test`, including the cross-process `--role` re-exec case, dead/recycled owner, lease reclaim of a `READY` slot, the `statvfs` refusal, hash refusal. **CPU-only, no engine.** |
| **S4.3.3** | the control protocol: `prefill_svc.hpp` + the pure state machine | new `include/strata/program/prefill_svc.hpp`, new `src/program/prefill_svc_test.cpp`, `CMakeLists.txt` | `serve_proto.hpp`, `serve_driver.hpp`, `generate.cpp`, `serve/*` | `prefill_svc_test`: every grammar + round-trip; the error-code → `Hold` table is total; the §4.5 state machine over every edge; nonce/`CLAIM`-before-`DONE` rules. **CPU-only, no engine.** |
| **S4.3.4** | `MtpDrafter::bind_kv_only` | `include/strata/core/mtp.hpp`, `src/core/mtp.cpp` | `generate.cpp`, `prefill.cpp`, anything in S4.3.1-3 | builds; `conversation_snapshot_test` still passes unchanged (the draft K/V format is untouched). The behavioural proof is the owner's `STRATA_STATE_HASH` run. |
| **S4.3.5** | the prefill instance: role plumbing, the job queue, the socket server, the arena writer | new `src/program/prefill_svc_server.cpp`, `generate.cpp` (the `--role` parse + the role gates around `:3748`, `:3944`, `:4014`, `:5220`), `CMakeLists.txt` | `serve/server.py`, `setup.py`, `serve_proto.hpp`, `serve_driver.hpp` | `prefill_svc_test` gains a server-side loop simulation against a fake socket; `cd build && ninja`; the serial proof scripts still pass (the role gates are an early return, never an edit of the serial text) |
| **S4.3.6** | the decode instance: `Phase::await_prefill`, the handoff client thread, the mount path | `include/strata/program/serve_driver.hpp` (the new `Phase`/`Step`/`Wait` values), `src/program/serve_driver_test.cpp`, `generate.cpp` (the `run_prefill_step` branch at `:7003-7008`), new `src/program/prefill_svc_client.cpp` | `src/core/*`, `serve/server.py`, `setup.py` | `serve_driver_test` extended: the dispatch table, `Wait::prefill`, an end-to-end simulation (full → full → DONE, both requests finish, no engine thread blocks, the retry bound bites). Serial proof byte-identical. |
| **S4.3.7** | the wait-queue seam: the `prefill` reason through `/status`, `/slots`, `/metrics` | `serve/server.py`, `serve/test_server.py` | `src/core/*`, `generate.cpp`, `setup.py` | `python3 -m pytest serve/test_server.py -q` green with the new reason pinned; `refusal_is_temporary` unchanged for it |
| **S4.3.8** | the operator surface: `setup.py` flags, `--help`, docs | `setup.py`, `src/program/generate.cpp` (the `usage()` block only, `:590-700`), `docs/DETAILS.md`, `README.md` | everything else | `setup.py --help` shows the flags; `engine_supports` refuses them for an old binary (`setup.py:1165-1202`); `--role bogus` exits 2 |
| **S4.3.9** | integration + the owner runbook | this document (a "what landed" section), `.megamind/` | new source | full CPU suite green; the §7.4 runbook is what the owner runs |

**Sequencing.** 1 ∥ 2 ∥ 3 (disjoint files, and 2 and 3 both depend only on 1's *format*, which is
specified here in full). 4 is independent of 1-3. 5 needs 1+2+3+4. 6 needs 1+2+3+5. 7 needs 6. 8 needs
5+6. **Never two agents in one file** — `generate.cpp` appears in 5, 6 and 8 and therefore those three
are strictly sequential.

**The S4.2 handoff.** `include/strata/program/serve_driver.hpp`,
`src/program/serve_driver_test.cpp`, `serve/server.py` and `serve/test_server.py` are **S4.2's files**
(`.megamind/stage4-plan.md:68`). S4.3.6 and S4.3.7 edit them, so they may not start until S4.2 has
landed — which is already the plan's rule ("S4.3 after S4.0 + S4.2", `.megamind/stage4-plan.md:72`).
S4.3.1-.4 do not touch any of them and can start immediately.

**What can be built and tested before the owner ever starts an engine:** S4.3.1, S4.3.2, S4.3.3,
S4.3.4 (build only), and the pure parts of S4.3.5/6's state machines. That is most of the risk — the
format, the lifecycle, the leases and the protocol — retired before a single GPU cycle is spent.

---

## 9. Open questions for the owner

These need a decision, not an implementation. Each says what the answer changes.

**OQ-S4-1 — does the prefill instance get its own GPU?**
One prefill instance needs its own weights (1466 MiB, `strata-iq3_s.log:11`), its own session arenas
(0.554/0.560/0.547 GiB per stage at 524288, `docs/STAGE3-CONCURRENCY.md:135`), its own MTP (836 MiB,
`strata-iq3_s.log:16`), its own prompt loan (4.95 GiB, `strata-iq3_s.log:33`) and its own expert cache.
The decode instances keep theirs. On three 24 GiB cards that is 1 prefill + 2 decode, or 1 + 1 with
room. **Answer changes:** the topology in §5.4, and whether `--layer-split` on the prefill instance
should be the same as the decode instances' (it must be, or `geom_hash` refuses — §3.10).

**OQ-S4-2 — how many handoff slots, and how big?**
The default proposal is 2 slots in each of 3 tier groups = 6 slots, sized per group = **≈ 8.2 GiB** of
tmpfs (§3.8). This box has 24 GiB free in `/dev/shm` and 3 GiB `MemAvailable` (`df -h /dev/shm`,
`free -g`). A single 524 288-token tier with 2 slots is **22.8 GiB**. tmpfs pages are RAM, and the
arena sits next to the 46.8 GiB expert arena. **Answer changes:** `--prefill-slots`,
`--prefill-slot-tiers`, and whether the owner adds RAM (the plan says they will,
`.megamind/stage4-plan.md:86`).

**OQ-S4-3 — a second conversation-cache budget.**
The prefill instance's `--conversation-cache-mib` is RAM the decode instances do not have. The owner's
config already asks for 20 000 MiB (`strata-iq3_s.json:37-38`). Two decode instances + one prefill
instance = three budgets. **Answer changes:** whether the prefill instance's reuse win (§1.3.3) is
actually affordable, or whether it should run with a small cache and re-read more.

**OQ-S4-4 — is the reverse handoff (S4.5) in scope?**
§1.3.3's limitation: the prefill instance's cache never contains tokens a decode instance generated,
so a chat that moves between decode instances re-reads. The fix is the same slot machinery in the
other direction (`TAKE` is reserved, §4.3). It is a real feature, not a detail, and it roughly doubles
the transport work. **Answer changes:** whether S4.3 ships with a per-chat instance affinity rule
(the cheap answer) or with the reverse handoff.

**OQ-S4-5 — one prefill instance, or several?**
One is the settled shape (`.megamind/stage4-plan.md:18-23`). Several is trivially supported by the
per-instance directory rule (§3.2) but nothing schedules across them. **Answer changes:** whether
`--prefill-endpoint` accepts a list (and then S4.3 needs a pick rule, which is a new scheduler and a
new set of bugs).

**OQ-S4-6 — does a decode instance ever read a prompt locally?**
`--prefill-fallback local` keeps `Prefill` + the 4.95 GiB loan alive in every decode instance, which is
VRAM and complexity, in exchange for surviving a prefill-instance outage by degrading instead of
holding. **Answer changes:** §1.5, and whether the decode role's VRAM budget must reserve the loan.

**OQ-S4-7 — the `--max-context` mismatch case.**
A decode instance at `--max-context 262144` and a prefill instance at `524288` produce different
`geometry` arrays? No — `geometry_key` does not include `max_cells`
(`src/core/conversation_state.cpp:14-19`), but the payload's K/V *is* sized by the token count, and
`conversation_kv_validate` compares `cells` (`src/core/conversation_snapshot.cpp:182-187`). A payload
read at 400 000 tokens cannot be mounted into a 262 144-cell session. **Answer:** the design refuses it
(`ERR toobig`, priced at `--prefill-max-tokens`, which defaults to the prefill instance's
`--max-context`), and the decode instance's own `prompt_never_fits` check
(`include/strata/program/serve_driver.hpp:1004-1007`) runs first. Confirm that "both `--max-context`
must match" is the operator rule, or make `--prefill-max-tokens` the only knob.

**OQ-S4-8 — Windows and HIP.**
The whole transport is POSIX (`flock`, `mmap`, `/proc`, `AF_UNIX`, `statvfs`). Stage 1 already refuses
a shared arena on Windows (`src/core/pinned.cu:256-263`, `.megamind/src/core/arena-notes.md:52-53`).
**Answer:** `--role prefill|decode` must be refused at start-up on `_WIN32` with a clear message, and
HIP is fine (the transport never touches the GPU API). Confirm that is acceptable, or specify a
Windows transport (out of scope for S4.3 either way).

**OQ-S4-9 — `MtpDrafter::bind_kv_only` or eat the 578 MiB?**
§1.4. The clean answer is a new method in `mtp.hpp`/`mtp.cpp`. The cheap answer is calling `bind()` in
the prefill role, which needs `NativeHead` loaded and costs 497 MiB + 81 MiB of VRAM for a head the
prefill instance never runs. **Answer changes:** S4.3.4's scope.

**OQ-S4-10 — does the prefill instance serve more than one model?**
No, and the arena directory makes it impossible (one `pack_hash` per arena, §3.2). Confirm, so nobody
builds a multiplexer.

---

## 10. Deliberately out of scope

* **Token-level batching of two decode slots into one verify window.** That is S4.1/S4.4
  (`docs/STAGE3-CONCURRENCY.md:61-67`, `:1267-1268`; `.megamind/stage4-plan.md:47-53`).
* **The reverse handoff** (decode → prefill), §9 OQ-S4-4. Reserved verb: `TAKE`.
* **Multi-segment handoffs** (`seq > 1`), incremental K/V, and slot sharing across requests. §3.9.
* **A prefill instance that preempts a running job.** §5.2.
* **Scheduling across several prefill instances.** §9 OQ-S4-5.
* **Persisting payloads to disk.** Snapshots stay host-RAM/tmpfs only
  (`docs/STAGE3-CONCURRENCY.md:77`, `:1270`).
* **Windows and HIP transports.** §9 OQ-S4-8.
* **Changing the client wire.** `serve_proto.hpp` is not edited (§1.6 rule 2).
* **A second `Prefill`, a second `Verifier`, or a second thread that touches a device in any process.**
  Risk R5, `docs/STAGE3-CONCURRENCY.md:283-287`, `:1229`.

---

## Appendix A — how the numbers in this document were produced

**No engine, no model, no GPU work was started.** Everything comes from reading source, reading the
live log read-only, and host-only arithmetic.

### A.1 Sources read

`.megamind/stage4-plan.md`; `docs/STAGE3-CONCURRENCY.md` (all 1299 lines); `docs/DETAILS.md:340-401`;
`include/strata/core/conversation_cache.hpp`; `conversation_buffer.hpp`; `conversation_snapshot.hpp`;
`conversation_memory.hpp`; `src/core/conversation_state.cpp:1-60`;
`src/core/conversation_snapshot.cpp:1-207`; `include/strata/core/pinned.hpp`; `src/core/pinned.cu:1-620`;
`src/core/expert_source.cpp:1355-1380`; `include/strata/program/serve_proto.hpp`;
`serve_swap.hpp`; `serve_driver.hpp` (all 1402 lines); `slot.hpp:1-140`;
`include/strata/core/mtp.hpp`; `src/core/mtp.cpp:142-240, 375-410, 413-560, 596-645, 679-760`;
`include/strata/prefill/prefill.hpp`; `src/prefill/prefill.cpp:137-250`;
`include/strata/core/session.hpp` via `docs/STAGE3-CONCURRENCY.md:108-113`;
`src/core/session.cpp:30-80`; `src/core/layer.cpp:527-660`;
`src/program/generate.cpp` (the `Options` block `:228-480`, the flag chain `:1240-1400`, the session
block `:2430-2530`, the cache/head block `:2600-2680`, the serve gate `:3748-3930`, the verifier block
`:3930-4030`, the conversation cache `:4036-4130`, `park_current` `:4240-4266`, the registry/wire
`:4268-4360`, `on_chunk` `:4412-4425`, the watchdog `:5190-5240`, the prompt-path helpers `:5245-5270`,
the swap plan `:5769`, `prep_request` `:6315`, `plan_prompt_segments` `:6853-6883`,
`reset_request_to_token0` `:6904-6930`, `run_prefill_step` `:6983-7037`, `finish_prefill`
`:7041-7100`); `setup.py:1140-1202, 1800-1930`; `serve/server.py:118-300, 360-440, 855-895,
1430-1540, 2540-2660`; `strata-iq3_s.json`; `CMakeLists.txt:629-695`; and the `.megamind` notes listed
in the task brief.

### A.2 Log-derived measurements (read-only `grep`/`awk` over `strata-iq3_s.log`)

`grep -o 'snapshot_bytes=[0-9]*' strata-iq3_s.log | cut -d= -f2 | sort -n` over **871** parks:

```
n=871  min=237413652  p50=1033605768  p90=2029940496  p99=2939786284  max=3180486108
```

Per-token rate over the 30 parks with more than 100 000 tokens:

```
min 20655 B/token   mean 22532 B/token   max 27342 B/token
```

(the "3123884 B/token" outlier is an 81-token park at 237 MB — that is the **fixed floor**, ≈ 236 MiB
of per-layer running state, not a rate. It is the same floor `docs/STAGE3-CONCURRENCY.md:144-148`
prices: 112 MiB of GDN + the RoPE table + buffers.)

Cost of moving one, from the engine's own lines:

```
parked 153983 tokens in 410.0 ms; snapshot_bytes=3180486108
swap …: ok in 220 ms; saved 2785504484 B, restored 0 B
swap …: ok in 313 ms; saved 2785539216 B, restored 799228852 B
conversation cache: restored 97934 tokens (checkpoint) in 310.1 ms
```

Prompt reuse, the fact §1.3.3 is built on:

```
prompt 245946 tokens = 245751 reused + 195 read in 1116 ms      (strata-iq3_s.log:6649)
prompt 222112 tokens = 221911 reused + 201 read in 1144 ms      (strata-iq3_s.log:6449)
```

### A.3 Host facts

```
df -h /dev/shm      ->  70G total, 47G used, 24G available
free -g             ->  Mem: 94 total, 90 used, 3 available
ls -l /dev/shm      ->  shared_experts.dat 50294992896 B, strata-core-leases 313 B
                        (both belong to live processes: read-only, never touched)
```

### A.4 The scratch-binary recipe

For anything the log does not say, use stage 3's recipe (`docs/STAGE3-CONCURRENCY.md:1291-1298`): link
a scratch binary against `build/libstrata_engine.a` + `libstrata_core.a` + `libstrata_kernels*.a` +
`libstrata_spec.a` + `build/ggml/src/libggml-{cpu,base}.a` and call the cost functions
(`session_bytes`, `qsa_state_bytes`, `conversation_snapshot_bytes`) with the real `ModelGeometry`.
**It never calls `session_init`** — that one does `cudaHostAlloc`
(`src/core/session.cpp:74`, `src/core/layer.cpp:631`).

### A.5 Guesses, labelled

* The ≈ 1.2 GiB freed by skipping the verifier + head + draft head in the prefill role (§1.3.2) is a
  **sum of logged allocations**, not a measured `cudaMemGetInfo` delta. The resulting "≈ 900 more
  expert slots" is arithmetic on the blob size the log implies — `expert cache auto: 15.51 GiB free,
  700 MiB reserved (+86 MiB for the draft head) -> 5945 slots` (`strata-iq3_s.log:20`) gives
  (15.51 − 0.786) GiB / 5945 ≈ 1 382 400 B, which is `expert_layout().max_blob`, the divisor used at
  `src/program/generate.cpp:2657`.
* The tier prices in §3.8 use the measured 20.7-22.5 KiB/token rate plus the 236 MiB floor. The real
  figure comes from `conversation_snapshot_bytes` at each tier, which S4.3.2 must print at start-up.
* "The copy-out of a 3 GB slot takes a few hundred ms" is inferred from the in-process park/restore
  times in A.2. A cross-process tmpfs read should be comparable (both are page-cache memcpys), but it
  is **not measured** and the owner's runbook step 5 measures it for the first time.
* `MtpDrafter::bind_kv_only` being sufficient is argued from `record_forward`'s early return at
  `src/core/mtp.cpp:486` and the graph inputs at `:600-616`. It is **not** proven until S4.3.4 builds
  and the owner runs the `STRATA_STATE_HASH` comparison.
