# Parked prefixes — the multi-prompt cache (stage 2, done)

Files: `include/strata/core/conversation_cache.hpp`,
`include/strata/program/conv_cache.hpp`, `src/program/generate.cpp`,
`src/core/conversation_cache_test.cpp`, `src/program/conv_cache_test.cpp`.

## Two different caches — do not confuse them

| | what it holds | policy |
|---|---|---|
| `--prompt-cache N` (6) | checkpoints inside **one** conversation branch (a prefix chain) | `program::conv_cache::eviction_victim()`: root pinned, rest LRU |
| `--conversation-cache-*` | **whole** conversations (independent branches) in host RAM | `ConversationCache`: 8 slots, LRU, no pin |

Stage 2 is the second one. It was already in the tree but off by default, capped
at 4, and pruned **FIFO** — which is exactly the owner's complaint (an agent that
switches prompts blows away the prefix it will need again).

## What changed

* `ConversationCache::default_slots = 8`; `Options::conversation_cache_slots`
  defaults to it (was 4).
* Per-entry use stamps in a parallel `std::vector<uint64_t> stamps_` (kept in sync
  by the private `erase_at()`), from a monotonic `clock_`. `best()` (a hit),
  `take()` (a mount) and `put()` (a park) advance it. `make_room()` evicts
  `lru_victim()` instead of `pop_front()`. Ties leave in park order, so the old
  FIFO answer is the degenerate case.
* **`best()` is no longer `const`.** Never introduce a `const ConversationCache&`.
* New: `slots()`, `budget()`, `clock()`, `use(i)`, `touch(i)`, `lru_victim()`.
  `touch()` exists for the checkpoint-chain resume path (a hit that did not mount
  the whole image is still evidence the branch is live).
* Deliberately **no pinned entry** at this level: parked conversations are
  independent branches, so pinning one is a guess about which client comes back
  and permanently steals a slot. The pin stays in `program/conv_cache.hpp`, where
  every retained item is a prefix of every other. Both now share an `lru_victim()`
  and a test asserts they agree on the same stamps.

## Defaults and the RAM trap

`conversation_cache_mib` default 0 → **8192**. But an 8 GiB budget on a machine
whose 47 GiB arena + 26.8 GB mapped PLE shard already ate the RAM is the review's
finding C1 in a new costume: `MemAvailable` counts reclaimable cache, so the
admission check passes and the OS pays for the snapshots by dropping the pages the
prompt path reads — the server ends up slower than the feature it gained.

So the **default** budget is `min(8 GiB, MemAvailable − 2.5 GiB floor − 4 GiB
request headroom)`, printed when it bites. An explicit `--conversation-cache-mib`
is never second-guessed. The INFO line reports `conversations.budget()` (what is
in force), not the option value.

### S3.1a: `--layer-split` parks now (stage 3's prerequisite)

The guard is **gone**. A parked conversation carries the first stage's state (CUDA0's, the
snapshot's `session` argument) plus one `ConversationStageSnapshot` per later stage plus the
MTP draft K/V, and `bytes()` counts all of it. `--conversation-cache-mib 0` still parks
nothing; an explicit budget is still never second-guessed by the machine-sized default.

What changed, and the traps:

* `SavedConversation::stage_parts` (new `ConversationStageSnapshot{layer_lo,layer_hi,state,kv}`)
  and a `core::ConversationStageSet` view over the engine's stage sessions. Every
  `conversation_snapshot_*` operation has a stage-set overload; the old signatures are those
  with an empty set. Full contract in `.megamind/src/core/s31a-parking-split-notes.md`.
* **`ConversationCheckpoint::stage_parts` is now required, not rejected.** `generate.cpp`'s
  `on_stage_chunk`/`checkpoint_at` already fill it on a split, so `view_validate` refusing any
  checkpoint with parts meant parking could never see a split's checkpoint chain at all. With a
  stage set the parts must be there, one per later stage, sized for that stage's carve.
* **`cudaDeviceSynchronize()` is per-device, and so is `kv_stream_reset`.** Every stage's save
  and restore is wrapped in `core::OnDevice(stage.device)`. A split mount does
  `1 + 1 + n_stages` syncs; the test counts them.
* **The draft state is not on CUDA0 on a split** - `MtpDrafter` binds to the last stage
  (`generate.cpp:3777`), so `ConversationStageSet::draft_device = mtp.device()`.
* Retention (`ConversationKvReuse`) stays single-GPU: the retained set is exactly the main
  session's QSA layers + the draft. A split's stage K/V is always captured fresh.
* A parked image is tied to the split it came from: change `--layer-split` and the images are
  refused, never half-mounted. The stage set must tile `[0, n_layers)` exactly.
* `session_init` allocates `qsa_states` with `n_qsa_layers()` entries and initialises only the
  range's - a synthetic session must do the same, and must build its `QsaState` array *after*
  allocating `idx_tail/idx_dead/idx_block_pos`, or the copies are stale.
* `ConversationBuffer` has no `data()`; compare with `operator==`/`read()`/`visit()`.

Verified: `conversation_validation_test` 1323 host-only / 2169 with the injected host backend
(two- and three-carve splits, a stage range with no QSA layer of its own, and every way a stage
part can be corrupt refused before any write); `conversation_snapshot_test` 2089 on a device
(was 1781); `conversation_cache_test` 4232 (was 4225). **No-drift proof:** the pre-change
snapshot test compiled against the new library passes 1781/1781.

## Tests

`g++ -std=c++20 -O1 -Iinclude -o /tmp/cc src/core/conversation_cache_test.cpp
src/core/conversation_memory.cpp && /tmp/cc` → 4225 checks (was 4149).
`src/program/conv_cache_test.cpp` → PASS, 10 checks (was 7).
Anti-vacuity: the new LRU assertions **fail** against a scratch header with the
victim forced back to FIFO, so they pin the policy, not just the outcome.
