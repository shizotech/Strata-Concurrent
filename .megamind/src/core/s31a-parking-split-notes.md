# S3.1a — parking across a layer split (DONE)

Files: `include/strata/core/conversation_cache.hpp`, `include/strata/core/conversation_snapshot.hpp`,
`src/core/conversation_state.cpp`, `src/program/generate.cpp` (guard + 5 call sites),
`src/core/conversation_validation_test.cpp`, `src/core/conversation_snapshot_test.cpp`,
`src/core/conversation_cache_test.cpp`, `docs/DETAILS.md`. **No CMakeLists.txt change** — the
existing `STRATA_BUILD_CONVERSATION_TESTS` targets already cover it.

## The contract the stage-3 scheduler (S3.1d/e) calls

```cpp
// core/conversation_snapshot.hpp
struct ConversationStage { SessionState* session = nullptr; int device = -1; };
struct ConversationStageSet {
    const ConversationStage* stages = nullptr;  // LATER stages only, in engine order
    int64_t count = 0;
    int main_device = -1;   // 0 on a split; -1 = "don't touch the current device" (single GPU / tests)
    int draft_device = -1;  // mtp.device(): the drafter is bound to the LAST stage
};

// core/conversation_cache.hpp
struct ConversationStageSnapshot {
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint state;      // that session's running state; .ids = live ids, .imgs empty
    std::vector<ConversationKv> kv;    // exactly qsa_alloc entries, NEVER a draft entry
};
struct SavedConversation { ...; std::vector<ConversationStageSnapshot> stage_parts; ... };
```

Every `conversation_snapshot_{bytes,capture_bytes,save,validate,restore}` has a stage-set
overload; the old signatures are those with an empty set. `generate.cpp` builds `conv_stages`
once next to `ConversationCache conversations` and passes it at all five call sites.

## Rules the implementation enforces

* `stage_parts.size()` **must** equal the stage count, both directions. A split image mounted
  without a stage set is refused; a single-GPU image mounted with one is refused. No silent
  format drift in either direction.
* The stage set must **tile** the model: main session `[0,K1)`, stage i `[K_i,K_{i+1})`, last
  `layer_hi == n_layers`. A split that skips or overlaps a layer is refused, not half-parked.
* `ConversationCheckpoint::stage_parts` (the upstream chain, already filled on a split by
  `on_stage_chunk`/`checkpoint_at`) is **required** and validated against each stage's carve
  when parking a split. Without a stage set it must stay empty — the old rule, unchanged.
* Retention (`ConversationKvReuse`) stays a single-GPU optimization: the retained set is exactly
  the main session's QSA layers + the draft. A split's stage K/V is always captured fresh.
* `bytes()` and the estimate include every stage part, its K/V directory, its ids, and each
  retained checkpoint's stage parts. A test pins `estimate == image.bytes()`.

## Traps hit (do not repeat)

* **`cudaDeviceSynchronize()` is per-device**, and so is `kv_stream_reset`. Every stage's save
  and restore is wrapped in `core::OnDevice(stage.device)`. A split mount does
  `1 (main) + 1 (draft) + n_stages` syncs — the test counts them.
* **The draft state is NOT on CUDA0 on a split.** `MtpDrafter` binds to the last stage
  (`generate.cpp:3777`), so its pools/host copies/stream are there: `draft_device = mtp.device()`.
* `session_init` allocates `qsa_states` with `n_qsa_layers()` entries and initialises only
  `[qsa_ord0, qsa_ord0+qsa_alloc)`; the rest are nulls. A synthetic session must do the same or
  the carve reads garbage. In `conversation_snapshot_test.cpp` the `QsaState` array must be
  built **after** `Fixture::alloc` of `idx_tail/idx_dead/idx_block_pos`, or the copies are stale
  and `checkpoint_targets` fails with "invalid indexer running-state target".
* A `Fixture` must not change `n_head_kv/head_dim/idx_key_dim` after construction — its pools
  were already sized from them.
* `ConversationBuffer` has no `data()`; compare with `operator==`, or `read()`/`visit()`.
* `ConversationCache::put` takes `SavedConversation&&` — an lvalue image needs an explicit move.

## Verification

* `cd build && ninja` clean (the `no .sframe will be created` ld lines are pre-existing noise).
* `conversation_cache_test` 4232 · `conversation_memory_test` 23 · `conv_cache_test` PASS ·
  `conversation_validation_test` 1323 host-only / 2169 with `CONVERSATION_TEST_TRANSFERS` ·
  `conversation_snapshot_test` **2089** on GPU 2 (was 1781).
* **No-drift proof:** the *pre-change* `conversation_snapshot_test.cpp` (`git show HEAD`)
  compiled against the *new* library passes 1781/1781.
* Anti-vacuity: deleting the stage-part validation loop fails "corrupt last-stage K/V rejected";
  skipping the stage-part restore fails "stage running state restored".
* `pool_test`/`expert_multi_test`: AVX-512 box dependency, pre-existing, not touched.

## Still unverifiable without a live split (owner action at a restart)

Real multi-device `OnDevice` switching, real `kv_stream_reset`/`kv_ring_restore` on the last
stage, and the wall-clock cost of a split park/mount (closes OQ3). `STRATA_STATE_HASH` after a
split mount is the check to run.
