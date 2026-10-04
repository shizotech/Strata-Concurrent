# S3.1d — mount/unmount a slot against the one live session

Files (mine):
```
include/strata/program/serve_swap.hpp   NEW  the ordered swap + the pure decisions (CPU-only)
src/program/serve_swap_test.cpp         NEW  105 CPU-only ordering/decision checks
src/program/generate.cpp                serve loop: the swap call site, per-slot records, ordering
CMakeLists.txt                          serve_swap_test under STRATA_BUILD_TESTS
docs/DETAILS.md                         the "Mounting and unmounting a conversation" section
```
NOT touched (per the contract): slot.hpp, serve_proto.hpp, conversation_state.cpp,
conversation_snapshot.hpp, serve/server.py.

## THE SWAP API — what S3.1e calls

```cpp
// generate.cpp, serve loop. Engine thread only.
bool swap_to(int64_t incoming_id, std::string& serr, bool& poisoned,
             bool restore_positions = true);
```
* `mount_image` (an `optional<SavedConversation>` in the loop) is the **input**: set it to the image
  taken from the `ConversationCache` before calling, or leave it empty for a hand-over with no
  restore. The swap consumes it, or puts it back.
* `restore_positions = false` when the hand-over is driven by a **request line** (the body rebuilds
  mrope/sfx/sampling itself). `true` (default) for a **scheduler-driven** hand-over — that is the
  form S3.1e will use, and it is the only one that restores the drafter history, the sampling
  params, the penalty window, the PCIe share, the position table and `mtp.set_prompt_len`.
* Returns false + `serr`. `poisoned` = the restore itself failed (session unusable -> `return 1`).
  `swap_wrote_session` = the session was written even though the swap failed: **do not retry**, and
  do not run the request. `mounted_id` is updated in that case too, because it must describe the
  session.
* State the loop keeps: `mounted_id` (the §5.3 active slot), `mrope_owner`, `conv_of(id)` ->
  `serve_swap::SlotConv&` (a `std::deque`, so references stay valid — a `vector` dangled),
  `prune_conv()` (drops records whose registry row is gone; call once per request).

## The order (fixed, in `serve_swap::kOrder`)

`drain_residency` (apply_pending(true)) -> `return_loan` (refill()) -> `validate` ->
`unmount` (snapshot save) -> `mount` (snapshot restore) -> `draft_kv` (mtp.kv_restore) ->
`adopt` (per-slot mirror) -> `device_state` (cvec + mrope upload).
The two drains **always** run, even for an empty plan. `validate` is before `unmount`: that is what
makes "a failed validation leaves the outgoing state intact" a property of the order.

## Decisions that were made (do not re-litigate)

* **The checkpoint chain is NOT copied per slot.** It is a shared prefix chain (PR #65's pinned
  system-prompt root); per-slot copies would break that reuse and cost ~118 MB each. The
  `ConversationCache` image already carries `live`/`live_imgs`/`checks`/`cvec`, and
  `conversation_snapshot_restore` puts them back. `SlotConv` mirrors them only so a mount with no
  image still has something, and for /slots.
* **`check_clock` is process-wide**, only ever moved forward by a mount — rewinding it would make a
  fresh checkpoint older than the chain it joins.
* **A hand-over with no restore clears `live`/`checks`/`live_ok`** for the incoming slot. The
  session's positional cells still hold the *outgoing* tokens; a checkpoint is only valid while the
  cells below it hold its tokens. Re-reading from 0 is the honest answer.
* **A slot whose branch was not parked is `resumable = false`** and may never be mounted again —
  that is the silent-garbage trap, and it is reported once on stderr.
* **`park_current` now returns `Saved{stored,skipped,failed}`**, not `bool`. 0.1.30 collapsed
  parked/refused/nothing into true; a hand-over cannot, because "not parked" changes what is
  mountable. The serial call site compares `== Saved::failed`, which is exactly 0.1.30's `!`.
* **After a successful swap the request body must NOT park** (`parked_by_swap = true`): the session
  now holds the *incoming* branch, so parking would snapshot what was just mounted (a GB of memcpy
  and a duplicate entry).
* **mrope (R7)**: `sync_positions()` runs at the top of every iteration, *before* the request body
  may rewrite the one host table — there is no way to recover the outgoing slot's positions after
  that. A `MropeScope` guard restores the mounted slot's table if the request bails out after
  touching it (bad embeddings file, prompt too long, token out of vocab).

## Traps hit (do not repeat)

* `conv_of` returning a reference into a `std::vector` that another `conv_of` reallocates. Use a
  `std::deque`.
* A local struct's destructor cannot name enclosing-function locals in its default member
  initialisers — build the `std::function` members after construction.
* `refill_one`/`refill` and `trace`/`tr` had to move **above** the request body (they only capture
  startup state) or the swap cannot return the loan before it saves.
* `park_bytes` must be set only when `conversations.put` actually stored; `parked_ok` is what
  decides `Saved::stored`, because `put` can refuse.

## Env switch

`STRATA_NO_SWAP=1` — tagged wire stays, hand-overs never happen, 0.1.30's serial path. Reported at
startup (`strata serve: slot swap on|off`) and in INFO as `slot_swap=0|1`. Documented in `--help`.

## Verification

* `cd build && ninja` clean (the `no .sframe` ld lines are pre-existing noise).
* `/tmp/s3build` ctest: 17/18 pass; only `expert_multi_test` fails (no AVX-512 on this box —
  pre-existing, untouched).
* `serve_swap_test` 105 · `slot_test` 176 · `serve_proto_test` 140 · `prefill_loan_test` 54 ·
  `conversation_cache_test` 4232 · `conv_cache_test` PASS · `conversation_memory_test` 23 ·
  `conversation_validation_test` 1323 host-only (built by hand against `build/libstrata_*.a` —
  `/tmp/s3build` is CUDA-OFF so it has no such target).
* `python3 -m unittest serve.test_server` -> 66 tests, OK (2 skipped).
* Anti-vacuity: moving `validate` after `unmount` in `run()` fails 6 of the 105 checks.

## Still unverifiable without a live engine — OWNER ACTION at a restart

A real swap moves gigabytes through real `SessionState`s on a 3-way split. The CPU tests pin the
order and the decisions, not the bytes. The check is `STRATA_STATE_HASH`: mount A, mount B, mount
A again, and the two A hashes must be identical.

```
# 1. stop the two live servers, then start one with concurrency and a parking budget that fits,
#    with the fingerprint and the step trace on:
STRATA_STATE_HASH=1 STRATA_TRACE=1 engine/strata --serve --model ... --layer-split auto \
  --serve-slots 3 --conversation-cache-mib 12288 --prompt-cache 8 --adapt-every 100000 \
  ... < /dev/null > /tmp/s.log 2>&1 &

# 2. three short requests with DIFFERENT prompts (ids 1/2/3), then repeat 1 and 2 so each
#    conversation is mounted, unmounted and mounted again.

# 3. the fingerprints and the swap bill
grep -E 'STATE_HASH|serve: swap|swap steps|slot swap' /tmp/s.log
```
Expected: for a conversation that is re-mounted, its `STATE_HASH` line matches the earlier one for
the same conversation — `gdn=`, `ple=`, `tail=`, `pooled=`, `kv=`, `mtp=`, `dead=` identical;
`stale=`/`pooled_full=` may differ (cells past L). `--adapt-every 100000` keeps residency static so
the comparison is strict (§7.2). The `serve: swap N: ... in X ms; saved B, restored B` lines close
**OQ3** at 32k / 128k / 512k tokens.

## For S3.1e specifically

* `mounted_id` (the engine's) and `Registry::active_id()` (the registry's) are **not the same
  variable and briefly disagree on purpose**. When a request finishes, `slot_finish` releases its row
  and clears `active_id`, but the session still *reflects that conversation* — `mounted_id` keeps
  naming it, and its record survives `prune_conv()` for exactly that reason. The next hand-over is
  what saves it. Do not "fix" this by setting `mounted_id = kNoId` at the end of a request: that
  would drop the outgoing branch on the floor instead of parking it.
* `swap_to` is safe with `mount_image` empty (a hand-over with no restore: the incoming slot re-reads
  from token 0). It is also safe to call when `incoming_id == mounted_id` — it returns true and does
  nothing, which is the zero-swap default §3.3 wants.
* `serve_swap::budget_of(conversations)` + `fits(b, incoming, held)` mirror
  `ConversationCache::can_fit` for the admission decision; use them rather than re-deriving it.
* `Plan`/`Hooks`/`Report` are there if you want to drive `serve_swap::run` directly, but `swap_to`
  already builds them; you should not need to.
