# S3.9 — a parked branch was being stolen by another request (regression after S3.8)

Owner's report: *"the parking doesnt seem to work now, as when a new request arrives it always kicks
the currently running request out with an 'it was decode but not parked' error."*

## This is NOT the save failing — the park succeeds

`strata-iq3_s.log` (22:37 run, `--serve-slots 3`, 15 048 MiB budget):

```
park: slot 14 parked 10394 tokens / 994 MiB; parked=5 entries, 1685 MiB of a 15048 MiB budget
swap 35: slot 14 -> 15 ok; saved 1042399212 B, restored 0 B, parked=5
conversation cache: restored 9849 tokens (checkpoint) in 25.7 ms; parked=4     ← slot 15 TOOK it
swap: slot 14's parked branch (10394 tokens) is no longer in the cache - pruned or replaced ...
slot 14 would step in decode with no conversation to step against - ending it
```

**`evictions=0` for the entire run** — nothing was ever pruned. The entry count drops 5→4 at the exact
moment another request mounts. 5 occurrences, every one the same shape.

## The mechanism

A parked entry is **one object**: its live branch, its checkpoint chain and its K/V pages. A new
request looks a branch up by **prefix** (`ConversationCache::best()` → `find()`), which happily matches
one of *another slot's* checkpoints — and the mount then `take()`s the **whole** entry.

Slot 15's prompt shares a 9 849-token prefix with slot 14's 10 394-token branch (same chat), so
`best()` matched slot 14's checkpoint, `take()` removed slot 14's live branch, and when slot 14 asked
for its own branch back `seek_mount_index` found nothing → `step_gate` correctly ended it.

The cache had **no concept of ownership**. Nothing in it distinguished "warm storage for the next
request of this chat" from "the only copy of a conversation that is still running".

## Why it appeared after S3.8

S3.8 made mid-**prefill** slots park (`publish_prefill_branch` + `outgoing_for(phase, at)`). Before
that, a prefill never parked anything, so far fewer entries were owned by live slots and the theft was
rare. The bug was always there; S3.8 put it on the common path.

## The fix — a claim, not a pin

`SavedConversation::owner` (`kNoOwner` = -1): the request whose branch this is, **while that request is
still running**.

* `find()` / `best()` / `best_exact()` take a `requester` and skip entries claimed by somebody else.
  `seek_mount_index` passes `s.id`, so a slot can always mount its own branch.
* `put(image, owner, held)` records the claim. One signature with defaults — an overload pair would
  make `put(img, 5)` ambiguous between `owner` and `held`.
* `release_owner(id)` / `release_conv_claims(id)` run from `drop_ctx`, the one thing that destroys a
  context, so every end path (finish, error, watchdog, refusal) releases. A claim can't leak.
* `return_mount_image` re-parks with the claim it came with.
* `victim_to_prune()`: eviction prefers **unclaimed** entries. A claimed entry is a live conversation —
  pruning it doesn't cost a re-read, it ends a request. When everything is claimed, LRU again: the
  budget is the hard limit and the owner gets the honest "pruned while it waited" line.

`park_owner_for(id)` decides the claim: `working_of(id) == finished` → `kNoOwner`, else `id`. That is
exactly "does this slot still owe a step", and it keeps stage 2 working — a **finished** conversation's
branch is parked unclaimed, so the next request of that chat still resumes from its prefix.

## Traps

* **A claim is not a pin.** It only refuses *mounting by a different request*. LRU may still evict it.
  Pinning would be a guess about which client comes back and would permanently steal a slot.
* **Do not claim for `req_id` at the `prep_request` park.** That park saves the branch the *session*
  holds, which belongs to the still-mounted slot, not to the request being prepped. Claiming it for
  `req_id` would hide another slot's conversation behind the wrong id and release it at the wrong time.
* **`kNoOwner` as the requester means "nobody in particular"** — a serial server, or a lookup before
  any slot owns anything. It hides nothing, so `--serve-slots 0/1` is untouched.
* `release_owner` must run **after** the branch is parked, not before, or the entry would be evictable
  in the window where it matters most.

## Verified

`conversation_cache_test` 4252 → **4271**; `serve_swap_test` 169 → **173**. ctest 18/19 (only the
pre-existing AVX-512 skip); serve_driver 377, conversation_memory 23, conv_cache, serve_proto 140,
slot 176, prefill_loan 128 all pass; `serve/test_server.py` 65 passed / 2 skipped. **Serial driver
byte-identical (3652 B)** and the output proof PASSes.

The new cache test reproduces the log exactly: an entry with live `{1,2,3,4,5}` claimed by slot 14 and
a checkpoint `{1,2,3}`; request 15 asks for `{1,2,3,99}` and must get **0**, with slot 14's entry still
in the cache.

Anti-vacuity, three mutations: removing the claim filter from `find`/`best_exact` → caught in both
suites; `release_owner` neutered → 2 failures (the leak shows up as stage 2's prefix reuse breaking);
`victim_to_prune` back to plain LRU → caught.

## Owner action

Restart. The `no longer in the cache` / `would step in decode with no conversation` pair should be
gone. A new request that shares a prefix with a **running** slot's parked branch now reads a bit more
of its own prompt instead of destroying that slot — expect a slightly lower `N reused` on those
requests, and no ended conversations.
