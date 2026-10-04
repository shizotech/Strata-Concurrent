# S3.8 — two concurrent prefills erase each other and never finish

Owner's report: *"when two prefills happen at the same time and they swap, they seem to erase the
other's previous progress, as after each swap they always start at 0 tokens again which makes them
never finish."*

## Confirmed in the owner's log (`strata-iq3_s.log` 24745-24860)

Slot 2 (decode) parks and resumes normally. Slot 3 (prefill, 81 tokens) never does:

```
swap 5: slot 2 -> 3 ok in 63 ms; saved 814820860 B, restored 0 B, parked=1 ...
slot 3 re-reading from token 0 (81 tokens were dropped with it)
slot 3 reset to token 0: 81 prompt tokens will be read again (0 checkpoints dropped)
swap 6: slot 3 -> 2 ok in 50 ms; saved 0 B, restored 814820860 B ...
slot 2 resumed from 2 tokens
```

`grep -c "reset to token 0"` = **101**. A livelock, not a slow run: the decode slot makes progress,
the prefill slot restarts forever, and the watchdog cannot see it because every pass really does read
tokens.

## The chain — five links, any one of which hides the bug

1. `publish_decode_branch` (generate.cpp) early-returns unless `phase == decode`. A mid-**prefill**
   slot is therefore never published: `live_ok` stays false (`prep_request` clears it for the whole
   request) and `live` still describes the previous conversation.
2. `park_current`'s first guard is `!live_ok` → `Saved::skipped` → the swap reports `saved 0 B`.
3. `serve_swap::invalidate_unparked(out)` → `resumable = false`.
4. Next pick: `step_gate(prefill, false, false)` → `Gate::re_read` → `reset_request_to_token0`.
5. **The licence:** `outgoing_for(prefill)` was **always** `re_readable`, which means "losing the
   session is survivable, the caller owes it a reset to token 0" — so the swap guard allowed the
   hand-over without a save. Re-reading is survivable **once**; on every swap it is a spin.

## The fix

* `serve_driver.hpp`: `outgoing_for(Phase, int64_t read_tokens)`. A prefill/prefill_end slot with
  `read_tokens > 0` is now **`must_park`** — it has progress the cache can save. Only a slot that has
  read *nothing* stays `re_readable` (re-reading that is genuinely free). The one-argument overload is
  kept and means "cursor 0", so an old call site cannot silently claim progress.
* `generate.cpp`: `publish_prefill_branch()` puts `ids[0, at)` into `live`/`live_ok` before a
  hand-over (guarded on `R.cancelled` — a read stopped between chunks has nothing clean to continue
  from, the same reason `finish_request` guards `live.swap(consumed)`); `publish_working_branch()`
  calls both publishes. Both swap sites publish **and** re-classify from the *current* cursor, since
  `working_set` was last armed before the slot's previous step.
* `restore_published_branch()` gained its prefill half: after a restore, `R.at` is re-derived from
  `live.size()` and `seg_i` re-planned, so the read resumes where it stopped.
* **Safety net:** `kMaxRereads = 4` + `reread_allowed()` + `reread_limit_line()`. A fifth reset is
  reported as a named ERR instead of restarting forever. This is the net under the fix, not the fix.

## Separate bug fixed on the same path

`arm_prompt_state` re-armed `cur` but **not `req_imgs`**, and `checkpoint_at` stamps
`c.imgs = imgs_below(req_imgs, L)`. A prefill resumed after a hand-over therefore stamped **another
conversation's picture keys** onto this branch's checkpoints — silent, and it corrupts the cache's
prefix comparison for every later request of that chat. Now `req_imgs = R.own_imgs` alongside `cur`.

## Traps

* **Both halves are needed.** Publishing without the classification still lets the driver treat the
  slot as throwaway; the classification without the publish leaves `park_current` with nothing to
  save. The end-to-end simulation in `serve_driver_test` exercises both through the real predicates.
* `R.at` is a segment end and every segment end is `<= n - 1`, so the publish stays inside the prompt.
  The `std::min` is belt-and-braces; 0.1.30's `finish_prefill` does the same cast.
* Do **not** re-derive `pp_next_check` in the restore: it travels with the request
  (`arm_prompt_state` copies it back), so the mid-prompt checkpoint schedule keeps its positions
  across a swap.
* `checks` is deliberately untouched by the publish: every checkpoint of a read is at a position
  `<= at`, so they are all prefixes of the branch being published.
* The "branch no longer in the cache" warning in `do_swap` is **decode-only** now. A prefill that
  finds nothing parked can still re-read (bounded); a decoder cannot, because its prompt tokens are
  already on the wire.
* Adding a `std::fprintf` inside a serve-scope lambda moves the count in
  `.shz_cmd/s31e2_output_proof.py` (it is textual, not reachability-based). `do_swap` already prints
  `slot N resumed from T tokens`, which for a prefill *is* the cursor — no new line needed.

## Verified

`serve_driver_test` 340 → **377** checks. ctest 18/19 (only `expert_multi_test`, no AVX-512,
pre-existing); conversation_cache 4252, conversation_memory 23, conv_cache, serve_swap 169,
serve_proto 140, slot 176, prefill_loan 128 all pass; `serve/test_server.py` 65 passed / 2 skipped.
**Serial driver byte-identical (3652 B)** and the output proof PASSes (109 serial-reachable calls,
unchanged) — none of this touches `--serve-slots 0/1`.

Anti-vacuity, four mutations (each reverted after running):
* `outgoing_for` back to phase-only (ignore the cursor) → **8** failures in `serve_driver_test`,
  including the end-to-end simulation ("both prefills finish", "never sent back to token 0");
* `reread_allowed` neutered to always-true → **3** failures (the simulation runs to its pass cap
  without `hit_bound`, so the bound is what makes the spin observable);
* `default: re_readable` in `outgoing_for` → **11** failures (the phase/cost invariant loop);
* `save_is_mandatory` neutered to always-false → **13** failures in `serve_swap_test` and **5** in
  `serve_driver_test` — the simulation's "both prefills finish" is among them, which is what proves
  the simulation drives off the real predicate rather than a parameter.

## Owner action

Restart and run two clients that both send a long prompt. Expect, per swap away from a prefill:
`park: slot N parked T tokens / X MiB` and on the way back in `slot N resumed from T tokens` —
instead of `saved 0 B` + `reset to token 0`. If a prefill genuinely cannot be parked (budget too
small), it now ends after 4 resets with a named ERR rather than spinning forever.
