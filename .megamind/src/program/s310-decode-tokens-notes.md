# S3.10 — `--decode-tokens N`: how many tokens a slot writes per turn

Owner's ask: *"a request generates 1-3 tokens or so and then switches to the next request, which is
too low … for example 10 tokens are generated when its a requests turn (for the decode)."*

## The mechanism (not a bug)

A decode **step** was exactly ONE verify window. A window commits `a + 1` tokens (`a` = MTP drafts
accepted, 1..8, usually 1-3). Every step boundary is a hand-over candidate, so the scheduler paid or
evaluated a full save+restore (R10: 237 MB-2.25 GB) every ~2 tokens.

## What landed

```
include/strata/program/serve_driver.hpp   decode_step_tokens(), decode_turn_done(); driver_line() gained `decode_tokens`
src/program/generate.cpp                  Options::decode_tokens, --decode-tokens parse + validation + usage,
                                          run_decode_step's window body wrapped in a per-turn loop,
                                          driver_line call site
src/program/serve_driver_test.cpp         2 new tests, 377 -> 395 checks
docs/DETAILS.md                           new paragraph under "Several requests at once" + cross-ref in S3.1e-2
docs/STAGE3-CONCURRENCY.md                new "### S3.10" section + R10 mitigation updated
```

`src/core/*`, `src/prefill/*`, `src/kernels/*`, `slot.hpp`, `serve_swap.hpp`, `serve/server.py`:
**not touched.**

## The shape

`run_decode_step` keeps 0.1.30's window body verbatim and wraps it in `for (;;)`:

* loop top: `decode_step_tokens(budget, produced_n, max_new) == 0` -> `Step::finished`
  (that is 0.1.30's `produced_n >= max_new` guard, stated once);
* `turn_windows > 0 && decode_turn_done(budget, produced_at_turn_start, produced_n)` ->
  `Step::progressed` (turn over, session may be taken);
* otherwise run one window; the existing `return Step::progressed` at the bottom now falls through to
  the loop top instead of leaving the lambda.

`budget = 0` (default) or `1` -> `decode_turn_done` is true after the first window -> exactly one
iteration -> today's behaviour by construction. The serial driver's
`while ((dst = run_decode_step(R)) == Step::progressed) {}` is unaffected.

## Decisions worth not re-litigating

1. **Token budget, not window count.** The owner's unit is tokens, and a window's yield varies with
   acceptance, so a window count would not be a fixed amount of work.
2. **Overshoot is allowed.** A window is the smallest unit of work that exists; `--decode-tokens 10`
   with 8-token acceptance gives 16. A floor, not a cap.
3. **The first window of a turn always runs.** A turn that runs nothing is a scheduler stall and would
   freeze the watchdog heartbeat.
4. **EOS / `--max-new` / `STOP` still end the request from inside the window**, so a long turn delays a
   cancel by at most one window (~24 ms).
5. **The pre-emption point is unchanged** — still between windows — so `swap_to` never interrupts a
   window and none of S3.1d/S3.6/S3.8/S3.9's parking logic had to move.
6. **Validation:** negative -> rc 2; `> --max_context` with `--serve-slots >= 2` -> rc 2 (a slot would
   hold the session for its whole answer, i.e. the serial engine in a slot costume).
7. **`--starve-ms` and `--decode-tokens` are the two ends of one trade**: the second sets how much work
   a slot does before it *can* be pre-empted, the first how long another slot waits before the
   pre-emption is *forced*. Raising only `--starve-ms` just delays the same 2-token hand-over.

## Verified

* `cd build && ninja` -> clean (the `.sframe` ld lines are pre-existing binutils noise).
* `/tmp/a2build` (tests ON) `ninja` -> clean; `serve_driver_test` **395 checks OK**,
  `serve_swap_test` 173, `slot_test` 176, `serve_proto_test` 140, `conv_cache_test` PASS,
  `prefill_loan_test` 128.
* `./build/strata --help` prints the flag; `--decode-tokens -5` -> rc 2 message;
  `--decode-tokens 5000 --max-context 128` -> rc 2 message.
* NOT run against a live model (the owner's standing "do not start the engine" constraint).

**Loop-shape simulation** (`.shz_cmd/s310_turn_shape.cpp`, links the real header, no GPU):

```
g++ -std=c++20 -O1 -Iinclude -o /tmp/t .shz_cmd/s310_turn_shape.cpp && /tmp/t
budget default (0)  turns=100 windows=100 tokens/turn=2.0
budget 1            turns=100 windows=100 tokens/turn=2.0
budget 10           turns= 19 windows=100 tokens/turn=10.5
budget 32           turns=  7 windows=100 tokens/turn=28.6
zero-acceptance: turns=5 windows=50 (terminates)
```

Same 100 windows of real work either way; the number of hand-over boundaries drops 5x at N=10. That is
the whole point of the flag, and it is the shape the owner asked for.

## Owner action

Add to the engine args in `strata-<model>.json`: `--decode-tokens 10`. The driver start-up line now
says `decode 10 tokens/turn` (vs `decode 1 window/turn`), and `swaps=` / `swap_ms=` in the activity
line should drop. That is OQ3/OQ4 measured for the first time.
