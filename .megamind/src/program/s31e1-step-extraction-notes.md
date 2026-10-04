# S3.1e-1 — extract the serve request body into resumable step functions

**Pure refactor, zero behaviour change, no scheduler.** Landed and verified.
Only `src/program/generate.cpp` changed.

## The step API — what S3.1e-2 calls

All in the serve scope of `src/program/generate.cpp`, all lambdas (the captures
are serve-scope state: `ver`, `mtp`, `sp`, `conversations`, `slots_reg`, `swap_to`, …).

```cpp
enum class Prep { ok, rejected, fatal };   // rejected: ERR printed -> next line
                                           // fatal:    ERR printed -> return 1
enum class Step { progressed, finished, cancelled, needs_swap, error, fatal_exit };
//   progressed  more of this step to do      finished  phase complete
//   cancelled   STOPped, fall through to the phase tail (NOT an error)
//   needs_swap  RESERVED for S3.1e-2 - nothing returns it yet
//   error       the ERR line is already printed, the caller must `return 1`
//   fatal_exit  #224 CUDA fault, already flushed, the caller must `_Exit(1)`

struct ReqCtx { /* the per-request locals: parsed line, resume point, sampling
                   dispatch, the segment cursor, the whole decode-loop state */ };

Prep prep_request(ReqCtx& R, const std::string& line);   // phases 1+2
void plan_prompt_segments(ReqCtx& R);                    // segment ends + cursor
Step run_prefill_step(ReqCtx& R);                        // phase 3, ONE segment
Step finish_prefill(ReqCtx& R);                          // phase 3 tail + phase 4 prologue
Step run_decode_step(ReqCtx& R);                         // phase 4, ONE verify window
Step finish_request(ReqCtx& R);                          // phase 5: DONE + state + metrics
```

## The serial call order (generate.cpp:6281)

```cpp
while (next_line(line)) {
    QUIT / BusyScope / stop_req / request_index / err.clear()
    ReqCtx R;                            // per iteration: the old locals' lifetime
    SlotGuard  slot_guard{&R, ...};      // destruct at END of the iteration, as before
    MropeScope mrope_scope;              // armed from inside prep_request via R
    Prep prep = prep_request(R, line);
    if (prep == Prep::rejected) continue;
    if (prep == Prep::fatal) return 1;
    plan_prompt_segments(R);
    Step st; while ((st = run_prefill_step(R)) == Step::progressed) {}
    if (st == Step::error) return 1;
    if (st == Step::fatal_exit) { fflush; fflush; _Exit(1); }
    if (finish_prefill(R) == Step::error) return 1;
    Step dst; while ((dst = run_decode_step(R)) == Step::progressed) {}
    if (dst == Step::error) return 1;
    if (finish_request(R) == Step::error) return 1;
}
```

Also hoisted to serve scope (they only ever captured serve-scope state, so
hoisting is a no-op): `windows_ok`, `read_windows`, `lend`, `dec_timing`,
`DecSnap`, `dec_snap`.

## What stayed inline, and why

* `QUIT`, `BusyScope` — the watchdog scope spans the whole iteration.
* The **instances** of `SlotGuard` and `MropeScope`. Their destructors must run at
  the end of the iteration, not at the end of `prep_request`. Only their struct
  definitions moved, so a body-local instance can be armed through `ReqCtx`
  (`R.slot_open`, `R.mrope_touched`). Destruction order is unchanged:
  mrope → slot → R → busy.
* The `while` loops that drive the steps — that *is* the serial path.

## For S3.1e-2

* `needs_swap` is the reserved channel for "this step wants the session handed to
  another slot". Nothing returns it today; add it where a step must pre-empt.
* `run_decode_step` is the natural pre-emption point (one window ≈ 16 ms) and it
  allocates nothing per token: `drafts/window/outv/dprob/sbuf/consumed` live in
  `ReqCtx`, sized once by `finish_prefill`.
* `run_prefill_step` advances one prompt segment; `R.at` is the cursor and
  `R.seg`/`R.seg_i` are the four segment ends 0.1.30 iterated.
* `ReqCtx` is per-request state only. The session, the cache, the registry and the
  prompt loan stay exactly where they were.

## Verification performed

* `cd build && ninja` clean after every phase and at the end (the `no .sframe`
  ld lines are pre-existing binutils noise).
* `/tmp/s3build`: `slot_test` 176 · `serve_proto_test` 140 · `serve_swap_test` 105 ·
  `prefill_loan_test` 54 · `conversation_cache_test` 4232 · `conv_cache_test` PASS ·
  `conversation_memory_test` 23 — all pass.
* `python3 -m pytest serve/test_server.py -q` → **64 passed, 2 skipped**.
* **Output-call proof** (`.shz_cmd/s31e1_output.py`): the serve block has **96
  `std::printf`/`std::fprintf` calls before and 96 after, in the same execution
  order**. The only two differing lines are statements whose trailing `continue`
  became `return Prep::rejected` — the printf on that line is byte-identical.
* **Per-phase statement diffs** (`.shz_cmd/s31e1_cmp2.py`): every changed line is
  one of (a) a `T& t = R.f;` alias, (b) a local declaration rewritten as
  `R.f = ...`, (c) an exit rewrite (`continue`→`return Prep::rejected`,
  `return 1`→`return Prep::fatal`/`Step::error`, `break`→`return Step::finished`/
  `cancelled`, `_Exit`→`return Step::fatal_exit`), or (d) the guard relocation.
* Phase A re-checked as a byte-exact move: 125 lines, uniform 4-space dedent.

## Trap found and fixed (do not repeat in S3.1e-2)

Splitting the segment `for` out of the body silently dropped its initialiser
`int64_t at = read_from;`. `-Wall` caught it as an unused variable; `R.at =
read_from;` is now set in `plan_prompt_segments`. **Any loop that initialises its
own cursor must carry that initialiser into the plan step, not the step loop.**

Second trap: the step lambdas were first inserted *inside* `while (next_line(...))`
because the insertion anchor was searched from the wrong end of the file. It still
compiled and still ran — but it rebuilt six `std::function`-free lambdas per
request and made the output-order proof nonsense. Check that a hoisted step
lambda really lands in the enclosing scope.
