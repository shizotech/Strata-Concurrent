# S4.3.3 — the prefill↔decode control protocol (LANDED, CPU-tested)

Files (new only, nothing existing touched, **no CMakeLists.txt edit**):
`include/strata/program/prefill_svc.hpp`, `src/program/prefill_svc_test.cpp`.
The parent's `EXISTS`-guarded block at `CMakeLists.txt:937-944` activated unchanged
(`add_executable(prefill_svc_test …)` + `include`, **links nothing** — the header is pure and
`serve_driver.hpp` is header-only).

`prefill_svc_test`: **2640 checks OK**, ~0.05 s, deterministic, no I/O at all (no file, no socket, no
`/dev/shm`, no GPU, no model, no engine). Built and run for real in `/tmp/s433build`
(`cmake -G Ninja -S . -B /tmp/s433build -DCMAKE_BUILD_TYPE=Release -DSTRATA_BUILD_TESTS=ON
-DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 && ninja -C /tmp/s433build prefill_svc_test`).
**`STRATA_ENABLE_CUDA=ON` is mandatory** or `strata_engine` — and with it the whole S4.3 block — is not
created (`CMakeLists.txt:35`).

Namespace `strata::program::prefill_svc`. It includes **nothing** from `strata/program/` and nothing
from `strata/core/` (D3 of §1.6: the client wire must not be able to drift). The test includes
`serve_driver.hpp` and asserts the shared vocabulary numerically.

## The public API (what S4.3.5 and S4.3.6 call)

### constants
```
kReadyToken = "PREFILL-READY-V1"   kProtoVersion = 1     kMaxLineBytes = 4096
kBackoffStartMs = 50               kBackoffCapMs = 2000  kDefaultPrefillRetries = 3
kDefaultPrefillQueue = 32
kWaitReasonPrefill = "prefill-busy"                     // §6.1 — S4.3.6 MUST use this string
kWaitTextPrefill   = "the prefill instance has no free handoff slot for a prompt this long"
kWaitPrefillValue  = 8                                  // Wait::prefill's slot in serve_driver.hpp
```

### lines: one formatter + one parser per verb, so they cannot drift
```cpp
enum class LineKind { unknown, hello, ready, ping, pong, req, cancel, ack, bye,
                      queued, claim, seg, done, err, info, take };
enum class Side { either, decode_to_prefill, prefill_to_decode };
struct Line { /* every field of every verb; `bad` non-empty ⇒ close the connection */ };

Line parse_line(const std::string& line);            // never throws, never I/O
bool line_is_well_formed(const std::string&, std::string& why);   // framing: ≤4096 B, ASCII, no CR/NUL
const char* line_name(LineKind);  Side line_side(LineKind);  bool line_from_peer(LineKind, Side);

std::string hello_line(proto_ver, client_id, pack_hash, geom_hash, max_ctx,
                       later_stage_layer_lo /*vector<u64>*/, cvec, vision);
std::string ready_line(const ReadyInfo&);            // {proto_ver, slots[], tiers[], max_tokens, pack, geom}
std::string ping_line(cookie) / pong_line(cookie);
std::string req_line(const ReqKeys&);                // {id, tokens, tier, ids_name, geni, cvec, resume}
std::string cancel_line(id) / ack_line(id, nonce) / bye_line(reason);
std::string queued_line(id, ahead) / claim_line(id, slot, nonce);
std::string seg_line(id, slot, seq, tokens_done, tokens_total, ms);
std::string done_line(id, slot, nonce, bytes, tokens, ms);
std::string err_line(id, ErrCode, message);           // empty message ⇒ the code's own sentence
std::string info_line(const std::vector<std::pair<std::string,std::string>>&);
std::string req_ids_path(dir, client_id, nonce, out, allow_absolute = true);  // req-<client>-<nonce>.ids
bool        ids_name_is_safe(name);                   // the WIRE-side rule: no absolute, no ".."
```
`slots=`/`tiers=` are index-aligned and **ascending** (S4.3.2 stores tiers ascending); `max_tokens`
must equal the largest tier or the handshake refuses.

### the error table (§4.4) — total, 14 codes, one place
```cpp
enum class ErrCode { none, full, queue, toobig, ids, novision, resume, geom, pack, cancel,
                     read, slotlost, shutting, badpayload, endpoint };
struct ErrClass { ErrCode code; int hold; bool drop_connection, retryable, decode_emitted;
                  const char* text; };
const std::vector<ErrClass>& err_table();   ErrClass err_class(ErrCode);
int  hold_for(ErrCode);   bool err_is_permanent / err_is_temporary / err_drops_connection /
    err_is_retryable / err_is_decode_emitted / err_reaches_the_client(ErrCode);
const char* err_name / err_text(ErrCode);   bool err_from_code(const std::string&, ErrCode&);
int prefill_svc_error_count();              // the table's size is asserted against this
namespace hold { run_now=0, wait=1, error=2; }   // mirrors serve_driver::Hold; asserted in the test
```
**7 permanent** (`toobig ids novision resume geom pack badpayload`) reach the client; **7 waits**
(`full queue read slotlost shutting endpoint cancel`) never do (D5). `geom`/`pack` also drop the
connection. `badpayload`/`endpoint` are decode-emitted. Only `read`/`slotlost` consume a retry.

### the handshake (§4.2)
```cpp
bool handshake_decision(const Line& hello, const Line& ready, std::string& why);  // decode side
bool hello_is_acceptable(const Line& hello, std::string& why);                    // prefill side
bool hello_requirements_met(const Line& hello, int cvec, int vision, std::string& why);
```
A version, pack or geometry mismatch is refused **at the handshake**, before any other line is read,
and the refusal prints both values. An old↔new pair can never mis-parse each other.

### the tier rule (§3.8) and admission pricing (§6.2)
```cpp
bool tier_for_tokens(tokens, ascending_tiers, uint64_t& tier);   // false ⇒ ERR toobig
bool prefill_prompt_too_big(tokens, max_prefill_tokens, arena_max_tokens, std::string& why);
bool tier_may_use(uint32_t job_tier, uint32_t slot_group);       // group >= tier, always
```
`prefill_prompt_too_big` runs at **admission**, before a `REQ` is ever sent.

### the decode-side machine (§4.5) — 13 states × 18 events, total by construction
```cpp
enum class HState { idle, queued, init, sent, held, accepted, prefilling, seg_ready,
                    handed_over, mounting, done, failed, cancelled };
enum class Event  { admit, req_sent, queued_line, claim_line, seg_line, done_line,
                    err_temporary, err_permanent, err_cancel, cancel_line, ack_sent, copy_done,
                    mount_ok, mount_fail, collect, retry, note_cancel, local_hit, last_event };
enum class Act    { none, write_req, write_cancel, write_ack, copy_out, mount, hold, err_client,
                    done_cancel, client_gone, finish, protocol_error, illegal };
kStateCount = 13;  kEventCount = 18;
const Transition* handoff_table(int& count);   // exactly 13*18 = 234 rows
Transition lookup(HState, Event);              // never null; a missing row says so by name
Event err_event_for(ErrCode);                  // cancel / temporary / permanent
bool state_is_live(HState);  bool state_holds_slot(HState);   // only H_SEG_READY holds a slot

class HandoffState {                          // one per request, on the handoff thread
  explicit HandoffState(int64_t retries = kDefaultPrefillRetries);
  bool admit_now(id);  bool begin(id, tokens, tier, why);  bool local_hit(id);
  StepResult on_line(id, const Line&);        // nonce + id + side guards run BEFORE the table
  StepResult note(id, Event);  StepResult note_cancel(id);
  StepResult mount_result(id, ok, after_first_write);       // the ENGINE thread's entry point
  std::string ack_line() const;               // "ACK <id> <nonce>" or empty
  bool may_release(int64_t slot, uint64_t nonce) const;     // and may_read(): the same rule
  bool retry_allowed() const;  bool reset_for_retry(std::string& why);
  bool may_send_req(int64_t now_ms) const;    // one outstanding REQ + the 50 ms→2 s backoff
  void note_now_ms(int64_t);                  // the machine has no clock of its own
  HState state(); uint64_t nonce(); int64_t slot(); ErrCode verdict(); bool poisoned();
  CancelOutcome cancel_outcome();             // client_done_cancel | client_gone
};
```
`Act` is what the caller switches on — one enum, so a call site cannot miss a case.

### the prefill-side job machine (§5.2, §4.6) — 9 states × 9 events, total
```cpp
enum class PJob   { queued, claimed, reading, writing, published, released, cancelling, cancelled,
                    failed };
enum class PEvent { claim_ok, read_start, seg_sent, publish_ok, ack, cancel, read_fail,
                    publish_lost, abort_done, last_event };
enum class PAct   { none, send_queued, send_claim, send_seg, send_done, err_read, err_slotlost,
                    err_cancel, release_slot, abort_claim, abort_err_cancel, abort_err_read, illegal };
kJobStateCount = 9;  kJobEventCount = 9;   // 81 rows
const JobTransition* prefill_job_table(int& count);   JobTransition job_lookup(PJob, PEvent);
ErrCode job_action_code(PAct);   bool cancel_after_done_is_normal(PJob);   // PUBLISHED ⇒ true
```
`S4.3.5`'s loop body is `job_lookup(state, event)` + the action it names — the fake prefill instance in
the test is written that way and is ~120 lines, which is the evidence that the table is enough.

### the log lines (§4.2, §4.3, §4.7)
`prefill_serving_line`, `endpoint_ready_line`, `endpoint_down_line`, `not_prefill_line`,
`hash_refusal_line`, `retry_limit_line`, `slot_released_line`, `activity_line`,
`hold_line_prefill`, `client_err_text`, `default_info_keys`, `info_value`, `info_shows_a_free_slot`
(the §6.2 wake point: an `INFO` whose `slots_free` is non-zero — note it is `"4/6"`, the spelling
`HandoffArena::free_slots()` produces).

## Deviations from docs/STAGE4-SPLIT-ROLES.md §4 (the header's banner numbers them **DEV-1 … DEV-14**;
the manager's decision D1–D5 in `.megamind/stage4-plan.md` are a DIFFERENT D-numbering, so the header
uses `DEV-n` to keep the two from colliding. ONE truth: DEV-14 is the one that bit during
implementation.)

1. **DEV-1: 13 states, not 8.** §4.5's diagram has six arrows into `H_REQ_SENT` and needs a per-state answer
   to `CANCEL`. Split into `H_INIT` / `H_SENT` / `H_HOLD`, and added `H_FAILED` + `H_CANCELLED`. No
   doc state was renamed or removed.
2. **`MOUNTING` is a state; "session poisoned" is not.** A restore that fails after its first write
   sets `poisoned()` and the caller must exit 1 — encoding it as a state would invite a recovery that
   must never happen.
3. **`H_QUEUED` has no handoff events** (the handoff thread must not touch it); `H_DONE` is terminal.
4. **`ERR cancel` ends the handoff; it does not format the client's `DONE … cancel`.** That line is
   `serve_proto`'s. The machine reports `cancel_outcome()` and S4.3.6 writes the client line.
5. **`Hold::run_now` is never produced by an error code** (every code is a refusal). The table is
   still total; `run_now` is reachable through `Act`.
6. **`prefill-busy` / the wait text live in this header** until S4.3.6 adds `Wait::prefill` to
   `serve_driver.hpp` (which S4.3.3 may not edit). Use `kWaitReasonPrefill` / `kWaitTextPrefill`
   verbatim; `serve/test_server.py` pins them.
7. **`ids=NAME` is validated, not just typed**: no `..` component, and a wire name may not be
   absolute. The only place the grammar is tighter than the doc, and the reason is that `ids=` names a
   file the prefill instance opens.
8. **An unknown VERB is `LineKind::unknown`, not a connection error.** Only a *known* verb with bad
   fields sets `Line::bad`. Forward compatibility, and the doc's own "a reader that ignores SEG is
   still correct".
9. **`HELLO n_stages` = the number of LATER stages**, and must equal `split=`'s length.
10. **`PING`/`PONG`'s cookie is `[A-Za-z0-9_-]+`** so a liveness probe cannot inject a line.
11. **`ERR` is three events** (`err_temporary` / `err_permanent` / `err_cancel`) because §4.4's table
    says an ERR means three different things and the difference is the code's class, not the verb.
12. **`PREFILL-READY-V1`'s fields after `proto_ver` are `k=v`**, exactly as §4.2 spells them
    (`slots=… max_tokens=… tiers=… pack_hash=… geom_hash=…`), not positional.
13. **The retry bound counts REQ lines**: `--prefill-retries 3` ⇒ at most 4 REQ lines per request.
14. **`SEG` carries no nonce** (§4.3's grammar has none), so it is gated by *state* (it is progress
    only, never a copy-out trigger) rather than by a nonce field it does not have. §4.5 rule 5 claims
    SEG carries the nonce, but §4.3's `SEG <id> <slot> <seq> <tokens_done> <tokens_total <ms>` does
    not, and adding a field would be a wire change. The payload a SEG precedes is still gated by the
    `DONE` nonce and then by the slot header's nonce inside `HandoffArena::read_slot`, so a stale SEG
    can delay nothing, mount nothing and release nothing.


## Traps found while implementing (do not reintroduce)

* **`SEG` has no nonce field.** The first draft nonce-checked `SEG` as well as `DONE` and rejected
  every legal `SEG` the fake prefill instance emitted. The grammar, not the intuition, is the spec.
* **`ERR`'s message is free text.** Re-joining it from tokens destroys spaces; keep the rest of the
  line (`detail::past_token`) so a log line quotes what the peer said. Same for `BYE <reason>` — and
  `past_token(line, 0)` is "past token 0", not "past token 1"; an off-by-one there ate the first word
  of every reason.
* **`strtod` accepts `"1e999"` (→ inf) and `"nan"`.** `!(v >= 0.0)` plus a magnitude ceiling is what
  refuses them; `v < 0.0` alone lets NaN through, and a formatter that emitted `inf` would produce a
  line the peer cannot parse.
* **A 4096-character `NAME` is a perfectly good token.** The parser will happily accept an over-long
  line; only the **framing** check (`line_is_well_formed`, run on the bytes read before the verb is
  looked at) enforces §4.2's limit. Do not rely on the parser for it.
* **A line cut *inside* its last token is legal grammar** (a shorter number). Asserting that the
  parser refuses it would be asserting it can see a byte boundary only the framer knows about. What
  must hold is that a truncated line re-formats to a **prefix** of the original — no field shifts.
* **`req_ids_path` builds the LOCAL path** from `--prefill-handoff-dir`, which is normally absolute;
  the *wire* rule (`ids_name_is_safe`) is the one that refuses an absolute name. Conflating them
  made every local path builder fail.
* **`HandoffState::begin()` must be legal from `H_QUEUED`** — that is the normal admission path — and
  `startable()` initially forgot it.
* **An `illegal` transition must never move the state.** Enforced in the table generator
  (`.shz_cmd/s433_gen_tables.py`), not by hand: a typo in one override would otherwise make a refusal
  look like a transition.
* **The tables are GENERATED as complete cross-products** (`.shz_cmd/s433_gen_tables.py` →
  `.shz_cmd/handoff_table.inc` + `job_table.inc`, pasted into the header). Hand-writing a
  13×18 table is how "total" becomes a claim instead of a fact. If you edit a cell, edit the
  generator and re-paste.
* **`check(!api(..., err), ("label: " + err).c_str())` is a bug** (unspecified argument order). The
  test uses `check_err` / `check_reason_contains` instead, and pins the reason text — a test that
  accepts "some refusal" is not a test of that refusal.

## What was verified (CPU-only; no engine, no model, no GPU, no sockets, no /dev/shm)

`prefill_svc_test`: **2640 checks OK.**
* **Grammar:** 25 lines format → parse → re-format **byte-identical**; every field read back; the
  version token is `PREFILL-READY-V1` and is *not* the client `READY`; the verb vocabularies are
  disjoint except the three documented name overlaps (`DONE`/`ERR`/`INFO`).
* **Malformed:** ~45 refused lines, each pinned to its reason; unknown verbs ignored; unknown trailing
  tokens ignored; a `k=v` value stops at the space (no smuggling); truncation swept at every byte of
  a `DONE` (three cases: verb destroyed / missing field / shorter number, with the prefix invariant);
  framing refuses empty, `\n`, `\r`, NUL, control bytes, DEL, non-ASCII, and >4096 bytes — and the
  test proves the parser alone would *accept* the over-long line, so the framer is load-bearing.
* **The error table:** 14 codes, one row each, every one `wait` or `error`; §4.4's permanent and
  temporary columns asserted row by row; the two classes partition the table (7/7); `geom`/`pack`
  drop the connection; `badpayload`/`endpoint` are decode-emitted; only `read`/`slotlost` retry; the
  code names round-trip; an unclassified code is an error, never a silent infinite hold.
* **No drift with `serve_driver.hpp`:** `Hold::run_now/wait/error` asserted equal to `hold::0/1/2`,
  `hold_name` compared with the real one, `Wait::prefill` pinned to `engine_busy + 1`, and the
  `prefill-busy` strings pinned.
* **The handshake:** version (both directions), pack, geometry, self-contradictory `max_tokens`, a
  0-slot tier, a non-prefill endpoint, `client_id 0`, and the cvec/vision requirements — each refused
  with both values on the line.
* **The state machine:** the table's size asserted to be exactly `13 × 18`; every cell has a reason
  and none is the MISSING sentinel; no pair duplicated; every state reachable from IDLE; every event
  legal somewhere (and `CANCEL` legal nowhere, deliberately); an illegal cell never moves the state;
  exactly one state may start a copy-out (that *is* CLAIM-before-DONE); SEG-before-CLAIM refused in
  every pre-CLAIM state; cancellation classified over all 13 states with exactly one refusal
  (MOUNTING) and the reason for it.
* **The nonces:** a second CLAIM nonce, a stale DONE nonce, a line for another id, and a line from
  the wrong side are all refused without moving the state; `may_release`/`may_read` refuse a stale
  nonce and a right-nonce-wrong-slot; the fake arena refuses the same at the slot level.
* **The retry bound:** the default fires at `retries+1` REQ lines and terminates before the pass cap;
  **neutering it makes the same simulation run to the pass cap without finishing** (the S3.8
  anti-vacuity shape), and the end-to-end version against the fake prefill instance shows both arms;
  the backoff doubles 50→100→200→400 and caps at 2 s; `may_send_req` gates on both the outstanding
  REQ and the backoff.
* **The twelve §4.7 failure modes**, each a data-driven scenario against a by-value fake arena +
  prefill loop + decode client: prefill dies (both in-flight handoffs hold, zero client ERRs); decode
  dies mid-handoff (READY lease reclaim, `flags |= abandoned`, immediate reclaim when the reader pid
  is registered, and **no** reclaim before the bound); decode dies after ACK (clean); slot leak
  visible in `slots_free=2/4` then recovered to `4/4`; partial write (a CLAIMED slot is never
  readable, reclaimed, rewritten); partial publish (CLAIMED-or-READY only, metadata zero until the
  publish, a stale/wrong-nonce publish refused); corrupt payload (hash mismatch stops the copy-out,
  the machine stays in H_SEG_READY so the slot still comes back, wrong client/nonce refused before
  anything is copied); restore-fails-after-the-first-write (poisoned, no retry, no local fallback);
  arena full (`ERR full` → hold, `ERR queue` → hold, the §3.8 tier rules including "a big job never
  steals a small slot" and "a small job cascades up"); tmpfs full (the refusal names both numbers;
  the decode side holds and starts anyway); bad line (framing or parser, then hold, never a client
  ERR); infinite retry (above).
* **End to end:** HELLO → PREFILL-READY-V1 → PING/PONG → REQ → QUEUED → CLAIM → SEG → DONE →
  copy-out → ACK → mount → collect, with the slot back to `6/6` **before** the mount (§4.5 rule 1),
  the engine thread stepping independently, and no `ACK-ED` echo. Two clients share one arena with
  per-client slots and a request id that is *not* globally unique (§2.4). A local cache hit skips the
  handoff entirely (zero REQ lines, zero nonces).
* **No-drift check:** `prefill_svc.hpp` includes only the 12 standard headers listed above. Existing
  CPU tests re-run in the same tree, all green: `serve_proto_test` 140, `serve_driver_test` 523,
  `slot_test` 176, `serve_swap_test` 173, `prefill_loan_test` 128, `saved_conv_wire_test` 483,
  `handoff_arena_test` 268. `git status` shows exactly two new untracked source files for this step.

**Mutation-tested: 74 mutations, 74 bite** (`.shz_cmd/s433_mutations.py` — it rebuilds the suite
against each mutated header and counts failures). Covered: every framing byte class, the length
limit, hex/NAME/ID/list typing, `ids=` required, the HELLO and READY self-consistency checks, the ERR
code vocabulary, free-text preservation, TAKE reserved, unknown-verb tolerance, every error-class
flip, both wait strings, every handshake check, the tier rules, DONE/SEG-without-CLAIM, a second REQ,
temporary↔permanent swaps, ACK before DONE, a second ACK, a copy-out with nothing published, a
cancellable mount, a retry from H_CANCELLED and from H_SEG_READY, a cancelled copy-out, a peer line
driving a queued request, a second DONE, both nonce rules, the id and side guards, the retry bound,
the backoff cap and growth, both `may_send_req` clauses, both `may_release` clauses, the poisoned
flag, six job-table cells (ACK of an unpublished slot, a second ACK, CANCEL-after-DONE, a publish
racing a cancel, the wrong ERR code, the lost slotlost answer), the ids path/name rules, and three
log-line wordings.

## For S4.3.5 (the prefill instance's socket loop)

* Read a line → `line_is_well_formed` (framing: close if false) → `parse_line` (close if `bad`) →
  `line_from_peer(kind, Side::prefill_to_decode)` (close if false).
* `HELLO`: `hello_is_acceptable` then `hello_requirements_met`; on false **close with no line**
  (§4.2). Then `ready_line(...)`, and `arena.note_client_pid(client_id, SO_PEERCRED pid)`
  (S4.3.2 deviation 2 — required for the immediate READY-lease reclaim).
* `REQ`: check `ids_name_is_safe(l.ids_name)` (refuse with `ERR ids`), `l.resume == 0`
  (`ERR resume`), `l.tokens <= arena.max_tokens()` (`ERR toobig`), the queue cap (`ERR queue`), then
  push the job and answer `queued_line(id, ahead)`.
* The engine thread drives the head job through `job_lookup(PJob, PEvent)`; the actions name the
  lines and the arena calls. `claim()` failing is `ERR full` — **temporary**, never a close.
* `CANCEL`: `job_lookup(state, PEvent::cancel)`. From `PUBLISHED` it is a **normal release**
  (`cancel_after_done_is_normal`), not an error.
* `ACK`: `arena.release(index, nonce)` — and only from `PUBLISHED`. There is no `RELEASE` alias and
  no `ACK-ED` echo; record `slot_released_line(...)` on stderr instead.
* `BYE` ends the connection; `PING` is answered `PONG <cookie>`.

## For S4.3.6 (the decode instance's handoff client thread)

* One `HandoffState(retries)` per request, owned by the handoff thread. `admit_now(id)` at
  admission, then the local `ConversationCache::best()` lookup **first**: a hit is `local_hit(id)`
  and no `REQ` is ever written (§1.5).
* `note(id, Event::req_sent)` → `Act::write_req` → send `req_line(...)`. `may_send_req(now_ms)` is
  the one-outstanding-REQ + backoff gate; the machine has no clock, so pass `note_now_ms`.
* Every peer line goes through `on_line(id, line)`. Switch on `StepResult::action`:
  `copy_out` (map → read → decode → checks 1-2, **never on the engine thread**), `write_ack`
  (send `st.ack_line()`), `hold` (S4.2's `WaitQueue` with `Wait::prefill` / `prefill-busy`),
  `err_client` (only for `err_reaches_the_client(verdict)`), `protocol_error` (close),
  `illegal` (log and continue).
* Before `HandoffArena::read_slot` / `release`, assert `st.may_read(slot, nonce)` /
  `st.may_release(slot, nonce)`. That is the stale-nonce rule at the protocol level; the arena
  enforces it again at the slot level.
* `mount_result(id, ok, after_first_write)` is the **engine thread's** call. `poisoned()` ⇒ return 1.
* `cancel_outcome()` decides `DONE … cancel` vs an ordinary `STOP`; this header never formats a
  client line.
* Add `Wait::prefill` to `serve_driver.hpp` at value `engine_busy + 1` with exactly
  `kWaitReasonPrefill` / `kWaitTextPrefill`, and do not add `prefill-busy` to
  `serve/server.py`'s `PERMANENT_REFUSALS` (§6.1).
