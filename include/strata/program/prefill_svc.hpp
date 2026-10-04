// include/strata/program/prefill_svc.hpp - S4.3.3: the PREFILL <-> DECODE CONTROL PROTOCOL.
//
// This is `docs/STAGE4-SPLIT-ROLES.md` §4 as code: the line grammar in both directions, the
// `PREFILL-READY-V1` handshake, the per-request handoff state machine (`H_*`), the nonce rules, the
// CLAIM-before-DONE rule, cancellation in every state, the retry bound, and the error-code table that
// maps every code onto `serve_driver::Hold` (wait vs error).
//
// WHY THIS IS A SEPARATE HEADER FROM `serve_proto.hpp` (manager decision D3, §1.6 rule 2).  The
// client-facing wire must not be able to drift.  `serve_proto.hpp` is NOT included here and NOT
// edited by S4.3; the prefill<->decode protocol has no `GEN`, no `T`, no ten-field `DONE`, and its
// version token is a different token.  "Mirrors the existing wire" means the same STYLE - one line per
// message, `KEYWORD` first, space-separated tokens, unknown trailing tokens ignored, versioned by a
// READY token - not a shared vocabulary.  A shared parser serving two protocols is exactly the drift
// D3 forbids, so `detail::f()` below is duplicated from `serve_proto.hpp:54-67` on purpose.
//
// WHY A HEADER AND WHY IT IS PURE.  Same rule as `slot.hpp`, `serve_swap.hpp`, `serve_driver.hpp` and
// `serve_proto.hpp`: it must build and be unit-tested on a CPU with no CUDA, no model, no engine and
// no socket (`src/program/prefill_svc_test.cpp`), and the engine includes it from `generate.cpp`.
// There is NO I/O here: every formatter RETURNS the line without the '\n' and the caller writes it;
// every parser takes a `std::string`.  Slots, nonces, tokens and hashes are VALUES.  Nothing in this
// file opens a file, maps a page, touches a socket, or calls into `core::HandoffArena` - the arena
// calls belong to S4.3.5 (writer) and S4.3.6 (reader), and this header only names what they must
// agree on.
//
// INCLUDES: `<array>`, `<charconv>`, `<cstdint>`, `<cstdarg>`, `<cstdio>`, `<cstdlib>`, `<cstring>`,
// `<map>`, `<sstream>`, `<string>`, `<utility>`, `<vector>`.  Nothing from `strata/program/` (D3) and
// nothing from `strata/core/`.  The TEST includes `serve_driver.hpp` and asserts that the `hold::`
// values and the `Wait::prefill` slot defined here are the real ones, so a vocabulary drift fails the
// test rather than the live server.
//
// ================================================================================================
// THE INTENDED CALL SEQUENCE (this is the contract S4.3.5 and S4.3.6 share)
// ================================================================================================
//
//   prefill instance (S4.3.5)                       decode instance (S4.3.6)
//   -------------------------                       ------------------------
//   core::handoff_arena_path(...)                   core::handoff_arena_path(...)
//   core::handoff_geom_hash(...)  == both sides ==
//   core::handoff_size_arena(...)
//   arena.open_writer(path, sizing, pack, geom)     (open_reader happens per connection)
//   listen(path/prefill.sock)
//
//   on accept:                                      on connect:
//     line = read()                                   send hello_line(...)          // HELLO ...
//     parse_line(line) -> kind == hello               line = read()
//     hello_is_acceptable(l, why)                     parse_line(line) -> kind == ready
//        false => close, NO line (§4.2)              handshake_decision(hello, ready, why)
//     send ready_line(ready)                             false => close, log `why`
//     arena.note_client_pid(client_id, peer_pid)      (S4.3.2 deviation 2: MUST be called at HELLO)
//
//   the socket reader thread:                       the handoff client thread:
//     parse_line -> req                               st.admit_now(id) / st.begin(id, tokens, tier)
//     jobs.push(ReqJob{id, tokens, tier, keys})       st.note(id, Event::req_sent) -> write req_line()
//     if (jobs.size() > queue_cap)                    on every peer line:
//        send err_line(id, ErrCode::queue, msg)         st.on_line(id, l) -> StepResult
//     the engine thread picks the head job (§5.2):      switch (r.action) {
//        arena.claim(tier, client_id, id, claim)          case Act::copy_out:  ... see below
//        send claim_line(id, claim.index, claim.nonce)    case Act::write_ack: send st.ack_line()
//        ... Prefill::run chunks ...                      case Act::hold:      S4.2 WaitQueue(prefill)
//        send seg_line(id, index, 1, done, total, ms)   }
//        conversation_wire_encode -> arena.publish(...)
//        send done_line(id, index, nonce, bytes, tokens, ms)
//     on CANCEL: stop flag; arena.abort_claim; err_line(id, cancel)
//     on ACK: arena.release(index, nonce)
//
//   one full handoff, in order:
//     -> REQ 7 10394 1 ids=req-0000000000000007-000000009f3c0001.ids cvec=1
//     <- QUEUED 7 1
//     <- CLAIM 7 3 1073741825          (the nonce is core::HandoffClaim::nonce)
//     <- SEG 7 3 1 10394 10394 812
//     <- DONE 7 3 1073741825 1033605768 10394 1196
//     [copy-out thread] arena.read_slot(3, nonce, client_id, payload)      // S4.3.2
//                       HandoffArena::payload_hash_of(payload) == payload.payload_hash
//                       conversation_wire_decode(...)                      // S4.3.1
//     -> ACK 7 1073741825                                                    // the ONLY release verb
//     [engine thread] mount_image -> serve_swap::run -> Phase::prefill -> Phase::decode
//
// `ACK` is sent as soon as the copy-out and §2.5 checks 1-2 pass, BEFORE the mount (§4.5 rule 1):
// holding a 3 GB slot across a mount is a slot leak with extra steps.
//
// ================================================================================================
// DEVIATIONS FROM docs/STAGE4-SPLIT-ROLES.md §4  (S4.3.5 and S4.3.6 must see ONE truth - this list
// is repeated in .megamind/src/program/s433-prefill-protocol-notes.md)
// ================================================================================================
//
// DEV-1. **The state machine has 13 states, not 8.**  §4.5's diagram names 8 `H_*` states
//     (H_REQ_SENT, H_ACCEPTED, H_PREFILLING, H_SEG_READY, H_HANDED_OVER, MOUNTING, QUEUED, IDLE) but
//     the same diagram has SIX arrows into `H_REQ_SENT` (ERR full/queue, a permanent ERR, QUEUED,
//     CLAIM, DONE, SEG) and §4.6 requires a different reaction to `CANCEL` in each state.  One state
//     with six incoming edges cannot answer "is CANCEL legal here, and what must the peer be sent?"
//     without an if-chain at the call site - which is the thing a state machine exists to remove.
//     Resolution: split `H_REQ_SENT` into the three shapes the peer's answer actually distinguishes -
//     `H_INIT` (nothing sent), `H_SENT` (REQ sent, no answer yet), `H_HELD` (a temporary ERR parked
//     us; the backoff timer owns the re-ask) - and add `H_FAILED` (a permanent ERR, waiting for the
//     caller to collect the verdict) and `H_CANCELLED` (cancelled, waiting for the release to be
//     confirmed).  `QUEUED`/`IDLE` are kept as `H_QUEUED`/`H_IDLE`.  Every state name from the doc
//     still exists with the same meaning; nothing was renamed, only added to.
//
// DEV-2. **`MOUNTING` is in the table, and "session poisoned" is NOT a state.**  §4.5 marks MOUNTING
//     `[engine]`.  It is kept as a state so the table is total and the engine-thread half of the
//     handoff is testable, but the machine never drives it: `mount_result()` is the engine thread's
//     entry point.  A failed validate is `H_FAILED` with `ErrCode::badpayload`; a mount that fails
//     AFTER its first write is not a state at all - it sets `HandoffState::poisoned()` and the caller
//     must exit 1 (0.1.30's rule, `conversation_snapshot.hpp:104-108`, `serve_swap.hpp:470-471`).
//     Encoding "fatal to the session" as a state would invite a recovery that must never happen.
//
// DEV-3. **`H_QUEUED` is a pre-handoff state with no handoff events.**  §4.5 puts `QUEUED` (S4.2's
//     WaitQueue head) at the top of the diagram.  It is not a handoff state - nothing has been sent
//     and the handoff thread must not touch it - so the machine accepts it only as the source of
//     `admit()` / `local_hit()`, and every peer line in it is refused.  `H_DONE` is terminal: the
//     only legal event is `collect()`.
//
// DEV-4. **`ERR <id> cancel` ends the handoff; it does not format the client's line.**  §4.6 says a
//     request cancelled before its first token gets `DONE … cancel` on the CLIENT wire.  That line is
//     `serve_proto`'s, not this protocol's, and this header must not format client lines (D3 of §1.6).
//     The machine therefore ends in `H_CANCELLED` and reports `cancel_outcome()` -
//     `client_done_cancel` or `client_gone` - and S4.3.6 writes the client line.
//
// DEV-5. **`Hold::run_now` is not produced by the error table.**  §4.4 asks for a mapping onto
//     `run_now`/`wait`/`error`.  Every code in that table is a refusal, so none of them is
//     `run_now`; the third value would be dead weight.  The table is still TOTAL (14 codes, each
//     exactly one entry, asserted against `prefill_svc_error_count()`), and `run_now` is reachable
//     through `Act` for the non-error events.  `cancel` maps to `wait` because §4.4 marks it "n/a"
//     and the honest reading is "not a failure at all": the caller must not error the client, and the
//     machine blocks a retry from `H_CANCELLED` so it cannot loop.
//
// DEV-6. **`prefill-busy` is defined HERE, as strings, because S4.3.3 may not edit
//     `serve_driver.hpp`.**  §6.1 wants `wait_reason(Wait::prefill) == "prefill-busy"` and
//     `wait_text(Wait::prefill) == "the prefill instance has no free handoff slot for a prompt this
//     long"`.  S4.3.6 owns `serve_driver.hpp` and adds the enum value; until then the only
//     authoritative spelling of the two strings is this header: `kWaitReasonPrefill` /
//     `kWaitTextPrefill`.  S4.3.6 MUST use these exact strings, and `serve/test_server.py` pins them.
//
// DEV-7. **`REQ`'s `ids=` key is required by the parser, and the file NAME is validated.**  §4.3 says
//     `ids=NAME` is REQUIRED and `NAME` matches `[A-Za-z0-9._/-]+`.  A name with a `..` component or
//     a leading `/` would let a peer point the prefill instance at any file on the box, so
//     `ids_name_is_safe()` refuses both and `req_ids_path()` refuses to BUILD such a name.  This is
//     the only place the grammar is tighter than the doc.
//
// DEV-8. **A line whose verb is unknown is `LineKind::unknown`, not a parse error.**  §4.3's rules are
//     "unknown TRAILING tokens are ignored" and "a line whose *known* tokens do not parse closes the
//     connection".  An unknown VERB is neither: it is the forward-compatibility case (a new prefill
//     instance may add a verb), and the doc itself says a reader that ignores `SEG` entirely is still
//     correct.  So `parse_line` returns `unknown` and the caller ignores it; only a KNOWN verb with
//     bad fields sets `Line::bad`, and only `bad` closes the connection (§4.7).
//
// DEV-9. **`HELLO`'s `split=LIST` is the layer_lo of every LATER stage** (the same set
//     `core::HandoffSplit::stages` carries), and `n_stages` must equal its length.  The doc does not
//     say what `n_stages` counts; §3.10's `geom_hash` covers "the layer_lo of every later stage, in
//     engine order, plus n_stages", so `n_stages` = the later-stage count, and a HELLO that
//     contradicts itself is refused at the handshake rather than after a 3 GB transfer.
//
// DEV-10. **`PING`/`PONG`'s cookie is typed `ID` (`[A-Za-z0-9_-]+`).**  §4.3 writes `PING <cookie>`
//     without a type.  Typing it stops a liveness probe being used to inject a line into the other
//     side's parser.
//
// DEV-11. **`ERR` is three events in the machine, not one.**  §4.4 says an `ERR` means three different
//     things and the difference is the code's CLASS, not the verb: temporary -> hold, permanent ->
//     error, `cancel` -> neither.  `err_event_for()` picks the variant and the table then says what
//     each state does with each class, so §4.6's "cancellation in every state" and D5's "a temporary
//     refusal never becomes an ERR" live in one place instead of at the call site.
//
// DEV-12. **`PREFILL-READY-V1`'s fields after `proto_ver` are `k=v`, not positional.**  §4.3's legend
//     says tokens are positional and unknown trailing tokens are ignored, but §4.2's grammar writes
//     `slots=LIST max_tokens=i64 tiers=LIST pack_hash=x64 geom_hash=x64` with the names attached.
//     The named form is what the doc spells, it is what `READY slots=` already does on the client
//     wire, and it is the only form under which "unknown trailing tokens ignored" can add a field
//     without moving the others.  `proto_ver` stays positional because it is the version.
//
// DEV-13. **The retry bound counts REQ lines, not DONEs.**  §4.7's "a fourth attempt ends the request"
//     with `--prefill-retries 3` means three RE-ASKS after the first, i.e. at most `retries + 1` REQ
//     lines.  `HandoffState::retry_allowed()` states it, and the test shows that neutering it runs
//     the simulation to its pass cap without finishing (the S3.8 anti-vacuity shape).
//
// DEV-14. **`SEG` carries NO nonce, so it is gated by STATE, not by a nonce.**  §4.5 rule 5 says "the
//     `CLAIM`, `SEG`, `DONE` and `ACK` lines for one handoff all carry the same nonce", but §4.3's
//     grammar for `SEG <id> <slot> <seq> <tokens_done> <tokens_total> <ms>` has no nonce field, and
//     adding one would be a wire change.  Resolution: keep §4.3's grammar (the doc's own rule is that
//     "a reader that ignores SEG entirely is still correct"), and pin SEG by state instead - it is
//     accepted only as progress in `H_PREFILLING`/`H_SEG_READY`/`H_HANDED_OVER`/`H_MOUNTING`/
//     `H_CANCELLED` and is a protocol error anywhere before a `CLAIM`.  The payload it precedes is
//     still gated twice: by the `DONE` nonce at the line level and by the slot header's nonce inside
//     `HandoffArena::read_slot`.  A stale SEG can therefore delay nothing, mount nothing and release
//     nothing.  (The first draft of this header nonce-checked SEG and rejected every legal SEG - which
//     is exactly how this rule got written down.)
//
// ================================================================================================
// THE INVARIANTS THIS FILE EXISTS TO KEEP (each one is a test in prefill_svc_test.cpp)
// ================================================================================================
//
//   1. One formatter, one parser, per line.  A line and its parser cannot drift.
//   2. `CLAIM` before `DONE`, always.  A `DONE` with no preceding `CLAIM` closes the connection.
//   3. One `nonce` ties `CLAIM`/`SEG`/`DONE`/`ACK` together.  A stale nonce cannot release,
//      cancel or read another request's slot.
//   4. At most ONE outstanding `REQ` per request, and at most `--prefill-retries` re-asks.  A
//      handoff loop is invisible to the watchdog (S3.8's shape), so the bound is not optional.
//   5. `CANCEL` is legal in every state, and its consequence differs per state (§4.6).  The table
//      says which.
//   6. A temporary refusal never becomes an `ERR` on the client's wire (D5).  `full`, `queue`,
//      `read`, `slotlost`, `shutting`, `endpoint` are waits; only `toobig`, `ids`, `novision`,
//      `resume`, `geom`, `pack`, `badpayload` are errors.
//   7. Version and hash mismatch are caught at the HANDSHAKE, never silently mis-parsed.
//   8. The machine is TOTAL: every state x every event is either handled or refused with a named
//      reason.  The test enumerates the whole table.

#pragma once

#include <array>
#include <charconv>
#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace strata::program::prefill_svc {

// ------------------------------------------------------------------ the version token (§4.2) -----

/// The handshake token a prefill instance emits FIRST.  A decode instance that does not see it knows
/// it is talking to something that is not a prefill instance, and says so:
/// `strata serve: the prefill endpoint does not speak PREFILL-READY-V1`.
inline constexpr const char* kReadyToken = "PREFILL-READY-V1";
/// The protocol revision carried in `HELLO` and in `PREFILL-READY-V1`.  Bumping it is a breaking
/// change: both sides refuse the connection rather than mis-parse each other (§4.2, D3 of §1.6).
inline constexpr uint64_t kProtoVersion = 1;
/// §4.2: "a line longer than 4096 bytes closes the connection".
inline constexpr size_t kMaxLineBytes = 4096;
/// §4.5 / §6.2: the re-ask backoff, "start 50 ms, cap 2 s".
inline constexpr int64_t kBackoffStartMs = 50;
inline constexpr int64_t kBackoffCapMs = 2000;
/// §4.7 / §1.2: `--prefill-retries`, the prefill analogue of `kMaxRereads`.
inline constexpr int64_t kDefaultPrefillRetries = 3;
/// §3.8 / §1.2: `--prefill-queue`; a job queue beyond it is `ERR queue`.
inline constexpr int64_t kDefaultPrefillQueue = 32;
/// §6.1, D6: the ONLY authoritative spelling of the new wait reason until S4.3.6 adds
/// `Wait::prefill` to `serve_driver.hpp`.  It must match `wait_reason()`/`wait_text()` there.
inline constexpr const char* kWaitReasonPrefill = "prefill-busy";
inline constexpr const char* kWaitTextPrefill =
    "the prefill instance has no free handoff slot for a prompt this long";
/// `Wait::prefill`'s slot in `serve_driver.hpp` once S4.3.6 adds it (after `engine_busy`, i.e. 8).
/// Pinned here so the two sides agree; the test asserts the real enum still has room for it.
inline constexpr int kWaitPrefillValue = 8;

// ------------------------------------------------------------------ formatting helpers -----------

namespace detail {
/// snprintf into a std::string.  Deliberately snprintf and not an ostream, and deliberately a COPY of
/// `serve_proto.hpp:54-67`'s helper rather than a shared one: neither header may pull the other in.
inline std::string f(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n < 0) return {};
    if ((size_t) n < sizeof buf) return std::string(buf, (size_t) n);
    std::string out((size_t) n, '\0');
    va_start(ap, fmt);
    std::vsnprintf(&out[0], (size_t) n + 1, fmt, ap);
    va_end(ap);
    return out;
}

/// Tokenise on whitespace.  Empty tokens are dropped, so a stray trailing space is harmless and a
/// double space is not mistaken for a missing field.
inline std::vector<std::string> split(const std::string& line) {
    std::vector<std::string> out;
    std::istringstream in(line);
    std::string t;
    while (in >> t) out.push_back(t);
    return out;
}

inline bool is_uint(const std::string& t, uint64_t& v) {
    if (t.empty()) return false;
    for (char c : t) if (c < '0' || c > '9') return false;
    const auto r = std::from_chars(t.data(), t.data() + t.size(), v);
    return r.ec == std::errc{} && r.ptr == t.data() + t.size();
}
inline bool is_int(const std::string& t, int64_t& v) {
    if (t.empty()) return false;
    size_t i = 0;
    if (t[0] == '-' || t[0] == '+') i = 1;
    if (i >= t.size()) return false;
    for (; i < t.size(); ++i) if (t[i] < '0' || t[i] > '9') return false;
    const auto r = std::from_chars(t.data(), t.data() + t.size(), v);
    return r.ec == std::errc{} && r.ptr == t.data() + t.size();
}
/// `x64`: exactly 16 LOWERCASE hex digits.  Uppercase is refused, because the doc pins the spelling
/// and a formatter that accepts both cannot be golden-tested against one.
inline bool is_hex64(const std::string& t, uint64_t& v) {
    if (t.size() != 16) return false;
    v = 0;
    for (char c : t) {
        if (c >= '0' && c <= '9') v = v * 16u + (uint64_t) (c - '0');
        else if (c >= 'a' && c <= 'f') v = v * 16u + (uint64_t) (c - 'a' + 10);
        else return false;
    }
    return true;
}
inline std::string hex64(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return std::string(b, 16);
}
inline bool is_dec(uint64_t v, const std::string& t) {
    return t == std::to_string(v);
}
/// `NAME`: `[A-Za-z0-9._/-]+`.
inline bool is_name(const std::string& t) {
    if (t.empty()) return false;
    for (char c : t) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                        c == '.' || c == '_' || c == '/' || c == '-';
        if (!ok) return false;
    }
    return true;
}
/// `ID`: `[A-Za-z0-9_-]+` (a PING cookie).
inline bool is_id_token(const std::string& t) {
    if (t.empty()) return false;
    for (char c : t) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                        c == '_' || c == '-';
        if (!ok) return false;
    }
    return true;
}
/// A comma-separated non-negative integer list (`N` or `N,N,...`).  An EMPTY list is legal and yields
/// no entries - that is `split=` with no layer split.  `tiers=`/`slots=` check emptiness themselves.
inline bool parse_list(const std::string& t, std::vector<uint64_t>& out) {
    out.clear();
    if (t.empty()) return true;
    size_t i = 0;
    for (;;) {
        const size_t comma = t.find(',', i);
        const size_t e = comma == std::string::npos ? t.size() : comma;
        if (e == i) return false;                       // empty element: "1,,2", a lead or tail comma
        uint64_t v = 0;
        if (!is_uint(t.substr(i, e - i), v)) return false;
        out.push_back(v);
        if (comma == std::string::npos) break;
        i = comma + 1;
    }
    return true;
}
inline std::string join_list(const std::vector<uint64_t>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) { if (i) s += ","; s += std::to_string(v[i]); }
    return s;
}
/// A `k=v` key: `[A-Za-z][A-Za-z0-9_]*`.
inline bool is_key(const std::string& t) {
    if (t.empty()) return false;
    if (!((t[0] >= 'a' && t[0] <= 'z') || (t[0] >= 'A' && t[0] <= 'Z'))) return false;
    for (char c : t) {
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                        c == '_';
        if (!ok) return false;
    }
    return true;
}
/// A `k=v` value: any non-space ASCII printable.  Messages and paths live here.
inline bool is_value(const std::string& t) {
    for (char c : t) if ((unsigned char) c < 0x21 || (unsigned char) c > 0x7e) return false;
    return true;
}
/// The offset in `line` just past the `nth` (0-based) whitespace-separated token.  Used to keep a
/// free-text field (an ERR message) byte-for-byte instead of re-joining it from tokens.
inline size_t past_token(const std::string& line, size_t nth) {
    size_t p = 0, seen = 0;
    while (p < line.size()) {
        while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
        if (p >= line.size()) return line.size();
        while (p < line.size() && line[p] != ' ' && line[p] != '\t') ++p;
        if (++seen > nth) return p;
    }
    return line.size();
}
}  // namespace detail

// ------------------------------------------------------------------ the wire characters (§4.2) ---

/// True for everything that may appear inside a line: printable ASCII, no CR, no NUL, no newline.
/// A line containing anything else is a FRAMING error, not a parse error - the caller closes the
/// connection rather than logging a bad verb.
inline bool line_is_well_formed(const std::string& line, std::string& why) {
    if (line.empty()) { why = "the line was empty"; return false; }
    if (line.size() > kMaxLineBytes) {
        why = "the line is " + std::to_string(line.size()) + " bytes and the limit is " +
              std::to_string(kMaxLineBytes);
        return false;
    }
    for (char c : line) {
        if (c == '\n') { why = "a line may not contain a newline"; return false; }
        if (c == '\r') { why = "a line may not contain a carriage return"; return false; }
        if (c == '\0') { why = "a line may not contain a NUL"; return false; }
        if ((unsigned char) c < 0x20 || (unsigned char) c > 0x7e) {
            why = "a line is ASCII printable only";
            return false;
        }
    }
    return true;
}

// ------------------------------------------------------------------ the verbs -------------------

enum class LineKind : uint8_t {
    unknown,        ///< a verb this build does not know: ignore it (D8), never close the connection
    hello,          ///< decode -> prefill, once per connection
    ready,          ///< prefill -> decode, once per connection ("PREFILL-READY-V1 ...")
    ping, pong,
    req,            ///< decode -> prefill: read this prompt
    cancel,         ///< decode -> prefill: stop that request
    ack,            ///< decode -> prefill: copied out and validated; the slot may be reused
    bye,            ///< decode -> prefill: orderly shutdown of this connection
    queued,         ///< prefill -> decode: accepted, no slot claimed yet
    claim,          ///< prefill -> decode: a slot was claimed
    seg,            ///< prefill -> decode: the read is done, the payload write started
    done,           ///< prefill -> decode: the payload is published
    err,            ///< prefill -> decode: this request failed
    info,           ///< prefill -> decode: process-wide, untagged
    take,           ///< RESERVED for S4.5's reverse handoff; refused in v1
};

inline const char* line_name(LineKind k) {
    switch (k) {
        case LineKind::unknown: return "unknown";
        case LineKind::hello: return "HELLO";
        case LineKind::ready: return "PREFILL-READY-V1";
        case LineKind::ping: return "PING";
        case LineKind::pong: return "PONG";
        case LineKind::req: return "REQ";
        case LineKind::cancel: return "CANCEL";
        case LineKind::ack: return "ACK";
        case LineKind::bye: return "BYE";
        case LineKind::queued: return "QUEUED";
        case LineKind::claim: return "CLAIM";
        case LineKind::seg: return "SEG";
        case LineKind::done: return "DONE";
        case LineKind::err: return "ERR";
        case LineKind::info: return "INFO";
        case LineKind::take: return "TAKE";
    }
    return "?";
}

/// Which side may SEND this verb.  `parse_line` does not know which side it is on - the caller checks
/// with `line_from_peer()`, and a line from the wrong side is a protocol violation.
enum class Side : uint8_t { either, decode_to_prefill, prefill_to_decode };
inline Side line_side(LineKind k) {
    switch (k) {
        case LineKind::hello: case LineKind::ping: case LineKind::req: case LineKind::cancel:
        case LineKind::ack: case LineKind::bye:
            return Side::decode_to_prefill;
        case LineKind::ready: case LineKind::pong: case LineKind::queued: case LineKind::claim:
        case LineKind::seg: case LineKind::done: case LineKind::err: case LineKind::info:
            return Side::prefill_to_decode;
        case LineKind::take: case LineKind::unknown: return Side::either;
    }
    return Side::either;
}
/// Is a line of kind `k` a legal message FROM THE PEER, given that this process is `me`?
inline bool line_from_peer(LineKind k, Side me) {
    const Side want = line_side(k);
    if (want == Side::either) return true;
    if (me == Side::decode_to_prefill) return want == Side::prefill_to_decode;   // I am decode
    if (me == Side::prefill_to_decode) return want == Side::decode_to_prefill;   // I am prefill
    return true;
}

// ------------------------------------------------------------------ the error codes (§4.4) ------

enum class ErrCode : uint8_t {
    none,
    full,        ///< no free slot and no queue room                     temporary
    queue,       ///< --prefill-queue is full                            temporary
    toobig,      ///< tokens > --prefill-max-tokens / the largest tier   permanent
    ids,         ///< the ids file is missing or unparsable              permanent
    novision,    ///< a GENI against a prefill instance without --vision permanent
    resume,      ///< resume != 0 in v1                                  permanent
    geom,        ///< geometry / split mismatch                          permanent, drop the connection
    pack,        ///< pack hash mismatch                                 permanent, drop the connection
    cancel,      ///< the request was cancelled before it ran            n/a
    read,        ///< the prompt read itself failed                      temporary, retry
    slotlost,    ///< the slot was reclaimed mid-write                   temporary, retry once then hold
    shutting,    ///< the prefill instance is exiting                    temporary
    badpayload,  ///< the decode instance's own §2.5 checks rejected it  permanent (decode-emitted)
    endpoint,    ///< the socket is down/closed/unparseable              temporary (decode-emitted)
};

inline const char* err_name(ErrCode c) {
    switch (c) {
        case ErrCode::none: return "none";
        case ErrCode::full: return "full";
        case ErrCode::queue: return "queue";
        case ErrCode::toobig: return "toobig";
        case ErrCode::ids: return "ids";
        case ErrCode::novision: return "novision";
        case ErrCode::resume: return "resume";
        case ErrCode::geom: return "geom";
        case ErrCode::pack: return "pack";
        case ErrCode::cancel: return "cancel";
        case ErrCode::read: return "read";
        case ErrCode::slotlost: return "slotlost";
        case ErrCode::shutting: return "shutting";
        case ErrCode::badpayload: return "badpayload";
        case ErrCode::endpoint: return "endpoint";
    }
    return "?";
}
inline bool err_from_code(const std::string& t, ErrCode& out) {
    for (int i = 1; i <= (int) ErrCode::endpoint; ++i) {
        const ErrCode c = (ErrCode) i;
        if (t == err_name(c)) { out = c; return true; }
    }
    out = ErrCode::none;
    return false;
}
/// The number of codes in the table.  The test asserts the table has exactly this many entries, so
/// adding a code without classifying it fails the test, not the server.
inline constexpr int prefill_svc_error_count() { return (int) ErrCode::endpoint; }

// ---- the mapping onto serve_driver::Hold ------------------------------------------------------
//
// §4.4: "the mapping lives in one table in `prefill_svc.hpp` so the two sides cannot drift".
// `serve_driver.hpp` owns `Hold` (`:885-889`) and `Wait` (`:902-911`), and S4.3.3 may not edit that
// file, so the vocabulary is restated here as an int with the SAME values and the SAME names.  The
// test restates it a third time from the real header and asserts the numbers match, so a drift is a
// test failure rather than a protocol surprise.
//
//   0 == serve_driver::Hold::run_now   (never produced by an error code - see D5)
//   1 == serve_driver::Hold::wait      (a temporary refusal: hold, re-ask, never ERR the client)
//   2 == serve_driver::Hold::error     (permanent: waiting would only hide it)
namespace hold {
inline constexpr int run_now = 0;
inline constexpr int wait = 1;
inline constexpr int error = 2;
inline const char* name(int h) {
    switch (h) {
        case run_now: return "run-now";
        case wait: return "wait";
        case error: return "error";
    }
    return "?";
}
}  // namespace hold

/// The classification of one error code.  `drop_connection` is §4.4's "also drop the connection"
/// column, which is a separate property: `geom`/`pack` are both permanent AND fatal to the socket,
/// while `badpayload` is permanent but says nothing about the socket, and `cancel` is neither.
/// `decode_emitted` marks the last two rows of §4.4's table, which the prefill instance never sends.
struct ErrClass {
    ErrCode code = ErrCode::none;
    int hold = hold::error;
    bool drop_connection = false;
    bool retryable = false;
    bool decode_emitted = false;
    const char* text = "";       ///< the sentence for the log / the client
};

/// The whole table, in one place.  Total: every code from `full` to `endpoint` appears exactly once.
inline const std::vector<ErrClass>& err_table() {
    static const std::vector<ErrClass> t = {
        {ErrCode::full,       hold::wait,  false, false, false,
            "the prefill instance has no free handoff slot for a prompt this long"},
        {ErrCode::queue,      hold::wait,  false, false, false,
            "the prefill instance's job queue is full"},
        {ErrCode::toobig,     hold::error, false, false, false,
            "the prompt is longer than the prefill instance can read"},
        {ErrCode::ids,        hold::error, false, false, false,
            "the prompt ids file is missing or unparsable"},
        {ErrCode::novision,   hold::error, false, false, false,
            "the prefill instance was started without --vision"},
        {ErrCode::resume,     hold::error, false, false, false,
            "the prefill instance owns the reuse decision; resume must be 0"},
        {ErrCode::geom,       hold::error, true,  false, false,
            "the geometry or layer split does not match the prefill instance"},
        {ErrCode::pack,       hold::error, true,  false, false,
            "the expert pack does not match the prefill instance"},
        {ErrCode::cancel,     hold::wait,  false, false, false,
            "the request was cancelled"},
        {ErrCode::read,       hold::wait,  false, true,  false,
            "the prompt read failed at the prefill instance"},
        {ErrCode::slotlost,   hold::wait,  false, true,  false,
            "the handoff slot was reclaimed while the payload was being written"},
        {ErrCode::shutting,   hold::wait,  false, false, false,
            "the prefill instance is exiting"},
        {ErrCode::badpayload, hold::error, false, false, true,
            "the received payload failed validation and was not mounted"},
        {ErrCode::endpoint,   hold::wait,  false, false, true,
            "the prefill endpoint is down"},
    };
    return t;
}

inline ErrClass err_class(ErrCode c) {
    for (const ErrClass& e : err_table()) if (e.code == c) return e;
    // An unknown code is a bug in THIS file, not a wire condition.  Say so loudly rather than
    // silently classifying it as a wait, which would turn a new code into an infinite hold.
    return {c, hold::error, false, false, false, "unknown prefill error code"};
}
inline int hold_for(ErrCode c) { return err_class(c).hold; }
inline bool err_is_permanent(ErrCode c) { return err_class(c).hold == hold::error; }
inline bool err_is_temporary(ErrCode c) { return err_class(c).hold == hold::wait; }
inline bool err_drops_connection(ErrCode c) { return err_class(c).drop_connection; }
inline bool err_is_retryable(ErrCode c) { return err_class(c).retryable; }
inline bool err_is_decode_emitted(ErrCode c) { return err_class(c).decode_emitted; }
inline const char* err_text(ErrCode c) { return err_class(c).text; }
/// D5's rule, restated: a temporary refusal must never become an `ERR` on the client's wire.  This is
/// the predicate S4.3.6 calls before it touches `serve_proto`.
inline bool err_reaches_the_client(ErrCode c) { return err_class(c).hold == hold::error; }

/// §6.1: exactly ONE new wait reason.  The strings are D6's, and they must match
/// `serve_driver.hpp`'s `wait_reason()`/`wait_text()` once S4.3.6 adds `Wait::prefill`.
inline const char* wait_reason_prefill() { return kWaitReasonPrefill; }
inline const char* wait_text_prefill() { return kWaitTextPrefill; }

// ------------------------------------------------------------------ the parsed line -------------

/// One parsed line: every field of every verb in one struct.  `bad` non-empty is the ONLY failure a
/// reader acts on by closing the connection (§4.3: "a line whose known tokens do not parse closes
/// the connection and logs the offending line").
struct Line {
    LineKind kind = LineKind::unknown;
    std::string verb;
    std::string raw;
    std::string bad;                       ///< non-empty => malformed; the reason
    std::vector<std::string> extra;        ///< unknown trailing tokens: ignored, kept for the log

    // HELLO
    uint64_t proto_ver = 0, client_id = 0, pack_hash = 0, geom_hash = 0;
    int64_t max_ctx = 0, n_stages = 0;
    std::vector<uint64_t> split;           ///< layer_lo of every LATER stage, engine order (D9)
    int cvec = 1, vision = 0;
    // PREFILL-READY-V1
    std::vector<uint64_t> slots, tiers;
    uint64_t max_tokens = 0;
    // PING / PONG
    std::string cookie;
    // REQ
    int64_t id = -1;
    uint64_t tokens = 0, tier = 0;
    std::string ids_name, geni;
    int64_t resume = 0;
    std::map<std::string, std::string> keys;
    // CANCEL / ACK / BYE
    uint64_t nonce = 0;
    std::string reason;
    // QUEUED
    uint64_t ahead = 0;
    // CLAIM / SEG / DONE / ERR
    uint64_t slot = 0, seq = 0, tokens_done = 0, tokens_total = 0, bytes = 0;
    double ms = 0.0;
    ErrCode code = ErrCode::none;
    std::string message;
    // INFO
    std::vector<std::pair<std::string, std::string>> info;

    bool ok() const { return kind != LineKind::unknown && bad.empty(); }
};

namespace detail {
/// The optional `k=v ...` keys of `REQ`/`HELLO`.  Stops at the first token that is not a key.
inline size_t parse_keys(const std::vector<std::string>& t, size_t i, Line& l) {
    for (; i < t.size(); ++i) {
        const std::string& tok = t[i];
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) break;
        const std::string key = tok.substr(0, eq);
        if (!is_key(key)) break;
        l.keys[key] = tok.substr(eq + 1);
    }
    return i;
}
}  // namespace detail

/// Parse one control line.  Never throws, never does I/O.  On an unknown verb `kind == unknown` and
/// `bad` stays empty (D8).  On a known verb with bad fields, `bad` names the reason and the caller
/// closes the connection and logs `raw`.
inline Line parse_line(const std::string& line) {
    Line l;
    l.raw = line;
    const std::vector<std::string> t = detail::split(line);
    if (t.empty()) { l.bad = "the line was empty"; return l; }
    l.verb = t[0];

    if (l.verb == "HELLO") l.kind = LineKind::hello;
    else if (l.verb == kReadyToken) l.kind = LineKind::ready;
    else if (l.verb == "PING") l.kind = LineKind::ping;
    else if (l.verb == "PONG") l.kind = LineKind::pong;
    else if (l.verb == "REQ") l.kind = LineKind::req;
    else if (l.verb == "CANCEL") l.kind = LineKind::cancel;
    else if (l.verb == "ACK") l.kind = LineKind::ack;
    else if (l.verb == "BYE") l.kind = LineKind::bye;
    else if (l.verb == "QUEUED") l.kind = LineKind::queued;
    else if (l.verb == "CLAIM") l.kind = LineKind::claim;
    else if (l.verb == "SEG") l.kind = LineKind::seg;
    else if (l.verb == "DONE") l.kind = LineKind::done;
    else if (l.verb == "ERR") l.kind = LineKind::err;
    else if (l.verb == "INFO") l.kind = LineKind::info;
    else if (l.verb == "TAKE") l.kind = LineKind::take;
    else { l.extra.assign(t.begin() + 1, t.end()); return l; }   // D8: unknown verb, not an error

    size_t i = 1;
    auto need = [&](const char* what) {
        l.bad = std::string(line_name(l.kind)) + " is missing " + what;
        return false;
    };
    auto num = [&](const char* what, uint64_t& dst) {
        if (i >= t.size()) return need(what);
        if (!detail::is_uint(t[i], dst)) {
            l.bad = std::string(line_name(l.kind)) + " has a non-numeric " + what + ": [" + t[i] + "]";
            return false;
        }
        ++i;
        return true;
    };
    auto snum = [&](const char* what, int64_t& dst) {
        if (i >= t.size()) return need(what);
        if (!detail::is_int(t[i], dst)) {
            l.bad = std::string(line_name(l.kind)) + " has a non-numeric " + what + ": [" + t[i] + "]";
            return false;
        }
        ++i;
        return true;
    };
    auto hex = [&](const char* what, uint64_t& dst) {
        if (i >= t.size()) return need(what);
        if (!detail::is_hex64(t[i], dst)) {
            l.bad = std::string(line_name(l.kind)) + " has a bad " + what +
                    " (want 16 lowercase hex digits): [" + t[i] + "]";
            return false;
        }
        ++i;
        return true;
    };
    auto kvlist = [&](const char* name, std::vector<uint64_t>& dst) {
        if (i >= t.size()) { l.bad = std::string(line_name(l.kind)) + " is missing " + name + "="; return false; }
        if (t[i].rfind(name + std::string("="), 0) != 0) {
            l.bad = std::string(line_name(l.kind)) + " wants " + name + "=LIST, got [" + t[i] + "]";
            return false;
        }
        if (!detail::parse_list(t[i].substr(std::strlen(name) + 1), dst)) {
            l.bad = std::string(line_name(l.kind)) + " has a bad " + name + "=: [" + t[i] + "]";
            return false;
        }
        ++i;
        return true;
    };
    auto kvnum = [&](const char* name, uint64_t& dst) {
        if (i >= t.size()) { l.bad = std::string(line_name(l.kind)) + " is missing " + name + "="; return false; }
        if (t[i].rfind(name + std::string("="), 0) != 0) {
            l.bad = std::string(line_name(l.kind)) + " wants " + name + "=, got [" + t[i] + "]";
            return false;
        }
        if (!detail::is_uint(t[i].substr(std::strlen(name) + 1), dst)) {
            l.bad = std::string(line_name(l.kind)) + " has a bad " + name + "=: [" + t[i] + "]";
            return false;
        }
        ++i;
        return true;
    };
    auto kvhex = [&](const char* name, uint64_t& dst) {
        if (i >= t.size()) { l.bad = std::string(line_name(l.kind)) + " is missing " + name + "="; return false; }
        if (t[i].rfind(name + std::string("="), 0) != 0) {
            l.bad = std::string(line_name(l.kind)) + " wants " + name + "=, got [" + t[i] + "]";
            return false;
        }
        if (!detail::is_hex64(t[i].substr(std::strlen(name) + 1), dst)) {
            l.bad = std::string(line_name(l.kind)) + " has a bad " + name +
                    "= (want 16 lowercase hex digits): [" + t[i] + "]";
            return false;
        }
        ++i;
        return true;
    };
    auto word = [&](const char* what, std::string& dst) {
        if (i >= t.size()) return need(what);
        dst = t[i];
        ++i;
        return true;
    };
    auto number = [&](const char* what, double& dst) {
        if (i >= t.size()) return need(what);
        const std::string& s = t[i];
        char* endp = nullptr;
        const double v = std::strtod(s.c_str(), &endp);
        // `!(v >= 0.0)` rejects NaN as well as a negative; the ceiling rejects "1e999" -> inf.  A
        // non-finite timing is a corrupt line, and a formatter that emits "inf" would then be
        // unparseable by the peer on the next hop.
        if (endp == s.c_str() || *endp != '\0' || !(v >= 0.0) || v > 1e15) {
            l.bad = std::string(line_name(l.kind)) + " has a bad " + what + ": [" + s + "]";
            return false;
        }
        dst = v;
        ++i;
        return true;
    };
    auto binary = [&](const std::map<std::string, std::string>& keys, const char* name, int& dst) {
        const auto it = keys.find(name);
        if (it == keys.end()) return true;
        if (it->second != "0" && it->second != "1") {
            l.bad = std::string(line_name(l.kind)) + " " + name + " must be 0 or 1";
            return false;
        }
        dst = it->second == "1";
        return true;
    };

    switch (l.kind) {
        case LineKind::hello: {
            if (!num("proto_ver", l.proto_ver)) return l;
            if (!num("client_id", l.client_id)) return l;
            if (!hex("pack_hash", l.pack_hash)) return l;
            if (!hex("geom_hash", l.geom_hash)) return l;
            if (!snum("max_ctx", l.max_ctx)) return l;
            if (!snum("n_stages", l.n_stages)) return l;
            if (i >= t.size()) { l.bad = "HELLO is missing split="; return l; }
            if (t[i].rfind("split=", 0) != 0) {
                l.bad = "HELLO wants split=LIST after n_stages, got [" + t[i] + "]";
                return l;
            }
            if (!detail::parse_list(t[i].substr(6), l.split)) {
                l.bad = "HELLO has a bad split=: [" + t[i] + "]";
                return l;
            }
            ++i;
            i = detail::parse_keys(t, i, l);
            for (; i < t.size(); ++i) l.extra.push_back(t[i]);
            if (!binary(l.keys, "cvec", l.cvec)) return l;
            if (!binary(l.keys, "vision", l.vision)) return l;
            if (l.n_stages < 0) { l.bad = "HELLO n_stages is negative"; return l; }
            if ((int64_t) l.split.size() != l.n_stages) {   // D9
                l.bad = "HELLO says n_stages=" + std::to_string(l.n_stages) + " but split= has " +
                        std::to_string(l.split.size()) + " entries";
                return l;
            }
            return l;
        }
        case LineKind::ready: {
            if (!num("proto_ver", l.proto_ver)) return l;
            if (!kvlist("slots", l.slots)) return l;
            if (!kvnum("max_tokens", l.max_tokens)) return l;
            if (!kvlist("tiers", l.tiers)) return l;
            if (!kvhex("pack_hash", l.pack_hash)) return l;
            if (!kvhex("geom_hash", l.geom_hash)) return l;
            for (; i < t.size(); ++i) l.extra.push_back(t[i]);   // unknown trailing keys: ignored
            if (l.slots.size() != l.tiers.size()) {
                l.bad = "PREFILL-READY-V1 slots= has " + std::to_string(l.slots.size()) +
                        " entries and tiers= has " + std::to_string(l.tiers.size());
                return l;
            }
            if (l.tiers.empty()) { l.bad = "PREFILL-READY-V1 tiers= is empty"; return l; }
            return l;
        }
        case LineKind::ping: case LineKind::pong: {
            if (!word("cookie", l.cookie)) return l;
            if (!detail::is_id_token(l.cookie)) {
                l.bad = std::string(line_name(l.kind)) + " cookie must be [A-Za-z0-9_-]+";
                return l;
            }
            for (; i < t.size(); ++i) l.extra.push_back(t[i]);
            return l;
        }
        case LineKind::req: {
            if (!snum("id", l.id)) return l;
            if (!num("tokens", l.tokens)) return l;
            if (!num("tier", l.tier)) return l;
            i = detail::parse_keys(t, i, l);
            for (; i < t.size(); ++i) l.extra.push_back(t[i]);
            const auto it_ids = l.keys.find("ids");
            if (it_ids == l.keys.end()) { l.bad = "REQ is missing ids=NAME"; return l; }
            l.ids_name = it_ids->second;
            if (!detail::is_name(l.ids_name)) {
                l.bad = "REQ ids= is not a NAME: [" + l.ids_name + "]";
                return l;
            }
            const auto it_g = l.keys.find("geni");
            if (it_g != l.keys.end()) {
                l.geni = it_g->second;
                if (!detail::is_name(l.geni)) {
                    l.bad = "REQ geni= is not a NAME: [" + l.geni + "]";
                    return l;
                }
            }
            if (!binary(l.keys, "cvec", l.cvec)) return l;
            const auto it_r = l.keys.find("resume");
            if (it_r != l.keys.end()) {
                if (!detail::is_int(it_r->second, l.resume)) {
                    l.bad = "REQ resume= is not an integer: [" + it_r->second + "]";
                    return l;
                }
            }
            return l;
        }
        case LineKind::cancel: {
            if (!snum("id", l.id)) return l;
            for (; i < t.size(); ++i) l.extra.push_back(t[i]);
            return l;
        }
        case LineKind::ack: {
            if (!snum("id", l.id)) return l;
            if (!num("nonce", l.nonce)) return l;
            for (; i < t.size(); ++i) l.extra.push_back(t[i]);
            return l;
        }
        case LineKind::bye: {
            // The reason is free text (§4.3's `BYE <reason>`): keep the rest of the line so a shutdown
            // reason with spaces survives the round trip.
            if (i < t.size()) {
                size_t p = detail::past_token(line, 0);   // past token 0 = past "BYE"
                while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
                l.reason = line.substr(p);
            }
            return l;
        }
        case LineKind::queued: {
            if (!snum("id", l.id)) return l;
            if (!num("ahead", l.ahead)) return l;
            for (; i < t.size(); ++i) l.extra.push_back(t[i]);
            return l;
        }
        case LineKind::claim: {
            if (!snum("id", l.id)) return l;
            if (!num("slot", l.slot)) return l;
            if (!num("nonce", l.nonce)) return l;
            for (; i < t.size(); ++i) l.extra.push_back(t[i]);
            return l;
        }
        case LineKind::seg: {
            if (!snum("id", l.id)) return l;
            if (!num("slot", l.slot)) return l;
            if (!num("seq", l.seq)) return l;
            if (!num("tokens_done", l.tokens_done)) return l;
            if (!num("tokens_total", l.tokens_total)) return l;
            if (!number("ms", l.ms)) return l;
            for (; i < t.size(); ++i) l.extra.push_back(t[i]);
            return l;
        }
        case LineKind::done: {
            if (!snum("id", l.id)) return l;
            if (!num("slot", l.slot)) return l;
            if (!num("nonce", l.nonce)) return l;
            if (!num("bytes", l.bytes)) return l;
            if (!num("tokens", l.tokens)) return l;
            if (!number("ms", l.ms)) return l;
            for (; i < t.size(); ++i) l.extra.push_back(t[i]);
            return l;
        }
        case LineKind::err: {
            if (!snum("id", l.id)) return l;
            std::string code;
            if (!word("code", code)) return l;
            if (!err_from_code(code, l.code)) {
                l.bad = "ERR has an unknown code: [" + code + "]";
                return l;
            }
            // The message is FREE TEXT (§4.4): keep the rest of the line byte-for-byte rather than
            // re-joining tokens, so a log line quotes what the peer actually said.
            size_t p = detail::past_token(line, 2);
            while (p < line.size() && (line[p] == ' ' || line[p] == '\t')) ++p;
            l.message = line.substr(p);
            if (l.message.empty()) { l.bad = "ERR is missing a message"; return l; }
            return l;
        }
        case LineKind::info: {
            for (; i < t.size(); ++i) {
                const size_t eq = t[i].find('=');
                if (eq == std::string::npos || !detail::is_key(t[i].substr(0, eq))) {
                    l.bad = "INFO wants k=v tokens, got [" + t[i] + "]";
                    return l;
                }
                l.info.emplace_back(t[i].substr(0, eq), t[i].substr(eq + 1));
            }
            return l;
        }
        case LineKind::take: {
            l.bad = "TAKE is reserved for the reverse handoff (S4.5) and is not part of v1";
            return l;
        }
        case LineKind::unknown: return l;
    }
    return l;
}

// ------------------------------------------------------------------ the formatters --------------

/// The `REQ` fields.  `ids_name` is REQUIRED (§4.3): the prompt ids live in a file, never on the wire.
struct ReqKeys {
    int64_t id = -1;
    uint64_t tokens = 0;
    uint64_t tier = 0;
    std::string ids_name;
    std::string geni;                 ///< empty = text
    int cvec = 1;
    int64_t resume = 0;
};

/// The §4.3 ids-file name (D7).  `dir` is the instance directory (§3.2); the file is
/// `req-<client>-<nonce>.ids`, mode 0600, deleted after `ACK`.  Returns false rather than build a
/// path a peer could point outside the instance directory.  This builds the LOCAL path from
/// `--prefill-handoff-dir`, which is normally absolute, so an absolute `dir` is allowed unless the
/// caller says otherwise - what is never allowed is a `..` component.  The wire-side rule (a name
/// arriving in `ids=`) is `ids_name_is_safe()`, which refuses an absolute name as well.
inline bool req_ids_path(const std::string& dir, uint64_t client_id, uint64_t nonce,
                         std::string& out, bool allow_absolute = true) {
    out.clear();
    if (dir.empty()) return false;
    if (!allow_absolute && dir[0] == '/') return false;
    for (size_t i = 0; i < dir.size(); ++i) {
        if (dir[i] == '.' && i + 1 < dir.size() && dir[i + 1] == '.') return false;
        const char c = dir[i];
        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                        c == '.' || c == '_' || c == '/' || c == '-';
        if (!ok) return false;
    }
    std::string name = detail::f("req-%016llx-%016llx.ids", (unsigned long long) client_id,
                                 (unsigned long long) nonce);
    out = dir;
    if (!out.empty() && out.back() != '/') out += '/';
    out += name;
    return true;
}
/// Is `name` a safe `ids=`/`geni=` value as it arrives on the wire?  Same rule as D7.
inline bool ids_name_is_safe(const std::string& name) {
    if (!detail::is_name(name)) return false;
    if (name.rfind('/', 0) == 0) return false;                 // absolute: refuse
    size_t i = 0;
    while (i < name.size()) {
        const size_t s = name.find('/', i);
        const std::string part = name.substr(i, s == std::string::npos ? std::string::npos : s - i);
        if (part == "..") return false;
        i = (s == std::string::npos) ? name.size() : s + 1;
    }
    return true;
}

/// decode -> prefill -------------------------------------------------------------------------------

inline std::string hello_line(uint64_t proto_ver, uint64_t client_id, uint64_t pack_hash,
                              uint64_t geom_hash, int64_t max_ctx,
                              const std::vector<uint64_t>& later_stage_layer_lo,
                              int cvec, int vision) {
    return detail::f("HELLO %llu %llu %s %s %lld %lld split=%s cvec=%d vision=%d",
                     (unsigned long long) proto_ver, (unsigned long long) client_id,
                     detail::hex64(pack_hash).c_str(), detail::hex64(geom_hash).c_str(),
                     (long long) max_ctx, (long long) later_stage_layer_lo.size(),
                     detail::join_list(later_stage_layer_lo).c_str(), cvec ? 1 : 0, vision ? 1 : 0);
}
inline std::string ping_line(const std::string& cookie) { return "PING " + cookie; }
inline std::string req_line(const ReqKeys& r) {
    std::string s = detail::f("REQ %lld %llu %llu ids=%s", (long long) r.id,
                              (unsigned long long) r.tokens, (unsigned long long) r.tier,
                              r.ids_name.c_str());
    if (!r.geni.empty()) s += " geni=" + r.geni;
    s += " cvec=" + std::string(r.cvec ? "1" : "0");
    if (r.resume != 0) s += " resume=" + std::to_string((long long) r.resume);
    return s;
}
inline std::string cancel_line(int64_t id) { return "CANCEL " + std::to_string((long long) id); }
inline std::string ack_line(int64_t id, uint64_t nonce) {
    return detail::f("ACK %lld %llu", (long long) id, (unsigned long long) nonce);
}
inline std::string bye_line(const std::string& reason) {
    return reason.empty() ? std::string("BYE") : "BYE " + reason;
}

/// prefill -> decode -----------------------------------------------------------------------------

struct ReadyInfo {
    uint64_t proto_ver = kProtoVersion;
    std::vector<uint64_t> slots;     ///< per tier, index-aligned with tiers
    std::vector<uint64_t> tiers;     ///< ASCENDING, the order the arena stored them in (S4.3.2)
    uint64_t max_tokens = 0;         ///< the largest tier == tiers.back()
    uint64_t pack_hash = 0, geom_hash = 0;
};
inline std::string ready_line(const ReadyInfo& r) {
    return detail::f("%s %llu slots=%s max_tokens=%llu tiers=%s pack_hash=%s geom_hash=%s",
                     kReadyToken, (unsigned long long) r.proto_ver,
                     detail::join_list(r.slots).c_str(), (unsigned long long) r.max_tokens,
                     detail::join_list(r.tiers).c_str(), detail::hex64(r.pack_hash).c_str(),
                     detail::hex64(r.geom_hash).c_str());
}
inline std::string pong_line(const std::string& cookie) { return "PONG " + cookie; }
inline std::string queued_line(int64_t id, uint64_t ahead) {
    return detail::f("QUEUED %lld %llu", (long long) id, (unsigned long long) ahead);
}
inline std::string claim_line(int64_t id, uint64_t slot, uint64_t nonce) {
    return detail::f("CLAIM %lld %llu %llu", (long long) id, (unsigned long long) slot,
                     (unsigned long long) nonce);
}
inline std::string seg_line(int64_t id, uint64_t slot, uint64_t seq, uint64_t tokens_done,
                            uint64_t tokens_total, double ms) {
    return detail::f("SEG %lld %llu %llu %llu %llu %.0f", (long long) id, (unsigned long long) slot,
                     (unsigned long long) seq, (unsigned long long) tokens_done,
                     (unsigned long long) tokens_total, ms);
}
inline std::string done_line(int64_t id, uint64_t slot, uint64_t nonce, uint64_t bytes,
                             uint64_t tokens, double ms) {
    return detail::f("DONE %lld %llu %llu %llu %llu %.0f", (long long) id, (unsigned long long) slot,
                     (unsigned long long) nonce, (unsigned long long) bytes,
                     (unsigned long long) tokens, ms);
}
inline std::string err_line(int64_t id, ErrCode code, const std::string& message) {
    const std::string m = message.empty() ? err_text(code) : message;
    return detail::f("ERR %lld %s %s", (long long) id, err_name(code), m.c_str());
}
inline std::string info_line(const std::vector<std::pair<std::string, std::string>>& kv) {
    std::string s = "INFO";
    for (const auto& e : kv) s += " " + e.first + "=" + e.second;
    return s;
}

// ------------------------------------------------------------------ the handshake (§4.2) --------

/// The decode instance's view of a `PREFILL-READY-V1`.  False + a reason when the two sides cannot
/// speak: a version mismatch, a pack mismatch, a geometry mismatch, or a self-contradictory READY
/// line.  §4.2: the decode instance "checks proto_ver, pack_hash and geom_hash here and refuses to
/// start a request until they match".  This is the ONLY place a version difference is decided, and it
/// is why an old decode instance against a new prefill instance can never mis-parse a line: the
/// connection is refused before any line beyond the handshake is read.
inline bool handshake_decision(const Line& hello, const Line& ready, std::string& why) {
    if (hello.kind != LineKind::hello) { why = "the first line out was not HELLO"; return false; }
    if (!hello.bad.empty()) { why = "HELLO did not parse: " + hello.bad; return false; }
    if (ready.kind != LineKind::ready) {
        why = std::string("the endpoint answered with [") + (ready.verb.empty() ? ready.raw : ready.verb) +
              "] instead of " + kReadyToken;
        return false;
    }
    if (!ready.bad.empty()) { why = "PREFILL-READY-V1 did not parse: " + ready.bad; return false; }
    if (ready.proto_ver != hello.proto_ver) {
        why = "the endpoint speaks protocol version " + std::to_string(ready.proto_ver) +
              " and this instance speaks " + std::to_string(hello.proto_ver) +
              "; both sides must run the same handoff protocol";
        return false;
    }
    if (ready.proto_ver != kProtoVersion) {
        why = std::string("the endpoint speaks protocol version ") + std::to_string(ready.proto_ver) +
              " and this build speaks " + std::to_string(kProtoVersion) + " (" + kReadyToken + ")";
        return false;
    }
    if (ready.pack_hash != hello.pack_hash) {
        why = "handoff refused: pack hash " + detail::hex64(hello.pack_hash) + " != " +
              detail::hex64(ready.pack_hash) + " (this instance and the prefill instance do not run "
              "the same expert pack)";
        return false;
    }
    if (ready.geom_hash != hello.geom_hash) {
        why = "handoff refused: geometry hash " + detail::hex64(hello.geom_hash) + " != " +
              detail::hex64(ready.geom_hash) + " (this instance and the prefill instance do not run "
              "the same layer split)";
        return false;
    }
    if (ready.slots.empty()) { why = "PREFILL-READY-V1 declares no slots"; return false; }
    if (ready.max_tokens == 0) { why = "PREFILL-READY-V1 declares max_tokens 0"; return false; }
    // §3.10/S4.3.2: the largest tier is the ceiling `--prefill-max-tokens` may claim, and tiers are
    // stored ASCENDING.  A READY line whose max_tokens contradicts its own tier list is a bug in the
    // peer, and it is cheaper to catch here than to price a `toobig` refusal against the wrong
    // ceiling later (that is how a 262144-token prompt ends up in a 131072-token slot).
    uint64_t top = 0;
    for (uint64_t v : ready.tiers) if (v > top) top = v;
    if (top != ready.max_tokens) {
        why = "PREFILL-READY-V1 says max_tokens=" + std::to_string(ready.max_tokens) +
              " but its largest tier is " + std::to_string(top);
        return false;
    }
    for (uint64_t s : ready.slots) if (s == 0) { why = "PREFILL-READY-V1 declares a tier with 0 slots"; return false; }
    return true;
}

/// The prefill instance's side: may this HELLO be served at all?  A prefill instance that does not
/// know the protocol version closes the connection WITHOUT a line (§4.2), so this returns false and
/// the caller closes; it does not format a refusal the other side cannot parse.
inline bool hello_is_acceptable(const Line& hello, std::string& why) {
    if (hello.kind != LineKind::hello) { why = "the first line in was not HELLO"; return false; }
    if (!hello.bad.empty()) { why = "HELLO did not parse: " + hello.bad; return false; }
    if (hello.proto_ver != kProtoVersion) {
        why = "the client speaks protocol version " + std::to_string(hello.proto_ver) +
              " and this instance speaks " + std::to_string(kProtoVersion);
        return false;
    }
    if (hello.client_id == 0) { why = "HELLO client_id 0 is reserved (a slot with no client, §3.4)"; return false; }
    if (hello.max_ctx <= 0) { why = "HELLO max_ctx must be positive"; return false; }
    return true;
}

/// The `cvec`/`vision` echo check (§4.2): a mismatch is caught at HELLO rather than after a 3 GB
/// transfer.  `vision` is one-way - a decode instance that needs images cannot be served by a prefill
/// instance started without `--vision`, and every later GENI is `ERR novision`.
inline bool hello_requirements_met(const Line& hello, int prefill_cvec, int prefill_vision,
                                   std::string& why) {
    if (hello.cvec && !prefill_cvec) {
        why = "the client asked for cvec=1 and this prefill instance runs without control vectors";
        return false;
    }
    if (hello.vision && !prefill_vision) {
        why = "the client asked for vision=1 and this prefill instance was started without --vision";
        return false;
    }
    return true;
}

// ------------------------------------------------------------------ the tier rule (§3.8) --------

/// The tier index a `tokens`-long prompt asks for, given the arena's ASCENDING tier token counts.
/// False when the prompt is bigger than the largest tier - which is `ERR toobig`, priced before any
/// GPU work (§4.4), and permanent.
inline bool tier_for_tokens(uint64_t tokens, const std::vector<uint64_t>& ascending_tiers,
                            uint64_t& tier) {
    if (ascending_tiers.empty()) return false;
    for (size_t i = 0; i < ascending_tiers.size(); ++i) {
        if (tokens <= ascending_tiers[i]) { tier = i; return true; }
    }
    return false;
}
/// `--prefill-max-tokens` and the arena ceiling are two different numbers and both refuse with
/// `toobig`.  This is the check S4.3.6 runs at ADMISSION, before a REQ is ever sent (§6.2: "a request
/// that can never run must not occupy a handoff slot"), and it is permanent: waiting will not make
/// the prefill instance able to read it.
inline bool prefill_prompt_too_big(uint64_t tokens, uint64_t max_prefill_tokens,
                                   uint64_t arena_max_tokens, std::string& why) {
    const uint64_t cap = max_prefill_tokens ? max_prefill_tokens : arena_max_tokens;
    if (cap == 0) { why = "the prefill endpoint declares no maximum prompt length"; return true; }
    if (tokens > cap) {
        why = "the prompt is " + std::to_string(tokens) + " tokens and the prefill instance reads at "
              "most " + std::to_string(cap) + " (--prefill-max-tokens; the arena ceiling is " +
              std::to_string(arena_max_tokens) + ")";
        return true;
    }
    return false;
}
/// §3.8's tier-starvation rule as a predicate the prefill instance can assert: a job of tier `i` may
/// claim a free slot in group `j` for any `j >= i`, preferring the SMALLEST such `j`.
inline bool tier_may_use(uint32_t job_tier, uint32_t slot_group) { return slot_group >= job_tier; }

// ------------------------------------------------------------------ the handoff state machine ---

/// The §4.5 states, per request, on the DECODE instance.  These are NOT the engine's `Phase` values:
/// they live in the handoff client thread, and `Phase::await_prefill` is the one `Phase` that means
/// "one of these is in progress".  `H_MOUNTING` is marked `[engine]` and is driven by the engine
/// thread through `mount_result()`, never by `on_line()`.  See deviation D1 for why there are 13.
enum class HState : uint8_t {
    idle,          ///< no handoff in progress
    queued,        ///< S4.2's WaitQueue head; nothing has been sent (D3)
    init,          ///< admitted, the local cache missed, no REQ on the wire yet
    sent,          ///< REQ sent, no answer yet                                  (§4.5 H_REQ_SENT)
    held,          ///< a temporary ERR parked us; the backoff timer owns the re-ask
    accepted,      ///< QUEUED received: accepted, no slot claimed              (§4.5 H_ACCEPTED)
    prefilling,    ///< CLAIM received: a slot exists, the read is running      (§4.5 H_PREFILLING)
    seg_ready,     ///< DONE received: the slot is READY, the copy-out runs HERE
    handed_over,   ///< ACK sent: the slot is FREE again, the image is ours     (§4.5 H_HANDED_OVER)
    mounting,      ///< [engine] validate -> unmount -> mount -> draft_kv -> adopt -> device_state
    done,          ///< the handoff finished and the mount succeeded
    failed,        ///< a permanent ERR; the caller collects the verdict
    cancelled,     ///< cancelled; waiting for the slot release to be confirmed
};

inline const char* state_name(HState s) {
    switch (s) {
        case HState::idle: return "IDLE";
        case HState::queued: return "H_QUEUED";
        case HState::init: return "H_INIT";
        case HState::sent: return "H_REQ_SENT";
        case HState::held: return "H_HOLD";
        case HState::accepted: return "H_ACCEPTED";
        case HState::prefilling: return "H_PREFILLING";
        case HState::seg_ready: return "H_SEG_READY";
        case HState::handed_over: return "H_HANDED_OVER";
        case HState::mounting: return "MOUNTING";
        case HState::done: return "H_DONE";
        case HState::failed: return "H_FAILED";
        case HState::cancelled: return "H_CANCELLED";
    }
    return "?";
}
/// Is this a state the handoff thread may still be working in?
inline bool state_is_live(HState s) {
    return s == HState::init || s == HState::sent || s == HState::held || s == HState::accepted ||
           s == HState::prefilling || s == HState::seg_ready || s == HState::handed_over ||
           s == HState::mounting;
}
/// Does this state still hold a handoff slot?  Only `H_SEG_READY` does: `CLAIM` took it and `ACK`
/// gave it back.  Everything after `ACK` owns a heap `SavedConversation`, not a slot.  This is the
/// predicate the slot-leak diagnostics hang on.
inline bool state_holds_slot(HState s) { return s == HState::seg_ready; }

/// The events the machine consumes.  The table is indexed by (state, event) and is TOTAL:
/// `handoff_table()` returns exactly `kStateCount * kEventCount` rows, one per pair, and the test
/// asserts that number.
///
/// `ERR` is THREE events, not one (D11), because §4.4's table says an ERR means three different
/// things and the difference is the code's CLASS, not the verb: a temporary code is a hold, a
/// permanent code is an error, and `cancel` is neither.  `err_event_for()` picks the variant.
enum class Event : uint8_t {
    admit,             ///< the WaitQueue let it through (H_QUEUED -> H_INIT)
    req_sent,          ///< the caller wrote REQ
    queued_line, claim_line, seg_line, done_line,
    err_temporary,     ///< ERR full/queue/read/slotlost/shutting/endpoint
    err_permanent,     ///< ERR toobig/ids/novision/resume/geom/pack/badpayload
    err_cancel,        ///< ERR cancel
    cancel_line,       ///< the caller wrote CANCEL (never received on this side)
    ack_sent,          ///< the caller wrote ACK
    copy_done,         ///< the copy-out and §2.5 checks 1-2 passed
    mount_ok, mount_fail,
    collect,           ///< the caller took the verdict out of H_FAILED / H_CANCELLED / H_DONE
    retry,             ///< a --prefill-retries re-ask
    note_cancel,       ///< the client cancelled; the machine decides what the peer is sent
    local_hit,         ///< the local ConversationCache::best() hit: NO handoff at all
    last_event,
};
inline const char* event_name(Event e) {
    switch (e) {
        case Event::admit: return "admit";
        case Event::req_sent: return "req_sent";
        case Event::queued_line: return "QUEUED";
        case Event::claim_line: return "CLAIM";
        case Event::seg_line: return "SEG";
        case Event::done_line: return "DONE";
        case Event::err_temporary: return "ERR-temporary";
        case Event::err_permanent: return "ERR-permanent";
        case Event::err_cancel: return "ERR-cancel";
        case Event::cancel_line: return "CANCEL";
        case Event::ack_sent: return "ack_sent";
        case Event::copy_done: return "copy_done";
        case Event::mount_ok: return "mount_ok";
        case Event::mount_fail: return "mount_fail";
        case Event::collect: return "collect";
        case Event::retry: return "retry";
        case Event::note_cancel: return "note_cancel";
        case Event::local_hit: return "local_hit";
        case Event::last_event: return "?";
    }
    return "?";
}
/// Which ERR event a code means.  `cancel` is checked first because §4.4 marks it "n/a" - it is not a
/// failure at all, and classifying it as temporary would put a cancelled request back in the queue.
inline Event err_event_for(ErrCode c) {
    if (c == ErrCode::cancel) return Event::err_cancel;
    return err_is_temporary(c) ? Event::err_temporary : Event::err_permanent;
}

inline constexpr int kStateCount = (int) HState::cancelled + 1;
inline constexpr int kEventCount = (int) Event::last_event;

/// What the caller must DO after an event.  One enum, so a switch at the call site cannot miss a
/// case, and so the whole machine is enumerable.
enum class Act : uint8_t {
    none,            ///< nothing to write, nothing to wait for
    write_req,       ///< send REQ (and only one at a time - §6.2)
    write_cancel,    ///< send CANCEL
    write_ack,       ///< send ACK: the ONLY release verb
    copy_out,        ///< map -> read -> rebuild -> §2.5 checks 1-2 (never on the engine thread)
    mount,           ///< post the image to the engine thread as mount_image
    hold,            ///< enter S4.2's wait queue with reason `prefill-busy`
    err_client,      ///< answer the client ERR (a permanent code only - D5)
    done_cancel,     ///< answer the client `DONE … cancel`
    client_gone,     ///< the request already produced its answer: an ordinary STOP, stage 3's business
    finish,          ///< the handoff is over
    protocol_error,  ///< §4.5 rule 4: close the connection and log it
    illegal,         ///< this event in this state: refused, `why` says why
};
inline const char* action_name(Act a) {
    switch (a) {
        case Act::none: return "none";
        case Act::write_req: return "write-req";
        case Act::write_cancel: return "write-cancel";
        case Act::write_ack: return "write-ack";
        case Act::copy_out: return "copy-out";
        case Act::mount: return "mount";
        case Act::hold: return "hold";
        case Act::err_client: return "err-client";
        case Act::done_cancel: return "done-cancel";
        case Act::client_gone: return "client-gone";
        case Act::finish: return "finish";
        case Act::protocol_error: return "protocol-error";
        case Act::illegal: return "illegal";
    }
    return "?";
}

/// One table entry.
struct Transition {
    HState from = HState::idle;
    Event event = Event::admit;
    HState to = HState::idle;
    Act action = Act::illegal;
    const char* why = "";
};

/// The whole machine, as data, defined out of line at the bottom of this header.  Total by
/// construction: one row per (state, event).
inline const Transition* handoff_table(int& count);

/// The one-cell answer for (state, event).  Never returns null and never falls through: a pair that
/// is not in the table is a bug in the table, and `lookup` says so rather than guessing.
inline Transition lookup(HState s, Event e) {
    int n = 0;
    const Transition* t = handoff_table(n);
    for (int i = 0; i < n; ++i)
        if (t[i].from == s && t[i].event == e) return t[i];
    return {s, e, s, Act::illegal, "MISSING FROM THE TABLE: this (state, event) pair has no row"};
}

// ------------------------------------------------------------------ one request's machine -----

/// Defined below with the log lines; `reset_for_retry` names it so the bound's message is ONE string
/// the test can pin (§4.7).
inline std::string retry_limit_line(int64_t id, int64_t attempts, int64_t retries, ErrCode last);

/// The per-request handoff machine.  Pure: it holds ids, nonces, counters and a backoff schedule, and
/// it decides what the caller must write.  It never touches a socket, an arena or a device, and it
/// has no clock of its own (the same rule as `serve_driver.hpp`: the caller passes `now_ms`).
class HandoffState {
  public:
    HandoffState() = default;
    explicit HandoffState(int64_t retries) : retries_(retries < 0 ? 0 : retries) {}

    int64_t retries() const { return retries_; }
    void set_retries(int64_t n) { retries_ = n < 0 ? 0 : n; }

    HState state() const { return st_; }
    int64_t id() const { return id_; }
    uint64_t nonce() const { return nonce_; }
    int64_t slot() const { return slot_; }
    uint64_t tokens() const { return tokens_; }
    uint64_t tier() const { return tier_; }
    uint64_t bytes() const { return bytes_; }
    int64_t req_lines() const { return req_lines_; }     ///< REQ lines written
    int64_t handoffs() const { return handoffs_; }       ///< handoffs that reached DONE
    bool cancelled() const { return cancel_; }
    bool poisoned() const { return poison_; }            ///< a mount that failed after its first write
    ErrCode verdict() const { return verdict_; }
    const std::string& verdict_text() const { return verdict_text_; }
    bool has_outstanding_req() const { return outstanding_; }
    int64_t backoff_ms() const { return backoff_ms_; }
    /// §4.5: "one outstanding REQ per request" and "a QUEUED/ERR full answer sets a backoff (start
    /// 50 ms, cap 2 s) so a full arena is not hammered".
    bool may_send_req(int64_t now_ms) const { return !outstanding_ && now_ms >= next_ask_ms_; }
    void note_now_ms(int64_t now_ms) { now_ms_hint_ = now_ms; }

    /// The local `ConversationCache::best()` hit path (§1.5): NO handoff at all, today's mount.
    bool local_hit(int64_t id) {
        if (!startable()) return false;
        reset_all(id);
        st_ = lookup(HState::idle, Event::local_hit).to;
        return true;
    }
    /// Admitted: the request is in S4.2's queue head and nothing has been sent.
    bool admit_now(int64_t id) {
        if (!startable()) return false;
        reset_all(id);
        st_ = HState::queued;
        return true;
    }
    /// Admitted and the local cache missed: the handoff is about to start.
    bool begin(int64_t id, uint64_t tokens, uint64_t tier, std::string& why) {
        if (!startable()) {
            why = std::string("cannot start a handoff from ") + state_name(st_);
            return false;
        }
        reset_all(id);
        tokens_ = tokens;
        tier_ = tier;
        st_ = HState::init;
        return true;
    }

    struct StepResult {
        bool ok = false;
        HState from = HState::idle;
        HState to = HState::idle;
        Act action = Act::illegal;
        std::string why;
        ErrCode code = ErrCode::none;
    };

    /// Feed one line from the peer.  The line must already have parsed and must name this request.
    /// A line about a different id, or a stale nonce, is refused with a named reason and the state
    /// does not move - that is §4.5 rule 5, and it is what stops a late `DONE` for a recycled slot
    /// from mounting somebody else's payload.
    StepResult on_line(int64_t id, const Line& l) {
        StepResult r;
        r.from = st_;
        if (!l.bad.empty()) { r.action = Act::protocol_error; r.why = l.bad; return r; }
        if (!line_from_peer(l.kind, Side::decode_to_prefill)) {
            r.action = Act::protocol_error;
            r.why = std::string(line_name(l.kind)) + " is not a line a prefill instance sends";
            return r;
        }
        if (l.id != id) {
            r.action = Act::illegal;
            r.why = std::string(line_name(l.kind)) + " names request " + std::to_string((long long) l.id) +
                    " and this machine is request " + std::to_string((long long) id);
            return r;
        }
        Event e;
        switch (l.kind) {
            case LineKind::queued: e = Event::queued_line; break;
            case LineKind::claim:  e = Event::claim_line; break;
            case LineKind::seg:    e = Event::seg_line; break;
            case LineKind::done:   e = Event::done_line; break;
            case LineKind::err:    e = err_event_for(l.code); break;
            default:
                r.action = Act::illegal;
                r.why = std::string(line_name(l.kind)) + " does not drive the handoff machine";
                return r;
        }
        // ---- the nonce rule, BEFORE the table (§4.5 rule 5) ------------------------------------
        // CLAIM sets it; DONE must match it.  SEG carries NO nonce field in §4.3's grammar, so it
        // cannot be checked at line level - what pins a SEG to this attempt is that it arrives while
        // the machine is in H_PREFILLING for this id, and the payload it precedes is still gated by
        // the DONE nonce and then by the slot header's nonce inside `read_slot`.
        if (e == Event::claim_line) {
            if (nonce_ != 0 && l.nonce != nonce_) {
                r.action = Act::protocol_error;
                r.why = "CLAIM carries a second nonce for one request (" + std::to_string(nonce_) +
                        " then " + std::to_string(l.nonce) + ")";
                return r;
            }
        } else if (e == Event::done_line && nonce_ != 0 && l.nonce != nonce_) {
            r.action = Act::protocol_error;
            r.why = std::string(line_name(l.kind)) + " nonce " + std::to_string(l.nonce) +
                    " does not match this handoff's nonce " + std::to_string(nonce_) +
                    " - a stale line about a slot that has already been reused";
            return r;
        }
        if (e == Event::claim_line && nonce_ == 0) nonce_ = l.nonce;
        if (e == Event::claim_line && slot_ < 0) slot_ = (int64_t) l.slot;
        if (e == Event::done_line && l.nonce == nonce_) { bytes_ = l.bytes; tokens_ = l.tokens; }
        if (e == Event::err_temporary || e == Event::err_permanent || e == Event::err_cancel) {
            if (verdict_ == ErrCode::none) verdict_ = l.code;
            verdict_text_ = l.message.empty() ? err_text(l.code) : l.message;
        }
        return step(id, e, l.code);
    }

    /// The caller's own moves: REQ written, ACK written, copy-out finished, cancel, collect, retry.
    StepResult note(int64_t id, Event e) { return step(id, e, ErrCode::none); }

    /// §4.6: cancellation is legal in EVERY state.  This is the entry point; the table decides what
    /// it means (write CANCEL, write ACK, or just end).
    StepResult note_cancel(int64_t id) { return step(id, Event::note_cancel, ErrCode::none); }

    /// The engine thread's entry point (D2).  `after_first_write` marks the case that is fatal to the
    /// session: 0.1.30's rule, and this stage does not soften it.  When it is true the machine does
    /// NOT offer a recovery - it sets `poisoned()` and the caller must exit 1.
    StepResult mount_result(int64_t id, bool ok, bool after_first_write) {
        if (!ok && after_first_write) {
            StepResult r;
            r.from = st_;
            r.to = st_;
            poison_ = true;
            verdict_ = ErrCode::badpayload;
            verdict_text_ = "the restore failed after its first write; the session is poisoned";
            r.action = Act::err_client;
            r.why = "the session is poisoned: return 1, do not attempt a recovery";
            r.code = ErrCode::badpayload;
            r.ok = true;
            return r;
        }
        return step(id, ok ? Event::mount_ok : Event::mount_fail, ErrCode::badpayload);
    }

    /// What the client gets when the handoff ended because the request was cancelled (D4).  The
    /// client line itself is `serve_proto`'s and is NOT formatted here.
    enum class CancelOutcome : uint8_t { none, client_done_cancel, client_gone };
    CancelOutcome cancel_outcome() const {
        if (!cancel_) return CancelOutcome::none;
        if (st_ == HState::done || st_ == HState::mounting || st_ == HState::handed_over)
            return CancelOutcome::client_gone;
        return CancelOutcome::client_done_cancel;
    }

    /// The `ACK` line for the slot this machine holds, or empty when it holds none.  The nonce is the
    /// one `CLAIM` gave, which is the only nonce that can release this slot (§4.5 rule 5).
    std::string ack_line() const {
        if (nonce_ == 0) return {};
        return prefill_svc::ack_line(id_, nonce_);
    }
    /// A stale nonce cannot release another request's slot.  This is the predicate S4.3.6 checks
    /// before it calls `HandoffArena::release(index, nonce)`.
    bool may_release(int64_t slot, uint64_t nonce) const {
        return nonce_ != 0 && nonce == nonce_ && slot_ == slot;
    }
    /// The same rule for the read: `read_slot(index, nonce, client_id, ...)` must be called with the
    /// nonce THIS handoff was given, never with a nonce from an earlier attempt.
    bool may_read(int64_t slot, uint64_t nonce) const { return may_release(slot, nonce); }

    /// The retry bound (§4.7, D13).  `true` means "ask again"; `false` means the request is over and
    /// the caller must answer the client.  This is the row stage 3's history says will be missed: a
    /// handoff loop is invisible to the watchdog because every pass really does move gigabytes.
    bool retry_allowed() const { return req_lines_ <= retries_; }

    /// Reset for a re-ask after a retryable failure.  False when the bound has been reached, which is
    /// the fourth attempt ending the request with a named ERR instead of spinning forever.
    bool reset_for_retry(std::string& why) {
        if (!retry_allowed()) {
            why = retry_limit_line(id_, req_lines_, retries_, verdict_);
            return false;
        }
        st_ = HState::init;
        nonce_ = 0;
        slot_ = -1;
        outstanding_ = false;
        verdict_ = ErrCode::none;
        verdict_text_.clear();
        return true;
    }

  private:
    bool startable() const {
        return st_ == HState::idle || st_ == HState::queued || st_ == HState::done ||
               st_ == HState::failed || st_ == HState::cancelled;
    }
    void reset_all(int64_t id) {
        id_ = id;
        st_ = HState::idle;
        nonce_ = 0;
        slot_ = -1;
        tokens_ = 0;
        tier_ = 0;
        bytes_ = 0;
        req_lines_ = 0;
        handoffs_ = 0;
        cancel_ = false;
        poison_ = false;
        outstanding_ = false;
        backoff_ms_ = 0;
        next_ask_ms_ = 0;
        verdict_ = ErrCode::none;
        verdict_text_.clear();
    }

    StepResult step(int64_t id, Event e, ErrCode code) {
        StepResult r;
        r.from = st_;
        r.code = code;
        if (id != id_) {
            r.action = Act::illegal;
            r.to = st_;
            r.why = "the machine is for request " + std::to_string((long long) id_) +
                    ", not " + std::to_string((long long) id);
            return r;
        }
        const Transition t = lookup(st_, e);
        r.to = t.to;
        r.action = t.action;
        r.why = t.why;
        r.ok = t.action != Act::illegal;
        if (t.action == Act::illegal) return r;

        // ---- the side effects the action names, applied here and nowhere else -------------------
        switch (e) {
            case Event::req_sent:
                outstanding_ = true;
                ++req_lines_;
                next_ask_ms_ = 0;
                break;
            case Event::queued_line:
                outstanding_ = false;
                break;
            case Event::done_line:
                outstanding_ = false;
                ++handoffs_;
                break;
            case Event::err_temporary:
                outstanding_ = false;
                // §4.5: "bounded backoff 50 ms -> 2 s, one outstanding REQ".
                backoff_ms_ = backoff_ms_ == 0 ? kBackoffStartMs
                              : (backoff_ms_ * 2 > kBackoffCapMs ? kBackoffCapMs : backoff_ms_ * 2);
                next_ask_ms_ = now_ms_hint_ + backoff_ms_;
                break;
            case Event::err_permanent:
                outstanding_ = false;
                if (verdict_ == ErrCode::none) verdict_ = code;
                break;
            case Event::err_cancel:
                outstanding_ = false;
                cancel_ = true;
                break;
            case Event::note_cancel:
                cancel_ = true;
                break;
            case Event::ack_sent:
                outstanding_ = false;
                break;
            case Event::retry:
                if (!reset_for_retry(r.why)) {
                    r.action = Act::illegal;
                    r.to = st_;
                    r.ok = false;
                    return r;
                }
                break;
            default:
                break;
        }
        st_ = t.to;
        return r;
    }

    int64_t id_ = -1;
    HState st_ = HState::idle;
    uint64_t nonce_ = 0;
    int64_t slot_ = -1;
    uint64_t tokens_ = 0, tier_ = 0, bytes_ = 0;
    int64_t req_lines_ = 0;
    int64_t handoffs_ = 0;
    int64_t retries_ = kDefaultPrefillRetries;
    int64_t backoff_ms_ = 0;
    int64_t next_ask_ms_ = 0;
    int64_t now_ms_hint_ = 0;
    bool cancel_ = false;
    bool poison_ = false;
    bool outstanding_ = false;
    ErrCode verdict_ = ErrCode::none;
    std::string verdict_text_;
};

// ------------------------------------------------------------------ the prefill-side job machine

/// The prefill instance's view of one job (§5.2).  The mirror of `HandoffState`, and it exists so
/// S4.3.5's socket loop cannot invent its own if-chain: §4.6's cancellation rules and §5.2's "one job
/// at a time" are table entries here.
enum class PJob : uint8_t {
    queued,      ///< accepted, no slot claimed
    claimed,     ///< a slot is claimed, the read has not started
    reading,     ///< Prefill::run is running
    writing,     ///< the payload write started (SEG emitted)
    published,   ///< DONE emitted, waiting for ACK
    released,    ///< ACK received: the slot is FREE
    cancelling,  ///< CANCEL arrived while work was in flight; the unwind owns the slot
    cancelled,   ///< the unwind finished
    failed,      ///< an ERR was emitted and the slot is back
};
inline const char* job_name(PJob j) {
    switch (j) {
        case PJob::queued: return "QUEUED";
        case PJob::claimed: return "CLAIMED";
        case PJob::reading: return "READING";
        case PJob::writing: return "WRITING";
        case PJob::published: return "PUBLISHED";
        case PJob::released: return "RELEASED";
        case PJob::cancelling: return "CANCELLING";
        case PJob::cancelled: return "CANCELLED";
        case PJob::failed: return "FAILED";
    }
    return "?";
}
enum class PEvent : uint8_t {
    claim_ok,      ///< HandoffArena::claim() succeeded
    read_start,    ///< Prefill::run is about to be called
    seg_sent,      ///< the read finished and the payload write started
    publish_ok,    ///< arena.publish() returned true
    ack,           ///< ACK received
    cancel,        ///< CANCEL received
    read_fail,     ///< the read or the payload write failed
    publish_lost,  ///< publish() refused: the claim was reclaimed under us (§4.7 slotlost)
    abort_done,    ///< abort_claim() finished: the slot is FREE again
    last_event,
};
inline const char* job_event_name(PEvent e) {
    switch (e) {
        case PEvent::claim_ok: return "claim_ok";
        case PEvent::read_start: return "read_start";
        case PEvent::seg_sent: return "seg_sent";
        case PEvent::publish_ok: return "publish_ok";
        case PEvent::ack: return "ack";
        case PEvent::cancel: return "cancel";
        case PEvent::read_fail: return "read_fail";
        case PEvent::publish_lost: return "publish_lost";
        case PEvent::abort_done: return "abort_done";
        case PEvent::last_event: return "?";
    }
    return "?";
}
enum class PAct : uint8_t {
    none,
    send_queued,       ///< QUEUED <id> <ahead>
    send_claim,        ///< CLAIM <id> <slot> <nonce>
    send_seg,          ///< SEG ...
    send_done,         ///< DONE ...
    err_read,          ///< ERR <id> read ...
    err_slotlost,      ///< ERR <id> slotlost ...
    err_cancel,        ///< ERR <id> cancel ...
    release_slot,      ///< arena.release(index, nonce) - the ACK path
    abort_claim,       ///< arena.abort_claim(index, nonce) - release WITHOUT publishing
    abort_err_cancel,  ///< abort_claim, then ERR cancel
    abort_err_read,    ///< abort_claim, then ERR read
    illegal,
};
inline const char* job_action_name(PAct a) {
    switch (a) {
        case PAct::none: return "none";
        case PAct::send_queued: return "send-QUEUED";
        case PAct::send_claim: return "send-CLAIM";
        case PAct::send_seg: return "send-SEG";
        case PAct::send_done: return "send-DONE";
        case PAct::err_read: return "send-ERR-read";
        case PAct::err_slotlost: return "send-ERR-slotlost";
        case PAct::err_cancel: return "send-ERR-cancel";
        case PAct::release_slot: return "release-slot";
        case PAct::abort_claim: return "abort-claim";
        case PAct::abort_err_cancel: return "abort-claim+ERR-cancel";
        case PAct::abort_err_read: return "abort-claim+ERR-read";
        case PAct::illegal: return "illegal";
    }
    return "?";
}
/// Which error code an action names, or `none`.  Lets the test assert that a cancelled job answers
/// `cancel` and never `read`, and that a lost slot answers `slotlost` and never `read`.
inline ErrCode job_action_code(PAct a) {
    switch (a) {
        case PAct::err_read: case PAct::abort_err_read: return ErrCode::read;
        case PAct::err_slotlost: return ErrCode::slotlost;
        case PAct::err_cancel: case PAct::abort_err_cancel: return ErrCode::cancel;
        default: return ErrCode::none;
    }
}
struct JobTransition {
    PJob from = PJob::queued;
    PEvent event = PEvent::claim_ok;
    PJob to = PJob::queued;
    PAct action = PAct::illegal;
    const char* why = "";
};
inline const JobTransition* prefill_job_table(int& count);

inline JobTransition job_lookup(PJob j, PEvent e) {
    int n = 0;
    const JobTransition* t = prefill_job_table(n);
    for (int i = 0; i < n; ++i)
        if (t[i].from == j && t[i].event == e) return t[i];
    return {j, e, j, PAct::illegal, "MISSING FROM THE TABLE: this (job state, event) pair has no row"};
}
inline constexpr int kJobStateCount = (int) PJob::failed + 1;
inline constexpr int kJobEventCount = (int) PEvent::last_event;

/// §4.5 rule 3: a `DONE` for a request the decode instance already cancelled is answered `ACK`
/// immediately and the slot is released WITHOUT being read.  The prefill instance must treat that as
/// a normal release, not a protocol error.
inline bool cancel_after_done_is_normal(PJob j) { return j == PJob::published; }

// ------------------------------------------------------------------ the startup / log lines -----

/// `strata prefill: serving on <sock>; <arena describe>` (§4.2).
inline std::string prefill_serving_line(const std::string& socket_path, const std::string& arena_desc) {
    return "strata prefill: serving on " + socket_path + "; " + arena_desc;
}
/// `strata serve: prefill endpoint <path>: PREFILL-READY-V1 (slots 2,2,2, max 131072 tokens); handoff on`
inline std::string endpoint_ready_line(const std::string& path, const ReadyInfo& r) {
    return "strata serve: prefill endpoint " + path + ": " + kReadyToken + " (slots " +
           detail::join_list(r.slots) + ", max " + std::to_string(r.max_tokens) + " tokens); handoff on";
}
/// §4.7: the prefill instance died.  The decode instance logs this and holds (or reads locally).
inline std::string endpoint_down_line(const std::string& why) {
    return "strata serve: the prefill instance died; holding new prompts" +
           (why.empty() ? std::string() : (" (" + why + ")"));
}
/// §4.2: the endpoint is not a prefill instance at all.
inline std::string not_prefill_line() {
    return std::string("strata serve: the prefill endpoint does not speak ") + kReadyToken;
}
/// §3.10: a hash mismatch, naming BOTH values, before any byte moves.
inline std::string hash_refusal_line(const std::string& which, uint64_t mine, uint64_t theirs,
                                     const std::string& note) {
    return "strata serve: handoff refused: " + which + " hash " + detail::hex64(mine) + " != " +
           detail::hex64(theirs) + (note.empty() ? std::string() : (" (" + note + ")"));
}
/// §4.7: the retry bound fired.  Named so the test pins the wording and the owner can grep it.
inline std::string retry_limit_line(int64_t id, int64_t attempts, int64_t retries, ErrCode last) {
    return "request " + std::to_string((long long) id) + " re-handoffed " +
           std::to_string((long long) attempts) + " times and never finished (last answer: " +
           err_name(last) + "); the bound is --prefill-retries " + std::to_string((long long) retries) +
           ". Raise --prefill-slots, lower --prefill-slot-tiers, or give /dev/shm more room.";
}
/// §4.3: the release the prefill instance records on ITS stderr (there is no `ACK-ED` echo).
inline std::string slot_released_line(int64_t slot, uint64_t client_id, uint64_t bytes,
                                      int64_t held_ms) {
    return "strata prefill: slot " + std::to_string((long long) slot) + " released by client " +
           detail::hex64(client_id) + ", " + std::to_string((long long) bytes) + " B, held " +
           std::to_string((long long) held_ms) + " ms";
}
/// §4.3: the periodic activity line.
inline std::string activity_line(int64_t jobs, int64_t done, int64_t err, int64_t queued,
                                 const std::string& slots_free, int64_t reused_tokens,
                                 int64_t read_tokens, int64_t handoff_ms, int64_t arena_mib,
                                 int64_t arena_total_mib) {
    return "strata prefill: activity: jobs=" + std::to_string((long long) jobs) +
           " done=" + std::to_string((long long) done) + " err=" + std::to_string((long long) err) +
           " queued=" + std::to_string((long long) queued) + " slots_free=" + slots_free +
           " reused=" + std::to_string((long long) reused_tokens) + " tokens read=" +
           std::to_string((long long) read_tokens) + " tokens handoff_ms=" +
           std::to_string((long long) handoff_ms) + " arena=" + std::to_string((long long) arena_mib) +
           "MiB/" + std::to_string((long long) arena_total_mib) + "MiB";
}
/// §4.3: the `ERR` the decode instance answers the CLIENT with, for a permanent code.  Kept as a
/// helper so the wording is one place; the line itself is still `serve_proto`'s to format.
inline std::string client_err_text(ErrCode c, const std::string& detail_note) {
    return err_text(c) + (detail_note.empty() ? std::string() : (": " + detail_note));
}
/// §6.1: the sentence a held request carries.  S4.3.6 passes this to `WaitQueue` as
/// `wait_text(Wait::prefill)` until the enum value lands.
inline std::string hold_line_prefill(int64_t id, int64_t queue_depth) {
    return "strata serve: slot " + std::to_string((long long) id) + " waits: " + kWaitTextPrefill +
           " (waiting=" + std::to_string((long long) queue_depth) + ")";
}

// ------------------------------------------------------------------ the INFO keys (§4.3) ------

inline std::vector<std::pair<std::string, std::string>> default_info_keys() {
    return {{"role", "prefill"}, {"slots", "0"}, {"slots_free", "0"}, {"jobs", "0"},
            {"jobs_done", "0"}, {"jobs_err", "0"}, {"reused_tokens", "0"}, {"read_tokens", "0"},
            {"arena_mib", "0"}, {"max_ctx", "0"}, {"pack", "0000000000000000"},
            {"geom", "0000000000000000"}};
}
/// Look one INFO key up.  Empty when the peer did not send it.
inline std::string info_value(const Line& l, const std::string& key) {
    for (const auto& e : l.info) if (e.first == key) return e.second;
    return {};
}
/// §6.2 wake point (c): a `DONE` for it, or an `INFO`/activity line showing a free slot.  This is the
/// predicate the decode instance uses to wake a `prefill-busy` waiter early instead of waiting for
/// the backoff.
inline bool info_shows_a_free_slot(const Line& l) {
    if (l.kind != LineKind::info) return false;
    const std::string free = info_value(l, "slots_free");
    if (free.empty()) return false;
    // `HandoffArena::free_slots()` spells it "4/6"; a bare "4" is accepted too.
    const size_t slash = free.find('/');
    const std::string head = slash == std::string::npos ? free : free.substr(0, slash);
    uint64_t v = 0;
    return detail::is_uint(head, v) && v > 0;
}

}  // namespace strata::program::prefill_svc

// ==================================================================================================
// THE TABLES, out of line so the class above stays readable.  Both are `inline` and both are
// generated as complete cross-products, so the header stays the only file (no .cpp, no link step) and
// "the machine is total" is a property of the table's size, not of how carefully it was written.
// ==================================================================================================

namespace strata::program::prefill_svc {

inline const Transition* handoff_table(int& count) {
    // Exactly kStateCount * kEventCount rows: one per (state, event).  The test asserts the
    // product, so "total" is a checked fact and not a claim in a comment.
    static const Transition t[] = {
        // ---- IDLE ----
        {HState::idle, Event::admit, HState::queued, Act::none,
            "admitted; the local cache has not been consulted yet"},
        {HState::idle, Event::req_sent, HState::idle, Act::illegal,
            "begin() first: there is no request to ask about"},
        {HState::idle, Event::queued_line, HState::idle, Act::illegal,
            "no handoff is in progress"},
        {HState::idle, Event::claim_line, HState::idle, Act::illegal,
            "no handoff is in progress"},
        {HState::idle, Event::seg_line, HState::idle, Act::protocol_error,
            "SEG with no CLAIM ever sent: S4.5 rule 4"},
        {HState::idle, Event::done_line, HState::idle, Act::protocol_error,
            "DONE with no CLAIM ever sent: S4.5 rule 4, close the connection"},
        {HState::idle, Event::err_temporary, HState::idle, Act::illegal,
            "no handoff is in progress"},
        {HState::idle, Event::err_permanent, HState::idle, Act::illegal,
            "no handoff is in progress"},
        {HState::idle, Event::err_cancel, HState::idle, Act::illegal,
            "no handoff is in progress"},
        {HState::idle, Event::cancel_line, HState::idle, Act::illegal,
            "CANCEL is a line the decode instance sends, not one it receives"},
        {HState::idle, Event::ack_sent, HState::idle, Act::illegal,
            "ACK with no slot held"},
        {HState::idle, Event::copy_done, HState::idle, Act::illegal,
            "there is nothing to copy out"},
        {HState::idle, Event::mount_ok, HState::idle, Act::illegal,
            "no handoff is in progress"},
        {HState::idle, Event::mount_fail, HState::idle, Act::illegal,
            "no handoff is in progress"},
        {HState::idle, Event::collect, HState::idle, Act::illegal,
            "no handoff is in progress"},
        {HState::idle, Event::retry, HState::idle, Act::illegal,
            "nothing has been asked, so there is nothing to retry"},
        {HState::idle, Event::note_cancel, HState::idle, Act::done_cancel,
            "cancelled before anything was asked: the client gets DONE ... cancel"},
        {HState::idle, Event::local_hit, HState::handed_over, Act::mount,
            "the local ConversationCache::best() hit: NO handoff at all (S1.5)"},
        // ---- QUEUED ----
        {HState::queued, Event::admit, HState::init, Act::none,
            "the wait queue let it through and the local cache missed"},
        {HState::queued, Event::req_sent, HState::queued, Act::illegal,
            "begin() first: a queued request has no REQ to write yet"},
        {HState::queued, Event::queued_line, HState::queued, Act::illegal,
            "the handoff thread must not touch a queued request"},
        {HState::queued, Event::claim_line, HState::queued, Act::illegal,
            "the handoff thread must not touch a queued request"},
        {HState::queued, Event::seg_line, HState::queued, Act::protocol_error,
            "SEG with no CLAIM: S4.5 rule 4"},
        {HState::queued, Event::done_line, HState::queued, Act::protocol_error,
            "DONE with no CLAIM: S4.5 rule 4"},
        {HState::queued, Event::err_temporary, HState::queued, Act::illegal,
            "the handoff thread must not touch a queued request"},
        {HState::queued, Event::err_permanent, HState::queued, Act::illegal,
            "the handoff thread must not touch a queued request"},
        {HState::queued, Event::err_cancel, HState::queued, Act::illegal,
            "the handoff thread must not touch a queued request"},
        {HState::queued, Event::cancel_line, HState::queued, Act::illegal,
            "CANCEL is not a line this side receives"},
        {HState::queued, Event::ack_sent, HState::queued, Act::illegal,
            "no slot held"},
        {HState::queued, Event::copy_done, HState::queued, Act::illegal,
            "nothing to copy"},
        {HState::queued, Event::mount_ok, HState::queued, Act::illegal,
            "the handoff thread must not touch a queued request"},
        {HState::queued, Event::mount_fail, HState::queued, Act::illegal,
            "the handoff thread must not touch a queued request"},
        {HState::queued, Event::collect, HState::queued, Act::illegal,
            "nothing to collect"},
        {HState::queued, Event::retry, HState::queued, Act::illegal,
            "nothing has been asked, so there is nothing to retry"},
        {HState::queued, Event::note_cancel, HState::idle, Act::done_cancel,
            "cancelled while queued: nothing was touched"},
        {HState::queued, Event::local_hit, HState::handed_over, Act::mount,
            "the local cache hit while it waited: no handoff"},
        // ---- INIT ----
        {HState::init, Event::admit, HState::init, Act::illegal,
            "the handoff has not been asked yet"},
        {HState::init, Event::req_sent, HState::sent, Act::write_req,
            "one outstanding REQ per request (S6.2)"},
        {HState::init, Event::queued_line, HState::accepted, Act::none,
            "accepted, no slot claimed yet"},
        {HState::init, Event::claim_line, HState::prefilling, Act::none,
            "CLAIM without QUEUED first is legal: the slot was free immediately"},
        {HState::init, Event::seg_line, HState::idle, Act::protocol_error,
            "SEG before CLAIM: S4.5 rule 4"},
        {HState::init, Event::done_line, HState::idle, Act::protocol_error,
            "DONE with no CLAIM: S4.5 rule 4"},
        {HState::init, Event::err_temporary, HState::init, Act::illegal,
            "ERR with no REQ in flight"},
        {HState::init, Event::err_permanent, HState::init, Act::illegal,
            "ERR with no REQ in flight"},
        {HState::init, Event::err_cancel, HState::init, Act::illegal,
            "ERR cancel with no REQ in flight"},
        {HState::init, Event::cancel_line, HState::init, Act::illegal,
            "CANCEL is not a line this side receives"},
        {HState::init, Event::ack_sent, HState::init, Act::illegal,
            "no slot held"},
        {HState::init, Event::copy_done, HState::init, Act::illegal,
            "nothing to copy"},
        {HState::init, Event::mount_ok, HState::init, Act::illegal,
            "the handoff has not been asked yet"},
        {HState::init, Event::mount_fail, HState::init, Act::illegal,
            "the handoff has not been asked yet"},
        {HState::init, Event::collect, HState::init, Act::illegal,
            "the handoff has not been asked yet"},
        {HState::init, Event::retry, HState::init, Act::none,
            "already ready to ask again"},
        {HState::init, Event::note_cancel, HState::idle, Act::done_cancel,
            "cancelled before REQ went out"},
        {HState::init, Event::local_hit, HState::handed_over, Act::mount,
            "the local cache hit after all"},
        // ---- SENT ----
        {HState::sent, Event::admit, HState::sent, Act::illegal,
            "already admitted"},
        {HState::sent, Event::req_sent, HState::sent, Act::illegal,
            "one outstanding REQ per request (S6.2)"},
        {HState::sent, Event::queued_line, HState::accepted, Act::none,
            "accepted, no slot claimed yet"},
        {HState::sent, Event::claim_line, HState::prefilling, Act::none,
            "the slot was free: CLAIM straight away"},
        {HState::sent, Event::seg_line, HState::idle, Act::protocol_error,
            "SEG before CLAIM: S4.5 rule 4"},
        {HState::sent, Event::done_line, HState::idle, Act::protocol_error,
            "DONE with no CLAIM: S4.5 rule 4, close the connection"},
        {HState::sent, Event::err_temporary, HState::held, Act::hold,
            "S4.4: a temporary ERR is a hold, never a client ERR (D5)"},
        {HState::sent, Event::err_permanent, HState::failed, Act::none,
            "S4.4: a permanent ERR ends the handoff; the caller collects the verdict"},
        {HState::sent, Event::err_cancel, HState::cancelled, Act::none,
            "the prefill instance answered ERR cancel: the job is gone"},
        {HState::sent, Event::cancel_line, HState::sent, Act::illegal,
            "CANCEL is not a line this side receives"},
        {HState::sent, Event::ack_sent, HState::sent, Act::illegal,
            "no slot held"},
        {HState::sent, Event::copy_done, HState::sent, Act::illegal,
            "nothing to copy"},
        {HState::sent, Event::mount_ok, HState::sent, Act::illegal,
            "the REQ is in flight; the peer's answer drives the state"},
        {HState::sent, Event::mount_fail, HState::sent, Act::illegal,
            "the REQ is in flight; the peer's answer drives the state"},
        {HState::sent, Event::collect, HState::sent, Act::illegal,
            "nothing to collect"},
        {HState::sent, Event::retry, HState::sent, Act::illegal,
            "a REQ is already in flight; wait for the answer"},
        {HState::sent, Event::note_cancel, HState::cancelled, Act::write_cancel,
            "S4.6: cancelled while the REQ is in flight"},
        {HState::sent, Event::local_hit, HState::sent, Act::illegal,
            "the cache lookup happens before REQ; not now"},
        // ---- HELD ----
        {HState::held, Event::admit, HState::held, Act::hold,
            "still held: the wait queue owns it"},
        {HState::held, Event::req_sent, HState::sent, Act::write_req,
            "the backoff expired: re-ask, still one outstanding REQ"},
        {HState::held, Event::queued_line, HState::accepted, Act::none,
            "a late QUEUED for the REQ we gave up on"},
        {HState::held, Event::claim_line, HState::prefilling, Act::none,
            "a late CLAIM for the REQ we gave up on"},
        {HState::held, Event::seg_line, HState::idle, Act::protocol_error,
            "SEG before CLAIM: S4.5 rule 4"},
        {HState::held, Event::done_line, HState::idle, Act::protocol_error,
            "DONE with no CLAIM in this attempt"},
        {HState::held, Event::err_temporary, HState::held, Act::hold,
            "a second temporary answer: hold again and the backoff doubles"},
        {HState::held, Event::err_permanent, HState::failed, Act::none,
            "a permanent ERR while held: the request is over"},
        {HState::held, Event::err_cancel, HState::cancelled, Act::none,
            "the job was cancelled at the prefill instance"},
        {HState::held, Event::cancel_line, HState::held, Act::illegal,
            "CANCEL is not a line this side receives"},
        {HState::held, Event::ack_sent, HState::held, Act::illegal,
            "no slot held"},
        {HState::held, Event::copy_done, HState::held, Act::illegal,
            "nothing to copy"},
        {HState::held, Event::mount_ok, HState::held, Act::illegal,
            "the hold is owned by S4.2's wait queue, not by the handoff thread"},
        {HState::held, Event::mount_fail, HState::held, Act::illegal,
            "the hold is owned by S4.2's wait queue, not by the handoff thread"},
        {HState::held, Event::collect, HState::held, Act::illegal,
            "nothing to collect"},
        {HState::held, Event::retry, HState::init, Act::none,
            "the retry bound allowed another handoff"},
        {HState::held, Event::note_cancel, HState::idle, Act::done_cancel,
            "S4.6: cancelled while held; no slot was ever claimed"},
        {HState::held, Event::local_hit, HState::handed_over, Act::mount,
            "the local cache hit while it held"},
        // ---- ACCEPTED ----
        {HState::accepted, Event::admit, HState::accepted, Act::illegal,
            "already admitted"},
        {HState::accepted, Event::req_sent, HState::accepted, Act::illegal,
            "one outstanding REQ per request"},
        {HState::accepted, Event::queued_line, HState::accepted, Act::none,
            "a second QUEUED is a progress update, not a new state"},
        {HState::accepted, Event::claim_line, HState::prefilling, Act::none,
            "the slot was claimed; the read is about to start"},
        {HState::accepted, Event::seg_line, HState::idle, Act::protocol_error,
            "SEG before CLAIM: S4.5 rule 4"},
        {HState::accepted, Event::done_line, HState::idle, Act::protocol_error,
            "DONE with no CLAIM: S4.5 rule 4, close the connection"},
        {HState::accepted, Event::err_temporary, HState::held, Act::hold,
            "S4.4: temporary -> hold, never a client ERR"},
        {HState::accepted, Event::err_permanent, HState::failed, Act::none,
            "S4.4: permanent -> the caller collects the verdict"},
        {HState::accepted, Event::err_cancel, HState::cancelled, Act::none,
            "the job was dropped before it ran (S4.6)"},
        {HState::accepted, Event::cancel_line, HState::accepted, Act::illegal,
            "CANCEL is not a line this side receives"},
        {HState::accepted, Event::ack_sent, HState::accepted, Act::illegal,
            "no slot held"},
        {HState::accepted, Event::copy_done, HState::accepted, Act::illegal,
            "nothing to copy"},
        {HState::accepted, Event::mount_ok, HState::accepted, Act::illegal,
            "the handoff is in flight"},
        {HState::accepted, Event::mount_fail, HState::accepted, Act::illegal,
            "the handoff is in flight"},
        {HState::accepted, Event::collect, HState::accepted, Act::illegal,
            "nothing to collect"},
        {HState::accepted, Event::retry, HState::accepted, Act::illegal,
            "the handoff is in flight; a retry would double it"},
        {HState::accepted, Event::note_cancel, HState::cancelled, Act::write_cancel,
            "S4.6: cancelled while QUEUED - the job is dropped, no slot touched"},
        {HState::accepted, Event::local_hit, HState::accepted, Act::illegal,
            "the handoff is already in flight"},
        // ---- PREFILLING ----
        {HState::prefilling, Event::admit, HState::prefilling, Act::illegal,
            "already admitted"},
        {HState::prefilling, Event::req_sent, HState::prefilling, Act::illegal,
            "one outstanding REQ per request"},
        {HState::prefilling, Event::queued_line, HState::prefilling, Act::illegal,
            "QUEUED after CLAIM is out of order"},
        {HState::prefilling, Event::claim_line, HState::prefilling, Act::illegal,
            "a second CLAIM for one request means the first slot was lost"},
        {HState::prefilling, Event::seg_line, HState::prefilling, Act::none,
            "progress; a reader that ignores SEG is still correct (S4.3)"},
        {HState::prefilling, Event::done_line, HState::seg_ready, Act::copy_out,
            "the payload is published; the copy-out starts HERE, never on the engine thread"},
        {HState::prefilling, Event::err_temporary, HState::held, Act::hold,
            "S4.4: read/slotlost/shutting are waits, not client errors"},
        {HState::prefilling, Event::err_permanent, HState::failed, Act::none,
            "S4.4: a permanent ERR ends the handoff"},
        {HState::prefilling, Event::err_cancel, HState::cancelled, Act::none,
            "S4.6: the read stopped at a chunk boundary and the slot was released without publishing"},
        {HState::prefilling, Event::cancel_line, HState::prefilling, Act::illegal,
            "CANCEL is not a line this side receives"},
        {HState::prefilling, Event::ack_sent, HState::prefilling, Act::illegal,
            "the slot is not READY; ACK would release an unwritten slot"},
        {HState::prefilling, Event::copy_done, HState::prefilling, Act::illegal,
            "nothing is published to copy"},
        {HState::prefilling, Event::mount_ok, HState::prefilling, Act::illegal,
            "the read is running at the prefill instance"},
        {HState::prefilling, Event::mount_fail, HState::prefilling, Act::illegal,
            "the read is running at the prefill instance"},
        {HState::prefilling, Event::collect, HState::prefilling, Act::illegal,
            "nothing to collect"},
        {HState::prefilling, Event::retry, HState::prefilling, Act::illegal,
            "the handoff is in flight; a retry would double it"},
        {HState::prefilling, Event::note_cancel, HState::cancelled, Act::write_cancel,
            "S4.6: cancelled while reading; the slot is released WITHOUT publishing"},
        {HState::prefilling, Event::local_hit, HState::prefilling, Act::illegal,
            "the handoff is already in flight"},
        // ---- SEG_READY ----
        {HState::seg_ready, Event::admit, HState::seg_ready, Act::illegal,
            "already admitted"},
        {HState::seg_ready, Event::req_sent, HState::seg_ready, Act::illegal,
            "one outstanding REQ per request"},
        {HState::seg_ready, Event::queued_line, HState::seg_ready, Act::illegal,
            "QUEUED after DONE is out of order"},
        {HState::seg_ready, Event::claim_line, HState::seg_ready, Act::illegal,
            "a second CLAIM while one slot is held"},
        {HState::seg_ready, Event::seg_line, HState::seg_ready, Act::none,
            "a late SEG is progress about the same payload"},
        {HState::seg_ready, Event::done_line, HState::seg_ready, Act::illegal,
            "a second DONE for one handoff means the slot was reused under us"},
        {HState::seg_ready, Event::err_temporary, HState::seg_ready, Act::illegal,
            "an ERR after DONE contradicts the published payload"},
        {HState::seg_ready, Event::err_permanent, HState::seg_ready, Act::illegal,
            "an ERR after DONE contradicts the published payload"},
        {HState::seg_ready, Event::err_cancel, HState::cancelled, Act::write_ack,
            "S4.5 rule 3: release the slot we hold, never read it"},
        {HState::seg_ready, Event::cancel_line, HState::seg_ready, Act::illegal,
            "CANCEL is not a line this side receives"},
        {HState::seg_ready, Event::ack_sent, HState::handed_over, Act::mount,
            "the caller wrote ACK itself; the image is ours"},
        {HState::seg_ready, Event::copy_done, HState::handed_over, Act::write_ack,
            "S4.5 rule 1: ACK as soon as the copy-out and checks 1-2 pass, BEFORE the mount"},
        {HState::seg_ready, Event::mount_ok, HState::seg_ready, Act::illegal,
            "not mounted yet"},
        {HState::seg_ready, Event::mount_fail, HState::seg_ready, Act::illegal,
            "not mounted yet"},
        {HState::seg_ready, Event::collect, HState::seg_ready, Act::illegal,
            "nothing to collect"},
        {HState::seg_ready, Event::retry, HState::seg_ready, Act::illegal,
            "release the slot first (ACK), then retry"},
        {HState::seg_ready, Event::note_cancel, HState::cancelled, Act::write_ack,
            "S4.6: cancelled after DONE, before ACK - send ACK, never mount"},
        {HState::seg_ready, Event::local_hit, HState::seg_ready, Act::illegal,
            "the payload is already in hand"},
        // ---- HANDED_OVER ----
        {HState::handed_over, Event::admit, HState::handed_over, Act::mount,
            "the engine thread may take it now"},
        {HState::handed_over, Event::req_sent, HState::handed_over, Act::illegal,
            "the handoff finished"},
        {HState::handed_over, Event::queued_line, HState::handed_over, Act::illegal,
            "QUEUED after DONE is out of order"},
        {HState::handed_over, Event::claim_line, HState::handed_over, Act::illegal,
            "a second CLAIM while the image is in hand"},
        {HState::handed_over, Event::seg_line, HState::handed_over, Act::none,
            "a late SEG is harmless"},
        {HState::handed_over, Event::done_line, HState::handed_over, Act::illegal,
            "a second DONE for one handoff"},
        {HState::handed_over, Event::err_temporary, HState::handed_over, Act::illegal,
            "an ERR after a successful handoff"},
        {HState::handed_over, Event::err_permanent, HState::handed_over, Act::illegal,
            "an ERR after a successful handoff"},
        {HState::handed_over, Event::err_cancel, HState::handed_over, Act::illegal,
            "an ERR cancel after a successful handoff"},
        {HState::handed_over, Event::cancel_line, HState::handed_over, Act::illegal,
            "CANCEL is not a line this side receives"},
        {HState::handed_over, Event::ack_sent, HState::handed_over, Act::illegal,
            "ACK was already sent; a second one would release somebody else's slot"},
        {HState::handed_over, Event::copy_done, HState::handed_over, Act::illegal,
            "already copied out"},
        {HState::handed_over, Event::mount_ok, HState::mounting, Act::mount,
            "post the image to the engine thread as mount_image"},
        {HState::handed_over, Event::mount_fail, HState::failed, Act::none,
            "the image could not be mounted before any write: ERR badpayload or a local re-read"},
        {HState::handed_over, Event::collect, HState::handed_over, Act::illegal,
            "nothing to collect"},
        {HState::handed_over, Event::retry, HState::handed_over, Act::illegal,
            "the handoff finished; a mount failure is retried from H_FAILED"},
        {HState::handed_over, Event::note_cancel, HState::cancelled, Act::none,
            "S4.6: cancelled after ACK - an ordinary STOP on the client wire, stage 3's business"},
        {HState::handed_over, Event::local_hit, HState::handed_over, Act::illegal,
            "the image is already in hand"},
        // ---- MOUNTING ----
        {HState::mounting, Event::admit, HState::mounting, Act::illegal,
            "the engine thread is already mounting"},
        {HState::mounting, Event::req_sent, HState::mounting, Act::illegal,
            "the handoff finished"},
        {HState::mounting, Event::queued_line, HState::mounting, Act::illegal,
            "QUEUED after DONE is out of order"},
        {HState::mounting, Event::claim_line, HState::mounting, Act::illegal,
            "a second CLAIM while mounting"},
        {HState::mounting, Event::seg_line, HState::mounting, Act::none,
            "a late SEG is harmless"},
        {HState::mounting, Event::done_line, HState::mounting, Act::illegal,
            "a second DONE for one handoff"},
        {HState::mounting, Event::err_temporary, HState::mounting, Act::illegal,
            "an ERR after a successful handoff"},
        {HState::mounting, Event::err_permanent, HState::mounting, Act::illegal,
            "an ERR after a successful handoff"},
        {HState::mounting, Event::err_cancel, HState::mounting, Act::illegal,
            "an ERR cancel after a successful handoff"},
        {HState::mounting, Event::cancel_line, HState::mounting, Act::illegal,
            "CANCEL is not a line this side receives"},
        {HState::mounting, Event::ack_sent, HState::mounting, Act::illegal,
            "ACK was already sent; the slot is FREE"},
        {HState::mounting, Event::copy_done, HState::mounting, Act::illegal,
            "already copied"},
        {HState::mounting, Event::mount_ok, HState::done, Act::finish,
            "validate, unmount, mount, draft_kv, adopt, device_state all passed"},
        {HState::mounting, Event::mount_fail, HState::failed, Act::none,
            "validate failed: ERR badpayload, or a local re-read with --prefill-fallback local"},
        {HState::mounting, Event::collect, HState::mounting, Act::illegal,
            "nothing to collect"},
        {HState::mounting, Event::retry, HState::mounting, Act::illegal,
            "finish the mount before retrying"},
        {HState::mounting, Event::note_cancel, HState::mounting, Act::illegal,
            "a mount is not cancellable: it has already written session state"},
        {HState::mounting, Event::local_hit, HState::mounting, Act::illegal,
            "already mounting"},
        // ---- DONE ----
        {HState::done, Event::admit, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::req_sent, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::queued_line, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::claim_line, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::seg_line, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::done_line, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::err_temporary, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::err_permanent, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::err_cancel, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::cancel_line, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::ack_sent, HState::done, Act::illegal,
            "the slot is already FREE"},
        {HState::done, Event::copy_done, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::mount_ok, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::mount_fail, HState::done, Act::illegal,
            "the handoff is over"},
        {HState::done, Event::collect, HState::idle, Act::finish,
            "the caller collected the result"},
        {HState::done, Event::retry, HState::done, Act::illegal,
            "a finished handoff is not retried; a new request begins a new machine"},
        {HState::done, Event::note_cancel, HState::done, Act::client_gone,
            "the request already has its payload: an ordinary STOP, stage 3's business"},
        {HState::done, Event::local_hit, HState::done, Act::illegal,
            "already handed over"},
        // ---- FAILED ----
        {HState::failed, Event::admit, HState::failed, Act::illegal,
            "already admitted"},
        {HState::failed, Event::req_sent, HState::failed, Act::illegal,
            "collect or retry first"},
        {HState::failed, Event::queued_line, HState::failed, Act::illegal,
            "the handoff already ended with a verdict"},
        {HState::failed, Event::claim_line, HState::failed, Act::illegal,
            "the handoff already ended with a verdict"},
        {HState::failed, Event::seg_line, HState::failed, Act::illegal,
            "the handoff already ended with a verdict"},
        {HState::failed, Event::done_line, HState::failed, Act::illegal,
            "the handoff already ended with a verdict"},
        {HState::failed, Event::err_temporary, HState::failed, Act::illegal,
            "the handoff already ended with a verdict"},
        {HState::failed, Event::err_permanent, HState::failed, Act::illegal,
            "the handoff already ended with a verdict"},
        {HState::failed, Event::err_cancel, HState::failed, Act::illegal,
            "the handoff already ended with a verdict"},
        {HState::failed, Event::cancel_line, HState::failed, Act::illegal,
            "the handoff already ended with a verdict"},
        {HState::failed, Event::ack_sent, HState::failed, Act::illegal,
            "no slot held"},
        {HState::failed, Event::copy_done, HState::failed, Act::illegal,
            "nothing to copy"},
        {HState::failed, Event::mount_ok, HState::failed, Act::illegal,
            "the handoff already ended with a verdict"},
        {HState::failed, Event::mount_fail, HState::failed, Act::illegal,
            "the handoff already ended with a verdict"},
        {HState::failed, Event::collect, HState::idle, Act::err_client,
            "the caller answers the client (permanent codes only - D5) and releases the row"},
        {HState::failed, Event::retry, HState::init, Act::none,
            "S4.5: a retry of a FAILED handoff re-enters H_REQ_SENT, bounded by --prefill-retries"},
        {HState::failed, Event::note_cancel, HState::failed, Act::done_cancel,
            "cancelled after a failure: the client gets DONE ... cancel, not ERR"},
        {HState::failed, Event::local_hit, HState::failed, Act::illegal,
            "the handoff already ended with a verdict"},
        // ---- CANCELLED ----
        {HState::cancelled, Event::admit, HState::cancelled, Act::illegal,
            "it is cancelled"},
        {HState::cancelled, Event::req_sent, HState::cancelled, Act::illegal,
            "it is cancelled"},
        {HState::cancelled, Event::queued_line, HState::cancelled, Act::write_cancel,
            "a QUEUED that races our CANCEL"},
        {HState::cancelled, Event::claim_line, HState::cancelled, Act::write_cancel,
            "a CLAIM that races our CANCEL: CANCEL again, the writer releases without publishing"},
        {HState::cancelled, Event::seg_line, HState::cancelled, Act::none,
            "progress about a job we killed"},
        {HState::cancelled, Event::done_line, HState::cancelled, Act::write_ack,
            "S4.5 rule 3: a DONE for a cancelled request is answered ACK, never read"},
        {HState::cancelled, Event::err_temporary, HState::cancelled, Act::none,
            "an ERR after we cancelled: nothing to do"},
        {HState::cancelled, Event::err_permanent, HState::cancelled, Act::none,
            "an ERR after we cancelled: nothing to do"},
        {HState::cancelled, Event::err_cancel, HState::idle, Act::done_cancel,
            "the prefill instance answered ERR cancel: the job is gone"},
        {HState::cancelled, Event::cancel_line, HState::cancelled, Act::illegal,
            "the request is cancelled"},
        {HState::cancelled, Event::ack_sent, HState::idle, Act::done_cancel,
            "the slot is released; the client gets DONE ... cancel"},
        {HState::cancelled, Event::copy_done, HState::cancelled, Act::illegal,
            "a cancelled handoff never copies out"},
        {HState::cancelled, Event::mount_ok, HState::cancelled, Act::illegal,
            "a cancelled handoff never mounts"},
        {HState::cancelled, Event::mount_fail, HState::cancelled, Act::illegal,
            "a cancelled handoff never mounts"},
        {HState::cancelled, Event::collect, HState::idle, Act::done_cancel,
            "no slot was ever claimed: nothing to release"},
        {HState::cancelled, Event::retry, HState::cancelled, Act::illegal,
            "a cancelled request is not re-asked"},
        {HState::cancelled, Event::note_cancel, HState::cancelled, Act::none,
            "already cancelled; a second CANCEL is idempotent"},
        {HState::cancelled, Event::local_hit, HState::cancelled, Act::illegal,
            "it is cancelled"},
    };
    count = (int) (sizeof t / sizeof t[0]);
    return t;
}



inline const JobTransition* prefill_job_table(int& count) {
    // Exactly kJobStateCount * kJobEventCount rows: one per (job state, event).
    static const JobTransition t[] = {
        // ---- QUEUED ----
        {PJob::queued, PEvent::claim_ok, PJob::claimed, PAct::send_claim,
            "a slot was claimed for it"},
        {PJob::queued, PEvent::read_start, PJob::queued, PAct::illegal,
            "no slot has been claimed"},
        {PJob::queued, PEvent::seg_sent, PJob::queued, PAct::illegal,
            "no slot has been claimed"},
        {PJob::queued, PEvent::publish_ok, PJob::queued, PAct::illegal,
            "no slot has been claimed"},
        {PJob::queued, PEvent::ack, PJob::queued, PAct::illegal,
            "no slot has been claimed"},
        {PJob::queued, PEvent::cancel, PJob::cancelled, PAct::err_cancel,
            "S4.6: cancelled while QUEUED - drop the job, no slot touched"},
        {PJob::queued, PEvent::read_fail, PJob::queued, PAct::illegal,
            "no slot has been claimed"},
        {PJob::queued, PEvent::publish_lost, PJob::queued, PAct::illegal,
            "no slot has been claimed"},
        {PJob::queued, PEvent::abort_done, PJob::queued, PAct::illegal,
            "no slot has been claimed"},
        // ---- CLAIMED ----
        {PJob::claimed, PEvent::claim_ok, PJob::claimed, PAct::illegal,
            "the read has not started"},
        {PJob::claimed, PEvent::read_start, PJob::reading, PAct::none,
            "Prefill::run begins"},
        {PJob::claimed, PEvent::seg_sent, PJob::claimed, PAct::illegal,
            "the read has not started"},
        {PJob::claimed, PEvent::publish_ok, PJob::claimed, PAct::illegal,
            "the read has not started"},
        {PJob::claimed, PEvent::ack, PJob::claimed, PAct::illegal,
            "the read has not started"},
        {PJob::claimed, PEvent::cancel, PJob::cancelling, PAct::abort_err_cancel,
            "S4.6: release the slot WITHOUT publishing"},
        {PJob::claimed, PEvent::read_fail, PJob::failed, PAct::abort_err_read,
            "the read failed before any byte was written"},
        {PJob::claimed, PEvent::publish_lost, PJob::claimed, PAct::illegal,
            "the read has not started"},
        {PJob::claimed, PEvent::abort_done, PJob::claimed, PAct::illegal,
            "the read has not started"},
        // ---- READING ----
        {PJob::reading, PEvent::claim_ok, PJob::reading, PAct::illegal,
            "the read is running"},
        {PJob::reading, PEvent::read_start, PJob::reading, PAct::illegal,
            "the read is running"},
        {PJob::reading, PEvent::seg_sent, PJob::writing, PAct::send_seg,
            "the read finished and the payload write started"},
        {PJob::reading, PEvent::publish_ok, PJob::published, PAct::send_done,
            "a zero-GPU reuse handoff may skip SEG (S1.3.3)"},
        {PJob::reading, PEvent::ack, PJob::reading, PAct::illegal,
            "the read is running"},
        {PJob::reading, PEvent::cancel, PJob::cancelling, PAct::abort_err_cancel,
            "S4.6: the stop flag lands at the next chunk boundary"},
        {PJob::reading, PEvent::read_fail, PJob::failed, PAct::abort_err_read,
            "a CUDA error or a lend failure: release without publishing"},
        {PJob::reading, PEvent::publish_lost, PJob::reading, PAct::illegal,
            "the read is running"},
        {PJob::reading, PEvent::abort_done, PJob::reading, PAct::illegal,
            "the read is running"},
        // ---- WRITING ----
        {PJob::writing, PEvent::claim_ok, PJob::writing, PAct::illegal,
            "the payload write is running"},
        {PJob::writing, PEvent::read_start, PJob::writing, PAct::illegal,
            "the payload write is running"},
        {PJob::writing, PEvent::seg_sent, PJob::writing, PAct::illegal,
            "the payload write is running"},
        {PJob::writing, PEvent::publish_ok, PJob::published, PAct::send_done,
            "the release fence and the ordered store landed"},
        {PJob::writing, PEvent::ack, PJob::writing, PAct::illegal,
            "the payload write is running"},
        {PJob::writing, PEvent::cancel, PJob::cancelling, PAct::abort_err_cancel,
            "cancelled mid-write: the payload is never published"},
        {PJob::writing, PEvent::read_fail, PJob::failed, PAct::abort_err_read,
            "the payload write failed"},
        {PJob::writing, PEvent::publish_lost, PJob::failed, PAct::err_slotlost,
            "the slot was reclaimed under us (S4.7): ERR slotlost, retry once"},
        {PJob::writing, PEvent::abort_done, PJob::writing, PAct::illegal,
            "the payload write is running"},
        // ---- PUBLISHED ----
        {PJob::published, PEvent::claim_ok, PJob::published, PAct::illegal,
            "the payload is published; only ACK or a cancel moves it"},
        {PJob::published, PEvent::read_start, PJob::published, PAct::illegal,
            "the payload is published; only ACK or a cancel moves it"},
        {PJob::published, PEvent::seg_sent, PJob::published, PAct::illegal,
            "the payload is published; only ACK or a cancel moves it"},
        {PJob::published, PEvent::publish_ok, PJob::published, PAct::illegal,
            "the payload is published; only ACK or a cancel moves it"},
        {PJob::published, PEvent::ack, PJob::released, PAct::release_slot,
            "the ONLY release verb"},
        {PJob::published, PEvent::cancel, PJob::released, PAct::release_slot,
            "S4.5 rule 3: CANCEL after DONE is a normal release, not a protocol error"},
        {PJob::published, PEvent::read_fail, PJob::published, PAct::illegal,
            "the payload is published; only ACK or a cancel moves it"},
        {PJob::published, PEvent::publish_lost, PJob::published, PAct::illegal,
            "the payload is published; only ACK or a cancel moves it"},
        {PJob::published, PEvent::abort_done, PJob::published, PAct::illegal,
            "the payload is published; only ACK or a cancel moves it"},
        // ---- RELEASED ----
        {PJob::released, PEvent::claim_ok, PJob::released, PAct::illegal,
            "the job is over"},
        {PJob::released, PEvent::read_start, PJob::released, PAct::illegal,
            "the job is over"},
        {PJob::released, PEvent::seg_sent, PJob::released, PAct::illegal,
            "the job is over"},
        {PJob::released, PEvent::publish_ok, PJob::released, PAct::illegal,
            "the job is over"},
        {PJob::released, PEvent::ack, PJob::released, PAct::illegal,
            "the job is over"},
        {PJob::released, PEvent::cancel, PJob::released, PAct::none,
            "already released"},
        {PJob::released, PEvent::read_fail, PJob::released, PAct::illegal,
            "the job is over"},
        {PJob::released, PEvent::publish_lost, PJob::released, PAct::illegal,
            "the job is over"},
        {PJob::released, PEvent::abort_done, PJob::released, PAct::illegal,
            "the job is over"},
        // ---- CANCELLING ----
        {PJob::cancelling, PEvent::claim_ok, PJob::cancelling, PAct::illegal,
            "the unwind is running"},
        {PJob::cancelling, PEvent::read_start, PJob::cancelling, PAct::illegal,
            "the unwind is running"},
        {PJob::cancelling, PEvent::seg_sent, PJob::cancelling, PAct::illegal,
            "the unwind is running"},
        {PJob::cancelling, PEvent::publish_ok, PJob::cancelling, PAct::abort_claim,
            "a publish that races a cancel must not land"},
        {PJob::cancelling, PEvent::ack, PJob::cancelling, PAct::illegal,
            "the unwind is running"},
        {PJob::cancelling, PEvent::cancel, PJob::cancelling, PAct::none,
            "a second CANCEL is idempotent"},
        {PJob::cancelling, PEvent::read_fail, PJob::cancelling, PAct::abort_claim,
            "the read failed while it was being cancelled"},
        {PJob::cancelling, PEvent::publish_lost, PJob::cancelling, PAct::abort_claim,
            "the slot was already taken back"},
        {PJob::cancelling, PEvent::abort_done, PJob::cancelled, PAct::err_cancel,
            "the unwind finished: the slot is FREE"},
        // ---- CANCELLED ----
        {PJob::cancelled, PEvent::claim_ok, PJob::cancelled, PAct::illegal,
            "the job is over"},
        {PJob::cancelled, PEvent::read_start, PJob::cancelled, PAct::illegal,
            "the job is over"},
        {PJob::cancelled, PEvent::seg_sent, PJob::cancelled, PAct::illegal,
            "the job is over"},
        {PJob::cancelled, PEvent::publish_ok, PJob::cancelled, PAct::illegal,
            "the job is over"},
        {PJob::cancelled, PEvent::ack, PJob::cancelled, PAct::illegal,
            "the job is over"},
        {PJob::cancelled, PEvent::cancel, PJob::cancelled, PAct::none,
            "idempotent"},
        {PJob::cancelled, PEvent::read_fail, PJob::cancelled, PAct::illegal,
            "the job is over"},
        {PJob::cancelled, PEvent::publish_lost, PJob::cancelled, PAct::illegal,
            "the job is over"},
        {PJob::cancelled, PEvent::abort_done, PJob::cancelled, PAct::none,
            "already unwound"},
        // ---- FAILED ----
        {PJob::failed, PEvent::claim_ok, PJob::failed, PAct::illegal,
            "the failure was already reported"},
        {PJob::failed, PEvent::read_start, PJob::failed, PAct::illegal,
            "the failure was already reported"},
        {PJob::failed, PEvent::seg_sent, PJob::failed, PAct::illegal,
            "the failure was already reported"},
        {PJob::failed, PEvent::publish_ok, PJob::failed, PAct::illegal,
            "the failure was already reported"},
        {PJob::failed, PEvent::ack, PJob::failed, PAct::illegal,
            "the failure was already reported"},
        {PJob::failed, PEvent::cancel, PJob::failed, PAct::illegal,
            "the failure was already reported"},
        {PJob::failed, PEvent::read_fail, PJob::failed, PAct::illegal,
            "the failure was already reported"},
        {PJob::failed, PEvent::publish_lost, PJob::failed, PAct::illegal,
            "the failure was already reported"},
        {PJob::failed, PEvent::abort_done, PJob::failed, PAct::illegal,
            "the failure was already reported"},
    };
    count = (int) (sizeof t / sizeof t[0]);
    return t;
}



}  // namespace strata::program::prefill_svc
