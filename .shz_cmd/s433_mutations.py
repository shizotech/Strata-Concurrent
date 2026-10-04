#!/usr/bin/env python3
"""S4.3.3 anti-vacuity: mutate prefill_svc.hpp, rebuild, and count how many mutations the suite catches.

Every mutation removes or weakens a check the header claims to make.  A mutation that changes nothing
means the corresponding test is vacuous.
"""
import os, shutil, subprocess, sys

SRC = "include/strata/program/prefill_svc.hpp"
TEST = "src/program/prefill_svc_test.cpp"
WORK = "/tmp/s433mut"

MUTS = [
    # ---- grammar / framing -----------------------------------------------------------------
    ("drop the max-line-length check",
     "if (line.size() > kMaxLineBytes) {", "if (false) {"),
    ("accept an embedded newline",
     "if (c == '\\n') { why = \"a line may not contain a newline\"; return false; }", ""),
    ("accept a CR",
     "if (c == '\\r') { why = \"a line may not contain a carriage return\"; return false; }", ""),
    ("accept a NUL",
     "if (c == '\\0') { why = \"a line may not contain a NUL\"; return false; }", ""),
    ("accept uppercase hex",
     "else if (c >= 'a' && c <= 'f') v = v * 16u + (uint64_t) (c - 'a' + 10);",
     "else if (c >= 'a' && c <= 'f') v = v * 16u + (uint64_t) (c - 'a' + 10);\n        else if (c >= 'A' && c <= 'F') v = v * 16u + (uint64_t) (c - 'A' + 10);"),
    ("accept a 15- or 17-digit hex hash",
     "if (t.size() != 16) return false;\n    v = 0;", "if (t.size() < 1) return false;\n    v = 0;"),
    ("let a list element be empty",
     "if (e == i) return false;                       // empty element", ""),
    ("let a NAME contain any byte",
     "inline bool is_name(const std::string& t) {\n    if (t.empty()) return false;",
     "inline bool is_name(const std::string& t) {\n    if (false) return false;"),
    ("let a PING cookie be any token",
     "inline bool is_id_token(const std::string& t) {\n    if (t.empty()) return false;\n    for (char c : t) {\n        const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||\n                        c == '_' || c == '-';\n        if (!ok) return false;\n    }\n    return true;\n}",
     "inline bool is_id_token(const std::string& t) { return !t.empty(); }"),
    ("make ids= optional",
     "if (it_ids == l.keys.end()) { l.bad = \"REQ is missing ids=NAME\"; return l; }", ""),
    ("skip the HELLO n_stages/split agreement check",
     "if ((int64_t) l.split.size() != l.n_stages) {   // D9", "if (false) {   // D9"),
    ("skip the READY slots/tiers length check",
     "if (l.slots.size() != l.tiers.size()) {", "if (false) {"),
    ("skip the ERR code vocabulary check",
     "if (!err_from_code(code, l.code)) {", "if (false) {"),
    ("re-join the ERR message from tokens (loses free text)",
     "size_t p = detail::past_token(line, 2);", "size_t p = line.size();"),
    ("accept TAKE as a normal verb",
     "l.bad = \"TAKE is reserved for the reverse handoff (S4.5) and is not part of v1\";", ""),
    ("reject an unknown verb as a connection error",
     "else { l.extra.assign(t.begin() + 1, t.end()); return l; }   // D8: unknown verb, not an error",
     "else { l.bad = \"unknown verb\"; return l; }"),
    # ---- the error table -------------------------------------------------------------------
    ("make `full` permanent (D5 violation)",
     "{ErrCode::full,       hold::wait,", "{ErrCode::full,       hold::error,"),
    ("make `endpoint` permanent (D5 violation)",
     "{ErrCode::endpoint,   hold::wait,", "{ErrCode::endpoint,   hold::error,"),
    ("make `toobig` temporary (an infinite hold for a fixed-length prompt)",
     "{ErrCode::toobig,     hold::error,", "{ErrCode::toobig,     hold::wait,"),
    ("drop geom's drop-connection flag",
     "{ErrCode::geom,       hold::error, true,", "{ErrCode::geom,       hold::error, false,"),
    ("make `read` non-retryable",
     "{ErrCode::read,       hold::wait,  false, true,", "{ErrCode::read,       hold::wait,  false, false,"),
    ("drop badpayload's decode-emitted flag",
     "{ErrCode::badpayload, hold::error, false, false, true,", "{ErrCode::badpayload, hold::error, false, false, false,"),
    ("drop the wait reason string",
     'inline constexpr const char* kWaitReasonPrefill = "prefill-busy";',
     'inline constexpr const char* kWaitReasonPrefill = "prefill";'),
    ("drop the wait text",
     'inline constexpr const char* kWaitTextPrefill =\n    "the prefill instance has no free handoff slot for a prompt this long";',
     'inline constexpr const char* kWaitTextPrefill = "busy";'),
    # ---- the handshake ---------------------------------------------------------------------
    ("skip the protocol-version check",
     "if (ready.proto_ver != hello.proto_ver) {", "if (false) {"),
    ("skip the pack-hash check",
     "if (ready.pack_hash != hello.pack_hash) {", "if (false) {"),
    ("skip the geometry-hash check",
     "if (ready.geom_hash != hello.geom_hash) {", "if (false) {"),
    ("skip the max_tokens-vs-largest-tier check",
     "if (top != ready.max_tokens) {", "if (false) {"),
    ("accept a HELLO from an old protocol",
     "if (hello.proto_ver != kProtoVersion) {\n        why = \"the client speaks protocol version",
     "if (false) {\n        why = \"the client speaks protocol version"),
    ("accept client_id 0",
     "if (hello.client_id == 0) {", "if (false) {"),
    ("skip the vision requirement check",
     "if (hello.vision && !prefill_vision) {", "if (false) {"),
    # ---- the tier rule ---------------------------------------------------------------------
    ("let a prompt bigger than the largest tier pick a tier anyway",
     "if (tokens <= ascending_tiers[i]) { tier = i; return true; }", "if (true) { tier = i; return true; }"),
    ("skip the toobig check",
     "if (tokens > cap) {", "if (false) {"),
    ("let a big job use a smaller tier group",
     "inline bool tier_may_use(uint32_t job_tier, uint32_t slot_group) { return slot_group >= job_tier; }",
     "inline bool tier_may_use(uint32_t, uint32_t) { return true; }"),
    # ---- the state machine -----------------------------------------------------------------
    ("allow DONE with no CLAIM (start the copy-out)",
     "{HState::sent, Event::done_line, HState::idle, Act::protocol_error,",
     "{HState::sent, Event::done_line, HState::seg_ready, Act::copy_out,"),
    ("allow DONE with no CLAIM from IDLE",
     "{HState::idle, Event::done_line, HState::idle, Act::protocol_error,",
     "{HState::idle, Event::done_line, HState::seg_ready, Act::copy_out,"),
    ("allow SEG with no CLAIM",
     "{HState::accepted, Event::seg_line, HState::idle, Act::protocol_error,",
     "{HState::accepted, Event::seg_line, HState::accepted, Act::none,"),
    ("allow a second REQ while one is outstanding",
     "{HState::sent, Event::req_sent, HState::sent, Act::illegal,",
     "{HState::sent, Event::req_sent, HState::sent, Act::write_req,"),
    ("let a temporary ERR reach the client",
     "{HState::sent, Event::err_temporary, HState::held, Act::hold,",
     "{HState::sent, Event::err_temporary, HState::failed, Act::err_client,"),
    ("let a permanent ERR become a hold (an infinite wait for a fixed-length prompt)",
     "{HState::sent, Event::err_permanent, HState::failed, Act::none,",
     "{HState::sent, Event::err_permanent, HState::held, Act::hold,"),
    ("allow ACK before DONE (free a slot mid-write)",
     "{HState::prefilling, Event::ack_sent, HState::prefilling, Act::illegal,",
     "{HState::prefilling, Event::ack_sent, HState::handed_over, Act::mount,"),
    ("allow a second ACK (release somebody else's slot)",
     "{HState::handed_over, Event::ack_sent, HState::handed_over, Act::illegal,",
     "{HState::handed_over, Event::ack_sent, HState::handed_over, Act::none,"),
    ("allow a copy-out with no published slot",
     "{HState::sent, Event::copy_done, HState::sent, Act::illegal,",
     "{HState::sent, Event::copy_done, HState::handed_over, Act::mount,"),
    ("allow a mount to be cancelled",
     "{HState::mounting, Event::note_cancel, HState::mounting, Act::illegal,",
     "{HState::mounting, Event::note_cancel, HState::cancelled, Act::write_ack,"),
    ("allow a retry from H_CANCELLED (a cancelled request re-asked)",
     "{HState::cancelled, Event::retry, HState::cancelled, Act::illegal,",
     "{HState::cancelled, Event::retry, HState::init, Act::none,"),
    ("allow a retry from H_SEG_READY (double the handoff, keep the slot)",
     "{HState::seg_ready, Event::retry, HState::seg_ready, Act::illegal,",
     "{HState::seg_ready, Event::retry, HState::init, Act::none,"),
    ("let a cancelled handoff copy out",
     "{HState::cancelled, Event::copy_done, HState::cancelled, Act::illegal,",
     "{HState::cancelled, Event::copy_done, HState::handed_over, Act::write_ack,"),
    ("let a peer line drive a queued request (D3)",
     "{HState::queued, Event::claim_line, HState::queued, Act::illegal,",
     "{HState::queued, Event::claim_line, HState::prefilling, Act::none,"),
    ("let a second DONE for one handoff through",
     "{HState::seg_ready, Event::done_line, HState::seg_ready, Act::illegal,",
     "{HState::seg_ready, Event::done_line, HState::seg_ready, Act::none,"),
    ("drop the CLAIM nonce-conflict rule",
     "if (nonce_ != 0 && l.nonce != nonce_) {\n                r.action = Act::protocol_error;\n                r.why = \"CLAIM carries a second nonce",
     "if (false) {\n                r.action = Act::protocol_error;\n                r.why = \"CLAIM carries a second nonce"),
    ("drop the DONE nonce-match rule",
     "} else if (e == Event::done_line && nonce_ != 0 && l.nonce != nonce_) {",
     "} else if (false) {"),
    ("drop the wrong-request-id guard",
     "if (l.id != id) {\n            r.action = Act::illegal;", "if (false) {\n            r.action = Act::illegal;"),
    ("drop the wrong-side line guard",
     "if (!line_from_peer(l.kind, Side::decode_to_prefill)) {", "if (false) {"),
    ("neuter the retry bound (the S3.8 livelock)",
     "bool retry_allowed() const { return req_lines_ <= retries_; }",
     "bool retry_allowed() const { return true; }"),
    ("make the backoff unbounded (hammer a full arena)",
     ": (backoff_ms_ * 2 > kBackoffCapMs ? kBackoffCapMs : backoff_ms_ * 2);",
     ": backoff_ms_ * 4;"),
    ("make the backoff never grow",
     "backoff_ms_ = backoff_ms_ == 0 ? kBackoffStartMs", "backoff_ms_ = 0 ? kBackoffStartMs"),
    ("let may_send_req ignore the outstanding REQ",
     "bool may_send_req(int64_t now_ms) const { return !outstanding_ && now_ms >= next_ask_ms_; }",
     "bool may_send_req(int64_t now_ms) const { return now_ms >= next_ask_ms_; }"),
    ("let may_send_req ignore the backoff",
     "return !outstanding_ && now_ms >= next_ask_ms_;", "return !outstanding_;"),
    ("let may_release accept any nonce",
     "return nonce_ != 0 && nonce == nonce_ && slot_ == slot;", "return true;"),
    ("let may_release ignore the slot",
     "return nonce_ != 0 && nonce == nonce_ && slot_ == slot;", "return nonce_ != 0 && nonce == nonce_;"),
    ("let a poisoned mount be retried",
     "r.action = Act::err_client;\n            r.why = \"the session is poisoned: return 1, do not attempt a recovery\";",
     "r.action = Act::none;\n            r.why = \"the session is poisoned\";"),
    ("drop the poisoned-session flag",
     "poison_ = true;", ""),
    # ---- the prefill job table -------------------------------------------------------------
    ("let ACK release an unpublished slot",
     "{PJob::reading, PEvent::ack, PJob::reading, PAct::illegal,",
     "{PJob::reading, PEvent::ack, PJob::released, PAct::release_slot,"),
    ("let a second ACK release again",
     "{PJob::released, PEvent::ack, PJob::released, PAct::illegal,",
     "{PJob::released, PEvent::ack, PJob::released, PAct::release_slot,"),
    ("let CANCEL after DONE be a protocol error instead of a release",
     "{PJob::published, PEvent::cancel, PJob::released, PAct::release_slot,",
     "{PJob::published, PEvent::cancel, PJob::published, PAct::illegal,"),
    ("let a cancelled job publish anyway",
     "{PJob::cancelling, PEvent::publish_ok, PJob::cancelling, PAct::abort_claim,",
     "{PJob::cancelling, PEvent::publish_ok, PJob::published, PAct::send_done,"),
    ("answer a cancelled read with ERR read instead of cancel",
     "{PJob::reading, PEvent::cancel, PJob::cancelling, PAct::abort_err_cancel,",
     "{PJob::reading, PEvent::cancel, PJob::cancelling, PAct::abort_err_read,"),
    ("lose the slotlost answer",
     "{PJob::writing, PEvent::publish_lost, PJob::failed, PAct::err_slotlost,",
     "{PJob::writing, PEvent::publish_lost, PJob::failed, PAct::err_read,"),
    # ---- the ids-file rule -----------------------------------------------------------------
    ("allow a .. component in an ids path",
     "if (dir[i] == '.' && i + 1 < dir.size() && dir[i + 1] == '.') return false;", ""),
    ("allow an absolute ids= name from the wire",
     'if (name.rfind(\'/\', 0) == 0) return false;                 // absolute: refuse', ""),
    ("allow a .. component in an ids= name",
     'if (part == "..") return false;', ""),
    # ---- the log lines ---------------------------------------------------------------------
    ("drop the retry-limit line's bound",
     '"); the bound is --prefill-retries "', '"); the bound is --prefill-retries " /*x*/ + "" + "'),
    ("drop the endpoint-down line",
     'return "strata serve: the prefill instance died; holding new prompts"',
     'return "strata serve: prefill note"'),
    ("drop the slots_free field from the activity line",
     '" queued=" + std::to_string((long long) queued) + " slots_free=" + slots_free +',
     '" queued=" + std::to_string((long long) queued) + " x=" + slots_free +'),
]

def build(src, out):
    r = subprocess.run(["g++", "-std=c++20", "-O0", "-Iinclude", "-o", out, TEST],
                       capture_output=True, text=True)
    return r.returncode == 0, r.stderr

def run(out):
    r = subprocess.run([out], capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr

os.makedirs(WORK, exist_ok=True)
orig = open(SRC).read()

ok, err = build(orig, "/tmp/prefill_svc_base")
if not ok:
    print("BASE BUILD FAILED\n", err); sys.exit(1)
rc, base_out = run("/tmp/prefill_svc_base")
print("baseline:", base_out.strip().splitlines()[-1] if base_out else "?", "rc", rc)
if rc != 0:
    print("baseline must pass"); sys.exit(1)

caught, missed, broken = 0, [], []
for i, (name, a, b) in enumerate(MUTS):
    if a not in orig:
        broken.append(name)
        continue
    mutated = orig.replace(a, b, 1)
    if mutated == orig:
        broken.append(name + " (no-op edit)")
        continue
    path = os.path.join(WORK, "hpp")
    # build against a copy of the tree with the mutated header
    shutil.copytree("include", os.path.join(WORK, "include"), dirs_exist_ok=True)
    open(os.path.join(WORK, "include/strata/program/prefill_svc.hpp"), "w").write(mutated)
    r = subprocess.run(["g++", "-std=c++20", "-O0", "-I" + WORK + "/include", "-o",
                        os.path.join(WORK, "t"), TEST], capture_output=True, text=True)
    if r.returncode != 0:
        caught += 1
        print("  CAUGHT (compile)  %2d  %s" % (i, name))
        continue
    rr = subprocess.run([os.path.join(WORK, "t")], capture_output=True, text=True, timeout=120)
    if rr.returncode != 0:
        caught += 1
        first = [l for l in (rr.stdout + rr.stderr).splitlines() if l.startswith("FAIL")]
        print("  CAUGHT (test)     %2d  %s   [%s]" % (i, name, first[0][:90] if first else ""))
    else:
        missed.append(name)
        print("  MISSED            %2d  %s" % (i, name))

print()
print("mutations: %d   caught: %d   missed: %d   bad-patch: %d"
      % (len(MUTS), caught, len(missed), len(broken)))
for m in missed:
    print("  missed:", m)
for m in broken:
    print("  bad patch:", m)
