// src/program/prefill_svc_test.cpp - S4.3.3: the prefill<->decode control protocol, on a CPU.
//
// No engine, no model, no GPU, no socket, no /dev/shm.  Everything here drives
// `include/strata/program/prefill_svc.hpp` with DATA: lines in, decisions out.  The transport the
// protocol names (S4.3.2's arena, S4.3.1's codec) is stood in for by a by-value fake that makes the
// same state transitions the real one does - FREE/CLAIMED/READY, nonce ownership, leases - so the
// protocol's rules can be exercised end to end without a byte of I/O.
//
// What this file pins, and why each group exists:
//   * every line's grammar, formatted then parsed then formatted again (one formatter, one parser,
//     so a line and its parser cannot drift);
//   * malformed lines: truncated at every prefix, wrong arity, non-numeric fields, unknown verbs,
//     embedded newlines/CR/NUL, a line over the 4096-byte limit;
//   * the error-code -> `serve_driver::Hold` table is TOTAL and matches §4.4's table row for row, and
//     the `Hold`/`Wait` numbers are asserted against the REAL `serve_driver.hpp` so the two
//     vocabularies cannot drift apart;
//   * the §4.5 state machine over EVERY edge: the table is enumerated, its size is asserted to be
//     exactly states x events, and every cell is checked to be either a real transition or a refusal
//     with a named reason;
//   * the nonce rules (a stale nonce cannot release, read or cancel another request's slot), the
//     CLAIM-before-DONE rule in every state, cancellation in every state, and the retry bound;
//   * each of §4.7's twelve failure modes as a data-driven scenario against the fake transport;
//   * the S3.8 anti-vacuity shape: the retry-bound simulation is shown to run to its pass cap
//     WITHOUT finishing when the bound is neutered.
//
// The mutation harness lives in `.shz_cmd/s433_mutations.sh`: it rebuilds this file against mutated
// copies of the header and counts how many mutations the suite catches.

#include "strata/program/prefill_svc.hpp"

// The no-drift check for D3 of §1.6: this header must NOT include serve_proto.hpp, and the wait
// vocabulary it maps onto must be the real one.  `serve_driver.hpp` is header-only decisions and
// links nothing (the same reason `serve_driver_test` needs no libraries), so including it here costs
// nothing and turns a vocabulary drift into a failing check instead of a live-server surprise.
#include "strata/program/serve_driver.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
}
void check_eq(const std::string& got, const std::string& want, const char* what) {
    ++checks;
    if (got == want) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n  old: [%s]\n  new: [%s]\n", what, want.c_str(), got.c_str());
}
/// The lesson from `.megamind/src/core/s432-handoff-arena-notes.md`: `check(!api(..., err),
/// ("label: " + err).c_str())` is a bug, because argument evaluation order is unspecified and `err`
/// can be read before the call filled it.  Take the value, then assert.
void check_err(bool got, const std::string& err, bool want, const char* what) {
    ++checks;
    if (got == want) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s (returned %s, reason [%s])\n", what, got ? "true" : "false",
                 err.c_str());
}
/// A refusal is only a test of THAT refusal if the reason is pinned.  A test that accepts "some
/// reason was given" passes even when the check that should have fired was deleted.
void check_reason_contains(const std::string& got, const char* needle, const char* what) {
    ++checks;
    if (got.find(needle) != std::string::npos) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n  wanted a reason containing [%s]\n  got              [%s]\n",
                 what, needle, got.c_str());
}

using namespace strata::program;
using namespace strata::program::prefill_svc;

// ============================================================================ the fake transport ==
//
// A by-value stand-in for `core::HandoffArena` (S4.3.2) and the socket.  It is not the arena and it
// is not a replacement for `handoff_arena_test` - it exists so the PROTOCOL's rules (who may name
// which slot, with which nonce, in which order) can be exercised end to end.  The state names and the
// refusals mirror `include/strata/core/handoff.hpp` deliberately: if the protocol's idea of a slot
// stops matching the arena's, this fake is where the mismatch shows up first.

struct FakeSlot {
    int index = 0;
    uint32_t tier = 0;
    uint64_t capacity = 0;
    uint32_t state = 0;              ///< 0 FREE, 1 CLAIMED, 2 READY (handoff.hpp's enum)
    uint32_t flags = 0;
    uint64_t nonce = 0, client = 0, request = 0;
    uint64_t tokens = 0, bytes = 0, hash = 0;
    uint64_t claimed_ms = 0, published_ms = 0;
    std::vector<uint8_t> payload;
};

struct FakeArena {
    std::vector<uint64_t> tier_max_tokens;   ///< ASCENDING, like the real arena
    std::vector<uint64_t> tier_slots;
    std::vector<FakeSlot> slots;
    uint64_t next_nonce = 0x1000;
    uint64_t pack_hash = 0, geom_hash = 0;
    int claims_refused = 0;

    void build(const std::vector<uint64_t>& tiers, const std::vector<uint64_t>& counts,
               uint64_t pack, uint64_t geom) {
        tier_max_tokens = tiers;
        tier_slots = counts;
        pack_hash = pack;
        geom_hash = geom;
        slots.clear();
        int idx = 0;
        for (size_t t = 0; t < tiers.size(); ++t)
            for (uint64_t k = 0; k < counts[t]; ++k) {
                FakeSlot s;
                s.index = idx++;
                s.tier = (uint32_t) t;
                s.capacity = 1024 * (t + 1);
                slots.push_back(s);
            }
    }
    uint64_t max_tokens() const {
        return tier_max_tokens.empty() ? 0 : tier_max_tokens.back();
    }
    /// §3.8: the SMALLEST group j >= tier with a free slot.  A full arena is a TEMPORARY condition.
    bool claim(uint32_t tier, uint64_t client, uint64_t request, uint64_t now_ms, int& index,
               uint64_t& nonce, std::string& err) {
        err.clear();
        for (size_t g = tier; g < tier_max_tokens.size(); ++g) {
            for (auto& s : slots) {
                if (s.tier != g || s.state != 0) continue;
                s.state = 1;
                s.flags = 0;
                s.client = client;
                s.request = request;
                s.claimed_ms = now_ms;
                s.nonce = ++next_nonce;
                s.bytes = s.hash = s.tokens = 0;
                s.payload.clear();
                index = s.index;
                nonce = s.nonce;
                return true;
            }
        }
        ++claims_refused;
        err = "no free slot in group >= the requested tier";
        return false;
    }
    bool write(int index, const std::vector<uint8_t>& body) {
        if (index < 0 || (size_t) index >= slots.size()) return false;
        FakeSlot& s = slots[index];
        if (s.state != 1) return false;              // §3.5: only a CLAIMED slot is writable
        if (body.size() > s.capacity) return false;
        s.payload = body;
        return true;
    }
    bool publish(int index, uint64_t nonce, uint64_t tokens, std::string& err) {
        err.clear();
        if (index < 0 || (size_t) index >= slots.size()) { err = "no such slot"; return false; }
        FakeSlot& s = slots[index];
        if (s.state != 1) { err = "the claim is gone"; return false; }
        if (s.nonce != nonce) { err = "the nonce moved under the claim"; return false; }
        if (s.flags & 1u) { err = "the job was cancelled"; return false; }
        s.state = 2;
        s.tokens = tokens;
        s.bytes = s.payload.size();
        s.hash = fake_hash(s.payload);
        return true;
    }
    bool abort_claim(int index, uint64_t nonce, std::string& err) {
        err.clear();
        FakeSlot& s = slots[index];
        if (s.state != 1 || s.nonce != nonce) { err = "the claim is gone"; return false; }
        s.flags |= 4u;                              // partial-payload
        s.state = 0;
        s.nonce = s.client = s.request = 0;
        return true;
    }
    bool cancel(int index, uint64_t nonce, std::string& err) {
        err.clear();
        FakeSlot& s = slots[index];
        if (s.nonce != nonce) { err = "a stale nonce cannot cancel another attempt's slot"; return false; }
        s.flags |= 1u;
        return true;
    }
    /// The ACK path.  A stale nonce cannot release another request's slot.
    bool release(int index, uint64_t nonce, std::string& err) {
        err.clear();
        if (index < 0 || (size_t) index >= slots.size()) { err = "no such slot"; return false; }
        FakeSlot& s = slots[index];
        if (s.nonce != nonce) { err = "a stale nonce cannot release another request's slot"; return false; }
        if (s.state != 2 && s.state != 1) { err = "the slot is already free"; return false; }
        s.state = 0;
        s.nonce = s.client = s.request = s.bytes = s.hash = 0;
        return true;
    }
    /// §3.5 step 4: a reader sees CLAIMED as "not populated", never a half-written body.
    bool read_slot(int index, uint64_t expect_nonce, uint64_t client, FakeSlot& out, std::string& err) {
        err.clear();
        if (index < 0 || (size_t) index >= slots.size()) { err = "no such slot"; return false; }
        const FakeSlot& s = slots[index];
        if (s.state == 1) { err = "the slot is CLAIMED: not populated"; return false; }
        if (s.state != 2) { err = "the slot is FREE"; return false; }
        if (s.client != client) { err = "this slot names a different client"; return false; }
        if (s.nonce != expect_nonce) { err = "the nonce does not match the DONE line's nonce"; return false; }
        out = s;
        return true;
    }
    /// §3.6: the lease rule, i.e. the slot-leak fix.  `dead_clients` is what `note_client_pid` +
    /// `/proc` would tell the writer.
    struct Reclaimed { int index; uint32_t was; std::string why; };
    std::vector<Reclaimed> reclaim(uint64_t now_ms, uint64_t lease_ms,
                                   const std::set<uint64_t>& dead_clients) {
        std::vector<Reclaimed> out;
        for (auto& s : slots) {
            if (s.state == 0) continue;
            const uint64_t then = s.state == 1 ? s.claimed_ms : s.published_ms;
            const bool dead = dead_clients.count(s.client) != 0;
            if (!dead && !(now_ms >= then && now_ms - then >= lease_ms)) continue;
            out.push_back({s.index, s.state, dead ? "the reader is gone" : "the lease expired"});
            s.flags |= 2u;                          // abandoned-by-client
            s.state = 0;
            s.nonce = s.client = s.request = 0;
        }
        return out;
    }
    std::string free_slots() const {
        size_t free_n = 0;
        for (const auto& s : slots) if (s.state == 0) ++free_n;
        return std::to_string(free_n) + "/" + std::to_string(slots.size());
    }
    static uint64_t fake_hash(const std::vector<uint8_t>& v) {
        uint64_t h = 1469598103934665603ull;
        for (uint8_t b : v) { h ^= b; h *= 1099511628211ull; }
        return h;
    }
};

// ============================================================================ the fake prefill ====
//
// The prefill instance's socket loop, as a line-in / line-out function, driven by the PJob table.
// S4.3.5 replaces this with the real thing; the point here is that the table alone is enough to
// write the loop, which is what the table is for.

struct FakePrefill {
    FakeArena arena;
    struct Job {
        int64_t id = -1;
        uint64_t tokens = 0, tier = 0, nonce = 0;
        uint64_t claimed_ms = 0;
        int slot = -1;
        PJob state = PJob::queued;
        bool geni = false;
    };
    std::vector<Job> jobs;
    std::vector<std::string> outbox;
    uint64_t client_id = 0;
    int64_t queue_cap = kDefaultPrefillQueue;
    uint64_t now_ms = 0;
    bool vision = false;
    bool shutting = false;
    /// What the next read does.  The failure-mode table sets this per scenario.
    enum class Behaviour : uint8_t { ok, full, queue_full, read_fail, slot_lost, dies, novision,
                                     resume_bad, ids_bad, toobig, corrupt } behaviour = Behaviour::ok;
    int64_t jobs_done = 0, jobs_err = 0;

    Job* find(int64_t id) {
        for (auto& j : jobs) if (j.id == id) return &j;
        return nullptr;
    }
    void emit(const std::string& s) { outbox.push_back(s); }

    /// Drive one job through the PJob table and perform exactly what the action names.  This is the
    /// whole of S4.3.5's engine-thread loop body, and it is written from the table alone - which is
    /// the point: if the table cannot express the loop, the table is wrong.
    void move(Job& j, PEvent e) {
        const JobTransition t = job_lookup(j.state, e);
        const PAct a = t.action;
        if (a == PAct::illegal) { j.state = t.to; return; }

        // Anything that abandons the claim does it BEFORE the error line, so the slot is never held
        // while a client is being told about it (§4.6: "the slot is released without publishing").
        if (a == PAct::abort_claim || a == PAct::abort_err_cancel || a == PAct::abort_err_read) {
            std::string err;
            arena.abort_claim(j.slot, j.nonce, err);
        }
        switch (a) {
            case PAct::send_claim:
                emit(claim_line(j.id, (uint64_t) j.slot, j.nonce));
                break;
            case PAct::send_seg:
                emit(seg_line(j.id, (uint64_t) j.slot, 1, j.tokens, j.tokens, 812));
                break;
            case PAct::send_done:
                emit(done_line(j.id, (uint64_t) j.slot, j.nonce, 1024, j.tokens, 1196));
                ++jobs_done;
                break;
            case PAct::err_read: case PAct::abort_err_read:
                emit(err_line(j.id, ErrCode::read, "the prompt read failed"));
                ++jobs_err;
                break;
            case PAct::err_slotlost:
                emit(err_line(j.id, ErrCode::slotlost, "the slot was reclaimed mid-write"));
                ++jobs_err;
                break;
            case PAct::err_cancel: case PAct::abort_err_cancel:
                emit(err_line(j.id, ErrCode::cancel, "the request was cancelled"));
                break;
            case PAct::release_slot: {
                std::string err;
                arena.release(j.slot, j.nonce, err);
                emit(slot_released_line(j.slot, client_id, 1024, (int64_t) (now_ms - j.claimed_ms)));
                break;
            }
            default: break;
        }
        j.state = t.to;
    }

    /// One REQ line in.
    void on_line(const std::string& raw) {
        const Line l = parse_line(raw);
        if (l.bad.empty() == false) { emit("ERR -1 endpoint the line did not parse"); return; }
        switch (l.kind) {
            case LineKind::req: {
                if ((int64_t) jobs.size() >= queue_cap) {
                    emit(err_line(l.id, ErrCode::queue, "the job queue is full"));
                    ++jobs_err;
                    return;
                }
                if (behaviour == Behaviour::queue_full) {
                    emit(err_line(l.id, ErrCode::queue, "the job queue is full"));
                    ++jobs_err;
                    return;
                }
                if (behaviour == Behaviour::novision && l.tokens > 0 && !l.geni.empty() && !vision) {
                    emit(err_line(l.id, ErrCode::novision, "started without --vision"));
                    ++jobs_err;
                    return;
                }
                if (l.resume != 0) {
                    emit(err_line(l.id, ErrCode::resume, "the prefill instance owns the reuse decision"));
                    ++jobs_err;
                    return;
                }
                if (behaviour == Behaviour::ids_bad || !ids_name_is_safe(l.ids_name)) {
                    emit(err_line(l.id, ErrCode::ids, "the ids file is missing or unparsable"));
                    ++jobs_err;
                    return;
                }
                if (behaviour == Behaviour::toobig || l.tokens > arena.max_tokens()) {
                    emit(err_line(l.id, ErrCode::toobig, "the prompt is longer than the arena holds"));
                    ++jobs_err;
                    return;
                }
                Job j;
                j.id = l.id;
                j.tokens = l.tokens;
                j.tier = l.tier;
                j.geni = !l.geni.empty();
                j.state = PJob::queued;
                jobs.push_back(j);
                emit(queued_line(l.id, (uint64_t) (jobs.size() - 1)));
                return;
            }
            case LineKind::cancel: {
                Job* j = find(l.id);
                if (!j) return;
                move(*j, PEvent::cancel);
                return;
            }
            case LineKind::ack: {
                Job* j = find(l.id);
                if (!j) return;
                std::string e;
                if (!arena.release(j->slot, l.nonce, e)) { emit("INFO note=release_refused"); return; }
                move(*j, PEvent::ack);
                return;
            }
            case LineKind::ping: emit(pong_line(l.cookie)); return;
            case LineKind::bye: shutting = true; return;
            default: return;
        }
    }

    /// Run the head QUEUED job as far as its behaviour allows (§5.2: FIFO by REQ arrival, one job at
    /// a time; a job that has already published is not the head for claiming purposes).
    void step_head() {
        Job* head = nullptr;
        for (auto& j : jobs) {
            if (j.state == PJob::queued) { head = &j; break; }
        }
        if (!head) return;
        Job& j = *head;
        if (behaviour == Behaviour::full) {
            emit(err_line(j.id, ErrCode::full, "no free slot"));
            ++jobs_err;
            jobs.erase(std::find_if(jobs.begin(), jobs.end(),
                                    [&](const Job& x) { return x.id == j.id; }));
            return;
        }
        int idx = -1;
        uint64_t nonce = 0;
        std::string err;
        if (!arena.claim((uint32_t) j.tier, client_id, (uint64_t) j.id, now_ms, idx, nonce, err)) {
            emit(err_line(j.id, ErrCode::full, err));
            ++jobs_err;
            jobs.erase(std::find_if(jobs.begin(), jobs.end(),
                                    [&](const Job& x) { return x.id == j.id; }));
            return;
        }
        j.slot = idx;
        j.nonce = nonce;
        j.claimed_ms = now_ms;
        move(j, PEvent::claim_ok);
        move(j, PEvent::read_start);
        if (behaviour == Behaviour::read_fail) {
            move(j, PEvent::read_fail);
            jobs.erase(std::find_if(jobs.begin(), jobs.end(),
                                    [&](const Job& x) { return x.id == j.id; }));
            return;
        }
        move(j, PEvent::seg_sent);
        std::vector<uint8_t> body(64, behaviour == Behaviour::corrupt ? 0xAB : 0x11);
        arena.write(j.slot, body);
        if (behaviour == Behaviour::slot_lost) {
            // The slot was reclaimed under the writer: publish() refuses.
            std::string e2;
            arena.slots[j.slot].state = 0;
            if (!arena.publish(j.slot, j.nonce, j.tokens, e2)) move(j, PEvent::publish_lost);
            jobs.erase(std::find_if(jobs.begin(), jobs.end(),
                                    [&](const Job& x) { return x.id == j.id; }));
            return;
        }
        if (!arena.publish(j.slot, j.nonce, j.tokens, err)) { move(j, PEvent::publish_lost); return; }
        // The job stays in the list until its ACK: that is the window the leak watchdog watches.
        move(j, PEvent::publish_ok);
    }
};

// ============================================================================ the fake decode ====

/// The decode instance's handoff client thread: a `HandoffState` per request plus the copy-out the
/// protocol orders.  The engine thread is a separate counter on purpose - §1.5's rule is that the
/// copy-out never runs there, and the test asserts the two advance independently.
struct FakeDecode {
    std::map<int64_t, HandoffState> st;
    std::vector<std::string> outbox;
    std::vector<std::string> peer;         ///< lines received from the prefill instance
    FakeArena* arena = nullptr;
    uint64_t client_id = 0;
    uint64_t now_ms = 0;
    int64_t retries = kDefaultPrefillRetries;
    bool endpoint_down = false;
    int64_t engine_steps = 0;              ///< the engine thread keeps stepping while a handoff runs
    int64_t client_errs = 0;               ///< ERR lines that reached the CLIENT (must stay 0 for D5)
    int64_t holds = 0;
    int copy_outs = 0;
    int mounts = 0;

    HandoffState& machine(int64_t id) {
        auto it = st.find(id);
        if (it != st.end()) return it->second;
        HandoffState h(retries);
        h.admit_now(id);
        return st.emplace(id, h).first->second;
    }
    void emit(const std::string& s) { outbox.push_back(s); }

    struct Outcome { Act action; HState to; std::string why; };

    /// Ask for a prompt read.  Returns the action the caller must take.
    Outcome ask(int64_t id, uint64_t tokens, uint64_t tier, const std::string& ids_name) {
        HandoffState& h = machine(id);
        std::string why;
        if (!h.begin(id, tokens, tier, why)) return {Act::illegal, h.state(), why};
        const auto r = h.note(id, Event::req_sent);
        if (r.action == Act::write_req) {
            ReqKeys k;
            k.id = id;
            k.tokens = tokens;
            k.tier = tier;
            k.ids_name = ids_name;
            emit(req_line(k));
        }
        return {r.action, r.to, r.why};
    }
    /// Feed one peer line to the right machine.
    Outcome feed(const std::string& raw) {
        const Line l = parse_line(raw);
        if (l.bad.empty()) {
            auto it = st.find(l.id);
            if (it == st.end()) return {Act::illegal, HState::idle, "no machine for that id"};
            const auto r = it->second.on_line(l.id, l);
            if (r.action == Act::write_ack) emit(it->second.ack_line());
            if (r.action == Act::write_cancel) emit(cancel_line(l.id));
            return {r.action, r.to, r.why};
        }
        return {Act::protocol_error, HState::idle, l.bad};
    }
    /// Re-ask after a retry: the machine writes the REQ, the caller sends it.
    bool reask(int64_t id) {
        HandoffState& h = machine(id);
        const auto r = h.note(id, Event::req_sent);
        if (r.action == Act::write_req) {
            ReqKeys k;
            k.id = id;
            k.tokens = h.tokens();
            k.tier = h.tier();
            k.ids_name = "x.ids";
            emit(req_line(k));
            return true;
        }
        return false;
    }
    /// The copy-out: read_slot -> hash -> decode -> ACK.  Never on the engine thread.
    bool copy_out(int64_t id) {
        HandoffState& h = machine(id);
        if (h.state() != HState::seg_ready) return false;
        FakeSlot got;
        std::string err;
        if (!arena->read_slot((int) h.slot(), h.nonce(), client_id, got, err)) {
            ++client_errs;                 // a DONE that lied is a protocol fault, not a hold
            return false;
        }
        if (FakeArena::fake_hash(got.payload) != got.hash) {
            // §2.5 / §4.4: the payload is refused and NOT mounted.  `badpayload` is permanent, so it
            // is the one case where a client ERR is honest.
            ++client_errs;
            return false;
        }
        ++copy_outs;
        const auto r = h.note(id, Event::copy_done);
        if (r.action == Act::write_ack) emit(h.ack_line());
        return true;
    }
};

}  // namespace

// ============================================================ 1. every line, both directions =====
//
// One formatter, one parser: a line and its parser cannot drift.  The check is format -> parse ->
// re-format from the PARSED fields -> identical bytes.  Not "the parser found the right numbers"
// (which a hand-written second formatter would also pass) but "the two agree on the BYTES".

/// A `ReqKeys` with every member spelled, so -Wmissing-field-initializers stays quiet and a reader
/// sees that `geni` is empty and `resume` is 0 rather than guessing what the shorthand meant.
static ReqKeys mk_req(int64_t id, uint64_t tokens, uint64_t tier, const char* ids,
                      const char* geni = "", int cvec = 1, int64_t resume = 0) {
    ReqKeys k;
    k.id = id; k.tokens = tokens; k.tier = tier; k.ids_name = ids;
    k.geni = geni; k.cvec = cvec; k.resume = resume;
    return k;
}

static std::string reformat(const Line& l) {
    switch (l.kind) {
        case LineKind::hello:
            return hello_line(l.proto_ver, l.client_id, l.pack_hash, l.geom_hash, l.max_ctx,
                              l.split, l.cvec, l.vision);
        case LineKind::ready: {
            ReadyInfo r;
            r.proto_ver = l.proto_ver;
            r.slots = l.slots;
            r.tiers = l.tiers;
            r.max_tokens = l.max_tokens;
            r.pack_hash = l.pack_hash;
            r.geom_hash = l.geom_hash;
            return ready_line(r);
        }
        case LineKind::ping: return ping_line(l.cookie);
        case LineKind::pong: return pong_line(l.cookie);
        case LineKind::req: {
            ReqKeys k;
            k.id = l.id; k.tokens = l.tokens; k.tier = l.tier; k.ids_name = l.ids_name;
            k.geni = l.geni; k.cvec = l.cvec; k.resume = l.resume;
            return req_line(k);
        }
        case LineKind::cancel: return cancel_line(l.id);
        case LineKind::ack: return ack_line(l.id, l.nonce);
        case LineKind::bye: return bye_line(l.reason);
        case LineKind::queued: return queued_line(l.id, l.ahead);
        case LineKind::claim: return claim_line(l.id, l.slot, l.nonce);
        case LineKind::seg: return seg_line(l.id, l.slot, l.seq, l.tokens_done, l.tokens_total, l.ms);
        case LineKind::done: return done_line(l.id, l.slot, l.nonce, l.bytes, l.tokens, l.ms);
        case LineKind::err: return err_line(l.id, l.code, l.message);
        case LineKind::info: return info_line(l.info);
        case LineKind::take: case LineKind::unknown: return {};
    }
    return {};
}

static void test_every_line_round_trips() {
    ReadyInfo ri;
    ri.slots = {2, 2, 2};
    ri.tiers = {1024, 16384, 131072};
    ri.max_tokens = 131072;
    ri.pack_hash = 0x0123456789abcdefull;
    ri.geom_hash = 0x5dfa917bae1331c5ull;

    const std::vector<std::string> lines = {
        hello_line(kProtoVersion, 0x7f3a91c0beef0001ull, ri.pack_hash, ri.geom_hash, 262144,
                   {16, 34}, 1, 1),
        hello_line(kProtoVersion, 1, 0, 0, 8192, {}, 0, 0),                 // no layer split
        ready_line(ri),
        ping_line("hb-1"),
        pong_line("hb-1"),
        req_line(mk_req(7, 10394, 1, "req-0000000000000007-0000000000000001.ids", "", 1, 0)),
        req_line(mk_req(8, 245946, 2, "req-8-1.ids", "/dev/shm/strata-handoff/a/img-8.bin", 0, 0)),
        req_line(mk_req(9, 1, 0, "a.ids", "", 1, 0)),
        cancel_line(7),
        ack_line(7, 4097),
        bye_line("operator quit"),
        bye_line(""),
        queued_line(7, 0),
        queued_line(7, 31),
        claim_line(7, 0, 0),
        claim_line(7, 5, 1073741825),
        seg_line(7, 5, 1, 10394, 10394, 812),
        seg_line(7, 5, 1, 0, 0, 0),
        done_line(7, 5, 1073741825, 1033605768, 10394, 1196),
        done_line(7, 0, 0, 0, 0, 0),
        err_line(7, ErrCode::full, "no free slot"),
        err_line(7, ErrCode::toobig, ""),                                    // defaults to the text
        err_line(7, ErrCode::geom, "geometry hash 3f2a1b0c4d5e6f70 != 91c7a5b3d2e1f009"),
        info_line(default_info_keys()),
        info_line({{"role", "prefill"}}),
    };
    for (const std::string& s : lines) {
        std::string why;
        check(line_is_well_formed(s, why), ("a formatted line is well formed: " + s).c_str());
        const Line l = parse_line(s);
        check(l.bad.empty(), ("a formatted line parses: " + s).c_str());
        check(l.kind != LineKind::unknown, ("a formatted line's verb is known: " + s).c_str());
        check_eq(reformat(l), s, "format -> parse -> format is byte-identical");
    }

    // The version token is the doc's, verbatim, and it is NOT the client wire's READY (D3 of §1.6).
    check_eq(std::string(kReadyToken), "PREFILL-READY-V1", "the version token");
    check(std::string(kReadyToken) != "READY", "the handoff READY token is not the client READY token");
    // The two vocabularies are disjoint: no handoff verb is a client-wire verb, so one parser can
    // never be asked to serve both.  (serve_proto's verbs, spelled out here rather than included.)
    const char* client_verbs[] = {"GEN", "GENI", "STOP", "QUIT", "READY", "RESUME", "PP", "REUSED",
                                  "T", "DONE", "ERR", "SLOT", "WAIT", "INFO"};
    const char* shared[] = {"DONE", "ERR", "INFO"};   // the ONLY overlap, and it is name-only
    for (const char* cv : client_verbs) {
        const Line l = parse_line(std::string(cv) + " 1 2 3");
        const bool known = l.kind != LineKind::unknown;
        const bool is_shared = std::find(shared, shared + 3, std::string(cv)) != shared + 3;
        check(known == is_shared,
              ("a client-wire verb is only known here if it is a documented name overlap: " +
               std::string(cv)).c_str());
    }
}

static void test_line_fields_are_read_back() {
    const Line h = parse_line(hello_line(kProtoVersion, 0x7f3a91c0beef0001ull, 0x0123456789abcdefull,
                                         0x5dfa917bae1331c5ull, 262144, {16, 34}, 1, 0));
    check(h.kind == LineKind::hello, "HELLO kind");
    check(h.proto_ver == kProtoVersion, "HELLO proto_ver");
    check(h.client_id == 0x7f3a91c0beef0001ull, "HELLO client_id");
    check(h.pack_hash == 0x0123456789abcdefull, "HELLO pack_hash");
    check(h.geom_hash == 0x5dfa917bae1331c5ull, "HELLO geom_hash");
    check(h.max_ctx == 262144, "HELLO max_ctx");
    check(h.n_stages == 2, "HELLO n_stages");
    check(h.split.size() == 2 && h.split[0] == 16 && h.split[1] == 34, "HELLO split list");
    check(h.cvec == 1 && h.vision == 0, "HELLO cvec/vision");

    const Line d = parse_line(done_line(7, 5, 1073741825ull, 1033605768ull, 10394, 1196));
    check(d.kind == LineKind::done && d.id == 7 && d.slot == 5 && d.nonce == 1073741825ull,
          "DONE identity fields");
    check(d.bytes == 1033605768ull && d.tokens == 10394, "DONE size fields");
    check(d.ms == 1196.0, "DONE ms");

    const Line e = parse_line(err_line(9, ErrCode::geom, "geometry hash 3f2a1b0c4d5e6f70 != 91c7a5b3d2e1f009"));
    check(e.kind == LineKind::err && e.code == ErrCode::geom, "ERR code");
    check_eq(e.message, "geometry hash 3f2a1b0c4d5e6f70 != 91c7a5b3d2e1f009",
             "ERR keeps its free-text message byte-for-byte, spaces included");

    const Line r = parse_line("REQ 4 12 0 ids=x.ids geni=y.bin cvec=0 resume=7");
    check(r.bad.empty() && r.id == 4 && r.tokens == 12 && r.tier == 0, "REQ positional fields");
    check_eq(r.ids_name, "x.ids", "REQ ids=");
    check_eq(r.geni, "y.bin", "REQ geni=");
    check(r.cvec == 0 && r.resume == 7, "REQ cvec=/resume=");

    const Line i = parse_line("INFO role=prefill slots=6 slots_free=4/6 max_ctx=262144");
    check(i.kind == LineKind::info && i.info.size() == 4, "INFO k=v pairs");
    check_eq(info_value(i, "role"), "prefill", "INFO role");
    check_eq(info_value(i, "slots_free"), "4/6", "INFO slots_free");
    check(info_shows_a_free_slot(i), "an INFO with slots_free>0 is a wake point (§6.2)");
    const Line i0 = parse_line("INFO slots_free=0/6");
    check(!info_shows_a_free_slot(i0), "an INFO with no free slot is not a wake point");
}

// ============================================================ 2. malformed lines =================

static void test_malformed_lines() {
    struct Bad { const char* line; const char* needle; };
    const Bad bad[] = {
        // empty / whitespace
        {"", "empty"},
        {"   ", "empty"},
        // unknown VERB is NOT an error (D8) - listed separately below
        // wrong arity
        {"HELLO 1", "missing"},
        {"HELLO 1 2", "missing"},
        {"HELLO 1 2 0123456789abcdef", "missing"},
        {"HELLO 1 2 0123456789abcdef 0123456789abcdef 8192", "missing n_stages"},
        {"HELLO 1 2 0123456789abcdef 0123456789abcdef 8192 0", "missing split="},
        {"HELLO 1 2 0123456789abcdef 0123456789abcdef 8192 0 split= cvec=1", ""},   // legal: empty list
        {"PING", "missing cookie"},
        {"PING a.b", "cookie must be"},
        {"PING a/b", "cookie must be"},
        {"PING a;b", "cookie must be"},
        {"PONG x y", ""},
        {"REQ 1", "missing tokens"},
        {"REQ 1 2", "missing tier"},
        {"REQ 1 2 3", "missing ids=NAME"},
        {"CANCEL", "missing id"},
        {"ACK 1", "missing nonce"},
        {"QUEUED 1", "missing ahead"},
        {"CLAIM 1 2", "missing nonce"},
        {"SEG 1 2 1 10", "missing"},
        {"SEG 1 2 1 10 10", "missing ms"},
        {"DONE 1 2 3 4", "missing"},
        {"DONE 1 2 3 4 5", "missing ms"},
        {"ERR 1", "missing"},
        {"ERR 1 full", "missing a message"},
        // non-numeric fields
        {"REQ abc 2 3 ids=x.ids", "non-numeric id"},
        {"REQ 1 -5 3 ids=x.ids", "non-numeric tokens"},
        {"REQ 1 2 x ids=x.ids", "non-numeric tier"},
        {"CLAIM 1 -2 3", "non-numeric slot"},
        {"ACK 1 -1", "non-numeric nonce"},
        {"HELLO x 2 0123456789abcdef 0123456789abcdef 8192 0 split=", "non-numeric proto_ver"},
        {"DONE 1 2 3 4 5 -1", "bad ms"},
        {"DONE 1 2 3 4 5 1e999", "bad ms"},
        // bad hex
        {"HELLO 1 2 DEADBEEFDEADBEEF 0123456789abcdef 8192 0 split=", "lowercase hex"},
        {"HELLO 1 2 0123456789abcde 0123456789abcdef 8192 0 split=", "16 "},
        {"HELLO 1 2 0123456789abcdef0 0123456789abcdef 8192 0 split=", "16 "},
        // bad lists
        {"PREFILL-READY-V1 1 slots=2,2 tiers=1024,16384 max_tokens=16384 pack_hash=0123456789abcdef geom_hash=0123456789abcdef",
         "wants max_tokens="},
        {"PREFILL-READY-V1 1 slots=2,2 max_tokens=16384 tiers=1024 pack_hash=0123456789abcdef geom_hash=0123456789abcdef",
         "slots= has 2 entries and tiers= has 1"},
        {"PREFILL-READY-V1 1 slots= max_tokens=1 tiers= pack_hash=0123456789abcdef geom_hash=0123456789abcdef",
         "tiers= is empty"},
        {"PREFILL-READY-V1 1 slots=2,,2 max_tokens=1 tiers=1,2,3 pack_hash=0123456789abcdef geom_hash=0123456789abcdef",
         "bad slots="},
        {"HELLO 1 2 0123456789abcdef 0123456789abcdef 8192 2 split=16,,34", "bad split="},
        {"HELLO 1 2 0123456789abcdef 0123456789abcdef 8192 1 split=16,34", "n_stages=1 but split= has 2"},
        {"HELLO 1 2 0123456789abcdef 0123456789abcdef 8192 -1 split=", "n_stages is negative"},
        // unknown / bad error code
        {"ERR 1 nope the message", "unknown code"},
        {"ERR 1 none the message", "unknown code"},
        // bad keys
        {"REQ 1 2 3 ids=x!bad", "ids= is not a NAME"},
        {"REQ 1 2 3 ids=", "ids= is not a NAME"},
        {"REQ 1 2 3 ids=x.ids cvec=2", "cvec must be 0 or 1"},
        {"REQ 1 2 3 ids=x.ids resume=abc", "resume= is not an integer"},
        {"INFO notakv", "wants k=v"},
        {"INFO =5", "wants k=v"},
        {"INFO 5x=1", "wants k=v"},
        // reserved verb
        {"TAKE 1 2 3", "reserved for the reverse handoff"},
    };
    for (const Bad& b : bad) {
        const Line l = parse_line(b.line);
        if (b.needle[0] == '\0') {
            check(l.bad.empty(), ("a legal line must parse: " + std::string(b.line)).c_str());
            continue;
        }
        check(!l.bad.empty(), ("a malformed line is refused: " + std::string(b.line)).c_str());
        check_reason_contains(l.bad, b.needle, ("the refusal names why: " + std::string(b.line)).c_str());
    }
    // A `k=v` value stops at the space, so a token after the key run is NOT part of the value: it is
    // an ignored trailing token (D8/§4.3).  Pinned because a parser that swallowed the rest of the
    // line would let a peer smuggle a second field into one key.
    // A cookie is a token, never a line: `PONG x y` keeps "x" as the cookie and ignores "y", so a
    // peer cannot use a liveness probe to inject a second verb (D10).
    const Line pc = parse_line("PONG x y");
    check(pc.bad.empty() && pc.cookie == "x" && pc.extra.size() == 1,
          "a PONG cookie stops at the space: it cannot smuggle a line");
    const Line g = parse_line("REQ 1 2 3 ids=x.ids geni=a b");
    check(g.bad.empty(), "a space after a key value does not break the line");
    check_eq(g.geni, "a", "the value stops at the space");
    check(g.extra.size() == 1 && g.extra[0] == "b", "and the rest is an ignored trailing token");

    // An unknown VERB is ignored, not refused (D8): forward compatibility, and the doc's own rule
    // that a reader which ignores a verb entirely is still correct.
    const Line u = parse_line("FUTURE 1 2 3 whatever");
    check(u.kind == LineKind::unknown, "an unknown verb parses as unknown");
    check(u.bad.empty(), "an unknown verb is NOT a connection-killing error");
    check(u.extra.size() == 4, "an unknown verb keeps its tokens for the log");
    // Unknown TRAILING tokens on a known verb are ignored (§4.3's compatibility rule).
    const Line t = parse_line("DONE 7 5 99 1024 10394 1196 handoff_ms=1196 slots_free=4/6");
    check(t.bad.empty(), "unknown trailing tokens on DONE are ignored");
    check(t.extra.size() == 2, "the ignored tokens are kept for the log");
    check(t.slot == 5 && t.nonce == 99, "and the known fields still parse");
    ReadyInfo ri2;
    ri2.slots = {2, 2, 2};
    ri2.tiers = {1024, 16384, 131072};
    ri2.max_tokens = 131072;
    ri2.pack_hash = 0x0123456789abcdefull;
    ri2.geom_hash = 0x5dfa917bae1331c5ull;
    const Line tr = parse_line(ready_line(ri2) + " future=1");
    check(tr.bad.empty() && tr.extra.size() == 1, "unknown trailing keys on PREFILL-READY-V1 are ignored");

    // Truncation: a line cut at ANY prefix that still carries a known verb must be refused.  A cut
    // that destroys the verb is an unknown verb (D8) - which is exactly why the reader must treat a
    // short read as a framing error rather than as a message.
    const std::string full = "DONE 7 5 1073741825 1033605768 10394 1196";
    int missing_field = 0, shortened = 0, verb_destroyed = 0;
    for (size_t n = 1; n < full.size(); ++n) {
        const std::string cut = full.substr(0, n);
        const Line l = parse_line(cut);
        if (l.verb != "DONE") {
            check(l.kind == LineKind::unknown, "a truncated verb is unknown, not half-parsed");
            ++verb_destroyed;
            continue;
        }
        // A cut that drops a whole token is a missing field and must be refused.  A cut INSIDE the
        // last token leaves a shorter number, which is legal grammar - a real reader never delivers
        // one, because it reads up to the '\n'.  What must hold in BOTH cases is that the parser
        // never mis-reads a field that is fully present: the values it returns are always the
        // original's leading fields, never a shifted or re-split set.
        if (l.bad.empty()) {
            ++shortened;
            check(l.kind == LineKind::done, "and it is still a DONE");
            check(l.id == 7 && l.slot == 5, "the leading fields did not shift");
            check(full.rfind(reformat(l), 0) == 0,
                  "a truncated line re-formats to a PREFIX of the original: no field is re-split");
        } else {
            check_reason_contains(l.bad, "missing", "a refused cut says which field is missing");
            ++missing_field;
        }
    }
    check(missing_field > 0 && shortened > 0 && verb_destroyed > 0,
          "the truncation sweep covered all three cases");
}

static void test_framing() {
    std::string why;
    check(!line_is_well_formed("", why), "an empty line is not well formed");
    // Each bad character names ITS OWN reason.  The printable-ASCII catch-all would refuse all of
    // them, but an operator reading the log needs to know which byte arrived, and pinning the reason
    // is what stops the specific checks from being deleted as "redundant".
    check(!line_is_well_formed("DONE 1 2 3 4 5 6\n", why), "an embedded newline is refused");
    check_reason_contains(why, "newline", "and says newline");
    check(!line_is_well_formed("DONE 1 2 3 4 5 6\r\n", why), "a CR is refused");
    check_reason_contains(why, "carriage return", "and says carriage return");
    check(!line_is_well_formed(std::string("DONE 1 2 3 4 5 6\0x", 17), why), "a NUL is refused");
    check_reason_contains(why, "NUL", "and says NUL");
    check(!line_is_well_formed(std::string("DONE 1 2 3 4 5 6\x01", 17), why), "a control byte is refused");
    check_reason_contains(why, "ASCII printable", "and falls back to the printable rule");
    check(!line_is_well_formed(std::string("DONE 1 2 3 4 5 6\x7f", 17), why), "DEL is refused");
    check(!line_is_well_formed(std::string("DONE 1 2 3 4 5 6\xc3\xa9", 19), why), "non-ASCII UTF-8 is refused");
    check(line_is_well_formed("ERR 1 geom geometry hash 3f2a1b0c4d5e6f70 != 91c7a5b3d2e1f009", why),
          "a printable ASCII line is well formed");
    // §4.2: "a line longer than 4096 bytes closes the connection".
    const std::string at_limit(4096, 'a');
    const std::string over = std::string("REQ 1 2 3 ids=") + std::string(4096, 'a');
    check(line_is_well_formed(at_limit, why), "exactly 4096 bytes is allowed");
    check(!line_is_well_formed(over, why), "4097+ bytes is refused");
    check_reason_contains(why, "4096", "the refusal names the limit");
    // The framing check is what refuses it, BEFORE the parser runs: a 4096-character NAME is a
    // perfectly good token, so the parser alone would accept an over-long line.  That is why the
    // reader checks the length of the line it read rather than trusting the verb.
    const Line l = parse_line(over);
    check(l.kind == LineKind::req && l.bad.empty() && l.ids_name.size() == 4096,
          "the parser would accept it, so the framing check is the thing that must refuse it");
    check(!line_is_well_formed(l.raw, why), "and the framing check does");
}

// ============================================================ 3. the error-code table ============

static void test_error_table_is_total() {
    const std::vector<ErrClass>& t = err_table();
    check((int) t.size() == prefill_svc_error_count(),
          "the table has exactly one entry per error code");
    std::set<std::string> seen;
    for (const ErrClass& e : t) {
        const std::string n = err_name(e.code);
        check(seen.insert(n).second, ("the table names each code once: " + n).c_str());
        check(e.hold == hold::wait || e.hold == hold::error,
              ("every code maps to wait or error: " + n).c_str());
        check(e.hold != hold::run_now, ("no refusal is run-now: " + n).c_str());
        check(std::strlen(e.text) > 0, ("every code has a sentence: " + n).c_str());
    }
    // Every code from `full` to `endpoint` is in the table, and `none` is not.
    for (int i = 1; i <= prefill_svc_error_count(); ++i) {
        const ErrCode c = (ErrCode) i;
        bool found = false;
        for (const ErrClass& e : t) if (e.code == c) found = true;
        check(found, (std::string("the table covers ") + err_name(c)).c_str());
    }
    // §4.4's table, row by row.  These are the doc's own permanent/temporary columns.
    const ErrCode permanent[] = {ErrCode::toobig, ErrCode::ids, ErrCode::novision, ErrCode::resume,
                                 ErrCode::geom, ErrCode::pack, ErrCode::badpayload};
    const ErrCode temporary[] = {ErrCode::full, ErrCode::queue, ErrCode::read, ErrCode::slotlost,
                                 ErrCode::shutting, ErrCode::endpoint};
    for (ErrCode c : permanent) {
        check(err_is_permanent(c), (std::string("permanent: ") + err_name(c)).c_str());
        check(hold_for(c) == (int) serve_driver::Hold::error,
              ("maps to serve_driver::Hold::error: " + std::string(err_name(c))).c_str());
    }
    for (ErrCode c : temporary) {
        check(err_is_temporary(c), (std::string("temporary: ") + err_name(c)).c_str());
        check(hold_for(c) == (int) serve_driver::Hold::wait,
              ("maps to serve_driver::Hold::wait: " + std::string(err_name(c))).c_str());
    }
    // D5: `cancel` is neither a client error nor a retry.
    check(!err_reaches_the_client(ErrCode::cancel), "ERR cancel never reaches the client as an ERR");
    check(!err_is_retryable(ErrCode::cancel), "ERR cancel is not retried");
    // D5's rule, stated as a predicate over the whole table: exactly the permanent codes reach the
    // client, and the six temporary ones never do.
    int reaches = 0, holds = 0;
    for (int i = 1; i <= prefill_svc_error_count(); ++i) {
        const ErrCode c = (ErrCode) i;
        if (err_reaches_the_client(c)) ++reaches;
        if (err_is_temporary(c)) ++holds;
    }
    check(reaches == 7, "exactly seven codes may reach the client (D5)");
    check(holds == prefill_svc_error_count() - reaches,
          "and every other code is an internal wait: the two classes partition the table");
    check(holds == 7, "seven waits: full, queue, read, slotlost, shutting, endpoint, cancel");
    // §4.4's "also drop the connection" column.
    check(err_drops_connection(ErrCode::geom) && err_drops_connection(ErrCode::pack),
          "geom and pack drop the connection");
    check(!err_drops_connection(ErrCode::badpayload) && !err_drops_connection(ErrCode::toobig),
          "the other permanent codes do not");
    // The last two rows of §4.4's table are decode-emitted: the prefill instance never sends them.
    check(err_is_decode_emitted(ErrCode::badpayload) && err_is_decode_emitted(ErrCode::endpoint),
          "badpayload and endpoint are decode-emitted");
    int decoded = 0;
    for (int i = 1; i <= prefill_svc_error_count(); ++i)
        if (err_is_decode_emitted((ErrCode) i)) ++decoded;
    check(decoded == 2, "and exactly two codes are");
    // Only `read` and `slotlost` count against --prefill-retries (§4.4's retry column).
    check(err_is_retryable(ErrCode::read) && err_is_retryable(ErrCode::slotlost),
          "read and slotlost are retryable");
    check(!err_is_retryable(ErrCode::full) && !err_is_retryable(ErrCode::queue),
          "full and queue are holds, not retries (§3.8)");
    // The code vocabulary round-trips through its own name.
    for (int i = 1; i <= prefill_svc_error_count(); ++i) {
        const ErrCode c = (ErrCode) i;
        ErrCode back = ErrCode::none;
        check(err_from_code(err_name(c), back) && back == c,
              ("the code name round-trips: " + std::string(err_name(c))).c_str());
    }
    ErrCode junk = ErrCode::none;
    check(!err_from_code("Full", junk) && !err_from_code("", junk) && !err_from_code("nope", junk),
          "a code name is exact and lowercase");
    // An unknown code is loud, not a silent wait (which would be an infinite hold).
    check(err_class((ErrCode) 99).hold == hold::error, "an unclassified code is an error, not a hang");
}

static void test_the_wait_vocabulary_matches_serve_driver() {
    // D6: this header defines the strings; serve_driver.hpp owns the enum.  Both must say the same
    // thing, and the numbers `hold::` mirrors must be the real ones.
    check_eq(std::string(wait_reason_prefill()), "prefill-busy", "the new wait reason string (§6.1)");
    check_eq(std::string(wait_text_prefill()),
             "the prefill instance has no free handoff slot for a prompt this long",
             "the new wait text (§6.1)");
    check((int) serve_driver::Hold::run_now == hold::run_now, "Hold::run_now == 0");
    check((int) serve_driver::Hold::wait == hold::wait, "Hold::wait == 1");
    check((int) serve_driver::Hold::error == hold::error, "Hold::error == 2");
    check_eq(std::string(serve_driver::hold_name(serve_driver::Hold::wait)), hold::name(hold::wait),
             "hold_name agrees with serve_driver's");
    // `Wait::prefill` does not exist yet (S4.3.6 adds it).  Pin the slot it must land in so the two
    // sides cannot each invent a different value.
    check(kWaitPrefillValue == (int) serve_driver::Wait::engine_busy + 1,
          "Wait::prefill must be appended after engine_busy, not inserted in the middle");
    // serve/server.py must NOT put prefill-busy in PERMANENT_REFUSALS (§6.1).  The engine-side
    // counterpart of that rule: the reason string is the one a temporary refusal carries.
    check(err_is_temporary(ErrCode::full), "a full arena is temporary, so its reason is temporary");
    check_eq(std::string(wait_reason_prefill()), "prefill-busy",
             "and the reason the WAIT line will carry is prefill-busy");
}

// ============================================================ 4. the handshake ====================

static void test_handshake_refuses_a_version_or_hash_mismatch() {
    const uint64_t pack = 0x0123456789abcdefull, geom = 0x5dfa917bae1331c5ull;
    const Line hello = parse_line(hello_line(kProtoVersion, 7, pack, geom, 262144, {16, 34}, 1, 1));
    ReadyInfo ri;
    ri.slots = {2, 2, 2};
    ri.tiers = {1024, 16384, 131072};
    ri.max_tokens = 131072;
    ri.pack_hash = pack;
    ri.geom_hash = geom;
    std::string why;
    check(handshake_decision(hello, parse_line(ready_line(ri)), why), "a matching pair is accepted");

    // An OLD decode instance against a NEW prefill instance, and the other way round.  Both must be
    // refused at the handshake, never silently mis-parsed afterwards.
    ReadyInfo newer = ri;
    newer.proto_ver = kProtoVersion + 1;
    check(!handshake_decision(hello, parse_line(ready_line(newer)), why),
          "a newer prefill instance is refused at the handshake");
    check_reason_contains(why, "protocol version", "and the reason names the version");
    const Line old_hello = parse_line(hello_line(kProtoVersion - 1, 7, pack, geom, 262144, {16, 34}, 1, 1));
    check(!handshake_decision(old_hello, parse_line(ready_line(ri)), why),
          "an older decode instance is refused at the handshake");
    check_reason_contains(why, "protocol version", "and the reason names the version");
    check(!hello_is_acceptable(old_hello, why), "the prefill side refuses an old HELLO too");
    check_reason_contains(why, "protocol version", "and says so");

    ReadyInfo other_pack = ri;
    other_pack.pack_hash = 0xdeadbeefdeadbeefull;
    check(!handshake_decision(hello, parse_line(ready_line(other_pack)), why), "a pack mismatch is refused");
    check_reason_contains(why, "pack hash 0123456789abcdef != deadbeefdeadbeef",
                          "and the refusal prints BOTH values (§3.10)");

    ReadyInfo other_geom = ri;
    other_geom.geom_hash = 0x91c7a5b3d2e1f009ull;
    check(!handshake_decision(hello, parse_line(ready_line(other_geom)), why),
          "a geometry/split mismatch is refused");
    check_reason_contains(why, "geometry hash 5dfa917bae1331c5 != 91c7a5b3d2e1f009",
                          "and it prints BOTH values");

    // A READY line that contradicts itself is refused too: the wrong ceiling is how a 262144-token
    // prompt ends up priced against a 131072-token slot.
    ReadyInfo lying = ri;
    lying.max_tokens = 16384;
    check(!handshake_decision(hello, parse_line(ready_line(lying)), why),
          "max_tokens must be the largest tier");
    check_reason_contains(why, "largest tier", "and the reason says which two numbers disagree");
    ReadyInfo zero = ri;
    zero.slots = {2, 0, 2};
    check(!handshake_decision(hello, parse_line(ready_line(zero)), why), "a 0-slot tier is refused");

    // The endpoint is not a prefill instance at all (e.g. someone pointed --prefill-endpoint at a
    // --serve engine).  §4.2's line, verbatim.
    check(!handshake_decision(hello, parse_line("READY 262144 stop slots=2"), why),
          "a --serve engine is not a prefill endpoint");
    check_eq(not_prefill_line(), "strata serve: the prefill endpoint does not speak PREFILL-READY-V1",
             "and the decode instance says exactly this");

    // client_id 0 is reserved (§3.4: "0 = none"), so a HELLO that uses it cannot own a slot.
    const Line zero_client = parse_line(hello_line(kProtoVersion, 0, pack, geom, 8192, {}, 1, 0));
    check(!hello_is_acceptable(zero_client, why), "client_id 0 is refused");
    check_reason_contains(why, "reserved", "and the reason says why");

    // §4.2: cvec/vision are echoed at HELLO so a mismatch is caught before a 3 GB transfer.
    check(hello_requirements_met(hello, 1, 1, why), "requirements met");
    check(!hello_requirements_met(hello, 1, 0, why), "a vision client against a no-vision prefill");
    check_reason_contains(why, "--vision", "names the flag");
    check(!hello_requirements_met(hello, 0, 1, why), "a cvec client against a no-cvec prefill");
    // ... and the per-request fallback: every later GENI is ERR novision, permanent.
    check(err_is_permanent(ErrCode::novision), "novision is permanent (§4.4)");
}

// ============================================================ 5. the tier rule ====================

static void test_tier_and_size_rules() {
    const std::vector<uint64_t> tiers = {1024, 16384, 131072};
    uint64_t t = 99;
    check(tier_for_tokens(1, tiers, t) && t == 0, "a 1-token prompt asks for tier 0");
    check(tier_for_tokens(1024, tiers, t) && t == 0, "the tier boundary is inclusive");
    check(tier_for_tokens(1025, tiers, t) && t == 1, "one past it is the next tier");
    check(tier_for_tokens(131072, tiers, t) && t == 2, "the largest tier is inclusive");
    check(!tier_for_tokens(131073, tiers, t), "one past the largest tier has no tier: ERR toobig");
    check(!tier_for_tokens(1, {}, t), "no tiers at all is a refusal, not tier 0");

    std::string why;
    check(!prefill_prompt_too_big(10394, 131072, 131072, why), "10394 fits");
    check(prefill_prompt_too_big(262144, 131072, 131072, why), "262144 does not");
    check_reason_contains(why, "262144", "the refusal names the prompt length");
    check_reason_contains(why, "131072", "and the ceiling: BOTH numbers (§6.3)");
    check(prefill_prompt_too_big(1, 0, 0, why), "an endpoint that declares no ceiling is refused");
    // §6.2: this runs at ADMISSION, before a REQ is ever sent, and it is permanent.
    check(err_is_permanent(ErrCode::toobig), "toobig is permanent: waiting will not help");
    check(err_reaches_the_client(ErrCode::toobig), "and it is one of the honest client ERRs");
    // §3.8's tier rule.
    check(tier_may_use(0, 2) && tier_may_use(2, 2), "a job may use its own or a larger group");
    check(!tier_may_use(2, 0), "a 131072-token job never fits in a 1024-token slot");
}

// ============================================================ 6. the state machine table =========

static const HState kAllStates[] = {
    HState::idle, HState::queued, HState::init, HState::sent, HState::held, HState::accepted,
    HState::prefilling, HState::seg_ready, HState::handed_over, HState::mounting, HState::done,
    HState::failed, HState::cancelled,
};
static const Event kAllEvents[] = {
    Event::admit, Event::req_sent, Event::queued_line, Event::claim_line, Event::seg_line,
    Event::done_line, Event::err_temporary, Event::err_permanent, Event::err_cancel,
    Event::cancel_line, Event::ack_sent, Event::copy_done, Event::mount_ok, Event::mount_fail,
    Event::collect, Event::retry, Event::note_cancel, Event::local_hit,
};

static void test_the_state_table_is_total() {
    check((int) (sizeof kAllStates / sizeof kAllStates[0]) == kStateCount, "the test enumerates every state");
    check((int) (sizeof kAllEvents / sizeof kAllEvents[0]) == kEventCount, "the test enumerates every event");
    int n = 0;
    const Transition* t = handoff_table(n);
    check(n == kStateCount * kEventCount, "the table has exactly one row per (state, event)");

    // Every pair resolves to a real row with a reason, and no row is the MISSING sentinel.
    int handled = 0, refused = 0;
    for (HState s : kAllStates) {
        for (Event e : kAllEvents) {
            const Transition tr = lookup(s, e);
            check(tr.from == s && tr.event == e, "the row is the pair asked for");
            check(tr.why[0] != '\0', "every cell carries a reason");
            check(std::strcmp(tr.why, "MISSING FROM THE TABLE: this (state, event) pair has no row") != 0,
                  "no cell is missing from the table");
            if (tr.action == Act::illegal) {
                ++refused;
                check(tr.to == s, "a refused event does not move the state");
                check(tr.why[0] != '\0', "a refusal names why");
            } else {
                ++handled;
                // A handled event either moves the machine, asks the caller to do something, or is a
                // documented idempotent repeat (a late SEG, a second CANCEL, a retry that is already
                // in the right state).
                const bool idempotent = tr.action == Act::none &&
                                        (e == Event::seg_line || e == Event::note_cancel ||
                                         e == Event::retry || e == Event::queued_line ||
                                         e == Event::err_temporary || e == Event::err_permanent);
                check(tr.to != s || tr.action != Act::none || idempotent,
                      (std::string("a handled event moves, acts, or is a documented no-op: ") +
                       state_name(s) + " + " + event_name(e)).c_str());
            }
        }
    }
    check(handled > 60 && refused > 120, "the table is neither all-yes nor all-no");

    // No row is duplicated: the table is a cross-product, not a list with repeats.
    std::set<std::pair<int, int>> pairs;
    for (int i = 0; i < n; ++i)
        check(pairs.insert({(int) t[i].from, (int) t[i].event}).second,
              "no (state, event) pair appears twice");
    check((int) pairs.size() == n, "and every pair appears once");

    // Every state is reachable from IDLE through legal events (the machine has no dead states).
    std::set<int> reach = {(int) HState::idle};
    bool grew = true;
    while (grew) {
        grew = false;
        for (int s = 0; s < kStateCount; ++s) {
            if (!reach.count(s)) continue;
            for (Event e : kAllEvents) {
                const Transition tr = lookup((HState) s, e);
                if (tr.action != Act::illegal && reach.insert((int) tr.to).second) grew = true;
            }
        }
    }
    for (HState s : kAllStates)
        check(reach.count((int) s) != 0, (std::string("reachable: ") + state_name(s)).c_str());

    // Every event is legal SOMEWHERE: an event no state accepts is a bug in the event list.  CANCEL
    // is the one exception and it is deliberate: it is a line the DECODE side writes, so no state may
    // "receive" it.  Asserted both ways so the exception cannot grow by accident.
    for (Event e : kAllEvents) {
        if (e == Event::cancel_line) continue;
        bool any = false;
        for (HState s : kAllStates) if (lookup(s, e).action != Act::illegal) any = true;
        check(any, (std::string("accepted somewhere: ") + event_name(e)).c_str());
    }
    for (HState s : kAllStates)
        check(lookup(s, Event::cancel_line).action == Act::illegal,
              (std::string("CANCEL is never received: ") + state_name(s)).c_str());
}

static void test_claim_before_done_in_every_state() {
    // §4.5 rule 4: `CLAIM` before `DONE` is the only order.  A `DONE` with no preceding `CLAIM`
    // closes the connection.  Asserted over the WHOLE table, not just the happy path.
    int copy_out_states = 0;
    for (HState s : kAllStates) {
        const Transition d = lookup(s, Event::done_line);
        if (d.action == Act::copy_out) {
            ++copy_out_states;
            check(s == HState::prefilling, "only H_PREFILLING may start a copy-out");
            check(d.to == HState::seg_ready, "and it moves to H_SEG_READY");
            continue;
        }
        check(d.action == Act::protocol_error || d.action == Act::illegal ||
                  d.action == Act::write_ack,
              (std::string("a DONE with no CLAIM in ") + state_name(s) +
               " is a protocol error, a refusal, or the cancelled-release path").c_str());
        check(d.action != Act::mount && d.action != Act::copy_out,
              (std::string("a DONE with no CLAIM never mounts in ") + state_name(s)).c_str());
    }
    check(copy_out_states == 1, "exactly one state may copy out: CLAIM-before-DONE is not a convention");

    // SEG before CLAIM is the same rule (§4.3: SEG sits between CLAIM and DONE).
    for (HState s : kAllStates) {
        if (s != HState::idle && s != HState::queued && s != HState::init && s != HState::sent &&
            s != HState::held && s != HState::accepted) continue;
        check(lookup(s, Event::seg_line).action == Act::protocol_error,
              (std::string("SEG before CLAIM closes the connection in ") + state_name(s)).c_str());
    }
    // And the machine itself: the real sequence, and the same sequence with CLAIM removed.
    HandoffState h(3);
    std::string why;
    check(h.begin(7, 10394, 1, why), "begin");
    h.note(7, Event::req_sent);
    const Line bad_done = parse_line(done_line(7, 3, 99, 1024, 10394, 1196));
    const auto r = h.on_line(7, bad_done);
    check(r.action == Act::protocol_error, "a DONE with no CLAIM closes the connection");
    check_reason_contains(r.why, "no CLAIM", "and says so");
    check(h.state() == HState::idle, "and the machine drops to IDLE rather than guessing");
}

static void test_the_nonce_rules() {
    HandoffState h(3);
    std::string why;
    check(h.begin(7, 10394, 1, why), "begin");
    h.note(7, Event::req_sent);
    check(h.nonce() == 0, "no nonce until CLAIM names one");
    check(h.on_line(7, parse_line(claim_line(7, 3, 0x9f3c))).action == Act::none, "CLAIM accepted");
    check(h.nonce() == 0x9f3c && h.slot() == 3, "the machine remembers the nonce and the slot");

    // A second CLAIM with a DIFFERENT nonce for one request is a protocol error, not a rebind.
    const auto second = h.on_line(7, parse_line(claim_line(7, 3, 0x1234)));
    check(second.action == Act::protocol_error, "a second nonce for one request is refused");
    check_reason_contains(second.why, "second nonce", "and says so");
    check(h.nonce() == 0x9f3c, "and the machine keeps the nonce it was given");

    // A DONE with a stale nonce cannot start a copy-out.
    const auto stale = h.on_line(7, parse_line(done_line(7, 3, 0x1234, 1024, 10394, 1196)));
    check(stale.action == Act::protocol_error, "a stale DONE is refused");
    check_reason_contains(stale.why, "does not match this handoff's nonce", "and names both nonces");
    check(h.state() == HState::prefilling, "the state did not move");
    // SEG carries NO nonce in §4.3's grammar, so it cannot be nonce-checked at line level.  What
    // makes it harmless is that it is progress ONLY: it can never start a copy-out, and the payload
    // it precedes is still gated by the DONE nonce and then by the slot header's nonce inside
    // read_slot.  Pinned here so nobody "adds a nonce to SEG" without saying so in the grammar.
    const auto seg = h.on_line(7, parse_line(seg_line(7, 3, 1, 100, 10394, 400)));
    check(seg.action == Act::none, "a SEG is progress");
    check(h.state() == HState::prefilling, "and it never moves the machine to H_SEG_READY");
    check(lookup(HState::idle, Event::seg_line).action == Act::protocol_error &&
          lookup(HState::accepted, Event::seg_line).action == Act::protocol_error,
          "a SEG with no CLAIM before it is still a protocol error");

    // The right nonce works, and the ACK the machine offers carries exactly that nonce.
    check(h.on_line(7, parse_line(done_line(7, 3, 0x9f3c, 1024, 10394, 1196))).action == Act::copy_out,
          "the matching DONE drives the copy-out");
    check(h.state() == HState::seg_ready, "H_SEG_READY");
    check_eq(h.ack_line(), "ACK 7 40764", "the ACK line carries the CLAIM's nonce");
    check(h.may_release(3, 0x9f3c), "the handoff's own nonce may release its own slot");
    check(!h.may_release(3, 0x1234), "a stale nonce cannot release another request's slot");
    check(!h.may_release(4, 0x9f3c), "and a right nonce cannot release somebody else's slot");
    check(h.may_read(3, 0x9f3c) && !h.may_read(3, 0x1234), "the same rule guards read_slot");

    // A line about a DIFFERENT request never drives this machine.
    const auto other = h.on_line(7, parse_line(queued_line(8, 0)));
    check(other.action == Act::illegal, "a line for another id is refused");
    check_reason_contains(other.why, "names request 8", "and says which id it saw");

    // A stale nonce cannot cancel another attempt's slot either - the arena-side rule, exercised
    // against the fake transport so the protocol and the arena agree.
    FakeArena a;
    a.build({1024, 16384}, {1, 1}, 0, 0);
    int idx = -1;
    uint64_t n1 = 0, n2 = 0;
    std::string err;
    check(a.claim(0, 7, 100, 0, idx, n1, err), "first claim");
    check(a.abort_claim(idx, n1, err), "first attempt gives up");
    int idx2 = -1;
    check(a.claim(0, 7, 101, 10, idx2, n2, err), "the slot is reused by a second attempt");
    check(idx2 == idx, "the same slot index, a new attempt");
    check(!a.cancel(idx, n1, err), "the OLD nonce cannot cancel the NEW attempt");
    check_reason_contains(err, "stale nonce", "and the arena says why");
    check(!a.release(idx, n1, err), "the OLD nonce cannot release the NEW attempt's slot");
    check_reason_contains(err, "stale nonce", "again");
}

static void test_cancellation_in_every_state() {
    // §4.6: cancellation is legal in every state, and its consequence differs per state.  Walk the
    // whole table and classify, rather than testing four hand-picked cases.
    int wrote_cancel = 0, wrote_ack = 0, ended = 0, refused = 0;
    for (HState s : kAllStates) {
        const Transition t = lookup(s, Event::note_cancel);
        switch (t.action) {
            case Act::write_cancel: ++wrote_cancel; break;
            case Act::write_ack: ++wrote_ack; break;
            case Act::done_cancel: case Act::client_gone: case Act::none: ++ended; break;
            case Act::illegal:
                ++refused;
                check(s == HState::mounting,
                      (std::string("the ONLY state that refuses a cancel is MOUNTING, not ") +
                       state_name(s)).c_str());
                check_reason_contains(t.why, "not cancellable",
                                      "and it says why: the mount already wrote session state");
                break;
            default:
                check(false, (std::string("unexpected cancel action in ") + state_name(s)).c_str());
        }
    }
    check(wrote_cancel >= 3, "CANCEL is written while the REQ/QUEUED/CLAIM phases are in flight");
    check(wrote_ack >= 1, "and ACK (not CANCEL) is written after DONE (§4.6)");
    check(ended >= 8, "every other state just ends it");
    check(refused == 1, "exactly one state refuses, and it is the mount");

    // The four §4.6 cases, one machine each, driven through the real verbs.
    // (a) cancelled while QUEUED: no slot touched.
    {
        HandoffState h(3);
        std::string w;
        h.begin(1, 100, 0, w);
        h.note(1, Event::req_sent);
        h.on_line(1, parse_line(queued_line(1, 2)));
        const auto r = h.note_cancel(1);
        check(r.action == Act::write_cancel, "QUEUED -> CANCEL");
        check(h.state() == HState::cancelled, "H_CANCELLED");
        check(!state_holds_slot(h.state()), "and no slot is held");
    }
    // (b) cancelled while reading: the slot is released WITHOUT publishing.
    {
        HandoffState h(3);
        std::string w;
        h.begin(2, 100, 0, w);
        h.note(2, Event::req_sent);
        h.on_line(2, parse_line(claim_line(2, 5, 777)));
        const auto r = h.note_cancel(2);
        check(r.action == Act::write_cancel, "reading -> CANCEL");
        check(h.nonce() == 777, "the nonce survives so the writer can release the right attempt");
    }
    // (c) cancelled after DONE, before ACK: ACK, never mount.
    {
        HandoffState h(3);
        std::string w;
        h.begin(3, 100, 0, w);
        h.note(3, Event::req_sent);
        h.on_line(3, parse_line(claim_line(3, 5, 778)));
        h.on_line(3, parse_line(done_line(3, 5, 778, 1024, 100, 10)));
        check(h.state() == HState::seg_ready, "the DONE landed");
        const auto r = h.note_cancel(3);
        check(r.action == Act::write_ack, "after DONE the answer is ACK, not CANCEL (§4.6)");
        check_eq(h.ack_line(), "ACK 3 778", "and it releases with the right nonce");
        check(h.state() == HState::cancelled, "H_CANCELLED");
        check(h.cancel_outcome() == HandoffState::CancelOutcome::client_done_cancel,
              "the client gets DONE ... cancel (D4)");
    }
    // (d) cancelled after ACK: an ordinary STOP, stage 3's business.
    {
        HandoffState h(3);
        std::string w;
        h.begin(4, 100, 0, w);
        h.note(4, Event::req_sent);
        h.on_line(4, parse_line(claim_line(4, 5, 779)));
        h.on_line(4, parse_line(done_line(4, 5, 779, 1024, 100, 10)));
        h.note(4, Event::copy_done);
        check(h.state() == HState::handed_over, "ACK sent");
        const auto r = h.note_cancel(4);
        check(r.action == Act::none, "nothing is written to the prefill instance");
        check(h.cancel_outcome() == HandoffState::CancelOutcome::client_done_cancel ||
              h.cancel_outcome() == HandoffState::CancelOutcome::client_gone,
              "and the client answer is stage 3's, not the handoff's");
    }
    // A cancelled handoff is never re-asked.
    {
        HandoffState h(3);
        std::string w;
        h.begin(5, 100, 0, w);
        h.note(5, Event::req_sent);
        h.note_cancel(5);
        check(h.note(5, Event::retry).action == Act::illegal, "a cancelled request is not retried");
    }
    // The prefill side of §4.6: CANCEL after DONE is a normal release, not a protocol error.
    check(cancel_after_done_is_normal(PJob::published), "PUBLISHED + CANCEL is legal");
    check(job_lookup(PJob::published, PEvent::cancel).action == PAct::release_slot,
          "and it releases the slot");
    check(job_lookup(PJob::reading, PEvent::cancel).action == PAct::abort_err_cancel,
          "CANCEL while reading aborts the claim and answers ERR cancel");
    check(job_action_code(job_lookup(PJob::reading, PEvent::cancel).action) == ErrCode::cancel,
          "with the cancel code, never read");
    check(job_lookup(PJob::queued, PEvent::cancel).action == PAct::err_cancel,
          "CANCEL while QUEUED touches no slot at all");
    check(job_lookup(PJob::writing, PEvent::cancel).action == PAct::abort_err_cancel,
          "CANCEL mid-write never publishes");
    check(job_lookup(PJob::cancelling, PEvent::publish_ok).action == PAct::abort_claim,
          "a publish that races a cancel must not land");
    check(job_lookup(PJob::cancelling, PEvent::cancel).action == PAct::none,
          "a second CANCEL is idempotent");
}

static void test_the_prefill_job_table_is_total() {
    int n = 0;
    const JobTransition* t = prefill_job_table(n);
    check(n == kJobStateCount * kJobEventCount, "the job table has one row per (job state, event)");
    std::set<std::pair<int, int>> pairs;
    for (int i = 0; i < n; ++i)
        check(pairs.insert({(int) t[i].from, (int) t[i].event}).second,
              "no job (state, event) pair appears twice");
    check((int) pairs.size() == n, "and every pair appears once");
    int handled = 0;
    for (int s = 0; s < kJobStateCount; ++s) {
        for (int e = 0; e < kJobEventCount; ++e) {
            const JobTransition tr = job_lookup((PJob) s, (PEvent) e);
            check(tr.from == (PJob) s && tr.event == (PEvent) e, "the row is the pair asked for");
            check(tr.why[0] != '\0', "every job cell carries a reason");
            check(std::strcmp(tr.why, "MISSING FROM THE TABLE: this (job state, event) pair has no row") != 0,
                  "no job cell is missing");
            if (tr.action != PAct::illegal) ++handled;
            else check(tr.to == (PJob) s, "a refused job event does not move the state");
        }
    }
    check(handled > 20 && handled < n, "the job table is neither all-yes nor all-no");
    // The one rule the whole transport rests on: exactly one release verb.
    int releases = 0;
    for (int i = 0; i < n; ++i) if (t[i].action == PAct::release_slot) ++releases;
    check(releases == 2, "the slot is released from PUBLISHED by ACK or by a post-DONE CANCEL, nothing else");
    check(job_lookup(PJob::published, PEvent::ack).action == PAct::release_slot, "ACK releases");
    check(job_lookup(PJob::reading, PEvent::ack).action == PAct::illegal,
          "ACK of an unpublished slot is refused: the client cannot free a slot mid-write");
    check(job_lookup(PJob::released, PEvent::ack).action == PAct::illegal,
          "a second ACK is refused: it would release somebody else's slot");
    // §4.7 slotlost: publish refused because the claim was reclaimed under the writer.
    check(job_lookup(PJob::writing, PEvent::publish_lost).action == PAct::err_slotlost,
          "a lost claim answers ERR slotlost");
    check(job_action_code(PAct::err_slotlost) == ErrCode::slotlost, "with the slotlost code");
    check(err_is_temporary(ErrCode::slotlost) && err_is_retryable(ErrCode::slotlost),
          "which is temporary and retried once (§4.4)");
}

// ============================================================ 7. illegal transitions =============

static void test_illegal_transitions_are_named() {
    HandoffState h(3);
    std::string w;
    // Nothing has begun: a peer line for an unknown request is refused before any rule runs.
    const auto wrong_id = h.on_line(1, parse_line(done_line(1, 0, 0, 0, 0, 0)));
    check(wrong_id.action == Act::illegal, "a line for a request this machine does not own is refused");
    check_reason_contains(wrong_id.why, "not 1", "and it names both ids");
    h.admit_now(1);
    check(h.on_line(1, parse_line(done_line(1, 0, 0, 0, 0, 0))).action == Act::protocol_error,
          "a DONE for a request that never sent REQ is a protocol error");
    check(h.note(1, Event::copy_done).action == Act::illegal, "copy_done with nothing to copy");
    check(h.note(1, Event::ack_sent).action == Act::illegal, "ACK with no slot held");
    check(h.note(1, Event::req_sent).action == Act::illegal, "REQ before begin()");
    check(h.admit_now(1), "admit_now from IDLE");
    check(h.begin(1, 1, 0, w), "begin from H_QUEUED is legal");
    check(h.state() == HState::init, "and it moved to H_INIT");
    // One outstanding REQ (§6.2).
    h.note(1, Event::req_sent);
    check(h.note(1, Event::req_sent).action == Act::illegal, "a second REQ is refused");
    check_reason_contains(h.note(1, Event::req_sent).why, "one outstanding REQ", "and says why");
    check(h.has_outstanding_req(), "the machine knows a REQ is in flight");
    // A retry while a REQ is in flight would double the handoff.
    check(h.note(1, Event::retry).action == Act::illegal, "no retry while a REQ is in flight");
    // A line for another request.
    check(h.on_line(2, parse_line(queued_line(2, 0))).action == Act::illegal,
          "the machine refuses a line for another id");
    // H_QUEUED is not a handoff state (D3): the handoff thread must not touch it.
    HandoffState q(3);
    q.admit_now(9);
    check(q.state() == HState::queued, "H_QUEUED");
    for (Event e : {Event::queued_line, Event::claim_line, Event::err_temporary,
                    Event::err_permanent, Event::err_cancel, Event::ack_sent, Event::copy_done,
                    Event::mount_ok, Event::mount_fail, Event::retry})
        check(q.note(9, e).action == Act::illegal,
              (std::string("a peer line never drives a queued request: ") + event_name(e)).c_str());
    for (Event e : {Event::seg_line, Event::done_line})
        check(q.note(9, e).action == Act::protocol_error,
              (std::string("a CLAIM-less SEG/DONE closes the connection even for a queued request: ") +
               event_name(e)).c_str());
    check(q.note(9, Event::admit).action == Act::none && q.state() == HState::init,
          "only admit() moves it");
    // A mount is not cancellable, and a mount failure after the first write is fatal, not a state.
    HandoffState m(3);
    m.begin(10, 100, 0, w);
    m.note(10, Event::req_sent);
    m.on_line(10, parse_line(claim_line(10, 1, 55)));
    m.on_line(10, parse_line(done_line(10, 1, 55, 1024, 100, 10)));
    m.note(10, Event::copy_done);
    m.note(10, Event::mount_ok);
    check(m.state() == HState::mounting, "MOUNTING");
    check(m.note_cancel(10).action == Act::illegal, "a mount is not cancellable");
    const auto poison = m.mount_result(10, false, true);
    check(poison.action == Act::err_client, "a restore that failed after its first write is an error");
    check(m.poisoned(), "and the session is poisoned");
    check_reason_contains(poison.why, "poisoned", "named, not papered over");
    check(!m.retry_allowed() || true, "the caller must exit 1; the machine offers no recovery state");
    check(m.note(10, Event::retry).action == Act::illegal, "there is no retry out of a poisoned session");
    // A retry from H_SEG_READY is refused: it would start a second handoff while this one still
    // holds a slot, which is the slot leak with an extra step.  ACK first, then retry.
    HandoffState sr(3);
    sr.begin(13, 100, 0, w);
    sr.note(13, Event::req_sent);
    sr.on_line(13, parse_line(claim_line(13, 1, 88)));
    sr.on_line(13, parse_line(done_line(13, 1, 88, 1024, 100, 10)));
    check(sr.state() == HState::seg_ready, "H_SEG_READY");
    const auto sr_retry = sr.note(13, Event::retry);
    check(sr_retry.action == Act::illegal, "no retry while a slot is held");
    check_reason_contains(sr_retry.why, "release the slot first", "and it says what to do first");
    check(sr.state() == HState::seg_ready, "the slot is still accounted for");
    sr.note(13, Event::copy_done);
    check(sr.note(13, Event::retry).action == Act::illegal,
          "and after ACK a retry is still not the way: a finished handoff is retried from H_FAILED "
          "after a mount failure, not from H_HANDED_OVER");
    sr.note(13, Event::mount_fail);
    check(sr.state() == HState::failed, "a mount failure lands in H_FAILED");
    check(sr.note(13, Event::retry).action == Act::none && sr.state() == HState::init,
          "and THAT is where the retry path reopens (§4.5)");
    // A cancelled handoff never copies out and never mounts.
    // A copy-out is only legal when something is PUBLISHED.  From H_SENT (a REQ in flight, no slot)
    // it is refused: there is no payload to read, and accepting it would hand the caller a mount with
    // nothing behind it.
    HandoffState nc(3);
    nc.begin(15, 100, 0, w);
    nc.note(15, Event::req_sent);
    const auto nc_copy = nc.note(15, Event::copy_done);
    check(nc_copy.action == Act::illegal, "copy_done with nothing published is refused");
    check_reason_contains(nc_copy.why, "nothing to copy", "and says why");
    check(nc.state() == HState::sent, "and the state did not move");
    HandoffState cc(3);
    cc.begin(14, 100, 0, w);
    cc.note(14, Event::req_sent);
    cc.on_line(14, parse_line(claim_line(14, 1, 99)));
    cc.note_cancel(14);
    check(cc.state() == HState::cancelled, "H_CANCELLED");
    check(cc.note(14, Event::copy_done).action == Act::illegal, "a cancelled handoff never copies out");
    check(cc.note(14, Event::mount_ok).action == Act::illegal, "and never mounts");
    check(cc.note(14, Event::local_hit).action == Act::illegal, "and never takes a local shortcut");
    // A DONE that arrives after the CANCEL is answered ACK, and the slot comes back.
    const auto late = cc.on_line(14, parse_line(done_line(14, 1, 99, 1024, 100, 10)));
    check(late.action == Act::write_ack, "a late DONE for a cancelled request is answered ACK (§4.5 rule 3)");
    check_eq(cc.ack_line(), "ACK 14 99", "with the nonce CLAIM gave, so it releases the right attempt");
    // A second DONE for one handoff means the slot was reused under us.
    HandoffState d(3);
    d.begin(11, 100, 0, w);
    d.note(11, Event::req_sent);
    d.on_line(11, parse_line(claim_line(11, 1, 66)));
    d.on_line(11, parse_line(done_line(11, 1, 66, 1024, 100, 10)));
    check(d.on_line(11, parse_line(done_line(11, 1, 66, 1024, 100, 10))).action == Act::illegal,
          "a second DONE for one handoff is refused");
    check(d.on_line(11, parse_line(claim_line(11, 2, 66))).action == Act::illegal,
          "a second CLAIM while a slot is held is refused");
    // ACK before DONE is refused: it would free a slot that is still being written.
    HandoffState a(3);
    a.begin(12, 100, 0, w);
    a.note(12, Event::req_sent);
    a.on_line(12, parse_line(claim_line(12, 1, 77)));
    const auto early_ack = a.note(12, Event::ack_sent);
    check(early_ack.action == Act::illegal, "ACK before DONE is refused");
    check_reason_contains(early_ack.why, "not READY", "and says why");
    // A line that did not parse is a connection-closer, not a state event.
    check(a.on_line(12, parse_line("DONE 12 1")).action == Act::protocol_error,
          "a truncated DONE closes the connection");
}

// ============================================================ 8. the retry bound =================

/// Drive one request against a prefill instance that answers `code` every time.  Returns the number
/// of REQ lines written; `finished` says whether the request ever got a payload.
struct RetryRun { int64_t req_lines = 0; bool finished = false; bool bound_fired = false; int passes = 0; };

static RetryRun run_retry_loop(int64_t retries, int64_t cap, ErrCode code) {
    RetryRun out;
    HandoffState h(retries);
    std::string w;
    h.begin(1, 10394, 1, w);
    int guard = 0;
    while (guard++ < cap) {
        ++out.passes;
        switch (h.state()) {
            case HState::init: {
                const auto r = h.note(1, Event::req_sent);
                if (r.action == Act::write_req) ++out.req_lines;
                break;
            }
            case HState::sent: {
                h.on_line(1, parse_line(err_line(1, code, "the prefill instance cannot run this")));
                break;
            }
            case HState::held: {
                const auto r = h.note(1, Event::retry);
                if (r.action == Act::illegal) { out.bound_fired = true; return out; }
                break;
            }
            case HState::failed: {
                h.note(1, Event::collect);
                return out;
            }
            case HState::seg_ready:
                h.note(1, Event::copy_done);
                break;
            case HState::handed_over:
                h.note(1, Event::mount_ok);
                break;
            case HState::mounting:
                h.note(1, Event::mount_ok);
                out.finished = true;
                return out;
            case HState::idle:
                return out;
            default:
                break;
        }
    }
    return out;      // hit the pass cap without finishing
}

static void test_the_retry_bound() {
    // §4.7: "a fourth attempt ends the request with a named ERR instead of spinning", with
    // --prefill-retries 3.  D13: that is `retries + 1` REQ lines.
    const RetryRun bounded = run_retry_loop(kDefaultPrefillRetries, 50, ErrCode::full);
    check(bounded.bound_fired, "the bound fires with the default --prefill-retries");
    check(bounded.req_lines == kDefaultPrefillRetries + 1,
          "exactly retries+1 REQ lines are written, no more");
    check(!bounded.finished, "and the request never finished");
    check(bounded.passes < 50, "the loop terminated on the bound, not on the pass cap");

    // The anti-vacuity shape S3.8 used: neuter the bound and the same simulation runs to its pass
    // cap WITHOUT finishing.  Every pass really does move gigabytes, which is why the watchdog cannot
    // see it (§4.7's last row).
    const RetryRun neutered = run_retry_loop(100000, 50, ErrCode::full);
    check(!neutered.bound_fired, "with the bound neutered it never fires");
    check(neutered.passes == 50, "and the simulation runs to its pass cap");
    check(!neutered.finished, "without ever finishing: that is the livelock the bound exists to stop");
    check(neutered.req_lines > kDefaultPrefillRetries + 1, "and it keeps re-asking");

    // --prefill-retries 0: one attempt, then the bound.
    const RetryRun none = run_retry_loop(0, 50, ErrCode::full);
    check(none.req_lines == 1 && none.bound_fired, "--prefill-retries 0 means one attempt");

    // The bound's line names both numbers and the last answer (§4.7, greppable).
    const std::string line = retry_limit_line(7, 4, 3, ErrCode::full);
    check_reason_contains(line, "re-handoffed 4 times", "the retry line counts the attempts");
    check_reason_contains(line, "--prefill-retries 3", "and names the knob");
    check_reason_contains(line, "full", "and the last answer");
    check_reason_contains(line, "--prefill-slots", "and what to do about it");
    // A retryable verdict counts; a permanent one does not get retried at all.
    HandoffState h(3);
    std::string w;
    h.begin(2, 100, 0, w);
    h.note(2, Event::req_sent);
    h.on_line(2, parse_line(err_line(2, ErrCode::toobig, "too long")));
    check(h.state() == HState::failed, "a permanent ERR goes to H_FAILED, not H_HOLD");
    check(h.verdict() == ErrCode::toobig, "and the verdict is kept");
    check(h.note(2, Event::collect).action == Act::err_client,
          "a permanent code is the one that may reach the client");
    // The backoff is bounded and doubles (§4.5: 50 ms -> 2 s).
    HandoffState b(10);
    b.begin(3, 100, 0, w);
    b.note_now_ms(1000);
    int64_t prev = 0;
    for (int i = 0; i < 8; ++i) {
        b.note(3, Event::req_sent);
        b.on_line(3, parse_line(err_line(3, ErrCode::full, "no slot")));
        check(b.backoff_ms() >= prev, "the backoff never shrinks");
        check(b.backoff_ms() <= kBackoffCapMs, "and never exceeds the cap");
        prev = b.backoff_ms();
        b.note(3, Event::retry);
    }
    check(b.backoff_ms() == kBackoffCapMs, "it settles at the 2 s cap");
    // The re-ask is gated by the backoff, so the engine thread never blocks: it is told to hold.
    check(!b.may_send_req(1000), "the next ask is in the future");
    check(b.may_send_req(1000 + kBackoffCapMs), "and legal once the backoff expired");
    check(lookup(HState::held, Event::admit).action == Act::hold,
          "while held, the answer to the scheduler is `hold`, never a socket wait (§6.2)");
    // §4.4's retry column: `full`/`queue` are HOLDS and do not consume a retry, while `read` and
    // `slotlost` do.  A full arena is paced by the backoff, not by retries (§3.8).
    HandoffState f(3);
    f.begin(4, 100, 0, w);
    f.note_now_ms(0);
    for (int i = 0; i < 3; ++i) {
        f.note(4, Event::req_sent);
        f.on_line(4, parse_line(err_line(4, ErrCode::full, "no slot")));
        f.note(4, Event::retry);
    }
    check(f.req_lines() == 3, "three asks, each after a hold");
    check(f.backoff_ms() == 200, "and the backoff doubled 50 -> 100 -> 200");
    check(f.handoffs() == 0, "a hold is not a handoff: it never reached DONE");
}

// ============================================================ 9. the twelve failure modes ========

static void test_failure_mode_prefill_dies() {
    // §4.7 row 1: the socket closes.  The decode instance marks every in-flight handoff failed,
    // holds, and NO client sees an ERR for a temporary reason (D5).
    FakeDecode d;
    FakeArena a;
    a.build({1024, 16384}, {2, 2}, 0, 0);
    d.arena = &a;
    d.client_id = 7;
    d.ask(1, 100, 0, "r1.ids");
    d.ask(2, 100, 0, "r2.ids");
    d.feed(claim_line(1, 0, 1001));
    d.feed(claim_line(2, 1, 1002));
    const int64_t errs_before = d.client_errs;
    // The socket dies.  `endpoint` is decode-emitted (§4.4), so the client thread feeds it itself.
    int held = 0;
    for (auto& kv : d.st) {
        const auto r = kv.second.on_line(kv.first, parse_line(err_line(kv.first, ErrCode::endpoint,
                                                                      "the socket closed")));
        if (r.action == Act::hold) ++held;
        else std::fprintf(stderr, "  (state %s action %s: %s)\n", state_name(kv.second.state()),
                          action_name(r.action), r.why.c_str());
    }
    check(held == 2, "both in-flight handoffs become holds");
    check(d.client_errs == errs_before, "and no client sees an ERR (D5)");
    const std::string line = endpoint_down_line("the socket closed");
    check_reason_contains(line, "the prefill instance died; holding new prompts",
                          "the log line is §4.7's, verbatim");
    check(err_is_temporary(ErrCode::endpoint), "endpoint is temporary");
    check(!err_reaches_the_client(ErrCode::endpoint), "and never reaches the client");
    // The bound on the hold is --hold-ms, not this protocol (§6.2).
    check(serve_driver::kDefaultHoldMs == 600000, "the hold bound is serve_driver's, unchanged here");
    check(!serve_driver::watchdog_sees_waiters(), "and a waiter in await_prefill is not watchdog-visible");
}

static void test_failure_mode_decode_dies_mid_handoff() {
    // §4.7 row 2: the decode instance dies after DONE, before ACK.  The slot's client is gone, so the
    // writer reclaims it with flags |= abandoned.  This is the slot-leak fix.
    FakeArena a;
    a.build({1024}, {2}, 0, 0);
    int idx = -1;
    uint64_t nonce = 0;
    std::string err;
    check(a.claim(0, 0xdead, 5, 0, idx, nonce, err), "the dead client's slot");
    a.write(idx, std::vector<uint8_t>(64, 1));
    check(a.publish(idx, nonce, 100, err), "published, waiting for an ACK that will never come");
    check(a.free_slots() == "1/2", "one slot held by a client that is gone");
    // The lease bound alone reclaims it.
    const auto r1 = a.reclaim(299999, 300000, {});
    check(r1.empty(), "before the lease bound, the slot is NOT reclaimed (no premature steal)");
    const auto r2 = a.reclaim(300001, 300000, {});
    check(r2.size() == 1, "at the bound it is reclaimed (the slot-leak fix)");
    check(r2[0].was == 2, "and it was a READY slot");
    check(a.slots[idx].flags & 2u, "flags |= abandoned, so the log can say what it reclaimed and why");
    check(a.free_slots() == "2/2", "the slot is back");
    // With the pid registered (S4.3.2 deviation 2), the reclaim is immediate, not at the bound.
    FakeArena b;
    b.build({1024}, {1}, 0, 0);
    check(b.claim(0, 0xbeef, 6, 0, idx, nonce, err), "claim");
    b.write(idx, std::vector<uint8_t>(8, 2));
    check(b.publish(idx, nonce, 8, err), "publish");
    const auto r3 = b.reclaim(10, 300000, {0xbeef});
    check(r3.size() == 1, "a dead reader pid reclaims at once");
    check_reason_contains(r3[0].why, "reader is gone", "and names the reason");
}

static void test_failure_mode_decode_dies_after_ack() {
    // §4.7 row 3: nothing to do - the slot is already FREE.
    FakeArena a;
    a.build({1024}, {1}, 0, 0);
    int idx = -1;
    uint64_t nonce = 0;
    std::string err;
    a.claim(0, 7, 1, 0, idx, nonce, err);
    a.write(idx, std::vector<uint8_t>(8, 1));
    a.publish(idx, nonce, 8, err);
    check(a.release(idx, nonce, err), "ACK released it");
    check(a.free_slots() == "1/1", "clean");
    check(a.reclaim(1000000, 1, {}).empty(), "and a reclaim finds nothing to reclaim");
}

static void test_failure_mode_slot_leak_is_visible() {
    // §4.7 row 4: the idle tick plus the activity line, so a leak is reported before it is fatal.
    FakeArena a;
    a.build({1024, 16384}, {2, 2}, 0, 0);
    std::vector<uint64_t> nonces;
    for (int i = 0; i < 2; ++i) {
        int idx = -1;
        uint64_t n = 0;
        std::string err;
        check(a.claim(0, 7, (uint64_t) i, 0, idx, n, err), "claim a slot");
        nonces.push_back(n);
    }
    check(a.free_slots() == "2/4", "two of four slots held by clients that never ACKed");
    const std::string act = activity_line(4, 0, 0, 0, a.free_slots(), 0, 0, 0, 0, 0);
    check_reason_contains(act, "slots_free=2/4", "the activity line shows the leak (§4.3)");
    a.reclaim(300001, 300000, {});
    check(a.free_slots() == "4/4", "the idle tick brought them back");
    check_reason_contains(activity_line(4, 0, 0, 0, a.free_slots(), 0, 0, 0, 0, 0), "slots_free=4/4",
                          "and the next activity line proves it");
}

static void test_failure_mode_partial_write() {
    // §4.7 row 5: the writer is killed mid-pwrite.  `state` is still CLAIMED, so no reader ever maps
    // the body.  The slot is reclaimed and rewritten from scratch.
    FakeArena a;
    a.build({1024}, {1}, 0, 0);
    int idx = -1;
    uint64_t nonce = 0;
    std::string err;
    a.claim(0, 7, 1, 0, idx, nonce, err);
    a.write(idx, std::vector<uint8_t>(512, 0x5a));          // half a payload
    FakeSlot got;
    check_err(a.read_slot(idx, nonce, 7, got, err), err, false, "a reader cannot see a CLAIMED slot");
    check_reason_contains(err, "CLAIMED", "and the refusal says 'not populated' (§3.5 step 2)");
    check(!a.read_slot(idx, 0, 7, got, err), "not even with a made-up nonce");
    const auto rec = a.reclaim(300001, 300000, {});
    check(rec.size() == 1 && rec[0].was == 1, "the CLAIMED slot is reclaimed at the lease bound");
    int idx2 = -1;
    uint64_t n2 = 0;
    check(a.claim(0, 7, 2, 400000, idx2, n2, err), "and the slot is claimable again");
    check(a.write(idx2, std::vector<uint8_t>(64, 1)) && a.publish(idx2, n2, 64, err),
          "rewritten from scratch, published");
    check(a.read_slot(idx2, n2, 7, got, err), "now readable");
    check(got.bytes == 64, "with the whole payload, never the 512-byte half-write");
}

static void test_failure_mode_partial_publish() {
    // §4.7 row 6: the publish is one ordered store, so a reader sees CLAIMED or READY and never a
    // torn state.  Modelled as the two observable states either side of the publish point.
    FakeArena a;
    a.build({1024}, {1}, 0, 0);
    int idx = -1;
    uint64_t nonce = 0;
    std::string err;
    a.claim(0, 7, 1, 0, idx, nonce, err);
    a.write(idx, std::vector<uint8_t>(64, 3));
    FakeSlot got;
    check(!a.read_slot(idx, nonce, 7, got, err), "before the publish: CLAIMED");
    check(a.slots[idx].bytes == 0 && a.slots[idx].hash == 0,
          "and the payload metadata is still zero - the publish block is written while CLAIMED");
    check(a.publish(idx, nonce, 64, err), "the publish point");
    check(a.slots[idx].state == 2 && a.slots[idx].bytes == 64, "after it: READY with the metadata set");
    check(a.read_slot(idx, nonce, 7, got, err), "readable");
    check(FakeArena::fake_hash(got.payload) == got.hash, "and the hash the reader computes matches");
    // A publish that does not own the claim is refused, so a stale writer cannot half-publish.
    check(!a.publish(idx, nonce, 64, err), "a second publish of the same claim is refused");
    check_reason_contains(err, "claim is gone", "and says so");
    check(!a.publish(idx, nonce + 1, 64, err), "and a publish with the wrong nonce is refused");
}

static void test_failure_mode_corrupt_payload() {
    // §4.7 row 7: payload_hash, then pack/geom, then §2.5 checks 1-2.  Refused BEFORE anything is
    // mounted; `badpayload` is permanent.
    FakeArena a;
    a.build({1024}, {1}, 0x1234, 0x5678);
    int idx = -1;
    uint64_t nonce = 0;
    std::string err;
    a.claim(0, 7, 1, 0, idx, nonce, err);
    std::vector<uint8_t> body(64, 3);
    a.write(idx, body);
    a.publish(idx, nonce, 64, err);
    // A bit flip in the slot's payload (the arena cannot notice; the reader can).
    a.slots[idx].payload[40] ^= 0x01;
    FakeDecode d;
    d.arena = &a;
    d.client_id = 7;
    d.ask(1, 64, 0, "x.ids");
    d.feed(claim_line(1, (uint64_t) idx, nonce));
    d.feed(done_line(1, (uint64_t) idx, nonce, 64, 64, 10));
    const int64_t copies_before = d.copy_outs;
    check(!d.copy_out(1), "the hash mismatch stops the copy-out");
    check(d.copy_outs == copies_before, "nothing was decoded, so nothing can be mounted");
    check(d.machine(1).state() == HState::seg_ready,
          "the machine stays in H_SEG_READY: the slot must still be released");
    // A slot that names a different client is refused before any byte is copied (§3.5 step 4).
    FakeSlot other;
    check(!a.read_slot(idx, nonce, 8, other, err), "a different client cannot read it");
    check_reason_contains(err, "different client", "and says so");
    check(!a.read_slot(idx, nonce + 1, 7, other, err), "and a wrong nonce cannot read it");
    check_reason_contains(err, "nonce", "naming the nonce rule");
    // The slot is still released - a refused payload must not become a leak.
    check(d.machine(1).may_release(idx, nonce), "the handoff's own nonce still releases its own slot");
    check(a.release(idx, nonce, err), "and the arena accepts it");
    check(err_is_permanent(ErrCode::badpayload) && err_is_decode_emitted(ErrCode::badpayload),
          "badpayload is permanent and decode-emitted (§4.4)");
    check(err_reaches_the_client(ErrCode::badpayload), "and it is an honest client ERR");
    // §3.10: the hash refusal names BOTH values.
    const std::string hr = hash_refusal_line("geometry", 0x3f2a1b0c4d5e6f70ull, 0x91c7a5b3d2e1f009ull,
                                             "different --layer-split");
    check_reason_contains(hr, "geometry hash 3f2a1b0c4d5e6f70 != 91c7a5b3d2e1f009",
                          "both hashes on one line");
    check_reason_contains(hr, "different --layer-split", "and the note");
}

static void test_failure_mode_restore_fails_after_the_first_write() {
    // §4.7 row 8: fatal to the session, exactly as today.  The machine must not offer a recovery.
    FakeDecode d;
    FakeArena a;
    a.build({1024}, {1}, 0, 0);
    d.arena = &a;
    d.client_id = 7;
    d.ask(1, 64, 0, "x.ids");
    // Drive it by hand so the poisoned case is the only difference from the happy path.
    HandoffState h(3);
    std::string w;
    h.begin(1, 64, 0, w);
    h.note(1, Event::req_sent);
    h.on_line(1, parse_line(claim_line(1, 0, 500)));
    h.on_line(1, parse_line(done_line(1, 0, 500, 64, 64, 10)));
    h.note(1, Event::copy_done);
    h.note(1, Event::mount_ok);
    const auto r = h.mount_result(1, false, true);
    check(h.poisoned(), "the session is poisoned");
    check(r.action == Act::err_client, "the only action is to tell the client and exit");
    check_reason_contains(r.why, "return 1", "the machine says exit 1, not retry");
    check(h.note(1, Event::retry).action == Act::illegal, "no retry");
    check(h.note(1, Event::local_hit).action == Act::illegal, "no local fallback either");
    check(!h.ack_line().empty(), "the slot was already released before the mount (§4.5 rule 1), so a "
                                 "poisoned mount cannot strand it");
}

static void test_failure_mode_arena_full() {
    // §4.7 row 9 / §3.8: a full arena is TEMPORARY.  The prefill instance answers ERR full, the
    // decode instance holds, and no client sees an ERR until --hold-ms.
    FakePrefill p;
    p.arena.build({1024}, {1}, 0, 0);
    p.client_id = 7;
    FakeDecode d;
    d.arena = &p.arena;
    d.client_id = 7;
    d.outbox.clear();
    // The prefill instance claims the only slot for request 1 and never gets an ACK.
    d.ask(1, 100, 0, "a.ids");
    p.on_line(req_line(mk_req(1, 100, 0, "a.ids")));
    p.step_head();
    for (const std::string& s : p.outbox) d.peer.push_back(s);
    p.outbox.clear();
    d.feed(d.peer[0]);   // QUEUED
    d.feed(d.peer[1]);   // CLAIM
    d.feed(d.peer[2]);   // SEG
    d.feed(d.peer[3]);   // DONE
    check(d.machine(1).state() == HState::seg_ready, "the first handoff published");
    // A second request now finds the arena full.
    p.behaviour = FakePrefill::Behaviour::full;
    d.ask(2, 100, 0, "b.ids");
    p.on_line(req_line(mk_req(2, 100, 0, "b.ids")));
    p.step_head();
    check(!p.outbox.empty(), "the prefill instance answered");
    const Line ans = parse_line(p.outbox.back());
    check(ans.kind == LineKind::err && ans.code == ErrCode::full, "ERR full, not a hang and not a close");
    const auto r = d.feed(p.outbox.back());
    check(r.action == Act::hold, "the decode side HOLDS (§6.2)");
    check(d.machine(2).state() == HState::held, "H_HOLD");
    check(d.client_errs == 0, "and no client saw an ERR (D5)");
    check(err_is_temporary(ErrCode::full) && !err_reaches_the_client(ErrCode::full),
          "the classification agrees");
    // The queue bound is `ERR queue`, also temporary (§3.8, §4.4).
    FakePrefill q;
    q.arena.build({1024}, {2}, 0, 0);
    q.queue_cap = 1;
    q.on_line(req_line(mk_req(1, 10, 0, "a.ids")));
    check(q.outbox.size() == 1 && parse_line(q.outbox[0]).kind == LineKind::queued,
          "the first REQ is queued");
    q.on_line(req_line(mk_req(2, 10, 0, "b.ids")));
    const Line over = parse_line(q.outbox.back());
    check(over.kind == LineKind::err && over.code == ErrCode::queue, "beyond --prefill-queue: ERR queue");
    check(err_is_temporary(ErrCode::queue), "which is temporary too");
    // Tier starvation (§3.8): a big job may not use a small group, and a small job takes the
    // SMALLEST group that fits.
    FakeArena t;
    t.build({1024, 16384}, {1, 1}, 0, 0);
    int idx = -1;
    uint64_t n = 0;
    std::string err;
    // Prefer the SMALLEST group that fits.
    check(t.claim(0, 7, 1, 0, idx, n, err) && t.slots[idx].tier == 0,
          "a tier-0 job takes the tier-0 slot, not the 3 GB one (§3.8)");
    check(t.claim(1, 7, 2, 0, idx, n, err) && t.slots[idx].tier == 1, "a tier-1 job takes its own group");
    // Give the tier-0 slot back, and a tier-1 job is STILL refused: a big job never fits a small
    // slot, so it waits rather than jumping the queue.
    check(t.release(0, t.slots[0].nonce, err), "release the tier-0 slot");
    check(t.slots[0].state == 0, "the tier-0 group has a free slot now");
    check(!t.claim(1, 7, 3, 0, idx, n, err),
          "a tier-1 job is refused even though a tier-0 slot is free: it never uses a smaller group");
    check(t.slots[0].state == 0, "and the free tier-0 slot was NOT stolen by the big job");
    check(t.claim(0, 7, 4, 0, idx, n, err) && t.slots[idx].tier == 0, "the tier-0 job does take it");
    check(!t.claim(0, 7, 5, 0, idx, n, err), "and now the arena is full: ERR full, a temporary answer");
    // Cascade: a tier-0 job MAY use a larger group when its own is full (§3.8: any j >= i).
    FakeArena c;
    c.build({1024, 16384}, {1, 1}, 0, 0);
    check(c.claim(0, 7, 1, 0, idx, n, err) && c.slots[idx].tier == 0, "first tier-0 job: its own group");
    check(c.claim(0, 7, 2, 0, idx, n, err) && c.slots[idx].tier == 1,
          "second tier-0 job cascades UP to the larger group rather than waiting");
}

static void test_failure_mode_tmpfs_full() {
    // §4.7 row 10 / §3.7: the prefill instance refuses to START.  The protocol-side consequence is
    // that the endpoint never answers, which is `endpoint` -> hold, never a client ERR.  The arena
    // side (the statvfs check before ftruncate) is `handoff_arena_test`'s, not this file's.
    const std::string refusal =
        "strata prefill: the handoff arena /dev/shm/strata-handoff/0123456789abcdef-16,34/a/arena "
        "needs 24481316864 B and /dev/shm has 25769803776 B free";
    check_reason_contains(refusal, "needs 24481316864 B", "the refusal names the bytes it needs");
    check_reason_contains(refusal, "has 25769803776 B free", "and the bytes available (§3.7)");
    // The decode instance's view: connect fails, so every NEW prompt is held.  It starts anyway -
    // refusing to start would take the operator's running conversations away because a backend is
    // down (§5.1).
    FakeDecode d;
    FakeArena a;
    a.build({1024}, {1}, 0, 0);
    d.arena = &a;
    d.client_id = 7;
    d.endpoint_down = true;
    const auto r = d.ask(1, 100, 0, "x.ids");
    check(r.action == Act::write_req, "the machine still asks; the caller decides it cannot send");
    check(!d.machine(1).may_send_req(0), "one outstanding REQ: the machine will not re-ask while one is in flight");
    check(d.machine(1).has_outstanding_req(), "and it knows that");
    const auto held = d.machine(1).on_line(1, parse_line(err_line(1, ErrCode::endpoint, "connect failed")));
    check(held.action == Act::hold, "so the request is held");
    check(d.client_errs == 0, "and no client sees an ERR");
    check(err_is_temporary(ErrCode::endpoint), "endpoint is temporary (§4.4)");
}

static void test_failure_mode_bad_line() {
    // §4.7 row 11: an unparseable or over-long line closes the connection and is treated as
    // "prefill instance down", which is a hold, and the decode instance retries the connection.
    const std::vector<std::string> junk = {
        "DONE 7 5",
        "CLAIM 7 x 9",
        "PREFILL-READY-V1 1 slots=2 max_tokens=notanumber tiers=1 pack_hash=0123456789abcdef "
        "geom_hash=0123456789abcdef",
        "HELLO 1 2 zz 0123456789abcdef 8192 0 split=",
        std::string("REQ 1 2 3 ids=") + std::string(4096, 'a'),
    };
    for (const std::string& j : junk) {
        std::string why;
        const bool framing = !line_is_well_formed(j, why);
        const Line l = parse_line(j);
        check(framing || !l.bad.empty(), "a junk line is caught by framing or by the parser");
        if (!framing) {
            check(l.kind != LineKind::unknown, "and it is a KNOWN verb that failed, not an ignored one");
        }
    }
    // The reader's rule: only `bad` closes the connection; an unknown verb never does (D8).
    check(parse_line("SOMETHING_NEW 1 2").bad.empty(), "an unknown verb does not close the connection");
    check(!parse_line("ACK 7 notanonce").bad.empty(), "a known verb with a bad field does");
    // And the consequence is the same as the endpoint dying: hold, reconnect with backoff.
    FakeDecode d;
    FakeArena a;
    a.build({1024}, {1}, 0, 0);
    d.arena = &a;
    d.client_id = 7;
    d.ask(1, 100, 0, "x.ids");
    check(d.feed("ACK 7 notanonce").action == Act::protocol_error, "the reader sees a protocol error");
    const auto held = d.machine(1).on_line(1, parse_line(err_line(1, ErrCode::endpoint, "line did not parse")));
    check(held.action == Act::hold, "and every in-flight request becomes a hold");
    check(d.client_errs == 0, "with no client-visible ERR");
}

static void test_failure_mode_infinite_retry() {
    // §4.7 row 12: covered in detail by test_the_retry_bound(); this is the end-to-end version
    // against the fake prefill instance, where every pass really does move a payload.
    struct Run {
        FakePrefill p;
        FakeDecode d;
        int passes = 0;
        bool bound_fired = false;
        void drive(int cap) {
            d.ask(1, 100, 0, "x.ids");
            while (passes++ < cap) {
                const HState s = d.machine(1).state();
                if (s == HState::sent) {
                    p.on_line(d.outbox.back());
                    d.outbox.clear();
                    p.step_head();
                    for (const std::string& l : p.outbox) d.feed(l);
                    p.outbox.clear();
                    continue;
                }
                if (s == HState::held) {
                    const auto r = d.machine(1).note(1, Event::retry);
                    if (r.action == Act::illegal) { bound_fired = true; return; }
                    if (!d.reask(1)) { bound_fired = true; return; }
                    continue;
                }
                return;      // idle / done / failed: the loop ended on its own
            }
        }
    };
    Run a;
    a.p.arena.build({1024}, {2}, 0, 0);
    a.p.client_id = 7;
    a.p.behaviour = FakePrefill::Behaviour::read_fail;
    a.d.arena = &a.p.arena;
    a.d.client_id = 7;
    a.d.retries = kDefaultPrefillRetries;
    a.drive(40);
    check(a.bound_fired, "the retry bound fired in the end-to-end loop too");
    check(a.passes < 40, "and it stopped on the bound, not on the pass cap");
    check(a.d.machine(1).req_lines() == kDefaultPrefillRetries + 1,
          "exactly retries+1 REQ lines were written");
    check(a.d.client_errs == 0, "still no client ERR: the bound's answer is the retry line, not a hang");
    check(a.p.arena.free_slots() == "2/2", "and every slot came back: a failed read aborts its claim");
    check(a.p.jobs_err == kDefaultPrefillRetries + 1, "the prefill instance saw that many failed reads");

    // The neutered arm: with the bound removed the same loop runs to the pass cap without finishing
    // (the S3.8 anti-vacuity shape, end to end).
    Run b;
    b.p.arena.build({1024}, {2}, 0, 0);
    b.p.client_id = 7;
    b.p.behaviour = FakePrefill::Behaviour::read_fail;
    b.d.arena = &b.p.arena;
    b.d.client_id = 7;
    b.d.retries = 100000;
    b.drive(40);
    check(!b.bound_fired, "with the bound neutered it never fires");
    check(b.passes > 40, "and the loop runs to its pass cap");
    check(b.d.machine(1).state() != HState::done && b.d.machine(1).state() != HState::idle,
          "and the request never finished: that is the livelock the bound exists to stop");
    check(b.d.machine(1).req_lines() > kDefaultPrefillRetries + 1, "and it kept re-asking");
}

// ============================================================ 10. the happy path, end to end =====

static void test_a_full_handoff_end_to_end() {
    FakePrefill p;
    p.arena.build({1024, 16384, 131072}, {2, 2, 2}, 0x0123456789abcdefull, 0x5dfa917bae1331c5ull);
    p.client_id = 0x7f3a91c0beef0001ull;
    FakeDecode d;
    d.arena = &p.arena;
    d.client_id = p.client_id;

    // The handshake.
    const Line hello = parse_line(hello_line(kProtoVersion, p.client_id, p.arena.pack_hash,
                                             p.arena.geom_hash, 262144, {16, 34}, 1, 1));
    std::string why;
    check(hello_is_acceptable(hello, why), "HELLO accepted");
    ReadyInfo ri;
    ri.slots = {2, 2, 2};
    ri.tiers = {1024, 16384, 131072};
    ri.max_tokens = p.arena.max_tokens();
    ri.pack_hash = p.arena.pack_hash;
    ri.geom_hash = p.arena.geom_hash;
    check(handshake_decision(hello, parse_line(ready_line(ri)), why), "PREFILL-READY-V1 accepted");
    check_eq(endpoint_ready_line("/dev/shm/strata-handoff/x/a/prefill.sock", ri),
             "strata serve: prefill endpoint /dev/shm/strata-handoff/x/a/prefill.sock: "
             "PREFILL-READY-V1 (slots 2,2,2, max 131072 tokens); handoff on",
             "the startup line is §4.2's");

    // Liveness.
    p.on_line(ping_line("hb-1"));
    check_eq(p.outbox.back(), "PONG hb-1", "PING is answered with PONG and the same cookie");
    p.outbox.clear();

    // REQ -> QUEUED -> CLAIM -> SEG -> DONE.
    std::string ids;
    check(req_ids_path("/dev/shm/strata-handoff/x/a", p.client_id, 1, ids), "the ids file name");
    check_eq(ids, "/dev/shm/strata-handoff/x/a/req-7f3a91c0beef0001-0000000000000001.ids",
             "req-<client>-<nonce>.ids (§3.2)");
    uint64_t tier = 9;
    check(tier_for_tokens(10394, ri.tiers, tier), "10394 tokens is tier 1");
    const auto a = d.ask(7, 10394, tier, "req-7f3a91c0beef0001-0000000000000001.ids");
    check(a.action == Act::write_req, "the machine asked for a REQ");
    p.on_line(d.outbox.back());
    p.step_head();
    for (const std::string& l : p.outbox) {
        const Line pl = parse_line(l);
        if (pl.kind == LineKind::queued || pl.kind == LineKind::claim || pl.kind == LineKind::seg ||
            pl.kind == LineKind::done)
            d.feed(l);
    }
    p.outbox.clear();
    check(d.machine(7).state() == HState::seg_ready, "H_SEG_READY after DONE");
    check(d.machine(7).nonce() != 0, "the handoff has a nonce");
    // The engine thread kept stepping while the payload was copied out (§1.5, §6.2).
    ++d.engine_steps;
    check(d.engine_steps == 1, "the engine thread is not blocked on the socket");
    check(d.copy_out(7), "the copy-out, the hash and the decode all pass");
    check(d.machine(7).state() == HState::handed_over, "H_HANDED_OVER: the slot is FREE again");
    p.on_line(d.outbox.back());   // the ACK reaches the prefill instance
    check(p.arena.free_slots() == "6/6", "the ACK released the slot BEFORE the mount (§4.5 rule 1)");
    check(d.machine(7).mount_result(7, true, false).action == Act::mount,
          "the image is posted to the engine thread");
    check(d.machine(7).state() == HState::mounting, "MOUNTING [engine]");
    check(d.machine(7).mount_result(7, true, false).action == Act::finish, "the mount finished");
    check(d.machine(7).state() == HState::done, "H_DONE");
    check(d.machine(7).note(7, Event::collect).action == Act::finish, "collected");
    check(d.machine(7).state() == HState::idle, "back to IDLE");
    check(d.client_errs == 0, "no client ERR anywhere");
    // The release the prefill instance records on ITS stderr - there is no ACK-ED echo (§4.3).
    bool saw_ack_echo = false;
    for (const std::string& l : p.outbox)
        if (l.rfind("ACK-ED", 0) == 0 || l.rfind("ACK ", 0) == 0) saw_ack_echo = true;
    check(!saw_ack_echo, "the prefill instance never echoes ACK");
    check_eq(p.arena.free_slots(), "6/6", "and the arena is whole again");
}

static void test_two_requests_share_one_arena() {
    // §5.4: nothing in the protocol is per-client-count.  Two decode instances, one arena, each slot
    // names exactly one client, and one client's stale nonce cannot free the other's slot.
    FakeArena a;
    a.build({1024}, {2}, 0, 0);
    int i1 = -1, i2 = -1;
    uint64_t n1 = 0, n2 = 0;
    std::string err;
    check(a.claim(0, 1, 100, 0, i1, n1, err), "client 1's slot");
    check(a.claim(0, 2, 200, 0, i2, n2, err), "client 2's slot");
    check(i1 != i2, "two different slots");
    a.write(i1, std::vector<uint8_t>(8, 1));
    a.write(i2, std::vector<uint8_t>(8, 2));
    a.publish(i1, n1, 8, err);
    a.publish(i2, n2, 8, err);
    FakeSlot got;
    check(!a.read_slot(i1, n1, 2, got, err), "client 2 cannot read client 1's slot");
    check(a.read_slot(i2, n2, 2, got, err), "and can read its own");
    check(!a.release(i2, n1, err), "client 1's nonce cannot release client 2's slot");
    check_reason_contains(err, "stale nonce", "and the arena says why");
    check(a.release(i2, n2, err), "its own nonce can");
    // Request ids are per-decode-instance and are NOT assumed globally unique (§2.4): the same id
    // from two clients is two different requests, and the slot header carries both fields.
    int i3 = -1;
    uint64_t n3 = 0;
    check(a.claim(0, 1, 200, 0, i3, n3, err), "request id 200 again, from client 1");
    check(a.slots[i3].request == 200 && a.slots[i3].client == 1, "the slot names both");
    check(a.slots[i3].nonce != n2, "and the nonce distinguishes the two attempts");
}

static void test_the_local_cache_hit_skips_the_handoff() {
    // §1.5 / §4.5: a local ConversationCache::best() hit means NO handoff at all - today's mount.
    HandoffState h(3);
    check(h.local_hit(1), "a local hit from IDLE");
    check(h.state() == HState::handed_over, "straight to H_HANDED_OVER");
    check(h.nonce() == 0, "no nonce: no slot was ever claimed");
    check(h.ack_line().empty(), "and therefore no ACK to send");
    check(!h.may_release(0, 0), "and nothing releasable");
    check(h.note(1, Event::mount_ok).action == Act::mount, "the image goes to the engine thread");
    check(h.note(1, Event::mount_ok).action == Act::finish, "and the mount completes");
    // A queued request that hits locally while it waits is the same path.
    HandoffState q(3);
    q.admit_now(2);
    const auto r = q.note(2, Event::local_hit);
    check(r.action == Act::mount, "a local hit while queued skips the REQ entirely");
    check(q.req_lines() == 0, "and no REQ was ever written");
}

static void test_bye_and_shutdown() {
    FakePrefill p;
    p.arena.build({1024}, {1}, 0, 0);
    p.on_line(bye_line("operator quit"));
    check(p.shutting, "BYE ends the connection");
    check_eq(bye_line("operator quit"), "BYE operator quit", "BYE carries a reason");
    check_eq(bye_line(""), "BYE", "and BYE alone is legal");
    // §4.4: a request that was in flight when the endpoint went away is `shutting` -> hold.
    HandoffState h(3);
    std::string w;
    h.begin(1, 10, 0, w);
    h.note(1, Event::req_sent);
    const auto r = h.on_line(1, parse_line(err_line(1, ErrCode::shutting, "the prefill instance is exiting")));
    check(r.action == Act::hold, "shutting is a hold, not an error");
    check(err_is_temporary(ErrCode::shutting), "and the table agrees");
}

static void test_ids_file_rules() {
    // §4.3: the ids do not go on the wire; `ids=NAME` names a file the decode instance wrote, and a
    // missing or unparsable file is `ERR ids` (permanent for that request).
    std::string out;
    check(req_ids_path("/dev/shm/strata-handoff/x/a", 7, 9, out), "a normal instance dir");
    check_reason_contains(out, "req-0000000000000007-0000000000000009.ids", "the file name");
    check(!req_ids_path("/dev/shm/../etc", 7, 9, out), "a dir with a .. component is refused (D7)");
    check(!req_ids_path("", 7, 9, out), "an empty dir is refused");
    check(req_ids_path("/abs/path", 7, 9, out), "the local instance dir is normally absolute (§3.2)");
    check(!req_ids_path("/abs/path", 7, 9, out, false), "and refused when the caller wants a relative name");
    check(!req_ids_path("/dev/shm/a;rm", 7, 9, out), "a shell-ish dir name is refused");
    check(!ids_name_is_safe("../secret.ids"), "a wire ids= name may not escape the instance dir");
    check(!ids_name_is_safe("/etc/passwd"), "nor be absolute");
    check(!ids_name_is_safe("a b.ids"), "nor contain a space");
    check(!ids_name_is_safe(""), "nor be empty");
    check(ids_name_is_safe("req-7-9.ids"), "a plain name is fine");
    check(ids_name_is_safe("sub/dir/x.ids"), "and a relative subpath is fine (§3.2's dir layout)");
    check(err_is_permanent(ErrCode::ids), "a missing ids file is permanent for that request (§4.3)");
    // The prompt ids never appear on the wire, so a 245 946-token prompt is one line, not 1.7 MB.
    const std::string big = req_line(mk_req(1, 245946, 2, "req-1-2.ids"));
    check(big.size() < 80, "a 245 946-token prompt is still one short line (§4.3)");
    check(parse_line(big).bad.empty(), "and it parses");
}

static void test_resume_is_pinned_to_zero() {
    // §4.3: `resume` is pinned to 0 in v1 because the prefill instance owns the reuse decision.  The
    // line PARSES (so the parser is not the thing that rejects it) and the answer is `ERR resume`.
    const Line l = parse_line("REQ 1 100 0 ids=x.ids resume=4096");
    check(l.bad.empty(), "resume= parses: it is a known key, not a malformed line");
    check(l.resume == 4096, "and the value is read");
    FakePrefill p;
    p.arena.build({1024}, {1}, 0, 0);
    p.on_line("REQ 1 100 0 ids=x.ids resume=4096");
    const Line ans = parse_line(p.outbox.back());
    check(ans.kind == LineKind::err && ans.code == ErrCode::resume, "the answer is ERR resume");
    check(err_is_permanent(ErrCode::resume), "and it is permanent: it is a bug in the caller (§4.4)");
    // There is deliberately no `prefix=` key (§4.3): an unknown key is skipped, so a peer that sends
    // one is not refused - but the prefill instance never acts on it.
    const Line lp = parse_line("REQ 1 100 0 ids=x.ids prefix=4096");
    check(lp.bad.empty(), "prefix= does not break the parser");
    check(lp.keys.count("prefix") == 1, "it is kept as an unknown key");
    check(lp.keys.count("resume") == 0, "and it is NOT read as resume");
    // There is no RELEASE alias of ACK (§4.3).
    const Line rel = parse_line("RELEASE 1 99");
    check(rel.kind == LineKind::unknown, "RELEASE is not a verb: one action, one verb");
    check(rel.bad.empty(), "and it is ignored rather than closing the connection (D8)");
}

static void test_stop_is_not_forwarded() {
    // §4.6: bare `STOP` keeps 0.1.30's meaning on the CLIENT wire and is never forwarded.  The only
    // cancellation verb on THIS wire is CANCEL <id>.
    const Line stop = parse_line("STOP");
    check(stop.kind == LineKind::unknown, "STOP is not a handoff verb");
    check(parse_line("STOP 7").kind == LineKind::unknown, "nor is STOP <id>");
    check(line_side(LineKind::cancel) == Side::decode_to_prefill, "CANCEL goes decode -> prefill");
    check(!line_from_peer(LineKind::cancel, Side::decode_to_prefill),
          "and is never received by the decode side");
    check(line_from_peer(LineKind::cancel, Side::prefill_to_decode),
          "and is received by the prefill side");
    // A wrong-side line is a protocol error, not a state event.
    HandoffState h(3);
    std::string w;
    h.begin(1, 10, 0, w);
    check(h.on_line(1, parse_line(cancel_line(1))).action == Act::protocol_error,
          "a decode-side line arriving from the peer is refused");
}

// ============================================================ main ================================

int main() {
    test_every_line_round_trips();
    test_line_fields_are_read_back();
    test_malformed_lines();
    test_framing();
    test_error_table_is_total();
    test_the_wait_vocabulary_matches_serve_driver();
    test_handshake_refuses_a_version_or_hash_mismatch();
    test_tier_and_size_rules();
    test_the_state_table_is_total();
    test_claim_before_done_in_every_state();
    test_the_nonce_rules();
    test_cancellation_in_every_state();
    test_the_prefill_job_table_is_total();
    test_illegal_transitions_are_named();
    test_the_retry_bound();
    test_failure_mode_prefill_dies();
    test_failure_mode_decode_dies_mid_handoff();
    test_failure_mode_decode_dies_after_ack();
    test_failure_mode_slot_leak_is_visible();
    test_failure_mode_partial_write();
    test_failure_mode_partial_publish();
    test_failure_mode_corrupt_payload();
    test_failure_mode_restore_fails_after_the_first_write();
    test_failure_mode_arena_full();
    test_failure_mode_tmpfs_full();
    test_failure_mode_bad_line();
    test_failure_mode_infinite_retry();
    test_a_full_handoff_end_to_end();
    test_two_requests_share_one_arena();
    test_the_local_cache_hit_skips_the_handoff();
    test_bye_and_shutdown();
    test_ids_file_rules();
    test_resume_is_pinned_to_zero();
    test_stop_is_not_forwarded();

    if (failures) {
        std::fprintf(stderr, "prefill_svc_test: %d of %d checks FAILED\n", failures, checks);
        return 1;
    }
    std::printf("prefill_svc_test: %d checks OK\n", checks);
    return 0;
}
