// src/program/slot_test.cpp - the stage-3 slot object, its state machine and the scheduler's pick rule
// (docs/STAGE3-CONCURRENCY.md §2/§3.3/§3.4, plan step S3.1b).  CPU only: no CUDA, no model, no engine.
//
// What is under test is the part that can be wrong without anyone noticing until a token is wrong: the
// transition table, the admission cap, the fairness bound, and the cancel-unwind ordering.
#include "strata/program/slot.hpp"

#include <cstdio>
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
    std::fprintf(stderr, "FAIL: %s\n  want: [%s]\n  got:  [%s]\n", what, want.c_str(), got.c_str());
}

using namespace strata::program;
using slot::State;
using slot::Slot;
using slot::Registry;
using slot::Pick;
using serve_proto::kNoId;

const State kAll[] = {State::idle, State::queued, State::prefilling, State::decoding,
                      State::parked, State::cancelling, State::error};

// ------------------------------------------------------------------ the state names -------------------
void test_state_names() {
    const char* names[] = {"idle", "queued", "prefilling", "decoding", "parked", "cancelling", "error"};
    for (int i = 0; i < 7; ++i) {
        check_eq(slot::state_name((State) i), names[i], "state_name matches the wire spelling of §6.2");
        State back = State::idle;
        check(slot::state_of(names[i], back) && back == (State) i, "state_of round-trips");
    }
    State junk = State::idle;
    check(!slot::state_of("Running", junk) && !slot::state_of("", junk),
          "an unknown state name is refused, not silently mapped to idle");
}

// ------------------------------------------------------------------ the transition table ------------
void test_transition_table() {
    // Every edge the design's diagram (§3.4) draws, one by one.
    check(slot::can_transition(State::idle, State::queued), "a new request queues");
    check(slot::can_transition(State::queued, State::prefilling), "admit -> prefill");
    check(slot::can_transition(State::queued, State::decoding), "a short read goes straight through windows");
    check(slot::can_transition(State::queued, State::cancelling), "STOP before it ever ran");
    check(slot::can_transition(State::queued, State::error), "refused at admission (budget, slots, context)");
    check(slot::can_transition(State::queued, State::idle), "a refused slot is dropped");
    check(slot::can_transition(State::prefilling, State::prefilling), "the next chunk");
    check(slot::can_transition(State::prefilling, State::decoding), "the prompt is read");
    check(slot::can_transition(State::prefilling, State::cancelling), "STOP mid-prompt");
    check(slot::can_transition(State::prefilling, State::error), "a chunk failed");
    check(slot::can_transition(State::prefilling, State::parked), "its prefix is kept for reuse, read unfinished");
    check(slot::can_transition(State::decoding, State::decoding), "the next window");
    check(slot::can_transition(State::decoding, State::parked), "finished, image kept (stage 2's parked prefix)");
    check(slot::can_transition(State::decoding, State::cancelling), "STOP mid-decode");
    check(slot::can_transition(State::decoding, State::error), "a window failed");
    check(slot::can_transition(State::decoding, State::idle), "DONE, image not kept");
    check(slot::can_transition(State::parked, State::decoding), "mount / continue");
    check(slot::can_transition(State::parked, State::prefilling), "mount and keep reading");
    check(slot::can_transition(State::parked, State::idle), "pruned or finished");
    check(slot::can_transition(State::cancelling, State::parked), "the unwind parks what it read");
    check(slot::can_transition(State::cancelling, State::idle), "the unwind finished, DONE");
    check(slot::can_transition(State::cancelling, State::error), "the unwind itself failed");
    check(slot::can_transition(State::error, State::idle), "the slot is destroyed, the process stays up");

    // The edges that must NOT exist.  Each one is a bug the table is there to catch.
    check(!slot::can_transition(State::idle, State::decoding), "nothing runs from idle");
    check(!slot::can_transition(State::idle, State::parked), "idle is not a parked conversation");
    check(!slot::can_transition(State::queued, State::parked), "a queued request allocated no state to park");
    check(!slot::can_transition(State::prefilling, State::queued), "a request does not go back to the queue");
    check(!slot::can_transition(State::parked, State::queued), "a parked image is not re-admitted as new");
    check(!slot::can_transition(State::cancelling, State::decoding), "a cancel is not un-stopable");
    check(!slot::can_transition(State::cancelling, State::prefilling), "nor restarted as a chunk");
    check(!slot::can_transition(State::cancelling, State::cancelling), "STOP twice is one cancel");
    check(!slot::can_transition(State::error, State::decoding), "a failed slot never runs again");
    check(!slot::can_transition(State::error, State::error), "and does not fail twice");
    check(!slot::can_transition(State::idle, State::idle), "idle does not self-transition");
    check(!slot::can_transition(State::parked, State::parked), "parked does not self-transition");

    // The table is the table: enumerate it and pin its size, so adding an edge is a deliberate act.
    // 1 (idle) + 5 (queued) + 6 (prefilling) + 5 (decoding) + 4 (parked) + 3 (cancelling) + 1 (error).
    int edges = 0;
    for (State a : kAll) for (State b : kAll) if (slot::can_transition(a, b)) ++edges;
    check(edges == 25, "the table has exactly 25 edges: §3.4's diagram plus the two step self-edges");
}

// ------------------------------------------------------------------ the state classes ---------------
void test_state_classes() {
    check(!slot::is_active(State::queued), "QUEUED holds no state, so it takes no active-slot permit (§3.4)");
    check(slot::is_active(State::prefilling) && slot::is_active(State::decoding) &&
          slot::is_active(State::cancelling), "the three states that occupy a slot");
    check(!slot::is_active(State::parked), "PARKED costs host RAM the parking budget governs (§2.3)");
    check(!slot::is_active(State::idle) && !slot::is_active(State::error), "idle/error occupy nothing");
    check(slot::has_work(State::queued), "a queued request is still work (admission)");
    check(!slot::has_work(State::parked) && !slot::has_work(State::idle), "parked and idle are not");
    check(slot::is_running(State::prefilling) && slot::is_running(State::decoding),
          "only prefill/decode are mid-step, so only they block a swap");
    check(!slot::is_running(State::cancelling), "a cancelling slot is not starting a new step");
}

// ------------------------------------------------------------------ registry basics -----------------
void test_registry_add_find_release() {
    Registry reg(2, 250);
    check(reg.max_active() == 2 && reg.concurrent(), "two slots = concurrency on");
    check(!Registry(1, 250).concurrent() && !Registry(0, 250).concurrent(),
          "0 or 1 slot = today's serial engine, untagged wire");
    check(Registry(99, 250).max_active() == slot::kMaxActive,
          "--serve-slots above the ceiling is clamped, never silently honoured (risk R1)");
    check(Registry(-3, -5).max_active() == 0 && Registry(-3, -5).starve_ms() == 0, "garbage in, safe out");

    Slot* a = reg.add(7, 1000);
    check(a != nullptr && a->id == 7 && a->state == State::queued, "add queues a request");
    check(a->created_ms == 1000 && a->started_ms == 0 && a->last_ran_ms == 0, "clocks start at 0");
    check(!a->cancel.load(), "not cancelled");
    check(a->parked_index == -1 && a->parked_bytes == 0, "no image yet");
    check(reg.find(7) == a && reg.find(8) == nullptr, "find by id");
    check(reg.add(7, 1100) == nullptr, "a duplicate live id is refused - two live requests with one id make "
                                       "the wire unparseable");
    check(reg.add(kNoId, 0) == nullptr, "an id-less request never reaches the registry");
    check(reg.count() == 1 && reg.active_count() == 0, "queued counts, but occupies no active slot");

    std::string err;
    check(reg.transition(7, State::prefilling, err), "queued -> prefilling");
    check(reg.active_count() == 1, "prefilling occupies an active slot");
    check(reg.find(7)->started_ms == 1000, "admission stamps started_ms from created_ms");
    check(reg.transition(7, State::prefilling, err), "a second chunk is legal");
    check(reg.release(7), "release");
    check(reg.find(7) == nullptr && reg.count() == 0, "the row is free again");
    check(reg.add(7, 2000) != nullptr, "and the id can be reused once the request is gone");
    check(!reg.release(999), "releasing nothing says so");
}

void test_illegal_transition_is_refused() {
    Registry reg(2, 250);
    reg.add(1, 0);
    std::string err;
    check(!reg.transition(1, State::parked, err), "queued -> parked is not in the table");
    check(reg.find(1)->state == State::queued, "and the state was NOT changed by the refusal");
    check(err.find("illegal transition") == 0, "the error names the reason");
    check(!reg.transition(42, State::queued, err), "no such slot");
}

void test_admission_cap() {
    Registry reg(2, 250);
    std::string err;
    reg.add(1, 0); reg.add(2, 0); reg.add(3, 0);
    check(reg.can_admit(), "nothing active yet");
    check(reg.transition(1, State::prefilling, err), "first admit");
    check(reg.transition(2, State::decoding, err), "second admit");
    check(!reg.can_admit(), "the cap is reached");
    check(!reg.transition(3, State::prefilling, err), "the third is refused");
    check(err == "no active slot free", "and says why, so the engine can ERR it (§3.4's refuse path)");
    check(reg.find(3)->state == State::queued, "still queued: refusal is not destruction");
    // a conversation with no pending work gives its permit back - that is the whole point of §2.3
    check(reg.transition(2, State::parked, err), "finished, image kept");
    check(reg.can_admit(), "the permit came back");
    check(reg.transition(3, State::decoding, err), "so the third is admitted");
    check(reg.parked_count() == 1, "and one conversation is parked");
    // an UNMOUNTED slot with work left keeps its permit (see is_active's note): the cap bounds host
    // images, and a paused prefill is exactly one of those.
    check(reg.active_count() == 2, "prefilling + decoding = two permits, the parked one is free");
}

void test_registry_full() {
    Registry reg(2, 250);
    for (int i = 1; i <= slot::kMaxSlots; ++i) check(reg.add(i, 0) != nullptr, "fill the registry");
    check(reg.add(100, 0) == nullptr, "and refuse one more (the engine ERRs, the process is fine)");
}

// ------------------------------------------------------------------ cancel / STOP -------------------
void test_cancel() {
    Registry reg(2, 250);
    std::string err;
    Slot* a = reg.add(1, 0);
    reg.transition(1, State::decoding, err);
    check(reg.cancel_request(1), "STOP 1 while decoding");
    check(a->cancel.load(), "the per-slot flag is set - today's single process-wide stop_req cannot say WHICH");
    check(a->state == State::cancelling, "and the state moved, so pick() sees it");
    check(reg.transition(1, State::idle, err), "the unwind ends in DONE");
    check(!reg.cancel_request(1), "a released slot cannot be cancelled");

    Slot* b = reg.add(2, 0);
    check(reg.cancel_request(2), "STOP 2 while queued");
    check(b->cancel.load() && b->state == State::queued, "a queued slot keeps its state and takes the flag");
    check(!reg.cancel_request(99), "STOP for an unknown id is refused, not silently ignored");
    check(!reg.cancel_request(kNoId), "and never for 'no id' - the caller resolves a bare STOP first");

    // a cancel mid-prompt unwinds through CANCELLING and can never start another step
    Slot* c = reg.add(3, 0);
    reg.transition(3, State::prefilling, err);
    reg.cancel_request(3);
    check(c->state == State::cancelling, "prefilling -> cancelling");
    check(!reg.transition(3, State::prefilling, err), "and it cannot resume: the step loop only unwinds");
    check(reg.transition(3, State::parked, err), "the unwind parks what it read");
    // STOP that arrives BEFORE admission: the slot is let in only far enough to unwind
    Slot* d = reg.add(4, 0);
    reg.cancel_request(4);
    check(reg.transition(4, State::decoding, err), "transition to a running state while cancelled ...");
    check(d->state == State::cancelling, "... lands in cancelling instead, never decoding");
}

void test_bare_stop_targets_the_newest_running() {
    Registry reg(3, 250);
    std::string err;
    reg.add(1, 0); reg.add(2, 10); reg.add(3, 20);
    reg.transition(1, State::decoding, err);
    reg.transition(2, State::decoding, err);
    reg.find(1)->started_ms = 5;
    reg.find(2)->started_ms = 15;
    check(reg.newest_id() == 2, "a bare STOP hits the newest RUNNING slot (0.1.30's meaning)");
    reg.cancel_request(reg.newest_id());
    check(reg.newest_id() == 1, "then the next newest running one");
    reg.cancel_request(reg.newest_id());
    check(reg.newest_id() == 3, "with nothing running, the newest queued one - the request that would run next");
    reg.cancel_request(reg.newest_id());
    check(reg.newest_id() == kNoId, "and once everything is cancelled, nothing");
}

// ------------------------------------------------------------------ pick(): the rules ---------------
void test_pick_idle() {
    Registry reg(2, 250);
    Pick p = reg.pick(0);
    check(p.action == Pick::Action::idle && p.id == kNoId, "nothing queued: idle, wait for a line");
    check_eq(p.why, "idle", "and says so");
}

void test_pick_stays_on_the_active_slot() {
    // One slot, nothing else waiting: zero swaps, i.e. today's behaviour exactly (§5.3: "the scheduler's
    // default is to stay on the active slot").
    Registry reg(1, 250);
    std::string err;
    reg.add(1, 0);
    reg.transition(1, State::decoding, err);
    reg.set_active(1);
    reg.note_ran(1, 100);
    Pick p = reg.pick(100);
    check(p.action == Pick::Action::run && p.id == 1, "the same slot's next window, no swap");
    check_eq(p.why, "active-decoding", "and it is not counted as a fairness decision");
}

void test_pick_prefers_prefill_over_decode() {
    // §3.3: "prefill before decode: it is the long pole".  Two shapes: the prefill already holds the
    // session (a run, no swap), and the session is elsewhere or unmounted (a swap/mount).
    Registry reg(3, 250);
    std::string err;
    reg.add(1, 0); reg.add(2, 0);
    reg.transition(1, State::prefilling, err);
    reg.transition(2, State::decoding, err);
    reg.set_active(1);
    reg.note_ran(1, 100);
    reg.find(2)->started_ms = 100;
    Pick p = reg.pick(100);
    check(p.action == Pick::Action::run && p.id == 1, "the active prefill keeps the session");
    check_eq(p.why, "prefill-next-chunk", "and the reason is the design's");

    // nothing mounted (the session was just unmounted): the prefill is the one to mount
    reg.set_active(kNoId);
    p = reg.pick(100);
    check(p.action == Pick::Action::swap && p.id == 1, "prefill is mounted before the decoder runs");
    check_eq(p.why, "prefill-next-chunk", "still the prefill rule, not fairness");
}

void test_pick_admits_before_running_more() {
    // A queued request with a free permit is admitted before the active slot gets another step: an admitted
    // request is the one that can start producing anything.
    Registry reg(2, 250);
    std::string err;
    reg.add(1, 0); reg.add(2, 50);
    reg.transition(1, State::prefilling, err);
    reg.set_active(1);
    reg.note_ran(1, 60);
    Pick p = reg.pick(60);
    check(p.action == Pick::Action::admit && p.id == 2, "admit 2");
    check(p.waited_ms == 10, "and report how long it waited");
    // with the cap reached it does NOT admit
    reg.transition(2, State::prefilling, err);
    p = reg.pick(70);
    check(p.action != Pick::Action::admit, "at --serve-slots 2 a third request cannot be admitted");
    // and admission is one per pick, oldest first: bringing a conversation in is itself a step
    Registry reg2(3, 250);
    reg2.add(1, 0); reg2.add(2, 5); reg2.add(3, 10);
    p = reg2.pick(10);
    check(p.action == Pick::Action::admit && p.id == 1, "the oldest queued request first (FIFO admission)");
}

void test_pick_fairness_bound() {
    // The active decoder keeps its slot until someone has waited past --starve-ms; then it is forced off.
    const int64_t starve = 250;
    Registry reg(2, starve);
    std::string err;
    reg.add(1, 0); reg.add(2, 10);
    reg.transition(1, State::decoding, err);
    reg.transition(2, State::decoding, err);
    reg.set_active(1);
    reg.note_ran(1, 100);
    reg.find(2)->last_ran_ms = 100;

    Pick p = reg.pick(100 + starve);
    check(p.action == Pick::Action::run && p.id == 1, "exactly at the bound: stay (the bound is strict)");
    p = reg.pick(100 + starve + 1);
    check(p.action == Pick::Action::swap && p.id == 2, "one ms past it: swap to the starving slot");
    check_eq(p.why, "fairness-starved", "and the reason says fairness, not round-robin");
    check(p.waited_ms == starve + 1, "with the wait reported");

    // --starve-ms 0 = never force a swap: the serial path stays serial even with two slots.
    Registry reg0(2, 0);
    reg0.add(1, 0); reg0.add(2, 10);
    reg0.transition(1, State::decoding, err);
    reg0.transition(2, State::decoding, err);
    reg0.set_active(1);
    reg0.note_ran(1, 100);
    reg0.find(2)->last_ran_ms = 100;
    p = reg0.pick(1000000);
    check(p.action == Pick::Action::run && p.id == 1, "starve-ms 0: the active slot never loses the session");

    // A long prefill must not starve a decoder.  This is the hole §3.3's T_starve exists to close, and the
    // design's pseudocode leaves it open (its rule 2 only fires when the active slot is DECODING, and its
    // rule 4 then hands the next chunk to the active PREFILLING slot forever).
    Registry reg3(2, starve);
    reg3.add(1, 0); reg3.add(2, 0);
    reg3.transition(1, State::prefilling, err);
    reg3.transition(2, State::decoding, err);
    reg3.set_active(1);
    reg3.note_ran(1, 0);
    reg3.find(2)->started_ms = 0;
    p = reg3.pick(100);
    check(p.action == Pick::Action::run && p.id == 1, "under the bound: the chunk still goes first");
    p = reg3.pick(starve + 1);
    check(p.action == Pick::Action::swap && p.id == 2, "past it: the decoder preempts the chunk stream");
    check_eq(p.why, "fairness-starved", "and it is the fairness rule that did it");

    // A slot that has NEVER run is starving by definition (created_ms is the clock).
    Registry reg4(2, starve);
    reg4.add(1, 0); reg4.add(2, 0);
    reg4.transition(1, State::decoding, err);
    reg4.transition(2, State::decoding, err);
    reg4.set_active(1);
    reg4.note_ran(1, 1000);
    p = reg4.pick(1000);
    check(p.action == Pick::Action::swap && p.id == 2, "a slot that has never had a window preempts one that has");
}

void test_pick_cancel_unwinds_first() {
    // §3.3 rule 1 and §4.4: a CANCELLING slot holds the prompt loan, so it unwinds before anything else
    // runs - including a prefill that is otherwise the long pole.
    Registry reg(3, 250);
    std::string err;
    reg.add(1, 0); reg.add(2, 0); reg.add(3, 0);
    reg.transition(1, State::decoding, err);
    reg.transition(2, State::prefilling, err);
    reg.set_active(1);
    reg.note_ran(1, 100);
    reg.find(2)->started_ms = 100;
    reg.cancel_request(3);                       // queued + cancelled
    Pick p = reg.pick(100000);
    check(p.action == Pick::Action::admit && p.id == 3, "the cancelled request is admitted so it can unwind");
    reg.transition(3, State::decoding, err);     // lands in cancelling (see transition())
    p = reg.pick(100000);
    check(p.action == Pick::Action::swap && p.id == 3, "then it is swapped to and unwound, before the prefill");
    check_eq(p.why, "cancel-unwind-first", "before the prefill, before fairness");
    // and if it is already the active slot, its unwind needs no swap
    reg.set_active(3);
    p = reg.pick(100000);
    check(p.action == Pick::Action::run && p.id == 3, "unwinding the active slot is a run, not a swap");
}

void test_pick_parked_is_not_work() {
    Registry reg(2, 250);
    std::string err;
    reg.add(1, 0);
    reg.transition(1, State::decoding, err);
    reg.transition(1, State::parked, err);
    reg.set_active(kNoId);
    Pick p = reg.pick(10000);
    check(p.action == Pick::Action::idle, "a parked conversation is not runnable work: the engine idles");
}

void test_pick_after_the_active_slot_dies() {
    // A step failed on the active slot: the process survives (§3.4 ERROR) and the session is free, so the
    // next slot runs without a swap.
    Registry reg(2, 250);
    std::string err;
    reg.add(1, 0); reg.add(2, 0);
    reg.transition(1, State::decoding, err);
    reg.transition(2, State::prefilling, err);
    reg.set_active(1);
    check(reg.transition(1, State::error, err), "a step failed");
    check(reg.release(1), "the slot is destroyed");
    check(reg.active_id() == kNoId, "release() clears the active marker - nothing is mounted");
    Pick p = reg.pick(500);
    check(p.action == Pick::Action::swap && p.id == 2, "slot 2 is mounted onto the free session");
}

// ------------------------------------------------------------------ a two-slot scenario -----------
void test_two_slot_interleaving() {
    // The shape S3.1e will run: A reads a long prompt, B joins and decodes; neither may starve the other,
    // and every pick must produce exactly one step.  A swap-out is NOT a state change (§5.3): the slot keeps
    // its work state and only `active_id` moves.
    Registry reg(2, 250);
    std::string err;
    reg.add(1, 0);
    reg.transition(1, State::prefilling, err);
    reg.set_active(1);

    int64_t now = 0;
    int a_steps = 0, b_steps = 0, swaps = 0, admits = 0;
    for (int i = 0; i < 40; ++i) {
        if (i == 5) reg.add(2, now);
        Pick p = reg.pick(now);
        if (p.action == Pick::Action::admit) {
            reg.transition(p.id, State::decoding, err);
            ++admits;
            now += 1;
            continue;
        }
        if (p.action == Pick::Action::idle) break;
        if (p.action == Pick::Action::swap) {
            reg.set_active(p.id);
            reg.note_swap(40);
            ++swaps;
        }
        reg.note_ran(p.id, now);
        if (p.id == 1) ++a_steps; else ++b_steps;
        now += 100;                       // a chunk or a window per step
    }
    check(admits == 1, "B was admitted exactly once");
    check(a_steps > 0 && b_steps > 0, "both slots got steps");
    check(a_steps < 40 && b_steps < 40, "neither monopolised the session");
    check(a_steps + b_steps == 39, "every other pick produced exactly one step");
    check(swaps > 0 && swaps == (int) reg.swaps(), "the engine's swaps are the ones the registry reports");
    check(reg.swap_ms() == 40 * reg.swaps(), "and their cost is accumulated for /metrics (§6.3)");
    // the fairness bound held: nobody waited more than the bound plus one step's length
    check(reg.max_starve_wait(now, reg.active_id()) <= 350,
          "no slot waited more than the bound plus one step's length");
}

void test_wait_ms_accounting() {
    Registry reg(2, 250);
    std::string err;
    reg.add(1, 0);
    reg.transition(1, State::decoding, err);
    reg.note_ran(1, 100, 24);      // one 24 ms window
    reg.note_ran(1, 200, 24);      // another, 76 ms after the first ended
    check(reg.find(1)->run_ms == 48, "run_ms accumulates only the time inside a step");
    check(reg.find(1)->wait_ms(200) == 152, "of a 200 ms life, 48 ms was work and 152 ms was waiting");
    check(reg.find(1)->last_ran_ms == 200, "and the fairness clock moves");
    reg.note_ran(1, 150, 0);       // a clock that goes backwards must not corrupt the fairness clock
    check(reg.find(1)->last_ran_ms == 200, "last_ran_ms is monotone by construction");
    // §7.2: a slot that waited for the GPU reports the wait, so its prompt_ms is not read as a slowdown
    Slot* b = reg.add(2, 0);
    reg.transition(2, State::decoding, err);
    reg.note_ran(2, 900, 100);
    check(b->wait_ms(1000) == 900, "900 ms of waiting, 100 ms of work, in a 1000 ms life");
    check(b->wait_ms(50) == 0, "and never negative, whatever the clock does");
}

void test_counters_feed_the_slot_line() {
    // The SLOT line's fields come straight off the Slot; pin the pair so /slots cannot drift.
    Registry reg(2, 250);
    Slot* s = reg.add(7, 0);
    s->ctx_used = 1200; s->ctx_cap = 524288; s->prompt_tokens = 35957; s->generated = 233;
    s->parked_bytes = 1073741824;
    const serve_proto::Out out(true);
    const std::string line = out.slot(s->id, s->state_name(), s->ctx_used, s->ctx_cap, s->prompt_tokens,
                                      s->generated, s->parked_bytes);
    check_eq(line, "SLOT 7 queued 1200 524288 35957 233 1073741824", "the §6.2 field order and spelling");
    // and the state token in it is one the server can map back to a State
    const size_t sp = line.find(' ', line.find(' ') + 1);
    const size_t sp2 = line.find(' ', sp + 1);
    State st = State::idle;
    check(slot::state_of(line.substr(sp + 1, sp2 - sp - 1), st) && st == State::queued,
          "the state token round-trips through state_of()");
}

void test_each_enumerates_the_rows() {
    Registry reg(2, 250);
    reg.add(5, 0); reg.add(9, 0); reg.release(5);
    std::vector<int64_t> ids;
    reg.each([&](const Slot& s) { ids.push_back(s.id); });
    check(ids.size() == 1 && ids[0] == 9, "each() walks the occupied rows, for /slots");
}

}  // namespace

int main() {
    test_state_names();
    test_transition_table();
    test_state_classes();
    test_registry_add_find_release();
    test_illegal_transition_is_refused();
    test_admission_cap();
    test_registry_full();
    test_cancel();
    test_bare_stop_targets_the_newest_running();
    test_pick_idle();
    test_pick_stays_on_the_active_slot();
    test_pick_prefers_prefill_over_decode();
    test_pick_admits_before_running_more();
    test_pick_fairness_bound();
    test_pick_cancel_unwinds_first();
    test_pick_parked_is_not_work();
    test_pick_after_the_active_slot_dies();
    test_two_slot_interleaving();
    test_wait_ms_accounting();
    test_counters_feed_the_slot_line();
    test_each_enumerates_the_rows();
    std::fprintf(stderr, "slot_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
