// src/program/serve_swap_test.cpp - stage 3 S3.1d: the slot mount/unmount ORDER and its decisions,
// on a CPU, with a fake session.
//
// What is actually being tested is the thing that cannot be tested any other way short of starting
// the engine: the ORDER of a hand-over, and what each failure does to the live session.  The snapshot
// machinery itself is covered by conversation_validation_test / conversation_snapshot_test; the
// scheduler that CHOOSES a swap is S3.1e.  Between them, the ordering rules of
// docs/STAGE3-CONCURRENCY.md §4.2/§4.4/§5.3 and risks R7/R8/R9 have no test at all - this file is
// that test.  `run()` performs the order through std::function hooks, so a fake session that records
// what it was asked to do, in what order, is enough to pin it.
#include "strata/program/serve_swap.hpp"
// S3.6: the swap's guard and the driver's step gate are ONE rule (§5.3) seen from two sides, so the
// swap test pins the pair together.  serve_driver.hpp includes serve_swap.hpp; including it here
// costs nothing and stops the two predicates drifting apart unnoticed.
#include "strata/program/serve_driver.hpp"

#include "strata/core/conversation_cache.hpp"

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

using namespace strata::program::serve_swap;
using Step = strata::program::serve_swap::Step;

/// The steps a plan is expected to run, in the canonical order.
std::vector<Step> expected(const Plan& p) {
    std::vector<Step> v;
    v.push_back(Step::drain_residency);
    v.push_back(Step::return_loan);
    if (p.park_guard) v.push_back(Step::park_guard);
    if (p.mount) v.push_back(Step::validate);
    if (p.save) v.push_back(Step::unmount);
    if (p.mount) v.push_back(Step::mount);
    if (p.draft_kv) v.push_back(Step::draft_kv);
    if (p.adopt) v.push_back(Step::adopt);
    if (p.device_state) v.push_back(Step::device_state);
    return v;
}

bool same(const std::vector<Step>& a, const std::vector<Step>& b) { return a == b; }

// ------------------------------------------------------------------------ the order ----------------

/// Every hook records its step name into one trace; the trace IS the assertion.
struct Recorder {
    std::vector<std::string> trace;
    Hooks hooks;
    std::string err;

    explicit Recorder(bool with_all) {
        auto add = [&](Step s) {
            return [this, s](std::string&) { trace.push_back(step_name(s)); return true; };
        };
        hooks.drain_residency = add(Step::drain_residency);
        hooks.return_loan = add(Step::return_loan);
        hooks.park_guard = add(Step::park_guard);
        if (with_all) {
            hooks.validate = add(Step::validate);
            hooks.unmount = add(Step::unmount);
            hooks.mount = add(Step::mount);
            hooks.draft_kv = add(Step::draft_kv);
            hooks.adopt = add(Step::adopt);
            hooks.device_state = add(Step::device_state);
        }
    }
};

void test_full_handover_order() {
    Recorder rec(true);
    Plan p{true, true, true, true, true, true};   // save, mount, draft_kv, adopt, device, park_guard
    const Report r = run(p, rec.hooks, rec.err);
    check(r.ok, "a hand-over whose hooks all succeed reports success");
    check(same(r.ran, expected(p)), "the hand-over ran exactly the canonical order");
    // The order spelled out, so a reordering fails loudly rather than silently.
    const std::vector<std::string> want = {"drain-residency", "return-loan", "park-guard", "validate",
                                           "unmount", "mount", "draft-kv", "adopt", "device-state"};
    check(rec.trace == want, "drain residency, return the loan, validate, save, restore, draft kv, adopt, device state");
    check(r.session_intact == false, "a completed hand-over did write the session (that was the point)");
    check(!r.session_poisoned, "a completed hand-over leaves a usable session");
}

void test_the_drains_always_run() {
    // A plan with nothing to do still drains: "no expert swap is pending" and "we forgot to drain"
    // are indistinguishable from the outside, and a swap over a half-applied residency table is
    // silent (risk R9).  Same for the loan (risk R8).
    Recorder rec(false);
    Plan p{};
    const Report r = run(p, rec.hooks, rec.err);
    check(r.ok, "an empty plan succeeds");
    const std::vector<std::string> want = {"drain-residency", "return-loan"};
    check(rec.trace == want, "both drains run even when the plan does nothing else");
    check(r.session_intact, "and nothing was written to the session");
}

void test_validate_runs_before_the_save() {
    // THE rule.  Validation writes nothing, so it goes first: a bad incoming image must cost the
    // outgoing conversation nothing.  If validate moved after unmount, this trace changes and the
    // "session intact" assertion below fails.
    Recorder rec(true);
    Plan p{true, true, false, false, false, true};
    const Report r = run(p, rec.hooks, rec.err);
    check(r.ok, "a save+restore plan succeeds");
    const std::vector<std::string> want = {"drain-residency", "return-loan", "park-guard", "validate",
                                           "unmount", "mount"};
    check(rec.trace == want, "validate happens before the outgoing state is saved");
}

void test_no_handover_when_the_slot_is_already_mounted() {
    check(!swap_needed(7, 7), "the mounted slot needs no hand-over");
    check(swap_needed(7, 9), "a different slot does");
    check(swap_needed(kNoId, 9), "nothing mounted and a slot wanted: hand over");
    check(!swap_needed(kNoId, kNoId), "nothing mounted, nothing wanted");
}

// ------------------------------------------------------------------------ the faults -----------------

struct Failing {
    Step at;
    std::vector<std::string> trace;
    Hooks hooks;
    std::string err{"the hook said no"};

    explicit Failing(Step where) : at(where) {
        auto add = [&](Step s, bool ok) {
            return [this, s, ok](std::string& e) {
                trace.push_back(step_name(s));
                if (ok) return true;
                e = "the " + std::string(step_name(s)) + " step failed";
                return false;
            };
        };
        auto step_ok = [&](Step s) { return s != at; };
        hooks.drain_residency = add(Step::drain_residency, step_ok(Step::drain_residency));
        hooks.return_loan = add(Step::return_loan, step_ok(Step::return_loan));
        hooks.park_guard = add(Step::park_guard, step_ok(Step::park_guard));
        hooks.validate = add(Step::validate, step_ok(Step::validate));
        hooks.unmount = add(Step::unmount, step_ok(Step::unmount));
        hooks.mount = add(Step::mount, step_ok(Step::mount));
        hooks.draft_kv = add(Step::draft_kv, step_ok(Step::draft_kv));
        hooks.adopt = add(Step::adopt, step_ok(Step::adopt));
        hooks.device_state = add(Step::device_state, step_ok(Step::device_state));
    }
};

void test_failed_validation_leaves_the_outgoing_state_intact() {
    // The contract in one test: a rejected image must not have cost anything.
    Failing f(Step::validate);
    Plan p{true, true, true, true, true};
    std::string err;
    const Report r = run(p, f.hooks, err);
    check(!r.ok, "a failed validation stops the hand-over");
    check(r.fault == Fault::validation, "and names validation as the fault");
    check(r.failed_at == Step::validate, "and where it stopped");
    check(r.session_intact, "the live session was never written: the outgoing branch is intact");
    check(!r.session_poisoned, "and it is still usable");
    check(f.trace == (std::vector<std::string>{"drain-residency", "return-loan", "validate"}),
          "nothing after validate ran - in particular no save and no restore");
    check(fault_is_recoverable(r.fault), "a validation failure is recoverable (discard the image)");
    check(!fault_poisons_session(r.fault), "and it does not poison the session");
}

void test_failed_save_aborts_before_any_restore() {
    Failing f(Step::unmount);
    Plan p{true, true, true, true, true, true};
    std::string err;
    const Report r = run(p, f.hooks, err);
    check(!r.ok, "a failed save stops the hand-over");
    check(r.fault == Fault::save, "and names save as the fault");
    check(r.session_intact, "the save only reads the session, so it is still intact");
    check(!r.session_poisoned, "and usable");
    check(f.trace == (std::vector<std::string>{"drain-residency", "return-loan", "park-guard",
                                               "validate", "unmount"}),
          "no restore was attempted after a failed save");
}

void test_failed_restore_poisons_the_session() {
    // 0.1.30's rule, kept: a restore that fails after its first write is fatal to the session, never
    // "permission to decode from a partially restored session".
    Failing f(Step::mount);
    Plan p{true, true, true, true, true, true};
    std::string err;
    const Report r = run(p, f.hooks, err);
    check(!r.ok, "a failed restore stops the hand-over");
    check(r.fault == Fault::restore, "and names restore as the fault");
    check(!r.session_intact, "the session was already being written");
    check(r.session_poisoned, "and may not be used for inference afterwards");
    check(fault_poisons_session(r.fault), "restore is the fatal fault");
    check(!fault_is_recoverable(r.fault), "and it is not recoverable");
    check(f.trace == (std::vector<std::string>{"drain-residency", "return-loan", "park-guard",
                                               "validate", "unmount", "mount"}),
          "nothing after the restore ran");
}

void test_failed_drain_stops_everything() {
    for (Step s : {Step::drain_residency, Step::return_loan}) {
        Failing f(s);
        Plan p{true, true, true, true, true};
        std::string err;
        const Report r = run(p, f.hooks, err);
        check(!r.ok, "a failed drain stops the hand-over");
        check(r.fault == Fault::residency, "and is classified as a residency fault");
        check(r.session_intact, "before anything was written");
        check(r.ran.size() == (size_t) ((int) s + 1), "and nothing after it ran");
    }
}

void test_a_later_step_means_the_session_was_written() {
    for (Step s : {Step::draft_kv, Step::adopt, Step::device_state}) {
        Failing f(s);
        Plan p{true, true, true, true, true};
        std::string err;
        const Report r = run(p, f.hooks, err);
        check(!r.ok, "a failure after the restore stops the hand-over");
        check(!r.session_intact, "the session had already been restored into");
    }
}

void test_order_predicate() {
    check(runs_before(Step::drain_residency, Step::mount), "the residency drain runs before the restore");
    check(runs_before(Step::return_loan, Step::unmount), "the loan goes back before the save");
    check(runs_before(Step::validate, Step::unmount), "validation runs before the save");
    check(runs_before(Step::unmount, Step::mount), "the save runs before the restore");
    check(runs_before(Step::mount, Step::draft_kv), "the draft ring is fixed after the restore");
    check(runs_before(Step::mount, Step::adopt), "the per-slot mirror moves after the restore");
    check(runs_before(Step::adopt, Step::device_state), "device state is re-established last");
    check(!runs_before(Step::mount, Step::validate), "never restore before validating");
    check(!runs_before(Step::device_state, Step::mount), "never upload positions before the session matches");
    int n = 0;
    for (Step s : kOrder) (void) s, ++n;
    check(n == (int) Step::count, "kOrder lists every step exactly once");
}

// ------------------------------------------------------------------ save / mount decisions ---------

void test_save_decision() {
    SlotConv out;
    out.id = 7;
    check(save_for(out, true) == Save::none, "a slot with nothing live parks nothing");
    out.live = {1, 2, 3};
    check(save_for(out, true) == Save::none, "a live branch that does not describe the session parks nothing");
    out.live_ok = true;
    check(save_for(out, true) == Save::park, "a live branch parks");
    check(save_for(out, false) == Save::none, "with parking off, nothing parks");
    out.live.clear();
    check(save_for(out, true) == Save::none, "an empty branch parks nothing (0.1.30's guard)");
}

void test_unparked_slot_cannot_be_mounted_again() {
    // THE silent-garbage trap.  If the outgoing conversation was not saved, the session's positional
    // cells are about to hold somebody else's tokens.  Its running state may still be in hand, but
    // mounting it would restore GDN/PLE/indexer state over stale K/V.
    SlotConv out;
    out.id = 7;
    out.live = {1, 2, 3};
    out.live_ok = true;
    check(can_mount(out), "a mounted slot is mountable");
    invalidate_unparked(out);
    check(!can_mount(out), "a slot whose conversation was not parked may not be mounted again");
    check(!out.live_ok, "and it is no longer the session's state");
    check(out.parked_index == -1 && out.parked_bytes == 0, "and it reports no parked image");
    check(out.live.size() == 3, "its ids survive as bookkeeping - the request still has to be answered");
}

void test_budget_matches_the_cache() {
    // The swap asks the cache's own question before it takes an image out of the cache: the outgoing
    // save has to fit while the incoming image is still held in RAM.
    strata::core::ConversationCache c(100000, 4);
    check(budget_of(c).budget == 100000 && budget_of(c).slots == 4, "the budget is read from the cache");
    check(fits(budget_of(c), 500), "500 fits an empty cache");
    check(!fits(budget_of(c), 150000), "150000 does not");
    strata::core::SavedConversation img;
    img.live.ids.assign(200, 7);
    const size_t held = img.bytes();
    check(c.put(std::move(img)), "the cache takes the image");
    const Budget b = budget_of(c);
    check(b.entries == 1, "and the mirror sees one entry");
    check(b.bytes == held, "and counts its bytes");
    check(fits(b, 100, held), "a small save fits while the incoming image is held");
    check(!fits(b, b.budget - b.bytes, held),
          "but not once the held image is counted against the same bytes - the same rule as can_fit");
    check(c.can_fit(b.budget - b.bytes, held) == fits(b, b.budget - b.bytes, held),
          "the free function and ConversationCache::can_fit agree");
    check(c.can_fit(100, held) == fits(b, 100, held), "and agree on the fitting case too");
    strata::core::ConversationCache off(0, 0);
    check(!fits(budget_of(off), 1), "a disabled cache fits nothing");
}

// ------------------------------------------------------------------------ mrope (R7) ---------------

void test_mrope_exclusivity() {
    SlotConv mounted;
    mounted.id = 7;
    check(mrope_exclusive_ok(mounted, 9, false), "an identity table belongs to nobody: any hand-over is fine");
    mounted.mrope_image = true;
    check(!mrope_exclusive_ok(mounted, 9, false),
          "a slot holding a non-identity table blocks a hand-over that would leave it mounted");
    check(mrope_exclusive_ok(mounted, 9, true),
          "a hand-over that unmounts its holder may take the table");
    check(mrope_exclusive_ok(mounted, 7, false), "the same slot keeps its own table");
    check(mrope_upload_needed(7, 7) == false, "the mounted owner needs no re-upload");
    check(mrope_upload_needed(7, 9), "a different owner must re-upload");
    check(mrope_upload_needed(-1, 9), "nobody owns it: upload");
}

void test_slot_conv_adopts_the_image() {
    // The four fields the 0.1.30 mount path assigns (generate.cpp:4820-4823), in one place.
    strata::core::SavedConversation img;
    img.live.ids = {5, 6, 7};
    img.live.imgs = {{1, 11}, {2, 22}};
    img.cvec = false;
    img.checkpoints.push_back(strata::core::ConversationCheckpoint{});
    img.checkpoints.back().ids = {5};
    SlotConv s;
    s.id = 3;
    s.live = {1, 2, 3, 4};
    s.cvec_cached = true;
    s.adopt_image(img);
    check(s.live == (std::vector<int32_t>{5, 6, 7}), "the live branch is the image's");
    check(s.live_imgs.size() == 2, "with the image's pictures");
    check(s.checks.size() == 1, "and its checkpoint chain");
    check(!s.cvec_cached, "and the control-vector state it was read with");
    check(s.live_ok, "a non-empty adopted branch describes the session");
    SlotConv e;
    strata::core::SavedConversation empty;
    e.adopt_image(empty);
    check(!e.live_ok, "an empty adopted branch does not");
}

void test_per_slot_state_is_not_shared() {
    // Two slots must not alias each other's conversation.  std::vector copies, so this is about the
    // record's shape, not about cleverness - but it is the shape S3.1e depends on.
    SlotConv a, b;
    a.id = 1; b.id = 2;
    a.live = {1, 2, 3};
    b.live = {9};
    a.checks.push_back(strata::core::ConversationCheckpoint{});
    a.checks.back().ids = {1};
    a.sfx_hist = {1, 2, 3};
    a.mrope_image = true;
    a.smpl.temperature = 0.7f;
    a.check_clock = 5;
    b.check_clock = 2;
    check(a.live != b.live, "each slot has its own live branch");
    check(a.checks.size() == 1 && b.checks.empty(), "each has its own checkpoint chain");
    check(a.sfx_hist.size() == 3 && b.sfx_hist.empty(), "the suffix drafter's history travels with the slot");
    check(a.mrope_image && !b.mrope_image, "so does whose positions are in the table");
    check(a.smpl.temperature != b.smpl.temperature, "and the sampling params");
    check(a.check_clock != b.check_clock, "and the checkpoint LRU clock");
}


// =========================================================== S3.6: the parking-collapse guard ==
//
// The bug these pin, in one sentence: a hand-over used to ask "can I park the outgoing slot?", get
// "no", destroy the conversation anyway, report SUCCESS, and let the outgoing slot's next step run
// against a session that described somebody else.  The three checks the contract demands:
//   * outgoing slot cannot be parked AND still has work  -> hand-over refused, session intact,
//     incoming slot ERR'd (not starved, not spun on);
//   * a slot in decode whose SlotConv is not resumable   -> no window runs (the driver's gate);
//   * a finished slot may be swapped out with no save.

void test_outgoing_cannot_be_parked_and_has_work_refuses_the_handover() {
    // THE refusal.  `park_guard` runs after the two drains and before `validate`/`unmount`, so a
    // refusal leaves the session exactly as it was: nothing was written, nothing was destroyed.
    struct Guard {
        std::vector<std::string> trace;
        Hooks hooks;
        ParkCheck chk;
        bool ran_guard = false;
        Guard() {
            auto add = [&](Step s) {
                return [this, s](std::string&) { trace.push_back(step_name(s)); return true; };
            };
            hooks.drain_residency = add(Step::drain_residency);
            hooks.return_loan = add(Step::return_loan);
            hooks.validate = add(Step::validate);
            hooks.unmount = add(Step::unmount);
            hooks.mount = add(Step::mount);
            hooks.draft_kv = add(Step::draft_kv);
            hooks.adopt = add(Step::adopt);
            hooks.device_state = add(Step::device_state);
            // The engine's hook: a 3.3 GiB snapshot against a 1.6 GiB budget, mid-decode.
            hooks.park_guard = [&](std::string& e) {
                ran_guard = true;
                trace.push_back(step_name(Step::park_guard));
                Budget b;
                b.bytes = 0; b.budget = (size_t) 1600 << 20; b.entries = 0; b.slots = 8;
                chk = park_fits(b, (size_t) 3346 << 20, 0, Outgoing::must_park, true);
                if (chk.ok()) return true;
                e = park_refusal_line(5, 6, chk);
                return false;
            };
        }
    };
    Guard g;
    Plan p{true, true, true, true, true, true};
    std::string err;
    const Report r = run(p, g.hooks, err);
    check(g.ran_guard, "the guard ran at all (it is in the plan, not an afterthought)");
    check(!r.ok, "a mid-decode slot that cannot be saved stops the hand-over");
    check(r.fault == Fault::park, "and names park as the fault");
    check(r.failed_at == Step::park_guard, "and where it stopped");
    check(r.session_intact, "the session was never written - the outgoing conversation survives");
    check(!r.session_poisoned, "and it is still usable");
    check(g.trace == (std::vector<std::string>{"drain-residency", "return-loan", "park-guard"}),
          "nothing after the guard ran: no validate, no save, no restore, no adopt");
    check(fault_is_recoverable(r.fault), "a park refusal is recoverable: the caller ERRs the incoming slot");
    check(!fault_poisons_session(r.fault), "and it never poisons the session");
    // The two numbers the owner asked for, on the one line.
    check(err.find("3346 MiB") != std::string::npos, "the refusal names the snapshot size");
    check(err.find("1600 MiB") != std::string::npos, "and the budget in force");
    check(err.find("NEVER be parked") != std::string::npos, "and says it is permanent, not transient");
    check(err.find("keeps the session") != std::string::npos, "and says who keeps the session");
}

void test_a_finished_slot_may_be_swapped_out_with_no_save() {
    // The other half of the rule, and the half that keeps 0.1.30 working: a request that is OVER
    // needs no save.  Its client has the answer and a later request finds the branch through
    // ConversationCache::best(), so dropping the in-session copy costs reuse, not a conversation -
    // even when the budget is zero.
    Budget b;
    b.bytes = 0; b.budget = 0; b.entries = 0; b.slots = 0;   // parking OFF
    const ParkCheck c = park_fits(b, (size_t) 3346 << 20, 0, Outgoing::finished, false);
    check(!c.mandatory, "a finished slot owes nothing to save");
    check(c.ok(), "so the hand-over proceeds whatever the budget");
    check(c.refusal == ParkRefusal::none, "and there is no refusal to report");
    check(save_is_mandatory(Outgoing::finished) == false, "finished is not mandatory");
    check(save_is_mandatory(Outgoing::must_park), "mid-decode is");
    check(save_is_mandatory(Outgoing::re_readable) == false, "mid-prompt is not, but it owes a re-read")
        ;
    check(save_is_optional(Outgoing::re_readable), "and the caller must rebuild it");
}

void test_the_budget_question_matches_the_cache() {
    // `budget_has_room` mirrors `ConversationCache::make_room`, NOT `can_fit`: make_room PRUNES the
    // least recently used conversation to make space, so a full cache is not a refusal.  A guard
    // built on `can_fit` would turn a normal eviction into a stalled conversation.
    strata::core::ConversationCache c(100000, 2);
    for (int i = 0; i < 2; ++i) {
        strata::core::SavedConversation img;
        img.live.ids.assign(400, (int32_t) (7 + i));   // distinct prefixes, so both survive
        check(c.put(std::move(img)), "the cache takes the image");
    }
    const Budget b = budget_of(c);
    check(b.entries == b.slots, "the cache is now full by entry count");
    check(!fits(b, 100), "can_fit says no - it would have to evict");
    check(budget_has_room(b, 100), "but make_room would evict the LRU and take it: not a refusal");
    check(!budget_has_room(b, b.budget + 1), "only something larger than the whole budget never fits");
    check(!budget_has_room(Budget{0, 0, 0, 0}, 1), "a budget of 0 parks nothing");
    // held: the incoming image is out of the cache but still resident during the exchange.
    check(budget_has_room(b, b.budget - 10, 10), "budget - held is the real ceiling");
    check(!budget_has_room(b, b.budget, 10), "and held counts against it");
    // the refusal kind, and that a mid-decode slot with a fitting snapshot is NOT refused.
    check(park_fits(b, 1000, 0, Outgoing::must_park, true).ok(),
          "a parkable mid-decode slot is never refused");
    const ParkCheck later = park_fits(b, 1000, 0, Outgoing::must_park, false);
    check(!later.ok() && later.refusal == ParkRefusal::not_saveable,
          "a session with no whole branch to save is the WAIT kind, not the NEVER kind");
    const ParkCheck never = park_fits(b, b.budget + 1, 0, Outgoing::must_park, true);
    check(!never.ok() && never.refusal == ParkRefusal::budget_too_small,
          "a snapshot bigger than the whole budget is the NEVER kind");
    check(park_fits(b, b.budget + 1, 0, Outgoing::re_readable, true).ok(),
          "the same oversized snapshot is allowed for a slot that can re-read its prompt");
}

void test_the_step_gate_catches_a_slot_the_session_does_not_describe() {
    // The contract's second requirement: "a slot in Phase::decode whose SlotConv is not resumable
    // must be impossible - add an assertion/guard that catches it and ends the slot with an explicit
    // ERR rather than running the window."
    SlotConv dead;
    dead.id = 5;
    dead.live = {1, 2, 3};
    dead.live_ok = true;
    check(can_mount(dead), "a mounted slot is resumable");
    invalidate_unparked(dead);
    check(!dead.resumable, "invalidate_unparked is the ONLY thing that clears it");
    // The gate's answers, over every phase and both facts.  `resumable` and `session_valid` ask
    // different questions: "does a parked image exist to come back to" and "does the live session
    // still hold this request's sequence".
    using Gate = strata::program::serve_driver::Gate;
    using Phase = strata::program::serve_driver::Phase;
    using strata::program::serve_driver::step_gate;
    check(step_gate(Phase::decode, false, true) == Gate::end,
          "THE CONTRACT CASE: a decoder with no parked image may not run a window, even while mounted");
    check(step_gate(Phase::decode, false, false) == Gate::end,
          "and it is checked before the session question, so it cannot be masked by a fresh mount");
    check(step_gate(Phase::decode, true, false) == Gate::end,
          "a decoder whose session went to another slot ends too - its tokens are already on the wire");
    check(step_gate(Phase::prefill, false, false) == Gate::re_read,
          "a mid-prompt slot with no session honours the promise: re-read from token 0");
    check(step_gate(Phase::prefill_end, false, false) == Gate::re_read,
          "and the same for the prefill tail");
    check(step_gate(Phase::prefill, false, true) == Gate::run,
          "a mid-prompt slot that still owns its session keeps reading: resumable gates a FUTURE "
          "hand-over, not the step in front of it");
    check(step_gate(Phase::queued, false, false) == Gate::run,
          "a queued slot runs no step of this kind, so the gate stays out of its way");
    check(step_gate(Phase::done, false, false) == Gate::run,
          "and so does a finished one");
    // The ERR must name the fix, not just the failure.
    const std::string why = strata::program::serve_driver::gate_end_reason(Phase::decode);
    check(why.find("decode") != std::string::npos, "the ERR says which phase it happened in");
    check(why.find("--conversation-cache-mib") != std::string::npos, "and what to change");
    check(why.find("no step may run against it") != std::string::npos, "and why");
}

void test_outgoing_for_maps_the_phase_to_the_cost() {
    using Phase = strata::program::serve_driver::Phase;
    using strata::program::serve_driver::outgoing_for;
    check(outgoing_for(Phase::decode) == Outgoing::must_park, "decode: the state must be saved");
    check(outgoing_for(Phase::prefill) == Outgoing::re_readable, "prefill: the prompt is still in hand");
    check(outgoing_for(Phase::prefill_end) == Outgoing::re_readable, "prefill-end: same");
    check(outgoing_for(Phase::queued) == Outgoing::finished, "queued: nothing to lose yet");
    check(outgoing_for(Phase::done) == Outgoing::finished, "done: the client has its answer");
}


}  // namespace

void test_dispatch_state_travels_with_the_slot() {
    // §3.5: sampling is applied at DISPATCH.  A hand-over that is not driven by a request line has
    // nothing else to re-dispatch from, so the record must be able to say "I have params" and carry
    // the penalty window and the PCIe share beside them.
    SlotConv a, b;
    a.id = 1; b.id = 2;
    check(!a.smpl_set, "a fresh record has no sampling of its own yet");
    a.smpl.temperature = 0.8f;
    a.smpl.top_k = 40;
    a.smpl.greedy = false;
    a.smpl_set = true;
    a.penalty_last_n = 512;
    a.pcie_num = 64;
    check(a.smpl_set && b.smpl_set == false, "only the slot that was asked carries it");
    check(a.smpl.top_k != b.smpl.top_k, "the sampler params do not alias");
    check(a.penalty_last_n == 512 && b.penalty_last_n == 0, "nor does the penalty window");
    check(a.pcie_num == 64 && b.pcie_num == 0, "nor the PCIe share");
}

void test_a_swap_without_a_restore_clears_the_branch() {
    // The rule the engine's adopt hook implements: no restore means the session's POSITIONAL cells
    // still hold the outgoing conversation's tokens, so nothing the incoming slot inherits from them
    // may be used - a checkpoint is only valid while the cells below it hold its tokens.
    SlotConv in;
    in.id = 9;
    in.checks.push_back(strata::core::ConversationCheckpoint{});
    in.checks.back().ids = {1, 2, 3};
    in.live = {1, 2, 3};
    in.live_ok = true;
    // what the engine does when plan.mount is false:
    in.live.clear();
    in.checks.clear();
    in.live_ok = false;
    check(in.live.empty() && in.checks.empty() && !in.live_ok,
          "a hand-over with no restore leaves the incoming slot nothing to resume from");
    check(in.resumable, "but the slot is still mountable - it describes the session, which is now empty")
        ;
}

void test_the_handover_mount_lookup_is_exact_not_a_prefix() {
    // S3.7 THE BUG THAT ENDED LIVE REQUESTS.  A driver handing the session back to a slot it
    // pre-empted asks the cache for that slot's own branch.  The prefix rule requires the token list
    // to be strictly SHORTER than the entry it matches, and a pre-empted slot's branch is exactly
    // EQUAL to what it parked - so the old lookup answered "nothing", the hand-over mounted nothing,
    // the adopt hook cleared the branch, and `step_gate` ended a running decoder with an ERR that
    // blamed the parking budget.  The owner's log: 80 tokens parked at 113 MiB against a 10 048 MiB
    // budget, and the request still died.
    strata::core::ConversationCache cache(1u << 20, 8);
    strata::core::SavedConversation img;
    img.cvec = true;
    img.live.ids = {7, 8, 9, 10};
    check(cache.put(std::move(img)), "park the branch a decoder leaves behind");

    SlotConv s;
    s.id = 3;
    s.live = {7, 8, 9, 10};
    s.live_ok = true;
    s.resumable = true;
    check(handover_seeks(s), "a resumable slot with a published branch has something to look for");
    check(seek_mount_index(cache, s) == 0, "and the hand-over finds exactly that entry");

    // The prefix rule, for contrast: the same cache, asked the way a NEW request asks.
    check(cache.best(std::vector<int32_t>{7, 8, 9, 10}, {}, true).tokens == 0,
        "the prefix rule cannot find a slot's own branch - that was the bug");
    check(cache.best(std::vector<int32_t>{7, 8, 9, 10, 11}, {}, true).tokens == 4,
        "it does find it as a prefix of a longer prompt, which is what it is for");

    // Nothing to mount, in each of the three ways.
    SlotConv gone = s;
    gone.resumable = false;
    check(seek_mount_index(cache, gone) == -1, "a slot whose image was invalidated seeks nothing");
    SlotConv empty = s;
    empty.live.clear();
    check(seek_mount_index(cache, empty) == -1, "a slot with no published branch seeks nothing");
    SlotConv other = s;
    other.live = {1, 2, 3};
    check(seek_mount_index(cache, other) == -1, "a different sequence is not this slot's branch");
    SlotConv longer = s;
    longer.live = {7, 8, 9, 10, 11};
    check(seek_mount_index(cache, longer) == -1,
          "a longer sequence is not it either - never restore a slot past tokens it never generated");

    // A checkpoint of the same entry is NOT mountable by a hand-over: a restore puts the whole entry
    // back, so matching the shorter checkpoint would resume the slot further along than it stopped.
    strata::core::ConversationCache chain_cache(1u << 20, 8);
    strata::core::SavedConversation with_cp;
    with_cp.cvec = true;
    with_cp.live.ids = {7, 8, 9, 10};
    strata::core::ConversationCheckpoint cp;
    cp.ids = {7, 8};
    with_cp.checkpoints.push_back(cp);
    check(chain_cache.put(std::move(with_cp)), "park a branch that also holds a shorter checkpoint");
    SlotConv at_cp = s;
    at_cp.live = {7, 8};
    check(seek_mount_index(chain_cache, at_cp) == -1,
          "the checkpoint is not a hand-over mount point");
    check(seek_mount_index(chain_cache, s) == 0, "the live branch is");

    // S3.9 THE CLAIM, from the hand-over's side.  The owner's regression: "when a new request arrives
    // it always kicks the currently running request out with an 'it was decode but not parked' error."
    // The park SUCCEEDED; a different request's prefix mount then removed the entry, because a parked
    // entry is one object and `take()` takes all of it.
    strata::core::ConversationCache claimed(1u << 20, 8);
    strata::core::SavedConversation mine;
    mine.cvec = true;
    mine.live.ids = {7, 8, 9, 10};
    check(claimed.put(std::move(mine), /*owner=*/3), "park it claimed for slot 3");
    check(seek_mount_index(claimed, s) == 0,
          "slot 3 still finds its own branch - the claim never hides an entry from its owner");
    SlotConv thief = s;
    thief.id = 9;
    check(seek_mount_index(claimed, thief) == -1,
          "slot 9 may not mount what slot 3 parked for itself");
    check(claimed.size() == 1, "and slot 3's conversation is still there to be handed back");

    // A disabled cache answers -1 rather than crashing or mounting garbage.
    strata::core::ConversationCache off(0, 0);
    check(seek_mount_index(off, s) == -1, "parking off -> nothing to mount");
}

int main() {
    test_full_handover_order();
    test_the_drains_always_run();
    test_validate_runs_before_the_save();
    test_no_handover_when_the_slot_is_already_mounted();
    test_failed_validation_leaves_the_outgoing_state_intact();
    test_failed_save_aborts_before_any_restore();
    test_failed_restore_poisons_the_session();
    test_failed_drain_stops_everything();
    test_a_later_step_means_the_session_was_written();
    test_order_predicate();
    test_save_decision();
    test_unparked_slot_cannot_be_mounted_again();
    test_budget_matches_the_cache();
    test_mrope_exclusivity();
    test_slot_conv_adopts_the_image();
    test_per_slot_state_is_not_shared();
    test_dispatch_state_travels_with_the_slot();
    test_a_swap_without_a_restore_clears_the_branch();
    test_outgoing_cannot_be_parked_and_has_work_refuses_the_handover();
    test_a_finished_slot_may_be_swapped_out_with_no_save();
    test_the_budget_question_matches_the_cache();
    test_the_step_gate_catches_a_slot_the_session_does_not_describe();
    test_outgoing_for_maps_the_phase_to_the_cost();
    test_the_handover_mount_lookup_is_exact_not_a_prefix();

    if (failures) {
        std::fprintf(stderr, "serve_swap_test: %d of %d checks FAILED\n", failures, checks);
        return 1;
    }
    std::printf("serve_swap_test: %d checks OK\n", checks);
    return 0;
}
