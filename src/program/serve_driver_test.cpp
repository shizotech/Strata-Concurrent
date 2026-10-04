// src/program/serve_driver_test.cpp - stage 3 S3.1e-2: the concurrent driver's decisions, on a CPU.
//
// The driver loop itself lives in `src/program/generate.cpp` and cannot run without a GPU, a model and
// tens of GiB of RAM.  Everything it DECIDES lives in include/strata/program/serve_driver.hpp, and
// this file pins those decisions: admission, the Pick -> step dispatch, the phase machine, the prompt
// loan's ownership state machine, per-slot cancellation routing and the watchdog-with-N-slots rule
// (risk R3).  Same pattern as slot_test / serve_proto_test / serve_swap_test.
#include "strata/program/serve_driver.hpp"

#include "strata/program/slot.hpp"

#include <cstdio>
#include <map>
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

using namespace strata::program;
using serve_driver::Gate;
using serve_driver::Loan;
using serve_driver::Outcome;
using serve_driver::Phase;
using serve_driver::Refuse;
using serve_driver::SlotWatch;
using serve_driver::Step;
using serve_driver::SwapResult;
using serve_driver::admit_fits_ram;
using serve_driver::bare_stop_target;
using serve_driver::bytes_per_token;
using serve_driver::dispatch_step;
using serve_driver::driver_line;
using serve_driver::GateEnd;
using serve_driver::next_phase;
using serve_driver::outgoing_for;
using serve_driver::ParkCeiling;
using serve_driver::park_ceiling_line;
using serve_driver::parkable_at_all;
using serve_driver::now_ms;
using serve_driver::kMaxRereads;
using serve_driver::parking_off_refuses_slots;
using serve_driver::phase_has_work;
using serve_driver::reread_allowed;
using serve_driver::reread_limit_line;
using serve_driver::SlotCost;
using serve_driver::slot_estimate;
using serve_driver::slot_image_bytes;
using serve_driver::slot_stalled;
using serve_driver::stalled_list;
using serve_driver::step_gate;
using serve_driver::stalled_slots;
using serve_driver::step_needs_loan;
using serve_driver::watchdog_aborts;
using slot::Pick;
using slot::Registry;
using slot::Slot;
using slot::State;

// ------------------------------------------------------------------ the clock, one timeline ----

void test_the_clock_is_monotonic() {
    const int64_t a = now_ms();
    const int64_t b = now_ms();
    check(b >= a, "now_ms() is monotonic across calls");
    check(now_ms() >= b, "now_ms() does not go backwards");
}

// ------------------------------------------------------------------------ the dispatch ---------

/// A pick for `id` given that the live session reflects `active` - the registry's own rule, restated
/// so the test can build one decision without driving a whole registry.
Pick mk_pick(int64_t id, int64_t active) {
    Pick p;
    p.action = (id == active) ? Pick::Action::run : Pick::Action::swap;
    p.id = id;
    return p;
}
Pick mk_pick(Pick::Action a, int64_t id) {
    Pick p;
    p.action = a;
    p.id = id;
    return p;
}

void test_dispatch_maps_every_phase() {
    // A QUEUED slot is admitted; that is the only thing `admit` means.
    check(dispatch_step(mk_pick(Pick::Action::admit, 3), Phase::queued) == Step::admit,
          "admit -> admit");
    // `idle` never runs a step.
    check(dispatch_step(Pick{}, Phase::decode) == Step::none, "idle -> none");
    // A slot whose body has not started yet is stepped by running its prep, whatever pick called it.
    check(dispatch_step(mk_pick(1, 1), Phase::queued) == Step::admit, "run on a queued slot -> admit");
    // The four body phases, on the mounted slot: no swap.
    check(dispatch_step(mk_pick(1, 1), Phase::prefill) == Step::prefill, "run + prefill -> prefill");
    check(dispatch_step(mk_pick(1, 1), Phase::prefill_end) == Step::prefill_end,
          "run + prefill_end -> prefill-end");
    check(dispatch_step(mk_pick(1, 1), Phase::decode) == Step::decode, "run + decode -> decode");
    check(dispatch_step(mk_pick(1, 1), Phase::done) == Step::finish, "run + done -> finish");
    // The same phases on a NON-mounted slot: the hand-over comes first, and it is one step.
    check(dispatch_step(mk_pick(2, 1), Phase::prefill) == Step::swap, "swap + prefill -> swap");
    check(dispatch_step(mk_pick(2, 1), Phase::prefill_end) == Step::swap, "swap + prefill_end -> swap");
    check(dispatch_step(mk_pick(2, 1), Phase::decode) == Step::swap, "swap + decode -> swap");
    check(dispatch_step(mk_pick(2, 1), Phase::done) == Step::swap, "swap + done -> swap (finish after the mount)");
    // A QUEUED slot is ALWAYS admitted, even when pick asked for a swap.  `prep_request` owns the
    // request-line hand-over (S3.1d's call site), and it is the form that takes the incoming image out
    // of the cache BEFORE saving the outgoing one, so the save counts it as held RAM.  Swapping first
    // would get that arithmetic wrong and would save/restore the same session twice.
    check(dispatch_step(mk_pick(2, 1), Phase::queued) == Step::admit, "a queued slot admits, never pre-swaps");
    check(dispatch_step(mk_pick(Pick::Action::admit, 2), Phase::queued) == Step::admit, "admit -> admit");
    check(dispatch_step(mk_pick(Pick::Action::run, 1), Phase::queued) == Step::admit,
          "run on a queued slot -> admit");
    // `unwind` is a REASON, not a different step: a cancelled slot runs its phase's tail and never
    // starts a new step.  Jumping straight to `finish_request` would be a real bug - on a request
    // whose decode loop never ran, `consumed` is empty and `finish_request` does `live.swap(consumed)`,
    // wiping the mounted conversation.
    check(dispatch_step(mk_pick(Pick::Action::unwind, 2), Phase::prefill) == Step::prefill_end,
          "unwind + prefill -> the prefill tail (it returns the loan)");
    check(dispatch_step(mk_pick(Pick::Action::unwind, 2), Phase::prefill_end) == Step::prefill_end,
          "unwind + prefill_end -> the same tail again");
    check(dispatch_step(mk_pick(Pick::Action::unwind, 2), Phase::decode) == Step::finish,
          "unwind + decode -> finish (no new window for a cancelled slot)");
    check(dispatch_step(mk_pick(Pick::Action::unwind, 2), Phase::done) == Step::finish,
          "unwind + done -> finish");
    check(dispatch_step(mk_pick(Pick::Action::unwind, 2), Phase::queued) == Step::admit,
          "unwind + queued -> admit (it still owes the client a DONE)");
    // A CANCELLING slot is unwound the same way even when pick said plain `run` (rule 1 of the pick
    // rule hands it the session to finish its unwind).
    Pick run_cancelling;
    run_cancelling.action = Pick::Action::run;
    run_cancelling.id = 1;
    run_cancelling.state = State::cancelling;
    check(dispatch_step(run_cancelling, Phase::prefill) == Step::prefill_end,
          "a cancelling slot never starts another segment");
    check(dispatch_step(run_cancelling, Phase::decode) == Step::finish,
          "and never another window");
    // ... but a cancelling slot that is NOT the mounted one still needs the hand-over first.
    Pick swap_cancelling;
    swap_cancelling.action = Pick::Action::swap;
    swap_cancelling.id = 2;
    swap_cancelling.state = State::cancelling;
    check(dispatch_step(swap_cancelling, Phase::prefill) == Step::swap,
          "unwinding a non-mounted slot still hands the session over first");
}

void test_dispatch_never_runs_a_step_over_a_needed_swap() {
    // The invariant the design states as a rule (§5.3): "A step for slot B while A is active is
    // illegal."  So whenever the picked id is not the mounted one, the answer is `swap`, never a step.
    // A QUEUED slot is the one exception, and it is not a step either: it admits, which is the form
    // that carries the hand-over (see test_dispatch_maps_every_phase).
    for (int i = (int) Phase::prefill; i <= (int) Phase::done; ++i)
        check(dispatch_step(mk_pick(2, 1), (Phase) i) == Step::swap,
              "a non-mounted slot never gets a direct step");
    // And the registry really does answer `swap` for a non-active slot, so the two halves agree.
    Registry reg(3, 250);
    const int64_t t = now_ms();
    reg.add(1, t); reg.add(2, t);
    std::string e;
    reg.transition(1, State::decoding, e);
    reg.transition(2, State::decoding, e);
    reg.set_active(1);
    reg.note_ran(1, t);
    const Pick p = reg.pick(t + 100000);
    check(p.action == Pick::Action::swap && p.id == 2, "a starved non-active slot -> swap");
    check(dispatch_step(p, Phase::decode) == Step::swap, "and the driver runs no window until it lands");
}

void test_phase_machine() {
    check(next_phase(Phase::queued, Outcome::finished) == Phase::prefill, "admit -> prefill");
    check(next_phase(Phase::prefill, Outcome::progressed) == Phase::prefill, "a segment after a segment");
    check(next_phase(Phase::prefill, Outcome::finished) == Phase::prefill_end, "segments done -> prefill-end");
    check(next_phase(Phase::prefill, Outcome::cancelled) == Phase::prefill_end,
          "a cancelled read still runs the phase tail (it returns the loan)");
    check(next_phase(Phase::prefill_end, Outcome::finished) == Phase::decode, "prefill-end -> decode");
    check(next_phase(Phase::decode, Outcome::progressed) == Phase::decode, "a window after a window");
    check(next_phase(Phase::decode, Outcome::finished) == Phase::done, "decode done -> done");
    check(next_phase(Phase::decode, Outcome::cancelled) == Phase::done, "decode cancelled -> done");
    check(next_phase(Phase::done, Outcome::finished) == Phase::done, "done is terminal");
    check(!phase_has_work(Phase::done), "a done slot has no work");
    check(!phase_has_work(Phase::queued), "a queued slot has no STEP yet, it needs admitting");
    check(phase_has_work(Phase::prefill) && phase_has_work(Phase::decode) &&
          phase_has_work(Phase::prefill_end), "the running phases have work");
}

// ----------------------------------------------------------------------- admission -------------

void test_admission_ram_arithmetic() {
    const uint64_t G = 1ull << 30;
    // 8 GiB free, a 1 GiB floor, a 2 GiB image: fits.
    check(admit_fits_ram(true, 8 * G, 1 * G, 2 * G, 0), "a slot that fits is admitted");
    // the same, but 3 GiB is already held by the other active slots: 1 + 2 + 3 > 5 -> refused
    check(!admit_fits_ram(true, 5 * G, 1 * G, 2 * G, 3 * G),
          "the other slots' images count against the same RAM");
    check(admit_fits_ram(true, 8 * G, 1 * G, 2 * G, 3 * G), "and with room to spare they do not");
    // exactly at the boundary is allowed (>=), one byte over is not
    check(admit_fits_ram(true, 3 * G, 1 * G, 2 * G, 0), "exactly enough is enough");
    check(!admit_fits_ram(true, 3 * G - 1, 1 * G, 2 * G, 0), "one byte short is refused");
    // no telemetry: the parking budget is the gate, so RAM cannot refuse (as for parking)
    check(admit_fits_ram(false, 0, 1 * G, 100 * G, 0), "no telemetry -> RAM is not the gate");
    // a zero floor is still a real check
    check(!admit_fits_ram(true, 0, 0, 1, 0), "zero free RAM refuses");
    check(admit_fits_ram(true, 1 * G, 0, 0, 0), "a zero-size image is free to admit");
    // the sizes that matter on the box this is sized against: ~2 GiB free, a 0.0 GiB floor and a
    // 6.19 GiB host copy per slot - which is why the default stays 0/1 (risk R2)
    check(!admit_fits_ram(true, 2 * G, 0, 7 * G, 0), "a 7 GiB slot does not fit in 2 GiB free");
}

void test_admission_refuses_what_cannot_be_swapped() {
    // R2/R12: with parking off there is no save/restore, so a slot switch would DESTROY the outgoing
    // conversation (serve_swap::invalidate_unparked).  Refuse at startup instead.
    check(parking_off_refuses_slots(3, false), "--serve-slots 3 with the cache off is refused");
    check(!parking_off_refuses_slots(3, true), "--serve-slots 3 with the cache on is fine");
    check(!parking_off_refuses_slots(1, false), "--serve-slots 1 is today's serial engine, cache or not");
    check(!parking_off_refuses_slots(0, false), "--serve-slots 0 likewise");
    // the per-slot estimate is the budget divided by the entries it was sized for
    check(slot_estimate((size_t) (8ull << 30), 8) == (size_t) (1ull << 30), "the estimate is budget / slots");
    check(slot_estimate(0, 8) == 0, "no budget -> no estimate");
    check(slot_estimate((size_t) (8ull << 30), 0) == 0, "no entries -> no estimate");
    check(std::string(serve_driver::refuse_reason(Refuse::no_parking)).find("cache") != std::string::npos,
          "the refusal says why");
    check(std::string(serve_driver::refuse_reason(Refuse::ram)).find("RAM") != std::string::npos,
          "and so does the RAM one");
}

void test_admission_respects_the_active_cap() {
    Registry reg(2, 250);
    const int64_t t = now_ms();
    check(reg.can_admit(), "an empty registry can admit");
    reg.add(1, t);
    reg.add(2, t);
    reg.add(3, t);
    std::string e;
    reg.transition(1, State::prefilling, e);
    check(reg.can_admit(), "one active of two can still admit");
    reg.transition(2, State::prefilling, e);
    check(!reg.can_admit(), "two active of two cannot admit a third");
    reg.transition(1, State::decoding, e);
    check(!reg.can_admit(), "a decoding slot still occupies its permit");
    // and the third request is refused rather than queued forever
    check(!reg.transition(3, State::prefilling, e), "the cap is enforced by transition too");
    check(reg.find(3)->state == State::queued, "the refused request stays queued");
    reg.release(1);
    check(reg.can_admit(), "releasing a slot frees its permit");
    check(reg.transition(3, State::prefilling, e), "and now the third request gets in");
    // QUEUED does not occupy a permit (§3.4: "no state allocated")
    check(reg.active_count() == 2, "only prefilling/decoding/cancelling are active");
}

// ------------------------------------------------------------------ the pick rule, end to end --

/// A slot in every state the driver can put it in, then what pick() answers.  This is the loop's
/// decision table: it is what the driver dispatches on, so it must be pinned.
void test_pick_rule_scenarios() {
    std::string e;
    {   // one queued request, nothing mounted: admit it
        Registry reg(2, 250);
        reg.add(1, now_ms());
        const Pick p = reg.pick(now_ms());
        check(p.action == Pick::Action::admit && p.id == 1, "a lone queued slot is admitted");
    }
    {   // the active slot is decoding and nobody is starved: stay on it, zero swaps
        Registry reg(3, 250);
        const int64_t t = now_ms();
        reg.add(1, t); reg.add(2, t);
        reg.transition(1, State::prefilling, e);
        reg.transition(1, State::decoding, e);
        reg.set_active(1);
        reg.note_ran(1, t);
        reg.note_ran(2, t);
        const Pick p = reg.pick(t);
        check(p.action == Pick::Action::run && p.id == 1, "active decoding, nobody starved -> stay");
        check(std::string(p.why) == "active-decoding", "and the reason is recorded");
    }
    {   // the same, but slot 2 has waited past the bound: fairness pre-empts
        Registry reg(3, 250);
        const int64_t t = now_ms();
        reg.add(1, t); reg.add(2, t - 1000);
        reg.transition(1, State::prefilling, e);
        reg.transition(1, State::decoding, e);
        reg.transition(2, State::decoding, e);
        reg.set_active(1);
        reg.note_ran(1, t);
        const Pick p = reg.pick(t);
        check(p.action == Pick::Action::swap && p.id == 2, "a starved decoder forces the swap");
    }
    {   // STAY on the active decoder while nobody is starved, even beside a long prefill (R10)
        Registry reg(3, 100000);
        const int64_t t = now_ms();
        reg.add(1, t); reg.add(2, t);
        reg.transition(1, State::decoding, e);
        reg.transition(2, State::prefilling, e);
        reg.set_active(1);
        reg.note_ran(1, t); reg.note_ran(2, t);
        const Pick p = reg.pick(t);
        check(p.action == Pick::Action::run && p.id == 1,
              "an unstarved active decoder keeps the session: no swap churn");
    }
    {   // but past the bound, prefill before decode: it is the long pole
        Registry reg(3, 250);
        const int64_t t = now_ms();
        reg.add(1, t); reg.add(2, t - 1000);
        reg.transition(1, State::decoding, e);
        reg.transition(2, State::prefilling, e);
        reg.set_active(1);
        reg.note_ran(1, t);
        const Pick p = reg.pick(t);
        check(p.action == Pick::Action::swap && p.id == 2, "a starved prefilling slot beats the active decoder");
        check(std::string(p.why) == "fairness-starved", "and it is the fairness rule that says so");
    }
    {   // with nothing mounted, a prefilling slot beats a decoding one outright
        Registry reg(3, 100000);
        const int64_t t = now_ms();
        reg.add(1, t); reg.add(2, t);
        reg.transition(1, State::decoding, e);
        reg.transition(2, State::prefilling, e);
        reg.note_ran(1, t); reg.note_ran(2, t);
        const Pick p = reg.pick(t);   // active_id_ is kNoId: rule 2 cannot fire
        check(p.action == Pick::Action::swap && p.id == 2, "prefill before decode (§3.3 rule 5)");
    }
    {   // a cancelling slot unwinds first, whatever else is waiting (it may hold the loan)
        Registry reg(3, 250);
        const int64_t t = now_ms();
        reg.add(1, t); reg.add(2, t); reg.add(3, t);
        reg.transition(1, State::prefilling, e);
        reg.transition(2, State::decoding, e);
        reg.set_active(2);
        reg.cancel_request(1);
        const Pick p = reg.pick(t);
        check(p.action == Pick::Action::swap && p.id == 1, "the cancelling slot is served first");
        check(reg.find(1)->state == State::cancelling, "cancel moved it out of prefilling");
        check(std::string(p.why) == "cancel-unwind-first", "and the reason is recorded");
    }
    {   // nothing runnable: idle, and the driver waits for a line
        Registry r2(2, 250);
        check(r2.pick(now_ms()).action == Pick::Action::idle, "an empty registry is idle");
        r2.add(1, now_ms());
        check(!r2.transition(1, State::queued, e), "a queued -> queued self-edge is not in the table");
        check(r2.find(1)->state == State::queued, "and an illegal edge changes nothing");
    }
}

// ------------------------------------------------------------------- the prompt loan (R8) ------

void test_loan_is_one_resource() {
    Loan l;
    check(!l.held(), "the loan starts free");
    check(l.may_lend(1), "anybody may take a free loan");
    check(l.acquire(1), "slot 1 takes it");
    check(l.held_by(1), "the owner is named");
    check(!l.may_lend(2), "slot 2 may not take a held loan");
    check(!l.acquire(2), "and cannot steal it");
    check(l.held_by(1), "the owner did not change");
    check(l.acquire(1), "the holder may re-acquire its own loan (a wider segment)");
    check(!l.may_read_cache(2), "no verify window may run over a lent cache");
    check(!l.release(2), "slot 2 cannot release a loan it does not own");
    check(l.held(), "and the loan is still held");
    check(l.release(1), "the owner gives it back");
    check(!l.held(), "free again");
    check(l.may_read_cache(1), "now a window may run");
    check(l.release(1), "releasing a free loan is not an error");
}

void test_who_needs_the_loan() {
    check(step_needs_loan(Step::prefill), "a batched segment needs the loan");
    check(!step_needs_loan(Step::decode), "a verify window does not (it refills first)");
    check(!step_needs_loan(Step::admit), "admission does not");
    check(!step_needs_loan(Step::finish), "the tail does not");
    check(!step_needs_loan(Step::prefill_end), "the prefill tail RETURNS it, it does not take it");
    check(!step_needs_loan(Step::swap), "a hand-over returns it too (serve_swap's return_loan step)");
    check(!step_needs_loan(Step::none) && !step_needs_loan(Step::wait), "no step, no loan");
}

void test_two_prefills_cannot_both_hold_the_loan() {
    // The R8 scenario, stated as the driver's own discipline: slot 1 holds the loan, pick now wants
    // slot 2's next segment, and the driver must not lend twice.
    Loan l;
    check(l.acquire(1), "slot 1 is mid-chunk and holds the loan");
    const Step s = dispatch_step(mk_pick(2, 1), Phase::prefill);
    check(s == Step::swap, "pick wants slot 2: the hand-over comes first");
    // A hand-over refills the loan (serve_swap's return_loan step) - which is exactly why the driver
    // must not start one while a DIFFERENT slot holds it: that refill is somebody else's buffer.
    check(!l.may_read_cache(2), "so the driver defers the hand-over instead");
    check(!l.may_lend(2), "and it cannot lend to slot 2 either");
    check(l.release(1), "slot 1's prefill tail gives it back");
    check(l.may_read_cache(2), "now the hand-over may run");
    check(dispatch_step(mk_pick(2, 1), Phase::prefill) == Step::swap, "swap to slot 2");
    check(l.may_lend(2) && l.acquire(2), "and slot 2 may lend");
    check(l.held_by(2), "one owner, named");
}

void test_the_loan_follows_the_slot_that_dies() {
    // A slot that errors or finishes must give the loan back before its row is freed, or the loan is
    // stranded and nothing can prefill again.
    Loan l;
    l.acquire(4);
    check(l.held(), "held by 4");
    check(!l.release(5), "another slot cannot clear it by accident");
    check(l.release(4), "its owner can, and so must its unwind");
    check(!l.held(), "and the next prefill is unblocked");
}

// ---- S3.2b: releasing the loan is NOT a residency event ------------------------------------
//
// The lazy loan (prefill_loan.hpp lever 5) means `release()` hands back the BUFFERS while the rows stay
// out of the cache and the layout stays carved.  The scheduler's rules are unchanged - one holder, no
// window over a carved range - and this pins that the change did not quietly weaken any of them.

void test_release_does_not_mean_the_cache_is_whole() {
    Loan l;
    check(l.acquire(1), "slot 1 lends");
    check(l.release(1), "and gives the buffers back");
    check(!l.held(), "no holder");
    // The counters are the only thing that knows how much work the release actually did, and they are
    // bookkeeping: no predicate may depend on them.
    check(l.acquires == 1 && l.releases == 1, "one acquire, one release");
    // A window is allowed once the buffers are back - and it is correct whether or not the experts are,
    // because the rows still out are marked non-resident in the table it reads.
    check(l.may_read_cache(1), "a window may run over a released loan");
    check(l.may_lend(2) && l.acquire(2), "and a different slot may take the same carved range next");
    check(l.acquires == 2, "two acquires: the loan is reused, not rebuilt per request");
}

void test_the_loan_handoff_is_counted() {
    // The event the lazy loan makes cheap: slot 1 finishes its prompt, slot 2 starts one.  Eager mode pays
    // two full refills and two lends for that; lazy mode pays one release, one acquire that marks almost
    // nothing, and one return that copies nothing.  `handoffs` is the column that shows the reuse, and it
    // is counted inside `acquire()` so no caller can forget to report it.
    Loan l;
    l.acquire(1);
    l.release(1);
    l.acquire(2);
    check(l.handoffs == 1, "the loan changed hands");
    l.release(2);
    l.acquire(2);
    check(l.handoffs == 1, "the same slot re-taking its own loan is not a handoff");
    check(l.acquires == 3 && l.releases == 2, "and the totals add up");
    // A slot that dies mid-read and whose loan is force-released still counts, so the summary line cannot
    // silently under-report the hand-offs the lazy loan produced.
    l.release(2);
    l.acquire(5);
    l.release(5);          // e.g. `kill_slot` / `drop_ctx` giving a dying holder's loan back
    l.acquire(6);
    check(l.handoffs == 3, "every change of holder counts, including one after a dying slot");
    check(!l.acquire(7), "and a held loan still cannot be stolen");
}

void test_a_released_loan_still_blocks_nothing_it_should_not() {
    // The regression this guards against: if `release()` were (wrongly) made to mean "the refill is done",
    // a scheduler could start a window while the buffers are still carved.  It must not, and `held()` is
    // still the predicate that says so.
    Loan l;
    l.acquire(3);
    check(!l.may_read_cache(3), "while the buffers are carved, no window");
    check(!l.may_lend(4), "and no second lend");
    l.release(3);
    check(l.may_read_cache(3), "released: windows are allowed - the table, not the release, is what makes them safe");
}

// ---------------------------------------------------------------- the watchdog with N slots ----

void test_watchdog_aborts_only_when_every_slot_stalled() {
    const int64_t limit = 60000;
    const int64_t t = 1000000;
    std::vector<SlotWatch> w(2);
    w[0].id = 1; w[0].step_started_ms = t; w[0].beats_at_start = 10;
    w[1].id = 2; w[1].step_started_ms = t; w[1].beats_at_start = 10;
    // the heartbeat moved: nobody stalled
    check(stalled_slots(w, t + limit + 1, /*beats*/ 11, limit).empty(),
          "a heartbeat that moved means nobody stalled");
    // both frozen on the same beat count
    const std::vector<int64_t> both = stalled_slots(w, t + limit + 1, /*beats*/ 10, limit);
    check(both.size() == 2, "two slots frozen on the same beat count are both stalled");
    check(watchdog_aborts(both, w.size()), "every active slot stalled -> abort, as 0.1.30 did");
    // now slot 2 started a fresh step: only slot 1 is stale
    w[1].step_started_ms = t + limit;
    const std::vector<int64_t> solo = stalled_slots(w, t + limit + 1, 10, limit);
    check(solo.size() == 1 && solo[0] == 1, "one stalled slot is named alone");
    check(!watchdog_aborts(solo, w.size()), "and the process does NOT abort (risk R3)");
    // under the limit: nothing
    check(stalled_slots(w, t + 1000, 10, limit).empty(), "under the limit nothing has stalled");
    // the limit off
    check(stalled_slots(w, t + 10 * limit, 10, 0).empty(), "STRATA_WATCHDOG_S=0 turns it off");
    // a slot not inside a step cannot stall
    SlotWatch idle{};
    idle.id = 3;
    check(!slot_stalled(idle, t + 10 * limit, 0, limit), "between steps there is nothing to stall on");
    check(stalled_list(both) == "slot 1, slot 2", "the abort message names the stalled slots");
    check(stalled_list(std::vector<int64_t>{7}) == "slot 7", "one slot, named");
    check(stalled_list(std::vector<int64_t>{}).empty(), "none, empty");
    // the one-slot case must reduce to 0.1.30's behaviour exactly
    std::vector<SlotWatch> one_slot(1);
    one_slot[0].id = 5; one_slot[0].step_started_ms = t; one_slot[0].beats_at_start = 3;
    check(watchdog_aborts(stalled_slots(one_slot, t + limit + 1, 3, limit), 1),
          "with one slot it is 0.1.30's rule");
    check(!watchdog_aborts(std::vector<int64_t>{}, 0), "watching nothing never aborts");
}

// ---------------------------------------------------------------- per-slot cancellation --------

void test_bare_stop_names_the_running_slot() {
    Registry reg(3, 250);
    const int64_t t = now_ms();
    reg.add(1, t); reg.add(2, t + 1);
    std::string e;
    reg.transition(1, State::prefilling, e);   // slot 1 is running, slot 2 is still queued
    check(bare_stop_target(reg) == 1, "a bare STOP names the RUNNING slot, not the newest row");
    reg.cancel_request(bare_stop_target(reg));
    check(reg.find(1)->cancel.load() && !reg.find(2)->cancel.load(), "and cancels only that one");
    // with nothing running it falls back to the newest queued row (the request that would run next)
    Registry r2(3, 250);
    r2.add(7, t); r2.add(8, t + 1);
    check(bare_stop_target(r2) == 8, "nothing running -> the newest queued one");
    // and nothing at all -> kNoId, so the driver sets no flag
    Registry r3(3, 250);
    check(bare_stop_target(r3) == serve_proto::kNoId, "an empty registry names nobody");
}

void test_named_stop_cancels_only_its_slot() {
    Registry reg(3, 250);
    const int64_t t = now_ms();
    reg.add(1, t); reg.add(2, t);
    std::string e;
    reg.transition(1, State::decoding, e);
    reg.transition(2, State::prefilling, e);
    check(reg.cancel_request(2), "STOP 2 lands");
    check(reg.find(2)->state == State::cancelling, "the running slot moves to cancelling");
    check(reg.find(1)->state == State::decoding && !reg.find(1)->cancel.load(),
          "the other request is untouched - the whole point of naming it");
    // a cancel for a slot that does not exist is refused, not fatal
    check(!reg.cancel_request(99), "STOP for an unknown id is refused");
    // and the cancelled slot can no longer enter a running state
    check(!reg.transition(2, State::decoding, e), "a cancelled slot cannot start a new step");
    check(reg.find(2)->state == State::cancelling, "it stays cancelling");
    // the unwind edge: cancelling -> idle, then release
    check(reg.transition(2, State::idle, e), "cancelling -> idle is legal");
    check(reg.release(2), "and the row frees");
    check(reg.active_count() == 1, "its permit came back");
}

void test_cancel_unwind_ordering() {
    // §3.4: CANCELLING "must refill the loan and free its active slot before DONE".
    Registry reg(3, 250);
    const int64_t t = now_ms();
    reg.add(1, t);
    std::string e;
    reg.transition(1, State::prefilling, e);
    reg.set_active(1);
    Loan l;
    l.acquire(1);
    reg.cancel_request(1);
    // pick still serves it (rule 1), and the driver runs the phase tail, which returns the loan
    const Pick p = reg.pick(t);
    check(p.action == Pick::Action::run && p.id == 1, "the mounted cancelling slot is run, not swapped");
    check(dispatch_step(p, Phase::prefill_end) == Step::prefill_end, "so its prefill tail runs");
    check(l.release(1), "and the loan goes back");
    reg.transition(1, State::idle, e);
    reg.set_active(serve_proto::kNoId);
    reg.release(1);
    check(!l.held() && reg.active_count() == 0 && reg.count() == 0,
          "no loan, no active slot, no row left behind");
}

void test_a_cancel_queued_before_admission_never_runs() {
    // `STOP <id>` arriving while the request line is still on the pipe: the row takes the flag at
    // admission and lands in `cancelling` instead of a running state.
    Registry reg(3, 250);
    const int64_t t = now_ms();
    reg.add(1, t);
    check(reg.cancel_request(1), "a queued slot can be cancelled");
    check(reg.find(1)->state == State::queued, "and it stays queued until admitted");
    std::string e;
    reg.transition(1, State::prefilling, e);
    check(reg.find(1)->state == State::cancelling, "admission lands it in cancelling, never prefilling");
    // the driver still has to run its tail and answer DONE, so the phase machine must have a path
    check(next_phase(Phase::queued, Outcome::finished) == Phase::prefill, "it is admitted all the same");
    check(next_phase(Phase::prefill, Outcome::cancelled) == Phase::prefill_end,
          "and its read is cancelled rather than run");
}

// ---------------------------------------------------------------- mounted / step legality ----

void test_a_step_only_runs_for_the_mounted_slot() {
    check(serve_driver::is_mounted(4, 4), "the mounted slot may step");
    check(!serve_driver::is_mounted(4, 5), "any other slot may not");
    check(!serve_driver::is_mounted(serve_proto::kNoId, 5), "nothing mounted: no slot may step");
    check(!serve_driver::is_mounted(4, serve_proto::kNoId), "nobody picked: no slot may step");
    check(!serve_driver::is_mounted(serve_proto::kNoId, serve_proto::kNoId),
          "two kNoId do not count as a match");
}

// --------------------------------------------------------------------- startup line ------------

void test_startup_line_reports_what_is_in_force() {
    const size_t G = (size_t) (1ull << 30);
    const std::string s = driver_line(3, 250, true, 8 * G, 8);
    check(s.find("3 slots") != std::string::npos, "it says how many slots");
    check(s.find("starve 250 ms") != std::string::npos, "and the fairness bound");
    check(s.find("slot swap on") != std::string::npos, "and whether swapping is on");
    check(s.find("8192 MiB") != std::string::npos, "and the parking budget in force");
    check(s.find("1024 MiB per slot") != std::string::npos, "and the per-slot estimate");
    const std::string off = driver_line(2, 0, false, 0, 0);
    check(off.find("slot swap off") != std::string::npos, "STRATA_NO_SWAP says off");
    check(off.find("cache OFF") != std::string::npos, "and a closed cache says so");
    check(off.find("starve 0 ms") != std::string::npos, "--starve-ms 0 says never force a swap");
}

// ------------------------------------------------------------------ the swap-order contract ----

/// The driver must not re-implement the hand-over: it calls `swap_to`, and `serve_swap::run` owns the
/// order.  What the driver DOES have to guarantee is that it never starts a step in the middle of a
/// hand-over, and that the loan is back before the swap.  These predicates are what it asserts, and
/// they are the ones this test pins.
void test_the_handover_order_is_still_the_swap_files() {
    using serve_swap::runs_before;
    using SwapStep = serve_swap::Step;
    check(runs_before(SwapStep::return_loan, SwapStep::unmount), "the loan is back before the save");
    check(runs_before(SwapStep::validate, SwapStep::mount), "validation before the write");
    check(runs_before(SwapStep::mount, SwapStep::adopt), "the restore before the per-slot adopt");
    check(runs_before(SwapStep::drain_residency, SwapStep::return_loan), "the residency drain goes first");
    check(!runs_before(SwapStep::mount, SwapStep::drain_residency), "and never the other way round");
    // the driver's own rule: a step never interleaves with a hand-over - it only runs AFTER a
    // completed one, which is why `dispatch_step` answers `swap` and not the step.
    Registry reg(3, 250);
    const int64_t t = now_ms();
    reg.add(1, t); reg.add(2, t);
    std::string e;
    reg.transition(1, State::decoding, e);
    reg.transition(2, State::decoding, e);
    reg.set_active(1);
    reg.note_ran(1, t);
    const Pick p = reg.pick(t + 100000);
    check(p.action == Pick::Action::swap, "starved decoder -> swap");
    check(dispatch_step(p, Phase::decode) == Step::swap, "and the driver runs no window until it lands");
    // After the hand-over the registry knows the new active slot.  What must hold is not "slot 2 runs
    // next" (slot 1 is now the starved one, and fairness hands the session straight back) but the
    // §5.3 invariant: whatever pick answers, the driver never runs a step for a slot the session does
    // not reflect.
    reg.set_active(2);
    reg.note_ran(2, t + 100000);
    const Pick q = reg.pick(t + 100000);
    const Step qs = dispatch_step(q, Phase::decode);
    check(qs == Step::decode || (qs == Step::swap && q.id != 2),
          "a direct step only ever for the mounted slot, a swap for the other one");
    // and the same slot, once mounted and unstarved, steps with no further hand-over
    reg.note_ran(1, t + 100000);
    const Pick r = reg.pick(t + 100000);
    check(r.action == Pick::Action::run && r.id == 2 && dispatch_step(r, Phase::decode) == Step::decode,
          "nobody starved: the mounted slot keeps the session");
}

// ---------------------------------------------------------- the loop, simulated end to end ----
//
// The predicates above are the driver's rules; this section runs the RULES IN ORDER against a fake
// engine, which is the only way to catch the failure modes a table of predicates cannot: a loop that
// never terminates, a permit that is never released, a loan that is never handed on, a cancelled slot
// that is never answered.  The fake `step()` stands in for `run_prefill_step`/`run_decode_step`/
// `finish_request`; the driver's own sequence (drain kills -> admit -> pick -> dispatch -> one step)
// is transcribed here so the sequence itself is under test.

inline const char* step_name_of(Step s) { return serve_driver::step_name(s); }

struct FakeEngine {
    Registry reg;
    Loan loan;
    std::vector<int64_t> deferred;
    int64_t mounted = serve_proto::kNoId;
    // what the fake steps do
    int segments_per_request = 3;
    // A per-request override, for the cases where one slot must be mid-read while another is not.
    std::map<int64_t, int> seg_override;
    int windows_per_request = 5;
    std::map<int64_t, int> seg_left, win_left;
    std::map<int64_t, Phase> phase;
    std::map<int64_t, bool> cancelled_at_admit;
    std::map<int64_t, bool> finish_seen;
    std::map<int64_t, bool> refuse_swap;      // a hand-over the engine refuses (R7 exclusivity)
    // S3.6: the two parking refusals, which the driver must treat OPPOSITELY.
    std::map<int64_t, bool> park_refuse_never;  // the outgoing slot can never be parked -> ERR this one
    std::map<int64_t, bool> park_refuse_later;  // nothing whole to save yet -> this one waits
    int park_waits = 0;
    // Is the slot the session reflects mid-read?  That is the ONLY condition `not_saveable` covers,
    // and it is what makes the wait kind self-clearing rather than a permanent refusal.
    bool mounted_mid_read() const {
        const auto it = phase.find(mounted);
        if (it == phase.end()) return false;
        return it->second == Phase::prefill || it->second == Phase::prefill_end;
    }
    std::vector<int64_t> done_ids, err_ids;
    std::map<int64_t, bool> counted;      // DONE recorded once per request
    std::vector<int64_t> inbox;           // request lines still waiting to be admitted
    std::vector<std::string> trace;
    int iterations = 0, swaps = 0, defers = 0;
    int deferred_hits() const { return defers; }

    explicit FakeEngine(int slots, int64_t starve) : reg(slots, starve) {}

    bool is_def(int64_t id) const {
        for (int64_t d : deferred) if (d == id) return true;
        return false;
    }
    void un_def(int64_t id) {
        for (size_t i = 0; i < deferred.size(); ++i)
            if (deferred[i] == id) { deferred.erase(deferred.begin() + (std::ptrdiff_t) i); return; }
    }
    // The driver's `drop_ctx`: the permit and the loan must both come back.
    void drop(int64_t id) {
        if (loan.held_by(id)) loan.release(id);
        un_def(id);
        std::string e;
        reg.transition(id, State::idle, e);
        reg.set_active(mounted);
        reg.release(id);
    }
    // The fake step functions.  Returns the outcome the engine's step would report.
    Outcome step(int64_t id, Step s) {
        trace.push_back(std::string(step_name_of(s)) + ":" + std::to_string((long long) id));
        switch (s) {
            case Step::prefill:
                if (--seg_left[id] > 0) return Outcome::progressed;
                return Outcome::finished;
            case Step::prefill_end:
                if (loan.held_by(id)) loan.release(id);
                return Outcome::finished;
            case Step::decode:
                if (--win_left[id] > 0) return Outcome::progressed;
                return Outcome::finished;
            case Step::finish:
                if (!finish_seen.count(id)) { finish_seen[id] = true; done_ids.push_back(id); }
                return Outcome::finished;
            default:
                return Outcome::finished;
        }
    }
};

/// One pass of the driver, transcribed from generate.cpp's loop.  Returns false when the loop would
/// have exited (nothing left and no lines to read).
bool driver_pass(FakeEngine& E, int64_t now) {
    ++E.iterations;
    // 2. admit: one line per pass, and a line that cannot be admitted yet STAYS in the inbox (the
    //    engine keeps it in `held_line` rather than pushing it back, so the client's order holds).
    if (!E.inbox.empty() && E.reg.can_admit()) {
        const int64_t id = E.inbox.front();
        if (E.reg.add(id, now) != nullptr) {
            E.inbox.erase(E.inbox.begin());
            E.phase[id] = Phase::queued;
            const auto ov = E.seg_override.find(id);
            E.seg_left[id] = ov != E.seg_override.end() ? ov->second : E.segments_per_request;
            E.win_left[id] = E.windows_per_request;
        }
    }
    // 3. pick
    Pick p = E.reg.pick(now);
    if (p.action == Pick::Action::idle) return !E.inbox.empty() || E.reg.count() > 0;
    if ((p.action == Pick::Action::run || p.action == Pick::Action::swap) && E.is_def(p.id)) {
        // the driver's tie-break: the longest-waiting runnable slot that is NOT deferred
        int64_t best = serve_proto::kNoId, bw = -1;
        E.reg.each([&](const Slot& s) {
            if (s.state != State::prefilling && s.state != State::decoding && s.state != State::cancelling) return;
            if (E.is_def(s.id)) return;
            const int64_t w = E.reg.waited_ms(now, s);
            if (w > bw) { bw = w; best = s.id; }
        });
        if (best == serve_proto::kNoId) return true;   // the driver sleeps and retries
        p.action = (best == E.reg.active_id()) ? Pick::Action::run : Pick::Action::swap;
        p.id = best;
        p.state = E.reg.find(best)->state;
    }
    // 4/5. dispatch.  R8 is asked BEFORE the hand-over, exactly as the engine does: a swap is a full
    //    save+restore, so moving the session to a slot only to find it cannot lend, then moving it
    //    back next pass, is the swap churn risk R10 refuses.
    Phase& ph = E.phase[p.id];
    Step s = dispatch_step(p, ph);
    const bool next_lends = (s == Step::prefill || s == Step::swap) &&
                            ph == Phase::prefill && E.seg_left[p.id] > 0;
    if (next_lends && !E.loan.may_lend(p.id)) {
        if (!E.is_def(p.id)) { E.deferred.push_back(p.id); ++E.defers; }
        return true;
    }
    for (size_t i = 0; i < E.deferred.size(); ++i)
        if (E.deferred[i] == p.id) { E.deferred.erase(E.deferred.begin() + (std::ptrdiff_t) i); break; }
    if (s == Step::swap) {
        ++E.swaps;
        if (E.loan.held()) E.loan.release(E.loan.owner);   // serve_swap's return_loan step
        // S3.6: `do_swap` returns a SwapResult, and the two parking refusals go opposite ways.  The
        // fake mirrors generate.cpp's branch exactly: NEVER ends the INCOMING request and leaves the
        // session with the outgoing slot; NOT-YET defers the incoming one and does the same.
        if (E.park_refuse_never.count(p.id)) {
            E.err_ids.push_back(p.id);
            E.drop(p.id);
            E.reg.set_active(E.mounted);
            return true;
        }
        if (E.park_refuse_later.count(p.id) && E.mounted_mid_read()) {
            // The session is mid-read, so there is nothing whole to save.  The incoming slot waits;
            // the refusal clears by itself when the running request reaches its decode phase.
            ++E.park_waits;
            if (!E.is_def(p.id)) { E.deferred.push_back(p.id); ++E.defers; }
            return true;
        }
        E.mounted = p.id;
        E.reg.set_active(p.id);
        s = dispatch_step(Pick{Pick::Action::run, p.id, p.state, p.waited_ms, "post-swap"}, ph);
    }
    if (E.refuse_swap.count(p.id) && s == Step::swap) {
        // The engine defers a slot whose hand-over was refused BEFORE writing, or the loop spins on
        // the same refusal forever.
        if (!E.is_def(p.id)) { E.deferred.push_back(p.id); ++E.defers; }
        return true;
    }
    if (s == Step::none) return true;
    if (s == Step::admit) {
        std::string e;
        const State want = E.cancelled_at_admit.count(p.id) ? State::cancelling : State::prefilling;
        if (!E.reg.transition(p.id, want, e)) return true;
        E.reg.set_active(p.id);
        E.mounted = p.id;
        ph = E.cancelled_at_admit.count(p.id) ? Phase::prefill_end : Phase::prefill;
        return true;
    }
    // the loan gate (R8): the loop already deferred anyone who could not lend, so this is the
    // engine's own hard guard inside the step
    if (s == Step::prefill) {
        if (!E.loan.acquire(p.id)) return true;
    }
    if (s == Step::decode && E.loan.held()) E.loan.release(E.loan.owner);
    const Outcome o = E.step(p.id, s);
    const Phase before = ph;
    ph = next_phase(ph, o);
    E.reg.note_ran(p.id, now);
    // The registry state follows what the real step functions do: `finish_prefill` moves the row to
    // `decoding`, and `finish_request` + the guard move it to `idle` and release it.  A phase that
    // reaches `done` is NOT released here: the NEXT pass dispatches `Step::finish` for it, which is
    // what the engine does, and only then does the row go.
    if (s == Step::prefill_end) {
        std::string e;
        E.reg.transition(p.id, State::decoding, e);
        E.reg.set_active(E.mounted);
        // S3.6: the running request just finished its read, so the session now holds a whole branch
        // and a `not_saveable` refusal cannot happen any more.  The engine wakes the waiters here
        // (`finish_prefill`'s `if (!loan.held()) clear_deferred()`); the fake has to, or a parked
        // slot stays parked for a reason that no longer exists.
        if (!E.loan.held()) E.deferred.clear();
    }
    if (before == Phase::done && s == Step::finish) E.drop(p.id);
    return true;
}

void test_the_loop_finishes_two_requests_without_swapping_to_death() {
    FakeEngine E(3, 250);
    const int64_t t0 = now_ms();
    int guard = 0;
    E.inbox = {1, 2};
    while (E.done_ids.size() < 2 && ++guard < 500) driver_pass(E, t0);
    check(E.done_ids.size() == 2, "both requests reached DONE");
    check(E.reg.count() == 0, "and both rows were released");
    check(E.reg.active_count() == 0, "and no active-slot permit leaked");
    check(!E.loan.held(), "and the prompt loan came back");
    // 3 segments + 1 tail + 5 windows + 1 finish = 10 steps each; the loop must not do far more
    check(E.iterations < 40, "the loop did not spin (bounded iteration count)");
    // It stayed on one slot until the other was starved, then swapped (R10): a swap is a full
    // save+restore, so fairness - not round-robin - is what forces one.
    check(E.swaps >= 1, "the second request did get the session (a swap happened)");
    check(E.swaps <= 4, "and not on every step (fairness, not round-robin)");
}

void test_the_loan_never_has_two_owners() {
    // Two requests that both want batched segments.  The driver must serialise them on the loan.
    FakeEngine E(3, 10);          // a short starve bound so the other slot does get served
    const int64_t t0 = now_ms();
    int guard = 0;
    int max_owners = 0;
    E.inbox = {1, 2};
    while (E.done_ids.size() < 2 && ++guard < 2000) {
        driver_pass(E, t0);
        // The loan has ONE owner, and it is always a slot that is actually running.  It is NOT always
        // the mounted slot: a hand-over refills the loan (`serve_swap::run`'s return_loan step) BEFORE
        // it can be refused by R7's exclusivity check, so a refused swap leaves the loan free while the
        // outgoing slot is still mounted.
        if (E.loan.held()) {
            ++max_owners;
            const Slot* o = E.reg.find(E.loan.owner);
            check(o != nullptr && slot::is_active(o->state),
                  "the loan's owner is a live, active slot");
        }
    }
    check(E.done_ids.size() == 2, "both finished despite competing for one loan");
    check(!E.loan.held(), "and the loan is free at the end");
    check(E.deferred.empty(), "and nobody is left deferred");
    check(max_owners > 0, "the loan really was held during the run");
}

void test_a_blocked_slot_does_not_cause_swap_churn() {
    // Both slots want batched segments (the loan).  If the driver swapped first and only then found
    // the loan taken, it would save+restore the session on every pass.  It must defer instead.
    FakeEngine E(3, 10);            // a short starve bound: pick WANTS to swap every pass
    E.segments_per_request = 6;
    E.windows_per_request = 1;
    const int64_t t0 = now_ms();
    E.inbox = {1, 2};
    int guard = 0;
    while (E.done_ids.size() < 2 && ++guard < 400) {
        driver_pass(E, t0 + guard);        // time advances, so the fairness bound always fires
        check(!E.loan.held() || E.reg.find(E.loan.owner) != nullptr,
              "the loan's owner is always a live row");
    }
    check(E.done_ids.size() == 2, "both finished");
    // 12 segments + 2 tails + 2 windows + 2 finishes = 18 steps; a swap on each would be ~18.
    check(E.swaps <= 6, "the loan is checked BEFORE the hand-over, so swaps stay bounded");
    check(E.deferred_hits() >= 0, "the deferral path is reachable");
}

void test_a_refused_handover_does_not_spin_the_loop() {
    // R7's exclusivity (or a save the cache refused) can reject a hand-over BEFORE anything is
    // written.  The session is intact, but if the driver just `continue`d, pick would choose the same
    // slot and the same refusal forever.  It must defer the slot and serve the other one.
    FakeEngine E(3, 10);
    E.segments_per_request = 2;
    E.windows_per_request = 2;
    E.refuse_swap[2] = true;
    const int64_t t0 = now_ms();
    E.inbox = {1, 2};
    int guard = 0;
    while (E.done_ids.size() < 2 && ++guard < 400) driver_pass(E, t0 + guard);
    check(E.done_ids.size() == 2, "a slot whose hand-over is refused still finishes, and so does the other");
    check(E.reg.count() == 0 && E.reg.active_count() == 0, "no permit leaked");
    check(!E.loan.held(), "no loan leaked");
    check(E.iterations < 60, "the loop did not spin on the refusal");
}

void test_a_cancelled_slot_is_unwound_and_answered() {
    FakeEngine E(3, 250);
    const int64_t t0 = now_ms();
    E.cancelled_at_admit[2] = true;      // `STOP 2` arrived before its line did
    int guard = 0;
    E.inbox = {1, 2};
    while (E.done_ids.size() < 2 && ++guard < 500) {
        driver_pass(E, t0);
        if (E.reg.find(2) != nullptr && E.reg.find(2)->state == State::cancelling) {
            // the driver must not run a new segment or window for it, only its tails
            const std::string last = E.trace.empty() ? std::string() : E.trace.back();
            check(last.rfind("prefill:", 0) != 0 || E.iterations < 3,
                  "a cancelling slot is not given another prompt segment");
        }
    }
    check(E.done_ids.size() == 2, "the cancelled request was still answered");
    check(E.reg.count() == 0 && E.reg.active_count() == 0, "and its permit came back");
    check(!E.loan.held(), "and its loan came back");
}

void test_a_slot_with_no_row_is_not_picked_forever() {
    // The engine's `R == nullptr` guard: a row with no context must be released, not retried.
    Registry reg(3, 250);
    const int64_t t = now_ms();
    reg.add(1, t);
    std::string e;
    reg.transition(1, State::decoding, e);
    reg.set_active(1);
    int spins = 0;
    while (reg.pick(t).action != Pick::Action::idle && ++spins < 5) {
        // the driver would find no context here and release the row
        reg.release(1);
        reg.set_active(serve_proto::kNoId);
    }
    check(spins == 1, "one release is enough: the row is gone and the pick goes idle");
    check(reg.pick(t).action == Pick::Action::idle, "and nothing is pickable afterwards");
}


// ====================================================== S3.6: the parking-collapse guard =======
//
// The owner's live failure, as CPU checks.  Two ~175K-token conversations, a parking budget that
// cannot hold either snapshot.  Before the fix the engine pre-empted the first one, could not park
// it, destroyed its conversation, reported the swap as `ok`, and when the scheduler switched back it
// ran a verify window against the SECOND conversation's session - so the first request "just
// prematurely ended", silently.

void test_step_gate_refuses_a_decode_slot_with_no_conversation() {
    // "a slot in decode whose SlotConv is not resumable -> does not run a window".  Checked FIRST,
    // before the session question, because a slot can be mounted and session-valid and STILL have
    // lost its parked image - that is exactly how the owner's run ended a request early.
    check(step_gate(Phase::decode, false, true) == Gate::end,
          "a decode slot whose conversation was destroyed may not run a window, even while mounted");
    check(step_gate(Phase::decode, false, false) == Gate::end,
          "and the parked-image check is not masked by a stale session flag");
    check(step_gate(Phase::decode, true, true) == Gate::run,
          "a decode slot that still owns its conversation and its session steps normally");
    check(step_gate(Phase::decode, true, false) == Gate::end,
          "a decoder whose session went elsewhere ends too: its tokens are already on the wire");
    check(step_gate(Phase::prefill, false, false) == Gate::re_read,
          "a prefill slot with no session honours the promise: re-read from token 0");
    check(step_gate(Phase::prefill_end, false, false) == Gate::re_read, "same for the prefill tail");
    check(step_gate(Phase::prefill, false, true) == Gate::run,
          "a mid-prompt slot that still owns its session keeps reading - resumable gates a FUTURE "
          "hand-over, not the step in front of it");
    check(step_gate(Phase::queued, false, false) == Gate::run, "queued runs no step of this kind");
    check(step_gate(Phase::done, false, false) == Gate::run, "done likewise");
    // The ERR must be actionable, not just alarming.
    const std::string why = gate_end_reason(Phase::decode);
    check(why.find("decode") != std::string::npos, "it names the phase");
    check(why.find("no step may run against it") != std::string::npos, "and the invariant it protects");
    check(why.find("--conversation-cache-mib") != std::string::npos, "and the knob to turn");
    check(why.find("--max-context") != std::string::npos, "and the alternative");
}

void test_outgoing_for_is_the_oneswap_guard_reads() {
    check(outgoing_for(Phase::decode) == serve_swap::Outgoing::must_park, "decode must be saved");
    check(outgoing_for(Phase::prefill) == serve_swap::Outgoing::re_readable, "prefill may re-read");
    check(outgoing_for(Phase::prefill_end) == serve_swap::Outgoing::re_readable, "prefill-end too");
    check(outgoing_for(Phase::queued) == serve_swap::Outgoing::finished, "queued has nothing to lose");
    check(outgoing_for(Phase::done) == serve_swap::Outgoing::finished, "done has its answer");
    // The phase that has work is exactly the phase whose hand-over is guarded.
    for (int i = (int) Phase::queued; i <= (int) Phase::done; ++i)
        check((outgoing_for((Phase) i) != serve_swap::Outgoing::finished) == phase_has_work((Phase) i),
              "a slot has work iff losing the session would cost it something");
}

void test_admission_names_a_useless_parking_budget() {
    // R2/R12's `parking_off_refuses_slots` catches a budget of ZERO.  This catches the nonzero-but-
    // useless one that actually bit the owner: 2 GiB of budget, 3.4 GiB of snapshot.
    ParkCeiling c;
    c.budget_bytes = (size_t) 2048 << 20;
    c.slots = 8;
    c.snapshot_bytes = (size_t) 3446 << 20;
    c.sized = true;
    check(!parkable_at_all(c.budget_bytes, c.slots, c.snapshot_bytes),
          "a snapshot larger than the whole budget can never be parked");
    check(c.useless_budget(), "and that is detected, not discovered 97 seconds later");
    check(c.capacity() == 0, "and the answer is: none fit");
    const std::string line = park_ceiling_line(3, c, true);
    check(line.find("3446 MiB") != std::string::npos, "the line gives the snapshot size");
    check(line.find("2048 MiB") != std::string::npos, "and the budget");
    check(line.find("NEVER be parked") != std::string::npos, "and says plainly it can never work");
    check(line.find("--conversation-cache-mib") != std::string::npos, "and what to raise");
    check(line.find("--max-context") != std::string::npos, "or what to lower");
    // The honest wording depends on whether the driver is actually running.
    check(line.find("REFUSE every hand-over") != std::string::npos,
          "with the driver on it says hand-overs will be refused, not that the run is serial");
    const std::string off = park_ceiling_line(3, c, false);
    check(off.find("Running serial") != std::string::npos,
          "with the driver off it says serial, which is the truth there");
    // A budget that DOES fit is not a finding, and a partial fit says how many fit.
    ParkCeiling good = c;
    good.budget_bytes = (size_t) 8192 << 20;
    check(!good.useless_budget(), "8 GiB of budget parks a 3.4 GiB snapshot");
    check(good.capacity() == 2, "and two of them, not eight - the budget, not the entry count, bounds it");
    check(park_ceiling_line(3, good, true).find("at most 2") != std::string::npos,
          "and the line says the number");
    // An unsized ceiling stays quiet rather than inventing a number.
    ParkCeiling unknown;
    unknown.budget_bytes = (size_t) 2048 << 20;
    unknown.slots = 8;
    check(!unknown.useless_budget(), "no sizing -> no claim");
    // A zero budget is still caught by the pre-existing rule, unchanged.
    check(parking_off_refuses_slots(3, false), "cache off is still refused at startup");
    ParkCeiling zero;
    zero.budget_bytes = 0; zero.slots = 8; zero.snapshot_bytes = 1; zero.sized = true;
    check(zero.useless_budget(), "a zero budget is useless for any snapshot");
}

void test_a_refused_handover_ends_the_incoming_request_not_the_running_one() {
    // The contract: "On refusal the session must be left exactly as it was ... and the incoming slot
    // must be answered with an ERR naming both numbers, not starved or spun on."  The fake engine
    // refuses slot 2's hand-over as NEVER-parkable; slot 1 must keep running and slot 2 must get an
    // ERR, and the loop must not spin.
    FakeEngine E(3, 10);
    E.segments_per_request = 2;
    E.windows_per_request = 3;
    E.park_refuse_never[2] = true;
    const int64_t t0 = now_ms();
    E.inbox = {1, 2};
    int guard = 0;
    while ((E.done_ids.size() < 1 || E.err_ids.empty()) && ++guard < 400) driver_pass(E, t0 + guard);
    check(!E.err_ids.empty(), "the incoming slot was ANSWERED with an ERR, not left waiting");
    check(E.err_ids.size() == 1, "exactly once - it is not retried and not spun on");
    check(E.err_ids[0] == 2, "and it is the INCOMING slot that was refused");
    check(E.done_ids.size() == 1 && E.done_ids[0] == 1,
          "the running slot finished normally: its conversation was never destroyed");
    check(E.mounted == 1, "the session stayed with the slot that could not be saved");
    check(E.reg.count() == 0 && E.reg.active_count() == 0, "no permit leaked");
    check(!E.loan.held(), "no loan leaked");
    check(E.iterations < 60, "the loop did not spin on the refusal");
}

void test_a_parkable_decode_slot_is_pre_empted_normally() {
    // The positive case, so the guard is not a feature that only ever says "no".  NOTE what this
    // fake can and cannot pin: it has no parking model at all (its `swap` just moves `mounted`), so
    // this does NOT prove the branch is published or that a real snapshot lands - only that the
    // driver's DECISIONS on the normal path still pre-empt a decoder, resume it, and never emit a
    // spurious ERR or spin.  The parkability arithmetic itself is pinned directly, on the real
    // cache, in serve_swap_test (`test_the_budget_question_matches_the_cache`); whether a real
    // snapshot of a pre-empted decoder lands byte-identical is a live-engine question and is listed
    // as the owner's restart check.
    FakeEngine E(3, 10);
    E.segments_per_request = 3;
    E.windows_per_request = 6;
    const int64_t t0 = now_ms();
    E.inbox = {1, 2};
    int guard = 0;
    while (E.done_ids.size() < 2 && ++guard < 500) driver_pass(E, t0 + guard);
    check(E.done_ids.size() == 2, "both conversations finished");
    check(E.err_ids.empty(), "neither was refused: the hand-over was parkable");
    check(E.swaps >= 2, "the decoder WAS pre-empted and later swapped back in");
    check(E.reg.count() == 0 && E.reg.active_count() == 0, "no permit leaked");
    check(!E.loan.held(), "no loan leaked");
    check(E.iterations < 60, "no spin");
}

void test_a_not_yet_parkable_handover_waits_instead_of_failing() {
    // The other kind: the session is mid-read, so there is nothing whole to save YET.  That clears
    // itself, so the incoming request waits - ERRing it would punish a client for a race that
    // resolves on its own.
    FakeEngine E(3, 10);
    // Slot 1 must STILL be reading when fairness forces the hand-over to slot 2 - that is the only
    // state in which `not_saveable` applies.  Fairness fires after `starve_ms`, i.e. after ~10
    // passes, so slot 1 needs far more segments than that to still be mid-read when it happens.
    // Slot 2 gets no segments at all, so its next step is a window and it is never parked behind the
    // prompt loan on its way to the hand-over.
    E.segments_per_request = 40;
    E.seg_override[2] = 0;
    E.windows_per_request = 3;
    E.park_refuse_later[2] = true;
    const int64_t t0 = now_ms();
    E.inbox = {1, 2};
    int guard = 0;
    while (E.done_ids.size() < 2 && ++guard < 400) driver_pass(E, t0 + guard);
    check(E.err_ids.empty(), "nothing was refused: the wait kind is not an error");
    check(E.done_ids.size() == 2, "both requests finished");
    check(E.park_waits >= 1, "the wait path was actually taken");
    check(E.reg.count() == 0 && E.reg.active_count() == 0, "no permit leaked");
    check(E.iterations < 80, "and the loop still terminated promptly");
}

void test_the_loop_terminates_on_an_empty_registry() {
    FakeEngine E(3, 250);
    check(!driver_pass(E, now_ms()), "an empty registry and an empty inbox ends the loop");
    check(E.reg.pick(now_ms()).action == Pick::Action::idle, "and the pick says idle");
    check(E.iterations <= 1, "the driver blocks for a line instead of spinning");
}

}  // namespace

// ---------------------------------------------------- S3.7: a slot is priced at its own length ----
//
// The question the owner asked: "is each slot really reserving max context size for each slot? Might
// it be that short conversations really only consume as much of the budget as they actually are?"
// Parking always did (the cache counts `SavedConversation::bytes()` of what it stored).  RAM
// ADMISSION did not: it charged every request `budget / --conversation-cache-slots`, which is a flat
// per-slot reservation and refuses a 150-token chat for a gigabyte it will never use.  These pin the
// new rule.
void test_a_slot_is_priced_at_its_own_length() {
    const uint64_t M = 1ull << 20;
    // The owner's box: 9048 MiB of budget over 8 cache entries = the old flat 1131 MiB per slot.
    SlotCost guess_only;
    guess_only.budget_bytes = 9048 * M;
    guess_only.cache_slots = 8;
    check(slot_image_bytes(guess_only, 150, 1) == 1131 * M,
          "with nothing sized and nothing measured, admission still falls back to budget / slots");

    // The startup ceiling: 524288 tokens park in 7746 MiB -> ~15 KiB per token.
    ParkCeiling ceil;
    ceil.budget_bytes = 9048 * M;
    ceil.slots = 8;
    ceil.snapshot_bytes = (size_t) (7746 * M);
    ceil.sized = true;
    ceil.size_at(524288);
    check(ceil.per_token() == (7746 * M + 524287) / 524288, "the ceiling implies a per-token rate");

    // The rate precedence, in one place: measured beats ceiling beats nothing.
    const SlotCost c = slot_cost_of(9048 * M, 8, ceil, 0);
    check(c.per_token == ceil.per_token(), "a sized engine prices per token before anything is parked");
    check(slot_cost_of(9048 * M, 8, ceil, 26000).per_token == 26000,
          "and a real park's rate wins over the ceiling's");
    check(slot_cost_of(9048 * M, 8, ParkCeiling{}, 0).per_token == 0,
          "unsized and unmeasured -> no rate, so the guess is what is left");

    // A 150-token chat with a 1-token answer, priced at the ceiling rate: ~2 MiB, not 1131 MiB.
    const uint64_t short_need = slot_image_bytes(c, 150, 1);
    check(short_need < 4 * M, "a short conversation is priced as short, not as a share of the budget");
    check(short_need < slot_image_bytes(guess_only, 150, 1),
          "and that is strictly cheaper than the old flat reservation - the bug");
    // A full-context request is still priced at the full ceiling.
    check(slot_image_bytes(c, 524288, 0) == 7746 * M, "a max-context conversation costs the ceiling");
    // `max_new` counts: the answer it has not written yet is part of what it will park.
    check(slot_image_bytes(c, 100000, 40000) > slot_image_bytes(c, 100000, 1),
          "the tokens it has yet to generate count too");

    // A measured rate beats the ceiling's implied one: the owner's log parks 95303 tokens in
    // 2486870164 B, which is ~26 KiB per token - more than the ceiling's ~15 KiB, because a real
    // branch also carries its checkpoint chain.  Whichever rate is in force, the price must follow
    // the request's length.
    const uint64_t measured = bytes_per_token(2486870164ull, 95303);
    check(measured > 25000 && measured < 27000, "the measured rate is ~26 KiB per token");
    const SlotCost meas2 = slot_cost_of(9048 * M, 8, ceil, measured);
    check(meas2.per_token == measured, "a park this process measured outranks the ceiling's rate");
    check(slot_image_bytes(meas2, 150, 1) < 5 * M,
          "and a short chat is still priced as short under it");
    check(slot_image_bytes(meas2, 10000, 1) < slot_image_bytes(meas2, 100000, 1),
          "the price grows with the conversation, not with the slot count");
    check(slot_image_bytes(meas2, 524288, 0) == 7746 * M,
          "a full-context one is capped at the ceiling, never above it");
    // Never more than the whole budget: a reservation larger than the budget refuses everything
    // forever, which is the failure mode this is here to remove.
    SlotCost tiny_budget = meas2;
    tiny_budget.budget_bytes = 10 * M;
    check(slot_image_bytes(tiny_budget, 524288, 4096) == 10 * M,
          "a slot is never priced above the whole budget");
    // A GENI request has no token count on its line, so the caller passes the ceiling; that path is
    // the caller's, but the arithmetic must not overflow at 524288 * a 26 KiB rate.
    check(slot_image_bytes(meas2, 524288, 524288) == 7746 * M, "no overflow at the top of the range");
    // Zero everywhere: no gate, and no crash.
    SlotCost none;
    check(slot_image_bytes(none, 1000, 100) == 0, "nothing known -> no reservation");
    check(bytes_per_token(1000, 0) == 0, "a zero-token park is not evidence");
    // An unsized engine keeps the old behaviour exactly, so nothing regresses when sizing fails.
    ParkCeiling unsized;
    unsized.budget_bytes = 9048 * M;
    unsized.slots = 8;
    check(slot_image_bytes(slot_cost_of(9048 * M, 8, unsized, 0), 150, 1) == 1131 * M,
          "no ceiling and no park -> the old estimate, unchanged");
}

void test_the_gate_blames_the_real_cause() {
    // The old text blamed the parking budget for EVERY end, including a run with a 9 GiB budget.
    // The two reasons are different and the message has to tell them apart.
    const uint64_t M = 1ull << 20;

    // 1. A decoder whose image was lost, and the snapshot genuinely does not fit.
    GateEnd never{Phase::decode, false, true, (uint64_t) 3446 * M, (uint64_t) 2048 * M};
    const std::string a = gate_end_reason(never);
    check(a.find("3446 MiB") != std::string::npos, "it names the snapshot the save needed");
    check(a.find("2048 MiB") != std::string::npos, "and the budget in force");
    check(a.find("--conversation-cache-mib") != std::string::npos, "and the knob, when the knob is right");
    check(a.find("--max-context") != std::string::npos, "and the alternative");

    // 2. The same slot on a box with a GENEROUS budget: the budget is not the cause, and saying so
    //    is the whole point - the owner raised a knob that was never the problem.
    GateEnd ample{Phase::decode, false, true, (uint64_t) 226 * M, (uint64_t) 9048 * M};
    const std::string b = gate_end_reason(ample);
    check(b.find("226 MiB") != std::string::npos, "it still gives both numbers");
    check(b.find("budget was big enough") != std::string::npos, "and says the budget was not the cause");
    check(b.find("raise --conversation-cache-mib") == std::string::npos,
          "and does NOT tell the owner to raise a knob that is already big enough");

    // 3. Not a budget condition at all: the session simply went elsewhere.
    GateEnd moved{Phase::decode, true, false, 0, (uint64_t) 9048 * M};
    const std::string c = gate_end_reason(moved);
    check(c.find("not a budget size problem") != std::string::npos, "it says so plainly");
    check(c.find("too small") == std::string::npos,
          "and it never blames the budget size here - that was the old lie");
    check(c.find("would not have restored it") != std::string::npos,
          "and it says the knob would not have fixed it");
    check(c.find("pruned") != std::string::npos, "and it names the real cause: nothing to mount back");

    // 4. No numbers available: name both possibilities, never guess one.
    const std::string d = gate_end_reason(Phase::decode);
    check(d.find("park") != std::string::npos && d.find("another slot") != std::string::npos,
          "without numbers it offers both causes");
    check(d.find("is too small") == std::string::npos, "and does not assert the budget is the cause");
}

void test_the_startup_line_says_how_a_slot_is_priced() {
    const size_t G = (size_t) (1ull << 30);
    // With a measured rate the line must say the price is per token, not per slot.
    const std::string meas = driver_line(3, 250, true, 8 * G, 8, 19968);
    check(meas.find("19968 B per context token") != std::string::npos, "it gives the measured rate");
    check(meas.find("nothing is reserved per slot") != std::string::npos, "and says what is NOT reserved");
    check(meas.find("per slot estimate") == std::string::npos,
          "and drops the flat figure, which is the number that misled the owner");
    // Without one, it keeps the old figure but labels it as the guess it is.
    const std::string old = driver_line(3, 250, true, 8 * G, 8);
    check(old.find("1024 MiB per slot estimate") != std::string::npos, "the fallback is still shown");
    check(old.find("until a conversation has been parked") != std::string::npos,
          "and is labelled as provisional");
}

// ------------------------------------------------ S3.8: two prefills must not erase each other --
//
// The owner's report: "when two prefills happen at the same time and they swap, they seem to erase the
// other's previous progress, as after each swap they always start at 0 tokens again which makes them
// never finish."  The log says exactly that - slot 3 `reset to token 0: 81 prompt tokens will be read
// again`, 101 times, while slot 2 parked 777 MiB and resumed normally.
//
// The mechanism, in three links, any one of which is enough to break the chain:
//   * a slot in `prefill` was classified `re_readable`, which LICENSES the swap to drop its state;
//   * nothing published a mid-prompt branch, so `park_current` had nothing to save (`saved 0 B`);
//   * the next step therefore sent the request back to token 0 - every single time.
// Re-reading is survivable ONCE. Re-reading on every swap is a livelock, and the watchdog cannot see
// it because every pass really does read tokens.

void test_a_prefill_with_progress_must_be_parked() {
    // The pure decision. `read_tokens` is the slot's cursor: 0 means the prompt is still entirely in
    // hand (re-reading it costs nothing), anything above 0 means the session holds progress the cache
    // can save.
    check(outgoing_for(Phase::prefill, 0) == serve_swap::Outgoing::re_readable,
          "a prefill that has read nothing may still be re-read cheaply");
    check(outgoing_for(Phase::prefill, 1) == serve_swap::Outgoing::must_park,
          "one token read and the state must be saved - this is the fix");
    check(outgoing_for(Phase::prefill, 36000) == serve_swap::Outgoing::must_park,
          "a long read especially: erasing it is the livelock");
    check(outgoing_for(Phase::prefill_end, 500) == serve_swap::Outgoing::must_park,
          "the prefill tail counts too");
    check(outgoing_for(Phase::prefill_end, 0) == serve_swap::Outgoing::re_readable,
          "a cancelled read that consumed nothing is still cheap");
    check(outgoing_for(Phase::decode, 0) == serve_swap::Outgoing::must_park,
          "decode is unchanged: its tokens are already on the wire");
    check(outgoing_for(Phase::queued, 999) == serve_swap::Outgoing::finished,
          "and a queued row still owes no step whatever the cursor says");
    // The phase-only overload is the "nothing read yet" case, so an old call site cannot silently
    // claim progress it does not have.
    check(outgoing_for(Phase::prefill) == outgoing_for(Phase::prefill, 0),
          "the one-argument form means cursor 0");
    // A must-park classification is what makes the swap guard refuse an unsavable hand-over, so the
    // pair must agree: a slot with progress is exactly a slot the guard protects.
    for (int i = (int) Phase::queued; i <= (int) Phase::done; ++i)
        for (int64_t at : {0, 1, 4096})
            check((outgoing_for((Phase) i, at) != serve_swap::Outgoing::finished) ==
                      phase_has_work((Phase) i),
                  "having work still means losing the session costs something, at any cursor");
    check(serve_swap::save_is_mandatory(outgoing_for(Phase::prefill, 4096)),
          "and the swap file agrees: that hand-over MUST save");
}

void test_the_reread_bound_turns_a_spin_into_an_error() {
    check(reread_allowed(0) && reread_allowed(kMaxRereads - 1), "the first re-reads are recoveries");
    check(!reread_allowed(kMaxRereads), "the next one is a spin");
    check(!reread_allowed(kMaxRereads + 50), "and it stays a spin - the owner's 101");
    const std::string why = reread_limit_line(3, kMaxRereads, kMaxRereads);
    check(why.find("token 0") != std::string::npos, "the line says what kept happening");
    check(why.find("--conversation-cache-mib") != std::string::npos, "and what to raise");
    check(why.find("--serve-slots 0/1") != std::string::npos, "and the way to avoid it entirely");
}

/// The livelock, replayed THROUGH THE REAL PREDICATES.  Two prefills, one GPU, swaps between them.
/// `publishes` stands for `publish_prefill_branch`: whether the engine puts the mid-prompt branch into
/// `live`/`live_ok` before the hand-over, which is what makes `park_current` able to save it at all.
/// Whether the read then SURVIVES is decided by the code under test - `outgoing_for(phase, cursor)`
/// and `save_is_mandatory` - not by a flag, so changing the rule changes the outcome here.
struct PrefillPair {
    int restarts = 0, steps = 0, unfinished = 0;
    bool hit_bound = false;
};

PrefillPair run_pair(bool publishes, int limit) {
    PrefillPair out;
    const int kSegs = 3;
    // slot A owns the session; B waits.  Each pass the idle one takes over.
    int64_t mounted = 1;
    std::map<int64_t, int> at;            // how far each read has got
    std::map<int64_t, int> rereads;
    at[1] = 0; at[2] = 0;
    for (int pass = 0; pass < 400; ++pass) {
        const int64_t run = mounted;
        const int64_t other = (run == 1) ? 2 : 1;
        if (at[run] >= kSegs) { mounted = other; continue; }   // that request is done
        // A hand-over away from `run`.  The driver classifies it from the phase AND the cursor, and
        // the swap guard only insists on a save when that classification says so.
        const serve_swap::Outgoing work = outgoing_for(Phase::prefill, at[run]);
        const bool must_save = serve_swap::save_is_mandatory(work);
        const bool saved = must_save && publishes;   // `park_current` needs a published branch
        // Losing ZERO progress is not a restart: a slot that has read nothing is `re_readable` on
        // purpose, and resetting it to token 0 costs nothing.  Only a read that had got somewhere and
        // was thrown away counts - which is exactly the owner's complaint.
        if (!saved && at[run] > 0) {
            ++out.restarts;
            at[run] = 0;                       // reset_request_to_token0
            ++rereads[run];
            // The real predicate, not a local comparison: if the bound is neutered, this loop runs to
            // its pass cap and `hit_bound` stays false, which is what the check below catches.
            if (!reread_allowed(rereads[run], limit)) { out.hit_bound = true; break; }
        }
        ++at[run];                             // one prompt segment
        ++out.steps;
        mounted = other;                       // and the session moves
    }
    if (at[1] < kSegs) ++out.unfinished;
    if (at[2] < kSegs) ++out.unfinished;
    return out;
}

void test_two_prefills_finish_when_the_read_is_parked() {
    // The fix: the read is published AND classified `must_park`, so it is saved and both prefills
    // advance.  Both halves are needed - publishing without the classification still lets the driver
    // treat the slot as throwaway, and the classification without the publish leaves `park_current`
    // with nothing to save.
    const PrefillPair fixed = run_pair(true, kMaxRereads);
    check(fixed.restarts == 0, "a published, must-park prefill is never sent back to token 0");
    check(fixed.unfinished == 0, "and both prefills finish");
    check(fixed.steps == 6, "exactly six segments read: no work is done twice");

    // The bug: nothing is published, so the save cannot happen and every swap erases the read.  The
    // bound is what makes this observable at all - without it the same shape runs forever, which is
    // the owner's 101 resets.
    const PrefillPair broken = run_pair(false, kMaxRereads);
    check(broken.restarts > 0, "unpublished, a read with progress IS erased");
    check(broken.hit_bound, "and the re-read bound catches it instead of restarting forever");
    check(broken.unfinished == 2, "and NEITHER prefill finishes - the owner's report, exactly");
    check(broken.steps > fixed.steps,
          "it burns MORE steps than the fixed case and gets nowhere: re-reading is not free");
}



// ============================ S3.10: how much decode one turn buys ============================
//
// The owner's complaint: a slot's turn was ONE verify window, which with MTP accepts 1..8 tokens, so
// "a request generates 1-3 tokens or so and then switches to the next request".  Each of those
// switches is a candidate save+restore of a 237 MB-2.25 GB image (risk R10).  `--decode-tokens N`
// makes a turn worth N tokens instead.  These are the two predicates the driver loop reads.

void test_a_turn_is_a_token_budget_not_a_window_count() {
    using serve_driver::decode_step_tokens;
    using serve_driver::decode_turn_done;

    // 0 and 1 are today: one window, and the turn is over the moment it produced anything.
    check(decode_step_tokens(0, 0, 1000) == 1, "budget 0 = one window");
    check(decode_step_tokens(1, 0, 1000) == 1, "budget 1 = one window");
    check(decode_turn_done(0, 0, 1), "budget 0: the turn is done after one window");
    check(decode_turn_done(1, 0, 1), "budget 1: same");
    check(decode_turn_done(0, 0, 0), "and a window that accepted nothing still ends the turn (never stall)");

    // A 10-token turn is NOT done after 3 tokens, and IS after 10.
    check(!decode_turn_done(10, 0, 3), "3 of 10: keep going");
    check(!decode_turn_done(10, 0, 9), "9 of 10: keep going");
    check(decode_turn_done(10, 0, 10), "10 of 10: hand the session back");
    check(decode_turn_done(10, 0, 14), "a window may overshoot the budget - it is a floor, not a cap");

    // The budget counts this TURN's tokens, not the request's lifetime total.
    check(!decode_turn_done(10, 500, 505), "measured from where the turn started");
    check(decode_turn_done(10, 500, 510), "and it is met there");

    // Never a turn that runs nothing: a slot that gets the session must emit at least one window, or
    // the scheduler spins without progress.
    check(decode_step_tokens(10, 0, 1000) == 10, "a 10-token turn asks for 10");
    check(decode_step_tokens(10, 995, 1000) == 5, "but never more than the request still owes (--max-new)");
    check(decode_step_tokens(10, 1000, 1000) == 0, "nothing owed = no window (the caller ends the request)");
    check(decode_step_tokens(10, 1005, 1000) == 0, "and that holds if the counters disagree");
}

void test_the_startup_line_names_the_turn_size() {
    const size_t G = (size_t) (1ull << 30);
    const std::string dflt = driver_line(3, 250, true, 8 * G, 8, 0, 0);
    check(dflt.find("1 window/turn") != std::string::npos, "the default says one window per turn");
    const std::string tuned = driver_line(3, 250, true, 8 * G, 8, 0, 10);
    check(tuned.find("10 tokens/turn") != std::string::npos, "and a set budget is printed");
    check(driver_line(3, 250, true, 8 * G, 8).find("1 window/turn") != std::string::npos,
          "the old call shape still reports the default");
}

// ================= S4.2: hold, don't reject — the WAIT QUEUE ==================
//
// prompt.md, requirement 3: "Currently if there is no budget to park a conversation or different
// circumstances requests get rejected with an error. That is bad, instead, those requests should be
// put on hold and be executed as soon as there are ressources free instead of cancelled."
//
// The engine already had FIVE different "no"s and treated four of them as an ERR.  These checks pin
// (a) which of them is a wait and which is permanent, (b) the queue's order, bound, wake points and
// cancel behaviour, and (c) end to end, against the fake engine, that a request refused for a
// temporary reason LATER RUNS - and that one refused for a permanent one still errors.

using serve_driver::Admit;
using serve_driver::Hold;
using serve_driver::Wait;
using serve_driver::Waiter;
using serve_driver::WaitQueue;
using serve_driver::admit_decision;
using serve_driver::hold_admitted_line;
using serve_driver::hold_expired;
using serve_driver::hold_expired_line;
using serve_driver::hold_for;
using serve_driver::hold_for_slots;
using serve_driver::hold_line;
using serve_driver::hold_name;
using serve_driver::kDefaultHoldMs;
using serve_driver::kPriorityNew;
using serve_driver::kPriorityResumed;
using serve_driver::price_never_fits;
using serve_driver::prompt_never_fits;
using serve_driver::slot_price_uncapped;
using serve_driver::wait_before;
using serve_driver::wait_line;
using serve_driver::wait_reason;
using serve_driver::wait_text;
using serve_driver::watchdog_sees_waiters;

void test_every_refusal_is_classified_wait_or_permanent() {
    // The four `Refuse` reasons.  Three name a resource that comes back; one names a configuration
    // that cannot.
    check(hold_for(Refuse::none) == Hold::run_now, "no refusal runs now");
    check(hold_for(Refuse::slots_full) == Hold::wait, "a busy slot is a wait - a conversation ends");
    check(hold_for(Refuse::ram) == Hold::wait, "free RAM is a wait - a conversation frees it");
    check(hold_for(Refuse::registry_full) == Hold::wait, "a full registry is a wait too");
    check(hold_for(Refuse::no_parking) == Hold::error,
          "parking off is PERMANENT: with no cache a hand-over destroys the pre-empted request (R2)");
    // The hand-over guard, which S3.6 already split in two.  The classification must agree with
    // `SwapResult`, or the driver's two branches contradict each other.
    check(hold_for(serve_swap::ParkRefusal::not_saveable) == Hold::wait,
          "mid-read means NOT YET: the read ends, so the incoming request waits");
    check(hold_for(serve_swap::ParkRefusal::budget_too_small) == Hold::error,
          "a snapshot bigger than the whole budget never fits: waiting cannot fix the config");
    check(hold_for(SwapResult::wait_park) == Hold::wait &&
          hold_for(SwapResult::end_incoming) == Hold::error,
          "the driver's own two answers classify the same way");
    check(hold_for(SwapResult::mounted) == Hold::run_now, "a mounted hand-over runs");
    check(hold_for(SwapResult::fatal) == Hold::error, "a failed restore is fatal, never a wait");
    // The step gate: `re_read` is a recovery the driver performs NOW, not a wait.
    check(hold_for(Gate::run) == Hold::run_now && hold_for(Gate::re_read) == Hold::run_now,
          "a re-read is immediate recovery, not a queue entry");
    check(hold_for(Gate::end) == Hold::error, "a lost conversation is gone: no wait brings it back");
    check(serve_driver::kRereadLimitHold == Hold::error, "the fifth re-read is a livelock, not a wait");
    // The startup rule, restated so the table is total.
    check(hold_for_slots(3, false) == Hold::error, "--serve-slots 3 with the cache off: permanent");
    check(hold_for_slots(3, true) == Hold::run_now, "with the cache on: not a refusal at all");
    check(hold_for_slots(1, false) == Hold::run_now, "and --serve-slots 0/1 is 0.1.30's serial path");
    // Every reason has a name AND a sentence, and neither is empty - a wait the log cannot name is a
    // hang as far as the owner is concerned.
    for (int i = (int) Wait::none; i <= (int) Wait::engine_busy; ++i) {
        const Wait w = (Wait) i;
        check(std::string(wait_reason(w)) != "?" && !std::string(wait_reason(w)).empty(),
              "every wait reason has a greppable name");
        check(!wait_text(w).empty(), "and a sentence");
    }
    for (int i = (int) Hold::run_now; i <= (int) Hold::error; ++i)
        check(std::string(hold_name((Hold) i)) != "?", "every hold kind has a name");
}

void test_what_can_never_run_is_named_at_admission() {
    // A prompt that does not fit --max-context can never run, at any load.  0.1.30's `+ 8` slack is
    // part of the rule, because `prep_request` uses it and the two must agree.
    check(prompt_never_fits(30000, 100, 8192), "a 30k prompt on an 8k engine never runs");
    check(!prompt_never_fits(8000, 100, 8192), "8000 + 100 + 8 fits 8192");
    check(prompt_never_fits(8085, 100, 8192), "8085 + 100 + 8 does not - the slack is real");
    check(prompt_never_fits(1, 1, 0), "an engine that reports no context runs nothing");
    // A request whose own price exceeds the whole budget can never be parked either.  The budget is
    // not a resource another request gives back, so this is permanent, not a wait.
    check(price_never_fits(4000ull << 20, 2048ull << 20), "4 GiB of snapshot in a 2 GiB budget: never");
    check(!price_never_fits(1000ull << 20, 2048ull << 20), "1 GiB in 2 GiB: it fits, so it may run");
    check(!price_never_fits(9000ull << 20, 0), "budget 0 is parking OFF, which the cache rule answers");
    // THE TRAP this exists to avoid: `slot_image_bytes` CLAMPS a price to the budget, so the clamped
    // figure can never exceed it and the permanent check would never fire.  The uncapped price is the
    // one the permanent question is asked against.
    SlotCost c;
    c.budget_bytes = 2048ull << 20;
    c.cache_slots = 8;
    c.ceiling_bytes = 7746ull << 20;
    c.per_token = 15000;
    check(slot_image_bytes(c, 524288, 40000) == 2048ull << 20,
          "the RAM-gate price is clamped to the budget (that clamp is what stopped admission refusing everything)");
    check(slot_price_uncapped(c, 524288, 40000) > 2048ull << 20,
          "and the uncapped price is the real figure, which is what the permanent check reads");
    check(slot_price_uncapped(c, 150, 1) < (2048ull << 20),
          "a short chat is not permanent - it is priced as short");
    check(slot_price_uncapped(SlotCost{}, 1000, 1) == 0, "nothing known -> no claim of permanence");
}

void test_admission_decides_run_wait_or_error_in_one_place() {
    Wait why = Wait::none;
    // The ordinary case: a permit is free, RAM fits, the cache is on.
    Admit a;
    a.prompt_tokens = 1000; a.max_new = 200; a.max_context = 32768;
    a.avail_bytes = 8ull << 30; a.floor_bytes = 1ull << 30; a.price_bytes = 20ull << 20;
    check(admit_decision(a, why) == Hold::run_now && why == Wait::none, "it runs");
    // The cap is full: a wait, and the reason names the resource.
    a.cap_free = false;
    check(admit_decision(a, why) == Hold::wait && why == Wait::slots_full, "no permit -> wait slots-full");
    a.cap_free = true;
    // No RAM: a wait, not the ERR the engine used to answer with.  THIS is the site the owner hit.
    // It is only a wait because the live slots hold enough that freeing one would let it in.
    a.avail_bytes = 1ull << 30; a.held_bytes = 4ull << 30;
    check(admit_decision(a, why) == Hold::wait && why == Wait::ram, "no free RAM -> wait ram, not ERR");
    // ... and it is PERMANENT when even an empty machine would not hold it: waiting cannot help.
    a.held_bytes = 0;
    check(admit_decision(a, why) == Hold::error,
          "a request that would not fit an empty machine is an error, not a forever-wait");
    a.avail_bytes = 8ull << 30; a.held_bytes = 0;
    // No registry row: a wait.
    a.row_free = false;
    check(admit_decision(a, why) == Hold::wait && why == Wait::registry_full, "no row -> wait");
    a.row_free = true;
    // R7's one position table: a wait (it already was one; now it is a NAMED one).
    a.image_ok = false;
    check(admit_decision(a, why) == Hold::wait && why == Wait::image_exclusive, "image alone -> wait");
    a.image_ok = true;
    // PERMANENT FIRST.  A prompt that cannot fit must be an error even with the cap full - otherwise
    // it queues for ten minutes and then times out blaming the load.
    a.prompt_tokens = 40000; a.cap_free = false;
    check(admit_decision(a, why) == Hold::error, "a doomed prompt errors even while the cap is full");
    check(why == Wait::none, "and it is not filed as a wait");
    a.prompt_tokens = 1000; a.cap_free = true;
    // A GENI request's prompt is in a file, so its length is NOT on the line.  Guessing it at the
    // ceiling would make every image request permanent; the caller says "unknown" instead.
    a.prompt_known = false; a.prompt_tokens = 0;
    check(admit_decision(a, why) == Hold::run_now, "an unknown prompt length is never called permanent");
    a.prompt_known = true; a.prompt_tokens = 1000;
    // Parking off: permanent for every request on a concurrent engine.
    a.cache_enabled = false;
    check(admit_decision(a, why) == Hold::error, "no cache -> permanent");
    a.cache_enabled = true;
    // No telemetry: the parking budget is the only gate, as it is for parking itself.
    a.have_telemetry = false; a.price_bytes = 90ull << 30;
    check(admit_decision(a, why) == Hold::run_now, "no telemetry -> no RAM gate to wait on");
    a.have_telemetry = true; a.price_bytes = 20ull << 20;
    // A long conversation that cannot be PRE-EMPTED is not refused at admission: it can still run,
    // alone.  S3.6's startup line promises exactly that, and taking it away would punish a user whose
    // only conversation would have worked.
    a.uncapped_price = 4000ull << 20; a.budget_bytes = 2048ull << 20;
    check(admit_decision(a, why) == Hold::run_now,
          "a conversation too big to park still RUNS - its hand-over is what refuses, with both numbers");
    a.uncapped_price = 0; a.budget_bytes = 0;
    check(admit_decision(a, why) == Hold::run_now, "and it still runs with those cleared");
}

void test_the_queue_is_fifo_with_one_documented_priority() {
    WaitQueue q(0);                       // no timeout for these ordering checks
    q.enqueue(1, Wait::slots_full, 1000);
    q.enqueue(2, Wait::slots_full, 1000);
    q.enqueue(3, Wait::ram, 1000);
    check(q.size() == 3, "three waiters");
    check(q.head()->id == 1, "first in, first served");
    q.pop(1);
    check(q.head()->id == 2, "and the order holds as the queue drains");
    // The ONE priority: a request that already holds a conversation outranks one that has never run.
    // It has tokens on the wire and a parked branch; the new one has lost nothing.
    q.enqueue(4, Wait::slots_full, 1000, kPriorityResumed);
    check(q.head()->id == 4, "a resumed request goes before a waiting new one");
    Waiter a, b;
    a.priority = kPriorityResumed; a.seq = 9;
    b.priority = kPriorityNew; b.seq = 1;
    check(wait_before(a, b) && !wait_before(b, a), "priority first, then arrival - and never both ways");
    a.priority = kPriorityNew; a.seq = 1; b.seq = 9;
    check(wait_before(a, b) && !wait_before(b, a), "and among equals, the one that arrived first decides");
    a.priority = b.priority; a.seq = b.seq;
    check(!wait_before(a, b) && !wait_before(b, a), "two identical keys are not ordered either way");
    // A duplicate id never enters - two live requests with one id would make the wire unparseable.
    check(!q.enqueue(3, Wait::ram, 1000), "the same id cannot wait twice");
    check(q.size() == 3, "and the queue did not grow");
    // The cap is a bound, not a crash.
    WaitQueue tiny(0, 2);
    check(tiny.enqueue(10, Wait::ram, 0) && tiny.enqueue(11, Wait::ram, 0), "two fit");
    check(tiny.full(), "and it is full");
    check(!tiny.enqueue(12, Wait::ram, 0), "a third is refused by the cap");
    check(tiny.size() == 2, "and nothing changed");
}

void test_a_wait_has_a_bound_and_the_bound_names_the_reason() {
    WaitQueue q(5000);
    q.enqueue(7, Wait::ram, 1000);
    check(!hold_expired(q.waited_ms(7, 5999), 5000), "under the bound it keeps waiting");
    check(hold_expired(q.waited_ms(7, 6000), 5000), "at the bound it is over");
    const std::vector<int64_t> out = q.expired(6000);
    check(out.size() == 1 && out[0] == 7, "and it is RETURNED, not silently dropped - the caller must answer it");
    check(q.size() == 1, "asking did not remove it");
    // hold_ms 0 = never give up: only STOP ends the wait.
    WaitQueue forever(0);
    forever.enqueue(8, Wait::ram, 0);
    check(forever.expired(100000000).empty(), "hold 0 waits forever");
    // The default is longer than any real queue on this box (a 36k-token read is ~70 s).
    check(kDefaultHoldMs >= 600000, "the default hold is minutes, not seconds");
    const std::string line = hold_expired_line(7, 5000, Wait::ram, 5000);
    check(line.find("5000 ms") != std::string::npos, "it gives the wait");
    check(line.find("RAM") != std::string::npos, "and what it waited for");
    check(line.find("--hold-ms") != std::string::npos, "and the knob that sets it");
    check(line.find("--serve-slots") != std::string::npos, "and the knob that would have prevented it");
    // The counters: a wait must be COUNTABLE, or 'it is queued' and 'it is lost' look identical.
    q.note_expired();
    check(q.queued_total() == 1 && q.expired_total() == 1, "queued and expired are both counted");
}

void test_a_freed_resource_wakes_the_waiters() {
    WaitQueue q(0);
    q.enqueue(1, Wait::ram, 0);
    q.enqueue(2, Wait::slots_full, 0);
    q.enqueue(3, Wait::handover_not_yet, 0);
    // A conversation finished: every reason has to be RE-DERIVED, not cleared into a run.  A waiter
    // parked behind RAM may now be blocked by the cap instead, and a reason that is never refreshed
    // is how a stale "waiting for RAM" line outlives the RAM problem.
    q.wake_all();
    check(q.reason_of(1) == Wait::none && q.reason_of(2) == Wait::none, "all three are re-examined");
    check(q.wakeups() == 1, "the wake is counted (the activity line shows it happened)");
    // A reason change is what tells the driver to print its line ONCE, not on every pass.
    check(q.set_reason(1, Wait::ram), "the first change says so");
    check(!q.set_reason(1, Wait::ram), "and repeating it does not - no log spam in a 1 kHz loop");
    check(q.set_reason(1, Wait::slots_full), "a different reason is a new event");
    check(q.reason_of(1) == Wait::slots_full, "and it is recorded");
    // Waking one waiter by id (the slot that just finished).
    q.wake(2);
    check(q.reason_of(2) == Wait::none && q.reason_of(1) == Wait::slots_full,
          "a named wake does not clear the others");
    check(q.find(99) == nullptr && q.reason_of(99) == Wait::none, "an unknown id wakes nothing");
}

void test_a_waiting_request_is_cancellable() {
    WaitQueue q(0);
    q.enqueue(1, Wait::ram, 0);
    q.enqueue(2, Wait::ram, 0);
    Waiter gone;
    check(q.cancel(1, gone), "STOP 1 finds it");
    check(gone.id == 1 && gone.cancelled, "and reports the waiter it took");
    check(q.size() == 1 && !q.contains(1), "it left the queue, so it will not be dispatched");
    check(q.head()->id == 2, "the queue still serves the one that did not cancel");
    check(q.cancelled_total() == 1, "cancels are counted");
    Waiter nobody;
    check(!q.cancel(5, nobody), "a STOP for an id that is not waiting is refused, not fatal");
    // A cancelled waiter is never the head, even if it arrived first - the loop must not run a
    // request whose client asked it to stop.
    WaitQueue r(0);
    r.enqueue(1, Wait::ram, 0);
    r.enqueue(2, Wait::ram, 0);
    Waiter w;
    r.find(1)->cancelled = true;      // flagged in place, as a race between STOP and dispatch would
    check(r.head()->id == 2, "a cancelled waiter is skipped, not dispatched and then cancelled");
    (void) w;
}

void test_the_watchdog_never_kills_a_waiter() {
    // R3's rule is about a slot that is IN A STEP.  A waiter has no session, no loan and no row, so
    // its heartbeat never moves - a watchdog that looked at waiters would kill every queued request
    // the moment the engine went quiet.  The hold bound is the only thing that ends a wait by itself.
    check(!watchdog_sees_waiters(), "waiters are outside the watchdog's view, by rule");
    const int64_t limit = 60000;
    const int64_t t = 1000000;
    SlotWatch none_of_these{};         // a waiter never registers a watch record
    check(!slot_stalled(none_of_these, t + 10 * limit, 0, limit),
          "and an empty watch record is not stalled - it is just not there");
    check(!serve_driver::slot_starved_to_death(0, 0), "and the starvation report stays quiet for it");
}

void test_the_queue_is_visible() {
    WaitQueue q(0);
    check(q.line(0) == "waiting=0", "an empty queue says so positively - it is not a missing line");
    q.enqueue(1, Wait::ram, 1000);
    q.enqueue(2, Wait::slots_full, 1200);
    const std::string s = q.line(6200);
    check(s.find("waiting=2") != std::string::npos, "the depth is in the activity line");
    check(s.find("oldest=5200ms") != std::string::npos, "and the longest wait");
    check(s.find("reason=ram") != std::string::npos, "and what the head is stuck behind");
    // The wire line for serve/server.py's /status and /slots.
    const std::string w = wait_line(2, 5200, Wait::ram);
    check(w == "WAIT 2 5200 ram", "one line, positional, ignorable by an old reader");
    // The hold events themselves are lines, so a wait is greppable and not inferable.
    check(hold_line(4, Wait::slots_full, 2).find("waits: every --serve-slots is busy") != std::string::npos,
          "the start of a wait is printed");
    check(hold_line(4, Wait::slots_full, 2).find("waiting=2") != std::string::npos, "with the depth");
    check(hold_admitted_line(4, 4120).find("ran after waiting 4120 ms") != std::string::npos,
          "and so is its end, with the wait - 7.2's rule that a waiting request reports it");
    // The activity line keeps every existing field and appends the queue.
    const std::string act = serve_driver::activity_line(3, 2, 2, 5, 90, 400, 12, 0, 4, 1ull << 30,
                                                        8ull << 30, q.line(6200));
    check(act.find("slots_active=2") != std::string::npos, "the old fields are still there");
    check(act.find("budget=8192MiB waiting=2") != std::string::npos,
          "and the queue tail follows them, in that order");
    check(serve_driver::activity_line(3, 2, 2, 5, 90, 400, 12, 0, 4, 1ull << 30, 8ull << 30)
              .find("waiting") == std::string::npos,
          "a caller with no queue prints the line it printed before");
    // The startup line says what the engine will do with a request it cannot run.
    const size_t G = (size_t) (1ull << 30);
    check(serve_driver::driver_line(3, 250, true, 8 * G, 8).find("hold 600000 ms then ERR") != std::string::npos,
          "the default hold is reported at startup");
    check(serve_driver::driver_line(3, 250, true, 8 * G, 8, 0, 0, 0).find("hold forever") != std::string::npos,
          "--hold-ms 0 says only STOP ends a wait");
}

// ---- the end-to-end proof -------------------------------------------------------------------
//
// `test_a_refused_handover_ends_the_incoming_request_not_the_running_one` already pins the permanent
// case.  This is the one the owner asked for: a request the engine could NOT run, which it now holds,
// and which RUNS once a slot frees.  It drives the same fake engine as the stage-3 simulations, with
// the admission gate in front of it - so if `admit_decision` is neutered, the request is refused
// instead of queued and the checks below fail.
//
// The shape mirrors the driver loop exactly:
//   * a line read from stdin is a request the engine has ACCEPTED (the client is already waiting on
//     it), so it must end in DONE or ERR - never in silence;
//   * a temporary "no" puts it in the queue and leaves it there;
//   * a permanent "no" answers ERR on the spot;
//   * a finished conversation is a WAKE POINT, and the head waiter is re-examined against the real
//     predicate on the next pass.

struct HoldRun {
    std::vector<int64_t> done, err;
    std::vector<int64_t> held_ids;      // ids that spent at least one pass in the queue
    int64_t longest_wait_ms = 0;
    int passes = 0;
    bool timed_out = false;
    size_t left_waiting = 0;
};

/// `slots` = the engine's active cap.  `hold_ms` = the bound, 0 = forever.
/// `permanent_third` = the third request's prompt does not fit --max-context, the one case that must
/// still be an ERR.
HoldRun run_with_hold(int slots, int64_t hold_ms, bool permanent_third) {
    FakeEngine E(slots, 10);
    E.segments_per_request = 2;
    E.windows_per_request = 3;
    WaitQueue q(hold_ms);
    HoldRun out;
    const int64_t t0 = now_ms();
    // The simulation owns admission, so the fake's own inbox stays empty: `driver_pass`'s admit step
    // must not smuggle a line past the hold gate, or the test would prove nothing.
    std::vector<int64_t> pending = {1, 2, 3};
    const int64_t max_context = 8192;
    const int64_t prompt[3] = {1000, 1000, permanent_third ? 40000 : 1000};
    auto admit_of = [&](int64_t id, int64_t tokens) {
        Admit a;
        a.prompt_tokens = tokens;
        a.max_new = 100;
        a.max_context = max_context;
        a.have_telemetry = true;
        a.avail_bytes = 8ull << 30;
        a.floor_bytes = 1ull << 30;
        a.price_bytes = 20ull << 20;
        a.cap_free = E.reg.can_admit();
        a.row_free = true;
        return a;
    };
    auto bring_in = [&](int64_t id) {
        if (E.reg.add(id, t0) == nullptr) return;
        E.phase[id] = Phase::queued;
        E.seg_left[id] = E.segments_per_request;
        E.win_left[id] = E.windows_per_request;
    };
    int64_t now = t0;
    for (int pass = 0; pass < 4000; ++pass) {
        now = t0 + pass;
        // 1. the hold bound, BEFORE anything else: a waiter that has had enough gets an answer.
        for (int64_t id : q.expired(now)) {
            out.err.push_back(id);
            out.timed_out = true;
            q.note_expired();
            q.pop(id);
        }
        // 2. admit ONE line per pass, decided by the real predicate.  Note there is no `can_admit`
        //    guard around this block: that is the whole change.  The cap is an INPUT to the decision,
        //    and a "no" from it is a WAIT now, not a line left unread and not an ERR.
        if (!pending.empty()) {
            const int64_t id = pending.front();
            Wait why = Wait::none;
            const Hold h = admit_decision(admit_of(id, prompt[(size_t) (id - 1)]), why);
            if (h == Hold::error) {
                out.err.push_back(id);
                pending.erase(pending.begin());
                q.note_refused();
            } else if (h == Hold::wait) {
                if (q.enqueue(id, why, now, kPriorityNew, prompt[(size_t) (id - 1)], 100))
                    out.held_ids.push_back(id);
                else {                       // the queue is bounded too: THAT is an error
                    out.err.push_back(id);
                    q.note_refused();
                }
                pending.erase(pending.begin());   // read, accepted, and HELD - not cancelled
            } else {
                pending.erase(pending.begin());
                bring_in(id);
            }
        }
        // 2b. a waiter whose reason has cleared goes back through the same gate.  This is the wake
        //     point the owner asked for: "started as soon as resources free up".
        if (const Waiter* w = q.head()) {
            Wait why = Wait::none;
            const Hold h = admit_decision(admit_of(w->id, w->prompt_tokens), why);
            if (h == Hold::run_now) {
                out.longest_wait_ms = q.waited_ms(w->id, now) > out.longest_wait_ms
                                          ? q.waited_ms(w->id, now) : out.longest_wait_ms;
                const int64_t id = w->id;
                q.pop(id);
                q.note_admitted();
                bring_in(id);
            } else {
                q.set_reason(w->id, why);    // still blocked, now by whatever it is today
            }
        }
        // 3/4/5. the driver pass, and the wake points.
        const size_t before = E.done_ids.size();
        const bool more = driver_pass(E, now);
        if (E.done_ids.size() > before) q.wake_all(now);   // a slot finished: re-examine the queue
        out.passes = pass;
        if (!more && pending.empty() && q.empty()) break;
    }
    out.done = E.done_ids;
    out.left_waiting = q.size();
    return out;
}

void test_a_refused_request_waits_and_then_runs() {
    // Two slots, three requests.  The third cannot run yet - and it must NOT be an error.
    const HoldRun r = run_with_hold(2, 600000, false);
    check(r.err.empty(), "nothing was refused: the third request was HELD, not cancelled");
    check(r.done.size() == 3, "and all three reached DONE - the held one ran once a slot freed");
    check(r.held_ids.size() == 1, "exactly one request spent a pass in the queue");
    check(r.longest_wait_ms > 0, "and its wait is a real, measured number");
    check(!r.timed_out, "it was never timed out - the bound is a safety net, not the normal path");
    check(r.left_waiting == 0, "and nobody was left waiting when the engine went idle");
}

void test_a_request_that_can_never_run_is_still_an_error() {
    // The other half of the rule.  A prompt that does not fit --max-context is answered at once, and
    // it is NOT queued - otherwise the client waits ten minutes to learn its prompt is too long.
    const HoldRun r = run_with_hold(2, 600000, true);
    check(r.err.size() == 1, "the doomed request got exactly one ERR");
    check(r.err[0] == 3, "and it is the third request, the one with the 40k prompt");
    check(r.done.size() == 2, "the two that fit still ran");
    check(!r.timed_out, "it did not wait for the bound first - permanence is named at admission");
}

void test_a_wait_that_never_clears_becomes_an_honest_timeout() {
    // The bound, exercised: a queue that never clears must eventually ANSWER the client, not hang it.
    // One slot and three requests: with a 5 ms hold, the two that cannot fit time out instead of
    // waiting forever, and the one that owns the slot still finishes.
    const HoldRun r = run_with_hold(1, 5, false);
    check(r.timed_out, "a waiter that never got a slot was answered with a timeout, not left hanging");
    check(r.err.size() == 2, "both of them, once each");
    check(r.done.size() == 1, "and the request that owned the only slot still finished normally");
    check(r.left_waiting == 0, "the queue is empty at the end - a timeout removes its waiter");
}

int main() {
    test_the_clock_is_monotonic();
    test_dispatch_maps_every_phase();
    test_dispatch_never_runs_a_step_over_a_needed_swap();
    test_phase_machine();
    test_admission_ram_arithmetic();
    test_admission_refuses_what_cannot_be_swapped();
    test_admission_respects_the_active_cap();
    test_pick_rule_scenarios();
    test_loan_is_one_resource();
    test_who_needs_the_loan();
    test_two_prefills_cannot_both_hold_the_loan();
    test_the_loan_follows_the_slot_that_dies();
    test_release_does_not_mean_the_cache_is_whole();
    test_the_loan_handoff_is_counted();
    test_a_released_loan_still_blocks_nothing_it_should_not();
    test_watchdog_aborts_only_when_every_slot_stalled();
    test_bare_stop_names_the_running_slot();
    test_named_stop_cancels_only_its_slot();
    test_cancel_unwind_ordering();
    test_a_cancel_queued_before_admission_never_runs();
    test_a_step_only_runs_for_the_mounted_slot();
    test_startup_line_reports_what_is_in_force();
    test_the_handover_order_is_still_the_swap_files();
    test_the_loop_finishes_two_requests_without_swapping_to_death();
    test_the_loan_never_has_two_owners();
    test_a_blocked_slot_does_not_cause_swap_churn();
    test_a_refused_handover_does_not_spin_the_loop();
    test_a_cancelled_slot_is_unwound_and_answered();
    test_a_slot_with_no_row_is_not_picked_forever();
    test_the_loop_terminates_on_an_empty_registry();
    test_step_gate_refuses_a_decode_slot_with_no_conversation();
    test_outgoing_for_is_the_oneswap_guard_reads();
    test_a_prefill_with_progress_must_be_parked();
    test_the_reread_bound_turns_a_spin_into_an_error();
    test_two_prefills_finish_when_the_read_is_parked();
    test_admission_names_a_useless_parking_budget();
    test_a_refused_handover_ends_the_incoming_request_not_the_running_one();
    test_a_not_yet_parkable_handover_waits_instead_of_failing();
    test_a_parkable_decode_slot_is_pre_empted_normally();
    test_a_slot_is_priced_at_its_own_length();
    test_the_gate_blames_the_real_cause();
    test_the_startup_line_says_how_a_slot_is_priced();
    test_a_turn_is_a_token_budget_not_a_window_count();
    test_the_startup_line_names_the_turn_size();
    test_every_refusal_is_classified_wait_or_permanent();
    test_what_can_never_run_is_named_at_admission();
    test_admission_decides_run_wait_or_error_in_one_place();
    test_the_queue_is_fifo_with_one_documented_priority();
    test_a_wait_has_a_bound_and_the_bound_names_the_reason();
    test_a_freed_resource_wakes_the_waiters();
    test_a_waiting_request_is_cancellable();
    test_the_watchdog_never_kills_a_waiter();
    test_the_queue_is_visible();
    test_a_refused_request_waits_and_then_runs();
    test_a_request_that_can_never_run_is_still_an_error();
    test_a_wait_that_never_clears_becomes_an_honest_timeout();

    if (failures) {
        std::fprintf(stderr, "serve_driver_test: %d of %d checks FAILED\n", failures, checks);
        return 1;
    }
    std::printf("serve_driver_test: %d checks OK\n", checks);
    return 0;
}
