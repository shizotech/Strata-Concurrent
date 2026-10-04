// src/program/prefill_loan_test.cpp - the serve loop's prefill fixed-cost decisions (CPU only).
//
// No CUDA, no model, no engine: these are the three predicates the --serve loan path decides on, and the
// bugs they are meant to prevent are all "the decision was wrong", which is checkable without a GPU.
#include "strata/program/prefill_loan.hpp"

#include <cstdio>
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

using namespace strata::program::prefill_loan;

// ------------------------------------------------------------------ lever 1: the sticky loan ---------
void test_relayout_predicate() {
    const LoanLayout base{8192, 5153};
    check(!needs_relayout(base, base), "the same layout needs no relayout");
    check(needs_relayout(base, LoanLayout{4096, 5153}), "a different chunk needs one");
    check(needs_relayout(base, LoanLayout{8192, 5000}), "a different first slot needs one");
    check(needs_relayout(LoanLayout{0, -1}, base), "nothing laid out yet needs one");
    check(needs_relayout(base, LoanLayout{0, -1}), "tearing the layout down is a change too");
}

void test_layout_follows_the_segment() {
    // There is deliberately no "keep the bigger layout for a smaller segment" rule: `part_slots` grows with
    // the chunk, so a wide layout evicts far more rows than a short prompt routes, and the refill of those
    // rows over PCIe dwarfs the two stream syncs a relayout costs.  This models the box's numbers: CUDA0
    // has 7 778 slots, an 8192-token chunk lends 2 625 of them and a 256-token chunk lends 82.
    struct Cache { int64_t slots; };
    const Cache c{7778};
    auto first_for = [&](int64_t chunk) {
        const int64_t k = chunk >= 8192 ? 2625 : chunk >= 4096 ? 1312 : chunk >= 1024 ? 328 : 82;
        return (int32_t) (c.slots - k);
    };
    auto rows_lent = [&](int64_t chunk) { return c.slots - first_for(chunk); };
    check(rows_lent(8192) > 30 * rows_lent(256),
          "keeping an 8192 layout for a 256-token segment would evict and refill 30x more experts");
    check(needs_relayout(LoanLayout{8192, first_for(8192)}, LoanLayout{256, first_for(256)}),
          "so a short segment after a long one does re-lay, and lends the narrow range");
    check(!needs_relayout(LoanLayout{256, first_for(256)}, LoanLayout{256, first_for(256)}),
          "two short segments in a row re-lay nothing - that is the sticky part");
}

void test_loan_grows() {
    // part_slots() is monotone in the chunk, so a bigger chunk lends a range starting no later.
    check(loan_grows(5153, 5153), "the same range is still a superset of itself");
    check(loan_grows(5153, 4096), "a wider range starts earlier");
    check(!loan_grows(5153, 6000), "a narrower range is not a superset: give the loan back first");
    check(!loan_grows(-1, 5153), "nothing was lent yet");
    check(!loan_grows(5153, -1), "nothing may be lent here");
    check(!loan_grows(-1, -1), "two participants that cannot lend do not grow");
}

void test_grow_keeps_the_lent_rows_a_superset() {
    // The scenario the engine walks through: chunk 8192 lends slots [5153, 7778); a 4096-token segment
    // lends [6466, 7778).  Growing from 4096 to 8192 must therefore only ADD rows, never drop one, or a
    // row would be marked non-resident while its slot is no longer covered by the buffers.
    struct Cache { int64_t slots; };
    const Cache c{7778};
    auto first_for = [&](int64_t chunk) {
        // part_slots() grows with the chunk; any monotone stand-in exercises the ordering
        const int64_t k = chunk / 8192 * 2625 + (chunk % 8192) / 4096 * 1312;
        return (int32_t) (c.slots - k);
    };
    const int32_t f4096 = first_for(4096), f8192 = first_for(8192);
    check(f8192 <= f4096, "a bigger chunk lends a range that starts no later");
    check(loan_grows(f4096, f8192), "so growing the loan in place is allowed here");
    // and the rows already lent (>= f4096) are all inside the new range (>= f8192)
    check(f8192 <= f4096, "every row already lent is still covered by the wider range");
}

// ------------------------------------------------------- lever 2: res_upload() only on change ----------
void test_residency_upload() {
    std::vector<int32_t> t{5, 6, -1, 9};
    ResidencyUpload up(4);
    check(up.due(t), "with no shadow yet, an upload is due - never skip against an unknown state");
    up.uploaded(t);
    check(!up.due(t), "the devices hold it: nothing is owed");
    t[2] = 7;
    check(up.due(t), "a host change makes the devices stale");
    t[2] = -1;
    check(!up.due(t), "and changing it back means the copy would move nothing");
    check(up.uploads() == 1, "one copy so far");
    up.uploaded(t);
    check(up.uploads() == 2, "the next real change copies once");
}

void test_residency_upload_lend_refill_round_trip() {
    // The case the lever exists for, in the order the engine now does it.  Rows 0..2 are resident in slots
    // inside the lend range; row 3 is not resident at all.  `lend` marks 0..2 non-resident but does NOT
    // upload - the batched prompt path reads the HOST table, so the devices still hold the pre-lend table.
    // `refill` marks the same rows resident in the same slots, and only then does anything read the device
    // table (the first verify window).  At that moment the host table is byte-identical to the devices':
    // the whole request costs zero uploads instead of two.
    std::vector<int32_t> t{10, 11, 12, -1};
    ResidencyUpload up(4);
    up.synced(t);                       // startup: the devices were loaded with this table
    check(!up.due(t), "nothing owed at startup");
    for (int i = 0; i < 3; ++i) t[i] = -1;              // lend
    up.change();                                        // recorded, not uploaded
    check(up.due(t), "while the loan is out, the devices are stale - and nothing may read them");
    for (int i = 0; i < 3; ++i) t[i] = i + 10;          // refill, same rows, same slots
    check(!up.due(t), "an exact lend/refill round trip owes no upload");
    check(up.uploads() == 0, "and it did not cost one");
}

void test_residency_upload_refill_after_an_adaptive_swap() {
    // A swap during decode moved an expert OUT of a slot the loan then refilled with its original expert:
    // the table is NOT what the devices hold, and the tracker must say so.
    std::vector<int32_t> t{10, 11};
    ResidencyUpload up(2);
    up.synced(t);
    t[0] = -1;                          // the adaptive tier evicted row 0
    up.uploaded(t);
    t[0] = 10;                          // the refill put the original expert back in that slot
    check(up.due(t), "that is a real change: it goes out");
}

void test_residency_upload_never_skips_an_untracked_change() {
    // The tracker's decision is a content comparison, so it is right even if a writer never called
    // change() - which is the property that makes skipping safe.
    std::vector<int32_t> t{1, 2, 3};
    ResidencyUpload up(3);
    up.synced(t);
    t[1] = 99;                          // a write nobody announced
    check(up.due(t), "still detected");
}

void test_residency_upload_empty_table() {
    ResidencyUpload up(0);
    std::vector<int32_t> empty;
    check(up.due(empty), "no shadow: due, even for an empty table");
    up.synced(empty);
    check(!up.due(empty), "shadow recorded");
}

void test_residency_upload_shadow_is_reused_in_place() {
    // The shadow is sized once and rewritten in place (no allocation between decode rounds).  A table that
    // changes size must therefore still be compared correctly, not read past the old length.
    ResidencyUpload up(4);
    std::vector<int32_t> a{1, 2, 3, 4};
    up.synced(a);
    check(!up.due(a), "equal");
    std::vector<int32_t> short_{1, 2};
    check(up.due(short_), "a shorter table is not the one the devices hold");
    up.uploaded(short_);
    check(!up.due(short_), "recorded");
    std::vector<int32_t> longer{1, 2, 9};
    check(up.due(longer), "a longer table is detected too");
    up.uploaded(longer);
    std::vector<int32_t> same{1, 2, 9};
    check(!up.due(same), "and the same content is not");
}

// ------------------------------------------------- lever 4: parking refusal backoff -------------------
void test_park_backoff_refusals() {
    ParkBackoff b{ParkBackoffPolicy{3, 64}};
    check(!b.quiet(0), "the first request asks");
    b.refused(0);
    b.refused(1);
    check(!b.quiet(2), "two refusals is not yet three");
    b.refused(2);
    check(b.quiet(3), "three consecutive refusals quiet parking down");
    check(b.went_quiet(), "and it says so once");
    b.clear_went_quiet();
    check(!b.went_quiet(), "only once");
    check(b.quiet(65), "quiet until request 66");
    check(!b.quiet(66), "then it asks again");
}

void test_park_backoff_success_resets() {
    ParkBackoff b{ParkBackoffPolicy{3, 64}};
    b.refused(0);
    b.refused(1);
    b.parked();                       // a park worked: the machine can take snapshots
    check(!b.quiet(2), "the backoff is over");
    check(b.consecutive() == 0, "and the streak is counted from scratch");
    b.refused(2);
    b.refused(3);
    check(!b.quiet(4), "two fresh refusals are not three");
}

void test_park_backoff_forced_requests_keep_the_streak() {
    // A request that starts a new conversation or mounts a parked one is FORCED: the call site asks
    // regardless of the window.  It must not clear the streak, or on a workload where a new conversation
    // arrives every ~13 requests three consecutive refusals would essentially never happen and the lever
    // would be a no-op.
    ParkBackoff b{ParkBackoffPolicy{3, 64}};
    b.refused(1);
    b.refused(2);
    check(!b.quiet(3), "two refusals is not three");
    b.refused(3);                        // a forced request refused too - it still counted
    check(b.quiet(4), "and the backoff engaged");
    b.reset();                           // only a successful park (or an operator) clears it
    check(!b.quiet(4), "reset() is the explicit clear, used by `parked()`");
}

void test_park_backoff_off() {
    ParkBackoff b{ParkBackoffPolicy{0, 64}};
    for (int i = 0; i < 100; ++i) b.refused(i);
    check(!b.quiet(100), "refusals=0 is today's behaviour: never quiet");
}

void test_park_backoff_never_runs_ahead() {
    // A quiet period must never extend backwards over requests already seen, and repeated refusals during
    // a quiet period (which cannot happen, but the invariant is cheap) must not move the window earlier.
    ParkBackoff b{ParkBackoffPolicy{1, 8}};
    b.refused(100);
    check(b.quiet(101) && b.quiet(107) && !b.quiet(108), "the window is [101, 108)");
    b.refused(101);
    check(b.quiet_until() == 109, "a later refusal moves the window later, never earlier");
}

void test_park_backoff_engages_on_a_chatty_workload() {
    // The shape this box's log has: a forced request (new conversation / mount) every ~13, and RAM-admission
    // refusals everywhere because the parked-prefix budget has collapsed.  `bench/prefill/
    // park_backoff_replay.py` measures the same pattern against the live log and gets ~53 % of the
    // estimates skipped; this is the same state machine on a synthetic version of it.
    ParkBackoff b{ParkBackoffPolicy{3, 64}};
    int paid = 0, skipped = 0;
    for (int64_t req = 1; req <= 400; ++req) {
        const bool forced = (req % 13) == 0;
        if (!forced && b.quiet(req)) { ++skipped; continue; }
        ++paid;
        b.refused(req);
    }
    check(skipped > paid, "the backoff engages even with a forced request every 13");
    check(b.consecutive() >= 3, "and the streak keeps counting through forced requests");
}

// ------------------------------------------------- S3.2b lever 5: the lazy loan ------------------------
//
// The state machine that replaces the end-of-request refill.  The bug class it has to prevent is a single
// one, and it is the worst one in this engine: a row marked RESIDENT whose slot does not hold that expert.
// That is not a crash, it is a plausible token.  So every case below asks one of two questions: "is this row
// marked resident, and who said so, and when?" or "could the pump have aimed at a slot a prompt segment is
// standing in?"

// The engine refers to these through `namespace pfl = strata::program::prefill_loan`; the alias is repeated
// here so the cases read the same way the engine code does.
namespace pfl = strata::program::prefill_loan;
using pfl::LazyRefillPolicy;
using pfl::LoanLedger;
using pfl::LoanRow;

void test_ledger_take_is_idempotent() {
    // The cross-request saving lives here.  A row the previous request left out of the cache is already
    // owned, so the next request's `lend()` does not count it, does not copy it and does not refill it.
    LoanLedger l;
    check(l.take(7, 100), "the first take owns the row");
    check(!l.take(7, 100), "taking it again is not a second debt");
    check(!l.take(7, 101), "and not a second debt with a different slot either");
    check(l.owed() == 1, "one row out");
    check(l.taken() == 1, "counted once");
    check(l.owns(7), "owned");
    check(!l.owns(8), "a row never taken is not owned");
}

void test_ledger_pump_marks_nothing_until_landed() {
    // `pump()` moves rows to in-flight; `landed()` is the ONLY thing that says they are home.  The engine
    // marks `host_res` from `landed()`'s return value, after confirming the copy with an event.
    LoanLedger l;
    for (int32_t i = 0; i < 5; ++i) l.take(i, 200 + i);
    const std::vector<LoanRow> b = l.pump(2);
    check(b.size() == 2, "a bounded batch");
    check(l.owed() == 3 && l.inflight() == 2, "two in flight, three still out");
    check(l.owns(0) && l.owns(2), "in flight is still owned: it is not home yet");
    check(l.returned_pumped() == 0, "nothing has landed yet");
    const std::vector<LoanRow> g = l.landed();
    check(g.size() == 2, "the batch landed");
    check(l.returned_pumped() == 2, "and is counted as returned");
    check(!l.owns(0), "no longer owned: the caller has marked it resident");
    check(l.owns(2), "the rest is still owed");
}

void test_ledger_one_batch_in_flight_at_a_time() {
    // The engine's pump predicate refuses to queue a second batch while one is unconfirmed, because the
    // confirmation is ONE event per cache: a second batch would be confirmed by the same event and marked
    // resident before its bytes arrived.  The ledger itself must make that policy expressible.
    LoanLedger l;
    for (int32_t i = 0; i < 10; ++i) l.take(i, 500 + i);
    check(l.inflight() == 0, "nothing in flight to start");
    l.pump(4);
    check(l.inflight() == 4, "a batch is in flight");
    check(l.pump(0).empty(), "a zero budget queues nothing");
    l.landed();
    check(l.inflight() == 0 && l.owed() == 6, "landed: the rest is still owed");
}

void test_ledger_requeue_after_a_failed_copy() {
    // A copy that could not be queued must NOT leave the row looking resident or looking home.
    LoanLedger l;
    l.take(3, 40);
    l.take(4, 41);
    l.pump(2);
    l.requeue_inflight();
    check(l.inflight() == 0 && l.owed() == 2, "back to owed");
    check(l.owns(3) && l.owns(4), "still owned, so still not resident");
    check(l.returned_pumped() == 0 && l.returned_eager() == 0, "and nothing was returned");
}

void test_ledger_eager_drain_clears_only_what_it_copied() {
    // `refill_one` copies the rows of ONE loan and calls `refilled()` for exactly those.  The rest of the
    // ledger stays owed - a mid-request hand-back must not silently become a full-ledger drain, or turning
    // `STRATA_PREFILL_STICKY_LOAN=0` would cost far more under the lazy rule than it does under the eager
    // one, and the A/B arm would stop measuring what it is supposed to.
    LoanLedger l;
    for (int32_t i = 0; i < 6; ++i) l.take(i, 900 + i);
    l.pump(2);                       // some of it is mid-flight when the eager refill runs
    int64_t copied = 0;
    for (int32_t i = 0; i < 3; ++i) { if (l.refilled(i)) ++copied; }
    check(copied == 3, "three rows reported home");
    check(l.returned_eager() == 3, "and counted as eagerly refilled, not pumped");
    check(l.owed() + l.inflight() == 3, "the other three are still out");
    check(!l.owns(0) && !l.owns(1) && !l.owns(2), "the copied ones are no longer owned");
    check(l.owns(3) && l.owns(4) && l.owns(5), "the rest still are");
    check(!l.refilled(0), "and reporting one home twice is not a thing");
}

void test_ledger_adaptive_tier_never_steals_an_owed_row() {
    // The trap the pump creates for the ADAPTIVE TIER.  `adapt()` promotes non-resident rows into other
    // rows' vacated slots; a row the loan left out looks exactly like a promotion candidate.  If it were
    // promoted, the pump would ALSO copy it into its original slot, and one of the two slots would be
    // orphaned forever (ExpertCache never re-hands a slot out).  `owns()` is the gate, and `drop()` is the
    // belt for the row that became owned after a swap was already queued.
    LoanLedger l;
    l.take(11, 60);
    check(l.owns(11), "adapt() must skip this candidate");
    check(l.drop(11), "apply_pending() found it home by another route");
    check(!l.owns(11), "so the pump will not copy it a second time");
    check(l.owed() == 0, "and it left the owed set");
    check(!l.drop(11), "dropping it twice is not a thing");
    l.take(12, 61);
    l.pump(1);
    check(l.drop(12), "an in-flight row can be dropped too");
    check(l.inflight() == 0 && l.owns(12) == false, "and it leaves both sets");
}

void test_pump_predicate_refuses_a_live_loan() {
    // THE one that keeps (I1) true while a prompt is reading.  A cache whose slots the prompt buffers
    // occupy must never be pumped: the copy would land in a slot that is scratch by the time the segment
    // finishes, and the ledger would then mark it resident.
    LazyRefillPolicy p;
    p.lazy = true;
    p.rows_per_step = 8;
    LoanLedger l;
    l.take(1, 10);
    check(pump_may_run(p, l, false), "no loan live: the pump may walk it home");
    check(!pump_may_run(p, l, true), "a loan IS live: never pump this cache");
    check(!pump_may_run(LazyRefillPolicy{false, 8}, l, false), "eager mode never pumps at all");
    check(!pump_may_run(LazyRefillPolicy{true, 0}, l, false), "rows_per_step=0 means never pump");
    LoanLedger empty;
    check(!pump_may_run(p, empty, false), "nothing owed: nothing to pump");
    l.pump(1);
    check(!pump_may_run(p, l, false), "a batch is already in flight: one at a time per cache");
}

void test_pump_batch_is_bounded_by_the_ledger() {
    LazyRefillPolicy p;
    p.lazy = true;
    p.rows_per_step = 8;
    LoanLedger l;
    check(pump_batch(p, l) == 0, "nothing owed");
    for (int32_t i = 0; i < 3; ++i) l.take(i, 70 + i);
    check(pump_batch(p, l) == 3, "the last batch is what is left, not the whole budget");
    p.rows_per_step = 0;
    check(pump_batch(p, l) == 0, "budget 0 queues nothing");
}

void test_pump_orders_by_routing_usage() {
    // The pump walks the HOTTEST rows home first, so a row decode never routes is never copied at all -
    // that is where the work is removed rather than moved.  `usage` is the same signal `adapt()` uses.
    LoanLedger l;
    for (int32_t i = 0; i < 5; ++i) l.take(i, 800 + i);
    std::vector<float> usage(5, 0.0f);
    usage[4] = 9.0f; usage[1] = 5.0f;
    const std::vector<LoanRow> b = l.pump(2, &usage);
    check(b.size() == 2, "a bounded batch");
    check(b[0].index == 4 && b[1].index == 1, "hottest first");
    check(l.owed() == 3, "the cold rows are still out");
    // and the order survives the next batch, because the sort is paid once per loan, not once per batch
    const std::vector<LoanRow> b2 = l.pump(1, &usage);
    check(b2.size() == 1, "the next batch continues");
    check(l.owed() == 2, "and the ledger shrank by one");
}

void test_pump_without_usage_keeps_insertion_order() {
    // The adaptive tier off means no usage vector.  The pump then walks the ledger in eviction order, which
    // is the honest fallback: it is the order the prompt evicted them, so it is roughly the order the
    // prompt will evict again.
    LoanLedger l;
    for (int32_t i = 0; i < 4; ++i) l.take(i, 300 + i);
    const std::vector<LoanRow> b = l.pump(2, nullptr);
    check(b[0].index == 0 && b[1].index == 1, "insertion order");
    std::vector<float> none;
    LoanLedger l2;
    for (int32_t i = 0; i < 3; ++i) l2.take(i, 1);
    const std::vector<LoanRow> b2 = l2.pump(2, &none);   // an empty vector: the tier is off
    check(b2[0].index == 0 && b2[1].index == 1, "an empty usage vector is the same as none");
}

// ---- THE ADVERSARIAL CASE the contract asks for: a row is marked non-resident and a decode window asks
// for it.  Modelled end to end, because the answer is a behaviour, not a comment.
void test_decode_window_never_reads_a_slot_that_does_not_hold_its_expert() {
    // A 4-row cache, slots 0..3, all resident.  The prompt lends slots >= 2, returns the loan LAZILY, and
    // the pump has not got the rows home yet.  Then a decode window routes all four experts.
    //
    // The engine's decode path asks exactly one question per routed expert - `host_res[i] >= 0`
    // (expert_source.cpp:1040 for the window's GPU/CPU split, :1098 for the fallback, and the device copy
    // `d_res` for `Verifier::resident_plan`) - and sends a non-resident row to the CPU pool.  So the test
    // is: does the table say the truth, and does the truth route the row away from the slot?
    std::vector<int32_t> host_res{0, 1, 2, 3};
    std::vector<int32_t> slot_holds{0, 1, 2, 3};   // which expert each slot's BYTES are
    LoanLedger led;
    LazyRefillPolicy p;
    p.lazy = true;
    p.rows_per_step = 2;

    // lend: rows in the range >= 2 stop being resident, and the ledger owns them.  Marked BEFORE the slot
    // is handed over, which is the direction that is always safe.
    for (int32_t i = 0; i < 4; ++i)
        if (host_res[i] >= 2) { led.take(i, host_res[i]); host_res[i] = -1; }
    // the prompt writes scratch into slots 2 and 3
    slot_holds[2] = -9; slot_holds[3] = -9;
    check(host_res[2] == -1 && host_res[3] == -1, "the lent rows are non-resident while the prompt runs");

    // finish_prefill: the LAZY return.  Nothing copied, nothing marked resident.
    check(led.owed() == 2, "two rows are out of the cache");

    // The window asks for every expert.  The rule the engine applies (`expert_source.cpp:1040` for the
    // window's GPU/CPU split, `Verifier::resident_plan` for the device copy): a row goes to the GPU only if
    // the table says it is resident - and then its slot must actually hold it.  A non-resident row goes to
    // the CPU pool, which is correct and is what every expert the cache never held already does.
    int to_gpu = 0, to_cpu = 0, broken = 0;
    for (int32_t i = 0; i < 4; ++i) {
        const bool resident = host_res[i] >= 0;
        if (!resident) { ++to_cpu; continue; }
        ++to_gpu;
        if (slot_holds[host_res[i]] != i) ++broken;
    }
    check(broken == 0, "no window row reads a slot that does not hold its expert");
    check(to_gpu == 2 && to_cpu == 2, "the two returned rows are CPU-side, the two untouched ones GPU-side");
    check(led.owns(2) && led.owns(3), "and the ledger still owns the CPU-side pair, so they will come home");

    // The pump walks one batch home.  The engine queues the copies, confirms them with an event, and only
    // then marks them - so the marking is what the test models.
    std::vector<LoanRow> b = led.pump(pump_batch(p, led));
    check(b.size() == 2, "one bounded batch");
    for (const LoanRow& r : b) {
        slot_holds[r.slot] = r.index;      // the copy landed (the engine confirmed it first)
        host_res[r.index] = r.slot;
    }
    led.landed();
    for (int32_t i = 0; i < 4; ++i)
        check(host_res[i] >= 0 && slot_holds[host_res[i]] == i, "the whole cache is true again");

    // And the dangerous direction, spelled out: if the engine marked a row resident BEFORE its copy landed,
    // this is exactly the check that would fire.
    std::vector<int32_t> lie = host_res;
    std::vector<int32_t> lie_slot{0, 1, -9, -9};
    int caught = 0;
    for (int32_t i = 0; i < 4; ++i)
        if (lie[i] >= 0 && lie_slot[lie[i]] != i) ++caught;
    check(caught == 2, "a row marked resident over a scratch slot IS detectable - and is the one thing the "
                       "ordering in settle_pump/refill_one makes unreachable");
}

void test_lazy_loan_survives_into_the_next_request() {
    // (C), the shared loan, as an emergent property of (A): because nothing was copied back, the second
    // request's lend finds its range already non-resident and takes on nothing new.  This is the 4.95 GiB
    // that stops being spent.
    std::vector<int32_t> host_res{0, 1, 2, 3, 4, 5};
    LoanLedger led;
    auto lend = [&](int32_t first) {
        int64_t took = 0;
        for (int32_t i = 0; i < 6; ++i)
            if (host_res[i] >= first) { led.take(i, host_res[i]); host_res[i] = -1; ++took; }
        return took;
    };
    const int64_t first_request = lend(3);
    check(first_request == 3, "the first request evicts three rows");
    // finish_prefill returns the loan lazily: no copy, no mark-back
    check(led.owed() == 3, "and they stay out");
    const int64_t second_request = lend(3);
    check(second_request == 0, "the second request evicts NOTHING - the loan is still lying in the cache");
    check(led.owed() == 3, "and the ledger still owns the same three rows, counted once");
    // a wider range only adds the new rows
    const int64_t third = lend(2);
    check(third == 1, "a wider loan adds only the row it newly covers");
    check(led.owed() == 4, "four out");
}

void test_lazy_lend_must_remark_and_eager_may_short_circuit() {
    // The rule that makes the sticky short-circuit UNSOUND once rows can come back on their own.  A pump
    // batch may have returned rows since the last segment; skipping the marking loop would then carve
    // prompt buffers into a slot the devices believe holds an expert.
    check(lend_must_remark(LazyRefillPolicy{true, 8}), "lazy: always re-mark");
    check(!lend_must_remark(LazyRefillPolicy{false, 0}), "eager: 0.1.30's short-circuit still applies");
    check(narrow_lend_refills_first(LazyRefillPolicy{false, 0}), "eager: a narrower range hands the loan back");
    check(!narrow_lend_refills_first(LazyRefillPolicy{true, 8}), "lazy: it just stays in the ledger");
}

void test_may_mark_resident_only_after_confirmation() {
    check(may_mark_resident(true), "a confirmed-landed copy may be marked");
    check(!may_mark_resident(false), "anything else may not - this is the whole of invariant (I1)");
}

}  // namespace

int main() {
    test_relayout_predicate();
    test_layout_follows_the_segment();
    test_loan_grows();
    test_grow_keeps_the_lent_rows_a_superset();
    test_residency_upload();
    test_residency_upload_lend_refill_round_trip();
    test_residency_upload_refill_after_an_adaptive_swap();
    test_residency_upload_never_skips_an_untracked_change();
    test_residency_upload_empty_table();
    test_residency_upload_shadow_is_reused_in_place();
    test_park_backoff_refusals();
    test_park_backoff_success_resets();
    test_park_backoff_forced_requests_keep_the_streak();
    test_park_backoff_off();
    test_park_backoff_never_runs_ahead();
    test_park_backoff_engages_on_a_chatty_workload();
    // S3.2b
    test_ledger_take_is_idempotent();
    test_ledger_pump_marks_nothing_until_landed();
    test_ledger_one_batch_in_flight_at_a_time();
    test_ledger_requeue_after_a_failed_copy();
    test_ledger_eager_drain_clears_only_what_it_copied();
    test_ledger_adaptive_tier_never_steals_an_owed_row();
    test_pump_predicate_refuses_a_live_loan();
    test_pump_batch_is_bounded_by_the_ledger();
    test_pump_orders_by_routing_usage();
    test_pump_without_usage_keeps_insertion_order();
    test_decode_window_never_reads_a_slot_that_does_not_hold_its_expert();
    test_lazy_loan_survives_into_the_next_request();
    test_lazy_lend_must_remark_and_eager_may_short_circuit();
    test_may_mark_resident_only_after_confirmation();

    if (failures) {
        std::fprintf(stderr, "prefill_loan_test: %d of %d checks FAILED\n", failures, checks);
        return 1;
    }
    std::printf("prefill_loan_test: %d checks OK\n", checks);
    return 0;
}
