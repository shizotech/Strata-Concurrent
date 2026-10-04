// include/strata/program/prefill_loan.hpp - the serve loop's PREFILL FIXED-COST levers, as pure logic.
//
// `--serve` pays a large per-REQUEST cost reading a prompt: measured on this box over 1 600 live requests,
// `prompt_ms ~= 9 500 (fixed) + 2.7 * fresh_tokens`, and the fixed term does NOT scale with the cached
// context (a 500-token read costs ~10 s at 5 k context and ~10 s at 250 k).  So it is overhead around the
// read, not the read.  The overhead is the prompt path's LOAN: the expert-cache slots it borrows for its
// chunk buffers, the experts that stops holding, the ones that have to stream back over PCIe afterwards,
// and the residency table that describes all of it to the devices.  See `bench/prefill/README.md` for the
// measurement and `bench/prefill/fixed-cost-changes.md` for what these levers change.
//
// Everything here is a predicate over plain integers, with no CUDA and no engine state, so it can be unit
// tested on a CPU (`src/program/prefill_loan_test.cpp`).  The engine in `src/program/generate.cpp` owns the
// state; these functions own the DECISIONS, and the decisions are the part that can be wrong.
//
// THE INVARIANT EVERY LEVER OBEYS: the set of experts resident during a decode window must be exactly what
// it is today.  Expert placement changes rounding, so a change there is a correctness change, not a
// performance one.  A lent slot must hold its expert again before any window reads it.
#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace strata::program::prefill_loan {

// ---------------------------------------------------------------- LEVER 1: the sticky loan ---------------
//
// The prompt path lays its chunk buffers out in the TAIL of one cache (`Prefill::relayout`), and the rows
// it covers stop being experts until they are refilled.  `lend()` runs per prompt segment and `refill()`
// runs at the end of every request, so a request that splits at a turn boundary can hand the loan back and
// take it again inside itself.

/// Where one prompt path's buffers are: the chunk they are carved for, and the first cache slot they cover.
/// `first < 0` = this participant never lends (its cache cannot, or the loan was refused at startup).
struct LoanLayout {
    int64_t chunk = 0;
    int32_t first = -1;
};

/// Does moving from the layout in place (`have`) to `want` need `Prefill::relayout`?  The carve is a
/// function of exactly these two numbers, so anything else about the request is irrelevant here.
inline bool needs_relayout(const LoanLayout& have, const LoanLayout& want) {
    return have.chunk != want.chunk || have.first != want.first;
}

// WHY THERE IS NO "KEEP THE BIGGER LAYOUT" RULE HERE.  `Prefill::run()` walks a segment in chunks of
// `min(m.T, remaining)`, so a layout carved for 8192 tokens would read a 256-token segment as one 256-token
// chunk - the same single chunk, the same kernel shapes, the same bits - and `relayout` could be skipped.
// It is still the wrong trade: the layout's chunk also decides how wide the slot range is (`part_slots`),
// and a wide range evicts far more experts than a short prompt routes.  A 256-token segment kept on an
// 8192-token layout would evict 2 625 of CUDA0's slots instead of 82 and then refill 2 625 of them over
// PCIe.  `relayout` is two stream syncs and a bump-pointer carve; the refill is gigabytes.  So the layout
// follows the segment, and only the parts that did not change are re-done.

/// Is the range `first_new..slots` a superset of the one already lent at `first_now`?  `part_slots` is
/// monotone in the chunk, so a BIGGER chunk always lends a range that starts no later: the rows already
/// lent are still lent, and only the rows the new range adds have to be taken.  That is the whole point of
/// growing a loan in place instead of refilling it and re-lending it.
inline bool loan_grows(int32_t first_now, int32_t first_new) {
    return first_now >= 0 && first_new >= 0 && first_new <= first_now;
}

// ------------------------------------------------------- LEVER 2: res_upload() on change ----------------
//
// The residency table is `n_layers * n_expert` int32 (24 576 entries = 96 KiB) and it lives on every
// device.  Today it is re-uploaded on every lend and every refill even when the lend took back exactly what
// it had just given.  The upload is synchronous and it is per device, so on a layer split it is a round
// trip on every link - including the 1.3 GB/s one.
//
// Skipping it is safe exactly when every device's copy is already byte-identical to the host table, and
// `res_upload()` always uploads the WHOLE table to EVERY device - so the invariant "every device holds
// `shadow_`" is one the tracker can maintain with one copy of the table, and `due()` can answer it by
// comparing content.
//
// Comparing content rather than trusting a "something changed" flag is deliberate.  A flag is only as good
// as the discipline of every writer; a comparison is right even if a writer forgot to say anything, and
// 96 KiB of `operator==` is a few microseconds against a synchronous 96 KiB copy per device.  The
// `change()` counters are for the report, not for the decision.
//
// It starts with no shadow: the devices were loaded with the table at startup, but the first decision after
// that must not be able to skip against an empty shadow.

class ResidencyUpload {
public:
    ResidencyUpload() = default;
    /// `n_layers` is not needed for the decision (the table is compared whole); it is kept so a caller can
    /// state what it is tracking, and so a future per-device form has the geometry to hand.
    explicit ResidencyUpload(int64_t n_layers) : n_layers_(n_layers) {}

    /// A host row changed.  Reporting only - `due()` does not depend on this being called, which is why
    /// the decision cannot be fooled by a writer that forgot to announce itself.
    void change() { ++changes_; }

    /// Does an upload have to happen?  False: every device already holds exactly `host`.
    bool due(const std::vector<int32_t>& host) const {
        return !shadow_valid_ || host.size() != shadow_.size() ||
               !std::equal(host.begin(), host.end(), shadow_.begin());
    }
    /// The devices now hold `host` (the caller has just copied it).
    void uploaded(const std::vector<int32_t>& host) {
        remember(host);
        ++uploads_;
    }
    /// Record the same fact without counting a copy, for the startup path where the table went out with the
    /// allocation that created it.
    void synced(const std::vector<int32_t>& host) { remember(host); }

    int64_t uploads() const { return uploads_; }
    int64_t changes() const { return changes_; }
    /// Decisions that turned out to need no copy, for the report.
    int64_t skipped() const { return skipped_; }
    void note_skipped() { ++skipped_; }
    int64_t layers() const { return n_layers_; }

private:
    /// The shadow is sized once and then written in place: `apply_pending()` calls this between decode
    /// rounds, and the engine's rule is no new per-token allocations on the decode path.
    void remember(const std::vector<int32_t>& host) {
        if (shadow_.size() != host.size()) shadow_.resize(host.size());
        std::copy(host.begin(), host.end(), shadow_.begin());
        shadow_valid_ = true;
    }
    int64_t n_layers_ = 0;
    std::vector<int32_t> shadow_;
    bool shadow_valid_ = false;
    int64_t uploads_ = 0, changes_ = 0, skipped_ = 0;
};

// ------------------------------ LEVER 5 (stage 3 S3.2b): the loan is returned LAZILY -------------------
//
// S0.3 left one number on the table and named it: the end-of-request refill.  `refill()` streams every row
// the prompt path evicted back from the arena into the same cache slot - on this box 2 625 + 1 971 + 1 885
// = 6 481 slots, 4.95 GiB per stage, over links measured at 2.9 / 1.3 / 24.5 GB/s, so CUDA1's share alone
// is ~3.8 s - and `--serve` prints `prompt_ms` AFTER it, so all of it is inside the fitted 9.5 s fixed
// term.  S0.3 could not remove it because its contract forbade changing what is refilled or when, and
// because a single-streaming request has to hand the cache back before its own decode loop.
//
// Stage 3's driver changes that, and the reason is not "the engine is busy".  It is that **`kNotResident`
// is a legal state the decode path already handles.**  A verify window routes an expert, looks the row up,
// and if it is not resident it goes to the CPU pool (`expert_source.cpp:929, 1098-1099` build the hit/miss
// `kind[]` from exactly that table).  Nothing in decode REQUIRES the cache to be whole; it only requires
// the table to be TRUE.  So the refill is not a correctness requirement, it is a hit-rate choice, and a
// hit-rate choice can be deferred, spread, or skipped.
//
// THE THREE MOVES, and which one removes work rather than hiding it:
//
//   (A) LAZY RETURN.  `finish_prefill()` does not copy anything back.  It leaves the rows marked
//       non-resident and records them in a per-cache LEDGER.  Decode runs the coldest tail on the CPU.
//   (C) THE LOAN SURVIVES THE REQUEST.  Because nothing was copied back, the layout (`first_now`,
//       `sp->chunk()`) and the ledger are still standing when the NEXT request prefills, so its `lend()`
//       re-marks almost nothing and re-lays nothing.  Two slots that both want to prefill share one loan
//       layout for the same reason.  (A) is what makes (C) possible; (C) is where the 4.95 GiB actually
//       goes away.
//   (B) THE PUMP.  The ledger is drained in bounded batches between decode steps, on its own non-blocking
//       stream, so a long decode gets its hit rate back without ever putting a copy on a prompt's critical
//       path.  (B) hides work; (A)+(C) remove it.  All three are here because the box needs both: the
//       fixed cost removed, and the cold tail eventually back in VRAM.
//
// THE ONE INVARIANT, and it is the only thing standing between this and plausible garbage:
//
//     (I1)  `host_res[i] >= 0`  <=>  the cache slot `host_res[i]` on the owning device holds expert `i`'s
//           bytes RIGHT NOW.
//     (I2)  at the start of every verify window, every device's `d_res` equals `host_res`.
//
// (I1) is enforced by ORDER, in one direction only: a row is marked non-resident BEFORE its slot is handed
// to the prompt path, and it is marked resident only AFTER its copy has been confirmed landed.  There is no
// window in which a row is marked resident and its slot holds scratch.  (I2) is enforced by
// `reconcile_residency()` running at the top of every window path.
//
// THE TRAP THIS CREATES, and it is the one a reader must not miss.  Once rows can come back on their own
// schedule, `lend()` may no longer short-circuit with "my current loan already covers this chunk": the
// pump may have returned some of those rows to the cache in the meantime, and skipping the marking loop
// would leave the device told that a slot holds an expert while the prompt buffers overwrite it.  So under
// the lazy rule `lend()` ALWAYS re-marks its range.  That is 24 576 int32 comparisons per participant per
// segment - microseconds against 3.8 s.

/// One row a cache lent to the prompt path and has not got back: the residency index (`layer * n_expert +
/// expert`) and the slot it must return to.  The slot is remembered, never re-derived: `ExpertCache` does
/// not evict and never re-hands-out a slot, so a ledger slot cannot belong to anybody else - but it also
/// cannot be found again from `host_res`, which says `kNotResident` for exactly these rows.
struct LoanRow {
    int32_t index = -1;   ///< the residency-table index
    int32_t slot = -1;    ///< the cache slot the expert belongs in
};

/// The per-cache ledger: which rows are OUT, which are copied-and-not-yet-confirmed, and the counters the
/// report reads.  One per `PfPart`, so a layer split refills each stage through its own cache.
///
/// It deliberately holds no CUDA and reads no clock: the engine decides WHEN a batch is queued and WHEN the
/// copy is confirmed, and tells this object the result.  That is what makes the state machine testable, and
/// the state machine is the part that can be wrong.
class LoanLedger {
public:
    /// The prompt path took this row out.  Idempotent: a row already owned is not owned twice, which is what
    /// keeps the ledger free of duplicates when a loan is grown, or re-taken after the pump returned part of
    /// it, or taken again by the NEXT request over a range that is still out.  Returns false if it was
    /// already owned - and that false is the whole cross-request saving: a row that has been out since the
    /// previous request is not counted, not copied, and not refilled again.
    bool take(int32_t index, int32_t slot) {
        if (owns(index)) return false;
        owed_.push_back(LoanRow{index, slot});
        mark(index);
        ordered_ = false;      // a new row has to be merged into the usage order on the next pump
        ++taken_;
        return true;
    }
    /// The row is owned by the ledger - owed or in flight - so something is already going to put it home.
    ///
    /// This is the gate the ADAPTIVE TIER has to respect.  `adapt()` treats every non-resident row with
    /// enough usage as a swap candidate, so a row the loan left out is exactly the kind of row it wants to
    /// promote - into some other row's vacated slot.  If it did, two owners would be moving the same expert:
    /// the pump would copy it into its ORIGINAL slot while `apply_pending()` marked it resident in the new
    /// one, and whichever won the race would orphan the other slot forever (`ExpertCache` never re-hands a
    /// slot out).  Both copies hold the right bytes, so that is a stranded slot and an unexplainable hit
    /// rate rather than wrong tokens - which is precisely the kind of drift this project keeps paying for.
    /// The pump is also strictly the better choice: it needs no victim, because the slot is already its own.
    ///
    /// O(1) by construction, because `adapt()` asks about all 24 576 rows every `--adapt-every` rounds and a
    /// linear scan of a 6 481-row ledger would make that quadratic.
    bool owns(int32_t index) const {
        return index >= 0 && (size_t) index < mark_.size() && mark_[(size_t) index] != 0;
    }

    size_t owed() const { return owed_.size(); }
    size_t inflight() const { return inflight_.size(); }
    bool empty() const { return owed_.empty() && inflight_.empty(); }

    /// Move up to `max` owed rows into the in-flight set: the engine has queued their copies and must
    /// confirm them before marking anything resident.  Returns the rows it took, so the caller can queue
    /// exactly those copies.
    ///
    /// WHICH rows, and why that is where the work is REMOVED rather than moved.  `usage` is the engine's
    /// per-row routing signal (`ExpertDrive::usage`, the same one `adapt()` promotes on): the ledger is
    /// ordered hottest-first, so the pump walks the rows decode actually routes home first, and a row decode
    /// never routes is never copied at all - the bytes it would have cost over PCIe are simply not spent.
    /// With no signal (the adaptive tier off, so the vector is empty) the ledger keeps insertion order, which
    /// is eviction order.
    ///
    /// The sort runs ONCE per loan, not once per batch: `take()` clears the flag whenever it adds a row,
    /// `pump()` pays it on the first batch after that.  `usage` decays by a constant factor every
    /// `--adapt-every` rounds, which cannot reorder anything, so re-sorting per batch would be the pump's own
    /// cost for no benefit.
    std::vector<LoanRow> pump(size_t max, const std::vector<float>* usage = nullptr) {
        std::vector<LoanRow> out;
        if (max == 0 || owed_.empty()) return out;
        if (!ordered_ && usage != nullptr && !usage->empty()) {
            std::stable_sort(owed_.begin(), owed_.end(), [&](const LoanRow& a, const LoanRow& b) {
                const float ua = (size_t) a.index < usage->size() ? (*usage)[(size_t) a.index] : 0.0f;
                const float ub = (size_t) b.index < usage->size() ? (*usage)[(size_t) b.index] : 0.0f;
                return ua > ub;
            });
            ordered_ = true;
        }
        const size_t n = max < owed_.size() ? max : owed_.size();
        out.assign(owed_.begin(), owed_.begin() + (std::ptrdiff_t) n);
        inflight_.insert(inflight_.end(), out.begin(), out.end());
        owed_.erase(owed_.begin(), owed_.begin() + (std::ptrdiff_t) n);
        return out;
    }
    /// The in-flight copies have been confirmed landed.  Those rows are back in the cache; the caller sets
    /// `host_res[index] = slot` for each, in the same order, and only then may a window read them.
    std::vector<LoanRow> landed() {
        std::vector<LoanRow> out;
        out.swap(inflight_);
        for (const LoanRow& r : out) unmark(r.index);
        returned_pumped_ += (int64_t) out.size();
        return out;
    }
    /// A batch that was queued but must be abandoned (a copy failed): the rows go back to the owed set, so
    /// a later attempt - or an eager refill - still puts them home.  They are NOT resident.
    void requeue_inflight() {
        owed_.insert(owed_.end(), inflight_.begin(), inflight_.end());
        inflight_.clear();
    }

    /// The row is home by ANOTHER route, so the ledger must stop owning it.  `adapt()` asks `owns()` first
    /// and never selects such a row, so this is the belt to that braces: it can only fire for a row that
    /// became owned after a swap was already queued.  Returns true if the row was owned.
    bool drop(int32_t index) { return erase_row(index, dropped_); }

    /// The row is home because an EAGER refill (`refill_one`) copied it back.  Same bookkeeping as `drop()`,
    /// a different counter, because "the adaptive tier promoted it" and "a mid-request hand-back refilled
    /// it" are different events the report has to tell apart.
    bool refilled(int32_t index) { return erase_row(index, returned_eager_); }

    int64_t taken() const { return taken_; }
    int64_t returned_eager() const { return returned_eager_; }
    int64_t returned_pumped() const { return returned_pumped_; }
    int64_t dropped() const { return dropped_; }

private:
    bool erase_row(int32_t index, int64_t& counter) {
        for (size_t k = 0; k < owed_.size(); ++k)
            if (owed_[k].index == index) {
                owed_.erase(owed_.begin() + (std::ptrdiff_t) k);
                unmark(index);
                ++counter;
                return true;
            }
        for (size_t k = 0; k < inflight_.size(); ++k)
            if (inflight_[k].index == index) {
                inflight_.erase(inflight_.begin() + (std::ptrdiff_t) k);
                unmark(index);
                ++counter;
                return true;
            }
        return false;
    }
    void mark(int32_t index) {
        if (index < 0) return;
        // `insert`, not `resize`: GCC 16 raises a false -Wstringop-overflow on a grow-by-one `resize` of a
        // uint8 vector when the size is a runtime value.
        if ((size_t) index >= mark_.size())
            mark_.insert(mark_.end(), (size_t) index + 1 - mark_.size(), 0);
        mark_[(size_t) index] = 1;
    }
    void unmark(int32_t index) {
        if (index >= 0 && (size_t) index < mark_.size()) mark_[(size_t) index] = 0;
    }
    std::vector<LoanRow> owed_;
    std::vector<LoanRow> inflight_;
    /// One byte per residency index: owned, however many rows are out.  Sized on first use to the highest
    /// index ever taken, so on a 48 x 512 model it is 24 KiB per cache - against the 4.95 GiB it tracks.
    std::vector<uint8_t> mark_;
    /// Is `owed_` in hottest-first order?  Cleared by `take()`, paid by the next `pump(usage)`.
    bool ordered_ = false;
    int64_t taken_ = 0, returned_eager_ = 0, returned_pumped_ = 0, dropped_ = 0;
};

/// The two knobs of lever 5, read once at startup like every other `STRATA_*` value.
struct LazyRefillPolicy {
    /// Return the loan lazily instead of streaming it back at the end of every prompt.
    bool lazy = true;
    /// Rows per cache per decode step the pump may have in flight.  0 = never pump: the ledger is only
    /// drained when something needs the cache whole (an eager refill, `STRATA_PREFILL_LAZY_LOAN=0`).
    int64_t rows_per_step = 8;
};

/// May the pump queue a batch for this cache right now?  Three conditions, all load-bearing:
///   * lazy mode is on (otherwise the ledger is always empty by the time decode starts);
///   * a budget exists;
///   * nothing is in flight - one batch per cache, because the confirmation is one event per cache and a
///     second batch queued before the first landed would be confirmed by the same event and marked
///     resident before its bytes arrived, which is exactly the silent-garbage failure mode;
///   * and NO batched prompt segment is live.  Its buffers ARE these slots: a copy landing inside a live
///     loan's range is overwritten by the segment that is still using them, and the next `lend()` would
///     re-mark the row only after the damage.  `settle_pump(wait = true)` at the top of `lend()` is what
///     makes the ordering safe; this predicate is what makes it unnecessary in the first place.
inline bool pump_may_run(const LazyRefillPolicy& p, const LoanLedger& led, bool loan_live) {
    return p.lazy && p.rows_per_step > 0 && !loan_live && led.inflight() == 0 && led.owed() > 0;
}

/// How many rows this step.  Bounded by the budget AND by what is owed, so the last batch of a ledger is
/// not a partial queue of nothing.
inline size_t pump_batch(const LazyRefillPolicy& p, const LoanLedger& led) {
    if (!p.lazy || p.rows_per_step <= 0) return 0;
    return (size_t) std::min<int64_t>(p.rows_per_step, (int64_t) led.owed());
}

/// Must `lend()` re-mark its range even when its current loan already covers the chunk it wants?  Under the
/// lazy rule YES, always: the pump may have returned rows since the last segment, and a row the pump
/// returned is resident again, so skipping the marking loop would hand the prompt buffers a slot the
/// devices believe holds an expert.  (Eager mode never has returned rows outstanding, so its short-circuit
/// is still sound and still taken.)
inline bool lend_must_remark(const LazyRefillPolicy& p) { return p.lazy; }

/// Does a segment that wants a NARROWER range than the loan in place have to hand its loan back first?
/// Eager mode: yes, that is 0.1.30.  Lazy mode: no - the rows outside the new range are already
/// non-resident and already in the ledger, so they stay there and the pump takes them home later.  The
/// relayout still happens (the buffers move), and the marking loop still runs.
inline bool narrow_lend_refills_first(const LazyRefillPolicy& p) { return !p.lazy; }

/// The one question the adversarial case asks: a decode window routes an expert whose row is OUT.  What is
/// allowed to happen?  It runs on the CPU.  That is not a fallback, it is the ordinary state of every
/// expert the cache has never held, and it is why the ledger can be drained at whatever rate the machine
/// can afford.  What is NEVER allowed is the other direction - a row marked resident whose slot does not
/// hold it - which is why `may_mark_resident()` exists as a named gate rather than a comment.
inline bool may_mark_resident(bool copy_confirmed_landed) { return copy_confirmed_landed; }

// ------------------------------------------------- LEVER 4: parking refusal backoff ---------------------
//
// `park_current()` runs whenever a request does not continue from the live session - which in a chat is
// almost every request, because the request resumes from a CHECKPOINT at the turn boundary.  Before it can
// say "no room", it validates the whole view, walks every retained checkpoint, estimates the snapshot and
// reads the RAM telemetry.  On a box whose parked-prefix budget has collapsed to 0.0 GiB that answer is the
// same every time, and it is paid every time.
//
// So: after N consecutive refusals caused by the physical-RAM admission floor, stop asking for a while.
// "A while" is a request count, not forever, and a park that worked clears it immediately.
//
// A request that STARTS a conversation or MOUNTS a parked one is different: parking is exactly what saves
// that request's re-read, so it asks regardless of the window (the call site passes `force`).  It does NOT
// clear the streak.  That distinction is the difference between a lever and a no-op: on this box's log
// roughly one request in thirteen starts a new conversation, so clearing the streak on those would mean
// three consecutive refusals essentially never happen.  `bench/prefill/park_backoff_replay.py` measures the
// pattern against the live log: 53 % of the estimates never have to be paid.

struct ParkBackoffPolicy {
    /// Consecutive RAM-admission refusals that quiet parking down.  0 = never (today's behaviour).
    int64_t refusals = 3;
    /// How many requests parking stays quiet for once it does.
    int64_t quiet_requests = 64;
};

class ParkBackoff {
public:
    ParkBackoff() = default;
    explicit ParkBackoff(ParkBackoffPolicy p) : p_(p) {}

    /// Skip the estimate + telemetry for this request?  Only ever true while a backoff is running.
    bool quiet(int64_t request_index) const {
        return p_.refusals > 0 && quiet_until_ > request_index;
    }

    /// A refusal caused by the physical-RAM admission floor.  `request_index` is the engine's request
    /// counter, so the quiet period is a window of requests, not a wall-clock timeout.
    void refused(int64_t request_index) {
        if (p_.refusals <= 0) return;
        ++consecutive_;
        if (consecutive_ >= p_.refusals) {
            const int64_t until = request_index + (p_.quiet_requests > 0 ? p_.quiet_requests : 1);
            if (until > quiet_until_) quiet_until_ = until;
            went_quiet_ = true;
        }
    }

    /// A park that worked: the machine can take snapshots, so the backoff is over.
    void parked() { reset(); }

    /// Clear the streak and the window.  The engine calls this from `parked()` only; a forced request does
    /// not, on purpose (see the note above).
    void reset() {
        consecutive_ = 0;
        quiet_until_ = 0;
        went_quiet_ = false;
    }

    /// True once per backoff, so the engine can say it went quiet without repeating itself every request.
    bool went_quiet() const { return went_quiet_; }
    void clear_went_quiet() { went_quiet_ = false; }

    int64_t consecutive() const { return consecutive_; }
    int64_t quiet_until() const { return quiet_until_; }
    const ParkBackoffPolicy& policy() const { return p_; }

private:
    ParkBackoffPolicy p_;
    int64_t consecutive_ = 0;
    int64_t quiet_until_ = 0;   ///< request index the quiet period ends at (0 = not quiet)
    bool went_quiet_ = false;
};

}  // namespace strata::program::prefill_loan
