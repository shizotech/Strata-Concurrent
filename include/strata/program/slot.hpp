// include/strata/program/slot.hpp - the stage-3 SLOT object, its state machine and the scheduler's pick
// rule, as pure logic (docs/STAGE3-CONCURRENCY.md §2, §3.3, §3.4; plan step S3.1b).
//
// NO CUDA, NO I/O, NO ENGINE STATE.  Like `prefill_loan.hpp`, this file owns the DECISIONS and the engine
// in `src/program/generate.cpp` owns the state; the decisions are the part that can be wrong, and every
// one of them is testable on a CPU (`src/program/slot_test.cpp`).
//
// WHAT A SLOT IS NOT.  A slot is not a second session and not a second set of CUDA graphs.  §2.3 of the
// design settles that by arithmetic: one session arena is 1.39 GiB of VRAM and 6.19 GiB of host RAM on
// the box this is sized against, against 457 MiB of free VRAM and a 0.0 GiB parking budget.  There is
// exactly ONE live session, the one the captured graphs point at, and a slot's sequence state lives in the
// ConversationCache image that `conversation_snapshot_save/restore` moves in and out of it (§5.3).  So a
// `Slot` is bookkeeping: an id, a state, the parsed request, counters, and the index of its parked image.
// Nothing here allocates on the decode path: the registry is a fixed array, `pick()` walks it without
// allocating, and the only containers are a request's own, filled once at admission.
//
// THE STATE MACHINE (§3.4) is the six llama.cpp shapes (`server-context.cpp:100-107`) plus the three
// Strata needs: PARKED (not a new store - it IS the ConversationCache entry), CANCELLING (the unwind must
// refill the prompt loan before the slot is freed) and ERROR (a failed step must destroy the slot and NOT
// the process - today a `park_current` failure does `return 1` at generate.cpp:4597, which is risk R4).
//
// THE PICK RULE (§3.3) is deterministic and small on purpose: it is the fairness bound, and a scheduler
// that is not deterministic can be neither unit-tested nor replayed against a log.
//
// -------------------------------------------------------------------------------- row lifetime -----
// A registry row is occupied while `id != kNoId`, in ANY state including `idle`.  `transition()` changes
// states and never frees a row; `release()` frees it.  The engine transitions a finished slot to `idle`
// (so /slots can still show it), then releases it once its DONE/ERR line is out and its image has been
// handed to the ConversationCache.
#pragma once

#include "strata/program/serve_proto.hpp"

#include <array>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::program::slot {

// ------------------------------------------------------------------------ the states -------------------

enum class State : uint8_t {
    idle,         ///< finished or never used; not in the active set, awaiting release()
    queued,       ///< parsed, not admitted; holds ids/max_new/sampling only - NO state allocated
    prefilling,   ///< reading its prompt (batched chunks or short-read windows); holds the loan per chunk
    decoding,     ///< running verify windows; the active slot, or parked in host RAM
    parked,       ///< a paused/finished conversation whose image is in the ConversationCache
    cancelling,   ///< STOP seen; runs until the current step's unwind finishes
    error,        ///< a step failed; ERR <id>, then the slot is destroyed - the process stays up
};

/// The wire spelling (§6.2's `SLOT` line): lower-case, one token, the same names as the diagram.
inline const char* state_name(State s) {
    switch (s) {
        case State::idle: return "idle";
        case State::queued: return "queued";
        case State::prefilling: return "prefilling";
        case State::decoding: return "decoding";
        case State::parked: return "parked";
        case State::cancelling: return "cancelling";
        case State::error: return "error";
    }
    return "idle";
}

inline bool state_of(const std::string& name, State& out) {
    for (int i = 0; i <= (int) State::error; ++i)
        if (name == state_name((State) i)) { out = (State) i; return true; }
    return false;
}

/// The transition table, as data, so the test can enumerate it and the engine cannot quietly invent an
/// edge.  Anything not listed is illegal.  The self-edges that ARE listed are the step loop re-entering a
/// state (a window after a window, a chunk after a chunk); a self-edge that is not listed is how a stray
/// double-transition gets noticed.
inline bool can_transition(State from, State to) {
    switch (from) {
        case State::idle:        return to == State::queued;
        case State::queued:      return to == State::prefilling || to == State::decoding ||
                                       to == State::cancelling || to == State::error || to == State::idle;
        case State::prefilling:  return to == State::prefilling || to == State::decoding ||
                                       to == State::cancelling || to == State::error ||
                                       to == State::parked || to == State::idle;
        case State::decoding:    return to == State::decoding || to == State::parked ||
                                       to == State::cancelling || to == State::error || to == State::idle;
        case State::parked:      return to == State::decoding || to == State::prefilling ||
                                       to == State::idle || to == State::error;
        case State::cancelling:  return to == State::parked || to == State::idle || to == State::error;
        case State::error:       return to == State::idle;
    }
    return false;
}

/// Does this state occupy an ACTIVE slot (a mounted or mountable conversation: it costs a host image and
/// queue residency)?  QUEUED does NOT - §3.4: it "holds only ids/max_new/sampling; **no state
/// allocated**" - and PARKED does not either: it costs host RAM the parking budget already governs (§2.3).
/// `--serve-slots` caps exactly this set (§2.3 policy 3), which is why pick() gates admission on
/// `active_count() < max_active()`.
///
/// PARKED vs an UNMOUNTED slot with work left.  §3.4's table calls PARKED "a finished or PAUSED
/// conversation whose image is in ConversationCache", which would leave the scheduler blind to a paused
/// prefill - a parked slot is by definition not runnable, so it could never be swapped back in.  The model
/// used here keeps the two facts apart, and S3.1d/S3.1e must follow it:
///
///   * a slot whose conversation is merely UNMOUNTED (saved to the cache, work still pending) KEEPS its
///     work state - `prefilling` or `decoding` - and records the image in `parked_index`/`parked_bytes`;
///   * `parked` is for a conversation with NO pending work that is kept for reuse: exactly stage 2's parked
///     prefix, which a later request finds through `ConversationCache::best()`.
///
/// That makes `pick()` total over the states that have work, and it means a swap-out is not a state change
/// at all - which is right, because §5.3 defines a swap as a save/restore, never a re-capture.
inline bool is_active(State s) {
    return s == State::prefilling || s == State::decoding || s == State::cancelling;
}
/// Does this state still have something for the scheduler to do with it (as opposed to resting in RAM or
/// being finished)?
inline bool has_work(State s) {
    return s == State::queued || s == State::prefilling || s == State::decoding || s == State::cancelling;
}
/// Is a step of this state running on the GPU right now (so a swap must wait for it to finish)?
inline bool is_running(State s) {
    return s == State::prefilling || s == State::decoding;
}

// --------------------------------------------------------------------------- the slot ------------------

/// One request's worth of bookkeeping.  Public fields, no invariants beyond the state machine: the engine
/// thread owns every write except `cancel`, which the stdin thread sets.
struct Slot {
    int64_t id = serve_proto::kNoId;
    State state = State::idle;

    /// The parsed request: prompt ids, the GENI embeddings file, sampling and tuning params.  Filled once,
    /// at admission; the decode path never touches it.  (`req.error` is unused - a slot exists only for a
    /// request that parsed.)
    serve_proto::Request req;

    /// The cancel flag.  Today's `stop_req` is ONE process-wide atomic (generate.cpp:4211); §3.5 makes it
    /// per slot and `STOP <id>` names which.  Atomic because the stdin thread sets it while the engine
    /// thread reads it between windows and between prompt chunks.
    std::atomic<bool> cancel{false};

    // ---- counters the wire and /slots report (§6.2) --------------------------------
    int64_t ctx_used = 0;          ///< tokens of context this conversation holds right now
    int64_t ctx_cap = 0;           ///< the engine's --max-context, copied at admission
    int64_t prompt_tokens = 0;     ///< prompt length as admitted
    int64_t generated = 0;         ///< tokens emitted so far
    int64_t reused = 0;            ///< prompt tokens the conversation cache already held
    int64_t parked_bytes = 0;      ///< size of this slot's image in the ConversationCache (0 = none)

    // ---- timing: milliseconds on ONE steady_clock, taken by the caller -------------
    int64_t created_ms = 0;        ///< when the request was parsed and queued
    int64_t started_ms = 0;        ///< when it was admitted (0 = never)
    int64_t last_ran_ms = 0;       ///< when it last completed a step (0 = never)
    int64_t run_ms = 0;            ///< total ms this slot spent INSIDE a step (see note_step)
    int64_t prompt_ms = 0, decode_ms = 0;   ///< the engine's own measurements, for DONE

    /// §7.2: a slot that waited for the GPU or for a swap must report it, or its `prompt_ms`/`decode_ms`
    /// look like a slowdown.  Everything the slot existed for that it did NOT spend inside a step.
    int64_t wait_ms(int64_t now_ms) const {
        const int64_t span = now_ms >= created_ms ? now_ms - created_ms : 0;
        return span > run_ms ? span - run_ms : 0;
    }

    /// The ConversationCache index holding this slot's image, or -1.  §3.4: PARKED *is* that entry, so the
    /// slot keeps its handle and nothing else.
    int64_t parked_index = -1;

    /// The row is taken (in any state, `idle` included) - see "row lifetime" above.
    bool occupied() const { return id != serve_proto::kNoId; }
    const char* state_name() const { return slot::state_name(state); }

    Slot() = default;
    // `cancel` is non-copyable, and a registry row is recycled in place rather than copied: the stdin
    // thread holds a pointer to it.  Reporting a row's VALUES is what the SLOT line is for.
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
};

// ---------------------------------------------------------------------- the scheduler's decision ------

/// What `pick()` decided, and why.  `why` is a stable string so the engine can log the decision and the
/// replay test can assert on it.
struct Pick {
    enum class Action : uint8_t {
        idle,     ///< nothing to run: wait for a line
        run,      ///< step `id` on the session already mounted - NO swap
        swap,     ///< unmount the active slot and mount `id`, then step it
        admit,    ///< bring `id` in from QUEUED (restore its image if it has one); it becomes prefilling
        unwind,   ///< finish `id`'s cancel unwind (refill the loan, free its active slot), then DONE
    };
    Action action = Action::idle;
    int64_t id = serve_proto::kNoId;
    State state = State::idle;
    int64_t waited_ms = 0;         ///< how long `id` has been waiting, for the log
    const char* why = "idle";
};

inline const char* action_name(Pick::Action a) {
    switch (a) {
        case Pick::Action::idle: return "idle";
        case Pick::Action::run: return "run";
        case Pick::Action::swap: return "swap";
        case Pick::Action::admit: return "admit";
        case Pick::Action::unwind: return "unwind";
    }
    return "idle";
}

// --------------------------------------------------------------------------- the registry ---------------

/// Registry rows, recycled in place.  Queued + active + just-finished-awaiting-release all take a row, so
/// this sits comfortably above `--serve-slots`; the ACTIVE cap is what bounds GPU work.
constexpr int kMaxSlots = 16;
/// The hard ceiling on `--serve-slots`.  The design's practical number on the box it is sized against is 2
/// (risk R2: slots are host-RAM objects and the parking budget there is 0.0 GiB); 8 is where the registry
/// stops being trivially enumerable.  `generate.cpp` refuses a larger value at startup rather than
/// over-committing (risk R1: "refuse `--serve-slots N` rather than overcommit").
constexpr int kMaxActive = 8;

class Registry {
public:
    /// `max_active` = `--serve-slots` (0 or 1 = today's serial behaviour: one active slot, no swaps).
    /// `starve_ms` = `--starve-ms`, the fairness bound (§3.3); 0 = never force a swap.
    explicit Registry(int max_active = 1, int64_t starve_ms = 250)
        : max_active_(max_active < 0 ? 0 : (max_active > kMaxActive ? kMaxActive : max_active)),
          starve_ms_(starve_ms < 0 ? 0 : starve_ms) {}

    int max_active() const { return max_active_; }
    /// Concurrency is on only at >= 2 slots: at 0 or 1 the engine must be byte-identical to 0.1.30, which
    /// also means the wire stays untagged (serve_proto::Out(false)) and no SLOT line is ever emitted.
    bool concurrent() const { return max_active_ >= 2; }
    int64_t starve_ms() const { return starve_ms_; }

    /// The one slot the live session currently reflects (§5.3), or kNoId when nothing is mounted.
    int64_t active_id() const { return active_id_; }
    /// The engine calls this only after the mount/unmount actually succeeded - the registry does not
    /// pretend it knows whether the snapshot machinery worked.
    void set_active(int64_t id) { active_id_ = id; }

    /// A new request: QUEUED, no state allocated.  nullptr when the registry is full, or when the id is
    /// already live - two live requests sharing an id would make the wire unparseable.  The engine ERRs and
    /// the process is unaffected.
    Slot* add(int64_t id, int64_t now_ms) {
        if (id == serve_proto::kNoId || find(id) != nullptr) return nullptr;
        for (Slot& s : slots_) {
            if (s.occupied()) continue;
            s.id = id;
            s.state = State::queued;
            s.req = serve_proto::Request{};
            s.cancel.store(false);
            s.created_ms = now_ms;
            s.started_ms = 0;
            s.last_ran_ms = 0;
            s.generated = s.reused = s.prompt_tokens = s.ctx_used = 0;
            s.parked_bytes = 0;
            s.parked_index = -1;
            s.prompt_ms = s.decode_ms = 0;
            s.run_ms = 0;
            ++count_;
            return &s;
        }
        return nullptr;
    }

    Slot* find(int64_t id) {
        if (id == serve_proto::kNoId) return nullptr;
        for (Slot& s : slots_) if (s.occupied() && s.id == id) return &s;
        return nullptr;
    }
    const Slot* find(int64_t id) const {
        if (id == serve_proto::kNoId) return nullptr;
        for (const Slot& s : slots_) if (s.occupied() && s.id == id) return &s;
        return nullptr;
    }

    /// Free a row.  The engine calls this after the slot's DONE/ERR line is out and its image has been
    /// handed to the ConversationCache (a PARKED conversation keeps its row until it is pruned; a finished
    /// one loses it immediately).
    bool release(int64_t id) {
        Slot* s = find(id);
        if (s == nullptr) return false;
        if (active_id_ == id) active_id_ = serve_proto::kNoId;
        s->state = State::idle;
        s->id = serve_proto::kNoId;
        s->req = serve_proto::Request{};
        s->cancel.store(false);
        --count_;
        return true;
    }
    void clear() {
        for (Slot& s : slots_) { s.state = State::idle; s.id = serve_proto::kNoId; s.cancel.store(false); }
        count_ = 0;
        active_id_ = serve_proto::kNoId;
    }

    /// Transition a slot, refusing an edge the table does not list.  Returns false and changes nothing in
    /// that case - the caller logs it; it does not "fix" the state by force.  Moving INTO an active state
    /// from a state that is not already active additionally needs an active-slot permit (§2.3 policy 3).
    ///
    /// A cancelled slot never enters a running state: if `cancel` is set and the caller asks for
    /// `prefilling` or `decoding`, the slot lands in `cancelling` instead (the table's edge is checked for
    /// THAT target).  This is what makes `STOP <id>` arriving before admission safe - the request is let in
    /// only far enough to unwind and answer, and it can never start a new step.
    bool transition(int64_t id, State to, std::string& err) {
        Slot* s = find(id);
        if (s == nullptr) { err = "no such slot"; return false; }
        if (s->cancel.load() && (to == State::prefilling || to == State::decoding)) to = State::cancelling;
        if (!can_transition(s->state, to)) {
            err = std::string("illegal transition ") + state_name(s->state) + " -> " + state_name(to);
            return false;
        }
        if (is_active(to) && !is_active(s->state) && active_count() >= max_active_) {
            err = "no active slot free";
            return false;
        }
        if (is_active(to) && !is_active(s->state) && s->started_ms == 0) s->started_ms = s->created_ms;
        s->state = to;
        return true;
    }

    /// Mark a cancel (`STOP <id>`).  Only a slot with work to do can be cancelled.  The
    /// prefilling/decoding -> cancelling edge is taken here so the pick rule sees it immediately; a QUEUED
    /// slot keeps its state and takes the flag, and pick() drops it at admission.
    bool cancel_request(int64_t id) {
        Slot* s = find(id);
        if (s == nullptr || !has_work(s->state)) return false;
        s->cancel.store(true);
        if (is_running(s->state)) s->state = State::cancelling;
        return true;
    }
    /// A bare `STOP` (no id) means 0.1.30's "stop whatever is running": the newest started slot that is
    /// actually running.  If nothing is running, the newest queued one - the request that would run next.
    int64_t newest_id() const {
        for (int pass = 0; pass < 2; ++pass) {
            int64_t best = serve_proto::kNoId, best_key = -1;
            for (const Slot& s : slots_) {
                if (!s.occupied() || s.cancel.load()) continue;
                if ((pass == 0) != is_running(s.state)) continue;
                const int64_t key = s.started_ms ? s.started_ms : s.created_ms;
                if (key >= best_key) { best_key = key; best = s.id; }
            }
            if (best != serve_proto::kNoId) return best;
        }
        return serve_proto::kNoId;
    }

    int count() const { return count_; }
    int active_count() const {
        int n = 0;
        for (const Slot& s : slots_) if (s.occupied() && is_active(s.state)) ++n;
        return n;
    }
    int parked_count() const {
        int n = 0;
        for (const Slot& s : slots_) if (s.occupied() && s.state == State::parked) ++n;
        return n;
    }
    /// Would admitting one more conversation fit the active cap?  §2.3 policy 3: `--slots` caps
    /// SIMULTANEOUSLY ACTIVE slots, independently of how many are parked.
    bool can_admit() const { return active_count() < max_active_; }

    /// How long `s` has been waiting for the GPU: since its last step if it has run, since it was created
    /// if it never has (a slot that has never run is starving by definition).
    int64_t waited_ms(int64_t now_ms, const Slot& s) const {
        const int64_t since = s.last_ran_ms ? s.last_ran_ms : (s.started_ms ? s.started_ms : s.created_ms);
        return now_ms >= since ? now_ms - since : 0;
    }

    /// A step of `id` started (`step_ms` is what it cost, in ms).  Keeps the fairness clock and the
    /// per-slot `wait_ms` honest: `run_ms` accumulates the time the slot actually held the session, and
    /// `Slot::wait_ms(now)` is everything else.  Monotone by construction - a clock reading that goes
    /// backwards (a caller that mixed clocks) is ignored rather than turned into a negative duration.
    void note_ran(int64_t id, int64_t now_ms, int64_t step_ms = 0) {
        if (Slot* s = find(id)) {
            if (step_ms > 0) s->run_ms += step_ms;
            if (now_ms > s->last_ran_ms) s->last_ran_ms = now_ms;
        }
    }

    int64_t swaps() const { return swaps_; }
    int64_t swap_ms() const { return swap_ms_; }
    void note_swap(int64_t ms) { ++swaps_; swap_ms_ += ms; }

    // ---------------------------------------------------------------- the pick rule (§3.3) ---------
    //
    // Deterministic, and in the design's order, with one hole in the pseudocode closed.  The design's rule
    // 2 ("stay on the active slot while nobody has waited past T_starve") only fires when the active slot is
    // DECODING, and its rule 4 ("prefill before decode") then hands the next chunk to the active PREFILLING
    // slot forever - so a decoder waiting beside a 36 k-token read would never run, which is exactly what
    // §3.3 says T_starve exists to prevent ("a decode slot that has not run for that long forces a swap").
    // The fix is rule 4 below: past the bound, the starved slot that already occupies an active slot is
    // served before anything else.  Nothing else about the order changes.
    //
    // The default remains to STAY on the active slot: a swap is a full save+restore of a 237 MB-2.25 GB
    // image (risk R10), so fairness - not round-robin - is what forces one.
    Pick pick(int64_t now_ms) const {
        Pick p;
        // 1. a CANCELLING slot unwinds first: it holds the prompt loan, and an unwind that waits can leak
        //    it (the loan is one loan, §4.4).
        for (const Slot& s : slots_) {
            if (!s.occupied() || s.state != State::cancelling) continue;
            p.action = s.id == active_id_ ? Pick::Action::run : Pick::Action::swap;
            p.id = s.id; p.state = s.state; p.why = "cancel-unwind-first";
            p.waited_ms = waited_ms(now_ms, s);
            return p;
        }
        const Slot* act = find(active_id_);
        const Slot* waiting = most_waiting(now_ms);      // any runnable non-active slot, the longest wait
        const bool over = starve_ms_ > 0 && waiting != nullptr && waited_ms(now_ms, *waiting) > starve_ms_;
        // 2. the active slot keeps running while nobody else is starved: zero swaps, today's behaviour.
        if (act != nullptr && act->state == State::decoding && !over) {
            p.action = Pick::Action::run; p.id = act->id; p.state = act->state;
            p.why = "active-decoding"; p.waited_ms = 0;
            return p;
        }
        // 3. admit a queued request (restore its image if it has one) while an active slot is free.  One
        //    admission per pick: bringing a conversation in is itself a step.
        for (const Slot& s : slots_) {
            if (!s.occupied() || s.state != State::queued) continue;
            if (!can_admit()) break;              // the cap is the cap: never exceed --serve-slots
            p.action = Pick::Action::admit; p.id = s.id; p.state = s.state;
            p.why = "admit-queued"; p.waited_ms = waited_ms(now_ms, s);
            return p;
        }
        // 4. fairness: a slot that already occupies an active slot and has waited past the bound runs next,
        //    whatever the active slot is doing.
        if (over && is_active(waiting->state)) {
            p.action = waiting->id == active_id_ ? Pick::Action::run : Pick::Action::swap;
            p.id = waiting->id; p.state = waiting->state; p.why = "fairness-starved";
            p.waited_ms = waited_ms(now_ms, *waiting);
            return p;
        }
        // 5. prefill before decode: it is the long pole (8-15 s a chunk against a ~24 ms window), and a
        //    request that cannot prefill cannot decode at all.
        for (const Slot& s : slots_) {
            if (!s.occupied() || s.state != State::prefilling) continue;
            p.action = s.id == active_id_ ? Pick::Action::run : Pick::Action::swap;
            p.id = s.id; p.state = s.state; p.why = "prefill-next-chunk";
            p.waited_ms = waited_ms(now_ms, s);
            return p;
        }
        // 6. a decoding slot that is not the active one: swap to it, then its next window.
        for (const Slot& s : slots_) {
            if (!s.occupied() || s.state != State::decoding) continue;
            p.action = s.id == active_id_ ? Pick::Action::run : Pick::Action::swap;
            p.id = s.id; p.state = s.state; p.why = "decode-next-window";
            p.waited_ms = waited_ms(now_ms, s);
            return p;
        }
        // 7. nothing runnable: idle, and the engine waits for a line.
        p.action = Pick::Action::idle; p.why = "idle";
        return p;
    }

    /// The runnable non-active slot that has waited longest (queued, prefilling or decoding), or nullptr.
    /// `pick()` compares its wait against `starve_ms()`; it is exposed because the engine logs it and the
    /// replay test asserts on it.
    const Slot* most_waiting(int64_t now_ms) const {
        const Slot* best = nullptr;
        int64_t bw = -1;
        for (const Slot& s : slots_) {
            if (!s.occupied() || s.id == active_id_) continue;
            if (s.state != State::queued && s.state != State::prefilling && s.state != State::decoding) continue;
            const int64_t w = waited_ms(now_ms, s);
            if (w > bw) { bw = w; best = &s; }
        }
        return best;
    }

    /// The longest wait among the slots the active one would starve.  Kept for the log line.
    int64_t max_starve_wait(int64_t now_ms, int64_t except_id) const {
        int64_t m = 0;
        for (const Slot& s : slots_) {
            if (!s.occupied() || s.id == except_id) continue;
            if (s.state != State::queued && s.state != State::prefilling && s.state != State::decoding) continue;
            const int64_t w = waited_ms(now_ms, s);
            if (w > m) m = w;
        }
        return m;
    }

    /// Enumerate the occupied rows, for /slots and the `SLOT` lines.
    template <class F>
    void each(F&& f) const { for (const Slot& s : slots_) if (s.occupied()) f(s); }

private:
    std::array<Slot, kMaxSlots> slots_;
    int max_active_ = 1;
    int64_t starve_ms_ = 250;
    int64_t active_id_ = serve_proto::kNoId;
    int count_ = 0;
    int64_t swaps_ = 0, swap_ms_ = 0;
};

}  // namespace strata::program::slot
