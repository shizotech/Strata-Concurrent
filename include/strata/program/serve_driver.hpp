// include/strata/program/serve_driver.hpp - stage 3 S3.1e-2: the DECISIONS of the concurrent serve
// driver, as pure logic (docs/STAGE3-CONCURRENCY.md §3.1-§3.5, risks R3/R7/R8/R12).
//
// WHY THIS FILE EXISTS.  S3.1e-1 turned one request body into resumable steps
// (`prep_request` / `run_prefill_step` / `finish_prefill` / `run_decode_step` / `finish_request`) and
// S3.1d turned a slot switch into one call (`swap_to`).  What is left is the LOOP that interleaves
// them, and a loop that only exists inside `generate.cpp` cannot be tested without a GPU, a model and
// 78 GiB of RAM.  So every decision the loop makes - may this request be admitted, which step does a
// `Pick` turn into, who holds the prompt loan, has every active slot stalled, which slot does a bare
// STOP name - lives here, next to `slot.hpp`, `serve_proto.hpp` and `serve_swap.hpp`, and the engine
// keeps only the state and the calls.  Same rule as the three headers before it (risk R13).
//
// WHAT IT IS NOT.  No CUDA, no I/O, no session, no clock of its own: the caller passes `now_ms` from
// the same steady clock it hands to `Registry::pick()`, so the fairness bound and the watchdog are
// measured against one timeline.
//
// THE GATE.  Everything in this file is reached ONLY when `Registry::concurrent()` is true
// (`--serve-slots >= 2`).  With 0/1 the engine takes 0.1.30's serial driver verbatim, and none of
// these predicates runs.
#pragma once

#include "strata/program/serve_swap.hpp"
#include "strata/program/slot.hpp"

#include <cstddef>
#include <cstdint>
#include <chrono>
#include <string>
#include <vector>

namespace strata::program::serve_driver {

using serve_proto::kNoId;
using slot::Pick;
using slot::Registry;
using slot::Slot;
using slot::State;

/// The driver's ONE clock.  `Registry::pick(now_ms)`, the fairness bound, the per-slot heartbeat and
/// the watchdog all read this, so a slot cannot be "not starved" by one clock and "stalled" by
/// another.  The same steady clock the serve loop already uses.
inline int64_t now_ms() {
    return (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ------------------------------------------------------------------ the step the loop runs ---------

/// A `Pick` says WHICH slot and WHETHER to hand the session over.  The engine's own phase machine
/// says WHICH step that slot is between.  This is the join of the two, and it is the whole of the
/// driver's dispatch: `admit` runs the request body's phases 1+2, the four `*_step` values run one
/// step each, `finish` runs the request's tail.
enum class Step : uint8_t {
    none,        ///< nothing to do: wait for a line
    admit,       ///< prep_request + plan_prompt_segments (this is where the image is mounted)
    prefill,     ///< run_prefill_step: ONE prompt segment
    prefill_end, ///< finish_prefill: return the loan, REUSED, set the decode loop up
    decode,      ///< run_decode_step: ONE verify window (the pre-emption point)
    finish,      ///< finish_request: DONE, the conversation state, the metrics
    unwind,      ///< a cancelled slot: run its phase tail and finish it
    wait,        ///< runnable, but the resource it needs is held by another slot
    swap,        ///< hand the session over first, then step the slot that was picked
};

inline const char* step_name(Step s) {
    switch (s) {
        case Step::none: return "none";
        case Step::admit: return "admit";
        case Step::prefill: return "prefill";
        case Step::prefill_end: return "prefill-end";
        case Step::decode: return "decode";
        case Step::finish: return "finish";
        case Step::unwind: return "unwind";
        case Step::wait: return "wait";
        case Step::swap: return "swap";
    }
    return "?";
}

/// Where a slot's resumable request body currently sits.  This is the engine's phase machine stated
/// as data: `ReqCtx::phase` holds one of these, and it is what makes `Pick` (which knows only the
/// slot's registry state) turn into a concrete step.
enum class Phase : uint8_t {
    queued,       ///< parsed, a registry row exists, `prep_request` has not run
    prefill,      ///< between prompt segments (`R.at` is the cursor)
    prefill_end,  ///< the segments are read or the read was cancelled; the loan goes back
    decode,       ///< between verify windows
    done,         ///< the request is over; `finish_request` ran
};

inline const char* phase_name(Phase p) {
    switch (p) {
        case Phase::queued: return "queued";
        case Phase::prefill: return "prefill";
        case Phase::prefill_end: return "prefill-end";
        case Phase::decode: return "decode";
        case Phase::done: return "done";
    }
    return "?";
}

/// Does a slot in this phase still have a step to run?
inline bool phase_has_work(Phase p) {
    return p == Phase::prefill || p == Phase::prefill_end || p == Phase::decode;
}

/// The step `pick()`'s decision turns into, for a slot in phase `ph`.
///
/// `needs_swap` is deliberately NOT resolved here: the caller has already asked `swap_to` (or is
/// about to), and a hand-over that fails must not be papered over by running a step against a
/// session that does not hold this slot's conversation.
inline Step dispatch_step(const Pick& p, Phase ph) {
    if (p.action == Pick::Action::idle) return Step::none;
    // A QUEUED slot is admitted, whatever else pick asked for.  Two reasons, both load-bearing:
    //   * `prep_request` owns the REQUEST-LINE hand-over (S3.1d's call site), and it is the form that
    //     takes the incoming image out of the cache BEFORE saving the outgoing conversation, so the
    //     outgoing save counts it as `held` RAM (`serve_swap::fits` / `ConversationCache::can_fit`).
    //     A driver that swapped first would get that budget arithmetic wrong and would save and
    //     restore the same session twice.
    //   * it is also the only thing that creates the registry row, the mount and the sampling dispatch.
    if (ph == Phase::queued) return Step::admit;
    // A cancelled slot never starts a NEW step.  §3.4: CANCELLING "runs until the current step's
    // unwind finishes", and `Registry::transition` already refuses to move it back into
    // prefilling/decoding.  So the unwind is a REASON, not a different step: the driver runs the
    // phase's tail, which is the part that returns the prompt loan (R8) and answers DONE.  Jumping
    // straight to `finish_request` would be a real bug: on a request whose decode loop never ran,
    // `consumed` is empty and `finish_request` does `live.swap(consumed)` - it would wipe the
    // mounted conversation.
    if (p.action == Pick::Action::unwind || p.state == State::cancelling) {
        Step t;
        switch (ph) {
            case Phase::prefill:     t = Step::prefill_end; break;  // the tail refills the loan
            case Phase::prefill_end: t = Step::prefill_end; break;
            case Phase::decode:      t = Step::finish;      break;  // no new window for a cancelled slot
            case Phase::done:        t = Step::finish;      break;
            default:                 t = Step::none;        break;
        }
        // §5.3 applies to an unwind too: a cancelled slot the session does not reflect must be handed
        // the session before it can finish.  `pick()`'s rule 1 already answers `swap` for that case.
        return (p.action == Pick::Action::swap && t != Step::none) ? Step::swap : t;
    }
    Step s;
    switch (ph) {
        case Phase::prefill:     s = Step::prefill; break;
        case Phase::prefill_end: s = Step::prefill_end; break;
        case Phase::decode:      s = Step::decode; break;
        case Phase::done:        s = Step::finish; break;
        default:                 s = Step::none; break;
    }
    // A hand-over is only needed when the picked slot is not the one the session reflects.  `run` on
    // the active slot is the zero-swap case, which is 0.1.30's behaviour and the default the design
    // insists on (risk R10: a swap is a 237 MB-2.25 GB memcpy).
    return (p.action == Pick::Action::swap && s != Step::none) ? Step::swap : s;
}

/// After a step returned `progressed`, which phase is the slot in for the NEXT pick?  `finished` and
/// `cancelled` move it on; `progressed` keeps it.  Kept as a table so the driver's phase machine is
/// one predicate rather than five `if`s at the call sites.
enum class Outcome : uint8_t { progressed, finished, cancelled };

inline Phase next_phase(Phase ph, Outcome o) {
    if (o == Outcome::progressed) return ph;
    switch (ph) {
        case Phase::prefill: return Phase::prefill_end;
        case Phase::prefill_end: return Phase::decode;
        case Phase::decode: return Phase::done;
        case Phase::queued: return Phase::prefill;
        case Phase::done: return Phase::done;
    }
    return Phase::done;
}

// ------------------------------------------------------------------------ admission ----------------

/// Why admission was refused.  The engine prints `ERR <id> <why>` for each of these and the process
/// stays up (§3.4's `refuse` edge) - which is itself the fix for risk R4, where a parking failure
/// used to `return 1` and kill the server.
enum class Refuse : uint8_t { none, slots_full, ram, no_parking, registry_full };

inline const char* refuse_reason(Refuse r) {
    switch (r) {
        case Refuse::none: return "ok";
        case Refuse::slots_full: return "all --serve-slots are busy";
        case Refuse::ram: return "not enough free RAM for another conversation";
        case Refuse::no_parking: return "the conversation cache is off, so a slot switch would lose the conversation";
        case Refuse::registry_full: return "no slot row free";
    }
    return "?";
}

/// The RAM side of admission, as a decision.  `avail_bytes` is the machine's free RAM (nullopt when
/// the platform has no telemetry - then the budget is the only gate, as it is for parking),
/// `floor_bytes` is `--conversation-cache-min-free-mib`, `image_bytes` is what this slot will hold
/// (its parked image, or the engine's own per-slot estimate), and `already_held` is what the other
/// active slots already hold.  The rule is the parking rule (`conversation_memory_admit`) with one
/// extra term: the new slot's own image, which is not in the cache yet and so is not in `budget`.
inline bool admit_fits_ram(bool have_telemetry, uint64_t avail_bytes, uint64_t floor_bytes,
                           uint64_t image_bytes, uint64_t already_held) {
    if (!have_telemetry) return true;   // no telemetry: the parking budget is the gate, as it is for parking
    const uint64_t need = image_bytes + already_held;
    return avail_bytes >= floor_bytes + need;
}

/// The FALLBACK per-slot host-RAM estimate: the parking budget divided by the number of images that
/// budget was sized for.  0 when parking is off, which `parking_off_refuses_slots` turns into a
/// refusal rather than a guess.
///
/// THIS IS A GUESS, AND IT IS THE WRONG ONE WHEN IT CAN BE AVOIDED.  It says "every conversation on
/// this box is `budget / --conversation-cache-slots` long", so an 8 GiB budget with 8 cache entries
/// charges a 150-token chat the same 1 GiB it charges a 200k-token one, and refuses the short one
/// because 1 GiB was not free.  Parking itself has never worked that way - `ConversationCache` counts
/// `SavedConversation::bytes()` of what it actually stored - so this figure is only used until the
/// engine has measured a real snapshot on this model (see `SlotCost::per_token`).
inline uint64_t slot_estimate(size_t budget_bytes, size_t cache_slots) {
    if (budget_bytes == 0 || cache_slots == 0) return 0;
    return (uint64_t) (budget_bytes / cache_slots);
}

/// Bytes of snapshot per context token, from a park that ACTUALLY happened.  This is the honest
/// unit: on the owner's box a parked conversation costs ~19.5 KiB per token, and the log prints the
/// token count and the byte count of every park, so the engine can divide them instead of guessing.
/// 0 for a zero-token park, which is not evidence of anything.
inline uint64_t bytes_per_token(uint64_t snapshot_bytes, uint64_t tokens) {
    return tokens == 0 ? 0 : (snapshot_bytes + tokens - 1) / tokens;
}

/// What one new slot is expected to hold, and what the engine knows about this model's cost.
///
///   * `per_token`     - bytes per context token, from the best rate the engine has: the largest
///                       snapshot this process has actually parked, else the rate the startup ceiling
///                       implies (`ParkCeiling::per_token`), else 0;
///   * `ceiling_bytes` - the startup worst case: what a conversation at this `--max_context` would
///                       need (`ParkCeiling::snapshot_bytes`, 0 = the engine could not size it);
///   * `budget_bytes` / `cache_slots` - the parking budget in force and its entry cap.
struct SlotCost {
    uint64_t per_token = 0;
    uint64_t ceiling_bytes = 0;
    uint64_t budget_bytes = 0;
    size_t cache_slots = 0;
};

/// The RAM to charge a request for, sized to THAT request rather than to the machine's average.
///
/// A parked conversation is demand-sized - `conversation_snapshot_bytes` measures it against
/// `live.size()`, the tokens it actually holds, and `ConversationCache::bytes_` counts what was
/// stored, never `--max_context`.  So the admission figure should be too:
///
///   1. with a rate, `prompt + max_new` tokens at that rate.  `max_new` is in there because the answer
///      the request has not written yet becomes part of the conversation, and a slot pre-empted
///      mid-decode parks all of it;
///   2. sized but rateless, the ceiling itself;
///   3. with nothing to go on, the old `budget / cache_slots` guess - the last resort, and the one
///      that made a 150-token chat look like a 1 GiB conversation;
///   4. never more than the ceiling (nothing on this box can cost more than a full-context
///      conversation) and never more than the whole budget (a reservation larger than the budget
///      refuses every request forever, which is the failure the owner hit).
///
/// A zero answer means "the engine cannot size this slot", and the caller treats it as no RAM gate
/// (the same rule `admit_fits_ram` applies when the platform has no memory telemetry).
inline uint64_t slot_image_bytes(const SlotCost& c, uint64_t prompt_tokens, uint64_t max_new) {
    uint64_t need = 0;
    if (c.per_token != 0) need = (prompt_tokens + max_new) * c.per_token;
    else if (c.ceiling_bytes != 0) need = c.ceiling_bytes;
    else if (c.budget_bytes != 0 && c.cache_slots != 0) need = slot_estimate(c.budget_bytes, c.cache_slots);
    if (c.ceiling_bytes != 0 && need > c.ceiling_bytes) need = c.ceiling_bytes;
    if (c.budget_bytes != 0 && need > c.budget_bytes) need = c.budget_bytes;
    return need;
}

/// Risk R2/R12, stated as a startup rule.  A slot switch is a save/restore (§5.3), and the save is
/// `park_current`.  With parking off, S3.1d's `invalidate_unparked` makes the outgoing conversation
/// un-mountable - so "concurrency" would silently destroy every request it pre-empted.  A box that
/// cannot park cannot host two conversations, and it is better to refuse at startup than to serve
/// garbage.  `kv_mode == 0` (no KV streaming, `--kv k8v4`) is the same argument in risk R12's form:
/// the image is 13 GiB and the swap path does not exist.
inline bool parking_off_refuses_slots(int slots, bool cache_enabled) {
    return slots >= 2 && !cache_enabled;
}

// ------------------------------------------------ the parking-collapse admission check (S3.6) --
//
// `parking_off_refuses_slots` catches a budget of ZERO.  The owner's box had a budget of 1.6 GiB,
// which is not zero and is still useless: a 175K-token conversation's snapshot is ~3.3 GiB, so it
// could never be parked, and the engine discovered that 97 seconds later by destroying the
// conversation mid-decode.  A nonzero-but-useless budget has to be named at admission, with both
// numbers, and it has to say what concurrency is actually possible here.
//
/// Can a snapshot of `snapshot_bytes` EVER be parked in this cache?  Mirrors
/// `ConversationCache::make_room` (which prunes LRU entries to make space), so a full cache is not a
/// refusal - only an image larger than the whole budget, or a budget of 0, can never fit.
inline bool parkable_at_all(size_t budget_bytes, size_t cache_slots, size_t snapshot_bytes) {
    return budget_bytes != 0 && cache_slots != 0 && snapshot_bytes <= budget_bytes;
}

/// The worst snapshot a conversation at this `--max_context` can need, from the engine's own sizing
/// (`conversation_snapshot_bytes` at `--max_context` tokens).  0 = the engine could not size it, in
/// which case there is nothing to compare and the check stays quiet.
struct ParkCeiling {
    size_t snapshot_bytes = 0;   ///< the largest snapshot this context can produce
    size_t budget_bytes = 0;     ///< the parking budget in force
    size_t slots = 0;            ///< --conversation-cache-slots
    bool sized = false;          ///< the engine could size it

    bool useless_budget() const { return sized && !parkable_at_all(budget_bytes, slots, snapshot_bytes); }
    /// How many conversations of this size the budget could hold, if any.
    int capacity() const {
        if (!sized || snapshot_bytes == 0 || budget_bytes < snapshot_bytes) return 0;
        const int n = (int) (budget_bytes / snapshot_bytes);
        return n < (int) slots ? n : (int) slots;
    }
    /// The bytes per context token this ceiling implies.  Until something has actually been parked
    /// this is the only rate the engine has, and it is what admission charges a request against
    /// (`slot_image_bytes` rule 2 caps the old `budget / slots` guess by the ceiling; this is the
    /// per-token form of the same number, so a short conversation is not charged for a long one).
    uint64_t per_token() const {
        if (!sized || snapshot_bytes == 0) return 0;
        return bytes_per_token(snapshot_bytes, ceiling_tokens_);
    }
    /// The token count `snapshot_bytes` was measured at - `--max_context`, unless the caller says
    /// otherwise.  0 makes `per_token()` 0, which puts admission back on the old guess.
    void size_at(uint64_t tokens) { ceiling_tokens_ = tokens; }

   private:
    uint64_t ceiling_tokens_ = 0;
};

/// The engine's `SlotCost`, assembled in the order the rates get better: a park this process actually
/// measured, else the rate the startup ceiling implies, else nothing at all - which is what puts
/// admission back on the `budget / slots` guess.  One place builds it so the engine and the test
/// cannot disagree about the precedence.
inline SlotCost slot_cost_of(uint64_t budget_bytes, size_t cache_slots, const ParkCeiling& ceiling,
                             uint64_t measured_per_token) {
    SlotCost c;
    c.budget_bytes = budget_bytes;
    c.cache_slots = cache_slots;
    c.ceiling_bytes = ceiling.sized ? (uint64_t) ceiling.snapshot_bytes : 0;
    c.per_token = measured_per_token != 0 ? measured_per_token : ceiling.per_token();
    return c;
}

/// What to print when the budget cannot hold the snapshot a conversation at this `--max_context`
/// would need.  Kept as a formatter so the wording, the two numbers and the test agree on one
/// spelling.  `driver_running` says whether the concurrent driver is actually on: the honest answer
/// differs, and a message that claims "running serial" while the driver is running is worse than no
/// message at all.
inline std::string park_ceiling_line(int slots, const ParkCeiling& c, bool driver_running) {
    const int cap = c.capacity();
    std::string s = "strata serve: --serve-slots " + std::to_string(slots) + ": a conversation at this "
                    "--max_context needs a snapshot of up to " +
                    std::to_string((long long) (c.snapshot_bytes >> 20)) + " MiB, but the parking budget is " +
                    std::to_string((long long) (c.budget_bytes >> 20)) + " MiB, so a conversation that long "
                    "can NEVER be parked";
    if (cap == 0) {
        s += " - no conversation at this --max_context can be swapped out and back in";
        if (driver_running)
            s += ". The concurrent driver is ON, but it will REFUSE every hand-over between two "
                 "conversations this long (the incoming request gets an ERR naming both numbers) and "
                 "serve them one at a time. Shorter conversations still swap normally.";
        else
            s += ". Running serial: one conversation at a time.";
        s += " For real concurrency here raise --conversation-cache-mib to at least " +
             std::to_string((long long) ((c.snapshot_bytes * (uint64_t) (slots > 0 ? slots : 1)) >> 20)) +
             " MiB for " + std::to_string(slots) + " slots, or lower --max-context.";
        // The number above is the worst case, and it is worth saying so: nothing is reserved per slot.
        // A parked conversation costs what it actually holds, so this figure only applies if every
        // conversation really runs to --max_context.
        s += " That figure assumes every conversation reaches --max_context: a parked conversation "
             "costs what it holds, not what it could grow to, so shorter ones need proportionally less.";
    } else {
        s += " - only snapshots up to " + std::to_string((long long) (c.budget_bytes >> 20)) +
             " MiB can be swapped, so at most " + std::to_string(cap) +
             " conversation(s) of this size fit the budget. A longer one is refused at its hand-over, "
             "never silently destroyed.";
    }
    return s;
}

// ------------------------------------------------------- the step gate (S3.6) -----------------
//
// THE INVARIANT (§5.3), as something the driver asserts before every step: a slot may only step when
// the mounted session describes it.  The parking-collapse bug was a slot in `Phase::decode` whose
// `SlotConv` had been invalidated (its conversation destroyed by a hand-over that could not park it)
// and which was stepped anyway - against another conversation's K/V and positions.  It sampled
// garbage, usually hit EOS at once, and reported nothing.
//
// The gate turns that into a named ERR.  Its input is `SlotConv::resumable`, the contract's own
// predicate, and it is the right one: `invalidate_unparked` is the ONLY thing that ever clears it, so
// `resumable == false` means precisely "this slot's conversation was destroyed by a hand-over that
// could not save it".  A healthy mid-decode slot has it set (the adopt hook sets it true for whatever
// the last hand-over mounted), so the gate never fires on a normal run - which is what makes it a
// guard rather than a new behaviour.
//
// WHY NOT `SlotConv::live_ok`.  It means "the session's positional cells hold exactly this token
// list", and `prep_request` deliberately clears it for the whole life of a request ("until this
// request has finished, the session is in between").  A gate that read it would end every decode slot
// in the process.
enum class Gate : uint8_t { run, re_read, end };

/// `ph` is the request's phase; `resumable` its `SlotConv::resumable`; `session_valid` whether the
/// live session still holds this request's sequence (`ReqCtx::session_valid`).
///
/// WHY BOTH.  They answer different questions and neither implies the other:
///   * `ph == decode && !resumable` is the CONTRACT's impossible case - the conversation was
///     destroyed by a hand-over that could not save it, and a decoder has nothing to re-read.  It is
///     checked FIRST, before `session_valid`, because a slot can be mounted, "valid" and still have
///     lost its parked image: that is precisely how the owner's run ended a request early.
///   * `!session_valid` is the ordinary pre-emption case: the session went to another slot.  A
///     mid-prompt request can honour "re-read from token 0"; a decoder cannot, so it ends.
///   * After `reset_request_to_token0` the session is zeroed and about to be read by THIS request, so
///     `session_valid` is true again - which is what stops the re-read from repeating forever.  A
///     re-read slot legitimately keeps `resumable == false` (nothing was parked), so the prefill
///     branch must not test it.
inline Gate step_gate(Phase ph, bool resumable, bool session_valid) {
    if (!phase_has_work(ph)) return Gate::run;   // queued/done run no step of this kind
    if (ph == Phase::decode && !resumable) return Gate::end;
    if (session_valid) return Gate::run;
    if (ph == Phase::decode) return Gate::end;
    // DECODE cannot be recovered: the tokens it already generated are on the wire, and re-reading the
    // prompt would emit them a second time.  Ending it with a named ERR is the only honest answer.
    if (ph == Phase::decode) return Gate::end;
    // PREFILL / PREFILL-END still hold their whole prompt in `ReqCtx::ids`, so the promise "re-read
    // from token 0" CAN be kept: reset the cursor, re-plan the segments, re-resume.
    return Gate::re_read;
}

/// Why the gate ended a slot, with the numbers that were actually true at that moment.
///
/// THE OLD WORDING WAS A LIE, and it sent the owner off to fix the wrong thing.  It blamed
/// "the parking budget is too small for a conversation this long" for EVERY `Gate::end`, but the gate
/// fires for two different reasons and only one of them is the budget:
///
///   * `!resumable` in `decode` - the conversation was destroyed by a hand-over that could not save
///     it.  THAT is a parking-budget (or parking-failure) condition, and the two numbers are the
///     evidence;
///   * `!session_valid` - the session went to another slot and nothing was restored into this one.
///     A generous budget does not prevent it, so blaming the budget is simply wrong.
///
/// So the reason carries the state it found and, when the caller has them, the snapshot the save
/// needed and the budget in force.  Without numbers it says what it actually observed instead of
/// guessing at a cause.
struct GateEnd {
    Phase ph = Phase::decode;
    bool resumable = false;   ///< did this slot still have a parked conversation to come back to?
    bool session_valid = false;  ///< does the mounted session describe this slot?
    uint64_t snapshot_bytes = 0;  ///< what its last save would have needed (0 = not measured)
    uint64_t budget_bytes = 0;    ///< the parking budget in force (0 = parking off)
};

inline std::string gate_end_reason(const GateEnd& g) {
    std::string s = "this slot's conversation was lost while it was " + std::string(phase_name(g.ph)) +
                    ": the mounted session no longer describes it, so no step may run against it.";
    if (g.ph == Phase::decode && !g.resumable) {
        s += " Its state was dropped on the last hand-over and a decoder cannot re-read its prompt, so"
             " it cannot be resumed.";
        if (g.snapshot_bytes != 0) {
            s += " The save needed " + std::to_string((long long) (g.snapshot_bytes >> 20)) +
                 " MiB and the parking budget is " + std::to_string((long long) (g.budget_bytes >> 20)) +
                 " MiB.";
            if (g.snapshot_bytes > g.budget_bytes)
                s += " The parking budget is too small for a conversation this long - raise"
                     " --conversation-cache-mib to at least " +
                     std::to_string((long long) (g.snapshot_bytes >> 20)) + " MiB or lower --max-context.";
            else
                s += " The budget was big enough, so the save itself failed (free RAM, or the cache"
                     " refused it) - check the park lines above this one.";
        } else {
            s += " See the `park:` line for this slot for the two numbers; if the snapshot exceeded the"
                 " budget, raise --conversation-cache-mib or lower --max-context.";
        }
    } else if (!g.session_valid) {
        s += " The session was handed to another slot without restoring this one's state - either it"
             " was never parked or the cache pruned it while it waited, so there is nothing to mount"
             " back. This is not a budget size problem: raising --conversation-cache-mib would not"
             " have restored it (a bigger budget, or fewer conversations competing for the same"
             " budget, makes pruning less likely, though).";
    }
    return s;
}

/// The one-argument form: the caller has no numbers, so it must not pick a cause.  Kept for tests and
/// for any site that cannot reach the state; the engine always uses the `GateEnd` form above.  A guess
/// at a cause is what made the old message useless, so this one names both possibilities.
inline std::string gate_end_reason(Phase ph) {
    return "this slot's conversation was lost while it was " + std::string(phase_name(ph)) +
           ": the mounted session no longer describes it, so no step may run against it."
           " Either its state was dropped on a hand-over that could not park it (raise"
           " --conversation-cache-mib or lower --max-context), or the session went to another slot"
           " without restoring it. The `park:` and `swap` lines above name which.";
}

// ------------------------------------------------ the hand-over refusal, two kinds (S3.6) -----
//
// `swap_to` can now refuse a hand-over for two different reasons, and the driver must treat them
// differently.  Getting this wrong either spins the loop or kills a request that could have waited.
//
//   * NEVER - the outgoing slot's snapshot can never fit the parking budget (`ParkRefusal::
//     budget_too_small`).  No amount of waiting changes that, so the INCOMING request is answered
//     with an ERR naming both numbers and its permit comes back.  This is the parking-collapse case,
//     and it is the one the owner hit: 3.3 GiB of snapshot against a 1.6 GiB budget, 733 times.
//   * LATER - the session is mid-read, so there is no whole branch to save yet (`ParkRefusal::
//     not_saveable`; `park_current`'s own first guard is `live_ok`).  The running request finishes
//     its read and becomes parkable, so the incoming one WAITS.  ERRing it would turn a normal
//     two-client race into a failure; swapping anyway is what used to destroy the half-read
//     conversation and then resume it against a foreign session.
enum class SwapResult : uint8_t {
    mounted,      ///< the session now reflects the slot: step it
    wait_park,    ///< refused as NOT-YET: the incoming slot waits, the outgoing one keeps the session
    end_incoming, ///< refused as NEVER: answer the incoming request with an ERR, session untouched
    fatal,        ///< the restore was attempted and failed: the session is finished for this process
};

inline const char* swap_result_name(SwapResult r) {
    switch (r) {
        case SwapResult::mounted: return "mounted";
        case SwapResult::wait_park: return "wait-park";
        case SwapResult::end_incoming: return "end-incoming";
        case SwapResult::fatal: return "fatal";
    }
    return "?";
}

/// The phase a slot is in decides what losing the session would cost it.  This is the ONE mapping the
/// driver and the swap guard share, so they cannot disagree about whether a hand-over must save.
///   queued / done - no step owed: `finished`
///   prefill / prefill_end, nothing read yet - the prompt is still entirely in hand: `re_readable`
///   prefill / prefill_end, tokens already read - progress the cache CAN save: `must_park`
///   decode        - the answer is already partly on the wire: `must_park`
///
/// S3.8 THE TWO-PREFILL LIVELOCK, and why `read_tokens` is now part of the question.  `re_readable`
/// means "losing this is survivable, the caller owes it a reset to token 0", and the driver honours
/// that - which is exactly why two concurrent prefills never finish.  A mid-prompt read has real
/// progress in the session, and the cache can save it, but the old rule classified it as throwaway,
/// so every swap dropped it and every swap-in sent the request back to token 0.  The owner's log:
/// slot 3 `reset to token 0: 81 prompt tokens will be read again`, 101 times, while slot 2 parked and
/// resumed normally.  Re-reading is survivable ONCE; re-reading on every swap is a livelock, and a
/// prompt that has consumed tokens must be parked instead.
inline serve_swap::Outgoing outgoing_for(Phase ph, int64_t read_tokens) {
    switch (ph) {
        case Phase::prefill:
        case Phase::prefill_end:
            return read_tokens > 0 ? serve_swap::Outgoing::must_park
                                   : serve_swap::Outgoing::re_readable;
        case Phase::decode:      return serve_swap::Outgoing::must_park;
        default:                 return serve_swap::Outgoing::finished;
    }
}

/// The phase alone, with no cursor: nothing has been read, so a prefill slot is still `re_readable`.
/// Kept for callers that genuinely do not know the cursor (a freshly admitted row) and for the
/// phase-total invariant test; the driver always uses the two-argument form.
inline serve_swap::Outgoing outgoing_for(Phase ph) { return outgoing_for(ph, 0); }

/// How many times one request may be sent back to token 0 before the driver calls it a livelock
/// instead of a recovery.  A re-read is a real answer when the branch was lost once; unbounded, it is
/// the two-prefill spin the owner hit, and it looks like progress to the watchdog because every pass
/// really does read tokens.
inline constexpr int kMaxRereads = 4;

/// Given the re-reads already spent, does another one count as progress or as a spin?
inline bool reread_allowed(int rereads, int limit = kMaxRereads) { return rereads < limit; }

/// The ERR for the bound.  Named so the test pins the wording and the owner can grep for it.
inline std::string reread_limit_line(int64_t id, int rereads, int limit) {
    return "slot " + std::to_string((long long) id) + " was sent back to token 0 " +
           std::to_string(rereads) + " times (limit " + std::to_string(limit) +
           "): its prompt read cannot be parked, so every hand-over erases it and it would restart "
           "forever. Raise --conversation-cache-mib, lower --max-context, or run this box with "
           "--serve-slots 0/1.";
}

/// The ERR line for the NEVER case, with both numbers in it.
inline std::string handover_err_line(int64_t out_id, uint64_t snapshot_bytes, uint64_t budget_bytes) {
    return "slot " + std::to_string((long long) out_id) + " cannot be parked: its snapshot is " +
           std::to_string((long long) (snapshot_bytes >> 20)) + " MiB and the parking budget is " +
           std::to_string((long long) (budget_bytes >> 20)) + " MiB. Raise --conversation-cache-mib "
           "to at least " + std::to_string((long long) (snapshot_bytes >> 20)) +
           " MiB, or lower --max-context, or run this box with --serve-slots 0/1.";
}

// ------------------------------------------------ the re-read / resume events (S3.6) ----------
//
// The old log promised "will be re-read from token 0" and never delivered an event for it.  Both
// outcomes are now lines of their own, so a conversation that lost its state is visible in the log
// rather than inferable from a suspiciously large `prompt ... = 0 reused` figure.
inline std::string reread_line(int64_t id, int64_t tokens_dropped) {
    return "strata serve: slot " + std::to_string((long long) id) + " re-reading from token 0 (" +
           std::to_string((long long) tokens_dropped) + " tokens were dropped with it)";
}
inline std::string resumed_line(int64_t id, int64_t tokens) {
    return "strata serve: slot " + std::to_string((long long) id) + " resumed from " +
           std::to_string((long long) tokens) + " tokens";
}

// ------------------------------------------------------- the driver trace (S3.6) --------------
//
// `STRATA_SERVE_TRACE=1`: the scheduler's decisions, on stderr, one line each.  It is behind ONE
// switch so the default log stays readable, and it is the only way to answer "why did the engine
// pick THAT slot" without a debugger.  `Pick::why` already existed and was thrown away.
inline bool serve_trace_on() {
    const char* v = std::getenv("STRATA_SERVE_TRACE");
    return v != nullptr && *v != '\0' && std::string(v) != "0";
}

inline std::string trace_pick(const char* source, const Pick& p) {
    return std::string("strata trace: pick[") + source + "] " + slot::action_name(p.action) +
           " slot " + std::to_string((long long) p.id) + " state " + slot::state_name(p.state) +
           " waited " + std::to_string((long long) p.waited_ms) + " ms why=" + p.why;
}
inline std::string trace_phase(int64_t id, Phase from, Phase to, const char* because) {
    if (from == to) return std::string();
    return "strata trace: slot " + std::to_string((long long) id) + " phase " + phase_name(from) +
           " -> " + phase_name(to) + " (" + because + ")";
}
inline std::string trace_step(int64_t id, Step s, const char* why) {
    return std::string("strata trace: step ") + step_name(s) + " slot " +
           std::to_string((long long) id) + " why=" + why;
}
inline std::string trace_loan_defer(int64_t id, int64_t owner) {
    return "strata trace: defer slot " + std::to_string((long long) id) + " on the prompt loan held by slot " +
           std::to_string((long long) owner);
}
inline std::string trace_park_wait(int64_t id, int64_t mounted) {
    return "strata trace: defer slot " + std::to_string((long long) id) + ": slot " +
           std::to_string((long long) mounted) + " is mid-read and cannot be saved yet";
}

/// The periodic activity line.  `slots_active` is only printed when it CHANGES, which makes a run
/// that never got past one conversation look exactly like a run that was idle: the two differ only by
/// the absence of a line.  This one is printed on a time bound regardless, and it carries the peak,
/// so `peak=2` distinguishes them even if the peak happened an hour ago.
///
/// S4.2 appends the hold queue's own state (`waiting=N oldest=Ms reason=R`) as a TRAILING defaulted
/// parameter, so every existing field keeps its position and meaning and a caller that has no queue
/// still prints the 0.1.30-era line.  `waiting=0` is a positive statement: it is what distinguishes
/// "the engine is idle" from "the engine stopped reporting".
inline std::string activity_line(int slots, int active, int peak, int64_t swaps, int64_t swap_ms,
                                 int64_t steps, int64_t admitted, int64_t refused, size_t parked,
                                 size_t parked_bytes, size_t budget_bytes,
                                 const std::string& wait_tail = std::string()) {
    std::string s = "strata serve: activity: slots=" + std::to_string(slots) +
           " slots_active=" + std::to_string(active) + " peak=" + std::to_string(peak) +
           " steps=" + std::to_string((long long) steps) +
           " admitted=" + std::to_string((long long) admitted) +
           " refused=" + std::to_string((long long) refused) +
           " swaps=" + std::to_string((long long) swaps) +
           " swap_ms=" + std::to_string((long long) swap_ms) +
           " parked=" + std::to_string((long long) parked) +
           " parked_bytes=" + std::to_string((long long) (parked_bytes >> 20)) + "MiB" +
           " budget=" + std::to_string((long long) (budget_bytes >> 20)) + "MiB";
    if (!wait_tail.empty()) s += " " + wait_tail;
    return s;
}

// ------------------------------------------------------------------- the prompt loan (R8) --------

/// The prompt loan is ONE resource: `pf_parts` lends the tail slots of each stage's expert cache to
/// that stage's `Prefill`, and two prefills holding the same tail would hand the same buffers to two
/// readers (risk R8).  `lend()`/`refill()` already do the work; what they do NOT do is name an owner,
/// and a scheduler that interleaves prefills needs one.
///
/// The rule this encodes:
///   * a batched prefill segment ACQUIRES the loan (it calls `lend`);
///   * a verify window may NOT start while the loan is held (it calls `refill` first);
///   * a hand-over may NOT start while the loan is held (`serve_swap::run`'s `return_loan` step
///     refills it, so the holder must give it up before the swap, not during it);
///   * the holder is cleared when the slot's `finish_prefill` returns it, or when the slot dies.
///
/// ---- S3.2b: what "gives it up" now MEANS, and what does not change. ---------------------------------
///
/// What does NOT change is the shape of this object.  The loan is still ONE resource with ONE holder,
/// because there is still one engine thread and one batched segment at a time (risk R8: two prefills
/// lending the same cache tail would hand the same buffers to two readers).  Ownership, the wait queue and
/// the deferral rule are exactly S3.1e-2's.
///
/// What DOES change is that `release()` no longer implies "the experts are back in the cache".  Under the
/// lazy rule (`prefill_loan::LoanLedger`, lever 5) releasing the loan means the rows stay out and the
/// layout stays carved, so the NEXT holder - possibly a different slot, possibly the same one on the next
/// request - inherits a loan that is already taken.  That is where the 4.95 GiB per stage goes: the second
/// request's `lend()` finds its range already non-resident and marks nothing, and its `finish_prefill`
/// returns nothing.
///
/// Two consequences the driver has to keep straight, and does:
///   * `held()` is about the BUFFERS, not about the rows.  A slot may not start a batched segment while
///     another holds the buffers (`may_lend`), and no window may run while it does (`may_read_cache`) -
///     both unchanged, because a carved range is still a carved range.
///   * `release()` is therefore NOT a residency event, and nothing in the scheduler may treat it as one.
///     The residency question is answered per row by `host_res`, which is why a decode window over a
///     released-but-unrefilled loan is correct: it routes the rows that are out to the CPU pool.
///
/// The counters exist so the owner can see the reuse the lazy loan is supposed to produce: a process whose
/// `acquires` is far larger than its `refills` is one where the loan is being shared instead of rebuilt.
struct Loan {
    int64_t owner = kNoId;
    /// Bookkeeping only - the scheduler's decisions never read these.
    int64_t acquires = 0;      ///< batched segments that took the loan
    int64_t releases = 0;      ///< loans given up (a lazy return, or an eager refill)
    int64_t handoffs = 0;      ///< acquires that followed a release by a DIFFERENT slot

    bool held() const { return owner != kNoId; }
    bool held_by(int64_t id) const { return owner == id; }
    /// May `id` start a batched prefill segment?  Only nobody else holds it.
    bool may_lend(int64_t id) const { return owner == kNoId || owner == id; }
    /// May a step that must not run over a lent cache run now?  (a verify window, a hand-over)
    bool may_read_cache(int64_t /*id*/) const { return owner == kNoId; }
    bool acquire(int64_t id) {
        if (owner != kNoId && owner != id) return false;
        if (owner == kNoId) {
            ++acquires;
            // The event the lazy loan makes cheap: one slot's prompt ended, another's starts.  Eager mode
            // pays two full refills and two lends for that; lazy mode pays one release, one acquire that
            // marks almost nothing, and one return that copies nothing.
            if (last_owner_ != kNoId && last_owner_ != id) ++handoffs;
        }
        owner = id;
        return true;
    }
    bool release(int64_t id) {
        if (owner == kNoId) return true;
        if (owner != id) return false;   // never release somebody else's loan
        last_owner_ = owner;
        owner = kNoId;
        ++releases;
        return true;
    }

private:
    int64_t last_owner_ = kNoId;
};

/// Does this step need the loan?  Only the batched prompt read does: a short part goes through the
/// verify windows (`--short-read`), and a window refills the loan before it runs.
inline bool step_needs_loan(Step s) { return s == Step::prefill; }

// --------------------------------------------------- how much decode one turn buys (S3.10) -----
//
// THE OWNER'S PROBLEM.  A slot's turn used to be exactly ONE verify window.  With MTP on, a window
// accepts 1..8 tokens, so a slot emitted ~1-3 tokens and then the scheduler moved on to the next
// conversation.  The swap is not free - it is a 237 MB-2.25 GB save+restore (risk R10) - so paying it
// every 2 tokens spends a request's whole budget on memcpy and the interleaving costs more than it
// buys.  `--decode-tokens N` says how many tokens a slot may generate before it gives the session
// back, which is the dial the owner asked for.
//
// WHY A TOKEN BUDGET AND NOT A WINDOW BUDGET.  The user's unit is tokens ("let it write 10 tokens"),
// and a window's yield varies with acceptance, so a fixed window count would not be a fixed amount of
// work.  The bound is therefore measured in tokens ACTUALLY PRODUCED, which is also what `max_new`
// and the wire count.
//
// `budget` is `--decode-tokens`: 0 keeps 0.1.30's behaviour (one window per turn), 1 is the same
// thing, and N asks for up to N tokens.  `produced` is how many tokens the REQUEST has already
// emitted and `max_new` its cap.  The answer is how many tokens a turn may still emit - never more
// than the request owes.  0 means "this request is over", which is 0.1.30's `produced_n >= max_new`
// guard stated once; the caller turns it into `Step::finished`.
inline int64_t decode_step_tokens(int64_t budget, int64_t produced, int64_t max_new) {
    const int64_t left = max_new - produced;          // tokens this request still owes
    if (left <= 0) return 0;
    if (budget <= 1) return 1;                        // today's behaviour: exactly one window
    const int64_t want = budget < left ? budget : left;
    return want;
}

/// Has a decode turn produced enough to hand the session over?  `budget` as above; `produced_before`
/// is the count when the turn started, `produced_now` the count after the window that just ran.
/// EOS, `max_new` and a cancel end the request inside `run_decode_step`, so this predicate only ever
/// decides "another window, or give the session back".
inline bool decode_turn_done(int64_t budget, int64_t produced_before, int64_t produced_now) {
    if (budget <= 1) return true;                     // one window per turn, as it always was
    return produced_now - produced_before >= budget;
}

// ------------------------------------------------------------------- per-slot cancellation ------

/// Which slot a bare `STOP` names.  0.1.30's meaning is "stop the running one", and it must stay
/// exactly that: the newest STARTED slot that is actually running.  Only when nothing is running does
/// it fall back to the newest queued one (the request that would run next).  `Registry::newest_id()`
/// already encodes that order; this wrapper exists so the driver's cancel routing is one named call
/// and the test can pin it against the registry's own answer.
inline int64_t bare_stop_target(const Registry& reg) { return reg.newest_id(); }

/// Is `id` the slot the live session reflects?  A step may only run for the mounted slot (§5.3).
/// `kNoId` on either side is never a match: "nothing is mounted" is not permission to step.
inline bool is_mounted(int64_t mounted_id, int64_t id) {
    return mounted_id != kNoId && id != kNoId && mounted_id == id;
}

// --------------------------------------------------------------------- the watchdog (R3) -------

/// Risk R3: today one stalled request `std::abort()`s the process, and with N slots that kills the
/// other N-1 conversations mid-answer.  The rule: a slot that stalled alone gets `ERR <id>` and its
/// slot is destroyed; the process aborts only when EVERY active slot stalled, because that is the one
/// case where the engine itself (not one request) is wedged.
///
/// `beats` is the process-wide heartbeat counter (`core::Progress::beats`), which every step of every
/// slot beats.  A slot is stalled when the counter has not moved since that slot last ran AND the
/// slot has been running for longer than the limit.  `running_ms` is how long the slot has been
/// inside its current step; `beats_moved` is whether the heartbeat advanced since the driver last
/// looked at this slot.
struct SlotWatch {
    int64_t id = kNoId;
    int64_t step_started_ms = 0;   ///< when the current step began (0 = not inside a step)
    uint64_t beats_at_start = 0;   ///< the heartbeat when it began
};

inline bool slot_stalled(const SlotWatch& w, int64_t now_ms, uint64_t now_beats, int64_t limit_ms) {
    if (limit_ms <= 0) return false;                 // STRATA_WATCHDOG_S=0
    if (w.step_started_ms == 0) return false;        // not inside a step: nothing to stall on
    if (now_beats != w.beats_at_start) return false; // somebody moved: not this slot's stall alone
    return now_ms - w.step_started_ms >= limit_ms;
}

/// The stalled slots, in registry order.  Empty means the watchdog stays quiet.
inline std::vector<int64_t> stalled_slots(const std::vector<SlotWatch>& w, int64_t now_ms,
                                          uint64_t now_beats, int64_t limit_ms) {
    std::vector<int64_t> out;
    for (const SlotWatch& s : w)
        if (slot_stalled(s, now_ms, now_beats, limit_ms)) out.push_back(s.id);
    return out;
}

/// Does the process abort?  Only when every slot the driver is watching stalled - the honest
/// generalisation of 0.1.30's rule, which is the same question with one slot.
inline bool watchdog_aborts(const std::vector<int64_t>& stalled, size_t watched) {
    return watched > 0 && stalled.size() >= watched;
}

/// The list for the abort message ("slot 3, slot 7" - the owner has to know WHICH request wedged the
/// engine, which is the whole point of naming requests).
inline std::string stalled_list(const std::vector<int64_t>& stalled) {
    std::string s;
    for (size_t i = 0; i < stalled.size(); ++i) {
        if (i) s += ", ";
        s += "slot " + std::to_string((long long) stalled[i]);
    }
    return s;
}

/// R3's second half: "a slot that stalls ALONE gets `ERR <id>` + slot destruction".
///
/// The watchdog above can only see a step that is IN FLIGHT, and with one engine thread there is at
/// most one of those - so a step that wedges wedges every slot, and aborting is 0.1.30's correct
/// answer.  The case the watchdog cannot see is a slot that is ACTIVE (it holds a permit and a
/// conversation) but never gets a step: the driver keeps deferring it.  That is a driver wedge, not a
/// GPU wedge, and killing the process for it would destroy the other conversations' work as well.
///
/// The bound is deliberately NOT `starve_ms`: a legitimate fairness wait is a few hundred
/// milliseconds, and a legitimate wait behind a 36 k-token read is tens of seconds.  `limit_ms` here
/// is the watchdog's limit multiplied by the driver's own factor (the caller passes it), so this only
/// fires on something that would otherwise hang a client forever.
inline bool slot_starved_to_death(int64_t waited_ms, int64_t limit_ms) {
    return limit_ms > 0 && waited_ms >= limit_ms;
}

// ================ hold, don't reject: the WAIT QUEUE (stage 4, requirement 3 / D5) =========
//
// THE OWNER'S ASK, verbatim from prompt.md: "Currently if there is no budget to park a conversation
// or different circumstances requests get rejected with an error. That is bad, instead, those
// requests should be put on hold and be executed as soon as there are ressources free instead of
// cancelled."
//
// WHAT THE ENGINE DID BEFORE THIS SECTION.  Every "no" the driver could give was the SAME kind of
// no: `refuse_line(...)` -> `ERR <id> <why>` -> the request is over.  But the reasons are not alike:
//
//   * "all --serve-slots are busy" is true for a moment.  A conversation finishes, and it is false.
//   * "not enough free RAM for another conversation" is true until one finishes or one is pruned.
//   * "the slot mid-read cannot be saved YET" is true until its read ends (`ParkRefusal::not_saveable`).
//   * "a prompt of 30 000 tokens does not fit --max-context 8192" is true FOREVER.
//
// Only the last kind may be an error.  Everything else is a WAIT, and a wait needs four things this
// section provides: an ORDER (who goes first), a BOUND (when a wait becomes an honest timeout),
// WAKE POINTS (what makes the engine look at the queue again), and VISIBILITY (so "why has my
// request not started?" is answerable from /status, /slots and the log instead of from a guess).
//
// WHY IT IS A SEPARATE SECTION AND NOT A CHANGE TO `Refuse`.  `Refuse`, `SwapResult`, `Gate` and
// `ParkRefusal` are the engine's existing vocabularies for "no", and every one of them already
// distinguishes some of the cases (S3.6 split the hand-over refusal into `wait_park` and
// `end_incoming` for exactly this reason).  Deleting or re-numbering them would break
// `serve_swap_test`, the driver's call sites and the log wording the owner greps for.  So the policy
// here is a CLASSIFICATION of those existing values plus the queue itself: one place that says, for
// every "no" the engine can produce, whether it is a wait or an error.
//
// STILL PURE.  No clock of its own (every method takes `now_ms` from the driver's one timeline, like
// `Registry::pick`), no I/O, no engine state, no CUDA.  `src/program/serve_driver_test.cpp` runs the
// whole thing, including an end-to-end simulation of a refused request that later runs.

/// What the driver should DO with a request it cannot run right now.
enum class Hold : uint8_t {
    run_now,  ///< nothing blocks it: admit/step it
    wait,     ///< blocked by something that comes back: keep it queued, wake it later
    error,    ///< blocked forever: answer `ERR` now, because waiting would only hide it
};

inline const char* hold_name(Hold h) {
    switch (h) {
        case Hold::run_now: return "run-now";
        case Hold::wait: return "wait";
        case Hold::error: return "error";
    }
    return "?";
}

/// Why a request is waiting.  Every value here names a resource that is held by somebody else and
/// will be given back; that is what makes the wait a wait rather than a refusal.
enum class Wait : uint8_t {
    none,             ///< not waiting
    slots_full,       ///< all --serve-slots are active (a conversation ending frees one)
    ram,              ///< no free host RAM for another conversation image
    registry_full,    ///< no registry row free for another waiter
    handover_not_yet, ///< the mounted slot cannot be saved YET: it is mid-read (`not_saveable`)
    loan,             ///< another slot holds the one prompt loan (risk R8)
    image_exclusive,  ///< an image request owns the one M-RoPE table (risk R7)
    engine_busy,      ///< (server side) the engine said no for a temporary reason
};

inline const char* wait_reason(Wait w) {
    switch (w) {
        case Wait::none: return "none";
        case Wait::slots_full: return "slots-full";
        case Wait::ram: return "ram";
        case Wait::registry_full: return "registry-full";
        case Wait::handover_not_yet: return "handover-not-yet";
        case Wait::loan: return "prompt-loan";
        case Wait::image_exclusive: return "image-alone";
        case Wait::engine_busy: return "engine-busy";
    }
    return "?";
}

/// The same reason as a sentence, for the ERR/DONE line and for /status.  Kept next to `wait_reason`
/// so the log, the wire and the test cannot drift apart.
inline std::string wait_text(Wait w) {
    switch (w) {
        case Wait::slots_full: return "every --serve-slots is busy";
        case Wait::ram: return "there is no free RAM for another conversation yet";
        case Wait::registry_full: return "the request registry is full";
        case Wait::handover_not_yet: return "the conversation that holds the session is mid-read and cannot be saved yet";
        case Wait::loan: return "another request holds the prompt loan";
        case Wait::image_exclusive: return "an image request is running alone (one position table)";
        case Wait::engine_busy: return "the engine is busy";
        case Wait::none: return "nothing";
    }
    return "unknown";
}

// ---- the classification of EVERY existing refusal site --------------------------------------
//
// One rule, stated once, so a call site cannot quietly decide on its own.  Anything not listed here
// is a bug: if the engine grows a new "no", it must appear in this table and in the test that
// enumerates it.

/// `admit_one`'s refusals.  `no_parking` is the only permanent one: with the conversation cache off
/// there is no way to hand the session to a second conversation at all, so a concurrent engine would
/// destroy every request it pre-empted (risk R2/R12).  The other three are resources that come back.
inline Hold hold_for(Refuse r) {
    switch (r) {
        case Refuse::none: return Hold::run_now;
        case Refuse::slots_full: return Hold::wait;
        case Refuse::ram: return Hold::wait;
        case Refuse::registry_full: return Hold::wait;
        case Refuse::no_parking: return Hold::error;
    }
    return Hold::error;
}

/// The hand-over guard (S3.6).  `budget_too_small` can never change at runtime - the snapshot is
/// bigger than the whole budget - so it stays an error; `not_saveable` clears when the running read
/// ends, so it is a wait.  `SwapResult` is the driver's own view of the same two answers.
inline Hold hold_for(serve_swap::ParkRefusal r) {
    switch (r) {
        case serve_swap::ParkRefusal::none: return Hold::run_now;
        case serve_swap::ParkRefusal::not_saveable: return Hold::wait;
        case serve_swap::ParkRefusal::budget_too_small: return Hold::error;
    }
    return Hold::error;
}

inline Hold hold_for(SwapResult r) {
    switch (r) {
        case SwapResult::mounted: return Hold::run_now;
        case SwapResult::wait_park: return Hold::wait;
        case SwapResult::end_incoming: return Hold::error;
        case SwapResult::fatal: return Hold::error;
    }
    return Hold::error;
}

/// The step gate.  `re_read` is NOT a wait: it is a recovery the driver performs immediately, and
/// putting it on a queue would only delay a request that is already runnable.  `end` means the
/// conversation is gone, which no amount of waiting brings back.
inline Hold hold_for(Gate g) {
    switch (g) {
        case Gate::run: return Hold::run_now;
        case Gate::re_read: return Hold::run_now;
        case Gate::end: return Hold::error;
    }
    return Hold::error;
}

/// The re-read bound (S3.8).  A fifth reset is a livelock, not a wait: the request cannot be parked,
/// so every hand-over erases it and it would restart forever.  Permanent.
inline constexpr Hold kRereadLimitHold = Hold::error;

/// A prompt that does not fit the context can never run on this engine, at any load: it is the one
/// refusal the client can fix by sending a shorter prompt.  0.1.30's `+ 8` slack is kept, because the
/// engine's own check (`prep_request`) uses it and the server's context guard must agree with it.
inline bool prompt_never_fits(int64_t prompt_tokens, int64_t max_new, int64_t max_context,
                              int64_t slack = 8) {
    return max_context <= 0 || prompt_tokens + max_new + slack > max_context;
}

/// A request whose own snapshot price exceeds the WHOLE parking budget can never be parked, so on a
/// concurrent engine it can never be pre-empted, so it can never share the machine.  This is the
/// "cannot fit even an empty machine" case: the budget is not something another request gives back,
/// it is the ceiling of the configuration.  An unsized budget (0) means parking is off, which
/// `hold_for(Refuse::no_parking)` already answers.
inline bool price_never_fits(uint64_t price_bytes, uint64_t budget_bytes) {
    return budget_bytes != 0 && price_bytes > budget_bytes;
}

/// The SAME price as `slot_image_bytes`, WITHOUT the two caps.  It exists because of a subtlety:
/// `slot_image_bytes` clamps a request to the budget ("never price a slot above the whole budget, or
/// admission refuses everything forever"), which is right for the RAM gate but makes the permanent
/// check above unable to fire.  So the permanent question is asked against the UNCAPTED figure: "what
/// would this conversation cost if nothing clamped it?"  If that is more than the budget, this
/// request can never be parked at any load, and waiting will not change that.
inline uint64_t slot_price_uncapped(const SlotCost& c, uint64_t prompt_tokens, uint64_t max_new) {
    if (c.per_token != 0) return (prompt_tokens + max_new) * c.per_token;
    if (c.ceiling_bytes != 0) return c.ceiling_bytes;
    if (c.budget_bytes != 0 && c.cache_slots != 0) return slot_estimate(c.budget_bytes, c.cache_slots);
    return 0;
}

/// `--serve-slots >= 2` with parking off: permanent for EVERY request, not a wait.  Same predicate as
/// the startup rule `parking_off_refuses_slots`, restated as a hold so the classification is total.
inline Hold hold_for_slots(int slots, bool cache_enabled) {
    return parking_off_refuses_slots(slots, cache_enabled) ? Hold::error : Hold::run_now;
}

/// The one question `admit_one` asks, as data, so the ordering of the checks - and therefore the
/// reason a request waits - is testable rather than an accident of the call site.  The order is
/// deliberate: a permanent reason must be named BEFORE a temporary one, or a request that can never
/// run sits in the queue for ten minutes and then times out with a message that blames the load.
struct Admit {
    int64_t prompt_tokens = 0;
    int64_t max_new = 0;
    int64_t max_context = 0;
    bool prompt_known = true;   ///< GENI: the prompt lives in a file, so its length is not on the line
    bool cache_enabled = true;  ///< the conversation cache is on (parking exists)
    bool have_telemetry = true; ///< the platform reports free RAM
    uint64_t avail_bytes = 0;   ///< free RAM
    uint64_t floor_bytes = 0;   ///< --conversation-cache-min-free-mib
    uint64_t price_bytes = 0;   ///< what THIS request's image will cost (`slot_image_bytes`)
    uint64_t uncapped_price = 0;///< the same price with no budget/ceiling clamp (0 = unknown)
    uint64_t held_bytes = 0;    ///< what the live slots already hold
    uint64_t budget_bytes = 0;  ///< the parking budget in force
    bool cap_free = true;       ///< an active-slot permit is free
    bool row_free = true;       ///< a registry row is free
    bool image_ok = true;       ///< R7: no image request is running alone
};

/// Decide, for one queued request, whether it runs now, waits (and why), or is an error (and why).
///
/// THE PERMANENT TESTS, and why there are exactly three of them:
///
///   1. parking off on a concurrent engine (`Refuse::no_parking`) - the configuration cannot host two
///      conversations at all, so every request here is permanent.  Waiting changes nothing.
///   2. a prompt that does not fit `--max_context` - the client can only fix it by sending a shorter
///      prompt, and it deserves to be told that now rather than after a hold timeout.
///   3. a request that fails the RAM test even if EVERY live slot gave back everything it was priced
///      for.  This is the "cannot fit even an empty machine" case, and it is the one that makes the
///      wait queue honest: waiting can only ever reduce `held_bytes` and grow `avail_bytes`, so a
///      request that is short by more than the whole held figure will never fit.  Ask the SAME
///      predicate (`admit_fits_ram`) with `held` moved into `avail` - one rule, two questions, so the
///      two can never disagree about the floor or the telemetry.
///
/// What is deliberately NOT here: a conversation whose snapshot would exceed the whole parking budget.
/// That one CAN run - alone, never pre-empted - and S3.6's startup line already promises exactly that
/// ("it will REFUSE every hand-over between two conversations this long and serve them one at a time").
/// Refusing it at admission would take a working long conversation away from a user because of a
/// concurrency limit it never asked for.  Its permanence belongs to the HAND-OVER
/// (`hold_for(ParkRefusal::budget_too_small) == Hold::error`), where the engine says so with both
/// numbers and leaves the mounted slot alone.  `price_never_fits` and `slot_price_uncapped` are
/// exported anyway, because S4.3's prefill/decode split needs to ask the question up front.
inline Hold admit_decision(const Admit& a, Wait& why) {
    why = Wait::none;
    // 1. permanent first.
    if (!a.cache_enabled) return Hold::error;              // Refuse::no_parking (R2/R12)
    if (a.prompt_known && prompt_never_fits(a.prompt_tokens, a.max_new, a.max_context)) return Hold::error;
    if (!admit_fits_ram(a.have_telemetry, a.avail_bytes + (a.held_bytes > ~0ull - a.avail_bytes
                                                               ? ~0ull - a.avail_bytes : a.held_bytes),
                        a.floor_bytes, a.price_bytes, 0))
        return Hold::error;                    // it would not fit even an EMPTY machine
    // 2. then the resources that come back.
    if (!a.image_ok) { why = Wait::image_exclusive; return Hold::wait; }
    if (!a.cap_free) { why = Wait::slots_full; return Hold::wait; }
    if (!a.row_free) { why = Wait::registry_full; return Hold::wait; }
    if (!admit_fits_ram(a.have_telemetry, a.avail_bytes, a.floor_bytes, a.price_bytes, a.held_bytes)) {
        why = Wait::ram;
        return Hold::wait;
    }
    return Hold::run_now;
}

// ---- the bound: when a wait becomes an honest timeout ---------------------------------------
//
// A wait with no bound is a hung client.  A bound that is too short is the old bug with a delay
// before it.  The default is deliberately LONGER than any real queue on this box: the owner's own
// worst case is a 36 k-token read at ~500 tok/s, which is ~70 s, and a request may queue behind two
/// of those.  `0` means "never give up": the client's own `STOP <id>` (or its disconnect) is the only
// thing that ends the wait, which is what a batch job wants.
inline constexpr int64_t kDefaultHoldMs = 600000;   // 10 minutes

inline bool hold_expired(int64_t waited_ms, int64_t hold_ms) {
    return hold_ms > 0 && waited_ms >= hold_ms;
}

/// THE WATCHDOG AND A WAITER.  `slot_stalled` reads the process heartbeat, and a heartbeat only moves
/// inside a step.  A waiter is by definition NOT inside a step - it has no session, no loan and no
/// row - so the watchdog can never see it, and it must never be killed by the watchdog either (that
/// would be R3's "one stalled request aborts the process" in a new costume).  The hold bound is the
/// ONLY thing that can end a wait on its own; `STOP <id>` and the client's disconnect are the others.
inline bool watchdog_sees_waiters() { return false; }

/// The ERR for an expired wait.  Named so the test pins the wording and the owner can grep it.  It
/// says what the request waited FOR, which is the difference between "the server is broken" and "the
/// server is full, and here is what is full".
inline std::string hold_expired_line(int64_t id, int64_t waited_ms, Wait why, int64_t hold_ms) {
    return "request " + std::to_string((long long) id) + " waited " +
           std::to_string((long long) waited_ms) + " ms and never got a slot (" + wait_text(why) +
           "); the hold limit is " + std::to_string((long long) hold_ms) +
           " ms (--hold-ms, 0 = wait forever). Raise --serve-slots, give the engine more RAM, or "
           "send fewer requests at once.";
}

/// The line printed once when a request STARTS waiting.  Without it, "hold, don't reject" is
/// indistinguishable from a hang: the client sees a slow first token and the log says nothing.
inline std::string hold_line(int64_t id, Wait why, size_t queue_depth) {
    return "strata serve: slot " + std::to_string((long long) id) + " waits: " + wait_text(why) +
           " (waiting=" + std::to_string((long long) queue_depth) + ")";
}

/// The line printed when a waiter is finally let through.  It carries the wait, because §7.2's rule
/// is that a request which waited must report it, or its `prompt_ms` reads like a slowdown.
inline std::string hold_admitted_line(int64_t id, int64_t waited_ms) {
    return "strata serve: slot " + std::to_string((long long) id) + " ran after waiting " +
           std::to_string((long long) waited_ms) + " ms";
}

/// `WAIT <n> <oldest_ms> <reason>` - the engine's hold queue, for `serve/server.py`'s /status and
/// /slots.  Untagged (it is about the process, not one request) and emitted ONLY on the tagged wire,
/// so `--serve-slots 0` never sends it.  A reader that does not know the line ignores it, which is
/// the same rule `SLOT` follows.
///
/// NOTE FOR THE NEXT READER: this formatter lives here rather than in `serve_proto.hpp`, which is
/// otherwise the only place wire lines are built.  It is here because stage 4's wait queue is the
/// driver's feature and the file ownership for S4.2 does not include `serve_proto.hpp`; S4.3, which
/// owns the protocol surface, should move it there if the line survives.
inline std::string wait_line(size_t n, int64_t oldest_ms, Wait why) {
    return "WAIT " + std::to_string((long long) n) + " " + std::to_string((long long) oldest_ms) +
           " " + wait_reason(why);
}

// ---- the queue itself -----------------------------------------------------------------------
//
// THE ORDER, stated because "FIFO" alone is not a rule when two requests are not alike:
//
//   1. A request that ALREADY holds a conversation outranks one that has never run
//      (`kPriorityResumed`).  The first has tokens on the wire and a parked branch; refusing it costs
//      the user an answer they are half-reading.  The second has lost nothing yet.  This is the only
//      priority the queue has, and it is a documented exception rather than a knob.
//   2. Among equals, strict FIFO by arrival (`seq`).  No length-based shortcut, no "shortest job
//      first": a scheduler whose order depends on the prompt would let a burst of long prompts starve
//      a short one forever, and it would make the order unguessable from the log.
//   3. A cancelled waiter is never dispatched.  It is answered where it is cancelled (`STOP <id>`),
//      so it cannot be "run" and then cancelled, which would emit tokens to a client that asked us to
//      stop.
inline constexpr int kPriorityNew = 0;
inline constexpr int kPriorityResumed = 1;

/// One request sitting in the hold queue.  Plain data: the driver owns the raw request line, this
/// owns the DECISION state (why it waits, since when, whether it has been cancelled).
struct Waiter {
    int64_t id = kNoId;
    Wait reason = Wait::none;
    int64_t since_ms = 0;       ///< when it started waiting (the hold bound measures from here)
    int64_t seq = 0;            ///< arrival order, the FIFO tie-break
    int priority = kPriorityNew;
    int64_t prompt_tokens = 0;
    int64_t max_new = 0;
    uint64_t price_bytes = 0;   ///< what it will cost if it runs (for the log and /slots)
    bool image = false;         ///< GENI: R7's exclusivity applies to it
    bool cancelled = false;     ///< STOP <id> seen while it waited
};

/// Is `a` served before `b`?  The whole ordering rule, in one predicate the test can enumerate.
inline bool wait_before(const Waiter& a, const Waiter& b) {
    if (a.priority != b.priority) return a.priority > b.priority;
    return a.seq < b.seq;
}

class WaitQueue {
public:
    explicit WaitQueue(int64_t hold_ms = kDefaultHoldMs, size_t cap = kDefaultCap)
        : hold_ms_(hold_ms < 0 ? 0 : hold_ms), cap_(cap < 1 ? 1 : cap) {}

    static constexpr size_t kDefaultCap = 64;

    int64_t hold_ms() const { return hold_ms_; }
    size_t cap() const { return cap_; }
    size_t size() const { return v_.size(); }
    bool empty() const { return v_.empty(); }
    bool full() const { return v_.size() >= cap_; }

    /// A duplicate id never enters: two live requests with one id would make the wire unparseable
    /// (the rule `Registry::add` already applies).  Returns false and changes nothing.  The same for a
    /// FULL queue: the queue is bounded by contract, and an unbounded one is not a wait queue, it is
    /// a memory leak with a nicer name.  The caller answers that request with an ERR (`Refuse::
    /// registry_full` is the closest existing reason) rather than dropping it silently.
    bool enqueue(int64_t id, Wait why, int64_t now_ms, int priority = kPriorityNew,
                 int64_t prompt_tokens = 0, int64_t max_new = 0, uint64_t price_bytes = 0,
                 bool image = false) {
        if (id == kNoId || find(id) != nullptr || full()) return false;
        Waiter w;
        w.id = id; w.reason = why; w.since_ms = now_ms; w.seq = ++seq_;
        w.priority = priority; w.prompt_tokens = prompt_tokens; w.max_new = max_new;
        w.price_bytes = price_bytes; w.image = image;
        v_.push_back(w);
        ++queued_total_;
        return true;
    }

    const Waiter* find(int64_t id) const {
        for (const Waiter& w : v_) if (w.id == id) return &w;
        return nullptr;
    }
    Waiter* find(int64_t id) {
        for (Waiter& w : v_) if (w.id == id) return &w;
        return nullptr;
    }
    bool contains(int64_t id) const { return find(id) != nullptr; }

    /// The next request to serve, or nullptr.  NOT the front of the vector: the priority rule can put
    /// a resumed request ahead of an older new one.
    const Waiter* head() const {
        const Waiter* best = nullptr;
        for (const Waiter& w : v_)
            if (!w.cancelled && (best == nullptr || wait_before(w, *best))) best = &w;
        return best;
    }

    /// Record why `id` waits.  True when the reason CHANGED, which is the driver's cue to print
    /// `hold_line` once rather than on every pass of a loop that runs thousands of times a second.
    bool set_reason(int64_t id, Wait why) {
        Waiter* w = find(id);
        if (w == nullptr || w->reason == why) return false;
        w->reason = why;
        return true;
    }
    /// The reason currently recorded, or `none`.
    Wait reason_of(int64_t id) const {
        const Waiter* w = find(id);
        return w == nullptr ? Wait::none : w->reason;
    }

    /// A resource came back (a slot finished, a park landed, the loan was released, a branch was
    /// pruned).  Clear every reason so the next pass re-evaluates from scratch: a waiter parked
    /// behind RAM may now be blocked by the cap instead, and a reason that is never re-derived is how
    /// a stale "waiting for RAM" line outlives the RAM problem.
    void wake_all(int64_t /*now_ms*/ = 0) {
        for (Waiter& w : v_) w.reason = Wait::none;
        ++wakeups_;
    }
    /// Wake ONE waiter (the resource that was named by id, e.g. the slot that just finished).
    void wake(int64_t id) {
        if (Waiter* w = find(id)) { w->reason = Wait::none; ++wakeups_; }
    }

    int64_t waited_ms(const Waiter& w, int64_t now_ms) const {
        return now_ms >= w.since_ms ? now_ms - w.since_ms : 0;
    }
    int64_t waited_ms(int64_t id, int64_t now_ms) const {
        const Waiter* w = find(id);
        return w == nullptr ? 0 : waited_ms(*w, now_ms);
    }
    /// The longest wait in the queue - the number /status and the activity line show.
    int64_t oldest_wait_ms(int64_t now_ms) const {
        int64_t m = 0;
        for (const Waiter& w : v_) { const int64_t t = waited_ms(w, now_ms); if (t > m) m = t; }
        return m;
    }
    /// The reason to REPORT: the head's, because that is the one the next request is stuck behind.
    Wait head_reason() const {
        const Waiter* w = head();
        return w == nullptr ? Wait::none : w->reason;
    }

    /// Who has waited past the bound.  Returned, not removed: the caller has to answer each one with
    /// `ERR <id>` before it forgets it, and forgetting first is how a client hangs.
    std::vector<int64_t> expired(int64_t now_ms) const {
        std::vector<int64_t> out;
        for (const Waiter& w : v_)
            if (hold_expired(waited_ms(w, now_ms), hold_ms_)) out.push_back(w.id);
        return out;
    }

    /// Take a waiter out (it is being admitted, it timed out, or it was answered).
    bool pop(int64_t id) {
        for (size_t i = 0; i < v_.size(); ++i)
            if (v_[i].id == id) { v_.erase(v_.begin() + (std::ptrdiff_t) i); return true; }
        return false;
    }

    /// `STOP <id>` while the request still waits.  It is removed from the queue AND reported, so the
    /// caller can answer the client (a DONE with finish `cancel`, the same answer a request cancelled
    /// at admission gets) instead of leaving it hanging on a pipe.
    bool cancel(int64_t id, Waiter& out) {
        Waiter* w = find(id);
        if (w == nullptr) return false;
        w->cancelled = true;
        out = *w;
        pop(id);
        ++cancelled_total_;
        return true;
    }
    bool cancelled(int64_t id) const {
        const Waiter* w = find(id);
        return w != nullptr && w->cancelled;
    }

    // ---- counters: the queue must be COUNTABLE, or "it is queued" and "it is lost" look the same.
    int64_t queued_total() const { return queued_total_; }
    int64_t admitted_total() const { return admitted_total_; }
    int64_t expired_total() const { return expired_total_; }
    int64_t cancelled_total() const { return cancelled_total_; }
    int64_t refused_total() const { return refused_total_; }
    int64_t wakeups() const { return wakeups_; }
    /// The caller calls this as it lets a waiter through, so `queued == admitted + expired +
    /// cancelled + refused` is an invariant the test can assert and the summary line can print.
    void note_admitted() { ++admitted_total_; }
    void note_expired() { ++expired_total_; }
    void note_refused() { ++refused_total_; }

    /// `waiting=2 oldest=4120ms reason=ram` - the activity line's tail, and /status's.  Empty queue ->
    /// `waiting=0`, which is the fact that distinguishes "nothing is queued" from "the engine stopped
    /// reporting".
    std::string line(int64_t now_ms) const {
        std::string s = "waiting=" + std::to_string((long long) v_.size());
        if (v_.empty()) return s;
        s += " oldest=" + std::to_string((long long) oldest_wait_ms(now_ms)) + "ms";
        s += " reason=" + std::string(wait_reason(head_reason()));
        return s;
    }

private:
    std::vector<Waiter> v_;
    int64_t hold_ms_ = kDefaultHoldMs;
    size_t cap_ = kDefaultCap;
    int64_t seq_ = 0;
    int64_t queued_total_ = 0, admitted_total_ = 0, expired_total_ = 0, cancelled_total_ = 0,
            refused_total_ = 0, wakeups_ = 0;
};

// ------------------------------------------------------------------ startup reporting ------------

/// The startup line's shape.  Kept as a formatter so the `--serve-slots` report and the test agree on
/// one spelling, and so the serial path's output is untouched (this only runs when concurrent()).
inline std::string driver_line(int slots, int64_t starve_ms, bool swaps_on, size_t budget_bytes,
                               size_t cache_slots, uint64_t per_token_bytes = 0,
                               int64_t decode_tokens = 0, int64_t hold_ms = kDefaultHoldMs) {
    std::string s = "strata serve: concurrent driver on (" + std::to_string(slots) + " slots; starve " +
                    std::to_string((long long) starve_ms) + " ms; slot swap " + (swaps_on ? "on" : "off");
    // S4.2: say what happens to a request that cannot run now.  This is the line that tells the owner
    // whether the engine holds or rejects, and what it holds for.
    s += "; hold " + std::string(hold_ms > 0
                                     ? std::to_string((long long) hold_ms) + " ms then ERR"
                                     : "forever (only STOP ends a wait)");
    // S3.10: say what a slot's TURN is.  The default is one verify window (~1-3 tokens with MTP), and
    // every turn boundary is a possible hand-over, so this number is how often the engine is willing to
    // pay a save+restore.  Printing it is the only way to tell a run that swaps every 2 tokens from one
    // that swaps every 20.
    s += "; decode " + std::string(decode_tokens > 0 ? std::to_string((long long) decode_tokens) + " tokens/turn"
                                                     : "1 window/turn");
    if (budget_bytes == 0 || cache_slots == 0) {
        s += "; conversation cache OFF";
    } else {
        s += "; parking budget " + std::to_string((long long) (budget_bytes >> 20)) + " MiB / " +
             std::to_string((long long) cache_slots) + " entries";
        // Say HOW a slot will be priced. The old line printed only `budget / entries`, which reads
        // like a reservation per slot - and it is not: parking charges what a conversation actually
        // holds. When the engine has a measured rate it prices each request at its own length.
        if (per_token_bytes != 0)
            s += " (" + std::to_string((long long) per_token_bytes) +
                 " B per context token, so a request is charged for its own length; nothing is"
                 " reserved per slot)";
        else
            s += " (" + std::to_string((long long) (slot_estimate(budget_bytes, cache_slots) >> 20)) +
                 " MiB per slot estimate until a conversation has been parked and measured)";
    }
    s += ")";
    return s;
}

}  // namespace strata::program::serve_driver
