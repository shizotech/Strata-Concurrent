// include/strata/program/serve_swap.hpp - stage 3 S3.1d: MOUNT and UNMOUNT one conversation slot in
// the ONE live session, as an ordered operation a scheduler can call between steps.
//
// WHY THIS FILE EXISTS.  docs/STAGE3-CONCURRENCY.md §2.3 settles by arithmetic that there is exactly
// ONE live session - the one the captured CUDA graphs point at - plus a bounded set of parked slot
// images in host RAM. §5.3 turns that into a rule: "Changing which conversation is active is a
// save/restore, never a re-capture." So the ONLY mechanism for putting a second conversation into the
// session is `conversation_snapshot_save` out of it and `conversation_snapshot_restore` back into it.
// `--serve` already does exactly that for ONE slot per request (generate.cpp's mount path); this file
// makes it one named operation with a fixed order, so S3.1e's scheduler can call it between steps and
// nobody has to re-derive the ordering at the call site.
//
// WHY THE ORDER IS CODE AND NOT A COMMENT.  Three of the steps exist only to make the other two safe:
//
//   * `apply_pending(true)` before the save, because a half-applied residency table is the "silent
//     plausible tokens" failure expert_cache.hpp:135-138 warns about (§4.2, risk R9);
//   * `refill()` before the save, because the prompt loan is the tail of one expert cache and a
//     snapshot taken while it is lent describes a cache that does not exist (§4.4, risk R8);
//   * `conversation_snapshot_validate` BEFORE the save, because validation is the only step that is
//     guaranteed to write nothing (conversation_snapshot.hpp:101-108: "Invalid images are rejected
//     before any CUDA call/write"). Validate first and a bad image costs you nothing; validate after
//     the save and it costs you the outgoing conversation.
//
// A restore that fails AFTER its first write is `transfer_failed` and is fatal to the session - that
// is 0.1.30's behaviour and this file keeps it, it does not soften it.
//
// WHAT THIS FILE IS NOT.  It owns no CUDA, no session, no I/O.  Like `prefill_loan.hpp` and
// `slot.hpp`, it owns the DECISIONS and the ORDER, and the engine in `src/program/generate.cpp` owns
// the state; `run()` drives the engine's own hooks through the order so the order itself is testable
// on a CPU with a fake session (`src/program/serve_swap_test.cpp`).  The scheduler that decides WHICH
// slot to mount is S3.1e; this file is the thing it calls.
//
// ---------------------------------------------------------------------------------------------
// S3.6 — THE PARKING-COLLAPSE FIX.  A live run (`--serve-slots 3`, two ~175K-token conversations)
// showed a slot being pre-empted, its snapshot too big for the collapsed parking budget to hold, its
// conversation destroyed anyway, and then the SAME slot being stepped again while the session held
// ANOTHER conversation's K/V and positional state.  It sampled garbage, usually hit EOS at once, and
// ended silently.  Diagnosis: `.megamind/src/program/stage3-parking-collapse.md`.
//
// Two rules were missing from this file, and both are now decisions rather than hopes:
//
//   * `save_is_mandatory()` — a slot that still has work to do may not be swapped out unless its
//     state CAN be saved.  The budget question is asked with the same arithmetic the save will use
//     (`budget_of`/`fits`), BEFORE the unmount hook runs, so the answer is known before anything is
//     destroyed.  A slot with no work left needs no save and may always go.
//   * `may_step()` — a slot in a stepping phase may only step when its record still describes the
//     session.  That is §5.3 stated as a predicate the driver can assert, and it is what turns the
//     silent garbage into one named error line.
//
// `invalidate_unparked()` is now a LAST resort, not the ordinary consequence of a refused park: the
// engine refuses the hand-over first.  It survives for the case that cannot be refused — the outgoing
// slot has no work left, so nothing is lost by dropping its image, and the save itself failed.
// ---------------------------------------------------------------------------------------------
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/program/serve_proto.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <string>
#include <vector>

namespace strata::program::serve_swap {

using serve_proto::kNoId;

// ------------------------------------------------------------------ the env switch -------------------
//
// STRATA_NO_SWAP=1: fall back to today's serial behaviour without recompiling.  It does not merely
// make the scheduler pick less often - it makes the engine's `swap_to` refuse a hand-over outright,
// so a box that turns out to swap itself to death (risk R10) can be started with the feature off and
// diagnosed.  The id-tagged wire stays on: what goes away is the mount/unmount, not the identity.
inline bool swaps_disabled_by_env() {
    const char* v = std::getenv("STRATA_NO_SWAP");
    return v != nullptr && *v != '\0' && std::string(v) != "0";
}

// ---------------------------------------------------- the conversation branch one slot owns ----------
//
// §3.5 lists what has to move from "process" to "slot".  Two of those lists must be kept apart:
//
//   * the SESSION's sequence state (`live`, `live_imgs`, `checks`, `cvec`) already travels with a
//     slot, because the vehicle for it is the ConversationCache image: `SavedConversation` carries
//     exactly those four and `conversation_snapshot_restore` puts them back.  They are mirrored here
//     so a slot that is mounted WITHOUT an image (still the session's own state, or being re-read)
//     keeps them, and so /slots can report them;
//   * what the image CANNOT carry lives here and nowhere else: the suffix drafter's history
//     (spec/suffix_drafter.hpp:47-48 - `hist_`/`table_` ARE the sequence), the sampling params and
//     penalty window (§3.5: applied at dispatch, never at parse time), the M-RoPE position table
//     (risk R7 - one device table for the whole process), the checkpoint LRU clock, and the drafter's
//     prompt length.
//
// `resumable` is the flag that keeps a hand-over honest.  If an unmount did NOT park the outgoing
// conversation, the session's positional cells are about to hold somebody else's tokens, so that
// slot may never be mounted again - it has to be re-read from zero.  Without this flag a later mount
// would happily restore running state over stale K/V and produce plausible garbage.
struct SlotConv {
    int64_t id = kNoId;

    // ---- the mirror of what the live session reflects while this slot is mounted ----
    std::vector<int32_t> live;                              ///< tokens the session has consumed
    std::vector<core::ConversationImageKey> live_imgs;      ///< the pictures among them
    std::vector<core::ConversationCheckpoint> checks;       ///< the prefix chain (shared by design)
    bool cvec_cached = true;                                ///< the control-vector state it was read with
    bool live_ok = false;                                   ///< `live` describes the session right now
    uint64_t check_clock = 0;                               ///< this chain's LRU clock

    // ---- what the snapshot image does not carry ----
    std::vector<int32_t> sfx_hist;                          ///< the suffix drafter's whole history
    strata::kernels::SamplerParams smpl;                    ///< this slot's sampling, re-dispatched on mount
    bool smpl_set = false;                                  ///< `smpl`/`penalty_last_n` were filled by a request
    int penalty_last_n = 0;                                 ///< the penalty window `smpl` asks for
    int pcie_num = 0;                                       ///< this slot's PCIe share of the missed experts
    bool mrope_image = false;                               ///< R7: non-identity positions
    std::vector<int32_t> mrope;                             ///< 3 x cells; EMPTY = the identity table
    int64_t prompt_len = 0;                                 ///< mtp.set_prompt_len() for this slot

    // ---- where its parked image is, and whether it may be mounted at all ----
    /// NOT a `ConversationCache` index: the cache is a deque that prunes, so an index is not stable
    /// across a `put`/`make_room` and must not be held.  It is here for the `SLOT` line's shape and
    /// for a caller that wants to name the entry it took; the engine leaves it -1 and finds a
    /// conversation again through `ConversationCache::best()` on its token prefix, as stage 2 does.
    int64_t parked_index = -1;
    int64_t parked_bytes = 0;
    bool resumable = true;                                  ///< false once its cells were handed over

    /// The image's own conversation branch, adopted on a mount (the same four fields the 0.1.30
    /// mount path assigns at generate.cpp:4820-4823).
    void adopt_image(const core::SavedConversation& image) {
        live = image.live.ids;
        live_imgs = image.live.imgs;
        checks = image.checkpoints;
        cvec_cached = image.cvec;
        live_ok = !live.empty();
    }
};

// ------------------------------------------------------------------ the decisions -------------------

/// Is any hand-over needed at all?  Staying on the mounted slot is the default and the cheap answer:
/// one swap is a full save+restore of a 237 MB-2.25 GB image (risk R10).
inline bool swap_needed(int64_t mounted_id, int64_t incoming_id) {
    return mounted_id != incoming_id;
}

enum class Save : uint8_t {
    none,   ///< nothing to save: no cache, or the slot has no live conversation
    park,   ///< save the outgoing conversation into the ConversationCache
};

/// What a snapshot save actually did.  `park_current` collapses all three of these into "true" for
/// 0.1.30's sake (a refusal is not an error: the request just re-reads), but a HAND-OVER cannot
/// collapse them: a branch that was not parked is a branch that may never be mounted again.
enum class Saved : uint8_t {
    stored,   ///< parked in the ConversationCache; the branch is resumable
    skipped,  ///< refused (budget, RAM admission, backoff) or nothing to save; NOT resumable
    failed,   ///< the snapshot save itself failed; the session is untouched but the swap must abort
};

/// Must the outgoing conversation be saved before the session is handed over?  This is 0.1.30's
/// `park_current` guard (generate.cpp:3921) stated as a decision: no cache, no live branch, or an
/// empty branch and there is nothing to write.
inline Save save_for(const SlotConv& out, bool cache_enabled) {
    if (!cache_enabled) return Save::none;
    if (!out.live_ok || out.live.empty()) return Save::none;
    return Save::park;
}

/// May this slot be mounted at all?  A slot whose conversation was never parked has to be re-read
/// from token 0, which is the request body's job, not the swap's.
inline bool can_mount(const SlotConv& s) { return s.resumable; }

/// Whether a hand-over has anything to look for: the slot must still be resumable, its mirror must
/// describe a real sequence, and that sequence is what the cache is asked for.
inline bool handover_seeks(const SlotConv& s) { return s.resumable && s.live_ok && !s.live.empty(); }

/// The cache index of the parked branch to mount for a hand-over, or -1 when there is none.
///
/// `best_exact`, NOT `best`.  A driver handing the session back to a slot it pre-empted wants the
/// parked branch that IS that slot's sequence.  `ConversationCache::best()` requires the token list to
/// be strictly SHORTER than the entry it matches - correct for a request that still has to read its
/// last token into the next verify window, and fatal here, because a pre-empted slot's branch is
/// exactly EQUAL to what it parked.  It answered 0 for the one conversation the cache was holding for
/// it, which is what produced the owner's "this slot's conversation was lost while it was decode":
/// 80 tokens parked at 113 MiB against a 10 048 MiB budget, the lookup found nothing, nothing was
/// mounted, the adopt hook cleared the branch, and the step gate ended a live request with an ERR
/// blaming the parking budget.  Nothing was ever too small.
///
/// `best_exact()` searches live branches only: a restore always puts the whole entry back, so
/// matching a shorter checkpoint would resume the slot past tokens it never generated.
inline int64_t seek_mount_index(core::ConversationCache& cache, const SlotConv& s) {
    if (!handover_seeks(s)) return -1;
    // S3.9: `s.id` is the requester.  A hand-over wants the entry ITS OWN slot parked, and the claim
    // rule lets it have that one even though it is claimed - while refusing an entry claimed by a
    // different running slot.
    const auto hit = cache.best_exact(s.live, s.live_imgs, s.cvec_cached, s.id);
    return hit.tokens > 0 ? (int64_t) hit.index : -1;
}

/// The ConversationCache's admission arithmetic (`conversation_cache.hpp:199-202`) as a free
/// function, so the swap can ask "will the OUTGOING save still fit while the INCOMING image is held
/// out of the cache" before it takes anything out.  `held` is that incoming image: it is no longer
/// counted in `parked_bytes`, but it is still in RAM during the exchange.
struct Budget {
    size_t bytes = 0;      ///< parked bytes, excluding the image being held
    size_t budget = 0;     ///< the cap in force (0 = parking off)
    size_t entries = 0;
    size_t slots = 0;      ///< --conversation-cache-slots (0 = off)
};
/// The cache's live accounting, read the one way the swap is allowed to read it.  `bytes()` includes
/// the retained-K/V buffers, which is exactly what `can_fit` counts, so the two agree by
/// construction (the test pins that against `ConversationCache::can_fit` itself).
inline Budget budget_of(const core::ConversationCache& c) {
    return Budget{c.bytes(), c.budget(), c.size(), c.slots()};
}
inline bool fits(const Budget& b, size_t incoming, size_t held = 0) {
    return b.budget != 0 && b.slots != 0 && held <= b.budget && incoming <= b.budget - held &&
           b.entries < b.slots && b.bytes <= b.budget - held - incoming;
}

/// Would the cache ACCEPT this snapshot at all?  This mirrors `ConversationCache::make_room` - the gate
/// `park_current` actually applies - and NOT `can_fit`: `make_room` EVICTS the least recently used
/// conversation to make space, so a full cache is not a refusal.  Only a snapshot larger than the
/// budget itself (or a budget of 0) can never be parked.  That is the question the parking-collapse
/// guard asks: "is this conversation EVER parkable here", and on the owner's box the answer for a
/// 175K-token conversation and a 1.6 GiB budget is no, permanently.
inline bool budget_has_room(const Budget& b, size_t snapshot, size_t held = 0) {
    return b.budget != 0 && b.slots != 0 && held <= b.budget && snapshot <= b.budget - held;
}

// ------------------------------------------------- the parking-collapse guard (S3.6) -------------
//
// THE BUG THIS KILLS.  A hand-over used to ask "can I park the outgoing slot?" and treat "no" as a
// consequence to live with: park_current refused, `invalidate_unparked` cleared the branch, the
// hand-over SUCCEEDED, and the outgoing slot's conversation was gone from the process.  Its request
// was still in the middle of decode, so the next pick stepped it against a session that described
// somebody else.  The log line promised "will be re-read from token 0" and nothing ever honoured it.
//
// The rule that was missing: a slot with work still to do may only be swapped out if its state can
// actually be saved.  A finished slot needs no save at all — its conversation is already in the cache
// or the client has its answer — so it may always go.
//
/// Does the outgoing slot still owe itself a step, and can it rebuild what a hand-over costs it?
/// The caller derives it from its own phase machine, because "has work" and "can re-read" are the
/// driver's business, not this file's.
///
///   * `finished`    - the request is over.  It needs no save: the client already has its answer, and
///                     a later request of the same chat finds the branch through
///                     `ConversationCache::best()` exactly as stage 2 always did.  It may always be
///                     swapped out, budget or no budget.  This is also what the serial path reports,
///                     which is what keeps `--serve-slots 0/1` behaving as 0.1.30 does.
///   * `re_readable` - mid-prompt.  Its whole prompt is still in the caller's hands, so losing the
///                     session's copy of it is survivable - but ONLY if the caller resets that request
///                     to token 0 before stepping it again.  `swap_to` says it allowed this by setting
///                     `Report::out_needs_reread`; a caller that ignores that flag is the bug.
///   * `must_park`   - mid-decode.  The tokens it generated are already on the wire and its prompt
///                     read cannot be redone without emitting them a second time, so its state MUST be
///                     saved.  If it cannot be, the hand-over is refused.
enum class Outgoing : uint8_t { finished, re_readable, must_park };

inline const char* outgoing_name(Outgoing o) {
    switch (o) {
        case Outgoing::finished: return "finished";
        case Outgoing::re_readable: return "re-readable";
        case Outgoing::must_park: return "must-park";
    }
    return "?";
}

/// Whether this hand-over MUST save the outgoing branch before it may write the session.
inline bool save_is_mandatory(Outgoing work) { return work == Outgoing::must_park; }
/// Whether the hand-over may proceed WITHOUT a save, provided the caller rebuilds the request.
inline bool save_is_optional(Outgoing work) { return work != Outgoing::must_park; }

/// Why a hand-over away from a `must_park` slot was refused.  The caller has to tell these apart:
///
///   * `budget_too_small` - the snapshot can NEVER fit this cache.  No amount of waiting changes
///     that, so the incoming request is answered with an ERR naming both numbers (the contract's
///     rule: refused, not starved and not spun on).
///   * `not_saveable` - the session is between two chunks of a prompt read, so there is no whole
///     branch for the cache to take: `park_current`'s own first guard is `live_ok`, and
///     `prep_request` keeps it false until `finish_request` sets it again.  A DECODE slot's branch is
///     published first (`publish_decode_branch`), so this only happens mid-prompt - where waiting
///     genuinely helps, because the read ends shortly.  The incoming request is deferred, not
///     refused: ERRing it would punish a client for a race that resolves itself.
enum class ParkRefusal : uint8_t { none, budget_too_small, not_saveable };

inline const char* park_refusal_name(ParkRefusal r) {
    switch (r) {
        case ParkRefusal::none: return "none";
        case ParkRefusal::budget_too_small: return "budget-too-small";
        case ParkRefusal::not_saveable: return "not-saveable";
    }
    return "?";
}

/// The budget question, asked BEFORE anything is destroyed.  `snapshot` is what this branch's
/// snapshot would cost (`conversation_snapshot_bytes`), `held` is the incoming image already taken
/// out of the cache but still in RAM.  The engine asks the cache's own arithmetic through
/// `budget_of`/`budget_has_room`, which mirror `ConversationCache::make_room` - the gate
/// `park_current` actually applies.  It is deliberately NOT `can_fit`: `make_room` prunes the least
/// recently used conversation to make space, so a full cache is not a refusal, and a guard that
/// refused there would turn a normal eviction into a stalled conversation.
///
/// `saveable` is the session's answer to "is there a whole branch down there to take at all"
/// (`save_for(out, enabled) == Save::park`).  A session mid-request says no, and no budget makes it
/// yes - which is why the two refusals are different questions, not one.
struct ParkCheck {
    Outgoing work = Outgoing::finished;
    bool mandatory = false;   ///< the outgoing slot is mid-decode: its state has to be saved
    bool saveable = true;     ///< the session holds a complete branch the cache could take
    bool fits = false;        ///< the budget would accept a snapshot of `snapshot` bytes
    ParkRefusal refusal = ParkRefusal::none;
    size_t snapshot = 0;      ///< the bytes the save would need
    size_t budget = 0;        ///< the parking budget in force (0 = parking off)
    size_t held = 0;          ///< the incoming image's bytes, still resident during the exchange
    size_t parked = 0;        ///< bytes already parked (LRU pruning may free them, so this is context)
    size_t entries = 0;       ///< parked conversations, and the cap on them
    size_t slots = 0;         ///< --conversation-cache-slots (0 = parking off)

    /// May the hand-over proceed?  Never-parkable is a refusal only for a slot that cannot re-read.
    bool ok() const { return !mandatory || (saveable && fits); }
};
inline ParkCheck park_fits(const Budget& b, size_t snapshot, size_t held, Outgoing work, bool saveable) {
    ParkCheck c;
    c.work = work;
    c.mandatory = save_is_mandatory(work);
    c.saveable = saveable;
    c.snapshot = snapshot;
    c.budget = b.budget;
    c.held = held;
    c.parked = b.bytes;
    c.entries = b.entries;
    c.slots = b.slots;
    c.fits = budget_has_room(b, snapshot, held);
    if (c.mandatory && !saveable) c.refusal = ParkRefusal::not_saveable;
    else if (c.mandatory && !c.fits) c.refusal = ParkRefusal::budget_too_small;
    return c;
}

/// The one-line explanation the engine prints when it refuses.  Kept here so the wording and the
/// numbers are the same everywhere the refusal appears, and so the test can pin it.  Both numbers the
/// owner asked for are on the line: the snapshot the conversation needs, and the budget in force.
inline std::string park_refusal_line(int64_t out_id, int64_t in_id, const ParkCheck& c) {
    std::string s = "strata serve: park: slot " + std::to_string((long long) out_id) + " snapshot " +
           std::to_string((long long) (c.snapshot >> 20)) + " MiB vs parking budget " +
           std::to_string((long long) (c.budget >> 20)) + " MiB (" +
           std::to_string((long long) (c.held >> 20)) + " MiB held for slot " +
           std::to_string((long long) in_id) + ", " + std::to_string((long long) c.entries) +
           "/" + std::to_string((long long) c.slots) + " entries, " +
           std::to_string((long long) (c.parked >> 20)) + " MiB parked)";
    if (!c.saveable)
        s += " - the session is mid-request, so there is no whole branch to save yet";
    else
        s += " - a conversation this long can NEVER be parked here";
    return s + " - hand-over REFUSED, slot " + std::to_string((long long) out_id) + " keeps the session";
}

/// THE INVARIANT, as a predicate (§5.3): a slot may only step when the mounted session describes it.
/// `live_ok` is exactly that fact — it is set when the branch was adopted or synced from the session
/// and cleared the moment the branch was invalidated.  A slot whose record does not describe the
/// session must be re-read from token 0 (the request body's job) or ended with an ERR; it must never
/// run a window or a prompt segment against somebody else's K/V.
inline bool may_step(const SlotConv& c) { return c.live_ok; }

/// The consequence of NOT saving.  The slot keeps its ids and its checkpoints as bookkeeping, but it
/// may never be mounted again: its positional cells belong to whoever mounts next.
///
/// After S3.6 the engine reaches this in ONE case only: the outgoing slot has NO work left (so no
/// request can be stepped against a stale session) and its save failed or was refused.  Reaching it
/// for a slot that still has work is the bug, and `save_is_mandatory` is what prevents it.
inline void invalidate_unparked(SlotConv& out) {
    out.live_ok = false;
    out.resumable = false;
    out.parked_index = -1;
    out.parked_bytes = 0;
}


/// R7.  `d_mrope` is ONE device table for the process.  The rule this encodes: the table describes
/// the MOUNTED slot, so a mount must (re)upload its owner's table, and a slot that is not mounted
/// must never be the owner.  `upload_needed` is true unless the device already holds exactly what
/// the incoming slot needs (same owner, and that owner is the mounted slot).
inline bool mrope_upload_needed(int64_t device_owner, int64_t incoming_id) {
    return device_owner != incoming_id;
}
/// The exclusivity check, as an acquired resource rather than an assumption: a non-identity table
/// may only be handed to another slot by a swap that unmounts its holder in the same operation.
/// A hand-over that leaves a non-identity holder mounted is refused.
inline bool mrope_exclusive_ok(const SlotConv& mounted, int64_t incoming_id, bool unmounts_mounted) {
    if (!mounted.mrope_image) return true;                 // the identity belongs to nobody
    if (mounted.id == incoming_id) return true;            // same slot: nothing to give up
    return unmounts_mounted;
}

// ------------------------------------------------------------------ the order --------------------

enum class Step : uint8_t {
    drain_residency,   ///< apply_pending(true): no expert swap may be half-applied across a hand-over
    return_loan,       ///< refill(): the prompt loan is the tail of one cache; give it back first
    park_guard,        ///< S3.6: can the OUTGOING branch be saved at all?  Refuse before destroying it
    validate,          ///< conversation_snapshot_validate - writes NOTHING, so it goes before the save
    unmount,           ///< conversation_snapshot_save of the outgoing branch into the cache
    mount,             ///< conversation_snapshot_restore of the incoming image into the session
    draft_kv,          ///< mtp.kv_restore(upto): the drafter's ring may hold cells past the resume point
    adopt,             ///< live / live_imgs / checks / cvec / checkpoint clock -> the incoming slot
    device_state,      ///< the per-slot device state the session depends on: M-RoPE, cvec, sampling
    count,
};

/// The canonical order, as data.  `run()` walks exactly this list, and the test asserts the trace it
/// produced equals it - so the order cannot drift out of the comment.
inline constexpr Step kOrder[] = {
    Step::drain_residency, Step::return_loan, Step::park_guard, Step::validate, Step::unmount,
    Step::mount, Step::draft_kv, Step::adopt, Step::device_state,
};

inline const char* step_name(Step s) {
    switch (s) {
        case Step::drain_residency: return "drain-residency";
        case Step::return_loan: return "return-loan";
        case Step::park_guard: return "park-guard";
        case Step::validate: return "validate";
        case Step::unmount: return "unmount";
        case Step::mount: return "mount";
        case Step::draft_kv: return "draft-kv";
        case Step::adopt: return "adopt";
        case Step::device_state: return "device-state";
        case Step::count: break;
    }
    return "?";
}

/// Is `a` strictly earlier than `b` in the canonical order?  The engine's own call sites and the
/// replay test both assert through this, so "the ordering rules" is a predicate and not prose.
inline bool runs_before(Step a, Step b) { return (int) a < (int) b; }

/// Where a hand-over stopped, and what it did to the session.  The distinction matters: only
/// `restore` may leave the session half-written, and only `restore` is fatal to it (that is
/// 0.1.30's rule, kept verbatim - generate.cpp:4802 "a failure here is fatal, never permission to
/// decode from a partially restored session").
///
/// `park` is S3.6's refusal: the OUTGOING slot still has work and its state cannot be saved, so the
/// hand-over is stopped before anything is destroyed.  It is recoverable in the sense that the
/// session is untouched - the caller answers the INCOMING request with an ERR and the outgoing slot
/// keeps running.
enum class Fault : uint8_t { none, residency, park, validation, save, restore, adopt, device };

inline const char* fault_name(Fault f) {
    switch (f) {
        case Fault::none: return "none";
        case Fault::residency: return "residency";
        case Fault::park: return "park";
        case Fault::validation: return "validation";
        case Fault::save: return "save";
        case Fault::restore: return "restore";
        case Fault::adopt: return "adopt";
        case Fault::device: return "device";
    }
    return "?";
}

/// A validation failure is recoverable exactly the way 0.1.30 treats it: the image is discarded and
/// the request falls back to the outgoing branch's existing prefix.
inline bool fault_is_recoverable(Fault f) { return f == Fault::validation || f == Fault::save || f == Fault::residency || f == Fault::park; }
/// A restore failure poisons the session; nothing may decode from it afterwards.
inline bool fault_poisons_session(Fault f) { return f == Fault::restore; }

/// Which steps this hand-over runs.  The two drains are NOT in the plan: they always run, because
/// "nothing is pending" and "we forgot to drain" look identical from the outside.
struct Plan {
    bool save = false;           ///< unmount the outgoing branch into the ConversationCache
    bool mount = false;          ///< restore an incoming image into the session
    bool draft_kv = false;       ///< mtp.kv_restore after the restore
    bool adopt = false;          ///< move the per-slot mirror in
    bool device_state = false;   ///< re-establish the mounted slot's device state
    /// S3.6: may the outgoing branch be saved at all?  Runs BEFORE `validate` and before `unmount`,
    /// so a conversation that cannot be parked is never destroyed.  Last in the struct (not in the
    /// order) so 0.1.30's five-field aggregate `Plan{...}` literals keep their meaning.
    bool park_guard = false;
};

/// The engine's side of the hand-over.  Every hook returns false to stop the swap with `err` set;
/// a hook may be empty (nothing to do).  They run on the engine thread, in `kOrder`.
struct Hooks {
    std::function<bool(std::string&)> drain_residency;
    std::function<bool(std::string&)> return_loan;
    std::function<bool(std::string&)> park_guard;
    std::function<bool(std::string&)> validate;
    std::function<bool(std::string&)> unmount;
    std::function<bool(std::string&)> mount;
    std::function<bool(std::string&)> draft_kv;
    std::function<bool(std::string&)> adopt;
    std::function<bool(std::string&)> device_state;
};

struct Report {
    bool ok = false;
    Fault fault = Fault::none;
    Step failed_at = Step::count;
    /// The restore never ran, so the live session is exactly as it was and the caller may retry the
    /// hand-over without one.  Once the restore has run - successfully or not - this is false: the
    /// session holds the incoming conversation, and a second hand-over would save THAT out as if it
    /// were the outgoing one.
    bool session_intact = true;
    /// The session may not be used for inference any more (a restore failed mid-write).
    bool session_poisoned = false;
    /// The steps that actually ran, in order.  The test asserts this equals `kOrder` filtered by the
    /// plan; the engine logs it under STRATA_TRACE.
    std::vector<Step> ran;
    /// `ms`, and the bytes moved, are the CALLER's: only the engine's hooks know them, and `run()`
    /// must not know what a snapshot costs.  The engine reports both on stderr per swap.
};

namespace detail {
inline bool call(const std::function<bool(std::string&)>& fn, Step s, Report& r, std::string& err, Fault f) {
    r.ran.push_back(s);
    if (!fn) return true;
    if (fn(err)) return true;
    r.fault = f;
    r.failed_at = s;
    r.ok = false;
    // `mount` is the step that writes the session, and everything after it implies that write
    // happened - so from `mount` on, the outgoing branch is gone and the caller must not retry the
    // hand-over.  A save only READS the session, so a failure at `unmount` or earlier leaves the
    // outgoing branch exactly as it was.  Only the restore itself can leave a session nothing may
    // decode from.
    if (runs_before(Step::mount, s) || s == Step::mount) {
        r.session_intact = false;
        r.session_poisoned = fault_poisons_session(f);
    }
    return false;
}
}  // namespace detail

/// Run one hand-over.  `run()` never touches a session, a device or stdout - it drives the hooks in
/// the order the design requires and classifies where it stopped.  That is the whole reason the
/// ordering is verifiable without a GPU.
inline Report run(const Plan& p, const Hooks& h, std::string& err) {
    Report r;
    r.ok = true;
    // 1-2. the drains, always.  A swap over a half-applied residency table or a lent cache slot is
    //      the failure R9 and R8 describe; both are silent.
    if (!detail::call(h.drain_residency, Step::drain_residency, r, err, Fault::residency)) return r;
    if (!detail::call(h.return_loan, Step::return_loan, r, err, Fault::residency)) return r;
    // 3. S3.6: the parking guard.  It runs after the loan is back (the save it is pricing needs the
    //    cache whole) and before ANYTHING that could destroy the outgoing branch.  A refusal here is
    //    the cheap one: the session has only been read, so the outgoing slot keeps it and the caller
    //    answers the incoming request instead of starving it.
    if (p.park_guard && !detail::call(h.park_guard, Step::park_guard, r, err, Fault::park)) return r;
    // 4. validate BEFORE anything is written.  This is what makes "a failed validation leaves the
    //    outgoing state intact" a property of the order rather than a hope.
    if (p.mount && !detail::call(h.validate, Step::validate, r, err, Fault::validation)) return r;
    // 5. unmount: the snapshot save.  It only READS the session, so a failure here leaves everything
    //    usable - the swap aborts and the outgoing conversation stays mounted.
    if (p.save && !detail::call(h.unmount, Step::unmount, r, err, Fault::save)) return r;
    // 6. mount: the snapshot restore.  Pre-validated above, so a failure here is a transfer failure
    //    and the session is finished for this process.
    if (p.mount && !detail::call(h.mount, Step::mount, r, err, Fault::restore)) return r;
    // 7-9. the per-slot state that the snapshot does not carry.
    if (p.draft_kv && !detail::call(h.draft_kv, Step::draft_kv, r, err, Fault::adopt)) return r;
    if (p.adopt && !detail::call(h.adopt, Step::adopt, r, err, Fault::adopt)) return r;
    if (p.device_state && !detail::call(h.device_state, Step::device_state, r, err, Fault::device)) return r;
    r.ok = true;
    r.fault = Fault::none;
    r.failed_at = Step::count;
    // A completed mount DID write the session - that was the point.  `session_intact` answers "is the
    // OUTGOING branch still there", and after a restore the honest answer is no.
    if (p.mount) r.session_intact = false;
    return r;
}

}  // namespace strata::program::serve_swap
