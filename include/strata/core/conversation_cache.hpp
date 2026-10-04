// CPU-only ownership and matching policy for --serve's parked conversations.
// Token equality, image identity, and steering mode are all required for reuse.
//
// TWO DIFFERENT CACHES LIVE IN THIS FEATURE - do not confuse them:
//   * strata::program::conv_cache::eviction_victim() picks which CHECKPOINTS
//     inside ONE conversation branch survive (a prefix chain with a pinned root);
//   * this class picks which WHOLE CONVERSATIONS survive.  A parked entry is one
//     client's complete prefix - live ids, its checkpoints, its K/V pages and its
//     draft K/V - so an agent that alternates between its own prompt and its
//     subagents' prompts keeps all of them warm instead of rebuilding one per
//     switch.
//
// Retention policy here: a FIXED number of slots (default 8) plus a byte budget,
// and the slot that leaves is the LEAST RECENTLY USED conversation, not the one
// parked first.  Each entry carries a use stamp from the cache's monotonic
// clock; `best()` (a hit), `take()` (a mount) and `put()` (a park) all advance
// the clock, and `put()`/`touch()` stamp the entry with the new value.  A hot
// conversation therefore survives arbitrarily many cold neighbours - every
// switch back to it refreshes it, and the cold ones are what get pruned.  The
// old first-in-first-out order is still the degenerate case: a set of entries
// that nobody has touched leaves in park order, because `lru_victim()` breaks
// stamp ties on the earlier (older parked) index.
//
// There is deliberately NO pinned entry at this level.  The pin in
// program/conv_cache.hpp protects the shared root of ONE chain, where every
// retained item is a prefix of every other one; parked conversations are
// independent branches, so pinning one would be a guess about which client comes
// back and would permanently steal a slot from the client that is active.  LRU
// is the whole policy here.
//
// Parking still serves ONE request at a time: it saves the re-read, it does not
// execute two requests concurrently.
#pragma once

#include "strata/core/conversation_buffer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <utility>
#include <vector>

namespace strata::core {

/// `SavedConversation::owner` for an entry no running request owns.  Request ids are the client's to
/// choose and the engine's own are never negative, so -1 is free and cannot alias a real slot.
constexpr int64_t kNoOwner = -1;

struct ConversationImageKey {
    int64_t start = 0;
    uint64_t hash = 0;
    bool operator==(const ConversationImageKey&) const = default;
};

struct ConversationCheckpoint {
    std::vector<int32_t> ids;
    std::vector<ConversationImageKey> imgs;
    std::vector<uint8_t> gdn, ple, tails, dead, block_pos;
    uint64_t used = 0; // upstream root-pinned/LRU checkpoint retention
    // **A `--layer-split` checkpoint: one part per LATER stage, in stage order.**  The checkpoint
    // itself is CUDA0's (the first stage's) part.  A stage part carries that stage session's own
    // running state for its layer carve, and its `imgs` is empty - the ids are what the state is
    // sized by.  Both the checkpoint chain and a parked conversation use this shape; see
    // `core::ConversationStageSet` in conversation_snapshot.hpp for what a snapshot requires.
    std::vector<ConversationCheckpoint> stage_parts;

    size_t bytes() const {
        size_t n = ids.capacity() * sizeof(int32_t) + imgs.capacity() * sizeof(ConversationImageKey) +
               gdn.capacity() + ple.capacity() + tails.capacity() + dead.capacity() + block_pos.capacity() +
               stage_parts.capacity() * sizeof(ConversationCheckpoint);
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
};

// Identity-layout K/V pages and completed indexer rows. For streamed layers the
// source is the authoritative host pool, NOT the replaceable VRAM slots.
struct ConversationKv {
    int format = 0;
    int64_t cells = 0, heads = 0, head_dim = 0, page_size = 0, pooled_rows = 0, idx_dim = 0;
    ConversationBuffer k, v, k_scale, v_scale, pooled;
    size_t bytes() const {
        return k.bytes() + v.bytes() + k_scale.bytes() + v_scale.bytes() + pooled.bytes();
    }
};

struct ConversationKvReuse {
    std::vector<ConversationKv> kv;
    // Original image extent for validation, and the earliest subsequent rewrite.
    int64_t captured_tokens = 0, unchanged_tokens = 0;
    size_t bytes() const {
        size_t n = kv.capacity() * sizeof(ConversationKv);
        for (const auto& layer : kv) n += layer.bytes();
        return n;
    }
};

// One LATER stage of a `--layer-split` inside a parked conversation: the state that stage's
// `SessionState` owns, plus the K/V of the QSA layers inside its carve.  Index i of
// `SavedConversation::stage_parts` is stage i, in the same order the engine walks its stages;
// the FIRST stage is CUDA0's session, which is the snapshot's main `live`/`kv` and is not here.
//
// `state.ids` carries the live tokens (the same shape a checkpoint-chain stage part has, see
// `ConversationCheckpoint::stage_parts`), `state.imgs` stays empty, and `kv` holds exactly
// `qsa_alloc` entries - never a draft entry, which lives in the main `kv`.
//
// A parked image keeps its stage running state HERE, in `stage_parts[i].state`, and leaves
// `live.stage_parts` empty; the checkpoints retained in `checkpoints` keep the upstream chain's
// shape, where each one's stage running state is `checkpoints[j].stage_parts[i]`.
struct ConversationStageSnapshot {
    int64_t layer_lo = 0, layer_hi = 0;   // the carve it was captured from; restore requires the same
    ConversationCheckpoint state;
    std::vector<ConversationKv> kv;

    size_t bytes() const {
        size_t n = state.bytes() + kv.capacity() * sizeof(ConversationKv);
        for (const auto& k : kv) n += k.bytes();
        return n;
    }
};

struct SavedConversation {
    // Runtime compatibility only; NOT a model/weights identity or disk schema.
    std::array<int64_t, 18> geometry{};
    // The session's layer carve the image was captured from ([0, n_layers) on one GPU); restore requires the same.
    int64_t layer_lo = 0, layer_hi = 0;
    ConversationCheckpoint live;
    std::vector<ConversationCheckpoint> checkpoints;
    std::vector<ConversationKv> kv; // main layers followed by the draft layer
    // Empty for a single-GPU image - the format and the byte accounting are then exactly as before.
    std::vector<ConversationStageSnapshot> stage_parts;
    bool cvec = true;
    // S3.9 THE CLAIM.  The request whose branch this is, while that request is still running;
    // `kNoOwner` for a branch that belongs to nobody in particular - a finished conversation's, or
    // anything on a server that cannot have two conversations at once.
    //
    // WHY IT EXISTS.  A parked entry is ONE object: its live branch, its checkpoint chain and its
    // K/V pages.  A new request looks a branch up by PREFIX (`best()`), which happily matches one of
    // another slot's checkpoints, and the mount then `take()`s the WHOLE entry - so reusing a
    // 9 849-token checkpoint destroyed the 10 394-token live branch a different, still-waiting slot
    // had parked for itself.  That slot then asked for its own branch back, found nothing, and was
    // ended.  The owner's log says it exactly: `park: slot 14 parked 10394 tokens`, then
    // `restored 9849 tokens (checkpoint) … parked=4`, then `slot 14's parked branch (10394 tokens) is
    // no longer in the cache`, with `evictions=0` for the whole run - nothing was ever pruned.
    //
    // A claim is not a pin.  LRU may still evict a claimed entry when the budget needs the room (the
    // owner then gets the honest "pruned while it waited" answer), and the claim is released the
    // moment its slot goes away, so stage 2's prefix reuse across requests of the same chat still
    // works.  It only says: do not hand this entry to a DIFFERENT request while its owner is alive.
    int64_t owner = kNoOwner;

    size_t bytes() const {
        size_t n = live.bytes() + checkpoints.capacity() * sizeof(ConversationCheckpoint) +
                   kv.capacity() * sizeof(ConversationKv) + stage_parts.capacity() * sizeof(ConversationStageSnapshot);
        for (const auto& c : checkpoints) n += c.bytes();
        for (const auto& k : kv) n += k.bytes();
        for (const auto& part : stage_parts) n += part.bytes();
        return n;
    }
};

template<class Token>
int64_t conversation_prefix(const ConversationCheckpoint& c, const std::vector<Token>& prompt,
                            const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    // The last prompt token always starts the next verify window.
    if (n == 0 || n >= prompt.size() || !std::equal(c.ids.begin(), c.ids.end(), prompt.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

// THE SAME BRANCH, NOT A PREFIX OF IT.  `conversation_prefix` deliberately refuses a match that
// consumes the whole prompt, because a request that RESUMES still has to read its last token into the
// next verify window.  That rule is wrong for the other lookup a driver makes: handing a slot back the
// conversation that was parked when it was pre-empted.  There the token list is not a prompt to read,
// it is the sequence the session must be put back into - and it is exactly equal, never shorter.
//
// This is not a corner case, it is the common one, and getting it wrong is what produced the owner's
// "this slot's conversation was lost while it was decode": a parked 80-token branch sat in the cache
// with 113 MiB against a 10 048 MiB budget, the driver asked for it with the prefix rule, got 0,
// mounted nothing, and the step gate correctly ended the request.  Nothing about the budget.
//
// EXACT, and only exact.  A shorter checkpoint of the same branch must not be mounted onto a slot
// that is mid-decode: the tokens between the checkpoint and the branch are ones it has ALREADY sent,
// and restoring the older state would make it generate them again.
template<class Token>
int64_t conversation_same_branch(const ConversationCheckpoint& c, const std::vector<Token>& branch,
                                 const std::vector<ConversationImageKey>& images) {
    const size_t n = c.ids.size();
    if (n == 0 || n != branch.size() || !std::equal(c.ids.begin(), c.ids.end(), branch.begin())) return 0;
    size_t j = 0;
    for (const auto& image : images) {
        if (image.start >= (int64_t) n) continue;
        if (j == c.imgs.size() || !(c.imgs[j++] == image)) return 0;
    }
    if (j != c.imgs.size()) return 0;
    return (int64_t) n;
}

class ConversationCache {
public:
    struct Match {
        size_t index = 0;
        int64_t tokens = 0;
        bool live = false;
    };

    // The number of parked conversations `--conversation-cache-slots` defaults
    // to.  Enough for one agent plus the subagent prompts it switches between,
    // and small enough that the byte budget - not the slot count - is normally
    // what bounds RAM.
    static constexpr size_t default_slots = 8;

    ConversationCache(size_t budget, size_t slots) : budget_(budget), slots_(slots) {}
    bool enabled() const { return budget_ != 0 && slots_ != 0; }
    size_t bytes() const { return bytes_ + reuse_.bytes(); }
    size_t size() const { return entries_.size(); }
    size_t slots() const { return slots_; }
    /// The byte budget actually in force. The driver may cap the requested one to what the machine can hand
    /// over, so this - not the option value - is what the startup line reports.
    size_t budget() const { return budget_; }
    size_t evictions() const { return evictions_; }
    // The cache's monotonic use clock. Exposed for the startup/telemetry line
    // only; stamps are meaningful relative to each other, never as wall time.
    uint64_t clock() const { return clock_; }
    // The last-use stamp of the entry at `index`. For tests and telemetry.
    uint64_t use(size_t index) const { return stamps_.at(index); }

    // Retain only the restored K/V buffers, not duplicate running checkpoints.
    // This optimization never evicts a parked conversation to make itself fit.
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens) {
        reuse_ = {};
        ConversationKvReuse candidate{std::move(kv), tokens, tokens};
        if (enabled() && candidate.bytes() <= budget_ - bytes_) reuse_ = std::move(candidate);
    }
    void limit_reuse(int64_t first_dirty) {
        reuse_.unchanged_tokens = std::min(reuse_.unchanged_tokens, first_dirty);
        if (reuse_.unchanged_tokens <= 0) reuse_ = {};
    }
    ConversationKvReuse take_reuse() { return std::exchange(reuse_, {}); }
    size_t retained_bytes() const { return reuse_.bytes(); }
    bool can_fit(size_t incoming, size_t held = 0) const {
        return enabled() && held <= budget_ && incoming <= budget_ - held &&
               entries_.size() < slots_ && bytes() <= budget_ - held - incoming;
    }

    // Least recently used, ties on the earlier index.  Pure and deterministic so
    // the CPU-only test can walk the scenarios by hand; the caller owns the
    // stamps and erases the returned index.  Empty input returns 0.
    static size_t lru_victim(const uint64_t* stamps, size_t n) {
        size_t v = 0;
        for (size_t i = 1; i < n; ++i)
            if (stamps[i] < stamps[v]) v = i;
        return v;
    }

    // NOT const: a match is a use, and it credits the entry that answered.
    // Without that, a conversation the client keeps switching back to would age
    // out exactly like one nobody asks for.
    //
    // `requester` is the id of the request asking, or `kNoOwner` when nobody in particular is asking
    // (a serial server, or a lookup made before any slot owns anything).  An entry CLAIMED by a
    // different running request is not offered: a mount `take()`s the whole entry, and that would
    // destroy the branch its owner parked for itself.  See `SavedConversation::owner`.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec,
               int64_t requester = kNoOwner) {
        const Match found = find(prompt, images, cvec, requester);
        if (found.tokens > 0) touch(found.index);
        return found;
    }

    /// STAGE 4 S4.2: the same prefix search, WITHOUT the stamp.  `best()` is "finding it is using it" -
    /// it advances the entry's use stamp, because a real mount is about to happen.  A caller that is
    /// merely ASKING whether a conversation is already parked must not refresh anything: the hold
    /// queue re-asks every waiter on every promotion pass, and a probe that touched the stamps would
    /// keep a cold conversation warm purely because it is queued, which is not a use.
    ///
    /// It is `const` for exactly that reason, and it applies the same claim rule as `find()` (an entry
    /// claimed by a DIFFERENT running request is not offered to this one).
    template<class Token>
    Match probe(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                bool cvec, int64_t requester = kNoOwner) const {
        return find(prompt, images, cvec, requester);
    }

    /// The lookup a HAND-OVER makes, as opposed to the one a new request makes: find the parked branch
    /// that IS this slot's sequence, so the session can be put back into it.  `best()` cannot answer
    /// this - its prefix rule requires the prompt to be LONGER than the branch, and a pre-empted
    /// slot's branch is exactly equal to what it parked - so a driver that used `best()` here would
    /// find its own conversation sitting in the cache, mount nothing, and lose the request.  See
    /// `conversation_same_branch`.
    ///
    /// It searches the entries' LIVE branches only, never their checkpoint chains, and that is
    /// deliberate rather than an oversight.  A restore always puts the whole entry back
    /// (`live = image.live.ids`), so matching a checkpoint whose ids happen to equal the query while
    /// its entry's live branch is LONGER would restore a sequence further along than the one the slot
    /// stopped at - the slot would resume past tokens it never generated.  A hand-over wants the exact
    /// branch or nothing.
    ///
    /// NOT const, for the same reason `best()` is not: finding it is using it.
    template<class Token>
    Match best_exact(const std::vector<Token>& branch, const std::vector<ConversationImageKey>& images,
                     bool cvec, int64_t requester = kNoOwner) {
        Match found;
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            if (claimed_by_other(i, requester)) continue;
            const int64_t n = conversation_same_branch(e.live, branch, images);
            if (n > found.tokens) found = {i, n, true};
        }
        if (found.tokens > 0) touch(found.index);
        return found;
    }

    // Credit an entry without mounting it: the caller resumed from a parked
    // conversation's checkpoint chain rather than its whole image, which is
    // still evidence that the branch is live.
    void touch(size_t index) {
        stamps_.at(index) = ++clock_;
    }

    /// Is this entry claimed by a running request other than `requester`?  Such an entry may be read
    /// and evicted, but it may not be MOUNTED by somebody else: mounting takes the whole entry out of
    /// the cache, and the entry is the other request's own parked state.
    bool claimed_by_other(size_t index, int64_t requester) const {
        const int64_t o = entries_.at(index).owner;
        return o != kNoOwner && requester != kNoOwner && o != requester;
    }

    /// Release every claim held by a request that has gone away.  Its parked branch stays in the cache
    /// and becomes reusable by the next request of that chat - stage 2's prefix mechanism, which must
    /// keep working.  Idempotent, and cheap: the cache holds at most `slots_` entries.
    void release_owner(int64_t id) {
        for (auto& e : entries_) if (e.owner == id) e.owner = kNoOwner;
    }

    /// The claim on the entry at `index`, or `kNoOwner`.
    int64_t owner(size_t index) const { return entries_.at(index).owner; }

    SavedConversation take(size_t index) {
        SavedConversation out = std::move(entries_.at(index));
        bytes_ -= out.bytes();
        erase_at(index);
        // The mount is a use, but the entry leaves the cache with its stamp.
        // Advance the clock anyway so the snapshot parked in its place is
        // strictly newer than everything still parked.
        ++clock_;
        return out;
    }

    // Reserve before allocating a snapshot. held is an incoming image removed
    // with take() but still alive during the exchange; count it against RAM too.
    bool make_room(size_t incoming, size_t held = 0) {
        if (!enabled() || held > budget_ || incoming > budget_ - held) return false;
        if (bytes() > budget_ - held - incoming) reuse_ = {};
        while (!entries_.empty() && (entries_.size() >= slots_ || bytes_ > budget_ - held - incoming)) {
            // Prune the conversation that has gone unused the longest, so the
            // prompts the client actually alternates between survive the ones it
            // abandoned. Ties leave in park order, which is the old FIFO answer.
            //
            // S3.9: but only among the conversations that belong to NOBODY, when any such thing
            // exists.  A claimed entry is a RUNNING request's own parked state: evicting it does not
            // cost that request a re-read, it destroys the conversation and the request gets ended.
            // Unclaimed entries are finished conversations - warm, but safe to lose.  When every
            // entry is claimed there is nothing safe to evict, and LRU is the answer again: the
            // budget is the hard limit and the owner's "pruned while it waited" line names it.
            const size_t victim = victim_to_prune();
            bytes_ -= entries_[victim].bytes();
            erase_at(victim);
            ++evictions_;
        }
        return true;
    }

    /// The entry `make_room` would prune: the least recently used unclaimed one, else the least
    /// recently used overall.  Exposed for the test, which can then pin the preference without
    /// driving the byte arithmetic.
    size_t victim_to_prune() const {
        if (entries_.empty()) return 0;
        size_t v = lru_victim(stamps_.data(), stamps_.size());
        for (size_t i = 0; i < entries_.size(); ++i) {
            if (entries_[i].owner != kNoOwner) continue;
            if (entries_[v].owner != kNoOwner || stamps_[i] < stamps_[v]) v = i;
        }
        return v;
    }

    /// Park an image and claim it for `owner` - the request whose branch this is, while it is still
    /// running.  `kNoOwner` (the default) means "belongs to the cache", which is what a finished
    /// conversation's snapshot is, and what every park on a serial server is.  One signature with
    /// defaults, deliberately: an overload pair here would make `put(img, 5)` ambiguous between an
    /// `owner` and a `held`.
    bool put(SavedConversation&& image, int64_t owner = kNoOwner, size_t held = 0) {
        const size_t n = image.bytes();
        if (!make_room(n, held)) return false;
        image.owner = owner;
        entries_.push_back(std::move(image));
        stamps_.push_back(++clock_);
        bytes_ += n;
        return true;
    }

private:
    // Mounting and pruning both drop an entry; one place does it so entries_ and
    // its stamp mirror can never disagree. Callers account for bytes_ themselves,
    // because take() measures the moved-out image.
    void erase_at(size_t index) {
        entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
        stamps_.erase(stamps_.begin() + (std::ptrdiff_t) index);
    }

    // The search itself, unchanged from the FIFO cache: the longest exact
    // prefix, ties to the entry parked last (the deque is still in park order -
    // touching an entry never moves it).
    template<class Token>
    Match find(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec,
               int64_t requester = kNoOwner) const {
        Match best;
        // Ties prefer the most recently parked branch. The caller prefers its
        // already-active state when that offers the same prefix length.
        for (size_t i = entries_.size(); i-- > 0;) {
            const auto& e = entries_[i];
            if (e.cvec != cvec) continue;
            // S3.9: an entry another running request parked for ITSELF is not up for reuse.  A hit
            // here would be followed by `take()`, which removes the whole entry - live branch,
            // checkpoints and K/V - and the other request would then be handed a session it cannot
            // rebuild.  Skipping it costs this request a shorter prefix; stealing it costs the other
            // one its conversation.
            if (claimed_by_other(i, requester)) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return best;
    }

    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0;
    uint64_t clock_ = 0; // monotonic use clock; 0 is never a stamp
    std::deque<SavedConversation> entries_; // park order, oldest first (unchanged)
    // entries_[i]'s last-use stamp, kept contiguous so lru_victim() takes a plain
    // pointer. Same length as entries_ - erase_at() is the only removal path.
    std::vector<uint64_t> stamps_;
    ConversationKvReuse reuse_;
};

} // namespace strata::core
