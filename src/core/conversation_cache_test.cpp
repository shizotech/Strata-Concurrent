#include "strata/core/conversation_cache.hpp"

#include <cstdio>
#include <cstdlib>

using namespace strata::core;

namespace {
int checks = 0;
void check(bool value, const char* description) {
    ++checks;
    if (!value) { std::fprintf(stderr, "FAIL: %s\n", description); std::exit(1); }
}
SavedConversation image(std::initializer_list<int32_t> ids, bool cvec = true) {
    SavedConversation s;
    s.live.ids = ids;
    s.live.gdn.resize(64, 7);
    s.cvec = cvec;
    return s;
}
}

// STAGE 4 S4.2: `probe()` is the hold queue's question - "does this conversation already have a
// parked branch?" - and it must NOT count as a use.  The queue re-asks every waiter on every
// promotion pass, so a probe that advanced the LRU stamps would keep a cold conversation warm purely
// because it is queued, which is not a use and would silently change which entry gets pruned.
void test_probe_is_a_use_free_lookup() {
    std::printf("\n== probe(): the hold queue's side-effect-free lookup (stage 4 S4.2) ==\n");
    {
        ConversationCache c(1u << 20, 4);
        c.put(image({1, 2, 3}));
        c.put(image({9, 8, 7}));
        const uint64_t u0 = c.use(0), u1 = c.use(1), clock = c.clock();
        // The prefix rule is the same one `find()` applies: the QUERY must be strictly longer than the
        // parked branch, because a request that resumes still reads its last token.
        check(c.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 3,
              "probe answers the same prefix search best() does");
        check(c.probe(std::vector<int32_t>{1, 2, 3}, {}, true).tokens == 0,
              "and it keeps the strict-prefix rule - an equal prompt is not a resume");
        check(c.probe(std::vector<int32_t>{9, 8, 7, 6}, {}, true).tokens == 3, "the other branch too");
        check(c.clock() == clock && c.use(0) == u0 && c.use(1) == u1,
              "and it credits NOTHING - asking is not using");
        // The contrast that makes the check non-vacuous: the very next `best()` DOES credit.
        check(c.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 3, "best() still finds it");
        check(c.clock() > clock && c.use(0) > u0, "best() credits; probe() did not");
    }
    {
        // The claim rule is the same as find()'s: another request's branch is not offered.
        ConversationCache c(1u << 20, 4);
        c.put(image({1, 2, 3}), /*owner=*/ 7);
        check(c.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true, 7).tokens == 3,
              "the claim's OWNER still gets its own branch");
        check(c.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true, 8).tokens == 0,
              "another requester does not (S3.9's claim, unchanged)");
        check(c.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 3,
              "a probe with NO requester hides nothing - S3.9's rule, so --serve-slots 0/1 is unaffected");
        c.release_owner(7);   // ... and only until its request goes away
        check(c.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true, 8).tokens == 3,
              "once released, another request may reuse the branch");
    }
    {
        ConversationCache off(0, 4);
        off.put(image({1, 2, 3}));
        check(off.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 0,
              "a disabled cache answers nothing, so the queue ranks every waiter as new");
    }
}

int main() {
    {
        ConversationBuffer bytes;
        const size_t first = ConversationBuffer::segment_bytes + 17;
        const size_t peak = bytes.allocation_peak(first);
        bytes.resize(first, 7);
        check(bytes.bytes() <= peak, "segmented payload and directory fit admitted bytes");
        uint8_t* original = nullptr;
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) { original = p; return true; });
        const size_t grown_peak = bytes.allocation_peak(first + 99);
        bytes.resize(first + 99, 9);
        bytes.visit(0, 1, [&](uint8_t* p, size_t, size_t) {
            check(p == original, "appending preserves existing payload addresses"); return true;
        });
        check(bytes.bytes() <= grown_peak, "growth including transient directory fits admission");
        std::vector<uint8_t> tail(116);
        check(bytes.read(tail.data(), ConversationBuffer::segment_bytes, tail.size()), "read spans segment boundaries");
        check(std::all_of(tail.begin(), tail.begin()+17, [](auto v){return v==7;}) &&
              std::all_of(tail.begin()+17, tail.end(), [](auto v){return v==9;}), "growth preserves prefix and initializes only suffix");
        auto copy = bytes;
        bytes.resize(ConversationBuffer::segment_bytes + 10);
        bytes.resize(first, 7); bytes.resize(first + 99, 9);
        check(bytes == copy, "equality ignores differing segmentation after rewind and regrowth");
        check(!bytes.read(tail.data(), bytes.size()-1, 2), "range check rejects a truncated payload");
        check(bytes.allocation_peak(SIZE_MAX) == SIZE_MAX, "allocation estimate rejects overflow");
    }
    {
        ConversationBuffer bytes;
        for (size_t n=1;n<=4096;++n) {
            const size_t peak = bytes.allocation_peak(n*17);
            bytes.resize(n*17,7);
            check(bytes.bytes() <= peak,"small append stays within the predicted allocation peak");
        }
        size_t segments = 0;
        bytes.visit(0,bytes.size(),[&](const uint8_t*,size_t,size_t){++segments;return true;});
        check(segments <= 4,"thousands of small turns do not create thousands of restore transfers");
    }
    {
        ConversationKv layer;
        layer.k.resize(400);
        std::vector<ConversationKv> layers;
        layers.push_back(std::move(layer));
        const size_t retained = layers.capacity()*sizeof(ConversationKv) + layers[0].bytes();
        const size_t parked = image({1,2,3}).bytes();
        ConversationCache cache(retained + parked, 4);
        check(cache.put(image({1,2,3})), "park before retaining active storage");
        cache.retain(std::move(layers), 16);
        check(cache.bytes() == retained + parked && cache.size() == 1, "active retained storage consumes bytes but no parked slot");
        cache.limit_reuse(9); cache.limit_reuse(12);
        auto reuse = cache.take_reuse();
        check(reuse.unchanged_tokens == 9, "a later continuation cannot undo a rewind's dirty boundary");
        check(cache.bytes() == parked, "taking retained buffers releases their budget accounting");
        cache.retain(std::move(reuse.kv), 16);
        check(!cache.can_fit(parked),"retained storage is counted when checking a capture without eviction");
        check(cache.retained_bytes() == retained && cache.size() == 1 && cache.evictions() == 0,
              "optional reuse admission does not evict or release anything");
        check(!cache.can_fit(SIZE_MAX) && !cache.can_fit(1,SIZE_MAX),"non-mutating reservation rejects overflow");
        check(cache.make_room(parked), "reservation can discard optional active buffers");
        check(cache.retained_bytes() == 0 && cache.size() == 1 && cache.evictions() == 0,
              "pressure drops retained storage before evicting parked conversations");
    }
    const std::vector<int64_t> a = {1, 2, 3, 4}, b = {9, 8, 7, 6};
    {
        // S3.1a: a parked conversation across a `--layer-split` carries one part per later stage.
        // The budget is only honest if those parts are counted, so pin the accounting here - this
        // test never touches CUDA, it only measures what the cache would be asked to hold.
        SavedConversation plain = image({1, 2, 3});
        SavedConversation split = image({1, 2, 3});
        ConversationStageSnapshot part;
        part.layer_lo = 4; part.layer_hi = 8;
        part.state.ids = split.live.ids;
        part.state.gdn.resize(64, 7); part.state.tails.resize(128, 9); part.state.dead.resize(32, 9);
        part.kv.emplace_back(); part.kv.back().k.resize(4096);
        split.stage_parts.push_back(std::move(part));
        check(split.bytes() > plain.bytes(), "a split snapshot's bytes include its stage parts");
        check(split.bytes() == plain.bytes() + split.stage_parts[0].bytes() +
                  sizeof(ConversationStageSnapshot),
              "the stage-part total is exactly the part plus its directory slot");
        ConversationCache cache(plain.bytes() + 8, 4);
        check(cache.put(SavedConversation{plain}), "the single-GPU image fits");
        check(!cache.put(std::move(split)), "a split image cannot slip past the budget it is not counted in");
        ConversationCache roomy(plain.bytes() * 2 + 4096, 4);
        check(roomy.bytes() == 0 && roomy.put(image({1, 2, 3})), "parked for the eviction check");
        SavedConversation chain = image({1, 2, 3});
        chain.checkpoints.push_back(chain.live);
        const size_t chain_without_parts = chain.bytes();
        ConversationCheckpoint stage_part;
        stage_part.ids = chain.live.ids;
        stage_part.gdn.resize(256, 3);
        chain.checkpoints.back().stage_parts.push_back(std::move(stage_part));
        check(chain.bytes() == chain_without_parts + chain.checkpoints.back().stage_parts[0].bytes() +
                  sizeof(ConversationCheckpoint),
              "a retained checkpoint's stage parts are counted too");
        check(roomy.put(std::move(chain)), "a split conversation with a checkpoint chain is admitted by its true size");
    }
    {
        ConversationCache cache(1024, 2);
        check(cache.put(image({1, 2, 3})), "park A");
        check(cache.put(image({9, 8, 7})), "park B");
        auto match = cache.best(a, {}, true);
        check(match.tokens == 3 && match.live, "A/B/A: recover A");
        auto restored = cache.take(match.index);
        check(restored.live.ids == std::vector<int32_t>({1, 2, 3}), "taking selected A preserves identity");
        check(cache.size() == 1 && cache.best(b, {}, true).tokens == 3, "B remains parked");
        check(cache.bytes() == image({9,8,7}).bytes(), "byte accounting after take");
        check(cache.put(std::move(restored)), "park returned A as newest");
        // CHANGED ON PURPOSE. This assertion used to read "evict oldest by slot
        // limit" / "B evicted before A": under the old first-in-first-out policy
        // B left because it was parked second. The outcome is the same but the
        // reason is not - B left because nothing has asked for it since it was
        // parked, while A was just mounted and re-parked. The blocks below pin
        // down the cases where the two policies genuinely disagree.
        check(cache.put(image({5, 6})), "the slot limit prunes the least recently used entry");
        check(cache.best(b, {}, true).tokens == 0 && cache.best(a, {}, true).tokens == 3, "idle B pruned, used A kept");
        check(cache.evictions() == 1, "eviction counter");
    }
    {
        // The owner's case: one client switching between its own prompt and its
        // subagents' prompts. The conversation that keeps being mounted must not
        // be the one that dies, however long ago it was parked.
        ConversationCache cache(1024, 2);
        check(cache.put(image({1,2,3})), "park A first");
        check(cache.put(image({9,8,7})), "park B second");
        for (int round = 0; round < 5; ++round) {
            const auto hit = cache.best(a, {}, true);
            check(hit.tokens == 3, "A answers every switch back to it");
            auto restored = cache.take(hit.index);
            check(cache.put(std::move(restored)), "A re-parks after every mount");
        }
        check(cache.size() == 2 && cache.evictions() == 0, "mounting A never reaches the slot limit");
        check(cache.use(0) < cache.use(1), "the re-parked entry is last in park order and newest in use");
        check(cache.put(image({5,6})), "the third prefix forces one prune");
        check(cache.best(b, {}, true).tokens == 0, "B - idle since it was parked - is the victim");
        check(cache.best(a, {}, true).tokens == 3, "A - parked first, used last - survives");
        check(cache.evictions() == 1, "one prune for one over-budget insert");
    }
    {
        // The fixed slot count: eight prefixes live side by side, and the ninth
        // pays for the conversation idle the longest, not for the first parked.
        ConversationCache cache(1u << 20, ConversationCache::default_slots);
        check(ConversationCache::default_slots == 8 && cache.slots() == 8, "eight parked prefixes by default");
        for (int i = 0; i < 8; ++i)
            check(cache.put(image({int32_t(100 + i), 1, 2})), "park a distinct prefix");
        check(cache.size() == 8 && cache.evictions() == 0, "eight slots hold eight prefixes");
        // Mount through the OLDEST parked prefix so it is no longer idle, and
        // leave prefix 101 untouched from here on.
        check(cache.best(std::vector<int64_t>{100,1,2,9}, {}, true).tokens == 3, "the oldest parked prefix still matches");
        check(cache.best(std::vector<int64_t>{100,1,2,9}, {}, true).index == 0,
              "crediting an entry never reorders park order - that is what ties are measured on");
        check(cache.put(image({200,1,2})), "the ninth prefix forces exactly one prune");
        check(cache.size() == 8 && cache.evictions() == 1, "the slot count stays fixed at eight");
        check(cache.best(std::vector<int64_t>{100,1,2,9}, {}, true).tokens == 3, "the just-used prefix survived");
        check(cache.best(std::vector<int64_t>{101,1,2,9}, {}, true).tokens == 0, "the idle prefix 101 is the LRU victim");
        for (int i = 2; i < 8; ++i) {
            std::vector<int64_t> p; p.push_back(int32_t(100 + i)); p.insert(p.end(), {1, 2, 9});
            check(cache.best(p, {}, true).tokens == 3, "the other idle prefixes are untouched by one prune");
        }
    }
    {
        // The shipped configuration at a realistic scale: a MiB-sized snapshot, a
        // budget of exactly eight of them, and the default slot count. The byte
        // budget and the slot limit agree on how many fit, and the ninth pays for
        // the conversation idle the longest rather than the first parked.
        const size_t one = image({1,2,3}).bytes();
        auto big = [&](int32_t head) {
            SavedConversation s = image({head, 1, 2});
            s.live.gdn.resize(one, 3);   // ~1 MiB per parked prefix
            return s;
        };
        const size_t snap = big(1).bytes();
        ConversationCache cache(snap*8, ConversationCache::default_slots);
        for (int i = 0; i < 8; ++i)
            check(cache.put(big(int32_t(300 + i))), "a MiB-scale prefix parks");
        check(cache.size() == 8 && cache.bytes() == snap*8, "the budget and the slot count both allow eight");
        check(!cache.can_fit(snap), "a ninth prefix does not fit without pruning");
        check(cache.best(std::vector<int64_t>{300,1,2,9}, {}, true).tokens == 3, "mount the oldest parked prefix");
        check(cache.make_room(snap), "reserving for the ninth prunes exactly one");
        check(cache.size() == 7 && cache.evictions() == 1, "the byte budget, not the slot count, set the limit");
        check(cache.best(std::vector<int64_t>{300,1,2,9}, {}, true).tokens == 3, "the just-used prefix survived");
        check(cache.best(std::vector<int64_t>{301,1,2,9}, {}, true).tokens == 0, "the prefix idle the longest is the victim");
        check(cache.put(big(400)) && cache.bytes() == snap*8, "the ninth prefix parks at full budget");
        check(cache.size() == 8, "back to eight parked prefixes");
    }
    {
        // A byte-budget prune picks the same victim as a slot prune: the policy
        // is one rule, not two.
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one*3, 16);
        cache.put(image({1,2,3})); cache.put(image({9,8,7})); cache.put(image({5,6,7}));
        check(cache.size() == 3 && cache.bytes() == one*3, "the budget holds exactly three snapshots");
        check(cache.best(a, {}, true).tokens == 3, "use the entry parked first");
        check(cache.put(image({4,5,6})), "the fourth snapshot only fits by pruning");
        check(cache.best(b, {}, true).tokens == 0, "the idle middle entry pays for the new one");
        check(cache.best(a, {}, true).tokens == 3 && cache.best(std::vector<int64_t>{5,6,7,9}, {}, true).tokens == 3,
              "the used entry and the newest entry both survive a budget prune");
        check(cache.bytes() == one*3, "byte accounting after a budget prune");
    }
    {
        // The policy itself, walked by hand.
        const uint64_t stamps[] = {5, 2, 4, 2, 9};
        check(ConversationCache::lru_victim(stamps, 5) == 1, "the least recent stamp is the victim");
        check(ConversationCache::lru_victim(stamps, 5) == ConversationCache::lru_victim(stamps, 5), "the policy is pure");
        const uint64_t fresh[] = {1, 2, 3, 4};
        check(ConversationCache::lru_victim(fresh, 4) == 0, "untouched entries still leave in park order");
        const uint64_t touched[] = {4, 1, 2, 3};
        check(ConversationCache::lru_victim(touched, 4) == 1, "one use moves the victim to the next oldest");
        const uint64_t tie[] = {3, 3, 3};
        check(ConversationCache::lru_victim(tie, 3) == 0, "equal stamps break to the earlier index, deterministically");
        const uint64_t single[] = {7};
        check(ConversationCache::lru_victim(single, 1) == 0, "a lone entry is its own victim");
    }
    {
        // Observability of the stamps, and the rule that only a hit credits.
        ConversationCache cache(1024, 2);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        const uint64_t before = cache.use(0), clock = cache.clock();
        check(cache.best(b, {}, true).tokens == 3, "B matches");
        check(cache.clock() > clock && cache.use(1) > clock, "a match advances the clock and credits its entry");
        check(cache.use(0) == before, "a match credits only the entry that answered");
        cache.touch(0);
        check(cache.use(0) > cache.use(1), "touch credits an entry without mounting it");
        check(cache.put(image({5,6})), "the slot limit now prunes the entry touch() did not credit");
        check(cache.best(a, {}, true).tokens == 3 && cache.best(b, {}, true).tokens == 0,
              "touch() saved A; B - used once and then left alone - is the victim");
        check(cache.best(std::vector<int64_t>{5,6,7,9}, {}, true).tokens == 2, "the new entry is parked");
    }
    {
        auto s = image({1, 2, 3});
        ConversationCheckpoint cp;
        cp.ids = {1, 2};
        s.checkpoints.push_back(cp);
        ConversationCache cache(4096, 4);
        cache.put(std::move(s));
        auto match = cache.best(std::vector<int64_t>{1, 2, 9, 4}, {}, true);
        check(match.tokens == 2 && !match.live, "edited suffix falls back to parked checkpoint");
        check(cache.best(a, {}, true).tokens == 3, "live prefix beats shorter checkpoint");
        check(cache.best(a, {}, false).tokens == 0, "steering mode is isolated");
        check(cache.best(std::vector<int64_t>{1, 2}, {}, true).tokens == 0, "equal-length checkpoint cannot consume last token");
        check(cache.best(std::vector<int64_t>{1}, {}, true).tokens == 0, "short prompt cannot match");
        check(cache.best(std::vector<int64_t>{}, {}, true).tokens == 0, "empty prompt cannot match");
        cache.put(image({1, 2, 3}));
        check(cache.best(a, {}, true).index == 1, "ties prefer most recently parked");
    }
    {
        auto s = image({1, 2, 3});
        s.live.imgs = {{1, 123}};
        ConversationCache cache(1024, 3);
        cache.put(std::move(s));
        check(cache.best(a, {{1,123}}, true).tokens == 3, "same image can resume");
        check(cache.best(a, {{1,124}}, true).tokens == 0, "different image pixels invalidate same pad tokens");
        check(cache.best(a, {}, true).tokens == 0, "missing image invalidates prefix");
        check(cache.best(a, {{1,123},{2,45}}, true).tokens == 0, "additional image in cached prefix invalidates");
        check(cache.best(a, {{1,123},{3,45}}, true).tokens == 3, "image after cached prefix does not invalidate");
    }
    {
        const size_t one = image({1,2,3}).bytes();
        ConversationCache cache(one*2, 8);
        cache.put(image({1,2,3})); cache.put(image({9,8,7}));
        check(cache.bytes() == one*2, "budget holds two exact-sized images");
        auto held = cache.take(cache.best(a, {}, true).index);
        check(cache.make_room(one, held.bytes()), "count in-flight image when reserving outgoing snapshot");
        check(cache.size() == 0, "in-flight reservation evicts otherwise fitting B");
        check(cache.put(image({5,6,7}), held.bytes()), "insert with in-flight accounting");
        check(cache.bytes()+held.bytes() <= one*2, "exchange obeys byte budget");
        check(!cache.make_room(one+1, one), "oversized exchange rejected");
        check(cache.size() == 1, "oversized snapshot does not evict useful entries");
        auto huge = image({1}); huge.live.gdn.resize(one*3);
        check(!cache.put(std::move(huge)), "oversized image rejected");
        check(cache.size() == 1, "oversized put leaves cache unchanged");
        check(!cache.make_room(0, one*2+1), "held larger than budget cannot underflow");
    }
    {
        ConversationCache disabled(0,4), no_slots(1024,0);
        check(!disabled.enabled() && !no_slots.enabled(), "both disable switches");
        check(!disabled.put(image({1,2,3})) && !no_slots.put(image({1,2,3})), "disabled cache stores nothing");
        check(disabled.best(a,{},true).tokens == 0, "disabled cache has no matches");
    }
    {
        // S3.9 THE CLAIM — a parked branch belongs to the request that is still running.
        //
        // The owner's regression: "the parking doesnt seem to work now, as when a new request arrives
        // it always kicks the currently running request out with an 'it was decode but not parked'
        // error."  The log shows the park SUCCEEDING (`park: slot 14 parked 10394 tokens`), then
        // `restored 9849 tokens (checkpoint) … parked=4`, then `slot 14's parked branch (10394 tokens)
        // is no longer in the cache` — with `evictions=0` for the entire run.  Nothing was pruned.  A
        // different request's PREFIX lookup matched one of the entry's checkpoints, and the mount that
        // followed `take()`'d the whole entry: live branch, checkpoint chain and K/V.  The entry is one
        // object, and the prefix rule does not care which part of it it hit.
        ConversationCache c(1 << 20, 8);
        auto owned = image({1, 2, 3, 4, 5});
        ConversationCheckpoint cp;
        cp.ids = {1, 2, 3};
        cp.gdn.resize(64, 7);
        owned.checkpoints.push_back(cp);
        check(c.put(std::move(owned), /*owner=*/14), "slot 14 parks its own branch");
        check(c.owner(0) == 14, "and the entry is claimed by the slot that parked it");

        // The theft, as it actually happened: request 15's prompt starts with the checkpoint's three
        // tokens, so the prefix rule matches — and mounting would remove slot 14's whole branch.
        const auto stolen = c.best(std::vector<int32_t>{1, 2, 3, 99}, {}, true, /*requester=*/15);
        check(stolen.tokens == 0,
              "a branch claimed by another running slot is NOT offered to this one - this is the fix");
        check(c.size() == 1, "and slot 14's conversation is still in the cache");

        // The owner may still have its own branch back.  Same entry, same claim, right requester.
        const auto back = c.best_exact(std::vector<int32_t>{1, 2, 3, 4, 5}, {}, true, /*requester=*/14);
        check(back.tokens == 5, "the claiming slot still finds its own branch");
        check(c.take(back.index).live.ids.size() == 5, "and may mount it - that is what the claim is for");
        check(c.size() == 0, "a mount still removes the entry, claim or no claim");

        // The claim is not a pin and not a leak: when its request is gone the branch is ordinary warm
        // storage again, and stage 2's prefix reuse works exactly as before.
        ConversationCache c2(1 << 20, 8);
        auto done = image({1, 2, 3, 4, 5});
        ConversationCheckpoint cp2;
        cp2.ids = {1, 2, 3};
        cp2.gdn.resize(64, 7);
        done.checkpoints.push_back(cp2);
        c2.put(std::move(done), 14);
        check(c2.best(std::vector<int32_t>{1, 2, 3, 99}, {}, true, 15).tokens == 0,
              "still hidden while slot 14 runs");
        c2.release_owner(14);
        check(c2.owner(0) == kNoOwner, "releasing the claim leaves the entry in the cache");
        const auto reuse = c2.best(std::vector<int32_t>{1, 2, 3, 99}, {}, true, 15);
        check(reuse.tokens == 3 && !reuse.live,
              "and now the next request of the chat resumes from the checkpoint - stage 2 intact");

        // A claim held by nobody in particular (a serial server, a finished conversation) hides
        // nothing: every existing behaviour is unchanged when `requester` is kNoOwner.
        ConversationCache c3(1 << 20, 8);
        c3.put(image({1, 2, 3}));
        check(c3.best(std::vector<int32_t>{1, 2, 3, 99}, {}, true).tokens == 3,
              "an unclaimed entry is offered to anyone, as always");
        check(c3.best(std::vector<int32_t>{1, 2, 3, 99}, {}, true, /*requester=*/77).tokens == 3,
              "and a requester never loses to an unclaimed entry");

        // Eviction prefers what belongs to nobody.  A claimed entry is a live conversation: pruning it
        // does not cost that request a re-read, it ends the request.
        ConversationCache c4(1 << 20, 4);
        c4.put(image({10, 1, 2}), 14);          // claimed, and the OLDER of the two
        c4.put(image({20, 1, 2}));              // unclaimed, and newer
        check(c4.victim_to_prune() == 1, "the unclaimed entry goes first, even though it is the newer one");
        c4.release_owner(14);
        check(c4.victim_to_prune() == 0, "once both are unclaimed it is plain LRU again");

        // When every entry is claimed there is nothing safe to evict, and the budget is still the hard
        // limit: LRU decides, and the owner gets the honest "pruned while it waited" line.
        ConversationCache c5(1 << 20, 3);
        c5.put(image({10, 1, 2}), 14);
        c5.put(image({20, 1, 2}), 15);
        c5.put(image({30, 1, 2}), 16);
        check(c5.victim_to_prune() == 0, "all claimed: LRU, the oldest parked");
        check(c5.put(image({40, 1, 2})), "and a park still succeeds by pruning something");
        check(c5.evictions() == 1, "exactly one prune");
        check(c5.size() == 3, "the cache stays at its slot limit");
        check(c5.owner(0) == 15 && c5.owner(1) == 16, "the two younger conversations survived");
    }

    {
        // S3.7 THE HAND-OVER LOOKUP.  A driver pre-empting a decoder parks its branch and later wants
        // THAT branch back.  `best()` cannot give it: its prefix rule requires the token list to be
        // strictly longer than the entry, and a slot's own branch is exactly equal to what it parked.
        // This is the bug behind the owner's "this slot's conversation was lost while it was decode" -
        // the branch was parked (80 tokens / 113 MiB against a 10 048 MiB budget), the lookup returned
        // 0, nothing was mounted, and the step gate ended a request whose budget was never the problem.
        const std::vector<int32_t> branch = {1, 2, 3};
        ConversationCache cache(4096, 4);
        check(cache.put(image({1, 2, 3})), "park the branch a slot will ask back");
        check(cache.best(branch, {}, true).tokens == 0,
              "best() cannot find a slot's own branch - the equal-length rule, and the bug");
        const auto back = cache.best_exact(branch, {}, true);
        check(back.tokens == 3 && back.live, "best_exact() finds it, as the live branch of its entry");
        check(back.index == 0, "and names the entry");
        auto restored = cache.take(back.index);
        check(restored.live.ids == branch, "taking it hands back the branch itself");
        check(cache.size() == 0, "and it leaves the cache, as a mount always does");

        // LIVE BRANCHES ONLY.  A restore always puts the whole entry back, so matching a checkpoint
        // whose ids happen to equal the query while its entry's live branch is LONGER would restore a
        // sequence further along than the one the slot stopped at - the slot would resume past tokens
        // it never generated.  A hand-over wants the exact branch or nothing.
        {
            auto s = image({1, 2, 3});
            ConversationCheckpoint cp;
            cp.ids = {1, 2};
            s.checkpoints.push_back(cp);
            ConversationCache c2(4096, 4);
            c2.put(std::move(s));
            check(c2.best_exact(std::vector<int32_t>{1, 2}, {}, true).tokens == 0,
                  "a checkpoint equal to the query is NOT mounted - its entry's live branch is longer");
            check(c2.best_exact(std::vector<int32_t>{1, 2, 3}, {}, true).tokens == 3,
                  "the live branch matches");
            check(c2.best_exact(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 0,
                  "a longer branch is not this one - no partial restore");
            check(c2.best_exact(std::vector<int32_t>{1, 2, 9}, {}, true).tokens == 0,
                  "a diverging branch matches nothing");
            check(c2.best_exact(std::vector<int32_t>{}, {}, true).tokens == 0,
                  "an empty query matches nothing");
        }
        // Steering mode and images are compared exactly as the prefix rule compares them.
        {
            auto s = image({1, 2, 3});
            s.live.imgs = {{1, 123}};
            ConversationCache c3(4096, 4);
            c3.put(std::move(s));
            check(c3.best_exact(branch, {}, true).tokens == 0, "a missing image invalidates the branch");
            check(c3.best_exact(branch, {{1, 124}}, true).tokens == 0, "different pixels invalidate it");
            check(c3.best_exact(branch, {{1, 123}}, true).tokens == 3, "the same image resumes it");
            check(c3.best_exact(branch, {{1, 123}}, false).tokens == 0, "steering mode is isolated");
        }
        // A hit is a use, so the entry that answered a hand-over is not the one that gets pruned next.
        {
            ConversationCache c4(1024, 2);
            c4.put(image({1, 2, 3}));
            c4.put(image({9, 8, 7}));
            const uint64_t u0 = c4.use(0), u1 = c4.use(1), clock = c4.clock();
            check(c4.best_exact(std::vector<int32_t>{1, 2, 3}, {}, true).tokens == 3,
                  "the hand-over lookup answers");
            check(c4.clock() > clock && c4.use(0) > clock, "and credits the entry it found");
            check(c4.use(1) == u1 && u1 < c4.use(0), "and credits only that entry");
            check(u0 < c4.use(0), "the credited entry is now the newest");
        }
        // A disabled cache answers nothing, exactly as `best()` does.
        {
            ConversationCache off(0, 4);
            off.put(image({1, 2, 3}));
            check(off.best_exact(std::vector<int32_t>{1, 2, 3}, {}, true).tokens == 0,
                  "a disabled cache has no matches for a hand-over either");
        }
    }
    test_probe_is_a_use_free_lookup();
    std::printf("conversation_cache_test: %d checks passed\n", checks);
}
