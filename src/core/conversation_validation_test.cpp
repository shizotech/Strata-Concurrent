// Host-only validation fixtures. CUDA is linked but must never be initialized:
// invalid restores must return before the first CUDA call or destination write.
#include "strata/core/conversation_snapshot.hpp"
#include "strata/kernels/kv_q4.hpp"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <limits>
#include <cstring>
#include <memory>
#include <vector>

#if defined(CONVERSATION_TEST_TRANSFERS)
// GNU/ELF link wrapping exercises the actual restore control flow without
// initializing CUDA or deliberately poisoning a real device context. This is
// host-backend fault injection, not a claim of hardware-failure recovery.
namespace {
int copy_calls = 0, sync_calls = 0, fail_copy = 0, fail_sync = 0;
size_t copied_bytes = 0;
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind) {
    if (++copy_calls == fail_copy) return cudaErrorInvalidValue;
    copied_bytes += n;
    std::memcpy(dst, src, n);
    return cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaDeviceSynchronize() {
    return ++sync_calls == fail_sync ? cudaErrorUnknown : cudaSuccess;
}
extern "C" cudaError_t __wrap_cudaGetLastError() { return cudaSuccess; }
#endif

using namespace strata::core;
namespace {
int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) { std::fprintf(stderr, "FAIL: %s\n", label); std::exit(1); }
}
struct Pools {
    QsaState st;
    std::array<std::vector<uint8_t>, 8> data;
    Pools(const ModelGeometry& g, int format) {
        st.max_cells = 96; st.n_pages = st.n_slots = 24; st.idx_pooled_rows = 26;
        st.kv_int8 = format == 1; st.kv_q4 = format == 2;
        const size_t per = format == 2 ? strata::kernels::kv_q4_bytes_per_head((int) g.head_dim)
                                      : g.head_dim * (format == 1 ? 1 : 2);
        data[0].resize(96 * g.n_head_kv * per, 0xa5); data[1] = data[0];
        data[2].resize(format == 1 ? 96 * g.n_head_kv * (g.head_dim / 64) * 2 : 0, 0xa5); data[3] = data[2];
        data[4].resize(26 * g.idx_key_dim * 4, 0xa5);
        data[5].resize(3 * g.idx_key_dim * 4, 0xa5);
        data[6].resize(g.idx_key_dim * 4, 0xa5); data[7].resize(4, 0xa5);
        if (format == 2) { st.k_q4 = data[0].data(); st.v_q4 = data[1].data(); }
        else if (format == 1) {
            st.k_q = (int8_t*) data[0].data(); st.v_q = (int8_t*) data[1].data();
            st.k_scale = (uint16_t*) data[2].data(); st.v_scale = (uint16_t*) data[3].data();
        } else { st.k_pool = (uint16_t*) data[0].data(); st.v_pool = (uint16_t*) data[1].data(); }
        st.idx_pooled = (float*) data[4].data(); st.idx_tail = (float*) data[5].data();
        st.idx_dead = (float*) data[6].data(); st.idx_block_pos = (int32_t*) data[7].data();
    }
    ConversationKv image(const ModelGeometry& g, bool index) const {
        ConversationKv k;
        k.format = qsa_kv_format(st); k.cells = 12; k.page_size = 4;
        k.heads = g.n_head_kv; k.head_dim = g.head_dim; k.idx_dim = g.idx_key_dim;
        k.pooled_rows = index ? 3 : 0;
        k.k.resize(data[0].size() / 8, 13); k.v = k.k;
        k.k_scale.resize(data[2].size() / 8, 13); k.v_scale = k.k_scale;
        k.pooled.resize(k.pooled_rows * g.idx_key_dim * 4, 13);
        return k;
    }
};

void fixture(int format, int experts, bool zero_qsa, bool ple) {
    ModelGeometry g;
    g.n_layers = zero_qsa ? 3 : 8; g.n_expert = experts;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    std::string error;
    ConversationStateSizes z;
    check(conversation_state_sizes(g, z, error), "checked canonical/pruned/zero-QSA sizing");
    Pools first(g, format), last(g, format), draft(g, format);
    std::array<QsaState, 2> layers{first.st, last.st};
    std::vector<uint8_t> gdn(z.gdn, 0xa5), history(ple ? z.ple : 0, 0xa5);
    SessionState ss;
    ss.max_cells = 96; ss.gdn_state = (float*) gdn.data();
    // the whole-model carve, as session_init leaves a one-GPU session
    ss.layer_hi = g.n_layers; ss.gdn_alloc = g.n_gdn_layers(); ss.qsa_alloc = g.n_qsa_layers();
    ss.ple_hist = ple ? (float*) history.data() : nullptr;
    ss.qsa_states = zero_qsa ? nullptr : layers.data();
    SavedConversation image;
    image.geometry = {g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,
                      g.ssm_d_conv,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,
                      g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
    image.layer_hi = g.n_layers;
    auto checkpoint = [&](size_t tokens) {
        ConversationCheckpoint c;
        c.ids.resize(tokens);
        for (size_t i = 0; i < tokens; ++i) c.ids[i] = (int32_t) i + 1;
        c.imgs = {{1, 44}};
        c.gdn.resize(z.gdn, 13); c.ple.resize(ple ? z.ple : 0, 13);
        c.tails.resize(g.n_qsa_layers() * z.tail, 13); c.dead.resize(g.n_qsa_layers() * z.dead, 13);
        c.block_pos.resize(g.n_qsa_layers() * z.block_pos, 13);
        return c;
    };
    image.live = checkpoint(9); image.checkpoints.push_back(checkpoint(5));
    image.kv.reserve((size_t) g.n_qsa_layers() + 1);
    if (!zero_qsa) { image.kv.push_back(first.image(g, true)); image.kv.push_back(last.image(g, true)); }
    image.kv.push_back(draft.image(g, false));
    check(conversation_snapshot_validate(image, ss, g, draft.st, error), "complete image validates without CUDA");
    size_t estimate = 0;
    check(conversation_snapshot_bytes({image.live.ids,image.live.imgs,image.checkpoints,true},ss,g,draft.st,estimate,error),
          "capture estimate works without CUDA");
    check(estimate == image.bytes(), "estimate covers checkpoint/indexer/spare payloads");
    auto unchanged = [&] {
        auto pristine = [](const auto& bytes) { return std::all_of(bytes.begin(),bytes.end(),[](uint8_t b){return b==0xa5;}); };
        if (!pristine(gdn) || !pristine(history) || ss.ple_prev[0] != -1 || ss.ple_prev[1] != -1) return false;
        for (const auto* p : {&first,&last,&draft}) for (const auto& bytes : p->data) if (!pristine(bytes)) return false;
        return true;
    };
    auto reject = [&](const std::function<void(SavedConversation&)>& mutate, const char* label) {
        auto bad = image; mutate(bad);
        check(conversation_snapshot_restore(bad,ss,g,draft.st,error) == ConversationRestore::invalid,label);
        check(unchanged(), "invalid restore did not touch any layer or running-state buffer");
    };
    reject([](auto& s){s.kv.back().k.pop_back();}, "late draft corruption rejected before first layer write");
    reject([](auto& s){s.geometry[16] = 128;}, "different expert geometry rejected");
    reject([](auto& s){s.live.stage_parts.emplace_back();}, "layer-split live state rejected before writes");
    reject([](auto& s){s.checkpoints[0].stage_parts.emplace_back();}, "layer-split checkpoint rejected before writes");
    reject([](auto& s){s.kv.pop_back();}, "missing KV layer rejected");
    reject([](auto& s){s.live.gdn.pop_back();}, "bad live recurrence rejected");
    reject([](auto& s){s.checkpoints.back().gdn.pop_back();}, "bad retained checkpoint rejected");
    reject([](auto& s){s.checkpoints.back().ids[0] = 99;}, "foreign checkpoint prefix rejected");
    reject([](auto& s){s.checkpoints.back().imgs[0].hash++;}, "foreign checkpoint image rejected");
    reject([](auto& s){s.live.imgs[0].start = -1;}, "negative image position rejected");
    reject([](auto& s){s.live.ids[0] = -1;}, "negative live token rejected");
    reject([](auto& s){s.live.ple.push_back(0);}, "PLE size/enabled mismatch rejected");
    reject([](auto& s){s.live.tails.push_back(0);}, "tail size/zero-QSA mismatch rejected");
    if (!zero_qsa) {
        reject([](auto& s){s.kv[1].pooled.pop_back();}, "late indexer spare row corruption rejected");
        reject([](auto& s){s.live.dead.pop_back();}, "missing indexer spare key rejected");
        reject([](auto& s){s.live.block_pos.pop_back();}, "missing indexer metadata rejected");
        ss.qsa_states = nullptr;
        check(!conversation_snapshot_validate(image,ss,g,draft.st,error), "missing QSA targets rejected");
        ss.qsa_states = layers.data();
        reject([](auto& s){s.layer_lo = 4;}, "snapshot from another layer range rejected");
        // #216's carve: a split stage owning layers [4, 8) holds GDN rows 3..5 and QSA ordinal 1 only
        SessionState stage = ss;
        stage.layer_lo = 4; stage.gdn_alloc = 3; stage.qsa_ord0 = 1; stage.qsa_alloc = 1;
        ConversationStateSizes zs;
        check(conversation_session_sizes(g,stage,zs,error) && zs.gdn == z.gdn / 6 * 3 && zs.tail == z.tail,
              "carved session sizes cover the owned rows only");
        check(!conversation_checkpoint_validate(image.live,stage,g,error), "whole-model checkpoint rejected by a carve");
        ConversationCheckpoint part = image.live;
        part.gdn.resize(zs.gdn); part.tails.resize(zs.tail); part.dead.resize(zs.dead);
        part.block_pos.resize(zs.block_pos);
        check(conversation_checkpoint_validate(part,stage,g,error), "carve-sized checkpoint validates");
        check(!conversation_snapshot_validate(image,stage,g,draft.st,error), "whole-model snapshot rejected by a carve");
        stage.qsa_alloc = 2;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the last QSA ordinal rejected");
        stage.qsa_alloc = 1; stage.gdn_alloc = 7;
        check(!conversation_session_sizes(g,stage,zs,error), "carve past the GDN rows rejected");
    }
    auto bad_geometry = g; bad_geometry.ssm_state_size = std::numeric_limits<int64_t>::max();
    check(!conversation_state_sizes(bad_geometry,z,error), "running-state arithmetic overflow rejected");
    bad_geometry = g; bad_geometry.qsa_interval = 0;
    check(!conversation_state_sizes(bad_geometry,z,error), "zero layer interval rejected before division");
    bad_geometry = g; bad_geometry.n_head_kv = std::numeric_limits<int64_t>::max();
    check(conversation_kv_bytes(draft.st,bad_geometry,9,false)==0, "KV byte overflow rejected");
#if defined(CONVERSATION_TEST_TRANSFERS)
    auto reset = [&] {
        for (auto* p : {&first,&last,&draft}) for (auto& bytes : p->data) std::fill(bytes.begin(),bytes.end(),0xa5);
        std::fill(gdn.begin(),gdn.end(),0xa5); std::fill(history.begin(),history.end(),0xa5);
        ss.ple_prev[0] = ss.ple_prev[1] = -1;
        copy_calls = sync_calls = fail_copy = fail_sync = 0;
    };
    reset(); fail_sync = 1;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "pre-transfer CUDA synchronization failure is fatal");
    check(copy_calls==0 && unchanged(), "pre-sync failure did not mutate state");
    reset(); fail_copy = 1;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "first CUDA transfer failure is fatal");
    check(unchanged(), "first-copy failure did not mutate state");
    reset(); fail_copy = 2;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "partial CUDA transfer failure is fatal, not an invalid-image fallback");
    check(!unchanged(), "fault fixture genuinely produced partial state");
    reset(); fail_sync = 2;
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::transfer_failed,
          "post-transfer CUDA synchronization failure is fatal");
    check(copy_calls>0 && !unchanged(), "post-sync failure occurred after state writes");
    reset();
    check(conversation_snapshot_restore(image,ss,g,draft.st,error)==ConversationRestore::restored,
          "injected host transfer backend completes a valid restore");
    check(gdn==image.live.gdn && history==image.live.ple && ss.ple_prev[0]==8 && ss.ple_prev[1]==9,
          "successful restore publishes correct running state and PLE window");
    uint64_t fingerprint = 0;
    check(conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back verifies complete draft payload");
    draft.data[0][0] ^= 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back detects corrupted draft byte");
    draft.data[0][0] ^= 1;
    fail_copy = copy_calls + 1;
    check(!conversation_kv_verify(image.kv.back(),draft.st,g,9,false,fingerprint,error), "read-back transfer failure is not a successful fingerprint");
    reset(); copied_bytes = 0;
    const ConversationView view{image.live.ids,image.live.imgs,image.checkpoints,true};
    SavedConversation full,incremental;
    check(conversation_snapshot_save(full,view,ss,g,draft.st,error), "full capture through host transfer backend");
    const size_t full_copies = copied_bytes;
    ConversationKvReuse reuse{full.kv,9,9};
    size_t peak = 0, reused = 0;
    check(conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.st,peak,error), "incremental capture admission without transfers");
    copied_bytes = 0;
    const bool saved_incrementally = conversation_snapshot_save(incremental,view,ss,g,draft.st,error,std::move(reuse),&reused);
    if (!saved_incrementally) std::fprintf(stderr,"capture error: %s\n",error.c_str());
    check(saved_incrementally, "incremental capture through host backend");
    check(reused > 0 && copied_bytes + reused == full_copies, "reused byte count measures transfers actually omitted");
    check(incremental.bytes() <= peak && incremental.live.gdn == full.live.gdn, "running state refreshed within admitted allocation");
    for (size_t i=0;i<full.kv.size();++i)
        check(incremental.kv[i].k==full.kv[i].k && incremental.kv[i].v==full.kv[i].v &&
              incremental.kv[i].k_scale==full.kv[i].k_scale && incremental.kv[i].v_scale==full.kv[i].v_scale &&
              incremental.kv[i].pooled==full.kv[i].pooled, "incremental and full capture agree across every payload");
    reuse = {full.kv,9,9};
    reuse.kv.back().k.pop_back();
    copy_calls = sync_calls = 0;
    check(!conversation_snapshot_capture_bytes(reuse,view,ss,g,draft.st,peak,error), "malformed retained draft rejected before admission");
    check(!copy_calls && !sync_calls, "invalid retained storage performs no CUDA calls");
    reuse = {full.kv,9,9};
    fail_copy = copy_calls + 1;
    SavedConversation unpublished;
    check(!conversation_snapshot_save(unpublished,view,ss,g,draft.st,error,std::move(reuse)), "incremental capture copy failure is reported");
    check(unpublished.kv.empty() && unpublished.live.ids.empty(), "failed incremental capture cannot publish a partial image");
#endif
}

// ===================== S3.1a: PARKING ACROSS A `--layer-split` =====================
//
// A split runs the model across several devices and each stage's `SessionState` owns only its
// layer carve (#216).  A parked conversation is therefore the FIRST stage's state (CUDA0's, the
// snapshot's `session` argument) PLUS one part per later stage PLUS the MTP draft state, which
// the engine binds to the last stage.  This fixture builds that shape out of host buffers - no
// CUDA is initialized - splits the model N ways, and checks that the byte accounting sees every
// part and that a corrupted stage part is refused before a single byte of any session is written.
// The park/mount round trip needs the injected host transfer backend, so it is under the same
// `CONVERSATION_TEST_TRANSFERS` switch the single-GPU round trip already uses.
struct Carve {
    ModelGeometry g;
    int64_t lo = 0, hi = 0;
    std::vector<Pools> layers;         ///< one per OWNED QSA ordinal
    std::vector<QsaState> qsa_store;   ///< n_qsa_layers entries, exactly as session_init allocates them
    std::vector<uint8_t> gdn, ple;
    SessionState ss;
    Carve(const ModelGeometry& geometry, int64_t from, int64_t to, int format)
        : g(geometry), lo(from), hi(to) {
        const int64_t I = std::max<int64_t>(g.qsa_interval, 1);
        const int64_t q_alloc = std::max<int64_t>(hi / I - lo / I, g.n_qsa_layers() > 0 ? 1 : 0);
        for (int64_t j = 0; j < q_alloc; ++j) layers.emplace_back(g, format);
        ss.max_cells = 96;
        ss.layer_lo = lo; ss.layer_hi = hi;
        ss.qsa_ord0 = lo / I; ss.qsa_alloc = q_alloc;
        ss.gdn_ord0 = lo - lo / I;
        ss.gdn_alloc = std::max<int64_t>((hi - lo) - (hi / I - lo / I), 0);
        ConversationStateSizes z;
        std::string ignored;
        conversation_session_sizes(g, ss, z, ignored);
        gdn.assign(z.gdn, 0xa5);
        ple.assign(z.ple, 0xa5);
        ss.gdn_state = (float*) gdn.data();
        ss.ple_hist = (float*) ple.data();
        // session_init allocates the WHOLE n_qsa_layers array and initializes only the range's
        // ordinals; the rest stay nulls a snapshot must never read.
        if (g.n_qsa_layers() > 0) {
            qsa_store.assign((size_t) g.n_qsa_layers(), QsaState{});
            for (int64_t j = 0; j < q_alloc; ++j) qsa_store[(size_t) (ss.qsa_ord0 + j)] = layers[(size_t) j].st;
            ss.qsa_states = qsa_store.data();
        }
    }
    void paint(uint8_t salt) {
        std::fill(gdn.begin(), gdn.end(), salt); std::fill(ple.begin(), ple.end(), salt);
        for (auto& p : layers) for (auto& bytes : p.data) std::fill(bytes.begin(), bytes.end(), salt);
    }
    bool pristine() const {
        auto all = [](const std::vector<uint8_t>& b) {
            return std::all_of(b.begin(), b.end(), [](uint8_t v) { return v == 0xa5; });
        };
        if (!all(gdn) || !all(ple)) return false;
        for (const auto& p : layers) for (const auto& bytes : p.data) if (!all(bytes)) return false;
        return true;
    }
};

#if defined(CONVERSATION_TEST_TRANSFERS)
bool equal(const ConversationKv& a, const ConversationKv& b) {
    return a.k == b.k && a.v == b.v && a.k_scale == b.k_scale && a.v_scale == b.v_scale && a.pooled == b.pooled;
}
#endif

/// `splits` are the cut points after the first stage: {4} is a two-carve split of 8 layers and
/// {2,5} a three-carve one.  The first stage is CUDA0's session - the snapshot's `session`
/// argument - and the rest are the stage set.
void split_fixture(int format, const std::vector<int64_t>& splits) {
    ModelGeometry g;
    g.n_layers = 8; g.n_expert = 256;
    g.ssm_state_size = 2; g.ssm_v_heads = 2; g.ssm_conv_channels = 8;
    g.n_head_kv = 1; g.head_dim = 64; g.idx_key_dim = 8;
    std::vector<int64_t> bounds{0};
    for (int64_t s : splits) bounds.push_back(s);
    bounds.push_back(g.n_layers);
    std::vector<std::unique_ptr<Carve>> carves;
    for (size_t i = 0; i + 1 < bounds.size(); ++i)
        carves.push_back(std::make_unique<Carve>(g, bounds[i], bounds[i + 1], format));
    Carve& main = *carves[0];
    Pools draft(g, format);
    std::vector<ConversationStage> stage_list;
    for (size_t i = 1; i < carves.size(); ++i) stage_list.push_back({&carves[i]->ss, (int) i});
    // main_device/draft_device stay -1: a host-only fixture, and OnDevice(-1) never asks CUDA
    // anything.  The engine passes 0 and the last stage's device instead.
    const ConversationStageSet stages{stage_list.empty() ? nullptr : stage_list.data(),
                                      (int64_t) stage_list.size(), -1, -1};
    std::vector<int32_t> ids(9);
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = (int32_t) i + 1;
    std::vector<ConversationImageKey> images{{1, 44}};
    auto part_of = [&](Carve& c) {
        ConversationCheckpoint p;
        p.ids = ids;
        ConversationStateSizes z;
        std::string ignored;
        conversation_session_sizes(g, c.ss, z, ignored);
        p.gdn.assign(z.gdn, 13); p.ple.assign(z.ple, 13);
        p.tails.assign((size_t) c.ss.qsa_alloc * z.tail, 13);
        p.dead.assign((size_t) c.ss.qsa_alloc * z.dead, 13);
        p.block_pos.assign((size_t) c.ss.qsa_alloc * z.block_pos, 13);
        return p;
    };
    std::vector<ConversationCheckpoint> checkpoints{part_of(main)};
    checkpoints[0].imgs = images;   // the images belong to the conversation, not to one stage
    for (size_t i = 1; i < carves.size(); ++i) checkpoints[0].stage_parts.push_back(part_of(*carves[i]));
    const ConversationView view{ids, images, checkpoints, true};
    std::string error;

    // A hand-built valid image: the host-only build cannot save, and validation must reject a
    // corrupt part whether or not CUDA is reachable.
    SavedConversation base;
    base.geometry = {g.n_embd,g.n_layers,g.qsa_interval,g.ssm_state_size,g.ssm_k_heads,g.ssm_v_heads,
                     g.ssm_d_conv,g.ssm_conv_channels,g.ssm_value_dim,g.n_head,g.n_head_kv,g.head_dim,
                     g.idx_q_heads,g.idx_key_dim,g.hc,g.hc_lr,g.n_expert,g.n_ff};
    base.layer_lo = main.lo; base.layer_hi = main.hi;
    base.live = part_of(main); base.live.imgs = images;
    base.checkpoints = checkpoints;
    base.kv.reserve(main.layers.size() + 1);
    for (const auto& p : main.layers) base.kv.push_back(p.image(g, true));
    base.kv.push_back(draft.image(g, false));
    base.stage_parts.reserve(carves.size() - 1);
    for (size_t i = 1; i < carves.size(); ++i) {
        ConversationStageSnapshot part;
        part.layer_lo = carves[i]->lo; part.layer_hi = carves[i]->hi;
        part.state = part_of(*carves[i]);
        part.kv.reserve(carves[i]->layers.size());
        for (const auto& p : carves[i]->layers) part.kv.push_back(p.image(g, true));
        base.stage_parts.push_back(std::move(part));
    }
    check(conversation_snapshot_validate(base, main.ss, stages, g, draft.st, error), "split image validates");
    size_t estimate = 0, plain = 0;
    check(conversation_snapshot_bytes(view, main.ss, stages, g, draft.st, estimate, error),
          "split capture estimate works without CUDA");
    check(estimate == base.bytes(), "the split estimate covers every stage part and its K/V");
    // A refused estimate must not leave a half-written number behind either.
    check(!conversation_snapshot_bytes(view, main.ss, {}, g, draft.st, plain, error),
          "a chain carrying stage parts cannot be parked without a stage set");
    check(plain == 0, "a refused single-GPU estimate reports no bytes");
    check(!conversation_snapshot_validate(base, main.ss, {}, g, draft.st, error),
          "a split image is refused without a stage set");
    check(base.stage_parts.size() == carves.size() - 1, "one stage part per later stage");
    for (size_t i = 0; i < base.stage_parts.size(); ++i) {
        const Carve& c = *carves[i + 1];
        check(base.stage_parts[i].layer_lo == c.lo && base.stage_parts[i].layer_hi == c.hi,
              "stage part records the carve it came from");
        check(base.stage_parts[i].kv.size() == (size_t) c.ss.qsa_alloc,
              "a stage part holds only its own carve's K/V, never a draft entry");
        check(base.stage_parts[i].state.gdn.size() < (size_t) 1 << 20, "a stage part is carve-sized");
    }
    // ---- all-or-nothing: every way a stage part can be wrong is refused with no CUDA and no write ----
    auto unchanged_all = [&] {
        auto all = [](const std::vector<uint8_t>& b) {
            return std::all_of(b.begin(), b.end(), [](uint8_t v) { return v == 0xa5; });
        };
        for (const auto& c : carves) if (!c->pristine()) return false;
        for (const auto& bytes : draft.data) if (!all(bytes)) return false;
        return true;
    };
    auto reject = [&](const std::function<void(SavedConversation&)>& mutate, const char* label) {
        auto bad = base; mutate(bad);
        check(conversation_snapshot_restore(bad, main.ss, stages, g, draft.st, error) == ConversationRestore::invalid,
              label);
        check(unchanged_all(), "invalid split restore did not touch any stage or the draft");
    };
    check(unchanged_all(), "the split fixture starts pristine");
    reject([](auto& s){ s.stage_parts.back().kv.back().k.pop_back(); },
           "corrupt last-stage K/V rejected before any write");
    reject([](auto& s){ s.stage_parts.front().state.gdn.pop_back(); },
           "corrupt first-stage running state rejected before any write");
    reject([](auto& s){ s.stage_parts.back().state.tails.push_back(0); },
           "wrong-size stage indexer payload rejected before any write");
    reject([](auto& s){ s.stage_parts.pop_back(); }, "a snapshot missing a stage part is rejected");
    reject([](auto& s){ s.stage_parts[0].layer_hi = s.stage_parts[0].layer_hi + 1; },
           "a stage part from another carve is rejected");
    reject([](auto& s){ s.checkpoints[0].stage_parts.back().dead.pop_back(); },
           "corrupt retained-checkpoint stage part rejected before any write");
    reject([](auto& s){ s.checkpoints[0].stage_parts.pop_back(); },
           "a retained checkpoint missing a stage part is rejected");
    reject([](auto& s){ s.kv.back().k.pop_back(); }, "corrupt draft rejected before any stage write");
    reject([](auto& s){ s.stage_parts.back().state.ids[0] = 99; },
           "a stage part that is not the live prefix is rejected");
    reject([](auto& s){ s.stage_parts.back().state.imgs.push_back({0, 7}); },
           "a stage part carrying images is rejected");
    reject([](auto& s){ s.live.stage_parts.push_back(s.stage_parts[0].state); },
           "a live checkpoint carrying stage parts is still refused");
    // A stage set that does not tile the model is a wiring bug, not a smaller conversation.
    {
        SessionState short_main = main.ss;   // the same buffers, a range that stops one layer early
        short_main.layer_hi = g.n_layers - 1;
        size_t refused = 1;
        check(!conversation_snapshot_bytes(view, short_main, stages, g, draft.st, refused, error),
              "stage set that does not cover the model is rejected");
        check(refused == 0, "a refused estimate reports no bytes");
        std::vector<ConversationStage> with_null = stage_list;
        with_null.push_back(ConversationStage{nullptr, -1});
        const ConversationStageSet null_set{with_null.data(), (int64_t) with_null.size(), -1, -1};
        check(!conversation_snapshot_bytes(view, main.ss, null_set, g, draft.st, refused, error),
              "a stage with no session behind it is rejected");
        check(refused == 0, "a refused stage-set estimate reports no bytes");
    }
#if defined(CONVERSATION_TEST_TRANSFERS)
    // ---- the round trip: park, scribble over every stage, mount, park again, compare ----
    SavedConversation image;
    if (!conversation_snapshot_save(image, view, main.ss, stages, g, draft.st, error))
        std::fprintf(stderr, "split capture error: %s\n", error.c_str());
    check(conversation_snapshot_save(image, view, main.ss, stages, g, draft.st, error), "split capture");
    check(image.stage_parts.size() == carves.size() - 1, "one stage part per later stage");
    check(image.live.stage_parts.empty(), "a parked image keeps its stage state in stage_parts");
    check(image.checkpoints[0].stage_parts.size() == carves.size() - 1,
          "the retained checkpoints keep the upstream chain's stage parts");
    check(image.bytes() == estimate, "a split capture's bytes match its admitted estimate");
    SavedConversation single_gpu;
    check(!conversation_snapshot_save(single_gpu, view, main.ss, g, draft.st, error),
          "a split checkpoint chain cannot be captured as a single-GPU image");
    for (auto& c : carves) c->paint(0x31);
    for (auto& bytes : draft.data) std::fill(bytes.begin(), bytes.end(), 0x31);
    check(!unchanged_all(), "the split fixture really scribbled over every stage");
    const int mount_c0 = copy_calls, mount_s0 = sync_calls;
    check(conversation_snapshot_restore(image, main.ss, stages, g, draft.st, error) == ConversationRestore::restored,
          "split restore mounts every stage");
    const int mount_copies = copy_calls - mount_c0, mount_syncs = sync_calls - mount_s0;
    check(mount_syncs == (int) carves.size() + 1,
          "split mount synchronizes the main device, the draft's and every stage's own");
    for (size_t i = 0; i < carves.size(); ++i) {
        const ConversationCheckpoint& saved = i == 0 ? image.live : image.stage_parts[i - 1].state;
        check(carves[i]->gdn == saved.gdn && carves[i]->ple == saved.ple, "stage running state restored");
    }
    check(main.ss.ple_prev[0] == 8 && main.ss.ple_prev[1] == 9, "PLE window restored from a split image");
    // A failure anywhere in the stage parts is fatal to the session, never a fallback to
    // "invalid image, keep the old one": the caller has already parked the outgoing conversation.
    // The state at this point is no longer the 0xa5 sentinel, so "did this restore change anything"
    // is answered by fingerprinting every buffer before and after.
    auto fingerprint_all = [&] {
        uint64_t h = 1469598103934665603ull;
        auto feed = [&](const std::vector<uint8_t>& b) {
            for (size_t i = 0; i < b.size(); ++i) { h ^= b[i]; h *= 1099511628211ull; }
        };
        for (const auto& c : carves) { feed(c->gdn); feed(c->ple); for (const auto& p : c->layers) for (const auto& b : p.data) feed(b); }
        for (const auto& b : draft.data) feed(b);
        return h;
    };
    auto scribble = [&] {
        for (auto& c : carves) c->paint(0x31);
        for (auto& bytes : draft.data) std::fill(bytes.begin(), bytes.end(), 0x31);
        std::fill(main.ss.ple_prev, main.ss.ple_prev + 2, -1);
    };
    scribble();
    copy_calls = sync_calls = 0; fail_sync = 1;
    const uint64_t before_pre_sync = fingerprint_all();
    check(conversation_snapshot_restore(image, main.ss, stages, g, draft.st, error) == ConversationRestore::transfer_failed,
          "split: pre-transfer synchronization failure is fatal");
    check(copy_calls == 0 && fingerprint_all() == before_pre_sync,
          "split: pre-sync failure wrote nothing to any stage");
    scribble();
    fail_sync = 0; copy_calls = sync_calls = 0; fail_copy = mount_copies;
    const uint64_t before_last_copy = fingerprint_all();
    check(conversation_snapshot_restore(image, main.ss, stages, g, draft.st, error) == ConversationRestore::transfer_failed,
          "split: a transfer failure in the last stage part is fatal, not an invalid-image fallback");
    check(fingerprint_all() != before_last_copy, "split: the last-copy failure genuinely left partial state");
    scribble();
    fail_copy = 0; copy_calls = sync_calls = 0; fail_sync = mount_syncs;
    const uint64_t before_stage_sync = fingerprint_all();
    check(conversation_snapshot_restore(image, main.ss, stages, g, draft.st, error) == ConversationRestore::transfer_failed,
          "split: a per-stage synchronization failure is fatal");
    check(sync_calls == mount_syncs && fingerprint_all() != before_stage_sync,
          "split: the per-stage synchronization really happens, after the stage writes");
    fail_sync = 0; copy_calls = sync_calls = 0;
    // Re-capturing the mounted state must reproduce the same bytes on every stage: the A/B/A
    // exactness the single-GPU test already demands, now across a carve.
    SavedConversation again;
    check(conversation_snapshot_save(again, view, main.ss, stages, g, draft.st, error),
          "re-capture after a split mount");
    check(again.stage_parts.size() == image.stage_parts.size(), "re-capture keeps the same stage count");
    for (size_t i = 0; i < image.stage_parts.size(); ++i) {
        const auto& x = again.stage_parts[i]; const auto& y = image.stage_parts[i];
        check(x.state.gdn == y.state.gdn && x.state.tails == y.state.tails &&
              x.state.dead == y.state.dead && x.state.block_pos == y.state.block_pos && x.state.ple == y.state.ple,
              "split A/B/A stage running state is byte-exact");
        for (size_t j = 0; j < y.kv.size(); ++j) check(equal(x.kv[j], y.kv[j]), "split A/B/A stage K/V is byte-exact");
    }
    check(again.live.gdn == image.live.gdn && again.live.dead == image.live.dead &&
          equal(again.kv[0], image.kv[0]) && equal(again.kv.back(), image.kv.back()),
          "split A/B/A main session and draft are byte-exact");
    // Retention stays the single-GPU optimization: the retained set is the main layers plus the
    // draft, never a stage's, and a split capture still honours the admitted bound.
    ConversationKvReuse reuse{image.kv, 9, 9};
    size_t peak = 0, reused = 0;
    check(conversation_snapshot_capture_bytes(reuse, view, main.ss, stages, g, draft.st, peak, error),
          "retained main K/V still admits a split capture");
    SavedConversation incremental;
    check(conversation_snapshot_save(incremental, view, main.ss, stages, g, draft.st, error,
                                     std::move(reuse), &reused), "incremental split capture");
    check(reused > 0 && incremental.bytes() <= peak, "incremental split capture stays within its admission");
    check(incremental.stage_parts.size() == image.stage_parts.size() && incremental.bytes() == image.bytes(),
          "an incremental split capture is the same image");
    reuse = {image.kv, 9, 9};
    reuse.kv.push_back(image.kv[0]);
    check(!conversation_snapshot_capture_bytes(reuse, view, main.ss, stages, g, draft.st, peak, error),
          "a retained set that is not exactly the main layers plus the draft is rejected");
#endif
}
}

int main() {
    for (int format : {0,1,2}) for (int experts : {256,512})
        for (bool zero_qsa : {false,true}) for (bool ple : {false,true}) fixture(format,experts,zero_qsa,ple);
    // S3.1a: two- and three-carve splits, and a stage range with no QSA layer of its own.
    for (int format : {0,1,2}) for (std::vector<int64_t> cuts : {std::vector<int64_t>{4},
                                                                 std::vector<int64_t>{2,5},
                                                                 std::vector<int64_t>{1,6}})
        split_fixture(format, cuts);
    std::printf("conversation_validation_test: %d host-only checks passed\n", checks);
}
