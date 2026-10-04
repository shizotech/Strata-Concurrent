#include "strata/core/conversation_snapshot.hpp"
#include "strata/core/on_device.hpp"
#include "conversation_checked.hpp"

#include <algorithm>
#include <array>
#include <limits>

namespace strata::core {
namespace {
using conversation_detail::add;
using conversation_detail::product;

std::array<int64_t, 18> geometry_key(const ModelGeometry& g) {
    return {g.n_embd, g.n_layers, g.qsa_interval, g.ssm_state_size, g.ssm_k_heads,
            g.ssm_v_heads, g.ssm_d_conv, g.ssm_conv_channels, g.ssm_value_dim,
            g.n_head, g.n_head_kv, g.head_dim, g.idx_q_heads, g.idx_key_dim,
            g.hc, g.hc_lr, g.n_expert, g.n_ff};
}

bool fail(std::string& error, const char* message) {
    error = std::string("conversation snapshot: ") + message;
    return false;
}

bool sync(std::string& error) {
    const auto status = cudaDeviceSynchronize();
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot synchronize: ") + cudaGetErrorString(status);
    return false;
}

bool copy(void* dst, const void* src, size_t bytes, std::string& error) {
    if (!bytes) return true;
    if (!dst || !src) return fail(error, "missing running-state buffer");
    const auto status = cudaMemcpy(dst, src, bytes, cudaMemcpyDefault);
    if (status == cudaSuccess) return true;
    error = std::string("conversation snapshot running-state copy: ") + cudaGetErrorString(status);
    return false;
}

// The session's carve (#216): `qsa_states` keeps global ordinals and holds the owned ones
// [qsa_ord0, qsa_ord0 + qsa_alloc); `gdn_state` holds `gdn_alloc` rows. A whole-model session owns every layer.
size_t owned_qsa(const SessionState& ss) { return (size_t) std::max<int64_t>(ss.qsa_alloc, 0); }
const QsaState& owned(const SessionState& ss, size_t j) { return ss.qsa_states[(size_t) ss.qsa_ord0 + j]; }

bool image_keys(const std::vector<ConversationImageKey>& images, size_t tokens) {
    int64_t previous = -1;
    for (const auto& image : images) {
        if (image.start <= previous || image.start < 0 || (uint64_t) image.start >= tokens) return false;
        previous = image.start;
    }
    return true;
}

bool checkpoint_targets(const SessionState& ss, const ModelGeometry& g, size_t tokens,
                        ConversationStateSizes& z, std::string& error) {
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (ss.max_cells < 0 || tokens > (uint64_t) ss.max_cells || (z.gdn && !ss.gdn_state) ||
        (owned_qsa(ss) && !ss.qsa_states)) return fail(error, "invalid session running-state targets");
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        const auto block = strata::kernels::qsa_real_shapes().idx_block;
        size_t pooled_bytes = 0;
        if (!st.idx_tail || !st.idx_dead || !st.idx_block_pos || !st.idx_pooled ||
            st.max_cells < 0 || tokens > (uint64_t) st.max_cells ||
            (tokens && tokens / (uint64_t) block >= (uint64_t) std::max<int64_t>(0, st.idx_pooled_rows)) ||
            !product(pooled_bytes, {tokens / (uint64_t) block + 1, (uint64_t) g.idx_key_dim, sizeof(float)}))
            return fail(error, "invalid indexer running-state target");
    }
    return true;
}

// ---- the layer split (S3.1a) ----
//
// An empty stage set is "no split": every single-GPU path is exactly what it was before.
int64_t stage_count(const ConversationStageSet& s) { return s.empty() ? 0 : s.count; }

// The caller's stage list must be usable before anything reads a session through it.
// Together with the main session (CUDA0's, the snapshot's `session` argument) the stages must
// tile the model exactly: increasing, non-overlapping, and covering [0, n_layers). A split that
// skips or double-covers a layer would park a conversation with state missing from it.
bool stage_set_validate(const ConversationStageSet& stages, const SessionState& main_ss,
                        const ModelGeometry& g, std::string& error) {
    if (stages.empty()) return true;
    if (stages.count > (int64_t) (std::numeric_limits<size_t>::max() / sizeof(ConversationStageSnapshot)))
        return fail(error, "too many layer-split stages");
    const int64_t n_qsa = g.n_qsa_layers();
    int64_t next = main_ss.layer_hi;
    for (int64_t i = 0; i < stages.count; ++i) {
        const SessionState* ss = stages.stages[i].session;
        if (ss == nullptr) return fail(error, "layer-split stage has no session");
        ConversationStateSizes z;
        if (!conversation_session_sizes(g, *ss, z, error)) return false;
        if (ss->layer_lo != next || ss->layer_lo >= ss->layer_hi || ss->layer_hi > g.n_layers)
            return fail(error, "layer-split stages do not tile the model");
        if (ss->qsa_ord0 < (int64_t) (ss->layer_lo / std::max<int64_t>(g.qsa_interval, 1)))
            return fail(error, "layer-split stage carve starts before its layer range");
        if (ss->qsa_alloc > n_qsa - ss->qsa_ord0) return fail(error, "layer-split stage carve past the last QSA layer");
        next = ss->layer_hi;
    }
    if (next != g.n_layers) return fail(error, "layer-split stages do not cover the whole model");
    return true;
}

// One stage's own contribution to a snapshot: its running state, its indexer payloads and the
// K/V of every QSA layer inside its carve.
bool stage_kv_bytes(const SessionState& ss, const ModelGeometry& g, int64_t upto, size_t& total, std::string& error) {
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (!checkpoint_targets(ss, g, (size_t) upto, z, error)) return false;
    const size_t layers = owned_qsa(ss);
    size_t directory = 0, tails = 0, dead = 0, positions = 0, ids = 0;
    if (!product(directory, {layers, sizeof(ConversationKv)}) || !product(tails, {layers, z.tail}) ||
        !product(dead, {layers, z.dead}) || !product(positions, {layers, z.block_pos}) ||
        !product(ids, {(uint64_t) upto, sizeof(int32_t)}) ||
        !add(total, directory) || !add(total, ids) || !add(total, tails) || !add(total, dead) ||
        !add(total, positions) || !add(total, z.gdn) || !add(total, ss.ple_hist ? z.ple : 0))
        return fail(error, "stage byte count overflow");
    for (size_t j = 0; j < layers; ++j) {
        const size_t n = conversation_kv_bytes(owned(ss, j), g, upto, true);
        if (!n || !add(total, n)) return fail(error, "invalid or overflowing stage K/V byte estimate");
    }
    return true;
}

// `stages` empty: the single-GPU rule, where a checkpoint may carry no stage parts at all and
// one that does is rejected (that is what parking used to be).  With a split the rule inverts:
// every checkpoint must carry exactly one part per later stage, sized for that stage's carve,
// because a checkpoint the engine can resume from is only complete when every stage can.
bool view_validate(const ConversationView& view, const SessionState& ss, const ConversationStageSet& stages,
                   const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, view.ids.size(), z, error)) return false;
    if (view.ids.empty() || !image_keys(view.images, view.ids.size()) ||
        std::any_of(view.ids.begin(), view.ids.end(), [](int32_t id) { return id < 0; }))
        return fail(error, "invalid live token/image metadata");
    for (const auto& c : view.checkpoints) {
        if (c.ids.empty() || c.ids.size() > view.ids.size() ||
            !std::equal(c.ids.begin(), c.ids.end(), view.ids.begin()))
            return fail(error, "checkpoint is not a live token prefix");
        if ((int64_t) c.stage_parts.size() != stage_count(stages))
            return fail(error, stages.empty() ? "layer-split parking is not supported"
                                              : "checkpoint is missing a layer-split stage part");
        if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
        for (size_t i = 0; i < c.stage_parts.size(); ++i) {
            const ConversationCheckpoint& part = c.stage_parts[i];
            // A stage part is the same shape as the checkpoint it belongs to, but for the stage's
            // carve: the ids say how far it goes, the images belong to the whole conversation.
            if (part.ids.size() != c.ids.size() || !part.imgs.empty() || !part.stage_parts.empty())
                return fail(error, "layer-split checkpoint part is not a live token prefix");
            if (!conversation_checkpoint_validate(part, *stages.stages[i].session, g, error)) return false;
        }
        size_t image = 0;
        for (const auto& key : view.images) {
            if ((uint64_t) key.start >= c.ids.size()) break;
            if (image >= c.imgs.size() || !(c.imgs[image++] == key))
                return fail(error, "checkpoint image identity differs");
        }
        if (image != c.imgs.size()) return fail(error, "checkpoint image prefix differs");
    }
    return true;
}

bool metadata_bytes(const ConversationCheckpoint& c, size_t& total) {
    size_t ids = 0, images = 0;
    if (!product(ids, {c.ids.size(), sizeof(int32_t)}) ||
        !product(images, {c.imgs.size(), sizeof(ConversationImageKey)})) return false;
    for (size_t n : {ids, images, c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()})
        if (!add(total, n)) return false;
    // A split's checkpoint carries one part per later stage; those parts are real running state,
    // plus the directory they live in (bytes() counts the same thing).
    if (!product(images, {c.stage_parts.capacity(), sizeof(ConversationCheckpoint)}) || !add(total, images)) return false;
    for (const auto& part : c.stage_parts)
        if (!metadata_bytes(part, total)) return false;
    return true;
}
} // namespace

bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& z, std::string& error) {
    z = {};
    const auto key = geometry_key(g);
    for (size_t i = 0; i < key.size(); ++i)
        if (key[i] < 0 || (i != 1 && key[i] == 0)) return fail(error, "invalid model geometry");
    size_t recurrence = 0, convolution = 0;
    if (!product(recurrence, {(uint64_t) g.ssm_state_size, (uint64_t) g.ssm_v_heads, (uint64_t) g.ssm_state_size}) ||
        !product(convolution, {(uint64_t) g.ssm_conv_channels, (uint64_t) (g.ssm_d_conv - 1)}) ||
        !add(recurrence, convolution) ||
        !product(z.gdn, {(uint64_t) g.n_gdn_layers(), recurrence, sizeof(float)}) ||
        !product(z.ple, {strata::kernels::NG_HIST, strata::kernels::NG_HC_DIM, sizeof(float)}) ||
        !product(z.tail, {(uint64_t) (strata::kernels::qsa_real_shapes().idx_block - 1),
                          (uint64_t) g.idx_key_dim, sizeof(float)}) ||
        !product(z.dead, {(uint64_t) g.idx_key_dim, sizeof(float)}))
        return fail(error, "running-state byte count overflow");
    z.block_pos = sizeof(int32_t);
    size_t total = 0;
    if (!product(total, {(uint64_t) g.n_qsa_layers(), z.tail}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.dead}) ||
        !product(total, {(uint64_t) g.n_qsa_layers(), z.block_pos}))
        return fail(error, "indexer byte count overflow");
    return true;
}

bool conversation_session_sizes(const ModelGeometry& g, const SessionState& ss, ConversationStateSizes& z,
                                std::string& error) {
    if (!conversation_state_sizes(g, z, error)) return false;
    const int64_t n_gdn = g.n_gdn_layers(), n_qsa = g.n_qsa_layers();
    if (ss.gdn_alloc < 0 || ss.gdn_alloc > n_gdn || ss.qsa_ord0 < 0 || ss.qsa_alloc < 0 ||
        ss.qsa_ord0 > n_qsa || ss.qsa_alloc > n_qsa - ss.qsa_ord0)
        return fail(error, "invalid session layer carve");
    // z.gdn is n_gdn whole rows; the session holds gdn_alloc of them (all of them for the default full range)
    z.gdn = n_gdn > 0 ? z.gdn / (size_t) n_gdn * (size_t) ss.gdn_alloc : 0;
    return true;
}

bool conversation_checkpoint_validate(const ConversationCheckpoint& c, const SessionState& ss,
                                      const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (c.gdn.size() != z.gdn || c.ple.size() != (ss.ple_hist ? z.ple : 0) ||
        c.tails.size() != layers * z.tail || c.dead.size() != layers * z.dead ||
        c.block_pos.size() != layers * z.block_pos || !image_keys(c.imgs, c.ids.size()))
        return fail(error, "invalid checkpoint running-state payload");
    return true;
}

bool conversation_checkpoint_save(ConversationCheckpoint& c, const SessionState& ss,
                                  const ModelGeometry& g, std::string& error) {
    ConversationStateSizes z;
    if (!checkpoint_targets(ss, g, c.ids.size(), z, error)) return false;
    const size_t layers = owned_qsa(ss);
    c.gdn.resize(z.gdn); c.ple.resize(ss.ple_hist ? z.ple : 0);
    c.tails.resize(layers * z.tail); c.dead.resize(layers * z.dead); c.block_pos.resize(layers * z.block_pos);
    if (!copy(c.gdn.data(), ss.gdn_state, c.gdn.size(), error) ||
        !copy(c.ple.data(), ss.ple_hist, c.ple.size(), error)) return false;
    for (size_t j = 0; j < layers; ++j) {
        const auto& st = owned(ss, j);
        if (!copy(c.tails.data() + j * z.tail, st.idx_tail, z.tail, error) ||
            !copy(c.dead.data() + j * z.dead, st.idx_dead, z.dead, error) ||
            !copy(c.block_pos.data() + j * z.block_pos, st.idx_block_pos, z.block_pos, error)) return false;
    }
    return true;
}

bool conversation_checkpoint_restore(const ConversationCheckpoint& c, SessionState& ss,
                                     const ModelGeometry& g, std::string& error) {
    if (!conversation_checkpoint_validate(c, ss, g, error)) return false;
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    if (!copy(ss.gdn_state, c.gdn.data(), c.gdn.size(), error) ||
        !copy(ss.ple_hist, c.ple.data(), c.ple.size(), error)) return false;
    for (size_t j = 0; j < owned_qsa(ss); ++j) {
        const auto& st = owned(ss, j);
        if (!copy(st.idx_tail, c.tails.data() + j * z.tail, z.tail, error) ||
            !copy(st.idx_dead, c.dead.data() + j * z.dead, z.dead, error) ||
            !copy(st.idx_block_pos, c.block_pos.data() + j * z.block_pos, z.block_pos, error)) return false;
        if (!c.ids.empty()) {
            const size_t row = c.ids.size() / strata::kernels::qsa_real_shapes().idx_block;
            if (!copy(st.idx_pooled + row * g.idx_key_dim, c.dead.data() + j * z.dead, z.dead, error)) return false;
        }
    }
    const size_t tokens = c.ids.size();
    ss.ple_prev[0] = tokens >= 2 ? c.ids[tokens - 2] : -1;
    ss.ple_prev[1] = tokens >= 1 ? c.ids[tokens - 1] : -1;
    return sync(error);
}

// The main session's own contribution: running state, indexer payloads, and the K/V of every
// QSA layer it owns plus the draft layer's K/V.
bool main_snapshot_bytes(const ConversationView& view, const SessionState& ss, const ModelGeometry& g,
                         const QsaState& draft, size_t& bytes, std::string& error) {
    ConversationStateSizes z;
    if (!conversation_session_sizes(g, ss, z, error)) return false;
    size_t ids = 0, images = 0, checkpoints = 0, layers = 0, tails = 0, dead = 0, positions = 0;
    const auto qsa = (uint64_t) owned_qsa(ss);
    if (!product(ids, {view.ids.size(), sizeof(int32_t)}) ||
        !product(images, {view.images.size(), sizeof(ConversationImageKey)}) ||
        !product(checkpoints, {view.checkpoints.size(), sizeof(ConversationCheckpoint)}) ||
        !product(layers, {qsa + 1, sizeof(ConversationKv)}) ||
        !product(tails, {qsa, z.tail}) || !product(dead, {qsa, z.dead}) ||
        !product(positions, {qsa, z.block_pos})) return fail(error, "snapshot metadata byte count overflow");
    for (size_t n : {ids, images, checkpoints, layers, tails, dead, positions, z.gdn, ss.ple_hist ? z.ple : 0})
        if (!add(bytes, n)) return fail(error, "snapshot byte count overflow");
    for (const auto& c : view.checkpoints)
        if (!metadata_bytes(c, bytes)) return fail(error, "checkpoint byte count overflow");
    const int64_t upto = (int64_t) view.ids.size(); // view_validate bounds this by signed max_cells
    for (uint64_t i = 0; i <= qsa; ++i) {
        const auto& st = i == qsa ? draft : owned(ss, (size_t) i);
        const size_t n = conversation_kv_bytes(st, g, upto, i != qsa);
        if (!n || !add(bytes, n)) return fail(error, "invalid or overflowing K/V byte estimate");
    }
    return true;
}

// The directory the stage parts themselves live in. Only counted when there is a split, so a
// single-GPU estimate is byte-identical to what it was before.
bool stage_directory_bytes(const ConversationStageSet& stages, size_t& bytes, std::string& error) {
    if (stages.empty()) return true;
    size_t directory = 0;
    if (!product(directory, {(uint64_t) stages.count, sizeof(ConversationStageSnapshot)}) ||
        !add(bytes, directory)) return fail(error, "layer-split stage directory overflow");
    return true;
}

bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss,
                                 const ConversationStageSet& stages, const ModelGeometry& g,
                                 const QsaState& draft, size_t& bytes, std::string& error) {
    bytes = 0;
    if (!stage_set_validate(stages, ss, g, error)) return false;
    if (!view_validate(view, ss, stages, g, error)) return false;
    if (!main_snapshot_bytes(view, ss, g, draft, bytes, error)) return false;
    if (!stage_directory_bytes(stages, bytes, error)) return false;
    const int64_t upto = (int64_t) view.ids.size();
    for (int64_t i = 0; i < stage_count(stages); ++i)
        if (!stage_kv_bytes(*stages.stages[i].session, g, upto, bytes, error)) return false;
    return true;
}

bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& ss,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error) {
    return conversation_snapshot_bytes(view, ss, {}, g, draft, bytes, error);
}

bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ConversationStageSet& stages,
                                         const ModelGeometry& g, const QsaState& draft,
                                         size_t& bytes, std::string& error) {
    if (!conversation_snapshot_bytes(view, ss, stages, g, draft, bytes, error)) return false;
    if (reuse.kv.empty()) return true;
    // Retention is the single-GPU optimization (the pages a mount left behind in this process).
    // A split's stage K/V is always captured fresh, so the retained set must be exactly the main
    // session's layers plus the draft - never a stage's.
    const size_t layers = owned_qsa(ss) + 1;
    if (reuse.kv.size() != layers || reuse.unchanged_tokens < 0 ||
        reuse.unchanged_tokens > reuse.captured_tokens || reuse.unchanged_tokens > int64_t(view.ids.size()))
        return fail(error, "invalid retained K/V prefix");
    for (size_t i = 0; i < layers; ++i) {
        const bool index = i + 1 != layers;
        const auto& st = index ? owned(ss, i) : draft;
        if (!conversation_kv_validate(reuse.kv[i], st, g, reuse.captured_tokens, index, error)) return false;
        const size_t fresh = conversation_kv_bytes(st, g, int64_t(view.ids.size()), index);
        size_t retained = 0;
        if (!conversation_kv_capture_bytes(reuse.kv[i], st, g, int64_t(view.ids.size()), index, retained, error)) return false;
        bytes -= fresh;
        if (!add(bytes, retained)) return fail(error, "retained K/V allocation overflow");
    }
    size_t directory = 0;
    if (!product(directory, {reuse.kv.capacity() - layers, sizeof(ConversationKv)}) || !add(bytes, directory))
        return fail(error, "retained K/V directory overflow");
    return true;
}

bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& ss, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error) {
    return conversation_snapshot_capture_bytes(reuse, view, ss, {}, g, draft, bytes, error);
}

bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& ss, const ConversationStageSet& stages,
                                const ModelGeometry& g, const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes) {
    size_t estimate = 0;
    if (!conversation_snapshot_capture_bytes(reuse, view, ss, stages, g, draft, estimate, error)) return false;
    // Build into a new object so a failure cannot publish a partial snapshot.
    SavedConversation captured;
    captured.geometry = geometry_key(g);
    captured.layer_lo = ss.layer_lo; captured.layer_hi = ss.layer_hi;
    captured.live.ids = view.ids; captured.live.imgs = view.images;
    captured.cvec = view.cvec; captured.checkpoints = view.checkpoints;
    const int64_t unchanged = reuse.kv.empty() ? 0 : reuse.unchanged_tokens;
    captured.kv = std::move(reuse.kv);
    const size_t layers = owned_qsa(ss);
    captured.kv.resize(layers + 1);
    const int64_t upto = (int64_t) view.ids.size();
    // The main session first, exactly the single-GPU order.  With a split the main session is
    // CUDA0's, so make that current rather than trusting the caller's.
    {
        const OnDevice on(stages.main_device);
        if (!sync(error)) return false;
        if (!conversation_checkpoint_save(captured.live, ss, g, error)) return false;
        for (size_t j = 0; j < layers; ++j)
            if (!conversation_kv_save(captured.kv[j], owned(ss, j), g, upto, true, error,
                                      unchanged, reused_bytes)) return false;
    }
    // The draft state is NOT on the main session's device when the model is split: `MtpDrafter`
    // binds to the LAST stage, so its pools, its host copies and its stream all live there.
    {
        const OnDevice on(stages.empty() ? -1 : stages.draft_device);
        if (!stages.empty() && !sync(error)) return false;
        // The draft's final cell may not have been computed when the output cap was
        // reached. Refresh that page even when the main prefix continued unchanged.
        if (!conversation_kv_save(captured.kv.back(), draft, g, upto, false, error,
                                  std::max<int64_t>(0, unchanged - 1), reused_bytes)) return false;
    }
    // Then every later stage, each with ITS OWN device current: the state lives there, and
    // `cudaDeviceSynchronize()` only ever reaches the device the caller is on.
    captured.stage_parts.resize((size_t) stage_count(stages));
    for (int64_t i = 0; i < stage_count(stages); ++i) {
        const ConversationStage& stage = stages.stages[i];
        const SessionState& stage_ss = *stage.session;
        ConversationStageSnapshot& part = captured.stage_parts[(size_t) i];
        part.layer_lo = stage_ss.layer_lo; part.layer_hi = stage_ss.layer_hi;
        part.state.ids = view.ids;
        const size_t stage_layers = owned_qsa(stage_ss);
        part.kv.resize(stage_layers);
        const OnDevice on(stage.device);
        if (!sync(error)) return false;
        if (!conversation_checkpoint_save(part.state, stage_ss, g, error)) return false;
        for (size_t j = 0; j < stage_layers; ++j)
            if (!conversation_kv_save(part.kv[j], owned(stage_ss, j), g, upto, true, error)) return false;
    }
    image = std::move(captured);
    return true;
}

bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& ss, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse, size_t* reused_bytes) {
    return conversation_snapshot_save(image, view, ss, {}, g, draft, error, std::move(reuse), reused_bytes);
}

bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss,
                                    const ConversationStageSet& stages, const ModelGeometry& g,
                                    const QsaState& draft, std::string& error) {
    if (!stage_set_validate(stages, ss, g, error)) return false;
    if (!image.live.stage_parts.empty()) return fail(error, "layer-split parking is not supported");
    if ((int64_t) image.stage_parts.size() != stage_count(stages))
        return fail(error, stages.empty() ? "layer-split parking is not supported"
                                          : "snapshot is missing a layer-split stage part");
    if (image.geometry != geometry_key(g)) return fail(error, "incompatible runtime geometry");
    // an image holds exactly one carve's running state and K/V: same layer range or nothing
    if (image.layer_lo != ss.layer_lo || image.layer_hi != ss.layer_hi)
        return fail(error, "snapshot from another session layer range");
    const ConversationView view{image.live.ids, image.live.imgs, image.checkpoints, image.cvec};
    if (!view_validate(view, ss, stages, g, error) || !conversation_checkpoint_validate(image.live, ss, g, error)) return false;
    const size_t layers = owned_qsa(ss);
    if (image.kv.size() != layers + 1) return fail(error, "invalid K/V layer count");
    const int64_t upto = (int64_t) image.live.ids.size();
    for (size_t j = 0; j < layers; ++j)
        if (!conversation_kv_validate(image.kv[j], owned(ss, j), g, upto, true, error)) return false;
    if (!conversation_kv_validate(image.kv.back(), draft, g, upto, false, error)) return false;
    // Every stage part validated too, before the first write anywhere. A stage whose carve moved
    // (a different --layer-split between park and mount) is a different conversation image.
    for (int64_t i = 0; i < (int64_t) image.stage_parts.size(); ++i) {
        const ConversationStageSnapshot& part = image.stage_parts[(size_t) i];
        const SessionState& stage_ss = *stages.stages[i].session;
        if (part.layer_lo != stage_ss.layer_lo || part.layer_hi != stage_ss.layer_hi)
            return fail(error, "layer-split stage part from another session layer range");
        if (!part.state.imgs.empty() || !part.state.stage_parts.empty() ||
            part.state.ids.size() != image.live.ids.size() ||
            !std::equal(part.state.ids.begin(), part.state.ids.end(), image.live.ids.begin()))
            return fail(error, "layer-split stage part is not a live token prefix");
        if (!conversation_checkpoint_validate(part.state, stage_ss, g, error)) return false;
        if (part.kv.size() != owned_qsa(stage_ss)) return fail(error, "invalid layer-split stage K/V layer count");
        for (size_t j = 0; j < part.kv.size(); ++j)
            if (!conversation_kv_validate(part.kv[j], owned(stage_ss, j), g, upto, true, error)) return false;
    }
    return true;
}

bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& ss,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error) {
    return conversation_snapshot_validate(image, ss, {}, g, draft, error);
}

ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                  const ConversationStageSet& stages, const ModelGeometry& g,
                                                  const QsaState& draft, std::string& error) {
    // Nothing below runs until the whole image - main session, every stage, every checkpoint -
    // has validated. A rejected image never leaves a half-mounted conversation.
    if (!conversation_snapshot_validate(image, ss, stages, g, draft, error)) return ConversationRestore::invalid;
    const int64_t upto = (int64_t) image.live.ids.size();
    {
        const OnDevice on(stages.main_device);
        if (!sync(error)) return ConversationRestore::transfer_failed;
        for (size_t j = 0; j < owned_qsa(ss); ++j)
            if (!conversation_kv_restore(image.kv[j], owned(ss, j), g, upto, true, error))
                return ConversationRestore::transfer_failed;
    }
    // The drafter lives on the last stage's device on a split; its ring refill (`kv_ring_restore`)
    // and its host pools are all there, not on CUDA0.  With no split `main_device`/`draft_device`
    // are both -1 and this is the same call, in the same place in the sequence, that the
    // single-GPU path always made.
    {
        const OnDevice on(stages.empty() ? -1 : stages.draft_device);
        if (!conversation_kv_restore(image.kv.back(), draft, g, upto, false, error))
            return ConversationRestore::transfer_failed;
    }
    {
        const OnDevice on(stages.main_device);
        if (!conversation_checkpoint_restore(image.live, ss, g, error)) return ConversationRestore::transfer_failed;
    }
    for (int64_t i = 0; i < (int64_t) image.stage_parts.size(); ++i) {
        const ConversationStage& stage = stages.stages[i];
        const ConversationStageSnapshot& part = image.stage_parts[(size_t) i];
        const OnDevice on(stage.device);
        for (size_t j = 0; j < part.kv.size(); ++j)
            if (!conversation_kv_restore(part.kv[j], owned(*stage.session, j), g, upto, true, error))
                return ConversationRestore::transfer_failed;
        if (!conversation_checkpoint_restore(part.state, *stage.session, g, error))
            return ConversationRestore::transfer_failed;
    }
    return ConversationRestore::restored;
}

ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& ss,
                                                   const ModelGeometry& g, const QsaState& draft, std::string& error) {
    return conversation_snapshot_restore(image, ss, {}, g, draft, error);
}
} // namespace strata::core
