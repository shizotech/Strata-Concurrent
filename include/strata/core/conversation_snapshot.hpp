#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

#include <string>

namespace strata::core {

// Caller synchronizes the device before saving, and after restoring all layers.
// include_index is false for the draft layer (its attention has no indexer).
size_t conversation_kv_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index);
// A nonzero unchanged_tokens is valid only for storage retained from an image
// actually restored into this session, bounded by every subsequent rewrite.
// Equal token IDs alone do not establish that its K/V bytes are unchanged.
bool conversation_kv_save(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error,
                          int64_t unchanged_tokens = 0, size_t* reused_bytes = nullptr);
bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                   int64_t upto, bool include_index, size_t& bytes, std::string& error);
// No CUDA calls or destination writes. Used for whole-session prevalidation.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, std::string& error);
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, std::string& error);
// Diagnostic read-back after a synchronized restore. Uses 64 KiB of stack
// workspace, compares authoritative bytes and resident draft-ring pages, and
// fingerprints the authoritative payload only. Never changes model state.
bool conversation_kv_verify(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, uint64_t& fingerprint, std::string& error);

struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
/// Whole-model sizes: `gdn` covers every GDN layer, the indexer sizes are per QSA layer.
bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& sizes, std::string& error);
/// The same for one session's layer carve (#216): `gdn` covers its `gdn_alloc` rows; the per-QSA-layer sizes
/// apply to each owned state [qsa_ord0, qsa_ord0 + qsa_alloc).  Rejects an inconsistent carve.  Checkpoints,
/// snapshots and their validation all use this: a session saves and restores only the state it owns.
bool conversation_session_sizes(const ModelGeometry& g, const SessionState& session, ConversationStateSizes& sizes,
                                std::string& error);
bool conversation_checkpoint_validate(const ConversationCheckpoint& checkpoint, const SessionState& session,
                                      const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_save(ConversationCheckpoint& checkpoint, const SessionState& session,
                                  const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_restore(const ConversationCheckpoint& checkpoint, SessionState& session,
                                     const ModelGeometry& g, std::string& error);

struct ConversationView {
    const std::vector<int32_t>& ids;
    const std::vector<ConversationImageKey>& images;
    const std::vector<ConversationCheckpoint>& checkpoints;
    bool cvec;
};

// ============================ A `--layer-split` PARKED CONVERSATION ============================
//
// A split runs the model across several devices, and each one's `SessionState` owns only its
// layer carve (see the carve note in core/session.hpp).  A parked conversation is the state of
// the WHOLE model, so on a split it is the first stage's state (the `session` argument, CUDA0)
// PLUS one part per later stage PLUS the MTP draft state, which lives on the LAST stage.
//
// `ConversationStageSet` is that list.  It is deliberately a view over the caller's sessions,
// not a copy: the engine owns them (`GpuStage::ss`), and the snapshot owns the bytes.
// `stages[i]` is stage i in the order the engine walks them, and it is the SAME order as
// `SavedConversation::stage_parts[i]` and as `ConversationCheckpoint::stage_parts[i]`.
// An empty set means "no split": every single-GPU path is byte-identical to what it was.
struct ConversationStage {
    SessionState* session = nullptr;
    int device = -1;   ///< the CUDA device that session's state lives on (-1: the caller's current one)
};
struct ConversationStageSet {
    const ConversationStage* stages = nullptr;
    int64_t count = 0;
    /// The device the MAIN session's state lives on.  On a split that is always CUDA0 (the engine
    /// builds `ss` under `OnDevice(0)`); -1 means "the caller's current device", which is what a
    /// single-GPU run - and a host-only test - wants.
    int main_device = -1;
    /// The device the draft `QsaState` lives on. On a split the drafter is bound to the LAST
    /// stage's device, so a save/restore must be issued there, not on the caller's.
    int draft_device = -1;

    bool empty() const { return stages == nullptr || count <= 0; }
};

bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error);
// The capture estimate includes retained capacity and transient segment directories;
// only estimate - reuse.bytes() requires additional physical RAM. Capture consumes
// the uniquely owned reusable buffers, including on failure.
// Caller admits the estimate before invoking capture. Allocation failures propagate
// to the RAM policy; the active session is never modified by capture.
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& session, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error);
enum class ConversationRestore { restored, invalid, transfer_failed };
// Invalid images are rejected before any CUDA call/write. Transfer failure may
// leave partial state: caller MUST NOT continue inference from that session.
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error);

// ---- the same five operations with a layer split (S3.1a) ----
//
// The single-GPU overloads above are exactly these with an empty `stages`, and they must stay
// that way: no format change, no new field written when there is no split.
//
// Contract, for every overload:
//   * `stages[i]` must be the stage whose sessions produced `SavedConversation::stage_parts[i]`
//     and `ConversationCheckpoint::stage_parts[i]` - same order, same carve.
//   * The draft state is the LAST stage's (that is where `MtpDrafter` binds), which is why
//     `ConversationStageSet::draft_device` exists.
//   * Validation is all-or-nothing: every stage's running state, every stage's K/V, every
//     checkpoint and the draft are validated BEFORE the first byte is written anywhere.
//     A restore that fails after the first write is `transfer_failed` and is fatal to the
//     session, exactly as the single-GPU path already is.
//   * `bytes()`/`capture_bytes()` include every stage part, so the RAM budget sees the truth.
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session,
                                 const ConversationStageSet& stages, const ModelGeometry& g,
                                 const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ConversationStageSet& stages,
                                         const ModelGeometry& g, const QsaState& draft,
                                         size_t& bytes, std::string& error);
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& session, const ConversationStageSet& stages,
                                const ModelGeometry& g, const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ConversationStageSet& stages, const ModelGeometry& g,
                                    const QsaState& draft, std::string& error);
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                  const ConversationStageSet& stages, const ModelGeometry& g,
                                                  const QsaState& draft, std::string& error);

} // namespace strata::core
