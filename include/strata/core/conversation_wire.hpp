// include/strata/core/conversation_wire.hpp - S4.3.1: the hand-off PAYLOAD CODEC.
//
// `SavedConversation` (include/strata/core/conversation_cache.hpp:124-162) is what the Stage-4 split has to
// move from the prefill instance to a decode instance: the live branch, its retained checkpoint chain, every
// layer's K/V, the draft layer's K/V, and on a `--layer-split` one part per later stage.  It is 237 MB ...
// 3.18 GB measured (docs/STAGE4-SPLIT-ROLES.md Appendix A.2) and it is NOT memcpy-able:
//
//   * every container is a `std::vector` / `std::deque` holding pointers into the writer's heap;
//   * `ConversationBuffer` (conversation_buffer.hpp:16-152) is a segment directory of `std::vector<uint8_t>`
//     with a 16 MiB segment cap and geometric growth.  It has NO `data()` and no contiguous payload - the
//     only accessors are `size()`, `empty()`, `bytes()`, `resize()`, `visit()`, `read()` and `operator==`;
//   * `bool` is not a wire type.
//
// So this is a RECORD STREAM, not a struct dump: every scalar at a fixed width, every `ConversationBuffer`
// reduced to a `{bytes, offset}` reference into one 64-byte-aligned blob pool, and the reader rebuilds real
// freshly-owned buffers with `resize()` + chunked `visit()` copies.  That is the same shape the snapshot's own
// save path uses (`src/core/conversation_snapshot.cpp:160-171` walks buffers with `visit`), so no new copy
// primitive is introduced anywhere.
//
// The format is specified in docs/STAGE4-SPLIT-ROLES.md §2.  Everything below implements it; §DEVIATIONS
// lists every place this file resolves an ambiguity or a mistake in that document, so S4.3.2/.5/.6 see ONE
// truth.  Read that section before calling anything here.
//
// ================================ THE PAYLOAD ================================
//
// Inside one hand-off slot, after the 4096-byte slot header (§3.4), the payload is
//
//     payload := header_record(256)  body_records  zero_pad  blob_pool
//
// `header_record` - the §2.3 field list, 232 bytes, padded to 256, little-endian, no padding holes:
//
//   offset   0  u32   magic          'S','R','4','H'  (0x53 0x52 0x34 0x48)
//   offset   4  u16   fmt_version    = 1
//   offset   6  u16   flags          bit0 cvec, bit1 has_stage_parts, bit2 partial (must be 0 here)
//   offset   8  u64   geometry[18]   SavedConversation::geometry, bit-for-bit
//   offset 152  u64   layer_lo
//   offset 160  u64   layer_hi
//   offset 168  i64   n_kv           kv.size()  (main layers, then the draft layer)
//   offset 176  i64   n_checkpoints  checkpoints.size()
//   offset 184  i64   n_stage_parts  stage_parts.size()   (0 = single-GPU image)
//   offset 192  i64   live_tokens    live.ids.size()
//   offset 200  u64   blob_pool_off  offset from the payload start of the blob pool
//   offset 208  u64   blob_pool_bytes
//   offset 216  u64   geom_hash      §3.10 - the caller's value, see deviation 3
//   offset 224  u64   pack_hash      §3.10 - the caller's value, see deviation 3
//   ---- the 24 bytes §2.3 leaves as padding (deviation 2) ----
//   offset 232  u64   payload_bytes  the TOTAL payload length, including this header and the pool
//   offset 240  u64   payload_hash   ALWAYS ZERO on the wire; see deviation 4
//   offset 248  u64   reserved       must be 0
//
// `body_records` - a fixed order, so a reader never searches (§2.3):
//
//   1. KV array          n_kv KV records, in SavedConversation::kv order (main layers, then the draft)
//   2. live              ONE CHECKPOINT record
//   3. checkpoints       n_checkpoints CHECKPOINT records, in vector order
//   4. stage_parts       n_stage_parts STAGE records, in engine order
//   5. END               a record whose kind IS the ASCII magic 'E','N','D','!'
//
// A record is `{u32 kind; u32 flags; u64 bytes}` (16 bytes) followed by exactly `bytes` body bytes.
// `flags` is reserved and must be 0.  `kind` 1 = KV, 2 = CHECKPOINT, 3 = STAGE, and kind 4 is spelled as
// the four ASCII bytes 'END!' so a scanner can find the terminator in a hex dump (deviation 5).
//
//   kind 1  KV          body = 136 bytes, fixed
//       i32 format; i32 pad(0)
//       i64 cells, heads, head_dim, page_size, pooled_rows, idx_dim
//       BLOBREF k, v, k_scale, v_scale, pooled
//   kind 2  CHECKPOINT
//       i64 n_ids; i32 ids[n_ids]; zero-pad to 8
//       i64 n_imgs; {i64 start; u64 hash}[n_imgs]
//       BLOBREF gdn, ple, tails, dead, block_pos
//       u64 used
//       i64 n_stage_parts; CHECKPOINT[n_stage_parts]      (recursive, same kind)
//   kind 3  STAGE
//       i64 layer_lo; i64 layer_hi
//       CHECKPOINT state                                   (a full record, header included)
//       i64 n_kv; KV[n_kv]                                 (a full record each)
//
// `BLOBREF` = `{u64 bytes; u64 offset}`, `offset` from the payload start.  A zero-length buffer is `{0, 0}`
// and has no pool entry.  Every pool entry starts on a 64-byte boundary and the gap between entries is
// zero-filled, so the payload is fully deterministic: encode(decode(x)) == x byte-for-byte, and
// decode(encode(x)) reproduces every field.
//
// ================================ DEVIATIONS FROM docs/STAGE4-SPLIT-ROLES.md §2 ================================
//
// 0. **§2.1's table is wrong about the checkpoint's running state.**  It says `ConversationCheckpoint`
//    carries "five `ConversationBuffer`s (`gdn`, `ple`, `tails`, `dead`, `block_pos`)".  It does not -
//    `conversation_cache.hpp:60` declares them as plain `std::vector<uint8_t>`.  Only `ConversationKv`
//    (conversation_cache.hpp:83) holds five real `ConversationBuffer`s.  The wire format is unchanged either
//    way - both become a BLOBREF into the same pool - but the codec handles the two storage types
//    separately, and a vector's payload IS contiguous.  Anyone writing S4.3.5/.6 should not go looking for
//    `visit()` on a checkpoint.
//
// 1. **`owner` is not on the wire at all.**  §2.1 lists it as "yes, but remapped", and §2.4 then says the
//    writer always writes `kNoOwner` and the reader "must ignore" it.  A field that is always the same
//    constant and must never be read is not a wire field - carrying it would only create a way to be wrong
//    (the S3.9 theft bug in a new costume).  `conversation_wire_encode` never reads `image.owner`;
//    `conversation_wire_decode` always yields `owner == kNoOwner`.  The receiver assigns the claim itself at
//    `ConversationCache::put(std::move(image), req_id, held)` (conversation_cache.hpp:388-396).  That is
//    exactly §2.4's rule, enforced by construction instead of by convention.
// 2. **The 24 bytes §2.3 leaves as padding are used, not left blank.**  The task contract requires a total
//    byte count and a content hash inside the payload; the §2.3 field list is pinned at 232 bytes and the
//    record is pinned at 256, so the two new fields live in the padding - the same trick the expert arena
//    header uses for its protocol fields ("the new protocol fields therefore live in the header's existing
//    reserved[] slots rather than after it", pinned.hpp:25-32).  `sizeof(WireHeaderRecord) == 256` and
//    `kWireHeaderSpecBytes == 232` are both asserted, so the §2.3 layout is still pinned exactly.
// 3. **`geom_hash` / `pack_hash` are supplied by the caller, not computed here.**  §3.10 defines them over
//    `geometry_key(g)` plus the canonical split spec, but `geometry_key` is file-local to
//    `src/core/conversation_state.cpp:14-19` and `handoff_split_spec()`/`handoff_geom_hash()` live in
//    S4.3.2's `handoff.hpp`.  The codec therefore takes them as opaque u64 `WireIdentity` values and only
//    compares them.  S4.3.5 must pass the SAME two values it wrote into the slot header (§3.4), and S4.3.6
//    must pass the same values it read from it - then the payload check and the slot check are the same test.
// 4. **`payload_hash` is zero on the wire and is carried out-of-band.**  The content hash is
//    `fnv1a64` over `[0, payload_bytes)` of the payload AS WRITTEN.  If the field itself held the value the
//    hash would be self-referential, and patching it after the fact would make the payload's hash differ
//    from the arena slot's `payload_hash` (§3.4), which S4.3.2 computes over the raw slot bytes.  Writing
//    zeros keeps the two numbers IDENTICAL: one hash, checked in both places, computed in one streaming
//    pass with no patch and no second pass.  `conversation_wire_encode` returns it in `WireEncoded`, the
//    caller puts it in the slot header and on the `DONE` line, and `conversation_wire_decode` re-computes
//    and compares it when `WireExpect::check_payload_hash` is set.  A non-zero field on the wire is refused.
// 5. **The END marker is a record whose kind is the ASCII magic.**  §2.3 says both "kind 4 END" and "u32
//    magic 'E','N','D','!'".  Those are the same thing read two ways: `kWireEnd = 0x21444E45`, whose four
//    little-endian bytes spell `END!`.  It is a normal 16-byte record header with `bytes == 0`.
// 6. **Blob alignment is 64 bytes relative to the PAYLOAD start.**  §2.3 says "relative to the slot start".
//    A slot's payload begins at `slot_base + 4096` and every `tier_slot_bytes` is `round_up_64(...) + 64 MiB`
//    (§3.8), so `slot_base + 4096` is itself 64-aligned and the two readings are the same number.  Payload-
//    relative is what the codec can actually guarantee on its own, and it is what every offset in the format
//    is measured from.
// 7. **`live.stage_parts` must be empty, and encode refuses an image whose it isn't.**  A parked image keeps
//    its stage state in `SavedConversation::stage_parts`, not in `live` (conversation_cache.hpp:109-111), and
//    the engine already refuses the other shape with "layer-split parking is not supported"
//    (src/core/conversation_state.cpp:432).  Refusing it at the codec gives the same answer one step earlier
//    and before any byte crosses a process boundary.
// 8. **Every count is bounded by the bytes actually available before anything is allocated.**  §2.3 has no
//    anti-corruption bound.  A receiver must not `resize()` from a number a corrupt payload invented, so the
//    decoder checks `count * sizeof(element) <= remaining record bytes` first, caps the recursive
//    `stage_parts` depth at `kWireMaxCheckpointDepth` (the real shape is depth 1: a checkpoint's stage parts
//    never nest), and rejects any blob reference that leaves the pool.  A payload that survives all of that
//    still cannot make the receiver allocate more than the payload itself is.
//
// ================================ WHAT S4.3.5 / S4.3.6 CALL ================================
//
//   prefill instance (writer) - after `HandoffArena::claim()`, before `publish()`:
//
//       WirePlan plan;
//       if (!conversation_wire_plan(image, plan, err)) { /* image cannot be handed off */ }
//       if (plan.total_bytes > claim.capacity) { /* pick a bigger tier; §3.8 */ }
//       SlotWireSink sink(fd, claim.payload_off, claim.capacity);
//       WireEncoded enc;
//       if (!conversation_wire_encode(image, sink, {geom_hash, pack_hash}, enc, err)) { /* abort_claim */ }
//       arena.publish(claim, tokens, enc.bytes, enc.payload_hash, err);   // the SAME hash, deviation 4
//
//   decode instance (reader) - after `HandoffArena::read_slot()`, before anything touches a session:
//
//       SpanWireSource src(payload.data, payload.payload_bytes);
//       WireExpect expect;
//       expect.geometry  = this_process_geometry_key;  expect.check_geometry = true;
//       expect.geom_hash = my_geom_hash; expect.pack_hash = my_pack_hash; expect.check_hashes = true;
//       expect.payload_hash = payload.payload_hash;    expect.check_payload_hash = true;
//       expect.stage_parts = my_stage_count;           // §2.5 step 2, both directions
//       expect.max_payload_bytes = slot_capacity;      // the "declared size exceeds the slot" refusal
//       SavedConversation image; WireDecoded got;
//       if (!conversation_wire_decode(src, image, got, expect, err)) { /* ERR badpayload, mount nothing */ }
//       conversations.put(std::move(image), req_id, held);   // THE RECEIVER CLAIMS IT (deviation 1)
//       // then §2.5 step 3: conversation_snapshot_validate(image, ss, conv_stages, g, mtp.kv_state(), err)
//
// `conversation_wire_decode` writes NOTHING into `image` unless the whole payload parsed: it builds into a
// local and moves it out on success, so a refusal costs the session nothing (the §2.5 "before the first
// write" rule, at the codec level).
//
// CPU-only by construction: no CUDA call, no session, no model, no file.  `fnv1a64` is the engine's existing
// FNV-1a (include/strata/core/pinned.hpp:173), reused rather than duplicated.
#pragma once

#include "strata/core/conversation_cache.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace strata::core {

// ================================ THE FORMAT CONSTANTS (an ABI - never move a value) ================================
/// 'S','R','4','H' - the four bytes at payload offset 0, read little-endian.
inline constexpr uint32_t kWireMagic = 0x48345253u;
inline constexpr uint16_t kWireFormatVersion = 1;
/// §2.3: the field list is 232 bytes; the record is padded to 256.
inline constexpr uint64_t kWireHeaderSpecBytes = 232;
inline constexpr uint64_t kWireHeaderRecordBytes = 256;
inline constexpr uint64_t kWireRecordHeaderBytes = 16;
/// §2.3: every pool entry starts on a 64-byte boundary.
inline constexpr uint64_t kWireBlobAlign = 64;
/// The FNV-1a 64 offset basis (`include/strata/core/pinned.hpp:173`), named so the encoder's running hash
/// and `conversation_wire_payload_hash()` cannot drift apart.
inline constexpr uint64_t kWireHashSeed = 1469598103934665603ull;
/// The largest single run the codec reads or writes at once.  A GiB payload is streamed through this.
inline constexpr uint64_t kWireChunkBytes = 1ull << 20;
/// Deviation 8: the recursive `stage_parts` bound.  The real shape is depth 1.
inline constexpr int kWireMaxCheckpointDepth = 8;
/// Structural caps.  Each is far above anything a real model produces and far below what a corrupt
/// payload would need to make the receiver allocate, and every one is also implied by the
/// "count * element size <= remaining bytes" rule - they exist so the reason names the field.
inline constexpr int64_t kWireMaxKvRecords = 4096;
inline constexpr int64_t kWireMaxCheckpoints = 1 << 20;
inline constexpr int64_t kWireMaxStageParts = 64;
inline constexpr int64_t kWireMaxImages = 1 << 16;

enum WireRecordKind : uint32_t {
    kWireRecordKv = 1,
    kWireRecordCheckpoint = 2,
    kWireRecordStage = 3,
    /// deviation 5: kind 4 spelled as the ASCII magic 'E','N','D','!'.
    kWireRecordEnd = 0x21444E45u,
};

enum WireHeaderFlags : uint16_t {
    kWireFlagCvec = 1u,
    kWireFlagHasStageParts = 2u,
    /// §3.5's slot-level "partial payload" flag.  A partial payload is never a valid payload, so a writer
    /// must never set it and a reader must refuse it.
    kWireFlagPartial = 4u,
    kWireFlagKnown = kWireFlagCvec | kWireFlagHasStageParts | kWireFlagPartial,
};

/// The §2.3 header record, exactly as it lies in the payload.  POD; every offset is `static_assert`ed
/// against the constants above, the `pinned.hpp:51-56` pattern.
struct WireHeaderRecord {
    uint32_t magic;
    uint16_t fmt_version;
    uint16_t flags;
    uint64_t geometry[18];
    uint64_t layer_lo;
    uint64_t layer_hi;
    int64_t n_kv;
    int64_t n_checkpoints;
    int64_t n_stage_parts;
    int64_t live_tokens;
    uint64_t blob_pool_off;
    uint64_t blob_pool_bytes;
    uint64_t geom_hash;
    uint64_t pack_hash;
    // ---- deviation 2: the 24 bytes §2.3 leaves as padding ----
    uint64_t payload_bytes;
    uint64_t payload_hash;   // deviation 4: always 0 on the wire
    uint64_t reserved;
};
static_assert(sizeof(WireHeaderRecord) == kWireHeaderRecordBytes);
static_assert(offsetof(WireHeaderRecord, magic) == 0);
static_assert(offsetof(WireHeaderRecord, fmt_version) == 4);
static_assert(offsetof(WireHeaderRecord, flags) == 6);
static_assert(offsetof(WireHeaderRecord, geometry) == 8);
static_assert(offsetof(WireHeaderRecord, layer_lo) == 152);
static_assert(offsetof(WireHeaderRecord, layer_hi) == 160);
static_assert(offsetof(WireHeaderRecord, n_kv) == 168);
static_assert(offsetof(WireHeaderRecord, n_checkpoints) == 176);
static_assert(offsetof(WireHeaderRecord, n_stage_parts) == 184);
static_assert(offsetof(WireHeaderRecord, live_tokens) == 192);
static_assert(offsetof(WireHeaderRecord, blob_pool_off) == 200);
static_assert(offsetof(WireHeaderRecord, blob_pool_bytes) == 208);
static_assert(offsetof(WireHeaderRecord, geom_hash) == 216);
static_assert(offsetof(WireHeaderRecord, pack_hash) == 224);
static_assert(offsetof(WireHeaderRecord, payload_bytes) == 232);
static_assert(offsetof(WireHeaderRecord, payload_hash) == 240);
static_assert(offsetof(WireHeaderRecord, reserved) == 248);
/// The §2.3 field list ends at 232; the 24 bytes after it are this file's deviation 2.
static_assert(offsetof(WireHeaderRecord, reserved) + 8 == kWireHeaderRecordBytes);
static_assert(offsetof(WireHeaderRecord, reserved) + 8 - 24 == kWireHeaderSpecBytes);

// ================================ THE SINK (encode output) ================================
/// A sink receives the payload strictly in order, in arbitrary chunks.  It must not assume a chunk
/// boundary lands anywhere useful - a 20 MiB `ConversationBuffer` arrives as twenty 1 MiB writes.
class WireSink {
public:
    virtual ~WireSink() = default;
    /// Append `bytes` at the current payload offset.  False aborts the encode.
    virtual bool write(const uint8_t* data, size_t bytes) = 0;
    /// Optional: WHY it refused.  The codec appends this to its own message, so an `errno` or "the slot is
    /// too small" reaches the log instead of a generic "the sink refused".
    virtual std::string refusal() const { return {}; }
};

/// Append to a byte vector.  For tests and for anything small enough to hold at once; a GiB payload should
/// use `SlotWireSink` so no second copy of it ever exists.
class VectorWireSink final : public WireSink {
public:
    explicit VectorWireSink(size_t reserve_bytes = 0) { bytes_.reserve(reserve_bytes); }
    bool write(const uint8_t* data, size_t bytes) override;
    const std::vector<uint8_t>& data() const { return bytes_; }
    std::vector<uint8_t>& data() { return bytes_; }

private:
    std::vector<uint8_t> bytes_;
};

/// Write straight into a hand-off slot's payload region through the arena fd - the S4.3.5 path.  There is
/// no intermediate buffer: the codec's chunk goes out as one `pwrite`.  `fd < 0` selects the mapped-base
/// form (`base` is then a pointer into the mapping), which S4.3.5 may prefer if it maps the slot.
class SlotWireSink final : public WireSink {
public:
    /// `fd` form: `offset` is the byte offset INSIDE the file where the payload starts
    /// (`HandoffClaim::payload_off`), and `capacity` is the slot's payload capacity.
    SlotWireSink(int fd, uint64_t offset, uint64_t capacity);
    /// mapping form: `mapping` is the payload's first byte.
    SlotWireSink(uint8_t* mapping, uint64_t capacity);
    bool write(const uint8_t* data, size_t bytes) override;
    std::string refusal() const override { return error_; }
    uint64_t written() const { return pos_; }
    /// The named refusal when a payload outgrew its slot.
    const std::string& error() const { return error_; }

private:
    int fd_ = -1;
    uint64_t at_ = 0;
    uint8_t* map_ = nullptr;
    uint64_t pos_ = 0, capacity_ = 0;
    std::string error_;
};

// ================================ THE SOURCE (decode input) ================================
/// Random access by payload offset.  `span()` lets a mapped slot hand the codec its own pages with no copy;
/// a source that cannot return a contiguous run returns null and the codec stages through its own buffer.
class WireSource {
public:
    virtual ~WireSource() = default;
    virtual uint64_t size() const = 0;
    /// Fill [offset, offset+bytes).  Must fail rather than short-read.
    virtual bool read(uint8_t* dst, uint64_t offset, size_t bytes) const = 0;
    virtual const uint8_t* span(uint64_t /*offset*/, size_t /*bytes*/) const { return nullptr; }
};

/// A payload already in memory - what a mapped hand-off slot is (S4.3.6), and what the test uses.
class SpanWireSource final : public WireSource {
public:
    SpanWireSource(const uint8_t* data, uint64_t bytes) : p_(data), n_(bytes) {}
    uint64_t size() const override { return n_; }
    bool read(uint8_t* dst, uint64_t offset, size_t bytes) const override;
    const uint8_t* span(uint64_t offset, size_t bytes) const override;

private:
    const uint8_t* p_ = nullptr;
    uint64_t n_ = 0;
};

/// A payload in a file (an arena slot read through the fd instead of a mapping).
class FileWireSource final : public WireSource {
public:
    FileWireSource(int fd, uint64_t offset, uint64_t bytes) : fd_(fd), at_(offset), n_(bytes) {}
    uint64_t size() const override { return n_; }
    bool read(uint8_t* dst, uint64_t offset, size_t bytes) const override;

private:
    int fd_ = -1;
    uint64_t at_ = 0;
    uint64_t n_ = 0;
};

// ================================ ENCODE ================================
/// The two identity values the payload carries (deviation 3).  Both must be the values the caller put in
/// the arena slot header, or the two checks stop agreeing.
struct WireIdentity {
    uint64_t geom_hash = 0;
    uint64_t pack_hash = 0;
};

/// Everything about a payload that is knowable without writing it.  S4.3.5 uses `total_bytes` to pick the
/// tier BEFORE claiming a slot, so a job that cannot fit is refused with a number instead of a partial
/// write.
struct WirePlan {
    uint64_t body_bytes = 0;        ///< the record stream, header excluded
    uint64_t blob_pool_off = 0;     ///< from the payload start
    uint64_t blob_pool_bytes = 0;   ///< including the alignment padding after the last blob
    uint64_t total_bytes = 0;       ///< the whole payload: header + records + pad + pool
    int64_t n_kv = 0, n_checkpoints = 0, n_stage_parts = 0, live_tokens = 0;
    uint16_t flags = 0;
    size_t blobs = 0;               ///< pool entries (zero-length buffers are not entries)
};

/// What an encode produced.
struct WireEncoded {
    uint64_t bytes = 0;             ///< == plan.total_bytes, the bytes actually written
    uint64_t payload_hash = 0;      ///< deviation 4: put this in the slot header and the DONE line
    uint64_t blob_pool_bytes = 0;
    size_t blobs = 0;
};

/// The size of the payload for `image`, with no writes and no allocation.  Refuses an image this codec
/// cannot carry (deviation 7, or an arithmetic overflow) with a named reason.
bool conversation_wire_plan(const SavedConversation& image, WirePlan& plan, std::string& error);

/// The same with the depth bound raised.  Production calls `conversation_wire_plan`; this exists so the
/// test can BUILD an over-deep payload and prove the receiver refuses it (deviation 8).
bool conversation_wire_plan(const SavedConversation& image, int max_checkpoint_depth, WirePlan& plan,
                            std::string& error);

/// Encode `image` into `sink`.  The sink receives exactly `plan.total_bytes` bytes in payload order.
/// `image.owner` is never read (deviation 1).  A sink that runs out of room fails with a reason naming
/// both numbers.
bool conversation_wire_encode(const SavedConversation& image, WireSink& sink, const WireIdentity& identity,
                              WireEncoded& out, std::string& error);
bool conversation_wire_encode(const SavedConversation& image, WireSink& sink, const WireIdentity& identity,
                              int max_checkpoint_depth, WirePlan& plan, WireEncoded& out, std::string& error);

// ================================ DECODE ================================
/// What the receiver insists the payload must look like BEFORE it writes anything.  Every check that fails
/// is a named refusal, and `image` is left untouched.
struct WireExpect {
    /// §2.5 step 2: the payload's geometry must equal this process's `geometry_key(g)`.
    std::array<int64_t, 18> geometry{};
    bool check_geometry = false;
    /// §3.10: both hashes, the same values the slot header carries (deviation 3).
    uint64_t geom_hash = 0, pack_hash = 0;
    bool check_hashes = false;
    /// deviation 4: the slot header's `payload_hash`, checked over the payload bytes.
    uint64_t payload_hash = 0;
    bool check_payload_hash = false;
    /// §2.5 step 2, BOTH directions: a split image without a stage set, or a single-GPU image with one, is
    /// refused.  -1 = the caller does not know its stage count and asks for no check.
    int64_t stage_parts = -1;
    /// The slot's payload capacity.  A declared `payload_bytes` above it is refused before anything is
    /// read.  0 = no bound (the caller already bounded the source).
    uint64_t max_payload_bytes = 0;
    /// Deviation 8.  Raise it only in a test.
    int max_checkpoint_depth = kWireMaxCheckpointDepth;
};

/// What a decoded payload says about itself, for the caller's log and for a second opinion.
struct WireDecoded {
    uint64_t bytes = 0;
    uint64_t payload_hash = 0;      ///< recomputed over the bytes that were read
    uint64_t geom_hash = 0, pack_hash = 0;
    std::array<int64_t, 18> geometry{};
    int64_t layer_lo = 0, layer_hi = 0;
    int64_t n_kv = 0, n_checkpoints = 0, n_stage_parts = 0, live_tokens = 0;
    uint16_t flags = 0;
    bool cvec = true;
    uint64_t blob_pool_off = 0, blob_pool_bytes = 0;
    size_t blobs = 0;
};

/// Parse a payload into a real `SavedConversation`.  All-or-nothing: on false, `image` is exactly as it was
/// passed in and `error` names the refusal.  `image.owner` comes back `kNoOwner` (deviation 1).
bool conversation_wire_decode(const WireSource& source, SavedConversation& image, WireDecoded& info,
                              const WireExpect& expect, std::string& error);

/// The convenience form for a payload already in memory.
bool conversation_wire_decode(const uint8_t* data, uint64_t bytes, SavedConversation& image,
                              WireDecoded& info, const WireExpect& expect, std::string& error);

/// Read just the header record - the "should I copy this slot out at all" check.  Fills `header` and
/// validates magic/version/flags/self-consistency, but NOT the geometry, the hashes or the body.
bool conversation_wire_peek(const uint8_t* data, uint64_t bytes, WireHeaderRecord& header, std::string& error);

/// deviation 4: the content hash of a payload as written.  This is the number that goes in the arena slot
/// header's `payload_hash` (§3.4) and that `HandoffArena::payload_hash_of()` produces, so the slot check
/// and the payload check are the same test.
uint64_t conversation_wire_payload_hash(const uint8_t* data, uint64_t bytes);

/// A one-line human-readable summary of a header record, for the refusal message.
std::string conversation_wire_describe(const WireHeaderRecord& header);

} // namespace strata::core
