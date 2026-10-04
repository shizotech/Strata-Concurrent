// include/strata/core/handoff.hpp - S4.3.2: the tmpfs HANDOFF ARENA.
//
// The Stage-4 split moves a `SavedConversation` (237 MB ... 3.18 GB measured, docs/STAGE4-SPLIT-ROLES.md
// Appendix A.2) out of the prefill instance and into a decode instance.  A socket cannot carry that - a
// pipe would serialise, copy twice and die on a 64 KiB buffer - so the payload goes through a shared
// page-cache-backed file and the socket only names the slot.  This header is that file's protocol.
//
// It is deliberately the SAME discipline stage 1 proved for the expert arena (`src/core/pinned.cu`,
// `include/strata/core/pinned.hpp`): a 4 KiB header whose offsets are an ABI, `magic`/`version`/
// `header_bytes` checked before anything else, a `state`/`owner`/`owner_start`/`nonce` claim, `flock`
// held for a DECISION and never across the write, publish through a release fence plus one ordered
// `pwrite` through the fd rather than a mapping, wait for a live owner / take over a dead one, refuse on
// a pack or geometry mismatch, and a `statvfs` free-space check BEFORE `ftruncate` because a sparse
// `ftruncate` on a too-small tmpfs SIGBUSes mid-write with no message at all.
//
// ================================ THE FILE ================================
//
//   <handoff-dir>/<pack16>-<split-spec>/<instance>/arena
//
//   arena := arena_header(4096)
//            + for each tier i: tier_slot_count[i] * ( slot_header(4096) + tier_slot_bytes[i] )
//
// One arena per prefill instance, always: the slot table has exactly ONE writer, and two prefill
// instances over one arena is the torn-arena bug stage 1 exists to prevent.  The prefill instance
// creates it; a decode instance opens it O_RDONLY and never writes a byte of it.
//
// ================================ THE ARENA HEADER (4096 B at offset 0) ================================
//
//   offset   0  char[16]  magic         "STRATA-HARENA-V1"
//   offset  16  u32       version       = 1
//   offset  20  u32       header_bytes  = 4096
//   offset  24  u32       tier_count    (1..8)
//   offset  28  u32       pad
//   offset  32  u64       slot_total    sum of tier_slot_count[i]
//   offset  40  u64       arena_bytes   the whole file
//   offset  48  u64       pack_hash
//   offset  56  u64       geom_hash
//   offset  64  u64       created_pid
//   offset  72  u64       created_start  /proc start time of created_pid
//   offset  80  u64       tier_max_tokens[8]   0 beyond tier_count
//   offset 144  u64       tier_slot_count[8]
//   offset 208  u64       tier_slot_bytes[8]   payload capacity AFTER a slot header
//   offset 272  u64       tier_slot_off[8]     byte offset of tier i's first slot group
//   offset 336  u64       reserved[8]
//
// ================================ A SLOT HEADER (4096 B at each slot's start) ================================
//
//   offset   0  char[16]  magic         "STRATA-HSLOT-V1"
//   offset  16  u32       version       = 1
//   offset  20  u32       header_bytes  = 4096
//   offset  24  u32       state         0 FREE, 1 CLAIMED, 2 READY
//   offset  28  u32       flags         bit0 cancelled, bit1 abandoned-by-client, bit2 partial-payload
//   offset  32  u32       tier          index into tier_max_tokens
//   offset  36  u32       pad
//   offset  40  u64       owner_pid     the prefill pid that wrote it, 0 = nobody
//   offset  48  u64       owner_start   its /proc start time, 0 = unknown
//   offset  56  u64       nonce         identifies THIS handoff attempt
//   offset  64  u64       client_id     the decode instance allowed to read it, 0 = none
//   offset  72  u64       request_id    the decode instance's request id
//   offset  80  u64       pack_hash
//   offset  88  u64       geom_hash
//   offset  96  u64       tokens        live branch tokens
//   offset 104  u64       payload_bytes bytes written after this header
//   offset 112  u64       payload_hash  FNV-1a over [4096, 4096+payload_bytes) of the slot
//   offset 120  u64       claimed_ms    monotonic ms since boot
//   offset 128  u64       published_ms  0 until published
//   offset 136  u64       seq           segment sequence (v1: always 1)
//   offset 144  u64       reserved[7]
//
// ================================ THE LIFECYCLE ================================
//
//   FREE --(claim under flock)--> CLAIMED --(write body, release fence, pwrite state)--> READY
//        --(ACK)--> FREE
//
// `state == READY` is the publish point and the ONLY state a reader may copy from.  A CLAIMED slot is
// "not populated": a reader that stumbles in mid-write sees that and never a half-written body.
//
// ================================ DEVIATIONS FROM docs/STAGE4-SPLIT-ROLES.md §3 ================================
//
// S4.3.5 (writer) and S4.3.6 (reader) must see ONE truth, so every place this implementation resolves
// an ambiguity or a mistake in the document is listed here and repeated in the .cpp:
//
// 1. **§3.5 step 3 is not implementable as written.**  It asks for "`payload_bytes`, `payload_hash` and
//    `published_ms` written in the same 32-byte `pwrite` as `state`", but with the §3.4 offsets those
//    fields are at 104/112/128 while `state` is at 24 - they are not contiguous, so no single store can
//    cover them.  Resolution, which keeps the invariant the sentence is really protecting (one ordered
//    store, never a page write another process can see half-done):
//      a. one `pwrite` of the contiguous 48-byte block [96,144) - `tokens`, `payload_bytes`,
//         `payload_hash`, `claimed_ms`, `published_ms`, `seq` - while `state` is STILL `CLAIMED`;
//      b. `std::atomic_thread_fence(release)`;
//      c. one `pwrite` of the 8 bytes [24,32) - `state` + `flags`.  **That store is the publish point.**
//    A reader therefore cannot see READY with stale payload metadata, and if the 8-byte store were ever
//    torn the payload hash check (§3.10) catches it before anything is mounted.
// 2. **`client_id` is opaque, so "the READY slot's client pid is dead" is not decidable from the slot
//    header alone.**  §3.6's lease table says a READY slot is reclaimed when "`client_id`'s pid is dead",
//    but §4.2 defines `client_id` as a u64 chosen by the decode instance, not a pid.  Resolution: the
//    writer registers the mapping with `note_client_pid(client_id, pid)` (it learns the pid from the
//    HELLO connection / `SO_PEERCRED`).  With no registration the time bound still reclaims the slot, so
//    the leak is bounded either way - the registration only makes it immediate.
// 3. **`geom_hash` is computed by the caller.**  §3.10 wants it over `geometry_key(g)` plus the canonical
//    split spec, but `geometry_key` is file-local to `src/core/conversation_state.cpp` and S4.3.2 may not
//    edit that file.  `handoff_geom_hash()` therefore takes the 18-int key and the split-spec string;
//    `handoff_split_spec()` gives the canonical spelling so both sides compute the same bytes.
// 4. **A second `open_writer()` on an arena a LIVE other process created is refused**, not adopted: one
//    arena, one writer.  If the recorded creator is dead (or its pid was recycled) the arena is
//    re-initialised from scratch - every slot header reset to FREE - because a restarted prefill instance
//    must not inherit slots whose payloads belong to a process that no longer exists.
// 5. `claimed_ms`/`published_ms` are `CLOCK_MONOTONIC` milliseconds, which is what "monotonic ms since
//    boot" means and is comparable across processes on one boot.
// 6. **`payload_hash` is over the slot's payload region only** - `[slot_base+4096,
//    slot_base+4096+payload_bytes)` - exactly as §3.4 says.  It is NOT over the slot header.
// 7. **`open_writer(..., bool allow_existing_writer = false)`** - the default is what production uses and
//    what §3.2 requires.  The flag skips ONLY the deviation-4 one-writer refusal, and exists so the CPU
//    test can put two PROCESSES in contention for one slot and prove the claim itself is exclusive across
//    address spaces (the mechanism the single-writer rule relies on).  The size, magic, version and both
//    hash checks all still run with it set.  S4.3.5 must never pass true.
//
// ================================ WHAT S4.3.5 / S4.3.6 CALL ================================
//
//   prefill instance (writer):
//     handoff_size_arena(tiers, pricer, sizing, err)   // pricer = conversation_snapshot_bytes wrapper
//     arena.open_writer(path, sizing, pack_hash, geom_hash, err)   // refuses with BOTH numbers if short
//     arena.note_client_pid(client_id, pid)
//     arena.claim(tier, client_id, request_id, claim, err)
//     arena.write_payload(claim, off, buf, n, err)      // no lock held
//     arena.publish(claim, tokens, bytes, hash, err)    // the release fence + the ordered store
//     arena.release(index, nonce, err)                  // on ACK
//     arena.reclaim(&err)                               // on the idle tick; claim() does it too
//
//   decode instance (reader):
//     arena.open_reader(path, pack_hash, geom_hash, err)
//     arena.read_slot(index, nonce, client_id, payload, err)   // maps PROT_READ, checks READY+owner+hash
//     (HandoffPayload unmaps itself; the reader never writes the arena)
//
// CPU-only by construction: no CUDA call, no model, no session.  `fnv1a64` is the engine's existing one
// (`include/strata/core/pinned.hpp`), reused rather than duplicated.
#pragma once

#include "strata/core/pinned.hpp"

#include <array>
#include <cstdint>
#include <cstddef>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace strata::core {

// ================================ THE CONSTANTS (an ABI - never move a value) ================================
inline constexpr uint64_t kHandoffHeaderBytes = 4096;       ///< arena header AND slot header, both 4 KiB
inline constexpr uint32_t kHandoffVersion = 1;
inline constexpr int kHandoffMaxTiers = 8;
/// §3.8: the per-slot slack over the priced payload, covering record headers, blob-pool alignment and the
/// segment-directory overhead `ConversationBuffer::bytes()` counts.
inline constexpr uint64_t kHandoffSlackBytes = 64ull << 20;
/// §3.6: the lease bound for BOTH states.  `STRATA_HANDOFF_LEASE_MS` overrides it (0 = lease immediately).
inline constexpr uint64_t kHandoffDefaultLeaseMs = 300000ull;

enum HandoffSlotState : uint32_t {
    kHandoffFree = 0,
    kHandoffClaimed = 1,
    kHandoffReady = 2,
};
enum HandoffSlotFlags : uint32_t {
    kHandoffFlagCancelled = 1u,
    kHandoffFlagAbandoned = 2u,
    kHandoffFlagPartial = 4u,
};

// ---- arena header offsets ----
inline constexpr uint64_t kArenaMagicOffset = 0;
inline constexpr uint64_t kArenaVersionOffset = 16;
inline constexpr uint64_t kArenaHeaderBytesOffset = 20;
inline constexpr uint64_t kArenaTierCountOffset = 24;
inline constexpr uint64_t kArenaSlotTotalOffset = 32;
inline constexpr uint64_t kArenaBytesOffset = 40;
inline constexpr uint64_t kArenaPackHashOffset = 48;
inline constexpr uint64_t kArenaGeomHashOffset = 56;
inline constexpr uint64_t kArenaCreatedPidOffset = 64;
inline constexpr uint64_t kArenaCreatedStartOffset = 72;
inline constexpr uint64_t kArenaTierMaxTokensOffset = 80;
inline constexpr uint64_t kArenaTierSlotCountOffset = 144;
inline constexpr uint64_t kArenaTierSlotBytesOffset = 208;
inline constexpr uint64_t kArenaTierSlotOffOffset = 272;
inline constexpr uint64_t kArenaReservedOffset = 336;

// ---- slot header offsets ----
inline constexpr uint64_t kSlotMagicOffset = 0;
inline constexpr uint64_t kSlotVersionOffset = 16;
inline constexpr uint64_t kSlotHeaderBytesOffset = 20;
inline constexpr uint64_t kSlotStateOffset = 24;
inline constexpr uint64_t kSlotFlagsOffset = 28;
inline constexpr uint64_t kSlotTierOffset = 32;
inline constexpr uint64_t kSlotOwnerOffset = 40;
inline constexpr uint64_t kSlotOwnerStartOffset = 48;
inline constexpr uint64_t kSlotNonceOffset = 56;
inline constexpr uint64_t kSlotClientOffset = 64;
inline constexpr uint64_t kSlotRequestOffset = 72;
inline constexpr uint64_t kSlotPackHashOffset = 80;
inline constexpr uint64_t kSlotGeomHashOffset = 88;
inline constexpr uint64_t kSlotTokensOffset = 96;
inline constexpr uint64_t kSlotPayloadBytesOffset = 104;
inline constexpr uint64_t kSlotPayloadHashOffset = 112;
inline constexpr uint64_t kSlotClaimedMsOffset = 120;
inline constexpr uint64_t kSlotPublishedMsOffset = 128;
inline constexpr uint64_t kSlotSeqOffset = 136;
inline constexpr uint64_t kSlotReservedOffset = 144;
/// The contiguous block the publish writes first (see deviation 1 above).
inline constexpr uint64_t kSlotPublishBlockOffset = 96;
inline constexpr uint64_t kSlotPublishBlockBytes = 48;
/// The publish point: one ordered 8-byte store of `state` + `flags`.
inline constexpr uint64_t kSlotPublishStateBytes = 8;

/// The on-disk arena header.  POD, and every offset above is `static_assert`ed against it (here AND in
/// the test, which restates the layout independently so a drift in one of the two stops the build).
struct HandoffArenaHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t tier_count;
    uint32_t pad;
    uint64_t slot_total;
    uint64_t arena_bytes;
    uint64_t pack_hash;
    uint64_t geom_hash;
    uint64_t created_pid;
    uint64_t created_start;
    uint64_t tier_max_tokens[kHandoffMaxTiers];
    uint64_t tier_slot_count[kHandoffMaxTiers];
    uint64_t tier_slot_bytes[kHandoffMaxTiers];
    uint64_t tier_slot_off[kHandoffMaxTiers];
    uint64_t reserved[kHandoffMaxTiers];
};
static_assert(kHandoffHeaderBytes == 4096);
static_assert(sizeof(HandoffArenaHeader) <= kHandoffHeaderBytes);
static_assert(offsetof(HandoffArenaHeader, magic) == kArenaMagicOffset);
static_assert(offsetof(HandoffArenaHeader, version) == kArenaVersionOffset);
static_assert(offsetof(HandoffArenaHeader, header_bytes) == kArenaHeaderBytesOffset);
static_assert(offsetof(HandoffArenaHeader, tier_count) == kArenaTierCountOffset);
static_assert(offsetof(HandoffArenaHeader, slot_total) == kArenaSlotTotalOffset);
static_assert(offsetof(HandoffArenaHeader, arena_bytes) == kArenaBytesOffset);
static_assert(offsetof(HandoffArenaHeader, pack_hash) == kArenaPackHashOffset);
static_assert(offsetof(HandoffArenaHeader, geom_hash) == kArenaGeomHashOffset);
static_assert(offsetof(HandoffArenaHeader, created_pid) == kArenaCreatedPidOffset);
static_assert(offsetof(HandoffArenaHeader, created_start) == kArenaCreatedStartOffset);
static_assert(offsetof(HandoffArenaHeader, tier_max_tokens) == kArenaTierMaxTokensOffset);
static_assert(offsetof(HandoffArenaHeader, tier_slot_count) == kArenaTierSlotCountOffset);
static_assert(offsetof(HandoffArenaHeader, tier_slot_bytes) == kArenaTierSlotBytesOffset);
static_assert(offsetof(HandoffArenaHeader, tier_slot_off) == kArenaTierSlotOffOffset);
static_assert(offsetof(HandoffArenaHeader, reserved) == kArenaReservedOffset);

/// The on-disk slot header.
struct HandoffSlotHeader {
    char magic[16];
    uint32_t version;
    uint32_t header_bytes;
    uint32_t state;
    uint32_t flags;
    uint32_t tier;
    uint32_t pad;
    uint64_t owner_pid;
    uint64_t owner_start;
    uint64_t nonce;
    uint64_t client_id;
    uint64_t request_id;
    uint64_t pack_hash;
    uint64_t geom_hash;
    uint64_t tokens;
    uint64_t payload_bytes;
    uint64_t payload_hash;
    uint64_t claimed_ms;
    uint64_t published_ms;
    uint64_t seq;
    uint64_t reserved[7];
};
static_assert(sizeof(HandoffSlotHeader) <= kHandoffHeaderBytes);
static_assert(offsetof(HandoffSlotHeader, magic) == kSlotMagicOffset);
static_assert(offsetof(HandoffSlotHeader, version) == kSlotVersionOffset);
static_assert(offsetof(HandoffSlotHeader, header_bytes) == kSlotHeaderBytesOffset);
static_assert(offsetof(HandoffSlotHeader, state) == kSlotStateOffset);
static_assert(offsetof(HandoffSlotHeader, flags) == kSlotFlagsOffset);
static_assert(offsetof(HandoffSlotHeader, tier) == kSlotTierOffset);
static_assert(offsetof(HandoffSlotHeader, owner_pid) == kSlotOwnerOffset);
static_assert(offsetof(HandoffSlotHeader, owner_start) == kSlotOwnerStartOffset);
static_assert(offsetof(HandoffSlotHeader, nonce) == kSlotNonceOffset);
static_assert(offsetof(HandoffSlotHeader, client_id) == kSlotClientOffset);
static_assert(offsetof(HandoffSlotHeader, request_id) == kSlotRequestOffset);
static_assert(offsetof(HandoffSlotHeader, pack_hash) == kSlotPackHashOffset);
static_assert(offsetof(HandoffSlotHeader, geom_hash) == kSlotGeomHashOffset);
static_assert(offsetof(HandoffSlotHeader, tokens) == kSlotTokensOffset);
static_assert(offsetof(HandoffSlotHeader, payload_bytes) == kSlotPayloadBytesOffset);
static_assert(offsetof(HandoffSlotHeader, payload_hash) == kSlotPayloadHashOffset);
static_assert(offsetof(HandoffSlotHeader, claimed_ms) == kSlotClaimedMsOffset);
static_assert(offsetof(HandoffSlotHeader, published_ms) == kSlotPublishedMsOffset);
static_assert(offsetof(HandoffSlotHeader, seq) == kSlotSeqOffset);
static_assert(offsetof(HandoffSlotHeader, reserved) == kSlotReservedOffset);
// The publish block must really be the contiguous run deviation 1 claims it is.
static_assert(kSlotPublishBlockOffset == kSlotTokensOffset);
static_assert(kSlotPublishBlockOffset + kSlotPublishBlockBytes == kSlotReservedOffset);

// ================================ THE IDENTITY (§3.2, §3.10) ================================

/// One layer carve of the engine's `--layer-split`, in engine order.
struct HandoffSplitStage {
    int64_t layer_lo = 0;
    int64_t layer_hi = 0;
};
/// The canonical split description.  Two instances with different splits produce different payloads for
/// the same conversation and must never mount each other's, so this string - and therefore `geom_hash` -
/// covers the main carve, every later carve in engine order, `n_stages`, and where the draft lives.
struct HandoffSplit {
    int64_t main_lo = 0;
    int64_t main_hi = 0;
    std::vector<HandoffSplitStage> stages;   ///< LATER stages only, engine order; empty = no split
    bool draft_on_last = true;               ///< the tiling rule: the drafter binds to the last stage
};
/// The canonical spelling.  Filename-safe (no space, no slash) so §3.2 can use it as a path component.
std::string handoff_split_spec(const HandoffSplit& split);

/// §3.10 `geom_hash`: FNV-1a over the 18-entry `geometry_key` and the canonical split spec.
uint64_t handoff_geom_hash(const std::array<int64_t, 18>& geometry, const std::string& split_spec);

/// §3.2 the arena path for one (pack, split, instance) triple.  `<pack16>` is 16 lowercase hex digits.
std::string handoff_arena_path(const std::string& handoff_dir, uint64_t pack_hash,
                               const std::string& split_spec, const std::string& instance);

inline uint64_t handoff_round_up_64(uint64_t n) { return (n + 63ull) & ~63ull; }

// ================================ SIZING (§3.7, §3.8) ================================

struct HandoffTierSpec {
    uint64_t max_tokens = 0;    ///< the tier's context length
    uint64_t slot_count = 0;    ///< how many slots in this tier's group
};

/// The computed layout.  `slot_off[i]` is written into the arena header and is the ONLY way a reader
/// finds a group: agreeing about arithmetic is how protocol bugs are born.
struct HandoffSizing {
    std::vector<HandoffTierSpec> tiers;
    std::vector<uint64_t> payload_bytes;   ///< as priced, before slack
    std::vector<uint64_t> slot_bytes;      ///< round_up_64(payload) + kHandoffSlackBytes
    std::vector<uint64_t> slot_off;        ///< file offset of tier i's first slot group
    uint64_t slot_total = 0;
    uint64_t arena_bytes = 0;
    uint64_t max_tokens = 0;               ///< the largest tier
    /// The start-up figure, spelled the way the engine spells its other numbers:
    /// `slots 2,2,2 (tiers 1024,16384,131072), max 131072 tokens, arena 8.20 GiB`.
    std::string describe;
};

/// `pricer(tier_max_tokens, payload_bytes, err)` is the engine's own arithmetic - in the prefill instance
/// it wraps `conversation_snapshot_bytes(view, session, stages, g, mtp.kv_state(), ...)` with a synthetic
/// view of exactly `tier_max_tokens` ids.  It is host-side only, so this whole path runs at start-up and
/// in a CPU test with no CUDA and no model.
using HandoffPricer = std::function<bool(uint64_t tier_tokens, uint64_t& payload_bytes, std::string& err)>;

/// The §3.8 rule: sized from the model's geometry and the tier token counts, NEVER from this machine's
/// RAM.  RAM only decides whether the arena FITS, and that is the `statvfs` check in `open_writer`.
bool handoff_size_arena(const std::vector<HandoffTierSpec>& tiers, const HandoffPricer& pricer,
                        HandoffSizing& out, std::string& err);
/// The same with the payload sizes already priced (what S4.3.5 uses after it called the engine itself).
bool handoff_size_arena(const std::vector<HandoffTierSpec>& tiers,
                        const std::vector<uint64_t>& payload_bytes, HandoffSizing& out, std::string& err);

// ================================ THE SLOT, AS READ ================================

struct HandoffSlotInfo {
    int index = -1;
    uint64_t base = 0;              ///< file offset of the slot header
    uint64_t capacity = 0;          ///< tier_slot_bytes[tier]
    uint32_t state = kHandoffFree;
    uint32_t flags = 0;
    uint32_t tier = 0;
    uint64_t owner_pid = 0, owner_start = 0, nonce = 0;
    uint64_t client_id = 0, request_id = 0;
    uint64_t pack_hash = 0, geom_hash = 0;
    uint64_t tokens = 0, payload_bytes = 0, payload_hash = 0;
    uint64_t claimed_ms = 0, published_ms = 0, seq = 0;
};

/// A claimed slot: everything the writer needs to `pwrite` into it.
struct HandoffClaim {
    int index = -1;
    uint64_t base = 0;              ///< file offset of the slot header
    uint64_t payload_off = 0;       ///< base + 4096
    uint64_t capacity = 0;          ///< the tier group's payload capacity
    uint32_t tier = 0;              ///< the group ACTUALLY used (may be larger than requested, §3.8)
    uint64_t nonce = 0;
};

/// What `reclaim()` took back, for the log line (§3.6: "the log can say what it reclaimed and why").
struct HandoffReclaim {
    int index = -1;
    uint32_t was_state = 0;
    uint32_t flags = 0;
    uint64_t owner_pid = 0, client_id = 0, nonce = 0, payload_bytes = 0;
    uint64_t age_ms = 0;
    std::string why;
};

/// A published payload mapped PROT_READ.  RAII: it unmaps itself.  The reader never holds the arena lock
/// across the copy, and never writes the arena at all.
struct HandoffPayload {
    const uint8_t* data = nullptr;
    uint64_t bytes = 0;                 ///< the published payload length (== payload_bytes)
    int index = -1;
    uint32_t tier = 0;
    uint64_t nonce = 0, client_id = 0, request_id = 0;
    uint64_t tokens = 0, payload_bytes = 0, payload_hash = 0, seq = 0;

    HandoffPayload() = default;
    ~HandoffPayload();
    HandoffPayload(const HandoffPayload&) = delete;
    HandoffPayload& operator=(const HandoffPayload&) = delete;
    HandoffPayload(HandoffPayload&& o) noexcept;
    HandoffPayload& operator=(HandoffPayload&& o) noexcept;
    bool valid() const { return data != nullptr; }
    /// Take ownership of a PROT_READ mapping of the slot's payload region.  `map`/`map_bytes` are what
    /// the destructor unmaps; `data` is the byte the reader starts copying at.
    void bind_mapping(void* map, uint64_t map_bytes, const uint8_t* data, uint64_t payload_bytes);
    void unbind();

  private:
    void* map_ = nullptr;
    uint64_t map_bytes_ = 0;
};

// ================================ THE ARENA ================================

class HandoffArena {
  public:
    /// The lease bound starts at `STRATA_HANDOFF_LEASE_MS` if set, else kHandoffDefaultLeaseMs.
    HandoffArena();
    ~HandoffArena();
    HandoffArena(const HandoffArena&) = delete;
    HandoffArena& operator=(const HandoffArena&) = delete;
    HandoffArena(HandoffArena&& o) noexcept;
    HandoffArena& operator=(HandoffArena&& o) noexcept;

    /// ---- the writer (the prefill instance) ----
    /// Creates the directory and the file, sizes it, and writes the header.  Refuses - with BOTH numbers
    /// and a named reason - when the filesystem cannot hold it, BEFORE any `ftruncate` (§3.7).  There is
    /// no fallback: unlike the expert arena, a handoff that does not fit is a start-up failure.
    ///
    /// `allow_existing_writer` (default false, and production must leave it false) skips only the
    /// "one arena, one writer" refusal of deviation 4.  It exists for the CPU test, which needs two
    /// PROCESSES to contend for one slot to prove the claim itself is exclusive across address spaces -
    /// the mechanism the single-writer rule relies on.  The size, magic, version and both-hash checks all
    /// still run, and the slot claim is still serialised by the arena lock.
    bool open_writer(const std::string& path, const HandoffSizing& sizing, uint64_t pack_hash,
                     uint64_t geom_hash, std::string& err, bool allow_existing_writer = false);
    /// The lease bound for this arena.  Default: `STRATA_HANDOFF_LEASE_MS`, else kHandoffDefaultLeaseMs.
    void set_lease_ms(uint64_t ms) { lease_ms_ = ms; }
    uint64_t lease_ms() const { return lease_ms_; }
    /// Deviation 2: the pid the writer saw for a `client_id`, so a READY slot whose reader died is
    /// reclaimed at once instead of at the lease bound.
    void note_client_pid(uint64_t client_id, uint64_t pid);

    /// §3.6: reclaim expired leases and dead/abandoned claims.  A DECISION, so it runs under the arena
    /// lock.  `claim()` calls it first; the prefill instance also calls it on its idle tick.
    std::vector<HandoffReclaim> reclaim(std::string* err = nullptr);

    /// §3.5 step 1 + §3.8's tier rule: a job of tier `i` takes the SMALLEST group `j >= i` that has a
    /// free (or reclaimable) slot, so a 1 024-token job never sits in a 3 GB slot while a 131 072-token
    /// job waits.  Exclusive: exactly one caller wins a slot.  `err` says "no free slot" - which is
    /// TEMPORARY (§3.8: a full arena is `QUEUED`, never an `ERR` on the client's wire).
    bool claim(uint32_t tier, uint64_t client_id, uint64_t request_id, HandoffClaim& out, std::string& err);

    /// §3.5 step 2.  No lock held.  Refuses anything past the tier's capacity with both numbers.
    bool write_payload(const HandoffClaim& c, uint64_t offset, const void* data, uint64_t bytes,
                       std::string& err);
    /// §3.5 step 3 + deviation 1.  Refuses when this process no longer owns the claim (state moved, nonce
    /// moved, or the client cancelled) - the `publish_shared_load()` rule, restated for a slot.
    bool publish(const HandoffClaim& c, uint64_t tokens, uint64_t payload_bytes, uint64_t payload_hash,
                 std::string& err);
    /// §3.5 step 5, the ACK path.  Does NOT zero the body: zeroing 3 GB would make the free path slower
    /// than the write path, and the next writer overwrites it.
    bool release(int index, uint64_t nonce, std::string& err);
    /// Give up a claim without publishing (a failed read, a cancelled job).  Sets the partial flag.
    bool abort_claim(int index, uint64_t nonce, std::string& err);
    /// Ask a running job to stop (§4.6).  The writer notices at `publish()`, which then refuses.
    bool cancel(int index, uint64_t nonce, std::string& err);

    /// ---- the reader (a decode instance) ----
    /// Opens O_RDONLY|O_CLOEXEC.  Never creates, never `ftruncate`s.  Checks magic/version/header_bytes
    /// first, then the size, then BOTH hashes - a mismatch is refused before any byte is copied, naming
    /// both values (§3.10).
    bool open_reader(const std::string& path, uint64_t pack_hash, uint64_t geom_hash, std::string& err);
    /// §3.5 step 4: shared flock for the header read only, then map.  Refuses unless
    /// `state == READY && client_id == me && nonce == the DONE line's nonce`, and unless the slot's own
    /// pack/geometry match the arena's.
    bool read_slot(int index, uint64_t expect_nonce, uint64_t client_id, HandoffPayload& out,
                   std::string& err);
    /// FNV-1a over the mapped payload, using the engine's `fnv1a64`.
    static uint64_t payload_hash_of(const HandoffPayload& p);

    /// ---- either side ----
    bool valid() const { return fd_ >= 0; }
    bool is_writer() const { return writer_; }
    const std::string& path() const { return path_; }
    const HandoffArenaHeader& header() const { return hdr_; }
    uint64_t arena_bytes() const { return hdr_.arena_bytes; }
    uint64_t slot_total() const { return hdr_.slot_total; }
    uint32_t tier_count() const { return hdr_.tier_count; }
    uint64_t tier_max_tokens(uint32_t tier) const;
    uint64_t tier_slot_bytes(uint32_t tier) const;
    uint64_t tier_slot_count(uint32_t tier) const;
    /// §3.3: the byte offset of tier i's first slot group, as the creator wrote it.  A reader uses THIS
    /// rather than recomputing the layout.
    uint64_t tier_slot_off(uint32_t tier) const;
    /// The largest tier, i.e. `--prefill-max-tokens`'s ceiling for this arena (§4.2's `max_tokens`).
    /// Tiers are stored ASCENDING, so this is the LAST one - not index 0.
    uint64_t max_tokens() const {
        return hdr_.tier_count == 0 ? 0 : hdr_.tier_max_tokens[hdr_.tier_count - 1];
    }
    /// The start-up line's figure: `slots 2,2,2 (tiers 1024,16384,131072), max 131072 tokens,
    /// arena 8.20 GiB, pack hash <x64>, geometry hash <x64>`.
    std::string describe() const;
    /// Read one slot header under a SHARED flock.  For diagnostics, the idle tick and the tests.
    bool slot_info(int index, HandoffSlotInfo& out) const;
    /// `free/total`, for the `INFO` and activity lines (§4.3).
    std::string free_slots(std::string* err = nullptr) const;

  private:
    void close_();
    bool read_header_(std::string& err);
    bool init_slots_(std::string& err);
    bool slot_base_(int index, uint64_t& base, uint32_t& tier, uint64_t& capacity) const;
    bool read_slot_header_(int index, HandoffSlotHeader& h, uint64_t& base, uint64_t& capacity) const;
    bool write_slot_fields_(uint64_t base, uint64_t off, const void* p, uint64_t n) const;
    bool lock_(bool exclusive, std::string& err) const;
    void unlock_() const;
    bool lease_expired_(uint64_t then_ms) const;
    bool client_dead_(uint64_t client_id) const;
    /// These assume `mu_` is ALREADY held.  `claim()` reclaims inside its own locked pass, and calling the
    /// public `reclaim()` from there would deadlock on a non-recursive mutex.
    std::vector<HandoffReclaim> reclaim_locked_(std::string* err);
    std::string free_slots_locked_() const;

    int fd_ = -1;
    bool writer_ = false;
    std::string path_;
    HandoffArenaHeader hdr_{};
    uint64_t lease_ms_ = kHandoffDefaultLeaseMs;
    std::vector<std::pair<uint64_t, uint64_t>> client_pids_;   ///< client_id -> pid (deviation 2)
    // The arena lock is per OPEN FILE DESCRIPTION, so two threads sharing this fd would share it and
    // silently re-enter.  Every decision therefore also takes this process-local mutex.
    mutable std::mutex mu_;
};

/// §3.6 helpers, exposed for the tests and for the prefill instance's own diagnostics.
bool handoff_pid_alive(uint64_t pid);
uint64_t handoff_proc_start_time(uint64_t pid);
bool handoff_claim_alive(uint64_t pid, uint64_t start_time);
/// CLOCK_MONOTONIC milliseconds since boot.
uint64_t handoff_now_ms();

}  // namespace strata::core
