// src/core/saved_conv_wire_test.cpp - S4.3.1: the payload codec's CPU-only test.
//
// NO ENGINE, NO MODEL, NO GPU, NO /dev/shm.  A `SavedConversation` is host storage end to end, so every
// interesting property of the hand-off format - a 20 MiB buffer spanning `ConversationBuffer` segments, a
// three-carve `--layer-split` image, a corrupt payload that would otherwise size a gigabyte allocation - is
// checkable against a synthetic image in well under a second.  That is the point of §7.3 of
// docs/STAGE4-SPLIT-ROLES.md: retire the format's risk before a single GPU cycle is spent.
//
// CUDA is LINKED (the codec sits on `fnv1a64`, which lives in `pinned.cu`) but never initialized: nothing
// here calls a CUDA function and the process never touches a device.  The one real file this test creates is
// a `memfd` - an anonymous in-memory descriptor - to exercise the `pwrite`/`pread` path S4.3.5 uses; it is
// not on any filesystem and it disappears with the process.
#include "strata/core/conversation_wire.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

using namespace strata::core;

namespace {

int checks = 0;
void check(bool ok, const char* label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", label);
        std::exit(1);
    }
}
/// `check(!api(..., err), ("label: " + err).c_str())` is a bug - argument evaluation order is unspecified,
/// so `err` can be read before the call filled it.  Capture first, then assert.
void check_ok(bool ok, const std::string& err, const char* label) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s -> %s\n", label, err.c_str());
        std::exit(1);
    }
}
void check_refusal(const std::string& err, const char* label) {
    ++checks;
    if (err.empty()) {
        std::fprintf(stderr, "FAIL: %s -> the refusal named no reason\n", label);
        std::exit(1);
    }
}
std::string hex(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return std::string(b);
}
uint64_t round_up_64_test(uint64_t n) { return (n + 63ull) & ~63ull; }

// ---------------------------------------------------------------- synthetic images
//
// The shape `conversation_snapshot_save` produces (src/core/conversation_state.cpp:360-417): the main
// session's K/V first then the draft's; `live.stage_parts` EMPTY with the later stages in
// `SavedConversation::stage_parts`; and each retained checkpoint carrying its own stage parts.

/// The 18 slots in `geometry_key`'s order (conversation_state.cpp:14-19).  Arbitrary values, but all
/// distinct, so swapping two of them is visible to the geometry check.
std::array<int64_t, 18> fake_geometry(int64_t n_layers, int64_t experts) {
    const int64_t v[18] = {1024, n_layers, 4, 64, 16, 32, 4, 512, 128, 8, 2, 128, 16, 128, 1, 1, experts, 2816};
    std::array<int64_t, 18> g{};
    for (int i = 0; i < 18; ++i) g[(size_t) i] = v[i];
    return g;
}

uint8_t pattern(size_t i, uint8_t salt) { return (uint8_t) ((i * 31u + salt * 7u + (i >> 9)) & 0xFFu); }

/// `ConversationBuffer` has no `data()`: `visit()` is the only way to touch its bytes.  Writing through it in
/// 64 KiB pieces is also what proves the buffer is addressable across its segment boundary.
void fill(ConversationBuffer& b, size_t n, uint8_t salt) {
    b.resize(n);
    size_t at = 0;
    while (at < n) {
        const size_t chunk = std::min<size_t>(64u * 1024u, n - at);
        const bool ok = b.visit(at, chunk, [&](uint8_t* p, size_t cnt, size_t abs) {
            for (size_t j = 0; j < cnt; ++j) p[j] = pattern(abs + j, salt);
            return true;
        });
        if (!ok) {
            std::fprintf(stderr, "FAIL: ConversationBuffer::visit refused its own segment\n");
            std::exit(1);
        }
        at += chunk;
    }
}

/// A checkpoint's running state is a plain `std::vector<uint8_t>` (deviation 0), so it fills directly.
void fill(std::vector<uint8_t>& v, size_t n, uint8_t salt) {
    v.resize(n);
    for (size_t i = 0; i < n; ++i) v[i] = pattern(i, salt);
}

/// How many pool entries an image should produce: every NON-zero-length buffer, in the order the record
/// stream emits them (§2.3: a zero-length buffer is `{0,0}` with no pool entry).
size_t expected_blobs(const SavedConversation& s) {
    size_t n = 0;
    auto cp = [&](const ConversationCheckpoint& c, auto&& self) -> void {
        for (size_t sz : {c.gdn.size(), c.ple.size(), c.tails.size(), c.dead.size(), c.block_pos.size()})
            if (sz) ++n;
        for (const auto& p : c.stage_parts) self(p, self);
    };
    auto kv = [&](const ConversationKv& k) {
        for (size_t sz : {k.k.size(), k.v.size(), k.k_scale.size(), k.v_scale.size(), k.pooled.size()})
            if (sz) ++n;
    };
    for (const auto& k : s.kv) kv(k);
    cp(s.live, cp);
    for (const auto& c : s.checkpoints) cp(c, cp);
    for (const auto& part : s.stage_parts) {
        cp(part.state, cp);
        for (const auto& k : part.kv) kv(k);
    }
    return n;
}

ConversationKv make_kv(int64_t cells, int64_t heads, int64_t head_dim, bool scales, bool pooled, uint8_t salt,
                       size_t k_bytes_override = 0) {
    ConversationKv k;
    k.format = scales ? 1 : 0;
    k.cells = cells;
    k.heads = heads;
    k.head_dim = head_dim;
    k.page_size = 4;
    k.pooled_rows = pooled ? 7 : 0;
    k.idx_dim = pooled ? 128 : 0;
    const size_t data = k_bytes_override ? k_bytes_override
                                         : (size_t) cells * (size_t) heads * (size_t) head_dim * 2;
    fill(k.k, data, salt);
    fill(k.v, data, (uint8_t) (salt + 1));
    const size_t sc = scales ? (size_t) cells * (size_t) heads * (size_t) (head_dim / 64) * 2 : 0;
    fill(k.k_scale, sc, (uint8_t) (salt + 2));
    fill(k.v_scale, sc, (uint8_t) (salt + 3));
    fill(k.pooled, pooled ? (size_t) k.pooled_rows * (size_t) k.idx_dim * 4 : 0, (uint8_t) (salt + 4));
    return k;
}

ConversationCheckpoint make_checkpoint(size_t tokens, size_t gdn, size_t ple, size_t tails, size_t dead,
                                       size_t block_pos, uint64_t used, uint8_t salt, size_t n_images = 2) {
    ConversationCheckpoint c;
    c.ids.resize(tokens);
    for (size_t i = 0; i < tokens; ++i) c.ids[i] = (int32_t) (i * 7 + 3);
    for (size_t i = 0; i < n_images && i * 3 + 1 < tokens; ++i)
        c.imgs.push_back(ConversationImageKey{(int64_t) (i * 3 + 1), 0x1000ull + i * 977ull + salt});
    fill(c.gdn, gdn, salt);
    fill(c.ple, ple, (uint8_t) (salt + 5));
    fill(c.tails, tails, (uint8_t) (salt + 6));
    fill(c.dead, dead, (uint8_t) (salt + 7));
    fill(c.block_pos, block_pos, (uint8_t) (salt + 8));
    c.used = used;
    return c;
}

/// `carves` are the LATER stages' layer ranges in engine order: {} is single-GPU, {{4,8}} a two-carve split,
/// {{2,5},{5,8}} a three-carve one.
SavedConversation make_image(int64_t n_layers, const std::vector<std::pair<int64_t, int64_t>>& carves,
                             size_t tokens, uint8_t salt, int64_t owner = kNoOwner) {
    SavedConversation s;
    s.geometry = fake_geometry(n_layers, 256);
    const int64_t main_hi = carves.empty() ? n_layers : carves.front().first;
    s.layer_lo = 0;
    s.layer_hi = main_hi;
    s.cvec = true;
    s.owner = owner;
    const size_t gdn = 4096, ple = 1024, tails = 256, dead = 512, block_pos = 64;
    s.live = make_checkpoint(tokens, gdn, ple, tails, dead, block_pos, 17, salt);
    auto cp = make_checkpoint(tokens / 2 + 1, gdn, ple, tails, dead, block_pos, 41, (uint8_t) (salt + 20));
    for (size_t i = 0; i < carves.size(); ++i) {
        auto part = make_checkpoint(tokens / 2 + 1, gdn / 2, ple, tails, dead, block_pos, 41,
                                    (uint8_t) (salt + 30 + i));
        part.imgs.clear();  // a stage part's imgs stay empty (conversation_cache.hpp:64-66)
        cp.stage_parts.push_back(std::move(part));
    }
    s.checkpoints.push_back(std::move(cp));
    s.checkpoints.push_back(make_checkpoint(tokens / 4 + 1, gdn, ple, tails, dead, block_pos, 42,
                                            (uint8_t) (salt + 21)));
    s.kv.push_back(make_kv(64, 2, 64, true, true, salt));
    if (main_hi > 4) s.kv.push_back(make_kv(64, 2, 64, false, true, (uint8_t) (salt + 10)));
    s.kv.push_back(make_kv(32, 2, 64, false, false, (uint8_t) (salt + 11)));  // the draft layer, last
    for (size_t i = 0; i < carves.size(); ++i) {
        ConversationStageSnapshot part;
        part.layer_lo = carves[i].first;
        part.layer_hi = carves[i].second;
        part.state = make_checkpoint(tokens, gdn / 2, ple, tails, dead, block_pos, 0, (uint8_t) (salt + 50 + i));
        part.state.imgs.clear();  // the ids are what the state is sized by
        part.kv.push_back(make_kv(64, 2, 64, true, true, (uint8_t) (salt + 60 + i)));
        s.stage_parts.push_back(std::move(part));
    }
    return s;
}

// ---------------------------------------------------------------- deep equality

bool equal(const ConversationKv& a, const ConversationKv& b) {
    return a.format == b.format && a.cells == b.cells && a.heads == b.heads && a.head_dim == b.head_dim &&
           a.page_size == b.page_size && a.pooled_rows == b.pooled_rows && a.idx_dim == b.idx_dim && a.k == b.k &&
           a.v == b.v && a.k_scale == b.k_scale && a.v_scale == b.v_scale && a.pooled == b.pooled;
}
bool equal(const ConversationCheckpoint& a, const ConversationCheckpoint& b) {
    if (a.ids != b.ids || a.imgs != b.imgs || a.gdn != b.gdn || a.ple != b.ple || a.tails != b.tails ||
        a.dead != b.dead || a.block_pos != b.block_pos || a.used != b.used ||
        a.stage_parts.size() != b.stage_parts.size())
        return false;
    for (size_t i = 0; i < a.stage_parts.size(); ++i)
        if (!equal(a.stage_parts[i], b.stage_parts[i])) return false;
    return true;
}
bool equal(const ConversationStageSnapshot& a, const ConversationStageSnapshot& b) {
    if (a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || !equal(a.state, b.state) ||
        a.kv.size() != b.kv.size())
        return false;
    for (size_t i = 0; i < a.kv.size(); ++i)
        if (!equal(a.kv[i], b.kv[i])) return false;
    return true;
}
/// Every field of `SavedConversation` except `owner`, which the codec deliberately never carries
/// (deviation 1 in the header).
bool equal_image(const SavedConversation& a, const SavedConversation& b) {
    if (a.geometry != b.geometry || a.layer_lo != b.layer_lo || a.layer_hi != b.layer_hi || a.cvec != b.cvec)
        return false;
    if (!equal(a.live, b.live) || a.checkpoints.size() != b.checkpoints.size() || a.kv.size() != b.kv.size() ||
        a.stage_parts.size() != b.stage_parts.size())
        return false;
    for (size_t i = 0; i < a.checkpoints.size(); ++i)
        if (!equal(a.checkpoints[i], b.checkpoints[i])) return false;
    for (size_t i = 0; i < a.kv.size(); ++i)
        if (!equal(a.kv[i], b.kv[i])) return false;
    for (size_t i = 0; i < a.stage_parts.size(); ++i)
        if (!equal(a.stage_parts[i], b.stage_parts[i])) return false;
    return true;
}

// ---------------------------------------------------------------- the sinks and sources the tests need

/// Records every write, so "a >16 MiB buffer went out as its segments and was never staged whole" is a fact
/// about the bytes that left the codec.
class RecordingSink final : public WireSink {
public:
    bool write(const uint8_t* data, size_t bytes) override {
        if (!bytes) return true;
        ++writes;
        max_write = std::max(max_write, bytes);
        total += bytes;
        bytes_.insert(bytes_.end(), data, data + bytes);
        return true;
    }
    size_t writes = 0, max_write = 0, total = 0;
    std::vector<uint8_t> bytes_;
};

/// Refuses once it is full - the slot-capacity path S4.3.5 hits when it claimed too small a tier.
class BoundedSink final : public WireSink {
public:
    explicit BoundedSink(size_t cap) : cap_(cap) {}
    bool write(const uint8_t* data, size_t bytes) override {
        if (bytes_.size() + bytes > cap_) {
            error = "the sink is full";
            return false;
        }
        bytes_.insert(bytes_.end(), data, data + bytes);
        return true;
    }
    std::string refusal() const override { return error; }
    size_t cap_;
    std::vector<uint8_t> bytes_;
    std::string error;
};

/// Never hands over a span and serves reads in bounded pieces: the mapped-slot case turned off, so the
/// codec's staging path is what runs.
class ChunkySource final : public WireSource {
public:
    ChunkySource(const std::vector<uint8_t>& d, size_t piece) : d_(d), piece_(piece) {}
    uint64_t size() const override { return d_.size(); }
    bool read(uint8_t* dst, uint64_t offset, size_t bytes) const override {
        if (offset + bytes > d_.size()) return false;
        size_t left = bytes;
        while (left) {
            const size_t n = std::min(left, piece_);
            std::memcpy(dst, d_.data() + offset + (bytes - left), n);
            dst += n;
            left -= n;
            ++pieces;  // what the codec actually asked for, in pieces
        }
        ++reads;
        return true;
    }
    const uint8_t* span(uint64_t, size_t) const override { return nullptr; }
    const std::vector<uint8_t>& d_;
    size_t piece_;
    mutable size_t reads = 0, pieces = 0;
};

/// A source whose header-sized region is real but whose tail was never written - the slot that was
/// published with a `payload_bytes` longer than what the writer actually got out.  `size()` is honest, the
/// READS past `real_` fail.
class ShortLyingSource final : public WireSource {
public:
    ShortLyingSource(const std::vector<uint8_t>& d, uint64_t real) : d_(d), real_(real) {}
    uint64_t size() const override { return d_.size(); }
    bool read(uint8_t* dst, uint64_t offset, size_t bytes) const override {
        if (offset + bytes > real_) return false;
        std::memcpy(dst, d_.data() + offset, bytes);
        return true;
    }
    const uint8_t* span(uint64_t offset, size_t bytes) const override {
        if (offset + bytes > real_) return nullptr;
        return d_.data() + offset;
    }
    const std::vector<uint8_t>& d_;
    uint64_t real_;
};

// ---------------------------------------------------------------- the record stream, for surgical corruption

/// Walk the record stream the way the decoder does, to find the offsets a corruption test needs.  If this
/// walker disagrees with the codec, the test fails loudly rather than patching the wrong byte.
struct Walk {
    const std::vector<uint8_t>& p;
    uint64_t pos = kWireHeaderRecordBytes;
    uint64_t u64(uint64_t at) const {
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= (uint64_t) p[(size_t) (at + (uint64_t) i)] << (i * 8);
        return v;
    }
    uint32_t u32(uint64_t at) const {
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= (uint32_t) p[(size_t) (at + (uint64_t) i)] << (i * 8);
        return v;
    }
    bool kv(uint64_t& first_blobref) {
        // kind/flags/len, then format+pad, then 6 i64, then 5 BLOBREFs
        if (first_blobref == (uint64_t) -1)
            first_blobref = pos + kWireRecordHeaderBytes + 4 + 4 + 6 * 8;
        pos += kWireRecordHeaderBytes + 4 + 4 + 6 * 8 + 5 * 16;
        return true;
    }
    bool checkpoint(uint64_t& ids_count_off, uint64_t& n_parts_off) {
        const uint64_t body = u64(pos + 8);
        const uint64_t end = pos + kWireRecordHeaderBytes + body;
        pos += kWireRecordHeaderBytes;
        if (ids_count_off == (uint64_t) -1) ids_count_off = pos;
        const int64_t n_ids = (int64_t) u64(pos);
        pos += 8 + (uint64_t) n_ids * 4 + ((n_ids * 4) % 8 ? 4 : 0);
        const int64_t n_imgs = (int64_t) u64(pos);
        pos += 8 + (uint64_t) n_imgs * 16 + 5 * 16 + 8;
        if (n_parts_off == (uint64_t) -1) n_parts_off = pos;
        pos += 8;
        while (pos < end) {
            if (!checkpoint(ids_count_off, n_parts_off)) return false;
        }
        pos = end;
        return true;
    }
    bool stage() {
        const uint64_t body = u64(pos + 8);
        const uint64_t end = pos + kWireRecordHeaderBytes + body;
        pos += kWireRecordHeaderBytes + 16;
        uint64_t a = (uint64_t) -1, b = (uint64_t) -1;
        if (!checkpoint(a, b)) return false;
        const int64_t n_kv = (int64_t) u64(pos);
        pos += 8;
        for (int64_t i = 0; i < n_kv; ++i) {
            uint64_t fb = (uint64_t) -1;
            if (!kv(fb)) return false;
        }
        pos = end;
        return true;
    }
};

/// Walk the whole body and report where the END record actually sits.  Deriving it as
/// `blob_pool_off - 16` is WRONG whenever the record stream is shorter than the pool boundary by padding,
/// and a corruption test that patches a pad byte passes for the wrong reason.
struct Landmarks { uint64_t end = 0, live_record = 0, first_stage = 0; };
Landmarks walk_landmarks(const std::vector<uint8_t>& p, const WirePlan& plan) {
    Walk w{p};
    uint64_t fb = (uint64_t) -1;
    for (int64_t i = 0; i < plan.n_kv; ++i) check(w.kv(fb), "the walk covered the KV array");
    const uint64_t live_record = w.pos;
    uint64_t a = (uint64_t) -1, b = (uint64_t) -1;
    check(w.checkpoint(a, b), "the walk covered the live checkpoint");
    for (int64_t i = 0; i < plan.n_checkpoints; ++i) check(w.checkpoint(a, b), "the walk covered a checkpoint");
    uint64_t first_stage = 0;
    for (int64_t i = 0; i < plan.n_stage_parts; ++i) {
        if (i == 0) first_stage = w.pos;
        check(w.stage(), "the walk covered a stage record");
    }
    check(w.pos < plan.blob_pool_off, "the walk ended inside the record region");
    return Landmarks{w.pos, live_record, first_stage};
}

// ---------------------------------------------------------------- the round trip

WireExpect expect_for(const SavedConversation& s, uint64_t geom_hash, uint64_t pack_hash,
                      uint64_t max_payload = 0) {
    WireExpect e;
    e.geometry = s.geometry;
    e.check_geometry = true;
    e.geom_hash = geom_hash;
    e.pack_hash = pack_hash;
    e.check_hashes = true;
    e.stage_parts = (int64_t) s.stage_parts.size();
    e.max_payload_bytes = max_payload;
    return e;
}

/// encode -> decode -> re-encode: the bytes must match and every field must survive.
void round_trip(const SavedConversation& in, const char* label) {
    const uint64_t geom = 0x1122334455667788ull, pack = 0xaabbccddeeff0011ull;
    WirePlan plan{};
    std::string err;
    check_ok(conversation_wire_plan(in, plan, err), err, label);
    check(plan.total_bytes >= kWireHeaderRecordBytes, "a plan is at least one header record");

    VectorWireSink sink((size_t) plan.total_bytes);
    WireEncoded enc{};
    check_ok(conversation_wire_encode(in, sink, {geom, pack}, enc, err), err, label);
    check(enc.bytes == plan.total_bytes, "the encoder wrote exactly what the plan predicted");
    check(sink.data().size() == plan.total_bytes, "the sink received the planned byte count");
    check(enc.payload_hash == conversation_wire_payload_hash(sink.data().data(), sink.data().size()),
          "the streaming hash equals an independent pass over the payload");
    check(plan.blob_pool_off % kWireBlobAlign == 0, "the blob pool starts 64-aligned");
    check(plan.blob_pool_off + plan.blob_pool_bytes == plan.total_bytes, "the pool ends the payload");
    check(plan.body_bytes + kWireHeaderRecordBytes <= plan.blob_pool_off, "the records fit before the pool");

    WireHeaderRecord hdr{};
    check_ok(conversation_wire_peek(sink.data().data(), sink.data().size(), hdr, err), err, label);
    check(hdr.magic == kWireMagic && hdr.fmt_version == kWireFormatVersion, "magic and version on the wire");
    check(hdr.n_kv == (int64_t) in.kv.size() && hdr.n_checkpoints == (int64_t) in.checkpoints.size() &&
              hdr.n_stage_parts == (int64_t) in.stage_parts.size() &&
              hdr.live_tokens == (int64_t) in.live.ids.size(),
          "the header's counts describe the image");
    check(hdr.payload_bytes == plan.total_bytes && hdr.payload_hash == 0,
          "the header carries the total and leaves the hash out-of-band");
    check((hdr.flags & kWireFlagCvec) != 0 ? in.cvec : !in.cvec, "the cvec flag matches the image");
    check((hdr.flags & kWireFlagHasStageParts) != 0 ? !in.stage_parts.empty() : in.stage_parts.empty(),
          "the stage-parts flag agrees with the vector");

    SavedConversation out;
    out.owner = 999;  // a decode must never inherit the caller's stale claim
    WireDecoded info{};
    WireExpect expect = expect_for(in, geom, pack, plan.total_bytes);
    expect.payload_hash = enc.payload_hash;
    expect.check_payload_hash = true;
    check_ok(conversation_wire_decode(sink.data().data(), sink.data().size(), out, info, expect, err), err, label);
    check(out.owner == kNoOwner, "a received entry arrives unclaimed (§2.4)");
    check(info.bytes == plan.total_bytes && info.blobs == plan.blobs, "the decode reports the same shape");
    check(info.payload_hash == enc.payload_hash, "the decode recomputed the same content hash");
    check(info.geometry == in.geometry && info.layer_lo == in.layer_lo && info.layer_hi == in.layer_hi,
          "the decoded header mirrors the image's identity");
    check(info.cvec == in.cvec, "the decoded cvec matches");
    check(equal_image(in, out), "every field of the decoded image equals the original");

    VectorWireSink again((size_t) plan.total_bytes);
    WireEncoded enc2{};
    check_ok(conversation_wire_encode(out, again, {geom, pack}, enc2, err), err, label);
    check(again.data() == sink.data(), "encode -> decode -> encode is byte-identical");
    check(enc2.payload_hash == enc.payload_hash, "the re-encode hashes to the same value");
}

// ---------------------------------------------------------------- the format's own invariants

void test_layout() {
    // Restated independently of the header, the way `handoff_arena_test` does it: if the two ever drift, the
    // test stops rather than the payload silently changing shape.
    check(sizeof(WireHeaderRecord) == 256, "the header record is 256 bytes");
    check(kWireHeaderSpecBytes == 232, "the documented field list is 232 bytes");
    check(offsetof(WireHeaderRecord, magic) == 0, "magic at 0");
    check(offsetof(WireHeaderRecord, fmt_version) == 4, "fmt_version at 4");
    check(offsetof(WireHeaderRecord, flags) == 6, "flags at 6");
    check(offsetof(WireHeaderRecord, geometry) == 8, "geometry at 8");
    check(offsetof(WireHeaderRecord, geometry) + 18 * 8 == 152, "geometry[18] ends at 152");
    check(offsetof(WireHeaderRecord, layer_lo) == 152, "layer_lo at 152");
    check(offsetof(WireHeaderRecord, layer_hi) == 160, "layer_hi at 160");
    check(offsetof(WireHeaderRecord, n_kv) == 168, "n_kv at 168");
    check(offsetof(WireHeaderRecord, n_checkpoints) == 176, "n_checkpoints at 176");
    check(offsetof(WireHeaderRecord, n_stage_parts) == 184, "n_stage_parts at 184");
    check(offsetof(WireHeaderRecord, live_tokens) == 192, "live_tokens at 192");
    check(offsetof(WireHeaderRecord, blob_pool_off) == 200, "blob_pool_off at 200");
    check(offsetof(WireHeaderRecord, blob_pool_bytes) == 208, "blob_pool_bytes at 208");
    check(offsetof(WireHeaderRecord, geom_hash) == 216, "geom_hash at 216");
    check(offsetof(WireHeaderRecord, pack_hash) == 224, "pack_hash at 224");
    check(offsetof(WireHeaderRecord, payload_bytes) == 232, "payload_bytes at 232 (deviation 2)");
    check(offsetof(WireHeaderRecord, payload_hash) == 240, "payload_hash at 240 (deviation 2)");
    check(offsetof(WireHeaderRecord, reserved) == 248, "reserved at 248 (deviation 2)");
    check(kWireMagic == 0x48345253u, "the magic constant");
    const uint8_t magic[4] = {0x53, 0x52, 0x34, 0x48};
    uint32_t assembled = 0;
    for (int i = 0; i < 4; ++i) assembled |= (uint32_t) magic[i] << (i * 8);
    check(assembled == kWireMagic, "the magic's four bytes are 'S','R','4','H'");
    check(kWireRecordEnd == 0x21444E45u, "the END kind is 0x21444E45");
    const uint32_t end_kind = kWireRecordEnd;
    const char* as_chars = reinterpret_cast<const char*>(&end_kind);
    check(as_chars[0] == 'E' && as_chars[1] == 'N' && as_chars[2] == 'D' && as_chars[3] == '!',
          "the END kind reads as 'END!' in a hex dump (deviation 5)");
    check(kWireRecordKv == 1 && kWireRecordCheckpoint == 2 && kWireRecordStage == 3, "the record kinds");
    check(kWireBlobAlign == 64 && kWireHeaderRecordBytes == 256 && kWireRecordHeaderBytes == 16,
          "the alignment and header sizes");
    check(kWireMaxCheckpointDepth == 8, "the recursion bound is a named constant");
}

void test_round_trips() {
    round_trip(make_image(8, {}, 65, 13), "single-GPU round trip");
    round_trip(make_image(8, {{4, 8}}, 65, 13), "two-carve layer-split round trip");
    round_trip(make_image(8, {{2, 5}, {5, 8}}, 65, 13), "three-carve layer-split round trip");
    round_trip(make_image(8, {{1, 3}, {3, 6}, {6, 8}}, 33, 7), "four-carve layer-split round trip");
    round_trip(make_image(8, {{4, 8}}, 1, 3), "one-token split round trip");

    // The empty conversation: no ids, no buffers, no K/V, no checkpoints, no stages.
    SavedConversation empty;
    empty.geometry = fake_geometry(8, 256);
    empty.layer_hi = 8;
    empty.cvec = false;
    empty.owner = 12345;
    round_trip(empty, "empty conversation round trip");
    {
        WirePlan p{};
        std::string err;
        check_ok(conversation_wire_plan(empty, p, err), err, "empty plan");
        // one CHECKPOINT record header + (n_ids, n_imgs, 5 BLOBREFs, used, n_stage_parts) + the END record
        check(p.body_bytes == kWireRecordHeaderBytes + (8 + 8 + 5 * 16 + 8 + 8) + kWireRecordHeaderBytes,
              "an empty payload is a live CHECKPOINT record and an END marker");
        check(p.total_bytes == round_up_64_test(kWireHeaderRecordBytes + p.body_bytes),
              "the payload is the header, the records, and the pad up to the pool");
        check(p.blob_pool_bytes == 0 && p.blobs == 0, "an empty payload has no pool");
        check(p.blob_pool_off == p.total_bytes, "an empty pool is a zero-length region at the end");
    }

    // Empty OPTIONAL buffers: no scales, no pooled rows, no PLE history, no images.
    {
        SavedConversation s = make_image(8, {}, 9, 5);
        s.live.ple.clear();
        s.kv[0].k_scale.resize(0);
        s.kv[0].v_scale.resize(0);
        s.kv[0].pooled.resize(0);
        s.kv.back().k_scale.resize(0);
        s.checkpoints[0].ple.clear();
        s.checkpoints[0].imgs.clear();
        s.live.imgs.clear();
        round_trip(s, "empty optional buffers round trip");
        WirePlan p{};
        std::string err;
        check_ok(conversation_wire_plan(s, p, err), err, "empty-optional plan");
        check(p.blobs == expected_blobs(s), "only non-empty buffers get a pool entry");
    }

    // cvec off, and negative scalars: proof the scalars travel bit-for-bit, not as a struct dump.
    {
        SavedConversation s = make_image(8, {}, 9, 6);
        s.cvec = false;
        s.layer_lo = -3;
        s.layer_hi = -1;
        s.geometry[0] = -1;
        s.geometry[17] = std::numeric_limits<int64_t>::min();
        round_trip(s, "cvec off and negative scalars round trip");
    }

    // A >16 MiB buffer, which is what `ConversationBuffer::segment_bytes` actually splits on.
    {
        SavedConversation s;
        s.geometry = fake_geometry(8, 256);
        s.layer_hi = 8;
        s.live = make_checkpoint(4, 16, 0, 0, 0, 0, 3, 9, 0);
        ConversationKv big;
        big.format = 0;
        big.cells = 1 << 20;
        big.heads = 1;
        big.head_dim = 20;
        big.page_size = 4;
        fill(big.k, (16u << 20) + 12345u, 77);  // 16 MiB + 12 KB: two segments
        fill(big.v, 20u << 20, 78);             // 20 MiB: two segments
        check(big.k.size() > ConversationBuffer::segment_bytes, "the fixture really spans segments");
        s.kv.push_back(std::move(big));
        s.kv.push_back(make_kv(8, 1, 64, false, false, 79));
        round_trip(s, "a >16 MiB buffer spanning segments round trip");

        WirePlan p{};
        std::string err;
        check_ok(conversation_wire_plan(s, p, err), err, "big plan");
        RecordingSink rs;
        WireEncoded enc{};
        check_ok(conversation_wire_encode(s, rs, {1, 2}, enc, err), err, "big encode");
        check(rs.total == p.total_bytes, "the recording sink saw the whole payload");
        check(rs.max_write <= ConversationBuffer::segment_bytes,
              "no single write is bigger than one ConversationBuffer segment");
        check(rs.max_write == ConversationBuffer::segment_bytes,
              "a >16 MiB buffer really went out as its own 16 MiB segment");
        check(rs.writes > 8, "the big payload was written in many pieces, not staged whole");

        ChunkySource src(rs.bytes_, 4096);
        SavedConversation out;
        WireDecoded info{};
        check_ok(conversation_wire_decode(src, out, info, expect_for(s, 1, 2), err), err, "chunky-source decode");
        check(equal_image(s, out), "the staged read path reproduces the big buffers byte for byte");
        check(src.pieces > 100, "the chunky source really served the bulk of the payload");
        check(src.reads < src.pieces, "the codec asked for chunked reads, not one giant one");
    }

    // A claimed image arrives unclaimed, whatever the sender's `owner` was (§2.4).
    {
        const SavedConversation s = make_image(8, {{4, 8}}, 9, 31, /*owner=*/4242);
        VectorWireSink sink;
        WireEncoded enc{};
        std::string err;
        check_ok(conversation_wire_encode(s, sink, {5, 6}, enc, err), err, "owner encode");
        SavedConversation out;
        out.owner = 777;
        WireDecoded info{};
        check_ok(conversation_wire_decode(sink.data().data(), sink.data().size(), out, info,
                                          expect_for(s, 5, 6), err), err, "owner decode");
        check(out.owner == kNoOwner, "the wire never carries a claim (deviation 1)");
    }
}

// ---------------------------------------------------------------- refusals

void patch(std::vector<uint8_t>& p, size_t off, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[off + (size_t) i] = (uint8_t) (v >> (i * 8));
}
void patch32(std::vector<uint8_t>& p, size_t off, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[off + (size_t) i] = (uint8_t) (v >> (i * 8));
}
void patch16(std::vector<uint8_t>& p, size_t off, uint16_t v) {
    p[off] = (uint8_t) v;
    p[off + 1] = (uint8_t) (v >> 8);
}
uint32_t plan32(const std::vector<uint8_t>& p, size_t off) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t) p[off + (size_t) i] << (i * 8);
    return v;
}

/// Locate the first KV record's first BLOBREF and the live checkpoint's `n_ids`, by walking the stream.
void find_offsets(const std::vector<uint8_t>& p, uint64_t& first_blobref, uint64_t& live_ids_count) {
    first_blobref = (uint64_t) -1;
    live_ids_count = (uint64_t) -1;
    Walk w{p};
    uint64_t n_kv = 0;
    for (int i = 0; i < 8; ++i) n_kv |= (uint64_t) p[168 + (size_t) i] << (i * 8);
    for (uint64_t i = 0; i < n_kv; ++i) check(w.kv(first_blobref), "the walk found a KV record");
    uint64_t ids = (uint64_t) -1, parts = (uint64_t) -1;
    check(w.checkpoint(ids, parts), "the walk found the live checkpoint");
    live_ids_count = ids;
}

void test_refusals() {
    const SavedConversation s = make_image(8, {{4, 8}}, 65, 13);
    const uint64_t geom = 0x1122334455667788ull, pack = 0xaabbccddeeff0011ull;
    VectorWireSink sink;
    WirePlan plan{};
    WireEncoded enc{};
    std::string err;
    check_ok(conversation_wire_encode(s, sink, {geom, pack}, kWireMaxCheckpointDepth, plan, enc, err), err,
             "refusal fixture");
    const std::vector<uint8_t>& good = sink.data();
    uint64_t first_blobref = 0, live_ids = 0;
    find_offsets(good, first_blobref, live_ids);
    check(first_blobref > kWireHeaderRecordBytes && first_blobref < plan.blob_pool_off,
          "the corruption fixture found a real BLOBREF");
    check(live_ids > first_blobref && live_ids < plan.blob_pool_off, "the fixture found the live id count");

    SavedConversation out;
    WireDecoded info{};

    // 1. truncated at several depths.
    for (size_t cut : {(size_t) 1, (size_t) 64, (size_t) 4096, good.size() - 1}) {
        std::vector<uint8_t> p(good.begin(), good.begin() + (ptrdiff_t) cut);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a truncated payload is refused");
        check(err.find("shorter than the header") != std::string::npos ||
                  err.find("shorter than its records claim") != std::string::npos ||
                  err.find("declares") != std::string::npos,
              "the truncation refusal says the payload is short, not something else");
    }
    WireHeaderRecord hdr{};
    check(!conversation_wire_peek(good.data(), kWireHeaderRecordBytes - 1, hdr, err),
          "a payload shorter than the header cannot be peeked");
    check_refusal(err, "short peek reason");

    // 2. truncated DEEPER, with the header's own numbers fixed to match: the refusal must come from the
    //    blob references, not from the length check.
    {
        std::vector<uint8_t> p = good;
        const size_t cut = p.size() - 8;
        p.resize(cut);
        patch(p, 232, cut);
        patch(p, 208, (uint64_t) (cut - plan.blob_pool_off));
        WireExpect e = expect_for(s, geom, pack);
        e.payload_hash = conversation_wire_payload_hash(p.data(), (uint64_t) p.size());
        e.check_payload_hash = true;  // so the hash is NOT what refuses
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, e, err),
              "a payload whose pool was chopped is refused by its blob references");
        check(err.find("past the end of the pool") != std::string::npos, "and it says which reference");
    }
    // 2b. a slot whose payload region is shorter than the header declares: the header's own numbers are
    //     consistent, so the refusal has to come from the reads themselves.
    {
        ShortLyingSource src(good, good.size() - 100);
        check(!conversation_wire_decode(src, out, info, expect_for(s, geom, pack), err),
              "a slot that cannot serve the bytes its header declares is refused");
        check_refusal(err, "short slot reason");
    }

    // 3. wrong version.
    for (uint16_t v : {(uint16_t) 0, (uint16_t) 2, (uint16_t) 65535}) {
        std::vector<uint8_t> p = good;
        patch16(p, 4, v);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a different wire format version is refused");
        check_refusal(err, "version reason");
        check(!conversation_wire_peek(p.data(), (uint64_t) p.size(), hdr, err), "peek refuses the version too");
    }
    // 4. wrong magic.
    {
        std::vector<uint8_t> p = good;
        patch32(p, 0, 0x44454144u);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "not a hand-off payload is refused");
        check_refusal(err, "magic reason");
        check(!conversation_wire_peek(p.data(), (uint64_t) p.size(), hdr, err), "peek refuses a bad magic");
    }
    // 5. wrong geometry, in every one of the 18 slots.
    for (int i = 0; i < 18; ++i) {
        std::vector<uint8_t> p = good;
        patch(p, 8 + (size_t) i * 8, (uint64_t) s.geometry[(size_t) i] + 1);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a different runtime geometry is refused");
        check_refusal(err, "geometry reason");
        WireExpect loose = expect_for(s, geom, pack);
        loose.check_geometry = false;
        check_ok(conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, loose, err), err,
                 "without check_geometry the payload still parses");
    }
    // 6. wrong geom_hash / pack_hash, naming BOTH values (§3.10).
    for (size_t off : {(size_t) 216, (size_t) 224}) {
        std::vector<uint8_t> p = good;
        patch(p, off, 0x0123456789abcdefull);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a foreign pack or geometry hash is refused");
        check(err.find("0123456789abcdef") != std::string::npos &&
                  err.find(hex(off == 216 ? geom : pack)) != std::string::npos,
              "the hash refusal prints BOTH values");
    }
    // 7. a declared size that exceeds the slot.
    {
        WireExpect e = expect_for(s, geom, pack);
        e.max_payload_bytes = plan.total_bytes - 1;
        check(!conversation_wire_decode(good.data(), (uint64_t) good.size(), out, info, e, err),
              "a payload bigger than its slot is refused");
        check_refusal(err, "slot-size reason");
        std::vector<uint8_t> p = good;
        patch(p, 232, std::numeric_limits<uint64_t>::max() / 2);
        WireExpect e2 = expect_for(s, geom, pack);
        e2.max_payload_bytes = 1u << 20;
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, e2, err),
              "an absurd declared size is refused before anything is read");
        check_refusal(err, "absurd size reason");
    }
    // 8. a corrupt count that would size a huge allocation.
    {
        std::vector<uint8_t> p = good;
        patch(p, (size_t) live_ids, 0x00007fffffffffffull);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a corrupt id count is refused before it sizes anything");
        check_refusal(err, "corrupt count reason");
        p = good;
        patch(p, (size_t) live_ids, (uint64_t) -1);  // negative
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a negative count is refused");
    }
    // 9. an absurd stage_parts nesting depth: build it with the test-only bound raised, refuse it with the
    //    production bound (deviation 8).
    {
        SavedConversation deep = make_image(8, {}, 5, 3);
        // Nest under a RETAINED checkpoint, which is legal (§2.3 says a checkpoint's stage_parts recurse);
        // `live.stage_parts` must stay empty (deviation 7), so that is not where to build it.
        ConversationCheckpoint* at = &deep.checkpoints[0];
        for (int i = 0; i < 40; ++i) {
            at->stage_parts.emplace_back();
            at = &at->stage_parts.back();
            at->ids = {1, 2, 3};
        }
        WirePlan dplan{};
        check(!conversation_wire_plan(deep, dplan, err), "plan() refuses an over-deep image");
        check_refusal(err, "deep plan reason");
        VectorWireSink ds;
        WirePlan dp{};
        WireEncoded de{};
        check_ok(conversation_wire_encode(deep, ds, {geom, pack}, 64, dp, de, err), err, "deep encode (test-only)");
        SavedConversation dout;
        WireExpect e = expect_for(deep, geom, pack);
        e.payload_hash = de.payload_hash;
        e.check_payload_hash = true;
        check(!conversation_wire_decode(ds.data().data(), (uint64_t) ds.data().size(), dout, info, e, err),
              "an over-deep stage_parts chain is refused");
        check(err.find("nested") != std::string::npos, "the depth refusal says so");
        e.max_checkpoint_depth = 64;
        check_ok(conversation_wire_decode(ds.data().data(), (uint64_t) ds.data().size(), dout, info, e, err), err,
                 "the same payload parses with the bound raised");
        check(equal_image(deep, dout), "an over-deep chain round-trips exactly when the bound allows it");
    }
    // 10. the END marker, the record kinds, and the record lengths.  `end_off` comes from walking the stream,
    //     NOT from `blob_pool_off - 16`: when the record stream is shorter than the pool boundary there is
    //     alignment padding in between, and patching a pad byte proves nothing about the END check.
    {
        const Landmarks lm = walk_landmarks(good, plan);
        const size_t end_off = (size_t) lm.end;
        check(end_off + kWireRecordHeaderBytes <= plan.blob_pool_off, "the END record is inside the records");
        check(plan32(good, end_off) == kWireRecordEnd, "the walk really landed on the END record");
        std::vector<uint8_t> p = good;
        patch32(p, end_off, 0x11223344u);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a payload with no END marker is refused");
        check(err.find("END") != std::string::npos, "and the refusal names the END marker");
        p = good;
        patch(p, end_off + 8, 8);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "an END record with a body is refused");
        check(err.find("END") != std::string::npos, "and that refusal names the END marker too");
        p = good;
        patch32(p, kWireHeaderRecordBytes, 99);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "an unexpected record kind is refused");
        check(err.find("kind") != std::string::npos, "and it says it was the kind");
        p = good;
        patch(p, kWireHeaderRecordBytes + 8, 1ull << 40);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a KV record of the wrong length is refused");
        check(err.find("KV array") != std::string::npos, "and it says the KV array was wrong");
        p = good;
        // A CHECKPOINT record declaring more bytes than the payload has - the generic record-length bound.
        patch(p, (size_t) lm.live_record + 8, 1ull << 40);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a record longer than the payload is refused");
        check(err.find("declares more bytes") != std::string::npos, "and it says the record is too long");
        p = good;
        patch32(p, kWireHeaderRecordBytes + 4, 1);  // a record's flags field
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a non-zero record flag is refused");
        check(err.find("flags") != std::string::npos, "and it says it was the flags field");
        // The live CHECKPOINT record's own kind, and a STAGE record's kind.
        p = good;
        patch32(p, (size_t) lm.live_record, 7);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a missing live CHECKPOINT record is refused");
        check(err.find("CHECKPOINT") != std::string::npos, "and the refusal names the record type");
        p = good;
        patch32(p, (size_t) lm.live_record + 4, 3);  // the live record's flags
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a CHECKPOINT record with flags set is refused");
        // A STAGE record whose declared body is far shorter than its contents: the inner CHECKPOINT must
        // still end where the header said, and it does not.
        p = good;
        if (lm.first_stage) {
            patch(p, (size_t) lm.first_stage + 8, 4);
            check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info,
                                            expect_for(s, geom, pack), err),
                  "a STAGE record shorter than its own contents is refused");
            check_refusal(err, "stage truncation reason");
            p = good;
            patch32(p, (size_t) lm.first_stage, kWireRecordKv);  // a stage slot holding the wrong kind
            check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info,
                                            expect_for(s, geom, pack), err),
                  "a missing STAGE record is refused");
            check(err.find("STAGE") != std::string::npos, "and the refusal names the record type");
        }
    }
    // 11. blob references that leave the pool or break the alignment.
    {
        std::vector<uint8_t> p = good;
        patch(p, (size_t) first_blobref + 8, plan.blob_pool_off + plan.blob_pool_bytes + 64);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a blob reference outside the pool is refused");
        check_refusal(err, "blobref reason");
        p = good;
        patch(p, (size_t) first_blobref + 8, plan.blob_pool_off + 3);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a blob reference that is not 64-aligned is refused");
        p = good;
        patch(p, (size_t) first_blobref, 0);  // bytes -> 0 while the offset stays non-zero
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a zero-length blob must be referenced as {0,0}");
    }
    // 12. the reserved / partial / unknown-flag fields must be exactly what the format says.
    for (size_t off : {(size_t) 240, (size_t) 248}) {
        std::vector<uint8_t> p = good;
        patch(p, off, 7);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a non-zero reserved field is refused");
        check_refusal(err, "reserved reason");
    }
    {
        // OR the bad bit in rather than replacing the field: overwriting it also clears cvec /
        // has_stage_parts, and the flag-vs-count consistency check would refuse the payload for a DIFFERENT
        // reason, so the flag check itself would never be exercised.
        std::vector<uint8_t> p = good;
        const uint16_t flags = (uint16_t) ((p[6] | ((uint16_t) p[7] << 8)) | kWireFlagPartial);
        patch16(p, 6, flags);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a partial payload is refused");
        check(err.find("partial") != std::string::npos, "and the refusal says partial");
        p = good;
        patch16(p, 6, (uint16_t) ((p[6] | ((uint16_t) p[7] << 8)) | (1u << 9)));
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "an unknown flag bit is refused, not ignored");
        check(err.find("unknown header flag") != std::string::npos, "and the refusal says unknown flag");
        p = good;
        patch(p, 200, plan.blob_pool_off + 8);  // the pool no longer starts 64-aligned
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "an unaligned blob pool is refused");
        check(err.find("64-byte boundary") != std::string::npos, "and the refusal says alignment");
    }
    // 13. the content hash, and the stage-count rule in BOTH directions (§2.5 step 2).
    {
        std::vector<uint8_t> p = good;
        p[(size_t) plan.blob_pool_off + 100] ^= 0x40;
        WireExpect e = expect_for(s, geom, pack);
        e.payload_hash = enc.payload_hash;
        e.check_payload_hash = true;
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, e, err),
              "one flipped byte fails the content hash");
        check(err.find("payload hash") != std::string::npos, "the hash refusal names itself");
        // ... and the same corruption is caught by the STRUCTURE when it lands on a length field.
        check_ok(conversation_wire_decode(good.data(), (uint64_t) good.size(), out, info, e, err), err,
                 "the uncorrupted payload passes the same hash check");
    }
    {
        WireExpect e = expect_for(s, geom, pack);
        e.stage_parts = 0;  // this decode instance is single-GPU
        check(!conversation_wire_decode(good.data(), (uint64_t) good.size(), out, info, e, err),
              "a split image is refused by a single-GPU instance");
        check_refusal(err, "split-into-single reason");
        const SavedConversation single = make_image(8, {}, 65, 13);
        VectorWireSink ss;
        WireEncoded se{};
        check_ok(conversation_wire_encode(single, ss, {geom, pack}, se, err), err, "single encode");
        WireExpect e2 = expect_for(single, geom, pack);
        e2.stage_parts = 1;  // this decode instance has a stage set
        check(!conversation_wire_decode(ss.data().data(), (uint64_t) ss.data().size(), out, info, e2, err),
              "a single-GPU image is refused by a split instance");
        check_refusal(err, "single-into-split reason");
    }
    // 14. a refused decode writes nothing into the caller's image.
    {
        SavedConversation untouched = make_image(8, {}, 9, 3);
        const SavedConversation pristine = untouched;
        std::vector<uint8_t> p = good;
        patch32(p, 0, 0xdeadbeefu);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), untouched, info, expect_for(s, geom, pack), err),
              "a refused decode returns false");
        check(equal_image(pristine, untouched) && untouched.owner == pristine.owner,
              "a refused decode left the caller's image exactly as it was");
    }
    // 15. the codec refuses an image it must not carry, before writing anything (deviation 7).
    {
        SavedConversation bad = make_image(8, {{4, 8}}, 9, 3);
        bad.live.stage_parts.push_back(bad.stage_parts[0].state);
        VectorWireSink bs;
        WireEncoded be{};
        check(!conversation_wire_encode(bad, bs, {geom, pack}, be, err),
              "a live checkpoint carrying stage parts is refused at encode");
        check_refusal(err, "live stage_parts reason");
        check(bs.data().empty(), "the refused encode wrote nothing");
        WirePlan bplan{};
        check(!conversation_wire_plan(bad, bplan, err), "plan() refuses it too");
    }
    // 16. a sink that runs out of room.
    {
        BoundedSink bs(plan.total_bytes - 1);
        WireEncoded be{};
        check(!conversation_wire_encode(s, bs, {geom, pack}, be, err), "a too-small slot refuses the encode");
        check(err.find("the sink is full") != std::string::npos, "the sink's own reason reaches the caller");
    }
    // 17. the header's `live_tokens` is not used to parse anything, so the ONLY thing that can catch a lie
    //     about it is the decode's declared-vs-parsed agreement check.
    for (uint64_t lie : {(uint64_t) 0, (uint64_t) 1, (uint64_t) s.live.ids.size() + 1, (uint64_t) 1 << 40}) {
        std::vector<uint8_t> p = good;
        patch(p, 192, lie);
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a header that lies about live_tokens is refused");
        check(err.find("live_tokens") != std::string::npos, "and the refusal names live_tokens");
    }
    // 18. a CHECKPOINT record whose declared body length is 8 bytes too long: the contents end early, and
    //     only the record's own length-vs-contents check can say so.
    {
        const Landmarks lm = walk_landmarks(good, plan);
        std::vector<uint8_t> p = good;
        uint64_t declared = 0;
        for (int i = 0; i < 8; ++i) declared |= (uint64_t) p[(size_t) lm.live_record + 8 + (size_t) i] << (i * 8);
        patch(p, (size_t) lm.live_record + 8, declared + 8);  // still inside the payload, but wrong
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a CHECKPOINT record whose length disagrees with its contents is refused");
        check(err.find("does not end where its header said") != std::string::npos,
              "and the refusal says the record's length disagreed");
    }
    // 19. the ids array's 4-byte alignment pad: the writer and the reader must agree on it, and a payload
    //     that omits it must be refused rather than silently mis-parsing the fields that follow.
    {
        const Landmarks lm = walk_landmarks(good, plan);
        // The live checkpoint has an ODD id count in this fixture, so its pad bytes are really there.
        check(s.live.ids.size() % 2 == 1, "the fixture exercises the ids pad");
        std::vector<uint8_t> p = good;
        const size_t ids_at = (size_t) lm.live_record + kWireRecordHeaderBytes;
        const uint64_t n_ids = s.live.ids.size();
        const size_t pad_at = ids_at + 8 + (size_t) n_ids * 4;
        check(p[pad_at] == 0 && p[pad_at + 1] == 0 && p[pad_at + 2] == 0 && p[pad_at + 3] == 0,
              "the ids pad bytes are zero on the wire");
        p[pad_at] = 0x7f;
        check(!conversation_wire_decode(p.data(), (uint64_t) p.size(), out, info, expect_for(s, geom, pack), err),
              "a non-zero ids pad byte is refused, not skipped");
        check(err.find("reserved field is not zero") != std::string::npos, "and it says which kind of byte");
    }
}

// ---------------------------------------------------------------- the slot sink, on a real fd

#if defined(__linux__) && defined(SYS_memfd_create)
void test_slot_sink() {
    // memfd: a real descriptor, real `pwrite`/`pread`, and nothing written to any filesystem - no /dev/shm,
    // no disk.  This is the exact path S4.3.5 uses to fill a hand-off slot.
    const int fd = (int) ::syscall(SYS_memfd_create, "strata-wire-test", MFD_CLOEXEC);
    check(fd >= 0, "memfd_create gave a descriptor");
    const SavedConversation s = make_image(8, {{2, 5}, {5, 8}}, 33, 17);
    WirePlan plan{};
    std::string err;
    check_ok(conversation_wire_plan(s, plan, err), err, "slot plan");
    const uint64_t slot_payload_off = 4096;  // what HandoffClaim::payload_off looks like
    check(::ftruncate(fd, (off_t) (slot_payload_off + plan.total_bytes)) == 0, "the memfd was sized");
    SlotWireSink sink(fd, slot_payload_off, plan.total_bytes);
    WireEncoded enc{};
    check_ok(conversation_wire_encode(s, sink, {7, 8}, enc, err), err, "slot encode");
    check(sink.written() == plan.total_bytes, "the slot sink wrote the whole payload");
    FileWireSource src(fd, slot_payload_off, plan.total_bytes);
    SavedConversation out;
    WireDecoded info{};
    WireExpect e = expect_for(s, 7, 8);
    e.payload_hash = enc.payload_hash;
    e.check_payload_hash = true;
    check_ok(conversation_wire_decode(src, out, info, e, err), err, "slot decode through pread");
    check(equal_image(s, out), "the fd path round-trips a three-carve image");
    // The bytes really are in the file at the right offset, which is what S4.3.2's slot hash covers.
    std::vector<uint8_t> raw(plan.total_bytes);
    check(::pread(fd, raw.data(), raw.size(), (off_t) slot_payload_off) == (ssize_t) raw.size(),
          "the slot's bytes read back");
    check(conversation_wire_payload_hash(raw.data(), raw.size()) == enc.payload_hash,
          "the slot's own FNV-1a equals the codec's (deviation 4)");
    SlotWireSink tight(fd, slot_payload_off, plan.total_bytes - 1);
    WireEncoded te{};
    check(!conversation_wire_encode(s, tight, {7, 8}, te, err), "a slot one byte short refuses");
    check(tight.error().find("claim a larger tier") != std::string::npos, "the slot refusal names the fix");
    ::close(fd);
}
#endif

void test_mapping_sink() {
    const SavedConversation s = make_image(8, {}, 17, 23);
    WirePlan plan{};
    std::string err;
    check_ok(conversation_wire_plan(s, plan, err), err, "mapping plan");
    std::vector<uint8_t> slot(kWireHeaderRecordBytes + plan.total_bytes + 1, 0xa5);  // +1: the sentinel byte
    SlotWireSink sink(slot.data() + kWireHeaderRecordBytes, plan.total_bytes);
    WireEncoded enc{};
    check_ok(conversation_wire_encode(s, sink, {9, 10}, enc, err), err, "mapping encode");
    VectorWireSink ref;
    WireEncoded refenc{};
    check_ok(conversation_wire_encode(s, ref, {9, 10}, refenc, err), err, "reference encode");
    check(ref.data().size() == plan.total_bytes, "the reference encode is the same size");
    check(std::equal(ref.data().begin(), ref.data().end(), slot.begin() + kWireHeaderRecordBytes),
          "the mapping sink produced the same bytes as the vector sink");
    check(slot[kWireHeaderRecordBytes - 1] == 0xa5, "the sink wrote nothing before the payload");
    check(slot[kWireHeaderRecordBytes + plan.total_bytes] == 0xa5, "the sink wrote nothing past the payload");
    SavedConversation out;
    WireDecoded info{};
    WireExpect e = expect_for(s, 9, 10);
    e.payload_hash = enc.payload_hash;
    e.check_payload_hash = true;
    check_ok(conversation_wire_decode(slot.data() + kWireHeaderRecordBytes, plan.total_bytes, out, info, e, err),
             err, "mapping decode");
    check(equal_image(s, out), "the mapping path round-trips");
}

} // namespace

int main() {
    test_layout();
    test_round_trips();
    test_refusals();
    test_mapping_sink();
#if defined(__linux__) && defined(SYS_memfd_create)
    test_slot_sink();
#endif
    std::printf("saved_conv_wire_test: %d checks passed\n", checks);
    return 0;
}
