// src/core/conversation_wire.cpp - S4.3.1: the SavedConversation payload codec.
//
// CPU-ONLY ON PURPOSE.  No CUDA call, no session, no model, no file this code opens: a `SavedConversation`
// is host storage end to end (`ConversationBuffer` is a directory of host `std::vector<uint8_t>` segments),
// so the whole format - including the parts that matter, a 20 MiB buffer spanning segments and a corrupt
// payload that would otherwise make the receiver allocate gigabytes - is testable in a fraction of a second.
// That is deliberate: the payload is 0.2-3 GB in production, and a format bug that only shows up there costs
// a GiB of copied bytes and a dead session.
//
// The one non-obvious structural decision: the record stream and the blob pool come from ONE traversal of
// the image (`Emitter`), run twice - once with no sink to lay the pool out, once with the sink to write it.
// Blob offsets are therefore assigned in exactly the order the BLOBREFs are emitted, in both passes, by the
// same code.  Two hand-written passes over the same struct is how format drift is born.
//
// Offsets are assigned RELATIVE TO THE POOL START and made absolute when they are written, because the pool's
// absolute start is not known until the record stream has been laid out.  The pool start is 64-aligned, so
// relative-64-alignment and absolute-64-alignment are the same statement (deviation 6 in the header).
#include "strata/core/conversation_wire.hpp"
#include "strata/core/pinned.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include <unistd.h>

namespace strata::core {
namespace {

// ---------------------------------------------------------------- diagnostics

bool fail(std::string& e, const char* m) {
    e = std::string("conversation wire: ") + m;
    return false;
}

std::string hex16(uint64_t v) {
    char b[24];
    std::snprintf(b, sizeof b, "%016llx", (unsigned long long) v);
    return std::string(b);
}

std::string dec(uint64_t v) { return std::to_string(v); }

bool add_checked(uint64_t& total, uint64_t n, std::string& error) {
    if (n > std::numeric_limits<uint64_t>::max() - total) return fail(error, "byte count overflow");
    total += n;
    return true;
}
bool mul_checked(uint64_t& out, uint64_t a, uint64_t b, std::string& error) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return fail(error, "byte count overflow");
    out = a * b;
    return true;
}
inline uint64_t round_up_64(uint64_t n) { return (n + 63ull) & ~63ull; }

// ---------------------------------------------------------------- little-endian scalars
//
// Written and read field by field rather than by dumping a struct: the format's promise is "fixed width,
// fixed alignment, little-endian", and a struct dump would silently depend on the compiler's padding and on
// the host's byte order.  `bool` never appears on the wire at all (§2.2).

void put_u16(uint8_t* p, uint16_t v) { p[0] = (uint8_t) v; p[1] = (uint8_t) (v >> 8); }
void put_u32(uint8_t* p, uint32_t v) {
    for (int i = 0; i < 4; ++i) p[i] = (uint8_t) (v >> (i * 8));
}
void put_u64(uint8_t* p, uint64_t v) {
    for (int i = 0; i < 8; ++i) p[i] = (uint8_t) (v >> (i * 8));
}
uint16_t get_u16(const uint8_t* p) { return (uint16_t) (p[0] | ((uint16_t) p[1] << 8)); }
uint32_t get_u32(const uint8_t* p) {
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= (uint32_t) p[i] << (i * 8);
    return v;
}
uint64_t get_u64(const uint8_t* p) {
    uint64_t v = 0;
    for (int i = 0; i < 8; ++i) v |= (uint64_t) p[i] << (i * 8);
    return v;
}

/// §2.3: a CHECKPOINT's ids are "4-byte aligned", so the array is followed by 0 or 4 pad bytes to put the
/// next i64 on 8.  Deterministic either way: the pad is written as zero and checked as zero.
uint64_t ids_padding(uint64_t n_ids) { return (n_ids * 4) % 8 ? 4 : 0; }

constexpr uint64_t kKvBodyBytes = 4 + 4 + 6 * 8 + 5 * 16;  // 136, §2.3
constexpr int kKvRefs = 5;
constexpr int kCpRefs = 5;

// ---------------------------------------------------------------- the sizes of things

bool size_checkpoint(const ConversationCheckpoint& c, int depth, int max_depth, uint64_t& body,
                     std::string& error) {
    if (depth > max_depth) return fail(error, "checkpoint stage_parts nesting exceeds the bound");
    uint64_t n = 0, t = 0;
    if (!mul_checked(t, (uint64_t) c.ids.size(), 4, error)) return false;
    if (!add_checked(n, 8 + t + ids_padding((uint64_t) c.ids.size()), error)) return false;
    if (!mul_checked(t, (uint64_t) c.imgs.size(), 16, error)) return false;
    if (!add_checked(n, 8 + t, error)) return false;
    if (!add_checked(n, kCpRefs * 16 + 8 + 8, error)) return false;  // 5 BLOBREFs, used, n_stage_parts
    for (const auto& part : c.stage_parts) {
        uint64_t sub = 0;
        if (!size_checkpoint(part, depth + 1, max_depth, sub, error)) return false;
        if (!add_checked(n, kWireRecordHeaderBytes + sub, error)) return false;
    }
    body = n;
    return true;
}

bool size_stage(const ConversationStageSnapshot& s, int depth, int max_depth, uint64_t& body,
                std::string& error) {
    uint64_t n = 16, cb = 0, t = 0;  // layer_lo, layer_hi
    if (!size_checkpoint(s.state, depth, max_depth, cb, error)) return false;
    if (!add_checked(n, kWireRecordHeaderBytes + cb, error)) return false;
    if (!mul_checked(t, (uint64_t) s.kv.size(), kWireRecordHeaderBytes + kKvBodyBytes, error)) return false;
    if (!add_checked(n, 8 + t, error)) return false;  // n_kv + the KV records
    body = n;
    return true;
}

// ---------------------------------------------------------------- the emitter

/// Feeds everything it is given to the running FNV-1a and forwards it to the real sink, so the payload's
/// content hash is produced in the SAME single pass that writes it (deviation 4) - no patch, no second pass.
class HashingSink final : public WireSink {
public:
    HashingSink(WireSink& inner, uint64_t& hash, uint64_t& bytes) : inner_(inner), hash_(hash), bytes_(bytes) {}
    bool write(const uint8_t* data, size_t n) override {
        if (n) {
            hash_ = fnv1a64(data, (uint64_t) n, hash_);
            bytes_ += n;
        }
        return inner_.write(data, n);
    }
    std::string refusal() const override { return inner_.refusal(); }

private:
    WireSink& inner_;
    uint64_t& hash_;
    uint64_t& bytes_;
};

class Emitter {
public:
    Emitter(const SavedConversation& image, const WireIdentity& identity, WireSink* sink, int max_depth)
        : image_(image), id_(identity), sink_(sink), max_depth_(max_depth) {}

    /// Lay the payload out (always), then write it (when a sink was given).
    bool run(WirePlan& plan, WireEncoded& enc);
    const std::string& error() const { return error_; }

private:
    bool out_bytes(const uint8_t* p, size_t n) {
        pos_ += n;
        if (layout_) return true;
        if (sink_->write(p, n)) return true;
        // The sink refused: keep ITS reason, so "the slot needs N bytes and holds M" reaches the log rather
        // than a generic "the sink refused".
        error_ = std::string("conversation wire: the sink refused the payload at byte ") + dec(pos_ - n);
        if (sink_ != nullptr) {
            const std::string why = sink_->refusal();
            if (!why.empty()) error_ += ": " + why;
        }
        return false;
    }
    bool pad_to(uint64_t align) {
        const uint64_t rem = pos_ % align;
        if (!rem) return true;
        static const uint8_t zeros[64] = {};
        return out_bytes(zeros, (size_t) (align - rem));
    }

    bool header();
    bool kv_record(const ConversationKv& k);
    bool checkpoint_record(const ConversationCheckpoint& c, int depth);
    bool stage_record(const ConversationStageSnapshot& s, int depth);
    bool body();
    bool pool();

    /// One pool entry: either a `ConversationKv`'s `ConversationBuffer` or a `ConversationCheckpoint`'s
    /// plain `std::vector<uint8_t>` (deviation 0 - §2.1 calls all ten "ConversationBuffers" and five of them
    /// are not).  Both become one BLOBREF and one pool entry; only the copy primitive differs.
    struct PoolEntry {
        const ConversationBuffer* buf = nullptr;
        const std::vector<uint8_t>* vec = nullptr;
        uint64_t rel = 0;   ///< pool-relative offset, assigned in the layout pass
        size_t n = 0;
    };

    /// Assign (layout pass) or look up (write pass) a blob's pool offset, in emission order.  Zero-length
    /// buffers are `{0,0}` with no pool entry (§2.3), in both passes.
    bool blob(const ConversationBuffer& b, uint64_t& absolute_offset) { return blob_of(&b, nullptr, absolute_offset); }
    bool blob(const std::vector<uint8_t>& v, uint64_t& absolute_offset) { return blob_of(nullptr, &v, absolute_offset); }
    bool blob_of(const ConversationBuffer* buf, const std::vector<uint8_t>* vec, uint64_t& absolute_offset) {
        absolute_offset = 0;
        const size_t n = buf ? buf->size() : vec->size();
        if (n == 0) return true;
        if (layout_) {
            const uint64_t rel = round_up_64(pool_rel_);
            if (!add_checked(pool_rel_, rel - pool_rel_ + n, error_)) return false;
            entries_.push_back(PoolEntry{buf, vec, rel, n});
            ++plan_.blobs;
            return true;
        }
        if (blob_index_ >= entries_.size()) return fail(error_, "internal: blob order changed between passes");
        absolute_offset = plan_.blob_pool_off + entries_[blob_index_++].rel;
        return true;
    }

    void push_u32(uint32_t v) {
        uint8_t b[4];
        put_u32(b, v);
        scratch_.insert(scratch_.end(), b, b + 4);
    }
    void push_u64(uint64_t v) {
        uint8_t b[8];
        put_u64(b, v);
        scratch_.insert(scratch_.end(), b, b + 8);
    }
    void push_i64(int64_t v) { push_u64((uint64_t) v); }

    const SavedConversation& image_;
    WireIdentity id_;
    WireSink* sink_ = nullptr;
    bool layout_ = true;   ///< true while assigning pool offsets, false while writing them out
    int max_depth_ = kWireMaxCheckpointDepth;
    uint64_t pos_ = 0;        ///< position in the payload so far
    uint64_t pool_rel_ = 0;   ///< position in the pool so far (pool-relative)
    std::vector<PoolEntry> entries_;      ///< the pool, in emission order, built by the layout pass
    size_t blob_index_ = 0;
    WirePlan plan_{};
    std::string error_;
    std::vector<uint8_t> scratch_;
};

bool Emitter::header() {
    // Built whole in a scratch buffer: it is small, and building it as one block means the offsets inside it
    // are the offsets of the record, not offsets into a stream.
    uint8_t raw[kWireHeaderRecordBytes];
    std::memset(raw, 0, sizeof raw);
    uint16_t flags = 0;
    if (image_.cvec) flags |= kWireFlagCvec;
    if (!image_.stage_parts.empty()) flags |= kWireFlagHasStageParts;
    put_u32(raw + 0, kWireMagic);
    put_u16(raw + 4, kWireFormatVersion);
    put_u16(raw + 6, flags);
    for (int i = 0; i < 18; ++i) put_u64(raw + 8 + (size_t) i * 8, (uint64_t) image_.geometry[(size_t) i]);
    put_u64(raw + 152, (uint64_t) image_.layer_lo);
    put_u64(raw + 160, (uint64_t) image_.layer_hi);
    put_u64(raw + 168, (uint64_t) image_.kv.size());
    put_u64(raw + 176, (uint64_t) image_.checkpoints.size());
    put_u64(raw + 184, (uint64_t) image_.stage_parts.size());
    put_u64(raw + 192, (uint64_t) image_.live.ids.size());
    put_u64(raw + 200, plan_.blob_pool_off);
    put_u64(raw + 208, plan_.blob_pool_bytes);
    put_u64(raw + 216, id_.geom_hash);
    put_u64(raw + 224, id_.pack_hash);
    put_u64(raw + 232, plan_.total_bytes);
    put_u64(raw + 240, 0);  // deviation 4: the content hash travels in the slot header, so this stays zero
    put_u64(raw + 248, 0);
    plan_.flags = flags;
    plan_.n_kv = (int64_t) image_.kv.size();
    plan_.n_checkpoints = (int64_t) image_.checkpoints.size();
    plan_.n_stage_parts = (int64_t) image_.stage_parts.size();
    plan_.live_tokens = (int64_t) image_.live.ids.size();
    return out_bytes(raw, sizeof raw);
}

bool Emitter::kv_record(const ConversationKv& k) {
    const ConversationBuffer* refs[kKvRefs] = {&k.k, &k.v, &k.k_scale, &k.v_scale, &k.pooled};
    scratch_.clear();
    scratch_.reserve(kKvBodyBytes);
    push_u32((uint32_t) (int32_t) k.format);
    push_u32(0);  // pad, so every i64 below is 8-aligned
    push_i64(k.cells);
    push_i64(k.heads);
    push_i64(k.head_dim);
    push_i64(k.page_size);
    push_i64(k.pooled_rows);
    push_i64(k.idx_dim);
    for (int i = 0; i < kKvRefs; ++i) {
        uint64_t off = 0;
        if (!blob(*refs[i], off)) return false;
        push_u64((uint64_t) refs[i]->size());
        push_u64(off);
    }
    if (scratch_.size() != kKvBodyBytes) return fail(error_, "internal: a KV record is the wrong size");
    return out_bytes(scratch_.data(), scratch_.size());
}

bool Emitter::checkpoint_record(const ConversationCheckpoint& c, int depth) {
    if (depth > max_depth_) return fail(error_, "checkpoint stage_parts nesting exceeds the bound");
    uint64_t body_bytes = 0;
    if (!size_checkpoint(c, depth, max_depth_, body_bytes, error_)) return false;
    uint8_t rh[kWireRecordHeaderBytes];
    put_u32(rh + 0, kWireRecordCheckpoint);
    put_u32(rh + 4, 0);
    put_u64(rh + 8, body_bytes);
    if (!out_bytes(rh, sizeof rh)) return false;

    scratch_.clear();
    push_u64((uint64_t) c.ids.size());
    if (!out_bytes(scratch_.data(), scratch_.size())) return false;
    // ids stream in chunks: a 524 288-token conversation would otherwise stage 2 MB for no reason.
    for (size_t i = 0; i < c.ids.size(); i += 4096) {
        const size_t n = std::min<size_t>(4096, c.ids.size() - i);
        scratch_.clear();
        for (size_t j = 0; j < n; ++j) push_u32((uint32_t) c.ids[i + j]);
        if (!out_bytes(scratch_.data(), scratch_.size())) return false;
    }
    if (ids_padding((uint64_t) c.ids.size())) {
        static const uint8_t z[4] = {};
        if (!out_bytes(z, 4)) return false;
    }
    scratch_.clear();
    push_u64((uint64_t) c.imgs.size());
    if (!out_bytes(scratch_.data(), scratch_.size())) return false;
    for (size_t i = 0; i < c.imgs.size(); i += 1024) {
        const size_t n = std::min<size_t>(1024, c.imgs.size() - i);
        scratch_.clear();
        for (size_t j = 0; j < n; ++j) {
            push_i64(c.imgs[i + j].start);
            push_u64(c.imgs[i + j].hash);
        }
        if (!out_bytes(scratch_.data(), scratch_.size())) return false;
    }
    const std::vector<uint8_t>* refs[kCpRefs] = {&c.gdn, &c.ple, &c.tails, &c.dead, &c.block_pos};
    for (int i = 0; i < kCpRefs; ++i) {
        uint64_t off = 0;
        if (!blob(*refs[i], off)) return false;
        uint8_t b[16];
        put_u64(b, (uint64_t) refs[i]->size());
        put_u64(b + 8, off);
        if (!out_bytes(b, sizeof b)) return false;
    }
    scratch_.clear();
    push_u64(c.used);
    push_u64((uint64_t) c.stage_parts.size());
    if (!out_bytes(scratch_.data(), scratch_.size())) return false;
    for (const auto& part : c.stage_parts)
        if (!checkpoint_record(part, depth + 1)) return false;
    return true;
}

bool Emitter::stage_record(const ConversationStageSnapshot& s, int depth) {
    uint64_t body_bytes = 0;
    if (!size_stage(s, depth, max_depth_, body_bytes, error_)) return false;
    uint8_t rh[kWireRecordHeaderBytes];
    put_u32(rh + 0, kWireRecordStage);
    put_u32(rh + 4, 0);
    put_u64(rh + 8, body_bytes);
    if (!out_bytes(rh, sizeof rh)) return false;
    uint8_t b[16];
    put_u64(b, (uint64_t) s.layer_lo);
    put_u64(b + 8, (uint64_t) s.layer_hi);
    if (!out_bytes(b, sizeof b)) return false;
    if (!checkpoint_record(s.state, depth)) return false;
    scratch_.clear();
    push_u64((uint64_t) s.kv.size());
    if (!out_bytes(scratch_.data(), scratch_.size())) return false;
    for (const auto& k : s.kv) {
        uint8_t rh2[kWireRecordHeaderBytes];
        put_u32(rh2 + 0, kWireRecordKv);
        put_u32(rh2 + 4, 0);
        put_u64(rh2 + 8, kKvBodyBytes);
        if (!out_bytes(rh2, sizeof rh2)) return false;
        if (!kv_record(k)) return false;
    }
    return true;
}

bool Emitter::body() {
    // §2.3's fixed order: the KV array, live, the checkpoints, the stage parts, END.
    for (const auto& k : image_.kv) {
        uint8_t rh[kWireRecordHeaderBytes];
        put_u32(rh + 0, kWireRecordKv);
        put_u32(rh + 4, 0);
        put_u64(rh + 8, kKvBodyBytes);
        if (!out_bytes(rh, sizeof rh)) return false;
        if (!kv_record(k)) return false;
    }
    if (!checkpoint_record(image_.live, 0)) return false;
    for (const auto& c : image_.checkpoints)
        if (!checkpoint_record(c, 0)) return false;
    for (const auto& s : image_.stage_parts)
        if (!stage_record(s, 0)) return false;
    uint8_t end[kWireRecordHeaderBytes];
    put_u32(end + 0, kWireRecordEnd);
    put_u32(end + 4, 0);
    put_u64(end + 8, 0);
    if (!out_bytes(end, sizeof end)) return false;
    return true;
}

bool Emitter::pool() {
    // Written from the SAME entry list the layout pass built, so an offset cannot drift from what the
    // BLOBREFs said.  The gaps are zero-filled, which is what makes the payload deterministic.
    uint64_t cursor = 0;  // pool-relative
    static const uint8_t zeros[4096] = {};
    for (const PoolEntry& e : entries_) {
        while (cursor < e.rel) {
            const size_t n = (size_t) std::min<uint64_t>(sizeof zeros, e.rel - cursor);
            if (!out_bytes(zeros, n)) return false;
            cursor += n;
        }
        if (e.vec) {
            // A checkpoint's running state is contiguous, but it is still written in bounded pieces so one
            // sink call is never larger than any other.
            for (size_t at = 0; at < e.n; at += kWireChunkBytes) {
                const size_t n = std::min<size_t>(kWireChunkBytes, e.n - at);
                if (!out_bytes(e.vec->data() + at, n)) return false;
            }
            cursor += e.n;
            continue;
        }
        // `visit` walks the segments in order and hands over each one's pointer: a 20 MiB buffer is written
        // as its own segments, with no temporary anywhere near its size.
        bool ok = true;
        e.buf->visit(0, e.n, [&](const uint8_t* p, size_t n, size_t) {
            if (!out_bytes(p, n)) ok = false;
            return ok;
        });
        if (!ok) return false;  // out_bytes already named the sink's reason
        cursor += e.n;
    }
    while (cursor < plan_.blob_pool_bytes) {
        const size_t n = (size_t) std::min<uint64_t>(sizeof zeros, plan_.blob_pool_bytes - cursor);
        if (!out_bytes(zeros, n)) return false;
        cursor += n;
    }
    return true;
}

bool Emitter::run(WirePlan& plan, WireEncoded& enc) {
    // Pass 1: lay out.  `out_bytes` only advances `pos_` and `blob()` assigns pool offsets, so nothing is
    // ever handed to the sink here - the sink may be a `pwrite` into a slot, and a layout pass must not write.
    layout_ = true;
    if (!header()) return fail(error_, "the layout pass failed on the header");
    if (!body()) return false;  // body() already named the reason
    plan_.body_bytes = pos_ - kWireHeaderRecordBytes;
    plan_.blob_pool_off = round_up_64(pos_);
    plan_.blob_pool_bytes = entries_.empty() ? 0 : round_up_64(pool_rel_);
    plan_.total_bytes = plan_.blob_pool_off + plan_.blob_pool_bytes;
    plan = plan_;
    if (sink_ == nullptr) return true;

    // Pass 2: write.  The header carries the real pool numbers, which are only knowable after the layout, so
    // the header is emitted here rather than before it.  The traversal is the same code, so the record
    // lengths and the blob order cannot disagree with pass 1 - and if they ever did, the checks below stop
    // the payload rather than shipping it.
    layout_ = false;
    blob_index_ = 0;
    pool_rel_ = 0;
    enc.payload_hash = kWireHashSeed;  // the running hash starts at FNV-1a's offset basis, not 0
    HashingSink hashed(*sink_, enc.payload_hash, enc.bytes);
    sink_ = &hashed;
    pos_ = 0;
    if (!header()) return false;  // a sink refusal, already named by out_bytes
    if (!body()) return false;
    if (!pad_to(kWireBlobAlign)) return false;
    if (!pool()) return false;
    if (pos_ != plan_.total_bytes) return fail(error_, "internal: written bytes disagree with the plan");
    if (blob_index_ != entries_.size()) return fail(error_, "internal: not every blob was written");
    enc.blob_pool_bytes = plan_.blob_pool_bytes;
    enc.blobs = plan_.blobs;
    return true;
}

// ---------------------------------------------------------------- the decoder

/// A bounded cursor over one region of the payload.  Every read checks the remaining bytes FIRST, so a
/// corrupt length can never make the reader walk past the payload or allocate from a number it invented.
/// It writes into the DECODER's error string, not its own: a refusal that loses its reason is a refusal the
/// operator cannot act on, and the caller's `err` is the only thing that reaches the log.
class Cursor {
public:
    Cursor(const WireSource& src, uint64_t begin, uint64_t end, std::string& error)
        : src_(src), pos_(begin), end_(end), error_(error) {}
    uint64_t pos() const { return pos_; }
    uint64_t left() const { return end_ - pos_; }

    bool take(uint8_t* dst, size_t n) {
        if (n > left()) return fail(error_, "the payload is shorter than its records claim");
        if (!n) return true;
        if (!src_.read(dst, pos_, n)) return fail(error_, "the source refused a read");
        pos_ += n;
        return true;
    }
    bool zeros(size_t n) {
        uint8_t buf[64];
        while (n) {
            const size_t k = std::min<size_t>(n, sizeof buf);
            if (!take(buf, k)) return false;
            for (size_t i = 0; i < k; ++i)
                if (buf[i]) return fail(error_, "a reserved field is not zero");
            n -= k;
        }
        return true;
    }
    bool u16(uint16_t& v) { uint8_t b[2]; if (!take(b, 2)) return false; v = get_u16(b); return true; }
    bool u32(uint32_t& v) { uint8_t b[4]; if (!take(b, 4)) return false; v = get_u32(b); return true; }
    bool u64(uint64_t& v) { uint8_t b[8]; if (!take(b, 8)) return false; v = get_u64(b); return true; }
    bool i64(int64_t& v) { uint64_t u = 0; if (!u64(u)) return false; v = (int64_t) u; return true; }

private:
    const WireSource& src_;
    uint64_t pos_, end_;
    std::string& error_;
};

/// Deviation 8: refuse before allocating.  `cap` is a structural bound and `room` is what the record
/// actually has left for this array - so a corrupt count cannot ask for more bytes than the payload holds.
bool checked_count(int64_t n, int64_t cap, const char* what, uint64_t room, uint64_t element,
                   std::string& error) {
    if (n < 0) return fail(error, what);
    if (n > cap) {
        error = std::string("conversation wire: ") + what + " " + dec((uint64_t) n) +
                " exceeds the bound " + dec((uint64_t) cap) + " (a corrupt payload cannot size an allocation)";
        return false;
    }
    if (element && (uint64_t) n > room / element) {
        error = std::string("conversation wire: ") + what + " " + dec((uint64_t) n) +
                " does not fit the record's remaining bytes";
        return false;
    }
    return true;
}

class Decoder {
public:
    Decoder(const WireSource& src, SavedConversation& target, WireDecoded& info, const WireExpect& expect)
        : src_(src), target_(target), out_(), info_(info), expect_(expect) {}

    /// All-or-nothing: `target_` is only assigned when the whole payload parsed.
    bool run(std::string& error);

private:
    bool header(std::string& error);
    bool hash(std::string& error);
    bool kv(Cursor& c, ConversationKv& k, std::string& error);
    bool checkpoint(Cursor& c, ConversationCheckpoint& cp, int depth, std::string& error);
    bool stage(Cursor& c, ConversationStageSnapshot& s, std::string& error);
    bool blobref(Cursor& c, ConversationBuffer& b, std::string& error);
    bool blobref(Cursor& c, std::vector<uint8_t>& v, std::string& error);

    const WireSource& src_;
    SavedConversation& target_;
    SavedConversation out_;
    WireDecoded& info_;
    const WireExpect& expect_;
    WireHeaderRecord hdr_{};
    uint64_t payload_bytes_ = 0, pool_begin_ = 0, pool_end_ = 0;
    size_t blobs_ = 0;
};

bool Decoder::header(std::string& error) {
    if (src_.size() < kWireHeaderRecordBytes) return fail(error, "the payload is shorter than the header record");
    uint8_t raw[kWireHeaderRecordBytes];
    if (!src_.read(raw, 0, sizeof raw)) return fail(error, "the source refused the header");
    const uint8_t* p = raw;
    hdr_.magic = get_u32(p + 0);
    hdr_.fmt_version = get_u16(p + 4);
    hdr_.flags = get_u16(p + 6);
    for (int i = 0; i < 18; ++i) hdr_.geometry[i] = get_u64(p + 8 + (size_t) i * 8);
    hdr_.layer_lo = get_u64(p + 152);
    hdr_.layer_hi = get_u64(p + 160);
    hdr_.n_kv = (int64_t) get_u64(p + 168);
    hdr_.n_checkpoints = (int64_t) get_u64(p + 176);
    hdr_.n_stage_parts = (int64_t) get_u64(p + 184);
    hdr_.live_tokens = (int64_t) get_u64(p + 192);
    hdr_.blob_pool_off = get_u64(p + 200);
    hdr_.blob_pool_bytes = get_u64(p + 208);
    hdr_.geom_hash = get_u64(p + 216);
    hdr_.pack_hash = get_u64(p + 224);
    hdr_.payload_bytes = get_u64(p + 232);
    hdr_.payload_hash = get_u64(p + 240);
    hdr_.reserved = get_u64(p + 248);

    if (hdr_.magic != kWireMagic) {
        error = "conversation wire: not a hand-off payload (magic 0x" + hex16(hdr_.magic).substr(8) +
                " != 0x53523448 = 'SR4H')";
        return false;
    }
    if (hdr_.fmt_version != kWireFormatVersion) {
        error = "conversation wire: payload format version " + dec(hdr_.fmt_version) + " != " +
                dec(kWireFormatVersion) + " (this engine speaks a different wire format)";
        return false;
    }
    if (hdr_.flags & kWireFlagPartial) return fail(error, "the payload says it is partial");
    if (hdr_.flags & (uint16_t) ~kWireFlagKnown) {
        error = "conversation wire: unknown header flag bits 0x" +
                hex16(hdr_.flags & (uint16_t) ~kWireFlagKnown).substr(12);
        return false;
    }
    if (hdr_.payload_hash != 0) return fail(error, "the header's payload_hash field is not zero (deviation 4)");
    if (hdr_.reserved != 0) return fail(error, "the header's reserved field is not zero");
    if (hdr_.payload_bytes < kWireHeaderRecordBytes) return fail(error, "the declared payload size is absurd");
    if (expect_.max_payload_bytes && hdr_.payload_bytes > expect_.max_payload_bytes) {
        error = "conversation wire: the payload declares " + dec(hdr_.payload_bytes) + " bytes, more than the " +
                dec(expect_.max_payload_bytes) + "-byte slot it arrived in";
        return false;
    }
    if (hdr_.payload_bytes > src_.size()) {
        error = "conversation wire: the payload declares " + dec(hdr_.payload_bytes) + " bytes but only " +
                dec(src_.size()) + " are present (truncated)";
        return false;
    }
    payload_bytes_ = hdr_.payload_bytes;
    if (hdr_.blob_pool_off < kWireHeaderRecordBytes || hdr_.blob_pool_off % kWireBlobAlign)
        return fail(error, "the blob pool does not start on a 64-byte boundary");
    if (hdr_.blob_pool_bytes > payload_bytes_ - hdr_.blob_pool_off)
        return fail(error, "the blob pool runs past the end of the payload");
    pool_begin_ = hdr_.blob_pool_off;
    pool_end_ = hdr_.blob_pool_off + hdr_.blob_pool_bytes;
    if ((hdr_.flags & kWireFlagHasStageParts) != 0 ? hdr_.n_stage_parts <= 0 : hdr_.n_stage_parts > 0)
        return fail(error, "the stage-parts flag and the stage-parts count disagree");

    if (expect_.check_geometry) {
        bool same = true;
        for (int i = 0; i < 18; ++i)
            if (hdr_.geometry[i] != (uint64_t) expect_.geometry[(size_t) i]) same = false;
        if (!same) {
            error = "conversation wire: incompatible runtime geometry - " + conversation_wire_describe(hdr_);
            return false;
        }
    }
    if (expect_.check_hashes) {
        if (hdr_.geom_hash != expect_.geom_hash) {
            error = "conversation wire: geometry hash " + hex16(hdr_.geom_hash) + " != " +
                    hex16(expect_.geom_hash) + " (a different engine build or layer split made this payload)";
            return false;
        }
        if (hdr_.pack_hash != expect_.pack_hash) {
            error = "conversation wire: pack hash " + hex16(hdr_.pack_hash) + " != " + hex16(expect_.pack_hash) +
                    " (a different model made this payload)";
            return false;
        }
    }
    if (expect_.stage_parts >= 0 && hdr_.n_stage_parts != expect_.stage_parts) {
        error = "conversation wire: the payload carries " + dec((uint64_t) hdr_.n_stage_parts) +
                " layer-split stage parts and this instance has " + dec((uint64_t) expect_.stage_parts) +
                " (a split image and a single-GPU image must never be mounted into each other)";
        return false;
    }
    if (!checked_count(hdr_.n_kv, kWireMaxKvRecords, "n_kv", 0, 0, error)) return false;
    if (!checked_count(hdr_.n_checkpoints, kWireMaxCheckpoints, "n_checkpoints", 0, 0, error)) return false;
    if (!checked_count(hdr_.n_stage_parts, kWireMaxStageParts, "n_stage_parts", 0, 0, error)) return false;
    if (hdr_.live_tokens < 0) return fail(error, "live_tokens is negative");

    info_.bytes = payload_bytes_;
    info_.geom_hash = hdr_.geom_hash;
    info_.pack_hash = hdr_.pack_hash;
    for (int i = 0; i < 18; ++i) info_.geometry[i] = (int64_t) hdr_.geometry[i];
    info_.layer_lo = (int64_t) hdr_.layer_lo;
    info_.layer_hi = (int64_t) hdr_.layer_hi;
    info_.n_kv = hdr_.n_kv;
    info_.n_checkpoints = hdr_.n_checkpoints;
    info_.n_stage_parts = hdr_.n_stage_parts;
    info_.live_tokens = hdr_.live_tokens;
    info_.flags = hdr_.flags;
    info_.cvec = (hdr_.flags & kWireFlagCvec) != 0;
    info_.blob_pool_off = hdr_.blob_pool_off;
    info_.blob_pool_bytes = hdr_.blob_pool_bytes;
    return true;
}

bool Decoder::hash(std::string& error) {
    if (!expect_.check_payload_hash) return true;
    uint64_t h = kWireHashSeed;
    std::vector<uint8_t> stage(kWireChunkBytes);
    for (uint64_t at = 0; at < payload_bytes_; at += kWireChunkBytes) {
        const size_t n = (size_t) std::min<uint64_t>(kWireChunkBytes, payload_bytes_ - at);
        if (const uint8_t* span = src_.span(at, n)) {
            h = fnv1a64(span, n, h);
            continue;
        }
        if (!src_.read(stage.data(), at, n)) return fail(error, "the source refused a read");
        h = fnv1a64(stage.data(), n, h);
    }
    if (h != expect_.payload_hash) {
        error = "conversation wire: payload hash " + hex16(h) + " != " + hex16(expect_.payload_hash) +
                " (the bytes here are not the bytes that were written)";
        return false;
    }
    info_.payload_hash = h;
    return true;
}

bool Decoder::blobref(Cursor& c, ConversationBuffer& b, std::string& error) {
    uint64_t bytes = 0, offset = 0;
    if (!c.u64(bytes) || !c.u64(offset)) return false;
    if (!bytes) {
        if (offset != 0) return fail(error, "a zero-length blob must be referenced as {0,0}");
        return true;
    }
    ++blobs_;
    if (offset < pool_begin_ || offset % kWireBlobAlign)
        return fail(error, "a blob reference is not on a 64-byte boundary inside the pool");
    if (offset > pool_end_ || bytes > pool_end_ - offset)
        return fail(error, "a blob reference runs past the end of the pool");
    // The allocation is bounded by the payload: the reference above has to land inside the pool, so a corrupt
    // `bytes` cannot make this process allocate more than the payload itself is.
    b.resize((size_t) bytes);
    // Chunked so no single source read exceeds `kWireChunkBytes`, and `visit` hands over one contiguous run
    // per SEGMENT inside each chunk, so the copy lands in the buffer's own memory: a 20 MiB K/V blob is
    // filled by twenty-odd <=1 MiB copies straight from the mapping or straight through `pread`.  No staging
    // buffer of any size exists on this path.
    bool ok = true;
    for (uint64_t at = 0; at < bytes && ok; at += kWireChunkBytes) {
        const size_t n = (size_t) std::min<uint64_t>(kWireChunkBytes, bytes - at);
        ok = b.visit((size_t) at, n, [&](uint8_t* p, size_t cnt, size_t abs) {
            if (const uint8_t* span = src_.span(offset + abs, cnt)) {
                std::memcpy(p, span, cnt);
                return true;
            }
            return src_.read(p, offset + abs, cnt);
        });
    }
    if (!ok) fail(error, "the source refused a blob read");
    return ok;
}

bool Decoder::blobref(Cursor& c, std::vector<uint8_t>& v, std::string& error) {
    uint64_t bytes = 0, offset = 0;
    if (!c.u64(bytes) || !c.u64(offset)) return false;
    if (!bytes) {
        if (offset != 0) return fail(error, "a zero-length blob must be referenced as {0,0}");
        return true;
    }
    ++blobs_;
    if (offset < pool_begin_ || offset % kWireBlobAlign)
        return fail(error, "a blob reference is not on a 64-byte boundary inside the pool");
    if (offset > pool_end_ || bytes > pool_end_ - offset)
        return fail(error, "a blob reference runs past the end of the pool");
    v.resize((size_t) bytes);
    // A checkpoint's running state is contiguous, so it is read in bounded pieces rather than one big read -
    // the same streaming rule the segmented case follows.
    bool ok = true;
    for (uint64_t at = 0; at < bytes && ok; at += kWireChunkBytes) {
        const size_t n = (size_t) std::min<uint64_t>(kWireChunkBytes, bytes - at);
        if (const uint8_t* span = src_.span(offset + at, n)) {
            std::memcpy(v.data() + at, span, n);
            continue;
        }
        ok = src_.read(v.data() + at, offset + at, n);
    }
    if (!ok) fail(error, "the source refused a blob read");
    return ok;
}

bool Decoder::kv(Cursor& c, ConversationKv& k, std::string& error) {
    uint32_t fmt = 0;
    if (!c.u32(fmt)) return false;
    if (!c.zeros(4)) return false;
    if (!c.i64(k.cells) || !c.i64(k.heads) || !c.i64(k.head_dim) || !c.i64(k.page_size) ||
        !c.i64(k.pooled_rows) || !c.i64(k.idx_dim))
        return false;
    k.format = (int) (int32_t) fmt;
    if (!blobref(c, k.k, error)) return false;
    if (!blobref(c, k.v, error)) return false;
    if (!blobref(c, k.k_scale, error)) return false;
    if (!blobref(c, k.v_scale, error)) return false;
    if (!blobref(c, k.pooled, error)) return false;
    return true;
}

bool Decoder::checkpoint(Cursor& c, ConversationCheckpoint& cp, int depth, std::string& error) {
    if (depth > expect_.max_checkpoint_depth) {
        error = "conversation wire: stage_parts nested " + dec((uint64_t) depth + 1) + " deep, past the bound " +
                dec((uint64_t) expect_.max_checkpoint_depth) +
                " (a corrupt payload cannot make this process allocate)";
        return false;
    }
    uint32_t kind = 0, rflags = 0;
    uint64_t rbytes = 0;
    if (!c.u32(kind) || !c.u32(rflags) || !c.u64(rbytes)) return false;
    if (kind != kWireRecordCheckpoint) {
        error = "conversation wire: expected a CHECKPOINT record, found kind 0x" + hex16(kind).substr(8);
        return false;
    }
    if (rflags != 0) return fail(error, "a record's flags field is not zero");
    if (rbytes > c.left()) return fail(error, "a record declares more bytes than the payload has");
    const uint64_t body_end = c.pos() + rbytes;

    int64_t n_ids = 0;
    if (!c.i64(n_ids)) return false;
    if (!checked_count(n_ids, std::numeric_limits<int64_t>::max() / 8, "n_ids", body_end - c.pos(), 4, error))
        return false;
    cp.ids.resize((size_t) n_ids);
    for (int64_t i = 0; i < n_ids; i += 2048) {
        const int64_t n = std::min<int64_t>(2048, n_ids - i);
        uint8_t buf[2048 * 4];
        if (!c.take(buf, (size_t) n * 4)) return false;
        for (int64_t j = 0; j < n; ++j)
            cp.ids[(size_t) (i + j)] = (int32_t) get_u32(buf + (size_t) j * 4);
    }
    if (ids_padding((uint64_t) n_ids) && !c.zeros(4)) return false;

    int64_t n_imgs = 0;
    if (!c.i64(n_imgs)) return false;
    if (!checked_count(n_imgs, kWireMaxImages, "n_imgs", body_end - c.pos(), 16, error)) return false;
    cp.imgs.resize((size_t) n_imgs);
    for (int64_t i = 0; i < n_imgs; ++i)
        if (!c.i64(cp.imgs[(size_t) i].start) || !c.u64(cp.imgs[(size_t) i].hash)) return false;

    if (!blobref(c, cp.gdn, error)) return false;
    if (!blobref(c, cp.ple, error)) return false;
    if (!blobref(c, cp.tails, error)) return false;
    if (!blobref(c, cp.dead, error)) return false;
    if (!blobref(c, cp.block_pos, error)) return false;
    if (!c.u64(cp.used)) return false;

    int64_t n_parts = 0;
    if (!c.i64(n_parts)) return false;
    if (!checked_count(n_parts, kWireMaxStageParts, "checkpoint stage_parts", body_end - c.pos(),
                       kWireRecordHeaderBytes, error))
        return false;
    cp.stage_parts.resize((size_t) n_parts);
    for (int64_t i = 0; i < n_parts; ++i)
        if (!checkpoint(c, cp.stage_parts[(size_t) i], depth + 1, error)) return false;
    if (c.pos() != body_end)
        return fail(error, "a CHECKPOINT record's body does not end where its header said");
    return true;
}

bool Decoder::stage(Cursor& c, ConversationStageSnapshot& s, std::string& error) {
    uint32_t kind = 0, rflags = 0;
    uint64_t rbytes = 0;
    if (!c.u32(kind) || !c.u32(rflags) || !c.u64(rbytes)) return false;
    if (kind != kWireRecordStage) {
        error = "conversation wire: expected a STAGE record, found kind 0x" + hex16(kind).substr(8);
        return false;
    }
    if (rflags != 0) return fail(error, "a record's flags field is not zero");
    if (rbytes > c.left()) return fail(error, "a record declares more bytes than the payload has");
    const uint64_t body_end = c.pos() + rbytes;
    if (!c.i64(s.layer_lo) || !c.i64(s.layer_hi)) return false;
    if (!checkpoint(c, s.state, 0, error)) return false;
    int64_t n_kv = 0;
    if (!c.i64(n_kv)) return false;
    if (!checked_count(n_kv, kWireMaxKvRecords, "stage n_kv", body_end - c.pos(),
                       kWireRecordHeaderBytes + kKvBodyBytes, error))
        return false;
    s.kv.resize((size_t) n_kv);
    for (int64_t i = 0; i < n_kv; ++i) {
        uint32_t k2 = 0, f2 = 0;
        uint64_t b2 = 0;
        if (!c.u32(k2) || !c.u32(f2) || !c.u64(b2)) return false;
        if (k2 != kWireRecordKv || f2 != 0 || b2 != kKvBodyBytes)
            return fail(error, "a stage's K/V array does not hold plain KV records");
        if (!kv(c, s.kv[(size_t) i], error)) return false;
    }
    if (c.pos() != body_end) return fail(error, "a STAGE record's body does not end where its header said");
    return true;
}

bool Decoder::run(std::string& error) {
    if (!header(error)) return false;
    if (!hash(error)) return false;
    Cursor c(src_, kWireHeaderRecordBytes, pool_begin_, error);
    for (int i = 0; i < 18; ++i) out_.geometry[(size_t) i] = (int64_t) hdr_.geometry[i];
    out_.layer_lo = (int64_t) hdr_.layer_lo;
    out_.layer_hi = (int64_t) hdr_.layer_hi;
    out_.cvec = info_.cvec;
    out_.owner = kNoOwner;  // deviation 1: the receiver claims it at put(), never from the wire

    out_.kv.resize((size_t) hdr_.n_kv);
    for (int64_t i = 0; i < hdr_.n_kv; ++i) {
        uint32_t kind = 0, f = 0;
        uint64_t b = 0;
        if (!c.u32(kind) || !c.u32(f) || !c.u64(b)) return false;
        if (kind != kWireRecordKv || f != 0 || b != kKvBodyBytes) {
            error = "conversation wire: the KV array does not hold plain KV records (kind 0x" +
                    hex16(kind).substr(8) + ", flags " + dec(f) + ", " + dec(b) + " body bytes)";
            return false;
        }
        if (!kv(c, out_.kv[(size_t) i], error)) return false;
    }
    if (!checkpoint(c, out_.live, 0, error)) return false;
    out_.checkpoints.resize((size_t) hdr_.n_checkpoints);
    for (int64_t i = 0; i < hdr_.n_checkpoints; ++i)
        if (!checkpoint(c, out_.checkpoints[(size_t) i], 0, error)) return false;
    out_.stage_parts.resize((size_t) hdr_.n_stage_parts);
    for (int64_t i = 0; i < hdr_.n_stage_parts; ++i)
        if (!stage(c, out_.stage_parts[(size_t) i], error)) return false;

    uint32_t kind = 0, f = 0;
    uint64_t b = 0;
    if (!c.u32(kind) || !c.u32(f) || !c.u64(b)) return fail(error, "the record stream has no END marker");
    if (kind != kWireRecordEnd || f != 0 || b != 0) return fail(error, "the END marker is malformed");
    if (c.pos() > pool_begin_)
        return fail(error, "the record stream runs past the start of the blob pool");
    // §2.3: the pool starts on a 64-byte boundary, so up to 63 zero bytes separate the END marker from it.
    // They are part of the payload and must be defined bytes, not whatever the slot held before.
    if (!c.zeros(pool_begin_ - c.pos())) return false;
    if (hdr_.live_tokens != (int64_t) out_.live.ids.size())
        return fail(error, "live_tokens disagrees with the live checkpoint's id count");
    // The other declared counts cannot disagree with what was parsed - the vectors are sized from them and
    // then filled exactly that many times - so there is no check here.  What CAN go wrong is the record
    // stream not ending where the pool begins, and the END marker's own shape, both checked above.

    info_.blobs = blobs_;
    target_ = std::move(out_);
    return true;
}

} // namespace

// ---------------------------------------------------------------- the sinks

bool VectorWireSink::write(const uint8_t* data, size_t bytes) {
    if (!bytes) return true;
    if (data == nullptr) return false;
    bytes_.insert(bytes_.end(), data, data + bytes);
    return true;
}

SlotWireSink::SlotWireSink(int fd, uint64_t offset, uint64_t capacity)
    : fd_(fd), at_(offset), map_(nullptr), pos_(0), capacity_(capacity) {}
SlotWireSink::SlotWireSink(uint8_t* mapping, uint64_t capacity)
    : fd_(-1), at_(0), map_(mapping), pos_(0), capacity_(capacity) {}

bool SlotWireSink::write(const uint8_t* data, size_t bytes) {
    if (!bytes) return true;
    if (pos_ > capacity_ || bytes > capacity_ - pos_) {
        error_ = "conversation wire: the payload needs " + dec(pos_ + bytes) + " bytes but the slot holds " +
                 dec(capacity_) + " (claim a larger tier)";
        return false;
    }
    if (map_) {
        std::memcpy(map_ + pos_, data, bytes);
        pos_ += bytes;
        return true;
    }
    size_t left = bytes;
    const uint8_t* p = data;
    while (left) {
        const ssize_t n = ::pwrite(fd_, p, left, (off_t) (at_ + pos_));
        if (n < 0) {
            if (errno == EINTR) continue;
            error_ = std::string("conversation wire: pwrite into the slot failed: ") + std::strerror(errno);
            return false;
        }
        if (n == 0) {
            error_ = "conversation wire: pwrite into the slot made no progress";
            return false;
        }
        p += (size_t) n;
        pos_ += (size_t) n;
        left -= (size_t) n;
    }
    return true;
}

bool SpanWireSource::read(uint8_t* dst, uint64_t offset, size_t bytes) const {
    if (p_ == nullptr || offset > n_ || bytes > n_ - offset) return false;
    std::memcpy(dst, p_ + offset, bytes);
    return true;
}
const uint8_t* SpanWireSource::span(uint64_t offset, size_t bytes) const {
    if (p_ == nullptr || offset > n_ || bytes > n_ - offset) return nullptr;
    return p_ + offset;
}

bool FileWireSource::read(uint8_t* dst, uint64_t offset, size_t bytes) const {
    if (offset > n_ || bytes > n_ - offset) return false;
    size_t left = bytes;
    while (left) {
        const ssize_t n = ::pread(fd_, dst, left, (off_t) (at_ + offset + (bytes - left)));
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        if (n == 0) return false;
        dst += (size_t) n;
        left -= (size_t) n;
    }
    return true;
}

// ---------------------------------------------------------------- the public entry points

bool conversation_wire_plan(const SavedConversation& image, WirePlan& plan, std::string& error) {
    return conversation_wire_plan(image, kWireMaxCheckpointDepth, plan, error);
}

bool conversation_wire_plan(const SavedConversation& image, int max_checkpoint_depth, WirePlan& plan,
                            std::string& error) {
    plan = WirePlan{};
    if (!image.live.stage_parts.empty())
        return fail(error, "a parked image must keep its stage state in stage_parts, not in live");
    Emitter e(image, WireIdentity{}, nullptr, max_checkpoint_depth < 1 ? 1 : max_checkpoint_depth);
    WireEncoded enc{};
    if (!e.run(plan, enc)) {
        error = e.error();
        return false;
    }
    return true;
}

bool conversation_wire_encode(const SavedConversation& image, WireSink& sink, const WireIdentity& identity,
                              WireEncoded& out, std::string& error) {
    WirePlan plan{};
    return conversation_wire_encode(image, sink, identity, kWireMaxCheckpointDepth, plan, out, error);
}

bool conversation_wire_encode(const SavedConversation& image, WireSink& sink, const WireIdentity& identity,
                              int max_checkpoint_depth, WirePlan& plan, WireEncoded& out, std::string& error) {
    out = WireEncoded{};
    plan = WirePlan{};
    if (!image.live.stage_parts.empty())
        return fail(error, "a parked image must keep its stage state in stage_parts, not in live");
    Emitter e(image, identity, &sink, max_checkpoint_depth < 1 ? 1 : max_checkpoint_depth);
    if (!e.run(plan, out)) {
        error = e.error();
        return false;
    }
    if (out.bytes != plan.total_bytes) return fail(error, "internal: the encoder wrote the wrong byte count");
    return true;
}

bool conversation_wire_decode(const WireSource& source, SavedConversation& image, WireDecoded& info,
                              const WireExpect& expect, std::string& error) {
    info = WireDecoded{};
    Decoder d(source, image, info, expect);
    return d.run(error);
}

bool conversation_wire_decode(const uint8_t* data, uint64_t bytes, SavedConversation& image,
                              WireDecoded& info, const WireExpect& expect, std::string& error) {
    SpanWireSource src(data, bytes);
    return conversation_wire_decode(src, image, info, expect, error);
}

bool conversation_wire_peek(const uint8_t* data, uint64_t bytes, WireHeaderRecord& header, std::string& error) {
    if (bytes < kWireHeaderRecordBytes) return fail(error, "the payload is shorter than the header record");
    const uint8_t* p = data;
    header.magic = get_u32(p + 0);
    header.fmt_version = get_u16(p + 4);
    header.flags = get_u16(p + 6);
    for (int i = 0; i < 18; ++i) header.geometry[i] = get_u64(p + 8 + (size_t) i * 8);
    header.layer_lo = get_u64(p + 152);
    header.layer_hi = get_u64(p + 160);
    header.n_kv = (int64_t) get_u64(p + 168);
    header.n_checkpoints = (int64_t) get_u64(p + 176);
    header.n_stage_parts = (int64_t) get_u64(p + 184);
    header.live_tokens = (int64_t) get_u64(p + 192);
    header.blob_pool_off = get_u64(p + 200);
    header.blob_pool_bytes = get_u64(p + 208);
    header.geom_hash = get_u64(p + 216);
    header.pack_hash = get_u64(p + 224);
    header.payload_bytes = get_u64(p + 232);
    header.payload_hash = get_u64(p + 240);
    header.reserved = get_u64(p + 248);
    if (header.magic != kWireMagic) return fail(error, "not a hand-off payload");
    if (header.fmt_version != kWireFormatVersion) return fail(error, "payload format version mismatch");
    if (header.flags & (uint16_t) ~kWireFlagKnown) return fail(error, "unknown header flag bits");
    if (header.payload_bytes < kWireHeaderRecordBytes) return fail(error, "the declared payload size is absurd");
    if (header.blob_pool_off < kWireHeaderRecordBytes || header.blob_pool_off % kWireBlobAlign)
        return fail(error, "the blob pool does not start on a 64-byte boundary");
    if (header.blob_pool_bytes > header.payload_bytes - header.blob_pool_off)
        return fail(error, "the blob pool runs past the end of the payload");
    return true;
}

uint64_t conversation_wire_payload_hash(const uint8_t* data, uint64_t bytes) {
    return fnv1a64(data, bytes, kWireHashSeed);
}

std::string conversation_wire_describe(const WireHeaderRecord& header) {
    std::string s = "geometry [";
    for (int i = 0; i < 18; ++i) {
        if (i) s += ",";
        s += dec((int64_t) header.geometry[i]);
    }
    s += "] layers [" + dec(header.layer_lo) + "," + dec(header.layer_hi) + ") kv " +
        dec((uint64_t) header.n_kv) + " checkpoints " + dec((uint64_t) header.n_checkpoints) + " stages " +
        dec((uint64_t) header.n_stage_parts) + " tokens " + dec((uint64_t) header.live_tokens) + " geom " +
        hex16(header.geom_hash) + " pack " + hex16(header.pack_hash);
    return s;
}

} // namespace strata::core
