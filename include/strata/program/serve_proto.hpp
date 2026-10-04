// include/strata/program/serve_proto.hpp - the `--serve` WIRE FORMAT, in one place (stage 3, S3.1c).
//
// `strata --serve` has spoken a line-oriented protocol since plan v0.3 P8: the client writes
// `GEN <max_new> [k=v ...] <id,id,...>` and the engine answers with `RESUME`, `PP`, `REUSED`, one `T` per
// token and a `DONE`.  That protocol has no request identity at all - `T 42` means "token 42 of the one
// request running" - so it cannot carry two requests in flight.  Stage 3 adds identity (§6 of
// docs/STAGE3-CONCURRENCY.md), and this header is the ONLY place either direction of the wire is
// formatted or parsed: a tagged line and an untagged line cannot drift, because they are the same call
// with one flag.
//
// WHY THIS IS A HEADER AND NOT A .cpp: the same rule as `prefill_loan.hpp` - it must build and be
// unit-tested on a CPU with no CUDA and no model (`src/program/serve_proto_test.cpp`), and the engine
// includes it from `src/program/generate.cpp`.  No I/O: every function RETURNS the line (without the
// '\n'); the caller writes it to stdout.  That is what makes the golden test in
// `src/program/serve_proto_test.cpp` possible at all.
//
// THE COMPATIBILITY RULE (docs/STAGE3-CONCURRENCY.md §6.4).  The protocol is versioned by one `READY`
// token and the default is OFF:
//
//   * `--serve-slots 0` (the default) => the engine emits EXACTLY the bytes 0.1.30 emitted, and accepts
//     exactly the lines 0.1.30 accepted.  `tagged=false` below.
//   * `--serve-slots N>=2` => per-request lines carry a trailing ` #<id>`, `READY` carries `slots=N`, and
//     the engine emits `SLOT` transition lines.  `tagged=true`.
//   * A request line is parsed the same way in both modes: the id is OPTIONAL, so an old client keeps
//     working against a new engine.
//
// The optional-id rule, stated exactly, because it is the one place a "clever" parser would break an old
// client: the token after `GEN`/`GENI` is a REQUEST ID only when the token after THAT is also a bare
// non-negative integer AND a third token exists.  `GEN 7 32` is therefore still 0.1.30's
// "max_new=7, ids=[32]" and not "id=7, max_new=32, no ids" - strictly more compatible than the rule in
// §6.1 of the design, and the reason is that a two-token request line is legal today.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdarg>
#include <cstdlib>
#include <cstring>
#include <charconv>
#include <sstream>
#include <string>
#include <vector>

namespace strata::program::serve_proto {

/// "this request has no id" - an untagged line, or a request from a client that predates stage 3.
constexpr int64_t kNoId = -1;

// ------------------------------------------------------------------ formatting helpers -----------------
namespace detail {
/// snprintf into a std::string.  Deliberately snprintf and not an ostream: the engine's 0.1.30 lines are
/// printf format strings, and the bit-exactness bar is "the same BYTES", which is only provable if the
/// same conversion is used on both sides of the golden test.
inline std::string f(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    const int n = std::vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if (n < 0) return {};
    if ((size_t) n < sizeof buf) return std::string(buf, (size_t) n);
    std::string out((size_t) n, '\0');
    va_start(ap, fmt);
    std::vsnprintf(&out[0], (size_t) n + 1, fmt, ap);
    va_end(ap);
    return out;
}
}  // namespace detail

/// The request-id tag, appended AFTER a line's own fields so every positional parser (server.py's
/// `_parse_done` indexes f[1..10]) keeps working when it ignores trailing tokens.
inline std::string tag(int64_t id) { return id == kNoId ? std::string() : " #" + std::to_string(id); }

// ---------------------------------------------------------------------- the engine -> client lines -----
//
// Every one of these is 0.1.30's printf format string, verbatim, plus the optional tag.  The golden test
// compares them against copies of the 0.1.30 `std::printf` calls.

class Out {
public:
    /// `tagged` = this engine speaks the stage-3 wire (`--serve-slots >= 2`).  With false, every method
    /// below produces the byte-for-byte 0.1.30 line.
    explicit Out(bool tagged) : tagged_(tagged) {}

    bool tagged() const { return tagged_; }

    /// READY <max_context> stop [slots=<N>]     ("stop": this engine honours STOP; "slots=": it also
    /// names requests.  A server enables multi-slot ONLY when it sees `slots=`.)
    std::string ready(int64_t max_context, int slots) const {
        return slots > 0 ? detail::f("READY %lld stop slots=%d", (long long) max_context, slots)
                         : detail::f("READY %lld stop", (long long) max_context);
    }

    /// INFO ... - process-wide, never tagged (§6.1).  The engine builds its own body; this only appends
    /// the stage-3 keys so the pair cannot drift between the two sides.
    std::string info_slots(int slots, int64_t active) const {
        return detail::f(" slots=%d slots_active=%lld concurrency=1", slots, (long long) active);
    }

    /// RESUME <n>          - this many prompt tokens are reused, sent before the read starts
    std::string resume(int64_t n, int64_t id = kNoId) const {
        return detail::f("RESUME %lld", (long long) n) + tag_for(id);
    }

    /// PP <pos> <prompt_tokens> <ms> <fresh tokens/s>   - one per prompt chunk, also a heartbeat
    std::string pp(int64_t pos, int64_t total, double ms, double tok_s, int64_t id = kNoId) const {
        return detail::f("PP %lld %lld %.0f %.1f", (long long) pos, (long long) total, ms, tok_s) + tag_for(id);
    }

    /// REUSED <n>          - the prompt is read; the first window comes next
    std::string reused(int64_t n, int64_t id = kNoId) const {
        return detail::f("REUSED %lld", (long long) n) + tag_for(id);
    }

    /// T <token_id>        - one per emitted token
    std::string token(int32_t t, int64_t id = kNoId) const {
        return detail::f("T %d", (int) t) + tag_for(id);
    }

    /// DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered>
    ///      <reused> <hits> <lookups>
    /// The field list is UNCHANGED from 0.1.30; the tag goes after it.
    std::string done(int64_t generated, int64_t prompt, double prompt_ms, double decode_ms,
                     const std::string& finish, int64_t drafts_accepted, int64_t drafts_offered,
                     int64_t reused, int64_t hits, int64_t lookups, int64_t id = kNoId) const {
        return detail::f("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld",
                         (long long) generated, (long long) prompt, prompt_ms, decode_ms, finish.c_str(),
                         (long long) drafts_accepted, (long long) drafts_offered, (long long) reused,
                         (long long) hits, (long long) lookups) + tag_for(id);
    }

    /// ERR <message>       - tagged when the message belongs to a named request, untagged when it does
    ///   not (a line the parser could not attribute to anything is still 0.1.30's bare ERR).
    std::string err(const std::string& msg, int64_t id = kNoId) const { return "ERR " + msg + tag_for(id); }

    /// SLOT <req_id> <state> <ctx_used> <ctx_cap> <prompt_tokens> <generated> <parked_bytes>
    /// Emitted on every state transition so serve/server.py can answer /slots without polling.
    /// `state` is one of the names in slot.hpp's state_name().  Always tagged: it only exists in the
    /// stage-3 wire at all, and an id-less SLOT line would be meaningless.
    std::string slot(int64_t id, const std::string& state, int64_t ctx_used, int64_t ctx_cap,
                     int64_t prompt_tokens, int64_t generated, int64_t parked_bytes) const {
        return detail::f("SLOT %lld %s %lld %lld %lld %lld %lld", (long long) id, state.c_str(),
                         (long long) ctx_used, (long long) ctx_cap, (long long) prompt_tokens,
                         (long long) generated, (long long) parked_bytes);
    }

private:
    std::string tag_for(int64_t id) const { return tagged_ ? tag(id) : std::string(); }
    bool tagged_;
};

// ---------------------------------------------------------------------- client -> engine lines -------

enum class Kind : uint8_t { other, gen, geni, stop, quit };

/// The engine's own defaults for the two per-request tuning keys; the caller passes them from its
/// Options so a request that does not name them keeps today's behaviour.
struct Defaults {
    double pcie_frac = 0.0;
    double spec_min_p = 0.0;
};

/// One parsed request line.  `error` non-empty means the line was rejected: the caller prints
/// `ERR <error>` (untagged, exactly as 0.1.30 does for a line it could not attribute to a request) and
/// reads the next line.  `id` is kNoId for every request from a client that does not speak stage 3.
struct Request {
    Kind kind = Kind::other;
    int64_t id = kNoId;
    int64_t max_new = 0;
    std::string emb_path;                 // GENI: the strata-vision embeddings file
    std::vector<int64_t> ids;             // the prompt
    // sampling / tuning keys, with 0.1.30's defaults: greedy, no penalties, cvec on
    float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
    float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
    int top_k = 20, penalty_last_n = 0, cvec = 1;
    unsigned long long seed = 0;
    double pcie_frac = 0.0, spec_min_p = 0.0;
    std::string error;
};

/// 0.1.30's ERR bodies, kept as constants so the golden test can pin them.
inline const std::string& err_expected() {
    static const std::string s = "expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>";
    return s;
}

/// Is `t` a bare non-negative integer (digits only)?  This is what the optional-id rule keys on, and it
/// deliberately does NOT accept a sign: `GEN -5 32 ...` is 0.1.30's malformed request, not an id.
inline bool is_bare_uint(const std::string& t, long long& v) {
    if (t.empty()) return false;
    for (char c : t) if (c < '0' || c > '9') return false;
    const auto r = std::from_chars(t.data(), t.data() + t.size(), v);
    return r.ec == std::errc{} && r.ptr == t.data() + t.size();
}

/// The comma/space separated prompt list.  This is 0.1.30's `parse_i64_list` (generate.cpp:614) moved
/// here verbatim - same splitter, same range check, same error text - so the id-less path cannot drift.
inline bool parse_id_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::string text(s);
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}

/// Parse one stdin line.  Accepts BOTH forms:
///
///   GEN  <max_new> [k=v ...] <ids>              (0.1.30)
///   GEN  <req_id> <max_new> [k=v ...] <ids>     (stage 3)
///   GENI ... same, with the embeddings file as the first token without an `=`
///   STOP            -> cancel the newest uncancelled request (0.1.30's meaning)
///   STOP <req_id>   -> cancel that request only
///   QUIT
///
/// `allow_id` MUST be false unless the engine was started with `--serve-slots >= 2`.  With false the
/// parser is 0.1.30's parser token for token - no request line can be read differently from the way
/// 0.1.30 read it - which is what turns the §6.4 row "old server / new engine" into a proof rather than a
/// hope.  With true, the id rule at the top of this file applies; the one shape whose meaning changes is
/// `GEN <a> <b> <c>` with all three bare integers, which 0.1.30 read as max_new=a and ids={b,c} and stage 3
/// reads as id=a, max_new=b, ids={c}.  serve/server.py only ever sends that shape after the `slots=`
/// handshake, so no existing client can hit it.
///
/// Unknown keys are skipped, as today: the ids start at the first token without an `=`.
inline Request parse_request(const std::string& line, const Defaults& d, bool allow_id = false) {
    Request r;
    r.pcie_frac = d.pcie_frac;
    r.spec_min_p = d.spec_min_p;
    if (line == "QUIT") { r.kind = Kind::quit; return r; }
    // STOP, and (only in id mode) STOP <id>.  With allow_id false the line must be EXACTLY "STOP", which
    // is 0.1.30's test (`if (l == "STOP")` at generate.cpp:4247): anything else falls through to the GEN
    // check and gets 0.1.30's "ERR expected: ..." - a slots=0 engine cannot be made to accept a line it
    // used to reject.
    if (line == "STOP" || (allow_id && line.rfind("STOP ", 0) == 0)) {
        r.kind = Kind::stop;
        std::string rest;
        if (line.size() > 5) rest = line.substr(5);
        // trim: "STOP " with nothing after it is a bare STOP, as a client that always appends a space
        // would otherwise send
        size_t b = rest.find_first_not_of(" \t");
        if (b != std::string::npos) {
            size_t e = rest.find_last_not_of(" \t");
            rest = rest.substr(b, e - b + 1);
        } else rest.clear();
        if (!rest.empty()) {
            long long v = 0;
            if (!is_bare_uint(rest, v)) { r.error = "bad request: STOP needs a request id"; return r; }
            r.id = v;
        }
        return r;
    }
    const bool geni = line.rfind("GENI ", 0) == 0;
    if (!geni && line.rfind("GEN ", 0) != 0) { r.kind = Kind::other; r.error = err_expected(); return r; }
    r.kind = geni ? Kind::geni : Kind::gen;
    const char* body = line.c_str() + (geni ? 5 : 4);

    // ---- the optional id, per the rule in the header comment ------------------
    const char* rest = body;
    if (allow_id) {
        const char* p = body;
        std::vector<std::string> tok;
        for (;;) {
            while (*p == ' ') ++p;
            const char* s = p;
            while (*p != '\0' && *p != ' ') ++p;
            if (p == s) break;
            tok.emplace_back(s, (size_t) (p - s));
        }
        long long a = 0, b = 0;
        if (tok.size() >= 3 && is_bare_uint(tok[0], a) && is_bare_uint(tok[1], b)) {
            r.id = a;
            r.max_new = b;
            // point `rest` at the third token, where the keys (or the ids) start
            const char* q = body;
            for (int k = 0; k < 2; ++k) {
                while (*q == ' ') ++q;
                while (*q != '\0' && *q != ' ') ++q;
            }
            rest = q;
        }
    }
    if (r.id == kNoId) {
        char* endp = nullptr;
        r.max_new = std::strtoll(body, &endp, 10);
        rest = endp;
    }

    // ---- the optional k=v keys ------------------------------------------------
    char* endp = nullptr;
    if (rest != nullptr) {
        endp = const_cast<char*>(rest);
        for (;;) {
            while (*endp == ' ') ++endp;
            const char* start = endp;
            while (*endp != '\0' && *endp != ' ') ++endp;
            if (endp == start) break;
            const std::string t(start, (size_t) (endp - start));
            const size_t eq = t.find('=');
            if (eq == std::string::npos) { endp = const_cast<char*>(start); break; }
            const std::string key = t.substr(0, eq);
            const float fv = std::strtof(t.c_str() + eq + 1, nullptr);
            if (key == "cvec") r.cvec = std::atoi(t.c_str() + eq + 1);
            else if (key == "temperature") r.temperature = fv;
            else if (key == "top_p") r.top_p = fv;
            else if (key == "top_k") r.top_k = std::atoi(t.c_str() + eq + 1);
            else if (key == "min_p") r.min_p = fv;
            else if (key == "penalty_last_n") r.penalty_last_n = std::atoi(t.c_str() + eq + 1);
            else if (key == "penalty_repeat") r.penalty_repeat = fv;
            else if (key == "penalty_freq") r.penalty_freq = fv;
            else if (key == "penalty_present") r.penalty_present = fv;
            else if (key == "seed") r.seed = std::strtoull(t.c_str() + eq + 1, nullptr, 10);
            else if (key == "pcie_frac") r.pcie_frac = (double) fv < 0.0 ? 0.0 : ((double) fv > 1.0 ? 1.0 : (double) fv);
            else if (key == "spec_min_p") r.spec_min_p = (double) fv < 0.0 ? 0.0 : ((double) fv > 1.0 ? 1.0 : (double) fv);
            // unknown keys are skipped: the ids start at the first token without '='
        }
    }
    if (geni && endp != nullptr) {
        while (*endp == ' ') ++endp;
        char* gap = std::strchr(endp, ' ');
        // 0.1.30's rule, kept exactly: the file is the token up to the next space, and a GENI line with no
        // space after max_new has no file at all - it is a bad request, not a file named after the ids.
        if (gap != nullptr) { r.emb_path.assign(endp, (size_t) (gap - endp)); endp = gap; }
    }
    std::string pe;
    if (r.max_new < 1 || endp == nullptr || (geni && r.emb_path.empty()) ||
        !parse_id_list(endp, r.ids, pe)) {
        r.error = "bad request: " + (pe.empty() ? std::string("max_new") : pe);
        return r;
    }
    return r;
}

/// Format a request line (the client's side of the wire).  Kept here so the round-trip test and
/// serve/server.py agree on one spelling; `id == kNoId` produces 0.1.30's id-less line.
inline std::string gen_line(Kind kind, int64_t id, int64_t max_new, const std::string& keys,
                            const std::string& emb_path, const std::vector<int64_t>& ids) {
    std::string s = kind == Kind::geni ? "GENI" : "GEN";
    if (id != kNoId) s += " " + std::to_string(id);
    s += " " + std::to_string(max_new);
    s += keys;
    if (kind == Kind::geni) s += " " + emb_path;
    s += " ";
    for (size_t i = 0; i < ids.size(); ++i) { if (i) s += ","; s += std::to_string(ids[i]); }
    return s;
}

inline std::string stop_line(int64_t id) { return id == kNoId ? std::string("STOP") : "STOP " + std::to_string(id); }

/// Read the ` #<id>` tag off an engine line.  Returns the id (kNoId when absent) and, when `body` is
/// non-null, the line with the tag removed - so one parser serves both wire versions.
inline int64_t parse_tag(const std::string& line, std::string* body) {
    const size_t h = line.rfind(" #");
    if (h == std::string::npos) { if (body) *body = line; return kNoId; }
    long long v = 0;
    if (!is_bare_uint(line.substr(h + 2), v)) { if (body) *body = line; return kNoId; }
    if (body) *body = line.substr(0, h);
    return v;
}

}  // namespace strata::program::serve_proto
