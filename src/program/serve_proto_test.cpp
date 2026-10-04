// src/program/serve_proto_test.cpp - the `--serve` wire format (stage 3, S3.1c).  CPU only: no CUDA, no
// model, no engine, no request.  The acceptance bar this file exists for is §7.1 of
// docs/STAGE3-CONCURRENCY.md: with `--serve-slots 0` the engine's stdout for a given sequence of requests
// must be BYTE-IDENTICAL to 0.1.30's.  That is proven here by formatting the same inputs twice - once
// through a verbatim copy of 0.1.30's `std::printf` calls (the `old_*` functions below, lifted from
// src/program/generate.cpp) and once through serve_proto - and comparing bytes.  Not by assertion prose.
#include "strata/program/serve_proto.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (ok) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", what);
}

void check_eq(const std::string& got, const std::string& want, const char* what) {
    ++checks;
    if (got == want) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n  old: [%s]\n  new: [%s]\n", what, want.c_str(), got.c_str());
}

using namespace strata::program;

// ============================================================ the 0.1.30 formatter, verbatim ============
//
// These are the engine's own printf calls as of 0.1.30 (generate.cpp:4352 READY, :4705 RESUME, :4024 and
// :4751 PP, :4987 REUSED, :5090 T, :5264 DONE, and the ERR bodies at :4381/:4434/:4439/:4531/:4537).
// Copied, not re-derived: the point of the test is that the new formatter emits the SAME BYTES.
std::string old_ready(int64_t max_context) {
    char b[128]; std::snprintf(b, sizeof b, "READY %lld stop", (long long) max_context); return b;
}
std::string old_resume(int64_t n) {
    char b[128]; std::snprintf(b, sizeof b, "RESUME %lld", (long long) n); return b;
}
std::string old_pp(int64_t pos, int64_t total, double ms, double tok_s) {
    char b[256]; std::snprintf(b, sizeof b, "PP %lld %lld %.0f %.1f", (long long) pos, (long long) total, ms, tok_s);
    return b;
}
std::string old_reused(int64_t n) {
    char b[128]; std::snprintf(b, sizeof b, "REUSED %lld", (long long) n); return b;
}
std::string old_token(int32_t t) {
    char b[64]; std::snprintf(b, sizeof b, "T %d", (int) t); return b;
}
std::string old_done(int64_t generated, int64_t prompt, double prompt_ms, double decode_ms,
                     const std::string& finish, int64_t da, int64_t dof, int64_t re, int64_t h, int64_t l) {
    char b[512];
    std::snprintf(b, sizeof b, "DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld",
                  (long long) generated, (long long) prompt, prompt_ms, decode_ms, finish.c_str(),
                  (long long) da, (long long) dof, (long long) re, (long long) h, (long long) l);
    return b;
}
std::string old_err(const std::string& msg) { return "ERR " + msg; }

// ============================================================ golden: slots=0 is 0.1.30 ===============
void test_golden_untagged_output() {
    const serve_proto::Out out(false);
    check(!out.tagged(), "an untagged Out is what --serve-slots 0 builds");

    check_eq(out.ready(524288, 0), old_ready(524288), "READY, slots=0: byte-identical");
    check_eq(out.resume(8000), old_resume(8000), "RESUME: byte-identical");
    check_eq(out.reused(8000), old_reused(8000), "REUSED: byte-identical");
    check_eq(out.token(248045), old_token(248045), "T: byte-identical");
    check_eq(out.token(-1), old_token(-1), "T with a negative id (never emitted, but the same conversion)");

    // PP: the two call sites use the same format; %.0f and %.1f rounding is part of the bytes.
    const double cases_ms[] = {0.0, 0.4, 0.5, 1.0, 24.0, 33080.0, 999999.9};
    const double cases_rate[] = {0.0, 1.0 / 3.0, 490.0, 1100.0, 12345.678};
    for (double ms : cases_ms)
        for (double r : cases_rate)
            check_eq(out.pp(4096, 35957, ms, r), old_pp(4096, 35957, ms, r), "PP: byte-identical");

    // DONE: every field, including the odd ones a real request produces.
    struct D { int64_t gen, prompt; double pms, dms; const char* fin; int64_t da, dof, re, h, l; };
    const D ds[] = {
        {1, 12000, 4000, 10, "stop", 0, 0, 8000, 0, 0},
        {0, 12000, 0, 0, "stop", 0, 0, 12000, 0, 0},
        {233, 35957, 33080.4, 5566.7, "length", 410, 900, 3072, 80679, 94780},
        {7, 40, 12.5, 168.2, "stop", 6, 9, 0, 3, 4},
        {1, 1, 0.4, 0.1, "cancel", 0, 1, 0, 0, 1},
    };
    for (const D& d : ds)
        check_eq(out.done(d.gen, d.prompt, d.pms, d.dms, d.fin, d.da, d.dof, d.re, d.h, d.l),
                 old_done(d.gen, d.prompt, d.pms, d.dms, d.fin, d.da, d.dof, d.re, d.h, d.l),
                 "DONE: byte-identical");

    // ERR: the five bodies 0.1.30 can print.
    check_eq(out.err(serve_proto::err_expected(), serve_proto::kNoId),
             old_err("expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>"),
             "ERR expected: byte-identical");
    check_eq(out.err("bad request: max_new", serve_proto::kNoId), old_err("bad request: max_new"),
             "ERR bad request max_new: byte-identical");
    check_eq(out.err("bad request: token list was empty", serve_proto::kNoId),
             old_err("bad request: token list was empty"), "ERR bad request empty: byte-identical");
    check_eq(out.err("bad request: invalid token id: expected an integer in [0, 2147483647]", serve_proto::kNoId),
             old_err("bad request: invalid token id: expected an integer in [0, 2147483647]"),
             "ERR bad token id: byte-identical");
    check_eq(out.err("this engine was started without --vision", serve_proto::kNoId),
             old_err("this engine was started without --vision"), "ERR no vision: byte-identical");
    check_eq(out.err("prompt (35957 tokens) + max_new (4096) exceeds the context (524288)", serve_proto::kNoId),
             old_err("prompt (35957 tokens) + max_new (4096) exceeds the context (524288)"),
             "ERR context: byte-identical");
    check_eq(out.err("a token id is outside the vocabulary", serve_proto::kNoId),
             old_err("a token id is outside the vocabulary"), "ERR vocab: byte-identical");
    // An untagged Out NEVER grows a tag, even if the caller passes an id: the flag, not the caller,
    // decides the wire.
    check_eq(out.token(7, 123), old_token(7), "an untagged Out ignores the id - it cannot drift to tagged");
    check_eq(out.done(1, 2, 3.0, 4.0, "stop", 0, 0, 0, 0, 0, 9),
             old_done(1, 2, 3.0, 4.0, "stop", 0, 0, 0, 0, 0), "same for DONE");
}

// ============================================================ the tagged wire (slots>=2) ===============
void test_tagged_output() {
    const serve_proto::Out out(true);
    check_eq(out.ready(524288, 2), "READY 524288 stop slots=2", "READY carries slots=N only when N>0");
    check_eq(out.resume(8000, 7), "RESUME 8000 #7", "RESUME tag after the field");
    check_eq(out.pp(4096, 35957, 33080.0, 490.0, 7), "PP 4096 35957 33080 490.0 #7", "PP tag");
    check_eq(out.reused(8000, 7), "REUSED 8000 #7", "REUSED tag");
    check_eq(out.token(248045, 7), "T 248045 #7", "T tag");
    check_eq(out.err("verify: layer 31 never rang", 7), "ERR verify: layer 31 never rang #7", "ERR tag");
    check_eq(out.done(233, 35957, 33080.4, 5566.7, "length", 410, 900, 3072, 80679, 94780, 7),
             "DONE 233 35957 33080.4 5566.7 length 410 900 3072 80679 94780 #7",
             "DONE: the tag goes AFTER the 10 fields, so _parse_done's f[1..10] still indexes them");
    // An id-less line in a tagged session stays untagged: process-wide lines have no request to name.
    check_eq(out.token(5, serve_proto::kNoId), "T 5", "kNoId never prints a tag");
    check_eq(out.slot(7, "decoding", 1200, 524288, 35957, 233, 1073741824),
             "SLOT 7 decoding 1200 524288 35957 233 1073741824", "SLOT line, §6.2's field order");
    check_eq(out.info_slots(2, 1), " slots=2 slots_active=1 concurrency=1", "the INFO additions (§6.1)");
}

// ============================================================ request parsing ==========================
const serve_proto::Defaults kDef{0.35, 0.05};

void test_old_form_requests() {
    // Everything 0.1.30 accepted must parse the same way, with id == kNoId.
    auto r = serve_proto::parse_request("GEN 32 1,2,3", kDef);
    check(r.error.empty(), "GEN <max_new> <ids>");
    check(r.kind == serve_proto::Kind::gen, "kind gen");
    check(r.id == serve_proto::kNoId, "no id in the old form");
    check(r.max_new == 32, "max_new 32");
    check(r.ids.size() == 3 && r.ids[2] == 3, "three ids");
    check(r.pcie_frac == 0.35 && r.spec_min_p == 0.05,
          "the tuning keys default to the engine's");

    r = serve_proto::parse_request("GEN 32 temperature=0.7 top_p=0.9 top_k=40 min_p=0.1 "
                                   "penalty_last_n=64 penalty_repeat=1.1 penalty_freq=0.2 "
                                   "penalty_present=0.3 seed=1234 cvec=0 pcie_frac=0.5 spec_min_p=0.02 "
                                   "1,2,3", kDef);
    check(r.error.empty(), "GEN with every key");
    check(r.temperature > 0.69f && r.temperature < 0.71f, "temperature");
    check(r.top_p > 0.89f && r.top_p < 0.91f, "top_p");
    check(r.top_k == 40 && r.min_p > 0.09f && r.min_p < 0.11f, "top_k / min_p");
    check(r.penalty_last_n == 64 && r.penalty_repeat > 1.09f && r.penalty_freq > 0.19f &&
          r.penalty_present > 0.29f, "the penalties");
    check(r.seed == 1234, "seed");
    check(r.cvec == 0, "cvec=0");
    check(r.pcie_frac == 0.5 && r.spec_min_p > 0.0199 && r.spec_min_p < 0.0201, "per-request tuning keys");
    check(r.ids.size() == 3, "the ids still start at the first token without an =");

    r = serve_proto::parse_request("GEN 32 unknown_key=9 1,2", kDef);
    check(r.error.empty() && r.ids.size() == 2, "an unknown key is skipped, as today");

    r = serve_proto::parse_request("GENI 64 emb.sve 5,6,7", kDef);
    check(r.error.empty() && r.kind == serve_proto::Kind::geni, "GENI <max_new> <file> <ids>");
    check(r.emb_path == "emb.sve", "the embeddings file is the first token without an =");
    check(r.max_new == 64 && r.ids.size() == 3, "GENI max_new / ids");

    r = serve_proto::parse_request("GENI 64 temperature=1 emb.sve 5,6", kDef);
    check(r.error.empty() && r.emb_path == "emb.sve" && r.ids.size() == 2, "GENI takes the same keys (#75)");

    // The two-token shape the design's literal rule would have broken: `GEN 7 32` is max_new=7, ids=[32].
    r = serve_proto::parse_request("GEN 7 32", kDef);
    check(r.error.empty() && r.id == serve_proto::kNoId && r.max_new == 7 && r.ids.size() == 1 && r.ids[0] == 32,
          "GEN 7 32 stays 0.1.30's max_new=7 ids=[32] - the id needs a THIRD token");

    r = serve_proto::parse_request("GEN 32 1 2 3", kDef);
    check(r.ids.size() == 3, "a space-separated id list still parses (0.1.30's splitter accepts it)");

    // THE COMPATIBILITY HALF OF THE PARSER: with --serve-slots 0 (allow_id = false) EVERY line means what
    // 0.1.30 read it to mean, including the three-integer shape that would otherwise take an id.
    r = serve_proto::parse_request("GEN 7 32 1,2,3", kDef, false);
    check(r.error.empty() && r.id == serve_proto::kNoId && r.max_new == 7 && r.ids.size() == 4 && r.ids[0] == 32,
          "allow_id=false: GEN 7 32 1,2,3 is 0.1.30's max_new=7 ids={32,1,2,3}");
    r = serve_proto::parse_request("GENI 4 emb.sve 1,2", kDef, false);
    check(r.error.empty() && r.id == serve_proto::kNoId && r.max_new == 4 && r.emb_path == "emb.sve",
          "allow_id=false: GENI is unchanged too");
}

void test_new_form_requests() {
    auto r = serve_proto::parse_request("GEN 7 32 1,2,3", kDef, true);
    check(r.error.empty(), "GEN <id> <max_new> <ids>");
    check(r.id == 7 && r.max_new == 32 && r.ids.size() == 3, "id 7, max_new 32");

    r = serve_proto::parse_request("GEN 7 32 temperature=0.7 1,2,3", kDef, true);
    check(r.error.empty() && r.id == 7 && r.max_new == 32 && r.temperature > 0.69f, "id + keys + ids");

    r = serve_proto::parse_request("GENI 9 64 emb.sve 5,6", kDef, true);
    check(r.error.empty() && r.id == 9 && r.max_new == 64 && r.emb_path == "emb.sve" && r.ids.size() == 2,
          "GENI <id> <max_new> <file> <ids>");

    // A single-token id list is the one shape whose meaning depends on the mode (`GEN 7 32 100`: 0.1.30's
    // max_new=7 ids={32,100}, stage 3's id=7 max_new=32 ids={100}).  Pinned in BOTH modes so it is a
    // decision on the record, and see the comment on parse_request for why no client can trip over it.
    r = serve_proto::parse_request("GEN 7 32 100", kDef, true);
    check(r.error.empty() && r.id == 7 && r.max_new == 32 && r.ids.size() == 1 && r.ids[0] == 100,
          "GEN 7 32 100 in id mode: the id wins");
    r = serve_proto::parse_request("GEN 7 32 100", kDef, false);
    check(r.error.empty() && r.id == serve_proto::kNoId && r.max_new == 7 && r.ids.size() == 2,
          "the same line with --serve-slots 0: still 0.1.30's meaning");

    // ids of 0 are legal tokens; a request id of 0 is not special-cased away
    r = serve_proto::parse_request("GEN 0 5 0,1,2", kDef, true);
    check(r.error.empty() && r.id == 0 && r.max_new == 5 && r.ids[0] == 0, "id 0 and token id 0");

    // a key-bearing second token is never an id: `GEN temperature=0.7 1,2` stays id-less
    r = serve_proto::parse_request("GEN temperature=0.7 1,2", kDef, true);
    check(!r.error.empty(), "max_new must still be a number in id mode");
}

void test_stop_and_quit() {
    auto r = serve_proto::parse_request("QUIT", kDef);
    check(r.kind == serve_proto::Kind::quit && r.error.empty(), "QUIT");

    r = serve_proto::parse_request("STOP", kDef);
    check(r.kind == serve_proto::Kind::stop && r.id == serve_proto::kNoId && r.error.empty(),
          "a bare STOP is 0.1.30's: no id, cancel whatever is running");
    r = serve_proto::parse_request("STOP", kDef, true);
    check(r.kind == serve_proto::Kind::stop && r.id == serve_proto::kNoId, "and it means the same in id mode");

    r = serve_proto::parse_request("STOP 7", kDef, true);
    check(r.kind == serve_proto::Kind::stop && r.id == 7, "STOP <id> names one request");

    r = serve_proto::parse_request("STOP 7 ", kDef, true);
    check(r.kind == serve_proto::Kind::stop && r.id == 7, "trailing space is trimmed");

    r = serve_proto::parse_request("STOP seven", kDef, true);
    check(r.kind == serve_proto::Kind::stop && !r.error.empty(), "STOP with a non-numeric id is an error");

    r = serve_proto::parse_request("STOP -7", kDef, true);
    check(!r.error.empty(), "a request id is never negative");

    r = serve_proto::parse_request("STOP 7 8", kDef, true);
    check(!r.error.empty(), "STOP takes exactly one id");

    // --serve-slots 0 accepts EXACTLY "STOP", as 0.1.30 did: a line it rejected must still be rejected,
    // with 0.1.30's message.
    r = serve_proto::parse_request("STOP 7", kDef, false);
    check(r.kind == serve_proto::Kind::other && r.error == serve_proto::err_expected(),
          "a slots=0 engine never accepts STOP <id>");
}

void test_malformed() {
    auto r = serve_proto::parse_request("", kDef);
    check(r.kind == serve_proto::Kind::other && r.error == serve_proto::err_expected(), "an empty line");
    r = serve_proto::parse_request("GEN", kDef);
    check(r.kind == serve_proto::Kind::other, "GEN alone is not a request");
    r = serve_proto::parse_request("GENARATE 1 2", kDef);
    check(r.kind == serve_proto::Kind::other, "GEN must be followed by a space");
    r = serve_proto::parse_request("GENI 32 1,2", kDef);
    check(!r.error.empty(), "GENI with no embeddings file");
    r = serve_proto::parse_request("GEN 0 1,2", kDef);
    check(!r.error.empty(), "max_new 0");
    r = serve_proto::parse_request("GEN -5 1,2", kDef);
    check(!r.error.empty(), "a negative max_new");
    r = serve_proto::parse_request("GEN 32", kDef);
    check(!r.error.empty() && r.error.find("token list was empty") != std::string::npos, "no ids");
    r = serve_proto::parse_request("GEN 32 1,abc", kDef);
    check(!r.error.empty() && r.error.find("invalid token id") != std::string::npos, "a non-numeric id");
    r = serve_proto::parse_request("GEN 32 1,-2", kDef);
    check(!r.error.empty(), "a negative token id");
    r = serve_proto::parse_request("GEN 32 2147483648", kDef);
    check(!r.error.empty(), "a token id above int32 (0.1.30's range check)");
    r = serve_proto::parse_request("gen 32 1,2", kDef);
    check(r.kind == serve_proto::Kind::other, "the verb is upper case, as today");
}

// ============================================================ round trips ==============================
void test_round_trip() {
    const std::vector<int64_t> ids{1, 2, 300000, 248045};
    // tagged request -> line -> parse -> the same fields
    const std::string line = serve_proto::gen_line(serve_proto::Kind::gen, 7, 32, " temperature=0.7", "", ids);
    check_eq(line, "GEN 7 32 temperature=0.7 1,2,300000,248045", "gen_line, tagged");
    auto r = serve_proto::parse_request(line, kDef, true);
    check(r.error.empty() && r.id == 7 && r.max_new == 32 && r.ids == ids && r.temperature > 0.69f,
          "the tagged request round-trips");

    // untagged request -> line -> parse -> the same fields, id absent
    const std::string old = serve_proto::gen_line(serve_proto::Kind::gen, serve_proto::kNoId, 32,
                                                  " temperature=0.7", "", ids);
    check_eq(old, "GEN 32 temperature=0.7 1,2,300000,248045", "gen_line, untagged = 0.1.30's spelling");
    r = serve_proto::parse_request(old, kDef);
    check(r.error.empty() && r.id == serve_proto::kNoId && r.max_new == 32 && r.ids == ids,
          "the untagged request round-trips");

    const std::string gi = serve_proto::gen_line(serve_proto::Kind::geni, 9, 64, "", "x/y.sve", ids);
    check_eq(gi, "GENI 9 64 x/y.sve 1,2,300000,248045", "gen_line GENI");
    r = serve_proto::parse_request(gi, kDef, true);
    check(r.error.empty() && r.id == 9 && r.emb_path == "x/y.sve", "the GENI round trip");

    check_eq(serve_proto::stop_line(serve_proto::kNoId), "STOP", "STOP, untagged");
    check_eq(serve_proto::stop_line(7), "STOP 7", "STOP, tagged");
}

void test_tag_parsing() {
    std::string body;
    check(serve_proto::parse_tag("T 42 #7", &body) == 7 && body == "T 42", "the tag comes off a T line");
    check(serve_proto::parse_tag("T 42", &body) == serve_proto::kNoId && body == "T 42",
          "an untagged line parses unchanged");
    check(serve_proto::parse_tag("DONE 1 2 3.0 4.0 stop 0 0 0 0 0 #7", &body) == 7 &&
          body == "DONE 1 2 3.0 4.0 stop 0 0 0 0 0", "the tag comes off a DONE line");
    check(serve_proto::parse_tag("ERR something bad", &body) == serve_proto::kNoId, "an untagged ERR");
    // A message that merely contains " #" is not a tag unless what follows is all digits.
    check(serve_proto::parse_tag("ERR bad # thing", &body) == serve_proto::kNoId &&
          body == "ERR bad # thing", "a '#' inside a message is not a tag");
    check(serve_proto::parse_tag("ERR bad #7x", &body) == serve_proto::kNoId, "digits then junk is not a tag");
    // and the tag survives the formatter/parser pair in both directions
    const serve_proto::Out out(true);
    for (int64_t id : {0LL, 1LL, 999999LL}) {
        std::string b;
        const int64_t got = serve_proto::parse_tag(out.token(1234, id), &b);
        check(got == id && b == "T 1234", "formatter -> parse_tag round-trips the id");
    }
}

// ============================================================ the compatibility matrix =================
void test_compatibility_matrix() {
    // §6.4, row "old server / new engine": an id-less request against a slots=0 engine, end to end.
    const serve_proto::Out out(false);
    auto r = serve_proto::parse_request("GEN 32 temperature=0.7 1,2,3", kDef);
    check(r.error.empty() && r.id == serve_proto::kNoId, "an old client's request parses");
    check_eq(out.token(42, r.id), "T 42", "and its answer is 0.1.30's bytes");
    check_eq(out.ready(524288, 0), "READY 524288 stop", "READY advertises no slots=, so the server stays serial");

    // row "new server / new engine": the handshake turns routing on.
    const serve_proto::Out out2(true);
    auto r2 = serve_proto::parse_request("GEN 3 32 1,2,3", kDef, true);
    check(r2.id == 3, "the new client's id is read");
    check_eq(out2.token(42, r2.id), "T 42 #3", "and every line of its answer is tagged");

    // row "new server / old engine": no slots= token on READY, so the server never sends an id.  The
    // engine side of that row is the slots=0 case above.
    check(out2.ready(524288, 0) == out.ready(524288, 0),
          "READY does not gain slots= unless the engine was started with --serve-slots");
}

}  // namespace

int main() {
    test_golden_untagged_output();
    test_tagged_output();
    test_old_form_requests();
    test_new_form_requests();
    test_stop_and_quit();
    test_malformed();
    test_round_trip();
    test_tag_parsing();
    test_compatibility_matrix();
    std::fprintf(stderr, "serve_proto_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
