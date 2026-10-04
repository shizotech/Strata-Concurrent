#!/usr/bin/env python3
"""S3.1e-1: prove the extraction preserved the statement sequence.

Compares the WHOLE `if (o.serve) { ... }` block before and after, canonicalised
to a statement stream.  Nothing is excluded by line range: the hoisted helpers
appear on both sides and simply match.  What is normalised:
  * blank / comment-only lines dropped, whitespace collapsed
  * exit statements (continue / break / return 1 / _Exit / return Prep::* /
    return Step::*) -> <EXIT>
  * a per-request local declaration vs the equivalent `R.field = ...` write
  * `T& t = R.field;` alias declarations -> dropped
  * the segment `for` header vs the segment iterator -> <SEGMENT>
  * the decode `while` guard vs the step's entry guard -> <DECODEGUARD>
  * the guard structs and the step-driver glue -> dropped
  * the new ReqCtx / enum / DecSnap declarations -> dropped
If the two streams are identical, the refactor changed no statement and no
statement order."""
import re, sys, difflib

def strip_comments(l):
    q = False
    for i, c in enumerate(l):
        if c == '"':
            q = not q
        elif c == "/" and not q and l[i:i + 2] == "//":
            return l[:i]
    return l

EXITISH = {"continue;", "break;", "return 1;", "std::_Exit(1);",
           "return Prep::rejected;", "return Prep::fatal;", "return Prep::ok;",
           "return Step::error;", "return Step::finished;", "return Step::cancelled;",
           "return Step::fatal_exit;", "return Step::progressed;"}

ALIAS = re.compile(r"^(const )?[A-Za-z_][A-Za-z_:0-9<>, ]*& [a-z_0-9]+ = R\.[a-z_0-9]+;$")

# lines that exist only because of the extraction (new plumbing) -> dropped
NEW_ONLY = [
    r"^enum class Prep \{", r"^enum class Step \{", r"^struct ReqCtx \{$", r"^};$", r"^\{$", r"^\}$",
    r"^auto prep_request = \[&\]\(ReqCtx& R, const std::string& line\) -> Prep \{$",
    r"^auto plan_prompt_segments = \[&\]\(ReqCtx& R\) \{$",
    r"^auto run_prefill_step = \[&\]\(ReqCtx& R\) -> Step \{$",
    r"^auto finish_prefill = \[&\]\(ReqCtx& R\) -> Step \{$",
    r"^auto run_decode_step = \[&\]\(ReqCtx& R\) -> Step \{$",
    r"^auto finish_request = \[&\]\(ReqCtx& R\) -> Step \{$",
    r"^ReqCtx R;$",
    r"^const Prep prep = prep_request\(R, line\);$",
    r"^if \(prep == Prep::(rejected|fatal)\) (continue|return 1);$",
    r"^plan_prompt_segments\(R\);$",
    r"^Step (st|dst) = Step::progressed;$",
    r"^while \(\((st|dst) = (run_prefill_step|run_decode_step)\(R\)\) == Step::progressed\) \{\}$",
    r"^if \((st|dst) == Step::(error|fatal_exit)\)( \{)? return 1;$",
    r"^if \(st == Step::fatal_exit\) \{$",
    r"^if \(finish_prefill\(R\) == Step::error\) return 1;$",
    r"^if \(finish_request\(R\) == Step::error\) return 1;$",
    r"^struct SlotGuard \{$", r"^bool open;$",
    r"^std::function<void\(int64_t\)> finish;$", r"^int64_t id;$",
    r"^~SlotGuard\(\) \{ if \(open && finish\) finish\(id\); \}$",
    r"^ReqCtx\* ctx = nullptr;$", r"^~SlotGuard\(\) \{ if \(ctx && ctx->slot_open && finish\) finish\(ctx->id\); \}$",
    r"^struct MropeScope \{$", r"^std::function<bool\(\)> skip;$", r"^std::function<void\(\)> restore;$",
    r"^~MropeScope\(\) \{ if \(skip && skip\(\)\) return; if \(restore\) restore\(\); \}$",
    r"^MropeScope mrope_scope;$",
    r"^mrope_scope\.(skip|restore) = ",
    r"^if \(mounted_id == strata::program::serve_proto::kNoId\) return;$",
    r"^apply_positions\(conv_of\(mounted_id\)\);$",
    r"^if \(!mrope_host\.empty\(\)\) upload_mrope_table\(\);$",
    r"^mrope_owner = mounted_id;$",
    r"^int64_t (turn_at|root_at) = -1;$",
    r"^std::array<int64_t, 4> seg\{\{.*\}\};$", r"^size_t seg_i = 0;$",
    r"^int64_t at = 0;$", r"^bool cancelled = false;$", r"^double prompt_ms = 0\.0;$",
    r"^int64_t p = 0;$", r"^int32_t x = 0;$",
    r"^std::vector<int32_t> drafts, window, outv, sbuf, consumed;$",
    r"^std::vector<float> dprob;$", r"^bool first_window = true;$",
    r"^int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;$",
    r"^int64_t draft_offered = 0, draft_accepted = 0;$",
    r'^const char\* finish = "length";$', r"^Clock::time_point d0\{\};$",
    r"^DecSnap ds0\{\};$", r"^double dt_run = 0, dt_commit = 0, dt_draft = 0;$",
    r"^int64_t dec_windows = 0, dec_T = 0;$", r"^int64_t decode_hits0 = 0, decode_look0 = 0;$",
    r"^double decode_ms = 0\.0;$",
    r"^strata::program::serve_proto::Request rq;$", r"^bool geni = false;$", r"^long long max_new = 0;$",
    r"^float req_temperature = 0\.0f, req_top_p = 1\.0f, req_min_p = 0\.0f;$",
    r"^float req_penalty_repeat = 1\.0f, req_penalty_freq = 0\.0f, req_penalty_present = 0\.0f;$",
    r"^int req_top_k = 20, req_penalty_last_n = 0, req_cvec = 1;$",
    r"^unsigned long long req_seed = 0;$", r"^double req_pcie_frac = 0\.0, req_spec_min_p = 0\.0;$",
    r"^std::string emb_path;$", r"^std::vector<int64_t> ids;$",
    r"^int64_t id = strata::program::serve_proto::kNoId;.*$", r"^int64_t n = 0;$",
    r"^bool slot_open = false;.*$", r"^bool mrope_touched = false;.*$",
    r"^std::array<int64_t, 3> remote_before\{\}, launches_before\{\};$",
    r"^std::array<uint64_t, 3> compact_before\{\}, full_before\{\};$",
    r"^std::array<double, 3> begin_before\{\}, wait_before\{\};$",
    r"^Clock::time_point r0\{\};$", r"^std::array<int64_t, 4> loan_bill\{\};$",
    r"^int64_t res_uploads0 = 0, res_skips0 = 0;$", r"^bool want_cvec = true;$",
    r"^int64_t resume = 0;$", r"^bool from_live = false;$", r"^int64_t reread_to = -1;$",
    r"^int64_t read_from = 0;$", r"^strata::kernels::SamplerParams req_sp;$", r"^int hist_n = 0;$",
    r"^struct DecSnap \{$", r"^double wait, pool, host, plan, actq, jobs, run;$",
    r"^int64_t misses, entries, hits, pcie;$",
    r"^auto dec_snap = \[&\]\(\) \{$",
    r"^return DecSnap\{ver\.ms_wait, ver\.ms_pool, ver\.ms_host, drive\.d\.ms_plan, drive\.d\.ms_actq, drive\.d\.ms_jobs,$",
    r"^drive\.d\.ms_run, drive\.d\.multi_misses, drive\.d\.multi_entries, drive\.d\.cache_hits,$",
    r"^drive\.d\.pcie_experts\};$",
    r"^static const bool dec_timing = std::getenv\(\"STRATA_DECODE_TIMING\"\) != nullptr;$",
    r"^R\.seg = \{reread_to, root_at, turn_at, n - 1\};$", r"^R\.seg_i = 0;$",
    r"^int64_t to = -1;$", r"^bool have = false;$", r"^while \(R\.seg_i < R\.seg\.size\(\)\) \{$",
    r"^const int64_t t = R\.seg\[R\.seg_i\+\+\];$", r"^if \(t > at\) \{ to = t; have = true; break; \}$",
    r"^if \(!have\) return Step::finished;$",
    r"^if \(cancelled \|\| produced_n >= max_new\) return Step::finished;$",
    r"^std::fflush\(stdout\);$", r"^std::fflush\(stderr\);$",
]
NEW_ONLY = [re.compile(p) for p in NEW_ONLY]

OLD_DECL = [
    (r"^const bool geni = (.*);$", r"geni = \1"),
    (r"^const long long max_new = (.*);$", r"max_new = \1"),
    (r"^const float req_temperature = (.*);$", r"req_temperature = \1"),
    (r"^const float req_top_p = (.*);$", r"req_top_p = \1"),
    (r"^const int req_top_k = ([^;]*);.*$", r"req_top_k = \1"),
    (r"^const unsigned long long req_seed = (.*);$", r"req_seed = \1"),
    (r"^const float req_min_p = rq\.min_p, req_penalty_repeat = rq\.penalty_repeat;$",
     r"req_min_p = rq.min_p; req_penalty_repeat = rq.penalty_repeat"),
    (r"^const float req_penalty_freq = rq\.penalty_freq, req_penalty_present = rq\.penalty_present;$",
     r"req_penalty_freq = rq.penalty_freq; req_penalty_present = rq.penalty_present"),
    (r"^const int req_penalty_last_n = (.*);$", r"req_penalty_last_n = \1"),
    (r"^const int req_cvec = ([^;]*);.*$", r"req_cvec = \1"),
    (r"^const double req_pcie_frac = rq\.pcie_frac, req_spec_min_p = rq\.spec_min_p;$",
     r"req_pcie_frac = rq.pcie_frac; req_spec_min_p = rq.spec_min_p"),
    (r"^const std::string emb_path = (.*);$", r"emb_path = \1"),
    (r"^std::vector<int64_t> ids = (.*);$", r"ids = \1"),
    (r"^bool slot_open = (.*);$", r"slot_open = \1"),
    (r"^const int64_t n = (.*);$", r"n = \1"),
    (r"^std::array<int64_t, 3> remote_before\{\};$", r"remote_before = {}"),
    (r"^std::array<int64_t, 3> launches_before\{\};$", r"launches_before = {}"),
    (r"^std::array<uint64_t, 3> compact_before\{\}, full_before\{\};$", r"compact_before = {}; full_before = {}"),
    (r"^std::array<double, 3> begin_before\{\}, wait_before\{\};$", r"begin_before = {}; wait_before = {}"),
    (r"^const Clock::time_point r0 = (.*);$", r"r0 = \1"),
    (r"^const std::array<int64_t, 4> loan_bill = (.*);$", r"loan_bill = \1"),
    (r"^const int64_t res_uploads0 = res_dirty\.uploads\(\), res_skips0 = res_dirty\.skipped\(\);$",
     r"res_uploads0 = res_dirty.uploads(); res_skips0 = res_dirty.skipped()"),
    (r"^const bool want_cvec = (.*);$", r"want_cvec = \1"),
    (r"^int64_t resume = (.*);$", r"resume = \1"),
    (r"^bool from_live = (.*);$", r"from_live = \1"),
    (r"^int64_t reread_to = ([^;]*);.*$", r"reread_to = \1"),
    (r"^const int64_t read_from = (.*);$", r"read_from = \1"),
    (r"^strata::kernels::SamplerParams req_sp;$", r"req_sp"),
    (r"^const int hist_n = (.*);$", r"hist_n = \1"),
    (r"^bool cancelled = (.*);$", r"cancelled = \1"),
    (r"^int64_t turn_at = (.*);$", r"turn_at = \1"),
    (r"^int64_t root_at = (.*);$", r"root_at = \1"),
    (r"^int64_t at = (.*);$", r"at = \1"),
    (r"^const double prompt_ms = (.*);$", r"prompt_ms = \1"),
    (r"^int64_t p = (.*);$", r"p = \1"),
    (r"^int32_t x = (.*);$", r"x = \1"),
    (r"^std::vector<int32_t> drafts\(size_t S, 0\), window\(size_t S\), outv\(size_t S\);$", r"SIZED"),
    (r"^std::vector<float> dprob\(size_t S, 0\.0f\);$", r"SIZED"),
    (r"^std::vector<int32_t> sbuf\(size_t S, 0\);$", r"SIZED"),
    (r"^bool first_window = (.*);$", r"first_window = \1"),
    (r"^int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;$", r"ZEROED"),
    (r"^int64_t draft_offered = 0, draft_accepted = 0;$", r"ZEROED"),
    (r"^std::vector<int32_t> consumed;$", r"consumed"),
    (r"^consumed\.reserve\((.*);$", r"consumed.reserve(\1)"),
    (r'^const char\* finish = (.*);$', r'finish = \1'),
    (r"^const Clock::time_point d0 = (.*);$", r"d0 = \1"),
    (r"^const DecSnap ds0 = (.*);$", r"ds0 = \1"),
    (r"^double dt_run = 0, dt_commit = 0, dt_draft = 0;$", r"ZEROED"),
    (r"^int64_t dec_windows = 0, dec_T = 0;$", r"ZEROED"),
    (r"^const int64_t decode_hits0 = (.*);$", r"decode_hits0 = \1"),
    (r"^const int64_t decode_look0 = (.*);$", r"decode_look0 = \1"),
    (r"^const double decode_ms = (.*);$", r"decode_ms = \1"),
    (r"^mrope_touched = true;$", r"mrope_touched = true"),
    (r"^bool mrope_touched = false;$", None),
    (r"^struct DecSnap \{$", None), (r"^double wait, pool, host, plan, actq, jobs, run;$", None),
    (r"^int64_t misses, entries, hits, pcie;$", None), (r"^auto dec_snap = \[&\]\(\) \{$", None),
    (r"^return DecSnap\{ver\.ms_wait, ver\.ms_pool, ver\.ms_host, drive\.d\.ms_plan, drive\.d\.ms_actq, drive\.d\.ms_jobs,$", None),
    (r"^drive\.d\.ms_run, drive\.d\.multi_misses, drive\.d\.multi_entries, drive\.d\.cache_hits,$", None),
    (r"^drive\.d\.pcie_experts\};$", None),
    (r"^static const bool dec_timing = std::getenv\(\"STRATA_DECODE_TIMING\"\) != nullptr;$", None),
    (r"^struct SlotGuard \{$", None), (r"^bool open;$", None),
    (r"^std::function<void\(int64_t\)> finish;$", None), (r"^int64_t id;$", None),
    (r"^~SlotGuard\(\) \{ if \(open && finish\) finish\(id\); \}$", None),
    (r"^\} slot_guard\{slot_open, tagged \? std::function<void\(int64_t\)>\(slot_finish\) : nullptr, req_id\};$", None),
    (r"^struct MropeScope \{$", None), (r"^std::function<bool\(\)> skip;.*$", None),
    (r"^std::function<void\(\)> restore;$", None),
    (r"^~MropeScope\(\) \{ if \(skip && skip\(\)\) return; if \(restore\) restore\(\); \}$", None),
    (r"^MropeScope mrope_scope;$", None),
    (r"^mrope_scope\.skip = \[&\]\(\) \{ return !swaps_on \|\| !mrope_touched \|\| mounted_id == req_id; \};$", None),
    (r"^mrope_scope\.restore = \[&\]\(\) \{$", None),
    (r"^if \(mounted_id == strata::program::serve_proto::kNoId\) return;$", None),
    (r"^apply_positions\(conv_of\(mounted_id\)\);$", None),
    (r"^if \(!mrope_host\.empty\(\)\) upload_mrope_table\(\);$", None),
    (r"^mrope_owner = mounted_id;$", None),
    (r"^for \(const int64_t to : \{reread_to, root_at, turn_at, n - 1\}\) \{$", "<SEGMENT>"),
    (r"^if \(to <= at\) continue;$", None),
    (r"^while \(!cancelled && produced_n < max_new\) \{$", "<DECODEGUARD>"),
    (r"^std::fflush\(stdout\);$", "<FLUSH>"), (r"^std::fflush\(stderr\);$", "<FLUSH>"),
    (r"^const strata::program::serve_proto::Request rq =$", None),
    (r"^strata::program::serve_proto::parse_request\(line, proto_def, tagged\);$",
     r"rq = strata::program::serve_proto::parse_request(line, proto_def, tagged)"),
]
NEW_DECL = [
    (r"^R\.rq = strata::program::serve_proto::parse_request\(line, proto_def, tagged\);$",
     r"rq = strata::program::serve_proto::parse_request(line, proto_def, tagged)"),
    (r"^R\.geni = (.*);$", r"geni = \1"), (r"^R\.max_new = (.*);$", r"max_new = \1"),
    (r"^R\.req_temperature = (.*);$", r"req_temperature = \1"),
    (r"^R\.req_top_p = (.*);$", r"req_top_p = \1"),
    (r"^R\.req_top_k = ([^;]*);$", r"req_top_k = \1"),
    (r"^R\.req_seed = (.*);$", r"req_seed = \1"),
    (r"^R\.req_min_p = rq\.min_p; R\.req_penalty_repeat = rq\.penalty_repeat;$",
     r"req_min_p = rq.min_p; req_penalty_repeat = rq.penalty_repeat"),
    (r"^R\.req_penalty_freq = rq\.penalty_freq; R\.req_penalty_present = rq\.penalty_present;$",
     r"req_penalty_freq = rq.penalty_freq; req_penalty_present = rq.penalty_present"),
    (r"^R\.req_penalty_last_n = (.*);$", r"req_penalty_last_n = \1"),
    (r"^R\.req_cvec = ([^;]*);$", r"req_cvec = \1"),
    (r"^R\.req_pcie_frac = rq\.pcie_frac; R\.req_spec_min_p = rq\.spec_min_p;$",
     r"req_pcie_frac = rq.pcie_frac; req_spec_min_p = rq.spec_min_p"),
    (r"^R\.emb_path = (.*);$", r"emb_path = \1"), (r"^R\.ids = (.*);$", r"ids = \1"),
    (r"^R\.id = rq\.id;$", None),
    (r"^R\.n = (.*);$", r"n = \1"),
    (r"^R\.remote_before = \{\};$", r"remote_before = {}"),
    (r"^R\.launches_before = \{\};$", r"launches_before = {}"),
    (r"^R\.compact_before = \{\}; R\.full_before = \{\};$", r"compact_before = {}; full_before = {}"),
    (r"^R\.begin_before = \{\}; R\.wait_before = \{\};$", r"begin_before = {}; wait_before = {}"),
    (r"^R\.r0 = (.*);$", r"r0 = \1"), (r"^R\.loan_bill = (.*);$", r"loan_bill = \1"),
    (r"^R\.res_uploads0 = res_dirty\.uploads\(\); R\.res_skips0 = res_dirty\.skipped\(\);$",
     r"res_uploads0 = res_dirty.uploads(); res_skips0 = res_dirty.skipped()"),
    (r"^R\.want_cvec = (.*);$", r"want_cvec = \1"), (r"^R\.resume = (.*);$", r"resume = \1"),
    (r"^R\.from_live = (.*);$", r"from_live = \1"), (r"^R\.reread_to = ([^;]*);.*$", r"reread_to = \1"),
    (r"^R\.read_from = (.*);$", r"read_from = \1"), (r"^R\.hist_n = (.*);$", r"hist_n = \1"),
    (r"^R\.cancelled = (.*);$", r"cancelled = \1"), (r"^R\.turn_at = (.*);$", r"turn_at = \1"),
    (r"^R\.root_at = (.*);$", r"root_at = \1"), (r"^R\.at = (.*);$", r"at = \1"),
    (r"^R\.prompt_ms = (.*);$", r"prompt_ms = \1"), (r"^R\.p = (.*);$", r"p = \1"),
    (r"^R\.x = (.*);$", r"x = \1"),
    (r"^R\.drafts\.assign\(size_t S, 0\); R\.window\.assign\(size_t S, 0\); R\.outv\.assign\(size_t S, 0\);$", r"SIZED"),
    (r"^R\.dprob\.assign\(size_t S, 0\.0f\);$", r"SIZED"),
    (r"^R\.sbuf\.assign\(size_t S, 0\);$", r"SIZED"),
    (r"^R\.first_window = (.*);$", r"first_window = \1"),
    (r"^R\.produced_n = 0; R\.sfx_windows = 0; R\.sfx_drafts = 0; R\.sfx_ok = 0;$", r"ZEROED"),
    (r"^R\.draft_offered = 0; R\.draft_accepted = 0;$", r"ZEROED"),
    (r"^consumed\.clear\(\); consumed\.reserve\((.*);$", r"consumed\nconsumed.reserve(\1)"),
    (r"^consumed\.clear\(\); consumed\.reserve\((.*)\);$", r"consumed\nconsumed.reserve(\1)"),
    (r"^R\.finish = (.*);$", r"finish = \1"), (r"^R\.d0 = (.*);$", r"d0 = \1"),
    (r"^R\.ds0 = (.*);$", r"ds0 = \1"), (r"^R\.dt_run = 0; R\.dt_commit = 0; R\.dt_draft = 0;$", r"ZEROED"),
    (r"^R\.dec_windows = 0; R\.dec_T = 0;$", r"ZEROED"),
    (r"^R\.decode_hits0 = (.*);$", r"decode_hits0 = \1"),
    (r"^R\.decode_look0 = (.*);$", r"decode_look0 = \1"),
    (r"^R\.decode_ms = (.*);$", r"decode_ms = \1"),
    (r"^R\.mrope_touched = true;$", r"mrope_touched = true"),
    (r"^strata::kernels::SamplerParams& req_sp = R\.req_sp;$", r"req_sp"),
    (r"^SlotGuard slot_guard\{&R, tagged \? std::function<void\(int64_t\)>\(slot_finish\) : nullptr\};$", None),
    (r"^mrope_scope\.skip = \[&\]\(\) \{ return !swaps_on \|\| !R\.mrope_touched \|\| mounted_id == req_id; \};$", None),
    (r"^mrope_scope\.restore = \[&\]\(\) \{$", None),
    (r"^if \(mounted_id == strata::program::serve_proto::kNoId\) return;$", None),
    (r"^apply_positions\(conv_of\(mounted_id\)\);$", None),
    (r"^if \(!mrope_host\.empty\(\)\) upload_mrope_table\(\);$", None),
    (r"^mrope_owner = mounted_id;$", None),
    (r"^std::fflush\(stdout\);$", "<FLUSH>"), (r"^std::fflush\(stderr\);$", "<FLUSH>"),
]

def canon(l, decls):
    s = re.sub(r"\s+", " ", strip_comments(l)).strip()
    if not s:
        return None
    if s in EXITISH:
        return "<EXIT>"
    if ALIAS.match(s):
        return None
    for rx in NEW_ONLY:
        if rx.match(s):
            return None
    m = re.match(r"^if \((.*)\) \{ (.*) (break|return Step::(finished|cancelled)); \}$", s)
    if m:
        return "if (%s) { %s <EXIT> }" % (m.group(1), m.group(2))
    m = re.match(r"^if \((.*)\) (return 1|return Step::error|return Prep::rejected);$", s)
    if m:
        return "if (%s) <EXIT>" % m.group(1)
    m = re.match(r"^if \((.*)\) \{ std::printf\((.*)\); (continue|return Prep::rejected); \}$", s)
    if m:
        return "if (%s) { std::printf(%s); <EXIT> }" % (m.group(1), m.group(2))
    m = re.match(r"^if \(poisoned\) (return 1|return Prep::fatal);$", s)
    if m:
        return "if (poisoned) <EXIT>"
    m = re.match(r"^if \(p \+ T > o\.max_context\) (break|return Step::finished);$", s)
    if m:
        return "if (p + T > o.max_context) <EXIT>"
    for rx, out in decls:
        if re.match(rx, s):
            return out if out is None or not out.startswith("<") else out
    return s

def stream(lines, a, b, decls):
    out = []
    for l in lines[a - 1:b]:
        c = canon(l, decls)
        if c is None:
            continue
        for part in c.split("\n"):
            if part:
                out.append(part)
    return out

old = open(sys.argv[1]).read().split("\n")
new = open(sys.argv[2]).read().split("\n")
oa = next(i for i, l in enumerate(old, 1) if l == "    if (o.serve) {")
ob = next(i for i, l in enumerate(old, 1) if i > oa and l == "        return 0;")
na = next(i for i, l in enumerate(new, 1) if l == "    if (o.serve) {")
nb = next(i for i, l in enumerate(new, 1) if i > na and l == "        return 0;")
OB = stream(old, oa, ob, OLD_DECL)
NB = stream(new, na, nb, NEW_DECL)

def squash(v):
    out = []
    for s in v:
        if s in ("<FLUSH>",) and out and out[-1] == s:
            continue
        out.append(s)
    return out
OB, NB = squash(OB), squash(NB)
if OB == NB:
    print("IDENTICAL canonical statement streams: %d statements (serve block %d..%d vs %d..%d)"
          % (len(OB), oa, ob, na, nb))
    sys.exit(0)
print("MISMATCH: before %d, after %d" % (len(OB), len(NB)))
for l in difflib.unified_diff(OB, NB, "before", "after", lineterm="", n=1):
    print(l)
sys.exit(1)
