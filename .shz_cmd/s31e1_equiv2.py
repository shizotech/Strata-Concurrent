#!/usr/bin/env python3
"""S3.1e-1: prove the extraction preserved the statement sequence.

Canonicalises both sides to a statement stream and requires equality.
Excluded from the stream (and checked separately):
  * the hoisted helper definitions (windows_ok / read_windows / lend / dec_snap)
  * the new ReqCtx / enum / guard struct declarations
  * ReqCtx alias declarations
Rewrites that ARE checked but normalised:
  * exit statements (continue/break/return 1/_Exit -> <EXIT>)
  * a local declaration vs the equivalent `R.field = ...` write
  * the segment `for` header vs the segment iterator
  * the decode `while` guard vs the step's entry guard
"""
import re, sys, difflib

def strip_comments(l):
    q = False
    for i, c in enumerate(l):
        if c == '"':
            q = not q
        elif c == "/" and not q and l[i:i + 2] == "//":
            return l[:i]
    return l

ALIAS = re.compile(r"^(const )?[A-Za-z_][A-Za-z_:0-9<>, ]*& [a-z_0-9]+ = R\.[a-z_0-9]+;$")

EXITISH = {"continue;", "break;", "return 1;", "std::_Exit(1);",
           "return Prep::rejected;", "return Prep::fatal;", "return Prep::ok;",
           "return Step::error;", "return Step::finished;", "return Step::cancelled;",
           "return Step::fatal_exit;", "return Step::progressed;"}

# old local declaration  ->  canonical
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
    (r"^consumed\.reserve\((.*)\);$", r"consumed.reserve(\1)"),
    (r'^const char\* finish = (.*);$', r'finish = \1'),
    (r"^const Clock::time_point d0 = (.*);$", r"d0 = \1"),
    (r"^const DecSnap ds0 = (.*);$", r"ds0 = \1"),
    (r"^double dt_run = 0, dt_commit = 0, dt_draft = 0;$", r"ZEROED"),
    (r"^int64_t dec_windows = 0, dec_T = 0;$", r"ZEROED"),
    (r"^const int64_t decode_hits0 = (.*);$", r"decode_hits0 = \1"),
    (r"^const int64_t decode_look0 = (.*);$", r"decode_look0 = \1"),
    (r"^const double decode_ms = (.*);$", r"decode_ms = \1"),
    (r"^mrope_touched = true;$", r"mrope_touched = true"),
    (r"^const strata::program::serve_proto::Request rq =$", r"SKIP"),
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
    (r"^R\.id = rq\.id;$", r"SKIP"),
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
    (r"^consumed\.clear\(\); consumed\.reserve\((.*)\);$", r"consumed\nconsumed.reserve(\1)"),
    (r"^R\.finish = (.*);$", r"finish = \1"), (r"^R\.d0 = (.*);$", r"d0 = \1"),
    (r"^R\.ds0 = (.*);$", r"ds0 = \1"), (r"^R\.dt_run = 0; R\.dt_commit = 0; R\.dt_draft = 0;$", r"ZEROED"),
    (r"^R\.dec_windows = 0; R\.dec_T = 0;$", r"ZEROED"),
    (r"^R\.decode_hits0 = (.*);$", r"decode_hits0 = \1"),
    (r"^R\.decode_look0 = (.*);$", r"decode_look0 = \1"),
    (r"^R\.decode_ms = (.*);$", r"decode_ms = \1"),
    (r"^R\.mrope_touched = true;$", r"mrope_touched = true"),
    (r"^strata::kernels::SamplerParams& req_sp = R\.req_sp;$", r"req_sp"),
]

DROP = {
    "for (const int64_t to : {reread_to, root_at, turn_at, n - 1}) {": "<SEGMENT>",
    "if (to <= at) continue;": None,
    "R.seg = {reread_to, root_at, turn_at, n - 1};": "<SEGMENT>",
    "R.seg_i = 0;": None, "int64_t to = -1;": None, "bool have = false;": None,
    "while (R.seg_i < R.seg.size()) {": None, "const int64_t t = R.seg[R.seg_i++];": None,
    "if (t > at) { to = t; have = true; break; }": None,
    "if (!have) return Step::finished;": None,
    "while (!cancelled && produced_n < max_new) {": "<DECODEGUARD>",
    "if (cancelled || produced_n >= max_new) return Step::finished;": "<DECODEGUARD>",
    "struct SlotGuard {": None, "bool open;": None,
    "std::function<void(int64_t)> finish;": None, "int64_t id;": None,
    "~SlotGuard() { if (open && finish) finish(id); }": None,
    "} slot_guard{slot_open, tagged ? std::function<void(int64_t)>(slot_finish) : nullptr, req_id};": "<GUARD>",
    "SlotGuard slot_guard{&R, tagged ? std::function<void(int64_t)>(slot_finish) : nullptr};": "<GUARD>",
    "bool mrope_touched = false;": None,
    "struct MropeScope {": None, "std::function<bool()> skip;": None,
    "std::function<void()> restore;": None,
    "~MropeScope() { if (skip && skip()) return; if (restore) restore(); }": None,
    "};": None, "{": None, "}": None,
    "MropeScope mrope_scope;": "<MROPE>",
    "mrope_scope.skip = [&]() { return !swaps_on || !mrope_touched || mounted_id == req_id; };": "<MROPE>",
    "mrope_scope.skip = [&]() { return !swaps_on || !R.mrope_touched || mounted_id == req_id; };": "<MROPE>",
    "mrope_scope.restore = [&]() {": "<MROPE>",
    "if (mounted_id == strata::program::serve_proto::kNoId) return;": "<MROPE>",
    "apply_positions(conv_of(mounted_id));": "<MROPE>",
    "if (!mrope_host.empty()) upload_mrope_table();": "<MROPE>",
    "mrope_owner = mounted_id;": "<MROPE>",
    "const Prep prep = prep_request(R, line);": None,
    "if (prep == Prep::rejected) continue;": None,
    "if (prep == Prep::fatal) return 1;": None,
    "plan_prompt_segments(R);": None,
    "Step st = Step::progressed;": None,
    "while ((st = run_prefill_step(R)) == Step::progressed) {}": None,
    "if (st == Step::error) return 1;": None,
    "if (st == Step::fatal_exit) {": None,
    "if (finish_prefill(R) == Step::error) return 1;": None,
    "Step dst = Step::progressed;": None,
    "while ((dst = run_decode_step(R)) == Step::progressed) {}": None,
    "if (dst == Step::error) return 1;": None,
    "if (finish_request(R) == Step::error) return 1;": None,
    "std::fflush(stdout);": "<FLUSH>", "std::fflush(stderr);": "<FLUSH>",
    "ReqCtx R;": None,
}

def canon(l, decls):
    s = re.sub(r"\s+", " ", strip_comments(l)).strip()
    if not s:
        return None
    if s in EXITISH:
        return "<EXIT>"
    if s in DROP:
        return DROP[s]
    if ALIAS.match(s):
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
            return re.sub(rx, out, s)
    return s

def stream(lines, ranges, decls):
    out = []
    for a, b in ranges:
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
OR = [(int(a), int(b)) for a, b in zip(sys.argv[3::2], sys.argv[4::2])]
NR = [(int(a), int(b)) for a, b in zip(sys.argv[5::2], sys.argv[6::2])]
OB = stream(old, OR, OLD_DECL)
NB = stream(new, NR, NEW_DECL)
# collapse runs of <MROPE>/<FLUSH> markers (they are guard plumbing, order-insensitive here)
def squash(v):
    out = []
    for s in v:
        if s in ("<MROPE>", "<FLUSH>") and out and out[-1] == s:
            continue
        out.append(s)
    return out
OB, NB = squash(OB), squash(NB)
if OB == NB:
    print("IDENTICAL canonical statement streams: %d statements" % len(OB))
    sys.exit(0)
print("MISMATCH: before %d, after %d" % (len(OB), len(NB)))
for l in difflib.unified_diff(OB, NB, "before", "after", lineterm="", n=1):
    print(l)
sys.exit(1)
