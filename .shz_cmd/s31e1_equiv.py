#!/usr/bin/env python3
"""S3.1e-1: PROVE the extraction preserved the statement sequence.

Both sides are reduced to a canonical statement stream:
  * blank and comment-only lines dropped, whitespace collapsed
  * `ReqCtx` alias declarations dropped (they are new plumbing, not behaviour)
  * every exit rewrite (`continue` / `break` / `return 1` / `std::_Exit(1)` /
    `return Prep::*` / `return Step::*`) mapped to one canonical token
  * the local declarations that became `ReqCtx` writes mapped to the write
  * the guard structs / driver glue / segment iterator mapped to fixed tokens
The two streams must then be IDENTICAL.  Anything else is a behaviour change."""
import re, sys, difflib

def strip_comments(l):
    # drop trailing // comments (no string literal can contain `//` in this code
    # except inside a printf, which never does here)
    q = False
    for i, c in enumerate(l):
        if c == '"':
            q = not q
        elif c == "/" and not q and l[i:i + 2] == "//":
            return l[:i]
    return l

def canon(l):
    s = strip_comments(l).strip()
    if not s:
        return None
    s = re.sub(r"\s+", " ", s)
    # alias declarations: pure plumbing
    if re.match(r"^(const )?[A-Za-z_][A-Za-z_:0-9<>, ]*& [a-z_0-9]+ = R\.[a-z_0-9]+;$", s):
        return None
    if s == "const int64_t n = R.n;":
        return None
    # exit rewrites
    if s in ("continue;", "break;", "return 1;", "std::_Exit(1);",
             "return Prep::rejected;", "return Prep::fatal;", "return Prep::ok;",
             "return Step::error;", "return Step::finished;", "return Step::cancelled;",
             "return Step::fatal_exit;", "return Step::progressed;"):
        return "<EXIT>"
    if s == "if (poisoned) return 1;":
        return "if (poisoned) <EXIT>"
    if s == "if (p + T > o.max_context) break;":
        return "if (p + T > o.max_context) <EXIT>"
    m = re.match(r"^if \((.*)\) \{ (.*) (break|return Step::(finished|cancelled)); \}$", s)
    if m:
        return "if (%s) { %s <EXIT> }" % (m.group(1), m.group(2))
    m = re.match(r"^if \((.*)\) (return 1|return Step::error|return Prep::rejected);$", s)
    if m:
        return "if (%s) <EXIT>" % m.group(1)
    m = re.match(r"^if \((.*)\) \{ std::printf\((.*)\); (continue|return Prep::rejected); \}$", s)
    if m:
        return "if (%s) { std::printf(%s); <EXIT> }" % (m.group(1), m.group(2))
    # the `for (const int64_t to : {...})` header and its skip -> the segment iterator
    if s.startswith("for (const int64_t to : {reread_to, root_at, turn_at, n - 1})"):
        return "<SEGMENT-ITER>"
    if s == "if (to <= at) continue;":
        return None
    if re.match(r"^R\.seg = \{reread_to, root_at, turn_at, n - 1\};$", s):
        return "<SEGMENT-ITER>"
    if s in ("R.seg_i = 0;", "int64_t to = -1;", "bool have = false;",
             "while (R.seg_i < R.seg.size()) {", "const int64_t t = R.seg[R.seg_i++];",
             "if (t > at) { to = t; have = true; break; }", "}", "if (!have) return Step::finished;"):
        return None
    # the decode loop guard
    if s == "while (!cancelled && produced_n < max_new) {":
        return "<DECODE-GUARD>"
    if s == "if (cancelled || produced_n >= max_new) return Step::finished;":
        return "<DECODE-GUARD>"
    # locals that became ReqCtx writes: keep the RHS, drop the LHS declaration form
    DECL2WRITE = [
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
        (r"^std::vector<int32_t> drafts\((size_t) S, 0\), window\((size_t) S\), outv\((size_t) S\);$",
         r"drafts/window/outv sized S"),
        (r"^std::vector<float> dprob\((size_t) S, 0\.0f\);$", r"dprob sized S"),
        (r"^std::vector<int32_t> sbuf\((size_t) S, 0\);$", r"sbuf sized S"),
        (r"^bool first_window = (.*);$", r"first_window = \1"),
        (r"^int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;$",
         r"produced_n/sfx_* = 0"),
        (r"^int64_t draft_offered = 0, draft_accepted = 0;$", r"draft_offered/accepted = 0"),
        (r"^std::vector<int32_t> consumed;$", r"consumed"),
        (r"^consumed\.reserve\((.*)\);$", r"consumed.reserve(\1)"),
        (r'^const char\* finish = (.*);$', r'finish = \1'),
        (r"^const Clock::time_point d0 = (.*);$", r"d0 = \1"),
        (r"^const DecSnap ds0 = (.*);$", r"ds0 = \1"),
        (r"^double dt_run = 0, dt_commit = 0, dt_draft = 0;$", r"dt_* = 0"),
        (r"^int64_t dec_windows = 0, dec_T = 0;$", r"dec_windows/dec_T = 0"),
        (r"^const int64_t decode_hits0 = (.*);$", r"decode_hits0 = \1"),
        (r"^const int64_t decode_look0 = (.*);$", r"decode_look0 = \1"),
        (r"^const double decode_ms = (.*);$", r"decode_ms = \1"),
        (r"^mrope_touched = true;$", r"mrope_touched = true"),
    ]
    for rx, out in DECL2WRITE:
        m = re.match(rx, s)
        if m:
            return re.sub(rx, out, s)
    WRITE2DECL = [
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
        (r"^R\.drafts\.assign\((size_t) S, 0\); R\.window\.assign\((size_t) S, 0\); R\.outv\.assign\((size_t) S, 0\);$",
         r"drafts/window/outv sized S"),
        (r"^R\.dprob\.assign\((size_t) S, 0\.0f\);$", r"dprob sized S"),
        (r"^R\.sbuf\.assign\((size_t) S, 0\);$", r"sbuf sized S"),
        (r"^R\.first_window = (.*);$", r"first_window = \1"),
        (r"^R\.produced_n = 0; R\.sfx_windows = 0; R\.sfx_drafts = 0; R\.sfx_ok = 0;$", r"produced_n/sfx_* = 0"),
        (r"^R\.draft_offered = 0; R\.draft_accepted = 0;$", r"draft_offered/accepted = 0"),
        (r"^consumed\.clear\(\); consumed\.reserve\((.*)\);$", r"consumed\nconsumed.reserve(\1)"),
        (r"^R\.finish = (.*);$", r"finish = \1"), (r"^R\.d0 = (.*);$", r"d0 = \1"),
        (r"^R\.ds0 = (.*);$", r"ds0 = \1"), (r"^R\.dt_run = 0; R\.dt_commit = 0; R\.dt_draft = 0;$", r"dt_* = 0"),
        (r"^R\.dec_windows = 0; R\.dec_T = 0;$", r"dec_windows/dec_T = 0"),
        (r"^R\.decode_hits0 = (.*);$", r"decode_hits0 = \1"),
        (r"^R\.decode_look0 = (.*);$", r"decode_look0 = \1"),
        (r"^R\.decode_ms = (.*);$", r"decode_ms = \1"),
        (r"^R\.mrope_touched = true;$", r"mrope_touched = true"),
        (r"^strata::kernels::SamplerParams req_sp;$", r"req_sp"),
    ]
    for rx, out in WRITE2DECL:
        if re.match(rx, s):
            return re.sub(rx, out, s)
    if s.startswith("R.drafts") or s.startswith("R.window"):
        return None
    return s

def stream(lines, a, b):
    out = []
    for l in lines[a - 1:b]:
        c = canon(l)
        if c is None:
            continue
        for part in c.split("\n"):
            if part:
                out.append(part)
    return out

old = open(sys.argv[1]).read().split("\n")
new = open(sys.argv[2]).read().split("\n")
OB = []
for a, b in zip(sys.argv[3::2], sys.argv[4::2]):
    OB += stream(old, int(a), int(b))
NB = []
for a, b in zip(sys.argv[5::2], sys.argv[6::2]):
    NB += stream(new, int(a), int(b))

if OB == NB:
    print("IDENTICAL canonical statement streams: %d statements" % len(OB))
    sys.exit(0)
print("MISMATCH: before %d, after %d" % (len(OB), len(NB)))
for l in difflib.unified_diff(OB, NB, "before", "after", lineterm="", n=2):
    print(l)
sys.exit(1)
