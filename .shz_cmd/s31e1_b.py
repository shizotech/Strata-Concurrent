#!/usr/bin/env python3
"""S3.1e-1 phase B: extract prep_request.

Strategy: the moved code keeps its ORIGINAL local names via reference aliases
into ReqCtx, so the body of prep_request is byte-identical to what it replaced
apart from the alias declarations and the exit statements."""
import re, sys

P = "src/program/generate.cpp"
src = open(P).read()
lines = src.split("\n")

def L(n):            # 1-based line n
    return lines[n - 1]

def idx(sub, start=1):
    for i in range(start - 1, len(lines)):
        if sub in lines[i]:
            return i + 1
    raise SystemExit("not found: " + sub)

# ---------------------------------------------------------------- locate the region
body_top = idx('        while (next_line(line)) {')
assert 'if (line == "QUIT") break;' in L(body_top + 1)
# phase 1+2 region: from the S3.1c parse comment to the end of the sampling block
p1 = idx("// ---- S3.1c: the request line, parsed by the one parser that knows both wire forms")
p2 = idx("c.pcie_num = drive.d.pcie_num;", idx("if (swaps_on) {\n".rstrip("\n"), idx("// request line (S3.1e pre-empting a decode) can put it back")))
assert L(p2).strip() == "c.pcie_num = drive.d.pcie_num;"
p2 += 1                                   # the closing `}` of the if (swaps_on) block
assert L(p2).strip() == "}"
assert L(p2 + 1).strip() == "bool cancelled = false;"

# SlotGuard block (comment + struct + instance)
sg_cmt = idx("// Whatever happens to this request from here - a validation ERR, a body ERR, a normal DONE - its")
sg_end = idx("} slot_guard{slot_open,")
# MropeScope block (comment + bool + struct + instance + lambdas)
ms_cmt = idx("// S3.1d R7, the guard.  The block below REWRITES the one host table and uploads it, for a")
ms_end = idx("mrope_owner = mounted_id;") + 1
assert L(ms_end).strip() == "};"

region = lines[p1 - 1:p2]

def cut(ls, a, b):
    """remove 1-based inclusive [a,b] from `ls` (which is indexed against `lines`)"""
    return [x for i, x in enumerate(ls) if not (a <= i + p1 <= b)]

region = cut(region, sg_cmt, sg_end)
region = cut(region, ms_cmt, ms_end)

# ---------------------------------------------------------------- renames inside the region
def sub_all(pat, rep, cnt_expected=None):
    global region
    n = 0
    out = []
    for l in region:
        n2, c = re.subn(pat, rep, l)
        n += c
        out.append(n2)
    region = out
    if cnt_expected is not None and n != cnt_expected:
        raise SystemExit("rename %r -> %r matched %d times, expected %s" % (pat, rep, n, cnt_expected))
    return n

# declarations that become ReqCtx writes + a same-named alias
sub_all(r"^            const bool geni = rq\.kind == strata::program::serve_proto::Kind::geni;$",
        "            R.geni = rq.kind == strata::program::serve_proto::Kind::geni;\n"
        "            const bool& geni = R.geni;")
sub_all(r"^            const long long max_new = \(long long\) rq\.max_new;$",
        "            R.max_new = (long long) rq.max_new;\n"
        "            const long long& max_new = R.max_new;")
for nm in ["temperature", "top_p", "top_k", "seed", "min_p", "penalty_repeat", "penalty_freq",
           "penalty_present", "penalty_last_n", "cvec", "pcie_frac", "spec_min_p"]:
    pass
sub_all(r"^            const float req_temperature = rq\.temperature;$",
        "            R.req_temperature = rq.temperature;\n            const float& req_temperature = R.req_temperature;")
sub_all(r"^            const float req_top_p = rq\.top_p;$",
        "            R.req_top_p = rq.top_p;\n            const float& req_top_p = R.req_top_p;")
sub_all(r"^            const int req_top_k = rq\.top_k;",
        "            R.req_top_k = rq.top_k;\n            const int& req_top_k = R.req_top_k;   // the sampler's own default; the sampled path REQUIRES top_k in 1..64")
sub_all(r"^            const unsigned long long req_seed = rq\.seed;$",
        "            R.req_seed = rq.seed;\n            const unsigned long long& req_seed = R.req_seed;")
sub_all(r"^            const float req_min_p = rq\.min_p, req_penalty_repeat = rq\.penalty_repeat;$",
        "            R.req_min_p = rq.min_p; R.req_penalty_repeat = rq.penalty_repeat;\n"
        "            const float& req_min_p = R.req_min_p; const float& req_penalty_repeat = R.req_penalty_repeat;")
sub_all(r"^            const float req_penalty_freq = rq\.penalty_freq, req_penalty_present = rq\.penalty_present;$",
        "            R.req_penalty_freq = rq.penalty_freq; R.req_penalty_present = rq.penalty_present;\n"
        "            const float& req_penalty_freq = R.req_penalty_freq; const float& req_penalty_present = R.req_penalty_present;")
sub_all(r"^            const int req_penalty_last_n = rq\.penalty_last_n;$",
        "            R.req_penalty_last_n = rq.penalty_last_n;\n            const int& req_penalty_last_n = R.req_penalty_last_n;")
sub_all(r"^            const int req_cvec = rq\.cvec;",
        "            R.req_cvec = rq.cvec;\n            const int& req_cvec = R.req_cvec;   // cvec=0|1: a loaded control vector for this request (on when absent)")
sub_all(r"^            const double req_pcie_frac = rq\.pcie_frac, req_spec_min_p = rq\.spec_min_p;$",
        "            R.req_pcie_frac = rq.pcie_frac; R.req_spec_min_p = rq.spec_min_p;\n"
        "            const double& req_pcie_frac = R.req_pcie_frac; const double& req_spec_min_p = R.req_spec_min_p;")
sub_all(r"^            const std::string emb_path = rq\.emb_path;$",
        "            R.emb_path = rq.emb_path;\n            const std::string& emb_path = R.emb_path;")
sub_all(r"^            std::vector<int64_t> ids = rq\.ids;$",
        "            R.ids = rq.ids;\n            std::vector<int64_t>& ids = R.ids;")
sub_all(r"^            req_id = rq\.id;$",
        "            req_id = rq.id;\n            R.id = rq.id;")
sub_all(r"^            bool slot_open = false;$", "            bool& slot_open = R.slot_open;")
sub_all(r"^            const int64_t n = \(int64_t\) ids\.size\(\);$",
        "            R.n = (int64_t) ids.size();\n            const int64_t n = R.n;")
sub_all(r"^            std::array<int64_t, 3> remote_before\{\};$", "            std::array<int64_t, 3>& remote_before = R.remote_before;")
sub_all(r"^            std::array<int64_t, 3> launches_before\{\};$", "            std::array<int64_t, 3>& launches_before = R.launches_before;")
sub_all(r"^            std::array<uint64_t, 3> compact_before\{\}, full_before\{\};$",
        "            std::array<uint64_t, 3>& compact_before = R.compact_before;\n            std::array<uint64_t, 3>& full_before = R.full_before;")
sub_all(r"^            std::array<double, 3> begin_before\{\}, wait_before\{\};$",
        "            std::array<double, 3>& begin_before = R.begin_before;\n            std::array<double, 3>& wait_before = R.wait_before;")
sub_all(r"^            const Clock::time_point r0 = Clock::now\(\);$",
        "            R.r0 = Clock::now();\n            const Clock::time_point& r0 = R.r0;")
sub_all(r"^            const std::array<int64_t, 4> loan_bill = loan_totals\(pf_parts\);$",
        "            R.loan_bill = loan_totals(pf_parts);\n            const std::array<int64_t, 4>& loan_bill = R.loan_bill;")
sub_all(r"^            const int64_t res_uploads0 = res_dirty\.uploads\(\), res_skips0 = res_dirty\.skipped\(\);$",
        "            R.res_uploads0 = res_dirty.uploads(); R.res_skips0 = res_dirty.skipped();\n"
        "            const int64_t& res_uploads0 = R.res_uploads0; const int64_t& res_skips0 = R.res_skips0;")
sub_all(r"^            const bool want_cvec = strata::kernels::cvec\(\)\.loaded\(\) \? req_cvec != 0 : true;$",
        "            R.want_cvec = strata::kernels::cvec().loaded() ? req_cvec != 0 : true;\n            const bool& want_cvec = R.want_cvec;")
sub_all(r"^            int64_t resume = 0;$", "            int64_t& resume = R.resume;")
sub_all(r"^            bool from_live = false;$", "            bool& from_live = R.from_live;")
sub_all(r"^            int64_t reread_to = -1;", "            int64_t& reread_to = R.reread_to;   // STRATA_CKPT_REREAD only: read [0, reread_to) again instead of restoring")
sub_all(r"^            const int64_t read_from = reread_to > 0 \? 0 : resume;$",
        "            R.read_from = reread_to > 0 ? 0 : resume;\n            const int64_t& read_from = R.read_from;")
sub_all(r"^            const int hist_n = std::min\(req_sp\.penalty_last_n, kPenaltyWindowCap\);$",
        "            R.hist_n = std::min(req_sp.penalty_last_n, kPenaltyWindowCap);\n            const int& hist_n = R.hist_n;")
sub_all(r"^            strata::kernels::SamplerParams req_sp;$", "            strata::kernels::SamplerParams& req_sp = R.req_sp;")
sub_all(r"^            const bool want_cvec_now", "            const bool want_cvec_now")   # untouched

# the request line parse: keep the name `rq` but store it in R
sub_all(r"^            const strata::program::serve_proto::Request rq =$",
        "            R.rq = strata::program::serve_proto::parse_request(line, proto_def, tagged);\n"
        "            const strata::program::serve_proto::Request& rq = R.rq;\n"
        "            if (false) {   // the original initialiser, kept out of the way\n")
# remove the original second line of the parse statement
sub_all(r"^                strata::program::serve_proto::parse_request\(line, proto_def, tagged\);$",
        "            }   // (end of the disabled original initialiser)")

# exit statements
sub_all(r"^                continue;$", "                return Prep::rejected;")
sub_all(r"^            continue;$", "            return Prep::rejected;")
sub_all(r"^                    return 1;$", "                    return Prep::fatal;")
sub_all(r"^                        return 1;$", "                        return Prep::fatal;")
sub_all(r"^            if \(bad\) \{ std::printf\(\"%s\\\\n\", sp_err\(\"a token id is outside the vocabulary\"\)\.c_str\(\)\); continue; \}$",
        "            if (bad) { std::printf(\"%s\\n\", sp_err(\"a token id is outside the vocabulary\").c_str()); return Prep::rejected; }")
sub_all(r"^                if \(!serr\.empty\(\)\) \{ std::printf\(\"%s\\\\n\", sp_err\(serr\)\.c_str\(\)\); continue; \}$",
        "                if (!serr.empty()) { std::printf(\"%s\\n\", sp_err(serr).c_str()); return Prep::rejected; }")
sub_all(r"^                        if \(poisoned\) return 1;$", "                        if (poisoned) return Prep::fatal;")

open("/tmp/region.txt", "w").write("\n".join(region))
print("region lines:", len(region))
print("p1=%d p2=%d body_top=%d" % (p1, p2, body_top))
