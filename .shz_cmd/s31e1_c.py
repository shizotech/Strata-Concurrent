#!/usr/bin/env python3
"""S3.1e-1 phase C: extract the prompt-segment loop into plan_prompt_segments /
run_prefill_step / finish_prefill.  Same technique as phase B: the moved code
keeps its original local names through references into ReqCtx."""
import sys

P = "src/program/generate.cpp"
lines = open(P).read().split("\n")

def L(n): return lines[n - 1]
def chk(n, sub):
    if sub not in L(n):
        raise SystemExit("line %d does not contain %r\n  got: %r" % (n, sub, L(n)))

R12 = "            "
A, B = 5744, 5875
chk(A, "bool cancelled = false;")
chk(B, 'if (cancelled) finish = "cancel";')
chk(B + 1, "while (!cancelled && produced_n < max_new) {")

# ---------------------------------------------------------------- replacement table
REP = {}
def rep(n, text): REP[n] = text

rep(A, R12 + "R.cancelled = false;")
chk(A + 1, 'tr("prompt start", n - 1);')
rep(A + 1, "")                                   # moves into plan_prompt_segments

# turn_at / root_at / at become ReqCtx members, aliased so the code is verbatim
rep(5750, R12 + "int64_t& turn_at = R.turn_at;")
rep(5759, R12 + "int64_t& root_at = R.root_at;")
rep(5766, R12 + "int64_t& at = R.at;")
chk(5767, "for (const int64_t to : {reread_to, root_at, turn_at, n - 1}) {")
rep(5767, R12 + "// ---- one segment (S3.1e-1): 0.1.30 ran these four back to back; the serial loop now")
rep(5768, R12 + "// calls `run_prefill_step` until it stops reporting `progressed`.")
# the whole for-loop body is re-indented by -4 when it moves; handled below by markers
chk(5768, "if (to <= at) continue;")
rep(5768, R12 + "R.seg = {reread_to, root_at, turn_at, n - 1};")
chk(5808, "}")
rep(5808, "")

# exit statements inside the segment loop
for n in [5773, 5777, 5806]:
    chk(n, "return 1;"); assert L(n).strip() == "return 1;", L(n)
    rep(n, L(n)[:len(L(n)) - len(L(n).lstrip())] + "return Step::error;")
chk(5796, "std::_Exit(1);")
rep(5796, "                            return Step::fatal_exit;")
chk(5798, "return 1;"); assert L(5798).strip() == "return 1;"
rep(5798, "                        return Step::error;")
chk(5800, "cancelled = true;")
rep(5800, R12 * 3 + "cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel")
chk(5801, "break;")
rep(5801, "                        return Step::cancelled;")

# finish_prefill starts here
chk(5809, "if (!refill(err)) {")
chk(5811, "return 1;"); assert L(5811).strip() == "return 1;"
rep(5811, "                return Step::error;")
chk(5827, "const double prompt_ms =")
rep(5827, R12 + "R.prompt_ms = std::chrono::duration<double, std::milli>(Clock::now() - r0).count();")
chk(5840, "int64_t p = n - 1;")
rep(5840, R12 + "R.p = n - 1;")
chk(5841, "int32_t x = (int32_t) ids[(size_t) (n - 1)];")
rep(5841, R12 + "R.x = (int32_t) ids[(size_t) (n - 1)];")
chk(5842, "std::vector<int32_t> drafts((size_t) S, 0), window((size_t) S), outv((size_t) S);")
rep(5842, R12 + "R.drafts.assign((size_t) S, 0); R.window.assign((size_t) S, 0); R.outv.assign((size_t) S, 0);")
chk(5843, "std::vector<float> dprob((size_t) S, 0.0f);")
rep(5843, R12 + "R.dprob.assign((size_t) S, 0.0f);")
chk(5844, "std::vector<int32_t> sbuf((size_t) S, 0);")
rep(5844, R12 + "R.sbuf.assign((size_t) S, 0);")
chk(5850, "bool first_window = true;")
rep(5850, R12 + "R.first_window = true;")
chk(5851, "int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;")
rep(5851, R12 + "R.produced_n = 0; R.sfx_windows = 0; R.sfx_drafts = 0; R.sfx_ok = 0;")
chk(5852, "int64_t draft_offered = 0, draft_accepted = 0;")
rep(5852, R12 + "R.draft_offered = 0; R.draft_accepted = 0;")
chk(5854, "std::vector<int32_t> consumed;")
rep(5854, R12 + "std::vector<int32_t>& consumed = R.consumed;")
chk(5855, "consumed.reserve(")
rep(5855, R12 + "consumed.clear();")
chk(5857, 'const char* finish = "length";')
rep(5857, R12 + 'R.finish = "length";')
chk(5858, "const Clock::time_point d0 = Clock::now();")
rep(5858, R12 + "R.d0 = Clock::now();")
chk(5860, "static const bool dec_timing =")
rep(5860, "")
chk(5861, "struct DecSnap {")
chk(5864, "};")
rep(5861, ""); rep(5862, ""); rep(5863, ""); rep(5864, "")
chk(5865, "auto dec_snap = [&]() {")
chk(5869, "};")
rep(5865, ""); rep(5866, ""); rep(5867, ""); rep(5868, ""); rep(5869, "")
chk(5870, "const DecSnap ds0 = dec_snap();")
rep(5870, R12 + "R.ds0 = dec_snap();")
chk(5871, "double dt_run = 0, dt_commit = 0, dt_draft = 0;")
rep(5871, R12 + "R.dt_run = 0; R.dt_commit = 0; R.dt_draft = 0;")
chk(5872, "int64_t dec_windows = 0, dec_T = 0;")
rep(5872, R12 + "R.dec_windows = 0; R.dec_T = 0;")
chk(5873, "const int64_t decode_hits0 = drive.d.cache_hits;")
rep(5873, R12 + "R.decode_hits0 = drive.d.cache_hits;")
chk(5874, "const int64_t decode_look0 =")
rep(5874, R12 + "R.decode_look0 = drive.d.cache_hits + drive.d.cache_admitted + drive.d.cache_refused;")
chk(5875, 'if (cancelled) finish = "cancel";')
rep(5875, R12 + 'if (R.cancelled) R.finish = "cancel";')

# ---------------------------------------------------------------- build the three pieces
dels = set()
for i in range(A, B + 1):
    if i in REP and REP[i] == "":
        dels.add(i)

def take(a, b):
    """lines a..b with the deletions and replacements applied"""
    out = []
    for i in range(a, b + 1):
        if i in dels:
            continue
        out.append(REP.get(i, L(i)))
    return out

# 1. plan_prompt_segments: A+1 (tr) .. 5766 (at)
plan_body = take(A + 2, 5766)
plan = []
plan.append("        // ---- phase 3 prologue: which prompt segments this request reads.")
plan.append("        // S3.1e-1: split out of the straight-line body so S3.1e-2 can plan a request's reads at")
plan.append("        // admit time and then drive them one step at a time.  Same four segment ends, same order.")
plan.append("        auto plan_prompt_segments = [&](ReqCtx& R) {")
plan.append("            std::vector<int64_t>& ids = R.ids;")
plan.append("            const int64_t n = R.n;")
plan.append("            const int64_t& resume = R.resume;")
plan.append("            const int64_t& read_from = R.read_from;")
plan.append("            const int64_t& reread_to = R.reread_to;")
plan.append("            int64_t& turn_at = R.turn_at;")
plan.append("            int64_t& root_at = R.root_at;")
plan.append("            int64_t& at = R.at;")
plan.append(R12 + 'tr("prompt start", n - 1);')
for l in plan_body:
    if l.strip() in ("int64_t& turn_at = R.turn_at;", "int64_t& root_at = R.root_at;", "int64_t& at = R.at;"):
        continue
    plan.append(l)
plan.append(R12 + "R.seg_i = 0;")
plan.append("        };")

# 2. run_prefill_step: the segment loop body 5769..5807, dedented by 4
seg_body = []
for i in range(5769, 5808):
    if i in dels:
        continue
    l = REP.get(i, L(i))
    if l.startswith("                "):
        l = l[4:]
    elif l.strip() and not l.startswith("            "):
        raise SystemExit("unexpected indent at %d: %r" % (i, L(i)))
    if l.strip():
        l = l[4:] if l.startswith("            ") else l
    seg_body.append(l)
step = []
step.append("        // ---- phase 3: read ONE prompt segment.  0.1.30 ran the four back to back; the serial")
step.append("        // loop calls this until it stops reporting `progressed`, which is the same sequence of")
step.append("        // lend / read / refill / checkpoint calls in the same order.")
step.append("        auto run_prefill_step = [&](ReqCtx& R) -> Step {")
step.append("            std::vector<int64_t>& ids = R.ids;")
step.append("            const int64_t& turn_at = R.turn_at;")
step.append("            const int64_t& root_at = R.root_at;")
step.append("            int64_t& at = R.at;")
step.append("            bool& cancelled = R.cancelled;")
step.append(R12 + "// the next segment with something to read - 0.1.30's `if (to <= at) continue;`")
step.append(R12 + "int64_t to = -1;")
step.append(R12 + "bool have = false;")
step.append(R12 + "while (R.seg_i < 4) {")
step.append(R12 + "    const int64_t t = R.seg[R.seg_i++];")
step.append(R12 + "    if (t > at) { to = t; have = true; break; }")
step.append(R12 + "}")
step.append(R12 + "if (!have) return Step::finished;")
step.extend(seg_body)
step.append(R12 + "return Step::progressed;")
step.append("        };")

# 3. finish_prefill: 5809..5875
fin = []
fin.append("        // ---- phase 3 tail + phase 4 prologue: give the loan back, report the prompt, and set up")
fin.append("        // the decode loop.  Runs for a finished prompt AND for a cancelled one, exactly as 0.1.30's")
fin.append("        // straight-line code did (the `break` out of the segment loop fell through to this).")
fin.append("        auto finish_prefill = [&](ReqCtx& R) -> Step {")
fin.append("            std::vector<int64_t>& ids = R.ids;")
fin.append("            const int64_t n = R.n;")
fin.append("            const long long& max_new = R.max_new;")
fin.append("            const Clock::time_point& r0 = R.r0;")
fin.append("            const std::array<int64_t, 4>& loan_bill = R.loan_bill;")
fin.append("            const int64_t& res_uploads0 = R.res_uploads0;")
fin.append("            const int64_t& res_skips0 = R.res_skips0;")
fin.append("            const int64_t& resume = R.resume;")
for i in range(5809, 5876):
    if i in dels:
        continue
    fin.append(REP.get(i, L(i)))
fin.append(R12 + "return Step::finished;")
fin.append("        };")

# ---------------------------------------------------------------- ReqCtx additions
ADD = r'''
            // ---- phase 3: the prompt segments and how far the read has got
            int64_t turn_at = -1, root_at = -1;
            std::array<int64_t, 4> seg{{-1, -1, -1, -1}};   // the four segment ends 0.1.30 iterated
            size_t seg_i = 0;
            int64_t at = 0;
            bool cancelled = false;
            double prompt_ms = 0.0;
            // ---- phase 4: the decode loop's state (set up by `finish_prefill`, advanced one window per
            // `run_decode_step`).  The vectors are sized once per request, never per token.
            int64_t p = 0;
            int32_t x = 0;
            std::vector<int32_t> drafts, window, outv, sbuf, consumed;
            std::vector<float> dprob;
            bool first_window = true;
            int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;
            int64_t draft_offered = 0, draft_accepted = 0;
            const char* finish = "length";
            Clock::time_point d0{};
            DecSnap ds0{};
            double dt_run = 0, dt_commit = 0, dt_draft = 0;
            int64_t dec_windows = 0, dec_T = 0;
            int64_t decode_hits0 = 0, decode_look0 = 0;
            double decode_ms = 0.0;
'''.split("\n")

DEC = r'''
        // STRATA_DECODE_TIMING=1: where a request's decode time goes (one line per request)
        static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;
        struct DecSnap {
            double wait, pool, host, plan, actq, jobs, run;
            int64_t misses, entries, hits, pcie;
        };
        auto dec_snap = [&]() {
            return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,
                           drive.d.ms_run, drive.d.multi_misses, drive.d.multi_entries, drive.d.cache_hits,
                           drive.d.pcie_experts};
        };
'''.split("\n")

# ---------------------------------------------------------------- the new loop-body region
GLUE = r'''
            // ---- S3.1e-1 phase 3: the prompt, one segment per step (0.1.30: one straight-line loop)
            plan_prompt_segments(R);
            Step st = Step::progressed;
            while ((st = run_prefill_step(R)) == Step::progressed) {}
            if (st == Step::error) return 1;
            if (st == Step::fatal_exit) {   // #224: a CUDA fault poisons the context for the whole process
                std::fflush(stdout);
                std::fflush(stderr);
                std::_Exit(1);
            }
            if (finish_prefill(R) == Step::error) return 1;
'''.split("\n")

# ---------------------------------------------------------------- assemble
out = lines[:A - 1] + GLUE + lines[B + 1:]

# insert DEC + ADD before `struct ReqCtx {`
def insert_before(ls, anchor, block):
    i = next(k for k, l in enumerate(ls) if l.strip() == anchor)
    return ls[:i] + block + ls[i:]

out = insert_before(out, "struct ReqCtx {", DEC)
# ADD goes after the last ReqCtx field line (`int hist_n = 0;`)
j = next(k for k, l in enumerate(out) if l.strip() == "int hist_n = 0;")
out = out[:j + 1] + ADD + out[j + 1:]
# plan/step/finish go right after the `};` that closes prep_request.  Find prep_request's end.
k = next(i for i, l in enumerate(out) if l.strip() == "return Prep::ok;")
k2 = next(i for i in range(k, len(out)) if out[i] == "        };")
out = out[:k2 + 1] + plan + step + fin + out[k2 + 1:]

open(P, "w").write("\n".join(out))
print("phase C done")
