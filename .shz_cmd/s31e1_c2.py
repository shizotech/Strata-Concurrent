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
def ind(n):
    l = L(n); return l[:len(l) - len(l.lstrip())]

R12 = "            "
A, B = 5744, 5875
chk(A, "bool cancelled = false;")
chk(B, 'if (cancelled) finish = "cancel";')
chk(B + 1, "while (!cancelled && produced_n < max_new) {")

REP = {}
def rep(n, text): REP[n] = text
def dele(n): REP[n] = ""

rep(A, R12 + "R.cancelled = false;")
dele(A + 1)                                        # tr("prompt start") moves into plan

rep(5750, R12 + "int64_t& turn_at = R.turn_at;")
rep(5759, R12 + "int64_t& root_at = R.root_at;")
rep(5766, R12 + "int64_t& at = R.at;")
dele(5767)                                         # for (const int64_t to : {...})
dele(5768)                                         # if (to <= at) continue;
dele(5808)                                         # the for's closing brace

for n in [5773, 5777, 5806]:
    chk(n, "return 1;"); assert L(n).strip() == "return 1;", L(n)
    rep(n, ind(n) + "return Step::error;")
chk(5796, "std::_Exit(1);")
rep(5796, ind(5796) + "return Step::fatal_exit;")
chk(5798, "return 1;"); assert L(5798).strip() == "return 1;"
rep(5798, ind(5798) + "return Step::error;")
chk(5800, "cancelled = true;")
rep(5800, ind(5800) + "cancelled = true;   // stopped while reading the prompt: refill the lent slots below, then DONE cancel")
chk(5801, "break;"); assert L(5801).strip() == "break;"
rep(5801, ind(5801) + "return Step::cancelled;")

chk(5809, "if (!refill(err)) {")
chk(5811, "return 1;"); assert L(5811).strip() == "return 1;"
rep(5811, ind(5811) + "return Step::error;")
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
rep(5855, R12 + "consumed.clear(); consumed.reserve((size_t) (n + max_new + S));")
chk(5857, 'const char* finish = "length";')
rep(5857, R12 + 'R.finish = "length";')
chk(5858, "const Clock::time_point d0 = Clock::now();")
rep(5858, R12 + "R.d0 = Clock::now();")
for n in range(5860, 5870):                        # dec_timing / DecSnap / dec_snap move to serve scope
    dele(n)
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
rep(5875, R12 + 'if (cancelled) finish = "cancel";')

def take(a, b, dedent=0):
    out = []
    for i in range(a, b + 1):
        if REP.get(i, None) == "":
            continue
        l = REP.get(i, L(i))
        if dedent and l.strip():
            assert l.startswith(" " * dedent), (i, repr(l))
            l = l[dedent:]
        out.append(l)
    return out

# ---------------------------------------------------------------- plan_prompt_segments
plan = [
    "        // ---- phase 3 prologue: decide which prompt segments this request reads.",
    "        // S3.1e-1: split out of the straight-line body so S3.1e-2 can plan a request's reads when it",
    "        // admits it and then drive them one step at a time.  Same four segment ends, same order.",
    "        auto plan_prompt_segments = [&](ReqCtx& R) {",
    R12 + "std::vector<int64_t>& ids = R.ids;",
    R12 + "const int64_t n = R.n;",
    R12 + "const int64_t& resume = R.resume;",
    R12 + "const int64_t& read_from = R.read_from;",
    R12 + "const int64_t& reread_to = R.reread_to;",
    R12 + "int64_t& turn_at = R.turn_at;",
    R12 + "int64_t& root_at = R.root_at;",
    R12 + "int64_t& at = R.at;",
    R12 + 'tr("prompt start", n - 1);',
]
for i in range(A + 2, 5767):
    l = REP.get(i, L(i))
    if l.strip() in ("int64_t& turn_at = R.turn_at;", "int64_t& root_at = R.root_at;", "int64_t& at = R.at;"):
        continue
    plan.append(l)
plan.append(R12 + "R.seg = {reread_to, root_at, turn_at, n - 1};")
plan.append(R12 + "R.seg_i = 0;")
plan.append("        };")

# ---------------------------------------------------------------- run_prefill_step
step = [
    "        // ---- phase 3: read ONE prompt segment.  0.1.30 ran the four segments back to back in one",
    "        // `for`; the serial loop now calls this until it stops reporting `progressed`, which is the",
    "        // same lend / read / refill / checkpoint calls in the same order.  S3.1e-2 interleaves them.",
    "        auto run_prefill_step = [&](ReqCtx& R) -> Step {",
    R12 + "std::vector<int64_t>& ids = R.ids;",
    R12 + "const int64_t& turn_at = R.turn_at;",
    R12 + "const int64_t& root_at = R.root_at;",
    R12 + "int64_t& at = R.at;",
    R12 + "bool& cancelled = R.cancelled;",
    R12 + "// the next segment with something to read - 0.1.30's `if (to <= at) continue;`",
    R12 + "int64_t to = -1;",
    R12 + "bool have = false;",
    R12 + "while (R.seg_i < R.seg.size()) {",
    R12 + "    const int64_t t = R.seg[R.seg_i++];",
    R12 + "    if (t > at) { to = t; have = true; break; }",
    R12 + "}",
    R12 + "if (!have) return Step::finished;",
]
step += take(5769, 5807, dedent=4)
step.append(R12 + "return Step::progressed;")
step.append("        };")

# ---------------------------------------------------------------- finish_prefill
fin = [
    "        // ---- phase 3 tail + phase 4 prologue: give the prompt loan back, report the prompt, and",
    "        // set the decode loop up.  Runs for a finished prompt AND for a cancelled one, exactly as",
    "        // 0.1.30's straight-line code did (its `break` fell through to here).",
    "        auto finish_prefill = [&](ReqCtx& R) -> Step {",
    R12 + "std::vector<int64_t>& ids = R.ids;",
    R12 + "const int64_t n = R.n;",
    R12 + "const long long& max_new = R.max_new;",
    R12 + "const Clock::time_point& r0 = R.r0;",
    R12 + "const std::array<int64_t, 4>& loan_bill = R.loan_bill;",
    R12 + "const int64_t& res_uploads0 = R.res_uploads0;",
    R12 + "const int64_t& res_skips0 = R.res_skips0;",
    R12 + "const int64_t& resume = R.resume;",
    R12 + "bool& cancelled = R.cancelled;",
    R12 + "const char*& finish = R.finish;",
]
fin += take(5809, 5875)
fin.append(R12 + "return Step::finished;")
fin.append("        };")

# ---------------------------------------------------------------- serve-scope additions
DEC = [
    "",
    "        // S3.1e-1: the decode-timing snapshot moved out of the request body with the decode step, so",
    "        // `ReqCtx` can hold one.  Same captures (`ver`, `drive`), same static flag, same fields.",
    '        static const bool dec_timing = std::getenv("STRATA_DECODE_TIMING") != nullptr;',
    "        struct DecSnap {",
    "            double wait, pool, host, plan, actq, jobs, run;",
    "            int64_t misses, entries, hits, pcie;",
    "        };",
    "        auto dec_snap = [&]() {",
    "            return DecSnap{ver.ms_wait, ver.ms_pool, ver.ms_host, drive.d.ms_plan, drive.d.ms_actq, drive.d.ms_jobs,",
    "                           drive.d.ms_run, drive.d.multi_misses, drive.d.multi_entries, drive.d.cache_hits,",
    "                           drive.d.pcie_experts};",
    "        };",
]
ADD = [
    "            // ---- phase 3: the prompt segments and how far the read has got",
    "            int64_t turn_at = -1, root_at = -1;",
    "            std::array<int64_t, 4> seg{{-1, -1, -1, -1}};   // the four segment ends 0.1.30 iterated",
    "            size_t seg_i = 0;",
    "            int64_t at = 0;",
    "            bool cancelled = false;",
    "            double prompt_ms = 0.0;",
    "            // ---- phase 4: the decode loop's state (`finish_prefill` sets it up, `run_decode_step`",
    "            // advances it one verify window at a time).  The vectors are sized once per request, never",
    "            // per token, so the decode hot path allocates nothing new.",
    "            int64_t p = 0;",
    "            int32_t x = 0;",
    "            std::vector<int32_t> drafts, window, outv, sbuf, consumed;",
    "            std::vector<float> dprob;",
    "            bool first_window = true;",
    "            int64_t produced_n = 0, sfx_windows = 0, sfx_drafts = 0, sfx_ok = 0;",
    "            int64_t draft_offered = 0, draft_accepted = 0;",
    '            const char* finish = "length";',
    "            Clock::time_point d0{};",
    "            DecSnap ds0{};",
    "            double dt_run = 0, dt_commit = 0, dt_draft = 0;",
    "            int64_t dec_windows = 0, dec_T = 0;",
    "            int64_t decode_hits0 = 0, decode_look0 = 0;",
    "            double decode_ms = 0.0;",
]

# ---------------------------------------------------------------- the loop-body glue
GLUE = [
    "            // ---- S3.1e-1 phase 3: the prompt, one segment per step (0.1.30: one straight-line loop)",
    "            plan_prompt_segments(R);",
    "            Step st = Step::progressed;",
    "            while ((st = run_prefill_step(R)) == Step::progressed) {}",
    "            if (st == Step::error) return 1;",
    "            if (st == Step::fatal_exit) {   // #224: a CUDA fault poisons the context for the whole process",
    "                std::fflush(stdout);",
    "                std::fflush(stderr);",
    "                std::_Exit(1);",
    "            }",
    "            if (finish_prefill(R) == Step::error) return 1;",
    "            // the request body's names for the phase-3/4 state the steps now own",
    "            bool& cancelled = R.cancelled;",
    "            const double& prompt_ms = R.prompt_ms;",
    "            int64_t& p = R.p;",
    "            int32_t& x = R.x;",
    "            std::vector<int32_t>& drafts = R.drafts;",
    "            std::vector<int32_t>& window = R.window;",
    "            std::vector<int32_t>& outv = R.outv;",
    "            std::vector<float>& dprob = R.dprob;",
    "            std::vector<int32_t>& sbuf = R.sbuf;",
    "            std::vector<int32_t>& consumed = R.consumed;",
    "            bool& first_window = R.first_window;",
    "            int64_t& produced_n = R.produced_n;",
    "            int64_t& sfx_windows = R.sfx_windows;",
    "            int64_t& sfx_drafts = R.sfx_drafts;",
    "            int64_t& sfx_ok = R.sfx_ok;",
    "            int64_t& draft_offered = R.draft_offered;",
    "            int64_t& draft_accepted = R.draft_accepted;",
    "            const char*& finish = R.finish;",
    "            const Clock::time_point& d0 = R.d0;",
    "            const DecSnap& ds0 = R.ds0;",
    "            double& dt_run = R.dt_run;",
    "            double& dt_commit = R.dt_commit;",
    "            double& dt_draft = R.dt_draft;",
    "            int64_t& dec_windows = R.dec_windows;",
    "            int64_t& dec_T = R.dec_T;",
    "            const int64_t& decode_hits0 = R.decode_hits0;",
    "            const int64_t& decode_look0 = R.decode_look0;",
]

out = lines[:A - 1] + GLUE + lines[B:]

i = next(k for k, l in enumerate(out) if l.strip() == "struct ReqCtx {")
out = out[:i] + DEC + out[i:]
j = next(k for k, l in enumerate(out) if l.strip() == "int hist_n = 0;")
out = out[:j + 1] + ADD + out[j + 1:]
k = next(i2 for i2, l in enumerate(out) if l.strip() == "return Prep::ok;")
k2 = next(i2 for i2 in range(k, len(out)) if out[i2] == "        };")
out = out[:k2 + 1] + plan + step + fin + out[k2 + 1:]

open(P, "w").write("\n".join(out))
print("phase C done")
