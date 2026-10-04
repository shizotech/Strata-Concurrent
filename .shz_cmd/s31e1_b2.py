#!/usr/bin/env python3
"""S3.1e-1 phase B: extract prep_request (serve phases 1+2).

Every moved line is kept byte-identical except for an explicit replacement
table; each replacement asserts the original line content first."""
import sys

P = "src/program/generate.cpp"
lines = open(P).read().split("\n")

def L(n):
    return lines[n - 1]

def chk(n, sub):
    if sub not in L(n):
        raise SystemExit("line %d does not contain %r\n  got: %r" % (n, sub, L(n)))

R12 = "            "          # request-body indent

# ------------------------------------------------------------------ region bounds
A, B = 5115, 5617
chk(A, "// ---- S3.1c: the request line, parsed by")
chk(B, "}")
chk(B + 1, "bool cancelled = false;")

DEL = [(5194, 5203), (5213, 5233)]      # SlotGuard block, MropeScope block
chk(5194, "// Whatever happens to this request from here")
chk(5203, "} slot_guard{slot_open,")
chk(5213, "// S3.1d R7, the guard.")
chk(5233, "};")
chk(5234, "if (geni || !mrope_identity) {")

# ------------------------------------------------------------------ replacement table
REP = {}
def rep(n, text):
    REP[n] = text

rep(5117, R12 + "R.rq = strata::program::serve_proto::parse_request(line, proto_def, tagged);\n"
              + R12 + "const strata::program::serve_proto::Request& rq = R.rq;")
chk(5118, "strata::program::serve_proto::parse_request(line, proto_def, tagged);")
rep(5118, "")

rep(5134, R12 + "R.geni = rq.kind == strata::program::serve_proto::Kind::geni;\n"
                + R12 + "const bool& geni = R.geni;")
rep(5135, R12 + "R.max_new = (long long) rq.max_new;\n" + R12 + "const long long& max_new = R.max_new;")
rep(5139, R12 + "R.req_temperature = rq.temperature;\n" + R12 + "const float& req_temperature = R.req_temperature;")
rep(5140, R12 + "R.req_top_p = rq.top_p;\n" + R12 + "const float& req_top_p = R.req_top_p;")
rep(5141, R12 + "R.req_top_k = rq.top_k;\n"
                + R12 + "const int& req_top_k = R.req_top_k;   // the sampler's own default; the sampled path REQUIRES top_k in 1..64")
rep(5142, R12 + "R.req_seed = rq.seed;\n" + R12 + "const unsigned long long& req_seed = R.req_seed;")
rep(5143, R12 + "R.req_min_p = rq.min_p; R.req_penalty_repeat = rq.penalty_repeat;\n"
                + R12 + "const float& req_min_p = R.req_min_p; const float& req_penalty_repeat = R.req_penalty_repeat;")
rep(5144, R12 + "R.req_penalty_freq = rq.penalty_freq; R.req_penalty_present = rq.penalty_present;\n"
                + R12 + "const float& req_penalty_freq = R.req_penalty_freq; const float& req_penalty_present = R.req_penalty_present;")
rep(5145, R12 + "R.req_penalty_last_n = rq.penalty_last_n;\n" + R12 + "const int& req_penalty_last_n = R.req_penalty_last_n;")
rep(5146, R12 + "R.req_cvec = rq.cvec;\n"
                + R12 + "const int& req_cvec = R.req_cvec;   // cvec=0|1: a loaded control vector for this request (on when absent)")
rep(5149, R12 + "R.req_pcie_frac = rq.pcie_frac; R.req_spec_min_p = rq.spec_min_p;\n"
                + R12 + "const double& req_pcie_frac = R.req_pcie_frac; const double& req_spec_min_p = R.req_spec_min_p;")
rep(5150, R12 + "R.emb_path = rq.emb_path;\n" + R12 + "const std::string& emb_path = R.emb_path;")
rep(5151, R12 + "R.ids = rq.ids;\n" + R12 + "std::vector<int64_t>& ids = R.ids;")
rep(5154, R12 + "req_id = rq.id;\n" + R12 + "R.id = rq.id;")
rep(5155, R12 + "bool& slot_open = R.slot_open;")
rep(5204, R12 + "R.n = (int64_t) ids.size();\n" + R12 + "const int64_t n = R.n;")
rep(5336, R12 + "std::array<int64_t, 3>& remote_before = R.remote_before;")
rep(5337, R12 + "std::array<int64_t, 3>& launches_before = R.launches_before;")
rep(5338, R12 + "std::array<uint64_t, 3>& compact_before = R.compact_before;\n"
                + R12 + "std::array<uint64_t, 3>& full_before = R.full_before;")
rep(5341, R12 + "std::array<double, 3>& begin_before = R.begin_before;\n"
                + R12 + "std::array<double, 3>& wait_before = R.wait_before;")
rep(5417, R12 + "R.r0 = Clock::now();\n" + R12 + "const Clock::time_point& r0 = R.r0;")
rep(5419, R12 + "R.loan_bill = loan_totals(pf_parts);\n" + R12 + "const std::array<int64_t, 4>& loan_bill = R.loan_bill;")
rep(5420, R12 + "R.res_uploads0 = res_dirty.uploads(); R.res_skips0 = res_dirty.skipped();\n"
                + R12 + "const int64_t& res_uploads0 = R.res_uploads0; const int64_t& res_skips0 = R.res_skips0;")
rep(5430, R12 + "R.want_cvec = strata::kernels::cvec().loaded() ? req_cvec != 0 : true;\n"
                + R12 + "const bool& want_cvec = R.want_cvec;")
rep(5431, R12 + "int64_t& resume = R.resume;")
rep(5432, R12 + "bool& from_live = R.from_live;")
rep(5518, R12 + "int64_t& reread_to = R.reread_to;   // STRATA_CKPT_REREAD only: read [0, reread_to) again instead of restoring")
rep(5567, R12 + "R.read_from = reread_to > 0 ? 0 : resume;\n" + R12 + "const int64_t& read_from = R.read_from;")
rep(5585, R12 + "strata::kernels::SamplerParams& req_sp = R.req_sp;")
rep(5605, R12 + "R.hist_n = std::min(req_sp.penalty_last_n, kPenaltyWindowCap);\n" + R12 + "const int& hist_n = R.hist_n;")

# ---- exit statements (assert each one is exactly a bare continue / return 1)
for n in [5123, 5128, 5132, 5165, 5211, 5310, 5331, 5396, 5413]:
    chk(n, "continue;")
    assert L(n).strip() == "continue;", L(n)
    ind = L(n)[:len(L(n)) - len(L(n).lstrip())]
    rep(n, ind + "return Prep::rejected;")
for n in [5386, 5469, 5478, 5485, 5558]:
    chk(n, "return 1;")
    assert L(n).strip() == "return 1;", L(n)
    ind = L(n)[:len(L(n)) - len(L(n).lstrip())]
    rep(n, ind + "return Prep::fatal;")
chk(5190, 'if (!serr.empty()) { std::printf("%s\\n", sp_err(serr).c_str()); continue; }')
rep(5190, R12 + 'if (!serr.empty()) { std::printf("%s\\n", sp_err(serr).c_str()); return Prep::rejected; }')
chk(5335, 'if (bad) { std::printf("%s\\n", sp_err("a token id is outside the vocabulary").c_str()); continue; }')
rep(5335, R12 + 'if (bad) { std::printf("%s\\n", sp_err("a token id is outside the vocabulary").c_str()); return Prep::rejected; }')
chk(5411, "if (poisoned) return 1;")
rep(5411, "                        if (poisoned) return Prep::fatal;")

# ------------------------------------------------------------------ build the moved body
dels = set()
for a, b in DEL:
    for i in range(a, b + 1):
        dels.add(i)
body = []
for i in range(A, B + 1):
    if i in dels:
        continue
    body.append(REP.get(i, L(i)))
body.append(R12 + "return Prep::ok;")

# ------------------------------------------------------------------ the new serve-scope declarations
DECL = r'''
        // ==================== S3.1e-1: the resumable request ====================
        //
        // Stage 3's scheduler (S3.1e-2) cannot run a request as one straight-line block: with two
        // conversations in one process it must give each of them ONE step at a time and swap the
        // session between them.  So the request body below is cut into the named steps
        // (`prep_request`, `run_prefill_step`, `finish_prefill`, `run_decode_step`,
        // `finish_request`), and the serial loop calls them in exactly the order the old straight-line
        // code ran.  This is a pure refactor: same order, same captures, same error paths, same
        // stdout bytes.
        //
        // WHY A STRUCT AND NOT A CLASS: `ReqCtx` is only the per-request state the old body kept in
        // its own locals.  It is constructed INSIDE `while (next_line(line))`, so its lifetime and its
        // construction/destruction order are exactly those locals' - nothing about when a conversation
        // image, a slot row or a CUDA buffer exists changes.  The steps are lambdas in the serve scope
        // because that is where every capture already lives (`ver`, `mtp`, `sp`, `conversations`,
        // `slots_reg`, ...); a class would have to be handed all of them.
        //
        // The moved code keeps its ORIGINAL local names as references into `ReqCtx`, so each step's
        // body is the old text verbatim apart from the alias declarations and the exit statements.
        enum class Prep { ok, rejected, fatal };   // rejected: the ERR line is out, take the next line
                                                   // fatal:    the ERR line is out, the engine exits 1
        enum class Step { progressed, finished, cancelled, needs_swap, error, fatal_exit };
        //   progressed  the step did work and there is more of the same step to do
        //   finished    the step is done; move on to the next phase
        //   cancelled   the request was STOPped; fall through to the phase's tail, not an error
        //   needs_swap  RESERVED for S3.1e-2: this step wants the session handed to another slot.
        //               The serial path never returns it and nothing handles it yet.
        //   error       the ERR line is already printed; the caller must `return 1` exactly as today
        //   fatal_exit  #224: a CUDA fault, already flushed; the caller must `std::_Exit(1)`
        struct ReqCtx {
            // ---- phase 1: the parsed request line
            strata::program::serve_proto::Request rq;
            bool geni = false;
            long long max_new = 0;
            float req_temperature = 0.0f, req_top_p = 1.0f, req_min_p = 0.0f;
            float req_penalty_repeat = 1.0f, req_penalty_freq = 0.0f, req_penalty_present = 0.0f;
            int req_top_k = 20, req_penalty_last_n = 0, req_cvec = 1;
            unsigned long long req_seed = 0;
            double req_pcie_frac = 0.0, req_spec_min_p = 0.0;
            std::string emb_path;
            std::vector<int64_t> ids;
            int64_t id = strata::program::serve_proto::kNoId;   // this request's id (== `req_id` while it runs)
            int64_t n = 0;
            bool slot_open = false;        // the registry row exists -> SlotGuard must finish it
            bool mrope_touched = false;    // this request rewrote the one position table
            // per-request metric baselines (phase 5 prints the deltas)
            std::array<int64_t, 3> remote_before{}, launches_before{};
            std::array<uint64_t, 3> compact_before{}, full_before{};
            std::array<double, 3> begin_before{}, wait_before{};
            // ---- phase 2: the resume point, the mount, the sampling dispatch
            Clock::time_point r0{};
            std::array<int64_t, 4> loan_bill{};
            int64_t res_uploads0 = 0, res_skips0 = 0;
            bool want_cvec = true;
            int64_t resume = 0;
            bool from_live = false;
            int64_t reread_to = -1;
            int64_t read_from = 0;
            strata::kernels::SamplerParams req_sp;
            int hist_n = 0;
        };
        // The two iteration guards.  They stay INSTANTIATED in the loop body: their destructors must
        // run at the end of the whole iteration, exactly as the old locals' did.  Only their definitions
        // move out, so a body-local instance can be armed from inside `prep_request` through `ReqCtx`.
        // The values they read at destruction (`slot_open`, `mrope_touched`) are never written after
        // setup, so reading them from the context is the same answer as the old constructor argument.
        struct SlotGuard {
            ReqCtx* ctx = nullptr;
            std::function<void(int64_t)> finish;
            ~SlotGuard() { if (ctx && ctx->slot_open && finish) finish(ctx->id); }
        };
        struct MropeScope {
            std::function<bool()> skip;     // this request IS the mounted slot: nothing to undo
            std::function<void()> restore;
            ~MropeScope() { if (skip && skip()) return; if (restore) restore(); }
        };
        // ---- phases 1 + 2: parse the line, resolve the resume point, mount, dispatch sampling.
        // `Prep::rejected` means the request was refused and its ERR line is already printed - the
        // old `continue` sites, in the same places.
        auto prep_request = [&](ReqCtx& R, const std::string& line) -> Prep {
'''.split("\n")

# ------------------------------------------------------------------ the new loop-body prologue
PRO = r'''
            // S3.1e-1: everything the old body kept as per-request locals.  Constructed here, once per
            // iteration, so its lifetime is those locals' lifetime.
            ReqCtx R;
            // Whatever happens to this request from here - a validation ERR, a body ERR, a normal DONE - its
            // row is finished at the end of the iteration.  A guard rather than a call at every exit: the loop
            // has a dozen `continue`s, and a leaked row would strand an active-slot permit, which under
            // --serve-slots is the resource that bounds everything (§2.3).
            SlotGuard slot_guard{&R, tagged ? std::function<void(int64_t)>(slot_finish) : nullptr};
            // S3.1d R7, the guard.  The block below REWRITES the one host table and uploads it, for a
            // request that may still bail out (a bad embeddings file, a prompt too long for the
            // context, a token outside the vocabulary).  If it does, the slot that is actually mounted
            // is left with somebody else's positions in the device - and it is still mounted, so no
            // hand-over will put them back.  This guard restores the mounted slot's table on the way
            // out whenever the request did not become the mounted one.  On the normal path the swap
            // sets `mounted_id == req_id` and the guard does nothing.
            MropeScope mrope_scope;
            mrope_scope.skip = [&]() { return !swaps_on || !R.mrope_touched || mounted_id == req_id; };
            mrope_scope.restore = [&]() {
                if (mounted_id == strata::program::serve_proto::kNoId) return;
                apply_positions(conv_of(mounted_id));
                if (!mrope_host.empty()) upload_mrope_table();
                mrope_owner = mounted_id;
            };
            const Prep prep = prep_request(R, line);
            if (prep == Prep::rejected) continue;
            if (prep == Prep::fatal) return 1;
            // the request body's own names for the state `prep_request` filled in
            const bool& geni = R.geni;
            const long long& max_new = R.max_new;
            const double& req_spec_min_p = R.req_spec_min_p;
            std::vector<int64_t>& ids = R.ids;
            const int64_t n = R.n;
            const int64_t& resume = R.resume;
            const int64_t& reread_to = R.reread_to;
            const int64_t& read_from = R.read_from;
            const Clock::time_point& r0 = R.r0;
            const std::array<int64_t, 4>& loan_bill = R.loan_bill;
            const int64_t& res_uploads0 = R.res_uploads0;
            const int64_t& res_skips0 = R.res_skips0;
            strata::kernels::SamplerParams& req_sp = R.req_sp;
            const int& hist_n = R.hist_n;
            std::array<int64_t, 3>& remote_before = R.remote_before;
            std::array<int64_t, 3>& launches_before = R.launches_before;
            std::array<uint64_t, 3>& compact_before = R.compact_before;
            std::array<uint64_t, 3>& full_before = R.full_before;
            std::array<double, 3>& begin_before = R.begin_before;
            std::array<double, 3>& wait_before = R.wait_before;
'''.split("\n")
PRO = ["        " + l if l.strip() else l for l in PRO]

# ------------------------------------------------------------------ assemble
tail = ["        };"]
new_region = DECL + body + tail + PRO

out = lines[:A - 1] + new_region + lines[B:]
open(P, "w").write("\n".join(out))
print("phase B: region %d..%d (%d lines) -> %d lines" % (A, B, B - A + 1, len(new_region)))
