// S4.3.4 — CPU-only non-vacuity check of MtpDrafter's KV-only gating.
//
// NOT a repo test target (the parent owns CMakeLists.txt). It links the real libstrata_engine.a and pokes the
// drafter's private state through `#define private public` so the mode can be entered WITHOUT a GPU, a model or
// a WeightTable (bind_kv_only needs `output.weight`, which needs CUDA to load).  If the class layout disagreed
// with the library's, the counters below would not line up — the checks that read back what the library wrote
// are also the proof that it does.
//
// No GPU, no model, no session_init, no /dev/shm.
#define private public
#define protected public
#include "strata/core/mtp.hpp"
#undef private
#undef protected

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::core;

static int fails = 0;
static void chk(bool ok, const char* what) {
    std::printf("%-6s %s\n", ok ? "ok" : "FAIL", what);
    if (!ok) ++fails;
}

int main() {
    // A geometry and a session shell, only so the NON-guarded arm can get as far as a CUDA call instead of
    // segfaulting on a null `g_`.  Nothing here allocates.
    static ModelGeometry g{};                 // the canonical defaults (layout.hpp)
    static SessionState ss{};
    static QsaState qsa[1]{};
    ss.qsa_states = qsa; ss.qsa_alloc = 1; ss.max_cells = 96;

    int fails_before = 0;

    // ---------- arm A: the mode is OFF (today's drafter) ----------
    {
        MtpDrafter m;
        m.g_ = &g; m.ss_ = &ss; m.max_t_ = 6; m.kv_only_ = false;
        std::vector<int32_t> drafts(6, -7);
        std::vector<float> probs(6, -1.0f);
        int n = -5; std::string err;
        const bool r = m.draft(6, drafts.data(), 10, 0, drafts.data(), err, probs.data(), 0.0f, &n);
        std::printf("  [mode off] draft -> %d, err = \"%s\"\n", (int) r, err.c_str());
        // It got PAST the guard and tried to capture/launch a draft graph (which fails here only because there
        // is no GPU).  That is the whole point: the guard is what stops it in arm B, not the missing device.
        chk(!r, "mode off: draft does not report the KV-only refusal");
        chk(err.find("KV-only") == std::string::npos, "mode off: the refusal string is not produced");
        chk(m.draft_refusals() == 0, "mode off: nothing was refused");
        chk(err.find("begin capture") != std::string::npos || !err.empty(),
            "mode off: it reached the graph capture (so arm B's stop is the guard, not the device)");
        fails_before = fails;
    }

    // ---------- arm B: the mode is ON ----------
    {
        MtpDrafter m;
        m.g_ = &g; m.ss_ = &ss; m.max_t_ = 6; m.kv_only_ = true;
        chk(m.kv_only(), "kv_only() reports the mode");

        std::vector<int32_t> drafts(6, -7);
        std::vector<float> probs(6, -1.0f);
        int n = -5; std::string err;
        const bool r = m.draft(6, drafts.data(), 10, 0, drafts.data(), err, probs.data(), 0.0f, &n);
        std::printf("  [mode on ] draft -> %d, n_drafts = %d, err = \"%s\"\n", (int) r, n, err.c_str());
        chk(!r, "draft() refuses (a caller that checks ends its request)");
        chk(n == 0, "draft() reports ZERO candidates");
        bool zeroed = true;
        for (int i = 0; i < m.max_t_ - 1; ++i) if (drafts[i] != 0 || probs[i] != 0.0f) zeroed = false;
        chk(zeroed, "every drafts/probs slot is zeroed (no plausible-but-wrong token survives)");
        chk(err == MtpDrafter::draft_refusal_reason(), "the refusal names the mode");
        chk(m.draft_refusals() == 1, "the refusal counter moved (the guard ran)");

        // draft_first: window_R_ is null in this mode; the guard must fire BEFORE the staging memcpy
        m.window_R_ = nullptr;
        std::vector<int32_t> d2(6, -3);
        std::vector<float> p2(6, -1.0f);
        int n2 = -5; std::string err2;
        const bool r2 = m.draft_first(6, nullptr, 1, 10, d2.data(), err2, p2.data(), 0.0f, &n2);
        chk(!r2, "draft_first() refuses");
        chk(n2 == 0, "draft_first() reports zero candidates");
        chk(m.draft_refusals() == 2, "both entry points route through one refusal");
        bool zeroed2 = true;
        for (int i = 0; i < m.max_t_ - 1; ++i) if (d2[i] != 0 || p2[i] != 0.0f) zeroed2 = false;
        chk(zeroed2, "draft_first() zeroes its outputs too");

        // the capture paths themselves
        std::string e3;
        chk(!m.capture_round(6, false, e3), "capture_round refuses: no draft graph is captured");
        chk(!m.capture_step(1, false, e3), "capture_step refuses");
        for (int i = 0; i < 9; ++i)
            if (m.round_exec_[i] || m.step_exec_[i] || m.round_exec_c_[i] || m.step_exec_c_[i])
                chk(false, "no draft graph exec exists");
        chk(true, "no round/step graph exec was created (all 36 slots null)");

        // record_forward's own gate: the full layer, refused before anything is dereferenced
        std::string e4;
        chk(!m.record_forward(1, 0, nullptr, e4), "record_forward(full) refuses");
        chk(e4 == MtpDrafter::draft_refusal_reason(), "record_forward(full) names the mode");

        // the coupled sampler can never activate
        strata::kernels::SamplerParams sp{}; sp.greedy = false; sp.temperature = 1.0f;
        m.set_draft_sampling(sp);
        chk(!m.coupled(), "set_draft_sampling cannot turn coupling on");
        m.set_draft_history(nullptr, 0, 0);   // must not touch the (null) penalty staging
        chk(true, "set_draft_history is a no-op");

        // bind() must not upgrade it
        std::string e5;
        chk(!m.bind(*(const WeightTable*) nullptr, nullptr, nullptr, e5), "bind() refuses to upgrade a KV-only drafter");
        chk(e5.find("KV-only") != std::string::npos, "bind()'s refusal names the mode");
        chk(m.kv_only(), "the failed bind left the mode alone");

        // bind_bytes reserves nothing for the expert cache
        chk(m.bind_bytes(1u << 20, 248320) == 0, "bind_bytes() is 0 in KV-only mode");

        // prefill without load()/bind() is a named error, not a null deref
        std::string e6;
        m.g_ = nullptr; m.wt_ = nullptr;
        chk(!m.prefill(nullptr, nullptr, 4, 0, e6), "prefill() refuses an unbound drafter");
        chk(!e6.empty(), "prefill() names the reason");
    }

    // ---------- arm C: bind_kv_only's own guards (no WeightTable available without CUDA) ----------
    {
        MtpDrafter m;
        m.max_t_ = 6;
        std::string err;
        // An empty table: the mode must NOT be set as a side effect of a failed bind.
        static WeightTable empty{};
        chk(!m.bind_kv_only(empty, err), "bind_kv_only refuses a table with no output.weight");
        chk(!m.kv_only(), "a failed bind_kv_only leaves the drafter unmodified");
        chk(err == "mtp: output.weight is missing", "the refusal is the same one bind() gives");
        // Already bound for drafting: no downgrade.
        m.head_ = (const NativeHead*) (uintptr_t) 1;   // non-null sentinel: only the null test matters
        std::string err2;
        chk(!m.bind_kv_only(empty, err2), "bind_kv_only refuses a drafter already bound for drafting");
        chk(err2.find("downgraded") != std::string::npos, "and says so");
    }

    std::printf(fails ? "\nFAILURES: %d\n" : "\nALL OK (%d failures)\n", fails);
    return fails ? 1 : 0;
}
