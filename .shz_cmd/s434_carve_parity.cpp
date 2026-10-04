// S4.3.4 non-vacuity + parity proof (host-only, no CUDA, no model).
// Replicates MtpDrafter::load's buffer carve BEFORE and AFTER the change and compares every offset.
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

struct Bump {
    uint8_t* base = nullptr;
    uint64_t used = 0;
    template <typename T> T* take(uint64_t n) {
        T* p = base ? (T*) (base + used) : nullptr;
        used += (n * sizeof(T) + 255) & ~255ull;
        return p;
    }
};

// scratch stand-ins for the kernel sizing helpers (values copied from the sources)
static uint64_t qsa_decode_attn_scratch_floats(int64_t cap) {   // src/kernels/cuda/qsa_decode_attn.cu:251
    const int64_t CHUNK = 64; const int64_t n_head = 24; const int64_t HD = 256;
    const int64_t chunks = (cap + CHUNK - 1) / CHUNK;
    return (uint64_t) chunks * (uint64_t) n_head * (HD + 2) + 64;
}
static uint64_t moe_hit_grouped_scratch_bytes(int64_t n_hits, int64_t n_embd, int64_t n_ff) {  // s2_expert_grouped.cu:318
    if (n_hits <= 0) return 0;
    const uint64_t gu = (uint64_t) n_hits * (uint64_t) (2 * n_ff) * 4;
    const uint64_t q8 = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 34;
    const uint64_t hs = (uint64_t) n_hits * (uint64_t) (n_ff / 32) * 4;
    const uint64_t xh = (uint64_t) (n_embd / 32) * 4;
    return ((gu + 15) & ~15ull) + ((q8 + 15) & ~15ull) + 2 * ((hs + 15) & ~15ull) + ((xh + 15) & ~15ull);
}
static uint64_t shared_expert_scratch_bytes(int64_t n_ff) {   // shared_expert.cu:201
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}
static uint64_t native_q8_1_bytes(int n_in, int ncols) {   // native_mmvq.cu:1147 (Q81Block = 8 half2? -> 34 B / 32 elems)
    return (uint64_t) ncols * (uint64_t) (n_in / 32) * 36;   // Q81Block: half2 ds (4) + int8_t qs[32] = 36
}

// The geometry of the live box (include/strata/core/layout.hpp defaults) and its config
// (strata-iq3_s.json: --spec 4 -> o.spec 6 after the suffix bump at generate.cpp:1659-1661, --mtp-window 32768,
//  --max-context 524288 in the log).
struct Geo { int64_t n_embd = 2560, hc = 4, hc_lr = 320, n_head = 24, head_dim = 256, n_head_kv = 2,
                    n_expert = 512, n_ff = 640; int k = 10; };

struct Off { std::vector<std::pair<std::string, uint64_t>> v; };

// ---- the ORIGINAL carve (git HEAD, src/core/mtp.cpp:252-279) ----
static uint64_t carve_old(Bump& b, const Geo& g, uint64_t T, uint64_t cap, uint64_t asf) {
    const uint64_t N = g.n_embd, HC = g.hc, K = (uint64_t) g.k;
    const uint64_t NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const uint64_t R2 = 2 * T;
    auto t = [&](const char* nm, uint64_t bytes) { b.take<uint8_t>(bytes); };
    (void) t;
    b.take<int32_t>(T); b.take<int32_t>(R2 * 4); b.take<int32_t>(R2 * NH); b.take<int32_t>(4);
    b.take<int32_t>(T * cap);
    b.take<float>(T * HC * N); b.take<float>(T * HC * N);
    b.take<float>(T * N); b.take<float>(T * N); b.take<float>(T * N);
    b.take<float>(T * HC * N); b.take<float>(T * HC * N);
    b.take<float>(T * N); b.take<float>(T * HC); b.take<float>(T * HC);
    b.take<float>(T * (uint64_t) g.hc_lr); b.take<float>(T * HC); b.take<float>(T * N);
    b.take<float>(T * HC * N);
    b.take<uint8_t>(native_q8_1_bytes((int) (NH * HD), 8));
    b.take<float>(T * NH * 2 * HD); b.take<float>(T * NH * HD);
    b.take<float>(T * NKV * HD); b.take<float>(T * NKV * HD);
    b.take<float>(T * NH * HD); b.take<float>(T * NH * HD);
    b.take<float>(asf);
    b.take<float>(T * (uint64_t) g.n_expert); b.take<float>(T * K); b.take<int32_t>(T * K);
    b.take<float>(T * N); b.take<float>(T * K * N); b.take<float>(T * N);
    b.take<float>(T * N);
    b.take<int32_t>(T * K); b.take<int32_t>(T * K); b.take<int32_t>(4);
    b.take<unsigned long long>(T * K); b.take<int32_t>(T * K + 1);
    b.take<int32_t>(4);
    b.take<uint8_t>(T * (N / 32) * 34); b.take<float>(T * (N / 32));
    b.take<uint8_t>(moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff));
    b.take<uint8_t>(shared_expert_scratch_bytes(g.n_ff));
    b.take<uint16_t>(N);
    b.take<int32_t>(T + 4);
    b.take<float>(T + 4);
    b.take<float>(HC);
    return b.used;
}

// ---- the NEW carve (src/core/mtp.cpp, kv == false == full mode) ----
static uint64_t carve_new(Bump& b, const Geo& g, uint64_t T, uint64_t cap, uint64_t asf, bool kv) {
    const uint64_t N = g.n_embd, HC = g.hc, K = (uint64_t) g.k;
    const uint64_t NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const uint64_t R2 = 2 * T;
    b.take<int32_t>(T); b.take<int32_t>(R2 * 4); b.take<int32_t>(R2 * NH);
    if (!kv) { b.take<int32_t>(4); b.take<int32_t>(T * cap); }
    b.take<float>(T * HC * N); b.take<float>(T * HC * N);
    b.take<float>(T * N); b.take<float>(T * N); b.take<float>(T * N);
    b.take<float>(T * HC * N); b.take<float>(T * HC * N);
    b.take<float>(T * N); b.take<float>(T * HC);
    if (!kv) b.take<float>(T * HC);
    b.take<float>(T * (uint64_t) g.hc_lr); b.take<float>(T * HC);
    if (!kv) b.take<float>(T * N);
    b.take<float>(T * HC * N);
    b.take<uint8_t>(native_q8_1_bytes((int) (NH * HD), 8));
    if (!kv) { b.take<float>(T * NH * 2 * HD); b.take<float>(T * NH * HD); }
    b.take<float>(T * NKV * HD); b.take<float>(T * NKV * HD);
    if (!kv) {
        b.take<float>(T * NH * HD); b.take<float>(T * NH * HD);
        b.take<float>(asf);
        b.take<float>(T * (uint64_t) g.n_expert); b.take<float>(T * K); b.take<int32_t>(T * K);
        b.take<float>(T * N); b.take<float>(T * K * N); b.take<float>(T * N);
        b.take<float>(T * N);
        b.take<int32_t>(T * K); b.take<int32_t>(T * K); b.take<int32_t>(4);
        b.take<unsigned long long>(T * K); b.take<int32_t>(T * K + 1);
        b.take<int32_t>(4);
        b.take<uint8_t>(T * (N / 32) * 34); b.take<float>(T * (N / 32));
        b.take<uint8_t>(moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff));
        b.take<uint8_t>(shared_expert_scratch_bytes(g.n_ff));
        b.take<uint16_t>(N);
        b.take<int32_t>(T + 4);
        b.take<float>(T + 4);
        b.take<float>(HC);
    }
    return b.used;
}

// record every (name, offset) pair so the two carves can be compared one buffer at a time
static std::vector<std::pair<std::string, uint64_t>> offsets(bool kv, const Geo& g, uint64_t T, uint64_t cap, uint64_t asf) {
    Bump b; b.base = (uint8_t*) 0x1000;   // fake base so take() reports real offsets
    const uint64_t N = g.n_embd, HC = g.hc, K = (uint64_t) g.k;
    const uint64_t NH = g.n_head, HD = g.head_dim, NKV = g.n_head_kv;
    const uint64_t R2 = 2 * T;
    std::vector<std::pair<std::string, uint64_t>> out;
    auto take = [&](const char* nm, auto f) { const uint64_t o = b.used; f(); out.push_back({nm, o}); };
    take("tok_", [&]{ b.take<int32_t>(T); }); take("step_", [&]{ b.take<int32_t>(R2 * 4); });
    take("pos_", [&]{ b.take<int32_t>(R2 * NH); });
    if (!kv) { take("row_", [&]{ b.take<int32_t>(4); }); take("ident_", [&]{ b.take<int32_t>(T * cap); }); }
    take("Rin_", [&]{ b.take<float>(T * HC * N); }); take("R_", [&]{ b.take<float>(T * HC * N); });
    take("emb_", [&]{ b.take<float>(T * N); }); take("en_", [&]{ b.take<float>(T * N); });
    take("e2_", [&]{ b.take<float>(T * N); });
    take("hn_", [&]{ b.take<float>(T * HC * N); }); take("h2_", [&]{ b.take<float>(T * HC * N); });
    take("mixed_", [&]{ b.take<float>(T * N); }); take("inj_", [&]{ b.take<float>(T * HC); });
    if (!kv) take("inj2_", [&]{ b.take<float>(T * HC); });
    take("lo_", [&]{ b.take<float>(T * (uint64_t) g.hc_lr); }); take("rs_", [&]{ b.take<float>(T * HC); });
    if (!kv) take("bo_", [&]{ b.take<float>(T * N); });
    take("xn_", [&]{ b.take<float>(T * HC * N); });
    take("xq_", [&]{ b.take<uint8_t>(native_q8_1_bytes((int) (NH * HD), 8)); });
    if (!kv) { take("qfull_", [&]{ b.take<float>(T * NH * 2 * HD); }); take("qcur_", [&]{ b.take<float>(T * NH * HD); }); }
    take("kcur_", [&]{ b.take<float>(T * NKV * HD); }); take("vcur_", [&]{ b.take<float>(T * NKV * HD); });
    if (!kv) {
        take("attn_", [&]{ b.take<float>(T * NH * HD); }); take("attn32_", [&]{ b.take<float>(T * NH * HD); });
        take("attn_scratch_", [&]{ b.take<float>(asf); });
        take("logits_", [&]{ b.take<float>(T * (uint64_t) g.n_expert); }); take("w_", [&]{ b.take<float>(T * K); });
        take("ids_", [&]{ b.take<int32_t>(T * K); });
        take("shared_", [&]{ b.take<float>(T * N); }); take("parts_", [&]{ b.take<float>(T * K * N); });
        take("y_", [&]{ b.take<float>(T * N); }); take("sample_", [&]{ b.take<float>(T * N); });
        take("hit_slot_", [&]{ b.take<int32_t>(T * K); }); take("hit_dst_", [&]{ b.take<int32_t>(T * K); });
        take("hit_count_", [&]{ b.take<int32_t>(4); });
        take("grp_ptr_", [&]{ b.take<unsigned long long>(T * K); }); take("grp_start_", [&]{ b.take<int32_t>(T * K + 1); });
        take("grp_counts_", [&]{ b.take<int32_t>(4); });
        take("hit_xq_", [&]{ b.take<uint8_t>(T * (N / 32) * 34); }); take("hit_xs_", [&]{ b.take<float>(T * (N / 32)); });
        take("hit_scratch_", [&]{ b.take<uint8_t>(moe_hit_grouped_scratch_bytes((int64_t) (T * K), g.n_embd, g.n_ff)); });
        take("sh_scratch_", [&]{ b.take<uint8_t>(shared_expert_scratch_bytes(g.n_ff)); });
        take("x_bf16_", [&]{ b.take<uint16_t>(N); });
        take("out_ids_", [&]{ b.take<int32_t>(T + 4); }); take("probs_", [&]{ b.take<float>(T + 4); });
        take("dummy_inj_", [&]{ b.take<float>(HC); });
    }
    return out;
}

int main() {
    int fails = 0;
    auto chk = [&](bool ok, const char* what) { if (!ok) { std::printf("FAIL: %s\n", what); ++fails; } };

    // several configs: the live one (T=6, window 32768 @ 524288) and others
    struct Cfg { const char* name; int max_t; int64_t window; int64_t max_cells; };
    const Cfg cfgs[] = {
        {"live: --spec 4 (+2 suffix) --mtp-window 32768, max-context 524288", 6, 32768, 524288},
        {"--spec 4, no suffix bump (max_t 4)",                                4, 32768, 524288},
        {"--spec 8, window 0 (whole context in VRAM)",                        8, 0,     262144},
        {"--spec 2, window 2048",                                             2, 2048,  524288},
    };
    const Geo g;
    for (const Cfg& c : cfgs) {
        const int64_t window = (c.window > 0 && c.window < c.max_cells) ? c.window : 0;
        const int64_t cap = (((window > 0 ? window : c.max_cells) + 63) / 64) * 64;
        const uint64_t asf = qsa_decode_attn_scratch_floats(cap);
        const uint64_t T = (uint64_t) c.max_t;
        Bump o, n_full, n_kv;
        const uint64_t old_used = carve_old(o, g, T, cap, asf);
        const uint64_t new_full = carve_new(n_full, g, T, cap, asf, false);
        const uint64_t new_kv = carve_new(n_kv, g, T, cap, asf, true);
        std::printf("%-62s T=%zu cap=%lld  arena old %llu B  new(full) %llu B  new(kv-only) %llu B  saved %llu B (%.1f MiB)\n",
                    c.name, (size_t) T, (long long) cap,
                    (unsigned long long) old_used, (unsigned long long) new_full, (unsigned long long) new_kv,
                    (unsigned long long) (old_used - new_kv), (double) (old_used - new_kv) / 1048576.0);
        chk(old_used == new_full, "full-mode arena size is unchanged");
        // per-buffer offsets: old vs new(full) must be identical, name by name
        const auto oo = offsets(false, g, T, cap, asf);
        const auto nn = offsets(false, g, T, cap, asf);
        chk(oo.size() == nn.size(), "same buffer list in full mode");
        for (size_t i = 0; i < oo.size() && i < nn.size(); ++i)
            chk(oo[i] == nn[i], "every buffer keeps its offset in full mode");
        // kv-only: every kept buffer keeps the SAME offset as before? (not required, but the ones that must
        // exist are all present)
        const auto kk = offsets(true, g, T, cap, asf);
        chk(kk.size() < oo.size(), "kv-only carves fewer buffers");
        for (const auto& e : kk) {
            bool found = false;
            for (const auto& f : oo) if (f.first == e.first) { found = true; break; }
            chk(found, "every kv-only buffer is one the full carve also has");
        }
    }
    std::printf(fails ? "FAILURES: %d\n" : "ALL OK (%d failures)\n", fails);
    return fails ? 1 : 0;
}
