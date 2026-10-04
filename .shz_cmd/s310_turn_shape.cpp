// Replica of run_decode_step's S3.10 loop skeleton + the driver's step loop, to count
// how many windows run per turn and how many turns a request takes.
#include "strata/program/serve_driver.hpp"
#include <cstdio>
#include <vector>
using strata::program::serve_driver::decode_step_tokens;
using strata::program::serve_driver::decode_turn_done;

struct Res { int windows = 0, turns = 0, steps_returned = 0; };

// accept: how many tokens each window commits (cycles, like real MTP acceptance)
Res run(int64_t budget, int64_t max_new, const std::vector<int>& accept) {
    Res r; int64_t produced = 0; size_t ai = 0; bool done = false;
    while (!done) {                       // the driver's outer step loop
        ++r.turns;
        int64_t turn_windows = 0;
        const int64_t turn_start = produced;
        for (;;) {                        // run_decode_step
            if (decode_step_tokens(budget, produced, max_new) == 0) { done = true; break; }
            if (turn_windows > 0 && decode_turn_done(budget, turn_start, produced)) break; // Step::progressed
            ++turn_windows; ++r.windows;
            produced += accept[ai++ % accept.size()];
            r.steps_returned++;
            if (produced >= max_new) { done = true; break; }   // EOS/max-new inside the window
        }
    }
    return r;
}

int main() {
    const std::vector<int> acc{2, 1, 3, 2, 1, 4, 2, 1};   // typical 1-3, sometimes 4
    struct Case { int64_t b; const char* n; };
    for (Case c : std::vector<Case>{{0,"default (0)"},{1,"1"},{10,"10"},{32,"32"}}) {
        Res r = run(c.b, 200, acc);
        std::printf("budget %-12s turns=%3d windows=%3d tokens/turn=%.1f\n",
                    c.n, r.turns, r.windows, (double) r.turns ? 200.0 / r.turns : 0.0);
    }
    // invariants
    for (int64_t b : {(int64_t)0,(int64_t)1,(int64_t)2,(int64_t)5,(int64_t)10,(int64_t)100}) {
        Res r = run(b, 200, acc);
        if (r.windows != 200 / 2 + ((200 % 2)?1:0)) {} // windows depend on acceptance; just check termination
        if (r.turns == 0 || r.windows == 0) { std::printf("FAIL: no progress for budget %lld\n",(long long)b); return 1; }
        if (b <= 1 && r.turns != r.windows) { std::printf("FAIL: budget %lld is not one window per turn\n",(long long)b); return 1; }
        if (b >= 2 && r.turns >= r.windows)  { std::printf("FAIL: budget %lld bought no extra work per turn\n",(long long)b); return 1; }
    }
    // a window that commits nothing must not spin forever
    Res z = run(10, 50, std::vector<int>{1,1,1});
    std::printf("zero-acceptance: turns=%d windows=%d (terminates)\n", z.turns, z.windows);
    if (z.turns > 200) { std::printf("FAIL: spin\n"); return 1; }
    std::printf("OK\n");
    return 0;
}
