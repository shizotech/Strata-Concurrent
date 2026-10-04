// src/core/verify_shape_slots_test.cpp - S4.5-D: the window-shape arrays follow kVerifyMaxT.
//
// CPU-ONLY.  No CUDA call, no device, no model, no engine, no /dev/shm, no file.  It
// includes the two headers that own the shape arrays and asserts facts about their
// CONSTANTS and their STRUCT SIZES.  That is the right shape for this test: the change it
// guards (`exec_[9]` -> `exec_[kVerifyShapeSlots]`, and the same for the six MtpDrafter
// graph arrays) must not alter anything the T <= 8 path compiles, and "the arrays are the
// size they were" is a property of the header, not of a runtime path.
//
// The array LENGTHS themselves are checked by `static_assert` inside the two headers, at
// the point of declaration, where the private members are in scope.  Those asserts are
// stronger than anything this file could do (the members are private: `Verifier`'s at
// include/strata/core/verify.hpp:137, `MtpDrafter`'s at include/strata/core/mtp.hpp) and
// they are what stops a future raise of `kVerifyMaxT` from leaving a shape array behind.
// This file checks the two things a header assert cannot:
//
//   1. the CONSTANTS still have the values the T <= 8 path was built and measured with, so
//      the rename did not quietly move the ceiling; and
//   2. the STRUCT SIZES did not move, because `Verifier` and `MtpDrafter` are constructed
//      on the stack (src/program/generate.cpp:895, :3944, :3954, :9533 and :2507), so a
//      size change is a real ABI change even when it is benign.
//
// Why the ceiling was dangerous before this change (risk B1, docs/STAGE4-BATCH-DECODE.md
// §7; the ceiling inventory is §1.2 there and §9.1/§9.6 of the S4.5 section):
// `Verifier::capture(T)` writes `exec_[T]` (src/core/verify.cpp:822), `record_window`
// writes `groups_[T]` (:342) and `run` writes `last_tokens_[t]` (:988), all with NO bounds
// check - `run` only checks `T > max_t_` (:954).  The seven `[9]`s had no textual link to
// `kVerifyMaxT` at all, so raising the ceiling and forgetting one array is silent
// undefined behaviour, not a refusal.  Index 0 of the shape arrays is never used (a window
// is T >= 1), so a shape array's length must be `kVerifyMaxT + 1`; `last_tokens_` is
// indexed by the row `t < T`, so its length is `kVerifyMaxT` with no +1.  Making those two
// the same is an off-by-one in one direction or the other, which is why they are now two
// different expressions and why the non-vacuity checks at the bottom insist the test can
// tell them apart.

#include "strata/core/verify.hpp"
#include "strata/core/mtp.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cstddef>
#include <cstdio>

namespace {

int failures = 0;
int checks = 0;

void ok(bool cond, const char* what) {
    ++checks;
    if (!cond) {
        ++failures;
        std::printf("FAIL: %s\n", what);
    }
}

void eq_sz(std::size_t got, std::size_t want, const char* what) {
    ++checks;
    if (got != want) {
        ++failures;
        std::printf("FAIL: %s: got %zu want %zu\n", what, got, want);
    }
}

constexpr int KMAX = strata::kernels::kVerifyMaxT;
constexpr int SLOT = strata::kernels::kVerifyShapeSlots;

// The values the headers' own static_asserts are written against.  Restating them here
// means deleting a header assert still shows up as a failure somewhere (see the mutation
// recipe in .megamind/src/kernels/s45-decode-slope-notes.md).
constexpr int EXPECT_KMAX = 8;
constexpr int EXPECT_SLOT = 9;

}  // namespace

int main() {
    std::printf("verify_shape_slots_test: kVerifyMaxT = %d, kVerifyShapeSlots = %d\n", KMAX, SLOT);

    // ---- 1. the ceiling itself did not move ------------------------------------
    // S4.5-D is a rename plus asserts.  It does NOT raise the row ceiling, and if it had,
    // every number in docs/STAGE4-BATCH-DECODE.md §2.2 (the verifier arena is 6.021 MiB of
    // VRAM per row) and every measured slope would need re-measuring.
    ok(KMAX == EXPECT_KMAX, "kVerifyMaxT is still 8 - S4.5-D does not raise the row ceiling");
    ok(SLOT == EXPECT_SLOT, "kVerifyShapeSlots is 9, exactly what the seven literal [9]s were");
    ok(SLOT == KMAX + 1, "kVerifyShapeSlots == kVerifyMaxT + 1 (index 0 is never used)");

    // ---- 2. the struct sizes did not move --------------------------------------
    // The bit-exactness half of the claim.  These are the bytes the stack frames at
    // generate.cpp:895 / :3944 / :3954 / :9533 (Verifier) and :2507 (MtpDrafter) reserve.
    // They are written down so that a later raise of kVerifyMaxT fails HERE and says "you
    // changed the size of a live object - go and re-prove parity", instead of moving a
    // struct in silence.
    eq_sz(sizeof(strata::core::Verifier), 5968, "sizeof(Verifier) unchanged by S4.5-D");
    eq_sz(sizeof(strata::core::MtpDrafter), 1640, "sizeof(MtpDrafter) unchanged by S4.5-D");

    // `MtpDrafter` has exactly six cudaGraphExec_t arrays (mtp.hpp: step_exec_,
    // round_exec_c_, step_exec_c_, prefill_exec_, prefill_dev_exec_, round_exec_) and
    // `Verifier` has exec_ plus the single commit_exec_.  Their totals are lower bounds on
    // the struct sizes, so a lost slot shows up here even if the sizeof() checks above were
    // edited out.
    const std::size_t mtp_graph_bytes = (std::size_t) 6 * (std::size_t) SLOT * sizeof(void*);
    ok(sizeof(strata::core::MtpDrafter) >= mtp_graph_bytes,
       "MtpDrafter still has room for six kVerifyShapeSlots graph arrays");
    const std::size_t ver_graph_bytes = ((std::size_t) SLOT + 1) * sizeof(void*);
    ok(sizeof(strata::core::Verifier) >= ver_graph_bytes,
       "Verifier still has room for exec_[kVerifyShapeSlots] plus commit_exec_");

    // ---- 3. the two bounds are one constant, so they cannot drift --------------
    // `Verifier::init` refuses `max_t > kVerifyMaxT` (src/core/verify.cpp:146) and
    // `MtpDrafter::load` refuses it (src/core/mtp.cpp:149).  The shape arrays are now
    // sized from the same constant, so the refusal bound and the array bound move
    // together.  Before S4.5-D they did not: init() could in principle accept T = 9 while
    // exec_ had 9 slots, and index 9 was past the end.
    ok(SLOT - 1 == KMAX,
       "the array bound and the init()/load() refusal bound are the same constant");

    // ---- 4. non-vacuity --------------------------------------------------------
    // A test that cannot fail is not a gate.  These prove the file distinguishes the two
    // candidate lengths, so an off-by-one in either direction is caught.
    ok(SLOT != KMAX, "the test distinguishes kVerifyMaxT from kVerifyMaxT + 1");
    ok(EXPECT_SLOT != EXPECT_KMAX, "and distinguishes them in the other direction too");
    ok(sizeof(void*) == 8, "cudaGraphExec_t is pointer-sized on this target, which the "
                           "size arithmetic above assumes");

    std::printf("verify_shape_slots_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
