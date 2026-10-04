# S4.5 — the marginal-row slope: is the decode window launch-bound?

**Question the owner asked:** S4.1 said B=2 is worth only +11-13 %. Is that an artifact of the 8-row
ceiling and an unexplained 8.4 ms/row slope, or is it real?

**The one number that decides it:** the marginal cost `b` of one more row in a verify window, and
whether `b` is bandwidth or overhead.

## Ownership / file discipline (R13)

* `docs/STAGE4-BATCH-DECODE.md` — **PARENT only.** The ceilings child writes its table to
  `.megamind/src/kernels/s45-ceilings-table.md`; the parent appends it as a new §9.
* `bench/decode-slope/*` — parent.
* `src/kernels/**`, `include/**`, `src/core/**` — ceilings child reads, nobody edits (except D, below).
* `src/program/generate.cpp`, `CMakeLists.txt` — **parent owns, do not edit.**

## What the log actually contains (verified, read-only)

`strata-iq3_s.log`, 17 508 lines, live, **read-only**. Per request, on **stderr**:

```
strata serve: prompt N tokens = R reused + F read in P ms (x tok/s), G generated in D ms (y tok/s),
              drafts accepted A of O, C checkpoints
strata serve: decode expert cache hit rate: H% (hits / lookups)
strata serve: KV streaming: X% of B block reads hit VRAM, M MiB read from RAM
strata serve: suffix drafts: W windows, a of o drafts accepted
```

`generate.cpp:7502-7505` (prompt line), `:7519-7522` (expert hit), `:7540-7543` (KV stream),
`:7546` (suffix drafts).

**There is NO per-window line in this log.** `STRATA_DECODE_TIMING=1` prints one line per *request*
(`generate.cpp:7344-7358`): `%lld windows, avg T %.2f, %.2f tokens/window, %.2f ms/window = verify … +
commit/emit … + draft …`. It was **not** set for any run in this log — `grep -c "decode timing"` = 0.
So Part A must *derive* the per-window rows, not read them.

## The derivation (this is the method, and its limits)

Per request, over its decode windows only (`decode_ms`, not `prompt_ms`):

* `W` = windows = `suffix drafts: W windows` when suffix drafting ran, else `O − A + 1`
  (`draft_offered += T−1`, `draft_accepted += a`, `generate.cpp:7253-7254`; the first window has T=1
  (`:7180`) and contributes 0 offered, so `Σ(T−1) = O` and `ΣT = O + W`).
* `ΣT = O + W`, `Σ(a+1) = tokens committed = G` (the last window's bonus token is counted in `produced_n`,
  `:7257-7260`).
* ⇒ **mean T per request = (O + W) / W**, and **mean ms/window = decode_ms / W**.
* `Σ a = A` ⇒ tokens/window = `G/W` = `1 + A/W`. Consistency check: `G` should equal `O + W − … `; the
  exact identity is `G = Σ(a+1)` over windows that ran, so `G = A + W` **unless** the request stopped
  early (EOS / max_new / cancel truncates the last window's emit).

That identity is the filter: keep requests where `G == A + W` exactly, so every window ran to
completion. `analyze.py` reports how many it dropped and why.

**What this can and cannot give.** It gives a *per-request mean* of T and of ms/window, not a joint
distribution. Regressing `ms/window` on `mean T` across requests is an **errors-in-variables**
regression: the within-request spread of T is measurement error in the regressor, which **biases the
slope toward 0**. So the fitted `b` from this route is a **lower bound** on the true marginal cost, and
the two-point S4.1 estimate (24 ms @ T=1 vs 66 ms @ T=6) is the upper anchor. Both must be reported.
The bucketed fit (requests whose mean T is pinned near an integer, e.g. acceptance ≈ 0 ⇒ T ≈ 1) is the
honest way to get a clean slope from this log.

## S4.1's two anchor points, re-checked

* `strata-iq3_s.log:106` — `prompt 9691 tokens = … 1 generated in 24 ms (42.2 tok/s), drafts accepted 0
  of 0` ⇒ W=1, T=1, 24 ms. **Clean.**
* `strata-iq3_s.log:8578` — `3305 generated in 50 085 ms (66.0 tok/s), drafts accepted 1959 of 2944`.
  W = 2944 − 1959 + 1 = 986; ΣT = 2944 + 986 = 3930 ⇒ **mean T = 3.99, not 6**; tokens/window =
  3305/986 = 3.35; **ms/window = 50 085/986 = 50.8 ms, not 66 ms.**
  ⇒ slope `(50.8 − 24) / (3.99 − 1)` = **6.7 ms/row**, and the intercept moves. **S4.1 §3.5's "T=6 ⇒
  66 ms" is wrong**: it assumed every window ran at the full `--spec 4` window (S_mtp = 6), but
  `--spec-min-p 0.5` truncates T per window (`generate.cpp:7176-7179`) and suffix drafts change it too.

## Part B findings (ceilings)

Full table: `.megamind/src/kernels/s45-ceilings-table.md` (paste-ready as a new §9 of the doc).

* **There are 13 hard 8-row limits, not 6.** §1.2 missed `kFusedGrMaxT` (`fused_gr.hpp:48`),
  `bf16_gemv_fp32_mmvf_multi` `n_tok>8` (`native_bf16.cu:139`), `shared_expert_multi` `n_tok>8`
  (`shared_expert.cu:172`), `kMaxWindowEntries=128` + `static_assert(MAXT*10<=128)`
  (`expert_source.cpp:974-975`), `CAP=MAXT*10` (`remote_experts.cpp:17`), `groups_[9]`/`last_tokens_[8]`
  (`verify.hpp:206,174`), and `moe_group_resident`'s `n<=128` (`s2_expert_grouped.cu:703`).
* **Zero-effect verdicts:** `--spec` clamp YES; `[9]` graph arrays YES; `MAX_NCOLS` YES (standalone only);
  `cpu::MAXT` NO (host RAM only — cheapest no); `kVerifyMaxT` NO; `GMAX` NO.
* **`GMAX` hard-fails the build at 24/32**, measured with `nvcc -c -arch=sm_86 -Xptxas -v`:
  69 120 B and 92 160 B against the 49 152 B static limit (`0x10e00/0x16800 vs 0xc000`). At 16 it
  compiles but occupancy drops 5 → 2 blocks/SM. **Corrected number:** `gu_grouped_kernel` is
  **10 560 B per GMAX entry**, not 2 560 — the batch-decode note (`batch-decode-notes.md:80`) is 4× low.
* **Arena slope, computed from `verify.cpp:229-262` and validated at 76.943 vs the log's 76.9:**
  **6.021 MiB/row** (5.441 buffers + 0.500 `scores_` + 0.080 `hit_scratch_`); fixed 40.625 MiB
  (`kStagingBlobs 16 × max_blob 2 662 400`). T=8 89.042, T=12 113.240, T=16 137.438, T=32 234.229 MiB.
  Drafter carve (`mtp.cpp:264-295`) 22.672 → 39.006 MiB. **B=2 at T=12 costs +26.9 MiB VRAM, +0.73 MiB
  pinned RAM** — memory is not the blocker.
* **`exec_[9]`/`groups_[9]`/`last_tokens_[8]` are OOB at T=9 with no check** (`verify.cpp:822,342,988`);
  `GMAX<T` silently drops entries (`s2_expert_grouped.cu:554,616`) — the worst failure mode.
* **Log fact for Part A:** 59 `captured the N-token window` lines, T=1..6 only (10/10/10/10/9/10), **zero**
  at T=7/8, over 12 process starts. Only 6 of the 8 `exec_` slots are ever instantiated.
* MMVQ: `g_multi_exact=true` (`native_mmvq.cu:997`) has **no caller in the engine**; ncols 9..16 is
  bit-exact by construction but `mmvq_multi_parity.cpp:73` tests only {1,2,3,4,5,6,8} — 7 and 9..16
  untested. The i-quant `iq_mmvq` (`iq_kernels.cu:633`) has **no ncols ceiling at all**.

## S4.5-D: the "compiled code is byte-identical" claim, proved

The claim in the headers' comments and in the commit message is that replacing the seven literal
`[9]`/`[8]` shape arrays with `kVerifyMaxT`-derived expressions changes **nothing** the T <= 8 path
compiles. That was asserted before it was checked; it is now checked.

**Method.** Compile `src/core/verify.cpp` twice at the real optimisation level, once against the
pre-change headers (`git show HEAD~1:...` staged into an override include dir) and once against the
committed ones, and compare object hashes:

```sh
mkdir -p /tmp/oldinc/strata/core /tmp/oldinc/strata/kernels
git show HEAD~1:include/strata/core/verify.hpp            > /tmp/oldinc/strata/core/verify.hpp
git show HEAD~1:include/strata/kernels/verify_kernels.hpp > /tmp/oldinc/strata/kernels/verify_kernels.hpp

nvcc -std=c++20 -O3 -arch=sm_86 -c src/core/verify.cpp -o NEW.o \
     -Iinclude -Ithird_party/ggml/include
nvcc -std=c++20 -O3 -arch=sm_86 -c src/core/verify.cpp -o OLD.o \
     -I/tmp/oldinc -Iinclude -Ithird_party/ggml/include      # override dir FIRST
md5sum OLD.o NEW.o
```

**Result (nvcc 13.2, sm_86, -O3):**

```
97b3330929e990a9e7238f5d277838ca  OLD.o   (162 752 B)
97b3330929e990a9e7238f5d277838ca  NEW.o   (162 752 B)
```

Identical, same size.

**Non-vacuity.** The same command with one array genuinely enlarged
(`exec_[9]` -> `exec_[12]`, everything else untouched) gives a **different** hash and a different
size:

```
4854706722050b6a0decfda143c89e04  BIG.o   (162 688 B)
```

So the comparison detects a real layout change; the match above is not two builds of the same file.

**Trap worth repeating:** the include order decides the result. `-Iinclude` before `-I/tmp/oldinc`
compiles the *new* headers in both arms and the hashes match trivially. Put the override directory
first, and confirm it took effect by checking that the mutation arm changes.
