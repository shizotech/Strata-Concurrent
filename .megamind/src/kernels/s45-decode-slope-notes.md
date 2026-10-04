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
