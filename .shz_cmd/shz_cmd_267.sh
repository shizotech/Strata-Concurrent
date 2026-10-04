cd /ssd/Strata && python3 - <<'PY'
p='docs/STAGE4-BATCH-DECODE.md'
s=open(p).read()
fixes=[
 ("| `token[]` | `h_tok_` (`verify.cpp:181`) | yes |",
  "| `token[]` | `h_tok_` (`include/strata/core/verify.hpp:181`, allocated `src/core/verify.cpp:190`) | yes |"),
 ("| `pos[]` | `h_pos_` (`verify.cpp:183`, `:965-969`) | yes in form, `pos0 + t` in policy |",
  "| `pos[]` | `h_pos_` (`include/strata/core/verify.hpp:183`, allocated `src/core/verify.cpp:192`, filled `:965-969`) | yes in form, `pos0 + t` in policy |"),
 ("per-row RoPE positions (q / kv / indexer) | `src/core/verify.cpp:965-969`",
  "per-row RoPE positions (q / kv / indexer) | `src/core/verify.cpp:965-969`"),
 ("(`h_step_` is `T × kStepCount`\nand `h_pos_` is `T × (n_head + n_head_kv + idx_q_heads)` (`src/core/verify.cpp:184`, `:212-213`)",
  "(`h_step_` is `T × kStepCount`\nand `h_pos_` is `T × (n_head + n_head_kv + idx_q_heads)`, `include/strata/core/verify.hpp:182-183`, allocated at\n`src/core/verify.cpp:191-192`)"),
 ("| pinned staging the graph reads *from* | `h_tok_/h_step_/h_pos_/h_commit_/h_ple_/h_out_/h_x_/h_ids_/h_w_/h_ymiss_/h_plan_` | `src/core/verify.cpp:187-204`, `:211-213` | **yes** |",
  "| pinned staging the graph reads *from* | `h_tok_/h_step_/h_pos_/h_commit_/h_ple_/h_out_/h_x_/h_ids_/h_w_/h_ymiss_/h_plan_` | `src/core/verify.cpp:190-203`, `:211` | **yes** |"),
 ("| the doorbell flags | `m_seq_/m_flag_/m_flagA_/m_flagB_` | `src/core/verify.cpp:199-201`; `include/strata/core/verify.hpp:190-193` | **yes** |",
  "| the doorbell flags | `m_seq_/m_flag_/m_flagA_/m_flagB_` | `src/core/verify.cpp:199-202`; `include/strata/core/verify.hpp:190-193` | **yes** |"),
 ("(`--spec-split`) already runs the recurrence over rows `[0, te)` while emitting outputs only for\n`[tb, te)` (`src/core/verify.cpp:473-475`, and the note at `:318-327`)",
  "(`--spec-split`) already runs the recurrence over rows `[0, te)` while emitting outputs only for\n`[tb, te)` (`src/core/verify.cpp:473-475`, `tb_`/`te_` at `:341`, and the split-window note at `:313-321`)"),
 ("| `Verifier::init` refuses `max_t > kVerifyMaxT` | `src/core/verify.cpp:146-147` |",
  "| `Verifier::init` refuses `max_t > kVerifyMaxT` | `src/core/verify.cpp:146-147` |"),
 ("the arena carve\n(`src/core/verify.cpp:225-263`) arithmetically",
  "the arena carve\n(`src/core/verify.cpp:224-263`) arithmetically"),
 ("Replicating `Verifier::init`'s `Bump` carve (`src/core/verify.cpp:225-263`)",
  "Replicating `Verifier::init`'s `Bump` carve (`src/core/verify.cpp:224-263`)"),
 ("**replicates `Verifier::init`'s arena carve**\n(`src/core/verify.cpp:225-263`) as the same `Bump`",
  "**replicates `Verifier::init`'s arena carve**\n(`src/core/verify.cpp:224-263`) as the same `Bump`"),
 ("`src/core/verify.cpp:330-793` reads only", "`src/core/verify.cpp:322-793` reads only"),
 ("`--spec-split` (`verify.hpp:122`, `:205`), the ordering documented at `verify.cpp:318-327`",
  "`--spec-split` (`verify.hpp:122`, `:205`), the ordering documented at `verify.cpp:313-321`"),
 ("`STRATA_DEC_BATCH=0` (the existing per-token fallback, `verify.cpp:340`)",
  "`STRATA_DEC_BATCH=0` (the existing per-token fallback, `verify.cpp:339`),"),
]
for a,b in fixes:
    if a not in s: print("MISS:", a[:70].replace("\n","\\n"))
    s=s.replace(a,b)
open(p,'w').write(s)
PY
grep -c "" docs/STAGE4-BATCH-DECODE.md
