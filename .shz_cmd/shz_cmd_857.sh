cd /ssd/Strata && python3 - <<'PY'
p='.megamind/src/core/s434-mtp-kv-only-notes.md'
s=open(p).read()
subs=[
 ("| `head_logits_` (draft logits) | `bind()` `mtp.cpp:422-427` never runs | 5.7 MiB at this box |",
  "| `head_logits_` (draft logits) | `bind()` `mtp.cpp:426-429` never runs | 5.7 MiB at this box |"),
 ("| `dhead_` + `dvocab_` (the draft head over `draft_vocab.bin`) | `bind()` `mtp.cpp:429-446` never runs | 81.2 MiB, `strata-iq3_s.log:35` |",
  "| `dhead_` + `dvocab_` (the draft head over `draft_vocab.bin`) | `bind()` `mtp.cpp:432-449` never runs | 81.2 MiB, `strata-iq3_s.log:35` |"),
 ("| `setup_coupled` (the coupled draft sampler: `cparams_`, `cring_`, `cscratch_`, `dinv_`, `h_cparams_`, `h_chist_`) | `bind()` `mtp.cpp:447` never runs | 0 here (`STRATA_SPEC_COUPLED` unset) |",
  "| `setup_coupled` (the coupled draft sampler: `cparams_`, `cring_`, `cscratch_`, `dinv_`, `h_cparams_`, `h_chist_`) | `bind()` `mtp.cpp:450` never runs | 0 here (`STRATA_SPEC_COUPLED` unset) |"),
 ("| `head_` → the whole `NativeHead` in the prefill role | `bind()`'s `--native` requirement (`mtp.cpp:~420`) gone | **497 MiB** (`strata-iq3_s.log:19`) — this is OQ-S4-9 |",
  "| `head_` → the whole `NativeHead` in the prefill role | `bind()`'s `--native` requirement (`mtp.cpp:425`) gone | **497 MiB** (`strata-iq3_s.log:19`) — this is OQ-S4-9 |"),
 ("| `round_exec_[]`, `step_exec_[]`, `round_exec_c_[]`, `step_exec_c_[]` | `capture_round` `mtp.cpp:728`, `capture_step` `mtp.cpp:763` refuse | 36 graph execs never captured |",
  "| `round_exec_[]`, `step_exec_[]`, `round_exec_c_[]`, `step_exec_c_[]` | `capture_round` `mtp.cpp:729`, `capture_step` `mtp.cpp:764` refuse | 36 graph execs never captured |"),
 ("| the draft-only device buffers | `load()`'s carve, `mtp.cpp:262-300` | see the numbers |",
  "| the draft-only device buffers | `load()`'s carve, `mtp.cpp:264-300` (`if (!kv)` groups), priced at `:301-306` | see the numbers |"),
 ("| `h_row_/m_row_`, `h_out_/m_out_`, `h_prob_/m_prob_` mapped staging | `load()` `mtp.cpp:255-258` | 240 B of pinned host |",
  "| `h_row_/m_row_`, `h_out_/m_out_`, `h_prob_/m_prob_` mapped staging | `load()` `mtp.cpp:255-258` | 240 B of pinned host |"),
 ("| `ident_` upload (`T × cap_` int32) | `load()` `mtp.cpp:313` (`if (ident_ != nullptr)`) | skipped with the buffer |",
  "| `ident_` upload (`T × cap_` int32) | `load()` `mtp.cpp:313` (`if (ident_ != nullptr)`) | skipped with the buffer |"),
 ("| `bind_bytes()` | `mtp.cpp:335` returns 0 | the expert cache keeps the reserve (`generate.cpp:2654`) |",
  "| `bind_bytes()` | `mtp.cpp:338` returns 0 | the expert cache keeps the reserve (`generate.cpp:2654`) |"),
 ("| `set_draft_sampling` | `mtp.cpp:397` forces `coupled_active_ = false` | can never write the null `*h_cparams_` |",
  "| `set_draft_sampling` | `mtp.cpp:400` forces `coupled_active_ = false` | can never write the null `*h_cparams_` |"),
 ("1. `draft()` `mtp.cpp:890` → `refuse_draft(...)`;\n2. `draft_first()` `mtp.cpp:950` → the same, **before** it stages into the null `window_R_`;\n3. `capture_round()` / `capture_step()` refuse, so no draft graph is ever captured;\n4. `record_forward()` refuses the `full` branch (`step_row0 >= 0`) **before dereferencing `g_`/`ss_`**, so even a\n   path that skipped 1-3 cannot run a draft forward. The `!full` branch is untouched.",
  "1. `draft()` `mtp.cpp:893` → `refuse_draft(...)`;\n2. `draft_first()` `mtp.cpp:951` → the same, **before** it stages into the null `window_R_` (and `:954` refuses a\n   null `window_R_` even when the mode is off — a new guard, not a rewrite);\n3. `capture_round()` `mtp.cpp:729` / `capture_step()` `mtp.cpp:764` refuse, so no draft graph is ever captured;\n4. `record_forward()` `mtp.cpp:526` refuses the `full` branch (`step_row0 >= 0`) **before dereferencing\n   `g_`/`ss_`**, so even a path that skipped 1-3 cannot run a draft forward. The `!full` branch is untouched."),
 ("`refuse_draft` (`mtp.cpp:503`)", "`refuse_draft` (`mtp.cpp:507`)"),
 ("The prompt path is **not touched at all**. `prefill()` (`src/core/mtp.cpp:790`), `capture_prefill` (`:711`),\n`capture_prefill_dev` (`:722`)",
  "The prompt path is **not touched at all**. `prefill()` (`src/core/mtp.cpp:791`, plus one new null guard at\n`:796`), `capture_prefill` (`:710`), `capture_prefill_dev` (`:721`)"),
]
miss=0
for a,b in subs:
    if a not in s: print("MISS:", a[:60]); miss+=1; continue
    s=s.replace(a,b,1)
open(p,'w').write(s); print("done, misses:",miss)
PY
