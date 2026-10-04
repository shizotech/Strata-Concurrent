cd /ssd/Strata && python3 - <<'PY'
p='docs/STAGE4-BATCH-DECODE.md'
s=open(p).read()
fixes=[
 ("`qsa.hpp:112-121`: \"the physical row for logical cell `t` and head `h` is\n  `page = table[t / page_size]`",
  "`qsa.hpp:68-74`: \"the physical row for logical cell `t` and head `h` is\n  `page = table[t / page_size]`"),
 ("(`qsa_decode_attn.hpp:5-14`, `qsa.hpp:112-121`:", "`qsa_decode_attn.hpp:5-14` and `qsa.hpp:68-74`:"),
 ("(`qsa.hpp:96-100` says exactly this: \"50 MiB\nat 32K\")", "(`qsa.hpp:83-84` says exactly this: \"128 B/token/layer = 1.5 KiB/token\nover the 12 QSA layers, i.e. 50 MiB at 32K\")"),
 ("(`qsa_select.hpp:16`, `qsa.hpp:149-152`).", "(`qsa_select.hpp:16`, `qsa.hpp:149-151`, `:156-162`)."),
 ("| `kv_append_*_step(…, page_table, step, …)` | `include/strata/kernels/kv_q8.hpp:29-32`, `qsa.hpp:176-186` | position and page come from `step[]` (device), so per-row |",
  "| `kv_append_*_step(…, page_table, step, …)` | `include/strata/kernels/kv_q8.hpp:29-32`, `include/strata/kernels/qsa.hpp:365` | position and page come from `step[]` (device), so per-row |"),
 ("(`kv_q8.hpp:29-32`).\n* **Indexer key append: no.**", "(`kv_q8.hpp:29-32`, `qsa.hpp:365`).\n* **Indexer key append: no.**"),
 ("indexed by cell index, no page table |\n| `native_qsa_indexer_append(raw, relative_pos_device, pos_base, …)`",
  "indexed by cell index, no page table (`qsa.hpp:197`: \"WHERE THE INDEXER'S **PER-SEQUENCE** STATE LIVES. One set per QSA layer per sequence\") |\n| `native_qsa_indexer_append(raw, relative_pos_device, pos_base, …)`"),
 ("(`include/strata/kernels/qsa.hpp:140-166`)", "(`include/strata/kernels/qsa.hpp:138-173`)"),
]
for a,b in fixes:
    if a not in s: print("MISS:", a[:70].replace("\n","\\n"))
    s=s.replace(a,b)
open(p,'w').write(s)
PY
grep -n "qsa.hpp:1[0-9][0-9]\|qsa.hpp:[6-9][0-9]" docs/STAGE4-BATCH-DECODE.md
