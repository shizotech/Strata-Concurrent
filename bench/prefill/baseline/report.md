# Strata prefill cost - 1 log file(s), 8,379 lines, 34 startup block(s), 30 engine restart(s), 1,600 requests
  (note: 34 startup block(s) but 30 reached `session is up` - the rest died during loading)
  (--limit: only the first 1,600 requests in log order are counted)

## Resolved settings, per restart
| # | chunk | CUDA0 loan slots | CUDA0 cache slots | PCIe GB/s | VRAM free MiB | kv_resident | max-context |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8192 | 2,200 / 3.85 GiB | 8,192 | 2.9 | 3,553 | 32,768 | - |
| 2 | 8192 | 2,200 / 3.85 GiB | 8,192 | 2.9 | 3,553 | 32,768 | - |
| 3 | 8192 | 2,200 / 3.85 GiB | 8,192 | 2.9 | 3,553 | 32,768 | - |
| 4 | 8192 | 2,200 / 3.85 GiB | 8,192 | 2.9 | 3,553 | 32,768 | - |
| 5 | 8192 | 2,200 / 3.85 GiB | 8,192 | 2.9 | 3,553 | 32,768 | - |
| 6 | 8192 | 2,200 / 3.85 GiB | 8,192 | 2.9 | 3,553 | 32,768 | - |
| 7 | 8192 | 2,119 / 3.84 GiB | 9,570 | 2.9 | 455 | 32,768 | - |
| 8 | 8192 | 2,416 / 4.56 GiB | 8,137 | 2.9 | 455 | 32,768 | - |
| 9 | 8192 | 2,416 / 4.56 GiB | 8,137 | 2.9 | 455 | 32,768 | - |
| 10 | 8192 | 2,417 / 4.56 GiB | 8,069 | 2.9 | 457 | 32,768 | - |
| 11 | 8192 | 2,416 / 4.56 GiB | 8,137 | 2.9 | 455 | 32,768 | - |
| 12 | 8192 | 2,625 / 4.95 GiB | 7,778 | 2.9 | 457 | 32,768 | 524,288 |
| 13 | 8192 | 2,337 / 4.23 GiB | 9,351 | 2.9 | 453 | 32,768 | 524,288 |
| 14 | 8192 | 2,337 / 4.23 GiB | 9,351 | 2.9 | 453 | 32,768 | 524,288 |
| 15 | 8192 | 2,337 / 4.23 GiB | 9,351 | 2.9 | 453 | 32,768 | 524,288 |
| 16 | 8192 | 2,246 / 4.23 GiB | 7,828 | 2.9 | 455 | 32,768 | 524,288 |
| 17 | 8192 | 2,246 / 4.23 GiB | 7,828 | 1.4 | 455 | 32,768 | 524,288 |
| 18 | 8192 | 2,625 / 4.95 GiB | 7,778 | 2.9 | 457 | 32,768 | 524,288 |
| 19 | 8192 | 2,625 / 4.95 GiB | 7,778 | 2.9 | 457 | 32,768 | 524,288 |
| 20 | 8192 | 2,625 / 4.95 GiB | 7,778 | 2.9 | 457 | 32,768 | 524,288 |
| 21 | 8192 | 2,625 / 4.95 GiB | 7,778 | 1.3 | 457 | 32,768 | 524,288 |
| 22 | 8192 | 2,625 / 4.95 GiB | 7,778 | 2.9 | 457 | 32,768 | 524,288 |
| 23 | 8192 | 2,625 / 4.95 GiB | 7,778 | 2.9 | 457 | 32,768 | 524,288 |
| 24 | 8192 | 2,626 / 4.95 GiB | 7,773 | 24.6 | 457 | 32,768 | 524,288 |
| 25 | 8192 | 2,625 / 4.95 GiB | 7,778 | 1.3 | 457 | 32,768 | 524,288 |
| 26 | 8192 | 2,625 / 4.95 GiB | 7,778 | 2.0 | 457 | 32,768 | 524,288 |
| 27 | 8192 | 2,625 / 4.95 GiB | 7,778 | 2.9 | 457 | 32,768 | 524,288 |
| 28 | 8192 | 2,625 / 4.95 GiB | 7,778 | 1.3 | 457 | 32,768 | 524,288 |
| 29 | 8192 | 2,625 / 4.95 GiB | 7,778 | 2.9 | 457 | 32,768 | 524,288 |
| 30 | 8192 | 2,625 / 4.95 GiB | 7,778 | 2.9 | 457 | 32,768 | 524,288 |

restart 1:
  other stage loans: CUDA1 1,971 of 9,216 slots (3.85 GiB), CUDA2 1,885 of 7,168 slots (3.85 GiB)
  stage PCIe probe: CUDA2 24.5 GB/s -> pcie_frac 0.55
  stage caches: CUDA2 layers 34-47, 7,168 slots (14.54 GiB), 7,168 of 7,168 pairs
  split auto: K=16,34, 24.0 ms/window predicted, caches hold 24,576 of 24,576 pairs (~100.0% of routed mass)
  expert cache auto: 18.04 GiB free, 700 MiB reserved (+0 MiB draft head) -> 7000 slots, the cache ended at 8,192 slots
  arena: 46.84 GiB at 0.51 GiB/s
  arena registration: partial: 8 slices pinned (7 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)

restart 2:
  other stage loans: CUDA1 1,971 of 9,216 slots (3.85 GiB), CUDA2 1,885 of 7,168 slots (3.85 GiB)
  stage PCIe probe: CUDA2 25.8 GB/s -> pcie_frac 0.55
  stage caches: CUDA2 layers 34-47, 7,168 slots (14.54 GiB), 7,168 of 7,168 pairs
  split auto: K=16,34, 24.0 ms/window predicted, caches hold 24,576 of 24,576 pairs (~100.0% of routed mass)
  expert cache auto: 18.04 GiB free, 700 MiB reserved (+0 MiB draft head) -> 7000 slots, the cache ended at 8,192 slots
  arena: 46.84 GiB at 0.50 GiB/s
  arena registration: partial: 8 slices pinned (7 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)

restart 3:
  other stage loans: CUDA1 1,971 of 9,216 slots (3.85 GiB), CUDA2 1,885 of 7,168 slots (3.85 GiB)
  stage PCIe probe: CUDA2 24.5 GB/s -> pcie_frac 0.55
  stage caches: CUDA2 layers 34-47, 7,168 slots (14.54 GiB), 7,168 of 7,168 pairs
  split auto: K=16,34, 24.0 ms/window predicted, caches hold 24,576 of 24,576 pairs (~100.0% of routed mass)
  expert cache auto: 18.04 GiB free, 700 MiB reserved (+0 MiB draft head) -> 7000 slots, the cache ended at 8,192 slots
  arena: 46.84 GiB at 0.51 GiB/s
  arena registration: partial: 8 slices pinned (7 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)

restart 4:
  other stage loans: CUDA1 1,971 of 9,216 slots (3.85 GiB), CUDA2 1,885 of 7,168 slots (3.85 GiB)
  stage PCIe probe: CUDA2 24.5 GB/s -> pcie_frac 0.55
  stage caches: CUDA2 layers 34-47, 7,168 slots (14.54 GiB), 7,168 of 7,168 pairs
  split auto: K=16,34, 24.0 ms/window predicted, caches hold 24,576 of 24,576 pairs (~100.0% of routed mass)
  expert cache auto: 18.04 GiB free, 700 MiB reserved (+0 MiB draft head) -> 7000 slots, the cache ended at 8,192 slots
  arena: 46.84 GiB at 0.54 GiB/s
  arena registration: partial: 8 slices pinned (7 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 5:
  other stage loans: CUDA1 1,971 of 9,216 slots (3.85 GiB), CUDA2 1,885 of 7,168 slots (3.85 GiB)
  stage PCIe probe: CUDA2 24.5 GB/s -> pcie_frac 0.55
  stage caches: CUDA2 layers 34-47, 7,168 slots (14.54 GiB), 7,168 of 7,168 pairs
  split auto: K=16,34, 24.0 ms/window predicted, caches hold 24,576 of 24,576 pairs (~100.0% of routed mass)
  expert cache auto: 18.04 GiB free, 700 MiB reserved (+0 MiB draft head) -> 7000 slots, the cache ended at 8,192 slots
  arena: 46.84 GiB at 0.53 GiB/s
  arena registration: partial: 8 slices pinned (7 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 6:
  other stage loans: CUDA1 1,971 of 9,216 slots (3.85 GiB), CUDA2 1,885 of 7,168 slots (3.85 GiB)
  stage PCIe probe: CUDA2 24.5 GB/s -> pcie_frac 0.55
  stage caches: CUDA2 layers 34-47, 7,168 slots (14.54 GiB), 7,168 of 7,168 pairs
  split auto: K=16,34, 24.0 ms/window predicted, caches hold 24,576 of 24,576 pairs (~100.0% of routed mass)
  expert cache auto: 18.04 GiB free, 700 MiB reserved (+0 MiB draft head) -> 7000 slots, the cache ended at 8,192 slots
  arena: 46.84 GiB at 0.53 GiB/s
  arena registration: partial: 8 slices pinned (7 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 7:
  other stage loans: CUDA1 1,930 of 8,440 slots (3.85 GiB)
  stage PCIe probe: CUDA1 1.4 GB/s -> pcie_frac 0.00
  stage caches: CUDA1 layers 24-47, 8,440 slots (16.89 GiB), 8,440 of 12,288 pairs
  split auto: K=24, 25.6 ms/window predicted, caches hold 18,190 of 24,576 pairs (~99.2% of routed mass)
  expert cache auto: 17.93 GiB free, 700 MiB reserved (+0 MiB draft head) -> 6954 slots, the cache ended at 9,570 slots
  arena: 46.84 GiB at 0.54 GiB/s
  arena registration: partial: 8 slices pinned (7 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 8:
  expert cache auto: 16.20 GiB free, 700 MiB reserved (+86 MiB draft head) -> 6223 slots, the cache ended at 8,137 slots
  arena: 46.84 GiB at 0.54 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 9:
  expert cache auto: 16.20 GiB free, 700 MiB reserved (+86 MiB draft head) -> 6223 slots, the cache ended at 8,137 slots
  arena: 46.84 GiB at 0.54 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 10:
  expert cache auto: 16.20 GiB free, 700 MiB reserved (+218 MiB draft head) -> 6171 slots, the cache ended at 8,069 slots
  arena: 46.84 GiB at 0.52 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)

restart 11:
  expert cache auto: 16.20 GiB free, 700 MiB reserved (+86 MiB draft head) -> 6223 slots, the cache ended at 8,137 slots
  arena: 46.84 GiB at 0.53 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)

restart 12:
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena: 46.84 GiB at 0.51 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 13:
  other stage loans: CUDA1 2,107 of 8,229 slots (4.23 GiB)
  stage PCIe probe: CUDA1 1.3 GB/s -> pcie_frac 0.00
  stage caches: CUDA1 layers 24-47, 8,229 slots (16.50 GiB), 8,229 of 12,288 pairs
  split auto: K=24, 25.7 ms/window predicted, caches hold 17,788 of 24,576 pairs (~99.1% of routed mass)
  expert cache auto: 17.53 GiB free, 700 MiB reserved (+0 MiB draft head) -> 6795 slots, the cache ended at 9,351 slots
  arena: 46.84 GiB at 0.54 GiB/s
  arena registration: partial: 8 slices pinned (7 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 14:
  other stage loans: CUDA1 2,107 of 8,229 slots (4.23 GiB)
  stage PCIe probe: CUDA1 1.3 GB/s -> pcie_frac 0.00
  stage caches: CUDA1 layers 24-47, 8,229 slots (16.50 GiB), 8,229 of 12,288 pairs
  split auto: K=24, 25.7 ms/window predicted, caches hold 17,788 of 24,576 pairs (~99.1% of routed mass)
  expert cache auto: 17.53 GiB free, 700 MiB reserved (+0 MiB draft head) -> 6795 slots, the cache ended at 9,351 slots
  arena: 46.84 GiB at 0.63 GiB/s
  arena registration: partial: 8 slices pinned (7 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 15:
  other stage loans: CUDA1 2,107 of 8,229 slots (4.23 GiB)
  stage PCIe probe: CUDA1 1.4 GB/s -> pcie_frac 0.00
  stage caches: CUDA1 layers 24-47, 8,229 slots (16.50 GiB), 8,229 of 12,288 pairs
  split auto: K=24, 25.7 ms/window predicted, caches hold 17,788 of 24,576 pairs (~99.1% of routed mass)
  expert cache auto: 17.53 GiB free, 700 MiB reserved (+0 MiB draft head) -> 6795 slots, the cache ended at 9,351 slots
  arena: 46.84 GiB at 0.52 GiB/s
  arena registration: partial: 8 slices pinned (7 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 16:
  expert cache auto: 15.61 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5986 slots, the cache ended at 7,828 slots
  arena: 46.84 GiB at 0.09 GiB/s
  arena registration: partial: 0 slices pinned (0 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 17:
  expert cache auto: 15.61 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5986 slots, the cache ended at 7,828 slots
  arena: 46.84 GiB at 0.07 GiB/s
  arena registration: partial: 0 slices pinned (0 GiB) -> pinned_share < 0.9 (lend cap 85 %, ring 96)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 18:
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena: 46.84 GiB at 0.51 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 19:
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena: 46.84 GiB at 0.51 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144

restart 20:
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena: 46.84 GiB at 0.49 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144

restart 21:
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena: 46.84 GiB at 0.51 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144

restart 22:
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena: 46.84 GiB at 0.52 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 23:
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena: 46.84 GiB at 0.51 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 24:
  expert cache auto: 15.51 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5945 slots, the cache ended at 7,773 slots
  arena: 46.84 GiB at 0.52 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144

restart 25:
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena: 46.84 GiB at 0.50 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 26:
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena: 46.84 GiB at 0.50 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 27:
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena: 46.84 GiB at 0.51 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144
  ! parking: no parking lines at all in this segment (the binary predates it, or it was off)

restart 28:
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena: 46.84 GiB at 0.49 GiB/s
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144

restart 29:
  parked-prefix budget 1.6 GiB (asked 8.0 GiB; 1.6 GiB free above a 2.5 GiB floor and 4.0 GiB headroom; 8 slots)
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144
  ! parking: the `parked=` byte totals in this segment reach 4.8 GiB but this restart's own budget is 1.6 GiB - the log has lines from ANOTHER live process interleaved here, so the parking counters for this restart are not trustworthy

restart 30:
  parked-prefix budget 0.0 GiB (asked 8.0 GiB; 0.0 GiB free above a 2.5 GiB floor and 4.0 GiB headroom; 8 slots)
  expert cache auto: 15.52 GiB free, 700 MiB reserved (+86 MiB draft head) -> 5949 slots, the cache ended at 7,778 slots
  arena registration: whole arena registered -> pinned_share >= 0.9 (lend cap 90 %, ring 384)
  rope yarn x2 against a trained context of 262,144
  ! parking: the `parked=` byte totals in this segment reach 5.8 GiB but this restart's own budget is 0.0 GiB - the log has lines from ANOTHER live process interleaved here, so the parking counters for this restart are not trustworthy

## Per restart
| # | reqs | fresh tok | reused tok | prompt wall s | prompt tok/s | decode tok/s | fit a (s) | fit marginal tok/s | R2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 | 0 | 0 | 0.0 | - | - | - | - | - |
| 2 | 0 | 0 | 0 | 0.0 | - | - | - | - | - |
| 3 | 0 | 0 | 0 | 0.0 | - | - | - | - | - |
| 4 | 28 | 124,601 | 82,209 | 189.1 | 659.1 | 74.9 | 0.58 | 515 | 0.985 |
| 5 | 36 | 164,520 | 1,205,480 | 255.3 | 644.4 | 87.4 | 2.65 | 515 | 0.974 |
| 6 | 3 | 160 | 152 | 2.4 | 67.5 | 109.8 | - | - | - |
| 7 | 10 | 11,864 | 60,759 | 52.4 | 226.5 | 73.2 | 1.25 | - | 0.985 |
| 8 | 13 | 50,645 | 101,259 | 185.1 | 273.5 | 56.7 | 4.97 | 360 | 0.939 |
| 9 | 3 | 160 | 152 | 6.5 | 24.8 | 53.9 | - | - | - |
| 10 | 0 | 0 | 0 | 0.0 | - | - | - | - | - |
| 11 | 0 | 0 | 0 | 0.0 | - | - | - | - | - |
| 12 | 11 | 10,135 | 102,423 | 43.6 | 232.7 | 54.0 | 0.71 | 41 | 0.984 |
| 13 | 37 | 99,302 | 1,294,940 | 449.4 | 221.0 | 71.1 | 4.13 | 328 | 0.933 |
| 14 | 9 | 103,715 | 775,771 | 246.8 | 420.2 | 72.9 | 4.15 | - | 0.997 |
| 15 | 19 | 462,890 | 771,007 | 828.5 | 558.7 | 72.7 | 3.95 | 470 | 0.998 |
| 16 | 3 | 160 | 152 | 49.8 | 3.2 | 27.2 | - | - | - |
| 17 | 37 | 62,318 | 337,389 | 355.9 | 175.1 | 45.3 | 3.89 | 550 | 0.907 |
| 18 | 7 | 10,454 | 29,152 | 55.7 | 187.7 | 58.4 | 3.41 | - | 0.955 |
| 19 | 0 | 0 | 0 | 0.0 | - | - | - | - | - |
| 20 | 0 | 0 | 0 | 0.0 | - | - | - | - | - |
| 21 | 0 | 0 | 0 | 0.0 | - | - | - | - | - |
| 22 | 52 | 219,446 | 189,087 | 1,404.1 | 156.3 | 24.0 | 5.31 | 102 | 0.864 |
| 23 | 12 | 30,062 | 38,871 | 165.9 | 181.2 | 41.5 | 2.69 | 133 | 0.925 |
| 24 | 0 | 0 | 0 | 0.0 | - | - | - | - | - |
| 25 | 10 | 19,384 | 19,348 | 94.9 | 204.3 | 40.9 | -0.45 | 16 | 0.542 |
| 26 | 32 | 70,271 | 1,187,287 | 383.1 | 183.4 | 61.1 | 7.01 | 396 | 0.857 |
| 27 | 840 | 2,432,302 | 114,972,108 | 12,121.0 | 200.7 | 56.4 | 8.62 | 527 | 0.997 |
| 28 | 0 | 0 | 0 | 0.0 | - | - | - | - | - |
| 29 | 39 | 78,269 | 736,811 | 672.9 | 116.3 | 49.1 | 7.98 | 206 | 0.688 |
| 30 | 399 | 1,804,172 | 25,758,460 | 11,419.8 | 158.0 | 48.6 | 13.55 | 239 | 0.938 |

## Aggregate
| metric | value |
| --- | ---: |
| requests | 1,600 (5 cancelled) |
| prompt tokens | 153,417,647 |
| reused (prefix hit) | 147,662,817 (96.2%) |
| fresh (the prompt path read) | 5,754,830 |
| prompt wall time | 28,982.1 s |
| effective prompt tok/s | 198.6 |
| generated tokens / decode wall | 1,664,293 / 30,968.3 s |
| decode tok/s | 53.7 |
| drafts accepted | 977,836 of 1,472,108 (66.4%) |
| checkpoints taken | 8,186 |
| expert-cache hit | 16,032,944 hits / 17,509,440 lookups = 91.6% |
| KV streaming block reads | 1,839,691,606 reads, 215,437.7 MiB read from RAM |
| parking: parked | 124 parks, 1,618,830 tokens, 92.1 s, 71.33 GiB of snapshots, 3211 evictions |
| parking: restored | 20 checkpoint + 6 live, 562,493 tokens, 4.9 s |
| parking: refused | 123 (physical-RAM admission, largest need 2,235 MiB) |
| suffix drafts | 61026 windows, 134,279 of 157,044 accepted |

Parking counters above come only from restart(s) 29, 30; the other 28 restart(s) logged no parking at all. Aggregate parking numbers are not comparable across restarts - read the per-restart rows.

## Prompt cost by fresh-token bucket
| fresh tokens | requests | fresh tok | wall s | effective tok/s | % of prompt wall | mean ms/req | mean fresh | mean reused | ms per fresh tok |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-64 (verify-window path) | 133 | 839 | 34.9 | 24.1 | 0.1 | 262 | 6 | 8,789 | 41.56 |
| 65-256 | 319 | 54,999 | 2,320.7 | 23.7 | 8.0 | 7,275 | 172 | 94,858 | 42.20 |
| 257-1024 | 666 | 351,539 | 7,285.3 | 48.3 | 25.1 | 10,939 | 528 | 128,062 | 20.72 |
| 1k-4k | 331 | 639,087 | 6,189.1 | 103.3 | 21.4 | 18,698 | 1,931 | 85,289 | 9.68 |
| 4k-16k | 113 | 952,216 | 4,105.8 | 231.9 | 14.2 | 36,335 | 8,427 | 23,000 | 4.31 |
| >16k | 38 | 3,756,150 | 9,046.4 | 415.2 | 31.2 | 238,062 | 98,846 | 3,018 | 2.41 |

**54.6% of all prompt wall time is in reads of <= 4 000 fresh tokens.**

## Fitted cost model (least squares over 1,600 requests)
```
prompt_ms = 9547 + 2.719 * fresh + -2.652e-06 * fresh * context
            ^9.5 s fixed    ^368 tok/s marginal  ^attention term        R2 = 0.902
residuals: median -1078 ms, mean |res| 6229 ms, p90 8741 ms
without the context term: a = 10278 ms, 459 tok/s marginal, R2 = 0.895
batched path only (1,467 of the requests): a = 10460 ms, 2.686 ms/fresh (372 tok/s), -2.531e-06*fresh*context, R2 = 0.905
```
`context` is the prompt length the engine reports (reused + fresh). **Do not read the third
coefficient as a measurement of the attention cost**: over this workload fresh and context
move together, so the term is not identified (its sign flips between restarts). The
context-control table below is the honest version of the same question.

## The two prompt paths (split at --short-read = 64)
| path | requests | fresh tok | wall s | % of prompt wall | mean fresh/req | mean ms/req | ms per fresh tok | effective tok/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| window | 133 | 839 | 34.9 | 0.1 | 6 | 262 | 41.56 | 24.1 |
| batched | 1,467 | 5,753,991 | 28,947.2 | 99.9 | 3,922 | 19,732 | 5.03 | 198.8 |

The window path borrows no expert-cache slots, so it pays no lend/refill. But this log only
ever sends ~6-token segments through it (the assistant header after a checkpoint), so its
cost per token is not separable from its fixed cost: mean 262 ms for 6 tokens.
The batched path costs ~20 s per request at a mean of 3922 fresh tokens. Raising
`--short-read` above 64 cannot be priced from this log - it needs the A/B script.

## Cost by the chunk the request actually lends
(`want = min(chunk, ceil(fresh/256)*256)`, generate.cpp:3418 - what `lend()` takes buffers for)
| want | mean want | requests | mean fresh | mean ms | ms per fresh tok | ms per want-token | effective tok/s | % of prompt wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| <=256 | 256 | 319 | 172 | 7,275 | 42.20 | 28.42 | 23.7 | 8.0 |
| <=512 | 512 | 382 | 369 | 9,762 | 26.46 | 19.07 | 37.8 | 12.9 |
| <=1024 | 879 | 284 | 742 | 12,522 | 16.89 | 14.25 | 59.2 | 12.3 |
| <=2048 | 1,577 | 213 | 1,443 | 18,647 | 12.92 | 11.83 | 77.4 | 13.7 |
| <=4096 | 2,944 | 118 | 2,811 | 18,790 | 6.68 | 6.38 | 149.6 | 7.7 |
| <=8192 | 7,448 | 151 | 31,181 | 87,101 | 2.79 | 11.69 | 358.0 | 45.4 |

`ms per want-token` is the prompt cost divided by the buffers the request asked for. A
request never lends more than its own prompt needs, so this table cannot show what a
different `--prefill` cap would have done - it shows how cost scales with the loan a request
already took. It is the shape the A/B script has to move along.

## Context, held inside one fresh-token bucket
| fresh tokens | context | requests | mean fresh | mean ms | ms per fresh tok |
| --- | ---: | ---: | ---: | ---: | ---: |
| 257-1024 | 0-20,000 | 45 | 549 | 9,352 | 17.05 |
| 257-1024 | 20,000-60,000 | 98 | 556 | 11,904 | 21.41 |
| 257-1024 | 60,000-120,000 | 165 | 535 | 10,579 | 19.78 |
| 257-1024 | 120,000-200,000 | 222 | 533 | 11,554 | 21.70 |
| 257-1024 | 200,000-inf | 136 | 485 | 10,200 | 21.05 |
| 1k-4k | 0-20,000 | 50 | 2,242 | 21,533 | 9.60 |
| 1k-4k | 20,000-60,000 | 84 | 2,120 | 18,860 | 8.90 |
| 1k-4k | 60,000-120,000 | 107 | 1,878 | 17,865 | 9.51 |
| 1k-4k | 120,000-200,000 | 67 | 1,630 | 18,681 | 11.46 |
| 1k-4k | 200,000-inf | 23 | 1,686 | 15,867 | 9.41 |
| 4k-16k | 0-20,000 | 63 | 9,260 | 40,124 | 4.33 |
| 4k-16k | 20,000-60,000 | 28 | 7,702 | 30,935 | 4.02 |
| 4k-16k | 60,000-120,000 | 21 | 6,919 | 32,513 | 4.70 |
| >16k | 20,000-60,000 | 11 | 32,019 | 78,992 | 2.47 |
| >16k | 60,000-120,000 | 15 | 84,976 | 241,514 | 2.84 |
| >16k | 120,000-200,000 | 3 | 132,966 | 263,803 | 1.98 |
| >16k | 200,000-inf | 7 | 242,051 | 510,945 | 2.11 |

Inside a bucket `fresh` barely moves, so a rise across the context rows is the attention
cost over the cached prefix. Where the rows are flat, context is not what costs time.

