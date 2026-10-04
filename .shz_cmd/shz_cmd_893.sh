cd /ssd/Strata && python3 - <<'PYEOF'
arena=46.84e9; slots=7777; blob=arena/slots; layers,topk=48,10
print("=== where the 8.4 ms marginal row goes ===")
for miss in (0.05,0.117,0.15):
    cpu_bytes = layers*topk*miss*blob
    # 5 workers, DDR4 dual channel ~40 GB/s effective, and a per-layer barrier
    for ram_bw in (30e9, 50e9):
        t = cpu_bytes/ram_bw*1000
        print(f"  miss {miss*100:4.1f}% -> {cpu_bytes/1e6:6.0f} MB from RAM = {t:5.2f} ms at {ram_bw/1e9:.0f} GB/s")
print()
print("  per-layer barrier: 48 layers x (doorbell round trip)")
for rt in (0.02,0.05,0.10):
    print(f"    at {rt*1000:.0f} us/layer -> {48*rt:.1f} ms/token")
print()
print("=== the ceiling if the slope were bandwidth-only (3.09 ms/row) ===")
for T,acc in ((6,0.665),(12,0.665),(24,0.665),(32,0.665)):
    for slope,label in ((8.4,"measured"),(3.09,"bandwidth floor")):
        ms = 24 + (T-1)*slope
        toks = T  # rows are tokens when B sequences each draft; approx
        print(f"  T={T:2d} {label:16s}: {ms:6.1f} ms for {T} rows -> {ms/T:5.2f} ms/row")
PYEOF
