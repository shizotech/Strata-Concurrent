cd /ssd/Strata && python3 - <<'PYEOF'
# Use the LOG's own numbers: 7777 slots = 14.75 GiB of VRAM
slots=7777; vram=14.75*2**30
blob=vram/slots
layers,topk=48,10
per_tok=layers*topk*blob
bw=936e9
print(f"expert blob            : {blob/1e6:.2f} MB   (7777 slots = 14.75 GiB, log:3808)")
print(f"experts per token      : {layers}x{topk} = {per_tok/1e6:.0f} MB")
print(f"pure HBM read          : {per_tok/bw*1000:.2f} ms/token")
print(f"measured               : 16.4 ms/token (61 tok/s)")
print(f"=> GPU bandwidth used  : {per_tok/bw/0.0164*100:.1f} %")
print()
miss=(1-0.883)
print(f"at 88.3% hit: {(per_tok*miss)/1e6:.0f} MB/token comes from host RAM")
for ram in (30e9,50e9):
    print(f"   = {per_tok*miss/ram*1000:.2f} ms at {ram/1e9:.0f} GB/s")
print()
print("=== the real prize: 93.6% of the GPU's decode bandwidth is IDLE ===")
print("rows that fit before the marginal cost overtakes the HBM term:")
for T in (6,8,12,16,24,32,48):
    for slope,label in ((8.4,"measured slope"),(1.05,"HBM-only slope")):
        ms=24+(T-1)*slope
        print(f"  T={T:2d} {label:16s}: {ms:6.1f} ms for {T} rows = {ms/T:5.2f} ms/row = {1000*T/ms:5.1f} tok/s")
    print()
PYEOF
