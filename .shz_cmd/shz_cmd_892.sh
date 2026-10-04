cd /ssd/Strata && python3 - <<'PYEOF'
# First-principles bound on what a decode window costs vs what the 3090 can deliver.
arena = 46.84e9            # GiB->B approx: the whole expert arena, IQ3_S
slots = 7777
slot_bytes = arena/slots   # per expert blob
layers, topk = 48, 10
per_tok_experts = layers*topk*slot_bytes
print(f"expert blob          : {slot_bytes/1e6:.2f} MB")
print(f"experts per token    : {layers}x{topk} = {per_tok_experts/1e6:.0f} MB")
bw = 936e9                 # RTX 3090 HBM bandwidth
print(f"pure HBM read time   : {per_tok_experts/bw*1000:.2f} ms/token")
print(f"measured decode rate : ~16.4 ms/token  (61 tok/s)")
print(f"=> GPU bandwidth utilisation: {per_tok_experts/bw/0.0164*100:.1f} %")
print()
# what the window actually costs
print("window fixed cost 24 ms, marginal 8.4 ms/row")
for T in (6,8,12,16,24,32):
    ms = 24 + (T-1)*8.4
    print(f"  T={T:2d} rows -> {ms:6.1f} ms/window")
PYEOF
