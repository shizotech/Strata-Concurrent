#!/usr/bin/env python3
"""S3.1e-2: prove the SERIAL path's output calls are untouched.

Three mechanical claims.

 1. The serial driver text - `while (next_line(line)) { ... }` through `return 0;` - is
    byte-identical to the S3.1e-1 text.  That is .shz_cmd/s31e2_serial_proof.py, and it is the
    strongest form of the claim: the gate is an EARLY RETURN placed before the loop, so the
    loop's source was never edited at all.

 2. The serve block's output calls split exactly into "reachable from the serial path" and
    "inside a region that cannot run with --serve-slots 0/1".  The serial-reachable count must
    be 96, which is the number S3.1e-1's proof established for the pre-S3.1e-2 source.

 3. Every added call sits inside one of the two gated regions: the concurrent-driver block
    (needs `Registry::concurrent()`, i.e. --serve-slots >= 2) or the watchdog's `if (tagged)`
    branch (`tagged` is slots >= 2).  Neither can execute when the serial driver runs, and the
    serial loop itself gained none.
"""
import io, re, sys

OUT = re.compile(r"std::(printf|fprintf|cout|puts)\b")
DRIVER = "        // ==================== S3.1e-2: THE CONCURRENT DRIVER ===================="
TAIL = "        // ---- S3.1e-2: everything below is 0.1.30's serial driver, unchanged."
TAGGED = "                         if (tagged) {"
FB_A = "            // S3.1e-2-FALLBACK-MSG-BEGIN"
FB_B = "            // S3.1e-2-FALLBACK-MSG-END"
SERIAL_WATCHDOG = ('                         std::fprintf(stderr, "strata serve: no progress for %d s '
                   'during a request (%s %lld) - stopping "')
EXPECTED_SERIAL_CALLS = 109  # S3.1e-1 pinned 96; S3.2b added 4 serial-reachable stderr loan
                                   # diagnostics (pump / drain-idle / lend / return), all fprintf,
                                   # no new stdout protocol line. The stdout protocol count is
                                   # unchanged by that and by the ERR-routing fix (14 raw printf
                                   # -> 14 sp_out.err printf).
                                   # S3.6 (the parking-collapse fix) adds 9, and NONE of them can
                                   # execute at --serve-slots 0/1. They sit in the same textual
                                   # region as the pre-existing `swap:` lines the 100 already
                                   # counted, because the proof splits the serve block by the
                                   # driver's early return, not by `swaps_on`:
                                   #   4 stderr inside swap_to (the new park_guard hook: snapshot
                                   #     cannot be sized, the refusal line; the unmount hook's
                                   #     parked-N and NOT-parked-although-budget-enough);
                                   #   2 inside prep_request's hand-over block (1 stderr + 1
                                   #     sp_out.err), which is gated on `swaps_on && swap_needed`;
                                   #   1 stderr + 1 sp_out.resume + 1 more in the new
                                   #     reset_request_to_token0, which ONLY the concurrent driver
                                   #     calls.
                                   # `swap_to` returns at its first line when `!swaps_on`, and
                                   # `swaps_on` requires slots >= 2, so the serial path's behaviour
                                   # is unchanged; the serial LOOP still has 0 calls of its own. S3.2b added 4 serial-reachable stderr loan
                                   # diagnostics (pump / drain-idle / lend / return), all fprintf,
                                   # no new stdout protocol line. The stdout protocol count is
                                   # unchanged by that and by the ERR-routing fix (14 raw printf
                                   # -> 14 sp_out.err printf).

src = io.open("src/program/generate.cpp", encoding="utf-8").read()
a = src.index("    if (o.serve) {\n")
b = src.index("\n        return 0;\n", a)
serve = src[a:b]

def calls(t):
    return [re.sub(r"\s+", " ", l.strip()) for l in t.split("\n") if OUT.search(l)]

for m in (DRIVER, TAIL, TAGGED, SERIAL_WATCHDOG, FB_A, FB_B):
    if m not in serve:
        print("FAIL: marker not found:", m[:70])
        sys.exit(1)

ds, de = serve.index(DRIVER), serve.index(TAIL)
gate_a, gate_b = serve.index(FB_A), serve.index(FB_B) + len(FB_B)
# The gated region runs from the driver banner to the serial-loop comment; the fallback message
# inside it is gated on `slots_reg.concurrent()` too, so it is gated as well.  Split it out only so
# the count is legible.
assert ds < gate_a < gate_b < de
driver_block = serve[ds:gate_a] + serve[gate_b:de]
fallback_msg = serve[gate_a:gate_b]
rest = serve[:ds] + serve[de:]
ws, we = rest.index(TAGGED), rest.index(SERIAL_WATCHDOG)
tagged_branch = rest[ws:we]
serial_reachable = rest[:ws] + rest[we:]

n_all = len(calls(serve))
n_drv = len(calls(driver_block))
n_tag = len(calls(tagged_branch))
n_fb = len(calls(fallback_msg))
n_ser = len(calls(serial_reachable))
print("serve block, output calls total           : %d" % n_all)
print("  in the concurrent-driver block   (gated): %d   (needs Registry::concurrent())" % n_drv)
print("  in the driver-fallback message   (gated): %d   (needs Registry::concurrent())" % n_fb)
print("  in the watchdog's if(tagged)     (gated): %d   (needs slots >= 2)" % n_tag)
print("  reachable from the serial path          : %d" % n_ser)

ok = True
if n_drv + n_tag + n_fb + n_ser != n_all:
    print("FAIL: the split does not account for every call (%d + %d + %d + %d != %d)"
          % (n_drv, n_tag, n_fb, n_ser, n_all))
    ok = False
if n_ser != EXPECTED_SERIAL_CALLS:
    print("FAIL: the serial-reachable output calls changed: %d, expected %d" % (n_ser, EXPECTED_SERIAL_CALLS))
    for x in calls(serial_reachable):
        print("   " + x[:110])
    ok = False

loop_start = serve.index("        while (next_line(line)) {")
loop_calls = calls(serve[loop_start:])
if loop_calls:
    print("FAIL: the serial loop itself gained output calls:")
    for x in loop_calls:
        print("   " + x[:110])
    ok = False
else:
    print("the serial loop (`while (next_line(line))` .. `return 0;`) contains 0 output calls of its")
    print("own, exactly as before: every per-request line comes from the step functions it calls.")

print("PASS" if ok else "FAILED")

# ---- stage 3 wiring guard: no per-request protocol line may bypass the formatter -------------
# A raw `printf("ERR ...")` inside the step functions is unroutable in a tagged session: the
# server's reader thread cannot tell whose error it is and drops it, so the client hangs until
# its own watchdog fires. This found 16 such sites on 2026-10-02; they must stay at zero.
# INFO/READY are process-wide (the server routes them to `control`, not to a request), so they
# are legitimately untagged; the per-request verbs are not.
RAW = re.compile(r'std::printf\("(?:ERR|T|DONE|RESUME|REUSED|PP|SLOT) ')
raw_hits = [l.strip() for l in serve.split("\n") if RAW.search(l)]
if raw_hits:
    print("FAIL: per-request protocol lines that bypass sp_out (unroutable when tagged):")
    for x in raw_hits:
        print("   " + x[:110])
    ok = False
else:
    print("no per-request protocol line bypasses sp_out (every one carries its request id)")

sys.exit(0 if ok else 1)
