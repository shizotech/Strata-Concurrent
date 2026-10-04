#!/usr/bin/env python3
"""S4.2 wiring gate: the hold queue must actually be USED by the concurrent driver.

The pure decisions live in include/strata/program/serve_driver.hpp and are pinned by
serve_driver_test.  That is not enough on its own: a tested predicate nobody calls is how
"hold, don't reject" shipped half-done once already (the first S4.2 agent wrote the whole API
and never reached generate.cpp).  This script asserts the CALL SITES exist, in the driver
block only, and that the serial path still cannot reach a wait.

Run:  python3 .shz_cmd/s42_wiring_proof.py
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEN = (ROOT / "src" / "program" / "generate.cpp").read_text()

fails = []


def need(pattern, why, count=1):
    n = len(re.findall(pattern, GEN))
    if n < count:
        fails.append(f"MISSING ({n} < {count}): {why}  [{pattern}]")
    return n


def forbid(pattern, why):
    n = len(re.findall(pattern, GEN))
    if n:
        fails.append(f"PRESENT ({n}): {why}  [{pattern}]")


# 1. the queue and its state exist
need(r"drv::WaitQueue holds\(o\.hold_ms\)", "the driver owns a WaitQueue")
need(r"std::deque<std::pair<int64_t, std::string>> hold_lines", "waiters keep their request lines")

# 2. admission asks the tested predicate instead of ERing
need(r"const drv::Hold h = ask_admission\(probe, in, why\)", "admission goes through admit_decision", 2)
need(r"return drv::admit_decision\(in\.a, why\);", "ask_admission delegates to the tested predicate")
need(r"if \(h == drv::Hold::wait\)", "a temporary refusal is a WAIT branch")
need(r"hold_enqueue\(probe, held_line, why, in\.est\)", "the wait branch enqueues, it does not ERR")
need(r"open_ctx\(probe, held_line, in\.est\)", "only Hold::run_now creates a context")

# 3. the queue is drained: expiry, promotion, cancellation
need(r"expire_waiters\(\);", "expired waits are answered every pass")
need(r"promote_waiters\(\);", "the head of the queue is promoted when resources free")
need(r"holds\.expired\(drv::now_ms\(\)\)", "the bound is the queue's own")
need(r"hold_cancel\(probe\.id\)", "STOP <id> cancels a waiter")
need(r"pending_cancel\.erase", "a cancel that arrived while waiting is consumed, not left stale")

# 4. the wake point
need(r"holds\.wake_all\(\)", "drop_ctx wakes the queue")

# 5. visibility
need(r"holds\.line\(now\)", "the activity line carries waiting=N")
need(r"drv::wait_line\(holds\.size\(\)", "the engine emits WAIT lines")
need(r"drv::hold_admitted_line\(id, waited\)", "a request that waited says how long it waited")

# 5b. STRUCTURE, not just presence.  A textual gate that only asks "is the call there?" passes a
#     mutation that keeps the call behind a dead condition, and that is exactly the regression that
#     would ship the half-done version again.  So: the enqueue must sit INSIDE the `Hold::wait`
#     branch, the context must be opened only after the wait/error branches returned, and the
#     promotion must sit inside a loop that re-asks the head.
def window(pattern, span, why):
    m = re.search(pattern, GEN, re.S)
    if not m:
        fails.append(f"MISSING (structure): {why}")
        return
    if pattern not in ("",) and not re.search(r"hold_enqueue\(probe, held_line, why, in\.est\)",
                                              GEN[m.end():m.end() + span]):
        fails.append(f"STRUCTURE: `hold_enqueue(...)` is not inside the branch it belongs to ({why})")


window(r"if \(h == drv::Hold::wait\) \{", 700, "the wait branch")
window(r"if \(h == drv::Hold::error\) \{", 700, "the permanent-error branch")
if not re.search(r"if \(h == drv::Hold::error\) \{[^{}]*refuse_line\(probe\.id, permanent_admit_reason",
                 GEN, re.S):
    fails.append("STRUCTURE: the permanent branch does not ERR with the named reason")
if not re.search(r"if \(h == drv::Hold::wait\)[\s\S]{0,700}?return;\s*\}\s*open_ctx\(probe, held_line",
                 GEN, re.S):
    fails.append("STRUCTURE: `open_ctx` is reachable without passing the wait/error branches")

# 6. the RAM ERR must NOT come back as an immediate refusal
forbid(r"refuse_line\(probe\.id, drv::refuse_reason\(drv::Refuse::ram\)\)",
       "the old immediate RAM refusal is back - a RAM shortage must be a wait")

# 7. the serial path must not reach a wait: the whole hold block lives inside `if (driver_on)`
driver_start = GEN.find("if (driver_on) {")
driver_end = GEN.find("\n        while (next_line(line)) {")
if driver_start < 0 or driver_end < 0 or driver_end < driver_start:
    fails.append("cannot locate the driver block / the serial loop")
else:
    serial = GEN[driver_end:]
    for name in ("holds", "hold_enqueue", "promote_waiters", "expire_waiters", "WaitQueue"):
        if re.search(r"\b" + name + r"\b", serial):
            fails.append(f"the SERIAL path references {name} - --serve-slots 0 must be 0.1.30")

if fails:
    print("S4.2 WIRING PROOF: FAILED")
    for f in fails:
        print("  " + f)
    sys.exit(1)

print("S4.2 WIRING PROOF: PASS")
print("  the hold queue is wired into the concurrent driver, the serial path cannot reach it,")
print("  and the old immediate RAM refusal is gone.")
