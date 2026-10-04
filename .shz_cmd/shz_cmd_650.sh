cd /ssd/Strata && python3 - <<'PY'
import re
p="src/program/prefill_svc_test.cpp"
s=open(p).read()
def rep(a,b):
    global s
    assert a in s, "MISSING: "+a[:70]
    s=s.replace(a,b,1)

rep("    p.arena.note_client_pid_unused_guard();\n","")
rep("    check(!h.begin(1, 1, 0, w), \"IDLE cannot begin twice - it has not started\");   // no-op guard\n    check(h.admit_now(1), \"admit_now from IDLE\");\n    check(!h.begin(1, 1, 0, w) == false, \"begin from H_QUEUED is legal\");",
    "    check(h.admit_now(1), \"admit_now from IDLE\");\n    check(h.begin(1, 1, 0, w), \"begin from H_QUEUED is legal\");\n    check(h.state() == HState::init, \"and it moved to H_INIT\")")
rep("    check(!d.machine(1).may_send_req(0) == false, \"the ask is gated by the backoff, not by the socket\");",
    "    check(d.machine(1).may_send_req(0), \"the machine is ready to ask: the socket, not the machine, is what failed\")")
rep("            check_reason_contains(tr.why, \"\", \"every cell carries a reason\");\n",
    "            check(tr.why[0] != '\\0', \"every cell carries a reason\");\n")
rep("        check(h.ack_line().empty() || true, \"the machine has a nonce-free release path\");\n","")
rep("    check(a.publish(idx, nonce, 100, err), \"published, waiting for an ACK that will never come\");",
    "    check(a.publish(idx, nonce, 100, err), \"published, waiting for an ACK that will never come\");\n    a.slots[idx].published_ms = 0;   // the lease runs from the publish")
rep("    check(a.publish(idx, nonce, 64, err), \"publish\");","    check(a.publish(idx, nonce, 64, err), \"publish\");")
rep("    check(b.publish(idx, nonce, 8, err), \"publish\");","    check(b.publish(idx, nonce, 8, err), \"publish\");")
rep("    d.feed(d.peer[3]);   // DONE","    d.feed(d.peer[3]);   // DONE\n    p.arena.slots[0].published_ms = 0;")
rep("    check(lookup(HState::held, Event::admit).action == Act::hold,\n          \"while held, the answer to the scheduler is `hold`, never a socket wait (§6.2)\");",
    "    check(lookup(HState::held, Event::admit).action == Act::hold,\n          \"while held, the answer to the scheduler is `hold`, never a socket wait (§6.2)\");\n"
    "    // §4.4's retry column: `full`/`queue` are HOLDS and do not consume a retry, while `read` and\n"
    "    // `slotlost` do.  A full arena is hammered by the backoff, not by retries (§3.8).\n"
    "    HandoffState f(1);\n"
    "    f.begin(4, 100, 0, w);\n"
    "    f.note_now_ms(0);\n"
    "    for (int i = 0; i < 3; ++i) {\n"
    "        f.note(4, Event::req_sent);\n"
    "        f.on_line(4, parse_line(err_line(4, ErrCode::full, \"no slot\")));\n"
    "        f.note(4, Event::retry);\n"
    "    }\n"
    "    check(f.req_lines() == 4, \"three retries plus the first ask\");\n"
    "    check(f.backoff_ms() == 400, \"and the backoff doubled 50 -> 100 -> 200 -> 400\");")
open(p,"w").write(s)
print("ok")
PY
