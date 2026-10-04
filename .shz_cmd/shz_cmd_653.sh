cd /ssd/Strata && python3 - <<'PY'
p="src/program/prefill_svc_test.cpp"
s=open(p).read()
def rep(a,b):
    global s
    assert a in s, "MISSING: "+a[:80]
    s=s.replace(a,b,1)

rep('''    const Line tr = parse_line(ready_line(ReadyInfo{{kProtoVersion, {2, 2, 2}, {1024, 16384, 131072},
                                                     131072, 0x0123456789abcdefull,
                                                     0x5dfa917bae1331c5ull}}) + " future=1");''',
    '''    ReadyInfo ri2;
    ri2.slots = {2, 2, 2};
    ri2.tiers = {1024, 16384, 131072};
    ri2.max_tokens = 131072;
    ri2.pack_hash = 0x0123456789abcdefull;
    ri2.geom_hash = 0x5dfa917bae1331c5ull;
    const Line tr = parse_line(ready_line(ri2) + " future=1");''')
rep('    check((int) kAllStates.length() == kStateCount, "the test enumerates every state");\n    check((int) kAllEvents.length() == kEventCount, "the test enumerates every event");',
    '    check((int) (sizeof kAllStates / sizeof kAllStates[0]) == kStateCount, "the test enumerates every state");\n    check((int) (sizeof kAllEvents / sizeof kAllEvents[0]) == kEventCount, "the test enumerates every event");')
rep('''    check(a.note(12, Event::ack_sent).action == Act::illegal, "ACK before DONE is refused");
    check_reason_contains(a.why, "not READY", "and says why");''',
    '''    const auto early_ack = a.note(12, Event::ack_sent);
    check(early_ack.action == Act::illegal, "ACK before DONE is refused");
    check_reason_contains(early_ack.why, "not READY", "and says why");''')
rep('''    check(d.machine(1).may_send_req(0), "the machine is ready to ask: the socket, not the machine, is what failed")
    const auto held''','''    check(d.machine(1).may_send_req(0), "the machine is ready to ask: the socket, not the machine, is what failed");
    const auto held''')
open(p,"w").write(s)
print("ok")
PY
g++ -std=c++20 -O0 -Iinclude -o /tmp/prefill_svc_test src/program/prefill_svc_test.cpp 2>&1 | head -40
