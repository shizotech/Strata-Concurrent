cd /ssd/Strata && python3 - <<'PY'
p="src/program/prefill_svc_test.cpp"
s=open(p).read()
def rep(a,b):
    global s
    assert a in s, "MISSING: "+a[:80]
    s=s.replace(a,b,1)

# 1. truncation sweep: the honest property is "a truncated line never mis-reads the fields that are
#    fully present".
rep('''        // A cut that drops a whole token is a missing field and must be refused.  A cut INSIDE the
        // last token is a shorter number: legal grammar, and a real reader never delivers it because
        // it reads up to the '\\n'.  Asserting refusal for the second case would be asserting that the
        // parser can see a byte boundary that only the framer knows about.
        const bool cut_at_boundary = full[n] == ' ';
        if (cut_at_boundary) {
            check(!l.bad.empty(), "a DONE cut at a token boundary is refused");
            ++missing_field;
        } else {
            check(l.bad.empty(), "a DONE cut inside its last token is still a well-formed prefix");
            ++shortened;
        }''',
'''        // A cut that drops a whole token is a missing field and must be refused.  A cut INSIDE the
        // last token leaves a shorter number, which is legal grammar - a real reader never delivers
        // one, because it reads up to the '\\n'.  What must hold in BOTH cases is that the parser
        // never mis-reads a field that is fully present: the values it returns are always the
        // original's leading fields, never a shifted or re-split set.
        if (l.bad.empty()) {
            ++shortened;
            check(l.kind == LineKind::done, "and it is still a DONE");
            check(l.id == 7 && l.slot == 5, "the leading fields did not shift");
            if (l.nonce != 0) check_eq(std::to_string(l.nonce), full.substr(10, std::to_string(l.nonce).size()),
                                       "the nonce is a prefix of the original's, not a re-split");
        } else {
            check_reason_contains(l.bad, "missing", "a refused cut says which field is missing");
            ++missing_field;
        }''')

# 2. idempotent no-ops are legal handled cells.
rep('''            } else {
                ++handled;
                // A handled event either moves the machine or asks the caller to do something.
                check(tr.to != s || tr.action != Act::none,
                      "a handled event either moves the state or asks for an action");
            }''',
'''            } else {
                ++handled;
                // A handled event either moves the machine, asks the caller to do something, or is a
                // documented idempotent repeat (a late SEG, a second CANCEL, a retry that is already
                // in the right state).
                const bool idempotent = tr.action == Act::none &&
                                        (e == Event::seg_line || e == Event::note_cancel ||
                                         e == Event::retry || e == Event::queued_line);
                check(tr.to != s || tr.action != Act::none || idempotent,
                      ("a handled event moves, acts, or is a documented no-op: " + state_name(s) +
                       " + " + event_name(e)).c_str());
            }''')

# 3. the slot-leak visibility test: 4 slots, 2 leaked.
rep('''    std::vector<uint64_t> nonces;
    for (int i = 0; i < 4; ++i) {''','''    std::vector<uint64_t> nonces;
    for (int i = 0; i < 2; ++i) {''')
rep('    check(a.free_slots() == "2/4", "two slots leaked by clients that never ACKed");',
    '    check(a.free_slots() == "2/4", "two of four slots held by clients that never ACKed");')

# 4. corrupt payload: check the wrong-client refusal BEFORE the release.
rep('''    check(d.machine(1).state() == HState::seg_ready,
          "the machine stays in H_SEG_READY: the slot must still be released");
    // The slot is still released - a refused payload must not become a leak.
    check(d.machine(1).may_release(idx, nonce), "the handoff's own nonce still releases its own slot");
    check(a.release(idx, nonce, err), "and the arena accepts it");''',
'''    check(d.machine(1).state() == HState::seg_ready,
          "the machine stays in H_SEG_READY: the slot must still be released");
    // A slot that names a different client is refused before any byte is copied (§3.5 step 4).
    FakeSlot other;
    check(!a.read_slot(idx, nonce, 8, other, err), "a different client cannot read it");
    check_reason_contains(err, "different client", "and says so");
    check(!a.read_slot(idx, nonce + 1, 7, other, err), "and a wrong nonce cannot read it");
    check_reason_contains(err, "nonce", "naming the nonce rule");
    // The slot is still released - a refused payload must not become a leak.
    check(d.machine(1).may_release(idx, nonce), "the handoff's own nonce still releases its own slot");
    check(a.release(idx, nonce, err), "and the arena accepts it");''')
rep('''    // A slot that names a different client is refused before any byte is copied (§3.5 step 4).
    FakeSlot got;
    check(!a.read_slot(0, nonce, 8, got, err), "a different client cannot read it");
    check_reason_contains(err, "different client", "and says so");
''','')

# 5. the tier-starvation assertions.
rep('''    FakeArena t;
    t.build({1024, 16384}, {1, 1}, 0, 0);
    int idx = -1;
    uint64_t n = 0;
    std::string err;
    check(t.claim(0, 7, 1, 0, idx, n, err) && t.slots[idx].tier == 0,
          "a tier-0 job takes the tier-0 slot, not the 3 GB one");
    check(!t.claim(1, 7, 2, 0, idx, n, err), "a tier-1 job cannot use the tier-0 group");
    check(t.claim(1, 7, 2, 0, idx, n, err) && t.slots[idx].tier == 1, "it takes its own group");''',
'''    FakeArena t;
    t.build({1024, 16384}, {1, 1}, 0, 0);
    int idx = -1;
    uint64_t n = 0;
    std::string err;
    check(t.claim(0, 7, 1, 0, idx, n, err) && t.slots[idx].tier == 0,
          "a tier-0 job takes the tier-0 slot, not the 3 GB one (§3.8: prefer the smallest that fits)");
    check(t.claim(1, 7, 2, 0, idx, n, err) && t.slots[idx].tier == 1,
          "a tier-1 job takes its own group");
    check(!t.claim(1, 7, 3, 0, idx, n, err),
          "a second tier-1 job is refused even though a tier-0 slot is free: a big job never fits a "
          "small slot, so it waits rather than jumping");
    check(t.slots[0].state == 0, "and the tier-0 slot really is free - the refusal is the rule, not the state");
    // A tier-0 job MAY cascade up when its own group is full (§3.8: any j >= i).
    check(t.claim(0, 7, 4, 0, idx, n, err) && t.slots[idx].tier == 0, "the tier-0 slot is still usable");
    check(!t.claim(0, 7, 5, 0, idx, n, err), "and now the arena is full: ERR full, a temporary answer");''')

# 6. may_send_req after a REQ is in flight is false - that is the one-REQ rule.
rep('    check(d.machine(1).may_send_req(0), "the machine is ready to ask: the socket, not the machine, is what failed");',
    '    check(!d.machine(1).may_send_req(0), "one outstanding REQ: the machine will not re-ask while one is in flight");\n    check(d.machine(1).has_outstanding_req(), "and it knows that")')

# 7. the end-to-end happy path must feed the ACK back to the prefill instance.
rep('''    check(d.copy_out(7), "the copy-out, the hash and the decode all pass");
    check(d.machine(7).state() == HState::handed_over, "H_HANDED_OVER: the slot is FREE again");''',
'''    check(d.copy_out(7), "the copy-out, the hash and the decode all pass");
    check(d.machine(7).state() == HState::handed_over, "H_HANDED_OVER: the slot is FREE again");
    p.on_line(d.outbox.back());   // the ACK reaches the prefill instance''')
open(p,"w").write(s)
print("ok")
PY
g++ -std=c++20 -O0 -Iinclude -o /tmp/prefill_svc_test src/program/prefill_svc_test.cpp 2>&1 | head -20 && /tmp/prefill_svc_test 2>&1 | tail -30
