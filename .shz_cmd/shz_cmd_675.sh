cd /ssd/Strata && python3 - <<'PY'
p="include/strata/program/prefill_svc.hpp"
s=open(p).read()
a='''            if (i < t.size()) {
                size_t p = detail::past_token(line, 1);'''
b='''            if (i < t.size()) {
                size_t p = detail::past_token(line, 0);   // past token 0 = past "BYE"'''
assert a in s
s=s.replace(a,b,1)
open(p,"w").write(s)

p="src/program/prefill_svc_test.cpp"
s=open(p).read()
a='''    // The prefill instance claims the only slot for request 1 and never gets an ACK.
    p.on_line(req_line({1, 100, 0, "a.ids"}));'''
b='''    // The prefill instance claims the only slot for request 1 and never gets an ACK.
    d.ask(1, 100, 0, "a.ids");
    p.on_line(req_line({1, 100, 0, "a.ids"}));'''
assert a in s
s=s.replace(a,b,1)
a='''    uint64_t n_slot0 = 0;
    {
        // re-read the nonce of the tier-0 claim so the release is legal
        n_slot0 = t.slots[0].nonce;
    }
    check(t.release(0, n_slot0, err), "release the tier-0 slot");'''
b='''    check(t.release(0, t.slots[0].nonce, err), "release the tier-0 slot");'''
assert a in s
s=s.replace(a,b,1)
a='''    check(t.release(idx == 0 ? 1 : 0, n, err) == false || true, "the release below is the real one");
'''
assert a in s
s=s.replace(a,'',1)
open(p,"w").write(s)
print("ok")
PY
g++ -std=c++20 -O0 -Iinclude -o /tmp/prefill_svc_test src/program/prefill_svc_test.cpp 2>&1 | grep -v sframe | head -20; echo "=== RUN ==="; /tmp/prefill_svc_test 2>&1 | tail -20
