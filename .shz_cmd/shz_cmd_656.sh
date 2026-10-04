cd /ssd/Strata && python3 - <<'PY'
p="include/strata/program/prefill_svc.hpp"
s=open(p).read()
def rep(a,b):
    global s
    assert a in s, "MISSING: "+a[:80]
    s=s.replace(a,b,1)

# BYE's reason is free text, like ERR's message: keep the rest of the line.
rep('''        case LineKind::bye: {
            if (i < t.size()) l.reason = t[i++];
            for (; i < t.size(); ++i) l.extra.push_back(t[i]);
            return l;
        }''',
'''        case LineKind::bye: {
            // The reason is free text (§4.3's `BYE <reason>`): keep the rest of the line so a shutdown
            // reason with spaces survives the round trip.
            if (i < t.size()) {
                size_t p = detail::past_token(line, 1);
                while (p < line.size() && (line[p] == \' \' || line[p] == \'\\t\')) ++p;
                l.reason = line.substr(p);
            }
            return l;
        }''')

# slots_free is "4/6" in the INFO line (the arena's own free_slots() spelling).
rep('''inline bool info_shows_a_free_slot(const Line& l) {
    if (l.kind != LineKind::info) return false;
    const std::string free = info_value(l, "slots_free");
    if (free.empty()) return false;
    uint64_t v = 0;
    return detail::is_uint(free, v) && v > 0;
}''',
'''inline bool info_shows_a_free_slot(const Line& l) {
    if (l.kind != LineKind::info) return false;
    const std::string free = info_value(l, "slots_free");
    if (free.empty()) return false;
    // `HandoffArena::free_slots()` spells it "4/6"; a bare "4" is accepted too.
    const size_t slash = free.find(\'/\');
    const std::string head = slash == std::string::npos ? free : free.substr(0, slash);
    uint64_t v = 0;
    return detail::is_uint(head, v) && v > 0;
}''')

# startable() must include H_QUEUED: begin() from the wait-queue head is the normal path.
rep('''    bool startable() const {
        return st_ == HState::idle || st_ == HState::done || st_ == HState::failed ||
               st_ == HState::cancelled;
    }''',
'''    bool startable() const {
        return st_ == HState::idle || st_ == HState::queued || st_ == HState::done ||
               st_ == HState::failed || st_ == HState::cancelled;
    }''')
open(p,"w").write(s)
print("header ok")
PY
