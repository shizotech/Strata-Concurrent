cd /ssd/Strata && python3 - <<'PYEOF'
from pathlib import Path
p = Path("/ssd/Strata/src/core/conversation_cache_test.cpp")
s = p.read_text()
old = '''        const uint64_t u0 = c.use(0), u1 = c.use(1), clock = c.clock();
        check(c.probe(std::vector<int32_t>{1, 2}, {}, true).tokens == 2,
              "probe answers the same prefix search best() does");
        check(c.probe(std::vector<int32_t>{1, 2, 3}, {}, true).tokens == 3, "including the full branch");
        check(c.clock() == clock && c.use(0) == u0 && c.use(1) == u1,
              "and it credits NOTHING - asking is not using");
        // The contrast that makes the check non-vacuous: the very next `best()` DOES credit.
        check(c.best(std::vector<int32_t>{1, 2, 3}, {}, true).tokens == 3, "best() still finds it");
        check(c.clock() > clock && c.use(0) > u0, "best() credits; probe() did not");'''
new = '''        const uint64_t u0 = c.use(0), u1 = c.use(1), clock = c.clock();
        // The prefix rule is the same one `find()` applies: the QUERY must be strictly longer than the
        // parked branch, because a request that resumes still reads its last token.
        check(c.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 3,
              "probe answers the same prefix search best() does");
        check(c.probe(std::vector<int32_t>{1, 2, 3}, {}, true).tokens == 0,
              "and it keeps the strict-prefix rule - an equal prompt is not a resume");
        check(c.probe(std::vector<int32_t>{9, 8, 7, 6}, {}, true).tokens == 3, "the other branch too");
        check(c.clock() == clock && c.use(0) == u0 && c.use(1) == u1,
              "and it credits NOTHING - asking is not using");
        // The contrast that makes the check non-vacuous: the very next `best()` DOES credit.
        check(c.best(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 3, "best() still finds it");
        check(c.clock() > clock && c.use(0) > u0, "best() credits; probe() did not");'''
assert old in s
s = s.replace(old, new)
old2 = '''        check(c.probe(std::vector<int32_t>{1, 2, 3}, {}, true, 7).tokens == 3,
              "its owner may probe its own branch");
        check(c.probe(std::vector<int32_t>{1, 2, 3}, {}, true, 8).tokens == 0,
              "another request's probe finds nothing (S3.9's claim, unchanged)");
        check(c.probe(std::vector<int32_t>{1, 2, 3}, {}, true).tokens == 0,
              "and an unclaimed probe hides nothing either");'''
new2 = '''        check(c.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true, 7).tokens == 0,
              "even its own owner does not get it back by PREFIX - the claim hides it from reuse");
        check(c.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true, 8).tokens == 0,
              "and another request certainly does not (S3.9's claim, unchanged)");
        check(c.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 3,
              "an unclaimed probe (no requester) sees it - which is what keeps stage 2's prefix reuse");'''
assert old2 in s
s = s.replace(old2, new2 + '''
        c.release_owner(7);   // ... and only until its request goes away
        check(c.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true, 8).tokens == 3,
              "once released, another request may reuse the branch");''')
old3 = '''        off.put(image({1, 2, 3}));
        check(off.probe(std::vector<int32_t>{1, 2, 3}, {}, true).tokens == 0,'''
new3 = '''        off.put(image({1, 2, 3}));
        check(off.probe(std::vector<int32_t>{1, 2, 3, 4}, {}, true).tokens == 0,'''
assert old3 in s
s = s.replace(old3, new3)
p.write_text(s)
print("ok")
PYEOF
grep -n "release_owner" include/strata/core/conversation_cache.hpp | head -3
