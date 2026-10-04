cd /ssd/Strata && python3 - <<'PY'
p="include/strata/program/prefill_svc.hpp"
s=open(p).read()
def rep(a,b):
    global s
    assert a in s, "MISSING: "+a[:80]
    s=s.replace(a,b,1)

# The LOCAL path builder may be absolute (that is what --prefill-handoff-dir is); the WIRE name may
# not.  Make that the documented split.
rep('''/// The §4.3 ids-file name (D7).  `dir` is the instance directory; the file is
/// `req-<client>-<nonce>.ids`, mode 0600, deleted after `ACK`.  Returns false rather than build a
/// path a peer could point outside the instance directory.''',
'''/// The §4.3 ids-file name (D7).  `dir` is the instance directory; the file is
/// `req-<client>-<nonce>.ids`, mode 0600, deleted after `ACK`.  This builds the LOCAL path from
/// `--prefill-handoff-dir`, which is normally absolute, so an absolute `dir` is allowed unless the
/// caller says otherwise - what is never allowed is a `..` component, in either direction.  The
/// wire-side rule (a name arriving in `ids=`) is `ids_name_is_safe()`, which refuses absolute names
/// as well.''')
rep('''inline bool req_ids_path(const std::string& dir, uint64_t client_id, uint64_t nonce,
                         std::string& out, bool allow_absolute = false) {''',
'''inline bool req_ids_path(const std::string& dir, uint64_t client_id, uint64_t nonce,
                         std::string& out, bool allow_absolute = true) {''')
open(p,"w").write(s)

# swap the regenerated tables
h=open(".shz_cmd/handoff_table.inc").read().rstrip("\n")
j=open(".shz_cmd/job_table.inc").read().rstrip("\n")
def swap(name, new):
    global s
    i = s.index(name)
    k = s.index("\n}\n", i) + 2
    s = s[:i] + new + "\n" + s[k:]
swap("inline const Transition* handoff_table(int& count) {", h)
swap("inline const JobTransition* prefill_job_table(int& count) {", j)
open(p,"w").write(s)
print("header ok")
PY
