cd /ssd/Strata && python3 - <<'PY'
import re
p="include/strata/program/prefill_svc.hpp"
lines=open(p).read().split("\n")
# rename the deviation labels D1..D14 -> DEV-1..DEV-14 within the banner block only
start=None; end=None
for i,l in enumerate(lines):
    if l.startswith("// D1. "): start=i
    if l.startswith("// THE INVARIANTS THIS FILE EXISTS TO KEEP"): end=i; break
assert start is not None and end is not None
for i in range(start,end):
    lines[i]=re.sub(r"^// (D)(1[0-4]|[1-9])\. ", r"// DEV-\2. ", lines[i])
    lines[i]=re.sub(r"^// (D)(1[0-4]|[1-9])\. \*\*", r"// DEV-\2. **", lines[i])
s="\n".join(lines)
# fix in-body references to the deviation numbers
s=s.replace("(D3 of §1.6)","(manager decision D3, §1.6 rule 2)")
s=s.replace("§4.2, D3 of §1.6","§4.2 and manager decision D3")
s=s.replace("nothing from `strata/program/` (D3) and","nothing from `strata/program/` (manager decision D3) and")
s=s.replace("nothing has been sent (D3)","nothing has been sent (DEV-3)")
s=s.replace("the file NAME is validated.**  §4.3 says","the file NAME is validated.**  §4.3 says")
s=s.replace("Same rule as D7.","Same rule as DEV-7.")
s=s.replace("(D7).  `dir` is the instance directory","(DEV-7).  `dir` is the instance directory")
s=s.replace("the parser returns `unknown`","the parser returns `unknown`")
s=s.replace("unknown verb, not an error","unknown verb, not an error")
s=s.replace("return l; }   // D8: unknown verb, not an error","return l; }   // DEV-8: unknown verb, not an error")
s=s.replace("if ((int64_t) l.split.size() != l.n_stages) {   // D9","if ((int64_t) l.split.size() != l.n_stages) {   // DEV-9")
s=s.replace("§4.4's table, row by row","§4.4's table, row by row")
s=s.replace("// D5's rule, restated","// DEV-5's rule, restated")
s=s.replace("/// §6.1, D6: the ONLY authoritative","/// §6.1, DEV-6: the ONLY authoritative")
s=s.replace("/// The §4.3 ids-file name (D7).","/// The §4.3 ids-file name (DEV-7).")
s=s.replace("Same rule as D7, applied","Same rule as DEV-7, applied")
s=s.replace("§4.2's grammar for `SEG`","§4.3's grammar for `SEG`")
s=s.replace("a version mismatch, a pack mismatch","a version mismatch, a pack mismatch")
s=s.replace("(D9).  The doc does not","(DEV-9).  The doc does not")
s=s.replace("§4.3's `NAME` matches","§4.3's `NAME` matches")
s=s.replace("/// The §4.5 states","/// The §4.5 states")
s=s.replace("See deviation D1 for why there are 13.","See DEV-1 for why there are 13.")
s=s.replace("(D11), because","(DEV-11), because")
s=s.replace("§4.4's table says an ERR","§4.4's table says an ERR")
s=s.replace("The engine thread's entry point (D2).","The engine thread's entry point (DEV-2).")
s=s.replace("What the client gets when the handoff ended because the request was cancelled (D4).","What the client gets when the handoff ended because the request was cancelled (DEV-4).")
s=s.replace("The retry bound (§4.7, D13).","The retry bound (§4.7, DEV-13).")
s=s.replace("§4.4's \"also drop the connection\"","§4.4's \"also drop the connection\"")
s=s.replace("(never produced by an error code - see D5)","(never produced by an error code - see DEV-5)")
s=s.replace("(a permanent code only - D5)","(a permanent code only - DEV-5)")
s=s.replace("the caller answers the client (permanent codes only - D5)","the caller answers the client (permanent codes only - DEV-5)")
s=s.replace("§4.3's rule is \"unknown TRAILING","§4.3's rule is \"unknown TRAILING")
s=s.replace("So `parse_line` returns `unknown`","So `parse_line` returns `unknown`")
s=s.replace("// D5's rule, restated: a temporary","// DEV-5's rule, restated: a temporary")
s=s.replace("/// D5's rule, restated","/// DEV-5's rule, restated")
s=s.replace("/// `Hold::run_now` is not produced","/// **`Hold::run_now` is not produced")
open(p,"w").write("\n".join(lines))
print("renamed")
PY
grep -n "D[0-9]" include/strata/program/prefill_svc.hpp | grep -v "DEV-" | grep -v "serve_proto\|manager decision\|§" | head -20
