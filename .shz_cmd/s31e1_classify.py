#!/usr/bin/env python3
"""S3.1e-1: classify every difference between the original request body and the
extracted step functions.  Any line that is not one of the allowed categories
is a behaviour change and must be investigated."""
import re, sys, difflib

def norm(ls):
    out = []
    for l in ls:
        s = l.strip()
        if not s or s.startswith("//"):
            continue
        out.append(re.sub(r"\s+", " ", s))
    return out

old = open(sys.argv[1]).read().split("\n")
new = open(sys.argv[2]).read().split("\n")
OB = norm(old[int(sys.argv[3]) - 1:int(sys.argv[4])])
NB = []
for a, b in zip(sys.argv[5::2], sys.argv[6::2]):
    NB += norm(new[int(a) - 1:int(b)])

ALLOW = [
    # exit-statement rewrites
    (re.compile(r"^return (Prep::(rejected|fatal)|Step::(error|finished|cancelled|fatal_exit|progressed));$"), "exit rewrite"),
    (re.compile(r"^if \(.*\) \{ .* return Step::(finished|cancelled); \}$"), "exit rewrite"),
    (re.compile(r"^if \(.*\) return Step::(finished|error|cancelled);$"), "exit rewrite"),
    (re.compile(r"^if \(poisoned\) return Prep::fatal;$"), "exit rewrite"),
    (re.compile(r"^return Prep::ok;$"), "exit rewrite"),
    # alias declarations into ReqCtx
    (re.compile(r"^(const )?[A-Za-z_:0-9<>,& ]+& [a-z_0-9]+ = R\.[a-z_0-9]+;"), "alias"),
    (re.compile(r"^const int64_t n = R\.n;$"), "alias"),
    # ReqCtx writes replacing a local declaration
    (re.compile(r"^R\.[a-z_0-9]+ = "), "ctx write"),
    # the serial driver glue
    (re.compile(r"^(plan_prompt_segments|prep_request|run_prefill_step|finish_prefill|run_decode_step|finish_request)\(R\)|^Step (st|dst) = |^while \(\(|^if \((st|dst|prep|finish_prefill)\b|^ReqCtx R;|^const Prep prep = |^if \(prep == |^\{$|^\}$"), "driver glue"),
    (re.compile(r"^SlotGuard slot_guard\{"), "guard"),
    (re.compile(r"^MropeScope mrope_scope;|^mrope_scope\.(skip|restore) = |^apply_positions\(conv_of\(mounted_id\)\);$|^if \(!mrope_host\.empty\(\)\) upload_mrope_table\;$|^mrope_owner = mounted_id;$"), "guard"),
    (re.compile(r"^std::_Exit\(1\);$|^std::fflush\((stdout|stderr)\);$"), "driver glue"),
    # the segment iterator that replaces the `for (const int64_t to : {...})`
    (re.compile(r"^R\.seg = \{|^R\.seg_i = 0;$|^int64_t to = -1;$|^bool have = false;$|^while \(R\.seg_i < R\.seg\.size\(\)\) \{$|^const int64_t t = R\.seg\[R\.seg_i\+\+\];$|^if \(t > at\) \{ to = t; have = true; break; \}$|^if \(!have\) return Step::finished;$"), "segment iterator"),
    (re.compile(r"^for \(const int64_t to : \{reread_to, root_at, turn_at, n - 1\}\) \{$|^if \(to <= at\) continue;$"), "segment iterator (old)"),
    # the decode-loop guard that replaces the `while (...)`
    (re.compile(r"^if \(cancelled \|\| produced_n >= max_new\) return Step::finished;$"), "loop guard (new)"),
    (re.compile(r"^while \(!cancelled && produced_n < max_new\) \{$"), "loop guard (old)"),
    # vectors moved to ReqCtx and sized once
    (re.compile(r"^R\.(drafts|window|outv|dprob|sbuf)\.assign\("), "ctx write"),
    (re.compile(r"^consumed\.clear\(\); consumed\.reserve\("), "ctx write"),
    (re.compile(r"^std::vector<int32_t> drafts\(.*\)|^std::vector<float> dprob\(|^std::vector<int32_t> sbuf\(|^std::vector<int32_t> consumed;$|^consumed\.reserve\("), "old decl"),
]

def classify(s):
    for rx, why in ALLOW:
        if rx.match(s):
            return why
    return None

sm = difflib.SequenceMatcher(None, OB, NB)
bad = 0
cats = {}
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        continue
    for s in OB[i1:i2]:
        c = classify(s)
        cats[c or "UNEXPLAINED-OLD"] = cats.get(c or "UNEXPLAINED-OLD", 0) + 1
        if c is None:
            print("UNEXPLAINED (old):", s); bad += 1
    for s in NB[j1:j2]:
        c = classify(s)
        cats[c or "UNEXPLAINED-NEW"] = cats.get(c or "UNEXPLAINED-NEW", 0) + 1
        if c is None:
            print("UNEXPLAINED (new):", s); bad += 1

print("\n--- categories:", cats)
print("--- before %d stmt-lines, after %d stmt-lines, UNEXPLAINED %d" % (len(OB), len(NB), bad))
sys.exit(1 if bad else 0)
