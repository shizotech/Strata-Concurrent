cd /ssd/Strata && python3 - <<'PYEOF'
import io
p='src/program/generate.cpp'
s=io.open(p,encoding='utf-8').read()
old='''                    if (free_permit) {
                        pass_moved = true;
                        const int64_t pid = P->id;      // `run_one` may drop the context (a rejected'''
new='''                    if (free_permit) {
                        pass_moved = true;
                        // S3.6: `prep_request` owns the REQUEST-LINE hand-over, and it cannot publish
                        // the outgoing slot's branch itself - at the point its body was written the
                        // driver's `live` deque does not exist yet, so the name `live` inside it is the
                        // serve-scope token vector.  Publish here, before the admit step, or the
                        // parking guard sees a mid-decode slot with nothing to save and refuses a
                        // hand-over that would otherwise have been perfectly parkable - turning a new
                        // client's first request into an ERR.
                        if (ReqCtx* OUT = ctx_of(mounted_id)) publish_decode_branch(*OUT);
                        const int64_t pid = P->id;      // `run_one` may drop the context (a rejected'''
assert s.count(old)==1
s=s.replace(old,new,1)
io.open(p,'w',encoding='utf-8').write(s)
print('ok')
PYEOF
cd /ssd/Strata/build && ninja 2>&1 | grep -E "error:" | head -3; echo BUILD_DONE
