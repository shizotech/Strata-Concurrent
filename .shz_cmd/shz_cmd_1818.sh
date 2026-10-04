cd /ssd/Strata && python3 - <<'PYEOF'
import io
p='src/program/generate.cpp'
s=io.open(p,encoding='utf-8').read()
old='''            // The prompt path's per-stage mid-prompt parts belong to the read that was just thrown
            // away; the next checkpoint of this request must not splice onto them.
            {
                std::lock_guard<std::mutex> lk(part_mu);
                part_at.clear();
                prompt_owner = strata::program::serve_proto::kNoId;
            }'''
new='''            // The prompt path's per-stage mid-prompt parts belong to the read that was just thrown
            // away; the next checkpoint of this request must not splice onto them.  `arm_prompt_state`
            // clears `part_at` when the prefilling request CHANGES, but a re-read is the same request,
            // so it has to be cleared here explicitly.
            {
                std::lock_guard<std::mutex> lk(part_mu);
                part_at.clear();
            }'''
assert s.count(old)==1
s=s.replace(old,new,1)
io.open(p,'w',encoding='utf-8').write(s)
print('ok')
PYEOF
cd /ssd/Strata/build && ninja 2>&1 | grep -E "error:" | head -5; echo BUILD_DONE
