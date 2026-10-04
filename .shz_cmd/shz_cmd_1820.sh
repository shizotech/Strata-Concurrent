cd /ssd/Strata && python3 - <<'PYEOF'
import io
p='src/program/generate.cpp'
s=io.open(p,encoding='utf-8').read()
old='''                const bool saveable =
                    strata::program::serve_swap::save_for(out, conversations.enabled()) ==
                        strata::program::serve_swap::Save::park;'''
new='''                // Read the SERVE-scope truth, not the slot record's mirror.  `park_current` asks
                // `live_ok && !live.empty()` on the serve-scope variables, and `SlotConv::live_ok` is
                // only refreshed by `sync_conversation()` at a few points, so the mirror can be stale
                // true for a slot whose request has since started reading.  A guard that trusted it
                // would let a hand-over through and then have `park_current` refuse it - which is the
                // exact disagreement this whole check exists to remove.
                const bool saveable = conversations.enabled() && live_ok && !live.empty();'''
assert s.count(old)==1
s=s.replace(old,new,1)
io.open(p,'w',encoding='utf-8').write(s)
print('ok')
PYEOF
cd /ssd/Strata/build && ninja 2>&1 | grep -E "error:" | head -5; echo BUILD_DONE
