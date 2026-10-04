cd /ssd/Strata && python3 - <<'PYEOF'
import io
p='src/program/generate.cpp'
s=io.open(p,encoding='utf-8').read()
old='''                } else {
                    // S3.6: this is now only reachable for a slot with NO work left (a finished
                    // conversation whose image the cache would not take).  A slot that still owes a
                    // step was refused above, before anything was destroyed - which is the whole fix.
                    // The promise this line used to make ("will be re-read from token 0") is now kept
                    // by `serve_driver::step_gate`: any slot whose record stopped describing the
                    // session is ended with a named ERR instead of running a window against a
                    // foreign one.
                    strata::program::serve_swap::invalidate_unparked(out);
                    swap_invalidated_out = true;
                    std::fprintf(stderr, "strata serve: swap: slot %lld NOT parked (%zu tokens, %lld MiB snapshot "
                                         "vs a %lld MiB budget) - its conversation is dropped.  It has no work "
                                         "left, so nothing will step against it.\\n",
                                 (long long) out.id, out.live.size(),
                                 (long long) (parkchk.snapshot >> 20),
                                 (long long) (parkchk.budget >> 20));
                }'''
new='''                } else {
                    // S3.6: the outgoing slot's branch is dropped without a save, and that is allowed
                    // for exactly TWO reasons, both decided by `out_state` before anything was
                    // destroyed:
                    //   * `finished`    - the request is over, so no step can ever run against a
                    //     stale session.  Dropping the in-session copy costs reuse, not a
                    //     conversation: a later request of the same chat finds the branch through
                    //     `ConversationCache::best()`, exactly as stage 2 always did.
                    //   * `re_readable` - the request is mid-prompt and its whole prompt is still in
                    //     `ReqCtx::ids`.  `serve_driver::step_gate` sees `resumable == false` and
                    //     sends it back to token 0 (`reset_request_to_token0`), which is the promise
                    //     this log line has always made and never kept.
                    // A slot in `decode` reaching this line is the bug; the guard above refuses it.
                    strata::program::serve_swap::invalidate_unparked(out);
                    swap_invalidated_out = true;
                    const bool over = out_state == strata::program::serve_swap::Outgoing::finished;
                    std::fprintf(stderr, "strata serve: swap: slot %lld NOT parked (%zu tokens, %lld MiB snapshot "
                                         "vs a %lld MiB budget) - %s\\n",
                                 (long long) out.id, out.live.size(),
                                 (long long) (parkchk.snapshot >> 20),
                                 (long long) (parkchk.budget >> 20),
                                 over ? "its request is over, so nothing will step against it and a later "
                                        "request of this chat re-reads from the cache"
                                      : "it is mid-prompt, so it will be RE-READ from token 0");
                    std::fflush(stderr);
                }'''
assert s.count(old)==1
s=s.replace(old,new,1)
io.open(p,'w',encoding='utf-8').write(s)
print('ok')
PYEOF
cd /ssd/Strata/build && ninja 2>&1 | grep -E "error:" | head -3; echo BUILD_DONE
