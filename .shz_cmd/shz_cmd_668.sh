cd /ssd/Strata && python3 - <<'PY'
p="src/program/prefill_svc_test.cpp"
s=open(p).read()
def rep(a,b):
    global s
    assert a in s, "MISSING: "+a[:80]
    s=s.replace(a,b,1)
rep('''    /// Run the head job as far as its behaviour allows.  Returns the lines the decode side must see.
    void step_head() {
        if (jobs.empty()) return;
        Job& j = jobs.front();
        if (j.state != PJob::queued) return;''',
'''    /// Run the head QUEUED job as far as its behaviour allows (§5.2: FIFO by REQ arrival, one job at
    /// a time; a job that has already published is not the head for claiming purposes).
    void step_head() {
        Job* head = nullptr;
        for (auto& j : jobs) {
            if (j.state == PJob::queued) { head = &j; break; }
        }
        if (!head) return;
        Job& j = *head;''')
rep('''        if (behaviour == Behaviour::full) {
            emit(err_line(j.id, ErrCode::full, "no free slot"));
            ++jobs_err;
            jobs.erase(jobs.begin());
            return;
        }''',
'''        if (behaviour == Behaviour::full) {
            emit(err_line(j.id, ErrCode::full, "no free slot"));
            ++jobs_err;
            jobs.erase(std::find_if(jobs.begin(), jobs.end(),
                                    [&](const Job& x) { return x.id == j.id; }));
            return;
        }''')
rep('''        if (!arena.claim((uint32_t) j.tier, client_id, (uint64_t) j.id, now_ms, idx, nonce, err)) {
            emit(err_line(j.id, ErrCode::full, err));
            ++jobs_err;
            jobs.erase(jobs.begin());
            return;
        }''',
'''        if (!arena.claim((uint32_t) j.tier, client_id, (uint64_t) j.id, now_ms, idx, nonce, err)) {
            emit(err_line(j.id, ErrCode::full, err));
            ++jobs_err;
            jobs.erase(std::find_if(jobs.begin(), jobs.end(),
                                    [&](const Job& x) { return x.id == j.id; }));
            return;
        }''')
rep('''        if (behaviour == Behaviour::read_fail) {
            move(j, PEvent::read_fail);
            jobs.erase(jobs.begin());
            return;
        }''',
'''        if (behaviour == Behaviour::read_fail) {
            move(j, PEvent::read_fail);
            jobs.erase(std::find_if(jobs.begin(), jobs.end(),
                                    [&](const Job& x) { return x.id == j.id; }));
            return;
        }''')
rep('''            if (!arena.publish(j.slot, j.nonce, j.tokens, e2)) move(j, PEvent::publish_lost);
            jobs.erase(jobs.begin());
            return;''',
'''            if (!arena.publish(j.slot, j.nonce, j.tokens, e2)) move(j, PEvent::publish_lost);
            jobs.erase(std::find_if(jobs.begin(), jobs.end(),
                                    [&](const Job& x) { return x.id == j.id; }));
            return;''')
rep('''        if (!arena.publish(j.slot, j.nonce, j.tokens, err)) { move(j, PEvent::publish_lost); return; }''',
'''        if (!arena.publish(j.slot, j.nonce, j.tokens, err)) { move(j, PEvent::publish_lost); return; }
        // The job stays in the list until its ACK: that is the window the leak watchdog watches.''')
open(p,"w").write(s)
print("ok")
PY
g++ -std=c++20 -O0 -Iinclude -o /tmp/prefill_svc_test src/program/prefill_svc_test.cpp 2>&1 | head -20 && /tmp/prefill_svc_test 2>&1 | tail -30
