"""S3.1e-2: prove the serial driver block is byte-identical to the S3.1e-1 text.

The expected text is the serial loop exactly as it read at the START of this task run
(generate.cpp:6281-6336, transcribed from the file before any S3.1e-2 edit).  The script
extracts everything from `while (next_line(line)) {` to the closing `return 0;` of the serve
block and compares it, byte for byte.
"""
import io, re, sys

EXPECTED = '''        while (next_line(line)) {
            if (line == "QUIT") break;
            // the watchdog watches a request from here until this iteration ends, whichever way it ends
            struct BusyScope {
                BusyScope() { strata::core::progress().busy.store(true); strata::core::progress_at("request"); }
                ~BusyScope() { strata::core::progress().busy.store(false); strata::core::progress_at("idle"); }
            } busy_scope;
            stop_req.store(false);   // a STOP that arrived between requests is stale
            ++request_index;         // S0.3 lever 4: the parking backoff's window is in requests
            err.clear();
            // S3.1e-1: everything the old body kept as per-request locals.  Constructed here, once per
            // iteration, so its lifetime is those locals' lifetime.
            ReqCtx R;
            // Whatever happens to this request from here - a validation ERR, a body ERR, a normal DONE - its
            // row is finished at the end of the iteration.  A guard rather than a call at every exit: the loop
            // has a dozen `continue`s, and a leaked row would strand an active-slot permit, which under
            // --serve-slots is the resource that bounds everything (§2.3).
            SlotGuard slot_guard{&R, tagged ? std::function<void(int64_t)>(slot_finish) : nullptr};
            // S3.1d R7, the guard.  The block below REWRITES the one host table and uploads it, for a
            // request that may still bail out (a bad embeddings file, a prompt too long for the
            // context, a token outside the vocabulary).  If it does, the slot that is actually mounted
            // is left with somebody else's positions in the device - and it is still mounted, so no
            // hand-over will put them back.  This guard restores the mounted slot's table on the way
            // out whenever the request did not become the mounted one.  On the normal path the swap
            // sets `mounted_id == req_id` and the guard does nothing.
            MropeScope mrope_scope;
            mrope_scope.skip = [&]() { return !swaps_on || !R.mrope_touched || mounted_id == req_id; };
            mrope_scope.restore = [&]() {
                if (mounted_id == strata::program::serve_proto::kNoId) return;
                apply_positions(conv_of(mounted_id));
                if (!mrope_host.empty()) upload_mrope_table();
                mrope_owner = mounted_id;
            };
            const Prep prep = prep_request(R, line);
            if (prep == Prep::rejected) continue;
            if (prep == Prep::fatal) return 1;

            // ---- S3.1e-1 phase 3: the prompt, one segment per step (0.1.30: one straight-line loop)
            plan_prompt_segments(R);
            Step st = Step::progressed;
            while ((st = run_prefill_step(R)) == Step::progressed) {}
            if (st == Step::error) return 1;
            if (st == Step::fatal_exit) {   // #224: a CUDA fault poisons the context for the whole process
                std::fflush(stdout);
                std::fflush(stderr);
                std::_Exit(1);
            }
            if (finish_prefill(R) == Step::error) return 1;
            // ---- S3.1e-1 phase 4: one verify window per step (0.1.30: one straight-line loop)
            Step dst = Step::progressed;
            while ((dst = run_decode_step(R)) == Step::progressed) {}
            if (dst == Step::error) return 1;
            // ---- S3.1e-1 phase 5: the request's tail (0.1.30: inline at the end of the body)
            if (finish_request(R) == Step::error) return 1;
        }
        return 0;
'''

path = "src/program/generate.cpp"
with io.open(path, encoding="utf-8") as f:
    src = f.read()

i = src.index("        while (next_line(line)) {")
j = src.index("        return 0;\n", i) + len("        return 0;\n")
got = src[i:j]

if got == EXPECTED:
    print("SERIAL DRIVER: byte-identical (%d bytes, %d lines)" % (len(got), got.count("\n")))
    sys.exit(0)

print("SERIAL DRIVER DIFFERS")
import difflib
for line in difflib.unified_diff(EXPECTED.splitlines(), got.splitlines(),
                                 "expected (S3.1e-1)", "now", lineterm="", n=2):
    print(line)
sys.exit(1)
