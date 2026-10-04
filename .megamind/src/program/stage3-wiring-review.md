# Stage 3 wiring review (2026-10-02)

Reviewed the S3.1a-e / S3.2b integration. One real bug, now fixed; the rest is sound.

## The bug: per-request protocol lines bypassing `sp_out`

`serve_proto::Out` is the single place that appends ` #<id>` to a per-request line
(`include/strata/program/serve_proto.hpp:148 tag_for()`). Sixteen `std::printf("ERR ...")`
sites inside the extracted step functions (`prep_request`, `run_prefill_step`,
`finish_prefill`, `run_decode_step`, `finish_request`) emitted **raw** lines.

Why that is a hang, not a cosmetic miss: `StrataEngine._pump` (`serve/server.py:260`) routes a
tagged session by parsing the tag; a line with no tag goes to `self.control`, and
`_stream`/`_register` only ever read the per-request queue. So the client's `ERR` is dropped and
the request sits until its own 10 s keep-alive loop or the engine watchdog fires.

Fixed by routing all sixteen through `sp_out.err(msg, R.id)`. Message text and the
`drive.d.failed && drive.d.fail ? drive.d.fail : err` conditional were preserved verbatim.
`sp_out.err()` is byte-identical to the old line when untagged, so `--serve-slots 0/1` is
unchanged.

## The guard

`.shz_cmd/s31e2_output_proof.py` now fails if any `std::printf("(ERR|T|DONE|RESUME|REUSED|PP|SLOT) `
appears in the serve block. `INFO`/`READY` are deliberately excluded: they are process-wide and
the server routes them to `control` on purpose. Anti-vacuity checked against raw `ERR`/`T`/
`DONE`/`RESUME` samples.

## Proof constants that moved (and why)

`EXPECTED_SERIAL_CALLS` 96 -> 100. The four additions are S3.2b's serial-reachable **stderr**
loan diagnostics (pump / drain-idle / lend / return). The stdout protocol count did not change
from that, nor from the ERR fix (14 raw -> 14 wrapped).

## Checked and found correct, no change made

- `settle_pump()` is the only place a lazily-returned row becomes resident, and only after
  `cudaEventQuery/Synchronize` confirms the copy landed; it re-checks `host_res[i] < 0` so a row
  the adaptive tier already promoted is not double-owned. `reconcile_residency()` runs at the top
  of every window path (`run_decode_step`, `read_windows`) *after* `pump_loans()`.
- `adapt()` defers to `led.owns()` so the pump and the adaptive tier never move the same expert.
- `watch_begin/watch_idle/watch_forget/watch_kill` are `static` (the watchdog thread is detached)
  and every access is under `watch_mu`. The escalation path (kill list not drained within one
  second -> abort) is the right reading of "the engine thread itself is wedged".
- `drop_ctx(id)` runs *after* `live.erase()` returns in the kill path, and `slot_finish` touches
  only the registry, so the `for (const ReqCtx& Q : live)` scans in `admit_one` cannot dangle.
- R7 image exclusivity is enforced at admission (waits, not refuses) and again by `MropeScope`.
- The loan guard in `run_one` is redundant with the deferral queue but deliberately kept: two
  slots lending the same cache rows is silent and produces plausible tokens.
- `serve/server.py` startup reads `INFO` before `READY` on the same fd, and `_pump` starts after
  that loop, so no line is lost at startup.
