#!/usr/bin/env python3
"""Replay a serve log's parking events through the S0.3 refusal-backoff state machine.

Lever 4 (`bench/prefill/fixed-cost-changes.md`) skips the snapshot estimate + RAM
telemetry after N consecutive physical-RAM-admission refusals, for a window of
REQUESTS.  Whether that helps is answerable OFFLINE: the log already records every park
and every refusal in order, so the state machine can be replayed over it without
starting anything.

This is the same state machine as `strata::program::prefill_loan::ParkBackoff`
(include/strata/program/prefill_loan.hpp), transcribed line for line, including:

  * the window is counted in REQUESTS, not in parking attempts.  A request only reaches
    `park_current` when it does not continue from the live session, so most requests
    never paid an estimate and are not in the sample - but they do advance the clock.
    The request index at a parking event is `requests closed so far + 1`, because
    `park_current` runs at the START of a request and the `prompt ... = X reused` line
    that closes it is printed at its END;
  * a request that starts a new conversation (`reused == 0`) or mounts a parked one
    (`conversation cache: restored`) always asks - `force = true` - and does NOT clear
    the refusal streak;
  * a park that stores clears the streak and the window.

    python3 bench/prefill/park_backoff_replay.py strata-iq3_s.log
    python3 bench/prefill/park_backoff_replay.py --refusals 5 --quiet 128 strata-iq3_s.log

Read-only.  Starts nothing.
"""
import argparse
import re
import sys

N = r"([0-9]+(?:\.[0-9]+)?)"
RE_REFUSE = re.compile(r"^strata serve: conversation cache: skip parking \("
                       r"(physical RAM admission|physical RAM floor after capture)")
RE_PARK = re.compile(r"^strata serve: conversation cache: (parked|skipped) " + N + r" tokens in " +
                     N + r" ms; parked=" + N)
RE_RESTORE = re.compile(r"^strata serve: conversation cache: restored ")
RE_PROMPT = re.compile(r"^strata serve: prompt " + N + r" tokens = " + N + r" reused")


def events(path):
    """Parking attempts in log order, as (kind, forced, request_index).

    kind: 'P' a park that stored, 'R' a physical-RAM-admission refusal, 'S' a park the
    budget check refused before the RAM check (the estimate was paid; it is neither a
    streak-breaker nor a refusal - exactly how the engine treats it).
    """
    out = []
    requests = 0            # requests already closed by a `prompt ...` line
    pending = []            # parking events awaiting the request line that closes them
    saw_restore = False
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = RE_REFUSE.match(line)
            if m:
                pending.append(["R", False, requests + 1])
                continue
            m = RE_PARK.match(line)
            if m:
                pending.append(["P" if m.group(1) == "parked" else "S", False, requests + 1])
                continue
            if RE_RESTORE.match(line):
                saw_restore = True
                continue
            m = RE_PROMPT.match(line)
            if m:
                requests += 1
                forced = saw_restore or int(m.group(2)) == 0
                for e in pending:
                    e[1] = e[1] or forced
                    out.append((e[0], e[1], e[2]))
                pending = []
                saw_restore = False
    for e in pending:
        out.append((e[0], e[1], e[2]))
    return out, requests


def replay(evs, refusals, quiet):
    """Returns (estimates still paid, estimates skipped)."""
    paid = consec = quiet_until = 0
    for kind, forced, req in evs:
        if refusals > 0 and quiet_until > req and not forced:
            continue                       # quiet: no estimate, no telemetry
        paid += 1
        if kind == "R":
            consec += 1
            if consec >= refusals:
                quiet_until = max(quiet_until, req + (quiet if quiet > 0 else 1))
        elif kind == "P":
            consec = 0                     # a park worked: ask every time again
            quiet_until = 0
    return paid, len(evs) - paid


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("log", nargs="+")
    ap.add_argument("--refusals", type=int, default=3, help="STRATA_PARK_REFUSALS (0 = never back off)")
    ap.add_argument("--quiet", type=int, default=64, help="STRATA_PARK_QUIET, in requests")
    a = ap.parse_args()

    evs, requests = [], 0
    for p in a.log:
        e, r = events(p)
        evs += e
        requests += r
    if not evs:
        print("no parking lines in this log (parking was off in these restarts)", file=sys.stderr)
        return 1

    refused = sum(1 for k, _, _ in evs if k == "R")
    parked = sum(1 for k, _, _ in evs if k == "P")
    forced = sum(1 for _, f, _ in evs if f)
    print(f"{requests} requests, {len(evs)} parking attempts: {refused} refused, {parked} parked, "
          f"{len(evs) - refused - parked} budget-refused; {forced} forced (new conversation / mount)")
    print(f"{'refusals':>9} {'quiet':>6}  {'paid':>5}  {'skipped':>8}")
    for r, q in ((a.refusals, a.quiet), (0, 0)):
        paid, skipped = replay(evs, r, q)
        tag = "engine default" if (r, q) == (a.refusals, a.quiet) else "off (today)"
        print(f"{r:>9} {q:>6}  {paid:>5}  {skipped:>5} ({100.0 * skipped / len(evs):.0f}%)  {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
