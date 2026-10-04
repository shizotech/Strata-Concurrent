#!/usr/bin/env python3
"""Generate the prefill_svc.hpp transition tables (total cross-products)."""

STATES = ["idle", "queued", "init", "sent", "held", "accepted", "prefilling", "seg_ready",
          "handed_over", "mounting", "done", "failed", "cancelled"]
EVENTS = ["admit", "req_sent", "queued_line", "claim_line", "seg_line", "done_line",
          "err_temporary", "err_permanent", "err_cancel", "cancel_line", "ack_sent",
          "copy_done", "mount_ok", "mount_fail", "collect", "retry", "note_cancel", "local_hit"]

T = {}

T["idle"] = ("idle", "illegal", "no handoff is in progress", {
    "admit": ("queued", "none", "admitted; the local cache has not been consulted yet"),
    "local_hit": ("handed_over", "mount", "the local ConversationCache::best() hit: NO handoff at all (S1.5)"),
    "note_cancel": ("idle", "done_cancel", "cancelled before anything was asked: the client gets DONE ... cancel"),
    "done_line": ("idle", "protocol_error", "DONE with no CLAIM ever sent: S4.5 rule 4, close the connection"),
    "seg_line": ("idle", "protocol_error", "SEG with no CLAIM ever sent: S4.5 rule 4"),
    "req_sent": ("idle", "illegal", "begin() first: there is no request to ask about"),
    "retry": ("idle", "illegal", "nothing has been asked, so there is nothing to retry"),
    "ack_sent": ("idle", "illegal", "ACK with no slot held"),
    "copy_done": ("idle", "illegal", "there is nothing to copy out"),
    "cancel_line": ("idle", "illegal", "CANCEL is a line the decode instance sends, not one it receives"),
})

T["queued"] = ("queued", "illegal", "the handoff thread must not touch a queued request", {
    "admit": ("init", "none", "the wait queue let it through and the local cache missed"),
    "local_hit": ("handed_over", "mount", "the local cache hit while it waited: no handoff"),
    "note_cancel": ("idle", "done_cancel", "cancelled while queued: nothing was touched"),
    "req_sent": ("queued", "illegal", "begin() first: a queued request has no REQ to write yet"),
    "retry": ("queued", "illegal", "nothing has been asked, so there is nothing to retry"),
    "collect": ("queued", "illegal", "nothing to collect"),
    "done_line": ("queued", "protocol_error", "DONE with no CLAIM: S4.5 rule 4"),
    "seg_line": ("queued", "protocol_error", "SEG with no CLAIM: S4.5 rule 4"),
    "cancel_line": ("queued", "illegal", "CANCEL is not a line this side receives"),
    "ack_sent": ("queued", "illegal", "no slot held"),
    "copy_done": ("queued", "illegal", "nothing to copy"),
})

T["init"] = ("init", "illegal", "the handoff has not been asked yet", {
    "req_sent": ("sent", "write_req", "one outstanding REQ per request (S6.2)"),
    "local_hit": ("handed_over", "mount", "the local cache hit after all"),
    "retry": ("init", "none", "already ready to ask again"),
    "note_cancel": ("idle", "done_cancel", "cancelled before REQ went out"),
    "claim_line": ("prefilling", "none", "CLAIM without QUEUED first is legal: the slot was free immediately"),
    "queued_line": ("accepted", "none", "accepted, no slot claimed yet"),
    "seg_line": ("idle", "protocol_error", "SEG before CLAIM: S4.5 rule 4"),
    "done_line": ("idle", "protocol_error", "DONE with no CLAIM: S4.5 rule 4"),
    "ack_sent": ("idle", "illegal", "no slot held"),
    "copy_done": ("idle", "illegal", "nothing to copy"),
    "cancel_line": ("idle", "illegal", "CANCEL is not a line this side receives"),
    "err_temporary": ("idle", "illegal", "ERR with no REQ in flight"),
    "err_permanent": ("idle", "illegal", "ERR with no REQ in flight"),
    "err_cancel": ("idle", "illegal", "ERR cancel with no REQ in flight"),
})

T["sent"] = ("sent", "illegal", "the REQ is in flight; the peer's answer drives the state", {
    "queued_line": ("accepted", "none", "accepted, no slot claimed yet"),
    "claim_line": ("prefilling", "none", "the slot was free: CLAIM straight away"),
    "seg_line": ("idle", "protocol_error", "SEG before CLAIM: S4.5 rule 4"),
    "done_line": ("idle", "protocol_error", "DONE with no CLAIM: S4.5 rule 4, close the connection"),
    "err_temporary": ("held", "hold", "S4.4: a temporary ERR is a hold, never a client ERR (D5)"),
    "err_permanent": ("failed", "none", "S4.4: a permanent ERR ends the handoff; the caller collects the verdict"),
    "err_cancel": ("cancelled", "none", "the prefill instance answered ERR cancel: the job is gone"),
    "req_sent": ("sent", "illegal", "one outstanding REQ per request (S6.2)"),
    "note_cancel": ("cancelled", "write_cancel", "S4.6: cancelled while the REQ is in flight"),
    "retry": ("sent", "illegal", "a REQ is already in flight; wait for the answer"),
    "admit": ("sent", "illegal", "already admitted"),
    "local_hit": ("sent", "illegal", "the cache lookup happens before REQ; not now"),
    "cancel_line": ("sent", "illegal", "CANCEL is not a line this side receives"),
    "ack_sent": ("idle", "illegal", "no slot held"),
    "copy_done": ("idle", "illegal", "nothing to copy"),
    "collect": ("idle", "illegal", "nothing to collect"),
})

T["held"] = ("held", "illegal", "the hold is owned by S4.2's wait queue, not by the handoff thread", {
    "req_sent": ("sent", "write_req", "the backoff expired: re-ask, still one outstanding REQ"),
    "retry": ("init", "none", "the retry bound allowed another handoff"),
    "admit": ("held", "hold", "still held: the wait queue owns it"),
    "local_hit": ("handed_over", "mount", "the local cache hit while it held"),
    "queued_line": ("accepted", "none", "a late QUEUED for the REQ we gave up on"),
    "claim_line": ("prefilling", "none", "a late CLAIM for the REQ we gave up on"),
    "seg_line": ("idle", "protocol_error", "SEG before CLAIM: S4.5 rule 4"),
    "done_line": ("idle", "protocol_error", "DONE with no CLAIM in this attempt"),
    "err_temporary": ("held", "hold", "a second temporary answer: hold again and the backoff doubles"),
    "err_permanent": ("failed", "none", "a permanent ERR while held: the request is over"),
    "err_cancel": ("cancelled", "none", "the job was cancelled at the prefill instance"),
    "note_cancel": ("idle", "done_cancel", "S4.6: cancelled while held; no slot was ever claimed"),
    "cancel_line": ("held", "illegal", "CANCEL is not a line this side receives"),
    "ack_sent": ("idle", "illegal", "no slot held"),
    "copy_done": ("idle", "illegal", "nothing to copy"),
    "collect": ("idle", "illegal", "nothing to collect"),
})

T["accepted"] = ("accepted", "illegal", "the handoff is in flight", {
    "claim_line": ("prefilling", "none", "the slot was claimed; the read is about to start"),
    "seg_line": ("idle", "protocol_error", "SEG before CLAIM: S4.5 rule 4"),
    "done_line": ("idle", "protocol_error", "DONE with no CLAIM: S4.5 rule 4, close the connection"),
    "queued_line": ("accepted", "none", "a second QUEUED is a progress update, not a new state"),
    "err_temporary": ("held", "hold", "S4.4: temporary -> hold, never a client ERR"),
    "err_permanent": ("failed", "none", "S4.4: permanent -> the caller collects the verdict"),
    "err_cancel": ("cancelled", "none", "the job was dropped before it ran (S4.6)"),
    "note_cancel": ("cancelled", "write_cancel", "S4.6: cancelled while QUEUED - the job is dropped, no slot touched"),
    "req_sent": ("accepted", "illegal", "one outstanding REQ per request"),
    "admit": ("accepted", "illegal", "already admitted"),
    "local_hit": ("accepted", "illegal", "the handoff is already in flight"),
    "retry": ("accepted", "illegal", "the handoff is in flight; a retry would double it"),
    "cancel_line": ("accepted", "illegal", "CANCEL is not a line this side receives"),
    "ack_sent": ("idle", "illegal", "no slot held"),
    "copy_done": ("idle", "illegal", "nothing to copy"),
    "collect": ("idle", "illegal", "nothing to collect"),
})

T["prefilling"] = ("prefilling", "illegal", "the read is running at the prefill instance", {
    "seg_line": ("prefilling", "none", "progress; a reader that ignores SEG is still correct (S4.3)"),
    "done_line": ("seg_ready", "copy_out", "the payload is published; the copy-out starts HERE, never on the engine thread"),
    "err_temporary": ("held", "hold", "S4.4: read/slotlost/shutting are waits, not client errors"),
    "err_permanent": ("failed", "none", "S4.4: a permanent ERR ends the handoff"),
    "err_cancel": ("cancelled", "none", "S4.6: the read stopped at a chunk boundary and the slot was released without publishing"),
    "claim_line": ("prefilling", "illegal", "a second CLAIM for one request means the first slot was lost"),
    "queued_line": ("prefilling", "illegal", "QUEUED after CLAIM is out of order"),
    "note_cancel": ("cancelled", "write_cancel", "S4.6: cancelled while reading; the slot is released WITHOUT publishing"),
    "req_sent": ("prefilling", "illegal", "one outstanding REQ per request"),
    "admit": ("prefilling", "illegal", "already admitted"),
    "local_hit": ("prefilling", "illegal", "the handoff is already in flight"),
    "retry": ("prefilling", "illegal", "the handoff is in flight; a retry would double it"),
    "cancel_line": ("prefilling", "illegal", "CANCEL is not a line this side receives"),
    "ack_sent": ("prefilling", "illegal", "the slot is not READY; ACK would release an unwritten slot"),
    "copy_done": ("prefilling", "illegal", "nothing is published to copy"),
    "collect": ("prefilling", "illegal", "nothing to collect"),
})

T["seg_ready"] = ("seg_ready", "illegal", "the slot is held until ACK", {
    "copy_done": ("handed_over", "write_ack", "S4.5 rule 1: ACK as soon as the copy-out and checks 1-2 pass, BEFORE the mount"),
    "ack_sent": ("handed_over", "mount", "the caller wrote ACK itself; the image is ours"),
    "note_cancel": ("cancelled", "write_ack", "S4.6: cancelled after DONE, before ACK - send ACK, never mount"),
    "done_line": ("seg_ready", "illegal", "a second DONE for one handoff means the slot was reused under us"),
    "claim_line": ("seg_ready", "illegal", "a second CLAIM while one slot is held"),
    "seg_line": ("seg_ready", "none", "a late SEG is progress about the same payload"),
    "queued_line": ("seg_ready", "illegal", "QUEUED after DONE is out of order"),
    "err_temporary": ("seg_ready", "illegal", "an ERR after DONE contradicts the published payload"),
    "err_permanent": ("seg_ready", "illegal", "an ERR after DONE contradicts the published payload"),
    "err_cancel": ("cancelled", "write_ack", "S4.5 rule 3: release the slot we hold, never read it"),
    "req_sent": ("seg_ready", "illegal", "one outstanding REQ per request"),
    "admit": ("seg_ready", "illegal", "already admitted"),
    "local_hit": ("seg_ready", "illegal", "the payload is already in hand"),
    "retry": ("seg_ready", "illegal", "release the slot first (ACK), then retry"),
    "cancel_line": ("seg_ready", "illegal", "CANCEL is not a line this side receives"),
    "mount_ok": ("seg_ready", "illegal", "not mounted yet"),
    "mount_fail": ("seg_ready", "illegal", "not mounted yet"),
    "collect": ("seg_ready", "illegal", "nothing to collect"),
})

T["handed_over"] = ("handed_over", "illegal", "the slot is FREE and the image is ours", {
    "mount_ok": ("mounting", "mount", "post the image to the engine thread as mount_image"),
    "admit": ("handed_over", "mount", "the engine thread may take it now"),
    "mount_fail": ("failed", "none", "the image could not be mounted before any write: ERR badpayload or a local re-read"),
    "note_cancel": ("cancelled", "none", "S4.6: cancelled after ACK - an ordinary STOP on the client wire, stage 3's business"),
    "local_hit": ("handed_over", "illegal", "the image is already in hand"),
    "ack_sent": ("handed_over", "illegal", "ACK was already sent; a second one would release somebody else's slot"),
    "copy_done": ("handed_over", "illegal", "already copied out"),
    "done_line": ("handed_over", "illegal", "a second DONE for one handoff"),
    "claim_line": ("handed_over", "illegal", "a second CLAIM while the image is in hand"),
    "seg_line": ("handed_over", "none", "a late SEG is harmless"),
    "queued_line": ("handed_over", "illegal", "QUEUED after DONE is out of order"),
    "err_temporary": ("handed_over", "illegal", "an ERR after a successful handoff"),
    "err_permanent": ("handed_over", "illegal", "an ERR after a successful handoff"),
    "err_cancel": ("handed_over", "illegal", "an ERR cancel after a successful handoff"),
    "req_sent": ("handed_over", "illegal", "the handoff finished"),
    "retry": ("handed_over", "illegal", "the handoff finished; a mount failure is retried from H_FAILED"),
    "cancel_line": ("handed_over", "illegal", "CANCEL is not a line this side receives"),
    "collect": ("handed_over", "illegal", "nothing to collect"),
})

T["mounting"] = ("mounting", "illegal", "the engine thread is mounting", {
    "mount_ok": ("done", "finish", "validate, unmount, mount, draft_kv, adopt, device_state all passed"),
    "mount_fail": ("failed", "none", "validate failed: ERR badpayload, or a local re-read with --prefill-fallback local"),
    "note_cancel": ("mounting", "illegal", "a mount is not cancellable: it has already written session state"),
    "seg_line": ("mounting", "none", "a late SEG is harmless"),
    "admit": ("mounting", "illegal", "the engine thread is already mounting"),
    "ack_sent": ("mounting", "illegal", "ACK was already sent; the slot is FREE"),
    "copy_done": ("mounting", "illegal", "already copied"),
    "collect": ("mounting", "illegal", "nothing to collect"),
    "retry": ("mounting", "illegal", "finish the mount before retrying"),
    "cancel_line": ("mounting", "illegal", "CANCEL is not a line this side receives"),
    "done_line": ("mounting", "illegal", "a second DONE for one handoff"),
    "claim_line": ("mounting", "illegal", "a second CLAIM while mounting"),
    "queued_line": ("mounting", "illegal", "QUEUED after DONE is out of order"),
    "err_temporary": ("mounting", "illegal", "an ERR after a successful handoff"),
    "err_permanent": ("mounting", "illegal", "an ERR after a successful handoff"),
    "err_cancel": ("mounting", "illegal", "an ERR cancel after a successful handoff"),
    "req_sent": ("mounting", "illegal", "the handoff finished"),
    "local_hit": ("mounting", "illegal", "already mounting"),
})

T["done"] = ("done", "illegal", "the handoff is over", {
    "collect": ("idle", "finish", "the caller collected the result"),
    "note_cancel": ("done", "client_gone", "the request already has its payload: an ordinary STOP, stage 3's business"),
    "retry": ("done", "illegal", "a finished handoff is not retried; a new request begins a new machine"),
    "admit": ("done", "illegal", "the handoff is over"),
    "local_hit": ("done", "illegal", "already handed over"),
    "ack_sent": ("done", "illegal", "the slot is already FREE"),
})

T["failed"] = ("failed", "illegal", "the handoff already ended with a verdict", {
    "collect": ("idle", "err_client", "the caller answers the client (permanent codes only - D5) and releases the row"),
    "retry": ("init", "none", "S4.5: a retry of a FAILED handoff re-enters H_REQ_SENT, bounded by --prefill-retries"),
    "note_cancel": ("failed", "done_cancel", "cancelled after a failure: the client gets DONE ... cancel, not ERR"),
    "admit": ("failed", "illegal", "already admitted"),
    "req_sent": ("failed", "illegal", "collect or retry first"),
    "ack_sent": ("failed", "illegal", "no slot held"),
    "copy_done": ("failed", "illegal", "nothing to copy"),
})

T["cancelled"] = ("cancelled", "illegal", "the request is cancelled", {
    "ack_sent": ("idle", "done_cancel", "the slot is released; the client gets DONE ... cancel"),
    "collect": ("idle", "done_cancel", "no slot was ever claimed: nothing to release"),
    "done_line": ("cancelled", "write_ack", "S4.5 rule 3: a DONE for a cancelled request is answered ACK, never read"),
    "claim_line": ("cancelled", "write_cancel", "a CLAIM that races our CANCEL: CANCEL again, the writer releases without publishing"),
    "queued_line": ("cancelled", "write_cancel", "a QUEUED that races our CANCEL"),
    "seg_line": ("cancelled", "none", "progress about a job we killed"),
    "err_temporary": ("cancelled", "none", "an ERR after we cancelled: nothing to do"),
    "err_permanent": ("cancelled", "none", "an ERR after we cancelled: nothing to do"),
    "err_cancel": ("idle", "done_cancel", "the prefill instance answered ERR cancel: the job is gone"),
    "note_cancel": ("cancelled", "none", "already cancelled; a second CANCEL is idempotent"),
    "admit": ("cancelled", "illegal", "it is cancelled"),
    "req_sent": ("cancelled", "illegal", "it is cancelled"),
    "retry": ("cancelled", "illegal", "a cancelled request is not re-asked"),
    "local_hit": ("cancelled", "illegal", "it is cancelled"),
    "copy_done": ("cancelled", "illegal", "a cancelled handoff never copies out"),
    "mount_ok": ("cancelled", "illegal", "a cancelled handoff never mounts"),
    "mount_fail": ("cancelled", "illegal", "a cancelled handoff never mounts"),
})

out = []
out.append("inline const Transition* handoff_table(int& count) {")
out.append("    // Exactly kStateCount * kEventCount rows: one per (state, event).  The test asserts the")
out.append("    // product, so \"total\" is a checked fact and not a claim in a comment.")
out.append("    static const Transition t[] = {")
n = 0
for s in STATES:
    dto, dact, dwhy, ov = T[s]
    out.append("        // ---- %s ----" % s.upper())
    for e in EVENTS:
        to, act, why = ov.get(e, (dto, dact, dwhy))
        # An illegal (refused) event must never move the state.  Enforced here so a typo in an
        # override cannot quietly make a refusal look like a transition.
        if act == "illegal":
            to = s
        out.append('        {HState::%s, Event::%s, HState::%s, Act::%s,' % (s, e, to, act))
        out.append('            "%s"},' % why)
        n += 1
out.append("    };")
out.append("    count = (int) (sizeof t / sizeof t[0]);")
out.append("    return t;")
out.append("}")
print("// rows:", n)
open(".shz_cmd/handoff_table.inc", "w").write("\n".join(out) + "\n")

# ------------------------------------------------------------------ the prefill job table -----
PJ = ["queued", "claimed", "reading", "writing", "published", "released", "cancelling", "cancelled", "failed"]
PE = ["claim_ok", "read_start", "seg_sent", "publish_ok", "ack", "cancel", "read_fail", "publish_lost", "abort_done"]

J = {}
J["queued"] = ("queued", "illegal", "no slot has been claimed", {
    "claim_ok": ("claimed", "send_claim", "a slot was claimed for it"),
    "cancel": ("cancelled", "err_cancel", "S4.6: cancelled while QUEUED - drop the job, no slot touched"),
})
J["claimed"] = ("claimed", "illegal", "the read has not started", {
    "read_start": ("reading", "none", "Prefill::run begins"),
    "cancel": ("cancelling", "abort_err_cancel", "S4.6: release the slot WITHOUT publishing"),
    "read_fail": ("failed", "abort_err_read", "the read failed before any byte was written"),
})
J["reading"] = ("reading", "illegal", "the read is running", {
    "seg_sent": ("writing", "send_seg", "the read finished and the payload write started"),
    "publish_ok": ("published", "send_done", "a zero-GPU reuse handoff may skip SEG (S1.3.3)"),
    "cancel": ("cancelling", "abort_err_cancel", "S4.6: the stop flag lands at the next chunk boundary"),
    "read_fail": ("failed", "abort_err_read", "a CUDA error or a lend failure: release without publishing"),
})
J["writing"] = ("writing", "illegal", "the payload write is running", {
    "publish_ok": ("published", "send_done", "the release fence and the ordered store landed"),
    "publish_lost": ("failed", "err_slotlost", "the slot was reclaimed under us (S4.7): ERR slotlost, retry once"),
    "cancel": ("cancelling", "abort_err_cancel", "cancelled mid-write: the payload is never published"),
    "read_fail": ("failed", "abort_err_read", "the payload write failed"),
})
J["published"] = ("published", "illegal", "the payload is published; only ACK or a cancel moves it", {
    "ack": ("released", "release_slot", "the ONLY release verb"),
    "cancel": ("released", "release_slot", "S4.5 rule 3: CANCEL after DONE is a normal release, not a protocol error"),
})
J["released"] = ("released", "illegal", "the job is over", {
    "cancel": ("released", "none", "already released"),
})
J["cancelling"] = ("cancelling", "illegal", "the unwind is running", {
    "abort_done": ("cancelled", "err_cancel", "the unwind finished: the slot is FREE"),
    "cancel": ("cancelling", "none", "a second CANCEL is idempotent"),
    "publish_ok": ("cancelling", "abort_claim", "a publish that races a cancel must not land"),
    "read_fail": ("cancelling", "abort_claim", "the read failed while it was being cancelled"),
    "publish_lost": ("cancelling", "abort_claim", "the slot was already taken back"),
})
J["cancelled"] = ("cancelled", "illegal", "the job is over", {
    "cancel": ("cancelled", "none", "idempotent"),
    "abort_done": ("cancelled", "none", "already unwound"),
})
J["failed"] = ("failed", "illegal", "the failure was already reported", {})

out = []
out.append("inline const JobTransition* prefill_job_table(int& count) {")
out.append("    // Exactly kJobStateCount * kJobEventCount rows: one per (job state, event).")
out.append("    static const JobTransition t[] = {")
m = 0
for s in PJ:
    dto, dact, dwhy, ov = J[s]
    out.append("        // ---- %s ----" % s.upper())
    for e in PE:
        to, act, why = ov.get(e, (dto, dact, dwhy))
        if act == "illegal":
            to = s
        out.append('        {PJob::%s, PEvent::%s, PJob::%s, PAct::%s,' % (s, e, to, act))
        out.append('            "%s"},' % why)
        m += 1
out.append("    };")
out.append("    count = (int) (sizeof t / sizeof t[0]);")
out.append("    return t;")
out.append("}")
print("job rows:", m)
open(".shz_cmd/job_table.inc", "w").write("\n".join(out) + "\n")
