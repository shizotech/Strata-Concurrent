cd /ssd/Strata && python3 - <<'PY'
p=".shz_cmd/s433_gen_tables.py"
s=open(p).read()
s=s.replace('PE = ["claim_ok", "read_start", "seg_sent", "publish_ok", "ack", "cancel", "read_fail", "publish_lost", "abort_claim"]',
            'PE = ["claim_ok", "read_start", "seg_sent", "publish_ok", "ack", "cancel", "read_fail", "publish_lost", "abort_done"]')
s=s.replace('"abort_claim": ("cancelled", "err_cancel", "the unwind finished: the slot is FREE")',
            '"abort_done": ("cancelled", "err_cancel", "the unwind finished: the slot is FREE")')
s=s.replace('"abort_claim": ("cancelled", "none", "already unwound")',
            '"abort_done": ("cancelled", "none", "already unwound")')
open(p,"w").write(s)
PY
python3 .shz_cmd/s433_gen_tables.py && grep -c "PAct::abort_done" .shz_cmd/job_table.inc; grep -c "PEvent::abort_done" .shz_cmd/job_table.inc
