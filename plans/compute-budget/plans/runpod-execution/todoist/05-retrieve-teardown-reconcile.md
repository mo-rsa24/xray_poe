# 05 · Retrieve, tear down, reconcile spend

[⌂ Index](00-INDEX.md) · [← prev 04](04-ldm-train-on-pod.md)

## Reference while you do it
- 📄 Plan: plans/compute-budget/plans/runpod-execution/plans/05-retrieve-teardown-reconcile.md

## Section context (paste into the Todoist section)
**Description:** Pull the VAE and LDM checkpoints plus logs from the pod's volume back to local `ckpts/`, verify they arrived intact, tear the pod down so billing stops, then reconcile actual spend against the cost memo's budget and note the variance.
**Objective:** Turn the rental into durable local artifacts, stop the meter, and close the loop with the cost memo so the budget forecast is checked against reality.
**Goal:** VAE + LDM checkpoints and logs retrieved to local `ckpts/` (integrity-verified), the pod terminated with no ongoing charge, and a spend-reconciliation note comparing actual vs budgeted with variance recorded.
**Verify (whole leaf):** `ls ckpts/` → `vae_*.pt` and `ldm_*.pt` present locally; `python -c "import torch; torch.load('ckpts/vae_*.pt', map_location='cpu'); torch.load('ckpts/ldm_*.pt', map_location='cpu'); print('load ok')"` → `load ok`; `runpodctl get pod` → pod no longer listed (terminated); `cat runs/spend_reconciliation.md` → "Actual $A vs budget $B (variance ±X%)".
**▶ Recommended prompt:** — custom; no skill fits (infra/ops teardown + spend bookkeeping).

## Tasks (one at a time)
- [ ] Retrieve `ckpts/vae_*.pt`, `ckpts/ldm_*.pt`, and the run logs from the volume to local `ckpts/` (`runpodctl receive` / `rsync`)
- [ ] Verify the retrieved checkpoints load locally and match the on-pod checksums
- [ ] Tear down the pod (and release the volume if no longer needed); confirm the RunPod console shows no active resource
- [ ] Reconcile actual spend (hours × $/hr + storage + egress) vs the cost memo's budget; write `runs/spend_reconciliation.md` with the variance
