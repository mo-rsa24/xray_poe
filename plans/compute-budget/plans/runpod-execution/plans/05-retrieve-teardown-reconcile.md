# 📥 Retrieve, Tear Down, Reconcile Spend

## Background
The final on-pod step, after both trains (`03`, `04`) have produced checkpoints on
the volume. Until the pod is torn down it keeps billing, so this plan is what
actually closes the rental.

## Description
Pull the VAE and LDM checkpoints plus their logs from the pod's volume back to
local `ckpts/`, verify they arrived intact, tear the pod down so billing stops,
then reconcile the actual spend (pod $/hr × hours + storage + egress) against the
cost memo's budget and note the variance.

## Purpose
This turns the rental into durable local artifacts the rest of Paper3 depends on,
stops the meter, and closes the loop with the cost memo so the budget forecast is
checked against reality. Serves Objective 5 and Definition-of-Done #5.

## Goal
VAE + LDM checkpoints and logs retrieved to local `ckpts/` (integrity-verified),
the pod terminated with no ongoing charge, and a spend-reconciliation note
comparing actual vs budgeted, with variance recorded.

## Tasks
- [ ] ⚠️ Retrieve `ckpts/vae_*.pt`, `ckpts/ldm_*.pt`, and the run logs from the volume to local `ckpts/` (`runpodctl receive` / `rsync`)
- [ ] ⚠️ Verify the retrieved checkpoints load locally and match the on-pod checksums
- [ ] ⚠️ Tear down the pod (and release the volume if no longer needed); confirm the RunPod console shows no active resource
- [ ] ⚠️ Reconcile actual spend (hours × $/hr + storage + egress) vs the cost memo's budget; write `runs/spend_reconciliation.md` with the variance

## Recommended skill
— custom; no skill fits (infra/ops teardown + spend bookkeeping).

## Engagement Instructions
```
# DO THIS — pull artifacts, tear down, reconcile
$ runpodctl receive <transfer-code>            # or: rsync the volume's ckpts/ + runs/ to local ckpts/
$ python -c "import torch; torch.load('ckpts/vae_*.pt', map_location='cpu'); torch.load('ckpts/ldm_*.pt', map_location='cpu'); print('load ok')"
# (then tear the pod down from the RunPod console / `runpodctl remove pod <id>`)

# GET THAT — artifacts local, no pod billing, spend reconciled
$ ls ckpts/                                     # expect: vae_*.pt and ldm_*.pt present locally
$ runpodctl get pod                             # expect: the pod no longer listed (terminated)
$ cat runs/spend_reconciliation.md
# expect: "Actual $A vs budget $B (variance ±X%)" — actual hours × $/hr + storage + egress
```
