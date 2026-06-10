# 01 · Provision pod + pin environment

[⌂ Index](00-INDEX.md) · [next → 02](02-corpus-transfer.md)

## Reference while you do it
- 📄 Plan: plans/compute-budget/plans/runpod-execution/plans/01-provision-and-env.md

## Section context (paste into the Todoist section)
**Description:** Provision the chosen RunPod GPU pod with an attached network volume at the cost-memo config, then bring the environment up reproducibly — git clone the Paper3 repo onto the volume and run one bootstrap bash script that installs pinned deps.
**Objective:** Stand up a reproducible, recorded environment as the foundation every later step (corpus, both trains, spend reconciliation) depends on.
**Goal:** A running pod at the chosen GPU + network volume, repo cloned to the volume, dependencies installed from a pinned spec, and a short provisioning record (GPU, volume size, image tag, $/hr, pod ID) written into the repo.
**Verify (whole leaf):** `nvidia-smi` shows the chosen GPU; `python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → pinned torch, True, the chosen GPU; `cat runs/pod_provision.md` shows GPU/volume/image/$/hr/pod ID/start time.
**▶ Recommended prompt:** — custom; no skill fits (RunPod infra/ops execution on rented hardware).

## Tasks (one at a time)
- [ ] Provision the pod at the cost-memo config (GPU tier + network-volume GB), using a CUDA/PyTorch base image matching the local `requirements.txt`
- [ ] On the pod, `git clone` the Paper3 repo onto the network volume (so code + ckpts survive a pod restart)
- [ ] Run the bootstrap bash script (`bash scripts/setup_pod.sh`) — installs pinned deps, verifies CUDA sees the GPU
- [ ] Record the provisioning facts (GPU, volume GB, image tag, $/hr, pod ID, start time) into `runs/pod_provision.md`
