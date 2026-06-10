# 🖥️ Provision Pod + Pin Environment

## Background
Gated on the cost memo's GO decision (`compute-budget/03`), which fixes the GPU
tier, the network-volume size, and the budget. This plan stands up that exact
config and makes the environment reproducible — the first of the five on-pod
steps.

## Description
Provision the chosen RunPod GPU pod with an attached network volume, then bring
the environment up the same way every time: `git clone` the Paper3 repo onto the
volume and run one bootstrap bash script that installs the pinned dependencies.
Record what was provisioned so the run is reproducible and the spend is traceable.

## Purpose
A reproducible, recorded environment is the foundation every later step stands on
— corpus transfer, both trains, and the spend reconciliation all assume the pod
exists at the costed config with the code and deps in place. Serves Objective 1
and Definition-of-Done #1.

## Goal
A running pod at the chosen GPU + network volume, with the repo cloned to the
volume, dependencies installed from a pinned spec, and a short provisioning record
(GPU, volume size, image tag, $/hr, pod ID) written into the repo.

## Tasks
- [ ] ⚠️ Provision the pod at the cost-memo config (GPU tier + network-volume GB), using a CUDA/PyTorch base image matching the local `requirements.txt`
- [ ] ⚠️ On the pod, `git clone` the Paper3 repo onto the network volume (so code + ckpts survive a pod restart)
- [ ] ⚠️ Run the bootstrap bash script (`bash scripts/setup_pod.sh`) — installs pinned deps, verifies CUDA sees the GPU
- [ ] ⚠️ Record the provisioning facts (GPU, volume GB, image tag, $/hr, pod ID, start time) into `runs/pod_provision.md`

## Recommended skill
— custom; no skill fits (RunPod infra/ops execution on rented hardware).

## Engagement Instructions
```
# DO THIS — on the freshly provisioned pod (the only on-pod steps are git clone + bash)
$ git clone <paper3-repo-url> /workspace/Paper3 && cd /workspace/Paper3
$ bash scripts/setup_pod.sh            # installs pinned deps; prints CUDA + GPU check at the end

# GET THAT — environment is live and pinned
$ nvidia-smi                            # expect: the chosen GPU listed, driver/CUDA present
$ python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# expect: pinned torch version, True, the chosen GPU name
$ cat runs/pod_provision.md             # expect: GPU, volume GB, image tag, $/hr, pod ID, start time recorded
```
