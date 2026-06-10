# RunPod Execution — Todoist leaf folder

Staged by `todoist-bridge` (folder mode) from
`plans/compute-budget/plans/runpod-execution/`. Each leaf below becomes one
Todoist task; its plan's task lines become the "Do" bullets inside that task.
Run `/todoist-publish` on this folder to write it into Todoist.

Work order is the file order (01 → 05) — the five on-pod steps are a linear
sequence, each gated on the one before. Themes below become Todoist sections.

## Setup
- [ ] [01 · Provision pod + pin environment](01-provision-and-env.md) — git clone + bash bootstrap on the costed config
- [ ] [02 · Corpus onto the volume + integrity check](02-corpus-transfer.md) — count + checksum vs the manifest

## Train
- [ ] [03 · Train the shared VAE on the pod](03-vae-train-on-pod.md) — overfit gate, then real-data train → ckpts/
- [ ] [04 · Train the single-disease LDM on the pod](04-ldm-train-on-pod.md) — latents + overfit gate, then CFG train → ckpts/

## Close-out
- [ ] [05 · Retrieve, tear down, reconcile spend](05-retrieve-teardown-reconcile.md) — pull artifacts local, kill the pod, reconcile vs budget
