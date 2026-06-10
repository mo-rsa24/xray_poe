# 🖥️ RunPod Execution

## Mission
Provision a RunPod GPU at the config chosen by the cost memo, set up the
environment, transfer the corpus, run the shared-VAE and single-disease-LDM
trains end-to-end behind their overfit-sanity gates, retrieve the checkpoints,
and tear down — turning the costed decision into the trained artifacts the rest
of Paper3 depends on. Owns running on rented infra, not the training-code design
(that lives in the vae and single-disease-ldm scopes).

## Objectives
1. Provision the chosen GPU + network volume on RunPod — template/image,
   CUDA/PyTorch env, mounts — reproducibly.
2. Transfer the clean corpus from data-foundation to the volume and verify
   integrity post-transfer.
3. Run VAE training on the pod behind its overfit-sanity gate; checkpoint to ckpts/.
4. Run single-disease LDM training on the pod behind its overfit-sanity gate;
   checkpoint to ckpts/.
5. Retrieve checkpoints + logs locally, tear down the pod, and reconcile actual
   spend against the budget.

## Goals
1. Pod provisioned at the chosen config; environment pinned (image/requirements) and recorded.
2. Corpus on the volume, integrity-verified (count/checksum).
3. VAE checkpoint + logs retrieved; recon gate passed (per the vae scope).
4. LDM checkpoint + logs retrieved; single-disease FID convincing (per the ldm scope).
5. Pod torn down (no ongoing charge); actual spend recorded vs. budget.

## Expected Outcome
The shared VAE and single-disease LDM checkpoints sit in ckpts/ locally,
produced on rented RunPod hardware at the costed config, with the pod torn down
and actual spend reconciled against the budget — so downstream composition work
has its trained models and no GPU is left billing.

## Definition of Done
1. RunPod pod provisioned at the chosen GPU + volume; environment pinned
   (image or requirements) and recorded.
2. Corpus transferred to the volume; post-transfer integrity check passes.
3. VAE overfit-sanity passed, then VAE trained on the pod; checkpoint + logs in ckpts/.
4. LDM overfit-sanity passed, then single-disease LDM trained on the pod;
   checkpoint + logs in ckpts/.
5. Pod torn down (no ongoing charge); actual spend reconciled against the cost
   memo's budget, variance noted.

## Sub-Scopes
(none yet — added by decompose-plan)

## Plans
- ⚠️ 00-profile-and-budget.md — FIRST: cheap 48 GB rental, `vae.profile` on noise → measured img/s, lock the `vae.budget` $ before the big commit
- ⚠️ 00b-ldm-pilot-and-budget.md — LDM cost on noise latents (early, kills §B EST) + a real-latent FID-slope pilot (after the VAE) → steps-parametric budget + cap
- ⚠️ 01-provision-and-env.md — provision the pod + network volume; git clone + bash bootstrap; pin & record env
- ⚠️ 02-corpus-transfer.md — corpus onto the volume; post-transfer count + checksum integrity check
- ⚠️ 03-vae-train-on-pod.md — VAE overfit gate, then real-data VAE train; checkpoint + ceiling to ckpts/
- ⚠️ 04-ldm-train-on-pod.md — latents + LDM overfit gate, then real-data CFG train; checkpoint + FID to ckpts/
- ⚠️ 05-retrieve-teardown-reconcile.md — pull checkpoints local, tear down the pod, reconcile spend vs budget
