# RunPod VAE Training Runbook

**Deployment assumption:** `git push` locally → `git pull` on pod → run steps in order.
**Estimated total wall-clock (A6000 48 GB):** ~1.5–2 hrs setup + ~25–200 hrs train (see §5).

---

## PRE-POD CHECKLIST (do this on your local machine before renting anything)

Gather these credentials — you cannot start without them:

- [ ] **Kaggle API token** — `kaggle.com → Settings → API → Create New Token` → downloads `kaggle.json`
- [ ] **W&B API key** — `wandb.ai → Settings → Danger Zone → API keys`
- [ ] **Git remote** — repo must be pushed and accessible on the pod (GitHub/GitLab SSH or HTTPS token ready)
- [ ] **RunPod account** — billing set up, credit loaded

---

## STEP 0 — Provision the pod (~5 min, ~$0)

1. Log into [runpod.io](https://www.runpod.io) → **Deploy**.
2. Choose GPU: **A6000 48 GB** ($0.44/hr) for minimum cost; **A100 80 GB** ($1.19/hr) for 2× batch.
3. Select template: `runpod/pytorch:2.5.1-py3.11-cuda12.4.1-devel-ubuntu22.04` (matches torch 2.5.1+cu124).
4. Attach a **network volume** → 50 GB (corpus 30 GB + checkpoints + headroom).
5. Set environment variables on the pod **before starting**:
   ```
   WANDB_API_KEY=<your-key>
   KAGGLE_USERNAME=<your-username>
   KAGGLE_KEY=<your-api-key>
   ```
6. Start the pod. Note the pod ID and $/hr in `runs/pod_provision.md`.

---

## STEP 0a — Configure passwordless SSH (~2 min, do once per pod)

RunPod gives you a command like:

```bash
ssh root@<ip-address> -p <port> -i ~/.ssh/id_rsa
```

**Add it to `~/.ssh/config` on your local machine** so you never type the full command again:

```text
Host runpod
    HostName <ip-address>
    User root
    Port <port>
    IdentityFile ~/.ssh/id_rsa
    StrictHostKeyChecking no
    ServerAliveInterval 60
    ServerAliveCountMax 10
```

Replace `<ip-address>` and `<port>` with the values from the RunPod "Connect" dialog.

After saving, test it:

```bash
ssh runpod
```

**Notes:**

- `StrictHostKeyChecking no` avoids the fingerprint prompt — fine for ephemeral pods.
- `ServerAliveInterval 60` keeps the connection alive through long training runs.
- When you rent a new pod the IP/port change — just update the two lines in `~/.ssh/config`.
- If RunPod gives you a custom key path instead of `~/.ssh/id_rsa`, update `IdentityFile` accordingly.

**Quick edit shortcut for future pod rentals:**

```bash
# Replace IP and port in one command (run from local machine):
sed -i "s/HostName .*/HostName <new-ip>/" ~/.ssh/config
sed -i "s/Port .*/Port <new-port>/" ~/.ssh/config
```

---

## STEP 0b — Connect VSCode to the pod (optional but recommended)

1. Install the **Remote - SSH** extension in VSCode locally.
2. In RunPod → your pod → **Connect** → copy the SSH command (looks like `ssh root@<ip> -p <port> -i ~/.ssh/id_rsa`).
3. In VSCode: `Ctrl+Shift+P` → **Remote-SSH: Connect to Host** → paste the host string.
4. Once connected, open `/workspace/Paper3` as the folder.
5. Install the **Claude Code** VSCode extension on the remote host if you want AI assistance on the pod — or just use `claude` in the pod's integrated terminal.

> The `.claude/settings.json` in the repo is already committed with pod-compatible (`python3`) paths, so Claude Code picks up the right permissions automatically.

---

## STEP 1 — Clone repo + bootstrap (~45–90 min, dominated by NIH download)

```bash
git clone <your-repo-url> /workspace/Paper3
cd /workspace/Paper3
bash scripts/setup_pod.sh
```

`setup_pod.sh` does in order (all blocking):
1. Verifies CUDA + GPU name/VRAM.
2. `pip install` all pinned deps from `requirements.txt` (torch CUDA wheel first).
3. Logs into W&B via `WANDB_API_KEY`.
4. Places `~/.kaggle/kaggle.json` from `KAGGLE_USERNAME` + `KAGGLE_KEY`.
5. **Downloads NIH ChestX-ray14 (~45 GB) — blocks until complete.**
   *(~30–60 min depending on RunPod bandwidth; ~112k images)*
6. Creates `ckpts/ runs/ figures/ logs/`.
7. Runs `python -m vae.sanity` — architecture gate.
8. Writes `runs/pod_provision.md`.

**Expected output:** `setup complete — ready for scripts/train_vae.sh`

---

## STEP 2 — Profile: measure img/s + lock the budget (~15 min, ~$0.11)

```bash
bash scripts/train_vae.sh --profile
```

Runs `vae.profile --res 512 --precision bf16 --grad-checkpoint --sweep`.

**Expected output:** `peak VRAM X.X GB | Y.Y img/s` + the max batch that fits.

**Action after this step:**
1. Fill in `runs/pod_provision.md` measured cells.
2. Update `configs/vae.yaml` `batch:` to the measured max batch.
3. Confirm cost: `python -m vae.budget --img-s <measured> --epochs 50 --n 112120 --rate 0.44`

---

## STEP 3 — Real-data overfit gate (~30 min, ~$0.22)

```bash
bash scripts/train_vae.sh --overfit
```

Runs 500 steps on the first 4 real NIH images (fp32, deterministic decode(μ)).
Logs recon images + loss curve to W&B run `paper3-vae`.

**Gate criterion:** recon loss must decrease by >10% from step 1 to step 500.
The script asserts this automatically and exits 1 if it fails.

**If the gate fails:**
- Check W&B recon images — are they all black / blown up?
- Common causes: bad image normalisation (check pixel range), bad CSV path, all-zero images.
- Do **not** proceed to Step 4 until this passes.

**Expected output:** `GATE PASSED — recon decreased. Proceed to full train.`

---

## STEP 4 — Full training run (~25–200 hrs depending on measured img/s)

```bash
bash scripts/train_vae.sh --config configs/vae.yaml
```

**What runs:**
- Real NIH data, 95/5 train/val split, frontal views only.
- bf16 autocast, AdamW, EMA, grad-checkpoint.
- Checkpoint every 5k steps → `ckpts/vae_step*.pt`.
- W&B: loss + grad norm + σ̄ every 100 steps; recon images every 500 steps; latent PCA every 5k steps.
- Stops at `--steps 150000` (configurable) or when you kill it — final checkpoint always written.

**Monitor via W&B:** open `wandb.ai/<your-entity>/paper3-vae`
- `loss/recon` should fall steadily for the first ~10k steps then plateau.
- `train/z_sigma_mean` should stay near 1.0 — if it climbs to >10, KL is not constraining.
- `recon/train` images should look progressively sharper.

**Recon gate (stops the run):** run `python -m vae.eval` on a held-out batch once
`loss/recon` plateaus — if SSIM > 0.85 and LPIPS < 0.15 the codec is good enough for the LDM.

**Resume after interruption (pod restart / preemption):**
```bash
bash scripts/train_vae.sh --config configs/vae.yaml
# train_vae.sh auto-detects the latest ckpts/vae_step*.pt and passes --resume
```

**Wall-clock estimates at 512² (A6000, measured after Step 2):**

| img/s | hrs/50 epochs | cost @$0.44 |
|------:|------:|------:|
| 8  | 194 h | $103 |
| 16 |  97 h |  $51 |
| 32 |  49 h |  $26 |

*(img/s is unknown until Step 2. The table is from `vae.budget`.)*

---

## STEP 5 — Retrieve checkpoints + teardown (~5 min)

```bash
# from your local machine
rsync -avz --progress <pod-ssh>:/workspace/Paper3/ckpts/ ./ckpts/
rsync -avz --progress <pod-ssh>:/workspace/Paper3/runs/  ./runs/
rsync -avz --progress <pod-ssh>:/workspace/Paper3/figures/ ./figures/
```

Then stop the pod from the RunPod console.

**Post-run checklist:**
- [ ] `ckpts/vae_final.pt` exists and `torch.load` succeeds
- [ ] W&B run shows a converging recon curve
- [ ] SSIM / LPIPS recon gate recorded in `runs/vae_train_*.log`
- [ ] `runs/pod_provision.md` has the measured img/s + total spend filled in
- [ ] Pod stopped (billing ends)

---

## QUICK REFERENCE — environment variables

| Variable | Where to set | Purpose |
|---|---|---|
| `WANDB_API_KEY` | RunPod pod env | W&B login (non-interactive) |
| `KAGGLE_USERNAME` | RunPod pod env | Kaggle download auth |
| `KAGGLE_KEY` | RunPod pod env | Kaggle download auth |
| `WANDB_PROJECT` | optional, train_vae.sh reads it | override project name |
| `VAE_PYTHON` | optional | override python binary path |

---

## TROUBLESHOOTING

**OOM on the full run:**
- Reduce `batch` in `configs/vae.yaml` by 1 and relaunch (it resumes).
- Already using `grad_checkpoint: true`? That's the main lever.

**NIH download interrupted:**
- Re-run `bash scripts/download_nih.sh` — Kaggle CLI resumes partial downloads.

**W&B not logging:**
- Check `WANDB_API_KEY` is set: `echo $WANDB_API_KEY`.
- Set `wandb_project: ""` in `configs/vae.yaml` to disable and proceed without it.

**`vae.sanity` fails on the pod:**
- Likely a CUDA/torch mismatch. Check `python -c "import torch; print(torch.version.cuda)"` matches the pod driver.
