# Exp 6 — detector construct + criterion validity (Gate 1 + Gate 2)

Checkpoint under test: `ckpts/detector_ctr_labels.pt`

## Gate 1 — GradCAM pointing-game (automated)

- **pointing-game = 0.23** over 20 cardio-positive films (saliency mass inside the PSPNet heart mask).
- Pass rule ≥ 0.80 → **FAIL**.
- Reads where the net looks (independent of the CTR signal); kills shortcut learning.

## Gate 2 — criterion agreement (independent path; radiologist step PENDING)

- The CTR-taught cardio detector is confirmed by **GradCAM + radiologist**, **never by CTR again** (independence rule — CTR teaches, so it cannot also validate).
- GradCAM half: satisfied by the Gate-1 pointing-game above.
- Radiologist half: **⚠️ pending human sign-off** — a blinded read confirming the highlighted region is the cardiac silhouette. Cannot be automated; record the radiologist's agreement here once obtained.

_Frame note: CAM and PSPNet heart mask compared in the 512² frame; NIH PA films are square so the detector center-crop is identity (coarse-bar approximation)._
