"""GPU overlay proof-images for the vae-v2 exhibit pack.

Reuses scripts/ctr_overlay.py (ctr_for_image geometry + _overlay drawing) on real/generated
films, then stamps a big plain-language banner. Three geometry anchors:
  1. cardio CTR  — p50 synth cardio, PSPNet heart/thorax lines, "CTR small vs real"
  2. effusion CP — a blunted effusion film, costophrenic-angle marker, "CP blunted"
  3. co-presence — an exp03 film with a big heart but a SHARP corner, "big heart, no fluid"
GPU lane, run once. Absolute paths (the /datasets symlink snapshot-reverts).
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, "/home-mscluster/mmolefe/Playground/PhD/Paper3")
import numpy as np
import torch
from PIL import Image, ImageDraw

import scripts.generate as G
from scripts.validate_individual_synthetics import generate_disease
import scripts.ctr_overlay as C
from metrics.grad_cam_utils import GradCAM, load_model, preprocess

CKPT = "/datasets/mmolefe/checkpoints/vae_v2_ldm_v4/model_ema_step0060000.safetensors"
VAE  = "/datasets/mmolefe/checkpoints/vae_v2/vae_v2_step0075000.pt"
SF   = "/datasets/mmolefe/data/latents_v2/scale_factor_perch.pt"
OUT  = Path("results/vae_v2/exhibits")
ANCH = OUT / "anchors"; ANCH.mkdir(parents=True, exist_ok=True)
CARDIO, EFFUSION = 1, 2
dev = torch.device("cuda")


def _banner(im: Image.Image, text: str, sub: str = "") -> Image.Image:
    """Stack a title band above the 512x512 overlay with a big plain label (fits width)."""
    band = 40 if not sub else 62
    canvas = Image.new("RGB", (im.width, im.height + band), (16, 16, 16))
    canvas.paste(im, (0, band))
    d = ImageDraw.Draw(canvas)
    d.text((8, 5), text, fill=(255, 235, 0), font=C._font(21))
    if sub:
        d.text((8, 34), sub, fill=(210, 210, 210), font=C._font(16))
    return canvas


def _score_and_overlay(seg, idx, path: Path, tint=True):
    m = C.ctr_for_image(seg, idx, path, str(dev), with_masks=True)
    return m, C._overlay(path, m, tint=tint)


def main():
    seg, idx = C._load_seg(str(dev))
    unet = G._load_unet(Path(CKPT), dev)
    vae = G._load_vae(Path(VAE), dev)
    sf = torch.load(SF, map_location="cpu", weights_only=False).float()

    # ---- 1. cardio CTR: p50 synth cardio ---------------------------------
    imgs = generate_disease(unet, vae, sf, CARDIO, 24, 4.0, "null", 50, 42, dev)
    rows = []
    for k, im in enumerate(imgs):
        p = ANCH / f"cardio_{k:02d}.png"; im.convert("L").save(p)
        m = C.ctr_for_image(seg, idx, p, str(dev))
        if m["ok"] and np.isfinite(m["ctr"]):
            rows.append((abs(m["ctr"] - 0.482), p, m["ctr"]))
    rows.sort()
    _, p50, ctr = rows[0]
    m, ov = _score_and_overlay(seg, idx, p50)
    _banner(ov, f"CTR {ctr:.2f} - small heart", "real cardio median 0.54 (generator under-draws)").save(OUT / "overlay_cardio_ctr.png")
    print(f"[cardio] p50 CTR={ctr:.3f} -> overlay_cardio_ctr.png")

    # ---- 2. effusion CP: a blunted film ----------------------------------
    imgs = generate_disease(unet, vae, sf, EFFUSION, 24, 4.0, "null", 50, 7, dev)
    best = None
    for k, im in enumerate(imgs):
        p = ANCH / f"effusion_{k:02d}.png"; im.convert("L").save(p)
        m = C.ctr_for_image(seg, idx, p, str(dev))
        cps = [c for c in (m["cp_left"].get("angle_deg"), m["cp_right"].get("angle_deg")) if c]
        cpm = max(cps) if cps else 0
        if cpm > 80 and (best is None or cpm > best[0]):
            best = (cpm, p, m)
    cpm, p_eff, m_eff = best
    _, ov = p_eff, C._overlay(p_eff, m_eff, tint=True)
    _banner(ov, f"CP {cpm:.0f} deg - blunted corner", "effusion fills the costophrenic angle").save(OUT / "overlay_effusion_cp.png")
    print(f"[effusion] blunted CP={cpm:.0f} -> overlay_effusion_cp.png")

    # ---- 3. co-presence: big heart, SHARP corner (the null) --------------
    p_co = Path("results/vae_v2/downstream_eval/exp03/compose/img_01017.png")
    m, ov = _score_and_overlay(seg, idx, p_co)
    cpl = m["cp_left"].get("angle_deg"); cpr = m["cp_right"].get("angle_deg")
    cpv = max([c for c in (cpl, cpr) if c] or [0])
    _banner(ov, f"big heart CTR {m['ctr']:.2f}, sharp corner CP {cpv:.0f} deg",
            "big heart, no fluid - the co-presence null").save(OUT / "overlay_copresence.png")
    print(f"[co-presence] CTR={m['ctr']:.2f} CP={cpv:.0f} -> overlay_copresence.png")

    # ---- 4. best-of-8 cardio GradCAM: detector fires on the heart --------
    model, ci, ei = load_model("ckpts/presence_classifier_finetuned.pt", str(dev))
    gcam = GradCAM(model, str(dev))
    imgs = generate_disease(unet, vae, sf, CARDIO, 8, 4.0, "null", 50, 123, dev)
    cands = []
    for k, im in enumerate(imgs):
        p = ANCH / f"bo8_cardio_{k:02d}.png"; im.convert("L").save(p)
        x, disp = preprocess(p)
        x = x.to(dev)
        with torch.no_grad():
            pc = float(torch.sigmoid(model(x)).squeeze()[ci])
        cands.append((pc, p, x, disp))
    cands.sort(key=lambda t: -t[0])
    pc, p_best, x_best, disp = cands[0]           # best-of-8 = highest cardio prob
    cam_c = gcam.cam(x_best, 0)
    base = np.asarray(disp, dtype=np.float32)
    if base.max() > 1.5:
        base /= 255.0
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    from PIL import Image as _I
    if cam_c.shape[:2] != base.shape[:2]:
        cam_c = np.asarray(_I.fromarray((cam_c * 255).astype(np.uint8)).resize(base.shape[1::-1]), dtype=np.float32) / 255.0
    heat = np.stack([cam_c, np.zeros_like(cam_c), np.zeros_like(cam_c)], axis=-1)
    blend = (0.55 * base + 0.45 * heat)
    ov = Image.fromarray((np.clip(blend, 0, 1)*255).astype(np.uint8))
    _banner(ov, f"detector: cardio {pc:.2f}", "GradCAM (red) sits on the heart - best-of-8").save(OUT / "overlay_gradcam_cardio.png")
    print(f"[gradcam] best-of-8 cardio prob={pc:.2f} -> overlay_gradcam_cardio.png")

    print("\nwrote 4 overlays to", OUT)


if __name__ == "__main__":
    main()
