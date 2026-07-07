"""Shared in-memory checkpoint evaluator + two overlay renderers.

This is the single source of truth for the generation-quality dashboard:
the in-training W&B hook (scripts/train_ldm.py) and the offline re-eval
(scripts/monitor_retrain.py) both call ``evaluate_checkpoint`` so their numbers
are identical by construction.

The instrument is scripts.ctr_overlay's PSPNet CTR/CP/border read, adapted here
to score an IN-MEMORY decoded generation (a [N,1,512,512] tensor) WITHOUT writing
to disk. ``score_decoded_array`` mirrors ctr_overlay.ctr_for_image's internals
(PSPNet logits → sigmoid → heart/Llung/Rlung masks → _max_width → _cp_read →
_peri_cardiac_opacity → _border_visible) but takes a uint8 512×512 grayscale array
instead of an image path. Aggregation (gate3_pass, cardio/effusion presence) reuses
scripts.gate3_yield so the offline path and the live path agree.

Heavy imports (torchxrayvision, the presence classifier) are kept lazy / passed in
so this module imports clean WITHOUT instantiating any CUDA/PSPNet model.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Real-data reference constants (measured in exp01/exp04). Re-exported so callers
# log identical ref/* lines.
REF = {
    "cardio_real_ctr_med": 0.535,
    "cardio_real_frac_gt0p5": 0.758,
    "effusion_floor": 0.341,
    "yield_target": 0.30,
    "cardio_gen_baseline_ctr": 0.430,
}
CARDIO_REAL_CTR = 0.535      # ref/cardio_real  (the flat reference line on the CTR panel)
CTR_THRESHOLD = 0.5          # ref/threshold

# Finding-label convention (matches generate.DISEASE_MAP / gate3_yield).
LABEL_NORMAL = 0
LABEL_CARDIO = 1
LABEL_EFFUSION = 2
LABEL_POE = "poe"            # composed both-disease (no single integer label)

# exp04 generated-image floor table (instrument false-positive floor).
CROSS_CSV = Path("results/exp04_leakage/cross_presence.csv")


# ---------------------------------------------------------------------------
# Per-image container the two renderers consume
# ---------------------------------------------------------------------------

@dataclass
class OverlayResult:
    """Everything the burned / layered renderers need for one decoded image.

    ``base`` is the 512×512 uint8 grayscale generation. ``masks`` holds the three
    PSPNet segment masks (bool arrays). The line/CP fields mirror ctr_overlay's
    ctr_for_image return dict so render_burned can reproduce its aesthetic exactly.
    """
    stem: str
    base: np.ndarray                          # (512,512) uint8 grayscale
    ctr: float
    heart_conf: float
    opacity: float
    border_visible: bool
    cp_left: dict = field(default_factory=dict)
    cp_right: dict = field(default_factory=dict)
    heart_line: list | None = None            # [x0,y,x1,y]
    thorax_line: list | None = None           # [x0,y,x1,y]
    heart_mask: np.ndarray | None = None
    llung_mask: np.ndarray | None = None
    rlung_mask: np.ndarray | None = None
    ok: bool = False


# ---------------------------------------------------------------------------
# In-memory CTR read (ctr_for_image adapted for a decoded array)
# ---------------------------------------------------------------------------

def _decoded_to_gray512(image: Any) -> np.ndarray:
    """Coerce a single decoded image into a (512,512) uint8 grayscale array.

    Accepts a torch tensor (1,512,512)/(512,512), a numpy array, or a PIL image.
    Matches how ctr_overlay resizes to 512 grayscale (via PIL.resize) so the
    instrument sees the same input distribution as on-disk PNGs.
    """
    from PIL import Image

    if hasattr(image, "detach"):              # torch tensor
        arr = image.detach().float().cpu().numpy()
    else:
        arr = np.asarray(image)
    arr = np.squeeze(arr)
    if arr.ndim == 3:                          # (C,H,W) or (H,W,C) → take first plane
        arr = arr[0] if arr.shape[0] <= 4 else arr[..., 0]
    if arr.dtype != np.uint8:
        a = arr.astype(np.float32)
        # tensors are typically in [0,1] (decode pipeline) or [-1,1]; map to [0,255]
        if a.min() < -0.01:
            a = (a + 1.0) / 2.0
        if a.max() <= 1.01:
            a = a * 255.0
        arr = np.clip(a, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr, mode="L")
    if pil.size != (512, 512):
        pil = pil.resize((512, 512))
    return np.asarray(pil, dtype=np.uint8)


def score_decoded_array(seg, idx, gray512: np.ndarray, device: str,
                        with_masks: bool = False) -> dict:
    """In-memory twin of ctr_overlay.ctr_for_image.

    Takes a (512,512) uint8 grayscale array (NOT a path) and returns the same
    dict ctr_for_image returns: ctr, heart_w, thorax_w, ok, heart_conf,
    peri_cardiac_opacity, border_visible, cp_left, cp_right, heart_line,
    thorax_line, _masks. The only difference is the input branch — everything
    downstream calls the very same ctr_overlay helpers.
    """
    import torch
    import torchxrayvision as xrv
    import scripts.ctr_overlay as C

    arr = xrv.datasets.normalize(gray512.astype(np.float32), maxval=255)     # → [-1024,1024]
    t = torch.from_numpy(arr)[None, None].to(device)                        # (1,1,512,512)
    with torch.no_grad():
        probs = torch.sigmoid(seg(t))[0].cpu().numpy()
    heart = C._mask(probs[idx[C.HEART]])
    llung, rlung = C._mask(probs[idx[C.LLUNG]]), C._mask(probs[idx[C.RLUNG]])
    lungs = llung | rlung
    gray = gray512.astype(np.float32)
    masks = {C.HEART: heart, C.LLUNG: llung, C.RLUNG: rlung} if with_masks else None
    if not heart.any() or not lungs.any():
        return {"ctr": float("nan"), "heart_w": 0, "thorax_w": 0, "ok": False,
                "heart_conf": 0.0, "peri_cardiac_opacity": float("nan"),
                "border_visible": False, "_masks": masks,
                "cp_left": C._cp_read(llung), "cp_right": C._cp_read(rlung)}
    hw, hy, hx0, hx1 = C._max_width(heart)
    tw, ty, tx0, tx1 = C._max_width(lungs)
    heart_conf = float(probs[idx[C.HEART]][heart].mean())
    opacity = C._peri_cardiac_opacity(gray, heart, lungs)
    return {
        "ctr": float(hw / tw) if tw else float("nan"),
        "heart_w": hw, "thorax_w": tw, "ok": True,
        "heart_conf": round(heart_conf, 3),
        "peri_cardiac_opacity": round(opacity, 3) if np.isfinite(opacity) else float("nan"),
        "border_visible": C._border_visible(opacity, heart_conf),
        "cp_left": C._cp_read(llung), "cp_right": C._cp_read(rlung),
        "heart_line": [hx0, hy, hx1, hy],
        "thorax_line": [tx0, ty, tx1, ty],
        "_masks": masks,
    }


def _row_from_score(m: dict) -> dict:
    """Build a gate3_yield-style row from a score dict, reusing gate3_yield's logic.

    Mirrors gate3_yield.score's per-image branch so gate3_pass / cardio_present /
    effusion_present are computed identically to the offline harness.
    """
    import scripts.gate3_yield as Y

    ctr, cl, cr = m["ctr"], m["cp_left"], m["cp_right"]
    cardio = bool(np.isfinite(ctr) and ctr > CTR_THRESHOLD)
    effusion = bool(cl["blunted"] or cr["blunted"])
    angs = [a for a in (cl["angle_deg"], cr["angle_deg"]) if a is not None and np.isfinite(a)]
    cp_max = max(angs) if angs else np.nan
    abstain = bool((np.isfinite(ctr) and abs(ctr - 0.5) <= Y.CTR_MARGIN) or
                   (angs and min(abs(a - 80) for a in angs) <= Y.CP_MARGIN))
    return {"ctr": ctr, "cp_max_angle": cp_max, "cardio_present": cardio,
            "effusion_present": effusion, "gate3_pass": cardio and effusion,
            "abstain": abstain}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _ctr_med(rows: list[dict]) -> float:
    a = np.array([r["ctr"] for r in rows if np.isfinite(r["ctr"])], dtype=np.float64)
    return float(np.median(a)) if len(a) else float("nan")


def _frac_gt05(rows: list[dict]) -> float:
    a = np.array([r["ctr"] for r in rows if np.isfinite(r["ctr"])], dtype=np.float64)
    return float((a > CTR_THRESHOLD).mean()) if len(a) else float("nan")


def _load_floor() -> dict:
    """exp04 generated-image instrument floor (no_finding false-positive rate).

    Returns {'cardio': frac, 'effusion': frac} from the no_finding_floor set's
    instrument presence columns. Used to floor-subtract presence so the dashboard
    reports excess-over-floor, not raw instrument false positives.
    """
    try:
        import pandas as pd
        df = pd.read_csv(CROSS_CSV)
        fl = df[df["set"] == "no_finding_floor"]
        if len(fl):
            return {"cardio": float(fl["cardio_present"].mean()),
                    "effusion": float(fl["effusion_present"].mean())}
    except Exception:
        pass
    return {"cardio": 0.0, "effusion": 0.0}


def evaluate_checkpoint(decoded_images, labels: list, seg, classifier=None) -> tuple:
    """Score a batch of DECODED generations and return (metrics, overlay_results).

    Parameters
    ----------
    decoded_images : Tensor[N,1,512,512] (or array/list of N images in [0,1]/[-1,1])
        The decoded generations to score — NOT the original/real images.
    labels : list[int|str]
        One label per image: 1=cardio, 2=effusion, "poe"=composed both-disease.
    seg :
        (PSPNet, idx) tuple from scripts.ctr_overlay._load_seg, ALREADY on the
        device you want inference to run on. Pass it pre-loaded so this module
        never instantiates CUDA itself.
    classifier : optional
        (model, cardio_idx, effusion_idx, ckpt) from
        metrics.presence_classifier.load_model. If given, presence/* keys are
        floor-subtracted using the exp04 floor; if None they are emitted as nan.

    Returns
    -------
    (metrics, overlay_results)
        metrics : dict with NAMESPACED keys (see module docstring / spec §1).
        overlay_results : list[OverlayResult] — one per input image, for the renderers.
    """
    # device the seg model lives on
    seg_model, idx = seg
    try:
        device = str(next(seg_model.parameters()).device)
    except Exception:
        device = "cpu"

    # normalise the batch into a list of single images
    if hasattr(decoded_images, "shape") and not isinstance(decoded_images, (list, tuple)):
        imgs = [decoded_images[i] for i in range(decoded_images.shape[0])]
    else:
        imgs = list(decoded_images)

    overlays: list[OverlayResult] = []
    rows_by_label: dict = {LABEL_NORMAL: [], LABEL_CARDIO: [], LABEL_EFFUSION: [], LABEL_POE: []}
    gray_by_label: dict = {LABEL_NORMAL: [], LABEL_CARDIO: [], LABEL_EFFUSION: [], LABEL_POE: []}

    for i, (img, lab) in enumerate(zip(imgs, labels)):
        gray = _decoded_to_gray512(img)
        m = score_decoded_array(seg_model, idx, gray, device, with_masks=True)
        row = _row_from_score(m)
        if lab in rows_by_label:
            rows_by_label[lab].append(row)
            gray_by_label[lab].append(gray)
        msk = m.get("_masks") or {}
        import scripts.ctr_overlay as C
        overlays.append(OverlayResult(
            stem=f"{_label_name(lab)}_{i:03d}",
            base=gray,
            ctr=m["ctr"], heart_conf=m.get("heart_conf", 0.0),
            opacity=m.get("peri_cardiac_opacity", float("nan")),
            border_visible=bool(m.get("border_visible")),
            cp_left=m.get("cp_left", {}), cp_right=m.get("cp_right", {}),
            heart_line=m.get("heart_line"), thorax_line=m.get("thorax_line"),
            heart_mask=msk.get(C.HEART), llung_mask=msk.get(C.LLUNG),
            rlung_mask=msk.get(C.RLUNG), ok=bool(m.get("ok")),
        ))

    cardio_rows = rows_by_label[LABEL_CARDIO]
    poe_rows = rows_by_label[LABEL_POE]
    eff_rows = rows_by_label[LABEL_EFFUSION]

    metrics: dict = {
        "ctr_med/cardio": _ctr_med(cardio_rows),
        "ctr_med/effusion": _ctr_med(eff_rows),
        "ctr_med/poe": _ctr_med(poe_rows),
        "frac_gt0.5/cardio": _frac_gt05(cardio_rows),
        "frac_gt0.5/poe": _frac_gt05(poe_rows),
        "border_frac/cardio": _border_frac(cardio_rows, overlays, LABEL_CARDIO),
        "border_frac/poe": _border_frac(poe_rows, overlays, LABEL_POE),
        "yield/poe_gate3": (float(np.mean([r["gate3_pass"] for r in poe_rows]))
                            if poe_rows else float("nan")),
        # cardio→effusion leak: effusion blunting falsely present in cardio-only samples
        "leak/cardio_to_effusion": (float(np.mean([r["effusion_present"] for r in cardio_rows]))
                                    if cardio_rows else float("nan")),
        "eval/instrument_ok": 1,
    }

    # presence (classifier) — floor-subtracted; nan when no classifier supplied
    pres = {"presence/cardio": float("nan"), "presence/effusion": float("nan"),
            "presence/both": float("nan")}
    if classifier is not None:
        try:
            pres = _presence_metrics(classifier, gray_by_label, device)
        except Exception as exc:  # presence is a soft add-on; never sink the eval
            print(f"  [eval_checkpoint] presence classifier failed: {exc}")
    metrics.update(pres)
    return metrics, overlays


def _label_name(lab) -> str:
    return {LABEL_CARDIO: "cardio", LABEL_EFFUSION: "effusion", LABEL_POE: "poe"}.get(lab, str(lab))


def _border_frac(rows: list[dict], overlays: list[OverlayResult], lab) -> float:
    """Fraction of this finding's images whose cardiac border is traceable."""
    vis = [o.border_visible for o in overlays if o.stem.startswith(_label_name(lab) + "_")]
    return float(np.mean(vis)) if vis else float("nan")


def _presence_metrics(classifier, gray_by_label: dict, device: str) -> dict:
    """Floor-subtracted presence + AUC from the fine-tuned classifier on decoded arrays.

    presence/* — floor-subtracted detection fraction (binary @ Youden threshold), kept
                  for backward compat with historical W&B panels.
    auc/*       — ROC-AUC treating generated-normal as negative class.
                  auc/cardio_gen: cardio-generated vs normal-generated, cardio logit.
                  auc/effusion_gen: effusion-generated vs normal-generated, effusion logit.
                  auc/poe_cardio: PoE-generated vs normal-generated, cardio logit.
                  auc/poe_effusion: PoE-generated vs normal-generated, effusion logit.
                  nan when normal images are absent from gray_by_label (< 10 eval steps).
    """
    from metrics import presence_classifier as P

    model, cardio_idx, effusion_idx, ckpt = classifier
    thr_c = float(ckpt.get("youden_threshold_cardio", 0.5))
    thr_e = float(ckpt.get("youden_threshold_effusion", 0.5))
    floor = _load_floor()

    def _probs(grays: list[np.ndarray]) -> np.ndarray:
        if not grays:
            return np.empty((0, 2), dtype=np.float32)
        return _predict_probs_arrays(model, grays, cardio_idx, effusion_idx, device)

    pn = _probs(gray_by_label.get(LABEL_NORMAL, []))
    pc = _probs(gray_by_label[LABEL_CARDIO])
    pe = _probs(gray_by_label[LABEL_EFFUSION])
    pp = _probs(gray_by_label[LABEL_POE])

    cardio = float((pc[:, 0] > thr_c).mean()) if len(pc) else float("nan")
    effusion = float((pe[:, 1] > thr_e).mean()) if len(pe) else float("nan")
    both = (float(((pp[:, 0] > thr_c) & (pp[:, 1] > thr_e)).mean())
            if len(pp) else float("nan"))
    # floor-subtract (clip at 0): excess over the generated-image false-positive floor
    if np.isfinite(cardio):
        cardio = max(0.0, cardio - floor.get("cardio", 0.0))
    if np.isfinite(effusion):
        effusion = max(0.0, effusion - floor.get("effusion", 0.0))

    # AUC: generated-normal as negative class, generated-disease as positive.
    def _auc(neg_p: np.ndarray, pos_p: np.ndarray, col: int) -> float:
        if len(neg_p) < 2 or len(pos_p) < 2:
            return float("nan")
        try:
            from sklearn.metrics import roc_auc_score
            y = [0] * len(neg_p) + [1] * len(pos_p)
            s = np.r_[neg_p[:, col], pos_p[:, col]]
            return float(roc_auc_score(y, s))
        except Exception:
            return float("nan")

    return {
        "presence/cardio": cardio,
        "presence/effusion": effusion,
        "presence/both": both,
        "auc/cardio_gen": _auc(pn, pc, 0),
        "auc/effusion_gen": _auc(pn, pe, 1),
        "auc/poe_cardio": _auc(pn, pp, 0),
        "auc/poe_effusion": _auc(pn, pp, 1),
    }


def _predict_probs_arrays(model, grays: list[np.ndarray], cardio_idx: int,
                          effusion_idx: int, device: str) -> np.ndarray:
    """predict_probs twin that takes in-memory 512×512 uint8 arrays.

    Mirrors presence_classifier._preprocess (xrv normalize → center-crop → resize224)
    but starts from an array so no temp PNG is written.
    """
    import torch
    import torchxrayvision as xrv

    tensors = []
    for g in grays:
        arr = xrv.datasets.normalize(g.astype(np.float32), maxval=255, reshape=True)  # (1,H,W)
        arr = xrv.datasets.XRayCenterCrop()(arr)
        arr = xrv.datasets.XRayResizer(224)(arr)
        tensors.append(torch.from_numpy(arr.copy()))
    batch = torch.stack(tensors).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(batch)).cpu().numpy()
    return np.column_stack([probs[:, cardio_idx], probs[:, effusion_idx]])


# OPEN GRAFT (deferred): wandb.Table + line_series custom panel. If the namespaced
# scalar view (ctr_med/*, frac_gt0.5/*, ...) proves too coarse for the supervisor,
# upgrade by accumulating per-step rows into a wandb.Table and emitting a
# wandb.plot.line_series so CTR-median / yield curves render as one multi-line panel
# instead of N separate scalar charts. Do NOT build the body until that need is real.


# ---------------------------------------------------------------------------
# Renderer 1 — matplotlib-burned overlay (ctr_overlay aesthetic)
# ---------------------------------------------------------------------------

def _caption(o: OverlayResult) -> str:
    """The shared caption string (burned + layered use the identical text)."""
    import scripts.ctr_overlay as C  # for the severity map, kept in lockstep
    _sev = {"sharp": "S", "mild": "M", "moderate": "Mo", "severe": "Sv", "n/a": "-"}
    bv = "Y" if o.border_visible else "N"

    def _cp(c):
        return f"{c.get('angle_deg')}({_sev.get(c.get('severity'), '?')})"

    ctr_s = f"{o.ctr:.3f}" if np.isfinite(o.ctr) else "nan"
    opac_s = f"{o.opacity:.2f}" if isinstance(o.opacity, (int, float)) and np.isfinite(o.opacity) else "nan"
    return (f"{o.stem} CTR={ctr_s} border={bv} opac={opac_s} "
            f"CP L={_cp(o.cp_left)} R={_cp(o.cp_right)}")


def render_burned(image, overlay: OverlayResult) -> np.ndarray:
    """Matplotlib overlay reproducing ctr_overlay.py's aesthetic exactly.

    Borrows ctr_overlay's PIL drawing path (red heart wash + line, blue Llung +
    thorax line, violet Rlung, green/orange CP circles, yellow caption band) so
    the in-training panel is pixel-faithful to the offline overlays. Returns an
    RGB uint8 (H,W,3) array (so it drops straight into wandb.Image).
    """
    from PIL import Image
    import scripts.ctr_overlay as C

    base = overlay.base if image is None else _decoded_to_gray512(image)
    im = Image.fromarray(base, mode="L").convert("RGB").resize((512, 512))

    # Rebuild the dict shape ctr_overlay._overlay expects, then reuse its drawing.
    m = {
        "ok": overlay.ok,
        "ctr": overlay.ctr,
        "border_visible": overlay.border_visible,
        "peri_cardiac_opacity": overlay.opacity,
        "heart_line": overlay.heart_line,
        "thorax_line": overlay.thorax_line,
        "cp_left": overlay.cp_left,
        "cp_right": overlay.cp_right,
        "_masks": {C.HEART: overlay.heart_mask, C.LLUNG: overlay.llung_mask,
                   C.RLUNG: overlay.rlung_mask},
    }
    if overlay.heart_mask is not None:
        C._tint_masks(im, m["_masks"])
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    if m["ok"] and m["heart_line"] and m["thorax_line"]:
        d.line(m["heart_line"], fill=(255, 60, 60), width=4)      # heart width (red)
        d.line(m["thorax_line"], fill=(60, 160, 255), width=4)    # thorax width (blue)
    C._cp_marker(d, m["cp_left"])
    C._cp_marker(d, m["cp_right"])

    # caption band (yellow text on black) — same layout as ctr_overlay._overlay
    bv = "Y" if overlay.border_visible else "N"
    _sev = {"sharp": "S", "mild": "M", "moderate": "Mo", "severe": "Sv", "n/a": "-"}

    def _cp(c):
        return f"{c.get('angle_deg')}({_sev.get(c.get('severity'), '?')})"

    ctr_s = f"{overlay.ctr:.3f}" if np.isfinite(overlay.ctr) else "nan"
    opac_s = f"{overlay.opacity:.2f}" if isinstance(overlay.opacity, (int, float)) and np.isfinite(overlay.opacity) else "nan"
    line1 = f"{overlay.stem}  CTR={ctr_s}  border={bv}"
    line2 = f"opac={opac_s}  CP L={_cp(overlay.cp_left)} R={_cp(overlay.cp_right)}"
    fnt = C._font(20)
    band = 90 if overlay.heart_mask is not None else 62
    d.rectangle([0, 512 - band, 512, 512], fill=(0, 0, 0))
    d.text((6, 512 - band + 4), line1, fill=(255, 235, 0), font=fnt)
    d.text((6, 512 - band + 32), line2, fill=(255, 235, 0), font=fnt)
    if overlay.heart_mask is not None:
        sfnt = C._font(16)
        d.text((6, 512 - 26), "seg:", fill=(220, 220, 220), font=sfnt)
        for x, (lab, col) in zip((52, 150, 260),
                                 (("heart", C._SEG_TINT[C.HEART]), ("Llung", C._SEG_TINT[C.LLUNG]),
                                  ("Rlung", C._SEG_TINT[C.RLUNG]))):
            d.text((x, 512 - 26), lab, fill=col, font=sfnt)
    return np.asarray(im, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Renderer 2 — W&B layered (independently-toggleable) image
# ---------------------------------------------------------------------------

def _line_stripe(canvas: np.ndarray, line: list | None, value: int, h: int = 3) -> None:
    """Paint a 3px-tall horizontal stripe of class id ``value`` along a CTR line."""
    if not line:
        return
    x0, y, x1, _ = line
    x0, x1 = (x0, x1) if x0 <= x1 else (x1, x0)
    y0 = max(0, y - h // 2)
    y1 = min(canvas.shape[0], y0 + h)
    canvas[y0:y1, x0:x1 + 1] = value


def render_layered(image, overlay: OverlayResult):
    """W&B layered image: every layer toggles independently in the UI.

    masks  : 'segmentation' (1=heart,2=Llung,3=Rlung) + 'ctr_lines'
             (1=heart_width, 2=thorax_width as 3px stripes at the measured rows).
    boxes  : 'cp_points' — one box per costophrenic vertex, domain='pixel',
             box_caption = the CP angle value.
    caption: identical text to the burned renderer.
    """
    import wandb
    import scripts.ctr_overlay as C

    base = overlay.base if image is None else _decoded_to_gray512(image)
    H, W = base.shape

    # segmentation layer
    seg_data = np.zeros((H, W), dtype=np.uint8)
    if overlay.heart_mask is not None:
        seg_data[overlay.heart_mask] = 1
    if overlay.llung_mask is not None:
        seg_data[overlay.llung_mask] = 2
    if overlay.rlung_mask is not None:
        seg_data[overlay.rlung_mask] = 3

    # ctr_lines layer (separate mask so it toggles on its own)
    line_data = np.zeros((H, W), dtype=np.uint8)
    _line_stripe(line_data, overlay.heart_line, 1)
    _line_stripe(line_data, overlay.thorax_line, 2)

    masks = {
        "segmentation": {"mask_data": seg_data,
                         "class_labels": {1: "heart", 2: "Llung", 3: "Rlung"}},
        "ctr_lines": {"mask_data": line_data,
                      "class_labels": {1: "heart_width", 2: "thorax_width"}},
    }

    # cp_points boxes (domain='pixel'), caption = CP angle value
    box_list = []
    for cp in (overlay.cp_left, overlay.cp_right):
        if cp and cp.get("vertex") is not None:
            x, y = cp["vertex"]
            r = cp.get("radius", 28)
            box_list.append({
                "position": {"minX": int(x - r), "maxX": int(x + r),
                             "minY": int(y - r), "maxY": int(y + r)},
                "class_id": 1,
                "box_caption": f"CP {cp.get('angle_deg')}",
                "domain": "pixel",
            })
    boxes = {"cp_points": {"box_data": box_list,
                           "class_labels": {1: "cp_corner"}}} if box_list else None

    return wandb.Image(base, masks=masks, boxes=boxes, caption=_caption(overlay))
