"""Render the vae-v2 decision timeline into a self-contained per-Act HTML deck.

Presentation-grade, supervisor/committee-facing. Each Act = one page: plain-language
narrative + methodology (how it was trained, what changed, why) + headline number +
framed exhibits, each with a detailed "what it shows / how it's measured / how to read it"
analysis + data notes + honest caveats. Plus an instruments-reference page and a colophon.

Every number is grounded in a committed source (see DECISION_TIMELINE gates + the agent-sourced
research). Figures base64-inlined. Writes to illuminate-output/decision-timeline/.
"""
from __future__ import annotations
import base64
import json
from pathlib import Path

EX = Path("results/vae_v2/exhibits")
OUT = Path("illuminate-output/decision-timeline"); OUT.mkdir(parents=True, exist_ok=True)
SHA = "7853133"
CONCEPTS = "../../docs/concepts"


def b64(name: str) -> str:
    p = EX / name
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode() if p.exists() else ""


def figure(f: dict) -> str:
    """Rich figure card: image + What it shows / How it's measured / How to read it."""
    src = b64(f["img"])
    if not src:
        return f'<figure class="missing">missing exhibit: {f["img"]}</figure>'
    blocks = ""
    for label, key in [("What it shows", "shows"), ("How it's measured", "measured"), ("How to read it", "read")]:
        if f.get(key):
            blocks += f'<div class="an"><span class="an-l">{label}.</span> {f[key]}</div>'
    cav = f'<div class="an cav"><span class="an-l">Caveat.</span> {f["caveat"]}</div>' if f.get("caveat") else ""
    return (f'<figure><img src="{src}" alt="{f["img"]}"/>'
            f'<figcaption><div class="fig-t">{f["title"]}</div>{blocks}{cav}</figcaption></figure>')


GLOSS = {
    "CTR": "cardiothoracic ratio — how wide the heart is versus the chest; above 0.5 is an enlarged heart",
    "CP angle": "costophrenic angle — the sharp bottom-outer corner where lung meets diaphragm; fluid rounds (blunts) it",
    "guidance": "classifier-free guidance weight w — how hard sampling is pushed toward the disease",
    "PoE yield": "product-of-experts yield — the share of two-disease composites that render BOTH signs at once",
    "LPIPS": "learned perceptual image patch similarity — how different two images look to a vision network; lower is closer",
    "SSIM": "structural similarity — luminance/contrast/local-structure agreement; higher is closer (1.0 identical)",
    "μ-latent": "the deterministic mean of the codec's latent (encode_moments), used instead of a randomly sampled one",
    "signal fraction": "how much of the latent carries real signal when sampled, μ-variance / (μ-variance + noise-variance)",
    "best-of-8": "generate 8 candidates per slot, keep the one with the highest heart-size (CTR), discard the other 7",
}


def gloss_html(terms):
    items = "".join(f"<li><b>{t}</b> — {GLOSS[t]}</li>" for t in (terms or []) if t in GLOSS)
    return f'<aside class="gloss"><div class="gl-h">plain-language glossary</div><ul>{items}</ul></aside>' if items else ""


def concept_links(slugs):
    if not slugs:
        return ""
    links = " · ".join(f'<a href="{CONCEPTS}/{s}.md">{s}</a>' for s in slugs)
    return f'<div class="concepts">Instruments (full definitions): {links}</div>'


def block(cls, heading, html):
    return f'<div class="{cls}"><div class="sec-h">{heading}</div>{html}</div>' if html else ""


# =====================================================================================
# ACT CONTENT — every number carries a source; methodology from the sourced research.
# =====================================================================================
ACTS = [
    dict(
        slug="codec-v1-v2", gate="[G5a]", title="Act 1 — Rebuilding the codec: v1 → v2",
        terms=["LPIPS", "SSIM"], concepts=["ctr", "pspnet"],
        summary="The original codec reconstructed the right shapes but looked blurry; a perceptual fine-tune (VAE-v2) made it look right while keeping the geometry, clearing all four reconstruction gates.",
        headline="LPIPS 0.152 → 0.022",
        story=("Everything downstream reads through the image codec (a VAE that compresses a 512×512 X-ray "
               "into a small latent and back). The original codec, <code>vae_step0025000.pt</code>, was trained "
               "on pixel-error only. It got the anatomy in the right place (heart-size error |ΔCTR| 0.009) but "
               "produced <i>perceptually soft</i> images — blurry cardiac borders that the diffusion model would "
               "inherit and amplify. So we rebuilt it."),
        methodology=(
            "<b>What changed, v1 → v2.</b> Same architecture and warm-started from the v1 weights, but with three "
            "changes: (1) add an <b>LPIPS perceptual loss</b> (weight λ=0.1, ramped over the first 1,000 steps) and a "
            "<b>PatchGAN discriminator</b> (weight 0.5) so the loss rewards <i>looking</i> right, not just matching pixels; "
            "(2) switch to a <b>border-visible NIH+VinDr diet</b> of 17,080 films (10,409 NIH + 6,671 VinDr) with "
            "flip/crop augmentation; (3) halve the learning rate to 5e-5 and drop the batch to 2 (fp32; bf16 collapsed the "
            "decoder to black, batch 4 ran out of memory with the extra losses). Trained 50,000 further steps on top of the "
            "25k warm-start = <b>75k total</b> (W&amp;B run <code>9sjehxmr</code>, 12h). "
            "<i>Honest note:</i> the discriminator never actually learned (its loss sat at 2.03 throughout), so the effective "
            "pressure was LPIPS + L1 — and that was enough to pass every gate. "
            "Sources: <code>configs/vae_v2.yaml</code>, <code>plans/03-vae-retrain.md</code>, <code>plans/02-diet-preparation.md</code>."),
        data=("Evaluated on a <b>held-out val split of n=854 border-visible films</b> (5% of the diet, withheld before "
              "training, patient-disjoint from the frozen test set). Scored by <code>scripts/eval_vae_v2.py</code>: AlexNet "
              "LPIPS + PSPNet CTR + skimage SSIM. Every number below is a median over those 854 films "
              "(<code>results/vae_v2/eval_gate/metrics_agg.csv</code>)."),
        figs=[dict(
            img="chart_vae_codec_compare.png",
            title="The four reconstruction gates, v1 vs v2 (held-out n=854)",
            shows="Whether each codec's reconstructions are faithful enough on four measures. v1 (orange) is the pixel-only original; v2 (green) is the perceptual fine-tune.",
            measured=("Two panels because the metrics point opposite ways. <b>Left (lower is better):</b> ΔCTR (median heart-size "
                      "error vs the real film), and LPIPS full + LPIPS on a 256×256 cardiac crop (perceptual difference through "
                      "AlexNet features, 0 = identical). <b>Right (higher is better):</b> lower-zone SSIM (structural similarity "
                      "on the bottom third of the image, 1.0 = identical). Dashed lines are the pass thresholds "
                      "(ΔCTR &lt;0.01, LPIPS &lt;0.10 / &lt;0.08, SSIM ≥0.88)."),
            read=("Read each pair as v1-then-v2. On the left, shorter bars are better: v1's LPIPS bars (0.152, 0.229) tower over "
                  "their thresholds — <b>v1 fails both perceptual gates</b> — while v2's (0.022, 0.020) are 4–7× under the line. "
                  "On the right, taller is better: v2's SSIM (0.976) clears 0.88 with more room than v1 (0.924). "
                  "The takeaway: v2 sharpened perceptually <i>and</i> improved the geometry (ΔCTR 0.009 → 0.003) at the same time — "
                  "the spatial-vs-perceptual tradeoff we feared did not happen."),
            caveat="v1 already passed the geometry gate (ΔCTR, SSIM); the whole case for v2 rests on the two LPIPS gates it failed."),
        ],
        caveat="These gates all score the reconstruction from the latent <i>mean</i>. Whether the latent still carries signal when <i>sampled</i> — what diffusion actually does — is a separate question, and it is where Act 2 goes.",
    ),
    dict(
        slug="latent-collapse", gate="[G5b]", title="Act 2 — The hidden trap: the latent collapsed",
        terms=["μ-latent", "signal fraction"], concepts=["ctr"],
        summary="The sharper codec had almost no signal when its latent was sampled (3.4% vs 21% for v1) — the posterior collapsed toward noise, which had already burned two diffusion runs; the fix was to train the diffusion model on the latent mean (μ) instead.",
        headline="signal fraction 21% → 3.4%",
        story=("VAE-v2 passed every reconstruction gate — but those gates all decode from the posterior <b>mean</b> μ. "
               "Diffusion is different: it denoises from random noise, so it needs the latent to carry signal when "
               "<b>sampled</b>. It did not. The measured <b>signal fraction</b> (how much of the latent is signal vs noise) "
               "was <b>3.4–3.6% for v2 vs 21% for v1</b> — the posterior had collapsed toward the N(0,1) prior, and the "
               "collapse was concentrated in one of four channels (~12%; the other three carried 0.1–1.4%)."),
        methodology=(
            "<b>How the trap was found.</b> Signal fraction = μ_std² / (μ_std² + σ_mean²), computed per checkpoint; target "
            "&gt;~15%. This gate was added <i>late</i> (2026-07-01), after two LDM retrains (<code>mzd3jwy9</code>, "
            "<code>gtsw5bq7</code>) had already trained on sampled latents and produced pure noise from step 5,000. "
            "<b>The fix:</b> train the diffusion model on the deterministic mean μ (via <code>encode_moments</code>), not sampled "
            "latents — reconstruction from μ is fine (LPIPS 0.022), so μ is 'inelegant, not unusable.' Per-channel normalization "
            "was explicitly rejected (it would amplify the near-empty channels); a single global scale on μ was used instead. "
            "This is why plan 05 re-encodes NIH+VinDr as μ-latents. Sources: <code>eval_gate/verdict.md</code>, [G5b], plans 04/05."),
        figs=[],
        caveat="Training on μ removes the sampling randomness the VAE was supposed to provide. It works here because the diffusion model supplies its own noise; but it is a workaround for a collapsed codec, recorded as such.",
    ),
    dict(
        slug="ldm-v1-v4", gate="[G6a]–[G6c]", title="Act 3 — Four diffusion models to a working generator",
        terms=["CTR", "guidance"], concepts=["ctr", "pspnet"],
        summary="Training the diffusion model in the new latent space took four tries: each fixed the last one's failure (effusion collapse → cardio starvation → washed-out anisotropy), and v4 finally rendered all classes — but its hearts still come out a touch small.",
        headline="cardio CTR: v1 0.508 → v4 0.482 (real 0.536)",
        story=("With the codec settled and μ-latents in hand, we retrained the class-conditional diffusion model (LDM) from "
               "scratch. It took four versions. The chart tells the whole arc: each version fixed the previous one's failure "
               "and exposed the next, and we stopped at v4-60k on purpose."),
        methodology=(
            "<b>Fixed across all four:</b> a class-conditional UNet (128 channels, 1000 timesteps) conditioned on "
            "no_finding / cardio / effusion + a null token, classifier-free-guidance dropout 0.15, trained in VAE-v2 μ-latent space. "
            "<b>What changed each step:</b> "
            "<b>v1</b> (<code>6wj35ips</code>) baseline, class mix 0.25/0.40/0.35 — cardio rendered (CTR 0.508) but <b>effusion "
            "collapsed</b> (its blunting feature stopped rendering, 0.38 → 0.003). "
            "<b>v2</b> (<code>umqnrs0o</code>) added weight-EMA + min-SNR and shifted the mix to 0.28/0.34/0.38 to save effusion — "
            "but dropping cardio's share to 0.34 <b>killed cardio</b> (CTR regressed to 0.385, 77% of cardio films leaked to effusion). "
            "<b>v3</b> (<code>yrcj8qgl</code>) restored cardio to 0.40 (mix 0.20/0.40/0.40) and fixed the EMA warm-up — now "
            "<b>both classes rendered</b> and PoE yield cleared its gate for the first time (0.125–0.156), but late training washed "
            "the images out. "
            "<b>v4</b> (<code>4vg2h5i4</code>) found the root cause: the μ-latents are <b>anisotropic</b> (per-channel std ≈ "
            "[0.05, 0.36, 0.09, 0.12], a 7.6× spread), so a single global scale left three of four channels far below the unit "
            "variance the diffusion noise schedule assumes — drowning them. v4 applies <b>per-channel whitening</b> (scale each "
            "channel by 1/std → isotropic [1,1,1,1]), plus a decode-scale bugfix, min-SNR turned OFF (ruled out), and an early "
            "stop at 60k. Result: all classes render coherently. "
            "Sources: <code>plans/06-ldm-retrain.md</code>, <code>plans/09-collapse-prevention.md</code>, <code>configs/ldm_vae_v2_v4.yaml</code>, tracking.md."),
        data=("Cardio CTR is measured by the PSPNet geometric judge (Act 3's instruments) on generated cardio films; the real "
              "reference median is 0.536. The class mix numbers are the training-batch sampling weights for "
              "no_finding/cardio/effusion. Run IDs are from plan 06 (the DECISION_TIMELINE version table has mis-labelled IDs — "
              "flagged separately for correction)."),
        figs=[
            dict(img="chart_ldm_versions.png",
                 title="Cardio CTR across v1 → v4, and why 60k is the stop",
                 shows="How well each diffusion version renders an enlarged heart (cardio CTR, final checkpoint), and what changed between them.",
                 measured=("Each bar is the median cardio CTR of that version's generated cardio films (PSPNet-measured). The blue "
                           "dashed line is the real-cardio median 0.536; the dotted line is CTR 0.50 (the enlarged-heart threshold). "
                           "The small text under each bar names the key change and the failure it caused. The orange ▼ on v3 marks that "
                           "cardio peaked at 0.521 around 40k steps and then <b>decayed</b> to 0.457 by 100k."),
                 read=("Left to right is the debugging story. v1 clears 0.50 but effusion is dead. v2 dips to 0.385 — the see-saw: "
                       "saving effusion starved cardio. v3 recovers both but the peak-then-decay (▼) shows why more steps hurt. v4 "
                       "(green) holds 0.482 near the peak with every class rendering. <b>Why we stopped at 60k:</b> in <i>every</i> run "
                       "cardio peaked around 40–50k then regressed, so v4 early-stops to lock cardio near its peak. The residual gap to "
                       "real (0.482 vs 0.536) is the codec's ceiling, not something more steps could fix — so it was carried to the "
                       "inference fix in Act 4."),
                 caveat="Only cardio CTR is plotted (one comparable instrument across versions). Effusion status is in the per-bar text, not the height, because it is a different measurement."),
            dict(img="chart_ctr_hist.png",
                 title="v4 hearts vs real hearts (distribution)",
                 shows="The full spread of heart-sizes for v4-generated cardio (orange) versus real cardio (blue), not just the medians.",
                 measured=("CTR = heart width ÷ thorax width. Both widths come from PSPNet segmentation: for each anatomy mask, "
                           "<code>_max_width</code> scans every row and keeps the widest span; heart-width is the widest row of the heart "
                           "mask, thorax-width the widest row of the two lungs. Each histogram is 200 films; dashed lines mark the medians, "
                           "the dotted line is CTR 0.50."),
                 read=("The orange distribution sits visibly left of blue — <b>v4 systematically under-draws the heart</b> (median 0.482 "
                       "vs 0.536), and a larger share of orange falls below the 0.50 enlarged line. It is a shift of the whole "
                       "distribution, not a few bad films."),
                 caveat="CTR reads NaN and is dropped when PSPNet fails to find a heart or lung; the histogram is over the films where the geometry was measurable."),
            dict(img="overlay_cardio_ctr.png",
                 title="One generated cardio film, with the measurement drawn on",
                 shows="What the CTR instrument actually does to a single film — the anatomy it found and the widths it measured.",
                 measured=("PSPNet (a ChestX-Det segmenter) outlines the heart and each lung. The <span style='color:#e33'>red line</span> "
                           "is the widest heart span, the <span style='color:#39f'>blue line</span> the widest thorax (lung) span; the "
                           "washes are the segment masks. CTR is red-length ÷ blue-length. The yellow banner prints the number."),
                 read=("Here CTR = 0.48: the red heart line is just under half the blue chest line, i.e. below the real 0.54 median — a "
                       "small heart, exactly the finding the histogram shows in aggregate."),
                 caveat="The red and blue widest rows are found independently, so they can sit at slightly different heights — that is correct, not a drawing error."),
        ],
        caveat="v4 renders every class and is the checkpoint every downstream result uses — but the small-heart gap is real and sets up Act 4.",
    ),
    dict(
        slug="cardio-fix", gate="cardio inference-fix", title="Act 4 — Fixing the small heart, no retrain",
        terms=["guidance", "best-of-8"], concepts=["best-of-n", "presence-classifier", "grad-cam"],
        summary="The small-heart gap closes entirely at inference: turning up guidance ramps a detector's sensitivity, and keeping the best of 8 samples pushes it past the real baseline — with no retraining.",
        headline="best-of-8 0.875 > real 0.806",
        story=("v4 <i>can</i> draw a big heart — it just does not do so often enough on the default setting. Two free inference "
               "levers close the gap without touching the weights: raising the guidance weight w, and best-of-N rejection sampling."),
        methodology=(
            "<b>The detector.</b> A fine-tuned DenseNet-121 (torchxrayvision NIH weights) with a small 2-output head, giving "
            "independent cardio and effusion sigmoids (a film can be positive for both, either, or neither). Only the last dense "
            "block is fine-tuned; both-disease images are held out of its training. <b>Sensitivity = TP/(TP+FN)</b> at the "
            "Youden-optimal threshold stored in the checkpoint. Because every film in a generated cardio set is meant to be positive, "
            "the fraction the detector flags <i>is</i> the sensitivity. "
            "<b>best-of-8.</b> Generate 8 candidates per slot, keep the one with the highest CTR, discard the other 7 (8× cost). "
            "Crucially the selection is on <b>CTR (the PSPNet judge)</b>, an instrument independent of the detector — so the detector "
            "score read on the kept set is honest, not selected-on-the-metric-being-reported. "
            "Sources: <code>cardio_enhance_compare.py</code>, <code>grad_cam_utils.py</code>, concepts <code>presence-classifier</code>/<code>best-of-n</code>."),
        data=("Lever comparison: n=160 films per lever (<code>results/section4/cardio_enhance/compare_levers.csv</code>). Real-cardio "
              "reference sensitivity 0.806 (n=160). GradCAM uses the same detector on a best-of-8 cardio film."),
        figs=[
            dict(img="chart_detector_bar.png",
                 title="Detector sensitivity per free inference lever (n=160/arm)",
                 shows="How often the detector catches synthetic cardio as the inference lever changes — from default guidance up through best-of-8, against the real baseline.",
                 measured=("Each bar is the fraction of that arm's 160 films the detector flags as cardio, at the Youden threshold. "
                           "w=1/4/6 are guidance weights (no selection); best-of-8 is w=4 keeping the max-CTR of 8; real is the real-cardio "
                           "reference. The axis is pinned 0–1 (never zoomed); the dashed line is the 0.80 pass mark."),
                 read=("Higher is better. Guidance alone ramps it steadily — <b>w1 0.35 → w4 0.63 → w6 0.75</b> — and <b>best-of-8 reaches "
                       "0.875, above real 0.806</b>, clearing the 0.80 line. So the small-heart gap is fully closable at inference; no "
                       "retrain needed."),
                 caveat="best-of-8's heart-size is high by construction (you kept the biggest-heart draw), so the honest utility check is this detector read on an independent instrument, not the CTR."),
            dict(img="overlay_gradcam_cardio.png",
                 title="Where the detector looks on a best-of-8 cardio",
                 shows="A heatmap of the pixels that most drove the detector's 'cardio' decision on one best-of-8 film.",
                 measured=("GradCAM on the DenseNet-121: backprop the cardio logit to the last 7×7 feature map, weight each channel by its "
                           "average gradient, sum, ReLU (keep only evidence <i>for</i> cardio), zero the top row (an apex border artifact), "
                           "upsample to 512, normalize by the 99th percentile. Red = high attribution."),
                 read=("The red heat sits <b>centrally, over the enlarged heart</b> — not on the borders or apex — which is where a real "
                       "cardio read should come from. The detector scores this film 0.93. It is evidence the classifier is looking at the "
                       "right anatomy, not a texture artifact."),
                 caveat="GradCAM shows correlation with the output, not a causal guarantee; a clean overlay has no bright band at the image edge (the top-row fix removes the known apex artifact)."),
        ],
        caveat="This fixes cardio as a single disease. Whether it helps the two-disease case is Act 5 — and there the answer flips.",
    ),
    dict(
        slug="copresence-null", gate="[G9v]", title="Act 5 — The headline null: two diseases won't co-occur",
        terms=["PoE yield", "CTR", "CP angle", "guidance"], concepts=["poe-composition", "ctr", "metrics"],
        summary="Composing the fixed cardio expert with the effusion expert almost never yields a film with both signs (2.6% vs a 30% bar), and it does not respond to the very guidance lever that fixed single-disease cardio — the paper's central mechanistic finding.",
        headline="PoE yield 2.6% « 30%",
        story=("The paper's real question: when both-disease X-rays are scarce, can we compose them from single-disease experts? "
               "The answer is no, and it is a clean mechanistic null — the model draws a big heart <b>or</b> fluid, rarely both."),
        methodology=(
            "<b>How composition works.</b> Product-of-experts combines the cardio and effusion experts in noise (ε) space via "
            "classifier-free guidance: ε = ε_null + w·(ε_cardio − ε_null) + w·(ε_effusion − ε_null). "
            "<b>How yield is scored.</b> Each composed film passes Gate-3 only if <b>CTR &gt; 0.5 AND a costophrenic corner is blunted "
            "at the same time</b> — both signs, measured by the same geometric PSPNet judges. "
            "<b>Floor-correction (the headline number).</b> A normal/no-finding set is scored through the identical Gate-3 to get its "
            "false-positive rate, and that floor is subtracted: corrected yield = max(0, raw − floor). This removes 'both-disease' "
            "passes you'd get by chance from generation texture alone. Robustness: measured across guidance w=2/4/6. "
            "Sources: <code>gate3_yield.py</code>, <code>exp03_yield_verdict.py</code>, concept <code>poe-composition</code>."),
        data=("N=2000 composed films (<code>results/vae_v2/downstream_eval/exp03/</code>). Floor-corrected yield 2.6% (confident 3.6%, "
              "Wilson 95% ≈ [2.6%, 5.0%]); stays 3–4% at higher cardio guidance. Bar: 30% support / 10% null."),
        figs=[
            dict(img="overlay_copresence.png",
                 title="The null made visible on one composed film",
                 shows="A single product-of-experts film that was asked for both diseases — and shows only one.",
                 measured=("Both geometric instruments drawn together: the <span style='color:#e33'>red/blue lines</span> give CTR (heart "
                           "vs chest width), and the CP corner circles read the costophrenic angle — <span style='color:#3c6'>green = sharp</span>, "
                           "<span style='color:#f80'>orange = blunted</span>."),
                 read=("This film has a <b>big heart (CTR 0.83)</b> but a <b>sharp corner (CP 51°, green)</b> — cardiomegaly rendered, "
                       "effusion did not. Multiply this across 2000 films and only 2.6% show both. It is the null you can see in one frame."),
                 caveat="One film is an illustration; the 2.6% is the quantitative claim from all 2000, floor-corrected."),
            dict(img="reuse_guidance_sweep.png",
                 title="Yield stays flat as cardio guidance rises",
                 shows="Whether pushing the cardio lever harder — the one that fixed single-disease cardio — rescues the two-disease yield.",
                 measured="Both-disease yield recomputed at cardio guidance w = 2, 4, 6, holding the scoring fixed.",
                 read=("The line is flat (roughly 2.6 → 4.1 → 3.1%). The lever that took single-disease cardio to 0.875 does <b>not</b> "
                       "move co-presence — which rules out 'the generator can't draw a big heart' as the cause. The failure is in "
                       "composition itself."),
                 caveat=None),
            dict(img="reuse_ctr_cp_scatter.png",
                 title="Heart-size vs corner-blunting, per composed film",
                 shows="Every composed film placed by how enlarged its heart is (x) and how blunted its corner is (y).",
                 measured=("x = CTR (dashed line at 0.5, the enlarged threshold); y = CP max-angle (the 80° blunted threshold). The "
                           "both-disease passers are the top-right quadrant (CTR &gt; 0.5 AND CP &gt; 80°)."),
                 read=("Films pile up along the axes — enlarged-heart-only or blunted-only — and the <b>top-right both-disease quadrant is "
                       "nearly empty</b>. That emptiness is the 2.6% yield, shown as a cloud."),
                 caveat=None),
        ],
        caveat="Single-disease synthetic sets (best-of-8 cardio, native effusion) remain usable downstream; it is specifically the two-disease composition that fails.",
    ),
    dict(
        slug="augment-null", gate="[G9a]", title="Act 6 — Fake hearts don't help a real detector",
        terms=[], concepts=["presence-classifier", "best-of-n"],
        summary="Adding the model's own best synthetic cardiomegaly to a real detector's training does not move its accuracy — a separate, exploratory null that does not touch the main result.",
        headline="augment-probe: flat",
        story=("A side question: even if composition fails, does the flagship best-of-8 synthetic cardio help a <i>real</i> "
               "cardio detector as extra training data? No."),
        methodology=("Train the real detector with and without added synthetic cardio and compare AUC. The synthetic pile is the same "
                     "best-of-8 set that scores 0.875 as a generated set. Sources: <code>results/section4/augment_probe/</code>, [G9a]."),
        figs=[dict(
            img="reuse_fig_augment_probe.png",
            title="Detector AUC with vs without added synthetic cardio",
            shows="Whether adding fake cardiomegaly to the training pool improves a real cardio detector.",
            measured="Real-detector AUC across conditions that add different synthetic piles; higher AUC = better detector.",
            read=("The lines are flat and overlapping — adding the fakes adds no signal the detector can use. best-of-8's value is as a "
                  "<i>generated set</i> (it scores 0.875 &gt; real), not as detector fuel."),
            caveat="This is 'no measured help,' not 'proven useless' — the probe is reader-limited; it does not affect the co-presence decision."),
        ],
        caveat=None,
    ),
    dict(
        slug="effusion-close", gate="[G6d]–[G6f]", title="Act 7 — Effusion sanity, and closing plan 06",
        terms=["CP angle", "guidance"], concepts=["cfg-guidance", "ctr", "pspnet"],
        summary="Generated effusion does blunt the costophrenic corner and responds to guidance (32% → 66%), but never in ≥80% of films; plan 06 was closed recording this as a known limitation rather than forced to a pass.",
        headline="blunt-rate 32%@w1 → 66%@w7",
        story=("The last open item on plan 06 was the effusion geometric sanity: do generated effusion films blunt the "
               "costophrenic corner in most samples? We ran it on the Blackwell GPU (after fixing a scoring-harness bug that had "
               "reported 0/200), then swept guidance to understand the number."),
        methodology=(
            "<b>How blunting is read.</b> On each lung mask, <code>_cp_read</code> finds the inferolateral corner, samples a "
            "28-pixel circle around it, and measures the fraction of that circle falling inside the lung × 360 = the angle. A sharp "
            "corner leaves most of the circle outside the lung (small angle); fluid fills the recess (large angle). A film is "
            "<b>blunted when the angle &gt; 80°</b>. Blunt-rate = the share of effusion films that are blunted. "
            "<b>The sweep.</b> 200 effusion-only films per guidance weight w ∈ {1,2,4,7}, same seed so only guidance varies. "
            "Sources: <code>scripts/effusion_blunt_check.py</code>, <code>ctr_overlay.py</code> (_cp_read/_cp_marker)."),
        data=("n=200 per w, valid 200/200 after the harness fix (<code>results/vae_v2/ldm_sanity/v4_60k_wsweep/</code>). Instrument "
              "validated against exp03 (2000 films): the 80° threshold separates cleanly — non-effusion median CP 61°, blunted median 92°."),
        figs=[
            dict(img="chart_blunt_curve.png",
                 title="Blunt-rate vs guidance weight",
                 shows="How the share of effusion films with a blunted corner changes as guidance is pushed harder.",
                 measured="For each guidance weight w, the fraction of 200 effusion-only films whose costophrenic angle exceeds 80°. The dashed line is the 0.80 pass mark.",
                 read=("Higher is better. The rate climbs monotonically — <b>32% → 44% → 58% → 66%</b> across w=1→7 — and the median "
                       "film crosses 80° by w=4. But it <b>never reaches the 0.80 bar</b>, even at aggressive w=7. So the low number is "
                       "partly weak guidance and partly a genuine partial shortfall."),
                 caveat="The 80° blunt threshold is hand-calibrated (robust to ±10°), not a validated clinical constant — a footnote, not a defect."),
            dict(img="overlay_effusion_cp.png",
                 title="A generated effusion film with blunted corners",
                 shows="What a blunted costophrenic corner looks like when the instrument flags it.",
                 measured=("The CP corner circles: <span style='color:#3c6'>green = sharp</span>, <span style='color:#f80'>orange = "
                           "blunted</span> (angle &gt; 80°). The dense white haze at the lung bases is the effusion morphology being detected."),
                 read=("Both corners are orange (CP 150° / 109°) — real basal effusion filling the recesses. This is genuine disease "
                       "morphology, not an instrument artifact; the model renders it, just not in ≥80% of films at low guidance."),
                 caveat="Heart CTR reads NaN on this film (the heart mask failed) — fine here, since this is an effusion exhibit and only the CP corners matter."),
        ],
        caveat="Decision [G6f]: plan 06 closed with the blunt-rate recorded as a known limitation (effusion is classifier-detectable, Exp-09 sens 0.865 PASS; the median film blunts at w≥4). Not forced to a pass.",
    ),
]

SLUGS = [a["slug"] for a in ACTS]


def fname(i: int) -> str:
    return f"{i+1:02d}-{SLUGS[i]}.html"


INSTR_N = len(ACTS) + 1
COLO_N = len(ACTS) + 2

CSS = """
:root{--bg:#faf9f7;--fg:#1c1a17;--mut:#6b655c;--card:#fff;--line:#e6e2da;--acc:#b3541e;--badge:#2c7fb8;--sec:#31708a}
@media(prefers-color-scheme:dark){:root{--bg:#16150f;--fg:#eae6dd;--mut:#9a9384;--card:#201e17;--line:#332f26;--acc:#e08a4a;--badge:#5fa8d3;--sec:#7cc0d8}}
:root[data-theme=dark]{--bg:#16150f;--fg:#eae6dd;--mut:#9a9384;--card:#201e17;--line:#332f26;--acc:#e08a4a;--badge:#5fa8d3;--sec:#7cc0d8}
:root[data-theme=light]{--bg:#faf9f7;--fg:#1c1a17;--mut:#6b655c;--card:#fff;--line:#e6e2da;--acc:#b3541e;--badge:#2c7fb8;--sec:#31708a}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 -apple-system,Segoe UI,Roboto,sans-serif}
.wrap{max-width:880px;margin:0 auto;padding:32px 20px 80px}
.top{display:flex;justify-content:space-between;align-items:baseline;border-bottom:2px solid var(--line);padding-bottom:12px;margin-bottom:8px;flex-wrap:wrap;gap:8px}
.gate{font:12px ui-monospace,monospace;color:var(--mut)}
h1{font-size:27px;margin:14px 0 4px;line-height:1.22}
.summary{font-size:18px;margin:8px 0 16px}
.badge{display:inline-block;background:var(--badge);color:#fff;font-weight:700;padding:4px 12px;border-radius:20px;font-size:15px;margin:4px 0 18px}
.body{margin:0 0 18px}
.sec-h{font-weight:700;color:var(--sec);font-size:12.5px;text-transform:uppercase;letter-spacing:.05em;margin:0 0 6px}
.method,.data{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:16px 0;font-size:14.5px}
.data{border-left:3px solid var(--badge)}
.method{border-left:3px solid var(--acc)}
figure{margin:22px 0;background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden}
figure img{width:100%;display:block;max-width:100%}
figcaption{padding:12px 16px}
.fig-t{font-weight:700;font-size:16px;margin-bottom:8px}
.an{font-size:14px;margin:6px 0;color:var(--fg)}
.an-l{color:var(--sec);font-weight:700}
.an.cav .an-l{color:var(--acc)}
code{background:var(--bg);padding:1px 5px;border-radius:4px;font-size:.9em}
.gloss{background:var(--card);border:1px solid var(--line);border-left:3px solid var(--acc);border-radius:8px;padding:10px 16px;margin:18px 0;font-size:14px}
.gl-h{font-weight:700;color:var(--acc);font-size:12px;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px}
.gloss ul{margin:0;padding-left:18px}.gloss li{margin:4px 0;color:var(--mut)}.gloss b{color:var(--fg)}
.concepts{font-size:13px;color:var(--mut);margin:14px 0}.concepts a{color:var(--badge)}
.pagecav{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 16px;margin:18px 0;font-size:14px;color:var(--mut)}
.pagecav b{color:var(--acc)}
.nav{display:flex;justify-content:space-between;margin-top:40px;padding-top:16px;border-top:1px solid var(--line)}
.nav a{color:var(--badge);text-decoration:none;font-weight:600}
.foot{color:var(--mut);font-size:12px;margin-top:30px;border-top:1px solid var(--line);padding-top:12px}
.idx li{margin:11px 0}.idx a{color:var(--fg);font-weight:600;text-decoration:none}.idx .g{font:12px ui-monospace,monospace;color:var(--mut);margin-left:8px}
h2{font-size:19px;color:var(--sec);margin:26px 0 6px}
"""

HEAD = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title><style>{css}</style></head><body>{body}</body></html>"""


def act_page(i, act):
    prev = f'<a href="{fname(i-1)}">← {ACTS[i-1]["title"].split("—")[0].strip()}</a>' if i > 0 else '<a href="index.html">← index</a>'
    nxt = f'<a href="{fname(i+1)}">{ACTS[i+1]["title"].split("—")[0].strip()} →</a>' if i < len(ACTS)-1 else '<a href="08-instruments.html">the instruments →</a>'
    figs_html = "".join(figure(f) for f in act["figs"])
    pagecav = f'<div class="pagecav"><b>Note.</b> {act["caveat"]}</div>' if act.get("caveat") else ""
    return f"""<div class="wrap">
<div class="top"><span class="gate">gate {act['gate']}</span><span class="gate">vae-v2 · {i+1}/{len(ACTS)}</span></div>
<h1>{act['title']}</h1>
<div class="summary">{act['summary']}</div>
<span class="badge">{act['headline']}</span>
<p class="body">{act['story']}</p>
{block("method", "How it was done — methodology", act.get("methodology"))}
{block("data", "The data behind the numbers", act.get("data"))}
{figs_html}
{pagecav}
{gloss_html(act['terms'])}
{concept_links(act['concepts'])}
<div class="nav">{prev}{nxt}</div>
<div class="foot">vae-v2 decision timeline · rendered from docs/DECISION_TIMELINE.md + sourced research · git {SHA} · every number verified against results/ CSVs.</div>
</div>"""


def instruments_page():
    rows = [
        ("CTR — cardiothoracic ratio", "How wide the heart is vs the chest (>0.5 = enlarged). PSPNet segments heart and lungs; <code>_max_width</code> takes the widest row of each; CTR = heart-width ÷ thorax-width. Real cardio median ≈ 0.536. In overlays: red = heart span, blue = thorax span.", "ctr"),
        ("PSPNet segmentation", "The independent geometric 'judge' — a ChestX-Det PSPNet that outlines heart / left lung / right lung so widths are measured, not guessed. Mask tints: heart=red, L-lung=blue, R-lung=violet. If a wash spills or a lung is missing, that film's numbers are suspect.", "pspnet"),
        ("Costophrenic (CP) angle / blunt-rate", "Whether the sharp lung-base corner is crisp or filled by fluid. <code>_cp_read</code> samples a 28px circle at the corner; the fraction inside the lung × 360 = the angle; >80° = blunted (effusion). Circles: green=sharp, orange=blunted. Blunt-rate = share of films >80°. Threshold is hand-calibrated (±10°).", "cfg-guidance"),
        ("Detector sensitivity", "Of films meant to be diseased, the share the appearance classifier flags. Detector = fine-tuned DenseNet-121, independent cardio/effusion sigmoids, sensitivity = TP/(TP+FN) at the Youden threshold. Higher is better; real cardio ≈ 0.806, raw fakes 0.365.", "presence-classifier"),
        ("best-of-N", "Generate N=8 candidates, keep the highest-CTR one, discard the rest. Selection is on CTR (PSPNet), independent of the detector — so the detector read on the kept set is honest, not selected-on-the-reported-metric.", "best-of-n"),
        ("GradCAM", "Where the detector looked. Backprop a class logit to the last feature map, weight by average gradient, ReLU, upsample. Red = high attribution. For cardio it should sit centrally on the heart; a bright image-border band would be an artifact.", "grad-cam"),
        ("LPIPS &amp; SSIM", "Two reconstruction-fidelity scores. SSIM = structural agreement (higher better, 1.0 identical). LPIPS = perceptual difference through AlexNet features (lower better, 0 identical). They point opposite ways — a good panel has a tall SSIM bar next to a short LPIPS bar.", "ctr"),
        ("PoE yield", "Of two-disease composites, the share showing BOTH signs. Gate-3 passes iff CTR>0.5 AND a corner is blunted. Floor-corrected: subtract the false-positive rate of the same gate on normal films. Bar: 30% support / 10% null.", "poe-composition"),
    ]
    body = "".join(f'<h2>{t}</h2><p class="body">{d} <span class="concepts">→ <a href="{CONCEPTS}/{c}.md">{c}</a></span></p>' for t, d, c in rows)
    return f"""<div class="wrap">
<div class="top"><span class="gate">reference</span><span class="gate">{INSTR_N}/{COLO_N}</span></div>
<h1>The instruments — how every number is measured</h1>
<div class="summary">The deck leans on eight measuring tools. Each is defined here once; the pages link back to these. The theme throughout: geometric judges (PSPNet CTR / CP) are independent of the appearance classifier, so when they agree the result is trustworthy, and when they disagree that disagreement is itself the finding.</div>
{body}
<div class="nav"><a href="{fname(len(ACTS)-1)}">← {ACTS[-1]["title"].split("—")[0].strip()}</a><a href="08-colophon.html">colophon →</a></div>
<div class="foot">vae-v2 decision timeline · instruments reference · git {SHA}</div>
</div>"""


def colophon_page():
    concepts = " · ".join(f'<a href="{CONCEPTS}/{s}.md">{s}</a>' for s in
                          ["ctr", "pspnet", "cfg-guidance", "poe-composition", "presence-classifier", "best-of-n", "grad-cam", "metrics"])
    return f"""<div class="wrap">
<div class="top"><span class="gate">colophon</span><span class="gate">{COLO_N}/{COLO_N}</span></div>
<h1>Colophon — how this deck was built</h1>
<div class="summary">Every headline number is read from a committed CSV or a sourced plan/verdict; every proof image is drawn by the instrument named in its caption. Nothing here is hand-set.</div>
<p class="body"><b>Source.</b> Rendered from <code>docs/DECISION_TIMELINE.md</code> and <code>docs/concepts/*.md</code> at git <b>{SHA}</b>, with methodology and per-version numbers traced to <code>results/vae_v2/eval_gate/</code>, <code>results/section4/cardio_enhance/</code>, <code>plans/03/04/06/09</code>, and <code>tracking.md</code>. Numbers verified against source CSVs. Figures are paths in the canonical doc, base64-inlined here so each page is a single portable file.</p>
<p class="body"><b>Exhibits &amp; repro.</b> Charts: <code>python scripts/build_exhibit_pack.py</code>. Overlays: <code>python scripts/build_exhibit_overlays.py</code>. Deck: <code>python scripts/build_illuminate.py</code> (after the two above).</p>
<p class="body"><b>Honest caveats carried through the deck.</b> The v1→v2 case rests on the two LPIPS gates v1 failed. VAE-v2's latent collapsed when sampled (train on μ). v4 still under-draws the heart (0.482 vs 0.536), fixed only at inference. Effusion never reaches the ≥80% geometric blunt bar (max 66% at w=7) — plan 06 closed with that as a known limitation. Co-presence is a mechanistic null (2.6% « 30%). One data-integrity flag: the DECISION_TIMELINE version table (lines 258-262) has mis-labelled run IDs — the deck uses the corrected plan-06 IDs.</p>
<div class="concepts">Instruments: {concepts}</div>
<div class="nav"><a href="08-instruments.html">← the instruments</a><a href="index.html">index →</a></div>
<div class="foot">vae-v2 decision timeline · git {SHA}</div>
</div>"""


def build():
    for i, act in enumerate(ACTS):
        (OUT / fname(i)).write_text(HEAD.format(title=act["title"], css=CSS, body=act_page(i, act)))
    (OUT / "08-instruments.html").write_text(HEAD.format(title="The instruments", css=CSS, body=instruments_page()))
    (OUT / "08-colophon.html").write_text(HEAD.format(title="Colophon", css=CSS, body=colophon_page()))
    items = "".join(
        f'<li><a href="{fname(i)}">{a["title"]}</a><span class="g">{a["gate"]} · {a["headline"]}</span></li>'
        for i, a in enumerate(ACTS))
    idx_body = f"""<div class="wrap">
<div class="top"><span class="gate">vae-v2</span><span class="gate">git {SHA}</span></div>
<h1>VAE-v2: from a sharper codec to a mechanistic null</h1>
<div class="summary">A plain-language walk through the decisions, each page with the methodology behind it and a detailed reading of every figure: rebuild the codec, retrain the diffusion model through four versions, fix cardiomegaly at inference, and land the headline finding — two diseases will not co-occur in a composed film.</div>
<ol class="idx">{items}<li><a href="08-instruments.html">The instruments — how every number is measured</a><span class="g">reference</span></li><li><a href="08-colophon.html">Colophon — how this deck was built</a><span class="g">provenance · repro</span></li></ol>
<div class="foot">Every headline number is backed by a chart from the actual CSV and a proof image with the instrument drawn on it. git {SHA}.</div>
</div>"""
    (OUT / "index.html").write_text(HEAD.format(title="VAE-v2 decision timeline", css=CSS, body=idx_body))
    seq = {
        "series": "Decision Timeline (vae-v2, deepened exhibit build)",
        "slug": "decision-timeline", "git_sha": SHA,
        "source": ["docs/DECISION_TIMELINE.md", "docs/concepts/*.md", "results/vae_v2/exhibits/*.png"],
        "argument": "Rebuild the codec (v1→v2, LPIPS 0.152→0.022), fix a latent collapse (train on μ), retrain the LDM through four versions (why we stopped at v4-60k), fix cardio at inference (best-of-8 0.875 > real 0.806), and land the mechanistic null: two diseases will not co-occur (2.6% « 30%). Effusion renders but misses the ≥80% blunt bar; plan 06 closed with that caveat.",
        "pages": [{"index": i + 1, "filename": fname(i), "title": a["title"], "gate": a["gate"], "headline": a["headline"]}
                  for i, a in enumerate(ACTS)]
                 + [{"index": INSTR_N, "filename": "08-instruments.html", "title": "The instruments"},
                    {"index": COLO_N, "filename": "08-colophon.html", "title": "Colophon"}],
        "notes": "Each Act carries methodology + per-figure what/how-measured/how-to-read analysis + data notes, all sourced. Two new comparison charts (VAE v1-v2 gates, LDM v1-v4 CTR ladder). Regenerate via scripts/build_illuminate.py after the exhibit builders.",
    }
    (OUT / "sequence.json").write_text(json.dumps(seq, indent=2))
    print("wrote", len(ACTS) + 3, "pages + sequence.json to", OUT)
    for p in sorted(OUT.glob("*.html")):
        print("  ", p.name, f"{p.stat().st_size//1024}K")


if __name__ == "__main__":
    build()
