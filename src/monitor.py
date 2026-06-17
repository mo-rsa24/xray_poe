"""MonitorBatch — fixed-seed latent noise for consistent visual logging.

Holds 4 seeds × 3 classes = 12 noise tensors, frozen at construction.
The same seeds produce the same images at every checkpoint so training
progress is directly comparable across steps in W&B.

Fixed seeds (hardcoded, never changed after construction):
    [42, 137, 256, 512]  — one per row in the output grid.

Grid layout (4 rows × 3 columns):
    col 0 = no_finding (label 0)
    col 1 = cardiomegaly (label 1)
    col 2 = effusion (label 2)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from PIL import Image as PILImage

MONITOR_SEEDS = [42, 137, 256, 512]
MONITOR_CLASSES = [0, 1, 2]                  # no_finding, cardiomegaly, effusion
CLASS_NAMES = ["no_finding", "cardiomegaly", "effusion"]
LATENT_SHAPE = (4, 128, 128)                 # (C, H, W) per sample


class MonitorBatch:
    """Fixed noise tensors for reproducible recon-grid logging.

    Parameters
    ----------
    device:
        Device to store the noise tensors on.  Must match the model device.
    latent_shape:
        (C, H, W) of one latent sample.  Defaults to (4, 128, 128).
    seeds:
        Four integer seeds, one per grid row.  Hard-coded; never mutate.
    """

    def __init__(
        self,
        device: torch.device | str,
        latent_shape: tuple[int, int, int] = LATENT_SHAPE,
        seeds: list[int] = MONITOR_SEEDS,
    ) -> None:
        self.device = torch.device(device)
        self.latent_shape = latent_shape
        self.seeds = seeds
        self.n_seeds = len(seeds)
        self.n_classes = len(MONITOR_CLASSES)

        # noise[seed_idx, class_idx] → (1, C, H, W)
        # Pre-allocate as a flat list; access via _idx(seed_i, cls_i)
        C, H, W = latent_shape
        self._noise: list[torch.Tensor] = []
        for seed in seeds:
            for cls_idx in MONITOR_CLASSES:
                g = torch.Generator(device=self.device).manual_seed(seed + cls_idx * 1000)
                z = torch.randn(1, C, H, W, generator=g, device=self.device)
                self._noise.append(z)

    def _idx(self, seed_i: int, cls_i: int) -> int:
        return seed_i * self.n_classes + cls_i

    def noise_for(self, seed_i: int, cls_i: int) -> torch.Tensor:
        """Return the frozen (1, C, H, W) noise for a given seed/class slot."""
        return self._noise[self._idx(seed_i, cls_i)]

    # ------------------------------------------------------------------
    # Decode grid
    # ------------------------------------------------------------------

    @torch.no_grad()
    def decode_grid(
        self,
        unet: nn.Module,
        ddim_scheduler,
        vae,
        cfg_weight: float = 1.0,
        null_token_idx: int = 3,
        steps: int = 50,
    ) -> "tuple[PILImage, list]":
        """Run DDIM denoising for all 12 slots; return labeled 4×3 PIL grid and uint8 cells.

        Returns
        -------
        grid_pil : PIL Image
            Labeled overview grid (dark header + seed labels baked in).
        cells : list[list[np.ndarray]]
            cells[seed_i][cls_i] — uint8 (H, W) arrays for per-class W&B panels.
        """
        import numpy as np
        from PIL import Image, ImageDraw, ImageFont

        ddim_scheduler.set_timesteps(steps)
        was_training = unet.training
        unet.eval()

        cell_images: list[list] = []  # [seed_i][cls_i] float32 [0,1]

        for seed_i in range(self.n_seeds):
            row_imgs = []
            for cls_i, cls_label in enumerate(MONITOR_CLASSES):
                z = self.noise_for(seed_i, cls_i).clone()
                label_cond = torch.tensor([cls_label], device=self.device)
                label_null = torch.tensor([null_token_idx], device=self.device)

                for t in ddim_scheduler.timesteps:
                    t_batch = torch.tensor([t], device=self.device)
                    eps_cond = unet(z, t_batch, label_cond)
                    if cfg_weight != 1.0:
                        eps_uncond = unet(z, t_batch, label_null)
                        eps = eps_uncond + cfg_weight * (eps_cond - eps_uncond)
                    else:
                        eps = eps_cond
                    z = ddim_scheduler.step(eps, t, z).prev_sample

                decoded = vae.decode(z)
                if hasattr(decoded, "sample"):
                    decoded = decoded.sample
                img = decoded[0, 0].float().clamp(-1, 1).cpu().numpy()
                img = (img + 1) / 2  # [0, 1]
                row_imgs.append(img)
            cell_images.append(row_imgs)

        if was_training:
            unet.train()

        # --- labeled canvas ---
        H_px, W_px = cell_images[0][0].shape
        LABEL_W = 120
        HEADER_H = 30
        COL_HEADERS = ["NO FINDING", "CARDIOMEGALY", "EFFUSION"]

        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16
            )
        except Exception:
            font = ImageFont.load_default()

        canvas_w = LABEL_W + W_px * self.n_classes
        canvas_h = HEADER_H + H_px * self.n_seeds
        labeled = Image.new("L", (canvas_w, canvas_h), color=30)
        draw = ImageDraw.Draw(labeled)

        # Header bar: column labels
        draw.rectangle([(0, 0), (canvas_w, HEADER_H - 1)], fill=20)
        for cls_i, header in enumerate(COL_HEADERS):
            x = LABEL_W + cls_i * W_px + W_px // 2 - len(header) * 4
            draw.text((x, 7), header, fill=215, font=font)

        # Rows: seed label + image cells
        for seed_i, row_imgs in enumerate(cell_images):
            y_top = HEADER_H + seed_i * H_px
            draw.rectangle([(0, y_top), (LABEL_W - 1, y_top + H_px - 1)], fill=20)
            draw.text((4, y_top + H_px // 2 - 8), f"seed {self.seeds[seed_i]}", fill=180, font=font)
            for cls_i, img_f in enumerate(row_imgs):
                cell = Image.fromarray((img_f * 255).clip(0, 255).astype(np.uint8), "L")
                labeled.paste(cell, (LABEL_W + cls_i * W_px, y_top))

        cells_uint8 = [
            [(img * 255).clip(0, 255).astype(np.uint8) for img in row]
            for row in cell_images
        ]
        return labeled, cells_uint8
