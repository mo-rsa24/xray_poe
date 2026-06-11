"""Latent Diffusion Model — wraps frozen VAE + UNet + DDPMScheduler."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import wandb
from diffusers import DDPMScheduler
from monai.networks.nets import AutoencoderKL


class LDM(nn.Module):
    """
    Latent Diffusion Model.

    Parameters
    ----------
    vae_ckpt:
        Path to a saved VAE state-dict or checkpoint dict with key "model_state".
    unet:
        Pre-built DiffusionModelUNet instance.  Mutually exclusive with unet_cfg.
    unet_cfg:
        Dict of kwargs forwarded to build_unet().  Ignored when unet is provided.
    scale_factor_path:
        Path to a .pt file containing a scalar tensor (latent scale factor).
    num_classes:
        Number of disease classes (default 3: no-finding, cardiomegaly, effusion).
        Label index num_classes is reserved as the CFG null/unconditional token.
    """

    NULL_TOKEN_OFFSET = 1  # null index = num_classes

    def __init__(
        self,
        vae_ckpt: str | Path,
        unet: nn.Module | None = None,
        unet_cfg: dict[str, Any] | None = None,
        scale_factor_path: str | Path | None = None,
        num_classes: int = 3,
        cfg_dropout_p: float = 0.15,
    ) -> None:
        super().__init__()

        # --- frozen VAE -------------------------------------------------------
        self.vae = self._load_vae(vae_ckpt)
        for p in self.vae.parameters():
            p.requires_grad_(False)
        self.vae.eval()

        # --- UNet -------------------------------------------------------------
        if unet is not None and unet_cfg is not None:
            raise ValueError("Provide unet or unet_cfg, not both.")
        if unet is not None:
            self.unet = unet
        elif unet_cfg is not None:
            from src.models.ldm_unet import build_unet
            self.unet = build_unet(**unet_cfg)
        else:
            raise ValueError("Either unet or unet_cfg must be provided.")

        # --- scale factor (non-trainable buffer) ------------------------------
        if scale_factor_path is not None:
            sf = torch.load(scale_factor_path, map_location="cpu")
            if not isinstance(sf, torch.Tensor):
                sf = torch.tensor(float(sf))
        else:
            sf = torch.tensor(1.0)
        self.register_buffer("scale_factor", sf.float())

        # --- scheduler --------------------------------------------------------
        self.scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
        )

        # --- class + null embedding -------------------------------------------
        self.num_classes = num_classes
        # Index num_classes is the CFG null/unconditional token.
        self.class_embed = nn.Embedding(num_classes + 1, 512)
        self.null_token_idx = num_classes
        self.cfg_dropout_p = cfg_dropout_p

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_vae(vae_ckpt: str | Path) -> AutoencoderKL:
        from vae.config import DEFAULT_CONFIG as vae_cfg
        model = AutoencoderKL(**vae_cfg.monai_kwargs())
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        state = ckpt.get("model_state", ckpt)
        model.load_state_dict(state)
        return model

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """One forward pass; returns scalar MSE loss.

        batch = (x, label)
          x     : (B, 1, 512, 512) normalised to [-1, 1]
          label : (B,)  integer class indices in [0, num_classes)
        """
        x, label = batch
        device = x.device

        # Encode to latent space
        with torch.no_grad():
            z = self.vae.encode(x).latent_dist.sample() * self.scale_factor  # (B,4,128,128)

        B = z.shape[0]

        # Sample timesteps and noise
        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (B,), device=device)
        eps = torch.randn_like(z)

        # Corrupt latents
        z_t = self.scheduler.add_noise(z, eps, t)

        # CFG dropout: replace label with null token
        drop_mask = torch.bernoulli(
            torch.full((B,), self.cfg_dropout_p, device=device)
        ).bool()
        label_input = label.clone()
        label_input[drop_mask] = self.null_token_idx

        # Forward through UNet
        context = self.class_embed(label_input).unsqueeze(1)  # (B, 1, 512)
        eps_pred = self.unet(z_t, t, context)

        return torch.nn.functional.mse_loss(eps_pred, eps)

    # ------------------------------------------------------------------
    # sampling
    # ------------------------------------------------------------------

    def sample(
        self,
        label_idx: int,
        n: int,
        w: float,
        steps: int = 50,
        anchor: str = "null",
    ) -> torch.Tensor:
        """DDIM CFG sampling.

        Returns raw latents of shape (n, 4, 128, 128).
        Decoded in the caller.
        """
        from src.inference.cfg import cfg_single

        device = self.scale_factor.device
        noise = torch.randn(n, 4, 128, 128, device=device)
        return cfg_single(self, noise, label_idx, w, steps=steps, anchor=anchor)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str | Path, step: int | None = None) -> None:
        """Save model state and metadata; log as W&B Artifact."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        unet_cfg: dict[str, Any] = {}
        if hasattr(self.unet, "config"):
            # diffusers-style config
            unet_cfg = dict(self.unet.config)
        elif hasattr(self.unet, "_cfg"):
            unet_cfg = self.unet._cfg

        torch.save(
            {
                "model_state": self.unet.state_dict(),
                "class_embed_state": self.class_embed.state_dict(),
                "unet_cfg": unet_cfg,
                "scale_factor": self.scale_factor,
                "num_classes": self.num_classes,
                "cfg_dropout_p": self.cfg_dropout_p,
            },
            path,
        )

        if wandb.run is not None:
            alias = f"step{step}" if step is not None else "latest"
            artifact = wandb.Artifact("ldm-ckpt", type="model")
            artifact.add_file(str(path))
            wandb.log_artifact(artifact, aliases=[alias, "latest"])

    @classmethod
    def from_artifact(
        cls,
        artifact_uri: str,
        vae_ckpt: str | Path,
        device: str | torch.device = "cpu",
    ) -> "LDM":
        """Load from a W&B artifact URI, e.g. 'entity/project/ldm-ckpt:step10000'."""
        api = wandb.Api()
        artifact = api.use_artifact(artifact_uri)
        download_dir = Path(artifact.download())
        ckpt_files = list(download_dir.glob("*.pt")) + list(download_dir.glob("*.pth"))
        if not ckpt_files:
            raise FileNotFoundError(f"No .pt/.pth file found in {download_dir}")
        ckpt_path = ckpt_files[0]

        ckpt = torch.load(ckpt_path, map_location=device)

        from src.models.ldm_unet import build_unet
        unet = build_unet(**(ckpt.get("unet_cfg") or {}))
        unet.load_state_dict(ckpt["model_state"])

        instance = cls(
            vae_ckpt=vae_ckpt,
            unet=unet,
            num_classes=ckpt["num_classes"],
            cfg_dropout_p=ckpt.get("cfg_dropout_p", 0.15),
        )
        instance.class_embed.load_state_dict(ckpt["class_embed_state"])
        instance.scale_factor.copy_(ckpt["scale_factor"].to(device))
        return instance.to(device)
