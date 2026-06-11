"""UNet backbone for the single-disease LDM.

Architecture
------------
Input latents:  (B, 4, 128, 128)
4 resolution levels:  128 → 64 → 32 → 16  (3 downsamples)
Channels per level:   128, 128, 128, 128   (all multiples of norm_num_groups=32)
Attention:            False, False, True, True  → 32×32 and 16×16 spatial
                      (map level indices to spatial: level 0=128, 1=64, 2=32, 3=16)
Cross-attention dim:  512  (class embedding output dimension)
Res-blocks per level: 2

Null-token decision
-------------------
The CFG unconditional path uses label index `num_classes` (= 3 for the default
three-class setup). This is a *learned* null embedding, not a zero vector —
keeping it learned lets the model express "truly unconditional" rather than
clamping it to the zero subspace. Index 3 is reserved; indices 0/1/2 map to
no_finding / cardiomegaly / effusion. This decision is documented here and must
match LDM.__init__ in src/models/ldm.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet


# Default architecture parameters — match configs/ldm_full.yaml model_channels=128
_DEFAULT_CHANNELS = (128, 128, 128, 128)
_DEFAULT_ATTENTION = (False, False, True, True)
_DEFAULT_NUM_HEAD_CHANNELS = 32   # 128 // 4 heads


class LDMUNet(nn.Module):
    """DiffusionModelUNet + class/null embedding.

    The embedding lives here (not inside MONAI's UNet) so that label index
    `num_classes` can serve as the learned CFG null token without touching
    MONAI's internal class-embedding path.

    Parameters
    ----------
    num_classes:
        Number of disease classes.  Index num_classes is the null token.
    model_channels:
        Base channel width.  All four levels use this width (no channel
        multiplier), keeping peak VRAM under 6 GB on an A4000.
    cross_attention_dim:
        Embedding dimension — must match the Embedding output size (512).
    """

    def __init__(
        self,
        num_classes: int = 3,
        model_channels: int = 128,
        cross_attention_dim: int = 512,
        num_res_blocks: int = 2,
        norm_num_groups: int = 32,
    ) -> None:
        super().__init__()

        channels = tuple(model_channels for _ in range(4))
        num_head_ch = max(1, model_channels // 4)   # 32 for model_channels=128

        self.unet = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=4,
            out_channels=4,
            num_res_blocks=num_res_blocks,
            channels=channels,
            attention_levels=(False, False, True, True),
            norm_num_groups=norm_num_groups,
            num_head_channels=num_head_ch,
            with_conditioning=True,
            cross_attention_dim=cross_attention_dim,
            transformer_num_layers=1,
        )

        # label index `num_classes` is the learned CFG null token
        self.class_embed = nn.Embedding(num_classes + 1, cross_attention_dim)
        self.null_token_idx: int = num_classes

        # store for checkpoint round-trip
        self._cfg = dict(
            num_classes=num_classes,
            model_channels=model_channels,
            cross_attention_dim=cross_attention_dim,
            num_res_blocks=num_res_blocks,
            norm_num_groups=norm_num_groups,
        )

    def forward(
        self,
        z_t: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Predict noise ε given noisy latent, timesteps, and class labels.

        Parameters
        ----------
        z_t       : (B, 4, 128, 128)
        timesteps : (B,)  integer timesteps in [0, T)
        labels    : (B,)  integer labels; use null_token_idx for unconditional

        Returns
        -------
        (B, 4, 128, 128)  predicted noise
        """
        context = self.class_embed(labels).unsqueeze(1)   # (B, 1, 512)
        return self.unet(z_t, timesteps, context=context)


def build_unet(
    num_classes: int = 3,
    model_channels: int = 128,
    cross_attention_dim: int = 512,
    num_res_blocks: int = 2,
    norm_num_groups: int = 32,
    **_ignored,
) -> LDMUNet:
    """Factory used by LDM.__init__ and from_artifact."""
    return LDMUNet(
        num_classes=num_classes,
        model_channels=model_channels,
        cross_attention_dim=cross_attention_dim,
        num_res_blocks=num_res_blocks,
        norm_num_groups=norm_num_groups,
    )
