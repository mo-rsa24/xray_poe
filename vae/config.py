"""The codec contract, as code.

Single source of truth for the architecture decided in
``plans/vae/architecture-decision.md`` (kl-f4-class AutoencoderKL, 512² grayscale
→ 4×128×128 latent at f=4). Every module — model, sanity, train, profile — reads
its shape/channel numbers from here so the contract can never drift between files.

The MONAI ``AutoencoderKL`` kwargs below map 1:1 onto the decision table:

    channels=(128,256,512)              ch=128, ch_mult=[1,2,4]; 2 downsamples → 512→256→128
    num_res_blocks=(2,2,2)              2 resblocks per stage
    latent_channels=4                   z=4  (f=4 ⇒ 4×128×128 latent)
    attention_levels=(False,False,False) attn_resolutions=[]: NO per-stage attention
    with_*_nonlocal_attn=True           mid-block self-attention ONLY
    norm_num_groups=32                  GroupNorm(32) + SiLU
    (raw decoder output — no tanh; data is normalized to [-1,1] instead)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VAEConfig:
    # --- spatial / shape contract -------------------------------------------
    spatial_dims: int = 2
    in_channels: int = 1
    out_channels: int = 1
    input_resolution: int = 512          # cost-gated; latent scales with this at fixed f=4
    downsample_factor: int = 4           # f=4 (informational; implied by len(channels)-1)

    # --- backbone schedule (kl-f4 skeleton) ---------------------------------
    channels: tuple[int, ...] = (128, 256, 512)
    num_res_blocks: tuple[int, ...] = (2, 2, 2)
    latent_channels: int = 4             # z=4. LEVER: z=8 buys recon headroom, costs
    #                                      latent diffusability — OFF by default (§4).
    norm_num_groups: int = 32
    norm_eps: float = 1e-6

    # --- attention: mid-block (lowest res, 128²) only -----------------------
    attention_levels: tuple[bool, ...] = (False, False, False)
    with_encoder_nonlocal_attn: bool = True
    with_decoder_nonlocal_attn: bool = True

    # --- memory / compute toggles -------------------------------------------
    use_checkpoint: bool = False         # gradient checkpointing (engage under VRAM pressure)
    use_convtranspose: bool = False      # decoder upsample style; raw conv output either way

    # --- loss weights (Zhao et al. 2017 mix; KL verbatim from LDM/SD) -------
    ms_ssim_weight: float = 0.84         # α — re-checked on CXR in plan-04 (currently nat-image α)
    l1_weight: float = 0.16              # = 1 - α
    kl_weight: float = 1e-6

    # LEVER (§4, OFF by default): a RadImageNet-LPIPS perceptual *training* term.
    # Never RGB VGG/AlexNet (out-of-domain on grayscale CXR). No adversarial term
    # exists at all — rejected by status, not just by default (§5).
    perceptual_weight: float = 0.0

    @property
    def latent_resolution(self) -> int:
        """Spatial size of the latent grid (input_resolution // f)."""
        return self.input_resolution // self.downsample_factor

    def monai_kwargs(self) -> dict:
        """Exactly the kwargs ``monai.networks.nets.AutoencoderKL`` expects."""
        return dict(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=self.channels,
            num_res_blocks=self.num_res_blocks,
            latent_channels=self.latent_channels,
            attention_levels=self.attention_levels,
            norm_num_groups=self.norm_num_groups,
            norm_eps=self.norm_eps,
            with_encoder_nonlocal_attn=self.with_encoder_nonlocal_attn,
            with_decoder_nonlocal_attn=self.with_decoder_nonlocal_attn,
            use_checkpoint=self.use_checkpoint,
            use_convtranspose=self.use_convtranspose,
        )

    def latent_shape(self, batch: int = 1) -> tuple[int, int, int, int]:
        r = self.latent_resolution
        return (batch, self.latent_channels, r, r)

    def input_shape(self, batch: int = 1) -> tuple[int, int, int, int]:
        r = self.input_resolution
        return (batch, self.in_channels, r, r)


DEFAULT_CONFIG = VAEConfig()
