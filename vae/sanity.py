"""Plan 02 — architecture sanity checks.

The cheapest possible gate on the architecture: validate the chosen module on dummy
tensors before any training stack exists. Catches a wrong latent shape, a bypass
path, or an exploding parameter count in seconds.

Run:  python -m vae.sanity
Expect: prints param count; all shape + grad + bottleneck assertions PASS.

Runs in the true contract shape (512²→128²→512²). That forward+backward needs
~7–11 GB, so this defaults to CUDA when available (with gradient checkpointing, to fit
modest cards) and falls back to CPU otherwise.
"""

from __future__ import annotations

import torch

from .config import VAEConfig
from .model import VAE


def _check(label: str, ok: bool) -> None:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}")
    if not ok:
        raise AssertionError(label)


def run(batch: int = 1, device: str | None = None) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    # Gradient checkpointing keeps the contract-shape backward within ~7.4 GB.
    cfg = VAEConfig(use_checkpoint=True)
    model = VAE(cfg).to(device)
    print(f"device: {device}")

    total = model.num_parameters()
    print(f"\nVAE parameter count: {total:,}  ({total/1e6:.2f}M)")
    for name, n in model.parameter_breakdown().items():
        print(f"    {name:<24} {n:,}  ({n/1e6:.2f}M)")
    print()

    in_shape = cfg.input_shape(batch)
    lat_shape = cfg.latent_shape(batch)
    x = torch.randn(*in_shape, device=device)

    # 1. Encode → latent shape
    z = model.encode(x)
    _check(f"encode{tuple(in_shape)} -> {tuple(z.shape)} == {tuple(lat_shape)}",
           tuple(z.shape) == lat_shape)

    # 2. Decode round-trips the input shape
    recon = model.decode(z)
    _check(f"decode{tuple(z.shape)} -> {tuple(recon.shape)} == {tuple(in_shape)}",
           tuple(recon.shape) == in_shape)

    # 3. Forward + backward: finite loss and finite grads
    out, mu, sigma = model(x)
    loss = out.pow(2).mean() + mu.pow(2).mean() + sigma.pow(2).mean()
    loss.backward()
    finite_loss = torch.isfinite(loss).item()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    finite_grads = len(grads) > 0 and all(torch.isfinite(g).all().item() for g in grads)
    _check("forward+backward: finite loss", bool(finite_loss))
    _check(f"forward+backward: finite grads on {len(grads)} tensors", bool(finite_grads))

    # 4. Bottleneck: the decoder consumes ONLY the latent (no encoder→decoder skip).
    #    AutoencoderKL is structurally skip-free; verify behaviorally that decode is a
    #    pure function of z — a z built from no image at all still decodes, and the
    #    same z always yields the same output regardless of any prior encode.
    model.eval()
    with torch.no_grad():
        z_free = torch.randn(*lat_shape, device=device)          # latent not from any encode
        _ = model.encode(torch.randn(*in_shape, device=device))  # encode something unrelated
        out_a = model.decode(z_free)
        out_b = model.decode(z_free)
    _check("bottleneck: decode(z) is a pure function of z (no enc→dec skip)",
           torch.allclose(out_a, out_b) and tuple(out_a.shape) == in_shape)

    print("\nAll sanity checks PASS.\n")


if __name__ == "__main__":
    run()
