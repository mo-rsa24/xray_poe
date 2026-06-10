# 🗜️ VAE Implementation (testable)

## Description
Implement the chosen architecture as a tested, modular codebase: encoder, decoder,
KL reparam bottleneck, the training loop (AdamW, bf16 autocast, grad-checkpointing
toggle, optional EMA, checkpoint save/load), and the eval code — encode/decode,
the SSIM/LPIPS reconstruction metric, and the ceiling-check script. Validated by
unit tests on tiny tensors.

## Purpose
The codec everything downstream depends on. "Testable" means it is exercised by
unit tests now and runs **unchanged on real data later** under runpod-execution.
The recon-metric + ceiling-check code is authored here; only its execution on real
data is deferred.

## Goal
A `vae` package whose unit tests pass: the model round-trips shapes, KL is finite,
a training step decreases loss on a fixed batch, and the eval metrics compute.

## Tasks
- [x] ✅ Implement encoder, decoder, and the KL reparam bottleneck per the decision note
- [x] ✅ Implement the training loop: AdamW, bf16 autocast, grad-checkpointing toggle, optional EMA, checkpoint save/load
- [x] ✅ Implement eval code: encode/decode, SSIM+LPIPS recon metric, ceiling-check script (real-data execution deferred to runpod-execution)
- [x] ✅ Write unit tests: shape round-trip, finite KL, one-step loss decrease, metric smoke test

## Recommended skill
custom; no skill fits — implementation + pytest.

## Engagement Instructions
```
$ pytest tests/                         # expect: all green
$ python -c "import torch, vae; m=vae.VAE(); z=m.encode(torch.randn(2,1,512,512)); \
print(z.shape, m.decode(z).shape)"      # expect: (2,4,128,128) (2,1,512,512)
```
