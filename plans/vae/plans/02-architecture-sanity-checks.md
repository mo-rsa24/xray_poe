# 🔬 Architecture Sanity Checks

## Description
Before building the full training stack, validate the chosen architecture as a
bare `nn.Module` on dummy tensors — shapes, parameter count, a forward+backward
pass, and the latent-bottleneck contract.

## Purpose
The cheapest possible gate on the architecture: catch a wrong latent shape, a
bypass path, or an exploding parameter count in seconds, before any training code,
data, or profiling.

## Goal
A passing sanity script proving `(B,1,512,512)→encode→(B,4,128,128)→decode→
(B,1,512,512)`, a finite forward+backward, a printed parameter count, and the
absence of any encoder→decoder skip.

## Tasks
- [x] ✅ Instantiate the model; print total + per-module parameter count
- [x] ✅ Assert `encode(x).shape == (B,4,128,128)` for `x=(B,1,512,512)`; assert `decode` round-trips the shape
- [x] ✅ Run one forward + backward on a dummy batch; assert finite loss + finite grads
- [x] ✅ Assert the decoder consumes only the latent (no encoder→decoder skip tensors)

## Recommended skill
custom; no skill fits — a small pytest/asserts script.

## Engagement Instructions
```
$ python -m vae.sanity
# expect: prints param count; all shape + grad + bottleneck assertions PASS
```
