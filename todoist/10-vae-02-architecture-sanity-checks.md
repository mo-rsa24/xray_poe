# VAE 2 · Architecture Sanity Checks

## Reference while you do it
- 📄 Plan: plans/vae/plans/02-architecture-sanity-checks.md

## Section context (paste into the Todoist subtask)
**Description:** Validate the architecture as a bare nn.Module on dummy tensors — shapes, param count, forward+backward, and the latent-bottleneck contract — before any training stack.
**Objective:** Cheapest gate on the architecture: catch a wrong latent shape, a bypass, or an exploding param count in seconds.
**Goal:** A passing script proving (B,1,512,512)→(B,4,128,128)→(B,1,512,512), finite fwd/bwd, printed params, no enc→dec skip.
**Verify (whole leaf):** `python -m vae.sanity` → prints param count; all shape/grad/bottleneck assertions PASS.

## Tasks (one at a time)
- [ ] Instantiate model; print total + per-module param count
- [ ] Assert encode shape (B,4,128,128) and decode round-trips shape
- [ ] One forward+backward on a dummy batch; assert finite loss + grads
- [ ] Assert decoder consumes only the latent (no enc→dec skip tensors)
