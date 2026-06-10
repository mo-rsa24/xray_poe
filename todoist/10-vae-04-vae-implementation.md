# VAE 4 · VAE Implementation (testable)

## Reference while you do it
- 📄 Plan: plans/vae/plans/04-vae-implementation.md

## Section context (paste into the Todoist subtask)
**Description:** Implement encoder/decoder/KL bottleneck, the training loop (AdamW, bf16, grad-checkpointing toggle, optional EMA, ckpt save/load), and eval code (encode/decode, SSIM/LPIPS recon metric, ceiling-check script). Unit-tested on tiny tensors.
**Objective:** The codec everything depends on — testable now, runs unchanged on real data later. Recon/ceiling code authored here; only execution deferred.
**Goal:** A `vae` package whose unit tests pass (shape round-trip, finite KL, one-step loss decrease, metric smoke test).
**Verify (whole leaf):** `pytest tests/` → all green; encode/decode round-trip prints (2,4,128,128)→(2,1,512,512).

## Tasks (one at a time)
- [ ] Implement encoder, decoder, KL reparam bottleneck per the decision note
- [ ] Implement training loop: AdamW, bf16, grad-checkpointing, optional EMA, ckpt save/load
- [ ] Implement eval: encode/decode, SSIM+LPIPS recon metric, ceiling-check script (real-data exec deferred)
- [ ] Unit tests: shape round-trip, finite KL, one-step loss decrease, metric smoke test
