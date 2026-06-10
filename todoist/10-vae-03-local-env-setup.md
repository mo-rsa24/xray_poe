# VAE 3 · Local Env Setup + launch.json

## Reference while you do it
- 📄 Plan: plans/vae/plans/03-local-env-setup.md

## Section context (paste into the Todoist subtask)
**Description:** Reproducible local run/test env — pinned deps, pytest, the `vae` package skeleton, and a `.vscode/launch.json` with overfit / train(noise) / profile / pytest configs.
**Objective:** Make the code runnable and breakpoint-debuggable here before profiling.
**Goal:** Deps pinned, pytest collects, `vae` imports, launch.json launches each entrypoint without import errors.
**Verify (whole leaf):** `python -c "import vae"` → ok; `pytest --collect-only` lists tests; Run-and-Debug ▸ "VAE: overfit" launches.
**▶ Recommended prompt:** `/update-config` ⚠️ (settings/launch wiring) or custom.

## Tasks (one at a time)
- [ ] Pin deps (torch, torchvision, einops, pyyaml, pytest, …)
- [ ] Create `vae` package skeleton (model, train, profile, sanity, eval + tests/)
- [ ] Author `.vscode/launch.json`: VAE: overfit / train (noise) / profile / Pytest
- [ ] Verify each config runs the entrypoint (or stub) without import errors
