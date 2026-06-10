# 🧰 Local Run/Test Environment + launch.json

## Description
Stand up a reproducible local environment to run, debug, and test the VAE here —
pinned dependencies, a `pytest` setup, the `vae` package skeleton, and a
`.vscode/launch.json` with configs for overfit / train(noise) / profile / pytest.

## Purpose
Make the code runnable and breakpoint-debuggable locally before profiling, so
iteration is fast and a fresh session can reproduce it. The launch.json lives in
this repo (`.vscode/launch.json`).

## Goal
Pinned deps installed, `pytest` collects, the `vae` package imports, and
`.vscode/launch.json` launches each entrypoint (even as a stub) without import errors.

## Tasks
- [x] ✅ Pin dependencies (torch, torchvision, einops, pyyaml, pytest, …) into requirements.txt / env
- [x] ✅ Create the `vae` package skeleton (model, train, profile, sanity, eval modules + tests/)
- [x] ✅ Author `.vscode/launch.json` with configs: sanity / overfit / train(noise) / profile / budget / `Pytest`  ✓ verified (.vscode/launch.json)
- [x] ✅ Verify each launch config runs the entrypoint (or stub) without import errors

## Recommended skill
`/update-config` ⚠️ (for settings/launch wiring) or custom — author launch.json + requirements.

## Engagement Instructions
```
$ python -c "import vae; print('ok')"   # expect: ok
$ pytest --collect-only                 # expect: tests listed, no import errors
# VS Code: Run and Debug ▸ "VAE: overfit" launches the entrypoint
```
