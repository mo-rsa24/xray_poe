# 🧮 AND-Composition Operators (MCMC + SuperDiff)

## Background
The PoE baseline is the existing `cfg_compose` score-sum. This plan adds two alternative AND-operators beside it — eval-only, the trained model untouched. SuperDiff-AND is conditional: build it only if MCMC-corrected alone doesn't settle the operator choice.

## Description
Implement the MCMC-corrected composition operator, then (gated) the SuperDiff-AND operator, as eval samplers beside `cfg_compose`.

## Purpose
Serves Objective 1 and DoD items 1–2 — the operators both scopes will import.

## Goal
At least the MCMC-corrected operator composing cardio∧<2nd-disease> on a saved checkpoint; SuperDiff-AND built or its omission recorded.

> `PY=/home/molefe/micromamba/envs/jaxstack/bin/python`

## Tasks
- [ ] ⚠️ **Implement MCMC-corrected composition**
  - `▶ prompt:` `Beside cfg_compose (src/inference/cfg.py) add an MCMC-corrected AND-operator: a Metropolis/Langevin correction step on the summed PoE score (Du et al. arXiv 2302.11552). Eval-only — no training change. Expose it as a selectable operator in the compose path.`
  - `✓ verify:` composes cardio∧<2nd-disease> and returns a sample on a saved ckpt; step-size/steps are config.
- [ ] ⚠️ **Implement SuperDiff-AND (gated)**
  - `Gate:` build ONLY if MCMC-corrected alone doesn't decide the operator choice (see `02`). Else record "MCMC sufficed — SuperDiff not built".
  - `▶ prompt:` `Add a SuperDiff-AND operator: Itô-density auto-reweighting of the two experts (Skreta et al. arXiv 2412.17762), beside cfg_compose + MCMC. Eval-only.`
  - `✓ verify:` SuperDiff-AND composes + returns a sample, OR `results/vindr_transfer/compose_ops/superdiff_skipped.md` records why it wasn't needed.

## Recommended skill
— custom; core composition code (`src/inference/cfg.py`).

## Engagement Instructions
```bash
$ $PY -c "from src.inference.cfg import compose; compose(op='mcmc', n=1); print('MCMC composes')"
$ ls results/vindr_transfer/compose_ops/ 2>/dev/null   # superdiff sample OR superdiff_skipped.md
# expect: MCMC operator composes a sample; SuperDiff built or its skip recorded
```
