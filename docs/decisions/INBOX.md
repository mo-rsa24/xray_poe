# Decision Timeline — Inbox (staging)

This file is written to **automatically** by the `Stop` hook
(`.claude/hooks/timeline_stub.sh`). Each line is a *stub*: a pointer saying "a decision
artifact changed — a gate may have resolved, go write it up." Each stub is tagged with the
source that fired it: `[verdict]`, `[diagnostic]`, `[data-audit]`, `[wandb-analysis]`
(tracking.md changed), or `[plan-reconcile]` (a MASTER_PLAN changed, e.g. after
`/sync-plan-tree`).

**What the hook cannot see — the deliberate path.** A hook runs a shell command; it cannot
read a Claude conversation, a raw W&B run not yet written to `tracking.md`, or an ad-hoc file.
Those are captured *deliberately*: at the end of a working session, **ask Claude to log any
gates decided in conversation**, and run `/analyze-run` so a W&B run lands in `tracking.md`
(which then trips the hook). The hook guarantees you never forget a file changed; it does not
author the narrative.

**Nothing here is the narrative.** Stubs are raw signals. When you resolve one:

1. Open the changed file, read the number.
2. Append a proper entry to [`../DECISION_TIMELINE.md`](../DECISION_TIMELINE.md) in the fixed
   five-part shape (Question / What we ran / What it said / What we decided / Fired next).
3. Run `/plain-speak --in-place` on that new entry **once** (not continuously).
4. Tick the stub's checkbox here, or delete it.

The hook only **stages**; promoting a stub into the stable record is always a deliberate step.
That separation is what keeps the timeline trustworthy — finished entries never get silently
rewritten.

---

## Stubs

_(none yet — the hook appends here when a decision artifact changes)_

### Sweep 2026-07-06 22:41 — 1 changed artifact(s)
- [x] `[diagnostic]` `results/section4/cardio_enhance/reachability.txt` → promoted into gate [C] (G8c) with reachability exhibit, 2026-07-07

### Sweep 2026-07-07 03:46 — 1 changed artifact(s)
- [x] `[verdict]` `results/section4/augment_probe/verdict.md` → promoted into gate [G9a] (augment-probe NULL), 2026-07-07

### Sweep 2026-07-07 03:51 — 1 changed artifact(s)
- [x] `[plan-reconcile]` `additional_plans/vae-v2/MASTER_PLAN.md` → sync-plan-tree this session: plans 07 + 08 → ✅, DoD-7 ✅; reflected in [G9a]/[G9v] plan-refs, 2026-07-07

### Sweep 2026-07-07 04:41 — 1 changed artifact(s)
- [x] `[verdict]` `results/vae_v2/downstream_eval/verdict.md` → promoted into gate [G9v] (co-presence mechanistic null, overall NO-GO), 2026-07-07

### Sweep 2026-07-07 05:50 — 2 changed artifact(s)
- [x] `[plan-reconcile]` `additional_plans/vae-v2/MASTER_PLAN.md` → sync-plan-tree this session: plans 07 + 08 → ✅, DoD-7 ✅; reflected in [G9a]/[G9v] plan-refs, 2026-07-07
- [x] `[verdict]` `results/vae_v2/downstream_eval/verdict.md` → promoted into gate [G9v] (co-presence mechanistic null, overall NO-GO), 2026-07-07

### Sweep 2026-07-07 05:56 — 1 changed artifact(s)
- [x] `[verdict]` `results/tables/table2_augmentation.md` → pre-registration row (2026-07-07), cited in gate [G9v]

### Sweep 2026-07-07 06:06 — 1 changed artifact(s)
- [x] `[plan-reconcile]` `additional_plans/vae-v2/MASTER_PLAN.md` → sync-plan-tree this session: plans 07 + 08 → ✅, DoD-7 ✅; reflected in [G9a]/[G9v] plan-refs, 2026-07-07

### Sweep 2026-07-07 07:11 — 2 changed artifact(s)
- [ ] `[plan-reconcile]` `additional_plans/vae-v2/MASTER_PLAN.md` changed → log/refresh this gate in DECISION_TIMELINE.md (Question / What we ran / What it said / What we decided / Fired next)
- [ ] `[verdict]` `results/vae_v2/eval_gate/verdict.md` changed → log/refresh this gate in DECISION_TIMELINE.md (Question / What we ran / What it said / What we decided / Fired next)

### Sweep 2026-07-07 07:26 — 1 changed artifact(s)
- [ ] `[plan-reconcile]` `additional_plans/vae-v2/MASTER_PLAN.md` changed → log/refresh this gate in DECISION_TIMELINE.md (Question / What we ran / What it said / What we decided / Fired next)
