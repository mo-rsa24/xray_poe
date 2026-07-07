# Composition Operators (subtask)

**Description:** Alternative AND-composition operators beside the PoE score-sum — MCMC-corrected + (conditional) SuperDiff-AND — plus the in-training 3-way grid; eval-only.
**Objective:** pick the best composition operator without changing the model or breaking PoE.
**Goal:** operators implemented, 3-way grid logged, best operator chosen + wired into 08.
**Verify:** operator_choice.md names the pick; PoE score-sum byte-identical to cfg_compose.

## Tasks
- Implement MCMC-corrected composition (Du 2302.11552)
- Implement SuperDiff-AND (gated on MCMC not deciding; Skreta 2412.17762)
- Add the in-training 3-way composition hook
- Choose the best operator + wire it into 08
- Expose the module for both scopes + confirm PoE untouched
