# BUILD-SPEC — vindr-transfer → Todoist (Option A: refresh in place)

**Account:** Molefe Molefe · momolefe24@gmail.com
**Project:** 🧑‍🎓 PhD · `6XHxp94hx7x3PMMR`
**Section:** 🫁🔁 External-Transfer Cardio Lever (VinDr) · `6gxG3h56P26Xc3QR`
**Label:** paper3 · `2183979036`
**Mapping:** B (one section) · source = current corrected plan files (rich + de-PadChest + gap-fixed)
**Date:** 2026-06-28

## Plan (create / update / retitle)
| Task | Existing id | Action |
|---|---|---|
| 01 · Acquire VinDr-CXR (512) | 6gxG3xw3C2VQFg8R | UPDATE (all-disease presence + Gap-B image-count verify) |
| 02 · VAE round-trip gate (go/no-go) | 6gxG3xxp2rrPcPM2 | UPDATE (de-PadChest) |
| 03 · Preprocess + photometric norm + border filter | 6gxG426f5rRPRqwR | UPDATE + RETITLE (PA/AP→PA-only+monochrome) |
| 04 · Data investigation | 6gxG42G232v67Px2 | UPDATE (de-PadChest + Phase-3 compare) |
| 05 · Stage-1 — class-conditional pretrain on VinDr | 6gxG42HWM2R8Qr92 | UPDATE + RETITLE (PadChest→VinDr + EMA) |
| 06 · Stage-2 — NIH appearance adaptation | 6gxG42VJW4GrFf52 | UPDATE (de-PadChest + Gap-A data prereq) |
| 08 · Eval — Gate-3 yield | 6gxG42g8XVf8jRJ2 | UPDATE (de-PadChest + chosen-operator) |
| 09 · Morbidity criteria + 2nd-disease pick | (none) | CREATE |
| 06.1 · Composition operators | (none) | CREATE subtask, parentId=6gxG42VJW4GrFf52 |
| 09.1 · Opacity instrument | (none) | CREATE subtask, parentId=<09 new id> |

## Result (executed 2026-06-28)
- UPDATED 7: 01/02/03/04/05/06/08 (03 + 05 retitled). de-PadChest + Gap-A/Gap-B applied.
- CREATED 09 · `6gxj9VjWQw3GwcfR`
- CREATED subtask 06.1 · `6gxj9VpRCPgwJfV2` (parent 06 `6gxG42VJW4GrFf52`)
- CREATED subtask 09.1 · `6gxj9X25W86PwQ9R` (parent 09 `6gxj9VjWQw3GwcfR`)

## Resume notes
- Eyeball self-check after task 01 update (read back), then proceed.
- Subtask 09.1 needs the new 09 task id (capture after 09 create).
- Re-run = idempotent (titles now match `NN · …`).
