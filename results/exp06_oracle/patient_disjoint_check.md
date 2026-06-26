# Exp 6 — patient-disjointness check (Oracle/Starved training pools)

Reserved test patients (Exp 1 `test_patient_ids.txt`): **317**

| pool | films before | films dropped | patients dropped | films after | shared after |
|---|---|---|---|---|---|
| both-disease (oracle pool) | 1063 | 506 | 317 | 557 | 0 |
| normal (base) | (missing dir) | — | — | — | — |
| cardio_only (base) | 1713 | 318 (drop at load) | 95 | 1395 | 0 (after load-time drop) |
| effusion_only (base) | 12254 | 583 (drop at load) | 116 | 11671 | 0 (after load-time drop) |

**Guarantee.** The Oracle/Starved both-disease pool (`oracle_both_pool.csv`) is written test-patient-free (assertion enforced above). The single-disease/normal base dirs are read-only NIH group folders; their reserved-patient films are dropped at load time by the detector data loader, which re-asserts an empty intersection before the first training step (fail-fast).

Verification: zero shared Patient IDs between every training pool and `data/splits/test_patient_ids.txt`.
