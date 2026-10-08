# Reproducibility comparison: traffic-trained vs non-traffic-trained Qwen Decision

The experiment compares the same Qwen3.5-0.8B/Clef decision architecture with an untrained decision head against Traffic-Qwen V1 Dual-Choice fine-tuned on guarded Jinan decisions. Both arms use the same CityFlow protocol, 3,600-second Jinan run, seed-paired Engine seed, metrics, and no guardrail or fallback.

## Paired outcomes

| Seed | Engine seed | Arm | Avg queue | Avg wait (s) | Avg travel (s) | Decisions | Phase switches | Guardrail | Fallbacks |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 3407 | 83 | Qwen Decision, no traffic training | 1751.96 | 999.11 | 1559.67 | 1549 | 0 | 0 | 0 |
| 3407 | 83 | Traffic-Qwen V1 | 156.68 | 37.71 | 283.66 | 1617 | 1565 | 0 | 0 |
| 3408 | 48 | Qwen Decision, no traffic training | 1422.10 | 943.71 | 1135.71 | 1409 | 0 | 0 | 0 |
| 3408 | 48 | Traffic-Qwen V1 | 137.29 | 30.90 | 271.82 | 1744 | 1692 | 0 | 0 |
| 3409 | 2 | Qwen Decision, no traffic training | 1422.10 | 943.71 | 1135.71 | 1080 | 0 | 0 | 0 |
| 3409 | 2 | Traffic-Qwen V1 | 148.36 | 30.95 | 278.96 | 1743 | 1694 | 0 | 0 |

## Mean paired metric reduction

Positive values mean a lower metric for Traffic-Qwen; percentages are paired by seed.

| Metric | Mean reduction | Sample SD across seeds | Per-seed values |
|---|---:|---:|---|
| Average queue | 90.32% | 0.74 pp | 91.06%, 90.35%, 89.57% |
| Average waiting time | 96.56% | 0.29 pp | 96.23%, 96.73%, 96.72% |
| Average travel time | 77.77% | 3.51 pp | 81.81%, 76.07%, 75.44% |

## Interpretation and limits

Across all three paired seeds, the non-traffic-trained decision head selected one phase exclusively and never switched. Traffic-Qwen selected all four phases and switched frequently, with zero fallback and guardrail corrections. The paired reductions show that specialization of the decision model materially improves this Jinan episode under the tested protocol.

This is evidence of a traffic-specialization effect within a fixed model family and decision architecture, not proof of generalization or a causal result across independent scenarios. The training labels come from JevLight with guardrail, only one Jinan episode is represented, and the V1 duration outputs remain collapsed toward 20 seconds. CityFlow's engine seed is selected from 100 possible values, so different requested seeds may collide; each trained/untrained pair nevertheless uses the exact same recorded engine seed.

Training runs used 792 train rows, 130 validation rows, 2 epochs, QLoRA 4-bit, LoRA rank/alpha 16/16, batch 4 × accumulation 8, and seed-specific initialization. The 143-row test split was not used for this comparison.

Quality checks: 15 unit tests run, 13 passed, 2 skipped for unavailable baseline/redistribution-gated data. The JevLight source benchmark and preserved project copy have matching SHA-256 hashes.
