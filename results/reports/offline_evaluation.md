# Traffic-Qwen V1 Dual-Choice offline evaluation

Labels imitate guarded JevLight actions; this is not zero-shot or RL. Temperatures were fitted on validation only. The held-out test is a chronological segment from the same episode, so it does not establish unseen-episode generalization.

Validation rows: 130; test rows: 143.

Phase temperature: 1.078 (validation NLL 0.4274 → 0.4265 test NLL).

Phase test: accuracy 0.790, balanced accuracy 0.769, macro-F1 0.776, top-2 1.000, ECE 0.063, Brier 0.260.

Duration temperature: 1.111. Duration test: exact accuracy 0.972, within 5s 1.000, MAE 0.14s, macro-F1 0.489, ECE 0.013, Brier 0.048, NLL 0.106.

Combined exact outcomes: `{'both_exact': 110, 'duration_only': 29, 'phase_only': 3, 'neither': 1}`. The duration accuracy is inflated by the strongly imbalanced label distribution; macro-F1 and per-class confusion must be read alongside it. See JSON for per-intersection and congestion metrics.
