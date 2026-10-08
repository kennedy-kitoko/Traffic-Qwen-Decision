# Evaluation

Offline evaluation predicts the two decisions on the chronological held-out segment. Temperatures are fit on validation only; the report gives pre/post-temperature metrics, confusion matrices, Brier scores, log loss, ECE, and per-intersection/congestion summaries. The test segment is from the training episode and is not evidence of generalization.

CityFlow runs use the same Jinan input, seed, horizon, and JevLight metric implementation for Traffic-Qwen. Main condition: no guardrail, no fallback. Secondary: guardrail and fallback enabled. The runner records decision distributions, proposals, applied actions, confidence, latency, interventions, fallback status, intersection, and simulation time.

Auxiliary maximum_non_service_time_seconds is inferred from phase command timestamps. It is not an official movement-level starvation metric. The plotted waiting series is a trace-derived proxy. Check raw measured metrics before drawing conclusions from these auxiliary series.
