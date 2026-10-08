# Results

See results/reports/final_report.md for details and results/results.json for machine-readable metrics. results/comparison.csv includes JevLight baselines and both Traffic-Qwen runs. results/benchmark_jinan_with_traffic_qwen_v1.json extends the original benchmark schema without overwriting the original.

Without guardrail Traffic-Qwen scored queue/wait/travel values 189.96 / 46.71 s / 303.12 s. Guarded Laya scored 191.39 / 47.36 s / 303.87 s. Traffic-Qwen without guardrail reduced those metrics by approximately 0.75%, 1.37%, and 0.25%, respectively, in this one run. Against MaxPressure it had a lower average queue and travel time but a substantially higher average waiting time. No multi-seed significance claim is made.
