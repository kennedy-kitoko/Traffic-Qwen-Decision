# Limitations

- Training and evaluation labels come from a single guarded Jinan episode.
- The test block is chronological and embargoed, but not from a separate episode or city.
- Labels imitate guarded JevLight behavior; they are not optimized with counterfactual outcomes or RL.
- Class imbalance is severe. Runtime selected only 20- and 40-second durations without guardrail; rare durations are not reliably learned.
- The maximum non-service duration is trace-derived at phase level and reached 900 seconds in the main run; it is not a validated movement-level starvation metric.
- Calibration was evaluated offline; runtime control uses uncalibrated probability distributions.
- Only one seed was run for the full horizon. Stochastic robustness and uncertainty are unknown.
- No merged model checkpoint was produced because base weights were unavailable locally and the offline merge helper tried Hugging Face.
- The local checkpoint cannot be obtained from this source repository; base model and checkpoint weights are excluded.
- Jinan traffic-data redistribution terms were not confirmed, so dataset splits are not bundled.
- Simulation results do not establish real-world safety, reliability, or deployment suitability.
