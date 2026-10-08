# Traffic-Qwen-V1-Dual-Choice final report

Model: `unsloth/Qwen3.5-0.8B`, QLoRA 4-bit rank 16, Clef head, two `choice` tasks (phase and duration) from one shared state. This is a JevLight imitation baseline trained on guarded Jinan decisions, not zero-shot, RL, self-evolving, or a generalization claim.

## Training

Train/validation/test rows: 792/130/143, preserving the supplied chronological splits. The splits share the same single Jinan episode; exact state keys do not overlap and an embargo was applied, but test is not an unseen episode. Training time: 637.0750449900006; peak training GPU memory GiB: 2.0765480995178223. Training metrics and versions are in `training/metrics.json`.

## Offline test

Calibration was fitted only on validation. Phase and duration metrics, confusion matrices, calibration and per-intersection/congestion results are in `offline_evaluation.json`; see `offline_evaluation.md` for headline scores. No test data was used for calibration.

## Interpretation of the CityFlow runs

The no-guardrail run used all four phases (440/450/211/198 decisions), switched phase 1,262 times, and retained the current phase on only 1.94% of decisions. This rules out the phase-locking failure seen with generic Laya in this one Jinan run. Duration behavior remains severely collapsed: 739 decisions selected 20 seconds and 560 selected 40 seconds; 15, 25, 30 and 35 seconds were never selected. This follows the imbalanced imitation labels and makes the apparent 97.2% offline duration accuracy misleading for minority durations.

Without guardrail, queue/wait/travel were 189.96 / 46.71 s / 303.12 s. Against guarded Laya (191.39 / 47.36 s / 303.87 s), the reductions were approximately 0.75% / 1.37% / 0.25%. With guardrail enabled, the corresponding values were 180.71 / 47.08 s / 297.60 s, with 25 corrections and zero fallbacks. The maximum non-service metric is a trace-derived proxy (900 s without guardrail; 855 s guarded), not an official JevLight score, and signals a residual starvation concern.

Classification: **partial transfer**. The model removes phase lock and closely matches the guarded imitation reference without a guardrail, but it has a severe duration-collapse and a high maximum non-service proxy. Only one seed and one episode were evaluated; the chronological test split is from that same episode, so these runs do not establish generalization.

The model export as a merged 16-bit checkpoint could not be completed offline: the installed Unsloth merge helper attempted to resolve the base model `unsloth/Qwen3.5-0.8B` through Hugging Face, and this WSL session has no network. The trained LoRA adapter plus Clef head remains saved and reloadable at `models/traffic-qwen-v1-dual-choice/`; no merged checkpoint is claimed.

## CityFlow benchmark comparison

The first table uses the official JevLight metrics (`results/benchmark_jinan.json` definitions). Values for Traffic-Qwen come from the same `OneLine.train` and `CityFlowEnv` metric calculations.

| Controller | Avg queue | Avg waiting time | Avg travel time |
|---|---:|---:|---:|
| Random | 630.59 | 35.55 | 594.16 |
| Fixedtime-30 | 431.37 | 50.7 | 451.45 |
| MaxPressure | 199.68 | 30.76 | 317.51 |
| PressLight | 697.76 | 41.33 | 630.93 |
| MPLight | 391.74 | 26.49 | 439.62 |
| CoLight | 866.49 | 51.12 | 750.91 |
| DynamicLight | 609.8 | 92.07 | 568.3 |
| Qwen2.5-7B-Instruct | 189.14 | 25.34 | 312.47 |
| Jev-API-revised-prompt | 206.2 | 47.76 | 312.75 |
| Laya-typed-decisions-local | 191.39305555555555 | 47.363854997235485 | 303.87307386814933 |
| Laya-typed-decisions-local-no-guardrail | 1823.4388888888889 | 943.759505192812 | 1395.3608748481167 |
| Traffic-Qwen-V1-Dual-Choice | 189.9638888888889 | 46.713642704376134 | 303.12120730738684 |
| Traffic-Qwen-V1-Dual-Choice-guardrail | 180.7111111111111 | 47.0805164282584 | 297.60381254964256 |

| Controller | Decisions | Fallbacks | Guardrail corrections | Runtime | Latency p50 (ms) | GPU memory (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| Random | None | None | None | None | None | None |
| Fixedtime-30 | None | None | None | None | None | None |
| MaxPressure | None | None | None | None | None | None |
| PressLight | None | None | None | None | None | None |
| MPLight | None | None | None | None | None | None |
| CoLight | None | None | None | None | None | None |
| DynamicLight | None | None | None | None | None | None |
| Qwen2.5-7B-Instruct | None | None | None | None | None | None |
| Jev-API-revised-prompt | 978 | 95 | 4 | None | None | None |
| Laya-typed-decisions-local | 1133 | 0 | 944 | 135.12187504768372 | None | None |
| Laya-typed-decisions-local-no-guardrail | None | 0 | 0 | None | None | None |
| Traffic-Qwen-V1-Dual-Choice | 1299 | 0 | 0 | 213.85281688200484 | 139.36392499454087 | 1.2923765182495117 |
| Traffic-Qwen-V1-Dual-Choice-guardrail | 1285 | 0 | 25 | 210.00462574500125 | 140.6811970009585 | 1.2923765182495117 |

## Relative metric changes

Positive values indicate lower (better) metrics than the named baseline.

```json
{
  "Traffic-Qwen-V1-Dual-Choice vs MaxPressure": {
    "Avg queue": 4.8658409009971475,
    "Avg waiting time": -51.864898258700045,
    "Avg travel time": 4.5317604776583895
  },
  "Traffic-Qwen-V1-Dual-Choice vs Qwen2.5-7B-Instruct": {
    "Avg queue": -0.43559738230354045,
    "Avg waiting time": -84.34744555791687,
    "Avg travel time": 2.9919008841210957
  },
  "Traffic-Qwen-V1-Dual-Choice vs Jev-API-revised-prompt": {
    "Avg queue": 7.873962711499072,
    "Avg waiting time": 2.190865359346448,
    "Avg travel time": 3.078750661107325
  },
  "Traffic-Qwen-V1-Dual-Choice vs Laya-typed-decisions-local": {
    "Avg queue": 0.7467181411144787,
    "Avg waiting time": 1.372802726672696,
    "Avg travel time": 0.24742783267751098
  },
  "Traffic-Qwen-V1-Dual-Choice vs Laya-typed-decisions-local-no-guardrail": {
    "Avg queue": 89.5821082874544,
    "Avg waiting time": 95.05025989700285,
    "Avg travel time": 78.2765008843765
  },
  "Traffic-Qwen-V1-Dual-Choice-guardrail vs MaxPressure": {
    "Avg queue": 9.499643874643878,
    "Avg waiting time": -53.05759567054095,
    "Avg travel time": 6.269467875140133
  },
  "Traffic-Qwen-V1-Dual-Choice-guardrail vs Qwen2.5-7B-Instruct": {
    "Avg queue": 4.456428512683133,
    "Avg waiting time": -85.79525030883346,
    "Avg travel time": 4.7576367172392455
  },
  "Traffic-Qwen-V1-Dual-Choice-guardrail vs Jev-API-revised-prompt": {
    "Avg queue": 12.361245823903435,
    "Avg waiting time": 1.4227042959413718,
    "Avg travel time": 4.842905659586713
  },
  "Traffic-Qwen-V1-Dual-Choice-guardrail vs Laya-typed-decisions-local": {
    "Avg queue": 5.58115570778575,
    "Avg waiting time": 0.5982168659912168,
    "Avg travel time": 2.0631184062155548
  },
  "Traffic-Qwen-V1-Dual-Choice-guardrail vs Laya-typed-decisions-local-no-guardrail": {
    "Avg queue": 90.0895438716223,
    "Avg waiting time": 95.01138625155995,
    "Avg travel time": 78.67191076415725
  }
}
```

The state trace records proposed and applied actions, probabilities, guardrail/fallback status, and latency. `comparison.csv` and `figures/` contain machine-readable and visual summaries. The original benchmark JSON is not overwritten.
