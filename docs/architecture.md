# Architecture

Traffic-Qwen V1 uses unsloth/Qwen3.5-0.8B as a decision backbone, loaded in 4-bit and trained with rank-16 QLoRA. A Clef typed-decision head receives a shared structured traffic state and returns two choice distributions in one inference: four signal phases and six green durations. The selected phase and duration are decoded directly; no text is generated during control.

The CityFlow path uses JevLight OneLine, CityFlowEnv, and the TrafficQwen agent integration. Signals advance at five-second simulator ticks. A five-second yellow interval occurs on phase changes. The optional JevLight-derived guardrail can modify phase or duration; the principal reported run disabled it and disabled fallback.

The input state is derived from JevLight structured movement detail. Fields absent from the original trace are not inferred or fabricated. The dataset builder marks unavailable downstream occupancy, exact vehicle counts, departures, and per-movement service age as unavailable.
