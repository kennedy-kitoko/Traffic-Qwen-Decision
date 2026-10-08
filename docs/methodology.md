# Methodology and JevLight protocol

The runtime depends on JevLight commit d92caa655e5a7487a49361626673df809eee9129 plus the patch in third_party/jevlight-runtime.patch. The patch adds TrafficQwen registration and structured-decision handling to the shared pipeline. The measured JevLight worktree also contained uncommitted changes; the patch captures the two runtime files needed by the measured run.

The simulation uses Jinan's 3×4 network, anon_3_4_jinan_real.json, 12 intersections, 3,600 simulated seconds, seed 3407, five-second decision cadence and five-second yellow transition. Official traffic metrics and reward use the same JevLight OneLine.train and CityFlowEnv path as its benchmark. Trace-derived auxiliary quantities are not official metrics.

The action is a pair of separate choices emitted from one state. The model selects phase and duration independently at inference; the pair is applied to CityFlow. The main result disables guardrail and fallback. The secondary result enables the JevLight guardrail and configured fallback; it recorded zero fallbacks and 25 guardrail corrections.

The complete protocol audit, actual run parameters and source commit are retained in results/reports and artifacts_manifest.
