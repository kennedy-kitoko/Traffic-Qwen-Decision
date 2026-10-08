# Reproducibility

The code commit, dependency revisions, seeds, configs, and measurements are recorded in artifacts_manifest/experiment_manifest.json and environment.json. scripts/install_jevlight_dependency.py checks out JevLight at its measured commit and applies a checked patch. CityFlow is pinned to its measured source revision. No model cache or base weights are embedded.

To reproduce, install the pinned environment, prepare an authorized dataset trace, install the JevLight code dependency, place authorized Jinan traffic/roadnet files at .deps/JevLight/data/Jinan/3_4/, place the matching local adapter/head at the configured path, run the smoke test, and then run bash scripts/reproduce_jinan.sh no-guardrail. Expected input hashes and commands are documented in the manifests.

The model revision could not be proven from the training log. A local Hugging Face cache ref observed during packaging pointed to 23c69c53358a07516b5827588b3fdb12ae78fd65; this is not proof of the exact training revision.
