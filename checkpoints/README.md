# Local checkpoints

traffic-qwen-v1-dual-choice contains the measured LoRA adapter, joint Clef head, tokenizer/config metadata, and training metadata. It is copied locally for convenient reproduction, but all weight/binary files are ignored by Git. Full Qwen3.5-0.8B base weights are not included. A merged checkpoint does not exist: export tried to resolve the base model through Hugging Face, but WSL had no network and no complete local base weights.

Base model identifier: unsloth/Qwen3.5-0.8B. The training log does not lock a model revision. A cache ref observed during packaging was 23c69c53358a07516b5827588b3fdb12ae78fd65, but this is not proof of the training revision. Loading requires the matching base model and Unsloth FastDecisionModel support.

Verify with python scripts/verify_artifacts.py --include-local-checkpoint. Do not force-add checkpoint files. A future model-hosting release requires separate license/redistribution review and published checksums.
