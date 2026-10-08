# Training

Measured configuration: Qwen3.5-0.8B; 4-bit QLoRA; rank 16, alpha 16, dropout 0; Clef head; max sequence length 2048; batch 4 with accumulation 8 (effective 32); two epochs; learning rate 2e-4, head learning rate 1e-4; cosine schedule; seed 3407; BF16; reporting disabled. Transformers 5.17.0 exposes warmup_steps; the trainer derived three warmup steps from the configured 0.05 ratio. Full measured runtime was about 637 seconds with 2.08 GiB peak VRAM.

Run a 10-step smoke check first with bash scripts/reproduce_smoke_test.sh. Then run python scripts/train_traffic_qwen.py --config configs/traffic_qwen_v1_dual_choice.json. Training refuses to overwrite a non-empty output checkpoint. Evaluate only after the training report says passed.

The recorded adapter and Clef head live locally under checkpoints/traffic-qwen-v1-dual-choice; all weight files are ignored by Git. The post-hoc temperature fit uses validation only and is offline analysis; the simulation agent does not apply those temperatures.
