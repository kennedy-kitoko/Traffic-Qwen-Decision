.PHONY: check test prepare smoke train offline cityflow-smoke cityflow-full verify
check:
	python scripts/check_environment.py
test:
	PYTHONPATH="$(CURDIR):$(CURDIR)/.deps/JevLight" python -m unittest discover -s tests -v
prepare:
	python scripts/prepare_data.py --trace "$(TRACE)"
smoke:
	bash scripts/reproduce_smoke_test.sh
train:
	python scripts/train_traffic_qwen.py --config configs/traffic_qwen_v1_dual_choice.json
offline:
	python scripts/evaluate_offline.py --config configs/traffic_qwen_v1_dual_choice.json
cityflow-smoke:
	bash scripts/reproduce_jinan.sh smoke
cityflow-full:
	bash scripts/reproduce_jinan.sh no-guardrail
verify:
	python scripts/verify_artifacts.py
