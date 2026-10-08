# Dataset creation

The original V1 splits contain 1,065 decisions: 792 training, 130 validation, and 143 chronological test examples. Labels are the phase and duration applied by JevLight after its guardrail. They imitate an existing policy and do not encode optimality or measured future consequences.

The split procedure used contiguous time blocks with a 60-second embargo at boundaries. This reduces adjacent-state leakage, but does not create an independent test episode. The dual-choice converter preserves the original splits and creates two one-hot gold distributions: phase and duration.

The split files are not included because redistribution rights for the underlying traffic trace were not confirmed. Obtain an authorized original state_action.json and use python scripts/prepare_data.py --trace PATH. The generated report should be compared with data/manifests/v1_dual_choice_dataset_report.json and the split hashes. Do not calibrate or tune against test.

Durations are strongly imbalanced: about 72% of source labels are 35 or 40 seconds; no 15-second decisions and only four 30-second decisions were present in the pre-dual V1 dataset. The trained controller's two-duration collapse is consistent with this limitation.
