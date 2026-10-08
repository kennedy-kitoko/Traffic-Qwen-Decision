# Data and reconstruction

No Jinan traffic or V1 split data are redistributed here. The source trace and traffic files are derived from the JevLight/LLMTSCS data tree; specific redistribution permissions were not verified. Obtain the traffic inputs and guarded state_action.json from an authorized source.

The measured V1 dual-choice dataset has 1,065 rows: train 792, validation 130, test 143. Splits were preserved from a chronological episode partition with a 60-second embargo. Labels are JevLight-with-guardrail applied decisions, converted to two one-hot choice labels: phase and duration. The dataset report documents distributions, missing state fields, overlap, and limitations.

Rebuild with:

~~~bash
python scripts/prepare_data.py --trace /path/to/authorized/state_action.json
~~~

The builder needs a 12-intersection trace in the expected JevLight schema. It does not synthesize missing samples. Compare rebuilt hashes and counts with manifests. A hash does not grant redistribution permission.

The sample folder intentionally contains only a note; even a small trace excerpt would remain derived traffic data.
