# Contributing

Keep experiment code, measured outputs, and interpretation separate. Do not overwrite baseline JSON or traces. Do not add model weights, data splits, caches, credentials, or private traffic traces to Git. Any proposed result must include its command, seed, config, environment, trace hash, and whether calibration, guardrail, or fallback affected the run.

Run python -m unittest discover -s tests -v and python scripts/verify_artifacts.py before submitting code. For JevLight adapter changes, test the runtime patch against the pinned commit in a fresh checkout. State clearly when a result was not remeasured.
