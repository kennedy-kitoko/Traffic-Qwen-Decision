"""Schema helpers for the Traffic-Qwen typed-decision dataset."""
from traffic_qwen.action_space import ACTION_IDS, criteria

QUESTION_NAME = "traffic_action"
QUESTION = {QUESTION_NAME: {
    "type": "choice",
    "instructions": "Choose one phase and green duration to serve traffic while reducing queues and waiting and preventing starvation.",
    "criteria": criteria(),
}}


def sample(state: dict, gold_action: str, metadata: dict) -> dict:
    if gold_action not in ACTION_IDS:
        raise ValueError(f"Invalid gold action: {gold_action}")
    return {"state": state, "questions": QUESTION,
            "gold": {QUESTION_NAME: {"label": gold_action}},
            "metadata": metadata}
