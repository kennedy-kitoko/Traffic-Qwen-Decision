"""Joint phase-duration action space used by Traffic-Qwen Decision."""
from dataclasses import dataclass

PHASES = (0, 1, 2, 3)
DURATIONS = (15, 20, 25, 30, 35, 40)
ACTION_IDS = tuple(f"phase_{p}_{d}" for p in PHASES for d in DURATIONS)


@dataclass(frozen=True)
class Action:
    phase: int
    duration_seconds: int


def encode(phase: int, duration_seconds: int) -> str:
    action = f"phase_{int(phase)}_{int(duration_seconds)}"
    if action not in ACTION_IDS:
        raise ValueError(f"Invalid Traffic-Qwen action: {action}")
    return action


def decode(action: str) -> Action:
    if action not in ACTION_IDS:
        raise ValueError(f"Invalid Traffic-Qwen action: {action}")
    _, phase, duration = action.split("_")
    return Action(int(phase), int(duration))


def criteria() -> dict[str, str]:
    return {encode(p, d): f"Activate JevLight phase {p + 1} for {d} seconds."
            for p in PHASES for d in DURATIONS}
