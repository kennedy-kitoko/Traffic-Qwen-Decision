"""Unit-explicit, normalized reward components for Traffic-Qwen V2."""

WEIGHTS = {
    "queue_reduction": 0.30,
    "waiting_reduction": 0.25,
    "departed_vehicles": 0.20,
    "switching_penalty": 0.05,
    "starvation_penalty": 0.10,
    "spillback_penalty": 0.10,
}
NORMALIZERS = {
    "queue_reduction": 25.0,                 # vehicles per 5-second sample
    "waiting_reduction": 5000.0,             # vehicle-seconds per 5-second sample
    "departed_vehicles": 10.0,               # vehicles per 5-second sample
    "switching_penalty": 2.0,                # switch events per 5-second sample
    "starvation_penalty": 1.0,               # bounded progressive penalty per 5-second sample
    "spillback_penalty": 10.0,               # saturated exit-lane samples
}


def normalize_component(name, value):
    if name not in NORMALIZERS:
        raise KeyError(name)
    sign = -1.0 if name.endswith("_penalty") else 1.0
    return sign * max(-1.0, min(1.0, float(value) / NORMALIZERS[name]))


def combine_components(raw, weights=None):
    """Return auditable raw, normalized, weighted values and scalar reward."""
    result = {}
    total = 0.0
    weights = weights or WEIGHTS
    for name, weight in weights.items():
        value = float(raw.get(name, 0.0))
        normalized = normalize_component(name, value)
        weighted = weight * normalized
        result[name] = {
            "raw": value,
            "unit": ("vehicles" if name == "queue_reduction" else
                     "vehicle_seconds" if name == "waiting_reduction" else
                     "vehicles" if name == "departed_vehicles" else
                     "switches" if name == "switching_penalty" else
                     "vehicle_seconds_after_60s" if name == "starvation_penalty" else
                     "saturated_lane_samples"),
            "normalized": normalized,
            "weight": weight,
            "weighted": weighted,
        }
        total += weighted
    result["total"] = total
    return result


def discounted_return(step_components, gamma=0.99, weights=None):
    if not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    return sum((gamma ** i) * combine_components(row, weights)["total"]
               for i, row in enumerate(step_components))


def spillback_lane_samples(lane_vehicle_counts, lane_lengths, lane_ids, threshold=0.80, storage_m=7.5):
    """Count exit lanes at/above storage occupancy; capacity is lane length / 7.5 m."""
    if not 0 < threshold <= 1 or storage_m <= 0:
        raise ValueError("threshold must be in (0, 1] and storage_m must be positive")
    ratios = []
    for lane in lane_ids:
        capacity = max(1.0, float(lane_lengths.get(lane, 0.0)) / storage_m)
        ratios.append(float(lane_vehicle_counts.get(lane, 0)) / capacity)
    return sum(ratio >= threshold for ratio in ratios), max(ratios, default=0.0)
