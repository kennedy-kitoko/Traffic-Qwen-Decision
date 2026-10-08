"""JevLight-compatible Traffic-Qwen V1 Dual-Choice agent."""
from __future__ import annotations

import json
import os
import threading
import time

from utils.my_utils import get_state_detail
from models.laya_agent import LayaAgent


class TrafficQwenAgent:
    _model = None
    _tokenizer = None
    _model_key = None
    _model_lock = threading.Lock()
    _infer_lock = threading.Lock()
    PHASE_OPTIONS = tuple(f"phase_{i}" for i in range(4))
    DURATIONS = (15, 20, 25, 30, 35, 40)

    def __init__(self, dic_agent_conf, dic_traffic_env_conf, intersection=None,
                 inter_name="0", phase_num=4, **_):
        if intersection is None:
            raise ValueError("TrafficQwenAgent requires intersection road metadata")
        if int(phase_num) != 4:
            raise ValueError("Traffic-Qwen V1 Dual-Choice requires exactly four phases")
        self.roads = intersection["roads"]
        self.inter_name = inter_name
        self.phase_num = 4
        self.duration_options = [int(v) for _, v in sorted(
            dic_traffic_env_conf.get("ACTION_DURATION", dict(enumerate(self.DURATIONS))).items(),
            key=lambda pair: int(pair[0]))]
        if tuple(self.duration_options) != self.DURATIONS:
            raise ValueError(f"Unsupported action durations: {self.duration_options}")
        self.model_path = dic_agent_conf.get("TRAFFIC_QWEN_MODEL_PATH", "models/traffic-qwen-v1-dual-choice")
        self.device = dic_agent_conf.get("TRAFFIC_QWEN_DEVICE", "cuda")
        self.seed = int(dic_agent_conf.get("SEED", 3407))
        self.model_revision = dic_agent_conf.get("TRAFFIC_QWEN_MODEL_REVISION")
        self.use_guardrail = bool(dic_agent_conf.get("TRAFFIC_QWEN_GUARDRAIL", False))
        self.use_fallback = bool(dic_agent_conf.get("TRAFFIC_QWEN_FALLBACK", False))
        self.min_confidence = float(dic_agent_conf.get("TRAFFIC_QWEN_MIN_CONFIDENCE", 0.0))
        self.last_response = None
        self.last_error = None
        self.last_guardrail = None
        self.last_confidence = None
        self.last_phase_probabilities = None
        self.last_duration_probabilities = None
        self.last_latency_ms = None
        self.last_fallback_used = False
        self.last_proposed_phase = None
        self.last_proposed_duration = None
        self.last_applied_phase = None
        self.last_applied_duration = None
        self.last_state_payload = None
        self.prev_phase = 0
        self.phase_age = 0
        self.last_decision_count = None
        self.phase_history = []
        self.duration_history = []
        self._ensure_model(self.model_path, self.device, self.seed, self.model_revision)

    @classmethod
    def _ensure_model(cls, model_path, device, seed=3407, revision=None):
        key = (os.path.abspath(model_path) if os.path.exists(model_path) else model_path,
               device, int(seed), revision)
        with cls._model_lock:
            if cls._model is None or cls._model_key != key:
                from unsloth import FastDecisionModel
                import torch
                dtype = torch.bfloat16 if torch.cuda.is_available() else None
                cls._model, cls._tokenizer = FastDecisionModel.from_pretrained(
                    model_path, max_seq_length=2048, dtype=dtype, load_in_4bit=True,
                    use_gradient_checkpointing="unsloth", local_files_only=True,
                    random_state=int(seed), revision=revision,
                )
                if hasattr(cls._model, "to") and device and device != "auto":
                    # Unsloth places quantized modules itself; avoid .to() for 4-bit models.
                    pass
                if not getattr(cls._model, "is_clef", False):
                    raise RuntimeError("Traffic-Qwen checkpoint is missing the Clef decision head")
                cls._model_key = key
        return cls._model, cls._tokenizer

    @classmethod
    def warmup(cls, model_path="models/traffic-qwen-v1-dual-choice", device="cuda", seed=3407,
               revision=None):
        from unsloth import FastDecisionModel
        model, tokenizer = cls._ensure_model(model_path, device, seed, revision)
        sample_state = {"task": "traffic_signal_control", "warmup": True}
        questions = cls._questions()
        with cls._infer_lock:
            return FastDecisionModel.predict(model, tokenizer, sample_state, questions)

    @staticmethod
    def _questions():
        from traffic_qwen.build_dual_choice_dataset import PHASES, DURATIONS, PHASE_DESCRIPTIONS
        return {
            "phase": {"type": "choice", "instructions": "Choose the signal phase that should be activated to reduce congestion and waiting while preventing starvation.",
                      "criteria": {p: PHASE_DESCRIPTIONS[i] for i, p in enumerate(PHASES)}},
            "duration": {"type": "choice", "instructions": "Choose the green duration appropriate for the selected phase and observed traffic conditions.",
                         "criteria": {d: f"Green time of {d} seconds." for d in DURATIONS}},
        }

    def _state(self, detail, incoming, mean_speed, count):
        if self.last_decision_count is not None:
            elapsed = max(0, int(count - self.last_decision_count) * 5)
            self.phase_age += elapsed
        self.last_decision_count = int(count or 0)
        movement = {name: {"cells": list(value.get("cells", ())),
                           "queue_vehicles": float(value.get("queue_len", 0)),
                           "avg_wait_seconds": float(value.get("avg_wait_time", 0))}
                    for name, value in detail.items()}
        incoming_proxy = {name: {"cells": list(value.get("cells", ())),
                                 "queue_proxy": float(value.get("queue_len", 0))}
                          for name, value in incoming.items()}
        state = {
            "intersection_id": self.inter_name,
            "simulation_time_seconds": int(count or 0) * 5,
            "current_phase_command_proxy": self.prev_phase,
            "seconds_since_phase_command_proxy": self.phase_age,
            "recent_phase_commands_proxy": self.phase_history[-4:],
            "recent_applied_durations_seconds": self.duration_history[-4:],
            "movement_lanes": movement,
            "incoming_approach_proxy": incoming_proxy,
            "approaching_speed": float(mean_speed),
            "available_fields": ["movement cells", "movement queue length", "movement average wait",
                                 "incoming cells/queue proxy", "approaching speed", "recent decisions from the trace"],
            "unavailable_fields": ["exact approaching vehicle count", "downstream occupancy", "recent departures",
                                   "per-vehicle max wait", "neighbor states", "CityFlow cur_phase at observation",
                                   "elapsed green from CityFlow", "phase service age by movement"],
        }
        self.last_state_payload = state
        return state

    def choose_action(self, env, state=None, count=None):
        from unsloth import FastDecisionModel
        detail, incoming, mean_speed = get_state_detail(self.roads, env)
        payload = self._state(detail, incoming, mean_speed, count)
        questions = self._questions()
        started = time.perf_counter()
        self.last_error = None
        self.last_guardrail = None
        self.last_fallback_used = False
        try:
            model, tokenizer = self._ensure_model(self.model_path, self.device, self.seed, self.model_revision)
            with self._infer_lock:
                response = FastDecisionModel.predict(model, tokenizer, payload, questions)
            self.last_latency_ms = (time.perf_counter() - started) * 1000
            self.last_response = response
            pa = response["phase"]
            da = response["duration"]
            phase_probs = {str(k): float(v) for k, v in pa["probabilities"].items()}
            duration_probs = {str(k): float(v) for k, v in da["probabilities"].items()}
            self._validate_probabilities(phase_probs, self.PHASE_OPTIONS)
            self._validate_probabilities(duration_probs, tuple(str(x) for x in self.DURATIONS))
            phase = self.PHASE_OPTIONS.index(str(pa["choice"]))
            duration_seconds = int(da["choice"])
            duration = self.duration_options.index(duration_seconds)
            prior_phase = self.prev_phase
            self.last_proposed_phase = phase
            self.last_proposed_duration = duration_seconds
            self.last_phase_probabilities = phase_probs
            self.last_duration_probabilities = duration_probs
            self.last_confidence = min(max(phase_probs.values()), max(duration_probs.values()))
            if self.min_confidence and self.last_confidence < self.min_confidence:
                raise ValueError(f"Decision confidence {self.last_confidence:.4f} is below threshold")
            applied_phase, applied_duration = self._apply_guardrail(phase, duration, detail)
            self.phase_history.append(applied_phase)
            self.duration_history.append(self.duration_options[applied_duration])
            self.prev_phase = applied_phase
            self.last_applied_phase = applied_phase
            self.last_applied_duration = self.duration_options[applied_duration]
            if applied_phase != prior_phase:
                self.phase_age = 0
            return applied_phase, applied_duration
        except Exception as exc:
            self.last_latency_ms = (time.perf_counter() - started) * 1000
            self.last_error = f"{type(exc).__name__}: {exc}"
            if not self.use_fallback:
                raise RuntimeError(f"Traffic-Qwen inference failed at {self.inter_name}: {exc}") from exc
            self.last_fallback_used = True
            phase, duration = self._fallback(detail)
            self.prev_phase = phase
            self.last_applied_phase = phase
            self.last_applied_duration = self.duration_options[duration]
            self.phase_history.append(phase)
            self.duration_history.append(self.duration_options[duration])
            return phase, duration

    @staticmethod
    def _validate_probabilities(probs, expected):
        if set(probs) != set(expected) or any(not (0 <= x <= 1) for x in probs.values()):
            raise ValueError(f"Invalid decision probabilities: {probs}")
        if abs(sum(probs.values()) - 1) > 0.02:
            raise ValueError(f"Probabilities do not sum to 1: {probs}")

    def _apply_guardrail(self, phase, duration, detail):
        if not self.use_guardrail:
            return phase, duration
        metrics = LayaAgent._phase_metrics(detail)
        scores = [metrics[f"phase_{i+1}"]["urgency_score"] for i in range(4)]
        best = max(range(4), key=lambda i: (scores[i], -i))
        original_phase, original_duration = phase, duration
        if scores[best] > 0 and scores[phase] < .60 * scores[best]:
            phase = best
        queue = metrics[f"phase_{phase+1}"]["queue_vehicles"]
        if queue >= 40:
            duration = len(self.duration_options) - 1
        elif queue >= 20:
            duration = max(duration, self.duration_options.index(30))
        if phase != original_phase or duration != original_duration:
            self.last_guardrail = {"model_phase": original_phase, "guardrail_phase": phase,
                                   "model_duration": self.duration_options[original_duration],
                                   "guardrail_duration": self.duration_options[duration], "phase_scores": scores}
        return phase, duration

    @staticmethod
    def _fallback(detail):
        from models.laya_agent import LayaAgent
        lanes = LayaAgent.PHASE_LANES
        loads = [sum(LayaAgent._lane_load(detail.get(lane, {})) for lane in pair) for pair in lanes]
        phase = max(range(4), key=lambda i: (loads[i], -i))
        duration = min(5, max(0, int(loads[phase] // 5)))
        return phase, duration
