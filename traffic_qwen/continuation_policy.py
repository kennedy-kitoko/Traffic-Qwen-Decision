"""Continuation policies for matched CityFlow counterfactual branches."""

from dataclasses import dataclass


@dataclass
class HistoricalOpenLoopContinuation:
    """Replay logged commands at their original timestamps (legacy comparator)."""
    raw: list
    target_index: int
    target_phase: int
    start_tick: int

    def actions(self, env, tick):
        from traffic_qwen.counterfactual_cityflow import _trace_command
        result = []
        for i in range(len(env.list_intersection)):
            if i == self.target_index:
                r = self.raw[i][tick]
                if tick == self.start_tick:
                    result.append(self.target_phase)
                elif r.get("decision_source") == "laya" and env.duration[i] <= 0:
                    result.append(_trace_command(r)[0])
                else:
                    result.append(-1)
            else:
                result.append(_trace_command(self.raw[i][tick])[0])
        return result


@dataclass
class MaxPressureContinuation:
    """Recompute JevLight's canonical 4-phase MaxPressure action every 5 s."""
    last_actions: list

    @staticmethod
    def choose(state, previous_action=0):
        if state["cur_phase"][0] == -1:
            return int(previous_action)
        pressure = state["traffic_movement_pressure_queue"]
        phase_scores = (pressure[1] + pressure[4], pressure[7] + pressure[10],
                        pressure[0] + pressure[3], pressure[6] + pressure[9])
        # np.argmax tie behavior in models/maxpressure_agent.py is first phase.
        return max(range(4), key=lambda i: (phase_scores[i], -i))

    def actions(self, env, tick=None):
        states, _ = env.get_state()
        self.last_actions = [self.choose(s, self.last_actions[i])
                             for i, s in enumerate(states)]
        return list(self.last_actions)
