# Traffic-Qwen V2 — Traffic Decision Dataset Protocol

This document defines the scientific and operational protocol used to construct the training dataset for **Traffic-Qwen V2**.

The objective is to define the traffic-control rules, safety constraints, counterfactual evaluation procedure, reward calculation, and target construction **before** launching large-scale data generation with DeepSeek and SUMO.

The central principle is:

> **DeepSeek proposes decisions. SUMO measures their consequences. The simulator determines the training target.**

```text
Real SUMO traffic state
        ↓
Safety constraints
        ↓
Admissible-action mask
        ↓
Structured DeepSeek proposal
        ↓
Counterfactual evaluation in SUMO
        ↓
Measured traffic consequences
        ↓
Reward calculation
        ↓
Validation and cleaning
        ↓
Soft target over 24 joint actions
        ↓
Traffic-Qwen V2 dataset
```

---

## 1. Decision Space

Each Traffic-Qwen decision is a joint combination of:

- one traffic-signal phase;
- one effective green duration.

### Traffic phases

| Phase | Served movements |
|---|---|
| `ETWT` | East-West through movements |
| `NTST` | North-South through movements |
| `ELWL` | East-West left-turn movements |
| `NLSL` | North-South left-turn movements |

### Green durations

```text
15 s, 20 s, 25 s, 30 s, 35 s, 40 s
```

The complete decision space therefore contains:

**4 phases × 6 durations = 24 joint actions**

Examples:

```text
ETWT_15
ETWT_20
...
NLSL_35
NLSL_40
```

Traffic-Qwen V2 uses a **single 24-class decision output** rather than two independent phase and duration outputs.

This preserves the relationship between the selected phase and its duration. A phase and a duration that are individually probable are not necessarily a good combination when executed together.

---

## 2. Operational Safety Constraints

Safety and operational constraints are enforced independently from DeepSeek and Traffic-Qwen.

The current configuration is:

| Constraint | Value |
|---|---:|
| Minimum effective green | 15 s |
| Maximum continuous green | 60 s |
| Priority waiting threshold | 60 s |
| Minimum queue for priority | 1 vehicle |
| Yellow transition | 5 s |

A model is never allowed to bypass these constraints.

---

## 3. Minimum Green: 15 Seconds

A newly activated phase must remain green for at least 15 seconds.

**Rule:**

```text
If current_green_time < 15 s:
    required_control = KEEP
```

No phase change is allowed before the minimum green has been completed.

`KEEP` is an operational controller command. It is **not currently added as a 25th Traffic-Qwen training class**.

This rule prevents:

- excessively rapid phase changes;
- unnecessary yellow transitions;
- oscillation between phases;
- interruption before vehicles have enough time to cross the intersection.

---

## 4. Maximum Continuous Green: 60 Seconds

A phase cannot remain continuously green for more than 60 seconds.

For an action that keeps the current phase:

```text
elapsed_green + requested_green ≤ 60 s
```

### Example

If the current phase has already been green for 25 seconds:

| Requested extension | Continuous green | Allowed |
|---:|---:|---|
| 15 s | 40 s | Yes |
| 20 s | 45 s | Yes |
| 25 s | 50 s | Yes |
| 30 s | 55 s | Yes |
| 35 s | 60 s | Yes |
| 40 s | 65 s | No |

The guardrail checks the continuous-green counter during execution so that repeated extensions cannot bypass the 60-second limit.

At the maximum continuous-green limit, the current phase must not be extended further when another phase with demand is available.

---

## 5. Starvation Prevention and Priority

A competing traffic movement becomes priority-eligible only when both conditions are satisfied:

```text
queue ≥ 1 vehicle
AND
average_waiting_time ≥ 60 s
```

Current parameters:

```text
priority_wait_seconds = 60
minimum_queue_for_priority = 1
```

An empty lane therefore cannot become urgent because of an obsolete waiting-time value.

A movement already served by the current phase is considered under service and does not compete against itself for priority.

If several competing movements exceed the priority threshold, they are ranked by:

1. highest observed average waiting time;
2. highest queue if waiting times are equal;
3. deterministic phase ordering if both remain equal.

Once the 15-second minimum green has been satisfied, the phase serving the urgent movement becomes mandatory.

Its admissible green durations remain candidates.

---

## 6. Rule Priority Order

At each control check, the rules are evaluated in the following order:

```text
1. Protect the 15 s minimum green
2. Detect an overdue competing movement
3. Prevent continuous green from exceeding 60 s
4. Apply the proposed controller action if it remains admissible
```

An urgent movement therefore cannot interrupt a phase that has just started.

It receives priority immediately after the minimum-green safety period has been completed.

---

## 7. Yellow Transition

When the selected action changes the active phase, JevLight/SUMO applies a 5-second yellow transition.

Example:

```text
Current phase: ETWT
Requested action: NLSL_20
```

Execution:

```text
5 s yellow
+
20 s effective NLSL green
=
25 s simulated time
```

The duration selected by Traffic-Qwen always represents **effective green time**.

Yellow time is not included in the selected green duration.

When the current phase is maintained, no additional yellow transition is inserted.

---

## 8. Dynamic Action Mask

Traffic-Qwen always keeps a fixed output space of 24 joint actions.

However, not every action is legal in every state.

The controller therefore constructs a dynamic action mask.

Example:

```json
{
  "eligible_actions": [
    "phase_3_15",
    "phase_3_20",
    "phase_3_25"
  ],
  "forbidden_actions": [
    "phase_0_15",
    "phase_1_15",
    "phase_2_15"
  ]
}
```

A forbidden action always receives:

```text
P(action | state) = 0
```

The remaining probability mass is normalized over admissible actions only.

---

## 9. State Provided to DeepSeek

DeepSeek receives a structured representation of the current traffic state.

It does not receive future information from the counterfactual simulations.

For the focal intersection, the state contains information such as:

- intersection ID;
- simulation time;
- current phase;
- elapsed effective green;
- queue per movement;
- observed average waiting time per movement;
- approaching vehicles;
- available neighboring-intersection information;
- available phases;
- available durations;
- applicable safety constraints.

A network summary additionally contains information such as:

- current phase of each intersection;
- total queue;
- total approaching vehicles;
- maximum observed lane waiting time.

DeepSeek therefore receives local traffic information and a compact network context without access to future simulation outcomes.

---

## 10. Exact Role of DeepSeek

DeepSeek acts as a **teacher for proposal and analysis**, not as the final source of truth.

### 10.1 Initial proposal

DeepSeek observes the current state and proposes a structured action.

Example:

```json
{
  "phase": "NLSL",
  "duration_seconds": 25
}
```

The response is validated with Pydantic.

The validation rejects:

- unknown phases;
- unsupported durations;
- unexpected fields;
- ambiguous numeric values;
- free-form text used directly as a control action.

If the teacher proposal violates the minimum green, maximum green, or priority rules, the proposal may be retained for audit but is not executed as a legal action.

### 10.2 Review after simulation

After the counterfactual actions have been simulated, DeepSeek may review the measured outcomes and explain why another action performed better.

This review is stored as metadata.

It does **not** determine the final training target.

The distinction is:

```text
DeepSeek proposal
        ↓
Candidate expert decision

SUMO evaluation
        ↓
Measured consequences

Training target
        ↓
Derived from measured simulation outcomes
```

---

## 11. Counterfactual Evaluation in SUMO

For every selected traffic state:

1. SUMO reaches the target simulation time.
2. The complete SUMO state is saved.
3. The required Python/JevLight controller state is also preserved.
4. Every admissible action restarts from the same restored state.
5. The candidate phase is applied.
6. Yellow is applied if the phase changes.
7. The candidate phase remains green for the requested effective duration.
8. MaxPressure controls the remaining evaluation horizon.
9. Every branch ends after exactly the same evaluation horizon.
10. Traffic consequences are measured.

The current evaluation horizon is:

```text
H = 120 s
```

All candidate actions are therefore compared using:

- the same initial state;
- the same traffic demand;
- the same random seed;
- the same evaluation horizon;
- the same continuation policy;
- the same reward configuration.

This is required for a fair counterfactual comparison.

---

## 12. Continuation Policy

After the candidate action has been executed, **MaxPressure** controls the remaining evaluation horizon.

The same continuation policy is used for every candidate action.

This prevents short actions from receiving an artificial advantage merely because they return earlier to a historical controller trajectory.

The measured return therefore represents the consequence of:

```text
candidate action
+
common continuation policy
+
fixed 120 s evaluation horizon
```

It should not be interpreted as proof of global optimality.

---

## 13. Interrupted Actions

An action can be legal when it starts but become incompatible with a safety or priority rule during execution.

For example, another movement may become overdue while a candidate action is still committed.

If an action is interrupted before its requested green duration has been fully executed, that branch is not used as a normal supervised example for that phase-duration combination.

It is rejected from the main training target and retained separately for audit.

Recommended audit metadata:

```json
{
  "requested_action": "phase_0_40",
  "executed_green_seconds": 20,
  "termination_reason": "priority_preemption",
  "valid_for_supervised_training": false
}
```

---

## 14. Measured Traffic Outcomes

Each counterfactual branch records several traffic metrics independently.

The current measurements include:

- mean queue;
- accumulated waiting time;
- vehicles reaching their destination;
- phase switches;
- starvation;
- downstream occupancy and spillback.

Raw components are preserved in the dataset metadata so that the final reward can be independently recalculated.

---

## 15. Reward Function

The current reward is configurable.

The present configuration is:

```text
R = -0.30 × Q
    -0.25 × W
    +0.20 × T
    -0.05 × S
    -0.10 × F
    -0.10 × B
```

Where:

| Variable | Meaning |
|---|---|
| `Q` | normalized mean queue |
| `W` | normalized waiting |
| `T` | normalized throughput / arrived vehicles |
| `S` | normalized phase switching |
| `F` | normalized starvation |
| `B` | normalized spillback |

The convention is:

```text
Higher reward = better action
```

If an implementation also stores a positive traffic cost, the relationship must remain explicit:

```text
reward = -cost
```

The reward sign must never be ambiguous when constructing the training targets.

---

## 16. Soft Training Targets

Traffic-Qwen V2 does not automatically treat the highest-return action as a perfectly correct hard label.

If two actions produce nearly identical outcomes, the dataset should preserve that uncertainty.

Example:

```text
Best return       = 0.285776
Second-best return = 0.282701
Difference         = 0.003075
```

A soft probability distribution is therefore constructed from the measured returns.

For every admissible action:

```text
logit(a) = (R(a) - R_max) / temperature
```

Then:

```text
P(a | s) = exp(logit(a)) / Σ exp(logit(a'))
```

For forbidden actions:

```text
P(a | s) = 0
```

The final distribution is normalized over admissible actions.

The temperature controls target sharpness:

- low temperature → strongly concentrated on the best actions;
- medium temperature → preserves plausible alternatives;
- high temperature → flatter distribution.

The temperature must be selected using training and validation data only.

The final test set must never be used to select it.

---

## 17. Dataset Example Structure

A Traffic-Qwen V2/V3 record follows the general structure below:

```json
{
  "state": {
    "simulation_time_seconds": 301,
    "focal_intersection_id": "intersection_1_1",
    "intersection": {
      "current_phase": "ETWT",
      "current_phase_elapsed_green_s": 25,
      "lanes": {}
    },
    "network_summary": {},
    "safety_constraints": {
      "minimum_green_seconds": 15,
      "maximum_continuous_green_seconds": 60,
      "priority_wait_seconds": 60,
      "minimum_queue_for_priority": 1
    }
  },
  "questions": {
    "traffic_action": {
      "type": "choice",
      "criteria": {
        "phase_0_15": "Activate ETWT for 15 seconds",
        "...": "...",
        "phase_3_40": "Activate NLSL for 40 seconds"
      }
    }
  },
  "gold": {
    "traffic_action": {
      "label": "phase_3_15",
      "choice": "phase_3_15",
      "probabilities": {
        "phase_0_15": 0.0,
        "...": 0.0,
        "phase_3_15": 0.31,
        "phase_3_20": 0.27
      }
    }
  },
  "metadata": {
    "teacher_proposal": {},
    "teacher_review": {},
    "eligible_actions": [],
    "forbidden_actions": [],
    "measured_returns": {},
    "reward_components": {},
    "seed": 3407,
    "episode_id": "...",
    "traffic_file": "...",
    "snapshot_sha256": "..."
  }
}
```

The main learned field is:

```text
traffic_action
```

with 24 joint phase-duration choices.

---

## 18. Train / Validation / Test Separation

Individual traffic states are **not randomly mixed across splits**.

Complete episodes are assigned to a split before state selection:

```text
Episodes
├── Train
├── Validation
└── Test
```

No state, snapshot, or temporal fragment from a test episode may be used to:

- train the model;
- choose the soft-target temperature;
- tune the reward;
- select thresholds;
- select hyperparameters.

Episode separation must occur before DeepSeek calls and before counterfactual evaluation.

---

## 19. Duplicate Detection

Every selected state receives a normalized state hash based on significant information such as:

- intersection;
- current phase;
- elapsed green;
- queues;
- waiting times;
- approaching traffic;
- network context;
- scenario;
- simulation time.

A minimum temporal distance is also applied to reduce long sequences of nearly identical states.

Equivalent or nearly equivalent states must not leak across train, validation, and test splits.

---

## 20. Dataset Cleaning and Rejection

A comparison is rejected if at least one critical validity condition fails.

Examples include:

- SUMO snapshot cannot be faithfully restored;
- restored observation differs from the original;
- green timer is missing;
- unknown phase or duration;
- invalid DeepSeek response;
- SUMO branch error;
- unequal evaluation horizons;
- different seeds;
- different continuation policies;
- different reward configurations;
- `NaN` or infinite values;
- missing reward components;
- reward cannot be recalculated;
- action is preempted before completing its commitment;
- overlap between train and test episodes;
- duplicate state;
- declared traffic file does not match the executed file;
- forbidden action receives positive target probability.

Rejected branches are retained separately for audit.

They are never silently converted into training examples.

---

## 21. Validation Before Large-Scale Generation

Validation is performed at several levels.

### 21.1 Unit tests

Tests cover at least:

- minimum green;
- 60-second maximum continuous green;
- empty lane without false priority;
- overdue lane with actual demand;
- selection between multiple overdue movements;
- dynamic action masking;
- yellow transition;
- reward calculation;
- starvation;
- spillback;
- Pydantic validation;
- target probability normalization.

### 21.2 SUMO smoke test

A real Jinan state must be successfully:

```text
saved
→ restored
→ controlled
→ simulated for 120 s
→ measured
```

without snapshot divergence or safety-rule violation.

### 21.3 Pilot dataset

Before large-scale generation, a small dataset from multiple episodes must be analyzed.

Important statistics include:

- number of admissible actions per state;
- diversity of winning phases;
- diversity of winning durations;
- frequency of overdue-lane priority;
- frequency of the 60-second green limit;
- target entropy;
- best-versus-runner-up margins;
- action-preemption rate;
- rejection rate;
- DeepSeek API cost;
- SUMO computational cost.

### 21.4 Unsloth / Clef validation

A small sample must be loaded through:

```python
FastDecisionModel.build_dataset(...)
```

The validation should verify:

- no silently ignored records;
- no unexpected truncation;
- exactly 24 target probabilities;
- probability sum approximately equal to 1;
- forbidden actions have probability 0;
- successful forward pass;
- successful backward pass;
- model save and reload.

---

## 22. Why SUMO Remains the Final Arbiter

A general-purpose LLM can produce a convincing traffic-control explanation while still selecting a suboptimal action.

Traffic-Qwen V2 is therefore **not designed to blindly imitate DeepSeek**.

The intended hierarchy is:

```text
DeepSeek
    ↓
Proposes and explains candidate decisions

SUMO
    ↓
Measures counterfactual consequences

Reward
    ↓
Ranks the measured outcomes

Soft targets
    ↓
Preserve relative action quality and uncertainty

Traffic-Qwen V2
    ↓
Learns a specialized traffic-control decision policy
```

DeepSeek contributes reasoning and expert proposals.

SUMO provides controlled evidence about the consequences of actions.

Traffic-Qwen V2 learns from those measured consequences and is intended to make fast specialized decisions without requiring the teacher model during deployment.

---

## 23. Reproducibility

Each generated example should preserve enough metadata to reproduce and audit the decision.

This includes:

```text
scenario
traffic file
episode ID
seed
simulation time
intersection ID
snapshot hash
current phase
elapsed green
safety configuration
eligible actions
forbidden actions
teacher proposal
teacher review
counterfactual returns
reward components
reward configuration
target probabilities
```

Important restored states should include a cryptographic fingerprint such as:

```text
snapshot_sha256
```

This allows accidental changes or restoration errors to be detected.

---

## 24. Current Protocol Status

The core protocol currently defines:

- 24 joint phase-duration actions;
- 15-second minimum effective green;
- 60-second maximum continuous green;
- starvation-prevention priority;
- 5-second yellow transition;
- dynamic action masking;
- SUMO state restoration;
- counterfactual action evaluation;
- MaxPressure continuation;
- multi-objective reward;
- soft targets;
- episode-level dataset splitting;
- rejection and audit rules;
- reproducibility metadata.

Traffic-Qwen V1 results and historical datasets remain unchanged.

Before large-scale teacher generation, the V3 dataset builder should fully support:

```text
dynamic action masking
+
zero probability for forbidden actions
+
normalized soft targets
+
raw SUMO returns
+
reward components
+
complete reproducibility metadata
+
explicit rejection of interrupted branches
```

Large-scale DeepSeek generation should begin only after this pipeline and its tests have been validated.

---

## Core Principle

```text
LLM proposes
      ↓
SUMO evaluates
      ↓
Reward ranks
      ↓
Traffic-Qwen learns
```

Traffic-Qwen V2 is not intended to become a smaller model that simply imitates a general-purpose LLM.

The objective is to use controlled and reproducible traffic simulations to train a **specialized decision model for adaptive traffic-signal control**.
