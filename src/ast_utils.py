"""Small, inspectable Adaptive Stress Testing utilities for METANET.

The proof of concept intentionally keeps four concerns separate:

* :class:`ASTEnvironment` advances the existing METANET simulator.
* :func:`assess_physical_failure` defines the current failure event.
* Action models define how an AST action changes the environment and how
  probable/costly that action is.
* :func:`run_mcts` searches action sequences without knowing traffic details.

That separation is deliberate. A later predictive-accuracy failure detector,
dynamic parameter schedule, or reinforcement-learning solver can replace one
piece without rewriting the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from simulation_utils import metanet_step


# ---------------------------------------------------------------------------
# Traffic state and failure detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrafficState:
    """Minimal state needed to continue one METANET rollout."""

    density: np.ndarray
    velocity: np.ndarray
    flow_origin: float
    queue: float
    time_index: int = 0


@dataclass(frozen=True)
class PhysicalFailureConfig:
    """Hard limits for the first physical/numerical failure definition."""

    max_density: float = 190.0  # veh/km/lane
    max_velocity: float = 250.0  # km/h


@dataclass(frozen=True)
class FailureAssessment:
    """Result of checking one state transition against the failure event.

    ``signed_margin`` is positive beyond a failure boundary, zero on the
    boundary, and negative on the safe side. ``distance`` is the non-negative
    distance to the nearest normalized boundary while the state remains safe.
    """

    failed: bool
    distance: float
    signed_margin: float
    subtype: str
    time_index: int
    segment: int
    raw_value: float
    reported_value: float
    threshold: float


def _first_nonfinite(array: np.ndarray) -> Optional[int]:
    locations = np.flatnonzero(~np.isfinite(array))
    return int(locations[0]) if locations.size else None


def assess_physical_failure(
    state: TrafficState,
    diagnostics: Mapping[str, np.ndarray],
    config: PhysicalFailureConfig,
) -> FailureAssessment:
    """Check raw and reported values for physical/numerical invalidity.

    Negative raw updates matter even though the normal simulator clips them.
    Upper limits are checked on the reported next state. All finite constraints
    are normalized so a single distance-to-failure can guide MCTS rollouts.
    """

    raw_density = np.asarray(diagnostics["raw_density"], dtype=float)
    raw_velocity = np.asarray(diagnostics["raw_velocity"], dtype=float)
    density = np.asarray(state.density, dtype=float)
    velocity = np.asarray(state.velocity, dtype=float)

    # Non-finite values are immediate numerical failures. Check raw values
    # first so clipping cannot conceal the source.
    arrays = (
        ("nonfinite_raw_density", raw_density, density),
        ("nonfinite_raw_velocity", raw_velocity, velocity),
        ("nonfinite_density", density, density),
        ("nonfinite_velocity", velocity, velocity),
    )
    for subtype, values, reported in arrays:
        segment = _first_nonfinite(values)
        if segment is not None:
            return FailureAssessment(
                failed=True,
                distance=0.0,
                signed_margin=math.inf,
                subtype=subtype,
                time_index=state.time_index,
                segment=segment,
                raw_value=float(values[segment]),
                reported_value=float(reported[segment]),
                threshold=math.nan,
            )

    # Each tuple is (normalized signed margin, subtype, segment, raw value,
    # reported value, threshold). The largest margin is the closest boundary.
    candidates: List[Tuple[float, str, int, float, float, float]] = []
    for segment in range(density.size):
        candidates.extend(
            [
                (
                    -raw_density[segment] / config.max_density,
                    "negative_raw_density",
                    segment,
                    float(raw_density[segment]),
                    float(density[segment]),
                    0.0,
                ),
                (
                    -raw_velocity[segment] / config.max_velocity,
                    "negative_raw_velocity",
                    segment,
                    float(raw_velocity[segment]),
                    float(velocity[segment]),
                    0.0,
                ),
                (
                    density[segment] / config.max_density - 1.0,
                    "excessive_density",
                    segment,
                    float(raw_density[segment]),
                    float(density[segment]),
                    config.max_density,
                ),
                (
                    velocity[segment] / config.max_velocity - 1.0,
                    "excessive_velocity",
                    segment,
                    float(raw_velocity[segment]),
                    float(velocity[segment]),
                    config.max_velocity,
                ),
            ]
        )

    margin, subtype, segment, raw_value, reported_value, threshold = max(
        candidates, key=lambda item: item[0]
    )
    failed = margin > 0.0
    return FailureAssessment(
        failed=failed,
        distance=0.0 if failed else -float(margin),
        signed_margin=float(margin),
        subtype=subtype,
        time_index=state.time_index,
        segment=segment,
        raw_value=raw_value,
        reported_value=reported_value,
        threshold=float(threshold),
    )


# ---------------------------------------------------------------------------
# Discrete action models used by the three proof-of-concept experiments
# ---------------------------------------------------------------------------


class ActionModel(Protocol):
    """Interface consumed by the simulator wrapper and generic MCTS search."""

    name: str
    actions: Tuple[Any, ...]

    def action_probability(self, action: Any) -> float:
        ...

    def intervention_cost(self, action: Any) -> float:
        ...

    def action_label(self, action: Any) -> str:
        ...

    def apply_block(
        self,
        action: Any,
        demand: np.ndarray,
        vsl_speeds: np.ndarray,
        start: int,
        end: int,
    ) -> None:
        ...


@dataclass(frozen=True)
class VSLAction:
    """One common VSL command applied to the configured segment group."""

    label: str
    speed: float


class VSLActionModel:
    """Discrete VSL actions with both probability and intervention scores."""

    name = "vsl"

    def __init__(
        self,
        actions: Sequence[VSLAction],
        probabilities: Sequence[float],
        costs: Sequence[float],
        controlled_segments: Optional[Sequence[int]] = None,
    ) -> None:
        if not (len(actions) == len(probabilities) == len(costs)):
            raise ValueError("VSL actions, probabilities, and costs must have equal length.")
        if not actions:
            raise ValueError("At least one VSL action is required.")

        probabilities_array = np.asarray(probabilities, dtype=float)
        if np.any(probabilities_array <= 0):
            raise ValueError("Every VSL action needs positive probability.")
        probabilities_array /= probabilities_array.sum()

        self.actions = tuple(actions)
        self._probabilities = {
            action: float(probability)
            for action, probability in zip(self.actions, probabilities_array)
        }
        self._costs = {
            action: float(cost) for action, cost in zip(self.actions, costs)
        }
        self.controlled_segments = (
            None if controlled_segments is None else tuple(int(i) for i in controlled_segments)
        )

    @classmethod
    def proof_of_concept(
        cls, controlled_segments: Optional[Sequence[int]] = None
    ) -> "VSLActionModel":
        """Return the agreed synthetic VSL model.

        ``unrestricted`` uses 1000 km/h, matching the repository's existing
        no-VSL convention. These probabilities and costs are deliberately easy
        to replace after review with the traffic-domain coauthor.
        """

        actions = (
            VSLAction("unrestricted", 1000.0),
            VSLAction("120", 120.0),
            VSLAction("100", 100.0),
            VSLAction("80", 80.0),
            VSLAction("60", 60.0),
        )
        return cls(
            actions=actions,
            probabilities=(0.50, 0.25, 0.15, 0.07, 0.03),
            costs=(0.0, 1.0, 2.0, 3.0, 4.0),
            controlled_segments=controlled_segments,
        )

    def action_probability(self, action: VSLAction) -> float:
        return self._probabilities[action]

    def intervention_cost(self, action: VSLAction) -> float:
        return self._costs[action]

    def action_label(self, action: VSLAction) -> str:
        return action.label

    def apply_block(
        self,
        action: VSLAction,
        demand: np.ndarray,
        vsl_speeds: np.ndarray,
        start: int,
        end: int,
    ) -> None:
        del demand  # VSL actions do not alter upstream inflow.
        if self.controlled_segments is None:
            vsl_speeds[start:end, :] = action.speed
        else:
            vsl_speeds[start:end, self.controlled_segments] = action.speed


@dataclass(frozen=True)
class GaussianInflowAction:
    """A discrete standard-normal multiplier used by MCTS."""

    z: float


class GaussianInflowActionModel:
    """Discrete approximation to additive Gaussian upstream-inflow noise."""

    name = "gaussian_inflow"

    def __init__(
        self,
        sigma: float,
        z_values: Iterable[float] = (-3, -2, -1, 0, 1, 2, 3),
    ) -> None:
        if sigma <= 0:
            raise ValueError("Gaussian inflow sigma must be positive.")
        self.sigma = float(sigma)
        self.actions = tuple(GaussianInflowAction(float(z)) for z in z_values)
        if not self.actions:
            raise ValueError("At least one Gaussian inflow action is required.")

        # Normalized normal-density weights form a simple categorical
        # approximation that is convenient for finite-action MCTS.
        weights = np.array([math.exp(-0.5 * action.z**2) for action in self.actions])
        weights /= weights.sum()
        self._probabilities = {
            action: float(probability)
            for action, probability in zip(self.actions, weights)
        }

    def action_probability(self, action: GaussianInflowAction) -> float:
        return self._probabilities[action]

    def intervention_cost(self, action: GaussianInflowAction) -> float:
        # The action-dependent part of a standard-normal negative log density.
        return 0.5 * action.z**2

    def action_label(self, action: GaussianInflowAction) -> str:
        return f"{action.z:+g} sigma"

    def apply_block(
        self,
        action: GaussianInflowAction,
        demand: np.ndarray,
        vsl_speeds: np.ndarray,
        start: int,
        end: int,
    ) -> None:
        del vsl_speeds  # Inflow actions leave VSL unrestricted.
        demand[start:end] = np.maximum(
            0.0, demand[start:end] + self.sigma * action.z
        )


class NominalActionModel:
    """Single no-disturbance action used for the mandatory baseline check."""

    name = "nominal"
    actions = ("nominal",)

    def action_probability(self, action: str) -> float:
        return 1.0

    def intervention_cost(self, action: str) -> float:
        return 0.0

    def action_label(self, action: str) -> str:
        return action

    def apply_block(
        self,
        action: str,
        demand: np.ndarray,
        vsl_speeds: np.ndarray,
        start: int,
        end: int,
    ) -> None:
        # Arrays are initialized to nominal demand and unrestricted VSL, so the
        # baseline action intentionally has no work to do.
        del action, demand, vsl_speeds, start, end


# ---------------------------------------------------------------------------
# Episode simulation and metrics
# ---------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """A complete or failure-terminated AST rollout."""

    actions: Tuple[Any, ...]
    action_labels: Tuple[str, ...]
    failed: bool
    failure: Optional[FailureAssessment]
    closest_assessment: Optional[FailureAssessment]
    log_likelihood: float
    intervention_cost: float
    simulated_steps: int
    time_step_hours: float
    density_history: np.ndarray
    velocity_history: np.ndarray
    raw_density_history: np.ndarray
    raw_velocity_history: np.ndarray

    @property
    def time_to_failure_minutes(self) -> Optional[float]:
        if not self.failed:
            return None
        return self.simulated_steps * self.time_step_hours * 60.0

    @property
    def closest_nonfailure_distance(self) -> Optional[float]:
        if self.closest_assessment is None:
            return None
        return self.closest_assessment.distance

    def summary(self) -> Dict[str, Any]:
        failure = self.failure
        closest = self.closest_assessment
        return {
            "failed": self.failed,
            "actions": list(self.action_labels),
            "simulated_steps": self.simulated_steps,
            "time_to_failure_minutes": self.time_to_failure_minutes,
            "log_likelihood": self.log_likelihood,
            "negative_log_likelihood": -self.log_likelihood,
            "intervention_cost": self.intervention_cost,
            "closest_nonfailure_distance": self.closest_nonfailure_distance,
            "closest_nonfailure_subtype": None if closest is None else closest.subtype,
            "closest_nonfailure_time_index": (
                None if closest is None else closest.time_index
            ),
            "closest_nonfailure_segment": None if closest is None else closest.segment,
            "closest_nonfailure_raw_value": (
                None if closest is None else closest.raw_value
            ),
            "closest_nonfailure_reported_value": (
                None if closest is None else closest.reported_value
            ),
            "closest_nonfailure_threshold": (
                None if closest is None else closest.threshold
            ),
            "closest_nonfailure_signed_margin": (
                None if closest is None else closest.signed_margin
            ),
            "failure_subtype": None if failure is None else failure.subtype,
            "failure_time_index": None if failure is None else failure.time_index,
            "failure_segment": None if failure is None else failure.segment,
            "failure_raw_value": None if failure is None else failure.raw_value,
            "failure_reported_value": None if failure is None else failure.reported_value,
            "failure_threshold": None if failure is None else failure.threshold,
            "failure_signed_margin": None if failure is None else failure.signed_margin,
        }


@dataclass
class ASTEnvironment:
    """A fixed METANET scenario stepped in one-minute AST action blocks."""

    T: float
    l: float
    initial_state: TrafficState
    nominal_demand: np.ndarray
    downstream_density: np.ndarray
    params: Dict[str, np.ndarray]
    lanes: Dict[int, float]
    failure_config: PhysicalFailureConfig = field(default_factory=PhysicalFailureConfig)
    action_interval_steps: int = 6
    unrestricted_speed: float = 1000.0

    def __post_init__(self) -> None:
        self.nominal_demand = np.asarray(self.nominal_demand, dtype=float)
        self.downstream_density = np.asarray(self.downstream_density, dtype=float)
        if self.nominal_demand.ndim != 1 or self.downstream_density.ndim != 1:
            raise ValueError("Demand and downstream density must be one-dimensional.")
        if self.nominal_demand.shape != self.downstream_density.shape:
            raise ValueError("Demand and downstream density must have the same length.")
        if self.nominal_demand.size % self.action_interval_steps != 0:
            raise ValueError("Simulation length must be divisible by action_interval_steps.")
        if self.initial_state.density.shape != self.initial_state.velocity.shape:
            raise ValueError("Initial density and velocity must have matching shapes.")
        if len(self.lanes) != self.initial_state.density.size:
            raise ValueError("Lane mapping must contain one entry per simulated segment.")

    @property
    def num_steps(self) -> int:
        return int(self.nominal_demand.size)

    @property
    def num_segments(self) -> int:
        return int(self.initial_state.density.size)

    @property
    def decision_horizon(self) -> int:
        return self.num_steps // self.action_interval_steps

    def run(self, actions: Sequence[Any], action_model: ActionModel) -> EpisodeResult:
        """Run one action sequence, stopping at the first failure."""

        if len(actions) > self.decision_horizon:
            raise ValueError("Action sequence is longer than this scenario's horizon.")

        demand = self.nominal_demand.copy()
        vsl_speeds = np.full(
            (self.num_steps, self.num_segments), self.unrestricted_speed, dtype=float
        )
        for decision, action in enumerate(actions):
            start = decision * self.action_interval_steps
            end = min(start + self.action_interval_steps, self.num_steps)
            action_model.apply_block(action, demand, vsl_speeds, start, end)

        state = TrafficState(
            density=np.asarray(self.initial_state.density, dtype=float).copy(),
            velocity=np.asarray(self.initial_state.velocity, dtype=float).copy(),
            flow_origin=float(self.initial_state.flow_origin),
            queue=float(self.initial_state.queue),
            time_index=0,
        )
        density_history = [state.density.copy()]
        velocity_history = [state.velocity.copy()]
        raw_density_history: List[np.ndarray] = []
        raw_velocity_history: List[np.ndarray] = []

        log_likelihood = 0.0
        intervention_cost = 0.0
        activated_decisions = 0
        closest: Optional[FailureAssessment] = None
        failure: Optional[FailureAssessment] = None
        simulated_steps = 0

        # Treat the initial condition as part of the nominal safety gate. Raw
        # and reported values are identical because no transition has occurred.
        initial_diagnostics = {
            "raw_density": state.density,
            "raw_velocity": state.velocity,
        }
        initial_assessment = assess_physical_failure(
            state, initial_diagnostics, self.failure_config
        )
        if initial_assessment.failed:
            failure = initial_assessment
        else:
            closest = initial_assessment

        for t in range(min(self.num_steps, len(actions) * self.action_interval_steps)):
            if failure is not None:
                break
            decision = t // self.action_interval_steps
            if decision >= activated_decisions:
                action = actions[decision]
                probability = action_model.action_probability(action)
                if probability <= 0:
                    raise ValueError("Action probabilities must be positive.")
                log_likelihood += math.log(probability)
                intervention_cost += action_model.intervention_cost(action)
                activated_decisions += 1

            step_result = metanet_step(
                t,
                state.density,
                state.velocity,
                state.queue,
                state.flow_origin,
                T=self.T,
                l=self.l,
                vsl_speeds=vsl_speeds,
                demand=demand,
                downstream_density=self.downstream_density,
                params=self.params,
                lanes=self.lanes,
                real_data=True,
                return_diagnostics=True,
            )
            density, velocity, queue, flow_origin, _flow, diagnostics = step_result
            simulated_steps = t + 1
            state = TrafficState(
                density=density,
                velocity=velocity,
                flow_origin=float(flow_origin),
                queue=float(queue),
                time_index=simulated_steps,
            )

            density_history.append(density.copy())
            velocity_history.append(velocity.copy())
            raw_density_history.append(diagnostics["raw_density"].copy())
            raw_velocity_history.append(diagnostics["raw_velocity"].copy())

            assessment = assess_physical_failure(state, diagnostics, self.failure_config)
            if (
                not assessment.failed
                and (closest is None or assessment.signed_margin > closest.signed_margin)
            ):
                closest = assessment
            if assessment.failed:
                failure = assessment
                break

        if closest is None and failure is None:
            raise ValueError("At least one action interval is required to evaluate an episode.")

        return EpisodeResult(
            actions=tuple(actions[:activated_decisions]),
            action_labels=tuple(
                action_model.action_label(action) for action in actions[:activated_decisions]
            ),
            failed=failure is not None,
            failure=failure,
            closest_assessment=closest,
            log_likelihood=log_likelihood,
            intervention_cost=intervention_cost,
            simulated_steps=simulated_steps,
            time_step_hours=self.T,
            density_history=np.asarray(density_history),
            velocity_history=np.asarray(velocity_history),
            raw_density_history=(
                np.asarray(raw_density_history)
                if raw_density_history
                else np.empty((0, self.num_segments))
            ),
            raw_velocity_history=(
                np.asarray(raw_velocity_history)
                if raw_velocity_history
                else np.empty((0, self.num_segments))
            ),
        )

    def run_nominal(self) -> EpisodeResult:
        """Run the required no-disturbance safety check."""

        model = NominalActionModel()
        return self.run([model.actions[0]] * self.decision_horizon, model)


# ---------------------------------------------------------------------------
# Generic finite-action Monte Carlo tree search
# ---------------------------------------------------------------------------


class SearchObjective(str, Enum):
    STANDARD_AST = "standard_ast"
    MIN_INTERVENTION = "minimum_intervention"


@dataclass(frozen=True)
class MCTSConfig:
    iterations: int = 2000
    exploration_constant: float = math.sqrt(2.0)
    failure_bonus: float = 1000.0
    miss_distance_weight: float = 100.0
    seed: int = 1


@dataclass
class _MCTSNode:
    action_sequence: Tuple[Any, ...]
    all_actions: Tuple[Any, ...]
    parent: Optional["_MCTSNode"] = None
    visits: int = 0
    total_reward: float = 0.0
    terminal: bool = False
    children: List["_MCTSNode"] = field(default_factory=list)
    untried_actions: List[Any] = field(init=False)

    def __post_init__(self) -> None:
        self.untried_actions = list(self.all_actions)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else -math.inf

    def best_ucb_child(self, exploration_constant: float) -> "_MCTSNode":
        if not self.children:
            raise ValueError("Cannot select a child from a leaf node.")

        log_parent_visits = math.log(max(1, self.visits))

        def ucb(child: _MCTSNode) -> float:
            if child.visits == 0:
                return math.inf
            exploration = exploration_constant * math.sqrt(
                log_parent_visits / child.visits
            )
            return child.mean_reward + exploration

        return max(self.children, key=ucb)


@dataclass
class SearchResult:
    objective: SearchObjective
    iterations: int
    best_episode: EpisodeResult
    best_failure: Optional[EpisodeResult]
    root_action_statistics: Tuple[Dict[str, Any], ...]

    def summary(self) -> Dict[str, Any]:
        return {
            "objective": self.objective.value,
            "iterations": self.iterations,
            "found_failure": self.best_failure is not None,
            "best_episode": self.best_episode.summary(),
            "root_action_statistics": list(self.root_action_statistics),
        }


def episode_reward(
    episode: EpisodeResult,
    objective: SearchObjective,
    config: MCTSConfig,
) -> float:
    """Scalar reward used for MCTS backups.

    Successful standard AST episodes retain their trajectory log-likelihood;
    minimum-intervention episodes retain the negative intervention cost.
    Non-failures also receive a smooth miss-distance penalty.
    """

    path_score = (
        episode.log_likelihood
        if objective == SearchObjective.STANDARD_AST
        else -episode.intervention_cost
    )
    if episode.failed:
        return config.failure_bonus + path_score
    if episode.closest_nonfailure_distance is None:
        raise ValueError("A non-failing episode must have a distance-to-failure.")
    return path_score - config.miss_distance_weight * episode.closest_nonfailure_distance


def _is_better_episode(
    candidate: EpisodeResult,
    incumbent: Optional[EpisodeResult],
    objective: SearchObjective,
) -> bool:
    """Rank final reported episodes without depending on reward scale."""

    if incumbent is None:
        return True
    if candidate.failed != incumbent.failed:
        return candidate.failed

    if candidate.failed:
        if objective == SearchObjective.STANDARD_AST:
            if not math.isclose(candidate.log_likelihood, incumbent.log_likelihood):
                return candidate.log_likelihood > incumbent.log_likelihood
        elif not math.isclose(
            candidate.intervention_cost, incumbent.intervention_cost
        ):
            return candidate.intervention_cost < incumbent.intervention_cost
        return candidate.simulated_steps < incumbent.simulated_steps

    if candidate.closest_nonfailure_distance is None:
        return False
    if incumbent.closest_nonfailure_distance is None:
        return True
    if not math.isclose(
        candidate.closest_nonfailure_distance,
        incumbent.closest_nonfailure_distance,
    ):
        return candidate.closest_nonfailure_distance < incumbent.closest_nonfailure_distance
    if objective == SearchObjective.STANDARD_AST:
        return candidate.log_likelihood > incumbent.log_likelihood
    return candidate.intervention_cost < incumbent.intervention_cost


def _sample_rollout_action(
    action_model: ActionModel,
    objective: SearchObjective,
    rng: np.random.Generator,
) -> Any:
    actions = action_model.actions
    if objective == SearchObjective.STANDARD_AST:
        probabilities = np.array(
            [action_model.action_probability(action) for action in actions], dtype=float
        )
        probabilities /= probabilities.sum()
        return actions[int(rng.choice(len(actions), p=probabilities))]

    # Uniform rollouts keep the minimum-intervention search exploratory. The
    # backed-up reward still prefers low-cost failures.
    return actions[int(rng.integers(len(actions)))]


def run_mcts(
    environment: ASTEnvironment,
    action_model: ActionModel,
    objective: SearchObjective,
    config: MCTSConfig = MCTSConfig(),
) -> SearchResult:
    """Search finite action sequences with a compact UCT implementation."""

    if config.iterations <= 0:
        raise ValueError("MCTS iterations must be positive.")
    if not action_model.actions:
        raise ValueError("MCTS requires at least one action.")

    rng = np.random.default_rng(config.seed)
    root = _MCTSNode(action_sequence=(), all_actions=action_model.actions)
    best_episode: Optional[EpisodeResult] = None
    best_failure: Optional[EpisodeResult] = None

    for _ in range(config.iterations):
        node = root

        # Selection: descend through fully expanded prefixes using UCB.
        while (
            len(node.action_sequence) < environment.decision_horizon
            and not node.terminal
            and not node.untried_actions
            and node.children
        ):
            node = node.best_ucb_child(config.exploration_constant)

        # Expansion: add one unvisited action to the selected prefix.
        if (
            len(node.action_sequence) < environment.decision_horizon
            and not node.terminal
            and node.untried_actions
        ):
            action_index = int(rng.integers(len(node.untried_actions)))
            action = node.untried_actions.pop(action_index)
            child = _MCTSNode(
                action_sequence=node.action_sequence + (action,),
                all_actions=action_model.actions,
                parent=node,
            )
            node.children.append(child)
            node = child

        # Rollout: complete the sequence with the action model's sampling rule.
        rollout_actions = list(node.action_sequence)
        while not node.terminal and len(rollout_actions) < environment.decision_horizon:
            rollout_actions.append(
                _sample_rollout_action(action_model, objective, rng)
            )

        episode = environment.run(rollout_actions, action_model)
        reward = episode_reward(episode, objective, config)

        # If a rollout failed before every action in the selected prefix became
        # active, attach the result to the matching earlier prefix. This avoids
        # crediting actions that occur after termination. A failure occurring
        # later in the random rollout must not mark the shorter tree prefix as
        # terminal; deeper actions were required to cause it.
        backup_node = node
        if episode.failed and len(episode.actions) <= len(node.action_sequence):
            while len(backup_node.action_sequence) > len(episode.actions):
                if backup_node.parent is None:
                    break
                backup_node = backup_node.parent
            backup_node.terminal = True

        # Backup the same terminal rollout result through its active prefixes.
        cursor: Optional[_MCTSNode] = backup_node
        while cursor is not None:
            cursor.visits += 1
            cursor.total_reward += reward
            cursor = cursor.parent

        if _is_better_episode(episode, best_episode, objective):
            best_episode = episode
        if episode.failed and _is_better_episode(episode, best_failure, objective):
            best_failure = episode

    if best_episode is None:
        raise RuntimeError("MCTS completed without evaluating an episode.")

    root_statistics = tuple(
        {
            "action": action_model.action_label(child.action_sequence[0]),
            "visits": child.visits,
            "mean_reward": child.mean_reward,
        }
        for child in sorted(root.children, key=lambda item: item.visits, reverse=True)
    )
    return SearchResult(
        objective=objective,
        iterations=config.iterations,
        best_episode=best_episode,
        best_failure=best_failure,
        root_action_statistics=root_statistics,
    )
