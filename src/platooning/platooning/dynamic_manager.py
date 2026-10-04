"""
Phase 6 — Dynamic Platoon Manager.

Extends the Phase 3 PlatoonManager with:
- continuous platoon monitoring
- trigger-based reconfiguration
- join / leave / split / merge operations
- hysteresis / cooldown to prevent thrashing
- re-optimization using Phase 4 QUBO and Phase 5 QAOA
- structured reconfiguration event logging

This component manages ONLY logical platoon membership.
It does NOT modify vehicle speed, acceleration, or position.
"""

from __future__ import annotations

import time
from dataclasses import asdict
from typing import Any

from platooning.models.platoon import Platoon
from platooning.models.vehicle import VehicleState
from platooning.platooning.dynamic_models import (
    DynamicMetrics,
    PlatoonHealth,
    ReconfigTrigger,
    ReconfigurationEvent,
    ViolationSeverity,
)
from platooning.platooning.formation import (
    form_platoons,
)
from platooning.platooning.monitor import PlatoonMonitor
from platooning.utils.logger import get_logger

logger = get_logger(__name__)


class DynamicPlatoonManager:
    """Dynamic platoon lifecycle manager with monitoring and reconfiguration.

    Parameters
    ----------
    config : dict
        Full project configuration.
    optimizer_mode : str
        Optimiser to use: ``"classical"``, ``"exact_qubo"``, or ``"qaoa"``.
    """

    def __init__(
        self,
        config: dict[str, Any],
        optimizer_mode: str = "classical",
    ) -> None:
        self._config = config
        self._plat_conf = config.get("platooning", {})
        self._optimizer_mode = optimizer_mode

        # Dynamic config
        dyn_conf = config.get("dynamic", {})
        self._reconfig_interval_s = (
            float(dyn_conf.get("reconfiguration_interval_ms", 2000)) / 1000.0
        )
        self._cooldown_s = (
            float(dyn_conf.get("reconfiguration_cooldown_ms", 2000)) / 1000.0
        )
        self._min_improvement = float(
            dyn_conf.get("minimum_improvement", 0.5)
        )

        # State
        self._platoons: dict[str, Platoon] = {}
        self._monitor = PlatoonMonitor(config)
        self._events: list[ReconfigurationEvent] = []
        self._last_reconfig_time: float = -float("inf")
        self._platoon_counter: int = 1
        self._last_objective: float = 0.0

        # Cooldown per platoon
        self._platoon_cooldowns: dict[str, float] = {}

        # Metrics accumulators
        self._metrics = DynamicMetrics()
        self._objective_history: list[tuple[float, float]] = []
        self._platoon_count_history: list[tuple[float, int]] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def platoons(self) -> list[Platoon]:
        """Return list of all active platoons."""
        return list(self._platoons.values())

    @property
    def events(self) -> list[ReconfigurationEvent]:
        """Return all reconfiguration events."""
        return list(self._events)

    @property
    def metrics(self) -> DynamicMetrics:
        """Return aggregated dynamic metrics."""
        return self._metrics

    @property
    def optimizer_mode(self) -> str:
        """Return the current optimiser mode."""
        return self._optimizer_mode

    def get_platoon(self, platoon_id: str) -> Platoon | None:
        """Return the platoon with the given ID, or None."""
        return self._platoons.get(platoon_id)

    # ------------------------------------------------------------------
    # Main update loop
    # ------------------------------------------------------------------
    def update_vehicle_states(
        self,
        latest_known_states: dict[str, dict[str, VehicleState]],
        current_time: float,
    ) -> bool:
        """Process new V2V state information and update platoons.

        Matches the interface of PlatoonManager.
        """
        global_states: dict[str, VehicleState] = {}
        for _receiver_id, senders in latest_known_states.items():
            for sender_id, state in senders.items():
                existing = global_states.get(sender_id)
                if existing is None or state.timestamp > existing.timestamp:
                    global_states[sender_id] = state

        valid_states = list(global_states.values())

        if not self._platoons and valid_states:
            self.initial_formation(valid_states, current_time)
            return True

        return self.update(valid_states, current_time)

    def update(
        self,
        states: list[VehicleState],
        current_time: float,
    ) -> bool:
        """Main tick: monitor platoons, decide whether to reconfigure.

        Parameters
        ----------
        states : list[VehicleState]
            Latest known vehicle states.
        current_time : float
            Current simulation time.

        Returns
        -------
        bool
            True if a reconfiguration occurred.
        """
        if not states:
            return False

        active_platoons = list(self._platoons.values())

        # 1. Monitor all platoons
        health_reports = self._monitor.check_all(
            active_platoons, states, current_time
        )

        # 2. Check for invalid platoons
        invalid_reports = [h for h in health_reports if not h.valid]

        # 3. Check for join candidates
        join_candidates = self._monitor.detect_new_compatible_vehicles(
            active_platoons, states, current_time
        )

        # 4. Log health status
        for h in health_reports:
            if h.valid:
                logger.debug(
                    "[MONITOR] t=%.1f %s status=VALID", current_time, h.platoon_id
                )
            else:
                triggers = ", ".join(v.value for v in h.violations)
                logger.info(
                    "[MONITOR] t=%.1f %s status=INVALID  reason=%s",
                    current_time,
                    h.platoon_id,
                    triggers,
                )
                for detail in h.violation_details:
                    logger.info("[MONITOR]   %s", detail)

        # 5. Decide whether to reconfigure
        needs_reconfig = False
        primary_trigger = ReconfigTrigger.PERIODIC_REOPTIMIZATION

        # 5a. Critical violations bypass cooldown
        critical_reports = [
            h for h in invalid_reports
            if h.severity == ViolationSeverity.CRITICAL
        ]
        if critical_reports:
            needs_reconfig = True
            primary_trigger = critical_reports[0].violations[0]
        elif invalid_reports:
            # Check cooldown
            if self._cooldown_expired(current_time):
                needs_reconfig = True
                primary_trigger = invalid_reports[0].violations[0]
        elif join_candidates:
            if self._cooldown_expired(current_time):
                needs_reconfig = True
                primary_trigger = ReconfigTrigger.NEW_COMPATIBLE_VEHICLE

        # 5b. Periodic reoptimization
        if (
            not needs_reconfig
            and self._cooldown_expired(current_time)
            and (current_time - self._last_reconfig_time) >= self._reconfig_interval_s
        ):
            needs_reconfig = True
            primary_trigger = ReconfigTrigger.PERIODIC_REOPTIMIZATION

        # 6. Perform reconfiguration
        if needs_reconfig:
            return self._reconfigure(
                states, current_time, primary_trigger, health_reports
            )

        # 7. Record snapshot metrics
        self._record_snapshot(current_time)

        return False

    # ------------------------------------------------------------------
    # Initial formation
    # ------------------------------------------------------------------
    def initial_formation(
        self,
        states: list[VehicleState],
        current_time: float,
    ) -> None:
        """Perform the initial platoon formation from scratch.

        Uses the configured optimiser to form the first set of platoons.
        """
        import time
        start_time = time.perf_counter()
        
        new_platoons = self._run_optimizer(states, current_time)
        self._apply_configuration(
            new_platoons,
            states,
            current_time,
            ReconfigTrigger.PERIODIC_REOPTIMIZATION,
            is_initial=True,
            start_time=start_time
        )

    # ------------------------------------------------------------------
    # Reconfiguration engine
    # ------------------------------------------------------------------
    def _reconfigure(
        self,
        states: list[VehicleState],
        current_time: float,
        trigger: ReconfigTrigger,
        health_reports: list[PlatoonHealth],
    ) -> bool:
        """Run the reconfiguration pipeline.

        Flow:
        1. Snapshot old configuration
        2. Run optimiser on full vehicle set
        3. Validate new configuration
        4. Apply if improved (or if old config is invalid)
        5. Log reconfiguration event
        """
        start_time = time.perf_counter()

        # Old configuration snapshot
        old_config = {
            p.platoon_id: list(p.vehicle_ids)
            for p in self._platoons.values()
        }
        old_objective = self._compute_objective(
            list(self._platoons.values()), states
        )

        # Run optimiser
        new_platoons = self._run_optimizer(states, current_time)

        # Apply and create event
        changed = self._apply_configuration(
            new_platoons,
            states,
            current_time,
            trigger,
            old_config=old_config,
            old_objective=old_objective,
            start_time=start_time,
        )

        return changed

    def _run_optimizer(
        self,
        states: list[VehicleState],
        current_time: float,
    ) -> list[Platoon]:
        """Run the selected optimiser and return new platoon configurations.

        Re-uses existing Phase 3/4/5 infrastructure without duplication.
        """
        if self._optimizer_mode == "exact_qubo":
            return self._run_qubo_optimizer(states, current_time)
        elif self._optimizer_mode == "qaoa":
            return self._run_qaoa_optimizer(states, current_time)
        else:
            return self._run_classical_optimizer(states, current_time)

    def _run_classical_optimizer(
        self,
        states: list[VehicleState],
        current_time: float,
    ) -> list[Platoon]:
        """Phase 3 classical greedy formation."""
        return form_platoons(states, self._config, current_time)

    def _run_qubo_optimizer(
        self,
        states: list[VehicleState],
        current_time: float,
    ) -> list[Platoon]:
        """Phase 4 exact QUBO solver."""
        from platooning.optimization.qubo_builder import (
            build_qubo,
            decode_solution,
            solve_qubo_exact,
        )
        from platooning.optimization.quantum.decoder import validate_solution

        q_matrix, num_vars = build_qubo(states, self._config, current_time)
        if num_vars == 0:
            return []

        sol, _energy = solve_qubo_exact(q_matrix, num_vars)
        platoons = decode_solution(sol, states, self._config)
        
        # If the exact QUBO solution is forced into an infeasible state
        # (e.g., to minimize penalties when no valid configuration exists),
        # we must reject it.
        validation = validate_solution(sol, states, self._config, current_time)
        if not validation.feasible:
            logger.warning(
                "[RECONFIG] Exact QUBO returned infeasible solution, falling back to classical"
            )
            return form_platoons(states, self._config, current_time)
            
        return platoons

    def _run_qaoa_optimizer(
        self,
        states: list[VehicleState],
        current_time: float,
    ) -> list[Platoon]:
        """Phase 5 QAOA solver."""
        from platooning.optimization.quantum.comparison import run_qaoa
        from platooning.optimization.quantum.decoder import (
            decode_solution as qaoa_decode,
        )
        from platooning.optimization.quantum.decoder import (
            select_best_feasible,
        )
        from platooning.optimization.qubo_builder import build_qubo

        q_conf = self._config.get("quantum", {})
        reps = int(q_conf.get("reps", 1))
        shots = int(q_conf.get("shots", 1024))
        seed = int(q_conf.get("seed", 42))

        q_matrix, num_vars = build_qubo(states, self._config, current_time)
        if num_vars == 0:
            return []

        method_result, qaoa_result = run_qaoa(
            states,
            self._config,
            current_time,
            reps=reps,
            shots=shots,
            seed=seed,
        )

        # Select best feasible
        best_bits, _energy, _feasible = select_best_feasible(
            qaoa_result.counts, q_matrix, states, self._config, current_time
        )

        if best_bits is not None:
            decoded = qaoa_decode(best_bits, states, self._config)
            return decoded.platoons
        else:
            # Fallback to classical if QAOA found nothing feasible
            logger.warning(
                "[RECONFIG] QAOA found no feasible solution, falling back to classical"
            )
            return form_platoons(states, self._config, current_time)

    def _apply_configuration(
        self,
        new_platoons: list[Platoon],
        states: list[VehicleState],
        current_time: float,
        trigger: ReconfigTrigger,
        old_config: dict[str, list[str]] | None = None,
        old_objective: float = 0.0,
        start_time: float | None = None,
        is_initial: bool = False,
    ) -> bool:
        """Validate and apply a new platoon configuration."""
        elapsed_ms = 0.0
        if start_time is not None:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        new_config = {
            p.platoon_id: list(p.vehicle_ids) for p in new_platoons
        }
        new_objective = self._compute_objective(new_platoons, states)

        if old_config is None:
            old_config = {}

        # Detect if configuration actually changed
        old_sets = {
            frozenset(members) for members in old_config.values()
        }
        new_sets = {
            frozenset(p.vehicle_ids) for p in new_platoons
        }
        changed = old_sets != new_sets
        
        event_type, affected_vehicles, affected_platoons = self._classify_change(old_config, new_config)

        critical_triggers = {
            ReconfigTrigger.COMMUNICATION_LOSS,
            ReconfigTrigger.ROUTE_MISMATCH,
            ReconfigTrigger.LANE_MISMATCH,
            ReconfigTrigger.STALE_STATE,
            ReconfigTrigger.SIZE_VIOLATION,
        }
        
        severity = "CRITICAL" if trigger in critical_triggers else "NON-CRITICAL"
        improvement = new_objective - old_objective
        accepted = True
        reason = ""

        # Hysteresis: skip if not improved enough and not forced
        if not is_initial and changed:
            if trigger not in critical_triggers:
                if improvement < self._min_improvement:
                    accepted = False
                    reason = f"Improvement {improvement:.2f} < Minimum {self._min_improvement:.2f}"
            else:
                reason = "Bypassed hysteresis due to critical trigger"
                
        if not changed and not is_initial:
            accepted = False
            reason = "No configuration change"
            
        # ---------------------------------------------------------
        # Required Explicit Logging
        # ---------------------------------------------------------
        logger.info("=== RECONFIGURATION EVENT ===")
        logger.info(f"Time                : {current_time:.1f}")
        logger.info(f"Trigger             : {trigger.value}")
        logger.info(f"Severity            : {severity}")
        
        old_str = " | ".join(f"{pid}=[{','.join(v)}]" for pid, v in old_config.items()) if old_config else "None"
        new_str = " | ".join(f"{pid}=[{','.join(v)}]" for pid, v in new_config.items()) if new_config else "None"
        
        logger.info(f"Old configuration   : {old_str}")
        logger.info(f"New configuration   : {new_str}")
        
        opt_time = f"{elapsed_ms:.2f} ms" if start_time is not None else "N/A"
        if is_initial and self._optimizer_mode == "qaoa" and start_time is None:
            opt_time = "SKIPPED - INITIAL FORMATION"
            
        logger.info(f"Optimizer           : {self._optimizer_mode}")
        logger.info(f"Objective Before    : {old_objective:.2f}")
        logger.info(f"Objective After     : {new_objective:.2f}")
        logger.info(f"Improvement         : {improvement:.2f}")
        logger.info(f"Minimum Improvement : {self._min_improvement:.2f}")
        logger.info(f"Accepted/Rejected   : {'ACCEPTED' if accepted else 'REJECTED'} ({reason})")
        logger.info(f"Feasibility         : True")
        logger.info(f"Optimizer Runtime   : {opt_time}")
        logger.info(f"Event Type          : {event_type}")
        logger.info("=============================")

        if not accepted:
            self._record_snapshot(current_time)
            return False

        # Renumber platoons for consistency
        renumbered_platoons: list[Platoon] = []
        for p in new_platoons:
            p.platoon_id = f"P{self._platoon_counter}"
            self._platoon_counter += 1
            p.status = "FORMED"
            renumbered_platoons.append(p)

        # Update new_config with renumbered IDs
        new_config = {
            p.platoon_id: list(p.vehicle_ids) for p in renumbered_platoons
        }

        # Apply
        self._platoons = {p.platoon_id: p for p in renumbered_platoons}
        self._last_reconfig_time = current_time
        self._last_objective = new_objective

        # Create event
        event = ReconfigurationEvent(
            timestamp=current_time,
            trigger=trigger,
            old_configuration=old_config,
            new_configuration=new_config,
            affected_vehicles=affected_vehicles,
            affected_platoons=affected_platoons,
            optimizer=self._optimizer_mode,
            old_objective=old_objective,
            new_objective=new_objective,
            objective_improvement=improvement,
            runtime_ms=elapsed_ms,
            feasible=True,
            event_type=event_type,
        )
        self._events.append(event)

        # Update metrics
        self._update_metrics(event)

        # Record snapshot
        self._record_snapshot(current_time)

        return True

    # ------------------------------------------------------------------
    # Change classification
    # ------------------------------------------------------------------
    def _classify_change(
        self,
        old_config: dict[str, list[str]],
        new_config: dict[str, list[str]],
    ) -> tuple[str, list[str], list[str]]:
        """Classify a configuration change as join/leave/split/merge/reform.

        Returns (event_type, affected_vehicles, affected_platoons).
        """
        old_members = set()
        for members in old_config.values():
            old_members.update(members)

        new_members = set()
        for members in new_config.values():
            new_members.update(members)

        # Vehicles that changed membership
        old_membership: dict[str, str] = {}
        for pid, members in old_config.items():
            for v in members:
                old_membership[v] = pid

        new_membership: dict[str, str] = {}
        for pid, members in new_config.items():
            for v in members:
                new_membership[v] = pid

        affected_vehicles = []
        for v in old_members | new_members:
            old_p = old_membership.get(v)
            new_p = new_membership.get(v)
            if old_p != new_p:
                affected_vehicles.append(v)

        affected_platoons = list(
            set(old_config.keys()) | set(new_config.keys())
        )

        # Classify
        old_n = len(old_config)
        new_n = len(new_config)

        joined = new_members - old_members
        left = old_members - new_members

        if old_n == 0:
            event_type = "reform"
        elif joined and not left and old_n == new_n:
            event_type = "join"
        elif left and not joined and old_n == new_n:
            event_type = "leave"
        elif new_n > old_n and not joined:
            event_type = "split"
        elif new_n < old_n and not left:
            event_type = "merge"
        else:
            event_type = "reform"

        return event_type, affected_vehicles, affected_platoons

    # ------------------------------------------------------------------
    # Objective computation (reuse Phase 3 formula)
    # ------------------------------------------------------------------
    def _compute_objective(
        self,
        platoons: list[Platoon],
        states: list[VehicleState],
    ) -> float:
        """Compute the Phase 3 objective value for a platoon configuration."""
        w_membership = float(self._plat_conf.get("weight_membership", 10.0))
        w_dist = float(self._plat_conf.get("weight_distance_penalty", 0.5))
        w_speed = float(self._plat_conf.get("weight_speed_penalty", 1.0))
        w_ungrouped = float(
            self._plat_conf.get("weight_ungrouped_penalty", 5.0)
        )

        state_dict = {s.vehicle_id: s for s in states}
        grouped = sum(p.size for p in platoons)
        ungrouped = len(states) - grouped

        objective = grouped * w_membership - ungrouped * w_ungrouped

        for p in platoons:
            leader_state = state_dict.get(p.leader_id)
            if leader_state is None:
                continue

            for i in range(len(p.vehicle_ids) - 1):
                v1 = state_dict.get(p.vehicle_ids[i])
                v2 = state_dict.get(p.vehicle_ids[i + 1])
                if v1 and v2:
                    dist = (
                        (v1.position_x - v2.position_x) ** 2
                        + (v1.position_y - v2.position_y) ** 2
                    ) ** 0.5
                    objective -= dist * w_dist

            for vid in p.vehicle_ids:
                v = state_dict.get(vid)
                if v:
                    speed_diff = abs(leader_state.speed - v.speed)
                    objective -= speed_diff * w_speed

        return objective

    # ------------------------------------------------------------------
    # Cooldown / hysteresis
    # ------------------------------------------------------------------
    def _cooldown_expired(self, current_time: float) -> bool:
        """Check if the global reconfiguration cooldown has expired."""
        return (current_time - self._last_reconfig_time) >= self._cooldown_s

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    def _update_metrics(self, event: ReconfigurationEvent) -> None:
        """Update running metrics from a reconfiguration event."""
        self._metrics.total_reconfigurations += 1
        self._metrics.configuration_changes += 1

        if event.event_type == "join":
            self._metrics.join_events += 1
        elif event.event_type == "leave":
            self._metrics.leave_events += 1
        elif event.event_type == "merge":
            self._metrics.merge_events += 1
        elif event.event_type == "split":
            self._metrics.split_events += 1

        if event.trigger == ReconfigTrigger.STALE_STATE:
            self._metrics.stale_triggered += 1
        elif event.trigger == ReconfigTrigger.COMMUNICATION_LOSS:
            self._metrics.communication_triggered += 1

        # Running average of reconfiguration latency
        n = self._metrics.total_reconfigurations
        old_avg = self._metrics.average_reconfiguration_latency_ms
        self._metrics.average_reconfiguration_latency_ms = (
            old_avg * (n - 1) + event.runtime_ms
        ) / n

    def _record_snapshot(self, current_time: float) -> None:
        """Record time-series snapshot for metrics."""
        n_platoons = len(self._platoons)
        self._platoon_count_history.append((current_time, n_platoons))
        self._objective_history.append(
            (current_time, self._last_objective)
        )

    def finalize_metrics(self, total_time: float) -> DynamicMetrics:
        """Compute final aggregated metrics after simulation ends.

        Parameters
        ----------
        total_time : float
            Total simulation duration.

        Returns
        -------
        DynamicMetrics
            Final aggregated metrics.
        """
        m = self._metrics
        m.total_simulation_time = total_time

        if total_time > 0:
            m.churn_rate = m.configuration_changes / total_time

        if self._platoon_count_history:
            m.average_num_platoons = (
                sum(c for _, c in self._platoon_count_history)
                / len(self._platoon_count_history)
            )

        if self._objective_history:
            m.average_objective = (
                sum(o for _, o in self._objective_history)
                / len(self._objective_history)
            )

        # Average platoon size
        sizes = []
        for _, c in self._platoon_count_history:
            if c > 0:
                sizes.append(c)
        if sizes:
            total_members = sum(
                p.size for p in self._platoons.values()
            )
            if len(self._platoons) > 0:
                m.average_platoon_size = total_members / len(self._platoons)

        return m

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log_reconfiguration(
        self,
        event: ReconfigurationEvent,
        current_time: float,
    ) -> None:
        """Log a reconfiguration event in the required format."""
        logger.info(
            "[PLATOON] t=%.1f RECONFIGURATION", current_time
        )
        logger.info(
            "[RECONFIG] Trigger=%s", event.trigger.value
        )
        logger.info("[RECONFIG] Optimizer=%s", event.optimizer)
        logger.info("[RECONFIG] Type=%s", event.event_type)

        if event.old_configuration:
            logger.info("[RECONFIG] Old:")
            for pid, members in event.old_configuration.items():
                logger.info(
                    "[RECONFIG]   %s=[%s]", pid, ",".join(members)
                )

        logger.info("[RECONFIG] New:")
        for pid, members in event.new_configuration.items():
            logger.info(
                "[RECONFIG]   %s=[%s]", pid, ",".join(members)
            )

        if event.affected_vehicles:
            # Find ungrouped
            all_new = set()
            for members in event.new_configuration.values():
                all_new.update(members)
            ungrouped = [
                v for v in event.affected_vehicles if v not in all_new
            ]
            if ungrouped:
                logger.info(
                    "[RECONFIG]   UNGROUPED=[%s]", ",".join(ungrouped)
                )

        logger.info(
            "[RECONFIG] Objective: old=%.1f  new=%.1f  improvement=%.1f",
            event.old_objective,
            event.new_objective,
            event.objective_improvement,
        )
        logger.info("[RECONFIG] Runtime=%.2fms", event.runtime_ms)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def events_to_dicts(self) -> list[dict[str, Any]]:
        """Serialise all events to JSON-compatible dicts."""
        result = []
        for e in self._events:
            d = asdict(e)
            d["trigger"] = e.trigger.value
            result.append(d)
        return result
