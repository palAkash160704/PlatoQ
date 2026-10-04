"""
Phase 5 QAOA Experiment — Reproducible comparison of Classical vs Exact QUBO vs QAOA.

Runs identical vehicle scenarios through all three approaches and saves
structured results.

Usage:
    python experiments/phase5_qaoa_experiment.py

Results are saved to:
    experiments/results/phase5/

IMPORTANT: This experiment does NOT claim quantum advantage.
All QAOA runs use a local simulator, and runtimes include classical
parameter optimisation and simulator overhead.
"""

import json
import os
import sys
from pathlib import Path

# Ensure src is in PYTHONPATH
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from platooning.models.vehicle import VehicleState
from platooning.optimization.quantum.comparison import (
    ComparisonResult,
    compare_all,
    comparison_to_dict,
)
from platooning.optimization.quantum.decoder import analyze_samples
from platooning.optimization.quantum.qaoa_solver import QAOASolver
from platooning.optimization.qubo_builder import build_qubo


# ======================================================================
# Scenario Definitions
# ======================================================================
def scenario_a_3vehicles() -> tuple[str, list[VehicleState], dict]:
    """Scenario A: 3 compatible vehicles close together."""
    states = [
        VehicleState(
            vehicle_id="V1",
            timestamp=10.0,
            position_x=100.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        ),
        VehicleState(
            vehicle_id="V2",
            timestamp=10.0,
            position_x=95.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        ),
        VehicleState(
            vehicle_id="V3",
            timestamp=10.0,
            position_x=90.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        ),
    ]
    return "3_vehicle_compatible", states, _default_config()


def scenario_b_4vehicles() -> tuple[str, list[VehicleState], dict]:
    """Scenario B: 4 compatible vehicles."""
    states = [
        VehicleState(
            vehicle_id=f"V{i}",
            timestamp=10.0,
            position_x=100.0 - i * 8.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        )
        for i in range(1, 5)
    ]
    return "4_vehicle_compatible", states, _default_config()


def scenario_c_5vehicles() -> tuple[str, list[VehicleState], dict]:
    """Scenario C: 5 compatible vehicles (baseline)."""
    states = [
        VehicleState(
            vehicle_id=f"V{i}",
            timestamp=10.0,
            position_x=100.0 - i * 6.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        )
        for i in range(1, 6)
    ]
    return "5_vehicle_baseline", states, _default_config()


def scenario_d_5vehicles_two_platoons() -> tuple[str, list[VehicleState], dict]:
    """Scenario D: 5 vehicles that should form two platoons (distance split)."""
    config = _default_config()
    config["platooning"]["max_formation_distance_m"] = 25.0

    states = [
        # Group 1: V1, V2
        VehicleState(
            vehicle_id="V1",
            timestamp=10.0,
            position_x=100.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        ),
        VehicleState(
            vehicle_id="V2",
            timestamp=10.0,
            position_x=90.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        ),
        # Group 2: V3, V4, V5 (far from group 1)
        VehicleState(
            vehicle_id="V3",
            timestamp=10.0,
            position_x=40.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        ),
        VehicleState(
            vehicle_id="V4",
            timestamp=10.0,
            position_x=32.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        ),
        VehicleState(
            vehicle_id="V5",
            timestamp=10.0,
            position_x=24.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        ),
    ]
    return "5_vehicle_two_platoons", states, config


def scenario_e_5vehicles_incompatible() -> tuple[str, list[VehicleState], dict]:
    """Scenario E: 5 vehicles with some incompatible (different routes)."""
    states = [
        VehicleState(
            vehicle_id="V1",
            timestamp=10.0,
            position_x=100.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        ),
        VehicleState(
            vehicle_id="V2",
            timestamp=10.0,
            position_x=94.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        ),
        VehicleState(
            vehicle_id="V3",
            timestamp=10.0,
            position_x=88.0,
            position_y=0.0,
            speed=30.0,
            route_id="r2",
            lane_id="l1",  # Different route
        ),
        VehicleState(
            vehicle_id="V4",
            timestamp=10.0,
            position_x=82.0,
            position_y=0.0,
            speed=30.0,
            route_id="r1",
            lane_id="l1",
        ),
        VehicleState(
            vehicle_id="V5",
            timestamp=10.0,
            position_x=76.0,
            position_y=0.0,
            speed=35.0,
            route_id="r1",
            lane_id="l2",  # Different lane
        ),
    ]
    return "5_vehicle_incompatible", states, _default_config()


def _default_config() -> dict:
    """Return the default experiment configuration."""
    return {
        "communication": {"stale_threshold_ms": 500.0},
        "platooning": {
            "enabled": True,
            "formation_interval_ms": 100,
            "max_formation_distance_m": 500.0,
            "max_speed_difference_mps": 10.0,
            "minimum_platoon_size": 2,
            "maximum_platoon_size": 5,
            "weight_membership": 10.0,
            "weight_distance_penalty": 0.5,
            "weight_speed_penalty": 1.0,
            "weight_ungrouped_penalty": 5.0,
        },
        "qubo": {
            "enabled": True,
            "solver": "exact",
            "max_exact_variables": 25,
            "weight_pairwise_distance": 0.1,
            "weight_pairwise_speed": 0.5,
            "penalty_assignment": 1000.0,
            "penalty_incompatibility": 1000.0,
            "penalty_size": 1000.0,
        },
        "quantum": {
            "backend": "aer_simulator",
            "shots": 1024,
            "reps": 1,
            "optimizer": "COBYLA",
            "seed": 42,
        },
    }


# ======================================================================
# Main Experiment
# ======================================================================
def run_experiment():
    """Run the complete Phase 5 experiment."""
    print("=" * 60)
    print("PHASE 5: QAOA EXPERIMENT")
    print("Classical Greedy vs Exact QUBO vs QAOA")
    print("=" * 60)
    print()

    # Output directory
    results_dir = Path(__file__).parent / "results" / "phase5"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    scenarios = [
        scenario_a_3vehicles,
        scenario_b_4vehicles,
        scenario_c_5vehicles,
        scenario_d_5vehicles_two_platoons,
        scenario_e_5vehicles_incompatible,
    ]

    all_results: list[dict] = []
    current_time = 10.1

    for scenario_fn in scenarios:
        name, states, config = scenario_fn()

        print(f"\n{'-' * 60}")
        print(f"Scenario: {name}")
        print(f"Vehicles: {len(states)}")

        q_matrix, num_vars = build_qubo(states, config, current_time)
        print(f"QUBO Variables: {num_vars}")
        print(f"{'-' * 60}")

        # Run comparison
        comparison = compare_all(
            states=states,
            config=config,
            current_time=current_time,
            scenario_name=name,
            qaoa_depths=[1, 2, 3],
            qaoa_shots=1024,
            qaoa_optimizer="COBYLA",
            qaoa_seed=42,
        )

        # Print results table
        _print_comparison(comparison)

        # Save to all_results
        result_dict = comparison_to_dict(comparison)
        all_results.append(result_dict)

        # Run detailed sample analysis for p=1
        print("\n  Sample Analysis (QAOA p=1, top 5):")
        solver = QAOASolver(reps=1, shots=1024, seed=42)
        qaoa_result = solver.solve(q_matrix, num_vars)
        analyses = analyze_samples(
            qaoa_result.counts, q_matrix, states, config, current_time, top_k=5
        )

        total_feasible = 0
        total_infeasible = 0
        for counts_str, count in qaoa_result.counts.items():
            bits = [int(b) for b in reversed(counts_str)]
            from platooning.optimization.quantum.decoder import validate_solution

            val = validate_solution(bits, states, config, current_time)
            if val.feasible:
                total_feasible += count
            else:
                total_infeasible += count

        total_shots = total_feasible + total_infeasible
        feas_rate = total_feasible / total_shots if total_shots > 0 else 0.0

        print(f"  Feasibility rate: {feas_rate:.1%} ({total_feasible}/{total_shots})")
        for a in analyses:
            feas_str = "Y" if a["feasible"] else "N"
            print(
                f"    {a['bitstring'][:20]:>20s}... : "
                f"count={a['count']:4d}  "
                f"energy={a['energy']:10.2f}  "
                f"feasible={feas_str}  "
                f"platoons={a['num_platoons']}"
            )

        result_dict["feasibility_rate"] = feas_rate

    # Save all results
    results_file = results_dir / "phase5_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {results_file}")

    # Generate plots
    _generate_plots(all_results, plots_dir)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


def _print_comparison(comp: ComparisonResult):
    """Print a formatted comparison table."""
    print(
        f"\n  {'Method':<20s} {'Objective':>10s} {'QUBO Energy':>12s} "
        f"{'Platoons':>8s} {'Feasible':>8s} {'Runtime':>10s}"
    )
    print(f"  {'-' * 70}")

    c = comp.classical
    print(
        f"  {'Classical':<20s} {c.objective:>10.2f} {'N/A':>12s} "
        f"{c.num_platoons:>8d} {'True':>8s} {c.runtime_seconds:>10.4f}s"
    )

    e = comp.exact_qubo
    print(
        f"  {'Exact QUBO':<20s} {e.objective:>10.2f} {e.qubo_energy:>12.2f} "
        f"{e.num_platoons:>8d} {str(e.feasible):>8s} {e.runtime_seconds:>10.4f}s"
    )

    for p, q in sorted(comp.qaoa_results.items()):
        label = f"QAOA p={p}"
        print(
            f"  {label:<20s} {q.objective:>10.2f} {q.qubo_energy:>12.2f} "
            f"{q.num_platoons:>8d} {str(q.feasible):>8s} {q.runtime_seconds:>10.4f}s"
        )

    # Metrics
    if comp.metrics:
        print("\n  Derived Metrics:")
        for key, m in comp.metrics.items():
            print(
                f"    {key}: obj_gap={m['objective_gap']:.2f}  "
                f"rel_gap={m['relative_objective_gap']:.4f}  "
                f"energy_gap={m['energy_gap']:.2f}  "
                f"feasible={m['feasible']}"
            )


def _generate_plots(all_results: list[dict], plots_dir: Path):
    """Generate comparison plots."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available — skipping plot generation.")
        return

    scenarios = [r["scenario"] for r in all_results]

    # 1. Objective comparison
    fig, ax = plt.subplots(figsize=(12, 6))

    classical_obj = [r["classical"]["objective"] for r in all_results]
    exact_obj = [r["exact_qubo"]["objective"] for r in all_results]

    x = range(len(scenarios))
    width = 0.15

    bars = [
        ax.bar(
            [xi - 2 * width for xi in x], classical_obj, width, label="Classical Greedy"
        ),
        ax.bar([xi - width for xi in x], exact_obj, width, label="Exact QUBO"),
    ]

    for p_idx, p in enumerate([1, 2, 3]):
        qaoa_obj = []
        for r in all_results:
            q = r["qaoa"].get(f"p{p}", {})
            qaoa_obj.append(q.get("objective", 0.0))
        bars.append(
            ax.bar(
                [xi + p_idx * width for xi in x], qaoa_obj, width, label=f"QAOA p={p}"
            )
        )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Objective Value")
    ax.set_title("Phase 5: Objective Value Comparison")
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "objective_comparison.png", dpi=150)
    plt.close()

    # 2. QAOA energy by depth
    fig, ax = plt.subplots(figsize=(10, 6))

    for p in [1, 2, 3]:
        energies = []
        for r in all_results:
            q = r["qaoa"].get(f"p{p}", {})
            energies.append(q.get("qubo_energy", float("inf")))
        ax.plot(scenarios, energies, marker="o", label=f"QAOA p={p}")

    exact_energies = [r["exact_qubo"]["qubo_energy"] for r in all_results]
    ax.plot(scenarios, exact_energies, marker="s", linestyle="--", label="Exact QUBO")

    ax.set_xlabel("Scenario")
    ax.set_ylabel("QUBO Energy")
    ax.set_title("Phase 5: QAOA Energy by Circuit Depth")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(plots_dir / "qaoa_energy_by_depth.png", dpi=150)
    plt.close()

    # 3. Runtime comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    exact_rt = [r["exact_qubo"]["runtime_seconds"] for r in all_results]
    ax.bar([xi - width for xi in x], exact_rt, width, label="Exact QUBO")

    for p_idx, p in enumerate([1, 2, 3]):
        qaoa_rt = []
        for r in all_results:
            q = r["qaoa"].get(f"p{p}", {})
            qaoa_rt.append(q.get("runtime_seconds", 0.0))
        ax.bar([xi + p_idx * width for xi in x], qaoa_rt, width, label=f"QAOA p={p}")

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title(
        "Phase 5: Runtime Comparison\n(Includes classical optimiser + simulator overhead)"
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(scenarios, rotation=45, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "runtime_comparison.png", dpi=150)
    plt.close()

    # 4. Feasibility rate
    if any("feasibility_rate" in r for r in all_results):
        fig, ax = plt.subplots(figsize=(10, 5))
        feas_rates = [r.get("feasibility_rate", 0.0) * 100 for r in all_results]
        ax.bar(scenarios, feas_rates, color="steelblue")
        ax.set_xlabel("Scenario")
        ax.set_ylabel("Feasibility Rate (%)")
        ax.set_title("Phase 5: QAOA Feasibility Rate (p=1)")
        ax.set_ylim(0, 105)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / "feasibility_rate.png", dpi=150)
        plt.close()

    print(f"Plots saved to: {plots_dir}")


if __name__ == "__main__":
    run_experiment()
