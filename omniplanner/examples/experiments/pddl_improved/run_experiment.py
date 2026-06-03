#!/usr/bin/env python3

import os
import sys
import time
import json
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path

from dsg_pddl.core.models import PddlDomain, PddlGoal
from omniplanner.compile_plan import collect_plans
from omniplanner.core import PlanRequest, full_planning_pipeline

EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(1, str(EXAMPLES_DIR))

from simple_compile_plan import compile_plan
from utils import (
    build_scalable_dsg,
    build_dsg_from_cache,
    DummyRobotPlanningAdaptor,
    generate_dnf_goal,
    load_omniplanner_pddl_domain,
    Predicate,
)
from utils_viz import visualize_dsg, visualize_plan

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)


def parse_args():
    parser = argparse.ArgumentParser(description="Run DSG-PDDL scalability experiment")

    # Graph generation
    parser.add_argument("--num-nodes", type=int, default=400)
    parser.add_argument("--num-objects", type=int, default=50)
    parser.add_argument("--num-regions", type=int, default=6)
    parser.add_argument("--graph-seed", type=int, default=1)

    parser.add_argument(
        "--graph-cache-dir",
        type=Path,
        default=None,
        help="Directory where generated DSG graphs are cached. Defaults to <output-dir>/graph_cache.",
    )

    parser.add_argument(
        "--force-regenerate-graph",
        action="store_true",
        help="Regenerate the DSG even if a cached graph exists.",
    )

    # Goal generation
    parser.add_argument("--goal-disjunctions", type=int, default=1)
    parser.add_argument("--goal-conjunctions", type=int, default=6)
    parser.add_argument("--goal-seed", type=int, default=1)

    # PDDL domain
    parser.add_argument(
        "--pddl-domain",
        choices=["derived", "explicit"],
        default="explicit",
        help="PDDL domain to use",
    )

    # PDDL symbolic sampler
    parser.add_argument(
        "--pddl-sampler",
        choices=["all", "paths", "compressed"],
        default="compressed",
        help="PDDL symbolic sampler to use",
    )

    # PDDL solver
    parser.add_argument(
        "--pddl-solver",
        choices=["lazy_ff", "wastar_ff", "astar_ff", "lmcut"],
        default="lazy_ff",
        help="PDDL solver to use",
    )

    parser.add_argument(
        "--pddl-timeout",
        type=int,
        default=60,
        help="PDDL solver timeout in seconds",
    )

    # Output
    default_output_dir = Path(os.getenv("RESULTS_OUTPUT_DIR", ""))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Output directory (default: {default_output_dir})",
    )

    args = parser.parse_args()

    if args.pddl_solver == "lmcut" and args.pddl_domain == "derived":
        parser.error(
            "'lmcut' is not supported with the 'derived' PDDL domain. "
            "Use '--pddl-domain explicit' or choose a different solver."
        )

    configure_pddl_solver_env(
        solver=args.pddl_solver,
        timeout=args.pddl_timeout,
    )

    return args


def create_experiment_dir(args):
    exp_name = (
        f"n{args.num_nodes}"
        f"_o{args.num_objects}"
        f"_r{args.num_regions}"
        f"_graph{args.graph_seed}"
        f"_d{args.goal_disjunctions}"
        f"_c{args.goal_conjunctions}"
        f"_goal{args.goal_seed}"
        f"_{args.pddl_domain}"
        f"_{args.pddl_sampler}"
        f"_{args.pddl_solver}"
    )

    exp_dir = args.output_dir / "results" / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    return exp_dir


def save_config(args, output_dir):
    config = vars(args)
    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
            f,
            indent=2,
        )


def configure_pddl_solver_env(solver: str, timeout: float) -> None:
    valid_pddl_solvers = {
        "lazy_ff": "suboptimal",
        "wastar_ff": "suboptimal",
        "astar_ff": "optimal",
        "lmcut": "optimal",
    }

    try:
        solver_type = valid_pddl_solvers[solver]
    except KeyError:
        valid = ", ".join(sorted(valid_pddl_solvers))
        raise ValueError(
            f"Invalid PDDL solver {solver!r}. Valid options: {valid}"
        )

    for kind in ("optimal", "suboptimal"):
        os.environ[f"PDDL_{kind.upper()}_SOLVER"] = "none"

    os.environ[f"PDDL_{solver_type.upper()}_SOLVER"] = solver
    os.environ[f"PDDL_{solver_type.upper()}_TIMEOUT"] = str(timeout)


def graph_cache_path(args):
    cache_dir = args.graph_cache_dir or (args.output_dir / "graph_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    return cache_dir / (
        f"dsg_n{args.num_nodes}"
        f"_o{args.num_objects}"
        f"_r{args.num_regions}"
        f"_seed{args.graph_seed}.json"
    )


def load_or_build_graph(args):
    cache_path = graph_cache_path(args)

    if cache_path.exists() and not args.force_regenerate_graph:
        print(f"Loading cached DSG from: {cache_path}\n")
        with open(cache_path, "r") as f:
            data = json.load(f)
        return build_dsg_from_cache(data)

    print(f"Generating DSG and saving to: {cache_path}\n")

    graph, data = build_scalable_dsg(
        num_nodes=args.num_nodes,
        valid_map_areas=[(0, 0, 20, 20)],
        num_objects=args.num_objects,
        num_regions=args.num_regions,
        seed=args.graph_seed,
    )

    tmp_path = cache_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)

    tmp_path.replace(cache_path)
    return graph


def move_debug_pddl_outputs(exp_dir):
    debug_output_dir = os.getenv("DEBUG_OUTPUT_DIR")
    if debug_output_dir:
        debug_output_dir = Path(debug_output_dir)

        files_to_move = {
            "plan.txt": exp_dir / "plan.txt",
            "problem.pddl": exp_dir / "problem.pddl",
        }

        for filename, destination in files_to_move.items():
            source = debug_output_dir / filename
            if source.exists():
                shutil.move(str(source), str(destination))


def main():
    args = parse_args()
    exp_dir = create_experiment_dir(args)
    save_config(args, exp_dir)

    # -------------------------------
    # Generate random 3D Scene Graph
    # -------------------------------
    start = time.perf_counter()
    graph = load_or_build_graph(args)
    end = time.perf_counter()
    build_dsg_time = end - start
    print(f"\nBuilding DSG took {build_dsg_time:.6f} seconds\n")

    fig = visualize_dsg(graph, show=False)
    fig.savefig(exp_dir / "dsg.png", dpi=300, bbox_inches="tight")
    fig.clf()

    # -------------------------------
    # Generate random PDDL Goal
    # -------------------------------
    predicates = [
        Predicate("at-poi", ["place"]),
        Predicate("at-object", ["dsg_object"]),
        Predicate("in-region", ["region"]),
        Predicate("holding", ["dsg_object"]),
        Predicate("safe", ["dsg_object"]),
        Predicate("object-in-place", ["dsg_object", "place"]),
        Predicate("visited-place", ["place"], can_be_negated=True),
        Predicate("visited-object", ["dsg_object"], can_be_negated=True),
        Predicate("visited-region", ["region"], can_be_negated=False),
    ]

    max_positive = {
        "holding": 1,
        ("at-poi", "at-object", "in-region"): 1,
        "safe": 0,
        ("in-region", "visited-region"): 0,
    }

    symbols_by_type = {
        "place": [f"p{i}" for i in range(args.num_nodes)],
        "dsg_object": [f"o{i}" for i in range(args.num_objects)],
        "region": [f"r{i}" for i in range(args.num_regions)],
    }

    pddl_goal = generate_dnf_goal(
        N=args.goal_disjunctions,
        K=args.goal_conjunctions,
        predicates=predicates,
        symbols_by_type=symbols_by_type,
        max_positive_per_pred=max_positive,
        seed=args.goal_seed,
    )

    if args.pddl_domain == "derived":
        domain = PddlDomain(load_omniplanner_pddl_domain(
            "RegionObjectRearrangementDomain_DerivedPredicates.pddl"
        ))
    else:
        domain = PddlDomain(load_omniplanner_pddl_domain(
            "RegionObjectRearrangementDomain_ExplicitState.pddl"
        ))

    if args.pddl_sampler == "compressed":
        from dsg_pddl.grounding.improved_region import (
            generate_region_rearrangement_pddl_compressed_graph as generate_symbols,
        )
    elif args.pddl_sampler == "paths":
        from dsg_pddl.grounding.improved_region import (
            generate_region_rearrangement_pddl_relevant_paths as generate_symbols,
        )
    else:
        from dsg_pddl.grounding.improved_region import (
            generate_region_rearrangement_pddl_all_symbols as generate_symbols,
        )

    # Save PDDL goal
    with open(exp_dir / "goal.txt", "w") as f:
        f.write(str(pddl_goal))

    adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")
    adaptors = {"euclid": adaptor}
    robot_poses = {"euclid": np.array([1.0, 1.0])}
    goal = PddlGoal(robot_id="euclid", pddl_goal=pddl_goal)

    req = PlanRequest(domain=domain, goal=goal, robot_states=robot_poses)

    start = time.perf_counter()
    _, symbols = generate_symbols(
        graph, goal.pddl_goal, robot_poses[goal.robot_id][:2], domain_name=domain.domain_name
    )
    simplified_symbols = [s.symbol for s in symbols]
    end = time.perf_counter()
    sampling_time = end - start
    print(f"Sampler took {sampling_time:.6f} seconds\n")

    fig = visualize_dsg(graph, nodes_to_show=simplified_symbols, show=False)
    fig.savefig(exp_dir / "sampled_symbols.png", dpi=300, bbox_inches="tight")
    fig.clf()

    # -------------------------------
    # Planning
    # -------------------------------
    start = time.perf_counter()
    plan = full_planning_pipeline(req, graph)
    end = time.perf_counter()
    planning_time = end - start
    print(f"Planning took {planning_time:.6f} seconds\n")

    move_debug_pddl_outputs(exp_dir)

    collected_plans = collect_plans(compile_plan(adaptors, "map", plan))
    fig = visualize_plan(
        collected_plans["euclid"],
        graph,
        #nodes_to_show=simplified_symbols,
        simplify_legend=True,
        show=False,
    )
    fig.savefig(exp_dir / "plan.png", dpi=300, bbox_inches="tight")
    fig.clf()

    # Determine whether planning succeeded
    plan_file = exp_dir / "plan.txt"
    success = (
        plan_file.exists()
        and plan_file.read_text().strip() != ""
    )

    # -------------------------------
    # Save experiment results
    # -------------------------------
    results = {
        "success": success,
        "sampling_time_sec": sampling_time,
        "planning_time_sec": planning_time,
        "num_sampled_symbols": len(simplified_symbols),
    }

    with open(exp_dir / "planning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved on: {exp_dir}\n")

if __name__ == "__main__":
    main()