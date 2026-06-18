#!/usr/bin/env python3

import os
import re
import sys
import time
import json
import shutil
import logging
import argparse
import numpy as np
from pathlib import Path

import spark_dsg
from dsg_pddl.core.models import PddlDomain, PddlGoal
from omniplanner.compile_plan import collect_plans
from omniplanner.core import PlanRequest, full_planning_pipeline

EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(1, str(EXAMPLES_DIR))

from simple_compile_plan import compile_plan
from utils import (
    DummyRobotPlanningAdaptor,
    generate_dnf_goal,
    load_omniplanner_pddl_domain,
    Predicate,
)
from utils_viz import visualize_dsg, visualize_plan

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)

DEFAULT_DSG_PATH = "../../resources/example_dsg.json"


REAL_DSG_SYMBOLS_BY_TYPE = {
    "place": [f"p{i}" for i in (
        '754', '1350', '1396', '1424', '1568', '1833', '2410', '2441',
        '2541', '3095', '3105', '3107', '3166', '3167', '3195', '4389',
        '6219', '6255', '6397', '6512', '6567', '6665', '6726', '7279',
        '7281', '7488', '8637', '8638', '9420', '9421', '10247', '10379',
        '15451', '15561', '15944', '21172', '21175', '21976', '22249',
        '22250', '22378', '22396', '22407', '22514', '22515', '22543',
        '22544', '24172', '24255', '24619', '24698', '25023', '25116',
        '25696', '25697', '25698', '26231', '26544', '26753', '26916',
        '27258', '27669', '27795', '28062', '28214', '28216', '28705',
        '28707', '29521', '29882', '29911', '30263', '30296', '56525',
        '56710', '57510', '58431', '58432', '58530', '58531', '59110',
        '59898', '60460', '60704', '60843', '61369', '67045', '67046',
        '67047', '67048', '67673', '67676', '68067', '68306', '75035',
        '75286'
    )],
    "dsg_object": [f"o{i}" for i in (
        '0', '2', '3', '7', '8', '9', '10', '17', '18', '19', '27', '29',
        '30', '36', '39', '43', '51', '53', '55', '58', '59', '63', '64',
        '66', '70', '75', '79', '82', '83', '84', '85', '88', '89', '95',
        '98', '100', '103', '112', '237', '238', '249', '251', '255',
        '257', '258', '263', '264', '265', '266', '269', '275', '277',
        '279', '282', '283', '285', '287', '291', '293', '300', '317',
        '344', '358', '363', '373'
    )],
    "region": [f"r{i}" for i in ('1', '2', '3', '4', '5')],
}


REAL_DSG_ROBOT_POSES = {"euclid": np.array([-20.5, -10.5])}


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3DSG-PDDL benchmark on a real DSG")

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

    if args.goal_disjunctions != 1:
        parser.error("Currently only goals with 1 disjunction are supported.")

    if args.pddl_solver == "lmcut" and args.pddl_domain == "derived":
        parser.error(
            "'lmcut' is not supported with the 'derived' PDDL domain. "
            "Use '--pddl-domain explicit' or choose a different solver."
        )

    configure_pddl_solver_env(
        solver=args.pddl_solver,
        timeout=args.pddl_timeout,
    )

    configure_pddl_sampler_env(
        sampler=args.pddl_sampler,
    )

    return args


def create_experiment_dir(args):
    exp_name = (
        f"real_dsg"
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
    config["robot_poses"] = {k: v.tolist() for k, v in REAL_DSG_ROBOT_POSES.items()}
    config["num_symbols_by_type"] = {
        key: len(value) for key, value in REAL_DSG_SYMBOLS_BY_TYPE.items()
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
            f,
            indent=2,
        )


def configure_pddl_sampler_env(sampler: str) -> None:
    valid_samplers = {"all", "paths", "compressed"}
    if sampler not in valid_samplers:
        raise ValueError(f"Invalid PDDL sampler {sampler!r}")

    os.environ["PDDL_SAMPLER"] = sampler


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
        raise ValueError(f"Invalid PDDL solver {solver!r}. Valid options: {valid}")

    for kind in ("optimal", "suboptimal"):
        os.environ[f"PDDL_{kind.upper()}_SOLVER"] = "none"

    os.environ[f"PDDL_{solver_type.upper()}_SOLVER"] = solver
    os.environ[f"PDDL_{solver_type.upper()}_TIMEOUT"] = str(timeout)


def load_real_graph(args):
    print(f"Loading real DSG from: {DEFAULT_DSG_PATH}\n")
    return spark_dsg.DynamicSceneGraph.load(DEFAULT_DSG_PATH)


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


def parse_plan_cost(plan_file: Path):
    """
    Parse the plan cost from a plan.txt file.

    Expected line format:
        ; cost = 75 (general cost)

    Returns:
        int or float if a cost is found,
        None otherwise.
    """
    if not plan_file.exists():
        return None

    plan_cost_re = re.compile(
        r"^\s*;\s*cost\s*=\s*"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
        r"\b",
        re.IGNORECASE | re.MULTILINE,
    )
    text = plan_file.read_text(errors="replace")
    matches = plan_cost_re.findall(text)

    if not matches:
        return None

    # Use the last cost line, in case the file contains multiple comments.
    cost = float(matches[-1])

    if cost.is_integer():
        return int(cost)

    return cost


def main():
    args = parse_args()
    exp_dir = create_experiment_dir(args)
    save_config(args, exp_dir)

    # -------------------------------
    # Load real 3D Scene Graph
    # -------------------------------
    start = time.perf_counter()
    graph = load_real_graph(args)
    end = time.perf_counter()
    load_dsg_time = end - start
    print(f"\nLoading DSG took {load_dsg_time:.6f} seconds\n")

    fig = visualize_dsg(graph, show=False)
    fig.savefig(exp_dir / "dsg.png", dpi=300, bbox_inches="tight")
    fig.clf()

    # -------------------------------
    # Generate random PDDL Goal
    # -------------------------------
    predicates = [
        Predicate("at-place", ["place"]),
        Predicate("at-object", ["dsg_object"]),
        Predicate("in-region", ["region"]),
        Predicate("holding", ["dsg_object"]),
        Predicate("safe", ["dsg_object"]),
        Predicate("object-in-place", ["dsg_object", "place"]),
        Predicate("visited-place", ["place"], can_be_negated=True),
        Predicate("visited-object", ["dsg_object"], can_be_negated=True),
        Predicate("visited-region", ["region"], can_be_negated=True),
    ]

    max_positive = {
        "holding": 1,
        ("at-place", "at-object", "in-region"): 1,
        "safe": 0,
        ("in-region", "visited-region"): 1,
    }

    pddl_goal = generate_dnf_goal(
        N=args.goal_disjunctions,
        K=args.goal_conjunctions,
        predicates=predicates,
        symbols_by_type=REAL_DSG_SYMBOLS_BY_TYPE,
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
    robot_poses = REAL_DSG_ROBOT_POSES
    goal = PddlGoal(robot_id="euclid", pddl_goal=pddl_goal)

    req = PlanRequest(domain=domain, goal=goal, robot_states=robot_poses)

    start = time.perf_counter()
    _, symbols = generate_symbols(
        graph,
        goal.pddl_goal,
        robot_poses[goal.robot_id][:2],
        domain_name=domain.domain_name,
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
    total_time = end - start
    print(f"Planning took {total_time:.6f} seconds\n")

    move_debug_pddl_outputs(exp_dir)

    collected_plans = collect_plans(compile_plan(adaptors, "map", plan))
    fig = visualize_plan(
        collected_plans["euclid"],
        graph,
        # nodes_to_show=simplified_symbols,
        simplify_legend=True,
        show=False,
    )
    fig.savefig(exp_dir / "plan.png", dpi=300, bbox_inches="tight")
    fig.clf()

    # Determine whether planning succeeded
    plan_file = exp_dir / "plan.txt"
    success = (
        plan_file.exists()
        and plan_file.read_text(errors="replace").strip() != ""
    )

    if success:
        plan_cost = parse_plan_cost(plan_file)

        if plan_cost is None:
            raise RuntimeError(
                f"Planning appears successful, but no plan cost was found in: {plan_file}"
            )
    else:
        plan_cost = None

    # -------------------------------
    # Save experiment results
    # -------------------------------
    results = {
        "success": success,
        "load_dsg_time_sec": load_dsg_time,
        "sampling_time_sec": sampling_time,
        "planning_time_sec": total_time - sampling_time,
        "total_time_sec": total_time,
        "num_sampled_symbols": len(simplified_symbols),
        "plan_cost": plan_cost,
    }

    with open(exp_dir / "planning_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved on: {exp_dir}\n")


if __name__ == "__main__":
    main()
