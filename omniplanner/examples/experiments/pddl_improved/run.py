import logging
import sys
import time
from pathlib import Path

import numpy as np

from dsg_pddl.core.models import PddlDomain, PddlGoal
from omniplanner.compile_plan import collect_plans
from omniplanner.core import PlanRequest, full_planning_pipeline

EXAMPLES_DIR = Path(__file__).resolve().parents[2]
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(1, str(EXAMPLES_DIR))

from simple_compile_plan import compile_plan
from utils import (
    build_scalable_dsg,
    DummyRobotPlanningAdaptor,
    generate_dnf_goal,
    load_omniplanner_pddl_domain,
    Predicate,
)
from utils_viz import visualize_dsg, visualize_plan

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)


def main():
    # Generate random 3D Scene Graph
    num_nodes = 400
    num_objects = 50
    num_regions = 6

    # valid_map_areas = generate_building(
    #     num_rooms=6,
    #     grid_size=(3, 2),
    #     seed=1
    # )

    graph = build_scalable_dsg(
        num_nodes=num_nodes,
        valid_map_areas=[(0, 0, 20, 20)],
        #valid_map_areas=valid_map_areas,
        num_objects=num_objects,
        num_regions=num_regions,
        seed=1,
    )
    #visualize_dsg(G, show_region_edges=False)

    # Generate random PDDL Goal
    predicates = [
        Predicate("at-poi", ["place"]),
        Predicate("at-object", ["dsg_object"]),
        #Predicate("in-region", ["region"]),
        Predicate("holding", ["dsg_object"]),
        Predicate("safe", ["dsg_object"]),
        Predicate("object-in-place", ["dsg_object", "place"]),
        Predicate("visited-place", ["place"], can_be_negated=True),
        Predicate("visited-object", ["dsg_object"], can_be_negated=True),
        #Predicate("visited-region", ["region"], can_be_negated=False),
    ]

    max_positive = {
        "holding": 1,
        #"safe": 0,
        ("at-poi", "at-object", "in-region"): 1,
    }

    symbols_by_type = {
        "place": [f"p{i}" for i in range(num_nodes)],
        "dsg_object": [f"o{i}" for i in range(num_objects)],
        "region": [f"r{i}" for i in range(num_regions)],
    }

    pddl_goal = generate_dnf_goal(
        N=1,
        K=6,
        predicates=predicates,
        symbols_by_type=symbols_by_type,
        max_positive_per_pred=max_positive,
        seed=2,
    )
    print(f"\nPDDL Goal: {pddl_goal}\n")

    adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")
    adaptors = {"euclid": adaptor}
    robot_poses = {"euclid": np.array([1.0, 1.0])}
    goal = PddlGoal(robot_id="euclid", pddl_goal=pddl_goal)

    domain = PddlDomain(
        load_omniplanner_pddl_domain(
            "RegionObjectRearrangementDomain_DerivedPredicates.pddl"
        )
    )

    req = PlanRequest(
        domain=domain,
        goal=goal,
        robot_states=robot_poses,
    )

    from dsg_pddl.grounding.improved_region import generate_region_rearrangement_pddl_relevant_paths
    _, symbols = generate_region_rearrangement_pddl_relevant_paths(
        graph,
        goal.pddl_goal,
        robot_poses[goal.robot_id][:2],
        domain_name=domain.domain_name,
    )
    simplified_symbols = [s.symbol for s in symbols]
    # visualize_dsg(graph, nodes_to_show=simplified_symbols)

    start = time.perf_counter()
    plan = full_planning_pipeline(req, graph)
    end = time.perf_counter()
    print(f"\nPlanning took {end - start:.6f} seconds\n")

    collected_plans = collect_plans(compile_plan(adaptors, "map", plan))
    print(collected_plans)
    visualize_plan(
        collected_plans["euclid"],
        graph,
        nodes_to_show=simplified_symbols,
        simplify_legend=True,
    )


if __name__ == "__main__":
    main()
