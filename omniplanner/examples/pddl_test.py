import logging

import time
import spark_dsg
import numpy as np

from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal
from omniplanner.compile_plan import collect_plans
from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from simple_compile_plan import compile_plan
from utils import (
    DummyRobotPlanningAdaptor, load_omniplanner_pddl_domain,
    build_scalable_dsg,
    generate_building,
    Predicate, generate_dnf_goal
)
from utils_viz import visualize_plan, visualize_dsg

import dsg_pddl.dsg_pddl_grounding_improved

logging.basicConfig()
logging.getLogger().setLevel(logging.WARN)


# G = spark_dsg.DynamicSceneGraph.load("resources/example_dsg.json")

# adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")
# adaptors = {"euclid": adaptor}

# robot_poses = {"euclid": np.array([-20.5, -10.5])}

# goal = PddlGoal(
#     robot_id="euclid",
#     pddl_goal="(and (at-poi p1396) (object-in-place o95 p2410) (visited-region r4) (not (visited-place p3167)))",
# )

# # Load the PDDL domain
# domain = PddlDomain(
#     load_omniplanner_pddl_domain(
#         "RegionObjectRearrangementDomain_DerivedPredicates.pddl"
#     )
# )

# # Build the plan request
# req = PlanRequest(
#     domain=domain,
#     goal=goal,
#     robot_states=robot_poses,
# )

# start = time.perf_counter()
# plan = full_planning_pipeline(req, G)
# end = time.perf_counter()
# print(f"Planning took {end - start:.6f} seconds")

# collected_plans = collect_plans(compile_plan(adaptors, "map", plan))
# visualize_plan(collected_plans['euclid'], G, simplify_legend=False)





# Generate random 3D Scene Graph
num_nodes = 400
num_objects = 50
num_regions = 6

# valid_map_areas = generate_building(
#     num_rooms=6,
#     grid_size=(3, 2),
#     seed=1
# )

G = build_scalable_dsg(
    num_nodes=num_nodes,
    valid_map_areas=[(0, 0, 20, 20)], #valid_map_areas,
    num_objects=num_objects,
    num_regions=num_regions,
    seed=1
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
    #Predicate("visited-region", ["region"], can_be_negated=True),
]

max_positive = {
    "holding": 1,
    "safe": 0,
    ("at-poi", "at-object", "in-region"): 1,
}

symbols_by_type = {
    "place": [f"p{i}" for i in range(num_nodes)],
    "dsg_object": [f"o{i}" for i in range(num_objects)],
    "region": [f"r{i}" for i in range(num_regions)],
}

pddl_goal = generate_dnf_goal(
    N=1, K=6,
    predicates=predicates,
    symbols_by_type=symbols_by_type,
    max_positive_per_pred=max_positive,
    seed=2
)
print(f"\nPDDL Goal: {pddl_goal}\n")

adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")
adaptors = {"euclid": adaptor}
robot_poses = {"euclid": np.array([1.0, 1.0])}
goal = PddlGoal(robot_id="euclid", pddl_goal=pddl_goal)

# import sys
# sys.exit()

# Load the PDDL domain
domain = PddlDomain(
    load_omniplanner_pddl_domain(
        "RegionObjectRearrangementDomain_DerivedPredicates.pddl"
    )
)

# Build the plan request
req = PlanRequest(
    domain=domain,
    goal=goal,
    robot_states=robot_poses,
)

from dsg_pddl.dsg_pddl_grounding_improved import (
    generate_region_rearrangement_pddl_relevant_paths,
)

_, symbols = generate_region_rearrangement_pddl_relevant_paths(
    G,
    goal.pddl_goal,
    robot_poses[goal.robot_id][:2],
    domain_name=domain.domain_name,
)
simplified_symbols = [s.symbol for s in symbols]
visualize_dsg(G, nodes_to_show=simplified_symbols)

start = time.perf_counter()
plan = full_planning_pipeline(req, G)
end = time.perf_counter()
print(f"\nPlanning took {end - start:.6f} seconds\n")

collected_plans = collect_plans(compile_plan(adaptors, "map", plan))
visualize_plan(collected_plans['euclid'], G,
               nodes_to_show=simplified_symbols,
               simplify_legend=True,
               )

# TODO:
# - multi robot problems are not adressed
# - region predicates in the goal make the solver performance mutch worse
# - solving for a optimal solution in some cases is very slow (WIP)
