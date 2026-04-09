import logging

import time
import spark_dsg
import numpy as np

from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal
from omniplanner.compile_plan import collect_plans
from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from omniplanner_ros.pddl_planner_ros import compile_plan
from utils import DummyRobotPlanningAdaptor, load_omniplanner_pddl_domain, build_scalable_dsg
from utils_viz import visualize_plan

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


G = spark_dsg.DynamicSceneGraph.load("resources/example_dsg.json")

adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")
adaptors = {"euclid": adaptor}

robot_poses = {"euclid": np.array([-20.5, -10.5])}

goal = PddlGoal(
    robot_id="euclid",
    pddl_goal="(and (at-poi p1396) (object-in-place o95 p2410) (visited-region r4) (not (visited-place p3167)))",
)

# Load the PDDL domain you want to use
domain = PddlDomain(load_omniplanner_pddl_domain("Test.pddl"))

# Build the plan request
req = PlanRequest(
    domain=domain,
    goal=goal,
    robot_states=robot_poses,
)

start = time.perf_counter()
plan = full_planning_pipeline(req, G)
end = time.perf_counter()
print(f"Planning took {end - start:.6f} seconds")

collected_plans = collect_plans(compile_plan(adaptors, "map", plan))
visualize_plan(collected_plans['euclid'], G)








# G = build_scalable_dsg(
#     grid_cols=10, grid_rows=10, cell_size=3.0,
#     num_objects=5, num_regions=4,
#     seed=123
# )

# adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")
# adaptors = {"euclid": adaptor}

# robot_poses = {"euclid": np.array([0.1, 0.1])}

# goal = PddlGoal(
#     robot_id="euclid",
#     pddl_goal="(and (at-poi p99) (object-in-place o4 p36) (not (visited-place p59)))",
#     #pddl_goal="(and (at-poi p99) (object-in-place o4 p36) (not (visited-object o0)))",
#     #pddl_goal="(and (at-poi p99) (object-in-place o4 p36) (not (visited-region r0)))",
#     #pddl_goal="(and (at-poi p99) (object-in-place o4 p36) (not (visited-place p59)) (not (visited-object o0)))",
#     #pddl_goal="(and (at-poi p99) (object-in-place o4 p36) (not (or (visited-place p59) (visited-object o0))))",
# )

# # Load the PDDL domain you want to use
# domain = PddlDomain(load_omniplanner_pddl_domain("Test.pddl"))

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
# visualize_plan(collected_plans['euclid'], G)