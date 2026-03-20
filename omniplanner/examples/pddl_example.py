import logging

import spark_dsg
import numpy as np

from dsg_pddl.pddl_grounding import PddlDomain, PddlGoal
from omniplanner.compile_plan import collect_plans
from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from omniplanner_ros.pddl_planner_ros import compile_plan
from utils import DummyRobotPlanningAdaptor, load_omniplanner_pddl_domain, visualize_plan

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


G = spark_dsg.DynamicSceneGraph.load("resources/example_dsg.json")

adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")
adaptors = {"euclid": adaptor}

robot_poses = {"euclid": np.array([-20.5, -10.5])}


# print("")
# print("================================")
# print("==   PDDL Domain (Simple)     ==")
# print("================================")
# print("")

# # TODO: Currently the simple domain has no notion of regions. I intend to add regions here,
# # but in a simple way that doesn't reflect the fact that a place is "in" a region.
# goal = PddlGoal(
#     robot_id="euclid",
#     pddl_goal="(and (visited-place p1350) (visited-object o95))",
# )

# # Load the PDDL domain you want to use
# domain = PddlDomain(load_omniplanner_pddl_domain("GotoObjectDomain.pddl"))

# # Build the plan request
# req = PlanRequest(
#     domain=domain,
#     goal=goal,
#     robot_states=robot_poses,
# )

# plan = full_planning_pipeline(req, G)
# # print("\nPlan from planning domain:")
# # print(plan)

# collected_plans = collect_plans(compile_plan(adaptors, "map", plan))
# # print("\nCollected plans:")
# # print(collected_plans)

# visualize_plan(collected_plans['euclid'], G)



# print("")
# print("================================")
# print("==   PDDL Domain (Pick/Place) ==")
# print("================================")
# print("")

# goal = PddlGoal(
#     robot_id="euclid",
#     pddl_goal="(and (object-in-place o95 p2410))",
# )

# # Load the PDDL domain you want to use
# domain = PddlDomain(load_omniplanner_pddl_domain("ObjectRearrangementDomain.pddl"))

# # Build the plan request
# req = PlanRequest(
#     domain=domain,
#     goal=goal,
#     robot_states=robot_poses,
# )

# plan = full_planning_pipeline(req, G)
# # print("\nPlan from planning domain:")
# # print(plan)

# collected_plans = collect_plans(compile_plan(adaptors, "map", plan))
# # print("\nCollected plans:")
# # print(collected_plans)

# visualize_plan(collected_plans['euclid'], G)



# print("")
# print("================================")
# print("==   PDDL Domain (Regions)    ==")
# print("================================")
# print("")

# goal = PddlGoal(
#     robot_id="euclid",
#     pddl_goal="(and (visited-region r4) (at-place p1396) (object-in-place o95 p2410))",
# )

# # Load the PDDL domain you want to use
# domain = PddlDomain(
#     load_omniplanner_pddl_domain("RegionObjectRearrangementDomain.pddl")
# )

# # Build the plan request
# req = PlanRequest(
#     domain=domain,
#     goal=goal,
#     robot_states=robot_poses,
# )

# plan = full_planning_pipeline(req, G)
# # print("\nPlan from planning domain:")
# # print(plan)

# collected_plans = collect_plans(compile_plan(adaptors, "map", plan))
# # print("\nCollected plans:")
# # print(collected_plans)

# visualize_plan(collected_plans['euclid'], G)



print("")
print("================================")
print("==   PDDL Domain (Test)       ==")
print("================================")
print("")

goal = PddlGoal(
    robot_id="euclid",
    pddl_goal="(and (visited-region r4) (at-poi p1396) (object-in-place o95 p2410))",
)

# Load the PDDL domain you want to use
domain = PddlDomain(load_omniplanner_pddl_domain("Test.pddl"))

# Build the plan request
req = PlanRequest(
    domain=domain,
    goal=goal,
    robot_states=robot_poses,
)

plan = full_planning_pipeline(req, G)
# print("\nPlan from planning domain:")
# print(plan)

collected_plans = collect_plans(compile_plan(adaptors, "map", plan))
# print("\nCollected plans:")
# print(collected_plans)

visualize_plan(collected_plans['euclid'], G)