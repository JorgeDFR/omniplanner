import numpy as np

from omniplanner.goto_points import GotoPointsDomain, GotoPointsGoal
from omniplanner.compile_plan import collect_plans
from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from omniplanner_ros.goto_points_ros import compile_plan
from utils import DummyRobotPlanningAdaptor, build_test_dsg


print("")
print("================================")
print("== Goto Points Domain, no DSG ==")
print("================================")
print("")

points = np.array(
    [
        [0.03350246, 0.27892633],
        [0.16300951, 0.16012492],
        [0.71635923, 0.5341003],
        [0.8763498, 0.43243519],
        [0.05777218, 0.51004976],
        [0.96980544, 0.00746369],
        [0.53927086, 0.75623442],
        [0.77329046, 0.66824145],
        [0.08683688, 0.49439621],
        [0.87066708, 0.50754294],
    ]
)

adaptor = DummyRobotPlanningAdaptor("spot", "spot", "map", "body")

robot_poses = np.array([0.0, 0.1])

goal = [1, 2, 3, 4]

req = PlanRequest(
    domain=GotoPointsDomain(),
    goal=[1, 2, 3, 4],
    robot_states=robot_poses
)

plan = full_planning_pipeline(req, points)

print("\nPlan from planning domain:")
print(plan)

collected_plans = collect_plans(compile_plan(adaptor, "map", plan))
print("\nCollected plans:")
print(collected_plans)



print("")
print("==================================")
print("== Goto Points Domain, with DSG ==")
print("==================================")
print("")

G = build_test_dsg()

adaptor = DummyRobotPlanningAdaptor("spot", "spot", "map", "body")

robot_poses = {"spot": np.array([0.0, 0.1])}

goal = GotoPointsGoal(["O(0)", "O(1)"], "spot")

req = PlanRequest(
    domain=GotoPointsDomain(),
    goal=goal,
    robot_states=robot_poses
)

plan = full_planning_pipeline(req, G)

print("\nPlan from planning domain:")
print(plan)

collected_plans = collect_plans(compile_plan(adaptor, "map", plan))
print("\nCollected plans:")
print(collected_plans)
