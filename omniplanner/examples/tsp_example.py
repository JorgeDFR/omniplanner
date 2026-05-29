import logging

import numpy as np

from omniplanner.domains.tsp import TspDomain, TspGoal
from omniplanner.compile_plan import collect_plans
from omniplanner.core import PlanRequest, full_planning_pipeline
from simple_compile_plan import compile_plan
from utils import DummyRobotPlanningAdaptor, build_test_dsg

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)


def main():
    print("")
    print("==========================")
    print("== TSP Domain           ==")
    print("==========================")
    print("")

    graph = build_test_dsg()
    adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")
    robot_poses = {"spot": np.array([0.0, 0.1])}
    goal = TspGoal(goal_points=["O(0)", "O(1)"], robot_id="spot")
    robot_domain = TspDomain(solver="2opt")

    req = PlanRequest(
        domain=robot_domain,
        goal=goal,
        robot_states=robot_poses,
    )

    plan = full_planning_pipeline(req, graph)

    print("\nPlan from planning domain:")
    print(plan)

    # collected_plans = collect_plans(compile_plan(adaptor, "map", plan))
    # print("\nCollected plans:")
    # print(collected_plans)


if __name__ == "__main__":
    main()
