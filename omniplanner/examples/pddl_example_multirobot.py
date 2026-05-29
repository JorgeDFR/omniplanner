#!/usr/bin/env python3

import logging
import numpy as np

from dsg_pddl.core.models import MultiRobotPddlDomain, PddlGoal
from omniplanner.compile_plan import collect_plans
from omniplanner.core import PlanRequest, full_planning_pipeline
from simple_compile_plan import compile_plan
from utils import (
    DummyRobotPlanningAdaptor,
    build_test_dsg,
    load_omniplanner_pddl_domain,
)

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


def main():
    graph = build_test_dsg()

    robot_poses = {
        "robot1": np.array([-1.2, 0.0]),
        "robot2": np.array([1.2, 0.0]),
        "robot3": np.array([0.0, 0.5]),
    }
    adaptors = {
        "robot1": DummyRobotPlanningAdaptor("euclid", "spot", "map", "euclid/body"),
        "robot2": DummyRobotPlanningAdaptor("hamilton", "spot", "map", "hamilton/body"),
        "robot3": DummyRobotPlanningAdaptor("gauss", "husky", "map", "husky/body"),
    }
    pddl_goal = "(and (visited-object o0) (visited-object o1))"

    print("")
    print("======================================")
    print("==     PDDL Domain (Multi-Robot)    ==")
    print("======================================")
    print("")

    print(f"PDDL goal: {pddl_goal}")
    goal = PddlGoal(robot_id="robot1", pddl_goal=pddl_goal)
    domain = MultiRobotPddlDomain(
        load_omniplanner_pddl_domain(
            "RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl"
        )
    )
    req = PlanRequest(
        domain=domain,
        goal=goal,
        robot_states=robot_poses,
    )

    plan = full_planning_pipeline(req, graph)

    print("\nSymbolic plan:")
    for action in plan.value.value.symbolic_actions:
        print(action)

    collected_plan = collect_plans(compile_plan(adaptors, "map", plan))
    print("\nCollected plans:")
    print(collected_plan)


if __name__ == "__main__":
    main()
