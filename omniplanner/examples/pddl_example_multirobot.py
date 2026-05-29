#!/usr/bin/env python3

import argparse
import logging

import numpy as np
import spark_dsg

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
logging.getLogger().setLevel(logging.INFO)

DEFAULT_GOAL = "(and (visited-object o0) (visited-object o1))"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the multirobot PDDL example.")
    parser.add_argument(
        "--scene-graph",
        help="Optional DSG JSON file. If omitted, a compact synthetic DSG is used.",
    )
    parser.add_argument(
        "--goal",
        default=DEFAULT_GOAL,
        help="PDDL goal string to solve.",
    )
    return parser.parse_args()


def load_graph(scene_graph_path):
    if scene_graph_path is None:
        return build_test_dsg()
    return spark_dsg.DynamicSceneGraph.load(scene_graph_path)


def main():
    args = parse_args()
    graph = load_graph(args.scene_graph)

    robot_poses = {
        "robot1": np.array([-1.2, 0.0]),
        "robot2": np.array([1.2, 0.0]),
        "robot3": np.array([0.0, 0.5]),
    }
    adaptors = {
        "robot1": DummyRobotPlanningAdaptor("euclid", "spot", "map", "euclid/body"),
        "robot2": DummyRobotPlanningAdaptor(
            "hamilton", "spot", "map", "hamilton/body"
        ),
        "robot3": DummyRobotPlanningAdaptor("gauss", "husky", "map", "husky/body"),
    }

    print("")
    print("======================================")
    print("==   PDDL Domain (Multi-Robot)      ==")
    print("======================================")
    print("")
    print(f"PDDL goal: {args.goal}")

    goal = PddlGoal(robot_id="robot1", pddl_goal=args.goal)
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

    compiled_plans = compile_plan(adaptors, "map", plan)
    collected_plan = collect_plans(compiled_plans)
    print("\nCollected plans:")
    print(collected_plan)


if __name__ == "__main__":
    main()
