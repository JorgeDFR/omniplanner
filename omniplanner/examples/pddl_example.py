import logging
from pathlib import Path

import numpy as np
import spark_dsg

from dsg_pddl.core.models import PddlDomain, PddlGoal
from omniplanner.compile_plan import collect_plans
from omniplanner.core import PlanRequest, full_planning_pipeline
from simple_compile_plan import compile_plan
from utils import DummyRobotPlanningAdaptor, load_omniplanner_pddl_domain

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

EXAMPLES_DIR = Path(__file__).resolve().parent


def run_pddl_case(graph, adaptors, robot_poses, domain_file, pddl_goal, title):
    print("")
    print("================================")
    print(f"==   {title}   ==")
    print("================================")
    print("")

    goal = PddlGoal(robot_id="euclid", pddl_goal=pddl_goal)
    domain = PddlDomain(load_omniplanner_pddl_domain(domain_file))
    req = PlanRequest(
        domain=domain,
        goal=goal,
        robot_states=robot_poses,
    )

    plan = full_planning_pipeline(req, graph)
    print("\nPlan from planning domain:")
    print(plan)

    # collected_plans = collect_plans(compile_plan(adaptors, "map", plan))
    # print("\nCollected plans:")
    # print(collected_plans)


def main():
    graph = spark_dsg.DynamicSceneGraph.load(
        str(EXAMPLES_DIR / "resources" / "example_dsg.json")
    )
    adaptor = DummyRobotPlanningAdaptor("euclid", "spot", "map", "body")
    adaptors = {"euclid": adaptor}
    robot_poses = {"euclid": np.array([-20.5, -10.5])}

    run_pddl_case(
        graph,
        adaptors,
        robot_poses,
        "GotoObjectDomain.pddl",
        "(and (visited-place p1350) (visited-object o95))",
        "PDDL Domain (Simple)",
    )

    run_pddl_case(
        graph,
        adaptors,
        robot_poses,
        "ObjectRearrangementDomain.pddl",
        "(and (object-in-place o95 p2410))",
        "PDDL Domain (Pick/Place)",
    )

    run_pddl_case(
        graph,
        adaptors,
        robot_poses,
        "RegionObjectRearrangementDomain.pddl",
        "(and (visited-region r4) (at-place p1396) (object-in-place o95 p2410))",
        "PDDL Domain (Regions)",
    )


if __name__ == "__main__":
    main()
