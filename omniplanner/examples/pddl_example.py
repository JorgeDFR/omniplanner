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


def run_pddl_case(graph, adaptors, robot_poses, title, domain_file, pddl_goal):
    print("")
    print("================================")
    print(title)
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
    collected_plans = collect_plans(compile_plan(adaptors, "map", plan))

    print("\nPlan from planning domain:")
    print(plan)
    print("\nCollected plans:")
    print(collected_plans)


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
        "==   PDDL Domain (Simple)     ==",
        "GotoObjectDomain.pddl",
        "(and (visited-place p1350) (visited-object o95))",
    )
    run_pddl_case(
        graph,
        adaptors,
        robot_poses,
        "==   PDDL Domain (Pick/Place) ==",
        "ObjectRearrangementDomain.pddl",
        "(and (object-in-place o95 p2410))",
    )
    run_pddl_case(
        graph,
        adaptors,
        robot_poses,
        "==   PDDL Domain (Regions)    ==",
        "RegionObjectRearrangementDomain.pddl",
        "(and (visited-region r4) (at-place p1396) (object-in-place o95 p2410))",
    )


if __name__ == "__main__":
    main()
