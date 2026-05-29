import logging
from importlib.resources import as_file, files
from pathlib import Path

import dsg_pddl.domains
import nlu_interface.resources
import numpy as np
from dsg_pddl.core.models import PddlDomain
from nlu_interface.llm_interface import OpenAIWrapper
from ruamel.yaml import YAML
from utils import DummyRobotPlanningAdaptor, build_test_dsg

from omniplanner.domains.language import LanguageDomain, LanguageGoal
from omniplanner.core import (
    PlanRequest,
    full_planning_pipeline,
)
from simple_compile_plan import compile_plan

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

yaml = YAML(typ="safe")
EXAMPLES_DIR = Path(__file__).resolve().parent


def run_goto_points_language_example():
    adaptor = DummyRobotPlanningAdaptor("spot", "spot", "map", "body")

    print("================================")
    print("== Goto Point Language Domain ==")
    print("================================")
    print("")

    goal = LanguageGoal("spot", "O(0) O(1)")
    robot_domain = LanguageDomain("goto_points", None, None)
    robot_poses = {"spot": np.array([0.0, 0.1])}

    req = PlanRequest(
        domain=robot_domain,
        goal=goal,
        robot_states=robot_poses,
    )

    graph = build_test_dsg()
    robot_plan = full_planning_pipeline(req, graph)

    print("Plan from planning domain:")
    print(robot_plan)

    compiled_plan = compile_plan(adaptor, "map", robot_plan)
    print("compiled plan:")
    print(compiled_plan)


def run_pddl_language_example():
    adaptor = DummyRobotPlanningAdaptor("spot", "spot", "map", "body")
    adaptors = {"euclid": adaptor}

    print("================================")
    print("==   PDDL Language Domain     ==")
    print("================================")
    print("")

    goal = LanguageGoal(command="Euclid, go to objects O(0) and O(1)", robot_id="")
    domain_type = "Pddl"
    robot_poses = {"euclid": np.array([0.0, 0.1])}

    with as_file(files(dsg_pddl.domains).joinpath("GotoObjectDomain.pddl")) as path:
        print(f"Loading domain {path}")
        with open(str(path), "r") as fo:
            domain = PddlDomain(fo.read())

    with open(EXAMPLES_DIR / "resources" / "llm_config.yaml", "r") as file:
        llm_config = yaml.load(file)

    with as_file(
        files(nlu_interface.resources).joinpath(llm_config["prompt"] + ".yaml")
    ) as path:
        print(f'Loading prompt from "{path}"')
        with open(str(path), "r") as file:
            prompt = yaml.load(file)

    llm_interface = OpenAIWrapper(
        model=llm_config["model"],
        mode=llm_config["mode"],
        prompt=prompt,
        num_incontext_examples=llm_config["num_incontext_examples"],
        temperature=llm_config["temperature"],
        api_timeout=llm_config["api_timeout"],
        seed=llm_config["seed"],
        api_key_env_var=llm_config["api_key_env_var"],
        debug=llm_config["debug"],
    )

    req = PlanRequest(
        domain=LanguageDomain(domain_type, domain, llm_interface),
        goal=goal,
        robot_states=robot_poses,
    )

    graph = build_test_dsg()
    plan = full_planning_pipeline(req, graph)

    print("Plan from planning domain:")
    print(plan)

    compiled_plan = compile_plan(adaptors, "map", plan)
    print("compiled plan:")
    print(compiled_plan)


def main():
    run_goto_points_language_example()
    run_pddl_language_example()


if __name__ == "__main__":
    main()
