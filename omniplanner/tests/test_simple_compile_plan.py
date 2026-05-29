import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from dsg_pddl.planning.parameterization import PddlPlan
from omniplanner.compile_plan import collect_plans
from omniplanner.core import MultiRobotWrapper, RobotWrapper, SymbolicContext
from omniplanner.domains.goto_points import GotoPointPrimitive, GotoPointsPlan
from simple_compile_plan import compile_plan


def test_simple_action_sequence_can_be_collected():
    adaptor = SimpleNamespace(name="spot")
    plan = GotoPointsPlan(
        [
            GotoPointPrimitive(np.array([0.0, 0.1]), np.array([1.0, 0.0])),
        ]
    )

    compiled = compile_plan(adaptor, "map", plan)

    assert collect_plans(compiled) == {"spot": compiled}


def test_multirobot_pddl_plan_is_split_and_collected_by_robot():
    plan = PddlPlan(
        domain=None,
        symbolic_actions=[
            ("goto-poi", "robot1", "p0", "p1"),
            ("inspect", "robot2", "o1"),
        ],
        parameterized_actions=["robot1-path", "robot2-path"],
        symbols={},
    )
    wrapper = MultiRobotWrapper(["robot1", "robot2"], SymbolicContext({}, plan))
    wrapper.set_name_remap("robot1", "robot1")
    wrapper.set_name_remap("robot2", "robot2")
    adaptors = {
        "robot1": SimpleNamespace(name="alpha"),
        "robot2": SimpleNamespace(name="beta"),
    }

    compiled = compile_plan(adaptors, "map", wrapper)

    assert [item.name for item in compiled] == ["robot1", "robot2"]
    assert all(isinstance(item, RobotWrapper) for item in compiled)
    assert compiled[0].value.robot_name == "alpha"
    assert compiled[0].value.actions[0].parameters == {
        "symbolic": ("goto-poi", "p0", "p1"),
        "path": "robot1-path",
    }
    assert compiled[1].value.robot_name == "beta"
    assert compiled[1].value.actions[0].parameters == {
        "symbolic": ("inspect", "o1"),
        "path": "robot2-path",
    }
    assert collect_plans(compiled) == {
        "robot1": compiled[0].value,
        "robot2": compiled[1].value,
    }
