from dataclasses import dataclass

import numpy as np
import pytest

import omniplanner.compile_plan as compiler
import omniplanner.language_planner as lang
import omniplanner.omniplanner as core
from omniplanner.goto_points import (
    GotoPointPrimitive,
    GotoPointsDomain,
    GotoPointsGoal,
    GotoPointsPlan,
    GroundedGotoPointsProblem,
    ground_problem as ground_goto,
    make_plan as make_goto_plan,
)
from omniplanner.language_planner import LanguageDomain, LanguageGoal
from omniplanner.utils import str_to_ns_value


@dataclass
class FakePlan:
    value: int


@dataclass
class FakeCompiled:
    robot: str
    value: int
    frame: str


@compiler.compile_plan.dispatch
def compile_plan(adaptor: str, plan_frame: str, plan: FakePlan):
    return FakeCompiled(adaptor, plan.value, plan_frame)


def test_robot_and_symbolic_wrappers_preserve_context_and_names():
    wrapper = core.RobotWrapper("spot", 2)
    assert core.extract(wrapper) == 2
    assert core.fmap(lambda v: v + 1, wrapper) == core.RobotWrapper("spot", 3)
    assert core.with_new_value(wrapper, 4) == core.RobotWrapper("spot", 4)

    symbolic = core.SymbolicContext({"p1": {"kind": "place"}}, wrapper)
    pushed = core.push(symbolic)
    assert pushed == core.RobotWrapper(
        "spot", core.SymbolicContext({"p1": {"kind": "place"}}, 2)
    )

    pushed_list = core.push(core.RobotWrapper("spot", [1, 2]))
    assert pushed_list == [
        core.RobotWrapper("spot", 1),
        core.RobotWrapper("spot", 2),
    ]


def test_multirobot_wrapper_remaps_and_can_receive_new_value():
    wrapper = core.MultiRobotWrapper(["Spot"], "plan")
    wrapper.set_name_remap("Spot", "spot")

    assert wrapper.remap_name_to_inner("Spot") == "spot"
    assert wrapper.remap_name_to_outer("spot") == "Spot"
    assert wrapper.remap_name_to_inner("missing") is None
    assert wrapper.remap_name_to_outer("missing") is None

    updated = core.with_new_value(wrapper, "new")
    assert updated.names == ["Spot"]
    assert updated.value == "new"
    assert updated.remap_name_to_inner("Spot") == "spot"


def test_dsg_context_provider_merges_explicit_and_dsg_context():
    class NodeSymbol:
        def __init__(self, char, index):
            self.char = char
            self.index = index

        def __eq__(self, other):
            return (self.char, self.index) == (other.char, other.index)

        def __hash__(self):
            return hash((self.char, self.index))

    class Node:
        attributes = type("Attrs", (), {"position": np.array([1.0, 2.0, 3.0])})()
        layer = type("Layer", (), {"layer": 1, "partition": 0})()

    class Labelspace:
        def get_node_category(self, node):
            return "chair"

    class Dsg:
        def __init__(self):
            self.node = Node()

        def find_node(self, ns):
            return self.node if ns == NodeSymbol("O", 7) else None

        def get_labelspace(self, layer, partition):
            return Labelspace()

    original = core.NodeSymbol
    core.NodeSymbol = NodeSymbol
    try:
        provider = core.DsgContextProvider(Dsg())
        provider["manual"] = {"color": "blue"}
        assert "o7" in provider
        assert "manual" in provider
        assert "absent" not in provider

        context = provider["o7"]
        assert np.array_equal(context["position"], np.array([1.0, 2.0, 3.0]))
        assert context["semantic_label"] == "chair"
        context["note"] = "ok"
        assert provider["o7"]["note"] == "ok"
        with pytest.raises(Exception, match="Cannot override DSG context"):
            context["position"] = np.zeros(3)
        with pytest.raises(TypeError):
            provider["bad"] = 1
    finally:
        core.NodeSymbol = original


def test_goto_points_grounding_and_plan_from_numpy_context(monkeypatch):
    monkeypatch.setattr("omniplanner.goto_points.time.sleep", lambda _: None)
    points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])
    start = np.array([-1.0, 0.0])

    grounded = ground_goto(GotoPointsDomain(), points, start, [1, 2])
    assert np.array_equal(grounded.start_point, start)
    assert np.array_equal(grounded.goal_points, points[[1, 2]])

    plan = make_goto_plan(grounded, points)
    assert isinstance(plan, GotoPointsPlan)
    assert len(plan.plan) == 2
    assert np.array_equal(plan.plan[0].start, start)
    assert np.array_equal(plan.plan[0].goal, points[1])
    assert np.array_equal(plan.plan[1].start, points[1])
    assert np.array_equal(plan.plan[1].goal, points[2])


def test_language_goto_points_dispatches_to_goto_grounder(monkeypatch):
    seen = {}

    class FakeDsg:
        pass

    @lang.ground_problem.dispatch
    def fake_ground(domain: GotoPointsDomain, dsg: FakeDsg, states: dict, goal: GotoPointsGoal, feedback=None):
        seen["goal"] = goal
        return "grounded"

    result = lang.ground_problem(
        LanguageDomain("goto_points"),
        FakeDsg(),
        {"spot": np.array([0.0, 0.0])},
        LanguageGoal("spot", "O(1) O(2)"),
    )

    assert result == "grounded"
    assert seen["goal"] == GotoPointsGoal(["O(1)", "O(2)"], "spot")


def test_language_pddl_uses_llm_response_and_feedback(monkeypatch):
    class FakePddlDomain:
        pass

    class FakeLLM:
        def request_plan_specification(self, command, dsg):
            assert command == "clean"
            return "{'spot': '(and (safe o1))'}"

    published = []
    feedback = type(
        "Feedback",
        (),
        {
            "plugin_feedback_collectors": {
                "language_planner": type(
                    "Collector",
                    (),
                    {"publish": {"llm_response": published.append}},
                )()
            }
        },
    )()

    @lang.ground_problem.dispatch
    def fake_pddl_ground(domain: FakePddlDomain, dsg: object, states: dict, goal: object, feedback=None):
        return (goal.robot_id, goal.pddl_goal)

    result = lang.ground_problem(
        LanguageDomain("Pddl", pddl_domain=FakePddlDomain(), llm_interface=FakeLLM()),
        object(),
        {"spot": np.zeros(2)},
        LanguageGoal("", "clean"),
        feedback,
    )

    assert result == [("spot", "(and (safe o1))")]
    assert published == ["{'spot': '(and (safe o1))'}"]
    with pytest.raises(Exception, match="Unexpected domain_type"):
        lang.ground_problem(LanguageDomain("bad"), object(), {}, LanguageGoal("", ""))


def test_str_to_ns_value_parses_dsg_symbol():
    assert isinstance(str_to_ns_value("O(7)"), int)


def test_compile_plan_recurses_through_wrappers_and_collections():
    adaptor = {"spot": "spot-adaptor", "drone": "drone-adaptor"}
    plan = core.SymbolicContext(
        {"p1": {}},
        [
            core.RobotWrapper("spot", FakePlan(1)),
            core.RobotWrapper("drone", FakePlan(2)),
        ],
    )

    compiled = compiler.compile_plan(adaptor, "map", plan)
    assert compiled == [
        core.RobotWrapper("spot", FakeCompiled("spot-adaptor", 1, "map")),
        core.RobotWrapper("drone", FakeCompiled("drone-adaptor", 2, "map")),
    ]
    assert compiler.collect_plans(compiled) == {
        "spot": FakeCompiled("spot-adaptor", 1, "map"),
        "drone": FakeCompiled("drone-adaptor", 2, "map"),
    }
    assert compiler.collect_plans(compiled[0]) == {
        "spot": FakeCompiled("spot-adaptor", 1, "map")
    }


def test_compile_plan_multirobot_remaps_adaptors():
    wrapper = core.MultiRobotWrapper(
        ["Spot"],
        core.SymbolicContext({"p": {}}, core.RobotWrapper("spot", FakePlan(9))),
    )
    wrapper.set_name_remap("Spot", "spot")

    compiled = compiler.compile_plan({"Spot": "spot-adaptor"}, "map", wrapper)
    assert compiled == core.MultiRobotWrapper(
        ["Spot"],
        core.RobotWrapper("spot", FakeCompiled("spot-adaptor", 9, "map")),
        {"Spot": "spot"},
        {"spot": "Spot"},
    )


@core.make_plan.dispatch
def make_plan(problem: FakePlan, map_context: object):
    return FakePlan(problem.value + 10)


@core.ground_problem.dispatch
def ground_problem(domain: core.PlanningDomain, map_context: object, robot_states: dict, goal: core.PlanningGoal, feedback=None):
    return FakePlan(robot_states["value"])


def test_full_planning_pipeline_uses_ground_and_make_dispatch():
    req = core.PlanRequest(core.PlanningDomain(), core.PlanningGoal(), {"value": 5})
    result = core.full_planning_pipeline(req, object())

    assert isinstance(result, core.SymbolicContext)
    assert result.value == FakePlan(15)
    assert core.full_planning_pipeline([req], object())[0].value == FakePlan(15)


def test_default_dispatch_errors_are_descriptive():
    assert "No matching specialization" in str(core.DispatchException("fn", object()))
    with pytest.raises(core.DispatchException, match="No matching specialization"):
        core.make_plan(core.GroundedProblem(), object())
    with pytest.raises(TypeError):
        core.push(core.RobotWrapper("spot", 1))
