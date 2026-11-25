"""Exercise adjacent manipulation and repair with the real Fast Downward solver."""

import shutil
from importlib.resources import files

import numpy as np
import pytest
import spark_dsg

from dsg_pddl.core.models import PddlDomain, PddlGoal, RegionMissionState
from dsg_pddl.grounding.improved_region import ground_improved_problem
from dsg_pddl.planning.parameterization import make_plan
from dsg_pddl.planning.solver import PddlSolveError

pytestmark = pytest.mark.skipif(not shutil.which("fast-downward"), reason="requires Fast Downward")
DOMAINS = ("ExplicitState", "DerivedPredicates")


@pytest.fixture
def graph():
    result = spark_dsg.DynamicSceneGraph()
    layers = spark_dsg.DsgLayers
    for layer in (layers.MESH_PLACES, layers.ROOMS, layers.OBJECTS):
        key = layers.name_to_layer_id(layer)
        result.add_layer(key.layer, key.partition, layer)
    for index, xy in enumerate(((0., 0.), (.5, 0.), (1., 0.), (1.5, 0.), (0., .5), (1., .5))):
        attrs = spark_dsg.PlaceNodeAttributes()
        attrs.position = (*xy, 0.)
        result.add_node(layers.MESH_PLACES, spark_dsg.NodeSymbol("P", index).value, attrs)
    for a, b in ((0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 2)):
        assert result.insert_edge(spark_dsg.NodeSymbol("P", a).value, spark_dsg.NodeSymbol("P", b).value)
    attrs = spark_dsg.ObjectNodeAttributes()
    attrs.position = (.5, 0., .1)
    attrs.semantic_label = 1
    result.add_node(layers.OBJECTS, spark_dsg.NodeSymbol("O", 1).value, attrs)
    return result


def solve(graph, domain, state, goal="(object-in-place o1 p3)", pose=(0., 0.)):
    source = files("dsg_pddl.domains").joinpath(f"RegionObjectRearrangementDomain_{domain}.pddl")
    request = PddlGoal(goal, "robot", state)
    problem = ground_improved_problem(PddlDomain(source.read_text()), graph, {"robot": np.array(pose)}, request)
    return make_plan(problem.value, graph)


@pytest.mark.parametrize("domain", DOMAINS)
@pytest.mark.parametrize("sampler", ("all", "paths", "compressed"))
def test_adjacent_manipulation_and_initial_pose(graph, domain, sampler, monkeypatch):
    monkeypatch.setenv("PDDL_SAMPLER", sampler)
    plan = solve(graph, domain, RegionMissionState())
    assert plan.symbolic_actions[0] == ("pick-object", "o1", "p1", "p0")
    assert plan.symbolic_actions[-1] == ("place-object", "o1", "p3", "p2")
    stance, target = plan.parameterized_actions[0]
    np.testing.assert_allclose(stance, (0., 0., 0.))
    np.testing.assert_allclose(target, (.5, 0., .1))


@pytest.mark.parametrize("domain", DOMAINS)
def test_repair_uses_detour_while_holding_and_preserves_progress(graph, domain):
    state = RegionMissionState(
        visited_places={"p1"}, holding="o1", blocked_edges={frozenset(("p0", "p1"))},
    )
    plan = solve(graph, domain, state, "(and (visited-place p1) (object-in-place o1 p3))")
    assert plan.symbolic_actions[-1] == ("place-object", "o1", "p3", "p2")
    assert all(action[0] != "pick-object" for action in plan.symbolic_actions)
    for action, points in zip(plan.symbolic_actions, plan.parameterized_actions):
        if action[0] == "goto-poi":
            np.testing.assert_allclose(points[0], plan.symbols[action[1]].position)
            np.testing.assert_allclose(points[-1], plan.symbols[action[2]].position)
    path = [tuple(point) for action, points in zip(plan.symbolic_actions, plan.parameterized_actions)
            if action[0] == "goto-poi" for point in points]
    assert (0., .5) in path and (.5, 0.) not in path

    plan = solve(graph, domain, state, pose=(1., 0.))
    assert plan.symbolic_actions == [("place-object", "o1", "p3", "p2")]
    complete = RegionMissionState(placements={"o1": "p3"})
    assert solve(graph, domain, complete).symbolic_actions == []
    repick = solve(graph, domain, complete, "(holding o1)", pose=(1., 0.))
    np.testing.assert_allclose(repick.parameterized_actions[0][1], (1.5, 0., 0.))


@pytest.mark.parametrize("domain", DOMAINS)
def test_reports_unsolvable_when_no_stance_is_in_reach(graph, domain, monkeypatch):
    monkeypatch.setenv("PDDL_MANIPULATION_REACH_M", "0.1")
    with pytest.raises(PddlSolveError, match="unsolvable"):
        solve(graph, domain, RegionMissionState())
