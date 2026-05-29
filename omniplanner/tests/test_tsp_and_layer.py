import networkx as nx
import numpy as np
import pytest
from types import SimpleNamespace

import omniplanner.tsp as tsp
import omniplanner.omniplanner as core
from omniplanner.tsp import (
    FollowPathPlan,
    GroundedTspProblem,
    LayerPlanner,
    TspDomain,
    TspGoal,
    make_plan,
    solve_tsp_2opt,
    two_opt,
)


class FakeAttrs:
    def __init__(self, position):
        self.position = np.array(position, dtype=float)


class FakeNodeId:
    def __init__(self, name, value):
        self._name = name
        self.value = value

    def str(self, include_prefix):
        return self._name


class FakeNode:
    def __init__(self, name, value, position):
        self.id = FakeNodeId(name, value)
        self.attributes = FakeAttrs(position)


class FakeDsg:
    def __init__(self):
        self.graph = nx.Graph()
        self.nodes = {
            1: FakeNode("P(1)", 1, [0.0, 0.0, 0.0]),
            2: FakeNode("P(2)", 2, [1.0, 0.0, 0.0]),
            3: FakeNode("P(3)", 3, [2.0, 0.0, 0.0]),
        }
        self.graph.add_edges_from([(1, 2), (2, 3)])

    def get_layer(self, layer):
        if layer == "fail":
            raise RuntimeError("missing")
        return "layer"

    def get_node(self, value):
        return self.nodes[value]


def test_layer_planner_paths_distances_and_external_projection(monkeypatch):
    fake = FakeDsg()
    monkeypatch.setattr(tsp, "spark_dsg", SimpleNamespace(DsgLayers=SimpleNamespace(MESH_PLACES="mesh")))
    monkeypatch.setattr(tsp.dsg_nx, "layer_to_networkx", lambda layer: fake.graph)

    planner = LayerPlanner(fake, "mesh", precompute_shortest_paths=True)
    assert planner.get_shortest_distance(1, 3) == 2
    assert planner.get_shortest_path(1, 3) == [1, 2, 3]
    assert planner.get_shortest_distance(1, 3, forbidden_nodes={2}) == np.inf
    with pytest.raises(nx.NetworkXNoPath):
        planner.get_shortest_path(1, 3, forbidden_nodes={1})

    point = np.array([1.1, 0.0])
    assert planner.get_closest_node_id(point) == 2
    assert np.array_equal(planner.get_closest_point(point), np.array([1.0, 0.0]))
    assert planner.get_external_distance(np.array([0.0, 0.0]), np.array([2.0, 0.0])) == 2
    assert len(planner.get_external_path(np.array([0.0, 0.0]), np.array([2.0, 0.0]))) == 5


def test_layer_planner_falls_back_to_numeric_mesh_layer(monkeypatch):
    fake = FakeDsg()
    calls = []

    def get_layer(layer):
        calls.append(layer)
        if layer == "mesh":
            raise RuntimeError("missing mesh")
        return "fallback"

    fake.get_layer = get_layer
    monkeypatch.setattr(tsp, "spark_dsg", SimpleNamespace(DsgLayers=SimpleNamespace(MESH_PLACES="mesh")))
    monkeypatch.setattr(tsp.dsg_nx, "layer_to_networkx", lambda layer: fake.graph)

    LayerPlanner(fake, "mesh")
    assert calls == ["mesh", 20]


def test_two_opt_and_tsp_plan(monkeypatch):
    costs = np.array(
        [
            [0, 1, 10, 1],
            [1, 0, 1, 10],
            [10, 1, 0, 1],
            [1, 10, 1, 0],
        ],
        dtype=float,
    )
    assert two_opt([0, 2, 1, 3], costs) == [0, 1, 2, 3]
    assert solve_tsp_2opt(costs) == [0, 1, 2, 3]

    class FakeLayerPlanner:
        def __init__(self, map_context, layer):
            pass

        def get_external_path(self, a, b):
            return [a, b]

    monkeypatch.setattr(tsp, "LayerPlanner", FakeLayerPlanner)
    problem = GroundedTspProblem(
        start_point=np.array([0.0, 0.0]),
        goal_points=np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
        distances=np.array([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=float),
        solver="2opt",
    )

    plan = make_plan(problem, object())
    assert isinstance(plan, FollowPathPlan)
    assert len(plan.steps) == 2
    assert np.array_equal(plan.steps[0].path[0], np.array([0.0, 0.0]))
    assert np.array_equal(plan.steps[0].path[1], np.array([1.0, 0.0]))
    assert np.array_equal(plan.steps[1].path[0], np.array([1.0, 0.0]))
    assert np.array_equal(plan.steps[1].path[1], np.array([2.0, 0.0]))

    with pytest.raises(NotImplementedError):
        make_plan(GroundedTspProblem(np.zeros(2), np.zeros((1, 2)), np.zeros((1, 1)), "bad"), object())


def test_tsp_domain_dataclass():
    assert TspDomain("2opt").solver == "2opt"


def test_tsp_ground_problem_from_symbolic_goal(monkeypatch):
    class FakeNode:
        def __init__(self, position):
            self.attributes = SimpleNamespace(position=np.array(position, dtype=float))

    class FakeDsg:
        def find_node(self, value):
            return {
                "o1": FakeNode([1.0, 0.0, 0.0]),
                "o2": FakeNode([2.0, 0.0, 0.0]),
            }.get(value)

    class FakeLayerPlanner:
        def __init__(self, graph, layer):
            pass

        def get_external_distance(self, a, b):
            return float(np.linalg.norm(a - b))

    monkeypatch.setattr(tsp, "str_to_ns_value", lambda symbol: symbol.lower().replace("(", "").replace(")", ""))
    monkeypatch.setattr(tsp, "LayerPlanner", FakeLayerPlanner)

    wrapped = core.ground_problem(
        TspDomain("2opt"),
        FakeDsg(),
        {"spot": np.array([0.0, 0.0, 0.0])},
        TspGoal(["O(1)", "O(2)"], "spot"),
    )
    assert wrapped.name == "spot"
    assert wrapped.value.distances[0, 2] == 2.0
    with pytest.raises(Exception, match="not in scene graph"):
        core.ground_problem(TspDomain("2opt"), FakeDsg(), {"spot": np.zeros(3)}, TspGoal(["O(9)"], "spot"))
