from types import SimpleNamespace

import numpy as np
import pytest
import spark_dsg

import dsg_pddl.grounding.legacy as grounding
import dsg_pddl.grounding.improved_region as improved
import dsg_pddl.grounding.multirobot as multirobot
import dsg_pddl.grounding.connectivity as grounding_connectivity
import dsg_pddl.grounding.containment as grounding_containment
import dsg_pddl.grounding.dsg_access as grounding_dsg_access
import dsg_pddl.grounding.improved_region as improved_impl
import dsg_pddl.grounding.legacy as grounding_impl
import dsg_pddl.grounding.multirobot as multirobot_impl
import dsg_pddl.grounding.symbols as grounding_symbols
import dsg_pddl.planning.parameterization as dsg_planning
import dsg_pddl.planning.parameterization as parameterization
import dsg_pddl.planning.solver as pddl_planning
from dsg_pddl.core.models import (
    GroundedPddlProblem,
    PddlDomain,
    PddlGoal,
    PddlProblem,
    PddlSymbol,
)
from dsg_pddl.core.domain_inspection import ensure_pddl_domain
from dsg_pddl.core.parsing import (
    ast_to_string,
    extract_facts,
    extract_negated_facts,
    lisp_string_to_ast,
    pddl_char_to_dsg_char,
    tokenize_lisp,
)


DOMAIN = """
(define (domain demo)
  (:requirements :strips)
  (:types place dsg_object)
  (:predicates (at-poi ?p - place))
  (:functions (total-cost))
  (:action goto-poi
    :parameters (?from - place ?to - place)
    :precondition (at-poi ?from)
    :effect (and (not (at-poi ?from)) (at-poi ?to))))
  (:derived (reachable ?p - place) (at-poi ?p))
)
"""


def test_lisp_helpers_and_fact_extraction():
    assert tokenize_lisp("(and (a b))") == ["(", "and", "(", "a", "b", ")", ")"]
    ast = lisp_string_to_ast("(and (visited-place p1) (not (visited-place p2)) (safe o1))")
    assert ast_to_string(ast) == "(and (visited-place p1) (not (visited-place p2)) (safe o1))"
    assert extract_facts(ast, "visited-place") == (
        ("visited-place", "p1"),
        ("visited-place", "p2"),
    )
    assert extract_negated_facts(ast, "visited-place") == (("visited-place", "p2"),)
    assert extract_facts("leaf", "visited-place") == ()
    assert pddl_char_to_dsg_char("o") == "O"
    assert pddl_char_to_dsg_char("p") == "P"
    assert pddl_char_to_dsg_char("x") == "x"
    with pytest.raises(SyntaxError):
        lisp_string_to_ast(")")


def test_pddl_domain_problem_and_symbols():
    domain = PddlDomain(DOMAIN)
    assert domain.domain_name == "demo"
    assert domain.requirements == (":strips",)
    assert "place" in domain.domain_types
    assert domain.predicates == (("at-poi", "?p", "-", "place"),)
    assert domain.functions == (("total-cost",),)
    assert domain.actions[0] == "goto-poi"
    assert isinstance(domain.derived, tuple)
    assert "(domain demo)" in domain.to_string()

    ensure_pddl_domain(domain.domain_ast)
    with pytest.raises(Exception, match="missing define"):
        ensure_pddl_domain(("bad",))

    symbol = PddlSymbol("p1", "place", [])
    assert symbol == PddlSymbol("p1", "place", [])
    assert symbol < PddlSymbol("p2", "place", [])
    assert len({symbol, PddlSymbol("p1", "place", [])}) == 1

    problem = PddlProblem(
        name="prob",
        domain="demo",
        objects={"place": ["p1", "p2"], "": ["robot"]},
        initial_facts=(("at-poi", "p1"), ("=", ("total-cost",), 0)),
        goal=("and", ("at-poi", "p2")),
        optimizing=True,
    )
    text = problem.to_string()
    assert "p1 p2 - place" in text
    assert "robot" in text
    assert "(:metric minimize (total-cost))" in text


def test_grounding_pure_helpers(monkeypatch):
    p1 = PddlSymbol("p1", "place", [], np.array([0.0, 0.0]))
    p2 = PddlSymbol("p2", "place", [], np.array([3.9, 0.0]))
    o1 = PddlSymbol("o1", "object", [], np.array([1.0, 0.0]))
    r1 = PddlSymbol("r1", "region", [])

    assert grounding.symbol_connectivity_to_pddl([(p2, p1, 3.9)]) == [
        ("connected", "p2", "p1"),
        ("=", ("distance", "p2", "p1"), 3),
        ("=", ("distance", "p1", "p2"), 3),
    ]
    assert grounding.extract_symbols_of_interest(
        object(), ("and", ("visited-place", "p3"), ("at-object", "o4"))
    ) == [PddlSymbol("p3", "place", []), PddlSymbol("o4", "object", [])]
    assert grounding.normalize_symbol("P(1)") == "p(1)"
    grounding.normalize_symbols([PddlSymbol("P1", "place", [])])
    assert grounding.generate_objects([p1, o1, r1]) == {
        "place": ["p1"],
        "dsg_object": ["o1"],
        "region": ["r1"],
    }

    class FakeLayerPlanner:
        def __init__(self, graph, layer):
            pass

        def get_external_distance(self, a, b):
            return float(np.linalg.norm(a - b))

    monkeypatch.setattr(grounding_impl, "LayerPlanner", FakeLayerPlanner)
    assert grounding.generate_symbol_connectivity(object(), [p1, p2]) == [(p2, p1, 3.9)]
    assert ("at-poi", "p1") in grounding.generate_init(object(), [p1, p2], p1)


def test_grounding_layer_helpers_with_fake_dsg(monkeypatch):
    class FakeId:
        def __init__(self, text, value):
            self.text = text
            self.value = value

        def str(self, include_prefix):
            return self.text

    class FakeNode:
        def __init__(self, text, value, position, siblings=(), parents=(), semantic_label=0):
            self.id = FakeId(text, value)
            self.attributes = SimpleNamespace(
                position=np.array(position, dtype=float),
                semantic_label=semantic_label,
            )
            self._siblings = siblings
            self._parents = parents

        def siblings(self):
            return self._siblings

        def parents(self):
            return self._parents

    class FakeLayer:
        def __init__(self, nodes):
            self.nodes = nodes

    class FakeNodeSymbol:
        def __init__(self, value_or_char, index=None):
            if index is None:
                self.text = "R(5)"
            else:
                self.text = f"{value_or_char}({index})"

        def str(self, include_prefix):
            return self.text

    fake_spark = SimpleNamespace(
        DsgLayers=SimpleNamespace(MESH_PLACES="places", OBJECTS="objects", ROOMS="rooms"),
        DynamicSceneGraph=object,
        LayerView=object,
        NodeSymbol=FakeNodeSymbol,
    )
    monkeypatch.setattr(grounding_impl, "spark_dsg", fake_spark)
    monkeypatch.setattr(grounding_connectivity, "spark_dsg", fake_spark)
    monkeypatch.setattr(grounding_containment, "spark_dsg", fake_spark)
    monkeypatch.setattr(grounding_dsg_access, "spark_dsg", fake_spark)
    monkeypatch.setattr(grounding_symbols, "spark_dsg", fake_spark)

    p1_node = FakeNode("P(1)", 1, [0.0, 0.0, 0.0], siblings=[2], parents=[5])
    p2_node = FakeNode("P(2)", 2, [1.0, 0.0, 0.0], siblings=[1], parents=[5])
    o1_node = FakeNode("O(1)", 10, [0.2, 0.0, 0.0])
    r1_node = FakeNode("R(5)", 5, [0.0, 0.0, 0.0])

    class FakeDsg:
        def get_layer(self, layer):
            return {
                "places": FakeLayer([p1_node, p2_node]),
                "objects": FakeLayer([o1_node]),
                "rooms": FakeLayer([r1_node]),
            }[layer]

        def get_node(self, value):
            return {1: p1_node, 2: p2_node, 10: o1_node, 5: r1_node}[value]

    graph = FakeDsg()
    lookup = {
        "p(1)": PddlSymbol("p(1)", "place", []),
        "p(2)": PddlSymbol("p(2)", "place", []),
        "o(1)": PddlSymbol("o(1)", "object", []),
        "r(5)": PddlSymbol("r(5)", "region", []),
    }

    assert grounding.explicit_edges_from_layer(lookup, graph, graph.get_layer("places")) == [
        (lookup["p(2)"], lookup["p(1)"], 1.0)
    ]
    assert grounding.implicit_edges_from_layers(
        lookup, graph.get_layer("objects"), graph.get_layer("places"), False, 1.0
    )[0][:2] == (lookup["o(1)"], lookup["p(1)"])
    assert grounding.generate_object_containment(graph) == [("object-in-place", "o(1)", "p(1)")]
    assert grounding.generate_place_containment(graph) == [
        ("place-in-region", "p(1)", "r(5)"),
        ("place-in-region", "p(2)", "r(5)"),
    ]
    assert {s.symbol for s in grounding.extract_all_symbols(graph)} == {"P(1)", "P(2)", "O(1)", "R(5)"}

    unresolved = PddlSymbol("p1", "place", [])
    monkeypatch.setattr(grounding_symbols.spark_dsg, "NodeSymbol", lambda char, idx: 1)
    grounding.add_symbol_positions(graph, [unresolved])
    assert np.array_equal(unresolved.position, np.array([0.0, 0.0]))


def test_grounding_pddl_generators_with_stubbed_graph(monkeypatch):
    all_symbols = [
        PddlSymbol("P1", "place", []),
        PddlSymbol("O1", "object", []),
        PddlSymbol("R1", "region", []),
    ]

    monkeypatch.setattr(grounding_impl, "extract_all_symbols", lambda graph: all_symbols[:])
    monkeypatch.setattr(grounding_impl, "add_symbol_positions", lambda graph, symbols: symbols)
    monkeypatch.setattr(grounding_impl, "generate_init", lambda graph, symbols, start: [("at-poi", start.symbol)])
    monkeypatch.setattr(grounding_impl, "generate_dense_init", lambda graph, symbols, start: [("dense", start.symbol)])
    monkeypatch.setattr(grounding_impl, "generate_dense_region_init", lambda graph, symbols, start: [("region", start.symbol)])

    inspection_text, inspection_symbols = grounding.generate_inspection_pddl(
        object(), "(and (visited-place p1) (at-object o1))", np.array([0.0, 0.0])
    )
    assert "goto-object-problem" in inspection_text
    assert {s.symbol for s in inspection_symbols} == {"pstart", "p1", "o1"}

    rearrange_text, rearrange_symbols = grounding.generate_rearrangement_pddl(
        object(), "(and (safe o1))", np.array([0.0, 0.0])
    )
    assert "object-rearrangement-domain" in rearrange_text
    assert {s.symbol for s in rearrange_symbols} == {"pstart", "p1", "o1"}

    region_text, region_symbols = grounding.generate_region_pddl(
        object(), "(and (visited-region r1))", np.array([0.0, 0.0])
    )
    assert "region-object-rearrangement-domain" in region_text
    assert {s.symbol for s in region_symbols} == {"pstart", "p1", "o1", "r1"}


def test_pddl_ground_problem_routes_legacy_and_improved_domains(monkeypatch):
    graph = spark_dsg.DynamicSceneGraph()
    states = {"spot": np.array([0.0, 0.0])}
    goal = PddlGoal("(and)", "spot")
    symbols = [PddlSymbol("pstart", "place", [], np.array([0.0, 0.0]))]

    monkeypatch.setattr(
        grounding_impl,
        "generate_inspection_pddl",
        lambda graph, goal, start: ("legacy-problem", symbols),
    )
    legacy_domain = PddlDomain(DOMAIN.replace("(domain demo)", "(domain goto-object-domain)"))
    legacy = grounding.ground_problem(legacy_domain, graph, states, goal)
    assert legacy.name == "spot"
    assert legacy.value.problem_str == "legacy-problem"

    monkeypatch.setattr(
        improved_impl,
        "generate_region_rearrangement_pddl_compressed_graph",
        lambda graph, goal, start, domain_name: ("improved-problem", symbols),
    )
    improved_domain = PddlDomain(
        DOMAIN.replace(
            "(domain demo)",
            f"(domain {improved.REGION_REARRANGEMENT_EXPLICIT_STATE_DOMAIN})",
        )
    )
    routed = grounding.ground_problem(improved_domain, graph, states, goal)
    assert routed.name == "spot"
    assert routed.value.problem_str == "improved-problem"


def test_improved_grounding_selection_helpers():
    goal = lisp_string_to_ast(
        "(and (at-poi p1) (object-in-place o1 p2) (visited-region r1) "
        "(not (visited-place p9)) (not (visited-object o9)))"
    )
    symbols, forbidden = improved.extract_goal_symbols(goal)
    assert {s.symbol for s in symbols} == {"p1", "p2", "o1", "o9", "p9", "r1"}
    assert {s.symbol for s in forbidden["places"]} == {"p9"}
    assert {s.symbol for s in forbidden["objects"]} == {"o9"}

    symbol_lookup = {
        "o1": PddlSymbol("o1", "object", []),
        "p1": PddlSymbol("p1", "place", []),
        "r1": PddlSymbol("r1", "region", []),
    }
    soi = [symbol_lookup["o1"]]
    forbidden = {"places": [], "objects": [symbol_lookup["o1"]], "regions": [symbol_lookup["r1"]]}
    improved.add_object_related_places(soi, forbidden, [("object-in-place", "o1", "p1")], symbol_lookup)
    improved.add_region_related_places(soi, forbidden, [("place-in-region", "p1", "r1")], symbol_lookup)
    assert symbol_lookup["p1"] in soi
    assert symbol_lookup["p1"] in forbidden["places"]

    planner = SimpleNamespace(symbol_to_node_value={"p1": 1, "p9": 9})
    assert improved.build_forbidden_nodes({"places": [PddlSymbol("p9", "place", [])]}, planner) == {9}

    suspicious, unsafe = improved.process_suspicious_objects(
        [("suspicious", "o1")],
        [("object-in-place", "o1", "p1")],
        [symbol_lookup["p1"]],
        symbol_lookup,
    )
    assert suspicious == [("suspicious", "o1")]
    assert unsafe == [("unsafe-place", "p1")]
    assert symbol_lookup["o1"] in [symbol_lookup["p1"], symbol_lookup["o1"]]


def test_improved_graph_compression_helpers():
    paths = {(1, 3): [1, 2, 3], (4, 3): [4, 2, 3]}
    secondary = improved.compute_secondary_nodes_from_paths(paths, {1, 3, 4})
    assert secondary == {2}
    segments = improved.build_segments_from_paths(paths, {1, 3, 4}, secondary, set())
    assert segments[(1, 2)] == [1, 2]
    assert segments[(2, 1)] == [2, 1]

    planner = SimpleNamespace(
        node_value_to_symbol={1: "p1", 2: "p2"},
    )
    lookup = {"p1": PddlSymbol("p1", "place", []), "p2": PddlSymbol("p2", "place", [])}
    assert improved.build_compressed_edges([(1, 2, 5, [1, 2])], list(lookup.values()), planner, lookup) == [
        (lookup["p1"], lookup["p2"], 5)
    ]


def test_improved_pddl_generators_with_stubbed_init(monkeypatch):
    all_symbols = [PddlSymbol("P1", "place", []), PddlSymbol("O1", "object", [])]
    monkeypatch.setattr(improved_impl, "extract_all_symbols", lambda graph: all_symbols[:])
    monkeypatch.setattr(improved_impl, "add_symbol_positions", lambda graph, symbols: symbols)
    monkeypatch.setattr(improved_impl, "generate_dense_places_init", lambda graph, symbols, start: [("at-poi", start.symbol)])
    monkeypatch.setattr(improved_impl, "generate_improved_places_init", lambda graph, symbols, start, forbidden: ([("at-poi", start.symbol)], symbols))
    monkeypatch.setattr(improved_impl, "generate_improved_places_init_v2", lambda graph, symbols, start, forbidden: ([("at-poi", start.symbol)], symbols))

    for generator in (
        improved.generate_region_rearrangement_pddl_all_symbols,
        improved.generate_region_rearrangement_pddl_relevant_paths,
        improved.generate_region_rearrangement_pddl_compressed_graph,
    ):
        text, symbols = generator(object(), "(and (safe o1))", np.array([0.0, 0.0]))
        assert improved.REGION_REARRANGEMENT_EXPLICIT_STATE_DOMAIN in text
        assert any(s.symbol == "pstart" for s in symbols)



def test_multirobot_helpers_and_grounding(monkeypatch, tmp_path):
    place_symbols = [
        PddlSymbol("p1", "place", [], np.array([0.0, 0.0])),
        PddlSymbol("p2", "place", [], np.array([5.0, 0.0])),
    ]
    assert multirobot.nearest_place_for_position(place_symbols, np.array([4.0, 0.0])) == "p2"
    assert multirobot.filter_goal_for_available_objects("(and (safe o1) (safe o2))", ["o2"]) == "(and (safe o2))"
    assert multirobot.filter_goal_for_available_objects("(and (safe o1))", []) == "(and)"

    monkeypatch.setenv("PDDL_DUMP_DIR", str(tmp_path))
    monkeypatch.setattr(multirobot_impl, "extract_all_symbols", lambda graph: place_symbols[:])
    monkeypatch.setattr(multirobot_impl, "add_symbol_positions", lambda graph, symbols: symbols)
    monkeypatch.setattr(multirobot_impl, "generate_dense_region_init_multirobot", lambda graph, symbols, states: [("at-poi", "spot", "pstartspot")])

    text, symbols = multirobot.generate_multirobot_region_pddl(
        object(), "(and (safe o2))", {"Spot": np.array([1.0, 2.0]), "Bad": None}
    )
    assert "multi-robot-problem" in text
    assert "Spot - robot" in text
    assert any(s.symbol == "pstartSpot" for s in symbols)
    assert (tmp_path / "mr_region_problem_latest.pddl").exists()


def test_dsg_planning_parameterization_and_solve(monkeypatch, tmp_path):
    symbols = {
        "p1": PddlSymbol("p1", "place", [], np.array([0.0, 0.0])),
        "p2": PddlSymbol("p2", "place", [], np.array([1.0, 0.0])),
        "o1": PddlSymbol("o1", "object", [], np.array([1.0, 1.0])),
    }

    class FakeLayerPlanner:
        def __init__(self, graph, layer):
            pass

        def get_external_path(self, a, b):
            return [a, b]

    monkeypatch.setattr(parameterization, "LayerPlanner", FakeLayerPlanner)
    monkeypatch.setattr(
        parameterization,
        "solve_pddl",
        lambda problem: [
            ("goto-poi", "p1", "p2"),
            ("inspect", "o1"),
            ("pick-object", "o1", "p2"),
            ("place-object", "o1", "p1"),
        ],
    )
    grounded = GroundedPddlProblem(PddlDomain(DOMAIN), "problem", symbols)
    plan = dsg_planning.make_plan(grounded, object())
    assert plan.symbolic_actions[0] == ("goto-poi", "p1", "p2")
    assert len(plan.parameterized_actions) == 4

    assert dsg_planning.drop_index(("a", "b", "c"), 1) == ("a", "c")
    assert dsg_planning.parameterize_goto_poi_multirobot(FakeLayerPlanner(None, None), symbols, ("goto-poi", "robot", "p1", "p2"))[1].tolist() == [1.0, 0.0]


def test_pddl_planning_selects_plan_without_fast_downward(monkeypatch, tmp_path):
    def fake_run(problem, domain, search_cmd, timeout, results, key):
        if key == "suboptimal":
            results[key] = ["(goto-poi p1 p2)\n", "; cost = 1\n"]
        else:
            results[key] = None

    monkeypatch.setenv("DEBUG_OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(pddl_planning, "_run_fd", fake_run)
    problem = GroundedPddlProblem(PddlDomain(DOMAIN), "(define (problem p))", {})
    assert pddl_planning.solve_pddl(problem) == [("goto-poi", "p1", "p2")]

    assert (tmp_path / "problem.pddl").read_text() == "(define (problem p))"
    assert (tmp_path / "plan.txt").read_text() == "(goto-poi p1 p2)\n; cost = 1\n"
