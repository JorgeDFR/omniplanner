import logging
from typing import Any

import spark_dsg
from plum import dispatch

from dsg_pddl.grounding.connectivity import (
    explicit_edges_from_layer,
    implicit_edges_from_layers,
    symbol_connectivity_to_pddl,
)
from dsg_pddl.grounding.containment import (
    generate_object_containment,
    generate_place_containment,
)
from dsg_pddl.grounding.dsg_access import get_places_layer
from dsg_pddl.grounding.symbols import (
    add_symbol_positions,
    extract_all_symbols,
    generate_objects,
    normalize_symbol,
    normalize_symbols,
)
from dsg_pddl.core.models import (
    GroundedPddlProblem,
    PddlDomain,
    PddlGoal,
    PddlProblem,
    PddlSymbol,
)
from dsg_pddl.core.parsing import (
    extract_facts,
    lisp_string_to_ast,
)
from omniplanner.core.wrappers import RobotWrapper
from omniplanner.domains.tsp import LayerPlanner

logger = logging.getLogger(__name__)

GOTO_OBJECT_DOMAIN = "goto-object-domain"
OBJECT_REARRANGEMENT_DOMAIN = "object-rearrangement-domain"
REGION_OBJECT_REARRANGEMENT_DOMAIN = "region-object-rearrangement-domain"


@dispatch
def ground_problem(
    domain: PddlDomain,
    dsg: spark_dsg.DynamicSceneGraph,
    robot_states: dict,
    goal: PddlGoal,
    feedback: Any = None,
) -> RobotWrapper[GroundedPddlProblem]:
    logger.info(f"Grounding PDDL Problem {domain.domain_name}")

    start = robot_states[goal.robot_id][:2]

    # TODO: TBD whether we want to check the domain here and choose how
    # to instantiate the PDDL problem, or if that should be in a separately
    # ground_problem function.
    match domain.domain_name:
        case "goto-object-domain":
            pddl_problem, symbols = generate_inspection_pddl(dsg, goal.pddl_goal, start)
        case "object-rearrangement-domain":
            pddl_problem, symbols = generate_rearrangement_pddl(
                dsg, goal.pddl_goal, start
            )
        case "region-object-rearrangement-domain":
            pddl_problem, symbols = generate_region_pddl(dsg, goal.pddl_goal, start)
        case _:
            return _ground_improved_pddl_problem(
                domain,
                dsg,
                robot_states,
                goal,
                feedback,
            )

    symbol_dict = {s.symbol: s for s in symbols}
    return RobotWrapper[GroundedPddlProblem](
        goal.robot_id, GroundedPddlProblem(domain, pddl_problem, symbol_dict)
    )


def _ground_improved_pddl_problem(domain, dsg, robot_states, goal, feedback=None):
    from dsg_pddl.grounding.improved_region import ground_improved_problem

    return ground_improved_problem(domain, dsg, robot_states, goal, feedback)


def generate_symbol_connectivity(G, symbols):
    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)

    connections = []
    for si in symbols:
        for sj in symbols:
            if si <= sj:
                continue

            distance = layer_planner.get_external_distance(si.position, sj.position)
            connections.append((si, sj, distance))

    return connections


def generate_init(G, symbols_of_interest, start_symbol):
    connectivity = generate_symbol_connectivity(G, symbols_of_interest)
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl = [("=", ("total-cost",), 0), ("at-poi", start_symbol.symbol)]
    initial_pddl += connectivity_pddl
    return initial_pddl


def add_start_symbol_edges(
    edges,
    symbols,
    layer_planner,
    start_symbol_key="pstart",
    threshold=3,
):
    symbol_lookup = {s.symbol: s for s in symbols}
    start_symbol = symbol_lookup[start_symbol_key]
    start_position = start_symbol.position

    for symbol in symbols:
        if symbol.symbol == start_symbol_key:
            continue

        distance = layer_planner.get_external_distance(start_position, symbol.position)
        if distance < threshold:
            edges.append((start_symbol, symbol, distance))


def generate_dense_symbol_connectivity(G, symbols, include_regions=False):
    symbol_lookup = {s.symbol: s for s in symbols}

    places_layer = get_places_layer(G)
    edges = []

    edges += explicit_edges_from_layer(symbol_lookup, G, places_layer)
    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)

    edges += implicit_edges_from_layers(
        symbol_lookup,
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        True,
        3,
        layer_planner,
    )

    edges += implicit_edges_from_layers(
        symbol_lookup,
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        places_layer,
        False,
        10,
        layer_planner,
    )

    if include_regions:
        # Region-level edges are not currently consumed by the PDDL domains.
        pass

    add_start_symbol_edges(edges, symbols, layer_planner)
    return edges


def generate_dense_region_symbol_connectivity(G, symbols):
    return generate_dense_symbol_connectivity(G, symbols, include_regions=True)


def generate_dense_init(G, symbols_of_interest, start_symbol):
    connectivity = generate_dense_symbol_connectivity(G, symbols_of_interest)
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl = [("=", ("total-cost",), 0), ("at-poi", start_symbol.symbol)]
    initial_pddl += connectivity_pddl

    containment_relations = generate_object_containment(G)
    initial_pddl += containment_relations
    return initial_pddl


def generate_dense_region_init(G, symbols_of_interest, start_symbol):
    connectivity = generate_dense_region_symbol_connectivity(G, symbols_of_interest)

    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl = [("=", ("total-cost",), 0), ("at-poi", start_symbol.symbol)]
    initial_pddl += connectivity_pddl

    containment_relations = generate_object_containment(G)
    containment_relations += generate_place_containment(G)
    initial_pddl += containment_relations
    return initial_pddl


def extract_symbols_of_interest(G, pddl_goal):
    place_facts = extract_facts(pddl_goal, "visited-place")
    place_facts += extract_facts(pddl_goal, "at-place")

    object_facts = extract_facts(pddl_goal, "visited-object")
    object_facts += extract_facts(pddl_goal, "at-object")

    place_symbols = [PddlSymbol(f[1], "place", []) for f in place_facts]
    object_symbols = [PddlSymbol(f[1], "object", []) for f in object_facts]

    return place_symbols + object_symbols


def simplify(pddl_goal):
    return pddl_goal


def generate_inspection_pddl(G, raw_pddl_goal_string, initial_position):
    problem_name = "goto-object-problem"
    problem_domain = "goto-object-domain"

    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)

    # ideally we check the goal here and see if we can run a more specialized planner based on the simplified goal
    goal_pddl = simplify(parsed_pddl_goal)

    goal_symbols = extract_symbols_of_interest(G, goal_pddl)
    normalize_symbols(goal_symbols)
    logger.debug(f"Extracted goal symbols: {goal_symbols}")

    start_place_symbol = PddlSymbol(
        "pstart", "place", ["at-poi"], position=initial_position
    )
    symbols = [start_place_symbol] + goal_symbols

    add_symbol_positions(G, symbols)

    pddl_objects = generate_objects(symbols)
    init = generate_init(G, symbols, start_place_symbol)

    problem = PddlProblem(
        name=problem_name,
        domain=problem_domain,
        objects=pddl_objects,
        initial_facts=init,
        goal=goal_pddl,
        optimizing=True,
    )

    return problem.to_string(), symbols


def generate_rearrangement_pddl(G, raw_pddl_goal_string, initial_position):
    problem_name = "object-rearrangement-domain"
    problem_domain = "object-rearrangement-domain"

    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)

    # ideally we check the goal here and see if we can run a more specialized planner based on the simplified goal
    goal_pddl = simplify(parsed_pddl_goal)

    all_symbols = extract_all_symbols(G)
    filtered_symbols = [s for s in all_symbols if s.layer in ["place", "object"]]
    normalize_symbols(filtered_symbols)

    start_place_symbol = PddlSymbol(
        "pstart", "place", ["at-poi"], position=initial_position
    )
    symbols = [start_place_symbol] + filtered_symbols

    add_symbol_positions(G, symbols)

    pddl_objects = generate_objects(symbols)
    init = generate_dense_init(G, symbols, start_place_symbol)

    problem = PddlProblem(
        name=problem_name,
        domain=problem_domain,
        objects=pddl_objects,
        initial_facts=init,
        goal=goal_pddl,
        optimizing=True,
    )

    return problem.to_string(), symbols


def generate_region_pddl(G, raw_pddl_goal_string, initial_position):
    problem_name = "region-object-rearrangement-domain"
    problem_domain = "region-object-rearrangement-domain"

    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)

    # ideally we check the goal here and see if we can run a more specialized planner based on the simplified goal
    goal_pddl = simplify(parsed_pddl_goal)

    all_symbols = extract_all_symbols(G)
    normalize_symbols(all_symbols)

    start_place_symbol = PddlSymbol(
        "pstart", "place", ["at-poi"], position=initial_position
    )
    symbols = [start_place_symbol] + all_symbols

    add_symbol_positions(G, symbols)

    pddl_objects = generate_objects(symbols)
    init = generate_dense_region_init(G, symbols, start_place_symbol)

    problem = PddlProblem(
        name=problem_name,
        domain=problem_domain,
        objects=pddl_objects,
        initial_facts=init,
        goal=goal_pddl,
        optimizing=True,
    )

    return problem.to_string(), symbols
