import logging
from typing import Any

import copy
import spark_dsg
import numpy as np
import networkx as nx
from plum import dispatch
from itertools import combinations

from dsg_pddl.pddl_grounding import (
    GroundedPddlProblem,
    PddlDomain,
    PddlGoal,
    PddlProblem,
    PddlSymbol,
)
from dsg_pddl.pddl_utils import extract_facts, extract_negated_facts, lisp_string_to_ast, pddl_char_to_dsg_char
from omniplanner.omniplanner import RobotWrapper
from omniplanner.tsp import LayerPlanner

logger = logging.getLogger(__name__)


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


def symbol_connectivity_to_pddl(connectivity):
    connections_init = []

    for info_s, info_t, dist in connectivity:
        s = info_s.symbol
        t = info_t.symbol
        d = int(dist)

        connected = ("connected", s, t)
        distance = ("=", ("distance", s, t), d)
        distance_rev = ("=", ("distance", t, s), d)

        connections_init.append(connected)
        connections_init.append(distance)
        connections_init.append(distance_rev)

    return connections_init


def generate_init(G, symbols_of_interest, start_symbol):
    connectivity = generate_symbol_connectivity(G, symbols_of_interest)
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl = [("=", ("total-cost",), 0), ("at-poi", start_symbol.symbol)]
    initial_pddl += connectivity_pddl
    return initial_pddl


def explicit_edges_from_layer(
    symbol_lookup: dict, G: spark_dsg.DynamicSceneGraph, layer: spark_dsg.LayerView
):
    """Get the edges corresponding to layer's edges"""
    edges = []
    for node in layer.nodes:
        p1 = node.attributes.position
        normalized_symbol = symbol_lookup[normalize_symbol(node.id.str(True))]
        for neighbor in node.siblings():
            if node.id.value < neighbor:
                continue
            n = G.get_node(neighbor)
            p2 = n.attributes.position
            normalized_symbol2 = symbol_lookup[normalize_symbol(n.id.str(True))]
            edges.append(
                (normalized_symbol, normalized_symbol2, np.linalg.norm(p1 - p2))
            )
    return edges


def implicit_edges_from_layers(
    symbol_lookup: dict,
    layer1: spark_dsg.LayerView,
    layer2: spark_dsg.LayerView,
    same_layer,
    connection_threshold,
    layer_planner=None,
):
    edges = []
    for n1 in layer1.nodes:
        p1 = n1.attributes.position
        normalized_symbol = symbol_lookup[normalize_symbol(n1.id.str(True))]
        for n2 in layer2.nodes:
            if same_layer and n1.id.value <= n2.id.value:
                continue
            p2 = n2.attributes.position
            d = np.linalg.norm(p1 - p2)
            if d > connection_threshold:
                continue

            if layer_planner is not None:
                d = layer_planner.get_external_distance(p1[:2], p2[:2])
                if d > connection_threshold:
                    continue

            normalized_symbol2 = symbol_lookup[normalize_symbol(n2.id.str(True))]
            edges.append((normalized_symbol, normalized_symbol2, d))
    return edges


def generate_dense_symbol_connectivity(G, symbols):
    symbol_lookup = {s.symbol: s for s in symbols}

    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    edges = []

    # Place <-> Place Edges
    edges += explicit_edges_from_layer(symbol_lookup, G, places_layer)

    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)

    # Object <-> Object Edges
    edges += implicit_edges_from_layers(
        symbol_lookup,
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        True,
        3,
        layer_planner,
    )

    # Object <-> Place Edges
    edges += implicit_edges_from_layers(
        symbol_lookup,
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        places_layer,
        False,
        10,
        layer_planner,
    )

    start_symbol = symbol_lookup["pstart"]
    start_position = start_symbol.position

    # Connection between starting place and other symbols
    start_connection_threshold = 3
    for s in symbols:
        if s.symbol == "pstart":
            continue

        d = layer_planner.get_external_distance(start_position, s.position)
        if d < start_connection_threshold:
            edges.append((start_symbol, s, d))

    return edges


def generate_dense_region_symbol_connectivity(G, symbols):
    symbol_lookup = {s.symbol: s for s in symbols}

    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    edges = []

    # Place <-> Place Edges
    edges += explicit_edges_from_layer(symbol_lookup, G, places_layer)

    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)

    # Object <-> Object Edges
    edges += implicit_edges_from_layers(
        symbol_lookup,
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        True,
        3,
        layer_planner,
    )

    # Object <-> Place Edges
    edges += implicit_edges_from_layers(
        symbol_lookup,
        G.get_layer(spark_dsg.DsgLayers.OBJECTS),
        places_layer,
        False,
        10,
        layer_planner,
    )

    # Region <-> Region Edges #TODO: currently, we don't actually utilize edges between regions?
    # region_layer =  G.get_layer(spark_dsg.DsgLayers.ROOMS)
    # edges += implicit_edges_from_layers(symbol_lookup, region_layer, region_layer, True, 20)

    start_symbol = symbol_lookup["pstart"]
    start_position = start_symbol.position

    # Connection between starting place and other symbols
    start_connection_threshold = 3
    for s in symbols:
        if s.symbol == "pstart":
            continue

        d = layer_planner.get_external_distance(start_position, s.position)
        if d < start_connection_threshold:
            edges.append((start_symbol, s, d))

    return edges


def generate_object_containment(G):
    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    containments = []

    centers = []
    symbols = []
    for node in places_layer.nodes:
        centers.append(node.attributes.position)
        symbols.append(normalize_symbol(node.id.str(True)))
    centers = np.array(centers)

    for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        closest_idx = np.argmin(
            np.linalg.norm(centers - node.attributes.position, axis=1)
        )
        closest_place = symbols[closest_idx]
        containments.append(
            ("object-in-place", normalize_symbol(node.id.str(True)), closest_place)
        )

    return containments


def generate_place_containment(G):
    try:
        places_layer_2d = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer_2d = G.get_layer(20)

    containments = []

    for node in places_layer_2d.nodes:
        parents = node.parents()
        for parent in parents:
            if parent is not None:
                containments.append(
                    (
                        "place-in-region",
                        normalize_symbol(node.id.str(True)),
                        normalize_symbol(spark_dsg.NodeSymbol(parent).str(True)),
                    )
                )

    return containments


def generate_suspicious_objects(G):
    suspicious_objects = []

    for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        # if not node.attributes.suspicious:
        #     continue

        if node.attributes.semantic_label in tuple(range(0, 40 + 1)):
            continue

        suspicious_objects.append(
            ("suspicious", normalize_symbol(node.id.str(True)))
        )

    return suspicious_objects


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


def simplify(pddl):
    return pddl


def add_symbol_positions(G, symbols):
    for s in symbols:
        if s.position is not None:
            continue
        else:
            pddl_symbol_char = s.symbol[0]
            dsg_symbol_char = pddl_char_to_dsg_char(pddl_symbol_char)
            ns = spark_dsg.NodeSymbol(dsg_symbol_char, int(s.symbol[1:]))
            position = G.get_node(ns).attributes.position[:2]
            if position is None:
                raise Exception(f"Could not find node {ns} in DSG")
            s.position = position
    return symbols


def normalize_symbols(symbols):
    for s in symbols:
        s.symbol = normalize_symbol(s)

def normalize_symbol(symbol):
    if isinstance(symbol, str):
        return symbol.lower()
    else:
        return symbol.symbol.lower()


def generate_objects(symbols):
    type_dict = {"place": [], "dsg_object": [], "region": []}
    for s in symbols:
        if s.layer == "place":
            type_dict["place"].append(s.symbol)
        elif s.layer == "object":
            type_dict["dsg_object"].append(s.symbol)
        elif s.layer == "region":
            type_dict["region"].append(s.symbol)

    return type_dict


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


def extract_all_symbols(G):
    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    place_symbols = []
    for node in places_layer.nodes:
        place_symbols.append(PddlSymbol(node.id.str(True), "place", []))

    region_symbols = []
    for node in G.get_layer(spark_dsg.DsgLayers.ROOMS).nodes:
        region_symbols.append(PddlSymbol(node.id.str(True), "region", []))

    object_symbols = []
    for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        object_symbols.append(PddlSymbol(node.id.str(True), "object", []))

    return place_symbols + object_symbols + region_symbols


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


# --------------------------------------------------------------------------- #
# ---------------------------------- DEVEL ---------------------------------- #
# --------------------------------------------------------------------------- #
def generate_dense_places_init(G, symbols_of_interest, start_symbol):
    connectivity = generate_dense_places_symbol_connectivity(G, symbols_of_interest)
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl = [("=", ("total-cost",), 0), ("at-poi", start_symbol.symbol)]
    initial_pddl += connectivity_pddl

    containment_relations = generate_object_containment(G)
    containment_relations += generate_place_containment(G)
    initial_pddl += containment_relations

    return initial_pddl

def generate_dense_places_symbol_connectivity(G, symbols):
    symbol_lookup = {s.symbol: s for s in symbols}

    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    edges = []

    # Place <-> Place Edges
    edges += explicit_edges_from_layer(symbol_lookup, G, places_layer)
    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)

    # Connection between starting place and other symbols
    start_symbol = symbol_lookup["pstart"]
    start_position = start_symbol.position
    start_connection_threshold = 3
    for s in symbols:
        if s.symbol == "pstart":
            continue

        d = layer_planner.get_external_distance(start_position, s.position)
        if d < start_connection_threshold:
            edges.append((start_symbol, s, d))

    return edges

def generate_test_pddl(G, raw_pddl_goal_string, initial_position):
    problem_name = "test-domain"
    problem_domain = "test-domain"

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
    init = generate_dense_places_init(G, symbols, start_place_symbol)

    problem = PddlProblem(
        name=problem_name,
        domain=problem_domain,
        objects=pddl_objects,
        initial_facts=init,
        goal=goal_pddl,
        optimizing=True,
    )

    return problem.to_string(), symbols

def generate_improved_places_init(G, symbols_of_interest, start_symbol, forbidden_symbols):
    object_containment_relations = generate_object_containment(G)
    place_containment_relations = generate_place_containment(G)
    suspicious_objects = generate_suspicious_objects(G)

    symbols = extract_all_symbols(G)
    normalize_symbols(symbols)
    symbol_lookup = {s.symbol: s for s in symbols}

    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    # Add to the symbols_of_interest the places were each object (currently present in the symbols_of_interest) is located
    for object_containment_relation in object_containment_relations:
        object_symbol = symbol_lookup[normalize_symbol(object_containment_relation[1])]
        place_symbol = symbol_lookup[normalize_symbol(object_containment_relation[2])]
        if object_symbol in symbols_of_interest and place_symbol not in symbols_of_interest:
            symbols_of_interest.append(place_symbol)

        if object_symbol in forbidden_symbols["objects"] and place_symbol not in forbidden_symbols["places"]:
            forbidden_symbols["places"].append(place_symbol)

    # Add to the symbols_of_interest all places belonging to the regions in the symbols_of_interest
    for place_containment_relation in place_containment_relations:
        place_symbol = symbol_lookup[normalize_symbol(place_containment_relation[1])]
        region_symbol = symbol_lookup[normalize_symbol(place_containment_relation[2])]
        if region_symbol in symbols_of_interest and place_symbol not in symbols_of_interest:
            symbols_of_interest.append(place_symbol)

        if region_symbol in forbidden_symbols["regions"] and place_symbol not in forbidden_symbols["places"]:
            forbidden_symbols["places"].append(place_symbol)

    # Add to the symbols_of_interest the place closest to the starting position
    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)
    start_node = layer_planner.get_closest_node_id(start_symbol.position)
    place_symbol = layer_planner.node_value_to_symbol[start_node]
    pddl_symbol = symbol_lookup[normalize_symbol(place_symbol)]
    if pddl_symbol not in symbols_of_interest:
        symbols_of_interest.append(pddl_symbol)

    # Add all places that are contained in the shortest path between each 2 places (currently present in the symbols_of_interest)
    improved_symbols_of_interest = copy.deepcopy(symbols_of_interest)
    forbidden_nodes = {
        layer_planner.symbol_to_node_value[s.symbol]
        for s in forbidden_symbols["places"]
    }
    for s1, s2 in combinations(symbols_of_interest, 2):
        if s1.symbol == "pstart" or s2.symbol == "pstart":
            continue

        if s1.layer != "place" or s2.layer != "place":
            continue

        s = layer_planner.symbol_to_node_value[s1.symbol]
        t = layer_planner.symbol_to_node_value[s2.symbol]

        try:
            path = layer_planner.get_shortest_path(s, t, forbidden_nodes)
        except nx.NetworkXNoPath:
            continue  # just ignore pairs with no path

        for node in path:
            place_symbol = layer_planner.node_value_to_symbol[node]
            pddl_symbol = symbol_lookup[normalize_symbol(place_symbol)]
            if pddl_symbol not in improved_symbols_of_interest:
                improved_symbols_of_interest.append(pddl_symbol)

    # Add to the symbols_of_interest all suspicious objects that are in the places (currently present in the symbols_of_interest)
    containment_map = {}
    for object_containment_relation in object_containment_relations:
        object_symbol = normalize_symbol(object_containment_relation[1])
        place_symbol = normalize_symbol(object_containment_relation[2])
        containment_map[object_symbol] = place_symbol

    relevant_suspicious_objects = []
    relevant_unsafe_places = []
    for suspicious_object in suspicious_objects:
        sus_object_symbol = normalize_symbol(suspicious_object[1])
        sus_object = symbol_lookup[sus_object_symbol]
        place_symbol = containment_map[sus_object_symbol]
        place = symbol_lookup[place_symbol]

        if sus_object in improved_symbols_of_interest:
            relevant_suspicious_objects.append(suspicious_object)
            relevant_unsafe_places.append(("unsafe-place", place_symbol))
            continue

        if place in improved_symbols_of_interest:
            improved_symbols_of_interest.append(sus_object)
            relevant_suspicious_objects.append(suspicious_object)
            relevant_unsafe_places.append(("unsafe-place", place_symbol))

    # Add to the symbols_of_interest all objects that are in the places (currently present in the symbols_of_interest)
    # for object_containment_relation in object_containment_relations:
    #     object_symbol = symbol_lookup[normalize_symbol(object_containment_relation[1])]
    #     place_symbol = symbol_lookup[normalize_symbol(object_containment_relation[2])]
    #     if place_symbol in improved_symbols_of_interest and object_symbol not in improved_symbols_of_interest:
    #         improved_symbols_of_interest.append(object_symbol)

    # Add to the symbols_of_interest the regions were each place (currently present in the symbols_of_interest) is located
    # for place_containment_relation in place_containment_relations:
    #     place_symbol = symbol_lookup[normalize_symbol(place_containment_relation[1])]
    #     region_symbol = symbol_lookup[normalize_symbol(place_containment_relation[2])]
    #     if place_symbol in improved_symbols_of_interest and region_symbol not in improved_symbols_of_interest:
    #         improved_symbols_of_interest.append(region_symbol)

    add_symbol_positions(G, improved_symbols_of_interest)

    edges = []

    # Place <-> Place Edges
    for node in places_layer.nodes:
        if symbol_lookup[normalize_symbol(node.id.str(True))] not in improved_symbols_of_interest:
            continue

        p1 = node.attributes.position
        normalized_symbol = symbol_lookup[normalize_symbol(node.id.str(True))]
        for neighbor in node.siblings():
            if node.id.value < neighbor:
                continue

            n = G.get_node(neighbor)
            if symbol_lookup[normalize_symbol(n.id.str(True))] not in improved_symbols_of_interest:
                continue

            p2 = n.attributes.position
            normalized_symbol2 = symbol_lookup[normalize_symbol(n.id.str(True))]
            edges.append(
                (normalized_symbol, normalized_symbol2, np.linalg.norm(p1 - p2))
            )

    # Connection between starting place and other symbols
    start_position = start_symbol.position
    start_connection_threshold = 3
    for s in improved_symbols_of_interest:
        if s.symbol == "pstart":
            continue

        d = layer_planner.get_external_distance(start_position, s.position)
        if d < start_connection_threshold:
            edges.append((start_symbol, s, d))

    connectivity_pddl = symbol_connectivity_to_pddl(edges)

    initial_pddl = [("=", ("total-cost",), 0), ("at-poi", start_symbol.symbol)]
    initial_pddl += connectivity_pddl

    # Consider only containment_relations between symbols_of_interest
    containment_relations = []
    for object_containment_relation in object_containment_relations:
        object_symbol = symbol_lookup[normalize_symbol(object_containment_relation[1])]
        place_symbol = symbol_lookup[normalize_symbol(object_containment_relation[2])]
        if object_symbol in improved_symbols_of_interest and place_symbol in improved_symbols_of_interest:
            containment_relations.append(object_containment_relation)

    for place_containment_relation in place_containment_relations:
        place_symbol = symbol_lookup[normalize_symbol(place_containment_relation[1])]
        region_symbol = symbol_lookup[normalize_symbol(place_containment_relation[2])]
        if place_symbol in improved_symbols_of_interest and region_symbol in improved_symbols_of_interest:
            containment_relations.append(place_containment_relation)

    initial_pddl += containment_relations
    initial_pddl += relevant_suspicious_objects
    initial_pddl += relevant_unsafe_places

    return initial_pddl, improved_symbols_of_interest

def extract_goal_symbols(pddl_goal):
    place_facts = extract_facts(pddl_goal, "at-poi")
    place_facts += extract_facts(pddl_goal, "at-place")
    place_facts += extract_facts(pddl_goal, "visited-place")
    place_facts_extra = extract_facts(pddl_goal, "object-in-place")

    object_facts = extract_facts(pddl_goal, "at-object")
    object_facts += extract_facts(pddl_goal, "visited-object")
    object_facts += extract_facts(pddl_goal, "suspicious")
    object_facts += extract_facts(pddl_goal, "holding")
    object_facts += extract_facts(pddl_goal, "safe")
    object_facts += extract_facts(pddl_goal, "object-in-place")

    region_facts = extract_facts(pddl_goal, "in-region")
    region_facts += extract_facts(pddl_goal, "visited-region")

    place_symbols = [PddlSymbol(f[1], "place", []) for f in place_facts]
    place_symbols += [PddlSymbol(f[2], "place", []) for f in place_facts_extra]
    object_symbols = [PddlSymbol(f[1], "object", []) for f in object_facts]
    region_symbols = [PddlSymbol(f[1], "region", []) for f in region_facts]

    goal_symbols = place_symbols + object_symbols + region_symbols

    forbidden_place_facts = extract_negated_facts(pddl_goal, "visited-place")
    forbidden_object_facts = extract_negated_facts(pddl_goal, "visited-object")
    forbidden_region_facts = extract_negated_facts(pddl_goal, "visited-region")

    forbidden_place_symbols = [PddlSymbol(f[1], "place", []) for f in forbidden_place_facts]
    forbidden_object_symbols = [PddlSymbol(f[1], "object", []) for f in forbidden_object_facts]
    forbidden_region_symbols = [PddlSymbol(f[1], "region", []) for f in forbidden_region_facts]

    return (
        goal_symbols,
        {
            "places": forbidden_place_symbols,
            "objects": forbidden_object_symbols,
            "regions": forbidden_region_symbols,
        }
    )

def generate_test_pddl_2(G, raw_pddl_goal_string, initial_position):
    problem_name = "test-domain"
    problem_domain = "test-domain"

    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)

    # ideally we check the goal here and see if we can run a more specialized planner based on the simplified goal
    goal_pddl = simplify(parsed_pddl_goal)

    goal_symbols, forbidden_symbols = extract_goal_symbols(goal_pddl)
    normalize_symbols(goal_symbols)
    normalize_symbols(forbidden_symbols["places"])
    normalize_symbols(forbidden_symbols["objects"])
    normalize_symbols(forbidden_symbols["regions"])

    start_place_symbol = PddlSymbol(
        "pstart", "place", ["at-poi"], position=initial_position
    )
    symbols_of_interest = [start_place_symbol] + goal_symbols

    init, symbols = generate_improved_places_init(G, symbols_of_interest, start_place_symbol, forbidden_symbols)
    pddl_objects = generate_objects(symbols)

    problem = PddlProblem(
        name=problem_name,
        domain=problem_domain,
        objects=pddl_objects,
        initial_facts=init,
        goal=goal_pddl,
        optimizing=True,
    )

    return problem.to_string(), symbols
# --------------------------------------------------------------------------- #
# ---------------------------------- DEVEL ---------------------------------- #
# --------------------------------------------------------------------------- #


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
        case "test-domain":
            pddl_problem, symbols = generate_test_pddl_2(dsg, goal.pddl_goal, start)
        case _:
            raise NotImplementedError(
                f"I don't know how to ground a domain of type {domain.domain_name}!"
            )

    symbol_dict = {s.symbol: s for s in symbols}
    return RobotWrapper(
        goal.robot_id, GroundedPddlProblem(domain, pddl_problem, symbol_dict)
    )
