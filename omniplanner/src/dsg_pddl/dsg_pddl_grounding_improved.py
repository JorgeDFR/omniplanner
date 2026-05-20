import logging

import copy
import spark_dsg
import numpy as np
import networkx as nx

from typing import Any
from plum import dispatch
from itertools import combinations
from collections import defaultdict

from dsg_pddl.pddl_grounding import (
    PddlGoal,
    PddlSymbol,
    PddlDomain,
    PddlProblem,
    GroundedPddlProblem
)
from dsg_pddl.dsg_pddl_grounding import (
    symbol_connectivity_to_pddl,
    generate_object_containment,
    generate_place_containment,
    explicit_edges_from_layer,
    simplify,
    extract_all_symbols,
    normalize_symbols,
    normalize_symbol,
    add_symbol_positions,
    generate_objects
)
from dsg_pddl.pddl_utils import (
    extract_facts,
    extract_negated_facts,
    lisp_string_to_ast
)
from omniplanner.omniplanner import RobotWrapper
from omniplanner.tsp import LayerPlanner

logger = logging.getLogger(__name__)


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
    match domain.domain_name:
        case "test-domain":
            pddl_problem, symbols = generate_test_pddl_v3(dsg, goal.pddl_goal, start)
        case _:
            raise NotImplementedError(
                f"I don't know how to ground a domain of type {domain.domain_name}!"
            )

    symbol_dict = {s.symbol: s for s in symbols}
    return RobotWrapper(
        goal.robot_id, GroundedPddlProblem(domain, pddl_problem, symbol_dict)
    )

# -------------------------------------------------------------------------
# PPDL Problem with all symbols
# -------------------------------------------------------------------------
def generate_test_pddl_v1(G, raw_pddl_goal_string, initial_position):
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


def generate_dense_places_init(G, symbols_of_interest, start_symbol):
    connectivity = generate_dense_places_symbol_connectivity(G, symbols_of_interest)
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl = [("=", ("total-cost",), 0), ("at-poi", start_symbol.symbol)]
    initial_pddl += connectivity_pddl

    containment_relations = generate_object_containment(G)
    containment_relations += generate_place_containment(G)
    initial_pddl += containment_relations

    return initial_pddl


def add_start_place_connection(
    edges,
    start_symbol,
    symbols,
    layer_planner,
    threshold=3.0,
):
    start_position = start_symbol.position
    for s in symbols:
        if s.symbol == "pstart":
            continue

        dist = layer_planner.get_external_distance(start_position, s.position)
        if dist < threshold:
            edges.append((start_symbol, s, int(np.ceil(dist))))


def generate_dense_places_symbol_connectivity(G, symbols):
    symbol_lookup = {s.symbol: s for s in symbols}

    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    edges = []

    # Place <-> Place Edges
    edges += explicit_edges_from_layer(symbol_lookup, G, places_layer)

    # Connection between starting place and other symbols
    start_symbol = symbol_lookup["pstart"]
    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)
    add_start_place_connection(edges, start_symbol, symbols, layer_planner)

    return edges



# -------------------------------------------------------------------------
# PPDL Problem with only the place symbols belonging to the cloests paths
# -------------------------------------------------------------------------
def generate_test_pddl_v2(G, raw_pddl_goal_string, initial_position):
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

    place_symbols = {PddlSymbol(f[1], "place", []) for f in place_facts}
    place_symbols |= {PddlSymbol(f[2], "place", []) for f in place_facts_extra}
    object_symbols = {PddlSymbol(f[1], "object", []) for f in object_facts}
    region_symbols = {PddlSymbol(f[1], "region", []) for f in region_facts}

    goal_symbols = place_symbols | object_symbols | region_symbols

    forbidden_place_facts = extract_negated_facts(pddl_goal, "visited-place")
    forbidden_object_facts = extract_negated_facts(pddl_goal, "visited-object")
    forbidden_region_facts = extract_negated_facts(pddl_goal, "visited-region")

    forbidden_place_symbols = {PddlSymbol(f[1], "place", []) for f in forbidden_place_facts}
    forbidden_object_symbols = {PddlSymbol(f[1], "object", []) for f in forbidden_object_facts}
    forbidden_region_symbols = {PddlSymbol(f[1], "region", []) for f in forbidden_region_facts}

    return (
        list(goal_symbols),
        {
            "places": list(forbidden_place_symbols),
            "objects": list(forbidden_object_symbols),
            "regions": list(forbidden_region_symbols),
        }
    )


def add_object_related_places(
    symbols_of_interest,
    forbidden_symbols,
    object_containment_relations,
    symbol_lookup,
):
    for relation in object_containment_relations:
        object_symbol = symbol_lookup[normalize_symbol(relation[1])]
        place_symbol = symbol_lookup[normalize_symbol(relation[2])]

        if (object_symbol in symbols_of_interest
            and place_symbol not in symbols_of_interest):
            symbols_of_interest.append(place_symbol)

        if (object_symbol in forbidden_symbols["objects"]
            and place_symbol not in forbidden_symbols["places"]):
            forbidden_symbols["places"].append(place_symbol)


def add_region_related_places(
    symbols_of_interest,
    forbidden_symbols,
    place_containment_relations,
    symbol_lookup,
):
    for relation in place_containment_relations:
        place_symbol = symbol_lookup[normalize_symbol(relation[1])]
        region_symbol = symbol_lookup[normalize_symbol(relation[2])]

        if (region_symbol in symbols_of_interest
            and place_symbol not in symbols_of_interest):
            symbols_of_interest.append(place_symbol)

        if (region_symbol in forbidden_symbols["regions"]
            and place_symbol not in forbidden_symbols["places"]):
            forbidden_symbols["places"].append(place_symbol)


def add_start_place(
    symbols_of_interest,
    start_symbol,
    layer_planner,
    symbol_lookup,
):
    start_node = layer_planner.get_closest_node_id(start_symbol.position)
    place_symbol = layer_planner.node_value_to_symbol[start_node]
    pddl_symbol = symbol_lookup[normalize_symbol(place_symbol)]

    if pddl_symbol not in symbols_of_interest:
        symbols_of_interest.append(pddl_symbol)

    return start_node


def add_relevant_place_nodes(
    symbols_of_interest,
    forbidden_nodes,
    layer_planner,
    symbol_lookup,
):
    improved_symbols_of_interest = copy.deepcopy(symbols_of_interest)
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

    return improved_symbols_of_interest

def build_forbidden_nodes(forbidden_symbols, layer_planner):
    forbidden_nodes = set()
    for s in forbidden_symbols["places"]:
        if s.symbol not in layer_planner.symbol_to_node_value:
            continue
        forbidden_nodes.add(layer_planner.symbol_to_node_value[s.symbol])

    return forbidden_nodes


def process_suspicious_objects(
    suspicious_objects,
    object_containment_relations,
    improved_symbols_of_interest,
    symbol_lookup,
):
    containment_map = {}
    for relation in object_containment_relations:
        object_symbol = normalize_symbol(relation[1])
        place_symbol = normalize_symbol(relation[2])
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

    return relevant_suspicious_objects, relevant_unsafe_places


def build_place_edges(
    G,
    improved_symbols_of_interest,
    symbol_lookup,
):
    try:
        places_layer = G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = G.get_layer(20)

    edges = []
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
            dist = np.linalg.norm(p1 - p2)
            edges.append(
                (normalized_symbol, normalized_symbol2, int(np.ceil(dist)))
            )

    return edges


def filter_containment_relations(
    object_containment_relations,
    place_containment_relations,
    improved_symbols_of_interest,
    symbol_lookup,
):
    containment_relations = []
    for relation in object_containment_relations:
        object_symbol = symbol_lookup[normalize_symbol(relation[1])]
        place_symbol = symbol_lookup[normalize_symbol(relation[2])]

        if (object_symbol in improved_symbols_of_interest
            and place_symbol in improved_symbols_of_interest):
            containment_relations.append(relation)

    for relation in place_containment_relations:
        place_symbol = symbol_lookup[normalize_symbol(relation[1])]
        region_symbol = symbol_lookup[normalize_symbol(relation[2])]

        if (place_symbol in improved_symbols_of_interest
            and region_symbol in improved_symbols_of_interest):
            containment_relations.append(relation)

    return containment_relations


def generate_improved_places_init(G, symbols_of_interest, start_symbol, forbidden_symbols):
    object_containment_relations = generate_object_containment(G)
    place_containment_relations = generate_place_containment(G)
    suspicious_objects = generate_suspicious_objects(G)

    symbols = extract_all_symbols(G)
    normalize_symbols(symbols)
    symbol_lookup = {s.symbol: s for s in symbols}

    add_object_related_places(
        symbols_of_interest,
        forbidden_symbols,
        object_containment_relations,
        symbol_lookup,
    )
    add_region_related_places(
        symbols_of_interest,
        forbidden_symbols,
        place_containment_relations,
        symbol_lookup,
    )

    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)
    add_start_place(
        symbols_of_interest,
        start_symbol,
        layer_planner,
        symbol_lookup,
    )
    forbidden_nodes = build_forbidden_nodes(forbidden_symbols, layer_planner)
    improved_symbols_of_interest = add_relevant_place_nodes(
        symbols_of_interest,
        forbidden_nodes,
        layer_planner,
        symbol_lookup,
    )

    relevant_suspicious_objects, relevant_unsafe_places = process_suspicious_objects(
        suspicious_objects,
        object_containment_relations,
        improved_symbols_of_interest,
        symbol_lookup,
    )

    add_symbol_positions(G, improved_symbols_of_interest)

    edges = build_place_edges(G, improved_symbols_of_interest, symbol_lookup)
    add_start_place_connection(
        edges,
        start_symbol,
        improved_symbols_of_interest,
        layer_planner,
    )
    connectivity_pddl = symbol_connectivity_to_pddl(edges)

    containment_relations = filter_containment_relations(
        object_containment_relations,
        place_containment_relations,
        improved_symbols_of_interest,
        symbol_lookup,
    )

    initial_pddl = [
        ("=", ("total-cost",), 0),
        ("at-poi", start_symbol.symbol),
    ]

    initial_pddl += connectivity_pddl
    initial_pddl += containment_relations
    initial_pddl += relevant_suspicious_objects
    initial_pddl += relevant_unsafe_places

    return initial_pddl, improved_symbols_of_interest



# -------------------------------------------------------------------------
# PPDL Problem with only most relevant symbols
# -------------------------------------------------------------------------
def generate_test_pddl_v3(G, raw_pddl_goal_string, initial_position):
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

    init, symbols = generate_improved_places_init_v2(G, symbols_of_interest, start_place_symbol, forbidden_symbols)
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

def build_primary_nodes(symbols_of_interest, layer_planner):
    primary_nodes = set()
    for s in symbols_of_interest:
        if s.layer != "place" or s.symbol == "pstart":
            continue

        try:
            node_id = layer_planner.symbol_to_node_value[s.symbol]
            primary_nodes.add(node_id)
        except KeyError:
            continue

    return primary_nodes


def add_important_place_symbols(
    improved_symbols_of_interest,
    important_nodes,
    layer_planner,
    symbol_lookup,
):
    for node_id in important_nodes:
        place_symbol = layer_planner.node_value_to_symbol[node_id]
        pddl_symbol = symbol_lookup[normalize_symbol(place_symbol)]

        if pddl_symbol not in improved_symbols_of_interest:
            improved_symbols_of_interest.append(pddl_symbol)


def build_shortest_paths(G, primary_nodes, forbidden_nodes, layer_planner):
    paths = {}
    primary_nodes = list(primary_nodes)
    for i in range(len(primary_nodes)):
        for j in range(i + 1, len(primary_nodes)):
            s = primary_nodes[i]
            t = primary_nodes[j]

            try:
                path = layer_planner.get_shortest_path(s, t, forbidden_nodes)
            except nx.NetworkXNoPath:
                continue

            if path:
                paths[(s, t)] = path

    return paths

def compute_secondary_nodes_from_paths(paths, primary_nodes):
    primary_nodes = set(primary_nodes)
    node_path_count = defaultdict(int)
    adjacency = defaultdict(set)
    for path in paths.values():
        seen = set()
        for i, node in enumerate(path):
            if node not in seen:
                node_path_count[node] += 1
                seen.add(node)

            if i > 0:
                prev = path[i - 1]
                adjacency[node].add(prev)
                adjacency[prev].add(node)

    secondary = set()
    for node in adjacency:
        if node in primary_nodes:
            continue

        degree = len(adjacency[node])
        path_count = node_path_count[node]
        if path_count >= 2 and degree >= 3:
            secondary.add(node)

    return secondary


def extract_protected_nodes(
    suspicious_objects,
    object_containment_relations,
    layer_planner,
):
    containment_map = {}
    for relation in object_containment_relations:
        object_symbol = normalize_symbol(relation[1])
        place_symbol = normalize_symbol(relation[2])
        containment_map[object_symbol] = place_symbol

    unsafe_place_nodes = set()
    for suspicious_object in suspicious_objects:
        sus_object_symbol = normalize_symbol(suspicious_object[1])
        if sus_object_symbol not in containment_map:
            continue

        place_symbol = containment_map[sus_object_symbol]
        if place_symbol not in layer_planner.symbol_to_node_value:
            continue

        node_id = layer_planner.symbol_to_node_value[place_symbol]
        unsafe_place_nodes.add(node_id)

    return unsafe_place_nodes


def build_segments_from_paths(paths, primary_nodes, secondary_nodes, protected_nodes):
    important = set(primary_nodes) | set(secondary_nodes) | set(protected_nodes)
    segments = {}
    lookup = {}
    for path in paths.values():
        current = [path[0]]
        for node in path[1:]:
            current.append(node)
            if node in important:
                start = current[0]
                end = current[-1]
                if start != end:
                    key = tuple(sorted((start, end)))
                    if key not in segments:
                        segments[key] = list(current)
                current = [node]

    for (a, b), seg in segments.items():
        lookup[(a, b)] = seg
        lookup[(b, a)] = list(reversed(seg))

    return lookup


def build_compressed_place_graph(
    G,
    primary_nodes,
    forbidden_nodes,
    protected_nodes,
    layer_planner,
):
    paths = build_shortest_paths(
        G,
        primary_nodes,
        forbidden_nodes,
        layer_planner,
    )

    if not paths:
        return set(), {}

    secondary_nodes = compute_secondary_nodes_from_paths(
        paths,
        primary_nodes,
    )

    segments = build_segments_from_paths(
        paths,
        primary_nodes,
        protected_nodes,
        secondary_nodes,
    )

    compressed_edges = []
    for (a, b), path in segments.items():
        total_dist = 0.0
        for i in range(len(path) - 1):
            n1 = G.get_node(path[i])
            n2 = G.get_node(path[i + 1])
            p1 = n1.attributes.position
            p2 = n2.attributes.position
            total_dist += np.linalg.norm(p1 - p2)
        compressed_edges.append((a, b, int(np.ceil(total_dist)), path))

    important_nodes = set(primary_nodes) | set(secondary_nodes)

    return important_nodes, compressed_edges


def build_compressed_edges(
    compressed_edges_raw,
    improved_symbols_of_interest,
    layer_planner,
    symbol_lookup,
):
    edges = []
    for start_node, end_node, total_dist, _ in compressed_edges_raw:
        s1 = symbol_lookup[normalize_symbol(layer_planner.node_value_to_symbol[start_node])]
        s2 = symbol_lookup[normalize_symbol(layer_planner.node_value_to_symbol[end_node])]

        if (s1 not in improved_symbols_of_interest
            or s2 not in improved_symbols_of_interest):
            continue

        edges.append((s1, s2, total_dist))

    return edges


def generate_improved_places_init_v2(
    G,
    symbols_of_interest,
    start_symbol,
    forbidden_symbols,
):
    object_containment_relations = generate_object_containment(G)
    place_containment_relations = generate_place_containment(G)
    suspicious_objects = generate_suspicious_objects(G)

    symbols = extract_all_symbols(G)
    normalize_symbols(symbols)

    symbol_lookup = {s.symbol: s for s in symbols}

    add_object_related_places(
        symbols_of_interest,
        forbidden_symbols,
        object_containment_relations,
        symbol_lookup,
    )
    add_region_related_places(
        symbols_of_interest,
        forbidden_symbols,
        place_containment_relations,
        symbol_lookup,
    )

    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)
    add_start_place(
        symbols_of_interest,
        start_symbol,
        layer_planner,
        symbol_lookup,
    )
    primary_nodes = build_primary_nodes(
        symbols_of_interest,
        layer_planner,
    )
    forbidden_nodes = build_forbidden_nodes(
        forbidden_symbols,
        layer_planner,
    )
    protected_nodes = extract_protected_nodes(
        suspicious_objects,
        object_containment_relations,
        layer_planner,
    )
    important_nodes, compressed_edges_raw = build_compressed_place_graph(
        G,
        primary_nodes,
        forbidden_nodes,
        protected_nodes,
        layer_planner,
    )

    improved_symbols_of_interest = copy.deepcopy(symbols_of_interest)
    add_important_place_symbols(
        improved_symbols_of_interest,
        important_nodes,
        layer_planner,
        symbol_lookup,
    )

    relevant_suspicious_objects, relevant_unsafe_places = process_suspicious_objects(
        suspicious_objects,
        object_containment_relations,
        improved_symbols_of_interest,
        symbol_lookup,
    )

    add_symbol_positions(G, improved_symbols_of_interest)

    edges = build_compressed_edges(
        compressed_edges_raw,
        improved_symbols_of_interest,
        layer_planner,
        symbol_lookup,
    )
    add_start_place_connection(
        edges,
        start_symbol,
        improved_symbols_of_interest,
        layer_planner,
    )
    connectivity_pddl = symbol_connectivity_to_pddl(edges)

    containment_relations = filter_containment_relations(
        object_containment_relations,
        place_containment_relations,
        improved_symbols_of_interest,
        symbol_lookup,
    )

    initial_pddl = [
        ("=", ("total-cost",), 0),
        ("at-poi", start_symbol.symbol),
    ]
    initial_pddl += connectivity_pddl
    initial_pddl += containment_relations
    initial_pddl += relevant_suspicious_objects
    initial_pddl += relevant_unsafe_places

    return initial_pddl, improved_symbols_of_interest
