import copy
import logging
import os
from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import spark_dsg

from dsg_pddl.core.models import (
    GroundedPddlProblem,
    PddlDomain,
    PddlGoal,
    PddlProblem,
    PddlSymbol,
)
from dsg_pddl.grounding.connectivity import (
    symbol_connectivity_to_pddl,
)
from dsg_pddl.grounding.containment import (
    generate_object_containment,
    generate_place_containment,
)
from dsg_pddl.grounding.dsg_access import get_places_layer
from dsg_pddl.grounding.legacy import simplify
from dsg_pddl.grounding.symbols import (
    add_symbol_positions,
    extract_all_symbols,
    generate_objects,
    normalize_symbols,
    normalize_symbol,
)
from dsg_pddl.core.parsing import (
    extract_facts,
    extract_negated_facts,
    extract_positive_facts,
    lisp_string_to_ast,
)
from omniplanner.core.wrappers import RobotWrapper
from omniplanner.domains.tsp import LayerPlanner

logger = logging.getLogger(__name__)

REGION_REARRANGEMENT_DERIVED_DOMAIN = (
    "region-object-rearrangement-derived-predicates-domain"
)
REGION_REARRANGEMENT_EXPLICIT_STATE_DOMAIN = (
    "region-object-rearrangement-explicit-state-domain"
)
REGION_REARRANGEMENT_PROBLEM_NAME = "region-object-rearrangement-improved-problem"



def get_region_sampler():
    sampler = os.getenv("PDDL_SAMPLER", "compressed").lower()

    if sampler == "compressed":
        return generate_region_rearrangement_pddl_compressed_graph
    if sampler == "paths":
        return generate_region_rearrangement_pddl_relevant_paths
    if sampler == "all":
        return generate_region_rearrangement_pddl_all_symbols

    raise ValueError(f"Invalid PDDL_SAMPLER={sampler!r}")


def ground_improved_problem(
    domain: PddlDomain,
    dsg: spark_dsg.DynamicSceneGraph,
    robot_states: dict,
    goal: PddlGoal,
    feedback=None,
) -> RobotWrapper[GroundedPddlProblem]:
    logger.info(f"Grounding PDDL Problem {domain.domain_name}")

    start = robot_states.get(goal.robot_id)
    if start is None:
        raise ValueError(f"No pose for robot {goal.robot_id}")
    supported_domains = {
        REGION_REARRANGEMENT_DERIVED_DOMAIN,
        REGION_REARRANGEMENT_EXPLICIT_STATE_DOMAIN
    }
    if domain.domain_name not in supported_domains:
        raise NotImplementedError(
            f"I don't know how to ground a domain of type {domain.domain_name}!"
        )

    generate_symbols = get_region_sampler()

    routes = {}
    pddl_problem, symbols = generate_symbols(
        dsg,
        goal.pddl_goal,
        start,
        domain_name=domain.domain_name,
        state=goal.mission_state,
        routes=routes,
    )

    symbol_dict = {s.symbol: s for s in symbols}
    return RobotWrapper[GroundedPddlProblem](
        goal.robot_id, GroundedPddlProblem(
            domain, pddl_problem, symbol_dict, routes, require_solution=goal.mission_state is not None,
        )
    )


def build_region_rearrangement_problem(
    domain_name,
    symbols,
    init,
    goal_pddl,
    problem_name=REGION_REARRANGEMENT_PROBLEM_NAME,
):
    pddl_objects = generate_objects(symbols)
    problem = PddlProblem(
        name=problem_name,
        domain=domain_name,
        objects=pddl_objects,
        initial_facts=init,
        goal=goal_pddl,
        optimizing=True,
    )
    return problem.to_string(), symbols


def make_start_place_symbol(initial_position):
    return PddlSymbol(
        "pstart",
        "place",
        ["at-place"],
        position=np.asarray(initial_position[:2]),
    )


# -------------------------------------------------------------------------
# PDDL problem with all DSG symbols
# -------------------------------------------------------------------------
def generate_region_rearrangement_pddl_all_symbols(
    G,
    raw_pddl_goal_string,
    initial_position,
    domain_name=REGION_REARRANGEMENT_EXPLICIT_STATE_DOMAIN,
    *, state=None, routes=None,
):
    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)

    goal_pddl = simplify(parsed_pddl_goal)

    all_symbols = extract_all_symbols(G)
    normalize_symbols(all_symbols)

    start_place_symbol = make_start_place_symbol(initial_position)
    symbols = [start_place_symbol] + all_symbols
    add_mission_objects(symbols, start_place_symbol, state)

    add_symbol_positions(G, symbols)

    init = generate_dense_places_init(G, symbols, start_place_symbol, domain_name, state, routes)
    return build_region_rearrangement_problem(domain_name, symbols, init, goal_pddl)


def generate_suspicious_objects(G):
    suspicious_objects = []

    for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes:
        # if not node.attributes.suspicious:
        #     continue

        # 10/50 -> 20 %
        if node.attributes.semantic_label in tuple(range(0, 40 + 1)):
            continue

        suspicious_objects.append(
            ("suspicious", normalize_symbol(node.id.str(True)))
        )

    return suspicious_objects


def effective_scene(G, state):
    containment = generate_object_containment(G)
    suspicious = generate_suspicious_objects(G)
    if state is None:
        return containment, suspicious

    known = {obj for _, obj, _ in containment}
    containment = [
        (predicate, obj, state.placements.get(obj, place))
        for predicate, obj, place in containment if obj != state.holding
    ]
    containment += [
        ("object-in-place", obj, place)
        for obj, place in state.placements.items()
        if obj not in known and obj != state.holding
    ]
    grounded_suspicion = [fact for fact in suspicious if fact[1] not in state.inspected
                          and fact[1] != state.holding]
    return containment, grounded_suspicion


def add_mission_objects(symbols, start, state):
    if state is None:
        return
    known = {normalize_symbol(symbol) for symbol in symbols}
    for obj in sorted(set(state.placements) | ({state.holding} if state.holding else set())):
        if obj not in known:
            symbols.append(PddlSymbol(obj, "object", [], position=start.position))


def apply_mission_state(init, symbols, start, layer_planner, domain_name, state):
    if state is None:
        return
    known = {symbol.symbol for symbol in symbols}
    visits = [("visited-place", state.visited_places)]
    if domain_name == REGION_REARRANGEMENT_EXPLICIT_STATE_DOMAIN:
        visits += [("visited-object", state.visited_objects), ("visited-region", state.visited_regions)]
    for predicate, visited in visits:
        init.extend((predicate, name) for name in sorted(visited & known))
    by_name = {symbol.symbol: symbol for symbol in symbols}
    for obj in sorted(known & state.placements.keys()):
        target = by_name.get(state.placements[obj])
        if target is not None:
            init.append(("placed", obj))
            by_name[obj].position_3d = target.position_3d
    if state.holding in known:
        init.extend([("holding", state.holding), ("hand-full",)])

    anchor = layer_planner.get_closest_node_id(start.position)
    place = layer_planner.node_value_to_symbol[anchor]
    if place in known and np.allclose(start.position, layer_planner.node_value_to_position[anchor]):
        init.remove(("at-place", start.symbol))
        init.append(("at-place", place))
        if domain_name == REGION_REARRANGEMENT_EXPLICIT_STATE_DOMAIN:
            init.extend(("at-object", fact[1]) for fact in init[:]
                        if fact[0] == "object-in-place" and fact[2] == place)
            init.extend(("in-region", fact[2]) for fact in init[:]
                        if fact[0] == "place-in-region" and fact[1] == place)


def region_layer_planner(G, state):
    planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)
    if state is not None:
        blocked = [
            tuple(planner.symbol_to_node_value[name] for name in edge)
            for edge in state.blocked_edges
            if len(edge) == 2 and edge <= planner.symbol_to_node_value.keys()
        ]
        planner.nx_layer.remove_edges_from(blocked)
    return planner


def dense_edges(symbols, layer_planner):
    lookup = {symbol.symbol: symbol for symbol in symbols if symbol.layer == "place"}
    edges = []
    for left, right in layer_planner.nx_layer.edges:
        source = lookup.get(layer_planner.node_value_to_symbol[left])
        target = lookup.get(layer_planner.node_value_to_symbol[right])
        if source is not None and target is not None:
            edges.append((source, target, int(np.ceil(np.linalg.norm(source.position - target.position)))))
    return edges


def start_connection(edges, start_symbol, symbols, layer_planner, routes, excluded=()):
    node = layer_planner.get_closest_node_id(start_symbol.position)
    graph = layer_planner.nx_layer
    excluded = set(excluded) - {node}
    allowed = nx.subgraph_view(graph, filter_node=lambda vertex: vertex not in excluded)
    paths = nx.single_source_shortest_path(allowed, node, cutoff=2)
    offset = np.linalg.norm(start_symbol.position - layer_planner.node_value_to_position[node])
    for symbol in symbols:
        if symbol.layer != "place" or symbol.symbol == start_symbol.symbol:
            continue
        target = layer_planner.symbol_to_node_value[symbol.symbol]
        path = paths.get(target)
        if path is None or offset + len(path) - 1 >= 3.0:
            continue
        cost = int(np.ceil(offset + len(path) - 1))
        edges.append((start_symbol, symbol, cost))
        if routes is not None:
            routes[(start_symbol.symbol, symbol.symbol)] = [
                start_symbol.position, *(layer_planner.node_value_to_position[vertex] for vertex in path),
            ]
    if routes is not None:
        for source, target, _ in edges:
            routes.setdefault((source.symbol, target.symbol), [source.position, target.position])
            routes.setdefault((target.symbol, source.symbol), list(reversed(routes[source.symbol, target.symbol])))


def generate_dense_places_init(G, symbols, start_symbol, domain_name, state=None, routes=None):
    layer_planner = region_layer_planner(G, state)
    edges = dense_edges(symbols, layer_planner)
    containment, suspicious = effective_scene(G, state)
    containment = filter_containment_relations(
        containment, generate_place_containment(G), symbols,
        {symbol.symbol: symbol for symbol in symbols},
    )
    relevant_suspicious, unsafe = process_suspicious_objects(
        suspicious, containment, symbols, {symbol.symbol: symbol for symbol in symbols},
    )
    start_connection(
        edges, start_symbol, symbols, layer_planner, routes,
        (layer_planner.symbol_to_node_value[place] for _, place in unsafe),
    )
    init = [("=", ("total-cost",), 0), ("at-place", start_symbol.symbol)]
    init += symbol_connectivity_to_pddl(edges) + containment + relevant_suspicious + unsafe
    init += interaction_facts(G, symbols, containment, start_symbol, layer_planner)
    apply_mission_state(init, symbols, start_symbol, layer_planner, domain_name, state)
    return init


# -------------------------------------------------------------------------
# PDDL problem with symbols along relevant shortest paths
# -------------------------------------------------------------------------
def generate_region_rearrangement_pddl_relevant_paths(
    G,
    raw_pddl_goal_string,
    initial_position,
    domain_name=REGION_REARRANGEMENT_EXPLICIT_STATE_DOMAIN,
    *, state=None, routes=None,
):
    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)

    goal_pddl = simplify(parsed_pddl_goal)

    goal_symbols, forbidden_symbols, declarations = extract_routing_goal(goal_pddl)

    start_place_symbol = make_start_place_symbol(initial_position)
    symbols_of_interest = [start_place_symbol] + goal_symbols

    init, symbols = generate_improved_places_init(
        G,
        symbols_of_interest,
        start_place_symbol,
        forbidden_symbols,
        domain_name,
        state,
        routes,
        declarations=declarations,
        needs_stances=any(
            extract_facts(goal_pddl, predicate)
            for predicate in ("object-in-place", "holding", "hand-full", "placed", "suspicious")
        ),
    )
    return build_region_rearrangement_problem(domain_name, symbols, init, goal_pddl)

def extract_routing_goal(pddl_goal):
    symbols, forbidden = extract_goal_symbols(pddl_goal, include_negated_visits=False)
    normalize_symbols(symbols)
    for excluded in forbidden.values():
        normalize_symbols(excluded)
    # The PDDL goal still needs declarations for symbols used only in exclusions.
    declarations = sorted(set().union(*forbidden.values()) - set(symbols))
    return symbols, forbidden, declarations


def extract_goal_symbols(pddl_goal, *, include_negated_visits=True):
    visit_facts = extract_facts if include_negated_visits else extract_positive_facts
    place_facts = extract_facts(pddl_goal, "at-place")
    place_facts += visit_facts(pddl_goal, "visited-place")
    place_facts_extra = extract_facts(pddl_goal, "object-in-place")

    object_facts = extract_facts(pddl_goal, "at-object")
    object_facts += visit_facts(pddl_goal, "visited-object")
    object_facts += extract_facts(pddl_goal, "suspicious")
    object_facts += extract_facts(pddl_goal, "holding")
    object_facts += extract_facts(pddl_goal, "safe")
    object_facts += extract_facts(pddl_goal, "object-in-place")

    region_facts = extract_facts(pddl_goal, "in-region")
    region_facts += visit_facts(pddl_goal, "visited-region")

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
        sorted(goal_symbols),
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


def retain_adjacent_stances(symbols, containment, layer_planner, symbol_lookup):
    """Retain approaches to the goal places and relevant object anchors."""
    known = {s.symbol for s in symbols}
    anchors = {s.symbol for s in symbols if s.layer == "place"}
    anchors.update(place for _, obj, place in containment if obj in known)

    retained = set(symbols)
    for anchor in sorted(anchors):
        node = layer_planner.symbol_to_node_value.get(anchor)
        if node is not None:
            for neighbor in layer_planner.nx_layer.neighbors(node):
                symbol = symbol_lookup[layer_planner.node_value_to_symbol[neighbor]]
                if symbol not in retained:
                    symbols.append(symbol)
                    retained.add(symbol)


def interaction_facts(G, symbols, containment, start, layer_planner):
    """Use original place edges and 3D geometry for local interactions."""
    places = {
        normalize_symbol(node.id.str(True)): node
        for node in get_places_layer(G).nodes
    }
    by_id = layer_planner.node_value_to_symbol
    positions = {name: node.attributes.position for name, node in places.items()}
    known = {symbol.symbol for symbol in symbols}
    objects = {
        normalize_symbol(node.id.str(True)): node.attributes.position
        for node in G.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes
    }
    objects_at = defaultdict(list)
    for predicate, obj, place in containment:
        if predicate == "object-in-place":
            objects_at[place].append(obj)
    start_anchor = by_id[layer_planner.get_closest_node_id(start.position)]
    positions[start.symbol] = np.r_[start.position, positions[start_anchor][2]]
    for symbol in symbols:
        symbol.position_3d = positions.get(symbol.symbol, objects.get(symbol.symbol))
    reach = float(os.getenv("PDDL_MANIPULATION_REACH_M", "0.75"))
    if not np.isfinite(reach) or reach <= 0:
        raise ValueError("PDDL_MANIPULATION_REACH_M must be finite and positive")
    facts = []
    for target in sorted(known & places.keys()):
        stances = {
            by_id[node] for node in layer_planner.nx_layer.neighbors(
                layer_planner.symbol_to_node_value[target]
            )
        }
        if start_anchor in stances and not np.allclose(start.position, positions[target][:2]):
            stances.add(start.symbol)
        for stance in sorted((stances & known) - {target}):
            facts.append(("adjacent", target, stance))
            if np.linalg.norm(positions[target] - positions[stance]) <= reach:
                facts.append(("place-reachable", target, stance))
            for obj in objects_at[target]:
                if obj in objects and np.linalg.norm(objects[obj] - positions[stance]) <= reach:
                    facts.append(("initial-pick-reachable", obj, stance))
    return facts


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
    symbols_of_interest,
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

        if sus_object in symbols_of_interest:
            relevant_suspicious_objects.append(suspicious_object)
            relevant_unsafe_places.append(("unsafe-place", place_symbol))
            continue

        if place in symbols_of_interest:
            symbols_of_interest.append(sus_object)
            relevant_suspicious_objects.append(suspicious_object)
            relevant_unsafe_places.append(("unsafe-place", place_symbol))

    return relevant_suspicious_objects, relevant_unsafe_places


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


def generate_improved_places_init(
    G,
    symbols_of_interest,
    start_symbol,
    forbidden_symbols,
    domain_name,
    state=None,
    routes=None,
    *,
    declarations=(),
    needs_stances=True,
):
    object_containment_relations, suspicious_objects = effective_scene(G, state)
    place_containment_relations = generate_place_containment(G)

    symbols = extract_all_symbols(G)
    add_mission_objects(symbols, start_symbol, state)
    normalize_symbols(symbols)
    if state is not None and state.holding:
        held = next(symbol for symbol in symbols if symbol.symbol == state.holding)
        if held not in symbols_of_interest:
            symbols_of_interest.append(held)
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

    layer_planner = region_layer_planner(G, state)
    add_start_place(
        symbols_of_interest,
        start_symbol,
        layer_planner,
        symbol_lookup,
    )
    forbidden_nodes = build_forbidden_nodes(forbidden_symbols, layer_planner)
    if needs_stances or (state is not None and state.holding):
        retain_adjacent_stances(
            symbols_of_interest, object_containment_relations, layer_planner, symbol_lookup,
        )
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

    edges = [
        (source, target, cost)
        for source, target, cost in dense_edges(improved_symbols_of_interest, layer_planner)
        if forbidden_nodes.isdisjoint(
            (layer_planner.symbol_to_node_value[source.symbol], layer_planner.symbol_to_node_value[target.symbol])
        )
    ]
    start_connection(
        edges, start_symbol, improved_symbols_of_interest, layer_planner, routes,
        forbidden_nodes | {
            layer_planner.symbol_to_node_value[place] for _, place in relevant_unsafe_places
        },
    )
    connectivity_pddl = symbol_connectivity_to_pddl(edges)

    undeclared = sorted(set(declarations) - set(improved_symbols_of_interest))
    add_symbol_positions(G, undeclared)
    improved_symbols_of_interest.extend(undeclared)
    containment_relations = filter_containment_relations(
        object_containment_relations,
        place_containment_relations,
        improved_symbols_of_interest,
        symbol_lookup,
    )

    initial_pddl = [
        ("=", ("total-cost",), 0),
        ("at-place", start_symbol.symbol),
    ]

    initial_pddl += connectivity_pddl
    initial_pddl += containment_relations
    initial_pddl += interaction_facts(
        G, improved_symbols_of_interest, containment_relations, start_symbol, layer_planner,
    )
    initial_pddl += relevant_suspicious_objects
    initial_pddl += relevant_unsafe_places

    apply_mission_state(initial_pddl, improved_symbols_of_interest, start_symbol, layer_planner, domain_name, state)
    return initial_pddl, improved_symbols_of_interest



# -------------------------------------------------------------------------
# PDDL problem with compressed graph over the most relevant symbols
# -------------------------------------------------------------------------
def generate_region_rearrangement_pddl_compressed_graph(
    G,
    raw_pddl_goal_string,
    initial_position,
    domain_name=REGION_REARRANGEMENT_EXPLICIT_STATE_DOMAIN,
    *, state=None, routes=None,
):
    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)

    goal_pddl = simplify(parsed_pddl_goal)

    goal_symbols, forbidden_symbols, declarations = extract_routing_goal(goal_pddl)

    start_place_symbol = make_start_place_symbol(initial_position)
    symbols_of_interest = [start_place_symbol] + goal_symbols

    init, symbols = generate_improved_places_init_v2(
        G,
        symbols_of_interest,
        start_place_symbol,
        forbidden_symbols,
        domain_name,
        state,
        routes,
        declarations=declarations,
        needs_stances=any(
            extract_facts(goal_pddl, predicate)
            for predicate in ("object-in-place", "holding", "hand-full", "placed", "suspicious")
        ),
    )
    return build_region_rearrangement_problem(domain_name, symbols, init, goal_pddl)

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


def build_segments_from_paths(paths, important_nodes):
    segments = {}
    lookup = {}
    for path in paths.values():
        current = [path[0]]
        for node in path[1:]:
            current.append(node)
            if node in important_nodes:
                start = current[0]
                end = current[-1]
                if start != end:
                    key = tuple(sorted((start, end)))
                    if key not in segments:
                        segments[key] = list(current if start == key[0] else reversed(current))
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
        return set(primary_nodes), []

    secondary_nodes = compute_secondary_nodes_from_paths(
        paths,
        primary_nodes,
    )

    inspection_stances = set()
    for path in paths.values():
        for left, right in zip(path, path[1:]):
            if right in protected_nodes:
                inspection_stances.add(left)
            if left in protected_nodes:
                inspection_stances.add(right)

    important_nodes = primary_nodes | secondary_nodes | protected_nodes | inspection_stances

    segments = build_segments_from_paths(
        paths,
        important_nodes,
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

    return important_nodes, compressed_edges


def build_compressed_edges(
    compressed_edges_raw,
    improved_symbols_of_interest,
    layer_planner,
    symbol_lookup,
):
    edges = []
    used = set()
    for start_node, end_node, total_dist, _ in compressed_edges_raw:
        key = frozenset((start_node, end_node))
        if key in used:
            continue
        used.add(key)
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
    domain_name,
    state=None,
    routes=None,
    *,
    declarations=(),
    needs_stances=True,
):
    object_containment_relations, suspicious_objects = effective_scene(G, state)
    place_containment_relations = generate_place_containment(G)

    symbols = extract_all_symbols(G)
    add_mission_objects(symbols, start_symbol, state)
    normalize_symbols(symbols)
    if state is not None and state.holding:
        held = next(symbol for symbol in symbols if symbol.symbol == state.holding)
        if held not in symbols_of_interest:
            symbols_of_interest.append(held)

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

    layer_planner = region_layer_planner(G, state)
    add_start_place(
        symbols_of_interest,
        start_symbol,
        layer_planner,
        symbol_lookup,
    )
    if needs_stances or (state is not None and state.holding):
        retain_adjacent_stances(
            symbols_of_interest, object_containment_relations, layer_planner, symbol_lookup,
        )
    forbidden_nodes = build_forbidden_nodes(forbidden_symbols, layer_planner)
    primary_nodes = build_primary_nodes(symbols_of_interest, layer_planner) - forbidden_nodes
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
    if routes is not None:
        for left, right, _, nodes in compressed_edges_raw:
            key = (layer_planner.node_value_to_symbol[left], layer_planner.node_value_to_symbol[right])
            routes[key] = [layer_planner.node_value_to_position[node] for node in nodes]
    start_connection(
        edges, start_symbol, improved_symbols_of_interest, layer_planner, routes,
        forbidden_nodes | {
            layer_planner.symbol_to_node_value[place] for _, place in relevant_unsafe_places
        },
    )
    connectivity_pddl = symbol_connectivity_to_pddl(edges)

    undeclared = sorted(set(declarations) - set(improved_symbols_of_interest))
    add_symbol_positions(G, undeclared)
    improved_symbols_of_interest.extend(undeclared)
    containment_relations = filter_containment_relations(
        object_containment_relations,
        place_containment_relations,
        improved_symbols_of_interest,
        symbol_lookup,
    )

    initial_pddl = [
        ("=", ("total-cost",), 0),
        ("at-place", start_symbol.symbol),
    ]
    initial_pddl += connectivity_pddl
    initial_pddl += containment_relations
    initial_pddl += interaction_facts(
        G, improved_symbols_of_interest, containment_relations, start_symbol, layer_planner,
    )
    initial_pddl += relevant_suspicious_objects
    initial_pddl += relevant_unsafe_places

    apply_mission_state(initial_pddl, improved_symbols_of_interest, start_symbol, layer_planner, domain_name, state)
    return initial_pddl, improved_symbols_of_interest
