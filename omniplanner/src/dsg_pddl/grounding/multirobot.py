import logging
import os
import time
from typing import Any

import numpy as np
import spark_dsg
from plum import dispatch

from dsg_pddl.core.models import (
    GroundedPddlProblem,
    MultiRobotPddlDomain,
    PddlGoal,
    PddlProblem,
    PddlSymbol,
)
from dsg_pddl.core.parsing import lisp_string_to_ast
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
from dsg_pddl.grounding.legacy import simplify
from dsg_pddl.grounding.symbols import (
    add_symbol_positions,
    extract_all_symbols,
    generate_objects,
    normalize_symbols,
)
from omniplanner.core.wrappers import MultiRobotWrapper
from omniplanner.domains.tsp import LayerPlanner

logger = logging.getLogger(__name__)


def generate_dense_region_symbol_connectivity_multirobot(G, symbols, robot_states):
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
    add_robot_start_edges(edges, symbols, symbol_lookup, robot_states, layer_planner)
    return edges


def add_robot_start_edges(
    edges,
    symbols,
    symbol_lookup,
    robot_states,
    layer_planner,
    threshold=50,
):
    for robot_id in robot_states.keys():
        start_symbol_key = f"pstart{robot_id}"
        if start_symbol_key in symbol_lookup:
            start_symbol = symbol_lookup[start_symbol_key]
            start_position = start_symbol.position

            for s in symbols:
                if s.symbol.startswith("pstart"):  # Skip other robot start positions
                    continue
                d = layer_planner.get_external_distance(start_position, s.position)
                if d < threshold:
                    edges.append((start_symbol, s, d))


def nearest_place_for_position(place_symbols: list[PddlSymbol], pos: np.ndarray) -> str:
    best_name = place_symbols[0].symbol
    best_d = float("inf")
    for p in place_symbols:
        d = float(np.linalg.norm(p.position - pos))
        if d < best_d:
            best_d = d
            best_name = p.symbol
    return best_name


# Multirobot init wrapper that reuses shared connectivity and containment helpers
def generate_dense_region_init_multirobot(
    G: spark_dsg.DynamicSceneGraph,
    symbols_of_interest: list[PddlSymbol],
    robot_states: dict[str, np.ndarray],
) -> list[tuple]:
    connectivity = generate_dense_region_symbol_connectivity_multirobot(
        G, symbols_of_interest, robot_states
    )
    connectivity_pddl = symbol_connectivity_to_pddl(connectivity)

    initial_pddl: list[tuple] = [("=", ("total-cost",), 0)]
    initial_pddl += connectivity_pddl

    # Containment relations (objects and places -> regions)
    initial_pddl += generate_object_containment(G)
    initial_pddl += generate_place_containment(G)

    # Add each valid robot's starting place using the pstart{robot_id} symbols
    for robot_id, pose in robot_states.items():
        if pose is None:
            continue
        start_symbol_key = f"pstart{robot_id}"
        initial_pddl.append(("at-poi", robot_id, start_symbol_key))

    return initial_pddl


def filter_goal_for_available_objects(
    goal_string: str, available_objects: list[str]
) -> str:
    """Filter goal string to only include objects that are available in the problem."""
    import re

    # Extract object names from goal string (e.g., o2, o3, o21, etc.)
    object_pattern = r"\bo\d+\b"
    goal_objects = re.findall(object_pattern, goal_string)

    # Filter to only include available objects
    available_goal_objects = [obj for obj in goal_objects if obj in available_objects]

    # Create new goal string with only available objects
    if not available_goal_objects:
        return "(and)"  # Empty goal if no objects available

    # For safety goals, create individual safe predicates
    safe_goals = [f"(safe {obj})" for obj in available_goal_objects]
    return f"(and {' '.join(safe_goals)})"


def generate_multirobot_region_pddl(
    G: spark_dsg.DynamicSceneGraph,
    raw_pddl_goal_string: str,
    robot_states: np.ndarray,
) -> tuple[str, list[PddlSymbol]]:
    """Generate a multi-robot PDDL problem for domain region-object-rearrangement-domain-multirobot-fd."""
    # Collect all places/objects/regions and positions
    symbols = extract_all_symbols(G)
    normalize_symbols(symbols)

    symbols_of_interest = symbols
    for robot in robot_states.keys():
        initial_position = robot_states[robot]
        if initial_position is None:
            logger.warning(
                f"Skipping robot {robot} due to missing initial position (None)."
            )
            continue
        start_place_symbol = PddlSymbol(
            "pstart" + str(robot),
            "place",
            ["at-poi"],
            position=np.array(initial_position[:2]),
        )
        # print("start_place_symbol: ", start_place_symbol)
        # print("initial_position: ", initial_position)
        symbols_of_interest = [start_place_symbol] + symbols_of_interest

    add_symbol_positions(G, symbols_of_interest)

    parsed_pddl_goal = lisp_string_to_ast(raw_pddl_goal_string)
    goal_pddl = simplify(parsed_pddl_goal)

    # Robot starts at nearest places to their given 2D states
    robot_ids = [rid for rid, pose in robot_states.items() if pose is not None]
    # Build init facts via shared helpers
    init_facts_tuples: list[tuple] = generate_dense_region_init_multirobot(
        G, symbols_of_interest, robot_states
    )

    # Ensure robot symbols exist (with positions) for downstream planners
    for rid, pose in robot_states.items():
        if pose is None:
            continue
        symbols_of_interest.append(
            PddlSymbol(rid, "robot", [], position=np.array(pose[:2]))
        )

    # Add suspicious facts for all objects
    object_symbols = [s for s in symbols_of_interest if s.layer == "object"]
    init_facts_tuples += [("suspicious", o.symbol) for o in object_symbols]

    # Build objects dict using shared generator and adding robots
    pddl_objects = generate_objects(symbols_of_interest)
    pddl_objects["robot"] = robot_ids
    problem = PddlProblem(
        name="multi-robot-problem",
        domain="region-object-rearrangement-domain-multirobot-fd",
        objects=pddl_objects,
        initial_facts=tuple(init_facts_tuples),
        goal=goal_pddl,
        optimizing=True,
    )
    problem_str = problem.to_string()

    try:
        debug_output_dir = os.getenv("DEBUG_OUTPUT_DIR", "")
        os.makedirs(debug_output_dir, exist_ok=True)
        debug_problem_fn = os.path.join(debug_output_dir, "multi-robot_problem.pddl")
        with open(debug_problem_fn, "w") as f:
            f.write(problem_str)
        logger.debug(f"Saved multi-robot PDDL problem to {debug_problem_fn}")
    except Exception as e:
        logger.warning(f"Failed to save multi-robot PDDL problem: {e}")

    return problem_str, symbols_of_interest


@dispatch
def ground_problem(
    domain: MultiRobotPddlDomain,
    dsg: spark_dsg.DynamicSceneGraph,
    robot_states: dict,
    goal: PddlGoal,
    feedback: Any = None,
) -> MultiRobotWrapper[GroundedPddlProblem]:
    logger.info(f"Grounding PDDL Problem {domain.domain_name}")

    pddl_compliant_robot_states = {k.lower(): v for k, v in robot_states.items()}
    match domain.domain_name:
        # case "goto-object-domain-multirobot-fd":
        #     pddl_problem, symbols = generate_multirobot_inspection_pddl(
        #         dsg, goal.pddl_goal, robot_states
        #     )

        case "region-object-rearrangement-domain-multirobot-fd":
            pddl_problem, symbols = generate_multirobot_region_pddl(
                dsg, goal.pddl_goal, pddl_compliant_robot_states
            )
            # logger.warning(f"!!!!!!!!!!!!!!pddl_problem: {pddl_problem}")
        case _:
            raise NotImplementedError(
                f"I don't know how to ground a domain of type {domain.domain_name}!"
            )

    symbol_dict = {s.symbol: s for s in symbols}

    # TODO: We don't actually want to rely on the robot_states, because
    # robot_states may contain information about robots we actually don't care
    # about for planning purposes.  Instead, we probably want this information
    # passed as part of the goal.

    valid_robot_names = [
        name for name, pose in robot_states.items() if pose is not None
    ]
    wrapper = MultiRobotWrapper[GroundedPddlProblem](
        valid_robot_names, GroundedPddlProblem(domain, pddl_problem, symbol_dict)
    )
    for outer_name in valid_robot_names:
        inner_name = outer_name.lower()
        wrapper.set_name_remap(outer_name, inner_name)
    return wrapper
