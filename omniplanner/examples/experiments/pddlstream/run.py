from __future__ import print_function

from pathlib import Path
import sys

import numpy as np
import spark_dsg

from pddlstream.utils import read
from pddlstream.algorithms.meta import solve
from pddlstream.language.constants import PDDLProblem, print_solution
from pddlstream.language.generator import from_gen_fn, from_test

from dsg_pddl.core.models import PddlSymbol
from dsg_pddl.grounding.legacy import (
    add_symbol_positions,
    generate_object_containment,
    generate_place_containment,
)
from dsg_pddl.core.parsing import extract_facts, lisp_string_to_ast
from omniplanner.domains.tsp import LayerPlanner

PDDL_IMPROVED_DIR = Path(__file__).resolve().parents[1] / "pddl_improved"
if str(PDDL_IMPROVED_DIR) not in sys.path:
    sys.path.insert(0, str(PDDL_IMPROVED_DIR))

from utils import build_scalable_dsg
from utils_viz import visualize_dsg


def retrieve_dsg_symbols(G):
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

    return place_symbols, object_symbols, region_symbols


def extract_symbols_of_interest(pddl_goal, containment_relations):
    place_facts = extract_facts(pddl_goal, "at-place")
    place_facts += extract_facts(pddl_goal, "visited-place")
    place_facts_extra = extract_facts(pddl_goal, "object-in-place")

    object_facts = extract_facts(pddl_goal, "at-object")
    object_facts += extract_facts(pddl_goal, "visited-object")
    object_facts += extract_facts(pddl_goal, "suspicious")
    object_facts += extract_facts(pddl_goal, "holding")
    object_facts += extract_facts(pddl_goal, "safe")
    object_facts += extract_facts(pddl_goal, "object-in-place")

    place_symbols = [PddlSymbol(f[1], "place", []) for f in place_facts]
    place_symbols += [PddlSymbol(f[2], "place", []) for f in place_facts_extra]
    object_symbols = [PddlSymbol(f[1], "object", []) for f in object_facts]

    for cont_type, symbol_1, symbol_2 in containment_relations:
        if cont_type == "object-in-place" and symbol_1 in [
            obj.symbol for obj in object_symbols
        ]:
            place_symbols += [PddlSymbol(symbol_2, "place", [])]

    containment_relations_of_interest = []
    for cont_type, symbol_1, symbol_2 in containment_relations:
        if cont_type == "object-in-place" and symbol_1 in [
            obj.symbol for obj in object_symbols
        ]:
            containment_relations_of_interest.append((cont_type, symbol_1, symbol_2))
        if cont_type == "place-in-region" and symbol_1 in [
            place.symbol for place in place_symbols
        ]:
            containment_relations_of_interest.append((cont_type, symbol_1, symbol_2))

    # region_facts = extract_facts(pddl_goal, "in-region")
    # region_facts += extract_facts(pddl_goal, "visited-region")

    return place_symbols, object_symbols, containment_relations_of_interest


def main():
    directory = Path(__file__).resolve().parent
    domain_pddl = read(str(directory / "domain.pddl"))
    stream_pddl = read(str(directory / "stream.pddl"))
    constant_map = {}

    G = build_scalable_dsg(
        num_nodes=100,
        valid_map_areas=[(0, 0, 30, 30)],
        num_objects=5,
        num_regions=4,
        seed=123,
    )

    pddl_goal = "(and (at-place p99) (object-in-place o4 p50) (visited-region r0))"
    goal = lisp_string_to_ast(pddl_goal)

    containment_relations = generate_object_containment(G)
    containment_relations += generate_place_containment(G)

    places, objects, containment_relations_of_interest = extract_symbols_of_interest(
        goal, containment_relations
    )
    all_places, _, regions = retrieve_dsg_symbols(G)

    pstart = PddlSymbol("pstart", "place", ["at-poi"], position=[0.0, 0.0])
    places.append(pstart)
    all_places.append(pstart)
    init = (
        [("place", p.symbol.lower()) for p in places]
        + [("dsg-object", o.symbol.lower()) for o in objects]
        + [("region", r.symbol.lower()) for r in regions]
        + [containment for containment in containment_relations_of_interest]
        + [("at-poi", pstart.symbol)]
    )

    layer_planner = LayerPlanner(G, spark_dsg.DsgLayers.MESH_PLACES)
    symbol_to_pddl = {p.symbol.lower(): p for p in all_places}
    add_symbol_positions(G, all_places)

    available_samples = [p.symbol.lower() for p in all_places]

    def sample_region(region):
        while True:
            for cont_type, symbol_1, symbol_2 in containment_relations:
                if (
                    cont_type == "place-in-region"
                    and symbol_2 == region
                    and symbol_1 in available_samples
                ):
                    available_samples.remove(symbol_1)
                    yield (symbol_1,)

    def check_connected(p1_pddl, p2_pddl):
        return get_distance(p1_pddl, p2_pddl) < np.inf

    def get_distance(p1_pddl, p2_pddl):
        p1 = symbol_to_pddl[p1_pddl]
        p2 = symbol_to_pddl[p2_pddl]
        return layer_planner.get_external_distance(p1.position, p2.position)

    stream_map = {
        "sample-region": from_gen_fn(sample_region),
        "check-connected": from_test(check_connected),
        "distance": get_distance,
    }

    problem = PDDLProblem(domain_pddl, constant_map, stream_pddl, stream_map, init, goal)

    solution = solve(
        problem,
        algorithm="adaptive",  # 'incremental' | 'focused' | 'binding' | 'adaptive'
        unit_costs=True,
        planner="ff-astar2",
        max_time=20.0,
        debug=False,
        verbose=False,
    )

    print_solution(solution)
    visualize_dsg(G)


if __name__ == "__main__":
    main()
