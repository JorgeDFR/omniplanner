import spark_dsg

from dsg_pddl.grounding.dsg_access import get_places_layer
from dsg_pddl.core.models import PddlSymbol
from dsg_pddl.core.parsing import pddl_char_to_dsg_char


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


def extract_all_symbols(G):
    places_layer = get_places_layer(G)

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
