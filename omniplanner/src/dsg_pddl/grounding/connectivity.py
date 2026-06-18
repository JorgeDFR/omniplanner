import numpy as np
import spark_dsg

from dsg_pddl.grounding.symbols import normalize_symbol


def symbol_connectivity_to_pddl(connectivity):
    connections_init = []

    for info_s, info_t, dist in connectivity:
        s = info_s.symbol
        t = info_t.symbol
        d = int(dist)

        connected = ("connected", s, t)
        distance = ("=", ("distance", s, t), d)
        connections_init.append(connected)
        connections_init.append(distance)

        # TODO: The reverse distance is not required by the compressed sampler
        distance_rev = ("=", ("distance", t, s), d)
        connections_init.append(distance_rev)

    return connections_init


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
