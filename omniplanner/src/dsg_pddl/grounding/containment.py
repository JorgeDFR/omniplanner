import numpy as np
import spark_dsg

from dsg_pddl.grounding.dsg_access import get_places_layer
from dsg_pddl.grounding.symbols import normalize_symbol


def generate_object_containment(G):
    places_layer = get_places_layer(G)

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
    places_layer_2d = get_places_layer(G)

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
