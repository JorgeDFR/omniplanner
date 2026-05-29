import spark_dsg


def get_places_layer(G):
    try:
        return G.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        return G.get_layer(20)
