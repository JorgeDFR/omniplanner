from dataclasses import dataclass
from importlib.resources import as_file, files

import random
import numpy as np

import spark_dsg
import dsg_pddl.domains


def load_omniplanner_pddl_domain(domain_name):
    with as_file(files(dsg_pddl.domains).joinpath(domain_name)) as path:
        print(f"Loading domain {path}")
        with open(str(path), "r") as fo:
            domain = fo.read()
    return domain


@dataclass
class DummyRobotPlanningAdaptor:
    name: str
    robot_type: str
    parent_frame: str
    child_frame: str


def build_test_dsg():
    """An example scene graph for testing"""
    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(2, "O", spark_dsg.DsgLayers.OBJECTS)
    G.add_layer(3, "p", spark_dsg.DsgLayers.PLACES)
    G.add_layer(4, "R", spark_dsg.DsgLayers.ROOMS)
    G.add_layer(5, "B", spark_dsg.DsgLayers.BUILDINGS)
    G.add_layer(20, "P", spark_dsg.DsgLayers.MESH_PLACES)

    room = spark_dsg.RoomNodeAttributes()
    room.position = np.array([0, 0, 0])
    room.semantic_label = 0

    G.add_node(spark_dsg.DsgLayers.ROOMS, spark_dsg.NodeSymbol("R", 0).value, room)

    place1 = spark_dsg.PlaceNodeAttributes()
    place1.position = np.array([-1, 0, 0])
    G.add_node(spark_dsg.DsgLayers.PLACES, spark_dsg.NodeSymbol("p", 0).value, place1)
    place2 = spark_dsg.PlaceNodeAttributes()
    place2.position = np.array([1, 0, 0])
    G.add_node(spark_dsg.DsgLayers.PLACES, spark_dsg.NodeSymbol("p", 1).value, place2)

    place1_2d = spark_dsg.PlaceNodeAttributes()
    place1_2d.position = np.array([-1.1, 0, 0])
    place1_2d.semantic_label = 4  # ground
    G.add_node(
        spark_dsg.DsgLayers.MESH_PLACES, spark_dsg.NodeSymbol("P", 0).value, place1_2d
    )
    place2_2d = spark_dsg.PlaceNodeAttributes()
    place2_2d.position = np.array([1.1, 0, 0])
    place2_2d.semantic_label = 4  # ground
    G.add_node(
        spark_dsg.DsgLayers.MESH_PLACES, spark_dsg.NodeSymbol("P", 1).value, place2_2d
    )

    obj1 = spark_dsg.ObjectNodeAttributes()
    obj1.position = np.array([-1.5, 0, 0])
    obj1.semantic_label = 34  # box
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", 0).value, obj1)
    obj2 = spark_dsg.PlaceNodeAttributes()
    obj2.position = np.array([1.5, 0, 0])
    obj2.semantic_label = 15  # rock
    G.add_node(spark_dsg.DsgLayers.OBJECTS, spark_dsg.NodeSymbol("O", 1).value, obj2)

    G.insert_edge(
        spark_dsg.NodeSymbol("R", 0).value, spark_dsg.NodeSymbol("p", 0).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("R", 0).value, spark_dsg.NodeSymbol("p", 1).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("p", 0).value, spark_dsg.NodeSymbol("p", 1).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("p", 0).value, spark_dsg.NodeSymbol("O", 0).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("p", 1).value, spark_dsg.NodeSymbol("O", 1).value
    )
    G.insert_edge(
        spark_dsg.NodeSymbol("P", 0).value, spark_dsg.NodeSymbol("P", 1).value
    )

    labelspaces = {
        "labelspaces": {
            "_l2p0": [
                [0, "unknown"],
                [1, "sky"],
                [2, "tree"],
                [3, "water"],
                [4, "ground"],
                [5, "grass"],
                [6, "sand"],
                [7, "sidewalk"],
                [8, "dock"],
                [9, "road"],
                [10, "path"],
                [11, "vehicle"],
                [12, "building"],
                [13, "shelter"],
                [14, "signal"],
                [15, "rock"],
                [16, "fence"],
                [17, "boat"],
                [18, "sign"],
                [19, "hill"],
                [20, "bridge"],
                [21, "wall"],
                [22, "floor"],
                [23, "ceiling"],
                [24, "door"],
                [25, "stairs"],
                [26, "pole"],
                [27, "rail"],
                [28, "structure"],
                [29, "window"],
                [30, "surface"],
                [31, "flora"],
                [32, "flower"],
                [33, "bed"],
                [34, "box"],
                [35, "storage"],
                [36, "barrel"],
                [37, "bag"],
                [38, "basket"],
                [39, "seating"],
                [40, "flag"],
                [41, "decor"],
                [42, "light"],
                [43, "appliance"],
                [44, "trash"],
                [45, "bicycle"],
                [46, "food"],
                [47, "clothes"],
                [48, "thing"],
                [49, "animal"],
                [50, "human"],
            ],
            "_l4p0": [
                [0, "unknown"],
                [1, "road"],
                [2, "field"],
                [3, "shelter"],
                [4, "indoor"],
                [5, "stairs"],
                [6, "sidewalk"],
                [7, "path"],
                [8, "boundary"],
                [9, "shore"],
                [10, "ground"],
                [11, "dock"],
                [12, "parking"],
                [13, "footing"],
            ],
        }
    }

    G.metadata.add(labelspaces)

    return G


def cluster_regions(mesh_nodes, num_regions, iterations=5, seed=None):
    """
    Perform simple K-means clustering over mesh node positions.

    mesh_nodes: dict[(r,c)] -> node_symbol
    returns: dict[node_symbol] -> region_id
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Convert mesh nodes to list with positions
    node_list = []
    for (r, c), symbol in mesh_nodes.items():
        pos = np.array([c, r])  # (x, y)
        node_list.append((symbol, pos))

    # -----------------------------
    # 1. Initialize random centers
    # -----------------------------
    initial = random.sample(node_list, num_regions)
    centers = [pos.copy() for (_, pos) in initial]

    # -----------------------------
    # 2. Iterate K-means
    # -----------------------------
    for _ in range(iterations):
        clusters = {i: [] for i in range(num_regions)}

        # Assignment step
        for symbol, pos in node_list:
            distances = [np.linalg.norm(pos - c) for c in centers]
            cluster_id = int(np.argmin(distances))
            clusters[cluster_id].append((symbol, pos))

        # Update step
        for i in range(num_regions):
            if len(clusters[i]) > 0:
                positions = np.array([p for (_, p) in clusters[i]])
                centers[i] = positions.mean(axis=0)

    # -----------------------------
    # 3. Final assignment
    # -----------------------------
    assignment = {}
    for i in range(num_regions):
        for symbol, _ in clusters[i]:
            assignment[symbol] = i

    return assignment, centers


def build_scalable_dsg(
    grid_rows=10, grid_cols=10, cell_size=1.0,
    num_objects=20, num_regions=2,
    seed=None,
):
    """
    Build a scalable Dynamic Scene Graph using:
    - Mesh places arranged in a grid
    - Objects sampled near mesh nodes
    - Regions defined as partitions of the grid
    """

    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(2, "O", spark_dsg.DsgLayers.OBJECTS)
    G.add_layer(3, "P", spark_dsg.DsgLayers.MESH_PLACES)
    G.add_layer(4, "R", spark_dsg.DsgLayers.ROOMS)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # -----------------------------
    # 1. Create Mesh Places (grid)
    # -----------------------------
    mesh_nodes = {}
    node_id = 0
    for r in range(grid_rows):
        for c in range(grid_cols):
            attr = spark_dsg.PlaceNodeAttributes()
            attr.position = np.array([c * cell_size, r * cell_size, 0.0])
            attr.semantic_label = 4  # ground

            symbol = spark_dsg.NodeSymbol("P", node_id).value
            G.add_node(spark_dsg.DsgLayers.MESH_PLACES, symbol, attr)

            mesh_nodes[(r, c)] = symbol
            node_id += 1

    # -----------------------------
    # 2. Connect Mesh Places (grid edges)
    # -----------------------------
    for r in range(grid_rows):
        for c in range(grid_cols):
            current = mesh_nodes[(r, c)]

            # Right neighbor
            if c + 1 < grid_cols:
                G.insert_edge(current, mesh_nodes[(r, c + 1)])

            # Down neighbor
            if r + 1 < grid_rows:
                G.insert_edge(current, mesh_nodes[(r + 1, c)])

    # -----------------------------
    # 3. Create Objects
    # -----------------------------
    object_nodes = []
    for i in range(num_objects):
        # sample a random grid cell
        r = random.randint(0, grid_rows - 1)
        c = random.randint(0, grid_cols - 1)

        base_pos = np.array([c * cell_size, r * cell_size, 0.0])

        # small random offset (to be "near" mesh node)
        offset = np.random.uniform(-0.3, 0.3, size=3)
        offset[2] = 0  # keep planar

        attr = spark_dsg.ObjectNodeAttributes()
        attr.position = base_pos + offset
        attr.semantic_label = random.randint(30, 40)  # random object class

        symbol = spark_dsg.NodeSymbol("O", i).value
        G.add_node(spark_dsg.DsgLayers.OBJECTS, symbol, attr)

        # connect to nearest mesh node
        nearest_mesh = mesh_nodes[(r, c)]
        G.insert_edge(nearest_mesh, symbol)

        object_nodes.append((symbol, r, c))

    # -----------------------------
    # 4. Create Regions (grid partitions)
    # -----------------------------
    region_nodes = []
    for region_id in range(num_regions):
        attr = spark_dsg.RoomNodeAttributes()
        attr.position = np.array([0, 0, 0])
        attr.semantic_label = region_id

        symbol = spark_dsg.NodeSymbol("R", region_id).value
        G.add_node(spark_dsg.DsgLayers.ROOMS, symbol, attr)
        region_nodes.append(symbol)

    # Cluster mesh nodes
    mesh_to_region, centroids = cluster_regions(mesh_nodes, num_regions, seed=seed)

    # Update region node position
    for region_id, centroid in zip(range(num_regions), centroids):
        node = G.get_node(region_nodes[region_id])
        node.attributes.position = np.array([centroid[0], centroid[1], 0.0])

    # Assign mesh nodes to regions
    for mesh_symbol, region_id in mesh_to_region.items():
        G.insert_edge(region_nodes[region_id], mesh_symbol)

    return G