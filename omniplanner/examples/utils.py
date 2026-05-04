import math
import random
import numpy as np

import spark_dsg
import dsg_pddl.domains

from dataclasses import dataclass
from importlib.resources import as_file, files


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


def build_scalable_dsg(
    num_nodes=100,
    valid_map_areas=[(0, 0, 10, 10)],
    num_objects=20,
    num_regions=2,
    seed=None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    G = spark_dsg.DynamicSceneGraph()
    G.add_layer(2, "O", spark_dsg.DsgLayers.OBJECTS)
    G.add_layer(3, "P", spark_dsg.DsgLayers.MESH_PLACES)
    G.add_layer(4, "R", spark_dsg.DsgLayers.ROOMS)

    # -------------------------------------------------
    # Compute work bounds (bounding square)
    # -------------------------------------------------
    xmin = min(r[0] for r in valid_map_areas)
    ymin = min(r[1] for r in valid_map_areas)
    xmax = max(r[2] for r in valid_map_areas)
    ymax = max(r[3] for r in valid_map_areas)

    side = max(xmax - xmin, ymax - ymin)

    work_bounds = (xmin, ymin, xmin + side, ymin + side)

    # -------------------------------------------------
    # Compute density-based threshold
    # -------------------------------------------------
    valid_area = sum((r[2] - r[0]) * (r[3] - r[1]) for r in valid_map_areas)
    min_dist = math.sqrt(valid_area / num_nodes)
    edge_threshold = 1.5 * min_dist

    # -------------------------------------------------
    # 1. Mesh Places (nodes)
    # -------------------------------------------------
    node_ids, positions = generate_valid_random_nodes(
        num_nodes=num_nodes,
        valid_map_areas=valid_map_areas,
        work_bounds=work_bounds,
        min_dist=min_dist,
        seed=seed,
    )

    mesh_nodes = []
    for i, pos in zip(node_ids, positions):
        attr = spark_dsg.PlaceNodeAttributes()
        attr.position = pos
        attr.semantic_label = 4

        symbol = spark_dsg.NodeSymbol("P", i).value
        G.add_node(spark_dsg.DsgLayers.MESH_PLACES, symbol, attr)

        mesh_nodes.append(symbol)

    # -------------------------------------------------
    # 2. Mesh Places (edges)
    # -------------------------------------------------
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if np.linalg.norm(positions[i] - positions[j]) < edge_threshold:
                G.insert_edge(mesh_nodes[i], mesh_nodes[j])

    # -------------------------------------------------
    # 3. Objects
    # -------------------------------------------------
    object_nodes = []
    objects_data = []
    for _ in range(num_objects):
        # sample a mesh node (anchor)
        node_idx = random.choice(range(len(mesh_nodes)))
        base_pos = positions[node_idx]

        # add small local offset
        offset = np.random.uniform(-0.3, 0.3, size=3)
        offset[2] = 0.0  # keep planar
        obj_pos = base_pos + offset
        semantic_label = random.randint(0, 50)

        objects_data.append((obj_pos, node_idx, semantic_label))

    # Sort objects spatially
    objects_data.sort(key=lambda x: morton_code(x[0][0], x[0][1]))
    for i, (obj_pos, node_idx, semantic_label) in enumerate(objects_data):
        attr = spark_dsg.ObjectNodeAttributes()
        attr.position = obj_pos
        attr.semantic_label = semantic_label

        symbol = spark_dsg.NodeSymbol("O", i).value
        G.add_node(spark_dsg.DsgLayers.OBJECTS, symbol, attr)

        mesh_symbol = mesh_nodes[node_idx]
        G.insert_edge(mesh_symbol, symbol)

        object_nodes.append(symbol)

    # -------------------------------------------------
    # 4. Regions
    # -------------------------------------------------
    region_nodes = []
    for region_id in range(num_regions):
        attr = spark_dsg.RoomNodeAttributes()
        attr.position = np.array([0, 0, 0])
        attr.semantic_label = region_id

        symbol = spark_dsg.NodeSymbol("R", region_id).value
        G.add_node(spark_dsg.DsgLayers.ROOMS, symbol, attr)
        region_nodes.append(symbol)

    mesh_nodes_dict = {i: positions[i] for i in range(len(mesh_nodes))}
    mesh_to_region, centroids = cluster_regions(mesh_nodes_dict, num_regions, seed=seed)

    for region_id, centroid in zip(range(num_regions), centroids):
        node = G.get_node(region_nodes[region_id])
        node.attributes.position = np.array([centroid[0], centroid[1], 0.0])

    for mesh_idx, region_id in mesh_to_region.items():
        G.insert_edge(region_nodes[region_id], mesh_nodes[mesh_idx])

    return G


def generate_valid_random_nodes(
    num_nodes,
    valid_map_areas,
    work_bounds,
    min_dist,
    seed=None,
    max_attempts=None,
):
    """
    Generates spatial nodes inside valid rectangular regions
    with a minimum distance constraint.
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    (xmin, ymin, xmax, ymax) = work_bounds

    def is_valid_point(x, y):
        for (x0, y0, x1, y1) in valid_map_areas:
            if x0 <= x <= x1 and y0 <= y <= y1:
                return True
        return False

    nodes = []
    positions = []

    if max_attempts is None:
        max_attempts = num_nodes * 50

    attempts = 0
    node_id = 0

    while len(nodes) < num_nodes and attempts < max_attempts:
        attempts += 1

        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)

        if not is_valid_point(x, y):
            continue

        pos = np.array([x, y, 0.0])

        # minimum distance constraint
        if any(np.linalg.norm(pos - p) < min_dist for p in positions):
            continue

        nodes.append(node_id)
        positions.append(pos)
        node_id += 1

    # Spatial sorting (bottom-left → top-right)
    indexed_positions = list(enumerate(positions))
    indexed_positions.sort(key=lambda x: morton_code(x[1][0], x[1][1]))

    positions_sorted = []
    for new_id, (old_id, pos) in enumerate(indexed_positions):
        positions_sorted.append(pos)

    nodes_sorted = list(range(len(positions)))

    return nodes_sorted, positions_sorted


def cluster_regions(mesh_nodes_dict, num_regions, iterations=5, seed=None):
    """
    K-means clustering over mesh node positions.

    Args:
        mesh_nodes_dict: dict[int -> np.ndarray([x,y])]
        num_regions: number of clusters
        iterations: k-means iterations
        seed: randomness control

    Returns:
        assignment: dict[node_id] -> region_id
        centers: list of cluster centroids
    """

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if len(mesh_nodes_dict) == 0 or num_regions <= 0:
        return {}, []

    # -----------------------------
    # 1. Convert dict to aligned arrays
    # -----------------------------
    node_ids = list(mesh_nodes_dict.keys())
    positions = np.array([mesh_nodes_dict[i] for i in node_ids])

    n = len(node_ids)

    # -----------------------------
    # 2. Initialize centers randomly
    # -----------------------------
    init_indices = random.sample(range(n), min(num_regions, n))
    centers = positions[init_indices].copy()

    # pad if needed
    while len(centers) < num_regions:
        centers = np.vstack([centers, positions[random.randint(0, n - 1)]])

    # -----------------------------
    # 3. K-means iterations
    # -----------------------------
    clusters = None

    for _ in range(iterations):
        clusters = {i: [] for i in range(num_regions)}

        # assignment step
        for idx, pos in enumerate(positions):
            dists = np.linalg.norm(centers - pos, axis=1)
            cluster_id = int(np.argmin(dists))
            clusters[cluster_id].append(idx)

        # update step
        for k in range(num_regions):
            if clusters[k]:
                centers[k] = positions[clusters[k]].mean(axis=0)

    # -----------------------------
    # 4. Build final mapping
    # -----------------------------
    assignment = {}

    for region_id, idxs in clusters.items():
        for idx in idxs:
            node_id = node_ids[idx]
            assignment[node_id] = region_id

    return assignment, centers.tolist()


def morton_code(x, y, scale=1000):
    x = int(x * scale)
    y = int(y * scale)

    def part1by1(n):
        n &= 0x0000ffff
        n = (n | (n << 8)) & 0x00FF00FF
        n = (n | (n << 4)) & 0x0F0F0F0F
        n = (n | (n << 2)) & 0x33333333
        n = (n | (n << 1)) & 0x55555555
        return n

    return part1by1(x) | (part1by1(y) << 1)