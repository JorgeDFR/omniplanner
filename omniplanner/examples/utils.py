from dataclasses import dataclass
from importlib.resources import as_file, files

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

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


def visualize_plan(collected_plan, DSG, show_objects=True):
    """
    Visualize a robot plan over the entire DSG graph.

    Args:
        collected_plan: ActionSequence object (e.g., collected_plans['euclid'])
        DSG: spark_dsg.DynamicSceneGraph
        show_objects: Whether to plot pick/place actions
    """
    actions = collected_plan.actions

    # --- 1. Print actions ---
    print("\n==== Action Sequence ====")
    for i, act in enumerate(actions):
        if act.__class__.__name__ == "Follow":
            print(f"{i+1}: Move along path until {act.path2d[-1][:2]}")
        elif act.__class__.__name__ == "Pick":
            print(f"{i+1}: Pick object '{act.object_id}' at {act.object_point[:2]}")
        elif act.__class__.__name__ == "Place":
            print(f"{i+1}: Place object '{act.object_id}' at {act.object_point[:2]}")
        else:
            print(f"{i+1}: {act}")

    # --- 2. Plot DSG graph ---
    plt.figure(figsize=(12, 10))
    plt.title("Robot Plan on DSG Graph")
    plt.xlabel("X")
    plt.ylabel("Y")

    # --- Plot places layer nodes ---
    try:
        places_layer = DSG.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = DSG.get_layer(20)

    node_positions = []
    for node in places_layer.nodes:
        pos = node.attributes.position[:2]
        node_positions.append(pos)
        plt.text(pos[0]+0.2, pos[1]+0.2, node.id, fontsize=6, color="black")
    node_positions = np.array(node_positions)
    if len(node_positions) > 0:
        plt.scatter(node_positions[:, 0], node_positions[:, 1],
                    label="DSG (Places Layer)", c="black", s=50, alpha=0.6)

    # Plot edges
    for src_node in places_layer.nodes:
        src_pos = src_node.attributes.position[:2]
        for neighbor in src_node.siblings():
            dst_node = DSG.get_node(neighbor)
            dst_pos = dst_node.attributes.position[:2]
            plt.plot([src_pos[0], dst_pos[0]], [src_pos[1], dst_pos[1]], 'gray', alpha=0.3)

    # --- 3. Plot robot path with colored segments ---
    cmap = cm.get_cmap('tab10')  # color map for different segments
    segment_idx = 0

    # Keep track of robot start position
    robot_start = None
    for act in actions:
        if act.__class__.__name__ == "Follow":
            path = np.array(act.path2d)
            if robot_start is None:
                robot_start = path[0]  # first point of the first Follow
                plt.scatter(robot_start[0], robot_start[1], c='blue', s=120, marker='*', label="Start")
            color = cmap(segment_idx % 10)
            plt.plot(path[:, 0], path[:, 1], '-o', color=color,
                     label=f"Segment {segment_idx+1}")
            segment_idx += 1
        elif show_objects and act.__class__.__name__ in ["Pick", "Place"]:
            obj_pos = np.array(act.object_point[:2])
            plt.scatter(obj_pos[0], obj_pos[1],
                        c='red' if act.__class__.__name__=="Pick" else 'green',
                        marker='s', s=100,
                        label=f"{act.__class__.__name__} '{act.object_id}'"
                        if f"{act.__class__.__name__} '{act.object_id}'" not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.text(obj_pos[0]+0.2, obj_pos[1]+0.2, f"{act.object_id}", fontsize=8)

    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()