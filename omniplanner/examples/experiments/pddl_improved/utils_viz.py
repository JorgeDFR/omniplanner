
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt

import spark_dsg


def _setup_plot(title="DSG Graph"):
    plt.figure(figsize=(12, 10))
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")


def _normalize_node_id(node):
    return node.id.str(True).lower()


def _is_visible(node, node_filter):
    if node_filter is None:
        return True
    return _normalize_node_id(node) in node_filter


def _plot_mesh_places(DSG, draw_text, node_filter=None):
    try:
        layer = DSG.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        layer = DSG.get_layer(20)

    if node_filter is not None:
        valid_nodes = [n for n in layer.nodes
                       if _normalize_node_id(n) in node_filter]
    else:
        valid_nodes = list(layer.nodes)
    valid_node_ids = {_normalize_node_id(n) for n in valid_nodes}

    node_positions = []
    for node in valid_nodes:
        pos = node.attributes.position[:2]
        node_positions.append(pos)
        if draw_text:
            plt.text(pos[0] + 0.2, pos[1] + 0.2, node.id, fontsize=6)

    if node_positions:
        node_positions = np.array(node_positions)
        plt.scatter(node_positions[:, 0], node_positions[:, 1],
                    c="black", s=20, alpha=0.6, label="DSG (Places)")

    # edges
    for src in valid_nodes:
        src_pos = src.attributes.position[:2]
        for neigh in src.siblings():
            dst = DSG.get_node(neigh)
            if _normalize_node_id(dst) not in valid_node_ids:
                continue
            dst_pos = dst.attributes.position[:2]
            plt.plot([src_pos[0], dst_pos[0]],
                     [src_pos[1], dst_pos[1]],
                     'gray', alpha=0.3)


def _plot_objects(DSG, draw_text, node_filter=None):
    layer = DSG.get_layer(spark_dsg.DsgLayers.OBJECTS)

    node_positions = []
    for node in layer.nodes:
        if not _is_visible(node, node_filter):
            continue
        pos = node.attributes.position[:2]
        node_positions.append(pos)
        if draw_text:
            plt.text(pos[0] + 0.2, pos[1] - 0.2, node.id, fontsize=6)

    if node_positions:
        node_positions = np.array(node_positions)
        plt.scatter(node_positions[:, 0], node_positions[:, 1],
                    c="blue", marker='^', s=50, alpha=0.6,
                    label="DSG (Objects)")


def _plot_regions(DSG, draw_edges, draw_text, node_filter=None):
    try:
        places_layer = DSG.get_layer(spark_dsg.DsgLayers.MESH_PLACES)
    except Exception:
        places_layer = DSG.get_layer(20)
    regions_layer = DSG.get_layer(spark_dsg.DsgLayers.ROOMS)
    spark_dsg.add_bounding_boxes_to_layer(DSG, spark_dsg.DsgLayers.ROOMS)

    if node_filter is not None:
        valid_region_nodes = [n for n in regions_layer.nodes
                              if _normalize_node_id(n) in node_filter]
        valid_place_nodes = [n for n in places_layer.nodes
                             if _normalize_node_id(n) in node_filter]
    else:
        valid_region_nodes = list(regions_layer.nodes)
        valid_place_nodes = list(places_layer.nodes)
    valid_place_ids = {_normalize_node_id(n) for n in valid_place_nodes}

    added_label = False
    for node in valid_region_nodes:
        center = node.attributes.bounding_box.world_P_center[:2]
        bb_min = node.attributes.bounding_box.min[:2] - [0.1, 0.1]
        bb_dim = node.attributes.bounding_box.dimensions[:2] + [0.2, 0.2]

        if draw_text:
            plt.text(center[0] + 0.2, center[1] - 0.2,
                    node.id, fontsize=8, color="red")

        label = "DSG (Regions)" if not added_label else None

        rect = plt.Rectangle(
            bb_min, bb_dim[0], bb_dim[1],
            fill=False, edgecolor='red', linestyle='--',
            linewidth=1.5, alpha=0.5, label=label
        )
        plt.gca().add_patch(rect)
        added_label = True

    if draw_edges:
        for src in valid_region_nodes:
            src_pos = src.attributes.bounding_box.world_P_center[:2]
            for child in src.children():
                dst = DSG.get_node(child)
                if _normalize_node_id(dst) not in valid_place_ids:
                    continue
                dst_pos = dst.attributes.position[:2]
                plt.plot([src_pos[0], dst_pos[0]],
                         [src_pos[1], dst_pos[1]],
                         'red', alpha=0.3)


def _plot_dsg_base(
    DSG, title="DSG Graph",
    draw_objects=True,
    draw_regions=True,
    draw_region_edges=False,
    draw_text=True,
    nodes_to_show=None
):
    node_filter = None
    if nodes_to_show is not None:
        node_filter = set(n.lower() for n in nodes_to_show)

    _setup_plot(title)
    _plot_mesh_places(DSG, draw_text=draw_text, node_filter=node_filter)
    if draw_objects:
        _plot_objects(DSG, draw_text=draw_text, node_filter=node_filter)
    if draw_regions:
        _plot_regions(
            DSG,
            draw_edges=draw_region_edges,
            draw_text=draw_text,
            node_filter=node_filter
        )


def visualize_dsg(
    DSG,
    show_objects=True, show_regions=True, show_region_edges=False,
    show_text=True,
    nodes_to_show=None,
    show=True,
):
    _plot_dsg_base(DSG, title="DSG Graph",
                   draw_objects=show_objects,
                   draw_regions=show_regions,
                   draw_region_edges=show_region_edges,
                   draw_text=show_text,
                   nodes_to_show=nodes_to_show)

    fig = plt.gcf()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_pad = 0.2 * (x_max - x_min)
    ax.set_xlim(x_min, x_max + x_pad)
    ax.set_ylim(y_min, y_max)

    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    if show:
        plt.show()

    return fig


def visualize_plan(
    collected_plan, DSG,
    show_objects=True, show_regions=True, show_region_edges=False,
    show_text=True, simplify_legend=False,
    nodes_to_show=None,
    show=True,
):
    if isinstance(collected_plan, dict):
        if len(collected_plan) != 1:
            raise ValueError("Pass a single robot plan or select one robot from the plan dict")
        collected_plan = next(iter(collected_plan.values()))

    actions = collected_plan.actions

    def _action_name(act):
        return getattr(act, "name", act.__class__.__name__)

    def _action_params(act):
        return getattr(act, "parameters", {})

    def _action_path(act):
        params = _action_params(act)
        if "path" in params:
            return np.array(params["path"])
        if "start" in params and "goal" in params:
            return np.array([params["start"], params["goal"]])
        if hasattr(act, "path2d"):
            return np.array(act.path2d)
        return None

    def _action_object_id(act):
        params = _action_params(act)
        symbolic = params.get("symbolic")
        if symbolic and len(symbolic) > 1:
            return symbolic[1]
        return getattr(act, "object_id", "")

    def _action_point(act):
        path = _action_path(act)
        if path is not None and len(path) > 0:
            return np.array(path[-1][:2])
        if hasattr(act, "object_point"):
            return np.array(act.object_point[:2])
        if hasattr(act, "gaze_point"):
            return np.array(act.gaze_point[:2])
        return None

    # --- Base DSG ---
    _plot_dsg_base(DSG, title="Robot Plan on DSG Graph",
                   draw_objects=show_objects,
                   draw_regions=show_regions,
                   draw_region_edges=show_region_edges,
                   draw_text=show_text,
                   nodes_to_show=nodes_to_show)

    # --- Plan overlay ---
    cmap = cm.get_cmap('tab10')
    segment_idx = 0
    robot_start = None

    for act in actions:
        name = _action_name(act)

        if name in ["goto-poi"]:
            path = _action_path(act)
            if path is None or len(path) == 0:
                continue

            if robot_start is None:
                robot_start = path[0]
                plt.scatter(*robot_start[:2],
                            c='blue', s=120, marker='*', label="Start")

            color = cmap(segment_idx % 10)
            if simplify_legend:
                plt.plot(path[:, 0], path[:, 1], '-o', color=color, linewidth=3.0)
            else:
                plt.plot(path[:, 0], path[:, 1], '-o',
                     color=color, linewidth=3.0, label=f"Segment {segment_idx+1}")
            segment_idx += 1

        elif name in ["pick-object"]:
            pos = _action_point(act)
            if pos is None:
                continue

            label = f"Pick '{_action_object_id(act)}'"
            existing_labels = plt.gca().get_legend_handles_labels()[1]

            plt.scatter(pos[0], pos[1],
                        c='red',
                        marker='s', s=100,
                        label=label if label not in existing_labels else "")

        elif name in ["place-object"]:
            pos = _action_point(act)
            if pos is None:
                continue

            label = f"Place '{_action_object_id(act)}'"
            existing_labels = plt.gca().get_legend_handles_labels()[1]

            plt.scatter(pos[0], pos[1],
                        c='green',
                        marker='s', s=100,
                        label=label if label not in existing_labels else "")

        elif name in ["inspect"]:
            pos = _action_point(act)
            if pos is None:
                continue

            label = f"Inspect '{_action_object_id(act)}'"
            existing_labels = plt.gca().get_legend_handles_labels()[1]

            plt.scatter(pos[0], pos[1],
                        c='green',
                        marker='D', s=80,
                        label=label if label not in existing_labels else "")

    # --- Final formatting ---
    fig = plt.gcf()
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x_pad = 0.2 * (x_max - x_min)
    ax.set_xlim(x_min, x_max + x_pad)
    ax.set_ylim(y_min, y_max)

    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    if show:
        plt.show()

    return fig
