"""Execution feedback and repair requests for the single-robot region domains."""

from dataclasses import dataclass, field

import numpy as np
import spark_dsg
from dsg_pddl.core.models import RegionMissionState
from dsg_pddl.grounding.containment import generate_place_containment
from dsg_pddl.grounding.improved_region import effective_scene
from omniplanner.domains.tsp import LayerPlanner
from omniplanner_msgs.msg import RegionPlanStatus, RegionReplanRequest
from rclpy.qos import HistoryPolicy, QoSProfile
from robot_executor_msgs.msg import ActionDoneMsg

ACTION_TYPES = {"goto-poi": "FOLLOW", "inspect": "GAZE",
                "pick-object": "PICK", "place-object": "PLACE"}


@dataclass
class Mission:
    goal: object
    solve: object
    state: RegionMissionState
    plan_id: str = ""
    actions: list = field(default_factory=list)
    paths: list = field(default_factory=list)
    object_positions: dict = field(default_factory=dict)
    place_positions: dict = field(default_factory=dict)
    next_action: int = 0
    retries: int = 0
    invalidated: bool = False
    finished: bool = False


class RegionRepair:
    def __init__(self, node):
        self.node = node
        self.missions = {}
        self.max_retries = node.declare_parameter("region_max_replans", 3).value
        self.move_threshold = node.declare_parameter("region_move_threshold_m", 0.10).value
        self.status = node.create_publisher(RegionPlanStatus, "~/region_plan_status", 10)
        node.create_subscription(RegionReplanRequest, "~/replan_request", self.request, 10)
        qos = QoSProfile(depth=10, history=HistoryPolicy.KEEP_ALL)
        if node.declare_parameter("region_indexed_feedback", False).value:
            # Import only when this executor interface is enabled.
            from robot_executor_msgs.msg import ActionResultMsg

            for robot in node.robot_adaptors:
                node.create_subscription(
                    ActionResultMsg, f"/{robot}/nav_executor/action_result", self.result, qos,
                )
        else:
            node.create_subscription(ActionDoneMsg, "/action_done", self.done, qos)

    def new_goal(self, goal, solve):
        old = self.missions.get(goal.robot_id)
        state = RegionMissionState()
        if old is not None:
            state.holding = old.state.holding
            state.placements = old.state.placements.copy()
            state.blocked_edges = old.state.blocked_edges.copy()
        mission = Mission(goal, solve, state)
        self.missions[goal.robot_id] = mission
        return mission

    def report(self, mission, status, reason=""):
        self.status.publish(RegionPlanStatus(
            robot_id=mission.goal.robot_id, plan_id=mission.plan_id, status=status, reason=reason,
        ))

    def request(self, msg):
        mission = self.missions.get(msg.robot_id)
        if mission is None or msg.plan_id != mission.plan_id:
            return
        if mission.finished:
            if mission.actions and mission.next_action == len(mission.actions):
                self.report(mission, RegionPlanStatus.SATISFIED)
            return
        if len(msg.blocked_sources) != len(msg.blocked_targets):
            self.node.get_logger().warning("Ignoring mismatched blocked-edge endpoint lists")
            return
        if msg.completed_actions > len(mission.actions):
            self.node.get_logger().warning("Ignoring invalid completed action count")
            return
        while mission.next_action < msg.completed_actions:
            index = mission.next_action
            self.feedback(msg.robot_id, msg.plan_id, index, ACTION_TYPES[mission.actions[index][0]], True, "")
        if mission.finished:
            self.report(mission, RegionPlanStatus.SATISFIED)
            return
        if mission.retries >= self.max_retries:
            mission.finished = True
            self.report(mission, RegionPlanStatus.RETRY_LIMIT, msg.reason)
            return
        mission.state.blocked_edges = {
            frozenset((source.lower(), target.lower()))
            for source, target in zip(msg.blocked_sources, msg.blocked_targets)
        }
        self.record_visits(mission.state, {place.lower() for place in msg.reached_places})
        mission.retries += 1
        mission.solve(mission.goal, mission)

    def dispatch(self, mission, message, plan, graph):
        layer = LayerPlanner(graph, spark_dsg.DsgLayers.MESH_PLACES)
        if len(plan.symbolic_actions) != len(message.actions):
            raise ValueError("Repair requires one executor action per symbolic action")
        paths = []
        for action, points in zip(plan.symbolic_actions, plan.parameterized_actions):
            path = []
            if action[0] == "goto-poi":
                for point in points:
                    node = layer.get_closest_node_id(point)
                    if np.allclose(layer.node_value_to_position[node], point):
                        name = layer.node_value_to_symbol[node]
                        if not path or path[-1] != name:
                            path.append(name)
            paths.append(path)
        mission.plan_id = message.plan_id
        mission.actions = plan.symbolic_actions
        mission.paths = paths
        mission.object_positions = self.object_positions(graph)
        place_names = {name for path in paths for name in path}
        place_names.update(name for action in mission.actions if action[0] != "goto-poi"
                           for name in action[2:])
        mission.place_positions = {
            name: layer.node_value_to_position[layer.symbol_to_node_value[name]].copy()
            for name in place_names if name in layer.symbol_to_node_value
        }
        mission.next_action = 0
        mission.invalidated = False

    @staticmethod
    def object_positions(graph):
        return {
            node.id.str(True).lower(): np.asarray(node.attributes.position).copy()
            for node in graph.get_layer(spark_dsg.DsgLayers.OBJECTS).nodes
        }

    def graph_updated(self, graph):
        active = [m for m in self.missions.values() if m.plan_id and not m.finished
                  and not m.invalidated and m.next_action < len(m.actions)]
        if not active:
            return
        positions = self.object_positions(graph)
        layer = LayerPlanner(graph, spark_dsg.DsgLayers.MESH_PLACES)
        for mission in active:
            nodes = layer.symbol_to_node_value
            routes = mission.paths[mission.next_action:]
            remaining = mission.actions[mission.next_action:]
            places = {name for path in routes for name in path}
            places.update(name for action in remaining if action[0] != "goto-poi"
                          for name in action[2:] if name != "pstart")
            if any(name not in nodes or np.linalg.norm(
                mission.place_positions[name] - layer.node_value_to_position[nodes[name]]
            ) > self.move_threshold for name in places) or any(
                not layer.nx_layer.has_edge(nodes[left], nodes[right])
                for path in routes for left, right in zip(path, path[1:])
            ):
                self.invalidate(mission, "remaining route changed")
                continue
            for action in remaining:
                if action[0] != "goto-poi" and action[3] != "pstart" and not layer.nx_layer.has_edge(
                    nodes[action[2]], nodes[action[3]],
                ):
                    self.invalidate(mission, "interaction stance changed")
                    break
                if action[0] not in ("pick-object", "inspect"):
                    continue
                obj = action[1]
                if obj in mission.state.placements or obj == mission.state.holding:
                    continue
                before, after = mission.object_positions.get(obj), positions.get(obj)
                if before is not None and (after is None or np.linalg.norm(before - after) > self.move_threshold):
                    self.invalidate(mission, f"target {obj} moved or was removed")
                    break

    def invalidate(self, mission, reason):
        if not mission.invalidated:
            mission.invalidated = True
            self.report(mission, RegionPlanStatus.INVALIDATED, reason)

    def result(self, msg):
        if msg.status == msg.STATUS_PREEMPTED:
            return
        self.feedback(msg.robot_name, msg.plan_id, msg.action_index, msg.action_type,
                      msg.status == msg.STATUS_SUCCESS, msg.detail)

    def done(self, msg):
        mission = self.missions.get(msg.robot_name)
        if mission is None or mission.next_action >= len(mission.actions):
            return
        action = mission.actions[mission.next_action]
        if msg.action_type != "FOLLOW" and msg.object_id.lower() != action[1]:
            return
        if msg.plan_id != mission.plan_id or mission.finished or msg.action_type != ACTION_TYPES[action[0]]:
            return
        if (not msg.success and msg.action_type == "PICK"
                and msg.header.frame_id == "operator_gave_up"):
            mission.finished = True
            self.report(mission, RegionPlanStatus.FAILED, "operator_gave_up")
            return
        reached = None
        if msg.action_type == "FOLLOW" and msg.points:
            layer = LayerPlanner(self.node.dsg_last, spark_dsg.DsgLayers.MESH_PLACES)
            reached = set()
            for point in msg.points:
                xy = (point.x, point.y)
                name = layer.get_closest_node_id(xy)
                if np.allclose(layer.node_value_to_position[name], xy):
                    reached.add(layer.node_value_to_symbol[name])
            if not msg.success:
                self.record_visits(mission.state, reached)
        self.feedback(msg.robot_name, msg.plan_id, mission.next_action, msg.action_type,
                      msg.success, f"{msg.action_type} failed", reached)

    def feedback(self, robot, plan_id, index, kind, success, reason, reached=None):
        mission = self.missions.get(robot)
        if (mission is None or mission.finished or plan_id != mission.plan_id
                or index != mission.next_action or index >= len(mission.actions)):
            return
        action = mission.actions[index]
        if kind != ACTION_TYPES[action[0]]:
            return
        if not success:
            self.invalidate(mission, reason)
            return

        state = mission.state
        if kind == "FOLLOW":
            self.record_visits(state, set(mission.paths[index]) if reached is None else reached)
        elif kind == "GAZE":
            state.inspected.add(action[1])
        elif kind == "PICK":
            state.holding = action[1]
        elif kind == "PLACE":
            state.holding = None
            state.placements[action[1]] = action[2]
        mission.next_action += 1
        if mission.next_action == len(mission.actions):
            mission.finished = True


    def record_visits(self, state, reached):
        if not reached:
            return
        graph = self.node.dsg_last
        state.visited_places.update(reached)
        containment, _ = effective_scene(graph, state)
        state.visited_objects.update(obj for _, obj, place in containment if place in reached)
        state.visited_regions.update(region for _, place, region in generate_place_containment(graph)
                                     if place in reached)
