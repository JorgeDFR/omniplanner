from dataclasses import dataclass


class PlanningDomain:
    pass


class PlanningGoal:
    pass


class ExecutionInterface:
    pass


@dataclass
class PlanRequest:
    domain: PlanningDomain
    goal: PlanningGoal
    robot_states: dict


@dataclass
class GroundedProblem:
    pass
    # initial_state: Any
    # goal_states: Any


@dataclass
class Plan:
    pass
