from dataclasses import dataclass

import numpy as np

from dsg_pddl.planning.parameterization import PddlPlan
from omniplanner.compile_plan import collect_plans, compile_plan
from omniplanner.domains.goto_points import GotoPointsPlan
from omniplanner.core import MultiRobotWrapper, RobotWrapper, SymbolicContext
from omniplanner.domains.tsp import FollowPathPlan


@dataclass
class SimpleAction:
    name: str
    parameters: dict


@dataclass
class SimpleActionSequence:
    robot_name: str
    frame: str
    actions: list[SimpleAction]


@collect_plans.dispatch
def collect_plans(plan: SimpleActionSequence):
    return {plan.robot_name: plan}


def _robot_name(adaptor):
    return getattr(adaptor, "name", getattr(adaptor, "robot_name", str(adaptor)))


@compile_plan.dispatch
def compile_plan(adaptor, plan_frame: str, plan: GotoPointsPlan):
    actions = [
        SimpleAction("goto-point", {"start": step.start, "goal": step.goal})
        for step in plan.plan
    ]
    return SimpleActionSequence(_robot_name(adaptor), plan_frame, actions)


@compile_plan.dispatch
def compile_plan(adaptor, plan_frame: str, plan: FollowPathPlan):
    actions = [
        SimpleAction("follow-path", {"path": np.asarray(step.path)})
        for step in plan.steps
    ]
    return SimpleActionSequence(_robot_name(adaptor), plan_frame, actions)


@compile_plan.dispatch
def compile_plan(adaptor, plan_frame: str, plan: PddlPlan):
    return compile_plan(adaptor, plan_frame, SymbolicContext({}, plan))


@compile_plan.dispatch
def compile_plan(
    adaptor, plan_frame: str, contextualized_plan: SymbolicContext[PddlPlan]
):
    plan = contextualized_plan.value
    actions = [
        SimpleAction(
            name=symbolic[0],
            parameters={"symbolic": symbolic, "path": parameters},
        )
        for symbolic, parameters in zip(
            plan.symbolic_actions, plan.parameterized_actions
        )
    ]
    return SimpleActionSequence(_robot_name(adaptor), plan_frame, actions)


@compile_plan.dispatch
def compile_plan(
    adaptors: dict,
    plan_frame: str,
    wrapper: MultiRobotWrapper[SymbolicContext[PddlPlan]],
):
    if not isinstance(wrapper.value, SymbolicContext) or not isinstance(
        wrapper.value.value, PddlPlan
    ):
        raise TypeError("Expected MultiRobotWrapper[SymbolicContext[PddlPlan]]")

    shared_plan = wrapper.value.value
    results = []
    for outer_name in wrapper.names:
        inner_name = wrapper.remap_name_to_inner(outer_name)
        robot_plan = PddlPlan(shared_plan.domain, [], [], shared_plan.symbols)
        for symbolic, parameters in zip(
            shared_plan.symbolic_actions, shared_plan.parameterized_actions
        ):
            if len(symbolic) > 1 and symbolic[1] == inner_name:
                robot_plan.symbolic_actions.append((symbolic[0],) + symbolic[2:])
                robot_plan.parameterized_actions.append(parameters)

        adaptor = adaptors[outer_name]
        results.append(
            RobotWrapper(
                outer_name,
                compile_plan(
                    adaptor,
                    plan_frame,
                    SymbolicContext(wrapper.value.context, robot_plan),
                ),
            )
        )

    return results
