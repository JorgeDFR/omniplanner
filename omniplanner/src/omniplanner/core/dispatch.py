from typing import Any, overload

from plum import dispatch

from omniplanner.core.api import GroundedProblem, Plan, PlanningDomain, PlanningGoal
from omniplanner.core.wrappers import fmap
from omniplanner.functor import Functor


class DispatchException(Exception):
    def __init__(self, function_name, *objects):
        arg_type_string = ", ".join(map(lambda x: x.__name__, map(type, objects)))
        super().__init__(
            f"No matching specialization for {function_name}({arg_type_string})"
        )


def register_builtin_domains():
    import dsg_pddl.grounding.improved_region  # noqa: F401
    import dsg_pddl.grounding.legacy  # noqa: F401
    import dsg_pddl.grounding.multirobot  # noqa: F401


def ensure_domain_dispatch_registered(domain):
    try:
        from dsg_pddl.core.models import MultiRobotPddlDomain, PddlDomain
    except Exception:
        return

    if isinstance(domain, (PddlDomain, MultiRobotPddlDomain)):
        register_builtin_domains()


@dispatch
def ground_problem(
    domain: PlanningDomain,
    map_context: Any,
    intial_state: Any,
    goal: PlanningGoal,
    feedback: Any = None,
) -> GroundedProblem:
    raise DispatchException(ground_problem, domain, map_context, goal, feedback)


@overload
@dispatch
def make_plan(grounded_problem: GroundedProblem, map_context: Any) -> Plan:
    raise DispatchException(make_plan, grounded_problem, map_context)


@dispatch
def make_plan(grounded_problem: Functor, map_context: Any):
    return fmap(lambda e: make_plan(e, map_context), grounded_problem)
