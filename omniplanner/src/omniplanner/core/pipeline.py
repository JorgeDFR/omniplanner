import logging
from typing import Any, overload

from plum import dispatch

from omniplanner.core.api import PlanRequest
from omniplanner.core.context import DsgContextProvider
from omniplanner.core.dispatch import (
    ensure_domain_dispatch_registered,
    ground_problem,
    make_plan,
)
from omniplanner.core.wrappers import SymbolicContext

logger = logging.getLogger(__name__)


@overload
@dispatch
def full_planning_pipeline(plan_request: PlanRequest, map_context: Any, feedback=None):
    ensure_domain_dispatch_registered(plan_request.domain)
    grounded_problem = ground_problem(
        plan_request.domain,
        map_context,
        plan_request.robot_states,
        plan_request.goal,
        feedback,
    )
    logger.debug("Grounded Problem")

    # TODO: it would be nice if we could incorporate additional symbol context
    # added during grounding...

    dsg_context = DsgContextProvider(map_context)
    contextualized_problem = SymbolicContext(dsg_context, grounded_problem)
    plan = make_plan(contextualized_problem, map_context)
    logger.debug(f"Made plan {plan}")
    return plan


@dispatch
def full_planning_pipeline(
    plan_requests: list[PlanRequest], map_context: Any, feedback=None
):
    return [full_planning_pipeline(pr, map_context, feedback) for pr in plan_requests]
