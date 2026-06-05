import logging
import os
import signal
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from dsg_pddl.core.models import GroundedPddlProblem
from dsg_pddl.core.parsing import lisp_string_to_ast

logger = logging.getLogger(__name__)


def get_solver_env_config():
    optimal_timeout = float(os.getenv("PDDL_OPTIMAL_TIMEOUT", "10"))
    suboptimal_timeout = float(os.getenv("PDDL_SUBOPTIMAL_TIMEOUT", "60"))

    optimal_solver = os.getenv("PDDL_OPTIMAL_SOLVER", "astar_ff").lower()
    suboptimal_solver = os.getenv("PDDL_SUBOPTIMAL_SOLVER", "lazy_ff").lower()

    return optimal_solver, optimal_timeout, suboptimal_solver, suboptimal_timeout


def _optimal_solvers(timeout: float) -> dict[str, str | None]:
    return {
        "none": None,
        "lmcut": f"astar(lmcut(), max_time={timeout})",
        "astar_ff": f"astar(ff(), max_time={timeout})",
    }


def _suboptimal_solvers(timeout: float) -> dict[str, str | None]:
    return {
        "none": None,
        "lazy_ff": (
            f"let(hff, ff(), "
            f"lazy_greedy([hff], preferred=[hff], max_time={timeout}))"
        ),
        "wastar_ff": (
            f"let(hff, ff(), "
            f"eager_wastar([hff], preferred=[hff], w=2, max_time={timeout}))"
        ),
    }


def _get_solver_config(kind: str, selected: str, timeout: float) -> str | None:
    solvers = (
        _optimal_solvers(timeout)
        if kind == "optimal"
        else _suboptimal_solvers(timeout)
    )

    if selected not in solvers:
        valid = ", ".join(sorted(solvers))
        raise ValueError(
            f"Invalid PDDL_{kind.upper()}_SOLVER={selected!r}. "
            f"Valid options are: {valid}"
        )

    return solvers[selected]


def _run_fd(problem, domain, search_cmd, timeout, result_container, key):
    """Run Fast Downward in a separate process."""

    with tempfile.TemporaryDirectory() as tmpdirname:
        problem_fn = os.path.join(tmpdirname, "problem.pddl")
        domain_fn = os.path.join(tmpdirname, "domain.pddl")
        plan_fn = os.path.join(tmpdirname, "plan.txt")

        with open(problem_fn, "w") as fo:
            fo.write(problem.problem_str)

        with open(domain_fn, "w") as fo:
            fo.write(domain.to_string())

        command = [
            "fast-downward",
            "--plan-file",
            plan_fn,
            domain_fn,
            problem_fn,
            "--search",
            search_cmd,
        ]

        try:
            proc = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=tmpdirname,
                start_new_session=True,
            )

            stdout, stderr = proc.communicate(timeout=timeout)

            logger.debug(f"{key} stdout: {stdout}")
            if proc.returncode == 0 and os.path.exists(plan_fn):
                with open(plan_fn, "r") as f:
                    result_container[key] = f.readlines()
            else:
                result_container[key] = None
                if stderr:
                    logger.debug(f"{key} stderr: {stderr}")

        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            result_container[key] = None
            logger.debug(f"{key} timed out")


def solve_pddl(problem: GroundedPddlProblem):
    """Configurable optimal + suboptimal PDDL solving."""

    # -----------------------
    # Define planners
    # -----------------------
    (pddl_optimal_solver, optimal_timeout,
     pddl_suboptimal_solver, suboptimal_timeout) = get_solver_env_config()

    optimal_search = _get_solver_config(
        "optimal",
        pddl_optimal_solver,
        optimal_timeout,
    )

    suboptimal_search = _get_solver_config(
        "suboptimal",
        pddl_suboptimal_solver,
        suboptimal_timeout,
    )

    # -----------------------
    # Threads
    # -----------------------
    results = {}
    threads = []

    if optimal_search is not None:
        threads.append(
            threading.Thread(
                target=_run_fd,
                args=(
                    problem,
                    problem.domain,
                    optimal_search,
                    optimal_timeout,
                    results,
                    "optimal",
                ),
            )
        )

    if suboptimal_search is not None:
        threads.append(
            threading.Thread(
                target=_run_fd,
                args=(
                    problem,
                    problem.domain,
                    suboptimal_search,
                    suboptimal_timeout,
                    results,
                    "suboptimal",
                ),
            )
        )

    if not threads:
        logger.warning("No PDDL solvers enabled.")
        chosen = []
    else:
        start = time.time()

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        elapsed = time.time() - start
        logger.debug(f"PDDL solving finished in {elapsed:.2f}s")

        if results.get("optimal"):
            logger.debug("Returning OPTIMAL plan")
            chosen = results["optimal"]
        elif results.get("suboptimal"):
            logger.debug("Returning SUBOPTIMAL plan")
            chosen = results["suboptimal"]
        else:
            logger.warning("Planning failed.")
            chosen = []

    # -----------------------
    # Debug output
    # -----------------------
    debug_output_dir = os.getenv("DEBUG_OUTPUT_DIR")

    if debug_output_dir:
        debug_dir = Path(debug_output_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        (debug_dir / "problem.pddl").write_text(problem.problem_str)
        (debug_dir / "domain.pddl").write_text(problem.domain.to_string())
        (debug_dir / "plan.txt").write_text("".join(chosen))

    # -----------------------
    # Parse plan
    # -----------------------
    plan = [lisp_string_to_ast(line) for line in chosen[:-1]]
    return plan
