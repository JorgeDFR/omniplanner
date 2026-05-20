
import os
import time
import signal
import logging
import tempfile
import threading
import subprocess

from dsg_pddl.pddl_grounding import GroundedPddlProblem
from dsg_pddl.pddl_utils import lisp_string_to_ast

logger = logging.getLogger(__name__)


OPTIMAL_TIMEOUT = float(os.getenv("PDDL_OPTIMAL_TIMEOUT", "10"))
SUBOPTIMAL_TIMEOUT = float(os.getenv("PDDL_SUBOPTIMAL_TIMEOUT", "60"))


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
            "--plan-file", plan_fn,
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
    """Parallel optimal + suboptimal PDDL solving (race strategy)."""

    # -----------------------
    # Define planners
    # -----------------------
    #optimal_search = f"astar(lmcut(), max_time={OPTIMAL_TIMEOUT})"
    optimal_search = f"astar(ff(), max_time={OPTIMAL_TIMEOUT})"
    #suboptimal_search = "let(hff, ff(), eager_wastar([hff], preferred=[hff], w=2, max_time={SUBOPTIMAL_TIMEOUT}))"
    suboptimal_search = f"let(hff, ff(), lazy_greedy([hff], preferred=[hff], max_time={SUBOPTIMAL_TIMEOUT}))"

    # -----------------------
    # Threads
    # -----------------------
    results = {}

    t_opt = threading.Thread(
        target=_run_fd,
        args=(problem, problem.domain, optimal_search,
              OPTIMAL_TIMEOUT, results, "optimal"),
    )

    t_sub = threading.Thread(
        target=_run_fd,
        args=(problem, problem.domain, suboptimal_search,
              SUBOPTIMAL_TIMEOUT, results, "suboptimal"),
    )

    start = time.time()

    t_opt.start()
    t_sub.start()

    t_opt.join()
    t_sub.join()

    elapsed = time.time() - start
    logger.debug(f"PDDL solving finished in {elapsed:.2f}s")

    # -----------------------
    # Selection logic
    # -----------------------
    chosen = None

    if results.get("optimal"):
        logger.debug("Returning OPTIMAL plan")
        chosen = results["optimal"]
    elif results.get("suboptimal"):
        logger.debug("Returning SUBOPTIMAL plan")
        chosen = results["suboptimal"]
    else:
        logger.warning(f"Planning failed.")
        chosen = []

    # -----------------------
    # Debug output
    # -----------------------
    debug_output_dir = os.getenv("DEBUG_OUTPUT_DIR", "")
    debug_problem_fn = os.path.join(debug_output_dir, "problem.pddl")
    debug_domain_fn = os.path.join(debug_output_dir, "domain.pddl")
    debug_plan_fn = os.path.join(debug_output_dir, "plan.txt")

    with open(debug_problem_fn, "w") as fo:
        fo.write(problem.problem_str)

    with open(debug_domain_fn, "w") as fo:
        fo.write(problem.domain.to_string())

    with open(debug_plan_fn, "w") as fo:
        fo.writelines(chosen)

    # -----------------------
    # Parse plan
    # -----------------------
    plan = [lisp_string_to_ast(line) for line in chosen[:-1]]
    return plan