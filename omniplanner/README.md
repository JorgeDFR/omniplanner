# Omniplanner

Omniplanner is a package for constructing planning requests,
grounding them against map or scene context, producing abstract plans, and
compiling those plans into robot-facing action sequences.

The package is built around runtime multiple dispatch. Planning domains register
implementations of `ground_problem` and `make_plan`; compilers register
implementations of `compile_plan`. At runtime, Omniplanner selects the
implementation that matches the domain, goal, map context, and intermediate plan
types.

## Package Layout

```text
src/
  omniplanner/
    core/              # public planning API, dispatch, wrappers, pipeline
    domains/           # built-in non-PDDL domains: goto-points, TSP, language
    compile_plan.py    # generic plan compilation and collection dispatch
    functor.py         # fmap support for wrappers and collections
    utils.py           # small shared utilities
  dsg_pddl/
    core/              # PDDL models, parsing, domain inspection
    grounding/         # DSG-to-PDDL grounding logic
    planning/          # PDDL solving and plan parameterization
    domains/           # bundled PDDL domain files
tests/                 # unit tests
examples/              # runnable demos and isolated experiments
```

## Core API

Most callers interact with the package through `PlanRequest` and
`full_planning_pipeline`:

```python
from omniplanner.core import PlanRequest, full_planning_pipeline

request = PlanRequest(
    domain=domain,
    goal=goal,
    robot_states=robot_states,
)

plan = full_planning_pipeline(request, map_context)
```

The pipeline performs:

1. domain-specific grounding with `ground_problem`
2. domain-specific planning with `make_plan`
3. optional context and robot wrapping via `SymbolicContext`, `RobotWrapper`, or
   `MultiRobotWrapper`

Plan compilation is a separate step:

```python
from omniplanner.compile_plan import collect_plans, compile_plan

compiled = compile_plan(adaptors, "map", plan)
plans_by_robot = collect_plans(compiled)
```

## Built-In Domains

Omniplanner currently includes:

- `GotoPointsDomain`: visits an explicit sequence of points or DSG symbols.
- `TspDomain`: orders goals with a TSP-style solver before producing a path.
- `LanguageDomain`: turns language goals into another supported domain, either
  directly for goto-points or through an LLM-generated PDDL goal.
- `PddlDomain` and `MultiRobotPddlDomain` from `dsg_pddl`: ground DSG context to
  PDDL, solve with Fast Downward, and parameterize symbolic actions.

## Examples

Runnable examples live directly in `examples/`:

```text
examples/goto_points_example.py
examples/tsp_example.py
examples/language_example.py
examples/pddl_example.py
examples/pddl_example_multirobot.py
```

Experimental workflows are isolated under `examples/experiments/`:

```text
examples/experiments/pddl_improved/
examples/experiments/pddlstream/
```

Run examples from the package directory after installing the package
dependencies:

```bash
python examples/goto_points_example.py
python examples/tsp_example.py
python examples/pddl_example_multirobot.py
```

The language example requires a valid API key configured through the environment
variable named in `examples/resources/llm_config.yaml`.

## Development

Create a local virtual environment, then install the package in editable
mode:

```bash
python -m venv .venv
.venv/bin/python -m pip install -e .
```

Run the unit tests with:

```bash
.venv/bin/python -m pytest
```

The current suite covers the core wrappers, dispatch pipeline, built-in domain
grounding/planning behavior, PDDL parsing and grounding units, TSP utilities,
and plan compilation recursion.

## Extending Omniplanner

To add a new planning domain:

1. Define domain, goal, grounded problem, and plan types.
2. Register a `ground_problem` implementation for your domain and goal.
3. Register a `make_plan` implementation for the grounded problem type.
4. Optionally register `compile_plan` implementations for robot-specific action
   output.

Example shape:

```python
from plum import dispatch

from omniplanner.core import PlanningDomain, PlanningGoal, ground_problem, make_plan


class MyDomain(PlanningDomain):
    pass


class MyGoal(PlanningGoal):
    pass


@dispatch
def ground_problem(domain: MyDomain, map_context, robot_states: dict, goal: MyGoal, feedback=None):
    return grounded_problem


@dispatch
def make_plan(grounded_problem: MyGroundedProblem, map_context):
    return plan
```
