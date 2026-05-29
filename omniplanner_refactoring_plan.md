# Improved Omniplanner Refactoring Plan

This plan supersedes the first generated refactoring plan. It incorporates a critical review of the package analysis, the root `README.md`, the `omniplanner_ros` consumers, examples, tests, and packaging metadata.

## Corrections To The First Plan

The original plan was directionally useful, but several assumptions were too aggressive or incomplete.

- `dsg_pddl` should not be treated as an internal implementation detail yet. `omniplanner_ros`, examples, tests, and likely downstream code import `dsg_pddl.*` directly and load PDDL assets with `importlib.resources.files(dsg_pddl.domains)`.
- Moving PDDL code under `omniplanner.domains.pddl` is a high-risk later option, not the default first refactor. The safer structure is to keep `dsg_pddl` as a public sibling package and improve its internal boundaries.
- The root `README.md` is more authoritative than `omniplanner/README.md`; it documents the intended architecture, plugin model, multi-dispatch extension mechanism, ROS plugin integration, and known instability.
- `omniplanner_ros` is not incidental. It is the main runtime integration point and constrains import paths, wrapper shapes, compile-plan dispatch behavior, PDDL asset paths, and plugin configuration names.
- `spark-config` is not unused at repository level. It is heavily used by `omniplanner_ros`; it is only unused by the non-ROS `omniplanner` package files inspected earlier.
- Several dependencies are outside `omniplanner/pyproject.toml` because they belong to ROS or examples: `rclpy`, ROS messages, `tf2_ros`, `robot_executor_interface`, `hydra_msgs`, `matplotlib`, and optional `pddlstream`.
- Renaming `compile_plan.py` to `compile.py` is not worth doing early. `compile_plan` is a clear public concept and is imported by ROS plugins.
- Populating `omniplanner.__init__` with broad exports may create dispatch/import side effects. A minimal compatibility-focused API is safer.

## Refactor Thesis

The right first refactor is not a package relocation. It is a boundary clarification while preserving public import paths.

Keep the current two-package runtime shape:

```text
omniplanner/src/
  omniplanner/   # orchestration, wrappers, built-in non-PDDL domains, compile dispatch
  dsg_pddl/      # PDDL models, PDDL assets, DSG-to-PDDL grounding, Fast Downward solving
```

Within that shape, split oversized modules and centralize shared logic. Only after import compatibility, ROS integration, and downstream usage are understood should the project consider moving `dsg_pddl` under `omniplanner`.

## Goals

- Preserve existing behavior and import paths used by `omniplanner_ros`, examples, tests, and downstream plugins.
- Make the non-ROS planning interfaces clearer: `PlanRequest`, `ground_problem`, `make_plan`, wrappers, symbolic context, and `compile_plan`.
- Make dispatch registration explicit enough to test and reason about, without replacing Plum in the first phase.
- Reduce duplication in DSG/PDDL symbol handling, layer access, containment facts, connectivity facts, and start-position handling.
- Isolate external integrations: Spark DSG access, Fast Downward execution, LLM request handling, and ROS-only compilation.
- Improve characterization tests before moving code.
- Separate behavior fixes from structural moves.

## Non-Goals

- Do not remove or rename public modules in the first refactor:
  - `omniplanner.omniplanner`
  - `omniplanner.compile_plan`
  - `omniplanner.goto_points`
  - `omniplanner.tsp`
  - `omniplanner.language_planner`
  - `dsg_pddl.pddl_grounding`
  - `dsg_pddl.pddl_utils`
  - `dsg_pddl.pddl_planning`
  - `dsg_pddl.dsg_pddl_*`
- Do not move PDDL asset files until old and new `importlib.resources` paths are both tested.
- Do not replace Plum dispatch yet. The root README explicitly identifies dispatch as an architectural choice.
- Do not rewrite the PDDL parser or domain renderer as part of the initial structure cleanup.
- Do not change ROS plugin names, Spark Config names, message contracts, or action compilation semantics.
- Do not silently change planner outputs, wrapper nesting, or robot-name mapping.

## Recommended Package Structure

### Phase 1 Target: Conservative In-Place Structure

Add internal modules while keeping existing modules as public facades:

```text
omniplanner/src/omniplanner/
  __init__.py
  omniplanner.py          # compatibility facade for core API
  core/
    __init__.py
    api.py                # PlanRequest, marker/protocol types
    dispatch.py           # ground_problem, make_plan, registration helpers
    pipeline.py           # full_planning_pipeline
    wrappers.py           # Wrapper, RobotWrapper, MultiRobotWrapper, SymbolicContext
    context.py            # DsgContextProvider, DsgNodeContext, symbol context lookup
    functor.py            # optional eventual home for current functor helpers
  compile_plan.py         # keep public dispatch object here, may delegate internally
  domains/
    __init__.py
    goto_points.py        # implementation, re-exported by omniplanner.goto_points
    tsp.py                # implementation, re-exported by omniplanner.tsp
    language.py           # implementation, re-exported by omniplanner.language_planner
  goto_points.py          # compatibility facade
  tsp.py                  # compatibility facade
  language_planner.py     # compatibility facade
  utils.py

omniplanner/src/dsg_pddl/
  __init__.py
  pddl_grounding.py       # compatibility facade for models/domain parsing
  pddl_utils.py           # compatibility facade for parsing/symbol helpers
  pddl_planning.py        # compatibility facade for Fast Downward solver
  dsg_pddl_planning.py    # compatibility facade for PDDL plan parameterization
  dsg_pddl_grounding.py   # compatibility facade for legacy single-robot grounding
  dsg_pddl_grounding_improved.py
  dsg_pddl_grounding_multirobot.py
  core/
    models.py             # PddlSymbol, PddlProblem, PddlGoal, PddlDomain
    parsing.py            # Lisp/PDDL AST parse/render helpers
    domain_inspection.py  # get_domain_* helpers and validation
  grounding/
    dsg_access.py         # layer lookup, node/symbol conversion
    symbols.py            # normalization and PddlSymbol construction
    connectivity.py       # edge/distance/init connectivity facts
    containment.py        # object/place/region containment facts
    legacy.py             # current dsg_pddl_grounding domain routing
    improved_region.py    # current improved region algorithms
    multirobot.py         # current multi-robot algorithms
  planning/
    solver.py             # Fast Downward process wrapper
    parameterization.py   # symbolic PDDL action to geometry/path parameters
  domains/
    *.pddl                # keep asset location for now
```

This is intentionally less disruptive than nesting all PDDL code under `omniplanner`.

### Phase 2 Optional Structure

After compatibility and downstream usage are known, consider a single namespace:

```text
omniplanner.domains.pddl
```

Only pursue this if the project wants a cleaner long-term API and can maintain `dsg_pddl` as a shim for at least one release cycle.

## Architectural Decisions

### Keep Plum, But Make Registration Auditable

Plum dispatch is central to the intended design. The problem is not Plum itself; it is hidden import-side-effect registration.

Plan:

- Keep `ground_problem`, `make_plan`, and `compile_plan` as public dispatch functions.
- Add `register_builtin_domains()` and `register_builtin_compilers()` helpers that import known built-ins.
- Keep automatic registration in `full_planning_pipeline` for backward compatibility.
- Add tests that prove a fresh process can import and run built-in domains without relying on accidental prior imports.

### Treat `compile_plan` As A Public Extension Point

ROS plugins register plan compilers with `@compile_plan.dispatch`. The dispatch object identity matters.

Plan:

- Do not rename `compile_plan.py` early.
- If internals are extracted, keep the public `compile_plan` function object in `omniplanner.compile_plan`.
- Add tests where external modules register a compiler after import and the generic wrapper recursion still finds it.

### Keep `dsg_pddl` Public For Now

`dsg_pddl` is not merely a helper folder. ROS planners load assets and types from it directly.

Plan:

- Split internals beneath `dsg_pddl/core`, `dsg_pddl/grounding`, and `dsg_pddl/planning`.
- Keep old modules importing/re-exporting from the new internals.
- Preserve `dsg_pddl.domains` package data paths.

### Separate ROS Constraints From Non-ROS Core

The non-ROS package should remain usable without importing ROS packages. ROS compilers and plugin adapters should stay in `omniplanner_ros`.

Plan:

- Do not move ROS-specific `compile_plan` overloads into `omniplanner`.
- Keep package dependencies for non-ROS core separate from ROS dependencies.
- Document which compilers live in examples, which live in ROS plugins, and which are package defaults.

## Concrete Refactoring Plan

### Step 0: Baseline And Characterization

No production code moves yet.

Add tests for:

- Current public imports used by examples and ROS:
  - `omniplanner.omniplanner`
  - `omniplanner.compile_plan`
  - `omniplanner.goto_points`
  - `omniplanner.tsp`
  - `omniplanner.language_planner`
  - `dsg_pddl.pddl_grounding`
  - `dsg_pddl.dsg_pddl_planning`
  - `dsg_pddl.domains`
- `compile_plan` external registration from a separate temporary module.
- `full_planning_pipeline` with fresh imports for built-in domains.
- Wrapper shape produced by single-robot, list-of-requests, language-to-PDDL, and multi-robot PDDL flows.
- `importlib.resources.files(dsg_pddl.domains).joinpath(...).pddl`.
- `PDDL_OPTIMAL_TIMEOUT`, `PDDL_SUBOPTIMAL_TIMEOUT`, `DEBUG_OUTPUT_DIR`, and `PDDL_DUMP_DIR` behavior.

Run:

```bash
cd omniplanner
pytest
```

If feasible, also add an import-only smoke test for `omniplanner_ros` modules in an environment with ROS dependencies installed. In the current non-ROS environment, keep ROS tests optional.

### Step 1: Extract Core Wrappers Without Changing Public Imports

Move implementation from `omniplanner.omniplanner` into `omniplanner/core/wrappers.py`:

- `Wrapper`
- `RobotWrapper`
- `MultiRobotWrapper`
- `SymbolicContext`
- `extract`
- `with_new_value`
- `fmap`
- `push`

Keep `omniplanner.omniplanner` re-exporting the same names.

Risk:

- Plum parametric dispatch may bind to classes by identity. Moving classes changes identities if done naively.

Mitigation:

- Move classes once and import them back from `omniplanner.omniplanner`; do not create duplicate class definitions.
- Run wrapper and compile tests immediately.

### Step 2: Extract Context Provider

Move to `omniplanner/core/context.py`:

- `string_as_nodesymbol`
- `DsgNodeContext`
- `DsgContextProvider`

Keep re-exports in `omniplanner.omniplanner`.

Add tests for:

- DSG-derived `position` and `semantic_label`.
- User context merging.
- Refusing to override DSG-provided keys.
- Non-DSG map contexts if pipeline is called with arrays or test doubles.

### Step 3: Extract API And Pipeline

Move to `omniplanner/core/api.py`:

- `PlanningDomain`
- `PlanningGoal`
- `ExecutionInterface`
- `PlanRequest`
- `GroundedProblem`
- `Plan`

Move to `omniplanner/core/dispatch.py`:

- `DispatchException`
- `ground_problem`
- `make_plan`
- `ensure_domain_dispatch_registered`
- new `register_builtin_domains`

Move to `omniplanner/core/pipeline.py`:

- `full_planning_pipeline`

Keep `omniplanner.omniplanner` as a compatibility facade.

Important compatibility rule:

- The public dispatch functions must remain the same objects seen by built-in domains. If this becomes awkward, leave the dispatch functions defined in `omniplanner.omniplanner` and only extract helper logic around them.

### Step 4: Move Built-In Non-PDDL Domains Behind Facades

Move implementations:

- `omniplanner.goto_points` implementation to `omniplanner/domains/goto_points.py`.
- `omniplanner.tsp` implementation to `omniplanner/domains/tsp.py`.
- `omniplanner.language_planner` implementation to `omniplanner/domains/language.py`.

Keep old files as facades.

Do not change:

- `GotoPointsDomain`, `GotoPointsGoal`, `GotoPointsPlan` constructor behavior.
- `TspDomain`, `TspGoal`, `FollowPathPlan` constructor behavior.
- `LanguageDomain("Pddl")` and `LanguageDomain("goto_points")` string behavior.

Behavior fixes such as removing `time.sleep(3)` or changing LLM response parsing should be separate PRs.

### Step 5: Clean PDDL Models And Parsing In Place

Create:

- `dsg_pddl/core/models.py`
- `dsg_pddl/core/parsing.py`
- `dsg_pddl/core/domain_inspection.py`

Move without behavior changes:

- `PddlSymbol`, `PddlProblem`, `PddlGoal`, `PddlDomain`, `MultiRobotPddlDomain`, `GroundedPddlProblem`.
- `tokenize_lisp`, `get_lisp_ast`, `lisp_string_to_ast`, `ast_to_string`.
- `ensure_pddl_domain`, `get_domain_name`, `get_domain_requirements`, `get_domain_types`, `get_domain_predicates`, `get_functions`, `get_derived`, `get_actions`.

Keep:

- `dsg_pddl.pddl_grounding` re-exporting model/domain names.
- `dsg_pddl.pddl_utils` re-exporting parsing utilities and symbol-char conversion.

Add golden tests for PDDL string output before and after the move.

### Step 6: Extract Fast Downward Solver Boundary

Create `dsg_pddl/planning/solver.py`.

Move:

- `_run_fd`
- `solve_pddl`
- timeout constants or a new internal config reader

Keep `dsg_pddl.pddl_planning.solve_pddl` available.

Do not change default behavior yet, even though there are issues:

- It waits for both solver threads.
- It writes debug files to `DEBUG_OUTPUT_DIR` or the current directory.
- It returns an empty plan when Fast Downward fails.

After the structural move, consider a behavior PR that introduces `FastDownwardConfig` and opt-in debug output.

### Step 7: Extract PDDL Plan Parameterization

Create `dsg_pddl/planning/parameterization.py`.

Move:

- `PddlPlan`
- `drop_index`
- `parameterize_*`
- `make_plan(GroundedPddlProblem, map_context)`

Keep `dsg_pddl.dsg_pddl_planning` as the public module.

Add tests for:

- Single-robot action parameterization.
- Multi-robot action projection assumptions.
- Unknown actions raising the same exception.
- `last_pose` behavior, even if imperfect.

### Step 8: Extract Shared DSG Grounding Utilities

Create:

- `dsg_pddl/grounding/dsg_access.py`
- `dsg_pddl/grounding/symbols.py`
- `dsg_pddl/grounding/connectivity.py`
- `dsg_pddl/grounding/containment.py`

Move shared helpers:

- `get_places_layer`
- `normalize_symbol`, `normalize_symbols`
- `add_symbol_positions`
- `extract_all_symbols`
- `generate_objects`
- `explicit_edges_from_layer`
- `implicit_edges_from_layers`
- `symbol_connectivity_to_pddl`
- `generate_object_containment`
- `generate_place_containment`

Keep original import paths from `dsg_pddl.dsg_pddl_grounding`.

This is where duplication should start decreasing, but avoid changing thresholds, edge direction, rounding, or symbol names.

### Step 9: Split Domain-Specific Grounding

Create:

- `dsg_pddl/grounding/legacy.py`
- `dsg_pddl/grounding/improved_region.py`
- `dsg_pddl/grounding/multirobot.py`

Move implementation gradually from:

- `dsg_pddl.dsg_pddl_grounding`
- `dsg_pddl.dsg_pddl_grounding_improved`
- `dsg_pddl.dsg_pddl_grounding_multirobot`

Keep old modules as dispatch-registration facades. They should import and register the same dispatch overloads.

Add a domain-name constants module, for example `dsg_pddl/domain_names.py`, containing:

- `GOTO_OBJECT_DOMAIN`
- `OBJECT_REARRANGEMENT_DOMAIN`
- `REGION_OBJECT_REARRANGEMENT_DOMAIN`
- `REGION_REARRANGEMENT_DERIVED_DOMAIN`
- `REGION_REARRANGEMENT_EXPLICIT_STATE_DOMAIN`
- `REGION_REARRANGEMENT_MULTIROBOT_FD_DOMAIN`

### Step 10: Revisit Side Effects And Behavior Fixes

Only after structural moves and characterization tests:

- Remove or gate `GotoPointsDomain.make_plan` sleep.
- Fix `GotoPointsPlan.append`.
- Add empty-goal handling for goto-points and TSP.
- Replace Python-literal LLM response parsing with JSON or a typed adapter, if accepted.
- Make Fast Downward debug output opt-in or explicit.
- Make multi-robot PDDL dumps opt-in.
- Revisit `last_pose` initialization in PDDL parameterization.
- Decide whether `PddlSymbol` equality should include layer or remain name-only.
- Decide canonical symbol formats and enforce them through one conversion module.

## Alternative Refactoring Approaches

### Option A: Conservative In-Place Modularization

This is the recommended approach.

Pros:

- Lowest risk to ROS and downstream imports.
- Keeps `dsg_pddl.domains` assets stable.
- Enables tests and cleanup without API churn.

Cons:

- Keeps two top-level packages.
- Some compatibility facades will remain for a while.

### Option B: Single Namespace Under `omniplanner`

Move `dsg_pddl` under `omniplanner.domains.pddl`.

Pros:

- Cleaner conceptual namespace.
- Easier long-term documentation.

Cons:

- High migration cost.
- Breaks direct `dsg_pddl` consumers unless extensive shims are maintained.
- Asset paths need careful migration.

Use only after Option A and after downstream import usage is known.

### Option C: Registry-Based Architecture Instead Of Plum

Replace Plum dispatch with explicit registries for grounding, planning, and compiling.

Pros:

- More inspectable and easier to document.
- Better error messages and plugin discovery.

Cons:

- Changes the core extension model documented in the README.
- Significant behavior and plugin migration risk.

Treat as a future design exploration, not this refactor.

## Dependencies And Integration Boundaries

### Non-ROS Package Dependencies

Declared in `omniplanner/pyproject.toml`:

- `numpy<2`
- `networkx`
- `parse`
- `plum-dispatch`
- `ruamel.yaml`
- `spark-dsg`
- `nlu-interface`
- `spark-config`

Review:

- `ruamel.yaml` is used by examples and ROS language config, not core non-ROS package logic.
- `spark-config` is used by `omniplanner_ros`, not the inspected non-ROS package.
- `nlu-interface` is only needed for `language_planner`.

Future packaging improvement:

- Split optional extras: `language`, `ros`, `examples`, `dev`, and possibly `pddlstream`.

### External Executables

- `fast-downward` is required for real PDDL solving.

Plan:

- Add a preflight helper that checks executable availability.
- Keep current empty-plan fallback until behavior change is approved.

### ROS Integration Dependencies

Used by `omniplanner_ros`:

- `rclpy`
- `tf2_ros`
- `tf_transformations`
- `nav_msgs`, `geometry_msgs`, `visualization_msgs`, `std_msgs`
- `omniplanner_msgs`, `robot_executor_msgs`, `hydra_msgs`
- `robot_executor_interface`, `robot_executor_interface_ros`
- `ros_system_monitor_msgs`
- `spark_config`

These should remain outside the core non-ROS package.

### Example-Only Dependencies

Examples use:

- `matplotlib`
- `pddlstream`
- possibly OpenAI-backed `nlu-interface` resources.

Do not promote these to core dependencies unless examples become tested package features.

## Backward Compatibility Contract

The following must continue to work through the refactor:

```python
from omniplanner.omniplanner import PlanRequest, full_planning_pipeline
from omniplanner.omniplanner import RobotWrapper, MultiRobotWrapper, SymbolicContext
from omniplanner.omniplanner import fmap, with_new_value
from omniplanner.compile_plan import compile_plan, collect_plans
from omniplanner.goto_points import GotoPointsDomain, GotoPointsGoal, GotoPointsPlan
from omniplanner.tsp import TspDomain, TspGoal, FollowPathPlan, LayerPlanner
from omniplanner.language_planner import LanguageDomain, LanguageGoal
from dsg_pddl.pddl_grounding import PddlDomain, MultiRobotPddlDomain, PddlGoal
from dsg_pddl.dsg_pddl_planning import PddlPlan
import dsg_pddl.domains
```

Also preserve Spark Config plugin names in `omniplanner_ros`:

- `GotoPoints`
- `Tsp`
- `Pddl`
- `LanguagePlanner`
- `MultiRobotPddl`

## Risk Register

- Dispatch object identity breaks external `@compile_plan.dispatch` or `@ground_problem.dispatch` registrations.
  - Mitigation: characterize external registration and keep dispatch objects stable.

- Class identity changes break Plum parametric dispatch.
  - Mitigation: move definitions once, re-export names, avoid duplicate class definitions.

- `dsg_pddl` asset moves break `importlib.resources`.
  - Mitigation: do not move assets until old and new resource paths are tested.

- ROS modules break due to import path changes.
  - Mitigation: include ROS import paths in compatibility tests; keep facades.

- PDDL output formatting changes break Fast Downward.
  - Mitigation: golden tests for generated PDDL.

- Symbol normalization cleanup changes planning behavior.
  - Mitigation: centralize first, change semantics later.

- Solver side-effect cleanup changes debugging workflows.
  - Mitigation: add explicit config while preserving current defaults until agreed.

- Optional dependencies get imported by core package.
  - Mitigation: keep ROS, LLM, plotting, and PDDLStream imports behind integration modules or examples.

## Testing Strategy

Required before any moves:

- Existing unit tests under `omniplanner/tests`.
- New import compatibility tests.
- New dispatch registration tests.
- New package-data resource tests for PDDL domains.
- New compile extension tests.

Required after each structural step:

```bash
cd omniplanner
pytest
```

Recommended optional tests:

- ROS import smoke tests in a ROS-enabled environment.
- Example smoke tests with Fast Downward and LLM calls stubbed.
- Golden PDDL problem snapshots for each built-in PDDL domain family.

## Migration Strategy

1. Add tests and documentation for current public API.
2. Extract internals under `omniplanner/core` while keeping `omniplanner.omniplanner`.
3. Extract non-PDDL domains under `omniplanner/domains` while keeping old modules.
4. Extract `dsg_pddl` internals under `dsg_pddl/core`, `dsg_pddl/grounding`, and `dsg_pddl/planning`.
5. Update internal imports to new modules.
6. Keep examples and ROS imports on old paths until shims are proven stable.
7. Document new preferred imports.
8. Only then update examples and ROS code to preferred imports.
9. Defer deprecation warnings until downstream users have been notified.

## Open Decisions

- Is `dsg_pddl` a permanent public package or a compatibility namespace for a future `omniplanner.domains.pddl`?
- Should the official extension API remain Plum dispatch, or should a registry API be added alongside it?
- Which exact imports are used by downstream repositories beyond `omniplanner_ros`?
- Should built-in registration be automatic, explicit, or both?
- Should language planner output be Python literal, JSON, or a typed object?
- Should Fast Downward failure return an empty plan, raise an exception, or return a structured failure?
- Should debug PDDL files be written by default?
- What is the canonical DSG/PDDL symbol naming convention?
- Should multi-robot participants come from `robot_states`, the goal, or planner config?

## Recommended First Implementation Slice

The first implementation slice should be:

1. Add import compatibility tests for current `omniplanner`, `dsg_pddl`, and `omniplanner_ros` import paths where dependencies are available.
2. Add a test proving third-party compiler registration against `omniplanner.compile_plan.compile_plan`.
3. Add PDDL asset resource tests for `dsg_pddl.domains`.
4. Extract only wrapper definitions and helper dispatches into `omniplanner/core/wrappers.py`.
5. Re-export them from `omniplanner.omniplanner`.
6. Run the full non-ROS test suite.

This slice improves structure while avoiding PDDL, solver, ROS, and asset-path risk.
