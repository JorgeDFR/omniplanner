# Omniplanner Package Overview

This document covers the Python package currently checked out under `omniplanner/src/omniplanner` and the closely coupled sibling package under `omniplanner/src/dsg_pddl`.

The user-requested path `omniplanner/omniplanner` does not exist in this repository. The installable package uses a `src` layout configured in `omniplanner/pyproject.toml`, with `omniplanner` and `dsg_pddl` discovered from `omniplanner/src`.

## Main Purpose

`omniplanner` is a planning orchestration package for robot planning pipelines over Dynamic Scene Graphs (DSGs), arrays of points, PDDL domains, and language-derived goals.

The core idea is:

1. A caller creates a `PlanRequest` with a planning domain, goal, and robot state map.
2. `full_planning_pipeline` dispatches to a domain-specific `ground_problem`.
3. The grounded problem is wrapped in symbolic DSG context.
4. `make_plan` dispatches to a problem-specific planner.
5. Optional `compile_plan` dispatch converts abstract plans into robot-executor-specific actions.

The package currently supports:

- Direct goto-point planning.
- TSP-style visit-order planning over DSG mesh-place paths.
- Single-robot PDDL planning over DSG symbols.
- Multi-robot PDDL planning over DSG symbols.
- Natural-language goals routed either to goto-points parsing or an LLM-backed PDDL goal generator.

## Current Structure

```text
omniplanner/
  pyproject.toml
  README.md
  src/
    omniplanner/
      __init__.py
      omniplanner.py
      functor.py
      compile_plan.py
      goto_points.py
      tsp.py
      language_planner.py
      utils.py
    dsg_pddl/
      __init__.py
      pddl_utils.py
      pddl_grounding.py
      pddl_planning.py
      dsg_pddl_grounding.py
      dsg_pddl_grounding_improved.py
      dsg_pddl_grounding_multirobot.py
      dsg_pddl_planning.py
      domains/
        GotoObjectDomain.pddl
        ObjectRearrangementDomain.pddl
        RegionObjectRearrangementDomain.pddl
        RegionObjectRearrangementDomain_DerivedPredicates.pddl
        RegionObjectRearrangementDomain_ExplicitState.pddl
        RegionObjectRearrangementDomain_MultiRobot_FD_Explore.pddl
  tests/
    test_core_planning.py
    test_tsp_and_layer.py
    test_pddl_units.py
  examples/
    goto_points_example.py
    tsp_example.py
    pddl_example.py
    language_example.py
    pddl_example_multirobot_FD_ominiplanner.py
    simple_compile_plan.py
    ...
```

`omniplanner/src/omniplanner/__init__.py` is empty, so public API exposure is currently by direct module imports rather than curated top-level exports.

## Packaging And Dependencies

`omniplanner/pyproject.toml` defines package name `omniplanner`, version `0.0.1`, Python `>=3.11`, and package discovery from `src`.

Runtime dependencies:

- `numpy<2`: coordinates, distances, arrays.
- `networkx`: graph shortest paths and graph filtering.
- `parse`: parsing DSG symbol strings such as `O(7)`.
- `plum-dispatch`: multiple dispatch for extension points.
- `ruamel.yaml`: used by examples/config loading.
- `spark-dsg`: Dynamic Scene Graph API and DSG-to-NetworkX conversion.
- `nlu-interface`: LLM interface used by `language_planner`.
- `spark-config`: listed dependency, not directly used in inspected package files.

External executable integration:

- `fast-downward` must be available on `PATH` for real PDDL solving in `dsg_pddl.pddl_planning.solve_pddl`.

Environment variables:

- `PDDL_OPTIMAL_TIMEOUT`: optimal Fast Downward search timeout, default `10`.
- `PDDL_SUBOPTIMAL_TIMEOUT`: suboptimal Fast Downward search timeout, default `60`.
- `DEBUG_OUTPUT_DIR`: where `solve_pddl` writes `problem.pddl`, `domain.pddl`, and `plan.txt`; default is empty string, meaning the current working directory.
- `PDDL_DUMP_DIR`: where multi-robot PDDL grounding writes persistent dumps; default is `pddl_dumps` under the current working directory.

## Core Module: `omniplanner.omniplanner`

Main abstractions:

- `PlanningDomain`: marker base class for planning domains.
- `PlanningGoal`: marker base class for planning goals.
- `ExecutionInterface`: empty marker, currently unused.
- `PlanRequest`: dataclass with `domain`, `goal`, and `robot_states`.
- `GroundedProblem`: empty base dataclass for grounded problems.
- `Plan`: empty base dataclass for plans.
- `DispatchException`: custom error for missing dispatch implementations.

Wrapper and context abstractions:

- `Wrapper`: marker base class for functor-style wrappers.
- `RobotWrapper[T]`: parametric dataclass containing one robot `name` and wrapped `value`.
- `MultiRobotWrapper[T]`: parametric dataclass containing robot `names`, wrapped `value`, and outer/inner robot-name remapping maps.
- `SymbolicContext[T]`: parametric dataclass containing a symbol `context` dictionary and wrapped `value`.
- `DsgContextProvider`: lazy dict-like context provider backed by a DSG. For keys that parse as DSG symbols, it can expose node `position` and `semantic_label`.
- `DsgNodeContext`: merged DSG-derived and user-provided context. It prevents overriding DSG-derived keys.

Dispatch extension points:

- `ground_problem(domain, map_context, robot_states, goal, feedback=None)`.
- `make_plan(grounded_problem, map_context)`.
- `full_planning_pipeline(plan_request, map_context, feedback=None)`.
- Wrapper helpers: `fmap`, `extract`, `with_new_value`, and `push`.

Important behavior:

- `full_planning_pipeline` calls `ensure_domain_dispatch_registered` before grounding. For `PddlDomain` and `MultiRobotPddlDomain`, this imports PDDL grounding modules so Plum dispatch registrations are loaded.
- Grounded problems are wrapped as `SymbolicContext(DsgContextProvider(map_context), grounded_problem)` before `make_plan`.
- `make_plan` has a generic functor overload that maps planning over wrappers and iterables.
- `push` swaps nested wrappers, for example `SymbolicContext[RobotWrapper[T]]` into `RobotWrapper[SymbolicContext[T]]`.

Fragile details:

- Several base classes are empty markers rather than protocols or abstract base classes.
- Dispatch registration depends on import side effects.
- `string_as_nodesymbol` assumes PDDL-style lower-case symbols and converts the first character with `pddl_char_to_dsg_char`.
- `ground_problem` has a typo in the parameter name `intial_state`.

## Functor Support: `omniplanner.functor`

`functor.py` provides reusable Plum-compatible functor machinery:

- `FunctorTrait`: marker for wrapper-like types.
- `Functor = Union[FunctorTrait, Iterable]`.
- `generic_inference`: custom type-parameter inference for parametric dataclasses.
- `dispatchable_parametric`: wraps a dataclass in Plum `parametric`.
- `fmap`: generic mapping over `List` and `Set`, with a fallback error for unsupported iterables.

`RobotWrapper`, `MultiRobotWrapper`, and `SymbolicContext` use `dispatchable_parametric`.

## Plan Compilation: `omniplanner.compile_plan`

`compile_plan.py` is a dispatch-based compilation layer from abstract plans to robot-specific execution objects. It does not define concrete robot commands itself.

Provided dispatch behavior:

- `compile_plan(adaptors, plan_frame, SymbolicContext[List[Any]])`: pushes symbolic context into list elements and compiles each element.
- `compile_plan(adaptors: dict, plan_frame, SymbolicContext[RobotWrapper[Any]])`: pushes context into a robot wrapper, selects the adaptor by robot name, then compiles.
- `compile_plan(adaptors: dict, plan_frame, list)`: compiles each list item.
- `compile_plan(adaptors: dict, plan_frame, RobotWrapper[Any])`: selects adaptor by robot name.
- `compile_plan(adaptors: dict, plan_frame, SymbolicContext[MultiRobotWrapper[Any]])`: pushes and delegates.
- `compile_plan(adaptors: dict, plan_frame, MultiRobotWrapper[SymbolicContext[Any]])`: remaps adaptor keys from outer robot names to inner names and maps compile over the shared wrapper value.
- `compile_plan(adaptor, plan_frame, Wrapper)`: fallback that strips unused wrappers.
- `collect_plans(List[RobotWrapper[Any]])` and `collect_plans(RobotWrapper[Any])`: produce `{robot_name: plan}` dictionaries.

Concrete compilation examples live in `examples/simple_compile_plan.py`, where `GotoPointsPlan`, `FollowPathPlan`, and `PddlPlan` are converted into `SimpleActionSequence`.

## Goto Points: `omniplanner.goto_points`

Domain and data types:

- `GotoPointsDomain`: marker subclass of `PlanningDomain`.
- `GotoPointsGoal`: dataclass with `goal_points: List[str]` and `robot_id`.
- `GroundedGotoPointsProblem`: holds `start_point` and `goal_points`.
- `GotoPointPrimitive`: dataclass with `start` and `goal` arrays.
- `GotoPointsPlan`: dataclass with `plan: list`.

Grounding behavior:

- For `DynamicSceneGraph`, `ground_problem` resolves each goal symbol with `str_to_ns_value`, fetches node positions, and returns `RobotWrapper(robot_id, GroundedGotoPointsProblem(...))`.
- For `np.ndarray`, `ground_problem` indexes the point array by the supplied goal index list.

Planning behavior:

- `make_plan(GroundedGotoPointsProblem, map_context)` creates a chain of `GotoPointPrimitive` segments from the robot start through the requested points.

Technical notes:

- `make_plan` sleeps for three seconds, which appears artificial and is patched out in tests.
- `GotoPointsPlan.append` appends `gpplan.append` instead of probably appending plan contents, which looks like a bug.
- Empty goal lists are not handled before indexing `goal_points[0]`.

## TSP And Layer Planning: `omniplanner.tsp`

`LayerPlanner` wraps DSG layers as NetworkX graphs:

- Converts `dsg.get_layer(layer)` with `spark_dsg.networkx.layer_to_networkx`.
- Falls back from `spark_dsg.DsgLayers.MESH_PLACES` to numeric layer `20`.
- Stores node ids, 2D positions, symbol mappings, and optional all-pairs shortest paths.
- Provides shortest distance/path methods, closest-node lookup, and external point-to-point distance/path through the DSG graph.

TSP data types:

- `TspDomain(solver: str)`.
- `TspGoal(goal_points: List[str], robot_id: str)`.
- `GroundedTspProblem(start_point, goal_points, distances, solver="2opt")`.
- `FollowPathPrimitive(path)`.
- `FollowPathPlan(steps)`.

Planning flow:

- `ground_problem(TspDomain, dsg, robot_states, TspGoal)` resolves symbols to positions, prepends robot start, computes pairwise DSG external distances, and returns `RobotWrapper`.
- `make_plan(GroundedTspProblem, map_context)` solves visit order with `solve_tsp_2opt`, then turns consecutive waypoints into DSG paths.

Technical notes:

- The only implemented TSP solver is `"2opt"`.
- `two_opt` mutates its route list.
- Shortest-path calls use unweighted NetworkX path length in some places while external distances combine graph path length and Euclidean endpoint offsets.

## Language Planner: `omniplanner.language_planner`

Data types:

- `LanguageDomain(domain_type, pddl_domain=None, llm_interface=None)`.
- `LanguageGoal(robot_id, command)`.

Behavior:

- `domain_type == "goto_points"`: splits the command on spaces and creates a `GotoPointsGoal`.
- `domain_type == "Pddl"`: calls `domain.llm_interface.request_plan_specification(command, dsg)`, parses the returned string with `ast.literal_eval`, publishes optional feedback, and grounds each returned robot-specific PDDL goal through the PDDL domain.

Integration points:

- Requires `nlu_interface.llm_interface.LLMInterface` or compatible object.
- Feedback, when present, is expected to have `plugin_feedback_collectors["language_planner"].publish["llm_response"]`.

Technical notes:

- Domain type strings are case-sensitive and inconsistent (`"Pddl"` rather than `"pddl"`).
- LLM responses must be Python-literal dictionaries, not JSON or structured objects.
- The local loop variable `goal` shadows the incoming `LanguageGoal`.

## Utilities: `omniplanner.utils`

`str_to_ns_value(s)` parses strings like `O(7)` with `parse.parse("{}({})", s)`, constructs `spark_dsg.NodeSymbol`, and returns its integer `.value`.

Technical notes:

- Invalid strings are not checked before accessing `p.fixed`.
- This utility returns `.value`, while other code sometimes works with `NodeSymbol` objects or lower-case PDDL symbols directly.

## PDDL Core: `dsg_pddl.pddl_utils`

Helpers:

- `extract_facts(goal, predicate)`: recursively extracts matching facts from a tuple/list AST.
- `extract_negated_facts(goal, predicate, negated=False)`: extracts facts under `not`.
- `tokenize_lisp`, `get_lisp_ast`, `lisp_string_to_ast`: very small Lisp/PDDL parser.
- `ast_to_string`: serializes tuple/list ASTs back to parenthesized strings.
- `pddl_char_to_dsg_char`: converts first-character PDDL symbols (`o`, `p`, `r`) to DSG-style characters (`O`, `P`, `R` where implemented).

Technical notes:

- The parser is intentionally minimal and does not handle comments, quoting, malformed EOF, or full PDDL grammar.
- `extract_facts` includes facts even under negation; callers must use `extract_negated_facts` separately where needed.

## PDDL Data Model: `dsg_pddl.pddl_grounding`

Data types:

- `PddlSymbol(symbol, layer, unary_predicates_to_apply, position=None)`: hashable and ordered by `symbol` only.
- `PddlProblem(name, domain, objects, initial_facts, goal, optimizing)`: renders Fast Downward-flavored PDDL problem strings.
- `PddlGoal(pddl_goal, robot_id)`.
- `PddlDomain(domain_str)`: parses a PDDL domain into `domain_ast`, `domain_name`, `requirements`, `domain_types`, `predicates`, `functions`, `derived`, and `actions`.
- `MultiRobotPddlDomain(PddlDomain)`: marker subclass for multi-robot grounding.
- `GroundedPddlProblem(domain, problem_str, symbols)`.

Domain parser assumptions:

- `ensure_pddl_domain` requires clauses in a fixed order: `define`, requirements, types, predicates, functions, then actions/derived clauses.
- Types, predicates, functions, derived predicates, and actions are parsed structurally but not semantically validated.

Technical notes:

- `PddlSymbol.__eq__` and `__hash__` ignore layer and position, so two symbols with the same name but different metadata collapse.
- `PddlProblem` accepts tuples/lists as ASTs and renders strings by simple recursion.

## PDDL Solving: `dsg_pddl.pddl_planning`

`solve_pddl(GroundedPddlProblem)`:

1. Builds two Fast Downward search commands: an "optimal" A* FF search and a "suboptimal" lazy greedy FF search.
2. Runs both in separate threads via `_run_fd`.
3. Each `_run_fd` creates a temporary directory, writes `domain.pddl` and `problem.pddl`, runs `fast-downward`, and reads the plan file if successful.
4. `solve_pddl` prefers the optimal plan if present, otherwise the suboptimal plan, otherwise an empty plan.
5. It writes debug copies to `DEBUG_OUTPUT_DIR`.
6. It parses plan lines, excluding the final cost line, with `lisp_string_to_ast`.

Technical notes:

- Despite the "race strategy" comment, both threads are joined before choosing a result; the first successful result does not cancel the other.
- Debug output writes to the current working directory by default.
- File writes use direct `open`, unlike most package code.
- If `fast-downward` is missing, both runs fail and an empty plan is returned.

## DSG/PDDL Grounding: `dsg_pddl.dsg_pddl_grounding`

This module registers `ground_problem(PddlDomain, DynamicSceneGraph, dict, PddlGoal)`.

Supported legacy domain names:

- `goto-object-domain`.
- `object-rearrangement-domain`.
- `region-object-rearrangement-domain`.

Unknown `PddlDomain` names are delegated to `dsg_pddl_grounding_improved.ground_improved_problem`.

Main helper groups:

- Connectivity:
  - `generate_symbol_connectivity`.
  - `symbol_connectivity_to_pddl`.
  - `explicit_edges_from_layer`.
  - `implicit_edges_from_layers`.
  - `generate_dense_symbol_connectivity`.
  - `generate_dense_region_symbol_connectivity`.
- DSG layer access and normalization:
  - `get_places_layer`.
  - `normalize_symbol`, `normalize_symbols`.
  - `add_symbol_positions`.
  - `extract_all_symbols`.
- Containment:
  - `generate_object_containment`.
  - `generate_place_containment`.
- PDDL problem construction:
  - `generate_inspection_pddl`.
  - `generate_rearrangement_pddl`.
  - `generate_region_pddl`.

Data flow:

- A raw PDDL goal string is parsed with `lisp_string_to_ast`.
- Relevant symbols are extracted or all DSG symbols are collected.
- Symbols are normalized to lower-case PDDL names.
- A synthetic start place `pstart` is added.
- Symbol positions are filled from DSG node ids.
- Connectivity and containment facts are generated.
- A `PddlProblem` is rendered and wrapped in `GroundedPddlProblem`.

Technical notes:

- Domain routing is by string matching on `domain.domain_name`.
- Several thresholds are hard-coded: object-object `3`, object-place `10`, start edge `3`.
- `simplify` is a stub.
- Symbol normalization mixes names like `p1`, `p(1)`, and lower-cased DSG strings, depending on upstream input.

## Improved PDDL Grounding: `dsg_pddl.dsg_pddl_grounding_improved`

This module handles newer region rearrangement domains:

- `region-object-rearrangement-derived-predicates-domain`.
- `region-object-rearrangement-explicit-state-domain`.

Main flow:

- `ground_improved_problem` validates the domain name and builds a compressed-graph region rearrangement PDDL problem.
- `generate_region_rearrangement_pddl_compressed_graph` parses the goal, extracts goal and forbidden symbols, creates `pstart`, and calls `generate_improved_places_init_v2`.
- `generate_improved_places_init_v2` adds related places, builds primary/forbidden/protected place nodes, compresses shortest paths into important graph edges, filters containment relations, marks suspicious/unsafe facts, and returns PDDL init facts plus relevant symbols.

Important helpers:

- `extract_goal_symbols`.
- `add_object_related_places`.
- `add_region_related_places`.
- `add_start_place`.
- `build_forbidden_nodes`.
- `process_suspicious_objects`.
- `build_place_edges`.
- `filter_containment_relations`.
- `build_primary_nodes`.
- `build_shortest_paths`.
- `compute_secondary_nodes_from_paths`.
- `extract_protected_nodes`.
- `build_segments_from_paths`.
- `build_compressed_place_graph`.
- `build_compressed_edges`.

Alternate generators remain present:

- `generate_region_rearrangement_pddl_all_symbols`.
- `generate_region_rearrangement_pddl_relevant_paths`.

Technical notes:

- There is substantial duplication with `dsg_pddl_grounding`.
- Suspicious object logic currently treats semantic labels outside `0..40` as suspicious.
- Compressed graph construction depends heavily on NetworkX shortest paths and symbol lookup consistency.
- Several functions mutate lists/dicts passed in, including `symbols_of_interest` and `forbidden_symbols`.

## Multi-Robot PDDL Grounding: `dsg_pddl.dsg_pddl_grounding_multirobot`

This module registers `ground_problem(MultiRobotPddlDomain, DynamicSceneGraph, dict, PddlGoal)`.

Supported domain:

- `region-object-rearrangement-domain-multirobot-fd`.

Main flow:

1. Robot state names are converted to lower-case for PDDL compliance.
2. `generate_multirobot_region_pddl` collects all DSG symbols, normalizes them, and creates one `pstart{robot}` place for each robot with a non-`None` pose.
3. It generates dense region init facts with robot-specific `at-poi` facts.
4. Robot symbols and suspicious object facts are added.
5. A `PddlProblem` is rendered.
6. A `MultiRobotWrapper` wraps the shared `GroundedPddlProblem`.
7. Outer robot names are remapped to inner lower-case names.

Helpers:

- `generate_dense_region_symbol_connectivity_multirobot`.
- `add_robot_start_edges`.
- `nearest_place_for_position`.
- `generate_dense_region_init_multirobot`.
- `filter_goal_for_available_objects`.
- `generate_multirobot_region_pddl`.

Technical notes:

- `generate_multirobot_region_pddl` writes PDDL dump files as a side effect.
- `filter_goal_for_available_objects` exists but is not used by `generate_multirobot_region_pddl`.
- Start-symbol naming depends on robot-name case and may produce mixed names before lower-case conversion.
- Valid planning robots are inferred from all non-`None` `robot_states`, not explicitly from the goal.

## PDDL Plan Parameterization: `dsg_pddl.dsg_pddl_planning`

`make_plan(GroundedPddlProblem, map_context)`:

1. Calls `solve_pddl`.
2. Creates a `LayerPlanner`.
3. Parameterizes symbolic PDDL actions into path-like geometry.
4. Returns `PddlPlan(domain, symbolic_actions, parameterized_actions, symbols)`.

Supported symbolic actions:

- `goto-poi`.
- `inspect`.
- `pick-object`.
- `place-object`.

Multi-robot detection:

- Uses substring check: `"multirobot" in grounded_problem.domain.domain_name`.
- Multi-robot parameterizers drop the robot argument at action index `1`.

Technical notes:

- `last_pose` starts at `np.zeros(2)` rather than the grounded robot start pose.
- Some parameterizers do not update `last_pose`.
- Action support is hard-coded in a `match` block.

## PDDL Domain Assets

PDDL files under `dsg_pddl/domains` are included as package data by `pyproject.toml`.

They define domains for:

- Goto-object inspection.
- Object rearrangement.
- Region object rearrangement.
- Region rearrangement with derived predicates.
- Region rearrangement with explicit state/action costs.
- Multi-robot region rearrangement for Fast Downward.

Examples load these assets through `importlib.resources`.

## Tests

Current tests cover:

- Wrapper behavior, context merging, dispatch recursion, and pipeline orchestration.
- Goto-points grounding and plan construction.
- Language planner routing and feedback publishing.
- TSP layer planning, fallback layer selection, 2-opt behavior, and symbolic TSP grounding.
- PDDL utilities, domain/problem rendering, grounding helper functions, improved grounding helper functions, multi-robot helper behavior, PDDL plan parameterization, and `solve_pddl` selection with Fast Downward stubbed.

The tests use monkeypatching heavily and include fake DSG/node objects to avoid requiring full external systems for most unit behavior.

## Public Interfaces And Entry Points

There are no configured console scripts or CLI entry points in `pyproject.toml`.

Public-facing Python APIs are currently direct imports from modules, including:

- `omniplanner.omniplanner.PlanRequest`.
- `omniplanner.omniplanner.full_planning_pipeline`.
- `omniplanner.omniplanner.ground_problem`.
- `omniplanner.omniplanner.make_plan`.
- `omniplanner.compile_plan.compile_plan`.
- `omniplanner.compile_plan.collect_plans`.
- Domain/goal classes from `goto_points`, `tsp`, `language_planner`, and `dsg_pddl.pddl_grounding`.

Examples are the main usage documentation.

## Key Data Flows

### Direct Goto Points

```text
PlanRequest(GotoPointsDomain, GotoPointsGoal, robot_states)
  -> full_planning_pipeline
  -> ground_problem(GotoPointsDomain, DSG, robot_states, GotoPointsGoal)
  -> RobotWrapper[GroundedGotoPointsProblem]
  -> SymbolicContext[DsgContextProvider, RobotWrapper[...]]
  -> make_plan via wrapper fmap/push
  -> RobotWrapper[SymbolicContext[GotoPointsPlan]]
  -> compile_plan(optional)
```

### TSP

```text
PlanRequest(TspDomain, TspGoal, robot_states)
  -> ground_problem resolves symbols and computes distance matrix with LayerPlanner
  -> RobotWrapper[GroundedTspProblem]
  -> make_plan solves 2-opt order and creates FollowPathPlan
  -> compile_plan(optional)
```

### Single-Robot PDDL

```text
PddlDomain + PddlGoal
  -> ensure_domain_dispatch_registered imports PDDL grounding modules
  -> ground_problem(PddlDomain, DSG, robot_states, PddlGoal)
  -> domain-name-specific PDDL problem generation
  -> RobotWrapper[GroundedPddlProblem]
  -> make_plan(GroundedPddlProblem, DSG)
  -> solve_pddl runs Fast Downward
  -> symbolic action tuples
  -> hard-coded action parameterization through LayerPlanner
  -> PddlPlan
```

### Multi-Robot PDDL

```text
MultiRobotPddlDomain + PddlGoal + robot_states
  -> ground_problem(MultiRobotPddlDomain, DSG, robot_states, PddlGoal)
  -> shared GroundedPddlProblem
  -> MultiRobotWrapper with outer/inner robot-name remaps
  -> make_plan produces shared PddlPlan
  -> compile_plan can split actions per robot in example compiler
```

### Language

```text
LanguageDomain("goto_points")
  -> split command into symbol strings
  -> GotoPointsGoal
  -> normal goto-points grounding

LanguageDomain("Pddl")
  -> LLMInterface.request_plan_specification
  -> ast.literal_eval response dict
  -> PddlGoal per robot
  -> PDDL grounding
```

## Architectural Patterns

- Multiple dispatch is the primary extensibility mechanism.
- Wrappers implement functor-like behavior so planning and compiling can transparently preserve robot identity and symbolic context.
- Domain selection is mostly class-based at dispatch level, then often string-based inside PDDL grounding.
- DSG/PDDL code uses procedural helper pipelines rather than object-oriented services.
- PDDL assets are static package data.
- Examples function as integration documentation and compiler extension examples.

## Known Technical Debt And Fragile Areas

- Import side effects are required for dispatch registrations.
- Two top-level packages, `omniplanner` and `dsg_pddl`, are tightly coupled but separately named.
- `omniplanner.omniplanner` contains base types, wrapper mechanics, context providers, dispatch functions, and orchestration in one module.
- PDDL parsing/rendering is a minimal custom implementation with fixed assumptions.
- Domain routing uses string literals spread across modules.
- Grounding code duplicates normalization, containment, connectivity, and start-symbol handling across legacy, improved, and multi-robot modules.
- Symbol naming conventions are inconsistent: examples include `O(1)`, `o1`, `pstart`, `pstartspot`, and lower-cased DSG strings.
- Several functions mutate arguments in place.
- Several file-system side effects happen during solving/grounding.
- Fast Downward integration is synchronous from the caller's perspective and always waits for both planner threads.
- `DEBUG_OUTPUT_DIR` default may write debug files to the process working directory.
- `GotoPointsPlan.append` appears incorrect.
- `GotoPointsPlan.make_plan` does not handle empty goals.
- `LanguageDomain` does not subclass `PlanningDomain`, and `LanguageGoal` does not subclass `PlanningGoal`, even though they participate in `ground_problem`.
- `LanguageDomain("Pddl")` expects Python literal output from the LLM.
- Multi-robot planning infers participating robots from `robot_states`, not from goal semantics.
- `dsg_pddl.dsg_pddl_planning` hard-codes action names and parameterization behavior.
- `__pycache__` files are present under package, examples, and tests in the working tree.
- README content is currently only `# Omniplanner` and `TBC`.

## What Future Developers Or Agents Need To Know

- Do not refactor by moving dispatch functions casually; Plum registrations are tied to imports and function objects.
- Before adding a new domain, decide whether it should be a new `PlanningDomain` subclass with its own `ground_problem`, or a new `PddlDomain.domain_name` branch.
- Before adding a new plan type, register `make_plan` for its grounded problem and `compile_plan` for its abstract plan if execution output is required.
- PDDL grounding assumes DSG layers for mesh places, objects, and rooms, and frequently falls back to numeric mesh-place layer `20`.
- The package currently has no stable top-level API contract in `__init__.py`; direct module paths are the de facto API.
- Tests rely on monkeypatching dispatch functions and external dependencies, so refactors should preserve dispatch names or provide compatibility shims.
- Examples may be used by downstream users despite not being package modules.
