#!/usr/bin/env python3
import csv
import json
import time
import math
import itertools
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batches of 3DSG-PDDL experiments"
    )

    parser.add_argument(
        "--script",
        type=Path,
        required=True,
        help="Path to the single-experiment script",
    )

    parser.add_argument(
        "--experiment-kind",
        choices=["synthetic", "real"],
        default="synthetic",
        help=(
            "Type of single-experiment script to run. Use 'synthetic' for the "
            "scalability script that accepts --num-nodes/--num-objects/"
            "--num-regions/--graph-seed. Use 'real' for the real-DSG script."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("batch_results"),
    )

    # Synthetic DSG experiment options.
    parser.add_argument("--num-nodes", type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--num-objects", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--num-regions", type=int, nargs="+", default=[4, 6])

    parser.add_argument(
        "--graph-scales",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional graph size scales for synthetic experiments. When provided, "
            "num-nodes, num-objects, and num-regions must each have exactly one "
            "value. Each scale is applied to all three base values."
        ),
    )

    parser.add_argument("--graph-seeds", type=int, nargs="+", default=[1])

    # Shared goal/PDDL options.
    parser.add_argument("--goal-seeds", type=int, nargs="+", default=[1, 2, 3])

    parser.add_argument("--goal-disjunctions", type=int, nargs="+", default=[1])
    parser.add_argument("--goal-conjunctions", type=int, nargs="+", default=[6])

    parser.add_argument(
        "--pddl-domains",
        nargs="+",
        choices=["derived", "explicit"],
        default=["explicit"],
    )

    parser.add_argument(
        "--pddl-samplers",
        nargs="+",
        choices=["all", "paths", "compressed"],
        default=["compressed"],
    )

    parser.add_argument(
        "--pddl-solvers",
        nargs="+",
        choices=["lazy_ff", "wastar_ff", "astar_ff", "lmcut"],
        default=["lazy_ff"],
    )

    parser.add_argument("--pddl-timeout", type=int, default=60)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.experiment_kind == "synthetic":
        if args.graph_scales is not None:
            if len(args.num_nodes) != 1:
                parser.error("--graph-scales requires exactly one --num-nodes value")
            if len(args.num_objects) != 1:
                parser.error("--graph-scales requires exactly one --num-objects value")
            if len(args.num_regions) != 1:
                parser.error("--graph-scales requires exactly one --num-regions value")
            if any(scale <= 0 for scale in args.graph_scales):
                parser.error("--graph-scales values must be positive")
    else:
        if args.graph_scales is not None:
            parser.error("--graph-scales is only valid with --experiment-kind synthetic")

    return args


def scaled_int(value, scale):
    return max(1, int(math.floor(value * scale + 0.5)))


def scale_tag(scale):
    return f"{scale:g}".replace(".", "p")


def experiment_name(cfg):
    if cfg["experiment_kind"] == "real":
        return (
            f"real_dsg"
            f"_d{cfg['goal_disjunctions']}"
            f"_c{cfg['goal_conjunctions']}"
            f"_goal{cfg['goal_seed']}"
            f"_{cfg['pddl_domain']}"
            f"_{cfg['pddl_sampler']}"
            f"_{cfg['pddl_solver']}"
        )

    name = (
        f"n{cfg['num_nodes']}"
        f"_o{cfg['num_objects']}"
        f"_r{cfg['num_regions']}"
    )

    if cfg.get("graph_scale") is not None:
        name += f"_s{scale_tag(cfg['graph_scale'])}"

    name += (
        f"_graph{cfg['graph_seed']}"
        f"_d{cfg['goal_disjunctions']}"
        f"_c{cfg['goal_conjunctions']}"
        f"_goal{cfg['goal_seed']}"
        f"_{cfg['pddl_domain']}"
        f"_{cfg['pddl_sampler']}"
        f"_{cfg['pddl_solver']}"
    )

    return name


def make_size_grid(args):
    if args.graph_scales is None:
        keys = ["num_nodes", "num_objects", "num_regions"]
        values = [args.num_nodes, args.num_objects, args.num_regions]

        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

        return

    base_num_nodes = args.num_nodes[0]
    base_num_objects = args.num_objects[0]
    base_num_regions = args.num_regions[0]

    for scale in args.graph_scales:
        yield {
            "num_nodes": scaled_int(base_num_nodes, scale),
            "num_objects": scaled_int(base_num_objects, scale),
            "num_regions": scaled_int(base_num_regions, scale),
            "graph_scale": scale,
            "base_num_nodes": base_num_nodes,
            "base_num_objects": base_num_objects,
            "base_num_regions": base_num_regions,
        }


def make_synthetic_grid(args):
    static_keys = [
        "graph_seed",
        "goal_disjunctions",
        "goal_conjunctions",
        "goal_seed",
        "pddl_domain",
        "pddl_sampler",
        "pddl_solver",
    ]

    static_values = [
        args.graph_seeds,
        args.goal_disjunctions,
        args.goal_conjunctions,
        args.goal_seeds,
        args.pddl_domains,
        args.pddl_samplers,
        args.pddl_solvers,
    ]

    for size_cfg in make_size_grid(args):
        for combo in itertools.product(*static_values):
            cfg = {
                "experiment_kind": "synthetic",
                **size_cfg,
                **dict(zip(static_keys, combo)),
            }

            # Match the single-run script restriction.
            if cfg["pddl_solver"] == "lmcut" and cfg["pddl_domain"] == "derived":
                continue

            yield cfg


def make_real_grid(args):
    static_keys = [
        "goal_disjunctions",
        "goal_conjunctions",
        "goal_seed",
        "pddl_domain",
        "pddl_sampler",
        "pddl_solver",
    ]

    static_values = [
        args.goal_disjunctions,
        args.goal_conjunctions,
        args.goal_seeds,
        args.pddl_domains,
        args.pddl_samplers,
        args.pddl_solvers,
    ]

    for combo in itertools.product(*static_values):
        cfg = {
            "experiment_kind": "real",
            **dict(zip(static_keys, combo)),
        }

        # Match the single-run script restriction.
        if cfg["pddl_solver"] == "lmcut" and cfg["pddl_domain"] == "derived":
            continue

        yield cfg


def make_grid(args):
    if args.experiment_kind == "real":
        yield from make_real_grid(args)
    else:
        yield from make_synthetic_grid(args)


def command_for(args, cfg):
    cmd = [
        "python",
        str(args.script),
    ]

    if cfg["experiment_kind"] == "synthetic":
        cmd.extend([
            "--num-nodes", str(cfg["num_nodes"]),
            "--num-objects", str(cfg["num_objects"]),
            "--num-regions", str(cfg["num_regions"]),
            "--graph-seed", str(cfg["graph_seed"]),
        ])

    cmd.extend([
        "--goal-disjunctions", str(cfg["goal_disjunctions"]),
        "--goal-conjunctions", str(cfg["goal_conjunctions"]),
        "--goal-seed", str(cfg["goal_seed"]),
        "--pddl-domain", cfg["pddl_domain"],
        "--pddl-sampler", cfg["pddl_sampler"],
        "--pddl-solver", cfg["pddl_solver"],
        "--pddl-timeout", str(args.pddl_timeout),
        "--output-dir", str(args.output_dir),
    ])

    return cmd


def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def write_csv(path, rows):
    if not rows:
        return

    keys = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_single_run_results(exp_dir):
    result_path = exp_dir / "planning_results.json"
    if not result_path.exists():
        return {}

    with open(result_path, "r") as f:
        return json.load(f)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_jsonl = args.output_dir / "batch_summary.jsonl"
    summary_csv = args.output_dir / "batch_summary.csv"

    rows = []
    grid = list(make_grid(args))

    print(f"Running {len(grid)} {args.experiment_kind} experiments")

    for i, cfg in enumerate(grid, start=1):
        name = experiment_name(cfg)
        exp_dir = args.output_dir / "results" / name
        result_file = exp_dir / "planning_results.json"

        if args.skip_existing and result_file.exists():
            print(f"[{i}/{len(grid)}] Skipping existing: {name}")
            single_results = load_single_run_results(exp_dir)
            run_success = single_results.get("success", False)
            row = {
                **cfg,
                "experiment": name,
                "status": "skipped_success" if run_success else "skipped_failed",
                **single_results,
            }
            rows.append(row)
            append_jsonl(summary_jsonl, row)
            write_csv(summary_csv, rows)
            continue

        cmd = command_for(args, cfg)

        print(f"\n[{i}/{len(grid)}] Running: {name}")
        print(" ".join(cmd))

        if args.dry_run:
            row = {
                **cfg,
                "experiment": name,
                "status": "dry_run",
                "command": " ".join(cmd),
            }
            rows.append(row)
            append_jsonl(summary_jsonl, row)
            write_csv(summary_csv, rows)
            continue

        start = time.perf_counter()
        completed = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
        )
        elapsed = time.perf_counter() - start

        single_results = load_single_run_results(exp_dir) if completed.returncode == 0 else {}
        run_success = (
            completed.returncode == 0
            and single_results.get("success", False)
        )

        row = {
            **cfg,
            "experiment": name,
            "status": "success" if run_success else "failed",
            "returncode": completed.returncode,
            "wall_time_sec": elapsed,
            "stdout_tail": completed.stdout[-2000:],
            "stderr_tail": completed.stderr[-2000:],
            **single_results,
        }

        if completed.returncode != 0:
            print(completed.stderr)

        rows.append(row)
        append_jsonl(summary_jsonl, row)
        write_csv(summary_csv, rows)

    write_csv(summary_csv, rows)
    print(f"\nBatch summary written to: {summary_csv}")
    print(f"JSONL log written to: {summary_jsonl}")


if __name__ == "__main__":
    main()
