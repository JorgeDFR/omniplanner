#!/usr/bin/env python3
import csv
import json
import time
import itertools
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batches of DSG-PDDL scalability experiments"
    )

    parser.add_argument(
        "--script",
        type=Path,
        required=True,
        help="Path to the single-experiment script",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("batch_results"),
    )

    parser.add_argument("--num-nodes", type=int, nargs="+", default=[100, 200, 400])
    parser.add_argument("--num-objects", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--num-regions", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--graph-seeds", type=int, nargs="+", default=[1])
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

    return parser.parse_args()


def experiment_name(cfg):
    return (
        f"n{cfg['num_nodes']}"
        f"_o{cfg['num_objects']}"
        f"_r{cfg['num_regions']}"
        f"_graph{cfg['graph_seed']}"
        f"_d{cfg['goal_disjunctions']}"
        f"_c{cfg['goal_conjunctions']}"
        f"_goal{cfg['goal_seed']}"
        f"_{cfg['pddl_domain']}"
        f"_{cfg['pddl_sampler']}"
        f"_{cfg['pddl_solver']}"
    )


def make_grid(args):
    keys = [
        "num_nodes",
        "num_objects",
        "num_regions",
        "graph_seed",
        "goal_disjunctions",
        "goal_conjunctions",
        "goal_seed",
        "pddl_domain",
        "pddl_sampler",
        "pddl_solver",
    ]

    values = [
        args.num_nodes,
        args.num_objects,
        args.num_regions,
        args.graph_seeds,
        args.goal_disjunctions,
        args.goal_conjunctions,
        args.goal_seeds,
        args.pddl_domains,
        args.pddl_samplers,
        args.pddl_solvers,
    ]

    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))

        # Match your single-run script restriction.
        if cfg["pddl_solver"] == "lmcut" and cfg["pddl_domain"] == "derived":
            continue

        yield cfg


def command_for(args, cfg):
    return [
        "python",
        str(args.script),
        "--num-nodes", str(cfg["num_nodes"]),
        "--num-objects", str(cfg["num_objects"]),
        "--num-regions", str(cfg["num_regions"]),
        "--graph-seed", str(cfg["graph_seed"]),
        "--goal-disjunctions", str(cfg["goal_disjunctions"]),
        "--goal-conjunctions", str(cfg["goal_conjunctions"]),
        "--goal-seed", str(cfg["goal_seed"]),
        "--pddl-domain", cfg["pddl_domain"],
        "--pddl-sampler", cfg["pddl_sampler"],
        "--pddl-solver", cfg["pddl_solver"],
        "--pddl-timeout", str(args.pddl_timeout),
        "--output-dir", str(args.output_dir),
    ]


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

    print(f"Running {len(grid)} experiments")

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
            continue

        cmd = command_for(args, cfg)

        print(f"\n[{i}/{len(grid)}] Running: {name}")
        print(" ".join(cmd))

        if args.dry_run:
            row = {
                **cfg,
                "experiment": name,
                "status": "dry_run",
            }
            rows.append(row)
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