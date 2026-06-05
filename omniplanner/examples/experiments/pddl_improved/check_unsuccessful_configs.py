#!/usr/bin/env python3
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box


CONFIG_COLUMNS = [
    "num_nodes",
    "num_edges",
    "num_objects",
    "num_regions",
    "goal_conjunctions",
    "goal_disjunctions",
    "graph_seed",
    "goal_seed",
    "pddl_domain",
    "pddl_sampler",
    "pddl_solver",
]

INTEGER_COLUMNS = {
    "num_nodes",
    "num_edges",
    "num_objects",
    "num_regions",
    "goal_conjunctions",
    "goal_disjunctions",
    "graph_seed",
    "goal_seed",
}

DIMENSIONS = {
    "graph_size": ["num_nodes", "num_edges", "num_regions"],
    "graph_seed": ["graph_seed"],
    "goal_shape": ["goal_conjunctions", "goal_disjunctions"],
    "goal_seed": ["goal_seed"],
    "planner": ["pddl_domain", "pddl_sampler", "pddl_solver"],
}

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print compact unsuccessful batch experiment configurations"
    )

    parser.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="Batch output directory containing results/*",
    )

    return parser.parse_args()


def load_results(batch_dir: Path) -> pd.DataFrame:
    rows = []

    for result_file in sorted((batch_dir / "results").glob("*/planning_results.json")):
        exp_dir = result_file.parent
        config_file = exp_dir / "config.json"

        if not config_file.exists():
            continue

        with open(config_file) as f:
            config = json.load(f)

        with open(result_file) as f:
            results = json.load(f)

        rows.append(
            {
                "experiment": exp_dir.name,
                **config,
                **results,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        raise RuntimeError(f"No results found in {batch_dir / 'results'}")

    if "success" not in df.columns:
        raise RuntimeError("Column 'success' was not found in results.")

    return df


def is_integer_like(value) -> bool:
    if pd.isna(value) or isinstance(value, bool):
        return False

    if isinstance(value, int):
        return True

    if isinstance(value, float):
        return value.is_integer()

    return False


def fmt(value, column: str | None = None) -> str:
    if pd.isna(value):
        return "-"

    if value == "*":
        return "[bold cyan]*[/bold cyan]"

    if column in INTEGER_COLUMNS:
        return str(int(value))

    if is_integer_like(value):
        return str(int(value))

    return str(value)


def build_candidate(
    df_all: pd.DataFrame,
    df_failed: pd.DataFrame,
    config_cols: list[str],
    wildcard_cols: tuple[str, ...],
    fixed_values: tuple,
):
    fixed_cols = [c for c in config_cols if c not in wildcard_cols]

    all_group = df_all.copy()
    failed_group = df_failed.copy()

    for col, value in zip(fixed_cols, fixed_values):
        all_group = all_group[all_group[col] == value]
        failed_group = failed_group[failed_group[col] == value]

    if failed_group.empty:
        return None

    wildcard_cols_list = list(wildcard_cols)

    all_variants = set(
        tuple(row[col] for col in wildcard_cols_list)
        for _, row in all_group[wildcard_cols_list]
            .drop_duplicates()
            .iterrows()
    )

    failed_variants = set(
        tuple(row[col] for col in wildcard_cols_list)
        for _, row in failed_group[wildcard_cols_list]
            .drop_duplicates()
            .iterrows()
    )

    if not all_variants or all_variants != failed_variants:
        return None

    pattern = {}

    for col, value in zip(fixed_cols, fixed_values):
        pattern[col] = value

    wildcard_count = 0

    for col in wildcard_cols:
        values = all_group[col].dropna().unique().tolist()

        if len(values) > 1:
            pattern[col] = "*"
            wildcard_count += 1
        else:
            pattern[col] = values[0] if values else "-"

    if wildcard_count == 0:
        return None

    return {
        "pattern": pattern,
        "covered_indices": set(failed_group.index.tolist()),
        "num_rows": len(failed_group),
        "wildcard_count": wildcard_count,
    }


def compact_failures(df_all: pd.DataFrame, df_failed: pd.DataFrame) -> pd.DataFrame:
    config_cols = [c for c in CONFIG_COLUMNS if c in df_all.columns]

    dimensions = {
        "graph_size": ["num_nodes", "num_edges", "num_regions"],
        "graph_seed": ["graph_seed"],
        "goal_shape": ["goal_conjunctions", "goal_disjunctions"],
        "goal_seed": ["goal_seed"],
        "planner": ["pddl_domain", "pddl_sampler", "pddl_solver"],
    }

    dimensions = {
        name: [c for c in cols if c in config_cols]
        for name, cols in dimensions.items()
    }
    dimensions = {name: cols for name, cols in dimensions.items() if cols}

    candidates = []

    dimension_items = list(dimensions.items())

    for size in range(len(dimension_items), 0, -1):
        for selected_dims in itertools.combinations(dimension_items, size):
            wildcard_cols = sorted(
                {
                    col
                    for _, cols in selected_dims
                    for col in cols
                },
                key=config_cols.index,
            )

            fixed_cols = [c for c in config_cols if c not in wildcard_cols]

            if not fixed_cols:
                fixed_groups = [()]
            else:
                fixed_groups = (
                    df_failed[fixed_cols]
                    .drop_duplicates()
                    .itertuples(index=False, name=None)
                )

            for fixed_values in fixed_groups:
                candidate = build_candidate(
                    df_all=df_all,
                    df_failed=df_failed,
                    config_cols=config_cols,
                    wildcard_cols=tuple(wildcard_cols),
                    fixed_values=fixed_values,
                )

                if candidate is not None:
                    candidates.append(candidate)

    candidates.sort(
        key=lambda c: (
            c["num_rows"],
            c["wildcard_count"],
        ),
        reverse=True,
    )

    used_indices = set()
    compact_rows = []

    for candidate in candidates:
        if candidate["covered_indices"] & used_indices:
            continue

        compact_rows.append(
            {
                **candidate["pattern"],
                "failed_runs": candidate["num_rows"],
            }
        )

        used_indices.update(candidate["covered_indices"])

    remaining = df_failed.loc[~df_failed.index.isin(used_indices)]

    for _, row in remaining.iterrows():
        compact_rows.append(
            {
                **{col: row[col] for col in config_cols},
                "failed_runs": 1,
            }
        )

    compact_df = pd.DataFrame(compact_rows)

    return compact_df.sort_values(
        by=[c for c in config_cols if c in compact_df.columns],
        key=lambda s: s.astype(str),
    ).reset_index(drop=True)


def overview_panel(df_all: pd.DataFrame, df_failed: pd.DataFrame, df_compact: pd.DataFrame):
    return Panel(
        f"[bold]Total runs found:[/bold] {len(df_all)}\n"
        f"[bold green]Successful runs:[/bold green] {len(df_all) - len(df_failed)}\n"
        f"[bold red]Unsuccessful runs:[/bold red] {len(df_failed)}\n"
        f"[bold cyan]Compact rows:[/bold cyan] {len(df_compact)}",
        title="Unsuccessful configuration overview",
        border_style="cyan",
    )


def compact_table(df_compact: pd.DataFrame) -> Table:
    table = Table(
        title="Compact unsuccessful configurations",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )

    visible_cols = [c for c in CONFIG_COLUMNS if c in df_compact.columns]

    for col in visible_cols:
        justify = "right" if col in INTEGER_COLUMNS else "left"
        table.add_column(col, justify=justify, overflow="fold")

    table.add_column("failed_runs", justify="right", style="bold red")

    for _, row in df_compact.iterrows():
        table.add_row(
            *[fmt(row[col], col) for col in visible_cols],
            fmt(row["failed_runs"], "failed_runs"),
        )

    return table


def main():
    args = parse_args()

    df_all = load_results(args.batch_dir)
    df_failed = df_all[df_all["success"] != True].copy()

    if df_failed.empty:
        console.print(
            Panel(
                "[bold green]No unsuccessful configurations found.[/bold green]",
                title="Result",
                border_style="green",
            )
        )
        return

    df_compact = compact_failures(df_all, df_failed)

    console.print(overview_panel(df_all, df_failed, df_compact))
    console.print(compact_table(df_compact))


if __name__ == "__main__":
    main()