#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


VARY_OPTIONS = [
    "num_nodes",
    "num_objects",
    "num_regions",
    "goal_conjunctions",
    "goal_disjunctions",
    "pddl_domain",
    "pddl_sampler",
    "pddl_solver",
]

STATIC_OPTIONS = VARY_OPTIONS.copy()

SEED_COLUMNS = [
    "graph_seed",
    "goal_seed",
]

METRICS = {
    "sampling_time_sec": "Sampler time (s)",
    "planning_time_sec": "Planning time (s)",
    "total_time_sec": "Total execution time (s)",
    "num_sampled_symbols": "Number of sampled symbols",
    "plan_cost": "Plan cost",
}

STATS = [
    "mean",
    "std",
    "min",
    "median",
    "max",
]

INTEGER_COLUMNS = {
    "num_nodes",
    "num_objects",
    "num_regions",
    "goal_conjunctions",
    "goal_disjunctions",
    "graph_seed",
    "goal_seed",
    "num_sampled_symbols",
    "plan_cost",
    "num_runs",
}

console = Console()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Print organized batch experiment results"
    )

    parser.add_argument(
        "--batch-dir",
        type=Path,
        required=True,
        help="Batch output directory containing results/*",
    )

    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help="Decimal precision for floating-point values.",
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

    return df


def is_integer_like(value) -> bool:
    if pd.isna(value):
        return False

    if isinstance(value, bool):
        return False

    if isinstance(value, int):
        return True

    if isinstance(value, float):
        return value.is_integer()

    return False


def fmt(value, precision: int, column: str | None = None) -> str:
    if pd.isna(value):
        return "-"

    if column in INTEGER_COLUMNS:
        return str(int(value))

    if is_integer_like(value) and column not in {
        "mean",
        "std",
        "min",
        "median",
        "max",
    }:
        return str(int(value))

    if isinstance(value, float):
        return f"{value:.{precision}f}"

    return str(value)


def metric_table(
    group: pd.DataFrame,
    vary: str,
    metric: str,
    metric_title: str,
    precision: int,
) -> Table:
    summary = (
        group.groupby(vary)[metric]
        .agg(STATS)
        .round(precision)
        .reset_index()
        .sort_values(vary)
    )

    table = Table(
        title=metric_title,
        show_lines=False,
    )

    table.add_column(vary, justify="right", style="bold")
    for stat in STATS:
        table.add_column(stat, justify="right")

    for _, row in summary.iterrows():
        table.add_row(
            fmt(row[vary], precision, vary),
            *[
                fmt(
                    row[stat],
                    precision,
                    metric,
                )
                for stat in STATS
            ],
        )

    return table


def counts_table(group: pd.DataFrame, vary: str) -> Table:
    counts = (
        group.groupby(vary)
        .size()
        .rename("num_runs")
        .reset_index()
        .sort_values(vary)
    )

    table = Table(title="Runs per value")
    table.add_column(vary, justify="right", style="bold")
    table.add_column("num_runs", justify="right")

    for _, row in counts.iterrows():
        table.add_row(
            fmt(row[vary], 0, vary),
            fmt(row["num_runs"], 0, "num_runs"),
        )

    return table


def static_config_panel(static_config: dict) -> Panel:
    lines = [f"[bold]{key}[/bold]: {value}" for key, value in static_config.items()]
    return Panel(
        "\n".join(lines),
        title="Static configuration",
        border_style="cyan",
    )


def sorted_unique_values(series: pd.Series):
    values = series.dropna().unique().tolist()
    return sorted(values, key=lambda x: str(x))


def parse_choice(raw: str, options: list):
    for option in options:
        if raw == str(option):
            return option
    return None


def ask_choice(column: str, options: list, prompt_prefix: str = "Choose value"):
    console.print(f"\n[bold cyan]{column}[/bold cyan]")
    console.print("Available values:")

    for option in options:
        console.print(f"  - {option}")

    while True:
        raw = input(f"{prompt_prefix} for {column}: ").strip()
        value = parse_choice(raw, options)

        if value is not None:
            return value

        console.print(
            f"[red]Invalid value.[/red] Please choose one of: "
            f"{', '.join(map(str, options))}"
        )


def config_overview_table(df: pd.DataFrame) -> Table:
    table = Table(title="Found experiment configuration parameters")
    table.add_column("Parameter", style="bold")
    table.add_column("# values", justify="right")
    table.add_column("Values")

    for col in VARY_OPTIONS:
        if col not in df.columns:
            continue

        values = sorted_unique_values(df[col])
        table.add_row(
            col,
            str(len(values)),
            ", ".join(map(str, values)),
        )

    return table


def main():
    args = parse_args()

    df_all = load_results(args.batch_dir)

    if "success" in df_all.columns:
        df_success = df_all[df_all["success"] == True].copy()
        if df_success.empty:
            raise RuntimeError("No successful runs found.")
    else:
        df_success = df_all.copy()

    available_metrics = {
        metric: title
        for metric, title in METRICS.items()
        if metric in df_success.columns
    }

    seed_cols = [c for c in SEED_COLUMNS if c in df_success.columns]

    console.print(
        Panel(
            "This script summarizes one comparable experiment group.\n\n"
            "First, choose which parameter should be analyzed as the varying "
            "parameter. Then, for every other configuration parameter that has "
            "multiple values, choose the fixed value to use.\n\n"
            "Seeds are not selected manually; they are averaged over automatically.",
            title="Interactive setup",
            border_style="cyan",
        )
    )

    total_runs = len(df_all)
    successful_runs = len(df_success)
    failed_runs = total_runs - successful_runs

    console.print(
        Panel(
            f"[bold]Total runs found:[/bold] {total_runs}\n"
            f"[bold]Successful runs:[/bold] {successful_runs}\n"
            f"[bold]Failed runs:[/bold] {failed_runs}",
            title="Loaded results",
            border_style="green",
        )
    )

    console.print(config_overview_table(df_success))

    possible_vary_options = [
        col
        for col in VARY_OPTIONS
        if col in df_success.columns and df_success[col].nunique(dropna=True) > 1
    ]

    single_value_vary_options = [
        col
        for col in VARY_OPTIONS
        if col in df_success.columns and df_success[col].nunique(dropna=True) == 1
    ]

    if possible_vary_options:
        vary = ask_choice(
            "varying parameter",
            possible_vary_options,
            prompt_prefix="Choose parameter to analyze",
        )
    else:
        if single_value_vary_options:
            vary = single_value_vary_options[0]

            console.print(
                Panel(
                    f"No parameter has more than one successful value.\n\n"
                    f"Using [bold]{vary}[/bold] as the grouping parameter anyway, "
                    f"so results will be printed with one row.",
                    title="Single-value results",
                    border_style="yellow",
                )
            )
        else:
            vary = "__all_results__"
            df_success[vary] = "all"
            df_all[vary] = "all"

            console.print(
                Panel(
                    "No configuration parameter values were available.\n\n"
                    "Using a synthetic grouping column so results will be printed "
                    "with one row.",
                    title="Single-value results",
                    border_style="yellow",
                )
            )

    static_cols = [
        c for c in STATIC_OPTIONS
        if c in df_success.columns and c != vary
    ]

    selected_config = {}

    console.print(
        Panel(
            f"You selected [bold]{vary}[/bold] as the varying parameter.\n\n"
            "Now choose fixed values for the remaining parameters whenever more "
            "than one value is available.",
            title="Select fixed configuration",
            border_style="cyan",
        )
    )

    filtered_success = df_success.copy()
    filtered_all = df_all.copy()

    for col in static_cols:
        options = sorted_unique_values(filtered_success[col])

        if len(options) > 1:
            selected_value = ask_choice(col, options)
            selected_config[col] = selected_value

            filtered_success = filtered_success[
                filtered_success[col] == selected_value
            ].copy()

            filtered_all = filtered_all[
                filtered_all[col] == selected_value
            ].copy()

        elif len(options) == 1:
            selected_config[col] = options[0]

    if filtered_success.empty:
        raise RuntimeError("No successful runs match the selected configuration.")

    if filtered_success[vary].nunique(dropna=True) <= 1:
        console.print(
            Panel(
                f"The selected configuration has only one value for "
                f"[bold]{vary}[/bold]: {filtered_success[vary].iloc[0]}\n\n"
                "Continuing anyway and printing one-row result tables.",
                title="Single-row results",
                border_style="yellow",
            )
        )

    selected_total_runs = len(filtered_all)

    if "success" in filtered_all.columns:
        selected_successful_runs = len(
            filtered_all[filtered_all["success"] == True]
        )
        selected_failed_runs = selected_total_runs - selected_successful_runs
    else:
        selected_successful_runs = len(filtered_success)
        selected_failed_runs = 0

    overview = (
        f"[bold]Successful runs used:[/bold] {selected_successful_runs}\n"
        f"[bold]Failed runs ignored:[/bold] {selected_failed_runs}\n"
        f"[bold]Varying parameter:[/bold] {vary}\n"
        f"[bold]Values:[/bold] "
        f"{', '.join(map(str, sorted_unique_values(filtered_success[vary])))}\n"
        f"[bold]Seeds averaged over:[/bold] "
        f"{', '.join(seed_cols) if seed_cols else 'none'}"
    )

    console.print(
        Panel(
            overview,
            title="Batch overview",
            border_style="green",
        )
    )

    console.print()
    console.rule("[bold cyan]Selected experiment group[/bold cyan]")
    console.print(static_config_panel(selected_config))

    for metric, title in available_metrics.items():
        console.print(
            metric_table(
                group=filtered_success,
                vary=vary,
                metric=metric,
                metric_title=title,
                precision=args.precision,
            )
        )

    console.print(counts_table(filtered_success, vary))


if __name__ == "__main__":
    main()