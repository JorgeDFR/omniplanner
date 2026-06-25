#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


GRAPH_SIZE_X_COLUMN = "graph_size"
GRAPH_SIZE_COLUMNS = [
    "num_nodes",
    "num_objects",
    "num_regions",
]
GRAPH_SIZE_TOLERANCE = 1e-9

VARY_OPTIONS = [
    GRAPH_SIZE_X_COLUMN,
    "num_nodes",
    "num_objects",
    "num_regions",
    "goal_conjunctions",
    "goal_disjunctions",
    "pddl_domain",
    "pddl_sampler",
    "pddl_solver",
]

# graph_size is derived, so it is not a fixed configuration option.
# When graph_size is the varying parameter, num_nodes/num_objects/num_regions
# are also not fixed manually; they are allowed to vary together as scale dimensions.
STATIC_OPTIONS = [option for option in VARY_OPTIONS if option != GRAPH_SIZE_X_COLUMN]

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

COLUMN_LABELS = {
    "graph_size": "Graph scale",
    "num_nodes": "Number of nodes",
    "num_objects": "Number of objects",
    "num_regions": "Number of regions",
    "goal_conjunctions": "Goal conjunctions",
    "goal_disjunctions": "Goal disjunctions",
    "pddl_domain": "PDDL domain",
    "pddl_sampler": "PDDL sampler",
    "pddl_solver": "PDDL solver",
    "graph_seed": "Graph seed",
    "goal_seed": "Goal seed",
}

console = Console()


class UserInputError(RuntimeError):
    """Raised for user-correctable CLI/configuration errors."""


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

        with open(config_file, encoding="utf-8") as f:
            config = json.load(f)

        with open(result_file, encoding="utf-8") as f:
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

    if isinstance(value, (np.integer, int)):
        return True

    if isinstance(value, (np.floating, float)):
        return float(value).is_integer()

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

    if isinstance(value, (np.floating, float)):
        return f"{float(value):.{precision}f}"

    return str(value)


def normalize_value(value) -> str:
    if pd.isna(value):
        return "<NA>"

    if isinstance(value, (np.bool_, bool)):
        return str(bool(value)).lower()

    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return str(int(value))

    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isfinite(value) and value.is_integer():
            return str(int(value))
        return str(value)

    return str(value)


def as_float_if_numeric(value):
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return float(value)

    if isinstance(value, (np.floating, float)) and math.isfinite(float(value)):
        return float(value)

    try:
        as_float = float(str(value))
    except (TypeError, ValueError):
        return None

    if math.isfinite(as_float):
        return as_float

    return None


def sorted_unique_values(series: pd.Series):
    values = series.dropna().unique().tolist()

    def key(value):
        numeric = as_float_if_numeric(value)
        if numeric is not None:
            return (0, numeric, normalize_value(value))
        return (1, normalize_value(value))

    return sorted(values, key=key)


def candidate_filter_values(raw: str) -> set[str]:
    candidates: list[object] = [raw]
    lower = raw.lower()

    if lower == "true":
        candidates.append(True)
    elif lower == "false":
        candidates.append(False)

    for caster in (int, float):
        try:
            candidates.append(caster(raw))
        except ValueError:
            pass

    try:
        candidates.append(json.loads(raw))
    except json.JSONDecodeError:
        pass

    return {normalize_value(value) for value in candidates}


def mask_equal_to_raw(series: pd.Series, raw: str) -> pd.Series:
    wanted = candidate_filter_values(raw)
    return series.map(normalize_value).isin(wanted)


def pretty_column(column: str) -> str:
    return COLUMN_LABELS.get(column, column.replace("_", " ").capitalize())


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

    table.add_column(pretty_column(vary), justify="right", style="bold")
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
    table.add_column(pretty_column(vary), justify="right", style="bold")
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
        "\n".join(lines) if lines else "none",
        title="Static configuration",
        border_style="cyan",
    )


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


def validate_graph_size_columns(df: pd.DataFrame) -> None:
    missing = [column for column in GRAPH_SIZE_COLUMNS if column not in df.columns]
    if missing:
        raise UserInputError(
            f"Cannot use {GRAPH_SIZE_X_COLUMN!r}; missing required column(s): "
            f"{', '.join(missing)}."
        )


def size_tuple_from_row(row: pd.Series) -> tuple[int | float, ...]:
    values = []
    for column in GRAPH_SIZE_COLUMNS:
        value = row[column]
        as_float = float(value)
        if math.isfinite(as_float) and as_float.is_integer():
            values.append(int(as_float))
        else:
            values.append(as_float)
    return tuple(values)


def format_size_tuple(size_tuple: tuple[int | float, ...]) -> str:
    return "(" + ", ".join(
        f"{column}={normalize_value(value)}"
        for column, value in zip(GRAPH_SIZE_COLUMNS, size_tuple)
    ) + ")"


def graph_size_tuples(df: pd.DataFrame) -> list[tuple[int | float, ...]]:
    validate_graph_size_columns(df)

    numeric = df.copy()
    for column in GRAPH_SIZE_COLUMNS:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

    numeric = numeric.dropna(subset=GRAPH_SIZE_COLUMNS)
    if numeric.empty:
        return []

    unique_sizes = (
        numeric[GRAPH_SIZE_COLUMNS]
        .drop_duplicates()
        .sort_values(GRAPH_SIZE_COLUMNS)
        .reset_index(drop=True)
    )

    return [size_tuple_from_row(row) for _, row in unique_sizes.iterrows()]


def graph_size_available(df: pd.DataFrame) -> bool:
    try:
        return len(graph_size_tuples(df)) > 1
    except UserInputError:
        return False


def parse_size_tuple_choice(raw: str) -> tuple[float, float, float] | None:
    cleaned = raw.strip().strip("()")
    if not cleaned:
        return None

    parts = [part.strip() for part in cleaned.split(",")]
    if len(parts) != 3:
        return None

    values: list[float] = []
    for part in parts:
        if "=" in part:
            part = part.split("=", maxsplit=1)[1].strip()
        try:
            value = float(part)
        except ValueError:
            return None
        if not math.isfinite(value) or value <= 0:
            return None
        values.append(value)

    return tuple(values)  # type: ignore[return-value]


def ask_graph_size_base(df: pd.DataFrame) -> dict[str, float]:
    available = graph_size_tuples(df)
    if not available:
        raise UserInputError("No numeric graph-size tuples were found.")

    console.print(f"\n[bold cyan]{GRAPH_SIZE_X_COLUMN} base[/bold cyan]")
    console.print(
        "Choose the scale-1 graph. Graph scale will be computed as "
        "num_nodes / base_num_nodes, while num_objects and num_regions are "
        "allowed to differ by up to 1 from the proportional rounded scale."
    )
    console.print("Available graph-size tuples:")

    for index, size_tuple in enumerate(available, start=1):
        console.print(f"  {index}. {format_size_tuple(size_tuple)}")

    while True:
        raw = input(
            "Choose scale-1 graph by number, or type nodes,objects,regions: "
        ).strip()

        if raw.isdigit():
            index = int(raw)
            if 1 <= index <= len(available):
                selected = tuple(float(value) for value in available[index - 1])
                return dict(zip(GRAPH_SIZE_COLUMNS, selected))

        parsed = parse_size_tuple_choice(raw)
        if parsed is not None:
            normalized_available = {
                tuple(float(value) for value in size_tuple)
                for size_tuple in available
            }
            if parsed in normalized_available:
                return dict(zip(GRAPH_SIZE_COLUMNS, parsed))

        console.print(
            "[red]Invalid value.[/red] Choose one of the listed numbers or "
            "enter an exact tuple such as 96,65,5."
        )


def add_graph_size_column_ratio(
    df: pd.DataFrame,
    graph_size_base: dict[str, float],
) -> pd.DataFrame:
    """Use num_nodes/base_num_nodes as scale, allowing rounded object/region counts."""
    scaled = df.copy()
    for column in GRAPH_SIZE_COLUMNS:
        scaled[column] = pd.to_numeric(scaled[column], errors="coerce")

    reference = scaled["num_nodes"] / graph_size_base["num_nodes"]
    valid = reference.notna() & (reference > 0)

    # Object and region counts often come from rounded scaled values. Allow an
    # absolute difference up to 1 count from the value predicted by num_nodes scale.
    for column in ["num_objects", "num_regions"]:
        expected = graph_size_base[column] * reference
        actual = scaled[column]
        valid = valid & actual.notna()
        valid = valid & ((actual - expected).abs() <= 1.0 + GRAPH_SIZE_TOLERANCE)

    scaled = scaled[valid].copy()
    if scaled.empty:
        base_tuple = tuple(graph_size_base[column] for column in GRAPH_SIZE_COLUMNS)
        raise UserInputError(
            f"No rows match a rounded proportional graph scale for base "
            f"{format_size_tuple(base_tuple)}."
        )

    scaled[GRAPH_SIZE_X_COLUMN] = reference.loc[scaled.index].round(6)
    return scaled


def config_overview_table(df: pd.DataFrame) -> Table:
    table = Table(title="Found experiment configuration parameters")
    table.add_column("Parameter", style="bold")
    table.add_column("# values", justify="right")
    table.add_column("Values")

    for col in VARY_OPTIONS:
        if col == GRAPH_SIZE_X_COLUMN:
            if graph_size_available(df):
                values = graph_size_tuples(df)
                values_text = ", ".join(format_size_tuple(value) for value in values)
                table.add_row(
                    GRAPH_SIZE_X_COLUMN,
                    str(len(values)),
                    "derived from " + values_text,
                )
            continue

        if col not in df.columns:
            continue

        values = sorted_unique_values(df[col])
        table.add_row(
            col,
            str(len(values)),
            ", ".join(map(str, values)),
        )

    return table


def possible_vary_options(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    multiple_values = []
    single_values = []

    for col in VARY_OPTIONS:
        if col == GRAPH_SIZE_X_COLUMN:
            try:
                value_count = len(graph_size_tuples(df))
            except UserInputError:
                continue
        elif col in df.columns:
            value_count = df[col].nunique(dropna=True)
        else:
            continue

        if value_count > 1:
            multiple_values.append(col)
        elif value_count == 1:
            single_values.append(col)

    return multiple_values, single_values


def main():
    args = parse_args()

    try:
        df_all = load_results(args.batch_dir)

        if "success" in df_all.columns:
            df_success = df_all[df_all["success"] == True].copy()  # noqa: E712
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
                f"Special option: choose [bold]{GRAPH_SIZE_X_COLUMN}[/bold] to "
                "analyze proportional graph scale instead of fixing num_nodes, "
                "num_objects, and num_regions separately.\n\n"
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

        multi_value_vary_options, single_value_vary_options = possible_vary_options(df_success)

        if multi_value_vary_options:
            vary = ask_choice(
                "varying parameter",
                multi_value_vary_options,
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

        selected_config = {}
        graph_size_base: dict[str, float] | None = None

        if vary == GRAPH_SIZE_X_COLUMN:
            graph_size_base = ask_graph_size_base(df_success)
            df_success = add_graph_size_column_ratio(df_success, graph_size_base)
            df_all = add_graph_size_column_ratio(df_all, graph_size_base)
            selected_config["graph_size_base"] = format_size_tuple(
                tuple(graph_size_base[column] for column in GRAPH_SIZE_COLUMNS)
            )

        static_cols = [
            c for c in STATIC_OPTIONS
            if c in df_success.columns and c != vary
        ]

        if vary == GRAPH_SIZE_X_COLUMN:
            static_cols = [c for c in static_cols if c not in GRAPH_SIZE_COLUMNS]

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
                    mask_equal_to_raw(filtered_success[col], normalize_value(selected_value))
                ].copy()

                filtered_all = filtered_all[
                    mask_equal_to_raw(filtered_all[col], normalize_value(selected_value))
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
                filtered_all[filtered_all["success"] == True]  # noqa: E712
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
            f"{', '.join(fmt(value, args.precision, vary) for value in sorted_unique_values(filtered_success[vary]))}\n"
            f"[bold]Seeds averaged over:[/bold] "
            f"{', '.join(seed_cols) if seed_cols else 'none'}"
        )

        if graph_size_base:
            overview += "\n[bold]Graph-size definition:[/bold] graph_size = num_nodes / base_num_nodes"

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

    except UserInputError as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()