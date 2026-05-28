"""Unpaced (no-stimulus) stabilisation plots.

Reads unpaced simulation outputs from DATA_DIR, writes all figures and intermediate
xlsx artefacts into OUTPUT_DIR. Both directories are resolved relative to this script
so the project works out of the box on a fresh clone.

The state-to-class mapping used by StrategyB and the heatmap functions is read from
STATE_CLASS_MAPPING (a 12-sheet xlsx with one sheet per model, columns: state, Class).
Time-to-stabilisation values themselves are computed fresh from simulation data via
`Stabilisation_Time_Across_Models`; the mapping file only contributes the Class column.
"""

# === imports ===
import os
import glob
import math
import re
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path

# === directories ===
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
# Where highthroughput_stabilisation_script.py writes its unpaced output.
DATA_DIR = os.path.join(_REPO_ROOT, "Outputs", "Unpaced")
# Plot + xlsx output directory; created lazily on first save.
OUTPUT_DIR = os.path.join(_REPO_ROOT, "Outputs", "Unpaced Plots")
# Static state -> functional class mapping (one sheet per model).
STATE_CLASS_MAPPING = os.path.join(_SCRIPT_DIR, "state_class_mapping.xlsx")


def _ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _savefig(fig, filename, **kwargs):
    _ensure_output_dir()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), **kwargs)


def _out_path(filename):
    """Return a full path inside OUTPUT_DIR for non-figure outputs (xlsx, etc.)."""
    _ensure_output_dir()
    return os.path.join(OUTPUT_DIR, filename)


def categorise_times_xlsx(times_xlsx, mapping_xlsx, output_xlsx):
    """Build a `categorised` xlsx by joining freshly-computed stabilisation times
    with the static state->class mapping.

    Each sheet in `times_xlsx` must have columns `state, time_to_stabilisation_ms`.
    Each sheet in `mapping_xlsx` (same sheet names) must have `state, Class`.
    Output sheets contain `state, time_to_stabilisation_ms, Class` — rows with no
    matching mapping entry are kept with Class=NaN."""
    times = pd.ExcelFile(times_xlsx)
    mapping = pd.ExcelFile(mapping_xlsx)
    mapping_sheets = set(mapping.sheet_names)
    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        for sheet in times.sheet_names:
            t_df = pd.read_excel(times_xlsx, sheet_name=sheet)
            if sheet in mapping_sheets:
                m_df = pd.read_excel(mapping_xlsx, sheet_name=sheet)[['state', 'Class']]
                merged = t_df.merge(m_df, on='state', how='left')
            else:
                merged = t_df.copy()
                merged['Class'] = pd.NA
            merged.to_excel(writer, sheet_name=sheet, index=False)
    return output_xlsx


# === helper functions ===

def _format_subscripts(label):
    def repl(match):
        return f"{match.group(1)}$_{{{match.group(2)}}}$"
    return re.sub(r"(\S+)_([^\s]+)", repl, label)


def _format_state_title(state_name):
    """Returns a bold LaTeX string with subscript formatting if the name contains an underscore."""
    if "_" in state_name:
        base, sub = state_name.split("_", 1)
        return rf"$\mathbf{{{base}_{{{sub}}}}}$"
    else:
        return rf"$\mathbf{{{state_name}}}$"


def state_stabilisation_plot(base_dir):
    """
    Plots steady-state stabilisation of all model state variables (no stimulus) across
    1-3 cell-type subfolders in base_dir. Each state variable gets its own subplot in an 8-column grid.
    """

    cell_type_dirs = [
        os.path.join(base_dir, d)
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]

    if not (1 <= len(cell_type_dirs) <= 3):
        raise ValueError("Expected between 1 and 3 cell-type folders.")

    cmap = matplotlib.colormaps["Set2"]
    colors = {
        "EPI": cmap(0 / 3),
        "ENDO": cmap(1 / 3),
        "M": cmap(2 / 3),
        "Generic": cmap(4 / 9),
    }

    allowed_cell_types = set(colors.keys())

    cell_dfs = {}

    for cell_dir in cell_type_dirs:

        # Extract cell type as a STRING (not the regex match object)
        match = re.search(r"(ENDO|EPI| M |Generic)", os.path.basename(cell_dir))
        if match is None:
            raise ValueError(
                f"Could not detect cell type in folder name: {cell_dir}"
            )

        cell_type = match.group(1)

        if cell_type not in allowed_cell_types:
            raise ValueError(f"Unsupported cell type: {cell_type}")

        parquet_files = sorted(glob.glob(os.path.join(cell_dir, "*.parquet")))
        if not parquet_files:
            raise ValueError(f"No parquet files found in {cell_dir}")

        df = pd.concat(
            (pd.read_parquet(f) for f in parquet_files),
            ignore_index=True
        )

        # Downsample: keep 1, skip next 5
        df = df.iloc[::6].reset_index(drop=True)

        cell_dfs[cell_type] = df

    example_df = next(iter(cell_dfs.values()))
    if "time" not in example_df.columns:
        raise ValueError("Column 'time' not found.")

    states = [c for c in example_df.columns if c != "time"]
    n_states = len(states)

    n_cols = 8
    n_rows = math.ceil(n_states / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4 * n_cols, 4 * n_rows),
        sharex=False
    )

    axes = axes.flatten()

    for i, state in enumerate(states):
        ax = axes[i]

        for ct, df in cell_dfs.items():
            ax.plot(
                df["time"],
                df[state],
                color=colors[ct],
                linewidth=1
            )

        ax.set_title(_format_state_title(state), fontsize=12)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Scientific notation without offset
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-3, 3))
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.get_offset_text().set_visible(False)

        ax.tick_params(axis="x", which="both", labelbottom=True)
        ax.set_ylabel("")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    legend_handles = [
        Line2D([0], [0], color=colors[ct], lw=2, label=ct)
        for ct in cell_dfs.keys()
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=len(legend_handles),
        frameon=False,
        fontsize=12
    )

    fig.supxlabel(
        "Time (ms)",
        fontsize=13,
        fontweight="bold"
    )

    plt.tight_layout(rect=[0, 0, 1, 1])

    return fig


def compute_stabilisation_time(time, delta, epsilon):
    """
    Compute stabilisation time given a normalised deviation trajectory.
    Right-censors at final time if never stabilised.
    """
    below = delta < epsilon

    for i in range(len(delta)):
        if np.all(below[i:]):
            return time[i]

    return time[-1]  # right-censored


# === plotting / IO functions ===

def determine_steady_state(folder_path, steady_window_ms=2000):
    """
    Determine steady-state values for each state variable by averaging
    over the last steady_window_ms of the simulation.
    """
    parquet_files = sorted(
        glob.glob(os.path.join(folder_path, "simulation_chunk_*.parquet"))
    )

    if len(parquet_files) == 0:
        raise FileNotFoundError("No parquet files found in the specified folder.")

    # Read and combine
    df = pd.concat(
        [pd.read_parquet(f) for f in parquet_files],
        ignore_index=True
    )

    if "time" not in df.columns:
        raise ValueError("Dataframe must contain a 'time' column.")

    # Identify steady-state window
    t_max = df["time"].max()
    df_ss = df[df["time"] >= (t_max - steady_window_ms)]

    # Exclude time column
    state_cols = [c for c in df_ss.columns if c != "time"]

    # Mean steady state
    steady_state = df_ss[state_cols].mean()

    return steady_state


def Plot_Normalised_deviation_trajetctories_all_cells(parent_folder_path, model_name, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    # Default colour scheme if none provided

    cmap = matplotlib.colormaps["Set2"]
    colors = {
        "EPI": cmap(0 / 3),
        "ENDO": cmap(1 / 3),
        "M": cmap(2 / 3)
    }

    n_states = None
    thin_step = 3 if "morotti" in model_name.lower() else 25

    # ---- Columns to drop per model ----
    cols_to_drop = set()
    if "morotti" in model_name.lower():
        cols_to_drop.update([
            "INa_mk1", "INa_mk2", "INa_mk3", "INa_mk4", "INa_mk5", "INa_mk6",
            "INa_mk7", "INa_mk8", "INa_mk9", "INa_mk10", "INa_mk11", "INa_mk12",
            "rtos",'f', 'fcaBj', 'fcaBsl','CaM_ecc', 'LCC_PKAp', 'RyR2809p', 'Ca2CaMB_dyad', 'Ca4CaMB_dyad', 'CaMB_dyad'
        ])
    if "bps" in model_name.lower():  # catches BPS and BPSLand
        cols_to_drop.update([
            "dD_dutta", "dval10", "dval7", "dval8", "dval9", "val2"
        ])
    if model_name.lower() == "doste†":
        cols_to_drop.update([
            "dcond1","dcond2","dcond3","dcond4","dcond5","dcond6","dcond7",
            "dcond8","dcond9","dcond10","dcond11","dcond12","dcond13","dcond14",
            "dcond15","dcond16","dcond17","dcond18","dcond19","dcond20","dcond21",
            "dcond22","dcond23","dcond24","dcond25","dcond26","dcond27","dcond28",
            "dcond29","dcond30","dcond31","dcond32","dcond33","dcond34","dcond35",
            "dcond36","dcond37","dcond38","dcond39","dcond40","dcond41","dcond42",
            "dcond43","dcond44","dcond45","dcond46","dcond47","dcond48","dcond49",
            "dcond50","dcond51","dcond52","dcond53","dcond54","dcond55","dcond56",
            "dcond57"
        ])

    fixed_at_one = set()

    subfolders = [
        os.path.basename(p) for p in glob.glob(
            os.path.join(parent_folder_path, "* sim chunks * at *ms")
        )
        if os.path.isdir(p)
    ]

    for cell_id in subfolders:
        cell_path = os.path.join(parent_folder_path, cell_id)

        parquet_files = glob.glob(
            os.path.join(cell_path, "simulation_chunk_*.parquet")
        )
        if not parquet_files:
            continue

        df = pd.concat(
            [pd.read_parquet(f) for f in parquet_files],
            ignore_index=True
        )

        drop_here = cols_to_drop & set(df.columns)
        if drop_here:
            df = df.drop(columns=list(drop_here))

        steady_state = determine_steady_state(cell_path)

        state_cols = [c for c in df.columns if c != "time"]
        if n_states is None:
            n_states = len(state_cols)
        df["time"] = df["time"] / 1000.0
        df_thinned = df.iloc[::thin_step].copy()

        # ---- Compute max deviation for normalisation ----
        max_dev = {}
        for state in state_cols:
            max_dev[state] = np.max(
                np.abs(df[state] - steady_state[state])
            )

        # ---- Infer cell type from folder name (token between 'chunks' and 'at') ----
        tokens = cell_id.split()
        cell_type = None
        try:
            chunks_idx = tokens.index("chunks")
            at_idx = tokens.index("at", chunks_idx + 1)
            result = " ".join(tokens[chunks_idx + 1:at_idx]).upper()
        except ValueError:
            result = cell_id.upper()
        if "EPI" in result:
            cell_type = "EPI"
        elif "ENDO" in result:
            cell_type = "ENDO"
        elif "M" in result:
            cell_type = "M"

        # ---- Plot each state directly (avoids melt + concat memory cost) ----
        time_vals = df_thinned["time"].values
        for state in state_cols:
            if max_dev[state] < 1e-15:
                delta = np.zeros(len(df_thinned))
            else:
                delta = (
                    np.abs(df_thinned[state].values - steady_state[state]) /
                    max_dev[state]
                )
                if delta[-1] >= 0.99:
                    fixed_at_one.add(state)
                    if "morotti" in model_name.lower():
                        print(f"  [{model_name}] {cell_id} | {state}: max_dev = {max_dev[state]:.6g}")
            ax.plot(
                time_vals,
                delta,
                color=colors[cell_type],
                linewidth=0.8,
                alpha=0.5
            )

    if fixed_at_one:
        print(f"[{model_name}] States fixed at normalised deviation = 1: {sorted(fixed_at_one)}")

    ax.set_title(f"{model_name}  ({n_states})", fontsize=23, fontweight="bold", loc="left")
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    return ax


def StrategyA(folder_path, model_order):
    """
    Parameters
    ----------
    folder_path : str
        Parent directory containing model subfolders.
    model_order : list of str
        Ordered list of model folder names to plot.
        Example: ["BARS", "BARSt", "Morotti", "Morottit"]
    """

    # Fixed grid: 6 rows × 2 columns
    fig, axes = plt.subplots(3, 4, figsize=(32, 18), sharex=True, sharey=True, dpi = 300)
    axes = axes.flatten()

    for ax, model_name in zip(axes, model_order):
        sub_folder = os.path.join(folder_path, model_name)

        if not os.path.isdir(sub_folder):
            raise ValueError(f"Model folder not found: {sub_folder}")
        print(model_name)
        Plot_Normalised_deviation_trajetctories_all_cells(
            parent_folder_path=sub_folder,
            model_name=model_name,
            ax=ax
        )

    # Remove unused axes if fewer than 12 models
    for ax in axes[len(model_order):]:
        ax.axis("off")

    # Shared axis labels
    fig.text(0.5, 0.04, "Time (s)", ha="center", fontsize=30, fontweight = "bold")
    fig.text(0.04,0.5,"Normalised Deviation (unitless)",fontweight = "bold",va="center",rotation="vertical",fontsize=30)
    cmap = matplotlib.colormaps["Set2"]
    colors = {
        "EPI": cmap(0 / 3),
        "ENDO": cmap(1 / 3),
        "M": cmap(2 / 3)
    }

    legend_handles = [
        mlines.Line2D([], [], color=colors["EPI"], label="EPI", linewidth=5),
        mlines.Line2D([], [], color=colors["ENDO"], label="ENDO", linewidth=5),
        mlines.Line2D([], [], color=colors["M"], label="M", linewidth=5),
    ]

    fig.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=35
    )

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.93])
    return fig


def Stabilisation_Time_Across_Models(model_dirs, model_names, epsilon=1e-3, steady_window_ms=2000):
    """
    Create violin plots with embedded boxplots of stabilisation times
    across models, aggregating across cell types.

    Returns:
        fig, ax
        stabilisation_times_per_model (list of dicts)
    """

    all_plot_data = []
    stabilisation_times_per_model = []

    for model_dir, model_name in zip(model_dirs, model_names):
        cell_dirs = [
        os.path.join(model_dir, d)
        for d in os.listdir(model_dir)
        if os.path.isdir(os.path.join(model_dir, d))
        and not d.startswith("OLD_")
        ]
        print(cell_dirs)

        # ---- Columns to drop per model (same rules as Plot function) ----
        cols_to_drop = set()
        if "morotti" in model_name.lower():
            cols_to_drop.update([
                "INa_mk1", "INa_mk2", "INa_mk3", "INa_mk4", "INa_mk5", "INa_mk6",
                "INa_mk7", "INa_mk8", "INa_mk9", "INa_mk10", "INa_mk11", "INa_mk12",
                "rtos", "f", "fcaBj", "fcaBsl", "CaM_ecc", "LCC_PKAp", "RyR2809p",
                "Ca2CaMB_dyad", "Ca4CaMB_dyad", "CaMB_dyad"
            ])
        if "bps" in model_name.lower():
            cols_to_drop.update([
                "dD_dutta", "dval10", "dval7", "dval8", "dval9", "val2"
            ])
        if model_name.lower() == "doste†":
            cols_to_drop.update([
                "dcond1","dcond2","dcond3","dcond4","dcond5","dcond6","dcond7",
                "dcond8","dcond9","dcond10","dcond11","dcond12","dcond13","dcond14",
                "dcond15","dcond16","dcond17","dcond18","dcond19","dcond20","dcond21",
                "dcond22","dcond23","dcond24","dcond25","dcond26","dcond27","dcond28",
                "dcond29","dcond30","dcond31","dcond32","dcond33","dcond34","dcond35",
                "dcond36","dcond37","dcond38","dcond39","dcond40","dcond41","dcond42",
                "dcond43","dcond44","dcond45","dcond46","dcond47","dcond48","dcond49",
                "dcond50","dcond51","dcond52","dcond53","dcond54","dcond55","dcond56",
                "dcond57"
            ])

        state_times_across_cells = {}

        for cell_dir in cell_dirs:
            parquet_files = glob.glob(
                os.path.join(cell_dir, "simulation_chunk_*.parquet")
            )
            if not parquet_files:
                continue

            # Read and combine
            df = pd.concat(
                [pd.read_parquet(f) for f in parquet_files],
                ignore_index=True
            )

            if "time" not in df.columns:
                raise ValueError("Missing 'time' column")

            drop_here = cols_to_drop & set(df.columns)
            if drop_here:
                df = df.drop(columns=list(drop_here))

            # Steady state
            steady_state = determine_steady_state(
                cell_dir, steady_window_ms
            )

            print(steady_state)
            state_cols = [c for c in df.columns if c != "time"]
            df_thinned = df.iloc[::11].copy()
            # Normalisation denominator
            max_dev = {
                s: np.max(np.abs(df_thinned[s] - steady_state[s]))
                for s in state_cols
            }

            # Compute stabilisation per state

            for s in state_cols:

                if max_dev[s] < 1e-13:
                    delta = pd.Series(0.0, index=df_thinned.index)
                else:
                    delta = (df_thinned[s] - steady_state[s]).abs() / max_dev[s]

                T = compute_stabilisation_time(
                    df_thinned["time"].values,
                    delta.values,
                    epsilon
                )

                state_times_across_cells.setdefault(s, []).append(T)

        # Aggregate across cells
        mean_times = {
            s: np.mean(times)
            for s, times in state_times_across_cells.items()
        }

        stabilisation_times_per_model.append(mean_times)

    return stabilisation_times_per_model


def StrategyB(excel_path, plot_type, log10_y, model_order):
    """
    plot_type options:
        - 'box_plot'
        - 'violin_plot'
        - 'scatter_plot'
        - 'violin_with_boxplot'

    model_order:
        List specifying the order of Excel sheet names on the x-axis.
        If None, the natural sheet order is used.
    """

    # ---- Read all sheets and build long dataframe ----
    xls = pd.ExcelFile(excel_path)

    records = []
    counts = {}  # store number of rows (states) per sheet

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Count rows (exclude header automatically since pandas handles it)
        counts[sheet_name] = df.shape[0]

        # Expect columns: state, time_to_stabilisation_ms
        for _, row in df.iterrows():
            records.append({
                "model": sheet_name,
                "state": row["state"],
                # convert ms -> seconds
                "time_s": row["time_to_stabilisation_ms"] / 1000.0
            })

    plot_df = pd.DataFrame(records)

    # ---- Apply user-defined order ----
    if model_order is not None:
        # Keep only models that exist in data
        model_order = [m for m in model_order if m in plot_df["model"].unique()]
    else:
        model_order = list(plot_df["model"].unique())

    # ---- Create plot ----
    fig, ax = plt.subplots(figsize=(15, 6))

    if plot_type == "box_plot":

        sns.boxplot(
            data=plot_df,
            x="model",
            y="time_s",
            order=model_order,
            ax=ax,
            color="lightgrey"
        )

    elif plot_type == "violin_plot":

        sns.violinplot(
            data=plot_df,
            x="model",
            y="time_s",
            order=model_order,
            linewidth=2,
            cut=0,
            ax=ax
        )

    elif plot_type == "violin_with_boxplot":

        # ---- Violin plot ----
        sns.violinplot(
            data=plot_df,
            x="model",
            y="time_s",
            order=model_order,
            cut=0,
            linewidth=3,
            width=0.9,
            inner=None,
            color="#c0392b",
            ax=ax
        )

        # ---- Embedded boxplot ----
        sns.boxplot(
            data=plot_df,
            x="model",
            y="time_s",
            order=model_order,
            width=0.05,
            showcaps=True,
            boxprops={
                "facecolor": "#0b1c2d",
                "edgecolor": "#0b1c2d",
                "linewidth": 1
            },
            whiskerprops={
                "color": "#0b1c2d",
                "linewidth": 1
            },
            capprops={
                "color": "#0b1c2d",
                "linewidth": 1
            },
            medianprops={
                "color": "yellow",
                "linewidth": 2,
                "linestyle": "--"
            },
            ax=ax
        )

    elif plot_type == "scatter_plot":

        rng = np.random.default_rng(seed=42)
        jitter_strength = 0.35

        models = model_order

        for i, model in enumerate(models):
            sub = plot_df[plot_df["model"] == model]

            x_jittered = (
                np.full(len(sub), i) +
                rng.uniform(-jitter_strength, jitter_strength, size=len(sub))
            )

            colors = np.where(
                sub["time_s"] == 0,
                "green",
                np.where(sub["time_s"] > (1_850_000 / 1000.0), "red", "grey")
            )

            ax.scatter(
                x_jittered,
                sub["time_s"],
                c=colors,
                alpha=0.7,
                s=25,
                edgecolors="none"
            )

        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models,size = 14,fontweight = "bold")

        legend_elements = [
            Line2D([0], [0], marker='o', color='none',
                   markerfacecolor='red', markersize=8, alpha=0.7,
                   label='No stabilisation'),

            Line2D([0], [0], marker='o', color='none',
                   markerfacecolor='green', markersize=8, alpha=0.7,
                   label='Instantaneous stabilisation'),

            Line2D([0], [0], marker='o', color='none',
                   markerfacecolor='grey', markersize=8, alpha=0.7,
                   label='Eventual stabilisation')
        ]

        ax.legend(
            handles=legend_elements,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.16),
            ncol=3,
            fontsize=11,
            frameon=False
        )

    else:
        raise ValueError(f"Unknown plot_type: {plot_type}")

    # ---- Axis formatting ----
    ax.set_ylabel("Time to stabilisation (s)",size = 11,fontweight = "bold")

    if log10_y:
        ax.set_yscale("log")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ---- Gridlines ----
    ax.yaxis.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.xaxis.grid(False)

    # ---- Annotate counts above each category ----
    ymax = plot_df["time_s"].max()
    y_text = ymax * 1.05 if not log10_y else ymax * 1.2

    for i, model in enumerate(model_order):
        if model in counts:
            ax.text(
                i,
                y_text,
                f"({counts[model]})",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight = "bold"
            )

    plt.tight_layout()
    return fig


def summarise_stabilisation_by_class(
    input_excel_path,
    output_excel_path
):
    """
    Reads an Excel file with multiple sheets (one per model), each containing:
      - 'state'
      - 'time_to_stabilisation_ms'
      - 'Class'

    Produces an Excel file with four sheets:
      1) Mean stabilisation time per class × model
      2) Max stabilisation time per class × model
      3) State counts per class × model
      4) State categories (comma-separated state names)
    """

    # Read all model sheets
    model_sheets = pd.read_excel(input_excel_path, sheet_name=None)

    models = list(model_sheets.keys())

    # Collect all functional classes across models
    all_classes = set()
    for df in model_sheets.values():
        if "Class" in df.columns:
            classes = (
                df["Class"]
                .dropna()
                .astype(str)
                .str.strip()
            )
            classes = classes[classes != ""]
            all_classes.update(classes)

    all_classes = sorted(all_classes)

    # Initialise output tables
    mean_df   = pd.DataFrame(index=all_classes, columns=models, dtype=float)
    max_df    = pd.DataFrame(index=all_classes, columns=models, dtype=float)
    states_df = pd.DataFrame(index=all_classes, columns=models, dtype=object)

    # Populate tables
    for model, df in model_sheets.items():

        if not {"state", "time_to_stabilisation_ms", "Class"}.issubset(df.columns):
            continue

        df = df.copy()
        df["Class"] = df["Class"].astype(str).str.strip()
        df = df[df["Class"] != ""]

        for cls, sub in df.groupby("Class"):

            mean_df.loc[cls, model] = sub["time_to_stabilisation_ms"].mean()
            max_df.loc[cls, model]  = sub["time_to_stabilisation_ms"].max()

            states_df.loc[cls, model] = ", ".join(
                sub["state"].astype(str)
            )

    # ---------- State counts ----------
    count_df = states_df.copy()

    for col in count_df.columns:
        count_df[col] = (
            count_df[col]
            .fillna("")
            .apply(lambda x: 0 if x.strip() == "" else len(x.split(",")))
        )

    # ---------- Row ordering by sparsity ----------
    # Fewer NaNs = more populated = appear earlier
    nan_counts = mean_df.isna().sum(axis=1)
    ordered_index = nan_counts.sort_values().index

    mean_df   = mean_df.loc[ordered_index]
    max_df    = max_df.loc[ordered_index]
    count_df  = count_df.loc[ordered_index]
    states_df = states_df.loc[ordered_index]

    # ---------- Write output ----------
    with pd.ExcelWriter(output_excel_path, engine="openpyxl") as writer:
        mean_df.to_excel(writer, sheet_name="Mean stabilisation time")
        max_df.to_excel(writer, sheet_name="Max stabilisation time")
        count_df.to_excel(writer, sheet_name="State counts")
        states_df.to_excel(writer, sheet_name="State categories")


def average_time_per_model(excel_path):
    """
    Computes the average time to stabilisation per model (Excel sheet).

    Steps:
    1. For each sheet, compute the mean time_to_stabilisation_ms per Class
    2. Take the mean of those class means to get one value per model

    Parameters
    ----------
    excel_path : str
        Path to the Excel file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - model_name
        - average_time_to_stabilisation_ms
    """

    # Load Excel file
    xls = pd.ExcelFile(excel_path)

    results = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Safety checks
        required_cols = {"time_to_stabilisation_ms", "Class"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Sheet '{sheet_name}' is missing required columns: {required_cols}"
            )

        # Mean per class
        class_means = (
            df.groupby("Class")["time_to_stabilisation_ms"]
              .mean()
        )

        # Mean of class means (one value per model)
        model_mean = class_means.mean()

        results.append({
            "model": sheet_name,
            "mean_time_s": model_mean/1000
        })

    return pd.DataFrame(results)


def weighted_average_time_per_model(excel_path):
    """
    Computes the weighted average time to stabilisation per model (Excel sheet).

    Steps:
    1. For each sheet, compute the mean time_to_stabilisation_ms per Class
    2. Count number of entities per Class
    3. Compute weighted average of class means using class counts

    Parameters
    ----------
    excel_path : str
        Path to the Excel file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - model
        - weighted_mean_time_s
    """

    # Load Excel file
    xls = pd.ExcelFile(excel_path)

    results = []

    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)

        # Safety checks
        required_cols = {"time_to_stabilisation_ms", "Class"}
        if not required_cols.issubset(df.columns):
            raise ValueError(
                f"Sheet '{sheet_name}' is missing required columns: {required_cols}"
            )

        # Mean per class
        class_means = (
            df.groupby("Class")["time_to_stabilisation_ms"]
              .mean()
        )

        # Number of entities per class
        class_counts = (
            df.groupby("Class")["time_to_stabilisation_ms"]
              .count()
        )

        # Weighted average across classes
        weighted_mean = (class_means * class_counts).sum() / class_counts.sum()

        results.append({
            "model": sheet_name,
            "weighted_mean_time_s": weighted_mean / 1000
        })

    return pd.DataFrame(results)


def create_heatmaps_from_summary_excel(
    excel_path,
    output_dir,
    model_order,
    model_stats_df,            # average
    model_weighted_stats_df,   # NEW: weighted average
    n_color_bins=5,            # NEW: discretisation control
    legend_min_s=0,          # NEW
    legend_max_s=2000,       # NEW
    figsize=(12, 8),
    dpi=300
):
    """
    model_stats_df must contain:
      - 'model'
      - 'mean_time_s'

    model_weighted_stats_df must contain:
      - 'model'
      - 'weighted_mean_time_s'
    """

    excel_path = Path(excel_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sheets = pd.read_excel(excel_path, sheet_name=None)

    # Load state counts sheet
    state_counts_df = sheets.get("State counts")
    if state_counts_df is None:
        raise ValueError("Excel file must contain a sheet named 'State counts'")

    state_counts_df = state_counts_df.copy()
    state_counts_df = state_counts_df.set_index(state_counts_df.columns[0])
    state_counts_df = state_counts_df.apply(pd.to_numeric, errors="coerce")

    # Continuous base colormap
    base_cmap = LinearSegmentedColormap.from_list(
        "GreenGoldRed",
        ["#1a9850", "#f1c232", "#d73027"],
        N=256
    )

    priority_rows = [
        "V",
        "Ca Dynamics",
        "Na Dynamics",
        "K Dynamics",
        "Ca flux/RyR Dynamics",
        "ICaL Gates",
        "IKr gates",
    ]

    for sheet_name, df in sheets.items():

        if sheet_name.lower() in ["state categories", "state counts"]:
            continue

        df = df.copy()
        df = df.set_index(df.columns[0])
        df_numeric = df.apply(pd.to_numeric, errors="coerce")

        # Convert ms → seconds
        df_numeric = df_numeric / 1000.0

        # Reorder columns
        cols = [c for c in model_order if c in df_numeric.columns]
        df_numeric = df_numeric[cols]

        # Reorder rows
        top_rows = [r for r in priority_rows if r in df_numeric.index]
        remaining_rows = [r for r in df_numeric.index if r not in top_rows]
        df_numeric = df_numeric.loc[top_rows + remaining_rows]

        # State-count annotations
        state_counts_plot = state_counts_df.loc[
            df_numeric.index,
            df_numeric.columns
        ]

        # Totals per model
        column_totals = state_counts_plot.sum(axis=0)

        # ---------- DISCRETISED COLOURBAR ----------
        # Fixed legend range (seconds)
        bounds = np.linspace(legend_min_s, legend_max_s, n_color_bins + 1)
        vmin, vmax = legend_min_s, legend_max_s


        cmap = LinearSegmentedColormap.from_list(
            "discrete_GreenGoldRed",
            base_cmap(np.linspace(0, 1, n_color_bins)),
            N=n_color_bins
        )
        norm = BoundaryNorm(bounds, cmap.N)

        # Plot
        fig = plt.figure(figsize=figsize)

        ax = sns.heatmap(
            df_numeric,
            cmap=cmap,
            norm=norm,
            annot=state_counts_plot,
            fmt=".0f",
            annot_kws={
                "fontsize": 9,
                "fontweight": "bold",
                "color": "black"
            },
            linewidths=1.2,           # BLACK GRIDLINES
            linecolor="black",
            cbar=True
        )

        # Colorbar styling
        cbar = ax.collections[0].colorbar
        cbar.set_ticks(bounds)
        cbar.ax.set_yticklabels([f"{b:.0f}" for b in bounds])
        cbar.ax.tick_params(labelsize=12, width=1.5)
        for label in cbar.ax.get_yticklabels():
            label.set_fontweight("bold")

        cbar.outline.set_edgecolor("black")
        cbar.outline.set_linewidth(1.8)
        cbar.ax.set_title("Time (s)", fontsize=13, fontweight="bold", pad=10)

        # X-axis labels with BOTH averages
        xticklabels = []
        for model in cols:
            avg_stats = model_stats_df.loc[
                model_stats_df["model"] == model
            ].iloc[0]

            weighted_stats = model_weighted_stats_df.loc[
                model_weighted_stats_df["model"] == model
            ].iloc[0]

            xticklabels.append(
                f"{model}\n"
                f"{avg_stats.mean_time_s:.0f} s\n"
                f"{weighted_stats.weighted_mean_time_s:.0f} s"
            )

        ax.set_xticklabels(
            xticklabels,
            rotation=0,
            ha="center",
            fontweight="bold"
        )

        # Y-axis formatting
        ylabels = [_format_subscripts(t.get_text()) for t in ax.get_yticklabels()]
        ax.set_yticklabels(
            ylabels,
            rotation=0,
            fontweight="bold"
        )

        ax.set_xlabel("")
        ax.set_ylabel("")

        # Totals above columns
        for i, total in enumerate(column_totals):
            ax.text(
                i + 0.5,
                -0.4,
                f"{int(total)}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                transform=ax.transData
            )

        plt.tight_layout()

        _savefig(fig, f"{sheet_name.replace(' ', '_')}_heatmap.png", dpi=dpi)
        plt.close(fig)


# === entry point ===
if __name__ == "__main__":
    STRATEGY_A_ORDER = ["Doste", "Doste†", "Morotti", "Morotti†", "BPS", "BPSLand",
                        "TOR", "TOR-DCl", "ORd", "Dutta", "TP06", "GB"]
    HEATMAP_ORDER = ["Doste", "Doste†", "Morotti", "Morotti†", "TOR", "TOR-DCl",
                     "Dutta", "ORd", "BPS", "BPSLand", "TP06", "GB"]

    # ---- Strategy A ----
    figure = StrategyA(DATA_DIR, STRATEGY_A_ORDER)
    _savefig(figure, "Strategy A plots - v6.png")

    # ---- Per-model state stabilisation plots (optional, slow) ----
    # Uncomment to render one big-grid figure per model showing every state variable's
    # trajectory across cell types. Requires per-model folders with cell-type subfolders.
    # state_models = ["Morotti 2021 BARS", "Morotti 2021 no BARS"]
    # for model in state_models:
    #     main_dir = os.path.join(DATA_DIR, model)
    #     figure = state_stabilisation_plot(main_dir)
    #     _savefig(figure, f"{model} state stabilisation plots.png")

    # ---- Compute fresh stabilisation times from current simulation data ----
    subfolders = [
        f for f in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, f))
    ]
    model_names = subfolders

    time_to_stabilisation = Stabilisation_Time_Across_Models(
        [os.path.join(DATA_DIR, s) for s in subfolders],
        model_names,
        epsilon=1e-2,
        steady_window_ms=2000,
    )

    times_xlsx = _out_path("time_to_stabilisation.xlsx")
    with pd.ExcelWriter(times_xlsx) as writer:
        for model_name, stab_dict in zip(model_names, time_to_stabilisation):
            model_name = os.path.basename(model_name)
            df_out = pd.DataFrame({
                "state": list(stab_dict.keys()),
                "time_to_stabilisation_ms": list(stab_dict.values()),
            })
            df_out.to_excel(writer, sheet_name=model_name, index=False)

    # ---- Inject the static state -> Class mapping to build the categorised xlsx ----
    categorised_xlsx = _out_path("categorised_time_to_stabilisation.xlsx")
    categorise_times_xlsx(times_xlsx, STATE_CLASS_MAPPING, categorised_xlsx)

    # ---- Strategy B ----
    figure = StrategyB(
        excel_path=categorised_xlsx,
        plot_type="scatter_plot",
        log10_y=False,
        model_order=HEATMAP_ORDER,
    )
    _savefig(figure, "Strategy B scatter plots v5.png", dpi=300)

    # ---- Strategy C: summary by class ----
    summary_xlsx = _out_path("stabilisation_summary_by_class.xlsx")
    summarise_stabilisation_by_class(
        input_excel_path=categorised_xlsx,
        output_excel_path=summary_xlsx,
    )

    # ---- Heatmaps ----
    model_avg_df = average_time_per_model(categorised_xlsx)
    model_weighted_avg_df = weighted_average_time_per_model(categorised_xlsx)
    create_heatmaps_from_summary_excel(
        summary_xlsx,
        OUTPUT_DIR,
        model_order=HEATMAP_ORDER,
        model_stats_df=model_avg_df,
        model_weighted_stats_df=model_weighted_avg_df,
        n_color_bins=8,
        legend_min_s=0,
        legend_max_s=2000,
        figsize=(16, 8),
        dpi=300,
    )
