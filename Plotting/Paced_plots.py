"""Paced stabilisation plots (action potential metrics, ion concentrations, Ca transients, etc.).

Reads paced simulation outputs from DATA_DIR; writes all figures into OUTPUT_DIR.
Both directories are resolved relative to this script's location so the project
works out of the box on a fresh clone.
"""

# === imports ===
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import seaborn as sns
import re
import glob
import math
from matplotlib.ticker import ScalarFormatter
from matplotlib.lines import Line2D

# === directories ===
# Repo root is the directory containing this script's parent (i.e. parent of Plotting/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Where highthroughput_stabilisation_script.py writes its paced output.
DATA_DIR = os.path.join(_REPO_ROOT, "Outputs", "Paced")
# Plot output directory; created lazily on first save.
OUTPUT_DIR = os.path.join(_REPO_ROOT, "Outputs", "Paced Plots")


def _ensure_output_dir():
    """Create OUTPUT_DIR (and parents) on first save call."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def _resolve_cols(df, names):
    """Return df columns matching `names` case-insensitively, preserving order.
    Each entry in `names` may be a string or a list/tuple of aliases (e.g. ['Cli','Cl_i']).
    Missing entries are skipped silently."""
    lower_to_actual = {c.lower(): c for c in df.columns}
    out = []
    for item in names:
        aliases = item if isinstance(item, (list, tuple)) else [item]
        for cand in aliases:
            actual = lower_to_actual.get(cand.lower())
            if actual is not None:
                out.append(actual)
                break
    return out


def _savefig(fig, filename, **kwargs):
    """Save `fig` into OUTPUT_DIR/<filename>, creating OUTPUT_DIR if needed."""
    _ensure_output_dir()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), **kwargs)


# === helper functions ===

def _format_state_title(state_name):
    """Returns a bold LaTeX string with subscript formatting if the name contains an underscore."""
    if "_" in state_name:
        base, sub = state_name.split("_", 1)
        return rf"$\mathbf{{{base}_{{{sub}}}}}$"
    else:
        return rf"$\mathbf{{{state_name}}}$"


# === plotting functions ===

def intramodel_AP_plots(folder_path, cell_types, model_name):
    """
    Generate APD, dV/dtmax, and RMP plots for multiple cell types and pacing frequencies.
    Each row = cell type; columns = 3 metrics repeated per frequency (500, 1000, 2000 ms).
    """

    all_files = [f for f in os.listdir(folder_path) if f.endswith('.parquet') and 'APD df' in f]
    freqs = ['500ms', '1000ms', '2000ms']

    nrows = len(cell_types)
    ncols_per_freq = 3
    ncols = ncols_per_freq * len(freqs)

    # Adjusted figsize to avoid squashing
    fig, axs = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), dpi=150)
    if nrows == 1:
        axs = axs.reshape(1, -1)

    plt.subplots_adjust(hspace=0.4, wspace=2.0)
    sns.set_style("whitegrid")

    line_width = 1.8
    color_palette = sns.color_palette("deep", 6)

    for row_idx, cell in enumerate(cell_types):
        for freq_idx, freq in enumerate(freqs):
            # Orchestrator filename: "{model} APD df {cell_type} at {cycle_length}ms.parquet"
            file_match = [f for f in all_files if (cell in f and f"at {freq}" in f)]
            if not file_match:
                continue
            file_path = os.path.join(folder_path, file_match[0])
            APD_df = pd.read_parquet(file_path)

            expected_cols = ['Cycle', 'APD30', 'APD50', 'APD60', 'APD90',
                             'dV/dtmax steepest', 'Resting Membrane Potential']
            missing_cols = [c for c in expected_cols if c not in APD_df.columns]
            if missing_cols:
                continue

            base_col = freq_idx * ncols_per_freq

            # 1. APD
            ax = axs[row_idx, base_col]
            labels = ["APD30", "APD50", "APD60", "APD90"]
            for i, label in enumerate(labels):
                ax.plot(APD_df['Cycle'], APD_df[label],
                        linewidth=line_width, color=color_palette[i])
            ax.legend(labels, frameon=False, loc='upper center', ncol=4, fontsize=8)
            ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('dimgray')
                ax.spines[spine].set_linewidth(1.2)

            # 2. dV/dtmax
            ax = axs[row_idx, base_col + 1]
            ax.plot(APD_df['Cycle'], APD_df['dV/dtmax steepest'],
                    linewidth=line_width, color='b')
            ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('dimgray')
                ax.spines[spine].set_linewidth(1.2)

            # 3. RMP
            ax = axs[row_idx, base_col + 2]
            ax.plot(APD_df['Cycle'], APD_df['Resting Membrane Potential'],
                    linewidth=line_width, color='y')
            ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)
            for spine in ['left', 'bottom']:
                ax.spines[spine].set_color('dimgray')
                ax.spines[spine].set_linewidth(1.2)

    # Column headers (APD, dV/dtmax, RMP)
    col_titles = ['APD', 'dV/dtmax', 'RMP']
    for freq_idx, freq in enumerate(freqs):
        base_col = freq_idx * ncols_per_freq
        for i, title in enumerate(col_titles):
            axs[0, base_col + i].set_title(title, fontsize=12, pad=15)

    # Frequency headers above each 3-column block
    for freq_idx, freq in enumerate(freqs):
        start_ax = axs[0, freq_idx * ncols_per_freq]
        end_ax = axs[0, freq_idx * ncols_per_freq + 2]
        bbox_start = start_ax.get_position()
        bbox_end = end_ax.get_position()
        center_x = (bbox_start.x0 + bbox_end.x1) / 2
        fig.text(center_x, 0.9, freq, ha='center', va='bottom', fontsize=20, fontweight='bold')

    # Row labels (cell types)
    for row_idx, cell in enumerate(cell_types):
        fig.text(0.05,
                 (nrows - row_idx - 0.6) / nrows,
                 cell, va='center', ha='center',
                 rotation=90, fontsize=26, fontweight='bold')

    fig.text(0.5, 0.04, "Time (ms)", ha='center', fontsize=24)
    fig.suptitle(f"{model_name}: Action Potential Properties", fontsize=30, y=1.02)

    plt.tight_layout(rect=[0.05, 0.06, 1, 0.95])
    plt.show()


def compare_celltype_stabilisation_across_models(parent_dir, cell_type, freq, plot_title):
    """
    Compare APD, dV/dtmax, and RMP stabilisation plots across multiple models for a given cell type and pacing frequency.
    """

    model_folders = [f for f in os.listdir(parent_dir)
                     if os.path.isdir(os.path.join(parent_dir, f))]

    if not model_folders:
        return

    valid_models = []
    model_data = []

    freq_str = f"at {freq}ms"

    for model_name in model_folders:
        model_path = os.path.join(parent_dir, model_name)

        # Shannon 2004 special case
        search_type = "cell_type" if "Shannon 2004" in model_name else cell_type

        all_files = [
            f for f in os.listdir(model_path)
            if f.endswith(".parquet") and "APD df" in f and search_type in f and freq_str in f
        ]

        if not all_files:
            continue

        file_path = os.path.join(model_path, all_files[0])
        try:
            APD_df = pd.read_parquet(file_path)
        except Exception:
            continue

        expected_cols = [
            'Cycle', 'APD30', 'APD50', 'APD60', 'APD90',
            'dV/dtmax steepest', 'Resting Membrane Potential'
        ]
        if not all(col in APD_df.columns for col in expected_cols):
            continue

        valid_models.append(model_name)
        model_data.append((model_name, APD_df))

    if not model_data:
        return

    nrows = len(model_data)
    ncols = 3

    fig, axs = plt.subplots(nrows, ncols, figsize=(14, 3.0 * nrows), dpi=150)
    if nrows == 1:
        axs = axs.reshape(1, -1)

    plt.subplots_adjust(hspace=0.5, wspace=0.8)
    sns.set_style("whitegrid")

    line_width = 1.8
    color_palette = sns.color_palette("deep", 6)

    for row_idx, (model_name, APD_df) in enumerate(model_data):

        ax = axs[row_idx, 0]
        labels = ["APD30", "APD50", "APD60", "APD90"]
        for i, label in enumerate(labels):
            ax.plot(APD_df['Cycle'], APD_df[label],
                    linewidth=line_width, color=color_palette[i])
        if row_idx == 0:
            ax.legend(labels, frameon=False, loc='upper center', ncol=4, fontsize=10)
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        ax = axs[row_idx, 1]
        ax.plot(APD_df['Cycle'], APD_df['dV/dtmax steepest'],
                linewidth=line_width, color='b')
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

        ax = axs[row_idx, 2]
        ax.plot(APD_df['Cycle'], APD_df['Resting Membrane Potential'],
                linewidth=line_width, color='y')
        ax.grid(True, color='lightgrey', linestyle='--', linewidth=0.5)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    col_titles = ['APD', 'dV/dtmax', 'RMP']
    for col_idx, title in enumerate(col_titles):
        axs[0, col_idx].set_title(title, fontsize=16, pad=20, fontweight='bold')

    for row_idx, (model_name, _) in enumerate(model_data):
        idx = row_idx
        if idx < len(axs):
            axs[idx, 0].set_ylabel(model_name, fontsize=12, fontweight="bold", rotation=0,
                                   labelpad=70, va="center")

    fig.text(0.5, 0.04, "Time (ms)", ha='center', fontsize=15, fontweight='bold')
    fig.suptitle(plot_title, fontsize=20, y=1.02, fontweight='bold')

    plt.tight_layout(rect=[0.06, 0.06, 1, 0.95])
    plt.show()


def intramodel_AP_plots_simplified(folder_path, model_name, cell_types):
    """
    Simplified version of intramodel_AP_plots:
    - Only plots APD90 (no APD30, 50, 60)
    - User provides cell_types (e.g. ['EPI','ENDO','M'])
    - Plots grouped by type: [APD90 500ms, 1000ms, 2000ms, dV/dtmax 500ms, ... , RMP 2000ms]
    """

    parquet_files = [f for f in os.listdir(folder_path) if f.endswith(".parquet") and "APD df" in f]
    if not parquet_files:
        raise FileNotFoundError("No matching parquet files found in the specified folder.")

    # Match "at {freq}ms" to avoid colliding with other numbers in filenames
    freq_pattern = re.compile(r"at (\d+)ms")
    freqs = sorted(set(int(freq_pattern.search(f).group(1)) for f in parquet_files if freq_pattern.search(f)))

    data_dict = {freq: {} for freq in freqs}
    for freq in freqs:
        for f in parquet_files:
            if f"at {freq}ms" in f:
                for ct in cell_types:
                    if re.search(rf"\b{ct}\b", f, re.IGNORECASE):
                        data_dict[freq][ct.upper()] = os.path.join(folder_path, f)

    n_freqs = len(freqs)
    plot_types = ['APD90', 'dV/dtmax', 'RMP']
    ncols = n_freqs * len(plot_types)
    cmap = matplotlib.colormaps["Set2"]
    colors = {ct: cmap(i / len(cell_types)) for i, ct in enumerate(cell_types)}

    group_indices = [0, n_freqs, 2*n_freqs]

    fig, axes = plt.subplots(1, ncols, figsize=(10 * n_freqs, 3.5), dpi=300)
    if ncols == 1:
        axes = [axes]

    y_values = {'APD90': [], 'dVdtmax': [], 'RMP': []}
    df_dict = {}

    for freq in freqs:
        df_dict[freq] = {}
        for ct, path in data_dict[freq].items():
            df = pd.read_parquet(path)
            df_dict[freq][ct] = df
            if 'APD90' in df.columns:
                y_values['APD90'].extend(df['APD90'].values)
            if 'dV/dtmax steepest' in df.columns:
                y_values['dVdtmax'].extend(df['dV/dtmax steepest'].values)
            if 'Resting Membrane Potential' in df.columns:
                y_values['RMP'].extend(df['Resting Membrane Potential'].values)

    # Per-model y-limit tweaks: Morotti 2021 and Grandi Bers 2010 have outlier ranges we clip
    if os.path.basename(folder_path) == "Morotti 2021":
        y_lims = {
            'APD90': (200, 600),
            'dV/dtmax': (340, max(y_values['dVdtmax'])),
            'RMP': (min(y_values['RMP']), -82.1)
        }
    elif os.path.basename(folder_path) == "Grandi Bers 2010":
        y_lims = {
            'APD90': (min(y_values['APD90']), max(y_values['APD90'])),
            'dV/dtmax': (min(y_values['dVdtmax']), 380),
            'RMP': (min(y_values['RMP']), max(y_values['RMP']))
        }
    else:
        y_lims = {
            'APD90': (min(y_values['APD90']), max(y_values['APD90'])),
            'dV/dtmax': (min(y_values['dVdtmax']), max(y_values['dVdtmax'])),
            'RMP': (min(y_values['RMP']), max(y_values['RMP']))
        }

    plot_index = 0
    for metric in plot_types:
        for freq in freqs:
            ax = axes[plot_index]
            for ct in cell_types:
                if freq in df_dict and ct.upper() in df_dict[freq]:
                    df = df_dict[freq][ct.upper()]
                    if metric == 'APD90' and 'APD90' in df.columns:
                        ax.plot(df['Cycle'], df['APD90'], label=ct, color=colors[ct], linewidth=2)
                    elif metric == 'dV/dtmax' and 'dV/dtmax steepest' in df.columns:
                        ax.plot(df['Cycle'], df['dV/dtmax steepest'], label=ct, color=colors[ct], linewidth=2)
                    elif metric == 'RMP' and 'Resting Membrane Potential' in df.columns:
                        ax.plot(df['Cycle'], df['Resting Membrane Potential'], label=ct, color=colors[ct], linewidth=2)

            ax.set_ylim(y_lims[metric])
            ax.grid(False)
            ax.set_xlabel("")
            ax.text(0.5, 1.02, f"{freq} ms", transform=ax.transAxes,
                    ha='center', fontsize=10, color='gray')

            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

            plot_index += 1

    # group titles centred over each metric block
    for i, metric in enumerate(plot_types):
        mid_index = i * n_freqs + (n_freqs - 1) / 2
        axes[int(mid_index)].set_title(metric, fontsize=16, fontweight='bold', pad=25)

    for i, ax in enumerate(axes):
        if i in [group_indices[1], group_indices[2]]:
            ax.set_position([
                ax.get_position().x0,
                ax.get_position().y0,
                ax.get_position().width,
                ax.get_position().height
            ])

    fig.text(0.5, -0.02, 'Time (ms)', ha='center', fontsize=14, fontweight='bold')
    fig.suptitle(f"{model_name}", fontsize=20, fontweight='bold', y=1.05)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02),
               ncol=len(cell_types), fontsize=12, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 1])
    return fig


def plot_final_AP(main_dir):
    """
    Plots the final action potential cycle (voltage V) for each cell type and pacing frequency
    found in subfolders of main_dir. Traces are trimmed to the last cycle duration.
    """

    base = os.path.basename(os.path.normpath(main_dir))
    match = re.search(r".*?\d{4}", base)
    model_name = match.group(0) if match else base

    folders = sorted([
        f for f in os.listdir(main_dir)
        if os.path.isdir(os.path.join(main_dir, f))
    ])

    if not folders:
        raise ValueError("No subfolders found in the directory.")

    variables = ['V']
    freq_order = ['500ms', '1000ms', '2000ms']

    data_by_freq = {}

    for folder in folders:
        folder_path = os.path.join(main_dir, folder)
        parquet_files = sorted(glob.glob(os.path.join(folder_path, "simulation_chunk_*.parquet")))
        if not parquet_files:
            continue

        df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

        freq_match = re.search(r"(500ms|1000ms|2000ms)", folder)
        cell_match = re.search(r"(ENDO|EPI| M |Generic)", folder)

        if not freq_match or not cell_match:
            continue

        freq = freq_match.group(1)
        cell_type = cell_match.group(1)

        if freq not in data_by_freq:
            data_by_freq[freq] = {}

        final_time = df['time'].iloc[-1]
        freq_val = int(freq[:-2])
        df = df[df['time'] >= (final_time - freq_val)]

        data_by_freq[freq][cell_type] = df

    freqs_present = [f for f in freq_order if f in data_by_freq]
    n_freqs = len(freqs_present)

    y_ranges = {var: [float('inf'), -float('inf')] for var in variables}

    for freq in data_by_freq:
        for ct in data_by_freq[freq]:
            df = data_by_freq[freq][ct]
            for var in variables:
                y_ranges[var][0] = min(y_ranges[var][0], df[var].min())
                y_ranges[var][1] = max(y_ranges[var][1], df[var].max())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=False, sharey=False, dpi=300)

    cmap = matplotlib.colormaps["Set2"]
    cell_colors = {
        "EPI": cmap(0/3),
        "ENDO": cmap(1/3),
        "M": cmap(2/3),
        "Generic": cmap(4/9)
    }

    handles_for_legend = {}

    for _, freq in enumerate(freqs_present):
        freq_data = data_by_freq[freq]

        for col, var in enumerate(variables):
            freq_idx = freqs_present.index(freq)
            ax_idx = col * n_freqs + freq_idx
            ax = axes[ax_idx]

            # Sort by variability so the most-variable CT is drawn first (and ends up in the back)
            var_scores = {
                ct: freq_data[ct][var].max() - freq_data[ct][var].min()
                for ct in freq_data
            }
            ordered_cts = sorted(var_scores, key=lambda x: var_scores[x], reverse=True)

            for ct in ordered_cts:
                df = freq_data[ct]
                line = ax.plot(
                    df['time'], df[var],
                    linewidth=2,
                    color=cell_colors[ct],
                    alpha=0.75,
                    label=ct
                )[0]

                if ct not in handles_for_legend:
                    handles_for_legend[ct] = line

            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            ax.set_ylim(y_ranges[var][0], y_ranges[var][1])

            ax.text(
                0.5, 1.02,
                f"{freq.replace('ms','')} ms",
                transform=ax.transAxes,
                ha='center',
                fontsize=13,
                color='gray'
            )

            ax.set_xlabel("")

    fig.suptitle(model_name, fontsize=18, fontweight='bold', y=0.995)
    fig.text(0.5, 0.02, "Time (ms)", ha="center", fontsize=14, fontweight='bold')

    fig.legend(
        handles_for_legend.values(),
        handles_for_legend.keys(),
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.965)
    )

    plt.tight_layout(rect=[0, 0, 1, 1])

    return fig


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


def plot_ca_transients(main_dir, subfolders):
    """
    Plots calcium transient metrics (Catmax, Catmin, Catamp) per beat across models and pacing frequencies.
    Reads APD parquet files named '{model} APD df {CT} at {cycle_length}ms.parquet' from each
    model subfolder. Catmax/Catmin/Catamp are columns inside the APD parquet (one row per cycle);
    the 'Cycle' column plays the role of the previous CSV's 'beat' column.
    """
    cmap = matplotlib.colormaps["Set2"]
    cell_colors = {
        "EPI": cmap(0/3),
        "ENDO": cmap(1/3),
        "M": cmap(2/3),
        "Generic": cmap(4/9)
    }

    freqs = [500, 1000, 2000]
    metrics = ["Catmax", "Catmin", "Catamp"]

    # scientific scaling for neat axes
    sci_powers = {"Catmax": -4, "Catmin": -5, "Catamp": -3}

    n_rows = len(subfolders)
    n_cols = 9

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2*n_cols, 3*n_rows),
        sharex=True, dpi=300
    )

    if n_rows == 1:
        axes = [axes]

    for row_idx, subfolder in enumerate(subfolders):
        folder_path = os.path.join(main_dir, subfolder)

        if not os.path.isdir(folder_path):
            continue

        files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".parquet") and "APD df" in f
        ]

        data = {freq: {} for freq in freqs}

        for file in files:
            match_cell = re.search(r"APD df (ENDO|EPI|M)", file)
            cell_type = match_cell.group(1) if match_cell else "Generic"

            match_freq = re.search(r"at (500|1000|2000)ms\.parquet$", file)
            if not match_freq:
                continue
            freq = int(match_freq.group(1))

            df = pd.read_parquet(os.path.join(folder_path, file))
            if {"Cycle", "Catmax", "Catmin", "Catamp"}.issubset(df.columns):
                data[freq][cell_type] = df

        # Compute shared y-limits for each metric block
        y_limits = {}
        for metric in metrics:
            vals = []
            for freq in freqs:
                for df in data[freq].values():
                    vals.extend(df[metric].values)
            if vals:
                y_limits[metric] = (min(vals), max(vals))
            else:
                y_limits[metric] = (None, None)

        for col_block, metric in enumerate(metrics):
            ymin, ymax = y_limits[metric]

            for i, freq in enumerate(freqs):
                ax = axes[row_idx][col_block*3 + i]

                for cell_type, df in data[freq].items():
                    color = cell_colors.get(cell_type, cell_colors["Generic"])
                    ax.plot(df["Cycle"], df[metric], lw=1.5, color=color)

                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)

                formatter = ScalarFormatter(useMathText=True)
                formatter.set_powerlimits((sci_powers[metric], sci_powers[metric]))
                ax.yaxis.set_major_formatter(formatter)

                if i != 0:
                    ax.set_yticklabels([])
                    ax.set_ylabel("")

                ax.set_title(f"{freq} ms", fontsize=10, color="grey", fontweight="normal")

                if row_idx == 0 and i == 1:
                    metric_title = {
                        "Catmax": r"$\mathbf{CaT_{max}}$",
                        "Catmin": r"$\mathbf{CaT_{min}}$",
                        "Catamp": r"$\mathbf{CaT_{amp}}$"
                    }
                    ax.text(
                        0.5, 1.25, metric_title[metric],
                        transform=ax.transAxes,
                        ha="center", va="bottom",
                        fontsize=22, fontweight="bold"
                    )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                if col_block == 0 and i == 0:
                    ax.set_ylabel(subfolder, fontsize=11, fontweight="bold")

                if row_idx != n_rows - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                else:
                    ax.tick_params(axis="x", labelbottom=True, labelsize=9)

                ax.grid(True, alpha=0.2)

    fig.supxlabel("Number of Cycles", fontsize=24, fontweight="bold", y=0.085)

    handles = [
        plt.Line2D([0], [0], color=cell_colors[c], lw=2)
        for c in ["EPI", "ENDO", "M"]
    ]
    labels = ["EPI", "ENDO", "M"]

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=3,
        frameon=False,
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    return fig


def plot_AP_morphology(main_dir, subfolders):
    """
    Plots AP upstroke morphology metrics (AP_max, AP_amp) per beat across models and pacing frequencies.
    Reads APD parquet files from each model subfolder.
    """
    cmap = matplotlib.colormaps["Set2"]
    cell_colors = {
        "EPI": cmap(0/3),
        "ENDO": cmap(1/3),
        "M": cmap(2/3),
        "Generic": cmap(4/9)
    }

    freqs = [500, 1000, 2000]
    metrics = ["AP_max", "AP_amp"]

    n_rows = len(subfolders)
    n_cols = 9

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2*n_cols, 3*n_rows),
        sharex=True, dpi=300
    )

    if n_rows == 1:
        axes = [axes]

    for row_idx, subfolder in enumerate(subfolders):
        folder_path = os.path.join(main_dir, subfolder)

        if not os.path.isdir(folder_path):
            continue

        files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".parquet") and "APD df" in f
        ]

        data = {freq: {} for freq in freqs}

        for file in files:
            match_cell = re.search(r"APD df (ENDO|EPI|M)", file)
            cell_type = match_cell.group(1) if match_cell else "Generic"

            match_freq = re.search(r"at (500|1000|2000)ms\.parquet$", file)
            if not match_freq:
                continue
            freq = int(match_freq.group(1))

            df = pd.read_parquet(os.path.join(folder_path, file))

            # Drop first few beats for Morotti 2021 (early transient is unrepresentative)
            if "Morotti 2021" in subfolder:
                df = df[df["Cycle"] >= 5]

            if {"Cycle", "AP_max", "AP_amp"}.issubset(df.columns):
                data[freq][cell_type] = df

        y_limits = {}
        for metric in metrics:
            vals = []
            for freq in freqs:
                for df in data[freq].values():
                    vals.extend(df[metric].values)
            if vals:
                y_limits[metric] = (min(vals), max(vals))
            else:
                y_limits[metric] = (None, None)

        for col_block, metric in enumerate(metrics):
            ymin, ymax = y_limits[metric]

            for i, freq in enumerate(freqs):
                ax = axes[row_idx][col_block*3 + i]

                for cell_type, df in data[freq].items():
                    color = cell_colors.get(cell_type, cell_colors["Generic"])
                    ax.plot(df["Cycle"], df[metric], lw=1.5, color=color)

                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)

                if i != 0:
                    ax.set_yticklabels([])
                    ax.set_ylabel("")

                ax.set_title(f"{freq} ms", fontsize=10, color="grey", fontweight="normal")

                if row_idx == 0 and i == 1:
                    metric_title = {
                        "AP_max": r"$\mathbf{AP_{max}}$",
                        "AP_amp": r"$\mathbf{AP_{amp}}$"
                    }
                    ax.text(
                        0.5, 1.25, metric_title[metric],
                        transform=ax.transAxes,
                        ha="center", va="bottom",
                        fontsize=22, fontweight="bold"
                    )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                if col_block == 0 and i == 0:
                    ax.set_ylabel(subfolder, fontsize=11, fontweight="bold")

                if row_idx != n_rows - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                else:
                    ax.tick_params(axis="x", labelbottom=True, labelsize=9)

                ax.grid(True, alpha=0.2)

    fig.supxlabel("Number of Cycles", fontsize=24, fontweight="bold", y=0.085)

    handles = [
        plt.Line2D([0], [0], color=cell_colors[c], lw=2)
        for c in ["EPI", "ENDO", "M"]
    ]
    labels = ["EPI", "ENDO", "M"]

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=3,
        frameon=False,
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    return fig


def plot_APD_stabilisation(main_dir, subfolders):
    """
    Plots APD90, dV/dtmax, and RMP per beat across models and pacing frequencies.
    Reads APD parquet files from each model subfolder.
    """
    cmap = matplotlib.colormaps["Set2"]
    cell_colors = {
        "EPI": cmap(0/3),
        "ENDO": cmap(1/3),
        "M": cmap(2/3),
        "Generic": cmap(4/9)
    }

    freqs = [500, 1000, 2000]
    metrics = ["APD90", "dV/dtmax steepest", "Resting Membrane Potential"]

    n_rows = len(subfolders)
    n_cols = 9

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2*n_cols, 3*n_rows),
        sharex=True, dpi=300
    )

    if n_rows == 1:
        axes = [axes]

    for row_idx, subfolder in enumerate(subfolders):
        folder_path = os.path.join(main_dir, subfolder)

        if not os.path.isdir(folder_path):
            continue

        files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".parquet") and "APD df" in f
        ]

        data = {freq: {} for freq in freqs}

        for file in files:
            match_cell = re.search(r"APD df (ENDO|EPI|M)", file)
            cell_type = match_cell.group(1) if match_cell else "Generic"

            match_freq = re.search(r"at (500|1000|2000)ms\.parquet$", file)
            if not match_freq:
                continue
            freq = int(match_freq.group(1))
            df = pd.read_parquet(os.path.join(folder_path, file))
            # Drop first few beats for Morotti 2021 (early transient is unrepresentative)
            if "Morotti 2021" in subfolder:
                df = df[df["Cycle"] >= 5]

            if {"Cycle", "APD90", "dV/dtmax steepest", "Resting Membrane Potential"}.issubset(df.columns):
                data[freq][cell_type] = df

        y_limits = {}
        for metric in metrics:
            vals = []
            for freq in freqs:
                for df in data[freq].values():
                    vals.extend(df[metric].values)
            if vals:
                y_limits[metric] = (min(vals), max(vals))
            else:
                y_limits[metric] = (None, None)

        for col_block, metric in enumerate(metrics):
            ymin, ymax = y_limits[metric]

            for i, freq in enumerate(freqs):
                ax = axes[row_idx][col_block*3 + i]

                for cell_type, df in data[freq].items():
                    color = cell_colors.get(cell_type, cell_colors["Generic"])
                    ax.plot(df["Cycle"], df[metric], lw=1.5, color=color)

                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)

                if i != 0:
                    ax.set_yticklabels([])
                    ax.set_ylabel("")

                ax.set_title(f"{freq} ms", fontsize=10, color="grey", fontweight="normal")

                if row_idx == 0 and i == 1:
                    metric_title = {
                        "APD90": r"$\mathbf{APD_{90}}$",
                        "dV/dtmax steepest": r"$\mathbf{dV/dt_{max}}$",
                        "Resting Membrane Potential": r"$\mathbf{RMP}$"
                    }
                    ax.text(
                        0.5, 1.25, metric_title[metric],
                        transform=ax.transAxes,
                        ha="center", va="bottom",
                        fontsize=22, fontweight="bold"
                    )

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                if col_block == 0 and i == 0:
                    ax.set_ylabel(subfolder, fontsize=11, fontweight="bold")

                if row_idx != n_rows - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                else:
                    ax.tick_params(axis="x", labelbottom=True, labelsize=9)

                ax.grid(True, alpha=0.2)

    fig.supxlabel("Number of Cycles", fontsize=24, fontweight="bold", y=0.085)

    handles = [
        plt.Line2D([0], [0], color=cell_colors[c], lw=2)
        for c in ["EPI", "ENDO", "M"]
    ]
    labels = ["EPI", "ENDO", "M"]

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=3,
        frameon=False,
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    return fig


def plot_chloride_traces(main_dir, subfolders):
    """
    Plots intracellular chloride (Cl_i / Cli) traces over time across models and pacing frequencies.
    Reads the single ion-trace parquet '{model} ion trace {CT} at {cycle_length}ms.parquet' written
    by the orchestrator. TOR_DCl2020 and Doste2022 use 'Cl_i'; other models may use 'Cli'.
    """
    cmap = matplotlib.colormaps["Set2"]
    cell_colors = {
        "EPI": cmap(0/3),
        "ENDO": cmap(1/3),
        "M": cmap(2/3),
        "Generic": cmap(4/9)
    }

    freqs = [500, 1000, 2000]

    n_rows = len(subfolders)
    n_cols = 3

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2*n_cols, 3*n_rows),
        sharex=False, dpi=300
    )

    if n_rows == 1:
        axes = [axes]

    for row_idx, subfolder in enumerate(subfolders):
        folder_path = os.path.join(main_dir, subfolder)

        if not os.path.isdir(folder_path):
            continue

        ion_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".parquet") and "ion trace" in f
        ]

        data = {freq: {} for freq in freqs}

        for fname in ion_files:
            match_cell = re.search(r"ion trace (ENDO|EPI|M)", fname)
            cell_type = match_cell.group(1) if match_cell else "Generic"

            match_freq = re.search(r"at (500|1000|2000)ms\.parquet$", fname)
            if not match_freq:
                continue
            freq = int(match_freq.group(1))

            df = pd.read_parquet(os.path.join(folder_path, fname))

            cols = _resolve_cols(df, ['time', ['Cli', 'Cl_i']])
            if len(cols) < 2:
                continue
            time_col, cl_col = cols[0], cols[1]

            df = df[[time_col, cl_col]].copy()
            df.columns = ["time", "Cl_i"]

            # Convert time from ms to seconds
            df["time"] = df["time"] / 1000.0

            # Thin data: keep 1 in every 12 points
            df = df.iloc[::12, :]

            data[freq][cell_type] = df

        # shared y-limits per row (across the 3 freqs)
        vals = []
        for freq in freqs:
            for df in data[freq].values():
                vals.extend(df["Cl_i"].values)
        if vals:
            ymin, ymax = min(vals), max(vals)
        else:
            ymin, ymax = None, None

        for i, freq in enumerate(freqs):
            ax = axes[row_idx][i]

            for cell_type, df in data[freq].items():
                color = cell_colors.get(cell_type, cell_colors["Generic"])
                ax.plot(df["time"], df["Cl_i"], lw=1.5, color=color)

            if ymin is not None and ymax is not None:
                ax.set_ylim(ymin, ymax)

            ax.set_title(f"{freq} ms", fontsize=10, color="grey", fontweight="normal")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if i == 0:
                ax.set_ylabel(subfolder, fontsize=11, fontweight="bold")
            else:
                ax.set_yticklabels([])

            if row_idx != n_rows - 1:
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.tick_params(axis="x", labelbottom=True, labelsize=9)

            ax.grid(True, alpha=0.2)

    fig.supxlabel("Time (ms)", fontsize=24, fontweight="bold", y=0.085)

    handles = [
        plt.Line2D([0], [0], color=cell_colors[c], lw=2)
        for c in ["EPI", "ENDO", "M"]
    ]
    labels = ["EPI", "ENDO", "M"]

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.03),
        ncol=3,
        frameon=False,
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    return fig


def plot_conc_stabilisation(main_dir, subfolders):
    """
    Plots Nai, Ki, and Cai traces over time across models and pacing frequencies.
    Reads the single ion-trace parquet '{model} ion trace {CT} at {cycle_length}ms.parquet' written
    by the orchestrator. For Morotti models, Ki is rounded to 2 dp to suppress floating-point
    quantization noise.
    """
    cmap = matplotlib.colormaps["Set2"]
    cell_colors = {
        "EPI": cmap(0/3),
        "ENDO": cmap(1/3),
        " M ": cmap(2/3),
        "Generic": cmap(4/9)
    }

    freqs = [500, 1000, 2000]
    metrics = ["Nai", "Ki", "Cai"]

    n_rows = len(subfolders)
    n_cols = 9  # 3 metrics x 3 freqs

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.2*n_cols, 3*n_rows),
        sharex=False, dpi=300
    )

    if n_rows == 1:
        axes = [axes]

    for row_idx, subfolder in enumerate(subfolders):
        folder_path = os.path.join(main_dir, subfolder)

        if not os.path.isdir(folder_path):
            continue

        ion_files = [
            f for f in os.listdir(folder_path)
            if f.endswith(".parquet") and "ion trace" in f
        ]

        data = {freq: {} for freq in freqs}

        for fname in ion_files:
            match_cell = re.search(r"ion trace (ENDO|EPI| M )", fname)
            cell_type = match_cell.group(1) if match_cell else "Generic"

            match_freq = re.search(r"at (500|1000|2000)ms\.parquet$", fname)
            if not match_freq:
                continue
            freq = int(match_freq.group(1))

            df = pd.read_parquet(os.path.join(folder_path, fname))

            cols = _resolve_cols(df, ['time', 'Cai', 'Nai', 'Ki', ['Cli', 'Cl_i']])
            # Require time + Nai + Ki + Cai at minimum
            needed = {'time', 'Nai', 'Ki', 'Cai'}
            present_lower = {c.lower() for c in cols}
            if not needed.issubset({n.lower() for n in present_lower}):
                continue

            # Normalise column names to canonical lower-case expected by the rest of the routine
            rename_map = {}
            for c in cols:
                lc = c.lower()
                if lc == "time":
                    rename_map[c] = "time"
                elif lc == "cai":
                    rename_map[c] = "Cai"
                elif lc == "nai":
                    rename_map[c] = "Nai"
                elif lc == "ki":
                    rename_map[c] = "Ki"
                elif lc in ("cli", "cl_i"):
                    rename_map[c] = "Cl_i"
            df = df[list(rename_map.keys())].rename(columns=rename_map).copy()

            df["time"] = df["time"] / 1000.0
            df = df.iloc[::3, :]
            if "Morotti" in subfolder:
                df["Ki"] = df["Ki"].round(2)

            data[freq][cell_type] = df

        y_limits = {}
        for metric in metrics:
            vals = []
            for freq in freqs:
                for df in data[freq].values():
                    vals.extend(df[metric].values)
            if vals:
                y_limits[metric] = (min(vals), max(vals))
            else:
                y_limits[metric] = (None, None)

        # Plot blocks (Nai | Ki | Cai)
        for col_block, metric in enumerate(metrics):
            ymin, ymax = y_limits[metric]

            for i, freq in enumerate(freqs):
                ax = axes[row_idx][col_block*3 + i]

                # Sort cell types by oscillation magnitude; most oscillatory drawn first (background)
                var_scores = {
                    ct: data[freq][ct][metric].max() - data[freq][ct][metric].min()
                    for ct in data[freq]
                }
                ordered_cts = sorted(var_scores, key=lambda ct: var_scores[ct], reverse=True)

                for ct in ordered_cts:
                    df_ct = data[freq][ct]
                    color = cell_colors.get(ct, cell_colors["Generic"])
                    # Add transparency only for Cai plots
                    if metric == "Cai":
                        ax.plot(df_ct["time"], df_ct[metric], lw=1.5, color=color, alpha=0.6)
                    else:
                        ax.plot(df_ct["time"], df_ct[metric], lw=1.5, color=color)

                if ymin is not None and ymax is not None:
                    ax.set_ylim(ymin, ymax)
                    # Scientific notation for Cai axis
                    if metric == "Cai":
                        ax.ticklabel_format(axis='y', style='sci', scilimits=(-4, -4))

                ax.set_title(f"{freq} ms", fontsize=10, color="grey", fontweight="normal")

                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)

                if i != 0:
                    ax.set_yticklabels([])
                    ax.set_ylabel("")

                if col_block == 0 and i == 0:
                    ax.set_ylabel(subfolder, fontsize=11, fontweight="bold")

                if row_idx != n_rows - 1:
                    ax.tick_params(axis="x", labelbottom=False)
                else:
                    ax.tick_params(axis="x", labelbottom=True, labelsize=9)

                if row_idx == 0 and i == 1:
                    metric_title = {
                        "Nai": r"$\mathbf{Na_i}$",
                        "Ki": r"$\mathbf{K_i}$",
                        "Cai": r"$\mathbf{Ca_i}$"
                    }
                    ax.text(
                        0.5, 1.25, metric_title[metric],
                        transform=ax.transAxes,
                        ha="center", va="bottom",
                        fontsize=22, fontweight="bold"
                    )

                ax.grid(True, alpha=0.2)

    # shared x label (seconds)
    fig.supxlabel("Time (s)", fontsize=24, fontweight="bold", y=0.085)

    handles = [
        plt.Line2D([0], [0], color=cell_colors[c], lw=2)
        for c in ["EPI", "ENDO", " M "]
    ]
    labels = ["EPI", "ENDO", " M "]

    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
        fontsize=18
    )

    plt.tight_layout(rect=[0, 0.09, 1, 1])
    return fig


# === entry point ===
if __name__ == "__main__":
    ALL_MODELS = [
        "BARS 2022", "BARS 2022_no BARS", "BPSLand 2022",
        "Morotti with BARS", "Morotti without BARS", "BPS 2020",
        "TOR-DynCl 2020", "TOR 2019", "CiPA 2017",
        "O'Hara Rudy 2010", "Grandi Bers 2010", "Ten Tusscher 2006"
    ]
    CHLORIDE_MODELS = ["BARS 2022", "BARS 2022_no BARS", "TOR-DynCl 2020"]

    # --- State stabilisation plots (reads unpaced sim data; lives here for legacy reasons) ---
    models = ["Morotti 2021", "Morotti 2021 - no BARS stim"]
    no_stim_dir = os.path.join(_REPO_ROOT, "Outputs", "Unpaced")
    for model in models:
        main_dir = os.path.join(no_stim_dir, model)
        figure = state_stabilisation_plot(main_dir)
        _savefig(figure, f'{os.path.basename(main_dir)} ENDO state stabilisation plots.png')

    # --- Ca transient stabilisation ---
    fig = plot_ca_transients(DATA_DIR, ALL_MODELS)
    _savefig(fig, 'Ca Transient stabilisation plots 0104 - final.png', transparent=True, dpi=300)

    # --- AP upstroke morphology stabilisation ---
    fig = plot_AP_morphology(DATA_DIR, ALL_MODELS)
    _savefig(fig, 'AP Upstroke Morphology stabilisation plots 0104 -final.png', transparent=True, dpi=300)

    # --- APD90/dV-dtmax/RMP stabilisation ---
    fig = plot_APD_stabilisation(DATA_DIR, ALL_MODELS)
    _savefig(fig, 'APD stabilisation plots_0104 - final.png', transparent=True)

    # --- Chloride traces ---
    fig = plot_chloride_traces(DATA_DIR, CHLORIDE_MODELS)
    _savefig(fig, 'Chloride Current stabilisation plots.png')

    # --- Ion concentration stabilisation ---
    fig = plot_conc_stabilisation(DATA_DIR, ALL_MODELS)
    _savefig(fig, 'Concentration stabilisation plots_test2.png', transparent=True, dpi=300)
