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
from matplotlib.ticker import ScalarFormatter

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


# === plotting functions ===

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
    # 2 metrics * 3 frequencies = 6 columns; keeping each subplot at the same 3.2 x 3 in
    # size as the other (9-col) figures means total width shrinks proportionally.
    n_cols = 6

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
                        "AP_max": r"$\mathbf{AP_{max}}$ (mV)",
                        "AP_amp": r"$\mathbf{AP_{amp}}$ (mV)"
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
            # if "Morotti 2021" in subfolder:
            #     df = df[df["Cycle"] >= 5]

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
                        "APD90": r"$\mathbf{APD_{90}}$ (ms)",
                        "dV/dtmax steepest": r"$\mathbf{dV/dt_{max}}$ (mV/ms)",
                        "Resting Membrane Potential": r"$\mathbf{RMP}$ (mV)"
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
    # Auto-detect models from whatever was produced in Outputs/Paced/.
    if not os.path.isdir(DATA_DIR):
        raise SystemExit(
            f"No simulation data found in {DATA_DIR!r}. "
            "Run highthroughput_stabilisation_script.py with protocol='paced' first."
        )
    ALL_MODELS = sorted(
        f for f in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, f))
    )
    if not ALL_MODELS:
        raise SystemExit(f"No model subfolders found in {DATA_DIR!r}.")
    print(f"Detected {len(ALL_MODELS)} model folder(s): {ALL_MODELS}")

    # Chloride-bearing models, matched as substrings so the BARS / no-BARS suffix
    # variants (e.g. 'Doste 2022 BARS', 'Doste 2022 no BARS') are picked up too.
    _chloride_substrings = ("Doste", "TOR-DCl", "BARS 2022")
    CHLORIDE_MODELS = [m for m in ALL_MODELS if any(s in m for s in _chloride_substrings)]

    # --- Ca transient stabilisation ---
    fig = plot_ca_transients(DATA_DIR, ALL_MODELS)
    _savefig(fig, 'Ca Transient stabilisation plots.png', transparent=True, dpi=300)

    # --- AP upstroke morphology stabilisation ---
    fig = plot_AP_morphology(DATA_DIR, ALL_MODELS)
    _savefig(fig, 'AP Upstroke Morphology stabilisation plots.png', transparent=True, dpi=300)

    # --- APD90 / dV-dtmax / RMP stabilisation ---
    fig = plot_APD_stabilisation(DATA_DIR, ALL_MODELS)
    _savefig(fig, 'APD stabilisation plots.png', transparent=True)

    # --- Chloride traces (only if any chloride-bearing model is present) ---
    if CHLORIDE_MODELS:
        fig = plot_chloride_traces(DATA_DIR, CHLORIDE_MODELS)
        _savefig(fig, 'Chloride Current stabilisation plots.png')
        print(f"Saved chloride plot for: {CHLORIDE_MODELS}")
    else:
        print("Skipping chloride plot — no Doste / TOR-DCl folders in Outputs/Paced.")

    # --- Ion concentration stabilisation ---
    fig = plot_conc_stabilisation(DATA_DIR, ALL_MODELS)
    _savefig(fig, 'Ionic Concentration stabilisation plots.png', transparent=True, dpi=300)
