## High-throughput stabilisation script: runs all models and saves outputs.

import pandas as pd
import matplotlib.pyplot as plt
from Ventricular_AP_Models.ORd import *
from Ventricular_AP_Models.GB import *
from Ventricular_AP_Models.Dutta2017 import *
from Ventricular_AP_Models.TOR_DCl2020 import *
from Ventricular_AP_Models.TOR2019 import *
from Ventricular_AP_Models.Morotti2021 import *
from Ventricular_AP_Models.TNNP06 import *
from Ventricular_AP_Models.Doste2022 import *
from Ventricular_AP_Models.BPS2020 import *
from Ventricular_AP_Models.BPSLand2022 import *
from calculate_AP_parameters import *
from conductances import *
import glob
import re
import os


def _resolve_cols(df, names):
    """Return df-column names matching `names` case-insensitively, preserving order;
    each entry in `names` may be a single str or a list/tuple of acceptable aliases
    (e.g. ['Cli', 'Cl_i']) — the first alias present in df is used; missing entries
    are skipped silently."""
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


models = ["Morotti 2021"] #, 'Ten Tusscher 2006', 'Grandi Bers 2010', "O'Hara Rudy 2010"

protocol = 'paced'  # 'unpaced'

if protocol == 'paced':
    cycle_lengths = [500, 1000, 2000]
else:
    cycle_lengths = [1000]

# When unpaced, override every model's amp default with 0 (silent stimulus).
# When paced, each model's baked-in amp default is used.
amp_override = {'amp': 0} if protocol == 'unpaced' else {}

cycles = 100
ISO_conc = 0.1
BARS = 'True'
plot_fig = False
baseline_K_concs = 5.4

## Output directory: <repo_root>/Outputs/<Paced|Unpaced>/.
## Resolved relative to this script so a fresh clone works without setup.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
dest_dir = os.path.join(_SCRIPT_DIR, "Outputs", "Paced" if protocol == "paced" else "Unpaced")


for model in models:

    destination_dir = os.path.join(dest_dir, model)
    os.makedirs(destination_dir, exist_ok=True)

    # identify cell types to loop over
    if model == 'Morotti 2021' or model == "BARS 2022":
        cell_types = ['ENDO', 'EPI']
    else:
        cell_types = ['ENDO', 'M', 'EPI']

    for cell_type in cell_types:
        print(cell_type)
        print(model)

        for cycle_length in cycle_lengths:

            print(f"Simulating {model} for {cell_type} cells at {cycle_length} ms CL")

            if model == "ORd 2010":
                df, _, stim_duration = run_ORd_Model(cycles, cycle_length, cell_type, baseline_K_concs, **amp_override)
            elif model == "Dutta 2017":
                df, _, stim_duration = run_Dutta_Model(cycles, cycle_length, cell_type, **amp_override)
            elif model == "TOR-DCl 2020":
                df, _, stim_duration = run_TOR_DCl_Model(cycles, cycle_length, cell_type, **amp_override)
            elif model == "TNNP06":
                df, _, stim_duration = run_TNNP06_model(cycles, cycle_length, cell_type, baseline_K_concs, **amp_override)
                stim_duration = 0.85
            elif model == "TOR 2019":
                df, _, stim_duration = run_TOR_Model(cycles, cycle_length, cell_type, **amp_override)
            elif model == "GB 2010":
                df, _, stim_duration = run_GB_model(cycles, cycle_length, cell_type, **amp_override)
            elif model == "Doste 2022":
                df, _, stim_duration = run_Doste_Model(cycles, cycle_length, cell_type, BARS, ISO_conc, **amp_override)
            elif model == "Morotti 2021":
                df, _, stim_duration = run_Morotti_model(cycles, cycle_length, cell_type,
                    flag_BARS=BARS, camkii_exp=1, stimDur=5, **amp_override)
            elif model == "BPS 2020":
                df, _, stim_duration = run_BPS_model(cycles, cycle_length, cell_type, **amp_override)
            elif model == "BPSLand 2022":
                df, _, stim_duration = run_BPSLand_model(cycles, cycle_length, cell_type, **amp_override)
            else:
                print('THERE IS A PROBLEM')

            if protocol == 'unpaced':
                # Unpaced: save the full df as chunked parquet (all columns, full duration).
                chunks_dir = os.path.join(destination_dir,
                                          f'{model} sim chunks {cell_type} at {cycle_length}ms')
                os.makedirs(chunks_dir, exist_ok=True)
                chunksize = 500_000
                for i in range(0, len(df), chunksize):
                    chunk = df.iloc[i:i + chunksize]
                    out_path = os.path.join(chunks_dir, f"simulation_chunk_{i // chunksize:03d}.parquet")
                    print(f"Saving {out_path} ...")
                    chunk.to_parquet(out_path, compression="zstd", index=False)
                print("Full df saved as Parquet chunks.")
            else:
                # Paced: slim ion trace (time, Cai, Nai, Ki, and Cli where present), full duration.
                # 'Cli' / 'Cl_i' accepted as aliases (TOR_DCl2020 + Doste2022 use 'Cl_i').
                ion_cols = _resolve_cols(df, ['time', 'Cai', 'Nai', 'Ki', ['Cli', 'Cl_i']])
                ion_path = os.path.join(destination_dir,
                                        f'{model} ion trace {cell_type} at {cycle_length}ms.parquet')
                df[ion_cols].to_parquet(ion_path, compression="zstd", index=False)
                print(f"Saved ion trace ({ion_cols}) -> {ion_path}")

            # APD analysis + APD parquet: only meaningful when paced.
            if protocol == 'paced':
                df_chunks, start_APD_time, target_voltage, max_APD_voltage, state_chunks = APD_df(df, cycle_length)
                APD_dfs = plot_APD(df_chunks, state_chunks, start_APD_time, target_voltage,
                                   max_APD_voltage, stim_duration, plot_fig)
                apd_path = os.path.join(destination_dir,
                                        f'{model} APD df {cell_type} at {cycle_length}ms.parquet')
                APD_dfs.to_parquet(apd_path, index=False)
                print(f"Saved APD parquet: {apd_path}")
