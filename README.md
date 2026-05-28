# AP Model Stability

A Python toolkit for running and comparing the stabilisation behaviour of human ventricular action-potential models. Ten widely-used models are bundled as a single Python package; one orchestrator script drives them in batch (paced or unpaced); two plotting scripts turn the resulting parquet files into stabilisation diagnostics, ion-trajectory plots, and per-class heatmaps.

## Repository layout

```
AP Model Stability/
├── highthroughput_stabilisation_script.py    # main entry point — runs the models
├── calculate_AP_parameters.py                # APD/AP-shape/Ca-transient extraction
├── conductances.py                           # per-model conductance lookup table
├── Ventricular_AP_Models/                    # the 10 ODE model implementations
│   ├── ORd.py            # O'Hara-Rudy 2010
│   ├── Dutta2017.py      # Dutta 2017 
│   ├── TNNP06.py         # Ten Tusscher-Noble-Noble-Panfilov 2006
│   ├── TOR2019.py        # Tomek-O'Hara-Rudy 2019
│   ├── TOR_DCl2020.py    # TOR 2019 with dynamic chloride
│   ├── GB.py             # Grandi-Bers 2010
│   ├── Doste2022.py      # Doste 2022 (TOR-ORd variant with β-adrenergic signalling)
│   ├── Doste2022_signaling.py
│   ├── Morotti2021.py    # Morotti 2021
│   ├── BPS2020.py        # BPS 2020
│   └── BPSLand2022.py    # BPS with Land 2017 contractile mechanics
├── Plotting/
│   ├── Paced_plots.py             # plots for paced runs (APD, ion traces, Ca transients)
│   ├── Unpaced_plots.py           # plots for unpaced runs (drift, stabilisation times)
│   └── state_class_mapping.xlsx   # state-name → functional-class lookup (see below)
└── Outputs/                       # created on first run (gitignored)
    ├── Paced/                     # raw paced simulation data
    ├── Unpaced/                   # raw unpaced simulation data
    ├── Paced Plots/               # paced figures
    └── Unpaced Plots/             # unpaced figures + per-class analysis
```

## Installation

Requires Python 3.10+ with the standard scientific stack:

```bash
pip install numpy scipy pandas matplotlib seaborn numba pyarrow openpyxl plotly
```

The model files use `@njit` decorators from [Numba](https://numba.pydata.org/) for speed — first invocation of each model JIT-compiles its ODE function (a one-off ~10–30 s cost per model).

## Quick start

1. Clone the repo and `cd` into it:
   ```bash
   git clone <your-fork-url>
   cd "AP Model Stability"
   ```
2. Open `highthroughput_stabilisation_script.py` and edit the config block near the top:
   ```python
   models = ["Morotti 2021", "TNNP06", "GB 2010", "ORd 2010"]
   protocol = 'paced'        # or 'unpaced'
   cycles = 2000             # number of beats per simulation
   BARS = True             # enable β-AR signalling for Doste/Morotti
   ```
3. Run the orchestrator:
   ```bash
   python highthroughput_stabilisation_script.py
   ```
   Outputs land in `Outputs/Paced/` or `Outputs/Unpaced/` depending on the protocol.
4. Generate plots:
   ```bash
   python Plotting/Paced_plots.py      # for paced runs
   python Plotting/Unpaced_plots.py    # for unpaced runs
   ```
   Figures and summary spreadsheets land in `Outputs/Paced Plots/` or `Outputs/Unpaced Plots/`.

## How `highthroughput_stabilisation_script.py` works

The script iterates a triple-nested loop over `(model, cell_type, cycle_length)`:

- `models`: which AP models to run (one of the names in the `Ventricular_AP_Models/` directory; see the elif tree in the script for the exact spellings).
- `cell_types`: automatically `['ENDO', 'EPI']` for Morotti / Doste, otherwise `['ENDO', 'M', 'EPI']`.
- `cycle_lengths`: `[500, 1000, 2000]` ms for paced (covers ~2, 1, 0.5 Hz pacing), `[1000]` for unpaced.

For each combination, the script calls the model's `run_*_Model(...)` function with the protocol-specific stimulus amplitude (the orchestrator omits `amp` when paced so the model's baked-in default is used; when unpaced it passes `amp=0` to silence the stimulus).

The model returns a pandas DataFrame of state-vector trajectories vs. time. The script then saves:

- **Paced**: a slim ion-trace parquet (`time`, `Cai`, `Nai`, `Ki`, and `Cl_i` where present), plus a per-cycle APD parquet computed by `calculate_AP_parameters.Calculate_APD` (APD30/50/60/90, dV/dt-max, resting potential, AP amplitude, Ca-transient max/min/amplitude).
- **Unpaced**: the **full** state-vector DataFrame, written as chunked parquet files (500 000 rows per chunk) into a `sim chunks` subdirectory. The full state is preserved because unpaced runs are used to study drift across every model state, not just APD-related metrics.

### Output folder structure

```
Outputs/
├── Paced/
│   └── {Model name}/
│       ├── {Model} APD df {cell_type} at {cycle_length}ms.parquet      # APD metrics per beat
│       └── {Model} ion trace {cell_type} at {cycle_length}ms.parquet   # time + ions
│
├── Unpaced/
│   └── {Model name}/
│       └── {Model} sim chunks {cell_type} at {cycle_length}ms/
│           ├── simulation_chunk_000.parquet                            # full state, chunked
│           ├── simulation_chunk_001.parquet
│           └── ...
│
├── Paced Plots/
│   ├── APD stabilisation plots.png
│   ├── Concentration stabilisation plots.png
│   ├── Ca Transient stabilisation plots.png
│   ├── AP Upstroke Morphology stabilisation plots.png
│   └── Chloride Current stabilisation plots.png # Only when Doste o Morotti models have been simuated
│
└── Unpaced Plots/
    ├── Strategy A plots.png                              # per-state deviation trajectories
    ├── Strategy B scatter plots.png                      # stabilisation-time distributions
    ├── time_to_stabilisation.xlsx                        # raw fresh times per state per model
    ├── categorised_time_to_stabilisation.xlsx            # the above + Class column joined
    ├── stabilisation_summary_by_class.xlsx               # mean/max times aggregated by class
    └── *_heatmap.png                                     # per-class heatmaps
```
## About `state_class_mapping.xlsx`

This is a static lookup file shipped with the repo. It has one sheet per model variant, each with two columns:

| Column | Meaning |
|---|---|
| `state` | The exact name of the state variable as it appears in the model's output DataFrame (e.g. `Cai`, `m`, `Pb_dyad`, `PKACII_PKI`). |
| `Class` | A human-readable functional class (e.g. `Voltage`, `Na Dynamics`, `Ca Dynamics`, `I_CaL Gates`, `CICR`, `CaMKII Dynamics`, `β-AR`, `Tension`). |

The class assignments are author-curated; they group hundreds of individual state variables into ~10–15 functional categories per model so the stabilisation-time results can be summarised meaningfully (e.g. "Morotti's β-AR module stabilises in ~1100 s while its I_Na gates stabilise in ~700 s").

**The file contains no time data** — the actual stabilisation times are always computed fresh from the user's own simulation output. The mapping file is a pure name → category lookup, and works the same way regardless of how long you run the simulations.

## About the dagger symbol (†)

Two sheets in `state_class_mapping.xlsx` are spelled with a trailing dagger (†):

| Model Name | Meaning |
|---|---|
| `Doste` | Doste 2022 model **with** β-adrenergic signalling enabled (`BARS = 'True'`). |
| `Doste†` | Doste 2022 model **with β-AR signalling disabled** (`BARS = 'False'`). |
| `Morotti` | Morotti 2021 model with `flag_BARS = True` (β-AR module driving phosphorylation). |
| `Morotti†` | Morotti 2021 model with `flag_BARS = False` (β-AR module clamped). |

## Citing the models

If you use this code, please also cite the original papers for whichever models you ran:

| Model | Citation |
|---|---|
| ORd 2011 | O'Hara T, Virág L, Varró A, Rudy Y. *Simulation of the undiseased human cardiac ventricular action potential.* PLoS Comput Biol 2011. |
| Dutta 2017 | Dutta S, et al. *Optimization of an in silico cardiac cell model for proarrhythmia risk assessment.* Front Physiol 2017. |
| TNNP06 | ten Tusscher KH, Panfilov AV. *Alternans and spiral breakup in a human ventricular tissue model.* Am J Physiol Heart Circ Physiol 2006. |
| TOR 2019 | Tomek J, et al. *Development, calibration, and validation of a novel human ventricular myocyte model in health, disease, and drug block.* eLife 2019. |
| GB 2010 | Grandi E, Pasqualini FS, Bers DM. *A novel computational model of the human ventricular action potential and Ca transient.* J Mol Cell Cardiol 2010. |
| Doste 2022 | Doste R, et al. (TOR-ORd-BARS implementation; see https://github.com/rdoste/ToR-ORd-BARS). |
| Morotti 2021 | Morotti S, et al. *A novel computational human ventricular myocyte model with β-adrenergic signalling and CaMKII activation.* See original MATLAB implementation in https://github.com/drgrandilab/Morotti-et-al-2021-Cross-species-translators-of-electrophysiological-response/tree/main/Updated%20human%20ventricular%20model |
| BPS 2020 / BPSLand 2022 | Bartolucci C, Passini E, Severi S. (BPS-Land 2022 https://pmc.ncbi.nlm.nih.gov/articles/PMC9198403/) |

