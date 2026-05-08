# ForMoSA Tutorials v2.0 — Design Spec

**Date:** 2026-05-08
**Branch:** `class_docs`
**Scope:** Six self-contained tutorial notebooks (+ one `.md`) covering the full ForMoSA v2.0 API. Tutorials are Python notebooks except the cluster deployment tutorial which is a Markdown file.

---

## Goal

Replace the two outdated v1.x demo notebooks with six completely new, beginner-friendly, self-contained tutorials that:
- Work from a `pip install ForMoSA` installation (no repo clone required)
- Download observation data and model grids automatically from GitHub Releases if not present
- Use the v2.0 dataclass API as the primary approach, with an INI/`ConfigLoader` alternative shown at the end of each notebook
- Explain the science behind each target and instrument mode, not just the API calls

---

## Decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | Format | `.ipynb` for tutorials 1–5; `.md` for tutorial 6 |
| 2 | Data hosting | GitHub Releases on `exoAtmospheres/ForMoSA`, tagged `tutorial-data-v1` |
| 3 | Grid hosting | Full BT-Settl / Exo-REM grids on GitHub Releases (1 GB+). One download, permanent value — users keep it for their own science |
| 4 | Grid download UX | Show progress bar (tqdm if available, else byte-count fallback). Skip if file already exists. Explicit note: "Keep this file — it works for all tutorials and your own analyses." |
| 5 | Config style | Dataclass API primary; INI file via `ConfigLoader` shown at end of each notebook (Section 7) |
| 6 | Setup structure | Separate cells per concern (Option B): environment check → workspace setup → data download → grid download |
| 7 | Tutorial 2 vsini | Omitted — SINFONI K (R ≈ 4000) is too low resolution for vsini. Only `rv` fitted |
| 8 | Tutorial 3 HCHR | Cover all required FITS extensions; `rv` and `vsini` included (HiRISE R ≈ 140,000 justifies it) |
| 9 | Tutorials 3 & 4 data | Not yet public — data download cells are `# TODO` stubs; all surrounding code is complete |
| 10 | Tutorial 5 | Uses pre-computed `ns_results.json` (Tutorial 2 results, committed to repo). No fitting required |
| 11 | Tutorial 6 | Two patterns: nohup (single-node) + SLURM (multi-node). Same Python run script for both |

---

## Directory & file map

```
docs/tutorials/
├── index.rst                                   UPDATE: list all 6
│
├── photo/vhs1256b/
│   ├── data/VHS1256b_photometry.fits           existing (keep)
│   └── tutorial_photometry.ipynb               NEW (replaces end_to_end_vhs1256b.ipynb)
│
├── spectroscopy/abpicb/
│   ├── data/ABPicb_SINFONI_K.fits              MOVE from sinfoni/abpicb/data/
│   └── tutorial_spectroscopy.ipynb             NEW (replaces end_to_end_abpicb.ipynb)
│
├── hchr/aflep/
│   └── tutorial_hchr.ipynb                     NEW — full code; data cells are TODO stubs
│
├── mosaic/betapicb/
│   └── tutorial_mosaic.ipynb                   NEW — full code; data cells are TODO stubs
│
├── plotting/
│   ├── ns_results.json                         NEW — pre-computed Tutorial 2 results (committed)
│   └── tutorial_advanced_plotting.ipynb        NEW
│
└── cluster/
    └── tutorial_cluster.md                     NEW
```

**Deleted:**
- `docs/tutorials/sinfoni/abpicb/end_to_end_abpicb.ipynb`
- `docs/tutorials/photo/vhs1256b/end_to_end_vhs1256b.ipynb`
- `docs/tutorials/sinfoni/` directory (renamed to `spectroscopy/`)

---

## Standard notebook anatomy

Every notebook (tutorials 1–5) follows this fixed section structure:

```
Preamble (markdown)
  Title, target, what you'll learn, estimated runtime

Section 0: Setup
  Cell A — Environment check
    import ForMoSA; print(ForMoSA.__version__)
    Assert minimum version. Clear ForMoSAError if not installed.

  Cell B — Workspace setup
    TUTORIAL_DIR = Path(".").resolve()
    Create data/, adapted_grid/, results/ next to the notebook

  Cell C — Data download
    DATA_FILE = TUTORIAL_DIR / "data" / "<target>.fits"
    DATA_URL   = "https://github.com/exoAtmospheres/ForMoSA/releases/download/tutorial-data-v1/<file>"
    If not present: download with progress bar (tqdm / urllib fallback)
    Validate FITS extensions: print a table of found vs expected
    [TODO stub for tutorials 3 & 4]

  Cell D — Grid download
    GRID_FILE = TUTORIAL_DIR / "grid" / "BT-Settl.nc"   # user can override
    GRID_URL   = "https://github.com/exoAtmospheres/ForMoSA/releases/download/tutorial-data-v1/BT-Settl.nc"
    Note: "This file is ~1 GB. Once downloaded it works for all tutorials and your own science."
    If not present: download with progress bar. Skip if already present.
    Print grid dimensions and parameter ranges.
    [TODO stub for tutorials 3 & 4 — Exo-REM grid]

Section 1: The science
  Markdown cells: what is this object, what instruments observed it,
  what parameters we are trying to measure, what we expect to find.
  Cite the relevant paper(s).

Section 2: Inspect the data
  Open FITS with astropy, print extension table, plot flux vs λ.

Section 3: Configure the analysis
  Primary: dataclass API
    ConfigPath(observation_path=[...], adapt_store_path=..., result_path=..., model_path=...)
    ConfigAdapt(...)
    ConfigInversion(...)
  Each argument annotated with an inline comment.

Section 4: Adapt the grid
  analysis = Analysis(config_path)
  analysis.adapt(config_adapt, config_inversion)
  Print expected runtime. Note on adapted=True for reruns.

Section 5: Run the fit
  ConfigParameters(par1=[...], par2=[...], ...)
  Each prior choice explained (why uniform, why these bounds).
  analysis.nested_sampling(config_parameters, config_adapt, config_inversion)
  Expected runtime and progress bar explanation.

Section 6: Results
  analysis.plot(analysis.ns.results)
  Explain what each output plot shows (corner, chains, radar, best-fit).

Section 7: INI file alternative
  ConfigGenerator().save(TUTORIAL_DIR, "config.ini")
  Open and edit the .ini manually (show key fields)
  cfg = ConfigLoader(TUTORIAL_DIR / "config.ini"); sections = cfg.load()
  Clearly marked as "alternative approach — same results".

Section 8: Next steps
  Links to the next tutorial, getting_started/ pages, API docs.
```

---

## Content plan per tutorial

### Tutorial 1 — Photometry: VHS 1256 b

**Target:** VHS 1256 b — extreme red L/T transition dwarf, 14 filters from SPHERE + NACO + JWST NIRCam/MIRI.
**Grid:** Full BT-Settl
**Parameters fitted:** `par1` (Teff), `par2` (log g), `r`, `d`, `alpha`
**Key teaching points:**
- What photometry tells us vs spectroscopy (SED shape, bolometric luminosity, basic Teff/log g)
- How SVO filter service works: ForMoSA downloads filter curves automatically from `FAC/INS/FILT` FITS extensions
- Physical vs analytical scaling: tutorial uses `alpha` (analytical) because absolute flux calibration across 14 heterogeneous filters is uncertain
- Why `r` and `d` are included even with analytical scaling: degeneracy discussion

**FITS extensions validated in Cell C:**
WAV, WAVE_UNIT, FLX, ERR, FAC, INS, FILT

---

### Tutorial 2 — Spectroscopy: AB Pic b

**Target:** AB Pic b — K-band VLT/SINFONI spectrum, R ≈ 4000.
**Grid:** Full BT-Settl
**Parameters fitted:** `par1` (Teff), `par2` (log g), `r`, `d`, `rv`
**Key teaching points:**
- What changes between photometry and spectroscopy: resolution adaptation, `wav_fit` window, continuum removal
- Why vsini is omitted: SINFONI K (R ≈ 4000) cannot resolve rotational broadening reliably; you need R > 50,000 (explained in science section)
- RV shift: how a Doppler shift appears in a spectrum and why it matters for companions
- The adapt → sample → plot loop from Tutorial 1 is the same; preamble says so explicitly

**FITS extensions validated in Cell C:**
WAV, WAVE_UNIT, FLX, ERR, RES

---

### Tutorial 3 — HCHR mode: AF Lep b

**Target:** AF Lep b — VLT/HiRISE observation (SPHERE + CRIRES+), R ≈ 140,000.
**Grid:** Exo-REM cloudless
**Parameters fitted:** `par1` (Teff), `par2` (log g), `r`, `d`, `rv`, `vsini`
**Data status:** TODO stub — data not yet publicly available.

**Key teaching points:**
- What HCHR mode is: simultaneous high contrast (star suppression) + high resolution (molecular lines)
- Required FITS extensions for HCHR — full table with explanation:

| Extension | Type | Description | Required for HCHR |
|-----------|------|-------------|:-----------------:|
| `WAV` | 1D array | Wavelength in μm | Yes |
| `WAVE_UNIT` | string | Wavelength unit (`um`, `nm`, `AA`) | Yes |
| `FLX` | 1D array | Planet flux (speckle-subtracted) | Yes |
| `ERR` | 1D array | Per-channel noise | Yes |
| `RES` | 1D array | Spectral resolution per wavelength point | Yes |
| `STAR_FLUX` | 2D array | Stellar speckle reference spectra (one column per nod/epoch) | **Yes — triggers `hc_mode=True`** |

- `hc_mode` is not set manually — the presence of a `STAR_FLUX` column triggers it automatically
- Consequences of `hc_mode=True`: continuum NOT removed from models; `_hc_modeling` applied; no analytical scaling
- Why `rv` and `vsini` are included here: HiRISE R ≈ 140,000 resolves individual CO and OH lines, making both parameters measurable
- Exo-REM vs BT-Settl: brief comparison of which grids are appropriate for young directly-imaged companions

**FITS extensions validated in Cell C:**
WAV, WAVE_UNIT, FLX, ERR, RES, STAR_FLUX
Cell C is a `# TODO` stub with clear comment indicating what file will go where once data is public.

---

### Tutorial 4 — MOSAIC mode: β Pic b

**Target:** β Pic b — combining a medium-resolution spectrum + multi-filter photometry.
**Grid:** Full BT-Settl
**Parameters fitted:** `par1` (Teff), `par2` (log g), `r`, `d`, `rv`, `alpha_0` (spectroscopy intercalibration), `alpha_1` (photometry intercalibration)
**Data status:** TODO stub — data not yet publicly available.

**Key teaching points:**
- What MOSAIC mode is: separate likelihood per instrument, combined meta-likelihood
- `ConfigPath(observation_path=["spec.fits", "photo.fits"])` — list of two FITS files
- MOSAIC-indexed parameters: `alpha_0` for the spectrum, `alpha_1` for the photometry
- When to use MOSAIC: heterogeneous datasets with potentially different flux calibrations
- Common pitfall: too many free `alpha` parameters with too few data points = overfitting
- How the meta-likelihood is formed: `log L_total = log L_spec + log L_photo`

**FITS extensions validated in Cell C:**
Both files validated separately. Spec: WAV, WAVE_UNIT, FLX, ERR, RES. Photo: WAV, WAVE_UNIT, FLX, ERR, FAC, INS, FILT.
Cell C is a `# TODO` stub.

---

### Tutorial 5 — Advanced plotting

**No fitting required.** Ships with `plotting/ns_results.json` — Tutorial 2 (AB Pic b) results pre-computed and committed to the repo (~100 KB JSON).

**Setup section:**
- Cell A: Environment check (same as other notebooks)
- Cell B: Load `ns_results.json` → `NSResults.from_dict(json.load(open("ns_results.json")))`
- No data/grid download cells needed

**Content:**

*5.1 — Reading results*
- `results.summary(sigma=1)` — printed table, what each column means
- `results.free_parameters` — list of fitted parameter names
- `results.median_parameters` — dict of weighted posterior medians
- `results._interval(sigma=1)` — 1σ credible intervals per parameter
- `results._quantile_parameters(q=0.16)` — arbitrary quantile access
- `results.logz` — Bayesian log-evidence: value and uncertainty; brief explanation of what it means for model comparison

*5.2 — Computing χ²_red manually*
```python
# Best logL from weighted posterior
best_logL = np.average(results.logl[results.burn_in:], weights=results.weights[results.burn_in:])
n_data    = len(obs.wave)
n_free    = len(results.free_parameters)
chi2_red  = -2 * best_logL / (n_data - n_free)
```

*5.3 — Corner plot customisation*
Access via `from ForMoSA.core.config import PLOTS_CONFIG`.
```python
PLOTS_CONFIG.CornerPlot.set_corner_plot_config(
    bins=40,          # histogram bin count (default 80)
    color='steelblue', # contour + histogram colour
    smooth=0.5,        # Gaussian smoothing of 2D contours
    show_titles=True,  # median ± σ in axis titles
    quantiles=(0.16, 0.5, 0.84),  # plotted quantile lines
)
```
Each argument explained with its effect on the plot.

*5.4 — Chains plot customisation*
```python
PLOTS_CONFIG.ChainsPlot.set_chains_plot_config(
    show_weights=False,       # overlay weight trace on each chain
    plot_best_value=True,     # horizontal line at posterior median
    color_chains='teal',
)
```
What burn-in is and how to read it from the plot.

*5.5 — Radar plot*
What the normalised radial axes represent. How to change the uncertainty quantiles:
```python
PLOTS_CONFIG.RadarPlot.set_radar_plot_config(quantiles=(0.05, 0.95))
```

*5.6 — Best-fit spectrum customisation*
```python
PLOTS_CONFIG.BestFitPlot.set_best_fit_plot_config(
    color_fit='royalblue',
    linewidth=1.5,
)
```
`plot_native_model=True` vs `False` — what the native (un-convolved) model shows.

*5.7 — Per-observation colour (MOSAIC context)*
```python
import matplotlib.cm as cm
for obs in analysis.ns.restricted_observations:
    obs.plot_config.set_plot_config(
        color=cm.inferno(analysis.observations.mcolors_normalize(obs.central_wavelength))
    )
```

*5.8 — Saving publication-quality figures*
```python
fig = analysis.plot(analysis.ns.results)  # returns Figure object
fig.savefig("corner_publication.pdf", dpi=300, bbox_inches="tight")
```

---

### Tutorial 6 — Cluster/MPI deployment (`.md`)

**Sections:**

1. **When you need this** — nestle runs in a single thread; PyMultiNest distributes likelihood evaluations across MPI ranks. Rule of thumb: > 3 free parameters → PyMultiNest.

2. **Prerequisites**
   - Open MPI installed on cluster
   - `mpi4py` and `pymultinest` installed in conda env
   - Module loading commands (cluster-specific, show generic example):
     ```bash
     module load openmpi/4.1.5 gcc/12.2.0
     conda activate env_formosa
     ```

3. **The Python run script** — fully annotated `run_formosa.py`:
   - MPI detection block (`from mpi4py import MPI` with ImportError fallback)
   - `IS_ROOT = (RANK == 0)` pattern
   - Step 1 (adapt): rank 0 only — already uses `ThreadPool` internally
   - `COMM.Barrier()` between steps — why it is needed
   - Step 2 (NS): all ranks participate — PyMultiNest handles inter-rank communication
   - Step 3 (plot): rank 0 only — `matplotlib.use("Agg")` for non-interactive backend
   - Why imports come after MPI init (fork-safety)
   - `adapted=True` flag for the NS step (reads from disk, no re-adaptation)

4. **Pattern A — nohup (single node)**
   ```bash
   nohup mpirun -np 12 python run_formosa.py > run.log 2>&1 &
   echo "PID: $!"
   tail -f run.log
   ```
   - Safe ceiling: ~85% of available cores (e.g., 12 on a 14-core machine)
   - Monitoring: `tail -f run.log`; stopping: `kill <PID>`

5. **Pattern B — SLURM (multi-node)**
   Full annotated `job.sh`:
   ```bash
   #!/bin/bash
   #SBATCH --job-name=formosa
   #SBATCH --nodes=2
   #SBATCH --ntasks-per-node=16
   #SBATCH --time=04:00:00
   #SBATCH --partition=compute
   #SBATCH --output=formosa_%j.log

   module load openmpi/4.1.5 gcc/12.2.0
   conda activate env_formosa

   mpirun -np $SLURM_NTASKS python run_formosa.py
   ```
   Submit: `sbatch job.sh`; monitor: `squeue -u $USER`

6. **Expected speedup** — brief guidance: PyMultiNest scales well up to ~32 ranks for most ForMoSA problems; beyond that, communication overhead dominates.

---

## Implementation order

1. `docs/tutorials/index.rst` — update to list all 6 tutorials
2. Delete old notebooks and `sinfoni/` directory; create new directory structure
3. `docs/tutorials/photo/vhs1256b/tutorial_photometry.ipynb`
4. `docs/tutorials/spectroscopy/abpicb/tutorial_spectroscopy.ipynb` (move FITS file from `sinfoni/`)
5. `docs/tutorials/hchr/aflep/tutorial_hchr.ipynb` (TODO stubs for data)
6. `docs/tutorials/mosaic/betapicb/tutorial_mosaic.ipynb` (TODO stubs for data)
7. `docs/tutorials/plotting/ns_results.json` — generate from Tutorial 2 results (or create synthetic)
8. `docs/tutorials/plotting/tutorial_advanced_plotting.ipynb`
9. `docs/tutorials/cluster/tutorial_cluster.md`
10. Final `make html` build check

---

## Constraints

- Do **not** modify any docstrings in `ForMoSA/` source
- Do **not** modify `whats_new.rst`
- All notebooks use the v2.0 dataclass API as primary; INI approach shown as Section 7
- Notebooks must run cell-by-cell from top to bottom without errors (excluding TODO stub cells)
- Tone: beginner-friendly but not condescending — explain the physics, not just the API calls
- GitHub Release tag: `tutorial-data-v1` (to be created by maintainer before publication)
- The `ns_results.json` for Tutorial 5 may be synthetic (realistic parameter values, correct structure) if Tutorial 2 has not been run by the time notebooks are written
