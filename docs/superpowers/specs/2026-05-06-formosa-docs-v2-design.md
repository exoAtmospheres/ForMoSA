# ForMoSA Documentation v2.0 — Design Spec

**Date:** 2026-05-06  
**Branch:** `class_docs`  
**Scope:** Update all non-tutorial documentation from v1.1.6 to v2.0.0. Tutorials are out of scope and will be handled separately.

---

## Goal

The existing docs were written for v1.1.6 (script-based API, `GlobFile` + `.ini` config). ForMoSA v2.0.0 is a complete rewrite around a class-based API (`Analysis`, Python dataclasses). The docs need to reflect this new API, remove obsolete content, improve navigability, and migrate away from `.ipynb` as the primary format for reference content.

---

## Decisions

| # | Question | Decision |
|---|----------|----------|
| 1 | Where does "Getting Started" live? | Rename `guidelines/` → `getting_started/`; build new hub content there |
| 2 | Where does "Scaling" live? | Standalone top-level section at same level as Getting Started |
| 3 | File format for new pages? | RST for index/toctree files; MD (MyST) for content-heavy subpages |
| 4 | Flowchart / code graph on welcome page? | Use `schema_ForMoSA.png` (from `paper/`) for flowchart; `.. todo::` placeholder for code graph |
| 5 | Config parameter detail level? | Full inline per-parameter notes in Getting Started (not pointer to API) |

---

## Execution strategy

**Option 2 — Structure first, content second.**

1. Build the skeleton: rename dirs, create stub files, update toctrees. Verify `make html` passes with no broken links.
2. Fill in each content page in sequence (Welcome → Installation → Getting Started pages → Scaling pages → Tutorials stub → API check).

Build command (in ForMoSA conda environment):
```bash
cd /Users/rajpoot/Karmabhumi/Packages/ForMoSA/docs
conda run -n env_formosa make html
```

---

## Directory & file map

```
docs/
├── conf.py                          UPDATE: release → '2.0.0'
├── index.rst                        UPDATE: full welcome page rewrite
├── installation.rst                 UPDATE: simplify + remove torch section
├── whats_new.rst                    NO CHANGE
│
├── getting_started/                 RENAME from guidelines/
│   ├── index.rst                    new hub page (RST, toctree only)
│   ├── folder_structure.md          NEW (was inline in guidelines/index.rst)
│   ├── data_formatting.md           NEW (replaces input_format/obs_format.ipynb)
│   ├── model_grid.md                NEW (replaces input_format/model_format.ipynb)
│   └── config_file.md               NEW (replaces input_format/config_format.ipynb)
│
├── scaling/                         NEW top-level section
│   ├── index.rst                    (RST, toctree)
│   ├── analytical_vs_physical.md    NEW
│   └── mosaic_best_practices.md     NEW
│
├── tutorials/                       RENAME from demos/ — stub only
│   └── index.rst                    placeholder with .. todo::
│
├── api/                             NO CHANGE to structure
│   └── index.rst                    minor: verify all modules listed
│
└── _static/
    └── schema_ForMoSA.png           COPY from paper/
```

**Retired:**
- `docs/guidelines/input_format/` — 3 old `.ipynb` files replaced by `.md` pages
- `docs/demos/` — renamed to `tutorials/`

---

## Content plan per page

### `conf.py`
- `release = '2.0.0'`
- Add all new contributors to `copyright` / `author` strings

### `index.rst` (Welcome page)
- **Short description:** "ForMoSA is an open-source Python package for modeling exoplanetary atmospheres using a forward modeling approach. It compares observed spectra and photometry against grids of atmospheric models via nested sampling to derive posterior distributions on physical parameters."
- **Features:** class-based API, MOSAIC multi-instrument, 3 NS back-ends (nestle/PyMultiNest/UltraNest), high-contrast mode, SVO filter service, configurable priors, comprehensive plotting, flexible config (INI or dataclasses)
- **Statement of need:** condensed from `paper/paper.md` — 3–4 sentences covering the gap ForMoSA fills
- **How does ForMoSA work?** — brief paragraph + embed `schema_ForMoSA.png` + `.. todo::` for code graph
- **Example plots:** embed existing `_static/` assets (accuracy_formosa, priors_teff_plot, rv, vsini) with captions
- **Attribution:** cite Petrus et al. (2023)
- **Acknowledgements:** updated from paper's Acknowledgements section (Code/Astro, ANR MIRAGES, ForM-X workshops, IPAG/Lagrange/MPIA)

### `installation.rst`
- Conda environment creation (keep macOS ARM64 note)
- **PyPI install** — single clean block: `pip install ForMoSA` + `conda install dask netCDF4 bottleneck`
- **Source install** — `git clone` + `pip install -e .` (remove the long manual per-package pip list)
- **PyMultiNest** — keep, still required for parallelisation
- ~~torch users~~ — **removed entirely**

### `getting_started/index.rst`
- One-paragraph intro: "Everything you need to run your first ForMoSA analysis."
- Toctree linking to all four subpages with short descriptions

### `getting_started/folder_structure.md`
- Why a structured workspace helps
- Folder tree (same as current `guidelines/index.rst`) with bullet explanations

### `getting_started/data_formatting.md`
- Primer: ForMoSA reads `.fits` table files; each extension is a NumPy array of equal length
- Full FITS extension table:

| Extension | Description | Spectroscopic | Photometric | HCHR |
|-----------|-------------|:---:|:---:|:---:|
| `WAV` | Wavelength array | Yes | Yes | Yes |
| `WAVE_UNIT` | Wavelength unit string | Yes | Yes | Yes |
| `FLX` | Flux array | Yes | Yes | Yes |
| `ERR` | 1-D flux uncertainty | Yes | Yes | Yes |
| `COV` | Covariance matrix (alternative to ERR) | Yes | No | Yes |
| `RES` | Spectral resolution per wavelength point | Yes | No | Yes |
| `FAC` | Observatory identifier (SVO-compatible) | No | Yes | No |
| `INS` | Instrument identifier (SVO-compatible) | No | Yes | No |
| `FILT` | Filter identifier (SVO-compatible) | No | Yes | No |
| `STAR_FLUX` | Stellar speckle reference spectrum | No | No | Yes |

- Code snippet: open a `.fits` file with `astropy`, print extensions, plot flux vs wavelength

### `getting_started/model_grid.md`
- What model grids are: pre-computed self-consistent atmospheric models stored as `xarray` `.nc` files
- Required xarray dimensions/coordinates (`par1`…`par4`, `wavelength`) and data variables (`flux`, `res`)
- List of publicly available grids with download links (ATMO, BT-Settl, ExoREM)
- Code snippet: open a grid with `xarray`, slice at a parameter point, plot a spectrum

### `getting_started/config_file.md`
- Two ways to configure: Python dataclasses (preferred) vs INI file loaded via `ConfigLoader`
- Table: which dataclass controls what
- **Mode validity matrix** — rows = parameters, columns = Standard / MOSAIC / Photometry / HCHR:

| Parameter | Standard | MOSAIC | Photometry | HCHR |
|-----------|:---:|:---:|:---:|:---:|
| `par1`–`par4` | Yes | Yes | Yes | Yes |
| `r` | Yes | Yes | Yes | Yes |
| `d` | Yes | Yes | Yes | Yes |
| `rv` | Yes | Yes | No | Yes |
| `vsini` | Yes | Yes | No | Yes |
| `ld` | Yes | Yes | No | No |
| `alpha` | Yes | Yes | Yes | Yes |
| `bb_T` | Yes | Yes | No | No |

- Detailed per-parameter notes (derived from `global_config.py` docstrings): prior syntax `["uniform","min","max"]`, meaning of each parameter, links to API for full type/validation info

### `scaling/index.rst`
- One-paragraph intro to the scaling question
- Toctree to two subpages

### `scaling/analytical_vs_physical.md`
- Physical scaling: `r` (radius in R_Jup) + `d` (distance in pc) — flux = model × (r/d)²
- Analytical scaling: `alpha` — flux = model × α
- When to use which: physical when you want to constrain radius; analytical when the absolute flux calibration is unreliable
- Worked example showing both configurations

### `scaling/mosaic_best_practices.md`
- What MOSAIC mode is: separate likelihood per instrument, combined meta-likelihood, per-instrument `alpha` intercalibration parameter
- When to use it: heterogeneous datasets (different instruments, resolutions, epochs)
- How to set it up: `observation_path` list + MOSAIC-indexed config params (`rv_0`, `rv_1`, …)
- Pitfalls: over-fitting with too many α parameters, resolution mismatch effects

### `tutorials/index.rst`
- Short intro noting tutorials are in progress
- `.. todo::` block listing all 6 planned tutorials: Photometry (VHS 1256 b), Spectroscopy (AB Pic b), HCHR (AF Lep b), MOSAIC (β Pic b), Advanced plotting, Cluster/MPI deployment

### `api/index.rst`
- Verify all modules are listed: `analysis`, `config`, `core`, `observation`, `grid`, `parameter`, `filter`, `transform`, `nested_sampling`, `plotting`, `main_utilities`

---

## Implementation order

1. `conf.py` — update release version
2. `_static/schema_ForMoSA.png` — copy from `paper/`
3. **Skeleton:** rename `guidelines/` → `getting_started/`, rename `demos/` → `tutorials/`, create `scaling/`, update `index.rst` toctree — run `make html` to verify no broken links
4. `index.rst` — welcome page full rewrite
5. `installation.rst` — simplify + remove torch section
6. `getting_started/index.rst` — hub page
7. `getting_started/folder_structure.md`
8. `getting_started/data_formatting.md`
9. `getting_started/model_grid.md`
10. `getting_started/config_file.md`
11. `scaling/index.rst` + `scaling/analytical_vs_physical.md` + `scaling/mosaic_best_practices.md`
12. `tutorials/index.rst` — stub
13. `api/index.rst` — verify
14. Final `make html` — check for warnings/errors

---

## Constraints

- Do **not** modify docstrings anywhere in `ForMoSA/` source
- Do **not** modify `whats_new.rst`
- Tone: clean, clear, concise, professional — with a light touch of wit where natural. Not casual, not robotic.
- All `.md` content pages use MyST Markdown (fenced code blocks, `{note}`, `{important}` directives)
- All `.rst` index files use standard Sphinx RST
