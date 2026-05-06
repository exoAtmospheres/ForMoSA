# Folder Structure

ForMoSA touches a lot of files — model grids, adapted sub-grids, observation
files, results, and plots. Keeping them organised from the start saves a
surprising amount of grief later.

## Recommended layout

```bash
~/formosa_desk/
├── atm_grids/              # Native model grids (.nc files)
├── project/
│   └── sample_name/        # A homogeneous set of observations (same wavelength grid)
│       ├── adapted_grid/   # Adapted sub-grids (written by analysis.adapt())
│       └── target_name/    # One target
│           ├── data/       # Input .fits observation files
│           └── results/    # NS results, plots, and saved state
├── (ForMoSA/)              # Source installation only
├── (PyMultiNest/)          # Source installation only
└── (MultiNest/)            # Source installation only
```

## What goes where

| Directory | Contents |
|-----------|----------|
| `atm_grids/` | Downloaded model grid `.nc` files (BT-Settl, Exo-REM, ATMO, …) |
| `sample_name/adapted_grid/` | Sub-grids written by `analysis.adapt()`. Re-usable across all targets observed with the same instrument setup, so you only pay the adaptation cost once. |
| `target_name/data/` | Your `.fits` observation files. One file per instrument in MOSAIC mode. |
| `target_name/results/` | Everything ForMoSA writes: `ns_results.json`, `.npz` sub-grids, corner plots, chain diagnostics. |

## Mapping the layout to `ConfigPath`

```python
from ForMoSA.config.global_config import ConfigPath

config_path = ConfigPath(
    model_path        = "~/formosa_desk/atm_grids/BT-Settl.nc",
    observation_path  = ["~/formosa_desk/project/sample_name/target_name/data/obs.fits"],
    adapt_store_path  = "~/formosa_desk/project/sample_name/adapted_grid/",
    result_path       = "~/formosa_desk/project/sample_name/target_name/results/",
)
```

```{note}
`adapt_store_path` is intentionally shared between targets in the same sample.
If two targets were observed with the same instrument and wavelength grid,
the adapted sub-grid is identical — no need to recompute it.
```
