# MOSAIC Best Practices

MOSAIC mode allows ForMoSA to fit **multiple datasets simultaneously**, each
with its own likelihood, while sharing a common set of physical parameters.
This page explains how MOSAIC works, when to use it, and how to avoid common
pitfalls.

## What MOSAIC does

In standard mode, ForMoSA evaluates a single log-likelihood against one
observation. In MOSAIC mode, it evaluates one log-likelihood per observation
and combines them into a **meta-likelihood**:

```{math}
\ln \mathcal{L}_\text{total} = \sum_{i=0}^{N-1} \ln \mathcal{L}_i
```

Each observation can have its own wavelength range (`wav_fit_i`), likelihood
type (`logL_type_i`), and intercalibration factor (`alpha_i`). Physical
parameters like T_eff, log g, and radius are shared across all observations.

## Setting up MOSAIC

Provide one `.fits` file per instrument in `observation_path`:

```python
from ForMoSA.config.global_config import ConfigPath, ConfigInversion, ConfigParameters

config_path = ConfigPath(
    observation_path = [
        "data/sphere_yjh.fits",   # index 0
        "data/gravity_k.fits",    # index 1
        "data/nircam_photo.fits", # index 2
    ],
    adapt_store_path = "adapted_grid/",
    result_path      = "results/",
    model_path       = "atm_grids/ExoREM.nc",
)
```

Use per-instrument suffixes for local parameters:

```python
config_parameters = ConfigParameters(
    par1    = ["uniform", "800",  "2000"],   # Teff — shared
    par2    = ["uniform", "3.0",  "5.5"],    # log g — shared
    r       = ["uniform", "0.5",  "3.0"],    # radius — shared
    d       = ["constant", "27.7"],           # distance — shared
    alpha_0 = ["uniform", "0.5", "2.0"],     # intercal. for SPHERE
    alpha_1 = ["uniform", "0.5", "2.0"],     # intercal. for GRAVITY
    alpha_2 = ["uniform", "0.5", "2.0"],     # intercal. for NIRCam
)
```

Per-instrument `logL_type` and `wav_fit` in `ConfigInversion`:

```python
config_inversion = ConfigInversion(
    ns_algo   = "pymultinest",
    npoints   = 300,
    logL_type = ["chi2", "chi2", "chi2"],
    wav_fit   = ["0.95, 1.65", "2.0, 2.45", "2.0, 5.0"],
)
```

## When to use MOSAIC

- **Heterogeneous datasets** — different instruments with different resolutions
  and calibration histories.
- **Broadband SED coverage** — combining optical photometry, near-IR
  spectroscopy, and mid-IR photometry.
- **Mitigating calibration biases** — per-instrument `alpha` parameters absorb
  systematic flux offsets without corrupting the shared physical posteriors.

```{note}
See [Ravet et al. (2025)](https://ui.adsabs.harvard.edu/abs/2023A%26A...670L...9P)
for a detailed study of how MOSAIC handles biases in heterogeneous datasets for
$\beta$ Pic b.
```

## Common pitfalls

**Too many `alpha` parameters**
: If every observation has a free `alpha`, the likelihood surface can become
  degenerate — especially when observations overlap in wavelength. As a rule
  of thumb, fix `alpha` for your most reliably calibrated dataset and let it
  float for the others.

**Resolution mismatch across datasets**
: The `target_res_mod` in `ConfigAdapt` must be set consistently. If one
  instrument sees the grid at R=4000 and another at R=100, the adapted
  sub-grids will have different wavelength samplings — which is correct —
  but make sure `wav_fit` for each observation only covers wavelengths where
  that instrument actually has data.

**Unequal dataset weights**
: A high-resolution spectrum with 2000 wavelength points will dominate the
  meta-likelihood over a 3-point photometric SED. Consider whether this is
  physically justified, or whether a noise-scaling likelihood
  (`chi2_noisescaling`) is more appropriate for the spectroscopic observation.

**MOSAIC vs independent runs**
: Running ForMoSA separately on each instrument and then comparing the
  posteriors is *not* equivalent to MOSAIC. MOSAIC enforces a shared physical
  model during sampling; independent runs cannot enforce that constraint.
