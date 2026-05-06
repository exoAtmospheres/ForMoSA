# Data Formatting

ForMoSA reads observations from **FITS table files** (`.fits`). Each extension
holds a NumPy array; all arrays in the same file must be the same length. This
page explains what extensions are required for each observation type and shows
how to verify your file before running an analysis.

## Required extensions

| Extension | Aliases accepted | Description | Spectroscopic | Photometric | HCHR |
|-----------|-----------------|-------------|:---:|:---:|:---:|
| `WAV` | `WAVELENGTH`, `WAVE`, `LAMBDA` | Wavelength array | Yes | Yes | Yes |
| `WAVE_UNIT` | — | Wavelength unit string (e.g. `"µm"`) | Yes | Yes | Yes |
| `FLX` | `FLUX` | Flux array | Yes | Yes | Yes |
| `ERR` | `ERROR`, `SIGMA` | 1-D flux uncertainty (use instead of `COV`) | Yes | Yes | Yes |
| `COV` | — | Full covariance matrix — shape `(N, N)` — alternative to `ERR` | Yes | No | Yes |
| `RES` | `RESOLUTION` | Spectral resolution λ/Δλ per wavelength point | Yes | No | Yes |
| `FAC` | `FACILITY` | Observatory identifier, must match [SVO](https://svo2.cab.inta-csic.es/theory/fps/) | No | Yes | No |
| `INS` | `INSTRUMENT` | Instrument identifier, must match [SVO](https://svo2.cab.inta-csic.es/theory/fps/) | No | Yes | No |
| `FILT` | `FILTER`, `FILTER_ID` | Filter identifier, must match [SVO](https://svo2.cab.inta-csic.es/theory/fps/) | No | Yes | No |
| `STAR_FLUX` | — | Stellar speckle reference spectrum (high-contrast mode only) | No | No | Yes |

```{note}
ForMoSA uses the **first matching alias** it finds. If your file uses `WAVE`
instead of `WAV`, that is fine — both are accepted.
```

```{important}
For photometric observations, `FAC`, `INS`, and `FILT` must be consistent with
the [SVO Filter Profile Service](https://svo2.cab.inta-csic.es/theory/fps/)
naming convention. ForMoSA uses these strings to automatically download and
cache the filter transmission curve.
```

## Creating a FITS file with astropy

```python
import numpy as np
from astropy.io import fits

# Example: K-band spectroscopic observation
wav  = np.linspace(2.0, 2.5, 500)      # µm
flx  = np.random.normal(1.0, 0.05, 500)
err  = np.full(500, 0.05)
res  = np.full(500, 4000.0)            # R ~ 4000

hdul = fits.HDUList([
    fits.PrimaryHDU(),
    fits.ImageHDU(wav,  name="WAV"),
    fits.ImageHDU(flx,  name="FLX"),
    fits.ImageHDU(err,  name="ERR"),
    fits.ImageHDU(res,  name="RES"),
])
# WAVE_UNIT stored as a header keyword on the WAV extension
hdul["WAV"].header["BUNIT"] = "µm"

hdul.writeto("my_observation.fits", overwrite=True)
```

## Inspecting and plotting your file

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

with fits.open("my_observation.fits") as hdul:
    hdul.info()          # prints all extensions and their shapes
    wav = hdul["WAV"].data
    flx = hdul["FLX"].data
    err = hdul["ERR"].data

plt.figure(figsize=(10, 4))
plt.plot(wav, flx, label="flux")
plt.fill_between(wav, flx - err, flx + err, alpha=0.3, label="1σ")
plt.xlabel("Wavelength (µm)")
plt.ylabel("Flux")
plt.legend()
plt.tight_layout()
plt.show()
```

## MOSAIC mode: multiple files

When combining multiple instruments, provide one `.fits` file per instrument and
list them all in `ConfigPath.observation_path`:

```python
from ForMoSA.config.global_config import ConfigPath

config_path = ConfigPath(
    observation_path=[
        "data/sphere_yjh.fits",    # SPHERE YJH low-res spectroscopy
        "data/gravity_k.fits",     # GRAVITY K-band spectroscopy
        "data/nircam_photo.fits",  # JWST NIRCam photometry
    ],
    adapt_store_path="adapted_grid/",
    result_path="results/",
    model_path="atm_grids/ExoREM.nc",
)
```

ForMoSA will assign each file an index (0, 1, 2, …) that you can use to set
per-instrument parameters such as `rv_0`, `rv_1`, etc. in `ConfigParameters`.
