# Data Formatting

ForMoSA reads observations from **FITS table files** (`.fits`). Each extension
holds a NumPy array; all arrays in the same file must be the same length. This
page explains what extensions are required for each observation type and shows
how to verify your file before running an analysis.

## Required extensions

The canonical name is what ForMoSA stores internally. Any alias in the **Accepted aliases** column
is equally valid when writing your FITS file — ForMoSA resolves them automatically.
`WAVELENGTH_UNIT` defaults to `"um"` (microns) if omitted.

| Canonical name | Accepted aliases | Description | Spectroscopic | Photometric | HCHR |
|----------------|-----------------|-------------|:---:|:---:|:---:|
| `WAVELENGTH` | `WAVE`, `WAV`, `LAMBDA` | Wavelength array | Required | Required | Required |
| `WAVELENGTH_UNIT` | `WAVE_UNIT`, `UNIT` | Wavelength unit string — one value repeated `N` times, e.g. `"um"`, `"nm"`, `"AA"`. Defaults to `"um"` if absent. | Optional | Optional | Optional |
| `FLUX` | `FLX` | Flux array | Required | Required | Required |
| `ERROR` | `ERR`, `SIGMA` | 1-σ flux uncertainty per point (use instead of `COVARIANCE`) | Required* | Required | Required* |
| `COVARIANCE` | `COV` | Full covariance matrix — shape `(N, N)` — alternative to `ERROR` | Required* | — | Required* |
| `RESOLUTION` | `RES` | Spectral resolution λ/Δλ per wavelength point | Required | — | Required |
| `FACILITY` | `FAC`, `Facility` | Observatory identifier, must match [SVO](https://svo2.cab.inta-csic.es/theory/fps/) | — | Required | — |
| `INSTRUMENT` | `INS` | Instrument identifier, must match [SVO](https://svo2.cab.inta-csic.es/theory/fps/) | — | Required | — |
| `FILTER_ID` | `FILT`, `FILTER`, `FILT_ID` | Filter identifier, must match [SVO](https://svo2.cab.inta-csic.es/theory/fps/) | — | Required | — |
| `STAR_FLUX` | `STAR_FLX` | Stellar speckle reference spectrum | — | — | Required |
| `TRANSMISSION` | `TRANSM` | Telluric or instrumental transmission spectrum applied to the model before comparison | Optional | — | Optional |
| `SYSTEMATICS` | `SYS` | One or more systematic noise components (shape `(N, M)` for `M` components) used for linear decorrelation in HCHR mode | — | — | Optional |

\* `ERROR` or `COVARIANCE` — one of the two is required for spectroscopic observations; `COVARIANCE` is not supported for photometry.

```{important}
For photometric observations, `FACILITY`, `INSTRUMENT`, and `FILTER_ID` must be consistent with
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

unit = np.full(500, "um", dtype="U10")   # one string per wavelength point

hdul = fits.HDUList([
    fits.PrimaryHDU(),
    fits.ImageHDU(wav,  name="WAVELENGTH"),
    fits.ImageHDU(unit, name="WAVELENGTH_UNIT"),
    fits.ImageHDU(flx,  name="FLUX"),
    fits.ImageHDU(err,  name="ERROR"),
    fits.ImageHDU(res,  name="RESOLUTION"),
])

hdul.writeto("my_observation.fits", overwrite=True)
```

## Inspecting and plotting your file

```python
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

with fits.open("my_observation.fits") as hdul:
    hdul.info()                    # prints all extensions and their shapes
    wav = hdul["WAVELENGTH"].data
    flx = hdul["FLUX"].data
    err = hdul["ERROR"].data

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
