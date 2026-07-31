# Knowing Your Model Grid

ForMoSA compares observations against **pre-computed, self-consistent atmospheric
model grids** stored as [xarray](https://docs.xarray.dev/en/stable/) `.nc` files.
This page explains what a grid file contains, which grids are available, and
how to inspect one before starting an analysis.

## What is a model grid?

A model grid is a multi-dimensional array of synthetic spectra computed by an
atmospheric code (e.g. BT-Settl, Exo-REM, ATMO) over a regular grid of physical
parameters such as effective temperature ($T_{eff}$), surface gravity ($\log g$),
metallicity ($\mathrm{[M/H]}$), and $\mathrm{[C/O]}$ ratio.

ForMoSA uses `xarray.Dataset` to represent grids. The dataset has:

- **Coordinates** — one per physical parameter (e.g. `par1` = $T_{eff}$, `par2` = $\log g$).
  The names `par1`–`par4` are fixed; what they *mean* depends on the grid.
- **Data variable `grid`** — the model fluxes, shape `(N_wavelength, N_par1, N_par2, …)`.
- **Attribute `res`** (`grid.attrs["res"]`) — the native spectral resolution at each
  wavelength point, shape `(N_wavelength,)`. Not a queryable data variable.
- **Coordinate `wavelength`** — the wavelength axis in microns.

## Available grids

The following grids have been pre-formatted for ForMoSA and are available for
download. Save them inside your `atm_grids/` directory.

| Grid | Reference | Download |
|------|-----------|----------|
| BT-Settl | [Allard et al. 2013](https://ui.adsabs.harvard.edu/abs/2013MSAIS..24..128A/abstract) | [Download](https://drive.google.com/file/d/1wvf4A-DupdVnYIpK_HmHE-fobqnYtvEz/view?usp=share_link) |
| Exo-REM | [Charnay et al. 2018](https://ui.adsabs.harvard.edu/abs/2018ApJ...854..172C/abstract) | [Download](https://drive.google.com/file/d/1k9SQjHLnMCwmGOHtraRnhCgiZ1-4J3Wk/view?usp=share_link) |
| ATMO | [Phillips et al. 2020](https://ui.adsabs.harvard.edu/abs/2020A%26A...637A..38P/abstract) | [Download](https://drive.google.com/file/d/1S1dcBD7UiuUCZIcNBNnJi6LMymrnkagM/view?usp=share_link) |

```{note}
Need a grid that isn't listed here? The ForMoSA team can generate custom
formatted grids on request — open an issue on
[GitHub](https://github.com/exoAtmospheres/ForMoSA/issues).
```

## Inspecting a grid

```python
import xarray as xr

# Open the grid (lazy-loads by default — no RAM spike)
grid = xr.open_dataset("atm_grids/BT-Settl.nc")
print(grid)
# Shows dimensions, coordinates, and data variables

# What parameter axes are available?
print(grid.coords)

# Range of Teff (par1) and logg (par2)
print("Teff range:", float(grid.par1.min()), "–", float(grid.par1.max()), "K")
print("logg range:", float(grid.par2.min()), "–", float(grid.par2.max()))
```

## Plotting a single spectrum

```python
import xarray as xr
import matplotlib.pyplot as plt

grid = xr.open_dataset("atm_grids/BT-Settl.nc")

# Select the model closest to Teff=1600 K, logg=4.0
spectrum = grid["grid"].sel(par1=1600, par2=4.0, method="nearest")

plt.figure(figsize=(10, 4))
plt.plot(grid.wavelength, spectrum, linewidth=0.8)
plt.xlabel("Wavelength (µm)")
plt.ylabel("Flux (model units)")
plt.title("BT-Settl: Teff = 1600 K, log g = 4.0")
plt.tight_layout()
plt.show()
```

## Checking resolution coverage

Before running `analysis.adapt()`, it is worth verifying that the grid's native
spectral resolution is higher than your observation's resolution at the relevant
wavelengths — otherwise the adaptation step cannot degrade the resolution correctly.

```python
import xarray as xr
import matplotlib.pyplot as plt

grid = xr.open_dataset("atm_grids/BT-Settl.nc")

# Native resolution of the grid across wavelength (stored as an attribute, not a data variable)
plt.figure(figsize=(10, 3))
plt.plot(grid.wavelength, grid.attrs["res"])
plt.xlabel("Wavelength (µm)")
plt.ylabel("Spectral resolution λ/Δλ")
plt.title("BT-Settl native resolution")
plt.tight_layout()
plt.show()
```
