# Analytical vs Physical Scaling

When ForMoSA evaluates the likelihood, it compares a transformed model spectrum
to the observed flux. Part of that transformation is a **scaling** step that
brings the model to the same flux level as the data. ForMoSA offers two main
approaches.


## Analytical scaling:

Analytical scaling multiplies the model by a factor `s` evaluated during the nested sampling:

```{math}
F_\text{obs}(\lambda) = F_\text{mod}(\lambda) \times s
```

This factor is computed by marginalizing the log-likelihood with respect to `s`. Mathematically,
this corresponds to:

```{math}
\frac{\partial \ln\mathcal{L}}{\partial s} = 0
```

For an uncorrelated log-likelihood, this corresponds to **Equ. 2** of [Cushing et al. 2008](https://iopscience.iop.org/article/10.1086/526489):

```{math}
s = \frac{ \sum_{i=1}^{N} f_i^\text{obs} f_i^\text{mod} / \sigma_i^2 }{ \sum_{j=1}^{N} (f_j^\text{mod} / \sigma_j)^2 }
```

```python
from ForMoSA.config.global_config import ConfigParameters

config_parameters = ConfigParameters(
    par1  = ["uniform", "800",  "2000"],
    par2  = ["uniform", "3.0",  "5.5"],
    # r and d are NOT set
)
```

**Use analytical scaling when:**
- The absolute flux calibration of your data is uncertain.
- You are fitting contrast spectra (e.g. from integral-field unit observations)
  where the flux level is not physically meaningful.
- You want a quick exploratory fit without committing to a radius prior.
- You are fiting continuum-substracted data.


## Physical scaling: `r` + `d`

Physical scaling applies the inverse-square law:

```{math}
F_\text{obs}(\lambda) = F_\text{mod}(\lambda) \times \left(\frac{r}{d}\right)^2
```

where `r` is the companion radius in Jupiter radii and `d` is the distance in
parsecs. This scaling is physically motivated and lets you retrieve the radius
as a free parameter.

```python
from ForMoSA.config.global_config import ConfigParameters

config_parameters = ConfigParameters(
    par1 = ["uniform", "800",  "2000"],   # Teff
    par2 = ["uniform", "3.0",  "5.5"],    # log g
    r    = ["uniform", "0.5",  "3.0"],    # radius (R_Jup) — free
    d    = ["constant", "27.7"],          # distance fixed to Gaia value (pc)
)
```

**Use physical scaling when:**
- Your flux calibration is reliable (flux-calibrated spectrum or photometry).
- You want to constrain the companion's physical radius.
- You can fix the distance (e.g. from Gaia parallax).


## Side-by-side comparison

| | Analytical | Physical (`r` + `d`) |
|---|---|---|
| Physical meaning | No | Yes |
| Requires distance | No | Yes (can be fixed) |
| Requires flux calibration | No | Yes |
| Adds free parameter? | No | Yes (`r`), or fix `d` |
| Recommended for | Contrast or continuum-substracted spectra | Photometry, flux-calibrated spectra |

## Special case: Parametric scaling `alpha`

When fitting different observations (see next page on MOSAIC), it can be usefull to allow
for individual scalings. In this case, corrective parameter(s) `alpha` can be defined:

```{math}
F_\text{obs}(\lambda) = F_\text{mod}(\lambda) \times \alpha \times \left(\frac{r}{d}\right)^2
```

```python
from ForMoSA.config.global_config import ConfigParameters

config_parameters = ConfigParameters(
    par1    = ["uniform", "800",  "2000"],   # Teff
    par2    = ["uniform", "3.0",  "5.5"],    # log g
    r       = ["uniform", "0.5",  "3.0"],    # radius (R_Jup) — free
    d       = ["constant", "27.7"],          # distance fixed to Gaia value (pc)
    alpha_0 = ["gaussian", "1", "0.1"],      # scaling parameter for the first obs
    alpha_1 = ["gaussian", "1", "0.1"],      # scaling parameter for the second obs
)
```

**Use physical scaling with `alpha` when:**
- You have noticeable flux calibration issues between different datasets
- You want to explore variability
