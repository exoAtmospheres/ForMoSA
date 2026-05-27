# Analytical vs Physical Scaling

When ForMoSA evaluates the likelihood, it compares a transformed model spectrum
to the observed flux. Part of that transformation is a **scaling** step that
brings the model to the same flux level as the data. ForMoSA offers two
approaches.

## Physical scaling: `r` + `d`

Physical scaling applies the inverse-square law:

```{math}
F_\text{obs}(\lambda) = F_\text{model}(\lambda) \times \left(\frac{r}{d}\right)^2
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

## Analytical scaling: `alpha`

Analytical scaling multiplies the model by a constant factor:

```{math}
F_\text{obs}(\lambda) = F_\text{model}(\lambda) \times \alpha
```

This is a pure nuisance parameter: it absorbs any flux-level offset without
making any physical claim about the radius or distance.

```python
from ForMoSA.config.global_config import ConfigParameters

config_parameters = ConfigParameters(
    par1  = ["uniform", "800",  "2000"],
    par2  = ["uniform", "3.0",  "5.5"],
    alpha = ["uniform", "0.0",  "10.0"],  # free scaling factor
    # r and d are NOT set
)
```

**Use analytical scaling when:**
- The absolute flux calibration of your data is uncertain.
- You are fitting contrast spectra (e.g. from integral-field unit observations)
  where the flux level is not physically meaningful.
- You want a quick exploratory fit without committing to a radius prior.

## Side-by-side comparison

| | Physical (`r` + `d`) | Analytical (`alpha`) |
|---|---|---|
| Physical meaning | Yes — retrieves radius | No — nuisance parameter |
| Requires distance | Yes (can be fixed) | No |
| Requires flux calibration | Yes | No |
| Adds free parameter? | Yes (`r`), or fix `d` | Yes (`alpha`) |
| Recommended for | Photometry, flux-calibrated spectra | Contrast spectra, exploratory fits |

## Combining both

You can set both `r`+`d` and `alpha` simultaneously — in that case ForMoSA
applies the physical scaling first and then multiplies by `alpha`. This is
occasionally useful when testing for residual systematic offsets on top of a
physical model, but in most cases you should pick one or the other.
