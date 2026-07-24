# Performances and accuracy

The computational cost of ForMoSA is primarily driven by the number of forward model evaluations 
required for the nested sampling algorithm to converge. 
While absolute execution time (inversion time) depends on the specific hardware, 
this page provides a comparative analysis to guide users in defining the configuration file and
includes exemples of the expected performance and accuracy of the code.

## Exemple setup

For this exercice, we generated an ensemble of K-band ($1.9-2.4\,\mu\mathrm{m}$) spectra at various
spectral resolution ($\text{R}_{\lambda}$ between 2 and 10,000) and signal-to-noise ratio (S/N between 1 and 100)
using the new Exo-REM k26 model grid ([Radcliffe et al. 2026](https://ui.adsabs.harvard.edu/abs/2026arXiv260529070R/abstract)).

## Performances

```{figure} ../_static/inversion_time_formosa_comp.png
:width: 100%
:align: center

Performance comparison between the nested sampling algorithms `PyMultiNest` (squares) and `Nestle` (crosses). Different colors show the number of free parameters used (from 1 to 5). From left to right: Inversion time as a function of spectral resolution (R$_{\lambda}$), signal-to-noise (S/N), and number of live points. The default setup is R$_{\lambda}$ = 368, S/N = 22, and 215 live points. This is intended to inform the user of the order of magnitude in time they should expect for their fit to converge.
```

To optimize the integration time of a ForMoSA inversion, we advise users to:


- Start low and run initial tests with a reduced number of live points (<100) to verify the model setup.  
- Use a spectral resolution that matches the physical information content of the data; over-sampling the spectrum increases the inversion time without improving accuracy.  
- Select the right algorithm for the inversion. For high-dimensional fits (>5 parameters), `PyMultiNest` is the recommended default.


```{note}
Still too slow? See [Tutorial 5](../tutorials/cluster/tutorial_cluster.md)
for instructions on setting up a parallelized inversion.
```

## Accuracy

The next figure illustrates the retrieval accuracy that users can expect for the five parameters of the Exo-REM k26 grid (T$_{eff}$, log(g), [M/H], C/O, and $f_{sed}$) as a function of spectral resolution, signal-to-noise ratio (S/N), and number of live points. 


The synthetic spectra simulate a K-band observation ($1.9-2.4\,\mu\mathrm{m}$), which significantly limits the coverage of the spectral energy distribution (SED). At low resolution and low S/N, the impact is most pronounced for parameters sensitive to the SED shape (T$_{eff}$, log(g), and $f_{sed}$). Overall, above ~30, the number of live points has minimal effect on the retrieval accuracy.


```{figure} ../_static/accuracy_formosa.png
:width: 100%
:align: center

Accuracy comparison using `PyMultiNest` with varying spectral resolution (R$_{\lambda}$), signal-to-noise (S/N), and number of live points. Each dotted red line represents the expected value and black points the retrieved posteriors for each parameter explored during the nested sampling. The default setup is R$_{\lambda}$ = 368, S/N = 22, and 215 live points. Each plot illustrates the effect of varying one parameter while keeping the others fixed.
```