# Tutorial 7 & 8 Notebook Reformat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reformat two Claude-generated notebooks into properly styled Tutorial 7 and Tutorial 8, matching the conventions of `tutorial_advanced_plotting.ipynb` (Tutorial 5).

**Architecture:** Write two new `.ipynb` files using the `Write` tool (full JSON), delete the two old files. No Python code changes — this is pure notebook content work.

**Tech Stack:** Jupyter notebook JSON (nbformat 4), Python 3.12, ForMoSA v2.0, matplotlib, pathlib.

---

### Task 1: Write Tutorial 7 — Advanced Plotting: Custom Figures

**Files:**
- Create: `docs/tutorials/plotting/tutorial_advanced_plotting_custom.ipynb`
- Reference (read for style): `docs/tutorials/plotting/tutorial_advanced_plotting.ipynb`
- Source content: `docs/tutorials/plotting/ForMoSA_Advanced_Plotting_Tutorial.ipynb`

- [ ] **Step 1: Write the complete notebook JSON**

Write to `docs/tutorials/plotting/tutorial_advanced_plotting_custom.ipynb`. Full structure below — every cell is included.

The notebook has these cells in order:

**Cell 0 — markdown: Title/intro**
```
# Tutorial 7 — Advanced Plotting: Custom Figures

**What you'll learn:**
- How to configure matplotlib globally with `rcParams` and locally with `rc_context`
- Three methods for loading ForMoSA nested-sampling results
- How to generate each plot type individually (best-fit, corner, radar, chains)
- The three config layers: `PLOTS_CONFIG`, per-observation `plot_config`, and `MAIN_PLOT`
- How to post-process matplotlib `Figure` / `Axes` objects for fine control
- How to save publication-quality figures (PDF/PNG)

**No fitting required.** This tutorial loads pre-computed results from the `results/`
folder committed alongside this notebook.

**Estimated runtime:** < 1 minute (no nested sampling).

**Prerequisites:** ForMoSA v2.0 installed. Familiarity with Tutorial 5 is helpful but not required.
```

**Cell 1 — markdown: Section 0**
```
## Section 0: Setup
```

**Cell 2 — code: ForMoSA version check**
```python
import sys
try:
    import ForMoSA
    print(f"ForMoSA {ForMoSA.__version__} — OK")
except ImportError:
    raise ImportError("pip install ForMoSA && conda install dask netCDF4 bottleneck")
print(f"Python {sys.version.split()[0]}")
```

**Cell 3 — markdown: rcParams intro**
```
### Matplotlib global styling with `rcParams`

`rcParams` is matplotlib's global settings dictionary. Set it once at the top of the
notebook and every plot you create afterwards inherits those values — fonts, line widths,
tick direction, DPI, and more. This is cleaner than repeating styling in every `plt.X` call.

For one-off overrides (a single figure in a different style), use `plt.rc_context(...)` —
it applies your changes only inside the `with` block and reverts them on exit.
```

**Cell 4 — code: rcParams**
```python
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size':        14,
    'font.family':      'serif',
    'font.serif':       ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',         # Computer Modern math (LaTeX-like, no LaTeX install)
    'axes.linewidth':   1.2,
    'xtick.labelsize':  12,
    'ytick.labelsize':  12,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    'legend.fontsize':  11,
    'legend.frameon':   False,
    'figure.dpi':       100,
    'savefig.dpi':      300,
})
print("rcParams set — all subsequent plots will inherit these defaults.")
```

**Cell 5 — markdown: rc_context**
```
### Scoped overrides with `rc_context`

If you want different fonts or styles for just one figure, wrap it in a context manager:

```python
with plt.rc_context({'font.size': 18, 'lines.linewidth': 2.0}):
    fig = plots.plot_corner()   # uses the overrides
# back to global defaults outside the block
```

### LaTeX rendering

Two paths:
- `mathtext.fontset='cm'` (set above): renders `$...$` math in Computer Modern.
  No LaTeX install needed. Recommended default.
- `plt.rcParams['text.usetex'] = True`: delegates to your system's LaTeX binary.
  Slower and requires a working install, but lets you use arbitrary LaTeX packages
  (e.g. `\textsc`, custom fonts) anywhere in a label string.
```

**Cell 6 — markdown: Section 1**
```
## Section 1: Loading results

ForMoSA stores fit output in two parallel files inside your `result_path`:

- `NS_results/results_<algo>.json` — the `NSResults` object (samples, weights,
  log-likelihoods, log-evidence, etc.)
- `NS_params/NS_params.json` — the nested-sampling configuration used for the run

Three loading patterns are shown below. Choose the one that matches what you have
and what you want to plot.
```

**Cell 7 — markdown: Method A**
```
### Method A — `NSResults` from JSON (fastest; corner, chains, radar only)

This is the quickest path and sufficient for every plot except the best-fit spectrum
(which needs the adapted model grid and the observations on disk too — see Method C).
Use this when you just want to explore or customise your posterior plots.
```

**Cell 8 — code: Method A load**
```python
import json
import logging
from pathlib import Path

from ForMoSA.nested_sampling.results  import NSResults
from ForMoSA.nested_sampling.plotting import Plotting

# Path to the pre-computed results included with this tutorial.
# If you are using your own fit, replace this with:
#   Path("/absolute/path/to/your/result_path") / "NS_results" / "results_pymultinest.json"
RESULTS_JSON = Path(".").resolve() / "results" / "NS_results" / "results_pymultinest.json"

if not RESULTS_JSON.exists():
    raise FileNotFoundError(
        f"Results file not found: {RESULTS_JSON}\n"
        "Make sure you are running this notebook from the plotting/ directory,\n"
        "or update RESULTS_JSON to point to your own result_path."
    )

with open(RESULTS_JSON) as f:
    data = json.load(f)

results = NSResults.from_dict(data)

logger = logging.getLogger("tutorial")
plots  = Plotting(results, logger=logger)

print(f"Results loaded: {results.samples.shape[0]} total samples")
print(f"Free parameters: {results.free_parameters}")
print(f"Burn-in index: {results.burn_in} (samples before this index are discarded)")
```

**Cell 9 — markdown: Method B**
```
### Method B — Directly from raw PyMultiNest output files

Use this if you only have the raw PyMultiNest output directory
(`RAW_.txt`, `RAW_ev.dat`, `RAW_stats.dat`) but not the JSON.
You need to tell ForMoSA which parameters you fitted so it can
label the columns correctly.
```

**Cell 10 — code: Method B load**
```python
# Uncomment and fill in your paths to use this method.

# from ForMoSA.nested_sampling.results import NSResults

# PYMULTINEST_DIR = Path(".").resolve() / "results" / "pymultinest"
# FREE_PARAMETERS = ['Teff', 'logg', 'M_H', 'C_O', 'r', 'd']   # match your fit exactly

# results = NSResults.from_pymultinest(
#     results_path    = str(PYMULTINEST_DIR),
#     free_parameters = FREE_PARAMETERS,
# )
print("Method B: uncomment the block above and fill in your paths to use this method.")
```

**Cell 11 — markdown: Method C**
```
### Method C — Full `Analysis` reload (required for the best-fit plot)

The best-fit spectrum plot needs more than just the posterior samples — it needs the
adapted model grid and the observation spectra that were used during the fit, plus
the best-fit model evaluated at the posterior median. All of this lives in the
`Analysis` object.

Setting `fitted=True` tells ForMoSA to skip the nested sampling run and instead
reconstruct `analysis.ns` and `analysis.ns_analysis` from the files already on disk.
Setting `adapted=True` skips the grid adaptation step too. Both must be True when
loading a completed fit.
```

**Cell 12 — code: Method C load**
```python
# Uncomment and fill in your config to use this method.
# This requires the adapted grid files to be present in adapt_store_path.

# from ForMoSA.analysis import Analysis
# from ForMoSA.config.global_config import ConfigPath, ConfigAdapt, ConfigInversion, ConfigParameters

# analysis = Analysis(
#     ConfigPath(
#         observation_path  = ["path/to/obs.fits"],
#         adapt_store_path  = "path/to/adapted_grid/",
#         result_path       = "path/to/results/",
#         model_path        = "path/to/model_grid.nc",
#     ),
#     adapted = True,   # adaptation already done on disk — don't redo it
#     fitted  = True,   # NS already run — load results from result_path
# )
print("Method C: uncomment the block above and fill in your config paths to use this method.")
```

**Cell 13 — markdown: Section 2**
```
## Section 2: Best-fit spectrum plot

The best-fit plot requires a full `Analysis` object loaded via Method C above —
it needs the adapted model grid and best-fit spectra, not just the posterior samples.

The subsections below show all the customisation options. Run them after loading
`analysis` and building `ns_analysis` from Method C.
```

**Cell 14 — markdown: 2.1 individual call**
```
### Section 2.1: Calling only the best-fit plot

`analysis.plot(results)` runs all four plot types at once. To generate just the
best-fit spectrum, bypass it and call `plots.plot_fit(...)` directly.
`NSAnalysis` computes the best-fit spectrum (the weighted posterior median model).
```

**Cell 15 — code: best-fit call**
```python
# Requires Method C above. Uncomment after loading analysis.

# from ForMoSA.nested_sampling.plotting    import Plotting
# from ForMoSA.nested_sampling.ns_analysis import NSAnalysis

# ns_analysis = NSAnalysis(analysis.ns, logger=analysis.logger)
# plots_bf    = Plotting(analysis.ns.results, analysis.logger)

# fig, ax, ax_filt, axr, axr2 = plots_bf.plot_fit(
#     analysis.ns.restricted_observations,
#     ns_analysis.best_fit,
# )

# Returned objects:
#   fig      — the matplotlib Figure
#   ax       — main spectrum panel (data + model)
#   ax_filt  — photometric filter transmission panel (None if no photometry)
#   axr      — residuals panel at the bottom
#   axr2     — residual histogram panel to the right of axr
print("Section 2.1: uncomment after loading analysis via Method C.")
```

**Cell 16 — markdown: 2.2 config**
```
### Section 2.2: Customising the best-fit line

`PLOTS_CONFIG.BestFitPlot` is a dataclass that controls the appearance of the model
line and residuals. Set it **before** calling `plot_fit` — it won't apply retroactively.

Available fields: `color_fit`, `color_residuals`, `linewidth`, `zorder`.
There is no `alpha` field in the dataclass — for transparency, edit post-hoc (Section 2.4).
```

**Cell 17 — code: best-fit config**
```python
from ForMoSA.core.config import PLOTS_CONFIG

PLOTS_CONFIG.BestFitPlot.set_best_fit_plot_config(
    color_fit       = '#E8844A',   # warm orange for the model line
    color_residuals = '#2C2C2C',   # near-black for residuals
    linewidth       = 1.5,
    zorder          = 200,         # draw model on top of data points
)
print("BestFitPlot config set. Call plots_bf.plot_fit(...) to apply.")
```

**Cell 18 — markdown: 2.3 per-obs colour**
```
### Section 2.3: Customising each observation's colour

In a multi-instrument (MOSAIC) fit, each observation is plotted on the same axes.
Every `Observation` object owns its own `plot_config`. Iterate over
`analysis.ns.restricted_observations` and call `set_plot_config(...)` on each.

Fields available from `ObsPlotConfig`:
`color`, `edgecolor`, `marker`, `markersize`, `linewidth`,
`errorbar_fmt`, `errorbar_alpha`, `errorbar_capsize`,
`zorder_data`, `zorder_error`, `label`.
```

**Cell 19 — code: per-obs colour**
```python
# Example: assign distinct colors by observation name.
# Requires Method C. Uncomment after loading analysis.

# custom_colors = {
#     'SINFONI_K':   '#4C72B0',
#     'HiRISE':      '#55A868',
#     'photometry':  '#C44E52',
# }

# for obs in analysis.ns.restricted_observations:
#     obs.plot_config.set_plot_config(
#         color          = custom_colors.get(obs.name, obs.plot_config.color),
#         linewidth      = 1.0,
#         errorbar_alpha = 0.5,   # make error bars slightly transparent
#     )
print("Section 2.3: uncomment after loading analysis via Method C.")
```

**Cell 20 — markdown: 2.4 MAIN_PLOT**
```
### Section 2.4: Figure-wide configuration (`MAIN_PLOT`)

`MAIN_PLOT` controls global layout properties that apply to the whole figure,
not to a single plot type. Set it before calling any plot function.
```

**Cell 21 — code: MAIN_PLOT**
```python
from ForMoSA.core.config import MAIN_PLOT

MAIN_PLOT.figsize         = (20, 9)    # wider figure for multi-instrument fits
MAIN_PLOT.legend_fontsize = 14
MAIN_PLOT.minor_ticks     = True
MAIN_PLOT.nb_minor_ticks  = 5         # number of minor ticks between major ticks
print("MAIN_PLOT config set.")
```

**Cell 22 — markdown: 2.5 post-hoc**
```
### Section 2.5: Post-hoc axis tweaks

ForMoSA's config dataclasses don't expose everything — axis-label font sizes,
tick label sizes, and best-fit-line alpha are not in the dataclass. You can set
these directly on the matplotlib `Axes` objects returned by `plot_fit`.

This is standard matplotlib: once you have an `Axes`, you can change anything.
```

**Cell 23 — code: post-hoc tweaks**
```python
# Requires Method C. Uncomment after calling plot_fit.

# for a in [ax, axr]:
#     a.tick_params(labelsize=14)
#     a.xaxis.label.set_size(15)
#     a.yaxis.label.set_size(15)

# # Apply alpha to the best-fit line (not in the config dataclass)
# for line in ax.get_lines():
#     if line.get_label() == 'Best fit':
#         line.set_alpha(0.85)

# # Redraw legend with explicit styling
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, frameon=False, loc='upper right', fontsize=13)
print("Section 2.5: uncomment after calling plot_fit.")
```

**Cell 24 — markdown: 2.6 param labels in legend**
```
### Section 2.6: Parameter values in the legend

`results.median_parameters` gives the weighted posterior median for each free
parameter as a `dict[str, float]`. `_interval(sigma=1)` gives the ±1σ asymmetric
credible interval as `dict[str, (low, high)]`. Use these to build a formatted
legend entry that shows the best-fit values directly in the plot.
```

**Cell 25 — code: param labels**
```python
import numpy as np

medians   = results.median_parameters
intervals = results._interval(sigma=1)

parts = []
for k, med in medians.items():
    lo, hi = intervals[k]
    parts.append(f'{k}=${med:.2f}_{{-{med-lo:.2f}}}^{{+{hi-med:.2f}}}$')

new_label = 'Best fit:  ' + ',  '.join(parts)
print("Legend label that would be applied:")
print(new_label)

# To apply it to the figure (requires Method C):
# for line in ax.get_lines():
#     if line.get_label() == 'Best fit':
#         line.set_label(new_label)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels, frameon=False, loc='upper right', fontsize=11)
```

**Cell 26 — markdown: 2.7 quantile bands**
```
### Section 2.7: Plotting posterior uncertainty bands (1σ, 2σ)

`ns_analysis.best_fit_interval(perc=...)` returns the lower and upper envelope
of the model at the given posterior credible level. Use `fill_between` to shade
the band on the spectrum axis.

> **Note:** `best_fit_interval` may return fluxes in native-model space rather
> than observation space for some setups. If the lengths don't match your
> observation's wavelength grid, use `ns_analysis.native_best_fit` instead,
> or draw samples manually (commented fallback below).
```

**Cell 27 — code: quantile bands**
```python
# Requires Method C and a live ns_analysis object.

# lower_1, higher_1 = ns_analysis.best_fit_interval(perc=0.68)
# lower_2, higher_2 = ns_analysis.best_fit_interval(perc=0.95)

# ax.fill_between(lower_1.wave, lower_1.flux, higher_1.flux,
#                 color='grey', alpha=0.4, zorder=150, label='1$\\sigma$')
# ax.fill_between(lower_2.wave, lower_2.flux, higher_2.flux,
#                 color='grey', alpha=0.2, zorder=140, label='2$\\sigma$')
# ax.legend(*ax.get_legend_handles_labels(), frameon=False, loc='upper right', fontsize=11)

# --- Fallback: manual quantile band by weighted posterior sampling ---
# import numpy as np
# n_draws = 500
# w       = results.weights[results.burn_in:]
# w_norm  = w / w.sum()
# idx     = np.random.choice(len(w_norm), size=n_draws, p=w_norm, replace=True)
# draws   = results.samples[results.burn_in:][idx]
# # For each draw, evaluate your model here and collect fluxes into an array.
# # fluxes = np.array([your_model(params) for params in draws])
# # lo1, hi1 = np.quantile(fluxes, [0.16, 0.84], axis=0)
# # lo2, hi2 = np.quantile(fluxes, [0.025, 0.975], axis=0)
# # ax.fill_between(wave_grid, lo1, hi1, color='grey', alpha=0.4)
# # ax.fill_between(wave_grid, lo2, hi2, color='grey', alpha=0.2)
print("Section 2.7: uncomment after loading analysis via Method C.")
```

**Cell 28 — markdown: Section 3**
```
## Section 3: Corner plot

`Plotting.plot_corner()` is a thin wrapper around the
[`corner`](https://corner.readthedocs.io) library.
The dataclass `PLOTS_CONFIG.CornerPlot` exposes most of `corner.corner`'s arguments.

A corner plot shows every pair of parameters as a 2D contour (off-diagonal panels)
and each parameter's marginal posterior as a 1D histogram (diagonal panels).
It is the standard way to visualise parameter correlations and posterior shapes
in Bayesian fitting.

The `Plotting` object was created from `results` in Section 1 (Method A).
We use it for all remaining plot types.
```

**Cell 29 — markdown: 3.1 colours**
```
### Section 3.1: Colors, contours, and fills

`fill_contours=True` shades the interior of each contour level.
`plot_density=True` adds a 2D density colour map behind the contours.
`smooth` applies a Gaussian filter in pixels — higher values give smoother contours
but can blur real structure; 0.5–1.5 is a typical range.
`levels` sets the 2D credible regions to draw; the values below correspond to
1σ, 2σ, and 3σ for a 2D Gaussian.
```

**Cell 30 — code: corner colours**
```python
from ForMoSA.core.config import PLOTS_CONFIG

PLOTS_CONFIG.CornerPlot.set_corner_plot_config(
    color         = '#2E5C8A',
    fill_contours = True,
    plot_density  = True,
    plot_contours = True,
    smooth        = 1.2,
    bins          = 60,
    levels        = [0.3935, 0.8647, 0.9889],   # 1σ / 2σ / 3σ for a 2D Gaussian
    hist_kwargs    = dict(color='#2E5C8A', histtype='stepfilled',
                          alpha=0.6, edgecolor='#143B66', linewidth=0.8),
    contour_kwargs = dict(colors='#2E5C8A', linewidths=0.8),
)

fig_corner = plots.plot_corner()
plt.show()
```

**Cell 31 — markdown: 3.2 fonts**
```
### Section 3.2: Fonts and label sizes

`title_kwargs` and `label_kwargs` are passed directly to `corner.corner`.
`title_fmt` controls the format string used in the median ± σ titles on the diagonal.
`max_n_ticks` limits how many tick marks appear on each axis (useful for crowded corners).
```

**Cell 32 — code: corner fonts**
```python
PLOTS_CONFIG.CornerPlot.set_corner_plot_config(
    title_kwargs = dict(fontsize=15),
    label_kwargs = dict(fontsize=15),
    title_fmt    = '.2f',
    max_n_ticks  = 4,
)

fig_corner = plots.plot_corner()
plt.show()
```

**Cell 33 — markdown: 3.3 custom labels**
```
### Section 3.3: Replacing parameter labels with custom LaTeX names

`corner` takes its labels from `results.free_parameters` (plain strings like `Teff`).
To use LaTeX labels (e.g. $T_{\rm eff}$), build the figure first, then overwrite
the axis labels on the returned `Axes` grid.

The corner figure has `n × n` axes in a square grid. The bottom row holds x-labels;
the leftmost column holds y-labels (skip the `[0, 0]` panel — it's a 1D histogram
with no y-label). Adjust `custom_labels` to match **your** `free_parameters` in order.
```

**Cell 34 — code: custom labels**
```python
import numpy as np

n = len(results.free_parameters)
print(f"Free parameters ({n}): {results.free_parameters}")

# >>> Replace these with your own LaTeX labels in the same order as free_parameters
custom_labels = [
    r'$T_{\mathrm{eff}}$ (K)',
    r'$\log g$',
    r'[M/H]',
    r'C/O',
    r'$R$ ($R_\mathrm{J}$)',
    r'$d$ (pc)',
][:n]   # truncate/extend to match n

fig_corner = plots.plot_corner()
axes = np.array(fig_corner.axes).reshape((n, n))

# Bottom row: x-labels
for i in range(n):
    axes[-1, i].set_xlabel(custom_labels[i], fontsize=15)

# Leftmost column: y-labels (skip the [0,0] diagonal histogram)
for i in range(1, n):
    axes[i, 0].set_ylabel(custom_labels[i], fontsize=15)

plt.show()
```

**Cell 35 — markdown: 3.4 quantile lines**
```
### Section 3.4: Quantile lines on the diagonal

The dashed vertical lines on the diagonal histograms mark specific posterior
quantiles. The default `(0.16, 0.5, 0.84)` corresponds to the median and ±1σ.
Change them to `(0.025, 0.5, 0.975)` for a ±2σ (95%) interval.
`show_titles=True` prints the median and interval above each diagonal panel.
```

**Cell 36 — code: quantile lines**
```python
PLOTS_CONFIG.CornerPlot.set_corner_plot_config(
    quantiles   = (0.025, 0.5, 0.975),   # 95% credible interval
    show_titles = True,
)

fig_corner = plots.plot_corner()
plt.show()
```

**Cell 37 — markdown: 3.5 tick density**
```
### Section 3.5: Tick density and label size

`max_n_ticks` in the config sets an approximate limit, but for precise control
you can set tick locators directly on each axis after the figure is built.
```

**Cell 38 — code: tick density**
```python
from matplotlib.ticker import MaxNLocator

fig_corner = plots.plot_corner()
for a in fig_corner.axes:
    a.tick_params(labelsize=11)
    a.xaxis.set_major_locator(MaxNLocator(4))
    a.yaxis.set_major_locator(MaxNLocator(4))

plt.show()
```

**Cell 39 — markdown: Section 4**
```
## Section 4: Radar plot

The radar plot shows every free parameter on a separate radial axis, all normalised
to [0, 1] relative to their sample range. The filled polygon marks the
median ± your chosen quantile interval. It gives a compact at-a-glance view of
which parameters are tightly constrained versus which still span most of their prior.

> **Known source typo:** The `RadarPlotConfig` dataclass has a misspelling in the
> ForMoSA source: the field is `fontisze_ticks` (not `fontsize_ticks`).
> Use the misspelled name or the call will silently ignore your value.
```

**Cell 40 — code: radar config**
```python
PLOTS_CONFIG.RadarPlot.set_radar_plot_config(
    color_radar        = '#7B3F8F',
    color_uncertainty  = '#B58CC2',
    color_quantiles    = '#7B3F8F',
    alpha_fill         = 0.4,
    linewidth          = 2.0,
    fontsize_names     = 13,
    fontisze_ticks     = 11,         # note: typo is in the ForMoSA source
    color_ticks        = '#24292E',
    quantiles          = (0.16, 0.84),
    size_quantiles     = 80,
    lw_quantiles       = 2.0,
)

fig_radar, ax_radar = plots.plot_radars()
plt.show()
```

**Cell 41 — markdown: 4.1 radar labels**
```
### Section 4.1: Replacing axis labels and adding annotations

The radar axes are standard matplotlib polar axes. Replace tick labels with
LaTeX strings using `set_xticklabels`, then add a note with `ax_radar.text(...)`.
```

**Cell 42 — code: radar labels and annotation**
```python
fig_radar, ax_radar = plots.plot_radars()

ax_radar.set_xticklabels(custom_labels, fontsize=13)

ax_radar.text(
    0.5, -0.12,
    'Shaded band: 16–84% posterior quantiles',
    transform=ax_radar.transAxes,
    ha='center', va='top',
    fontsize=11, color='grey',
)

plt.show()
```

**Cell 43 — markdown: Section 5**
```
## Section 5: Chains plot

The chains plot shows the evolution of each sampled parameter over the course
of the nested-sampling run — one panel per parameter, with samples in order.

Three overlays help you read convergence:
- **Burn-in marker** (vertical dashed line): samples to the left are discarded;
  only those to the right enter the posterior.
- **Importance weights** (right y-axis, grey trace): shows which part of the
  chain carries most of the posterior weight. A spike near the end is normal for NS.
- **Best-value line** (horizontal line): marks the weighted posterior median.
```

**Cell 44 — code: chains config**
```python
PLOTS_CONFIG.ChainsPlot.set_chains_plot_config(
    color_chains         = '#5E3C99',
    alpha_chains         = 0.6,

    color_plot_burn_in   = '#E66101',
    fontsize_burn_in     = 13,
    linestyle_burn_in    = '--',

    show_weights         = True,
    color_plot_weights   = '#1F1F1F',
    alpha_weights        = 0.35,
    fontsize_weights     = 13,

    plot_best_value      = True,
    color_best_value     = 'black',
    linestyle_best_value = '-.',
)

fig_chains, axs = plots.plot_chains()
plt.show()
```

**Cell 45 — markdown: 5.1 chains labels**
```
### Section 5.1: Replacing y-labels

`axs` is a list of `Axes` objects, one per free parameter, in the same order as
`results.free_parameters`. Zip them with your custom labels to replace them.
```

**Cell 46 — code: chains labels**
```python
fig_chains, axs = plots.plot_chains()

for a, name in zip(axs, custom_labels):
    a.set_ylabel(name, fontsize=13)
    a.tick_params(labelsize=11)

fig_chains.tight_layout()
plt.show()
```

**Cell 47 — markdown: Section 6**
```
## Section 6: Saving figures

All four plot functions return a `Figure` object. Call `savefig` directly on it
to control format, DPI, and bounding box.

**PDF vs PNG:**
- PDF (vector): preferred for journal submission. Lines and text scale to any size.
  DPI only affects raster fall-backs (e.g. very dense scatter plots embedded in the PDF).
- PNG (raster): use for slides or talks where a vector renderer isn't guaranteed.
  Use `dpi=300` minimum for print; `dpi=200` is fine for screen.

**Bounding box:** `bbox_inches='tight'` trims whitespace around the figure.
Almost always what you want for publication figures.
```

**Cell 48 — code: saving**
```python
from pathlib import Path

out_dir = Path(".").resolve()

fig_corner = plots.plot_corner()
fig_corner.savefig(out_dir / "corner_custom.pdf", dpi=300, bbox_inches="tight")
fig_corner.savefig(out_dir / "corner_custom.png", dpi=200, bbox_inches="tight")
print(f"Saved: {out_dir / 'corner_custom.pdf'}")
print(f"Saved: {out_dir / 'corner_custom.png'}")

fig_radar, _ = plots.plot_radars()
fig_radar.savefig(out_dir / "radar_custom.pdf", dpi=300, bbox_inches="tight")

fig_chains, _ = plots.plot_chains()
fig_chains.savefig(out_dir / "chains_custom.pdf", dpi=300, bbox_inches="tight")
```

**Cell 49 — markdown: Section 7**
```
## Section 7: Next steps

- **Tutorial 8 — Statistical Tests and Model Selection:** Compute reduced χ²,
  log-evidence, effective sample size, AIC/BIC, and Bayes factors from your
  ForMoSA results. Compare models and report them correctly in a paper.
- **API docs:** `docs/api/` for `Plotting`, `NSResults`, `PlotsConfig`,
  `MAIN_PLOT`, and all config dataclasses with their full parameter lists.
```

- [ ] **Step 2: Verify the file was written**

Run:
```bash
python -c "import json; nb=json.load(open('docs/tutorials/plotting/tutorial_advanced_plotting_custom.ipynb')); print(len(nb['cells']), 'cells')"
```
Expected: prints `50 cells` (or close, depending on exact count)

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/plotting/tutorial_advanced_plotting_custom.ipynb
git commit -m "docs: add Tutorial 7 — Advanced Plotting: Custom Figures"
```

---

### Task 2: Write Tutorial 8 — Statistical Tests and Model Selection

**Files:**
- Create: `docs/tutorials/plotting/tutorial_statistical_tests.ipynb`
- Reference: `docs/tutorials/plotting/tutorial_advanced_plotting.ipynb`
- Source content: `docs/tutorials/plotting/ForMoSA_Statistical_Tests_Tutorial.ipynb`

- [ ] **Step 1: Write the complete notebook JSON**

Write to `docs/tutorials/plotting/tutorial_statistical_tests.ipynb`. All cells:

**Cell 0 — markdown: Title/intro**
```
# Tutorial 8 — Statistical Tests and Model Selection

**What you'll learn:**
- How to check whether your nested-sampling run has converged
- How to compute the effective sample size (ESS) from importance weights
- How to calculate goodness-of-fit (reduced χ²) per observation
- How to compute AIC and BIC information criteria
- How to compare two ForMoSA fits using Bayes factors (log Bayes factor from logZ)
- Practical worked examples: fixed vs free parameter, and different atmospheric grids

**No fitting required.** This tutorial loads pre-computed results from the `results/`
folder committed alongside this notebook.

**Estimated runtime:** < 1 minute (no nested sampling).

**Prerequisites:** ForMoSA v2.0 installed. Tutorial 5 or 7 recommended for context,
but not required.
```

**Cell 1 — markdown: autocorrelation note**
```
### A note on autocorrelation and nested sampling

Many MCMC tutorials warn you to check chain autocorrelation length as a convergence
diagnostic. **Do not do this with nested sampling.** NS samples are not a Markov
chain in the relevant sense — they are nested-shell draws, each weighted by an
importance weight that reflects how much posterior probability it carries. Computing
autocorrelation on NS samples is conceptually wrong, and the number it produces has
no meaningful interpretation for NS convergence.

Use NS-appropriate diagnostics instead: the `logZ` uncertainty, the effective sample
size (ESS), and optionally repeating the run with a different random seed.
```

**Cell 2 — markdown: Section 0**
```
## Section 0: Setup
```

**Cell 3 — code: setup**
```python
import sys
try:
    import ForMoSA
    print(f"ForMoSA {ForMoSA.__version__} — OK")
except ImportError:
    raise ImportError("pip install ForMoSA && conda install dask netCDF4 bottleneck")

import numpy as np
print(f"Python {sys.version.split()[0]}, numpy {np.__version__}")
```

**Cell 4 — markdown: Section 1**
```
## Section 1: Loading results

We load the pre-computed results included with this tutorial.
The same `NSResults` object is used throughout all sections below.
If you want to use your own fit, replace `RESULTS_JSON` with the path
to your `result_path/NS_results/results_pymultinest.json`.

> **nestle users:** replace `results_pymultinest.json` with `results_nestle.json`.
```

**Cell 5 — code: load results**
```python
import json
from pathlib import Path
from ForMoSA.nested_sampling.results import NSResults

RESULTS_JSON = Path(".").resolve() / "results" / "NS_results" / "results_pymultinest.json"

if not RESULTS_JSON.exists():
    raise FileNotFoundError(
        f"Results file not found: {RESULTS_JSON}\n"
        "Make sure you are running this notebook from the plotting/ directory,\n"
        "or update RESULTS_JSON to point to your own result_path."
    )

with open(RESULTS_JSON) as f:
    data = json.load(f)

results = NSResults.from_dict(data)

print(f"Results loaded: {results.samples.shape[0]} total samples")
print(f"Free parameters: {results.free_parameters}")
print(f"Burn-in index: {results.burn_in}")
print(f"Post-burn-in samples: {results.samples.shape[0] - results.burn_in}")
```

**Cell 6 — markdown: Section 2**
```
## Section 2: Nested-sampling convergence diagnostics

Before trusting your posterior, check that the sampler has converged.
Two numbers tell you most of what you need: the **log-evidence uncertainty**
and the **effective sample size**.
```

**Cell 7 — markdown: 2.1 logZ**
```
### Section 2.1: Log-evidence and its uncertainty

`results.logz` is a two-element list `[logZ, logZ_err]` produced directly by
the NS algorithm (PyMultiNest or UltraNest). `logZ` is the log of the Bayesian
evidence — the integral of the likelihood over the entire prior. `logZ_err` is
the numerical integration uncertainty.

**Rule of thumb:** `logZ_err ≲ 0.5` is typical for a well-converged PyMultiNest
run with default `npoints=500`. If it is substantially larger, re-run with more
live points (try `npoints=1000` or `npoints=2000`).
```

**Cell 8 — code: logZ**
```python
logZ, logZ_err = results.logz
print(f"log Z = {logZ:.3f} ± {logZ_err:.3f}")

if logZ_err > 0.5:
    print("WARNING: logZ_err > 0.5 — consider re-running with more live points.")
else:
    print("logZ_err looks good (< 0.5).")
```

**Cell 9 — markdown: 2.2 ESS**
```
### Section 2.2: Effective sample size (ESS)

The posterior is a *weighted* sample — each draw has an importance weight that
reflects how much probability it carries. A few high-weight samples can dominate
the posterior even if you technically have tens of thousands of raw draws.

The **effective sample size** corrects for this. It answers the question:
"How many *independent, equally-weighted* samples would give the same statistical
resolution as my weighted set?" The formula is:

$$\mathrm{ESS} = \frac{1}{\sum_i \tilde{w}_i^2}$$

where $\tilde{w}_i = w_i / \sum_j w_j$ are the normalised weights.

**Interpretation:**
- ESS > 1000 → posterior quantiles are well-resolved at the 1σ level
- ESS 100–1000 → medians are reliable; 2σ tails are noisy
- ESS < 100 → re-run with more live points; your posterior is dominated by a few samples
```

**Cell 10 — code: ESS**
```python
w      = results.weights[results.burn_in:]
w_norm = w / w.sum()

ess        = 1.0 / np.sum(w_norm**2)
efficiency = ess / len(w)

print(f"Raw post-burn-in samples: {len(w)}")
print(f"ESS:                      {ess:.0f}")
print(f"Sampling efficiency:      {efficiency:.2%}")

if ess < 100:
    print("WARNING: ESS < 100. Re-run with more live points.")
elif ess < 1000:
    print("NOTE: ESS in 100–1000 range. Medians are reliable; tail quantiles may be noisy.")
else:
    print("ESS looks good (> 1000).")
```

**Cell 11 — markdown: 2.3 re-run**
```
### Section 2.3: The most honest convergence check — re-run with a different seed

The most rigorous convergence test is to run the same fit twice with different
random-number-generator seeds and compare the results. If `logZ` values agree
within their stated uncertainties, and the posterior medians agree within ~ 0.5σ,
the run has converged. There is no programmatic shortcut — just run the fit twice
and compare.

To change the PyMultiNest seed, set `seed` in your `ConfigNestle` or
`ConfigPyMultiNest` dataclass (check the API docs for the exact field name).
```

**Cell 12 — markdown: Section 3**
```
## Section 3: Goodness of fit — reduced χ²

Reduced χ² measures how well the best-fit model explains the data, accounting for
the number of free parameters.

$$\chi^2_\nu = \frac{\chi^2}{N - k}$$

where $N$ is the total number of data points across all observations, $k$ is the
number of free parameters, and $\chi^2 = \sum_i \left(\frac{d_i - m_i}{\sigma_i}\right)^2$.

**Interpretation:**
- $\chi^2_\nu \approx 1$ — fit is consistent with the stated error bars
- $\chi^2_\nu \gg 1$ — model underfits (bad model or underestimated errors)
- $\chi^2_\nu \ll 1$ — errors are overestimated, or model has too many parameters

**Caveat:** χ² is only meaningful for likelihoods of the form `chi2` or
`chi2_noisescaling`. For CCF-based likelihoods (e.g. HiRISE in MOSAIC mode),
residuals do not reduce to standard χ² — skip or restrict this section to your
χ²-likelihood observations only.

The code below requires a full `Analysis` object with `fitted=True`.
See Tutorial 7, Section 1, Method C for how to load one.
```

**Cell 13 — code: chi2_red from logL (standalone)**
```python
# Standalone version: compute chi2_red from the weighted posterior log-likelihood.
# This works with just NSResults (no Analysis object needed).
#
# This is an approximation: it uses the *weighted average* log-likelihood from the
# posterior, not the true maximum-likelihood point. It is fast and useful for a
# quick sanity check, but the per-observation breakdown below (which needs Analysis)
# is more informative.

n_data = 450    # approximate number of data points — replace with your actual value
n_free = len(results.free_parameters)

logl_post = results.logl[results.burn_in:]
w_post    = results.weights[results.burn_in:]
best_logL = np.average(logl_post, weights=w_post)

chi2      = -2 * best_logL
chi2_red  = chi2 / (n_data - n_free)

print(f"n_data   = {n_data}  (update this to your actual number of spectral bins)")
print(f"n_free   = {n_free}")
print(f"-2 log L = {chi2:.2f}")
print(f"χ²_red   = {chi2_red:.3f}")
```

**Cell 14 — code: chi2_red per-observation (needs Analysis)**
```python
# Full per-observation breakdown — requires Method C from Tutorial 7.
# Uncomment after loading analysis and ns_analysis.

# chi2_total = 0.0
# ndata      = 0

# for i, obs in enumerate(analysis.ns.restricted_observations):
#     model_flux = ns_analysis.best_fit[i].flux
#     residuals  = (obs.flux - model_flux) / obs.err
#     chi2_i     = np.sum(residuals**2)
#     n_i        = len(obs.flux)

#     chi2_total += chi2_i
#     ndata      += n_i

#     print(f"{obs.name:20s}  chi2={chi2_i:10.2f}  N={n_i:5d}  chi2/N={chi2_i/n_i:.3f}")

# n_free   = len(results.free_parameters)
# dof      = ndata - n_free
# chi2_red = chi2_total / dof
# print()
# print(f"Total chi2 = {chi2_total:.2f}")
# print(f"DOF        = {dof}")
# print(f"chi2_red   = {chi2_red:.3f}")
print("Per-observation chi2: uncomment after loading analysis via Method C (Tutorial 7).")
```

**Cell 15 — markdown: Section 4**
```
## Section 4: Information criteria — AIC and BIC

Information criteria penalise model complexity to prevent overfitting.
Lower values are better. Use them to compare two fits to the **same dataset**
with **different numbers of free parameters**.

$$\mathrm{AIC} = 2k - 2\ln\hat{L}$$
$$\mathrm{BIC} = k\ln n - 2\ln\hat{L}$$

where $k$ is the number of free parameters, $n$ is the number of data points,
and $\hat{L}$ is the maximum likelihood.

**AIC vs BIC:**
- AIC penalises complexity weakly. It tends to prefer slightly richer models.
- BIC penalises complexity proportionally to $\ln n$. For large datasets
  ($n \gtrsim 8$), BIC penalises more harshly than AIC.
- In practice, report both and note if they agree.

**ΔAIC interpretation** (Burnham & Anderson 2002):

| ΔAIC   | Support for the higher-AIC model     |
|--------|--------------------------------------|
| 0 – 2  | Substantial (models nearly equivalent) |
| 4 – 7  | Considerably less                    |
| > 10   | Essentially none                     |
```

**Cell 16 — code: AIC/BIC**
```python
# Best log-likelihood: use the maximum over the post-burn-in chain.
best_logL = results.logl[results.burn_in:].max()

k = len(results.free_parameters)
n = n_data   # from Section 3 above — update to your actual value

AIC = 2*k - 2*best_logL
BIC = k * np.log(n) - 2*best_logL

print(f"best log L = {best_logL:.3f}")
print(f"k = {k},  n = {n}")
print(f"AIC = {AIC:.2f}")
print(f"BIC = {BIC:.2f}")
print()
print("To compare two fits, compute ΔAIC = AIC_A - AIC_B (positive means B is preferred).")
```

**Cell 17 — markdown: Section 5**
```
## Section 5: Bayesian model comparison — Bayes factors

The Bayes factor compares two models by their *marginal likelihoods*
(= the Bayesian evidence, Z). Nested sampling gives you logZ directly,
so the log Bayes factor falls straight out:

$$\ln B_{12} = \ln Z_1 - \ln Z_2$$

A positive value means model 1 is favoured. The uncertainty propagates in quadrature:

$$\sigma_{\ln B} = \sqrt{\sigma_{\ln Z_1}^2 + \sigma_{\ln Z_2}^2}$$

**Jeffreys / Kass-Raftery interpretation scale:**

| $\ln B_{12}$ | $2\ln B_{12}$ | Evidence for model 1          |
|-------------|--------------|-------------------------------|
| 0 – 1       | 0 – 2        | Not worth more than a mention |
| 1 – 3       | 2 – 6        | Positive                      |
| 3 – 5       | 6 – 10       | Strong                        |
| > 5         | > 10         | Very strong                   |

Sources (cite whichever convention you adopt):
- Kass & Raftery (1995), *JASA* 90, 773. DOI: 10.1080/01621459.1995.10476572
- Jeffreys (1961), *Theory of Probability*, Oxford Univ. Press, 3rd ed.
- Trotta (2008), *Contemp. Phys.* 49, 71 (astronomy focus). DOI: 10.1080/00107510802066753

> The exact thresholds vary slightly across sources. Pick one convention and cite it consistently.
```

**Cell 18 — code: Bayes factor template**
```python
# Template for computing log B from two completed fits.
# Replace the paths with your own result_path locations.

# results_json_1 = Path(".").resolve() / "path/to/model_1/NS_results/results_pymultinest.json"
# results_json_2 = Path(".").resolve() / "path/to/model_2/NS_results/results_pymultinest.json"

# results_1 = NSResults.from_dict(json.load(open(results_json_1)))
# results_2 = NSResults.from_dict(json.load(open(results_json_2)))

# logB     = results_1.logz[0] - results_2.logz[0]
# logB_err = np.hypot(results_1.logz[1], results_2.logz[1])
# print(f"ln B_12 = {logB:.2f} ± {logB_err:.2f}")
# if logB > 5:
#     print("Very strong evidence for model 1.")
# elif logB > 3:
#     print("Strong evidence for model 1.")
# elif logB > 1:
#     print("Positive evidence for model 1.")
# elif logB > -1:
#     print("Inconclusive.")
# else:
#     print("Evidence favors model 2.")
print("Bayes factor template: fill in your paths and uncomment.")
```

**Cell 19 — markdown: 5.1 caveat**
```
### Section 5.1: Bayes factors are prior-dependent — always sanity-check

`logZ` integrates the likelihood over the *entire prior volume*. If two models
have very different prior ranges (or one model has extra parameters with wide,
unconstrained priors), the Bayes factor reflects the Occam penalty from the
unused prior volume — not necessarily a real fit-quality difference.

**Always cross-check against χ² and information criteria.** If `logB_12` says
"model 1 strongly preferred" but `chi2_red` and AIC say the two are nearly
identical, the Bayes factor is penalising model 2's broader prior — mathematically
correct but worth being explicit about in your paper.

**For a clean model comparison:**
1. Use the *same* free-parameter set with the *same* prior ranges across both models
   (differ only in the thing you're testing).
2. Use identical wavelength coverage and data points.
3. Report logZ, logZ_err, chi2_red, and AIC/BIC together.
```

**Cell 20 — markdown: Section 6**
```
## Section 6: Practical use cases

The two subsections below show the complete workflow for the most common
model-comparison scenarios in atmospheric retrieval.
```

**Cell 21 — markdown: 6.1 fixed vs free**
```
### Section 6.1: Fixed vs free parameter (e.g. surface gravity `logg`)

Run two ForMoSA fits on the same data:
- **Fixed fit**: `logg` set as a `constant` parameter at a literature value.
- **Free fit**: `logg` given a `uniform` prior over a physically reasonable range.

Then compare logZ values.

**Reading the result:**
- `logB > 3` and the `logg` posterior is well-constrained (narrow, away from prior edges)
  → the data constrain `logg`; keep it free.
- `logB ≈ 0` and the `logg` posterior spans most of the prior range
  → the data don't constrain `logg`; fixing it is justified and the simpler model
  is preferred on Occam grounds.
- `logB < -3` (fixed strongly preferred) → check your prior range on the free fit;
  a too-wide prior penalises the free model unfairly and may not be physically motivated.
```

**Cell 22 — code: fixed vs free logg**
```python
# Template. Fill in your result paths.

# results_json_fixed = Path(".").resolve() / "path/to/fixed_logg/NS_results/results_pymultinest.json"
# results_json_free  = Path(".").resolve() / "path/to/free_logg/NS_results/results_pymultinest.json"

# results_fixed = NSResults.from_dict(json.load(open(results_json_fixed)))
# results_free  = NSResults.from_dict(json.load(open(results_json_free)))

# logB     = results_free.logz[0] - results_fixed.logz[0]
# logB_err = np.hypot(results_free.logz[1], results_fixed.logz[1])
# print(f"ln B(free vs fixed logg) = {logB:.2f} ± {logB_err:.2f}")

# # Also show the logg posterior from the free fit
# if 'logg' in results_free.free_parameters:
#     j = results_free.free_parameters.index('logg')
#     s = results_free.samples[results_free.burn_in:, j]
#     w = results_free.weights[results_free.burn_in:]
#     w_norm = w / w.sum()
#     med = np.average(s, weights=w_norm)
#     lo  = np.percentile(s, 16)
#     hi  = np.percentile(s, 84)
#     print(f"logg (weighted median):    {med:.2f}")
#     print(f"logg (16th–84th perc.):    [{lo:.2f}, {hi:.2f}]")
print("Section 6.1: fill in your result paths and uncomment.")
```

**Cell 23 — markdown: 6.2 grid comparison**
```
### Section 6.2: Comparing atmospheric model grids (e.g. BT-Settl vs Sonora-Bobcat)

Run the same observations through two different atmospheric grids and compare their
logZ values.

**Important caveats before comparing grids:**

1. **Same free-parameter set, same prior ranges.** If grid A uses `[M/H]` and grid B
   uses `[Fe/H]` with a different prior range, the evidence difference reflects prior
   volume, not fit quality.
2. **Identical wavelength coverage.** logZ scales with the number of data points.
   Fitting different wavelength windows makes logZ values incomparable.
3. **Also compare χ²_red.** Grid differences often show up most clearly in how well
   the model reproduces individual spectral lines — that's easier to see in χ² than
   in logZ, which integrates over the whole prior.
```

**Cell 24 — code: grid comparison**
```python
# Template. Fill in your result paths.

# results_json_A = Path(".").resolve() / "path/to/grid_A/NS_results/results_pymultinest.json"
# results_json_B = Path(".").resolve() / "path/to/grid_B/NS_results/results_pymultinest.json"

# results_A = NSResults.from_dict(json.load(open(results_json_A)))
# results_B = NSResults.from_dict(json.load(open(results_json_B)))

# logZ_A, errA = results_A.logz
# logZ_B, errB = results_B.logz

# print(f"log Z (grid A) = {logZ_A:.2f} ± {errA:.2f}")
# print(f"log Z (grid B) = {logZ_B:.2f} ± {errB:.2f}")
# print(f"ln B_AB        = {logZ_A - logZ_B:.2f}  (positive favors A)")
print("Section 6.2: fill in your result paths and uncomment.")
```

**Cell 25 — markdown: Section 7**
```
## Section 7: Reporting checklist

When reporting model selection in a paper, give **all four numbers** together.
They tell different stories, and a referee will ask for any you omit:

| Quantity | What it tells you |
|----------|-------------------|
| `logZ ± logZ_err` | Bayesian evidence; accounts for prior complexity |
| `chi2_red` | Goodness of fit; intuitive and comparable across papers |
| `AIC` | Fit quality penalised weakly for complexity |
| `BIC` | Fit quality penalised strongly for complexity (preferred when n is large) |

**Also report:**
- The prior ranges you used (logZ is prior-dependent)
- The number of live points (`npoints`) and the algorithm (PyMultiNest / UltraNest / nestle)
- The ESS, so readers know how well the posterior is resolved

## Section 8: Next steps

- **API docs:** `docs/api/` for `NSResults`, `NSAnalysis`, and all config dataclasses.
- **Tutorial 7 — Advanced Plotting:** For customising corner, radar, chains, and
  best-fit plots with ForMoSA's config system and direct matplotlib post-processing.
```

- [ ] **Step 2: Verify the file was written**

```bash
python -c "import json; nb=json.load(open('docs/tutorials/plotting/tutorial_statistical_tests.ipynb')); print(len(nb['cells']), 'cells')"
```
Expected: prints the cell count (> 20)

- [ ] **Step 3: Commit**

```bash
git add docs/tutorials/plotting/tutorial_statistical_tests.ipynb
git commit -m "docs: add Tutorial 8 — Statistical Tests and Model Selection"
```

---

### Task 3: Delete the old source files

**Files:**
- Delete: `docs/tutorials/plotting/ForMoSA_Advanced_Plotting_Tutorial.ipynb`
- Delete: `docs/tutorials/plotting/ForMoSA_Statistical_Tests_Tutorial.ipynb`

- [ ] **Step 1: Remove old files**

```bash
git rm docs/tutorials/plotting/ForMoSA_Advanced_Plotting_Tutorial.ipynb
git rm docs/tutorials/plotting/ForMoSA_Statistical_Tests_Tutorial.ipynb
```

- [ ] **Step 2: Commit**

```bash
git commit -m "docs: remove old unformatted notebook drafts (replaced by Tutorial 7 & 8)"
```

---

### Task 4: Verify final state

- [ ] **Step 1: Check plotting directory**

```bash
ls docs/tutorials/plotting/
```

Expected output contains:
```
tutorial_advanced_plotting.ipynb           ← Tutorial 5 (unchanged)
tutorial_advanced_plotting_custom.ipynb    ← Tutorial 7 (new)
tutorial_statistical_tests.ipynb           ← Tutorial 8 (new)
results/
ns_results.json
corner_publication.pdf
corner_publication.png
```

Does NOT contain `ForMoSA_Advanced_Plotting_Tutorial.ipynb` or `ForMoSA_Statistical_Tests_Tutorial.ipynb`.

- [ ] **Step 2: Spot-check notebook structure**

```bash
python -c "
import json
for name in ['tutorial_advanced_plotting_custom.ipynb', 'tutorial_statistical_tests.ipynb']:
    nb = json.load(open(f'docs/tutorials/plotting/{name}'))
    first_cell = nb['cells'][0]['source']
    first_line = first_cell[0] if isinstance(first_cell, list) else first_cell
    print(name, '->', first_line.strip())
"
```

Expected: both print their `# Tutorial N — ...` header lines.

---

## Self-Review

**Spec coverage:**
- ✅ Tutorial 7: header, setup, rcParams, 3 loading methods, best-fit, corner, radar, chains, saving, next steps
- ✅ Tutorial 8: header, setup, convergence (logZ + ESS), chi2_red, AIC/BIC, Bayes factors, use cases, reporting checklist
- ✅ Paths: all use `Path(".").resolve() / "results" / "NS_results" / "results_pymultinest.json"` with FileNotFoundError guard
- ✅ Section numbering: `## Section N: Title` throughout
- ✅ Old files deleted via `git rm`
- ✅ Newbie explanations: every non-trivial quantity has a plain-English paragraph before the code

**Placeholder scan:** No TBD/TODO in notebook cells — all `>>> REPLACE` comments now have working defaults pointing at the committed results folder, with commented blocks for user's own paths.

**Type consistency:** `results.free_parameters`, `results.samples`, `results.weights`, `results.burn_in`, `results.logz`, `results.logl`, `results.median_parameters`, `results._interval` — consistent across all tasks.
