# Radar Plot Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign `plot_radars` to use a Cartesian polygon spider chart with per-axis tab10 colouring, value labels at the outer spoke tip, optional tick labels, and no fill.

**Architecture:** Switch from matplotlib's polar projection to a regular `Axes` with explicit Cartesian maths — this is the only way to get true straight-line polygon grids without fighting matplotlib internals. All drawing uses `ax.plot` / `ax.scatter` / `ax.text`. No `fill_between`. New config fields added to `RadarPlotConfig`; existing fields kept for backward compatibility.

**Tech Stack:** Python 3.12, matplotlib, numpy, pytest

---

## File Map

| File | What changes |
|------|-------------|
| `ForMoSA/core/config.py` | Add 10 new fields to `RadarPlotConfig` (lines 329–390) |
| `ForMoSA/nested_sampling/plotting.py` | Add `_format_val` static method; full rewrite of `plot_radars` (lines 162–330) |
| `tests/test_plotting_radar.py` | New test file (create) |

---

## Task 1 — New `RadarPlotConfig` fields

**Files:**
- Modify: `ForMoSA/core/config.py:338–358`
- Test: `tests/test_plotting_radar.py`

- [ ] **Step 1: Create the test file with a failing test**

```python
# tests/test_plotting_radar.py
import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ForMoSA.core.config import RadarPlotConfig
from ForMoSA.core.errors import ForMoSAError
from ForMoSA.nested_sampling.results import NSResults
from ForMoSA.nested_sampling.plotting import Plotting


# ── shared fixture ────────────────────────────────────────────────────────
@pytest.fixture
def ns_results_5p():
    """NSResults with 5 parameters, 200 samples, uniform weights."""
    rng = np.random.default_rng(42)
    N, P = 200, 5
    samples = rng.uniform(
        [1500, 3.0, 0.8, 30.0, -5.0],
        [2800, 5.0, 1.8, 60.0,  30.0],
        (N, P),
    )
    weights = np.ones(N) / N
    return NSResults(
        samples=samples,
        weights=weights,
        logl=rng.uniform(-100, 0, N),
        logvol=np.linspace(0, -10, N),
        logz=[-50.0, 0.1],
        free_parameters=['Teff', 'log g', 'r', 'd', 'rv'],
    )


@pytest.fixture
def plotting(ns_results_5p):
    return Plotting(results=ns_results_5p, logger=None)


# ── Task 1 tests ──────────────────────────────────────────────────────────
def test_radar_config_new_fields_exist():
    cfg = RadarPlotConfig()
    assert cfg.polygon_grid is True
    assert cfg.n_rings == 4
    assert cfg.show_quantile_lines is True
    assert cfg.alpha_quantile_lines == 0.5
    assert cfg.linewidth_quantile_lines == 1.0
    assert cfg.use_axis_colors is True
    assert cfg.fontsize_values == 8
    assert cfg.show_tick_labels is False
    assert cfg.fontsize_tick_labels == 8
    assert cfg.alpha_tick_labels == 0.5


def test_radar_config_set_via_method():
    cfg = RadarPlotConfig()
    cfg.set_radar_plot_config(polygon_grid=False, show_tick_labels=True, n_rings=5)
    assert cfg.polygon_grid is False
    assert cfg.show_tick_labels is True
    assert cfg.n_rings == 5


def test_radar_config_unknown_key_raises():
    cfg = RadarPlotConfig()
    with pytest.raises(ForMoSAError):
        cfg.set_radar_plot_config(nonexistent_key=True)
```

- [ ] **Step 2: Run — expect FAIL (AttributeError on missing fields)**

```bash
cd ~/Karmabhumi/Packages/ForMoSA
pytest tests/test_plotting_radar.py::test_radar_config_new_fields_exist -v
```

Expected: `FAILED — AttributeError: 'RadarPlotConfig' object has no attribute 'polygon_grid'`

- [ ] **Step 3: Add the 10 new fields to `RadarPlotConfig`**

Open `ForMoSA/core/config.py`. After line 358 (`lw_quantiles: float = 2.0`), add:

```python
    # Polygon grid
    polygon_grid: bool = True          # True → polygon rings; False → legacy circular polar
    n_rings: int = 4                   # number of concentric rings

    # Uncertainty bounds — replace fill_between
    show_quantile_lines: bool = True   # draw dashed q_low / q_high polygons
    alpha_quantile_lines: float = 0.5
    linewidth_quantile_lines: float = 1.0

    # Per-axis tab10 colouring
    use_axis_colors: bool = True       # True → tab10 per axis; False → use color_quantiles

    # Value annotation (placed at outer spoke tip, next to axis label)
    fontsize_values: int = 8

    # Tick labels along each spoke
    show_tick_labels: bool = False     # opt-in; off by default
    fontsize_tick_labels: int = 8
    alpha_tick_labels: float = 0.5
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_plotting_radar.py::test_radar_config_new_fields_exist \
       tests/test_plotting_radar.py::test_radar_config_set_via_method \
       tests/test_plotting_radar.py::test_radar_config_unknown_key_raises -v
```

Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add ForMoSA/core/config.py tests/test_plotting_radar.py
git commit -m "feat(radar): add 10 new RadarPlotConfig fields for polygon grid, tick labels, and axis colours"
```

---

## Task 2 — `_format_val` helper

**Files:**
- Modify: `ForMoSA/nested_sampling/plotting.py` (add static method after `__init__`)
- Test: `tests/test_plotting_radar.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/test_plotting_radar.py`:

```python
# ── Task 2 tests ──────────────────────────────────────────────────────────
def test_format_val_large_positive():
    assert Plotting._format_val(2345.6) == '2346'

def test_format_val_large_negative():
    assert Plotting._format_val(-1234.5) == '-1235'

def test_format_val_medium():
    assert Plotting._format_val(23.456) == '23.5'

def test_format_val_small():
    assert Plotting._format_val(1.234) == '1.23'

def test_format_val_boundary_10():
    # Exactly 10 → medium branch
    assert Plotting._format_val(10.0) == '10.0'

def test_format_val_boundary_1000():
    # Exactly 1000 → large branch
    assert Plotting._format_val(1000.0) == '1000'
```

- [ ] **Step 2: Run — expect FAIL**

```bash
pytest tests/test_plotting_radar.py::test_format_val_large_positive -v
```

Expected: `FAILED — AttributeError: type object 'Plotting' has no attribute '_format_val'`

- [ ] **Step 3: Add `_format_val` to `Plotting`**

In `ForMoSA/nested_sampling/plotting.py`, insert after the `ns_results` property (after line ~65):

```python
    @staticmethod
    def _format_val(v: float) -> str:
        """Format a scalar for the radar value annotation.

        Returns 0 decimal places for |v| >= 1000,
                1 decimal place  for |v| >= 10,
                2 decimal places otherwise.
        """
        if abs(v) >= 1000:
            return f'{v:.0f}'
        if abs(v) >= 10:
            return f'{v:.1f}'
        return f'{v:.2f}'
```

- [ ] **Step 4: Run — expect PASS**

```bash
pytest tests/test_plotting_radar.py -k "format_val" -v
```

Expected: `6 passed`

- [ ] **Step 5: Commit**

```bash
git add ForMoSA/nested_sampling/plotting.py tests/test_plotting_radar.py
git commit -m "feat(radar): add _format_val static helper for value label formatting"
```

---

## Task 3 — Rewrite `plot_radars`

**Files:**
- Modify: `ForMoSA/nested_sampling/plotting.py:162–330`
- Test: `tests/test_plotting_radar.py`

- [ ] **Step 1: Add integration tests**

Append to `tests/test_plotting_radar.py`:

```python
# ── Task 3 tests ──────────────────────────────────────────────────────────
def test_plot_radars_returns_fig_axes(plotting):
    fig, ax = plotting.plot_radars()
    import matplotlib.figure
    import matplotlib.axes
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(ax, matplotlib.axes.Axes)
    plt.close(fig)


def test_plot_radars_no_fill(plotting):
    """No filled polygon — ax.collections must be empty."""
    fig, ax = plotting.plot_radars()
    assert len(ax.collections) == 0, (
        f"Expected no filled collections, got {len(ax.collections)}"
    )
    plt.close(fig)


def test_plot_radars_axis_labels_present(plotting):
    """All 5 parameter names appear as text objects."""
    fig, ax = plotting.plot_radars()
    text_strings = [t.get_text() for t in ax.texts]
    for param in ['Teff', 'log g', 'r', 'd', 'rv']:
        assert any(param in s for s in text_strings), (
            f"Parameter '{param}' not found in ax.texts: {text_strings}"
        )
    plt.close(fig)


def test_plot_radars_axis_off(plotting):
    """Axis frame must be hidden."""
    fig, ax = plotting.plot_radars()
    assert not ax.axison
    plt.close(fig)


def test_plot_radars_aspect_equal(plotting):
    """Axes must have equal aspect ratio."""
    fig, ax = plotting.plot_radars()
    assert ax.get_aspect() == 'equal'
    plt.close(fig)


def test_plot_radars_tab10_colors_on_markers(plotting):
    """With use_axis_colors=True (default), the 5 marker dots use tab10 colours."""
    import matplotlib.pyplot as plt
    tab10 = plt.cm.tab10.colors
    fig, ax = plotting.plot_radars()
    # PathCollections from ax.scatter — skip the white halo ones (facecolor white)
    colored = [
        c for c in ax.collections
        # no collections expected — markers are scatter → PathCollection
    ]
    # Actually scatter returns PathCollections stored in ax.collections
    # Filter: facecolor != white
    import numpy as np
    marker_colors = []
    for coll in ax.collections:
        fc = coll.get_facecolor()
        if fc is not None and len(fc) > 0:
            rgba = fc[0]
            if not np.allclose(rgba[:3], [1, 1, 1]):
                marker_colors.append(tuple(rgba[:3]))
    assert len(marker_colors) == 5, f"Expected 5 coloured markers, got {len(marker_colors)}"
    plt.close(fig)
```

- [ ] **Step 2: Run — expect FAILs**

```bash
pytest tests/test_plotting_radar.py -k "plot_radars" -v 2>&1 | tail -20
```

Expected: multiple failures (the current polar-based implementation produces collections from `fill_between`, uses polar axes, etc.)

- [ ] **Step 3: Replace `plot_radars` entirely**

In `ForMoSA/nested_sampling/plotting.py`, replace the full body of `plot_radars` (lines 162–330, keeping the `def plot_radars(self)` signature and docstring) with:

```python
    def plot_radars(self) -> tuple[Figure, Axes]:
        '''
        Radar plot the posterior samples as a polygon spider chart.

        Draws a Cartesian spider chart (polygon grid, no circular projection).
        Configure via ``PLOTS_CONFIG.RadarPlot`` (a :class:`RadarPlotConfig`
        instance).

        Returns
        -------
        tuple[matplotlib.figure.Figure, matplotlib.axes._axes.Axes]
            Figure and Axes objects.

        Notes
        -----
        Authors: Paulina Palma-Bifani, Matthieu Ravet, Allan Denis, Bhavesh Rajpoot
        '''
        self._logger.info('    Plotting radar plot of the chains')

        samples = self.ns_results.samples[self.ns_results.burn_in:]
        weights = self.ns_results.weights[self.ns_results.burn_in:]
        config  = PLOTS_CONFIG.RadarPlot
        params  = self.ns_results.free_parameters
        N       = len(params)

        # ── Weighted quantiles ────────────────────────────────────────────
        q_low, q_med, q_high = [], [], []
        for i in range(samples.shape[1]):
            q_low.append(self.ns_results._weighted_quantile(
                samples[:, i], weights, config.quantiles[0]))
            q_med.append(self.ns_results._weighted_quantile(
                samples[:, i], weights, 0.5))
            q_high.append(self.ns_results._weighted_quantile(
                samples[:, i], weights, config.quantiles[1]))

        q_low  = np.array(q_low)
        q_med  = np.array(q_med)
        q_high = np.array(q_high)

        prior_mins = np.min(samples, axis=0)
        prior_maxs = np.max(samples, axis=0)

        # ── Normalise each parameter to [0, 1] over sample range ─────────
        def _norm(vals: np.ndarray) -> np.ndarray:
            out = np.empty(N)
            for i in range(N):
                span = prior_maxs[i] - prior_mins[i]
                span = span if span != 0.0 else 1.0
                out[i] = (vals[i] - prior_mins[i]) / span
            return out

        q_low_n  = _norm(q_low)
        q_med_n  = _norm(q_med)
        q_high_n = _norm(q_high)

        # ── Cartesian polar coordinates ───────────────────────────────────
        # Angles: clock-face, starting from top (-π/2), going clockwise.
        R      = 1.0
        angles = np.array([-np.pi / 2 + 2 * np.pi * i / N for i in range(N)])
        cos_a  = np.cos(angles)
        sin_a  = np.sin(angles)

        def _xy(radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return radii * cos_a, radii * sin_a

        def _closed(x: np.ndarray, y: np.ndarray):
            return np.append(x, x[0]), np.append(y, y[0])

        # ── Figure ────────────────────────────────────────────────────────
        label_pad = 0.18   # axis name clearance beyond outer ring
        value_pad = 0.15   # additional offset for value text
        lim = R + label_pad + value_pad + 0.12

        fig, ax = plt.subplots(figsize=config.figsize)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # ── Polygon grid rings ────────────────────────────────────────────
        for ring in range(1, config.n_rings + 1):
            r        = R * ring / config.n_rings
            rx, ry   = _xy(np.full(N, r))
            xs, ys   = _closed(rx, ry)
            is_outer = ring == config.n_rings
            ax.plot(xs, ys,
                    color='gray',
                    linewidth=1.5 if is_outer else 0.8,
                    linestyle='-' if is_outer else '--',
                    alpha=0.55 if is_outer else 0.28,
                    zorder=1)

        # ── Axis spokes ───────────────────────────────────────────────────
        ox, oy = _xy(np.full(N, R))
        for i in range(N):
            ax.plot([0.0, ox[i]], [0.0, oy[i]],
                    color='gray', linewidth=0.8, alpha=0.40, zorder=1)

        # ── Tick labels along each spoke (opt-in) ────────────────────────
        if config.show_tick_labels:
            for i in range(N):
                perp  = angles[i] + np.pi / 2   # perpendicular direction for nudge
                min_v = prior_mins[i]
                max_v = prior_maxs[i]
                for ring in range(1, config.n_rings):
                    r   = R * ring / config.n_rings
                    val = min_v + (max_v - min_v) * ring / config.n_rings
                    x   = r * cos_a[i] + 0.06 * np.cos(perp)
                    y   = r * sin_a[i] + 0.06 * np.sin(perp)
                    ax.text(x, y, self._format_val(val),
                            fontsize=config.fontsize_tick_labels,
                            alpha=config.alpha_tick_labels,
                            ha='center', va='center', zorder=5)

        # ── Quantile bound lines (replaces fill_between) ──────────────────
        if config.show_quantile_lines:
            for q_n in (q_low_n, q_high_n):
                qx, qy = _xy(q_n)
                ax.plot(*_closed(qx, qy),
                        color=config.color_radar,
                        linewidth=config.linewidth_quantile_lines,
                        linestyle='--',
                        alpha=config.alpha_quantile_lines,
                        zorder=2)

        # ── Median line ───────────────────────────────────────────────────
        mx, my = _xy(q_med_n)
        ax.plot(*_closed(mx, my),
                color=config.color_radar,
                linewidth=config.linewidth,
                solid_capstyle='round',
                zorder=3)

        # ── Axis colours ──────────────────────────────────────────────────
        if config.use_axis_colors:
            axis_colors = [plt.cm.tab10.colors[i % 10] for i in range(N)]
        else:
            axis_colors = [config.color_quantiles] * N

        # ── Markers ───────────────────────────────────────────────────────
        for i in range(N):
            x, y = mx[i], my[i]
            # White halo for contrast
            ax.scatter(x, y, s=config.size_quantiles + 40,
                       color='white', edgecolors='none', zorder=4)
            # Coloured dot
            ax.scatter(x, y, s=config.size_quantiles,
                       color=axis_colors[i],
                       edgecolors='white', linewidths=config.lw_quantiles,
                       zorder=5)

        # ── Axis labels + value annotations at outer spoke tip ────────────
        lx, ly  = _xy(np.full(N, R + label_pad))
        vx, vy  = _xy(np.full(N, R + label_pad + value_pad))

        for i in range(N):
            # Parameter name — bold, axis colour
            ax.text(lx[i], ly[i], params[i],
                    fontsize=config.fontsize_names, fontweight='bold',
                    color=axis_colors[i],
                    ha='center', va='center', zorder=6)

            # Value annotation — same colour, smaller font
            med_s  = self._format_val(q_med[i])
            low_s  = self._format_val(q_med[i] - q_low[i])
            high_s = self._format_val(q_high[i] - q_med[i])
            ax.text(vx[i], vy[i],
                    f'${med_s}_{{-{low_s}}}^{{+{high_s}}}$',
                    fontsize=config.fontsize_values,
                    color=axis_colors[i],
                    ha='center', va='center', zorder=6)

        return fig, ax
```

Also remove the now-unused `path_effects` import at the top of the file if it is no longer used anywhere else:

```bash
grep -n "path_effects" ForMoSA/nested_sampling/plotting.py
```

If the only occurrence is the import line and the (now deleted) `plot_radars` body, remove the import line.

- [ ] **Step 4: Run all Task 3 tests**

```bash
pytest tests/test_plotting_radar.py -k "plot_radars" -v
```

Expected: `6 passed`

- [ ] **Step 5: Run the full test suite to check no regressions**

```bash
cd ~/Karmabhumi/Packages/ForMoSA && pytest -v 2>&1 | tail -20
```

Expected: all previously passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add ForMoSA/nested_sampling/plotting.py tests/test_plotting_radar.py
git commit -m "feat(radar): rewrite plot_radars — polygon grid, tab10 colours, value labels at outer edge, no fill"
```

---

## Task 4 — Tick labels feature test + cleanup

**Files:**
- Test: `tests/test_plotting_radar.py`
- Modify: `ForMoSA/core/config.py` (docstring only, if needed)

- [ ] **Step 1: Add tick-label feature tests**

Append to `tests/test_plotting_radar.py`:

```python
# ── Task 4 tests ──────────────────────────────────────────────────────────
def test_plot_radars_no_tick_labels_by_default(plotting):
    """show_tick_labels=False by default — text count = 2*N (names + values)."""
    from ForMoSA.core.config import PLOTS_CONFIG
    PLOTS_CONFIG.RadarPlot.show_tick_labels = False   # ensure default
    fig, ax = plotting.plot_radars()
    # 5 axis-name texts + 5 value texts = 10
    assert len(ax.texts) == 10, (
        f"Expected 10 texts (5 names + 5 values), got {len(ax.texts)}: "
        + str([t.get_text() for t in ax.texts])
    )
    plt.close(fig)


def test_plot_radars_tick_labels_enabled(plotting):
    """show_tick_labels=True adds (n_rings-1)*N extra text objects."""
    from ForMoSA.core.config import PLOTS_CONFIG
    PLOTS_CONFIG.RadarPlot.show_tick_labels = True
    PLOTS_CONFIG.RadarPlot.n_rings = 4
    fig, ax = plotting.plot_radars()
    # 5 names + 5 values + (4-1)*5 tick labels = 10 + 15 = 25
    assert len(ax.texts) == 25, (
        f"Expected 25 texts, got {len(ax.texts)}"
    )
    PLOTS_CONFIG.RadarPlot.show_tick_labels = False   # reset
    plt.close(fig)


def test_plot_radars_use_axis_colors_false(plotting):
    """use_axis_colors=False falls back to uniform color_quantiles colour."""
    from ForMoSA.core.config import PLOTS_CONFIG
    import numpy as np
    PLOTS_CONFIG.RadarPlot.use_axis_colors = False
    fig, ax = plotting.plot_radars()
    # All 5 coloured markers should share the same RGB
    coloured_rgs = []
    for coll in ax.collections:
        fc = coll.get_facecolor()
        if fc is not None and len(fc) > 0:
            rgba = fc[0]
            if not np.allclose(rgba[:3], [1, 1, 1]):
                coloured_rgs.append(tuple(np.round(rgba[:3], 4)))
    assert len(set(coloured_rgs)) == 1, (
        f"Expected all markers same colour, got: {set(coloured_rgs)}"
    )
    PLOTS_CONFIG.RadarPlot.use_axis_colors = True   # reset
    plt.close(fig)


def test_plot_radars_quantile_lines_disabled(plotting):
    """show_quantile_lines=False produces exactly 1 + n_rings lines (median + grid)."""
    from ForMoSA.core.config import PLOTS_CONFIG
    PLOTS_CONFIG.RadarPlot.show_quantile_lines = False
    PLOTS_CONFIG.RadarPlot.n_rings = 4
    fig, ax = plotting.plot_radars()
    # grid: 4 ring polygons + 5 spokes = 9; median = 1 → total 10
    assert len(ax.lines) == 10, (
        f"Expected 10 lines, got {len(ax.lines)}"
    )
    PLOTS_CONFIG.RadarPlot.show_quantile_lines = True   # reset
    plt.close(fig)
```

- [ ] **Step 2: Run — expect PASS**

```bash
pytest tests/test_plotting_radar.py -v
```

Expected: all tests pass (the implementation already handles all these cases).

- [ ] **Step 3: Run full suite one final time**

```bash
cd ~/Karmabhumi/Packages/ForMoSA && pytest -v
```

Expected: all tests pass.

- [ ] **Step 4: Final commit**

```bash
git add tests/test_plotting_radar.py
git commit -m "test(radar): add tick-label, axis-colour, and quantile-line feature tests"
```

---

## Self-Review

**Spec coverage:**
- ✅ Value label positioned next to axis label at outer edge (Task 3, `lx/vx` computation)
- ✅ Polygon boundary, inner dashed / outer solid (Task 3, `polygon grid rings` block)
- ✅ No fill — quantile lines instead (Task 3, `show_quantile_lines` block; `fill_between` removed)
- ✅ Tick labels opt-in, α=0.5, small font, along spoke (Task 3, `show_tick_labels` block)
- ✅ Tab10 per axis: label text + value text + marker; `color_radar` for median line (Task 3, `axis_colors` block)
- ✅ `_format_val` helper (Task 2)
- ✅ New config fields + `set_radar_plot_config` compat (Task 1)
- ✅ `path_effects` import cleanup noted in Task 3 Step 3

**Placeholder scan:** None found.

**Type consistency:** `_format_val(v: float) -> str` used identically in Tasks 2 and 3. `axis_colors` is `list[tuple]` in both marker and label loops. Config field names match between Task 1 (definition) and Task 3 (usage) for all 10 new fields.
