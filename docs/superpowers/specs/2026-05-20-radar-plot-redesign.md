# Radar Plot Redesign — Design Spec
_2026-05-20_

## Summary

Redesign `Plotting.plot_radars` and `RadarPlotConfig` to fix label positioning, replace the circular polar grid with a polygon spider chart, drop the filled area, add optional per-axis tick labels, and colour each axis with a distinct tab10 colour.

---

## Changes at a Glance

| # | Item | Current | After |
|---|------|---------|-------|
| 1 | Value label position | Radially offset + angular drift (`angle+0.15`) | Next to axis label at outer spoke tip |
| 2 | Grid boundary | Circular (matplotlib polar projection) | Straight-line polygon; inner rings dashed, outer solid |
| 3 | Uncertainty area | `fill_between` filled polygon | No fill; q_low and q_high drawn as thin dashed polygons |
| 4 | Tick labels | Commented-out dead code | Optional; along each spoke, α=0.5, small font |
| 5 | Axis colours | Uniform `color_quantiles` | Tab10 colour per axis: label text + value text + marker dot; `color_radar` stays for median line |

---

## Architecture

### `RadarPlotConfig` (`ForMoSA/core/config.py`)

Add new fields; keep all existing fields so `set_radar_plot_config` calls stay valid.

```python
# Grid
polygon_grid: bool = True       # True → polygon spider; False → legacy circular polar
n_rings: int = 4                # how many concentric rings to draw

# Uncertainty — replaces fill
show_quantile_lines: bool = True
alpha_quantile_lines: float = 0.5
linewidth_quantile_lines: float = 1.0

# Axis colours
use_axis_colors: bool = True    # True → tab10 per axis; False → use color_quantiles for all

# Value labels (now at outer edge next to axis label)
fontsize_values: int = 8        # replaces fontsize_ticks for the value annotation

# Tick labels along spokes
show_tick_labels: bool = False  # off by default; opt-in
fontsize_tick_labels: int = 8
alpha_tick_labels: float = 0.5
```

Existing fields kept as-is: `color_radar`, `linewidth`, `color_uncertainty`, `alpha_fill`, `color_quantiles`, `color_ticks`, `fontsize_names`, `fontsize_ticks`, `size_quantiles`, `lw_quantiles`, `quantiles`, `figsize`.

---

### `Plotting.plot_radars` (`ForMoSA/nested_sampling/plotting.py`)

**Switch from polar axes to regular axes.** The circular polar projection cannot produce a true polygon grid without fighting matplotlib internals. Using a regular axes with explicit Cartesian maths gives full control and simpler code.

#### Coordinate system

```
angles[i] = -π/2 + 2πi/N      # clock-face, starting from top
R = 1.0                         # outer ring radius in data units

outer vertex i:
  x = R · cos(angles[i])
  y = R · sin(angles[i])

data point i:
  x = q_med_norm[i] · cos(angles[i])
  y = q_med_norm[i] · sin(angles[i])
```

#### Drawing order (back → front)

1. **Polygon grid rings** — for `ring` in 1…`n_rings`:
   - inner rings (`ring < n_rings`): dashed, low alpha
   - outer ring (`ring == n_rings`): solid, higher alpha
2. **Axis spokes** — gray lines from `(0,0)` to each outer vertex
3. **Tick labels** — if `show_tick_labels`: at each inner ring × each spoke; physical value = `prior_mins[i] + (prior_maxs[i]-prior_mins[i]) × ring/n_rings`; nudged slightly perpendicular to spoke; `globalAlpha = alpha_tick_labels`
4. **Quantile lines** — if `show_quantile_lines`: dashed polygon at q_low_norm and q_high_norm; colour = `color_radar` + `alpha_quantile_lines`
5. **Median line** — solid polygon at q_med_norm; colour = `color_radar`, `linewidth`
6. **Markers** — per-axis tab10 colour (or `color_quantiles` if `use_axis_colors=False`); white halo ring behind each dot
7. **Axis labels** — bold text at `(R + label_pad) · (cos, sin)`; colour = axis colour
8. **Value annotations** — `$med_{-low}^{+high}$` at `(R + label_pad + value_pad) · (cos, sin)` on same spoke; smaller font; same axis colour

#### Label placement detail

```
label_pad  = 0.18   # axis name clearance beyond outer ring
value_pad  = 0.15   # additional offset for value text below name
```

Both offsets are along the spoke direction `(cos(a), sin(a))`. No perpendicular drift.

#### Axis limits

```python
ax.set_aspect('equal')
ax.axis('off')
lim = R + label_pad + value_pad + 0.12   # ~1.45 with defaults
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
```

#### Helper: `_format_val(v)`

Internal static method; returns `f'{v:.0f}'` if `|v| ≥ 1000`, `f'{v:.1f}'` if `|v| ≥ 10`, else `f'{v:.2f}'`. Eliminates duplicated formatting logic.

#### Return

Same as before: `tuple[Figure, Axes]`.

---

## What Is Not Changing

- Quantile computation (`_weighted_quantile`)
- Normalisation logic (`prior_mins` / `prior_maxs` from sample range)
- The closing-element trick (`append(x[0])`) — not needed for Cartesian; will be removed
- Public API: `plot_radars()` signature unchanged
- `set_radar_plot_config` validation logic

---

## Files Touched

| File | Change |
|------|--------|
| `ForMoSA/core/config.py` | Add 7 new fields to `RadarPlotConfig` |
| `ForMoSA/nested_sampling/plotting.py` | Full rewrite of `plot_radars`; add `_format_val` helper |

No other files change.
