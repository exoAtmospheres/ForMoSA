# Design: Reformat Tutorial 7 & 8 Notebooks

**Date:** 2026-05-19  
**Status:** Approved  
**Branch:** class_docs

---

## Goal

Reformat two Claude-generated notebooks in `docs/tutorials/plotting/` to match the style, structure, and pedagogical quality of the existing `tutorial_advanced_plotting.ipynb` (Tutorial 5). Add them to the tutorial series as Tutorial 7 and Tutorial 8.

---

## Source files

| Old filename | New filename | Tutorial # |
|---|---|---|
| `ForMoSA_Advanced_Plotting_Tutorial.ipynb` | `tutorial_advanced_plotting_custom.ipynb` | 7 |
| `ForMoSA_Statistical_Tests_Tutorial.ipynb` | `tutorial_statistical_tests.ipynb` | 8 |

Old files are deleted after new ones are written.

---

## Style reference: `tutorial_advanced_plotting.ipynb` (Tutorial 5)

Key patterns to replicate:

1. **Header cell**: `# Tutorial N — Title` + "What you'll learn:" bullets + "Estimated runtime:" + "Prerequisites:" note.
2. **Section 0 — Setup**: `try: import ForMoSA` version check cell, followed by results-loading cell using `Path(".").resolve()` with `FileNotFoundError` guard.
3. **Section labels**: `## Section N: Title` in every markdown heading.
4. **Path pattern**: `Path(".").resolve() / "results" / "NS_results" / "results_pymultinest.json"` (relative to the `plotting/` directory where the notebooks live).
5. **Inline explanations**: Every non-trivial code block is preceded by a markdown cell explaining the *why*, not just the *what*. Suitable for a newbie who has run Tutorial 2 but has no stats background.
6. **Final section**: "Next steps" with links to the next tutorial and API docs.

---

## Tutorial 7 — Advanced Plotting: Custom Figures

**File:** `docs/tutorials/plotting/tutorial_advanced_plotting_custom.ipynb`

### Sections

- **Section 0 — Setup**: ForMoSA version check + `rcParams` primer (what it is, why it comes first, how `rc_context` scopes it).
- **Section 1 — Loading results**: Three methods explained with newbie context. Paths fixed to actual results folder.
  - Method A: `NSResults` from JSON (fastest; corner/chains/radar only)
  - Method B: From raw PyMultiNest output files
  - Method C: Full `Analysis` reload (required for best-fit plot)
- **Section 2 — Best-fit plot**: Individual call, config, per-observation colours, figure-wide config (`MAIN_PLOT`), post-hoc axis tweaks, parameter labels in legend, 1σ/2σ bands, interactive view.
- **Section 3 — Corner plot**: Colors/contours, fonts/labels, custom LaTeX labels, quantile lines, axis range overrides, tick density.
- **Section 4 — Radar plot**: Full config, label replacement, annotations. Note on `fontisze_ticks` source typo.
- **Section 5 — Chains plot**: Full config, y-label replacement.
- **Section 6 — Saving figures**: `savefig` with PDF vs PNG guidance, journal submission advice.
- **Section 7 — Next steps**: Tutorial 8 + API docs.

---

## Tutorial 8 — Statistical Tests and Model Selection

**File:** `docs/tutorials/plotting/tutorial_statistical_tests.ipynb`

### Sections

- **Section 0 — Setup**: ForMoSA version check, numpy import. Intro note: why autocorrelation is wrong for nested sampling (NS samples are not a Markov chain — explained for newbies).
- **Section 1 — Loading results**: Single loading block pointing to `results/NS_results/results_pymultinest.json`.
- **Section 2 — NS convergence**: logZ + uncertainty, ESS formula explained step by step, re-run advice.
- **Section 3 — Goodness of fit**: χ²_red formula and code, per-observation breakdown, interpretation table, CCF-likelihood caveat.
- **Section 4 — Information criteria**: AIC/BIC formulas, plain-English explanation of each penalty, ΔAIC table.
- **Section 5 — Bayesian model comparison**: logB from logZ, Jeffreys/Kass-Raftery interpretation table with citations, prior-dependence caveat.
- **Section 6 — Practical use cases**: Fixed vs free `logg` walkthrough; different atmospheric grids. Clear interpretation guidance at each step.
- **Section 7 — Reporting checklist**: "When writing a paper, report all four: logZ, logZ_err, χ²_red, AIC or BIC."

---

## Cross-cutting rules

- All `'PATH/TO/...'` placeholders → `Path(".").resolve() / "results" / "NS_results" / "results_pymultinest.json"` with `FileNotFoundError` guard.
- Every non-trivial code cell has a preceding markdown explanation written for a newbie: what the quantity measures, how to read the output, what to do if the number looks wrong.
- Section headings: `## Section N: Title`.
- No bare `import` blocks without explanation of what the imported objects do.
- Use the `results/NS_results/results_pymultinest.json` path for the main demo; use `results/NS_results/results_nestle.json` as a note for nestle users.
