# NuGrid-Tools

NuGrid-Tools is a modernization fork of NuGridPy for nuclear astrophysics
analysis. The goal is to preserve the scientific value and familiar workflows of
NuGridPy while making the package easier to install, easier to use, compatible
with modern Python, and capable of producing publication-quality plots by
default.

This fork starts from the existing NuGridPy codebase and will evolve toward a
maintainable Python 3 package with stronger tests, clearer APIs, and modern
visualization defaults.

## Goals

- Support current Python 3 versions and keep compatibility work visible as new
  Python versions are released.
- Modernize packaging with `pyproject.toml`, standard dependency metadata, and
  reliable installation workflows.
- Replace legacy Python 2 compatibility layers and deprecated scientific Python
  APIs.
- Preserve existing NuGridPy user workflows where practical, especially common
  notebook and interactive-analysis patterns.
- Improve the plotting system so nuclear astrophysics figures are clear,
  beautiful, colorblind-aware, and suitable for papers, talks, and reports.
- Add tests around core scientific utilities, data readers, and plotting
  behavior so future refactors are safer.
- Improve documentation with modern examples for MESA, NuGrid `se` files,
  abundance profiles, Kippenhahn diagrams, HR diagrams, isotope charts, and
  publication exports.

## Compatibility Direction

NuGrid-Tools is expected to target actively maintained Python versions. The exact
support window will be defined in project metadata and CI, but the intended
policy is:

- Test against multiple supported Python 3 releases.
- Avoid deprecated NumPy, SciPy, Matplotlib, and h5py APIs where possible.
- Prefer explicit compatibility shims over hidden version-specific behavior.
- Keep import-time side effects minimal so the package remains stable in
  scripts, notebooks, batch jobs, and documentation builds.

## Development Setup

NuGrid-Tools uses `pyproject.toml` for package metadata and dependency groups.
Use a virtual environment for local development, then install the package in
editable mode with the development and documentation extras:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-dev.txt
```

Run the test suite with:

```bash
python3 -m pytest
```

Runtime-only installs can use:

```bash
python3 -m pip install -r requirements.txt
```

## Plotting Direction

Plotting modernization is a first-class goal, not an afterthought. NuGrid-Tools
will add a coherent plotting layer with:

- Publication-oriented Matplotlib styles.
- Consistent figure sizing and export helpers.
- Colorblind-safe palettes and scientifically meaningful colormaps.
- Cleaner defaults for labels, legends, tick marks, line widths, and DPI.
- APIs that return `fig` and `ax` objects so users can customize figures
  without fighting global plotting state.
- Backward-compatible wrappers for established plotting methods where feasible.

## Migration Principles

- Modernize in small, reviewable steps.
- Keep scientific behavior stable unless a change is intentional and documented.
- Prefer additive APIs before removing legacy behavior.
- Mark breaking changes clearly.
- Maintain examples that can be run by users and by CI.

## Implementation Plan

### Phase 1 — Spring Cleaning

`nugridpy/data_plot.py` is currently a single ~5,300-line mixin class
(`DataPlot`) inherited by MESA, PPN, and NuGrid `se` classes alike, so
stellar-evolution classes end up carrying nucleosynthesis-only methods and
vice versa. Before adding features, split it by concern:

- `nu_plots/` — plotting for `nuppn`/nucleosynthesis-network output (abundance
  charts, flux charts, isotope evolution), with its own `io.py`, `utils.py`,
  and `plotting.py`.
- `mesa_plots/` — plotting for MESA stellar-evolution output (Kippenhahn, HR
  diagrams, profiles), with its own `io.py`, `utils.py`, and `plotting.py`.
- A small shared module for anything genuinely used by both (colour palettes,
  figure/style helpers) so it isn't duplicated.

This phase is pure extraction — no behavior changes — checked against the
existing regression tests (`nugridpy/regression_tests/`) before Phase 2
begins.

### Phase 2 — Core Nucleosynthesis Plotting

Most of this functionality already exists in `data_plot.py`/`ppn.py`
(`abu_chart`, `abu_flux_chart`/`flux_solo`, `iso_abund`, `plot_xtime`), so
this phase is extract-and-polish, not a rebuild from scratch. New
functionality is called out explicitly:

- **Abundance chart** (extracted/cleaned `abu_chart`): plot by cycle number
  from `abu_vector`/`iso_massf`; line of stability highlighted in bold
  blocks; single-colour mode; abundance threshold cutoff.
- **Flux chart**: same treatment as the abundance chart, for flux files.
- **Abundance distribution** (X vs. A): extracted/polished `iso_abund`.
- **Abundance evolution** (X vs. time): extracted/polished `plot_xtime`,
  reading from `x-time.dat`.
- **Ratio abundance chart** *(new)*: compare two `iso_massf` files, with a
  residual-difference option and a diverging/polarizing colour scheme.
- **Created / destroyed / enhanced / depleted isotope tracker** *(new)*:
  classify isotopes by how their abundance changed over the evolution of the
  explosion.

### Phase 3 — Reaction Network Introspection

Needs new data plumbing (a network/rate-file parser), separate from the
plotting extraction above:

- Reactions active in the network at a given cycle.
- Reactions present in the network as a whole.
- Reactions affecting a given isotope, given an explicit, documented
  criterion (e.g., contributes more than some threshold to
  production/destruction flux) — the criterion needs to be defined before
  implementation starts.

### Phase 4 — Rate Uncertainty & Sensitivity Analysis

Scope as its own module rather than folding into `nu_plots/` — it has
different data-management needs (many repeated PPN runs and per-reaction
multiplier bookkeeping, not just reading a single output file):

- **Factor-based sensitivity**: from a set of factored runs, extract a target
  isotope's abundance over a chosen range and plot the abundance ratio
  (factored/unfactored) against the factor applied — one scatter plot per
  reaction.
- **Multi-reaction/multi-isotope tables**: reproduce Iliadis et al. (2002)
  Table 6–10-style summaries across many factored reactions and abundance
  ratios.
- **Most-uncertain-reactions ranking**: needs an explicit criterion (e.g.,
  rank by change in output abundance per unit change in rate factor) —
  define before implementing.
- **Temperature dependence of sensitivity**: rerun the sensitivity analysis
  at multiple representative cycles/temperatures rather than assuming one
  global factor per reaction — rate uncertainties are known to be
  temperature-dependent, which is why evaluations such as STARLIB/REACLIB
  report temperature-binned lognormal uncertainty factors.
- **Monte Carlo upgrade path**: replace one-factor-at-a-time scans with Monte
  Carlo sampling of lognormal rate-uncertainty factors per reaction
  (Longland/Iliadis-style). This yields the full output distribution per
  isotope, correlations between isotopes, and lets reactions be ranked by
  rank correlation (e.g., Spearman) between sampled rate and output
  abundance — more informative, and more efficient than a full factorial
  grid, once more than a handful of reactions are varied at once.

## Relationship To NuGridPy

NuGrid-Tools is a fork intended for modernization work. It should credit and
preserve the original NuGridPy project history while providing a place to make
larger Python 3 and plotting improvements without destabilizing legacy users.
