#
# NuGridpy - Tools for accessing and visualising NuGrid data.
#
# Copyright 2007 - 2014 by the NuGrid Team.
# All rights reserved. See LICENSE.
#

"""
nu_plots.sensitivity

Factor-based reaction-rate sensitivity analysis: given a baseline PPN
run and a set of runs with one reaction's rate multiplied by a factor
(the layout `tools/ppn_runner.py` produces, and the one real
sensitivity-study data this was developed against uses --
`<runs_root>/<reaction_name>/fact_<factor>/`), compute how much a
target isotope's abundance changes per factor, tabulate that across
many reactions and isotopes (an Iliadis et al. 2002 Table 8-style
summary), and rank reactions by how much they matter for a given
isotope.

**Important limitation**: all ratios here are computed from *raw,
undecayed* `iso_massf*.DAT` abundances. Published sensitivity
comparisons (e.g. Iliadis et al. 2002) use *decayed* abundances (short
-lived species decayed to their stable daughters before comparing) --
that decay step is a separate post-processing pass this module does
not perform. Treat these ratios as "how did the raw network output
change," not as directly comparable to a decayed-abundance reference
table; expect the same sign and rough magnitude, not exact agreement.

Deliberately not included: Monte Carlo sampling of rate-uncertainty
factors (the natural next step once this deterministic factor-scan
version is in use -- needs a decay solver and a sampling/statistics
layer, a bigger follow-up).
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import re
from glob import glob

from nugridpy import ppn


_FACTOR_DIR_RE = re.compile(r'^fact_([0-9.eE+-]+)$')

# isotope name, either "Be-7" (nu_plots.plotting._isotope_name's own
# output format) or "7Be" (the format used in reaction_plan.json-style
# configs, and in Iliadis-style tables generally: mass number first).
_ISO_NAME_HYPHEN_RE = re.compile(r'^([A-Za-z]+)-(\d+)$')
_ISO_NAME_MASSFIRST_RE = re.compile(r'^(\d+)([A-Za-z]+)$')


def parse_factor_dirname(name):
    '''
    "fact_10" -> 10.0, "fact_0.01" -> 0.01, "baseline" -> 1.0.

    Returns None if `name` doesn't match either pattern (callers use
    this to filter a directory listing down to factor runs).
    '''
    if name == 'baseline':
        return 1.0
    m = _FACTOR_DIR_RE.match(name)
    if not m:
        return None
    return float(m.group(1))


def discover_factored_runs(reaction_dir):
    '''
    Given a directory such as `runs_VITAL/23Na_pg_24Mg/`, return
    `{factor: path}` for every `fact_*` subdirectory found (skipping
    anything else -- e.g. an `NPDATA` symlink sits alongside these in
    real run trees).
    '''
    runs = {}
    for entry in sorted(glob(os.path.join(reaction_dir, 'fact_*'))):
        if not os.path.isdir(entry):
            continue
        factor = parse_factor_dirname(os.path.basename(entry))
        if factor is not None:
            runs[factor] = entry
    return runs


def _parse_isotope_name(name, elements_names):
    '''(z, a) for an isotope name in either "Be-7" or "7Be" form.'''
    m = _ISO_NAME_HYPHEN_RE.match(name)
    if m:
        symbol, a = m.group(1), int(m.group(2))
    else:
        m = _ISO_NAME_MASSFIRST_RE.match(name)
        if not m:
            raise ValueError("Cannot parse isotope name {!r}".format(name))
        a, symbol = int(m.group(1)), m.group(2)
    symbol_upper = symbol.upper()
    for z, sym in enumerate(elements_names):
        if sym and sym.upper() == symbol_upper:
            return z, a
    raise ValueError("Unknown element symbol {!r} in isotope {!r}".format(symbol, name))


def _as_abu_vector(run_dir_or_instance):
    if isinstance(run_dir_or_instance, str):
        return ppn.abu_vector(run_dir_or_instance)
    return run_dir_or_instance


def isotope_abundance(run_dir, isotope, cycle=None):
    '''
    Raw mass fraction of `isotope` (e.g. 'Mg-24' or '24Mg') in a PPN
    run directory, at `cycle` (default: the last available cycle).

    Parameters
    ----------
    run_dir : string or ppn.abu_vector
        A run directory path, or an already-loaded abu_vector (pass an
        instance when calling this repeatedly against the same run --
        e.g. the baseline -- to avoid re-parsing its output files).
    isotope : string
    cycle : integer, optional

    Returns
    -------
    float. 0.0 if the isotope isn't present in the network at all.
    '''
    p = _as_abu_vector(run_dir)
    if cycle is None:
        cycle = max(p.get('mod'))
    z, a = _parse_isotope_name(isotope, p.elements_names)
    grid = p._abu_grid(cycle)
    return grid.get((a - z, z), 0.0)


def reaction_sensitivity(baseline_dir, reaction_dir, isotope, cycle=None):
    '''
    For one reaction, the abundance ratio (factored/baseline) of
    `isotope` across every factor tested for that reaction.

    Parameters
    ----------
    baseline_dir : string or ppn.abu_vector
    reaction_dir : string
        A directory such as `runs_VITAL/23Na_pg_24Mg/`, containing
        `fact_*` subdirectories (see `discover_factored_runs`).
    isotope : string
    cycle : integer, optional

    Returns
    -------
    dict {factor: ratio}, sorted by factor.
    '''
    baseline = _as_abu_vector(baseline_dir)
    baseline_abund = isotope_abundance(baseline, isotope, cycle=cycle)
    runs = discover_factored_runs(reaction_dir)
    result = {}
    for factor, path in sorted(runs.items()):
        factored_abund = isotope_abundance(path, isotope, cycle=cycle)
        if baseline_abund == 0:
            result[factor] = float('inf') if factored_abund > 0 else float('nan')
        else:
            result[factor] = factored_abund / baseline_abund
    return result


def plot_reaction_sensitivity(baseline_dir, reaction_dir, isotopes, cycle=None,
                               reaction_name=None, show=True, savefig=False, path=None):
    '''
    Log-log scatter/line plot of abundance ratio (factored/baseline)
    vs. rate factor, one series per isotope, for a single reaction --
    the standard "how sensitive is isotope X to this reaction's rate"
    plot.

    Parameters
    ----------
    baseline_dir : string or ppn.abu_vector
    reaction_dir : string
    isotopes : list of string
    cycle : integer, optional
    reaction_name : string, optional
        Used in the title/filename; defaults to `basename(reaction_dir)`.
    show, savefig, path : as in the nu_plots chart methods.
    '''
    import matplotlib.pylab as pl

    baseline = _as_abu_vector(baseline_dir)
    if reaction_name is None:
        reaction_name = os.path.basename(os.path.normpath(reaction_dir))

    fig = pl.figure()
    ax = pl.axes([0.15, 0.12, 0.8, 0.78])
    for isotope in isotopes:
        sens = reaction_sensitivity(baseline, reaction_dir, isotope, cycle=cycle)
        factors = sorted(sens)
        ratios = [sens[f] for f in factors]
        ax.plot(factors, ratios, marker='o', linestyle='-', label=isotope)
    ax.axhline(1.0, color='k', linewidth=0.5, linestyle=':')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('rate factor')
    ax.set_ylabel('X / X$_{baseline}$')
    ax.set_title('Sensitivity: ' + reaction_name)
    ax.legend(loc='best', fontsize='small')

    if savefig:
        graphname = 'sensitivity-' + reaction_name
        if path is not None:
            graphname = os.path.join(path, graphname)
        fig.savefig(graphname)
        print(graphname, 'is done')
    if show:
        pl.show()


def sensitivity_table(baseline_dir, runs_root, reactions, cycle=None):
    '''
    Build an Iliadis-et-al.-2002-Table-8-style sensitivity table
    across many reactions and isotopes.

    Parameters
    ----------
    baseline_dir : string or ppn.abu_vector
    runs_root : string
        Directory containing one subdirectory per reaction name (e.g.
        `runs_VITAL/`), each holding `fact_*` run directories.
    reactions : list of dict
        Each entry needs a `'name'` key (matching a subdirectory of
        `runs_root`) and either an `'affected_isotopes'` or
        `'isotopes'` key (list of isotope name strings) -- accepts a
        `reaction_plan.json`-style `"reactions"` list directly.
    cycle : integer, optional

    Returns
    -------
    list of dict, one per (reaction, isotope) pair:
    `{'reaction': name, 'isotope': iso, <factor>: ratio, ...}` --
    long format, matching a real study's `results.csv` shape, so it
    can be written straight out with `csv.DictWriter` or handed to a
    dataframe library if the caller has one.
    '''
    baseline = _as_abu_vector(baseline_dir)
    rows = []
    for reaction in reactions:
        name = reaction['name']
        isotopes = reaction.get('affected_isotopes', reaction.get('isotopes'))
        reaction_dir = os.path.join(runs_root, name)
        if not os.path.isdir(reaction_dir):
            continue
        for isotope in isotopes:
            row = {'reaction': name, 'isotope': isotope}
            row.update(reaction_sensitivity(baseline, reaction_dir, isotope, cycle=cycle))
            rows.append(row)
    return rows


def rank_reactions(table, isotope, metric='max_abs_log_ratio'):
    '''
    Rank the reactions in a `sensitivity_table` result by how much
    they affect `isotope`.

    Parameters
    ----------
    table : list of dict, as returned by `sensitivity_table`.
    isotope : string
        Only rows for this isotope are considered.
    metric : string, optional
        'max_abs_log_ratio' (default): rank by the largest
        `|log10(ratio)|` seen across all tested factors for that
        reaction -- the biggest swing away from the baseline,
        regardless of which factor caused it or which direction.

    Returns
    -------
    list of (reaction_name, metric_value), sorted most-sensitive
    first.
    '''
    if metric != 'max_abs_log_ratio':
        raise ValueError("Unsupported metric: {!r}".format(metric))

    from math import log10

    scores = []
    for row in table:
        if row['isotope'] != isotope:
            continue
        ratios = [v for k, v in row.items() if k not in ('reaction', 'isotope')
                  and v is not None and v > 0]
        if not ratios:
            continue
        score = max(abs(log10(r)) for r in ratios)
        scores.append((row['reaction'], score))
    scores.sort(key=lambda item: item[1], reverse=True)
    return scores
