#
# NuGridpy - Tools for accessing and visualising NuGrid data.
#
# Copyright 2007 - 2014 by the NuGrid Team.
# All rights reserved. See LICENSE.
#

"""
nu_plots.network_audit

Higher-level consistency checks and diffs built on top of
`nu_plots.network`'s parsers -- answers "can I trust this network
configuration/run" rather than "what reactions are in it" (that's
`network.py`'s job; this module consumes its parsed output).

This exists because none of these checks exist anywhere else. NuPPN's
own Fortran source (`physics/source/vital.F90`) performs exactly one
of them internally (an active reaction referencing an inactive
isotope) -- but only as a non-fatal printed warning, easy to miss in
run output, with no equivalent pre-flight check available before
spending compute on a run. Nothing anywhere diffs two
`ppn_physics.input`/`networksetup.txt` snapshots, cross-checks
`isotopedatabase.txt` coverage, flags a silently-collapsed network
output, or fingerprints a run's actual input files for provenance.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import hashlib
import os

from nugridpy import ppn


def _as_abu_vector(run_dir_or_instance):
    if isinstance(run_dir_or_instance, str):
        return ppn.abu_vector(run_dir_or_instance)
    return run_dir_or_instance


def diff_physics_input(isotopes_a, reactions_a, isotopes_b, reactions_b):
    '''
    Compare two parsed ppn_physics.input results (as returned by
    `network.parse_ppn_physics_input`) and report which isotope/
    reaction indices changed their active (T/F) flag.

    Parameters
    ----------
    isotopes_a, reactions_a : the first config's parsed isotopes/reactions.
    isotopes_b, reactions_b : the second config's parsed isotopes/reactions.

    Returns
    -------
    dict with keys 'isotopes_turned_off', 'isotopes_turned_on',
    'reactions_turned_off', 'reactions_turned_on' -- each a sorted list
    of indices present in both configs whose active flag differs.
    '''
    def _flag_diff(a, b):
        a_by_index = {x.index: x.active for x in a}
        b_by_index = {x.index: x.active for x in b}
        turned_off = sorted(i for i in a_by_index
                             if i in b_by_index and a_by_index[i] and not b_by_index[i])
        turned_on = sorted(i for i in a_by_index
                            if i in b_by_index and not a_by_index[i] and b_by_index[i])
        return turned_off, turned_on

    iso_off, iso_on = _flag_diff(isotopes_a, isotopes_b)
    reac_off, reac_on = _flag_diff(reactions_a, reactions_b)
    return {
        'isotopes_turned_off': iso_off,
        'isotopes_turned_on': iso_on,
        'reactions_turned_off': reac_off,
        'reactions_turned_on': reac_on,
    }


def check_physics_input_consistency(isotopes, reactions):
    '''
    Flag any active reaction that references an inactive isotope.

    A structured, pre-flight reimplementation of the one check
    `vital.F90`'s `read_physics_input_data` performs internally -- but
    there it's a non-fatal printed warning; here it's a plain list you
    can check before spending compute on a run.

    Parameters
    ----------
    isotopes : list of network.PhysicsInputIsotope
    reactions : list of network.PhysicsInputReaction

    Returns
    -------
    list of dict, one per problem found:
    `{'reaction_index': int, 'isotope_name': str}`.
    '''
    active_isotope_names = {iso.name for iso in isotopes if iso.active}
    problems = []
    for r in reactions:
        if not r.active:
            continue
        for species in (r.reactant, r.projectile, r.product1, r.product2):
            name = species.name.strip()
            if name == 'OOOOO':
                continue
            if name not in active_isotope_names:
                problems.append({
                    'reaction_index': r.index,
                    'isotope_name': name,
                })
    return problems


def diff_networksetup(reactions_a, reactions_b):
    '''
    Compare two `network.parse_networksetup()` results by index --
    verifies the *compiled* network nuppn actually built matches what
    the input config implies, rather than what you configured.

    Returns
    -------
    dict with keys 'added' (indices only in b), 'removed' (indices
    only in a), 'activated' (index active in b but not a), and
    'deactivated' (index active in a but not b) -- each a sorted list
    of indices.
    '''
    a_by_index = {r.index: r for r in reactions_a}
    b_by_index = {r.index: r for r in reactions_b}
    added = sorted(set(b_by_index) - set(a_by_index))
    removed = sorted(set(a_by_index) - set(b_by_index))
    activated = sorted(i for i in a_by_index if i in b_by_index
                        and not a_by_index[i].active and b_by_index[i].active)
    deactivated = sorted(i for i in a_by_index if i in b_by_index
                          and a_by_index[i].active and not b_by_index[i].active)
    return {
        'added': added,
        'removed': removed,
        'activated': activated,
        'deactivated': deactivated,
    }


def check_isotope_coverage(reactions, database_isotopes):
    '''
    Cross-check isotopes referenced by active reactions against an
    isotope database.

    Parameters
    ----------
    reactions : list of network.Reaction or network.PhysicsInputReaction
        Only active (`.active is True`) reactions are considered.
    database_isotopes : list of network.DatabaseIsotope,
        as returned by `network.parse_isotopedatabase`.

    Returns
    -------
    dict with keys 'referenced_but_missing' (set of (z, a) referenced
    by an active reaction but absent from the database) and
    'unreferenced_in_database' (set of (z, a) present and active in
    the database but never referenced by any active reaction).
    '''
    referenced = set()
    for r in reactions:
        if not r.active:
            continue
        for species in (r.reactant, r.projectile, r.product1, r.product2):
            name = species.name.strip()
            if name == 'OOOOO' or species.z is None or species.a is None:
                continue
            referenced.add((species.z, species.a))
    database = {(iso.z, iso.a) for iso in database_isotopes if iso.active}
    return {
        'referenced_but_missing': referenced - database,
        'unreferenced_in_database': database - referenced,
    }


def check_network_collapse(run_dir_or_abu_vector, cycle, expected_isotopes,
                            floor=1e-99, floor_fraction_threshold=0.9):
    '''
    Flag signs that a run's output network has silently collapsed --
    the failure mode `vital.F90`'s own header comment warns about
    (disabling VITAL species/reactions leaves "active species collapsed
    by >90%, effectively zero reaction flow" with no fatal error).

    Parameters
    ----------
    run_dir_or_abu_vector : string or ppn.abu_vector
        A run directory path, or an already-loaded abu_vector.
    cycle : integer
    expected_isotopes : list of network.PhysicsInputIsotope
        The isotopes expected to be present and tracked. This should
        normally be a *known-good reference* run's active isotope list
        (e.g. a trusted baseline's `parse_ppn_physics_input` isotopes)
        -- not necessarily this same run's own `ppn_physics.input`,
        since a misconfigured run's own file may itself claim nothing
        is active (exactly the case that silently collapsed
        `test_vital_iso_reac_off` -- checking its output against its
        *own* config would trivially find nothing "missing").
    floor : float, optional
        The numerical underflow floor mass fractions get pinned to.
        The default is 1e-99.
    floor_fraction_threshold : float, optional
        If the fraction of isotopes actually present in the output at
        `floor` meets or exceeds this, it's flagged. The default is
        0.9.

    Returns
    -------
    dict, empty if no problems found. Possible keys:
    'missing_isotopes' (names expected active but absent from the
    output at this cycle) and 'floor_fraction' (the actual fraction
    found at/below `floor`, only present if it meets the threshold).
    '''
    p = _as_abu_vector(run_dir_or_abu_vector)
    grid = p._abu_grid(cycle)

    expected = [(iso.a - iso.z, iso.z, iso.name)
                for iso in expected_isotopes if iso.active]
    missing = [name for (n, z, name) in expected if (n, z) not in grid]

    present_values = list(grid.values())
    floor_fraction = (
        sum(1 for v in present_values if v <= floor) / len(present_values)
        if present_values else 1.0
    )

    problems = {}
    if missing:
        problems['missing_isotopes'] = missing
    if floor_fraction >= floor_fraction_threshold:
        problems['floor_fraction'] = floor_fraction
    return problems


def fingerprint_files(run_dir, filenames=('ppn_physics.input', 'isotopedatabase.txt')):
    '''
    SHA-256 of each of `filenames` as found in `run_dir`, for
    provenance comparisons -- confirms two runs meant to differ in only
    one input actually did, or catches when they didn't (e.g. a shared
    input file silently left in a stale state between runs).

    Parameters
    ----------
    run_dir : string
    filenames : tuple of string, optional

    Returns
    -------
    dict {filename: hexdigest}. A filename missing from `run_dir` is
    omitted from the result rather than raising.
    '''
    fingerprints = {}
    for name in filenames:
        path = os.path.join(run_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, 'rb') as f:
            fingerprints[name] = hashlib.sha256(f.read()).hexdigest()
    return fingerprints
