#
# NuGridpy - Tools for accessing and visualising NuGrid data.
#
# Copyright 2007 - 2014 by the NuGrid Team.
# All rights reserved. See LICENSE.
#

"""
nu_plots.network

Parses NuPPN's `networksetup.txt` (the reaction-network table a ppn/
mppnp run writes to its run directory), `ppn_physics.input` (the
isotope/reaction on-off input a run reads), and `isotopedatabase.txt`
into structured records, to answer:

- what reactions are in the network as a whole
- what reactions structurally affect a given isotope
- what a run's actual input configuration turned on/off

This is data introspection, not plotting -- it lives in `nu_plots`
because it's still nucleosynthesis-network-specific, parallel to how
`mesa_plots` holds MESA-specific code, not because it draws figures.
Higher-level consistency checks/diffs built on top of these parsers
(is this configuration internally consistent, did a run's output
collapse, do two configs actually differ the way they're supposed to)
live in the separate `nu_plots.network_audit` module.

Deliberately not included: correlating `flux_NNNNN.DAT` rows (a
cycle's reaction fluxes, already read by
`nu_plots.plotting.abu_flux_chart`) back to specific named reactions
from this table. Row count and a spot-check against real production
data (`networksetup.txt` vs. `flux_00020.DAT` from an actual run)
don't line up cleanly enough to resolve with confidence -- the risk of
silently misattributing a flux value to the wrong reaction is worse
than not offering the feature. Revisit if/when a validated
ground-truth example (or the PPN Fortran source that writes both
files) is available to confirm the correspondence.
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import re
from collections import namedtuple


Species = namedtuple('Species', ['count', 'z', 'a', 'name'])
Reaction = namedtuple('Reaction', [
    'index', 'active', 'reactant', 'projectile', 'product1', 'product2',
    'source', 'reaction_type',
])

# Matches one reaction-table line, e.g.:
#       1 T  2  PROT   +  0  OOOOO  ->  1  H   2  +  0  OOOOO   4.344E-21  VITAL  (p,g)   5   1.000E+00   2.146E+18
_REACTION_RE = re.compile(
    r"^\s*(\d+)\s+([TF])\s+(\d+)\s+(.{5})\s+\+\s+(\d+)\s+(.{5})"
    r"\s+->\s+(\d+)\s+(.{5})\s+\+\s+(\d+)\s+(.{5})\s+\S+\s+(\S+)\s+(\S+)\s+\d+"
)

PhysicsInputIsotope = namedtuple('PhysicsInputIsotope', [
    'index', 'name', 'a', 'z', 'active', 'halflife', 'halflife_unit',
])
PhysicsInputReaction = namedtuple('PhysicsInputReaction', [
    'index', 'active', 'reactant', 'projectile', 'product1', 'product2',
    'q_value', 'source', 'reaction_type',
])

# Matches one ppn_physics.input isotope-table line, e.g.:
#       6 BE  7   7.  4. T  5.329d+01 d
# Species name captured non-greedily up to the "A. Z." numeric pair --
# validated against every isotope row in 4 real ppn_physics.input
# files (no false matches against reaction-table rows, which don't
# have this A./Z./flag shape at all).
_PHYSICS_INPUT_ISOTOPE_RE = re.compile(
    r"^\s*(\d+)\s+(.+?)\s+(-?\d+)\.\s+(-?\d+)\.\s+([TF])\s+(\S+)\s+(\S+)\s*$"
)

# Matches one ppn_physics.input reaction-table line, e.g.:
#       1 T  2 PROT    0 OOOOO  0 OOOOO   1 H   2     1.179   VITAL  (p,g)   5   1.000E+00
# Same count+5-char-species-name shape as networksetup.txt's
# _REACTION_RE, but this table has no literal '+'/'->' separators and
# puts the T/F flag right after the index instead of at the end.
_PHYSICS_INPUT_REACTION_RE = re.compile(
    r"^\s*(\d+)\s+([TF])\s+(\d+)\s+(.{5})\s+(\d+)\s+(.{5})"
    r"\s+(\d+)\s+(.{5})\s+(\d+)\s+(.{5})\s+(\S+)\s+(\S+)\s+(\S+)\s+\d+\s+\S+\s*$"
)

DatabaseIsotope = namedtuple('DatabaseIsotope', ['z', 'a', 'name', 'stable_a', 'active'])

_SPECIAL_SPECIES = {
    'PROT': (1, 1),
    'NEUT': (0, 1),
    'OOOOO': (0, 0),   # no second reactant/product (e.g. a decay, or (x,g))
}

_NAME_RE = re.compile(r'^([A-Za-z]+)\s*(\d+)$')


def _symbol_to_z(elements_names):
    '''{symbol.upper(): Z} from an elements_names list (index = Z).'''
    return {sym.upper(): z for z, sym in enumerate(elements_names) if sym}


def _parse_species(field, count, symbol_to_z):
    text = field.strip()
    if text in _SPECIAL_SPECIES:
        z, a = _SPECIAL_SPECIES[text]
        return Species(count, z, a, text)
    match = _NAME_RE.match(text)
    if not match:
        return Species(count, None, None, text)
    symbol, mass = match.group(1).upper(), int(match.group(2))
    return Species(count, symbol_to_z.get(symbol), mass, text)


def parse_networksetup(path, elements_names):
    '''
    Parse a networksetup.txt file's reaction table into a list of
    Reaction records.

    Parameters
    ----------
    path : string
        Path to a networksetup.txt file.
    elements_names : list
        The same isotope-database element-symbol list an
        abu_vector/se instance exposes (`self.elements_names`, index =
        atomic number Z), used to resolve species names in the file to
        Z. A species whose symbol isn't found there gets `z=None`
        (still returned, just not resolvable to an atomic number).

    Returns
    -------
    list of Reaction. A physical reaction can appear more than once --
    an active ('T') row plus one or more superseded 'F' rows, when a
    newer rate evaluation replaces an older one -- both are returned
    here; use `active_only=True` in `reactions_in_network`/
    `reactions_for_isotope` to keep only the preferred entries.
    '''
    symbol_to_z = _symbol_to_z(elements_names)
    reactions = []
    with open(path, 'r') as f:
        for line in f:
            m = _REACTION_RE.match(line)
            if not m:
                continue
            (index, active, rcount, rname, pcount, pname,
             p1count, p1name, p2count, p2name, source, rtype) = m.groups()
            reactions.append(Reaction(
                index=int(index),
                active=(active == 'T'),
                reactant=_parse_species(rname, int(rcount), symbol_to_z),
                projectile=_parse_species(pname, int(pcount), symbol_to_z),
                product1=_parse_species(p1name, int(p1count), symbol_to_z),
                product2=_parse_species(p2name, int(p2count), symbol_to_z),
                source=source,
                reaction_type=rtype,
            ))
    return reactions


def parse_ppn_physics_input(path, elements_names):
    '''
    Parse a ppn_physics.input file's isotope and reaction tables.

    Unlike networksetup.txt (a compiled *output* snapshot of the
    network), this is the actual *input* a ppn/mppnp run reads: each
    isotope and each reaction carries its own independent T/F
    "considered" flag here, and NuPPN's own source
    (`physics/source/vital.F90`) only checks -- and only warns,
    non-fatally -- that an active reaction's isotopes are also active;
    it doesn't otherwise validate this file.

    Isotope and reaction rows are told apart by shape, not by file
    position (an isotope row's second field is a name; a reaction
    row's second field is the T/F flag itself) -- validated against
    every row of 4 real ppn_physics.input files with zero ambiguous or
    unmatched non-comment/non-namelist lines.

    Parameters
    ----------
    path : string
        Path to a ppn_physics.input file.
    elements_names : list
        As in `parse_networksetup` -- used to resolve reaction-table
        species names to Z.

    Returns
    -------
    (isotopes, reactions) : (list of PhysicsInputIsotope, list of PhysicsInputReaction)
    '''
    symbol_to_z = _symbol_to_z(elements_names)
    isotopes = []
    reactions = []
    with open(path, 'r') as f:
        for line in f:
            m = _PHYSICS_INPUT_REACTION_RE.match(line)
            if m:
                (index, active, rcount, rname, pcount, pname,
                 p1count, p1name, p2count, p2name,
                 q_value, source, rtype) = m.groups()
                reactions.append(PhysicsInputReaction(
                    index=int(index),
                    active=(active == 'T'),
                    reactant=_parse_species(rname, int(rcount), symbol_to_z),
                    projectile=_parse_species(pname, int(pcount), symbol_to_z),
                    product1=_parse_species(p1name, int(p1count), symbol_to_z),
                    product2=_parse_species(p2name, int(p2count), symbol_to_z),
                    q_value=float(q_value),
                    source=source,
                    reaction_type=rtype,
                ))
                continue
            m = _PHYSICS_INPUT_ISOTOPE_RE.match(line)
            if m:
                index, name, a, z, active, halflife, halflife_unit = m.groups()
                isotopes.append(PhysicsInputIsotope(
                    index=int(index),
                    name=name.strip(),
                    a=int(a),
                    z=int(z),
                    active=(active == 'T'),
                    # Kept as the raw Fortran-formatted string (e.g.
                    # '5.329d+01', 'd' exponent, not 'e') rather than a
                    # float -- nothing in this module needs it
                    # numerically yet.
                    halflife=halflife,
                    halflife_unit=halflife_unit,
                ))
    return isotopes, reactions


def parse_isotopedatabase(path):
    '''
    Parse an isotopedatabase.txt file.

    Simple whitespace-delimited format: Z, A, element symbol, the
    stable reference mass number for that element (not this row's own
    A), and a T/F "considered" flag. Deliberately a standalone parser
    here rather than reusing `utils.Utils._read_isotopedatabase`, which
    is private, NumPy-array-coupled, and built for a different purpose
    (decay-index bookkeeping).

    Parameters
    ----------
    path : string
        Path to an isotopedatabase.txt file.

    Returns
    -------
    list of DatabaseIsotope.
    '''
    isotopes = []
    with open(path, 'r') as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            z, a, name, stable_a, active = parts
            isotopes.append(DatabaseIsotope(
                z=int(z), a=int(a), name=name, stable_a=int(stable_a),
                active=(active == 'T'),
            ))
    return isotopes


def reactions_in_network(reactions, active_only=True):
    '''
    All reactions in the network.

    Parameters
    ----------
    reactions : list of Reaction, as returned by `parse_networksetup`.
    active_only : boolean, optional
        If True (default), only the active ('T') entries -- i.e. the
        preferred rate for each physical reaction. If False, every
        row including superseded duplicates.
    '''
    if active_only:
        return [r for r in reactions if r.active]
    return list(reactions)


def reactions_for_isotope(reactions, z, a, active_only=True):
    '''
    Reactions in which the isotope (z, a) participates, as reactant,
    projectile, product1, or product2.

    The criterion is plain structural participation: every reaction
    that physically involves this isotope on either side, not a flux
    or significance threshold -- there's no cycle-dependent "how much
    does this reaction matter" ranking here (see the module docstring
    for why that's deliberately not implemented yet).

    Parameters
    ----------
    reactions : list of Reaction, as returned by `parse_networksetup`.
    z, a : integers
        Atomic number and mass number of the isotope to look up.
    active_only : boolean, optional
        As in `reactions_in_network`. The default is True.
    '''
    pool = reactions_in_network(reactions, active_only=active_only)
    return [r for r in pool
            if any(s.z == z and s.a == a
                   for s in (r.reactant, r.projectile, r.product1, r.product2))]
