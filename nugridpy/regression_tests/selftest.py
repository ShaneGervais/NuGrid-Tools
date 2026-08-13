from __future__ import absolute_import
from builtins import str
from builtins import range
import matplotlib
matplotlib.use('agg')
import unittest
import os
import numpy as np

from .tempdir.tempfile_ import TemporaryDirectory
from nugridpy.nu_plots.plotting import NuPlotMixin

class TestModuleImports(unittest.TestCase):

    def test_import_ascii_table(self):
        import nugridpy.ascii_table

    def test_import_astronomy(self):
        import nugridpy.astronomy

    def test_import_data_plot(self):
        import nugridpy.data_plot

    def test_import_plot_common(self):
        import nugridpy.plot_common

    def test_import_nu_plots(self):
        import nugridpy.nu_plots.plotting

    def test_import_mesa_plots(self):
        import nugridpy.mesa_plots.plotting

    def test_import_grain(self):
        import nugridpy.grain

    def test_import_h5T(self):
        import nugridpy.h5T

    def test_import_mesa(self):
        import nugridpy.mesa

    def test_import_nugridse(self):
        import nugridpy.nugridse

    def test_import_ppn(self):
        import nugridpy.ppn

    def test_import_utils(self):
        import nugridpy.utils

    def test_import_network(self):
        import nugridpy.nu_plots.network

    def test_import_sensitivity(self):
        import nugridpy.nu_plots.sensitivity

class TestPlotMixinComposition(unittest.TestCase):

    def test_data_plot_shim_aliases_plot_common(self):
        from nugridpy import data_plot, plot_common
        self.assertIs(data_plot.DataPlot, plot_common.PlotCommon)

    def test_nu_plots_classes_inherit_nu_plot_mixin(self):
        from nugridpy import ppn, nugridse
        from nugridpy.nu_plots.plotting import NuPlotMixin
        self.assertTrue(issubclass(ppn.xtime, NuPlotMixin))
        self.assertTrue(issubclass(ppn.abu_vector, NuPlotMixin))
        self.assertTrue(issubclass(nugridse.se, NuPlotMixin))

    def test_mesa_classes_inherit_expected_mixins(self):
        from nugridpy import mesa
        from nugridpy.plot_common import PlotCommon
        from nugridpy.mesa_plots.plotting import MesaPlotMixin
        self.assertTrue(issubclass(mesa.mesa_profile, PlotCommon))
        self.assertFalse(issubclass(mesa.mesa_profile, MesaPlotMixin))
        self.assertTrue(issubclass(mesa.history_data, MesaPlotMixin))
        self.assertTrue(issubclass(mesa.star_log, MesaPlotMixin))


class _FakeAbuVector(NuPlotMixin):
    '''
    Synthetic stand-in for `ppn.abu_vector`, for exercising the
    plotting/classification logic in nu_plots.plotting without a real
    ppn run directory or a network fetch. Carries just enough of a
    real instance's surface (`get`, `_classTest`, `elements_names`,
    `stable_el`) for `NuPlotMixin`'s abundance-chart family to work.
    '''

    # index by Z; '' placeholders for elements not used in test data
    ELEMENTS_NAMES = ['', 'H', 'He', '', '', '', 'C', 'N', 'O']
    # stable_el[0] is skipped by _draw_stable_boxes; entries are
    # [symbol, mass_number, mass_number, ...]
    STABLE_EL = [
        [''],
        ['H', 1, 2],
        ['He', 3, 4],
        ['C', 12, 13],
        ['N', 14, 15],
        ['O', 16, 17, 18],
    ]

    def __init__(self, cycles_data):
        '''cycles_data: {cycle_number: {'A':.., 'Z':.., 'ABUNDANCE_MF':.., 'ISOM':..}}'''
        self._cycles_data = cycles_data
        self.elements_names = self.ELEMENTS_NAMES
        self.stable_el = self.STABLE_EL

    def get(self, attri, cycle=None, *a, **k):
        if attri == 'mod':
            return np.array(sorted(self._cycles_data.keys()))
        return self._cycles_data[cycle][attri]

    def _classTest(self):
        return 'PPN'


def _synthetic_cycle(scale=1.0, drop=None):
    A = np.array([1, 4, 12, 13, 14, 16])
    Z = np.array([1, 2, 6, 6, 7, 8])
    MF = np.array([0.7, 0.28, 1e-3, 1e-8, 1e-4, 1e-3]) * scale
    ISOM = np.array([1, 1, 1, 1, 1, 1])
    if drop is not None:
        mask = np.arange(len(A)) != drop
        A, Z, MF, ISOM = A[mask], Z[mask], MF[mask], ISOM[mask]
    return {'A': A, 'Z': Z, 'ABUNDANCE_MF': MF, 'ISOM': ISOM}


class TestNewAbundanceCharts(unittest.TestCase):
    '''
    Covers the Phase 2 additions to nu_plots.plotting: single_colour/
    threshold on abu_chart, the abu_flux_chart/flux_solo merge, and
    the two new chart types (abu_ratio_chart, abu_evolution_classify).
    Uses synthetic in-memory data throughout -- no network fetch, no
    dependency on a real ppn run directory.
    '''

    def setUp(self):
        import matplotlib.pylab as mpy
        self.mpy = mpy

    def test_abu_chart_default_single_colour_threshold_imagic(self):
        p = _FakeAbuVector({0: _synthetic_cycle()})
        self.mpy.figure()
        p.abu_chart(0, show=False)
        self.mpy.figure()
        p.abu_chart(0, show=False, single_colour='steelblue')
        self.mpy.figure()
        p.abu_chart(0, show=False, threshold=-5)
        self.mpy.figure()
        p.abu_chart(0, show=False, imagic=True)

    def test_abu_flux_chart_merge_and_flux_solo_wrapper(self):
        p = _FakeAbuVector({0: _synthetic_cycle()})
        with TemporaryDirectory() as tdir:
            import os
            cwd = os.getcwd()
            os.chdir(tdir)
            try:
                with open('flux_00000.DAT', 'w') as f:
                    f.write("header line\n")
                    f.write("1 1 1 0 0 1 2 1 1 1.0e-3 1.0e-4\n")
                    f.write("2 2 2 0 0 2 3 2 2 5.0e-5 5.0e-6\n")
                self.mpy.figure()
                p.abu_flux_chart(0, show=False)
                self.mpy.figure()
                p.abu_flux_chart(0, show=False, show_abundance=False)
                self.mpy.figure()
                p.flux_solo(0, show=False)
                self.mpy.figure()
                p.abu_flux_chart(0, show=False, threshold=-5, imagic=True)
            finally:
                os.chdir(cwd)

    def test_abu_flux_chart_flux_extent_beyond_abundance_network(self):
        # Regression test: the flux panel's own (N,Z) extent is
        # computed separately from, and can be larger than, the
        # abundance panel's -- a flux row referencing a nuclide outside
        # the loaded isotope network's own N/Z range (e.g. a late
        # cycle's flux file touching nuclides the abundance side never
        # sees) used to crash with an IndexError from sizing the flux
        # array to the smaller, abundance-derived bound.
        p = _FakeAbuVector({0: _synthetic_cycle()})
        with TemporaryDirectory() as tdir:
            import os
            cwd = os.getcwd()
            os.chdir(tdir)
            try:
                with open('flux_00000.DAT', 'w') as f:
                    f.write("header line\n")
                    f.write("1 1 1 0 0 1 2 1 1 1.0e-3 1.0e-4\n")
                    # Z_k5=15, A_k5=30: far outside this synthetic
                    # network's own Z/N range (max Z=8, max N=8).
                    f.write("2 1 1 0 0 15 30 1 1 1.0e-3 1.0e-4\n")
                self.mpy.figure()
                p.abu_flux_chart(0, show=False)
            finally:
                os.chdir(cwd)

    def test_abu_ratio_chart(self):
        p_self = _FakeAbuVector({0: _synthetic_cycle(scale=1.0)})
        p_other = _FakeAbuVector({0: _synthetic_cycle(scale=0.1, drop=3)})
        self.mpy.figure()
        result = p_self.abu_ratio_chart(p_other, 0, show=False)
        # 5 isotopes shared (C-13 dropped from p_other); each scaled by
        # exactly 10x, so log10(ratio) should be 1.0 for all of them.
        self.assertEqual(len(result), 5)
        for val in result.values():
            self.assertAlmostEqual(val, 1.0, places=6)

        self.mpy.figure()
        result_r = p_self.abu_ratio_chart(p_other, 0, show=False, residual=True)
        for val in result_r.values():
            self.assertAlmostEqual(val, 9.0, places=6)

    def test_abu_evolution_classify(self):
        p = _FakeAbuVector({
            0: {
                'A': np.array([1, 4, 12, 16]),
                'Z': np.array([1, 2, 6, 8]),
                'ABUNDANCE_MF': np.array([0.7, 1e-3, 1e-2, 1e-3]),
                'ISOM': np.array([1, 1, 1, 1]),
            },
            38: {
                'A': np.array([1, 4, 14, 16]),
                'Z': np.array([1, 2, 7, 8]),
                'ABUNDANCE_MF': np.array([0.07, 0.2, 1e-2, 1e-3]),
                'ISOM': np.array([1, 1, 1, 1]),
            },
        })
        self.mpy.figure()
        categories = p.abu_evolution_classify(cycle_start=0, cycle_end=38, show=False)
        self.assertEqual(categories['created'], ['N-14'])
        self.assertEqual(categories['destroyed'], ['C-12'])
        self.assertEqual(categories['enhanced'], ['He-4'])

        # default cycle_start/cycle_end resolved from self.get('mod')
        self.mpy.figure()
        categories_default = p.abu_evolution_classify(show=False)
        self.assertEqual(categories_default, categories)

        # plot=False returns the same data without touching matplotlib
        categories_noplot = p.abu_evolution_classify(cycle_start=0, cycle_end=38, plot=False)
        self.assertEqual(categories_noplot['created'], ['N-14'])


class TestNetwork(unittest.TestCase):
    '''
    Covers nu_plots.network's networksetup.txt parsing against a small
    fragment of real production reaction-table lines (verbatim from an
    actual ppn run's networksetup.txt), including the active/superseded
    ('T'/'F') duplicate pattern and the PROT/OOOOO/two-letter-symbol
    species-name conventions.
    '''

    ELEMENTS_NAMES = ['', 'H', 'He', '', '', '', 'C', 'N', 'O']

    NETWORKSETUP_FRAGMENT = (
        "************************************************************************\n"
        "\n"
        "          REACTION NETWORK                                   V(i) Nasv(*rho)\n"
        "\n"
        "          4  =  NGIR           0  = + REAZ ISOMERS\n"
        "      1 T  2  PROT   +  0  OOOOO  ->  1  H   2  +  0  OOOOO   4.344E-21  VITAL  (p,g)   5   1.000E+00   2.146E+18\n"
        "      3 T  2  HE  3  +  0  OOOOO  ->  1  HE  4  +  2  PROT    1.982E-15  VITAL  (v,v)  99   1.000E+00   1.241E+19\n"
        "     14 F  1  C  12  +  1  PROT   ->  1  N  13  +  0  OOOOO   5.666E-22  VITAL  (p,g)   5   1.000E+00   0.000E+00\n"
        "     15 T  1  C  12  +  1  PROT   ->  1  N  13  +  0  OOOOO   5.666E-22  JINAC  (p,g)   5   1.000E+00   0.000E+00\n"
    )

    def _write_fragment(self, tdir):
        import os
        path = os.path.join(tdir, 'networksetup.txt')
        with open(path, 'w') as f:
            f.write(self.NETWORKSETUP_FRAGMENT)
        return path

    def test_parse_networksetup(self):
        from nugridpy.nu_plots import network
        with TemporaryDirectory() as tdir:
            path = self._write_fragment(tdir)
            reactions = network.parse_networksetup(path, self.ELEMENTS_NAMES)
        self.assertEqual(len(reactions), 4)

        r1 = reactions[0]
        self.assertEqual(r1.index, 1)
        self.assertTrue(r1.active)
        self.assertEqual((r1.reactant.z, r1.reactant.a), (1, 1))   # PROT
        self.assertEqual((r1.projectile.z, r1.projectile.a), (0, 0))  # OOOOO
        self.assertEqual((r1.product1.z, r1.product1.a), (1, 2))   # H-2
        self.assertEqual(r1.source, 'VITAL')
        self.assertEqual(r1.reaction_type, '(p,g)')

        # no species should fail to resolve against the given elements_names
        for r in reactions:
            for s in (r.reactant, r.projectile, r.product1, r.product2):
                self.assertIsNotNone(s.z, msg='unresolved species: {}'.format(s))

    def test_reactions_in_network_active_filter(self):
        from nugridpy.nu_plots import network
        with TemporaryDirectory() as tdir:
            path = self._write_fragment(tdir)
            reactions = network.parse_networksetup(path, self.ELEMENTS_NAMES)

        self.assertEqual(len(network.reactions_in_network(reactions, active_only=False)), 4)
        active = network.reactions_in_network(reactions, active_only=True)
        # index 14 (F) is a superseded duplicate of index 15 (T) -- both
        # are C-12(p,g)N-13, so active_only should keep 15 and drop 14.
        self.assertEqual(sorted(r.index for r in active), [1, 3, 15])

    def test_reactions_for_isotope(self):
        from nugridpy.nu_plots import network
        with TemporaryDirectory() as tdir:
            path = self._write_fragment(tdir)
            reactions = network.parse_networksetup(path, self.ELEMENTS_NAMES)

        # He-4 (Z=2, A=4): appears only as product1 of reaction 3
        he4 = network.reactions_for_isotope(reactions, z=2, a=4)
        self.assertEqual([r.index for r in he4], [3])

        # proton (Z=1, A=1): reactant of 1 (count=2), product2 of 3,
        # reactant of 15 (active C-12(p,g)N-13; 14 is superseded)
        proton = network.reactions_for_isotope(reactions, z=1, a=1)
        self.assertEqual(sorted(r.index for r in proton), [1, 3, 15])

        # an isotope absent from every reaction here: O-16 (Z=8, A=16)
        o16 = network.reactions_for_isotope(reactions, z=8, a=16)
        self.assertEqual(o16, [])


class TestSensitivity(unittest.TestCase):
    '''
    Covers nu_plots.sensitivity: pure-function factor-dirname parsing
    and directory discovery need only real (possibly empty) temp
    directories; the abundance-ratio/table/ranking logic reuses the
    existing _FakeAbuVector/_synthetic_cycle fixtures, either passed
    directly as an instance or resolved from a monkey-patched
    `ppn.abu_vector` for the string-path code path real usage takes
    (restored in a finally block so it doesn't leak into other tests).
    '''

    def test_parse_factor_dirname(self):
        from nugridpy.nu_plots import sensitivity
        self.assertEqual(sensitivity.parse_factor_dirname('fact_10'), 10.0)
        self.assertEqual(sensitivity.parse_factor_dirname('fact_0.01'), 0.01)
        self.assertEqual(sensitivity.parse_factor_dirname('baseline'), 1.0)
        self.assertIsNone(sensitivity.parse_factor_dirname('NPDATA'))

    def test_discover_factored_runs(self):
        from nugridpy.nu_plots import sensitivity
        import os
        with TemporaryDirectory() as tdir:
            for name in ('fact_0.5', 'fact_2', 'fact_10', 'NPDATA', 'clean_output'):
                os.mkdir(os.path.join(tdir, name))
            runs = sensitivity.discover_factored_runs(tdir)
        self.assertEqual(sorted(runs.keys()), [0.5, 2.0, 10.0])

    def test_isotope_abundance_hyphen_and_massfirst_naming(self):
        from nugridpy.nu_plots import sensitivity
        baseline = _FakeAbuVector({0: _synthetic_cycle(scale=1.0)})
        abund = sensitivity.isotope_abundance(baseline, 'He-4')
        self.assertAlmostEqual(abund, 0.28, places=6)
        # mass-first naming, as used in reaction_plan.json, resolves the same isotope
        self.assertEqual(abund, sensitivity.isotope_abundance(baseline, '4He'))

    def test_reaction_sensitivity(self):
        from nugridpy.nu_plots import sensitivity
        import os
        baseline = _FakeAbuVector({0: _synthetic_cycle(scale=1.0)})

        with TemporaryDirectory() as tdir:
            reaction_dir = os.path.join(tdir, 'some_reaction')
            os.makedirs(os.path.join(reaction_dir, 'fact_10'))
            os.makedirs(os.path.join(reaction_dir, 'fact_0.1'))
            path_to_instance = {
                os.path.join(reaction_dir, 'fact_10'): _FakeAbuVector({0: _synthetic_cycle(scale=10.0)}),
                os.path.join(reaction_dir, 'fact_0.1'): _FakeAbuVector({0: _synthetic_cycle(scale=0.1)}),
            }
            real_abu_vector = sensitivity.ppn.abu_vector
            sensitivity.ppn.abu_vector = lambda path: path_to_instance[path]
            try:
                sens = sensitivity.reaction_sensitivity(baseline, reaction_dir, 'He-4')
            finally:
                sensitivity.ppn.abu_vector = real_abu_vector

        self.assertAlmostEqual(sens[10.0], 10.0, places=6)
        self.assertAlmostEqual(sens[0.1], 0.1, places=6)

    def test_sensitivity_table_and_rank_reactions(self):
        from nugridpy.nu_plots import sensitivity
        import os
        baseline = _FakeAbuVector({0: _synthetic_cycle(scale=1.0)})

        with TemporaryDirectory() as tdir:
            paths = {}
            for reaction, scale in [('rxn_big', 10.0), ('rxn_small', 1.1)]:
                d = os.path.join(tdir, reaction, 'fact_2')
                os.makedirs(d)
                paths[d] = _FakeAbuVector({0: _synthetic_cycle(scale=scale)})

            real_abu_vector = sensitivity.ppn.abu_vector
            sensitivity.ppn.abu_vector = lambda path: paths[path]
            try:
                table = sensitivity.sensitivity_table(
                    baseline, tdir,
                    [{'name': 'rxn_big', 'affected_isotopes': ['He-4']},
                     {'name': 'rxn_small', 'affected_isotopes': ['He-4']}])
            finally:
                sensitivity.ppn.abu_vector = real_abu_vector

        self.assertEqual(len(table), 2)
        ranked = sensitivity.rank_reactions(table, 'He-4')
        self.assertEqual(ranked[0][0], 'rxn_big')
        self.assertGreater(ranked[0][1], ranked[1][1])


class TestRealSeData(unittest.TestCase):
    '''
    Exercises nugridse.se against the real se.h5 files bundled in
    tests/data/ (a 3 Msun, Z=0.02 AGB model, 300 real cycles) --
    in-repo, portable, no absolute paths or network fetch needed.

    This is the regression net for three real bugs a debugging pass
    against this data found and fixed: two h5T.py float()-on-a-
    length-1-array crashes in the HDF5 age-attribute reader (blocked
    loading entirely), a missing `from numpy import *` in nugridse.py
    (a real regression from the Phase 1 data_plot.py split -- it used
    to get numpy names transitively through `from .data_plot import
    *`, which this refactor replaced with targeted imports), and an
    off-by-one between `xticks`/`labelsx` in iso_abund that only
    triggers when the isotope mass range span is <=100 (this bundled
    model's reduced isotope set hits that branch; the wider PPN
    baseline data used elsewhere doesn't).
    '''

    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'tests', 'data')

    def setUp(self):
        import matplotlib.pylab as mpy
        self.mpy = mpy

    def test_load_real_se_data(self):
        from nugridpy import nugridse
        s = nugridse.se(self.DATA_DIR)
        self.assertEqual(len(s.se.cycles), 300)

    def test_iso_abund_on_real_data(self):
        # exercises the xticks/labelsx off-by-one (mass range span <=100
        # here) and nugridse.py's bare array/zeros/where usage
        from nugridpy import nugridse
        s = nugridse.se(self.DATA_DIR)
        self.mpy.figure()
        s.iso_abund(3, stable=True, show=False)

    def test_plotprofMulti_on_real_data(self):
        from nugridpy import nugridse
        s = nugridse.se(self.DATA_DIR)
        with TemporaryDirectory() as tdir:
            cwd = os.getcwd()
            os.chdir(tdir)
            try:
                s.plotprofMulti(1, 3, 2, ['H-1', 'He-4'], 0, 3, -6, 0)
            finally:
                os.chdir(cwd)

    def test_movie_abu_chart_and_iso_abund_on_real_data(self):
        from nugridpy import nugridse
        s = nugridse.se(self.DATA_DIR)
        with TemporaryDirectory() as tdir:
            cwd = os.getcwd()
            os.chdir(tdir)
            try:
                s.movie([1, 3], plotstyle='abu_chart')
                s.movie([1, 3], plotstyle='iso_abund')
            finally:
                os.chdir(cwd)


class TestAbuChart(unittest.TestCase):

    def test_abu_chart(self):
        from nugridpy import utils,ppn,data_plot
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pylab as mpy
        import os

        # Perform tests within temporary directory
        with TemporaryDirectory() as tdir:
            # wget the data for a ppn run from the CADC VOspace
            n = 3
            for cycle in range(0,n):
                cycle_str = str(cycle).zfill(2)
                os.system("wget -q --content-disposition --directory '" + tdir + "' "
                          + "'http://www.canfar.phys.uvic.ca/vospace/synctrans?TARGET="\
                          + "vos%3A%2F%2Fcadc.nrc.ca%21vospace%2Fnugrid%2Fdata%2Fprojects%2Fppn%2Fexamples%2F"\
                          + "ppn_Hburn_simple%2Fiso_massf000" + cycle_str + ".DAT&DIRECTION=pullFromVoSpace&PROTOCOL"\
                          + "=ivo%3A%2F%2Fivoa.net%2Fvospace%2Fcore%23httpget'")

            # test_data_dir should point to the correct location of a set of abundances data file
            #nugrid_dir= os.path.dirname(os.path.dirname(ppn.__file__))
            #NuPPN_dir= nugrid_dir + "/NuPPN"
            #test_data_dir= NuPPN_dir + "/examples/ppn_C13_pocket/master_results"

            p=ppn.abu_vector(tdir) # TODO: this function fails to raise an exception if path is not found!
            mp=p.get('mod')
            if len(mp) == 0:
                raise IOError("Cannot locate a set of abundance data files")
            sparse=10
            cycles=mp[:1000:sparse]
            form_str='%6.1F'
            form_str1='%4.3F'

            i=0
            for cyc in cycles:
                T9  = p.get('t9',fname=cyc)
                Rho = p.get('rho',fname=cyc)
                mod = p.get('mod',fname=cyc)
                # time= p.get('agej',fname=cyc)*utils.constants.one_year
                time= p.get('agej',fname=cyc)
                mpy.close(i);mpy.figure(i);i += 1
                p.abu_chart(cyc,mass_range=[0,41],plotaxis=[-1,22,-1,22],lbound=(-6,0),show=False)
                mpy.title(str(mod)+' t='+form_str%time+'yr $T_9$='+form_str1%T9+' $\\rho$='+str(Rho))
                png_file='abu_chart_'+str(cyc).zfill(len(str(max(mp))))+'.png'
                mpy.savefig(png_file)
                self.assertTrue(os.path.exists(png_file))
                os.remove(png_file)


    def test_abu_evolution(self):
        from nugridpy import ppn, utils
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pylab as mpy
        import os

        # Perform tests within temporary directory
        with TemporaryDirectory() as tdir:
            # wget the data for a ppn run from the CADC VOspace
            os.system("wget -q --content-disposition --directory '" + tdir +  "' "\
                          + "'http://www.canfar.phys.uvic.ca/vospace/synctrans?TARGET="\
                          + "vos%3A%2F%2Fcadc.nrc.ca%21vospace%2Fnugrid%2Fdata%2Fprojects%2Fppn%2Fexamples%2F"\
                          + "ppn_Hburn_simple%2Fx-time.dat&DIRECTION=pullFromVoSpace&PROTOCOL"\
                          + "=ivo%3A%2F%2Fivoa.net%2Fvospace%2Fcore%23httpget'")

            #nugrid_dir= os.path.dirname(os.path.dirname(ppn.__file__))
            #NuPPN_dir= nugrid_dir + "/NuPPN"
            #test_data_dir= NuPPN_dir + "/examples/ppn_Hburn_simple/RUN_MASTER"

            symbs=utils.symbol_list('lines2')
            x=ppn.xtime(tdir)
            specs=['PROT','HE  4','C  12','N  14','O  16']
            i=0
            for spec in specs:
                x.plot('time',spec,logy=True,logx=True,shape=utils.linestyle(i)[0],show=False,title='')
                i += 1
            mpy.ylim(-5,0.2)
            mpy.legend(loc=0)
            mpy.xlabel('$\log t / \mathrm{min}$')
            mpy.ylabel('$\log X \mathrm{[mass fraction]}$')
            abu_evol_file = 'abu_evolution.png'
            mpy.savefig(abu_evol_file)
            self.assertTrue(os.path.exists(abu_evol_file))


class ImageCompare(unittest.TestCase):

    def test_ppnHburn_abucharts(self):
        from .ImageCompare.abu_chart import load_chart_files
        from .ImageCompare.compare_image_entropy import compare_images
        with TemporaryDirectory() as tdir:
            load_chart_files(tdir)
            compare_images(tdir)

if __name__ == '__main__':
    unittest.main()
