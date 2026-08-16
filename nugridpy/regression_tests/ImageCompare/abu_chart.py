from builtins import str
from builtins import range
from ... import ppn as p
import os
import os.path
import shutil

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_THIS_DIR, '..', 'data', 'ppn_Hburn_simple')
REFERENCE_DIR = os.path.join(DATA_DIR, 'reference_abucharts')

# Matches the bundled iso_massf000{00,10,20,30}.DAT cycles.
CYCLES = list(range(0, 39, 10))


def load_chart_files(path='.'):
    '''
    Populate `path` with the bundled ppn_Hburn_simple abundance data
    plus the committed MasterAbuChart*.png reference images, then
    regenerate AbuChart*.png there from that data so `compare_images`
    can diff the freshly-drawn charts against the reference ones.

    This used to `wget` both the input data and the reference images
    from the CADC VOspace; both are now bundled locally under
    `regression_tests/data/ppn_Hburn_simple/` so the test has no
    network dependency. See `reference_abucharts/README.md` for how
    the reference images were produced.
    '''
    for cycle in CYCLES:
        cycle_str = str(cycle).zfill(2)
        iso_massf_name = 'iso_massf000' + cycle_str + '.DAT'
        shutil.copy(os.path.join(DATA_DIR, iso_massf_name),
                    os.path.join(path, iso_massf_name))
        master_name = 'MasterAbuChart' + cycle_str + '.png'
        shutil.copy(os.path.join(REFERENCE_DIR, master_name),
                    os.path.join(path, master_name))
    a = p.abu_vector(path)
    a.abu_chart(CYCLES, plotaxis=[-1, 16, -1, 15], savefig=True, path=path)


if __name__ == "__main__":
    load_chart_files()
