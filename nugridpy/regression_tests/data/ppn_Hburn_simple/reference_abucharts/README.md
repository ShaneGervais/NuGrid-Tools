# Reference abundance-chart images

These `MasterAbuChart{00,10,20,30}.png` files are the reference images
used by `ImageCompare::test_ppnHburn_abucharts` (via
`nugridpy/regression_tests/ImageCompare/compare_image_entropy.py`).

They were originally downloaded from the CADC VOspace alongside the
`iso_massf*.DAT` example data (see the old `wget`-based
`load_chart_files` for the URLs), but those reference images are not
otherwise available on this machine, so this test could not be
de-networked against the original ground truth.

Instead, these images were regenerated locally from the bundled,
typo-corrected `../iso_massf000{00,10,20,30}.DAT` data using
`abu_vector.abu_chart(cycles, plotaxis=[-1,16,-1,15], savefig=True)`
on already-verified, bug-fixed NuGrid-Tools code. The test therefore
now checks for *self-consistency* (does the current code still
produce the same chart it produced when these images were captured)
rather than matching an external, independently-authored reference.

If `abu_chart`'s rendering intentionally changes, regenerate these
images the same way and review the diff visually before committing.
