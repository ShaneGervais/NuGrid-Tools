"""
nugridpy/styles.py

A reusable "publication-quality" matplotlib style, aiming for the same
look as Julia's CairoMakie + a LaTeX theme: serif, LaTeX-typeset text
and math, clean minimal axes, high-resolution `savefig` defaults.

Every existing NuGrid-Tools plotting method already writes its math
labels as plain matplotlib mathtext (e.g. `'$\\log_{10}(X)$'`) rather
than through a real LaTeX toolchain -- no method currently sets
`text.usetex=True`. So the Computer-Modern-style look comes from
switching mathtext's own font set (`mathtext.fontset='cm'`), the same
way CairoMakie's LaTeX theme typesets math itself (via
MathTeXEngine.jl) rather than shelling out to a system `latex`
binary. That keeps this style dependency-free and portable. A real
`text.usetex=True` toolchain is still offered as an explicit opt-in
for users who have LaTeX installed and want literal LaTeX rendering,
but it is slower, and fails outright on machines without a LaTeX
distribution -- hence off by default.

Usage
-----
>>> from nugridpy.styles import use_latex_style
>>> use_latex_style()
>>> # ... make plots as usual ...

This only sets global `matplotlib.rcParams` -- it does not touch
`figsize`/`fontsize`/`color_map` arguments that individual plotting
methods pass explicitly, which still take precedence for whichever
plot calls them.
"""
import matplotlib

_LATEX_STYLE_RCPARAMS = {
    'font.family': 'serif',
    'mathtext.fontset': 'cm',
    'font.size': 11,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'axes.linewidth': 0.8,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.frameon': False,
    'figure.figsize': (6.0, 4.5),
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
}


def use_latex_style(usetex=False):
    '''
    Apply a publication-quality, LaTeX-typeset plotting style.

    Sets serif text and Computer-Modern-style math (via matplotlib's
    `mathtext`, not a system LaTeX install), thin boxed axes, inward
    ticks, a frameless legend, and high-resolution `savefig` defaults.
    Affects all matplotlib plotting from the point it's called, until
    `reset_style()` is called or the process exits.

    Parameters
    ----------
    usetex : boolean, optional
        If True, additionally sets `text.usetex=True` so labels render
        through a real system LaTeX toolchain instead of mathtext. This
        requires LaTeX (plus `dvipng`) to already be installed and is
        slower and less portable -- most users should leave this False
        and get the same visual style from mathtext alone. The default
        is False.

    Returns
    -------
    dict
        The previous values of every rcParam this function changes, in
        a form that can be passed straight to
        `matplotlib.rcParams.update(...)` to undo the change (see
        `reset_style`).
    '''
    params = dict(_LATEX_STYLE_RCPARAMS)
    if usetex:
        params['text.usetex'] = True

    previous = {key: matplotlib.rcParams[key] for key in params}
    matplotlib.rcParams.update(params)
    return previous


def reset_style(previous):
    '''
    Undo a previous `use_latex_style()` call.

    Parameters
    ----------
    previous : dict
        The dict returned by `use_latex_style()`.
    '''
    matplotlib.rcParams.update(previous)
