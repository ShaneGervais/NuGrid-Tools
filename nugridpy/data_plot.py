#
# NuGridpy - Tools for accessing and visualising NuGrid data.
#
# Copyright 2007 - 2014 by the NuGrid Team.
# All rights reserved. See LICENSE.
#

"""
data_plot.py

DEPRECATED: this module is a backward-compatibility shim. The
`DataPlot` god-class that used to live here has been split into
`nugridpy.plot_common.PlotCommon` (generic + dual-concern plotting
helpers), `nugridpy.nu_plots.plotting.NuPlotMixin` (nucleosynthesis
network plots), and `nugridpy.mesa_plots.plotting.MesaPlotMixin` (MESA
plots). `DataPlot` here is kept as an alias of `PlotCommon` only so
that `from nugridpy import data_plot` and `from nugridpy.data_plot
import DataPlot` keep working; new code should import from
`plot_common`/`nu_plots`/`mesa_plots` directly.
"""
from __future__ import absolute_import

from .plot_common import PlotCommon as DataPlot
from .plot_common import _padding_model_number
from .nu_plots.plotting import flux_chart
