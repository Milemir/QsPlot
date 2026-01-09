"""
QsPlot - High-performance Visualization for Quantitative Finance
"""

import warnings

# Core Python API
from .processor import DataProcessor
from .core import Visualizer
from .utils.imputer import FastImputer

# Expose C++ module components
try:
    from . import qsplot_engine
    Renderer = qsplot_engine.Renderer
    RendererConfig = qsplot_engine.RendererConfig
except (ImportError, AttributeError):
    try:
        import qsplot_engine
        Renderer = qsplot_engine.Renderer
        RendererConfig = qsplot_engine.RendererConfig
    except (ImportError, AttributeError):
        Renderer = None
        RendererConfig = None

__version__ = "0.2.0"
__all__ = [
    "Visualizer",
    "DataProcessor", 
    "FastImputer",
    "Renderer",
    "RendererConfig",
]
