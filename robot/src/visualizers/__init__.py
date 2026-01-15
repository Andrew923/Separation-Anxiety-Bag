"""
Visualization modules for robot sensor data.

Provides OpenCV-based GUI components for debugging and integration testing.
"""

from .vfh_gui import VFHVisualizer, VisualizerConfig, InfoPanelRenderer
from .depth_preprocessor_viz import DepthPreprocessorVisualizer, DepthVisualizerConfig

__all__ = [
    'VFHVisualizer',
    'VisualizerConfig',
    'InfoPanelRenderer',
    'DepthPreprocessorVisualizer',
    'DepthVisualizerConfig',
]
