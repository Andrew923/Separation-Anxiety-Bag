"""
Visualization modules for robot sensor data.

Provides OpenCV-based GUI components for debugging and integration testing.
"""

from .depth_preprocessor_viz import DepthPreprocessorVisualizer, DepthVisualizerConfig

__all__ = [
    'DepthPreprocessorVisualizer',
    'DepthVisualizerConfig',
]
