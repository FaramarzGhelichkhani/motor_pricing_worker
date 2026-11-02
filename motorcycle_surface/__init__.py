"""USED motorcycle price surfaces. Only Publication Policy V15 is active.

Factory data is excluded before feature generation, modeling and publication.
"""
from .core import MotorcyclePriceSurface, SurfaceConfig, FEATURES_24, GROUP, CELL
__all__ = ['MotorcyclePriceSurface','SurfaceConfig','FEATURES_24','GROUP','CELL']
__version__ = '15.0.0'
