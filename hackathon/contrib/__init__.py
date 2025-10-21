"""
Hackathon contrib module for protein-ligand binding site prediction.
"""

from .simple_binding_predictor import predict_binding_sites, predict_binding_sites_batch

__all__ = ['predict_binding_sites', 'predict_binding_sites_batch']
