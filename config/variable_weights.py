"""
Variable-specific weights for CNP model loss function.

This module provides configurable weights for different variables in the CNP model,
allowing for fine-tuning the loss function to prioritize specific variables.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

# Global variable to store loaded weights from JSON file
_loaded_weights: Optional[Dict[str, Dict[str, float]]] = None

def load_variable_weights_from_json(json_path: str) -> Optional[Dict[str, Dict[str, float]]]:
    """
    Load variable weights from a JSON file.
    
    Supports two formats:
    1. Direct format (legacy):
       {
           "pft1d_weights": {"var1": 1.0, "var2": 2.0, ...},
           "soil2d_weights": {"var1": 1.0, "var2": 2.0, ...},
           "scalar_weights": {"var1": 1.0, "var2": 2.0, ...}
       }
    
    2. Unified format (new):
       {
           "variable_weights": {
               "pft1d_weights": {"var1": 1.0, ...},
               "soil2d_weights": {"var1": 1.0, ...},
               "scalar_weights": {"var1": 1.0, ...}
           },
           "tail_aware_weights": {...},
           ...
       }
    
    Args:
        json_path: Path to JSON file containing variable weights
        
    Returns:
        Dictionary with keys 'pft1d_weights', 'soil2d_weights', 'scalar_weights',
        or None if file doesn't exist or is invalid
    """
    global _loaded_weights
    if json_path is None or not os.path.exists(json_path):
        return None
    
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            # Check if it's unified format (has 'variable_weights' key)
            if 'variable_weights' in data:
                weights = data['variable_weights']
                if isinstance(weights, dict):
                    _loaded_weights = weights
                    return weights
            # Otherwise, assume direct format (legacy)
            elif 'pft1d_weights' in data or 'soil2d_weights' in data or 'scalar_weights' in data:
                _loaded_weights = data
                return data
            else:
                print(f"Warning: JSON file {json_path} doesn't contain expected variable_weights structure")
                return None
    except Exception as e:
        print(f"Warning: Failed to load variable weights from {json_path}: {e}")
        return None
    
    return None


def get_pft1d_variable_weights(variables: List[str] = None, json_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Get variable-specific weights for PFT1D variables.
    
    Args:
        variables: List of PFT1D variable names (if None, returns default weights)
        
    Returns:
        Dictionary mapping variable names to their weights
    """
    # Default weights for key variables
    default_weights = {
        'xsmrpool': 5.0,   # Higher weight for problematic variables
        'cpool': 2.0,
        'npool': 2.0,
        'ppool': 2.0,
        'tlai': 3.0,       # Leaf area index is important
        'totvegc': 2.0,    # Total vegetation carbon
        'litr2p_vr': 5.0,  # Litter 2 phosphorus - variable weight
        'litr2n': 5.0,     # Litter 2 nitrogen - variable weight
        'litr2n_vr': 5.0,  # Litter 2 nitrogen (variant) - variable weight
    }
    
    # Use JSON weights if provided, otherwise use defaults
    if json_weights is not None:
        # Merge JSON weights with defaults (JSON takes precedence)
        merged_weights = default_weights.copy()
        merged_weights.update(json_weights)
        default_weights = merged_weights
    elif _loaded_weights is not None and 'pft1d_weights' in _loaded_weights:
        # Use globally loaded weights
        merged_weights = default_weights.copy()
        merged_weights.update(_loaded_weights['pft1d_weights'])
        default_weights = merged_weights
    
    # If no variables provided, return default weights
    if variables is None:
        return default_weights
    
    # Create weights dictionary for the provided variables
    weights = {}
    for var in variables:
        if var in default_weights:
            weights[var] = default_weights[var]
        else:
            weights[var] = 1.0  # Default weight for other variables
    
    return weights

def get_soil2d_variable_weights(variables: List[str] = None, json_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Get variable-specific weights for soil2D variables.
    
    Args:
        variables: List of soil2D variable names (if None, returns default weights)
        
    Returns:
        Dictionary mapping variable names to their weights
    """
    # Default weights for key soil variables
    default_weights = {
        'primp_vr': 3.0,
        'sminn_vr': 3.0,
        'smin_no3_vr': 3.0,
        'smin_nh4_vr': 3.0,
        'labilep_vr': 2.5,
        'secondp_vr': 2.5,
        'litr2p_vr': 5.0,  # Litter 2 phosphorus - variable weight
        'litr2n_vr': 5.0,  # Litter 2 nitrogen - variable weight
    }
    
    # Use JSON weights if provided, otherwise use defaults
    if json_weights is not None:
        # Merge JSON weights with defaults (JSON takes precedence)
        merged_weights = default_weights.copy()
        merged_weights.update(json_weights)
        default_weights = merged_weights
    elif _loaded_weights is not None and 'soil2d_weights' in _loaded_weights:
        # Use globally loaded weights
        merged_weights = default_weights.copy()
        merged_weights.update(_loaded_weights['soil2d_weights'])
        default_weights = merged_weights
    
    # If no variables provided, return default weights
    if variables is None:
        return default_weights
    
    # Create weights dictionary for the provided variables
    weights = {}
    for var in variables:
        if var in default_weights:
            weights[var] = default_weights[var]
        else:
            weights[var] = 1.0  # Default weight for other variables
    
    return weights

def get_scalar_variable_weights(variables: List[str] = None, json_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """
    Get variable-specific weights for scalar variables.
    
    Args:
        variables: List of scalar variable names (if None, returns default weights)
        
    Returns:
        Dictionary mapping variable names to their weights
    """
    # Default weights for scalar variables
    default_weights = {
        'GPP': 1.5,
        'NPP': 1.5,
        'AR': 1.2,
        'HR': 1.2,
    }
    
    # Use JSON weights if provided, otherwise use defaults
    if json_weights is not None:
        # Merge JSON weights with defaults (JSON takes precedence)
        merged_weights = default_weights.copy()
        merged_weights.update(json_weights)
        default_weights = merged_weights
    elif _loaded_weights is not None and 'scalar_weights' in _loaded_weights:
        # Use globally loaded weights
        merged_weights = default_weights.copy()
        merged_weights.update(_loaded_weights['scalar_weights'])
        default_weights = merged_weights
    
    # If no variables provided, return default weights
    if variables is None:
        return default_weights
    
    # Create weights dictionary for the provided variables
    weights = {}
    for var in variables:
        if var in default_weights:
            weights[var] = default_weights[var]
        else:
            weights[var] = 1.0  # Default weight for other variables
    
    return weights
