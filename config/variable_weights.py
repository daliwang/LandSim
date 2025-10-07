"""
Variable-specific weights for CNP model loss function.

This module provides configurable weights for different variables in the CNP model,
allowing for fine-tuning the loss function to prioritize specific variables.
"""

from typing import Dict, List, Optional, Any

def get_pft1d_variable_weights(variables: List[str] = None) -> Dict[str, float]:
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
    }
    
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

def get_soil2d_variable_weights(variables: List[str] = None) -> Dict[str, float]:
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
    }
    
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

def get_scalar_variable_weights(variables: List[str] = None) -> Dict[str, float]:
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
