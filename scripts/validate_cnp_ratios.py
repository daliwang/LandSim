#!/usr/bin/env python3
"""
Validate CNP stoichiometric ratios in model predictions.

This script checks whether predicted CNP variables follow the expected
stoichiometric relationships defined in CNP_STOICHIOMETRIC_RELATIONSHIPS.md.

Usage:
    python scripts/validate_cnp_ratios.py --results-dir cnp_results/run_20260212_162802_experiment_2
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# PFT-specific ratio values (from model_variable_quantities.txt)
DEADWDCN = np.array([1, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 500, 
                     0, 0, 0, 0, 0, 500, 500, 500, 500, 500, 500, 500, 500])
DEADWDCP = 3000.0  # Constant for all PFTs

LEAFCN = np.array([1, 35, 40, 25, 30, 30, 25, 25, 25, 30, 25, 25, 25, 25, 25, 25, 
                   25, 25, 25, 25, 25, 25, 25, 25, 25])
LEAFCP = np.array([1, 525, 400, 250, 600, 450, 500, 375, 250, 450, 375, 250, 250, 
                   375, 375, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275])

FROOTCN = np.array([1, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 42, 
                    42, 42, 42, 42, 42, 42, 42, 42, 42])
FROOTCP = 1000.0  # Constant

LIVEWDCN = np.array([1, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 0, 0, 0, 0, 0, 
                     50, 50, 50, 50, 50, 50, 50, 50])
LIVEWDCP = 3000.0  # Constant

# Soil layer ratios (constant)
SOIL1_CN = 12.0
SOIL1_CP = 360.0  # cn_s1_new * np_s1_new = 12 * 30
SOIL2_CN = 12.0
SOIL2_CP = 360.0
SOIL3_CN = 10.0
SOIL3_CP = 500.0  # cn_s3_new * np_s3_new = 10 * 50
SOIL4_CN = 10.0
SOIL4_CP = 500.0


def load_predictions(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load predictions and ground truth from results directory."""
    predictions = {}
    
    # Load PFT 1D predictions
    pft_pred_dir = results_dir / 'cnp_predictions' / 'pft_1d_predictions'
    pft_gt_dir = results_dir / 'cnp_predictions' / 'pft_1d_ground_truth'
    
    if pft_pred_dir.exists():
        for pred_file in pft_pred_dir.glob('*.csv'):
            # Extract variable name from "predictions_Y_varname.csv" or "Y_varname.csv"
            stem = pred_file.stem
            if stem.startswith('predictions_'):
                var_name = stem.replace('predictions_', '').replace('Y_', '')
                # Find matching ground truth: "ground_truth_Y_varname.csv"
                gt_file = pft_gt_dir / f"ground_truth_Y_{var_name}.csv"
            else:
                var_name = stem.replace('Y_', '')
                gt_file = pft_gt_dir / pred_file.name
            
            if gt_file.exists():
                try:
                    predictions[f'pft_1d_{var_name}_pred'] = pd.read_csv(pred_file)
                    predictions[f'pft_1d_{var_name}_gt'] = pd.read_csv(gt_file)
                    logger.debug(f"Loaded pft_1d: {var_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {var_name}: {e}")
    
    # Load soil 2D predictions
    soil_pred_dir = results_dir / 'cnp_predictions' / 'soil_2d_predictions'
    soil_gt_dir = results_dir / 'cnp_predictions' / 'soil_2d_ground_truth'
    
    if soil_pred_dir.exists():
        for pred_file in soil_pred_dir.glob('*.csv'):
            # Extract variable name from "predictions_Y_varname.csv" or "Y_varname.csv"
            stem = pred_file.stem
            if stem.startswith('predictions_'):
                var_name = stem.replace('predictions_', '').replace('Y_', '')
                # Find matching ground truth: "ground_truth_Y_varname.csv"
                gt_file = soil_gt_dir / f"ground_truth_Y_{var_name}.csv"
            else:
                var_name = stem.replace('Y_', '')
                gt_file = soil_gt_dir / pred_file.name
            
            if gt_file.exists():
                try:
                    predictions[f'soil_2d_{var_name}_pred'] = pd.read_csv(pred_file)
                    predictions[f'soil_2d_{var_name}_gt'] = pd.read_csv(gt_file)
                    logger.debug(f"Loaded soil_2d: {var_name}")
                except Exception as e:
                    logger.warning(f"Failed to load {var_name}: {e}")
    
    logger.info(f"Loaded {len(predictions) // 2} prediction/ground truth pairs")
    return predictions


def check_pft_1d_ratio(
    c_pred: np.ndarray,
    n_pred: np.ndarray,
    p_pred: np.ndarray,
    c_gt: np.ndarray,
    n_gt: np.ndarray,
    p_gt: np.ndarray,
    cn_ratio: np.ndarray,
    cp_ratio: float,
    woody: np.ndarray = None,
    var_name: str = ""
) -> Dict[str, float]:
    """
    Check CNP ratios for 1D PFT variables.
    
    Args:
        c_pred, n_pred, p_pred: Predictions [n_samples, n_pfts]
        c_gt, n_gt, p_gt: Ground truth [n_samples, n_pfts]
        cn_ratio: C:N ratio per PFT [n_pfts]
        cp_ratio: C:P ratio (constant)
        woody: Woody flag per PFT [n_pfts], optional
        var_name: Variable name for logging
    
    Returns:
        Dictionary with ratio violation statistics
    """
    results = {}
    
    # Apply woody mask if provided
    if woody is not None:
        woody_mask = (woody > 0.5).astype(float)
        # For non-woody PFTs, N and P should be 0
        n_pred_masked = n_pred * woody_mask
        p_pred_masked = p_pred * woody_mask
        n_gt_masked = n_gt * woody_mask
        p_gt_masked = p_gt * woody_mask
    else:
        woody_mask = np.ones_like(c_pred)
        n_pred_masked = n_pred
        p_pred_masked = p_pred
        n_gt_masked = n_gt
        p_gt_masked = p_gt
    
    # Ensure cn_ratio and woody_mask can broadcast with c_pred
    # c_pred shape: (n_samples, n_pfts)
    # cn_ratio shape: should be (1, n_pfts) or (n_pfts,)
    if len(cn_ratio.shape) == 1:
        cn_ratio = cn_ratio.reshape(1, -1)
    if woody_mask is not None and len(woody_mask.shape) == 1:
        woody_mask = woody_mask.reshape(1, -1)
    
    # Compute expected N and P from C predictions
    n_expected = c_pred / (cn_ratio + 1e-8)
    if woody_mask is not None:
        n_expected = n_expected * woody_mask
    
    p_expected = c_pred / (cp_ratio + 1e-8)
    if woody_mask is not None:
        p_expected = p_expected * woody_mask
    
    # Compute expected N and P from C ground truth
    n_expected_gt = c_gt / (cn_ratio + 1e-8) * woody_mask
    p_expected_gt = c_gt / (cp_ratio + 1e-8) * woody_mask
    
    # Calculate ratio violations (relative error)
    # For predictions
    valid_mask_pred = (c_pred > 1e-8) & (woody_mask > 0.5)
    if valid_mask_pred.sum() > 0:
        cn_ratio_pred = np.where(valid_mask_pred, c_pred / (n_pred_masked + 1e-8), np.nan)
        cp_ratio_pred = np.where(valid_mask_pred, c_pred / (p_pred_masked + 1e-8), np.nan)
        
        cn_ratio_expected = np.where(valid_mask_pred, cn_ratio, np.nan)
        cp_ratio_expected = np.where(valid_mask_pred, cp_ratio, np.nan)
        
        cn_error = np.abs(cn_ratio_pred - cn_ratio_expected) / (cn_ratio_expected + 1e-8)
        cp_error = np.abs(cp_ratio_pred - cp_ratio_expected) / (cp_ratio_expected + 1e-8)
        
        results['cn_ratio_mae_pred'] = np.nanmean(np.abs(n_pred_masked - n_expected))
        results['cp_ratio_mae_pred'] = np.nanmean(np.abs(p_pred_masked - p_expected))
        results['cn_ratio_rel_error_pred'] = np.nanmean(cn_error)
        results['cp_ratio_rel_error_pred'] = np.nanmean(cp_error)
        results['cn_ratio_rmse_pred'] = np.sqrt(np.nanmean((n_pred_masked - n_expected) ** 2))
        results['cp_ratio_rmse_pred'] = np.sqrt(np.nanmean((p_pred_masked - p_expected) ** 2))
    else:
        results['cn_ratio_mae_pred'] = np.nan
        results['cp_ratio_mae_pred'] = np.nan
        results['cn_ratio_rel_error_pred'] = np.nan
        results['cp_ratio_rel_error_pred'] = np.nan
        results['cn_ratio_rmse_pred'] = np.nan
        results['cp_ratio_rmse_pred'] = np.nan
    
    # For ground truth (to verify ratios are correct in GT)
    valid_mask_gt = (c_gt > 1e-8) & (woody_mask > 0.5)
    if valid_mask_gt.sum() > 0:
        cn_ratio_gt = np.where(valid_mask_gt, c_gt / (n_gt_masked + 1e-8), np.nan)
        cp_ratio_gt = np.where(valid_mask_gt, c_gt / (p_gt_masked + 1e-8), np.nan)
        
        cn_ratio_expected_gt = np.where(valid_mask_gt, cn_ratio, np.nan)
        cp_ratio_expected_gt = np.where(valid_mask_gt, cp_ratio, np.nan)
        
        cn_error_gt = np.abs(cn_ratio_gt - cn_ratio_expected_gt) / (cn_ratio_expected_gt + 1e-8)
        cp_error_gt = np.abs(cp_ratio_gt - cp_ratio_expected_gt) / (cp_ratio_expected_gt + 1e-8)
        
        results['cn_ratio_rel_error_gt'] = np.nanmean(cn_error_gt)
        results['cp_ratio_rel_error_gt'] = np.nanmean(cp_error_gt)
    
    return results


def check_soil_2d_ratio(
    c_pred: np.ndarray,
    n_pred: np.ndarray,
    p_pred: np.ndarray,
    c_gt: np.ndarray,
    n_gt: np.ndarray,
    p_gt: np.ndarray,
    cn_ratio: float,
    cp_ratio: float,
    var_name: str = ""
) -> Dict[str, float]:
    """
    Check CNP ratios for 2D soil variables.
    
    Args:
        c_pred, n_pred, p_pred: Predictions [n_samples, n_layers]
        c_gt, n_gt, p_gt: Ground truth [n_samples, n_layers]
        cn_ratio: C:N ratio (constant)
        cp_ratio: C:P ratio (constant)
        var_name: Variable name for logging
    
    Returns:
        Dictionary with ratio violation statistics
    """
    results = {}
    
    # Compute expected N and P from C
    n_expected = c_pred / (cn_ratio + 1e-8)
    p_expected = c_pred / (cp_ratio + 1e-8)
    
    n_expected_gt = c_gt / (cn_ratio + 1e-8)
    p_expected_gt = c_gt / (cp_ratio + 1e-8)
    
    # Calculate ratio violations
    valid_mask = c_pred > 1e-8
    
    if valid_mask.sum() > 0:
        cn_ratio_pred = np.where(valid_mask, c_pred / (n_pred + 1e-8), np.nan)
        cp_ratio_pred = np.where(valid_mask, c_pred / (p_pred + 1e-8), np.nan)
        
        cn_error = np.abs(cn_ratio_pred - cn_ratio) / (cn_ratio + 1e-8)
        cp_error = np.abs(cp_ratio_pred - cp_ratio) / (cp_ratio + 1e-8)
        
        results['cn_ratio_mae_pred'] = np.nanmean(np.abs(n_pred - n_expected))
        results['cp_ratio_mae_pred'] = np.nanmean(np.abs(p_pred - p_expected))
        results['cn_ratio_rel_error_pred'] = np.nanmean(cn_error)
        results['cp_ratio_rel_error_pred'] = np.nanmean(cp_error)
        results['cn_ratio_rmse_pred'] = np.sqrt(np.nanmean((n_pred - n_expected) ** 2))
        results['cp_ratio_rmse_pred'] = np.sqrt(np.nanmean((p_pred - p_expected) ** 2))
    else:
        results['cn_ratio_mae_pred'] = np.nan
        results['cp_ratio_mae_pred'] = np.nan
        results['cn_ratio_rel_error_pred'] = np.nan
        results['cp_ratio_rel_error_pred'] = np.nan
        results['cn_ratio_rmse_pred'] = np.nan
        results['cp_ratio_rmse_pred'] = np.nan
    
    return results


def validate_cnp_ratios(results_dir: Path, output_file: Optional[Path] = None) -> Dict:
    """Main validation function."""
    logger.info(f"Loading predictions from {results_dir}")
    predictions = load_predictions(results_dir)
    
    # Load config to get variable lists
    config_path = results_dir / 'cnp_config.json'
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_info = config.get('data_info', {})
    pft_1d_vars = data_info.get('variables_1d_pft', [])
    soil_2d_vars = data_info.get('variables_2d_soil', [])
    
    validation_results = {}
    
    # Validate PFT 1D variables
    logger.info("Validating PFT 1D variables...")
    
    # Dead stem
    if 'deadstemc' in pft_1d_vars:
        c_pred_key = f'pft_1d_deadstemc_pred'
        n_pred_key = f'pft_1d_deadstemn_pred'
        p_pred_key = f'pft_1d_deadstemp_pred'
        
        if all(k in predictions for k in [c_pred_key, n_pred_key, p_pred_key]):
            c_pred_df = predictions[c_pred_key]
            n_pred_df = predictions[n_pred_key]
            p_pred_df = predictions[p_pred_key]
            c_gt_df = predictions[c_pred_key.replace('pred', 'gt')]
            n_gt_df = predictions[n_pred_key.replace('pred', 'gt')]
            p_gt_df = predictions[p_pred_key.replace('pred', 'gt')]
            
            # Extract PFT columns from each dataframe independently
            # Each variable has different column names (Y_deadstemc_pft1 vs Y_deadstemn_pft1)
            def get_pft_cols(df):
                pft_cols = [c for c in df.columns if 'pft' in str(c).lower()]
                if not pft_cols:
                    # Fallback: exclude coordinate columns
                    pft_cols = [c for c in df.columns if c.lower() not in ['longitude', 'latitude', 'lon', 'lat', 'unnamed: 0', 'index']]
                
                # Sort PFT columns numerically (pft1, pft2, ..., pft16)
                def pft_sort_key(col):
                    import re
                    match = re.search(r'pft(\d+)', str(col).lower())
                    return int(match.group(1)) if match else 999
                
                return sorted(pft_cols, key=pft_sort_key)
            
            c_pft_cols = get_pft_cols(c_pred_df)
            n_pft_cols = get_pft_cols(n_pred_df)
            p_pft_cols = get_pft_cols(p_pred_df)
            
            c_pred = c_pred_df[c_pft_cols].values
            n_pred = n_pred_df[n_pft_cols].values
            p_pred = p_pred_df[p_pft_cols].values
            c_gt = c_gt_df[c_pft_cols].values
            n_gt = n_gt_df[n_pft_cols].values
            p_gt = p_gt_df[p_pft_cols].values
            
            # Debug: check shapes
            logger.debug(f"deadstemc shapes: c_pred={c_pred.shape}, n_pred={n_pred.shape}, p_pred={p_pred.shape}")
            
            # Determine number of PFTs from actual data shape
            n_pfts = c_pred.shape[1] if len(c_pred.shape) > 1 else 1
            
            # Use PFT indices 1-16 (skip PFT0)
            if n_pfts == 16:
                cn_ratio = DEADWDCN[1:17]
                woody = np.ones(16)
            else:
                # Use first n_pfts ratios from PFT indices 1 onwards
                cn_ratio = DEADWDCN[1:min(n_pfts+1, len(DEADWDCN))]
                if len(cn_ratio) < n_pfts:
                    cn_ratio = np.pad(cn_ratio, (0, n_pfts - len(cn_ratio)), constant_values=500.0)
                woody = np.ones(n_pfts)
            
            # Ensure cn_ratio and woody have correct shape for broadcasting
            # They should be (n_pfts,) to broadcast with (n_samples, n_pfts)
            cn_ratio = np.array(cn_ratio).reshape(1, -1) if len(cn_ratio.shape) == 1 else cn_ratio
            woody = np.array(woody).reshape(1, -1) if len(woody.shape) == 1 else woody
            
            results = check_pft_1d_ratio(
                c_pred, n_pred, p_pred, c_gt, n_gt, p_gt,
                cn_ratio, DEADWDCP, woody, 'deadstemc'
            )
            validation_results['deadstemc'] = results
            logger.info(f"deadstemc CN ratio error: {results.get('cn_ratio_rel_error_pred', 'N/A'):.4f}")
            logger.info(f"deadstemc CP ratio error: {results.get('cp_ratio_rel_error_pred', 'N/A'):.4f}")
    
    # Leaf
    if 'leafc' in pft_1d_vars:
        c_pred_key = f'pft_1d_leafc_pred'
        n_pred_key = f'pft_1d_leafn_pred'
        p_pred_key = f'pft_1d_leafp_pred'
        
        if all(k in predictions for k in [c_pred_key, n_pred_key, p_pred_key]):
            c_pred_df = predictions[c_pred_key]
            n_pred_df = predictions[n_pred_key]
            p_pred_df = predictions[p_pred_key]
            c_gt_df = predictions[c_pred_key.replace('pred', 'gt')]
            n_gt_df = predictions[n_pred_key.replace('pred', 'gt')]
            p_gt_df = predictions[p_pred_key.replace('pred', 'gt')]
            
            # Extract PFT columns
            def get_pft_cols(df):
                pft_cols = [c for c in df.columns if 'pft' in str(c).lower()]
                if not pft_cols:
                    pft_cols = [c for c in df.columns if c.lower() not in ['longitude', 'latitude', 'lon', 'lat', 'unnamed: 0', 'index']]
                def pft_sort_key(col):
                    import re
                    match = re.search(r'pft(\d+)', str(col).lower())
                    return int(match.group(1)) if match else 999
                return sorted(pft_cols, key=pft_sort_key)
            
            c_pft_cols = get_pft_cols(c_pred_df)
            n_pft_cols = get_pft_cols(n_pred_df)
            p_pft_cols = get_pft_cols(p_pred_df)
            
            c_pred = c_pred_df[c_pft_cols].values
            n_pred = n_pred_df[n_pft_cols].values
            p_pred = p_pred_df[p_pft_cols].values
            c_gt = c_gt_df[c_pft_cols].values
            n_gt = n_gt_df[n_pft_cols].values
            p_gt = p_gt_df[p_pft_cols].values
            
            n_pfts = c_pred.shape[1] if len(c_pred.shape) > 1 else 1
            
            cn_ratio = LEAFCN[1:min(n_pfts+1, len(LEAFCN))]
            if len(cn_ratio) < n_pfts:
                cn_ratio = np.pad(cn_ratio, (0, n_pfts - len(cn_ratio)), constant_values=25.0)
            cn_ratio = np.array(cn_ratio).reshape(1, -1) if len(cn_ratio.shape) == 1 else cn_ratio
            
            # Leaf CP is PFT-specific, use average for now
            cp_ratio = np.mean(LEAFCP[1:min(n_pfts+1, len(LEAFCP))])
            
            results = check_pft_1d_ratio(
                c_pred, n_pred, p_pred, c_gt, n_gt, p_gt,
                cn_ratio, cp_ratio, None, 'leafc'
            )
            validation_results['leafc'] = results
            logger.info(f"leafc CN ratio error: {results.get('cn_ratio_rel_error_pred', 'N/A'):.4f}")
    
    # Fine root
    if 'frootc' in pft_1d_vars:
        c_pred_key = f'pft_1d_frootc_pred'
        n_pred_key = f'pft_1d_frootn_pred'
        p_pred_key = f'pft_1d_frootp_pred'
        
        if all(k in predictions for k in [c_pred_key, n_pred_key, p_pred_key]):
            c_pred_df = predictions[c_pred_key]
            n_pred_df = predictions[n_pred_key]
            p_pred_df = predictions[p_pred_key]
            c_gt_df = predictions[c_pred_key.replace('pred', 'gt')]
            n_gt_df = predictions[n_pred_key.replace('pred', 'gt')]
            p_gt_df = predictions[p_pred_key.replace('pred', 'gt')]
            
            # Extract PFT columns
            def get_pft_cols(df):
                pft_cols = [c for c in df.columns if 'pft' in str(c).lower()]
                if not pft_cols:
                    pft_cols = [c for c in df.columns if c.lower() not in ['longitude', 'latitude', 'lon', 'lat', 'unnamed: 0', 'index']]
                def pft_sort_key(col):
                    import re
                    match = re.search(r'pft(\d+)', str(col).lower())
                    return int(match.group(1)) if match else 999
                return sorted(pft_cols, key=pft_sort_key)
            
            c_pft_cols = get_pft_cols(c_pred_df)
            n_pft_cols = get_pft_cols(n_pred_df)
            p_pft_cols = get_pft_cols(p_pred_df)
            
            c_pred = c_pred_df[c_pft_cols].values
            n_pred = n_pred_df[n_pft_cols].values
            p_pred = p_pred_df[p_pft_cols].values
            c_gt = c_gt_df[c_pft_cols].values
            n_gt = n_gt_df[n_pft_cols].values
            p_gt = p_gt_df[p_pft_cols].values
            
            n_pfts = c_pred.shape[1] if len(c_pred.shape) > 1 else 1
            
            cn_ratio = FROOTCN[1:min(n_pfts+1, len(FROOTCN))]
            if len(cn_ratio) < n_pfts:
                cn_ratio = np.pad(cn_ratio, (0, n_pfts - len(cn_ratio)), constant_values=42.0)
            cn_ratio = np.array(cn_ratio).reshape(1, -1) if len(cn_ratio.shape) == 1 else cn_ratio
            
            results = check_pft_1d_ratio(
                c_pred, n_pred, p_pred, c_gt, n_gt, p_gt,
                cn_ratio, FROOTCP, None, 'frootc'
            )
            validation_results['frootc'] = results
            logger.info(f"frootc CN ratio error: {results.get('cn_ratio_rel_error_pred', 'N/A'):.4f}")
    
    # Validate soil 2D variables
    logger.info("Validating soil 2D variables...")
    
    # Soil layer 1
    if 'soil1c_vr' in soil_2d_vars:
        c_pred_key = f'soil_2d_soil1c_vr_pred'
        n_pred_key = f'soil_2d_soil1n_vr_pred'
        p_pred_key = f'soil_2d_soil1p_vr_pred'
        
        if all(k in predictions for k in [c_pred_key, n_pred_key, p_pred_key]):
            c_pred = predictions[c_pred_key].values
            n_pred = predictions[n_pred_key].values
            p_pred = predictions[p_pred_key].values
            c_gt = predictions[c_pred_key.replace('pred', 'gt')].values
            n_gt = predictions[n_pred_key.replace('pred', 'gt')].values
            p_gt = predictions[p_pred_key.replace('pred', 'gt')].values
            
            results = check_soil_2d_ratio(
                c_pred, n_pred, p_pred, c_gt, n_gt, p_gt,
                SOIL1_CN, SOIL1_CP, 'soil1c_vr'
            )
            validation_results['soil1c_vr'] = results
            logger.info(f"soil1c_vr CN ratio error: {results.get('cn_ratio_rel_error_pred', 'N/A'):.4f}")
            logger.info(f"soil1c_vr CP ratio error: {results.get('cp_ratio_rel_error_pred', 'N/A'):.4f}")
    
    # Save results
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(validation_results, f, indent=2)
        logger.info(f"Validation results saved to {output_file}")
    
    return validation_results


def main():
    parser = argparse.ArgumentParser(description='Validate CNP ratios in predictions')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='Path to results directory containing predictions')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file for validation results')
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return
    
    output_file = Path(args.output) if args.output else results_dir / 'analysis' / 'cnp_ratio_validation.json'
    
    validation_results = validate_cnp_ratios(results_dir, output_file)
    
    # Print summary
    print("\n" + "="*80)
    print("CNP Ratio Validation Summary")
    print("="*80)
    for var_name, results in validation_results.items():
        print(f"\n{var_name}:")
        if 'cn_ratio_rel_error_pred' in results:
            print(f"  CN Ratio Relative Error: {results['cn_ratio_rel_error_pred']:.4f}")
        if 'cp_ratio_rel_error_pred' in results:
            print(f"  CP Ratio Relative Error: {results['cp_ratio_rel_error_pred']:.4f}")
        if 'cn_ratio_rmse_pred' in results:
            print(f"  CN Ratio RMSE: {results['cn_ratio_rmse_pred']:.6f}")
        if 'cp_ratio_rmse_pred' in results:
            print(f"  CP Ratio RMSE: {results['cp_ratio_rmse_pred']:.6f}")


if __name__ == '__main__':
    main()
