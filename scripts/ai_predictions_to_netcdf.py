#!/usr/bin/env python3
"""
Convert AI predictions to NetCDF format compatible with restart_variable_plot.py

This script takes AI predictions from the cnp_predictions directory and converts them
to a NetCDF file that can be used with the restart_variable_plot.py script for
creating comparison maps.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
import sys
from typing import Dict, List, Any, Optional

# Project imports
sys.path.append(str(Path(__file__).resolve().parents[1]))
from config.training_config import parse_cnp_io_list


def parse_variable_list_file(variable_list_path: str) -> Dict[str, List[str]]:
    """Parse the CNP IO list file to extract scalar, PFT1D, and soil2D variables."""
    print(f"Parsing variable list file: {variable_list_path}")
    
    variables = {
        'scalar': [],
        'pft1d': [],
        'soil2d': []
    }
    
    try:
        with open(variable_list_path, 'r') as f:
            lines = f.readlines()
        
        current_section = None
        for line in lines:
            line = line.strip()
            
            # Detect sections - handle variations in section names
            if 'SCALAR VARIABLES' in line or 'SCALAR VARIABLES (1D' in line:
                current_section = 'scalar'
                print(f"  Found scalar section: {line}")
                continue
            elif '1D PFT VARIABLES' in line or '1D PFT VARIABLES (' in line:
                current_section = 'pft1d'
                print(f"  Found PFT1D section: {line}")
                continue
            elif '2D VARIABLES' in line or '2D VARIABLES (' in line:
                current_section = 'soil2d'
                print(f"  Found 2D section: {line}")
                continue
            elif 'OUTPUT VARIABLES' in line:
                current_section = None
                continue
            
            # Parse variables in current section
            if current_section and line.startswith('•'):
                # Extract variable names from the line
                # Handle multiple variables per line separated by commas
                var_part = line.replace('•', '').strip()
                var_names = [v.strip() for v in var_part.split(',')]
                
                for var_name in var_names:
                    if var_name:  # Skip empty strings
                        # Remove Y_ prefix if present
                        if var_name.startswith('Y_'):
                            var_name = var_name[2:]
                        
                        if current_section == 'scalar':
                            variables['scalar'].append(var_name)
                        elif current_section == 'pft1d':
                            variables['pft1d'].append(var_name)
                        elif current_section == 'soil2d':
                            variables['soil2d'].append(var_name)
        
        print(f"  Scalar variables: {len(variables['scalar'])}")
        print(f"  PFT1D variables: {len(variables['pft1d'])}")
        print(f"  Soil2D variables: {len(variables['soil2d'])}")
        
        # Print the actual variables found for debugging
        print(f"  Scalar: {variables['scalar']}")
        print(f"  PFT1D: {variables['pft1d']}")
        print(f"  Soil2D: {variables['soil2d']}")
        
        return variables
        
    except Exception as e:
        print(f"Warning: Could not parse variable list file: {e}")
        # Return default variables if parsing fails
        return {
            'scalar': ['GPP', 'NPP', 'AR', 'HR'],
            'pft1d': ['deadcrootc', 'deadcrootn', 'deadcrootp', 'deadstemc', 'deadstemn', 'deadstemp', 'frootc', 'frootc_storage', 'leafc', 'leafc_storage', 'totcolp', 'totlitc', 'totvegc', 'tlai'],
            'soil2d': ['cwdc_vr', 'cwdn_vr', 'cwdp_vr', 'litr1c_vr', 'litr2c_vr', 'litr3c_vr', 'litr1n_vr', 'litr2n_vr', 'litr3n_vr', 'litr1p_vr', 'litr2p_vr', 'litr3p_vr', 'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr', 'soil1c_vr', 'soil1n_vr', 'soil1p_vr', 'soil2c_vr', 'soil2n_vr', 'soil2p_vr', 'soil3c_vr', 'soil3n_vr', 'soil3p_vr', 'soil4c_vr', 'soil4n_vr', 'soil4p_vr']
        }


def _extract_variables_from_config(config_path: Path) -> Optional[Dict[str, List[str]]]:
    """Extract variable lists from a cnp_config.json produced during training.
    Returns a dict with keys: 'scalar', 'pft1d', 'soil2d' or None on failure.
    """
    try:
        with open(config_path, 'r') as f:
            cfg = json.load(f)
        if 'data_info' not in cfg:
            return None
        di = cfg['data_info']
        variables = {
            'scalar': di.get('x_list_scalar_columns', []) or [],
            'pft1d': di.get('variables_1d_pft', []) or [],
            'soil2d': di.get('x_list_columns_2d', []) or []
        }
        # Ensure they are lists of strings
        for k in list(variables.keys()):
            variables[k] = [str(v) for v in variables[k]]
        return variables
    except Exception:
        return None


def _auto_detect_variables_from_predictions(predictions_dir: Path) -> Optional[Dict[str, List[str]]]:
    """Find a nearby cnp_config.json relative to predictions_dir and extract variables.
    Searches predictions_dir and its parents.
    """
    try:
        # Typical structure: .../run_.../cnp_inference_entire_dataset/cnp_predictions
        # We search upward for cnp_config.json
        for parent in [predictions_dir] + list(predictions_dir.parents):
            cfg = parent / 'cnp_config.json'
            if cfg.exists():
                vars_from_cfg = _extract_variables_from_config(cfg)
                if vars_from_cfg:
                    print(f"Auto-detected variables from {cfg}")
                    return vars_from_cfg
        return None
    except Exception:
        return None


def _discover_variables_from_files(predictions_dir: Path) -> Optional[Dict[str, List[str]]]:
    """Fallback: derive variable lists by inspecting prediction CSVs."""
    try:
        result = {'scalar': [], 'pft1d': [], 'soil2d': []}
        # Scalar
        scalar_csv = predictions_dir / 'predictions_scalar.csv'
        if scalar_csv.exists():
            cols = list(pd.read_csv(scalar_csv, nrows=0).columns)
            result['scalar'] = [c[2:] for c in cols if c.startswith('Y_')]
        # PFT 1D
        pft_dir = predictions_dir / 'pft_1d_predictions'
        if pft_dir.exists():
            for p in sorted(pft_dir.glob('predictions_*.csv')):
                var = p.stem.replace('predictions_Y_', '')
                if var:
                    result['pft1d'].append(var)
        # Soil 2D
        soil_dir = predictions_dir / 'soil_2d_predictions'
        if soil_dir.exists():
            for p in sorted(soil_dir.glob('predictions_*.csv')):
                var = p.stem.replace('predictions_Y_', '')
                if var:
                    result['soil2d'].append(var)
        # Ensure at least some variables were discovered
        if any(result[k] for k in result):
            print("Auto-discovered variables from prediction files:")
            print(f"  Scalar: {len(result['scalar'])}")
            print(f"  PFT1D: {len(result['pft1d'])}")
            print(f"  Soil2D: {len(result['soil2d'])}")
            return result
        return None
    except Exception:
        return None


def load_ai_predictions(
    predictions_dir: Path,
    soil_2d_bias_corrected_subdir: Optional[str] = None,
) -> Dict[str, Any]:
    """Load AI model predictions from the predictions directory.

    If soil_2d_bias_corrected_subdir is set (e.g. soil_2d_predictions_5P_bias_corrected_phase2),
    after loading soil from soil_2d_predictions we also load any
    predictions_Y_<var>_bias_corrected.csv from that subdir and overlay them onto preds['soil_2d'],
    so that the NetCDF uses bias-corrected 5P where applied.
    """
    print(f"Loading AI predictions from: {predictions_dir}")
    
    preds = {}
    
    # Load scalar predictions
    scalar_path = predictions_dir / 'predictions_scalar.csv'
    if scalar_path.exists():
        scalar_df = pd.read_csv(scalar_path)
        lon, lat = _extract_coords(scalar_df)
        preds['scalar_coords'] = (lon, lat)
        preds['scalar'] = _drop_coords(scalar_df)
        print(f"  Loaded scalar predictions: {preds['scalar'].shape}")
        # Print sample locations if available
        if 'Longitude' in preds['scalar'] and 'Latitude' in preds['scalar']:
            sample_locs = preds['scalar'][['Longitude', 'Latitude']].drop_duplicates().head(5)
            print(f"  Sample locations (first 5 unique):\n{sample_locs}")
    
    # Load 1D PFT predictions
    pft_dir = predictions_dir / 'pft_1d_predictions'
    if pft_dir.exists():
        preds['pft_1d'] = {}
        preds['pft1d_coords'] = {}
        for p in sorted(pft_dir.glob('predictions_*.csv')):
            # Extract variable name from filename (e.g., predictions_Y_tlai.csv -> tlai)
            var_name = p.stem.replace('predictions_Y_', '')
            df = pd.read_csv(p)
            lon, lat = _extract_coords(df)
            preds['pft1d_coords'][var_name] = (lon, lat)
            preds['pft_1d'][var_name] = _drop_coords(df)
            print(f"  Loaded PFT predictions for {var_name}: {df.shape}")
            # Print sample locations if available
            if 'Longitude' in df and 'Latitude' in df:
                sample_locs = df[['Longitude', 'Latitude']].drop_duplicates().head(3)
                print(f"    Sample locations (first 3 unique):\n{sample_locs}")
    
    # Load 2D soil predictions
    soil_dir = predictions_dir / 'soil_2d_predictions'
    if soil_dir.exists():
        preds['soil_2d'] = {}
        preds['soil2d_coords'] = {}
        for p in sorted(soil_dir.glob('predictions_*.csv')):
            # Skip bias-corrected filenames here; they are loaded from the bias-corrected subdir if given
            if '_bias_corrected' in p.stem:
                continue
            # Extract variable name from filename (e.g., predictions_Y_cwdc_vr.csv -> cwdc_vr)
            var_name = p.stem.replace('predictions_Y_', '')
            df = pd.read_csv(p)
            lon, lat = _extract_coords(df)
            preds['soil2d_coords'][var_name] = (lon, lat)
            preds['soil_2d'][var_name] = _drop_coords(df)
            print(f"  Loaded soil predictions for {var_name}: {df.shape}")
            # Print sample locations if available
            if 'Longitude' in df and 'Latitude' in df:
                sample_locs = df[['Longitude', 'Latitude']].drop_duplicates().head(3)
                print(f"    Sample locations (first 3 unique):\n{sample_locs}")
    # Overlay bias-corrected soil 2D predictions if subdir provided (e.g. 5P from apply_5p_bias_scale_correction.py)
    if soil_2d_bias_corrected_subdir:
        bias_dir = predictions_dir / soil_2d_bias_corrected_subdir
        if bias_dir.exists():
            if 'soil_2d' not in preds:
                preds['soil_2d'] = {}
                preds['soil2d_coords'] = {}
            for p in sorted(bias_dir.glob('predictions_Y_*_bias_corrected.csv')):
                # predictions_Y_<var>_bias_corrected.csv -> <var>
                var_name = p.stem.replace('predictions_Y_', '').replace('_bias_corrected', '')
                df = pd.read_csv(p)
                lon, lat = _extract_coords(df)
                preds['soil2d_coords'][var_name] = (lon, lat)
                preds['soil_2d'][var_name] = _drop_coords(df)
                print(f"  Overlaid bias-corrected soil predictions for {var_name}: {df.shape}")
        else:
            print(f"  Warning: soil 2D bias-corrected subdir not found: {bias_dir}")
    
    # Load static inverse mapping for coordinates
    static_inv = predictions_dir / 'test_static_inverse.csv'
    if static_inv.exists():
        static_df = pd.read_csv(static_inv)
        preds['test_static_inverse'] = static_df
        print(f"  Loaded static inverse mapping: {static_df.shape}")
        # Print sample locations
        sample_locs = static_df[['Longitude', 'Latitude']].drop_duplicates().head(5)
        print(f"  Sample locations from static inverse (first 5 unique):\n{sample_locs}")
    
    return preds

def _select_base_coords(ai_preds: Dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Select base coordinates from CSV-derived predictions."""
    # Prefer scalar coords if available (should be full grid)
    if 'scalar_coords' in ai_preds and ai_preds['scalar_coords'][0] is not None:
        return ai_preds['scalar_coords']
    # Fall back to any soil2d/pft1d coords
    if 'soil2d_coords' in ai_preds and ai_preds['soil2d_coords']:
        _, coords = next(iter(ai_preds['soil2d_coords'].items()))
        if coords[0] is not None:
            return coords
    if 'pft1d_coords' in ai_preds and ai_preds['pft1d_coords']:
        _, coords = next(iter(ai_preds['pft1d_coords'].items()))
        if coords[0] is not None:
            return coords
    raise ValueError("No CSV coordinates found in predictions (Longitude/Latitude columns missing?)")


def create_netcdf_structure(ai_preds: Dict[str, Any], variable_list: Dict[str, List[str]],
                           output_path: Path) -> xr.Dataset:
    """Create NetCDF structure compatible with restart_variable_plot.py."""
    print("Creating NetCDF structure...")

    lon, lat = _select_base_coords(ai_preds)
    n_samples = len(lon)

    # Create coordinate variables
    coords = {
        'gridcell': np.arange(n_samples),
        'pft': np.arange(1, 17),  # PFT1-16 (excluding PFT0)
        'column': np.arange(1),    # Only first column
        'levgrnd': np.arange(10),  # Only first 10 layers
    }

    # Create the dataset
    ds = xr.Dataset(coords=coords)

    # Add coordinate variables that restart_variable_plot.py expects
    # Grid coordinates: 1D arrays for gridcell dimension
    ds['grid1d_lon'] = xr.DataArray(lon, dims=['gridcell'])
    ds['grid1d_lat'] = xr.DataArray(lat, dims=['gridcell'])

    # PFT coordinates: For PFT variables, we need to create proper mapping
    # Each PFT gets assigned to the first gridcell (index 1, 1-based)
    pft_lon = np.full(16, float(lon[0]), dtype=float)
    pft_lat = np.full(16, float(lat[0]), dtype=float)
    ds['pfts1d_lon'] = xr.DataArray(pft_lon, dims=['pft'])
    ds['pfts1d_lat'] = xr.DataArray(pft_lat, dims=['pft'])

    # Column coordinates: For column variables, we need to create proper mapping
    # Each column gets assigned to the first gridcell (index 1, 1-based)
    col_lon = np.full(1, float(lon[0]), dtype=float)
    col_lat = np.full(1, float(lat[0]), dtype=float)
    ds['cols1d_lon'] = xr.DataArray(col_lon, dims=['column'])
    ds['cols1d_lat'] = xr.DataArray(col_lat, dims=['column'])

    # Add gridcell indices (1-based as expected by restart_variable_plot.py)
    # All PFTs and columns are assigned to gridcell 1
    ds['pfts1d_gridcell_index'] = xr.DataArray(np.ones(16, dtype=int), dims=['pft'])
    ds['cols1d_gridcell_index'] = xr.DataArray(np.ones(1, dtype=int), dims=['column'])

    print(f"  Created base structure with {n_samples} gridcells")
    print(f"  PFT coordinates: {pft_lon.shape} (all assigned to first gridcell)")
    print(f"  Column coordinates: {col_lon.shape} (all assigned to first gridcell)")
    return ds


def add_scalar_variables(ds: xr.Dataset, ai_preds: Dict[str, Any], 
                        variable_list: Dict[str, List[str]]) -> None:
    """Add scalar variables to the dataset."""
    if 'scalar' not in ai_preds:
        return
    
    print("Adding scalar variables...")
    scalar_df = ai_preds['scalar']
    
    for var_name in variable_list['scalar']:
        # Look for column with Y_ prefix
        col_name = f'Y_{var_name}'
        if col_name in scalar_df.columns:
            values = scalar_df[col_name].values
            ds[var_name] = xr.DataArray(values, dims=['gridcell'])
            print(f"  Added {var_name}: {values.shape}")


def add_pft_variables(ds: xr.Dataset, ai_preds: Dict[str, Any], 
                     variable_list: Dict[str, List[str]]) -> None:
    """Add PFT variables to the dataset."""
    if 'pft_1d' not in ai_preds:
        print("  Warning: No PFT predictions found in ai_preds")
        print(f"    Available keys: {list(ai_preds.keys())}")
        return
    
    print("Adding PFT variables...")
    print(f"  PFT variables to add: {variable_list['pft1d']}")
    print(f"  Available PFT predictions: {list(ai_preds['pft_1d'].keys())}")
    
    for var_name in variable_list['pft1d']:
        print(f"    Processing PFT variable: {var_name}")
        if var_name in ai_preds['pft_1d']:
            pft_df = ai_preds['pft_1d'][var_name]
            print(f"      Found PFT dataframe: {pft_df.shape}")
            print(f"      Columns: {list(pft_df.columns)[:5]}...")  # Show first 5 columns
            
            # PFT variables should have columns like Y_leafc_pft1, Y_leafc_pft2, etc.
            pft_cols = [c for c in pft_df.columns if c.startswith(f'Y_{var_name}_pft')]
            print(f"      PFT columns starting with Y_{var_name}_pft: {len(pft_cols)}")
            
            if pft_cols:
                # Sort PFT columns numerically
                def pft_num(col):
                    try:
                        return int(col.split('_pft')[-1])
                    except:
                        return 999
                
                pft_cols = sorted(pft_cols, key=pft_num)
                
                # Create array with shape (pft, gridcell)
                # This matches what restart_variable_plot.py expects for PFT variables
                pft_data = np.zeros((16, len(pft_df)), dtype=float)
                
                for i, col in enumerate(pft_cols):
                    if i < 16:  # PFT1-16
                        pft_data[i, :] = pft_df[col].values
                
                ds[var_name] = xr.DataArray(pft_data, dims=['pft', 'gridcell'])
                print(f"      Added {var_name}: {pft_data.shape}")
                print(f"      PFT columns found: {len(pft_cols)}")
                print(f"      PFT range: {pft_cols[0]} to {pft_cols[-1]}")
            else:
                print(f"      Warning: No PFT columns found for {var_name}")
                print(f"      Available columns: {list(pft_df.columns)[:10]}...")
        else:
            print(f"      Warning: {var_name} not found in PFT predictions")
            print(f"      Available PFT variables: {list(ai_preds['pft_1d'].keys())}")


def add_soil_variables(ds: xr.Dataset, ai_preds: Dict[str, Any], 
                      variable_list: Dict[str, List[str]]) -> None:
    """Add soil 2D variables to the dataset."""
    if 'soil_2d' not in ai_preds:
        print("  Warning: No soil predictions found in ai_preds")
        print(f"    Available keys: {list(ai_preds.keys())}")
        return
    
    print("Adding soil 2D variables...")
    print(f"  Soil variables to add: {variable_list['soil2d']}")
    print(f"  Available soil predictions: {list(ai_preds['soil_2d'].keys())}")
    
    for var_name in variable_list['soil2d']:
        print(f"    Processing soil variable: {var_name}")
        if var_name in ai_preds['soil_2d']:
            soil_df = ai_preds['soil_2d'][var_name]
            # Only use first 10 columns (after dropping long/lat)
            layer_cols = list(soil_df.columns)[:10]
            print(f"      Using columns for layers: {layer_cols}")
            soil_data = np.zeros((1, 10, len(soil_df)), dtype=float)
            for i, col in enumerate(layer_cols):
                soil_data[0, i, :] = soil_df[col].values
            ds[var_name] = xr.DataArray(soil_data, dims=['column', 'levgrnd', 'gridcell'])
            print(f"      Added {var_name}: {soil_data.shape}")
            print(f"      First column layer columns found: {len(layer_cols)}")
            print(f"      Layer range: {layer_cols[0]} to {layer_cols[-1]}")
        else:
            print(f"      Warning: {var_name} not found in soil predictions")
            print(f"      Available soil variables: {list(ai_preds['soil_2d'].keys())}")


def _extract_coords(df):
    for col in ['Longitude', 'Long', 'long']:
        if col in df.columns:
            lon = df[col].values
            break
    else:
        lon = None
    for col in ['Latitude', 'Lat', 'lat']:
        if col in df.columns:
            lat = df[col].values
            break
    else:
        lat = None
    return lon, lat

def _drop_coords(df):
    for col in ['Longitude', 'Long', 'long', 'Latitude', 'Lat', 'lat']:
        if col in df.columns:
            df = df.drop(columns=[col])
    return df

def _check_coords_match(coords1, coords2, label1, label2):
    if coords1[0] is None or coords1[1] is None or coords2[0] is None or coords2[1] is None:
        print(f"  Warning: Missing coordinates for {label1} or {label2}")
        return
    if not (np.allclose(coords1[0], coords2[0]) and np.allclose(coords1[1], coords2[1])):
        raise ValueError(f"Coordinate mismatch between {label1} and {label2}")


def _wrap_coords(lon: np.ndarray) -> np.ndarray:
    """Wrap longitudes from 0-360 to -180-180."""
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0


def main():
    parser = argparse.ArgumentParser(
        description='Convert AI predictions to NetCDF format compatible with restart_variable_plot.py'
    )
    parser.add_argument('--ai-predictions', default='cnp_inference_entire_dataset/cnp_predictions',
                       help='Path to AI predictions directory (cnp_predictions)')
    parser.add_argument('--variable-list', required=False,
                       help='Path to CNP_IO_list file (optional; will auto-detect from nearby cnp_config.json or CSVs)')
    parser.add_argument('--output', default='comparison_results/ai_predictions_for_plotting.nc',
                       help='Output NetCDF file path')
    parser.add_argument('--examples', action='store_true', 
                       help='Show example usage and exit')
    parser.add_argument('--wrap-longitude', action='store_true', default=False,
                       help='Wrap longitudes from 0–360 to -180–180 (default: on). Use --no-wrap-longitude to disable if shell supports).')
    parser.add_argument('--soil-2d-bias-corrected-subdir', default=None,
                       help='Subdir under --ai-predictions containing bias-corrected soil 2D CSVs '
                            '(e.g. soil_2d_predictions_5P_bias_corrected_phase2 from apply_5p_bias_scale_correction.py). '
                            'Files named predictions_Y_<var>_bias_corrected.csv will overlay raw soil 2D for those variables.')
    
    args = parser.parse_args()
    
    if args.examples:
        print("""
Examples:
  # Convert AI predictions to NetCDF
  python ai_predictions_to_netcdf.py \\
    --ai-predictions cnp_results/run_20250814_193455/cnp_predictions \\
    --variable-list CNP_IO_list1.txt \\
    --output ai_predictions.nc

  # Use with restart_variable_plot.py
  python restart_variable_plot.py  # Edit FILE_NEW to point to ai_predictions.nc
        """)
        return
    
    # Validate inputs
    ai_predictions_dir = Path(args.ai_predictions)
    if not ai_predictions_dir.exists():
        parser.error(f'AI predictions directory not found: {ai_predictions_dir}')
    
    variable_list_path = args.variable_list
    if variable_list_path and not Path(variable_list_path).exists():
        parser.error(f'Variable list file not found: {variable_list_path}')
    
    output_path = Path(args.output)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("AI Predictions to NetCDF Converter")
    print("=" * 60)
    print(f"AI predictions: {ai_predictions_dir}")
    print(f"Variable list: {variable_list_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Parse or auto-detect variable list
    if args.variable_list:
        variable_list = parse_variable_list_file(variable_list_path)
    else:
        # Try auto-detect via training config near predictions dir
        variable_list = _auto_detect_variables_from_predictions(ai_predictions_dir)
        if not variable_list:
            # Fallback: infer from prediction file names/headers
            variable_list = _discover_variables_from_files(ai_predictions_dir)
        if not variable_list:
            parser.error('Could not auto-detect variables. Provide --variable-list to proceed.')
    
    # Load AI predictions (optionally overlay 5P bias-corrected from apply_5p_bias_scale_correction.py)
    ai_preds = load_ai_predictions(
        ai_predictions_dir,
        soil_2d_bias_corrected_subdir=getattr(args, 'soil_2d_bias_corrected_subdir', None),
    )

    # Report current lon/lat ranges from CSV coordinates
    try:
        base_lon, base_lat = _select_base_coords(ai_preds)
        print(f"  Before wrapping - grid1d_lon min/max: {float(np.min(base_lon))}, {float(np.max(base_lon))}")
        print(f"  Before wrapping - grid1d_lat min/max: {float(np.min(base_lat))}, {float(np.max(base_lat))}")
    except Exception as _e:
        print(f"  Warning: Failed to compute pre-wrap lon/lat ranges: {_e}")

    # Optional longitude wrapping 0–360 -> -180–180
    if getattr(args, 'wrap_longitude', False):
        try:
            if 'scalar_coords' in ai_preds and ai_preds['scalar_coords'][0] is not None:
                lon, lat = ai_preds['scalar_coords']
                ai_preds['scalar_coords'] = (_wrap_coords(lon), lat)
            if 'pft1d_coords' in ai_preds:
                for k, coords in ai_preds['pft1d_coords'].items():
                    if coords[0] is not None:
                        ai_preds['pft1d_coords'][k] = (_wrap_coords(coords[0]), coords[1])
            if 'soil2d_coords' in ai_preds:
                for k, coords in ai_preds['soil2d_coords'].items():
                    if coords[0] is not None:
                        ai_preds['soil2d_coords'][k] = (_wrap_coords(coords[0]), coords[1])
            base_lon, base_lat = _select_base_coords(ai_preds)
            print(f"  After wrapping - grid1d_lon min/max: {float(np.min(base_lon))}, {float(np.max(base_lon))}")
            print(f"  Latitude min/max: {float(np.min(base_lat))}, {float(np.max(base_lat))}")
        except Exception as _e:
            print(f"  Warning: Failed to wrap longitudes: {_e}")
    
    # Check coordinate matching
    if 'scalar_coords' in ai_preds and 'pft1d_coords' in ai_preds:
        for var, coords in ai_preds['pft1d_coords'].items():
            _check_coords_match(ai_preds['scalar_coords'], coords, 'scalar', f'pft1d:{var}')
    if 'scalar_coords' in ai_preds and 'soil2d_coords' in ai_preds:
        for var, coords in ai_preds['soil2d_coords'].items():
            _check_coords_match(ai_preds['scalar_coords'], coords, 'scalar', f'soil2d:{var}')

    # Create NetCDF structure
    ds = create_netcdf_structure(ai_preds, variable_list, output_path)
    
    # Add variables
    add_scalar_variables(ds, ai_preds, variable_list)
    add_pft_variables(ds, ai_preds, variable_list)
    add_soil_variables(ds, ai_preds, variable_list)
    
    # Save to NetCDF
    print(f"\nSaving NetCDF file to: {output_path}")
    ds.to_netcdf(output_path)
    
    print("\nNetCDF file created successfully!")
    print(f"You can now use this file with restart_variable_plot.py")
    print(f"Edit the FILE_NEW variable in restart_variable_plot.py to point to: {output_path}")
    
    # Print summary
    print(f"\nDataset summary:")
    print(f"  Gridcells: {ds.sizes['gridcell']}")
    print(f"  PFTs: {ds.sizes['pft']} (PFT1-16)")
    print(f"  Columns: {ds.sizes['column']} (first column only)")
    print(f"  Soil layers: {ds.sizes['levgrnd']} (first 10 layers only)")
    print(f"  Variables: {len(ds.data_vars)}")
    
    # Print variable details
    print(f"\nVariable details:")
    for var_name, var_data in ds.data_vars.items():
        if var_name not in ['grid1d_lon', 'grid1d_lat', 'pfts1d_lon', 'pfts1d_lat', 
                           'cols1d_lon', 'cols1d_lat', 'pfts1d_gridcell_index', 'cols1d_gridcell_index']:
            print(f"  {var_name}: {var_data.dims} {var_data.shape}")
    
    # Print coordinate details
    print(f"\nCoordinate details:")
    for coord_name, coord_data in ds.coords.items():
        print(f"  {coord_name}: {coord_data.dims} {coord_data.shape}")


if __name__ == '__main__':
    main()
