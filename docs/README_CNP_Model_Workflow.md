# CNP Model Workflow: Complete Training to Validation Pipeline

## Overview

This document describes the complete workflow for training, validating, and comparing the CNP (Carbon-Nitrogen-Phosphorus) AI model against ELM (Ecosystem Land Model) simulation results. The pipeline includes data preprocessing, model training, inference, validation, and comprehensive comparison with ground truth data.

## Table of Contents

1. [Workflow Overview](#workflow-overview)
2. [File Structure](#file-structure)
3. [Prerequisites](#prerequisites)
4. [Step-by-Step Workflow](#step-by-step-workflow)
5. [Data Normalization & Denormalization](#data-normalization--denormalization)
6. [Validation & Comparison](#validation--comparison)
7. [Output Files](#output-files)
8. [Troubleshooting](#troubleshooting)
9. [Technical Details](#technical-details)

## Workflow Overview

The CNP model workflow consists of eight main stages:

```
1. Configuration → 2. Training → 3. Validation → 4. Inference → 5. NetCDF Export → 6. Model Comparison → 7. Restart File Ingestion → 8. Restart File Verification
```

**Stage 1**: Create variable list configuration (`CNP_IO_demo1.txt`)
**Stage 2**: Train CNP model with specified variables and parameters
**Stage 3**: Validate model performance on test dataset
**Stage 4**: Run inference on entire dataset
**Stage 5**: Export AI predictions to NetCDF format
**Stage 6**: Compare AI predictions with ELM model results
**Stage 7**: Ingest AI predictions into model restart file  
**Stage 8**: Verify and compare AI-enhanced restart file with original

## File Structure

```
cnp_results/run_YYYYMMDD_HHMMSS/
├── cnp_config.json                   # Training configuration for this run
├── cnp_training_YYYYMMDD_HHMMSS.log  # Training logs
├── cnp_training_losses.csv           # Training loss history
├── cnp_metrics.json                  # Model performance metrics
├── cnp_predictions/                  # Training predictions and ground truth
│   ├── model.pth                     # Trained model weights
│   ├── scalers/                      # Saved scalers and metadata
│   │   ├── scaler_metadata.json      # Summary of saved scalers
│   │   ├── time_series_scaler.pkl    # (if group scalers saved)
│   │   ├── static_scaler.pkl         # (if group scalers saved)
│   │   ├── pft_param_scaler.pkl      # (if group scalers saved)
│   │   ├── scalar/                   # Individual scalar scalers
│   │   ├── y_scalar/                 # Individual Y_scalar scalers
│   │   ├── pft_1d/                   # Individual PFT1D scalers
│   │   ├── y_pft_1d/                 # Individual Y_PFT1D scalers
│   │   ├── soil_2d/                  # Individual Soil2D scalers (per layer)
│   │   └── y_soil_2d/                # Individual Y_Soil2D scalers (per layer)
│   ├── predictions_scalar.csv        # Scalar predictions (denormed when saving)
│   ├── ground_truth_scalar.csv       # Scalar ground truth (denormed when saving)
│   ├── pft_1d_predictions/           # Per-variable PFT1D predictions CSVs
│   ├── pft_1d_ground_truth/          # Per-variable PFT1D ground truth CSVs
│   ├── soil_2d_predictions/          # Per-variable Soil2D predictions CSVs
│   │   └── predictions_Y_<var>.csv   # Columns: Y_<var>_col1_layer1..layer10 (1×10)
│   ├── soil_2d_ground_truth/         # Per-variable Soil2D ground truth CSVs
│   │   └── ground_truth_Y_<var>.csv  # Columns: Y_<var>_col1_layer1..layer10 (1×10)
│   └── test_static_inverse.csv       # Denormalized static features for test set
├── cnp_inference_entire_dataset/     # Full dataset inference results
│   ├── inference_config.json         # Inference configuration
│   └── cnp_predictions/              # Inference predictions (same structure as above)
├── comparison_results/               # Comparison data preparation
│   └── ai_predictions_for_plotting.nc  # AI predictions aggregated to NetCDF
├── ai_model_comparison_plots/        # Comparison visualization outputs
│   └── comparison_CNP_IO_demo1/      # Plots for specific variable list
│       ├── tlai_pft1.png             # PFT variable plots
│       ├── tlai_pft2.png
│       ├── cwdc_vr_lev0.png          # Soil variable plots
│       ├── cwdc_vr_lev4.png
│       └── cwdc_vr_lev9.png
├── updated_restart_CNP_IO_*.nc       # AI-enhanced restart files
├── ai_restart_comparison_plots/      # Restart file verification plots
│   ├── tlai_pft1.png                 # PFT comparison plots
│   ├── cwdc_vr_lev0.png              # Soil layer comparison plots
│   └── ...                           # Additional variable plots
└── plots/                            # Additional training plots
```

## Prerequisites

### Required Python Packages
```bash
pip install torch torchvision torchaudio
pip install xarray numpy matplotlib cartopy scipy
pip install pandas scikit-learn
```

### Required Data
- **Training Dataset**: Enhanced CNP dataset in pickle format
- **ELM Model Results**: NetCDF file with model simulation outputs
- **Variable List Template**: CNP_IO template for variable specification

### System Requirements
- **GPU**: CUDA-compatible GPU (recommended: NVIDIA H100 or equivalent)
- **Memory**: Minimum 32GB RAM, recommended 64GB+
- **Storage**: Sufficient space for model outputs and comparison plots

## Step-by-Step Workflow

### Step 1: Create Variable List Configuration

Create `CNP_IO_demo1.txt` using the CNP_IO template:

```bash
# Copy and modify the CNP_IO template
cp CNP_IO_template.txt CNP_IO_demo1.txt

# Edit the file to specify your variables
# Example variables included:
# - Scalar: GPP, NPP, AR, HR
# - PFT 1D: tlai (Total Leaf Area Index)
# - Soil 2D: cwdc_vr, cwdn_vr, cwdp_vr (Carbon, Nitrogen, Phosphorus)
```

**Variable Categories**:
- **Time Series Variables**: FLDS, PSRF, FSDS, QBOT, PRECTmms, TBOT
- **Static Variables**: Latitude, Longitude, AREA, landfrac, PFT percentages, soil properties
- **PFT Parameters**: 44 PFT-specific parameters (e.g., pft_deadwdcn, pft_frootcn)
- **Scalar Variables**: GPP, NPP, AR, HR (Gross Primary Production, Net Primary Production, etc.)
- **PFT 1D Variables**: tlai (Total Leaf Area Index)
- **Soil 2D Variables**: cwdc_vr, cwdn_vr, cwdp_vr (Carbon, Nitrogen, Phosphorus in soil layers)

### Step 2: Train CNP Model

```bash
python train_cnp_model.py \
    --variable-list CNP_IO_demo1.txt \
    --epoch 100 \
    --batch 128
```

**Training Process**:
- **Data Loading**: Loads enhanced CNP dataset from specified paths
- **Data Preprocessing**: Normalizes all variables using MinMaxScaler
- **Model Architecture**: CNP combined model with LSTM, CNN, and Transformer components
- **Training**: 100 epochs with AdamW optimizer and cosine learning rate scheduler
- **Output**: Trained model, scalers, and training metrics

**Key Training Parameters**:
- **Learning Rate**: 0.0001
- **Batch Size**: 128 (configurable)
- **Epochs**: 100 (configurable)
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing with step size 10

### Step 3: Navigate to Results Directory

```bash
# Navigate to the generated results directory
cd cnp_results/run_YYYYMMDD_HHMMSS
# Example: cd cnp_results/run_20250815_205419
```

### Step 4: Validate Model Performance

```bash
python ../../scripts/cnp_result_validationplot.py .
```

**Validation Process**:
- **Test Dataset**: Uses held-out test data (30% of total dataset)
- **Metrics**: RMSE, NRMSE, R² for each variable type
- **Visualization**: Scatter plots comparing predictions vs. ground truth
- **Output**: Validation plots and performance metrics

**Validation Metrics**:
- **Scalar Variables**: GPP, NPP, AR, HR performance
- **PFT Variables**: tlai performance across 16 PFTs
- **Soil Variables**: cwdc_vr, cwdn_vr, cwdp_vr performance across soil layers

### Step 4.5: Generate Comprehensive Prediction Quality Report

```bash
python ../../scripts/generate_prediction_quality_report.py
```

**Prediction Quality Analysis**:
- **Quality Categorization**: Classifies each prediction as "good", "ok", or "bad" based on statistical thresholds
- **Comprehensive Metrics**: Uses R², relative RMSE, and relative MAE for evaluation
- **Variable-Level Analysis**: Provides detailed quality breakdown for each variable
- **Visualization**: Generates multiple charts including stacked bar charts, pie charts, and scatter plots
- **HTML Report**: Creates an interactive HTML report for easy viewing

**Quality Classification Criteria**:
- **Good**: R² ≥ 0.9, Relative RMSE ≤ 0.1, Relative MAE ≤ 0.1
- **OK**: R² ≥ 0.7, Relative RMSE ≤ 0.25, Relative MAE ≤ 0.25
- **Bad**: Below OK thresholds

**Output Files**:
- `analysis/detailed_quality_assessment.csv`: Full dataset with quality categorization
- `analysis/variable_quality_summary.csv`: Summary statistics for each variable
- `analysis/quality_summary_report.txt`: Text report with overall statistics
- `analysis/prediction_quality_report.html`: Interactive HTML report
- `analysis/prediction_quality_by_variable.png`: Bar chart showing quality distribution
- `analysis/overall_prediction_quality.png`: Pie chart of overall quality
- `analysis/r2_vs_rmse.png`: Scatter plot of R² vs Relative RMSE

**Customizable Parameters**:
```bash
python ../../scripts/generate_prediction_quality_report.py \
  --r2-good 0.85 \
  --rmse-good 0.15 \
  --output-dir custom_analysis
```

**Detailed Documentation**: See `docs/README_prediction_quality.md` for comprehensive usage instructions and advanced features.

### Step 5: Run Inference on Entire Dataset

```bash
python ../../scripts/run_inference_all.py \
    --variable-list ../../CNP_IO_demo1.txt
```

**Inference Process**:
- **Model Loading**: Loads trained model from `cnp_predictions/model.pth`
- **Full Dataset**: Processes entire enhanced CNP dataset
- **Denormalization**: Applies inverse transformation using saved scalers
- **Output**: Creates `cnp_inference_entire_dataset/` folder with predictions

**Inference Outputs**:
- **Scalar Predictions**: GPP, NPP, AR, HR for all gridcells
- **PFT Predictions**: tlai for all PFTs and gridcells
- **Soil Predictions**: cwdc_vr, cwdn_vr, cwdp_vr for all soil layers and gridcells

### Step 6: Export to NetCDF Format

```bash
python ../../scripts/ai_predictions_to_netcdf.py \
    --variable-list ../../CNP_IO_demo1.txt
```

**NetCDF Export Process**:
- **Data Aggregation**: Combines all variable predictions
- **Coordinate System**: Preserves geographic coordinates (grid1d_lon, grid1d_lat)
- **Variable Structure**: Organizes data by variable type and dimensions
- **Output**: `comparison_results/ai_predictions_for_plotting.nc`

**NetCDF Structure**:
- **Dimensions**: gridcell (20,975), pft (16), levgrnd (10)
- **Variables**: All predicted variables with proper coordinate mapping
- **Metadata**: Coordinate system and variable information

### Step 7: Compare with ELM Model Results

```bash
python ../../scripts/ai_model_comparison_plot.py \
    --variable-list ../../CNP_IO_demo1.txt
```

**Comparison Process**:
- **Data Loading**: Loads AI predictions and ELM model results
- **Coordinate Alignment**: Uses model coordinates as master system
- **Spatial Mapping**: Maps AI gridcells to model gridcell positions
- **Visualization**: Generates tri-panel comparison plots

**Output Structure**:
- **Location**: `./ai_model_comparison_plots/comparison_CNP_IO_demo1/`
- **Plot Types**: PFT plots, soil layer plots, scalar variable plots
- **Format**: High-resolution PNG files with statistical information

### Step 8: Ingest AI Predictions into Model Restart File

```bash
python ../../scripts/ai_predictions_to_restart.py \
    --variable-list ../../CNP_IO_demo1.txt
```

**Restart File Ingestion Process**:
- **File Preparation**: Creates copy of original restart file
- **Variable Selection**: Updates only PFT1D and soil2D variables from CNP_IO list
- **Spatial Mapping**: Uses geographic coordinates to map AI data to model gridcells
- **Data Replacement**: 
  - PFT1D: Updates PFT1-PFT16 (skipping PFT0) in first column of each gridcell
  - Soil2D: Updates first column and first 10 layers of each gridcell
- **Output**: `updated_restart_CNP_IO_demo1_*.nc` file

**Key Features**:
- **Direct NetCDF Manipulation**: Bypasses xarray encoding issues
- **Attribute Preservation**: Maintains all original file attributes and metadata
- **Coordinate System**: Uses model coordinates as master for compatibility
- **Selective Updates**: Only modifies specified variables, leaves others unchanged

### Step 9: Verify AI-Enhanced Restart File

```bash
# Option 1: Use AI restart comparison script (recommended)
python ../../scripts/ai_restart_comparison.py \
    --variable-list ../../CNP_IO_demo1.txt \
    --layers 0,5,9 \
    --pfts 1,2,4,5,6,7,8,9

# Option 2: Use general restart variable plot script
python ../../scripts/restart_variable_plot.py \
    --model final \
    --new ./updated_restart_CNP_IO_demo1_*.nc

# Option 3: Manual verification using NetCDF tools
ncdump -h updated_restart_CNP_IO_demo1_*.nc | head -50
ncdump -v tlai updated_restart_CNP_IO_demo1_*.nc | head -20
```

**Verification Process**:
- **Data Integrity**: Confirms AI predictions were correctly ingested
- **Variable Comparison**: Compares AI-enhanced values with original model values
- **Spatial Alignment**: Verifies coordinate system and gridcell mapping
- **Statistical Analysis**: Provides RMSE, NRMSE, R² metrics for each variable

**Output Structure**:
- **Location**: `./ai_restart_comparison_plots/`
- **Plot Types**: Four-panel comparison plots (AI Enhanced, Original Model, Difference, Percent Difference)
- **Variable Coverage**: All PFT1D and soil2D variables from CNP_IO list
- **Layer Selection**: Configurable soil layers (default: 0, 5, 9)
- **PFT Selection**: Configurable PFTs (default: 1,2,4,5,6,7,8,9)

**Verification Metrics**:
- **Spatial Distribution**: Geographic patterns and variability
- **Value Comparison**: Statistical correlation between AI and model
- **Difference Analysis**: Spatial patterns of AI enhancements
- **Percent Difference**: Categorical analysis (0-10%, 10-30%, 30%+)

## Data Normalization & Denormalization

### Normalization Process

**Training Phase**:
1. **Data Loading**: Raw data loaded from enhanced dataset
2. **Scaler Fitting**: MinMaxScaler fitted on training data (70% of dataset)
3. **Data Transformation**: All variables normalized to [0, 1] range
4. **Scaler Persistence**: Scalers saved for inference and denormalization

**Variable-Specific Normalization**:
- **Time Series**: 6 variables (FLDS, PSRF, FSDS, QBOT, PRECTmms, TBOT)
- **Static Variables**: 49 variables (coordinates, PFT percentages, soil properties)
- **PFT Parameters**: 44 variables (PFT-specific parameters)
- **Scalar Variables**: 4 variables (GPP, NPP, AR, HR)
- **PFT 1D Variables**: 1 variable (tlai)
- **Soil 2D Variables**: 3 variables (cwdc_vr, cwdn_vr, cwdp_vr)

### Denormalization Process

**Inference Phase**:
1. **Model Prediction**: AI model outputs normalized predictions
2. **Scaler Loading**: Loads saved scalers for each variable type
3. **Inverse Transformation**: Applies `scaler.inverse_transform()` to restore original scale
4. **Data Export**: Denormalized predictions saved for comparison

**Denormalization Examples**:
```python
# Scalar variables
gpp_denorm = scalar_scaler.inverse_transform(gpp_norm)

# PFT variables  
tlai_denorm = pft_1d_scaler.inverse_transform(tlai_norm)

# Soil variables
cwdc_denorm = soil_2d_scaler.inverse_transform(cwdc_norm)
```

## Validation & Comparison

### Model Validation

**Test Dataset Performance**:
- **Split Ratio**: 70% training, 30% testing
- **Metrics**: RMSE, NRMSE, R² for each variable type
- **Visualization**: Scatter plots with regression lines
- **Statistical Analysis**: Correlation analysis and error distribution

**Performance Targets**:
- **Scalar Variables**: R² > 0.95 (GPP, NPP, AR, HR)
- **PFT Variables**: R² > 0.90 (tlai across PFTs)
- **Soil Variables**: R² > 0.85 (carbon, nitrogen, phosphorus in soil)

### Model Comparison

**AI vs ELM Comparison**:
- **Coordinate System**: Model coordinates as master for ingestion compatibility
- **Spatial Mapping**: Nearest-neighbor mapping from AI to model gridcells
- **Statistical Analysis**: RMSE, NRMSE, R² between AI predictions and model results
- **Visualization**: Tri-panel plots (AI, Model, Difference)

**Comparison Metrics**:
- **Spatial Alignment**: Geographic coordinate correspondence
- **Value Comparison**: Statistical correlation and error metrics
- **Pattern Analysis**: Spatial distribution and variability patterns

## Output Files

### Training Outputs

**Model Files**:
- `model.pth`: Trained PyTorch model weights
- `cnp_config.json`: Complete training configuration
- `resolved_config.json`: Resolved variable lists and parameters

**Scaler Files**:
- `scalers/`: Directory containing all normalization scalers
- `scaler_metadata.json`: Comprehensive scaler information and parameters

**Performance Files**:
- `cnp_training_losses.csv`: Training loss history
- `cnp_metrics.json`: Model performance metrics
- `cnp_training_YYYYMMDD_HHMMSS.log`: Detailed training logs

### Inference Outputs

**Prediction Files**:
- `cnp_inference_entire_dataset/cnp_predictions/`: Full dataset predictions
- `comparison_results/ai_predictions_for_plotting.nc`: NetCDF format predictions

**Validation Files**:
- `cnp_predictions/predictions_*.csv`: Training dataset predictions
- `cnp_predictions/ground_truth_*.csv`: Training dataset ground truth

### Comparison Outputs

**Visualization Files**:
- `ai_model_comparison_plots/comparison_CNP_IO_demo1/`: Comparison plots
- **PFT Plots**: `tlai_pft1.png`, `tlai_pft2.png`, etc.
- **Soil Plots**: `cwdc_vr_lev0.png`, `cwdc_vr_lev4.png`, etc.

### Restart File Outputs

**AI-Enhanced Restart Files**:
- `updated_restart_CNP_IO_demo1_*.nc`: AI-enhanced model restart file
- **File Size**: Same as original restart file (~3.1 GB)
- **Content**: Original restart file with AI predictions integrated
- **Compatibility**: Ready for ELM model simulations

**Restart Verification Plots**:
- `ai_restart_comparison_plots/`: Comprehensive verification plots
- **Four-Panel Layout**: AI Enhanced, Original Model, Difference, Percent Difference
- **Variable Coverage**: All PFT1D and soil2D variables from CNP_IO list
- **Statistical Analysis**: RMSE, NRMSE, R² metrics for each comparison

## Troubleshooting

### Common Issues

#### 1. Memory Errors
**Problem**: CUDA out of memory during training
**Solution**: Reduce batch size, use gradient accumulation, or enable mixed precision

#### 2. Data Loading Issues
**Problem**: Dataset not found or corrupted
**Solution**: Verify data paths, check file permissions, validate data format

#### 3. Coordinate Mismatch
**Problem**: AI and model coordinates don't align
**Solution**: Use coordinate analysis utility, verify coordinate systems

#### 4. Variable Not Found
**Problem**: Variable missing from comparison
**Solution**: Check variable names, ensure both datasets contain required variables

### Debug Information

**Training Logs**:
- Detailed training progress and metrics
- Data loading and preprocessing information
- Model architecture and parameter details

**Validation Outputs**:
- Performance metrics for each variable type
- Error analysis and distribution information
- Data quality and coverage statistics

## Technical Details

### Model Architecture

**CNP Combined Model**:
- **LSTM Encoder**: Processes time series variables (6 features)
- **Surface Encoder**: Processes static variables (49 features)
- **PFT Parameter Encoder**: CNN-based PFT parameter processing (44 parameters × 17 PFTs)
- **Scalar Encoder**: Processes scalar variables (4 features)
- **PFT 1D Encoder**: Processes PFT variables (1 feature)
- **Soil 2D Encoder**: Processes soil variables (3 features × 10 layers)
- **Feature Fusion**: Concatenates all encoder outputs (432 total features)
- **Output Heads**: Separate prediction heads for each variable type

**Model Parameters**:
- **Total Parameters**: ~1.35M trainable parameters
- **Architecture**: Hybrid CNN-Transformer with attention mechanisms
- **Activation**: ReLU and GELU activation functions
- **Regularization**: Dropout (0.1) and weight decay (0.01)

### Data Processing Pipeline

**Input Processing**:
1. **Data Loading**: Multi-file dataset loading with batch processing
2. **Feature Engineering**: Variable selection and preprocessing
3. **Normalization**: MinMaxScaler application to all variables
4. **Data Splitting**: Training/validation/test split with stratification

**Output Processing**:
1. **Model Prediction**: Forward pass through trained model
2. **Denormalization**: Inverse transformation using saved scalers
3. **Data Export**: CSV and NetCDF format outputs
4. **Coordinate Mapping**: Spatial alignment for model comparison

### Performance Optimization

**Training Optimization**:
- **Mixed Precision**: Optional FP16 training for memory efficiency
- **Gradient Accumulation**: Effective larger batch sizes
- **Learning Rate Scheduling**: Cosine annealing for convergence
- **Early Stopping**: Prevents overfitting with patience mechanism

**Inference Optimization**:
- **Batch Processing**: Efficient inference on large datasets
- **Memory Management**: GPU memory optimization
- **Parallel Processing**: Multi-worker data loading

### Restart File Integration

**Data Ingestion Process**:
- **File Copying**: Creates backup of original restart file
- **Direct NetCDF Manipulation**: Uses netCDF4 for low-level data access
- **Spatial Mapping**: Geographic coordinate-based gridcell alignment
- **Selective Variable Updates**: Only modifies CNP_IO-specified variables

**Verification Methods**:
- **Automated Comparison**: AI restart comparison script with statistical analysis
- **General Restart Plotting**: Standard restart variable plot script
- **Manual Inspection**: NetCDF command-line tools for detailed verification
- **Statistical Validation**: RMSE, NRMSE, R² metrics for quality assessment

## Future Enhancements

### Planned Improvements

1. **Advanced Architectures**: Transformer-based models for better sequence modeling
2. **Multi-task Learning**: Joint optimization of multiple variable types
3. **Uncertainty Quantification**: Prediction confidence intervals
4. **Real-time Inference**: Streaming data processing capabilities
5. **Model Compression**: Quantization and pruning for deployment

### Integration Opportunities

1. **ELM Model Integration**: Direct ingestion of AI predictions into restart files
2. **Real-time Monitoring**: Continuous model performance tracking
3. **Automated Retraining**: Trigger-based model updates
4. **Distributed Training**: Multi-GPU and multi-node training
5. **Cloud Deployment**: Scalable inference infrastructure
6. **Restart File Validation**: Automated quality checks and verification
7. **Model Simulation Pipeline**: End-to-end AI-enhanced model runs

---

## Contact & Support

For questions, issues, or contributions to the CNP model workflow:

- **Repository**: [AI4BGC CNP Model Repository]
- **Documentation**: [Technical Documentation]
- **Issues**: [GitHub Issues]
- **Contributions**: [Contributing Guidelines]

---

*Last Updated: August 2025*
*Version: 1.0*
