# Single Point Restart File Extraction

This guide shows how to extract single point restart files:
1. How to Get Single Point 20-year Ad-Spin Up Restart File
2. How to Get Single Point AI-Updated Ad-Spin Up Restart File
3. How to Validate
## Target Site Coordinates

The Amazon site coordinates in TRENDY coordinate system:
- **Longitude**: 303.75°
- **Latitude**: -17.4246°

## 1. How to Get Single Point 20-year Ad-Spin Up Restart File

Extract single point from the original 20-year ad-spin up restart file:

```bash
python scripts/extract_elm_restart_point.py \
  --restart-file <path/to/input_20year_ad_spinup_restart_file.nc> \
  --lat -17.4246 \
  --lon 303.75 \
  --output-file <path/to/output_single_point_file.nc>
```

This command extracts the single point data and saves it to the specified output directory.

## 2. How to Get Single Point AI-Updated Ad-Spin Up Restart File

**Step 1:** Follow `CNP_pipeline_runbook.md` to train a model and generate an AI-updated ad-spin up restart file across all sites.

**Step 2:** Extract single point from the AI-updated ad-spin up restart file:

```bash
python scripts/extract_elm_restart_point.py \
  --restart-file <path/to/AI-updated_input_20year_ad_spinup_restart_file.nc> \
  --lat -17.4246 \
  --lon 303.75 \
  --output-file <path/to/output_AI-updated_single_point_file.nc>
```

This command extracts the single point data from the AI-predicted restart file and saves it to the specified output directory.

## 3. How to Validate

Compare the original single point restart file with the AI-updated single point restart file to validate the differences:

```bash
python3 scripts/compare_nc2.py \
  --file_name1 <path/to/output_single_point_file.nc> \
  --file_name2 <path/to/output_AI-updated_single_point_file.nc>
```

This command compares the two files and reports differences in variables, data types, shapes, and values.
