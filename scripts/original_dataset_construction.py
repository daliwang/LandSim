import netCDF4 as nc
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

file_path1 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/surfdata_0.9x1.25_hist_1700_17pfts_c240731.nc'
file_path2 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/20250117_trendytest_ICB1850CNRDCTCBC_ad_spinup.elm.h0.0021-01-01-00000.nc'
file_path3 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/20250117_trendytest_ICB1850CNPRDCTCBC.elm.h0.0401-01-01-00000.nc'
file_path4 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_FLDS_1901-2023_z01.nc'
file_path5 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_PSRF_1901-2023_z01.nc'
file_path6 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_FSDS_1901-2023_z01.nc'
file_path7 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_QBOT_1901-2023_z01.nc'
file_path8 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_PRECTmms_1901-2023_z01.nc'
file_path9 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data/forcings/crujra.v2.5.5d_TBOT_1901-2023_z01.nc'
file_path10 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/20250117_trendytest_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc'
file_path11 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0401-01-01-00000.nc'

file_path12 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.h0.0701-01-01-00000.nc'
file_path13 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.h0.0721-01-01-00000.nc'
file_path14 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.h0.0741-01-01-00000.nc'
file_path15 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.h0.0761-01-01-00000.nc'
file_path16 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.h0.0781-01-01-00000.nc'
file_path17 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0701-01-01-00000.nc'
file_path18 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0721-01-01-00000.nc'
file_path19 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0741-01-01-00000.nc'
file_path20 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0761-01-01-00000.nc'
file_path21 = '/home/UNT/dg0997/all_gdw/0_oak_weather/dataset/ornl_data_700/output/20250117_trendytest_ICB1850CNPRDCTCBC.elm.r.0781-01-01-00000.nc'


lat1, lon1 = 90, 0
lat2, lon2 = -90, 360

ds1 = nc.Dataset(file_path1)
ds2 = nc.Dataset(file_path2)
ds3 = nc.Dataset(file_path3)
ds4 = nc.Dataset(file_path4)
ds5 = nc.Dataset(file_path5)
ds6 = nc.Dataset(file_path6)
ds7 = nc.Dataset(file_path7)
ds8 = nc.Dataset(file_path8)
ds9 = nc.Dataset(file_path9)
ds10 = nc.Dataset(file_path10)
ds11 = nc.Dataset(file_path11)

ds_h0_list = [nc.Dataset(fp) for fp in [file_path12, file_path13, file_path14, file_path15, file_path16]]
ds_r_list  = [nc.Dataset(fp) for fp in [file_path17, file_path18, file_path19, file_path20, file_path21]]


lats = ds2.variables['lat'][:]
lons = ds2.variables['lon'][:]
landmask = ds2.variables['landmask'][:]

lat_indices = np.where((lats >= lat2) & (lats <= lat1))[0]
lon_indices = np.where((lons >= lon1) & (lons <= lon2))[0]

filtered_coordinates = [(i, j) for i in lat_indices for j in lon_indices if landmask[i, j] == 1]
print("Number of filtered coordinates with landmask=1:", len(filtered_coordinates))

batch_size = 1000
save_path = "training_data_landfrac_batched.pkl"
batch_number = 1


print("Pre-calculating indices, please wait...")

query_coords = np.array([(lats[i], lons[j]) for i, j in filtered_coordinates])

gridcell_lat = ds10.variables['grid1d_lat'][:]
gridcell_lon = ds10.variables['grid1d_lon'][:]
restart_grid_coords = np.vstack((gridcell_lat, gridcell_lon)).T

print("Building KDTree for restart file grid...")
restart_tree = cKDTree(restart_grid_coords)

_, all_restart_indices = restart_tree.query(query_coords, k=1)
print("Index calculation for restart file grid is complete.")

lats4 = ds4.variables['LATIXY'][:]
lons4 = ds4.variables['LONGXY'][:]
forcing_grid_coords = np.vstack((lats4, lons4)).T

print("Building KDTree for meteorological forcing field grid...")
forcing_tree = cKDTree(forcing_grid_coords)

_, all_forcing_indices = forcing_tree.query(query_coords, k=1)
print("Index calculation for meteorological forcing field grid is complete.")

print("Starting final optimized data preloading...")

pft_based_vars = [
    'totvegc', 'deadstemn', 'deadcrootn', 'deadstemp', 'deadcrootp',
    'leafc', 'leafc_storage', 'frootc', 'frootc_storage',
    'deadcrootc', 'deadstemc', 'tlai',
    # 'H2OCAN', 'T_VEG', 'T10_VALUE'
    'leafn', 'leafn_storage', 'frootn','frootn_storage',
    'leafp', 'leafp_storage', 'frootp','frootp_storage',
    'livestemc', 'livestemc_storage', 
    'livestemn', 'livestemn_storage', 
    'livestemp', 'livestemp_storage',

    'deadcrootc_storage', 'deadstemc_storage', 
    'livecrootc', 'livecrootc_storage', 
    'deadcrootn_storage', 'deadstemn_storage', 
    'livecrootn', 'livecrootn_storage', 
    'deadcrootp_storage', 'deadstemp_storage', 
    'livecrootp', 'livecrootp_storage'
]

col_based_1d_vars = ['cwdp', 'totcolp', 'totlitc',
                    #  'H2OSFC', 'H2OSNO', 'TH2OSFC', 'T_GRND', 'T_GRND_R', 'T_GRND_U'
                     ]

col_based_2d_vars = [
    'cwdn_vr', 'secondp_vr', 'cwdp_vr', 'soil3c_vr', 'soil4c_vr', 'cwdc_vr',
    'soil1c_vr', 'soil1n_vr', 'soil1p_vr',
    'soil2c_vr', 'soil2n_vr', 'soil2p_vr',
    'soil3n_vr', 'soil3p_vr',
    'soil4n_vr', 'soil4p_vr',
    'litr1c_vr', 'litr2c_vr', 'litr3c_vr',
    'litr1n_vr', 'litr2n_vr', 'litr3n_vr',
    'litr1p_vr', 'litr2p_vr', 'litr3p_vr',
    'sminn_vr', 'smin_no3_vr', 'smin_nh4_vr',
    # 'H2OSOI_LIQ', 'H2OSOI_ICE', 'T_SOISNO', 'LAKE_SOILC', 'T_LAKE'
    'labilep_vr', 'occlp_vr', 'primp_vr'
    # 'LABILEP_vr', 'OCCLP_vr', 'PRIMP_vr'
]

# landunit_based_vars = [
#     'taf'
# ]

# topounit_based_vars = [
#     'TS_TOPO'
# ]

all_x_vars = pft_based_vars + col_based_1d_vars + col_based_2d_vars 

x_values = {}
print("--- Loading X variables ---")
for var_name in all_x_vars:
    print(f"  > Loading X: {var_name}")
    x_values[var_name] = ds10.variables[var_name][:]

stacked_y_values = {}
print("--- Loading and stacking Y variables ---")
for var_name in all_x_vars:
    print(f"  > Loading and stacking Y: {var_name}")
    list_of_arrays = [ds_r.variables[var_name][:] for ds_r in ds_r_list]
    stacked_y_values[var_name] = np.stack(list_of_arrays, axis=0)

pft_gridcell_index = ds10.variables['pfts1d_gridcell_index'][:]
column_gridcell_index = ds10.variables['cols1d_gridcell_index'][:]
landunit_gridcell_index = ds10.variables['land1d_gridcell_index'][:]
# topounit_gridcell_index = ds10.variables['topo1d_gridcell_index'][:]


pft_map = {}
column_map = {}
# landunit_map = {}
# topounit_map = {}

unique_gridcell_ids = np.unique(pft_gridcell_index)
for grid_id in unique_gridcell_ids:
    pft_map[grid_id] = np.where(pft_gridcell_index == grid_id)[0]
    column_map[grid_id] = np.where(column_gridcell_index == grid_id)[0]
    # landunit_map[grid_id] = np.where(landunit_gridcell_index == grid_id)[0]
    # topounit_map[grid_id] = np.where(topounit_gridcell_index == grid_id)[0]

print("All data is ready! Entering high-speed processing loop.")

for start_idx in range(0, len(filtered_coordinates), batch_size):
    end_idx = min(start_idx + batch_size, len(filtered_coordinates))
    batch_coords = filtered_coordinates[start_idx:end_idx]
    batch_restart_indices = all_restart_indices[start_idx:end_idx]
    batch_forcing_indices = all_forcing_indices[start_idx:end_idx]
    print(f"Processing batch {start_idx} to {end_idx}")

    data_dict = {
        'landfrac':[],
        'Latitude': [],
        'Longitude': [],
        'FLDS': [], 'PSRF': [], 'FSDS': [], 'QBOT': [], 'PRECTmms': [], 'TBOT': [],
        'LANDFRAC_PFT': [], 'PCT_NATVEG': [], 'AREA': [], 'peatf': [], 'abm': [],
        'SOIL_COLOR': [], 'SOIL_ORDER': [], 'PCT_NAT_PFT': [], 'PCT_SAND': [],
        'soil3c_vr': [], 'soil4c_vr': [], 'cwdc_vr': [], 'deadcrootc': [], 'deadstemc': [],'tlai': [],
        'GPP': [],
        'Y_soil3c_vr': [], 'Y_soil4c_vr': [], 'Y_cwdc_vr': [],  'Y_deadcrootc': [],  'Y_deadstemc': [], 'Y_tlai': [],
        'Y_GPP': [],
        'SCALARAVG_vr': [],
        'PCT_CLAY': [],
        'SNOWDP': [],
        'H2OSOI_10CM': [],
        'HR': [],  'AR': [],  'NPP': [], 'COL_FIRE_CLOSS': [],
        'Y_HR': [],  'Y_AR': [],  'Y_NPP': [], 'Y_COL_FIRE_CLOSS': [],
        'OCCLUDED_P': [],
        'SECONDARY_P': [],
        'LABILE_P': [],
        'APATITE_P': [],

        'cwdn_vr': [], 'secondp_vr': [], 'cwdp_vr': [],'cwdp': [], 'totcolp': [], 'totvegc': [], 'deadstemn': [], 'deadcrootn': [],
        'deadstemp': [], 'deadcrootp': [], 'leafc': [], 'leafc_storage': [], 'frootc': [], 'frootc_storage': [],
        'Y_cwdn_vr': [], 'Y_secondp_vr': [], 'Y_cwdp_vr': [], 'Y_cwdp': [], 'Y_totcolp': [], 'Y_totvegc': [], 'Y_deadstemn': [], 'Y_deadcrootn': [],
        'Y_deadstemp': [], 'Y_deadcrootp': [], 'Y_leafc': [], 'Y_leafc_storage': [], 'Y_frootc': [], 'Y_frootc_storage': [],
        'totlitc': [],

    'leafn': [], 'leafn_storage': [], 'frootn': [],'frootn_storage': [],
    'leafp': [], 'leafp_storage': [], 'frootp': [],'frootp_storage': [],
    'livestemc': [], 'livestemc_storage': [], 
    'livestemn': [], 'livestemn_storage': [], 
    'livestemp': [], 'livestemp_storage': [],
    'labilep_vr': [], 'occlp_vr': [], 'primp_vr': [],

    'deadcrootc_storage': [], 'deadstemc_storage': [], 
    'livecrootc': [], 'livecrootc_storage': [], 
    'deadcrootn_storage': [], 'deadstemn_storage': [], 
    'livecrootn': [], 'livecrootn_storage': [], 
    'deadcrootp_storage': [], 'deadstemp_storage': [],
    'livecrootp': [], 'livecrootp_storage': [], 

    'Y_leafn': [], 'Y_leafn_storage': [], 'Y_frootn': [],'Y_frootn_storage': [],
    'Y_leafp': [], 'Y_leafp_storage': [], 'Y_frootp': [],'Y_frootp_storage': [],
    'Y_livestemc': [], 'Y_livestemc_storage': [], 
    'Y_livestemn': [], 'Y_livestemn_storage': [], 
    'Y_livestemp': [], 'Y_livestemp_storage': [],
    'Y_labilep_vr': [], 'Y_occlp_vr': [], 'Y_primp_vr': [],

    'Y_deadcrootc_storage': [], 'Y_deadstemc_storage': [], 
    'Y_livecrootc': [], 'Y_livecrootc_storage': [], 
    'Y_deadcrootn_storage': [], 'Y_deadstemn_storage': [], 
    'Y_livecrootn': [], 'Y_livecrootn_storage': [], 
    'Y_deadcrootp_storage': [], 'Y_deadstemp_storage': [],
    'Y_livecrootp': [], 'Y_livecrootp_storage': [], 
        'Y_totlitc': [],
        # 'H2OCAN': [], 'T_VEG': [], 'T10_VALUE': [],
        # 'Y_H2OCAN': [], 'Y_T_VEG': [], 'Y_T10_VALUE': [],
        # 'H2OSFC': [], 'H2OSNO': [], 'TH2OSFC': [], 'T_GRND': [], 'T_GRND_R': [], 'T_GRND_U': [],
        # 'Y_H2OSFC': [], 'Y_H2OSNO': [], 'Y_TH2OSFC': [], 'Y_T_GRND': [], 'Y_T_GRND_R': [], 'Y_T_GRND_U': [],
        # 'H2OSOI_LIQ': [], 'H2OSOI_ICE': [], 'T_SOISNO': [], 'LAKE_SOILC': [], 'T_LAKE': [],
        # 'Y_H2OSOI_LIQ': [], 'Y_H2OSOI_ICE': [], 'Y_T_SOISNO': [], 'Y_LAKE_SOILC': [], 'Y_T_LAKE': [],
        # 'taf': [],
        # 'Y_taf': [],
        # 'TS_TOPO': [],
        # 'Y_TS_TOPO': [],
        'soil1c_vr': [], 'soil1n_vr': [], 'soil1p_vr': [],
        'soil2c_vr': [], 'soil2n_vr': [], 'soil2p_vr': [],
        'soil3n_vr': [], 'soil3p_vr': [],
        'soil4n_vr': [], 'soil4p_vr': [],
        'litr1c_vr': [], 'litr2c_vr': [], 'litr3c_vr': [],
        'litr1n_vr': [], 'litr2n_vr': [], 'litr3n_vr': [],
        'litr1p_vr': [], 'litr2p_vr': [], 'litr3p_vr': [],
        'sminn_vr': [], 'smin_no3_vr': [], 'smin_nh4_vr': [],
        'Y_soil1c_vr': [], 'Y_soil1n_vr': [], 'Y_soil1p_vr': [],
        'Y_soil2c_vr': [], 'Y_soil2n_vr': [], 'Y_soil2p_vr': [],
        'Y_soil3n_vr': [], 'Y_soil3p_vr': [],
        'Y_soil4n_vr': [], 'Y_soil4p_vr': [],
        'Y_litr1c_vr': [], 'Y_litr2c_vr': [], 'Y_litr3c_vr': [],
        'Y_litr1n_vr': [], 'Y_litr2n_vr': [], 'Y_litr3n_vr': [],
        'Y_litr1p_vr': [], 'Y_litr2p_vr': [], 'Y_litr3p_vr': [],
        'Y_sminn_vr': [], 'Y_smin_no3_vr': [], 'Y_smin_nh4_vr': []
    }

    new_pft_list = ['totvegc', 'deadstemn','deadcrootn','deadstemp','deadcrootp','leafc','leafc_storage','frootc','frootc_storage']

    gridcell_lat = ds10.variables['grid1d_lat'][:]
    gridcell_lon = ds10.variables['grid1d_lon'][:]
    print('gridcell_lat',gridcell_lat)
    print('gridcell_lon',gridcell_lon)
    pft_gridcell_index = ds10.variables['pfts1d_gridcell_index'][:]
    column_gridcell_index = ds10.variables['cols1d_gridcell_index'][:]

    soil3c_vr_values = ds10.variables['soil3c_vr'][:]
    soil4c_vr_values = ds10.variables['soil4c_vr'][:]
    cwdc_vr_values = ds10.variables['cwdc_vr'][:]
    cwdn_vr_values = ds10.variables['cwdn_vr'][:]
    secondp_vr_values = ds10.variables['secondp_vr'][:]
    cwdp_vr_vr_values = ds10.variables['cwdp_vr'][:]
    cwdp_vr_values = ds10.variables['cwdp'][:]
    totcolp_vr_values = ds10.variables['totcolp'][:]

    deadcrootc_values = ds10.variables['deadcrootc'][:]
    deadstemc_values = ds10.variables['deadstemc'][:]
    tlai_values = ds10.variables['tlai'][:]

    totvegc_values = ds10.variables['totvegc'][:]
    deadstemn_values = ds10.variables['deadstemn'][:]
    deadcrootn_values = ds10.variables['deadcrootn'][:]
    deadstemp_values = ds10.variables['deadstemp'][:]
    deadcrootp_values = ds10.variables['deadcrootp'][:]
    leafc_values = ds10.variables['leafc'][:]
    leafc_storage_values = ds10.variables['leafc_storage'][:]
    frootc_values = ds10.variables['frootc'][:]
    frootc_storage_values = ds10.variables['frootc_storage'][:]


    for k, (i, j) in enumerate(batch_coords):
        closest_index = batch_restart_indices[k]
        gridcell_id = closest_index + 1

        target_lat = lats[i]
        target_lon = lons[j]

        closest_index = batch_restart_indices[k]
        gridcell_id = closest_index + 1

        mask_s = column_gridcell_index == gridcell_id

        closest_restart_index = batch_restart_indices[k]
        nearest_forcing_index = batch_forcing_indices[k]

        gridcell_id = closest_restart_index + 1

        pft_indices_for_cell = pft_map.get(gridcell_id, [])
        column_indices_for_cell = column_map.get(gridcell_id, [])

        for var_name in pft_based_vars:
            x_val = x_values[var_name][pft_indices_for_cell]
            data_dict[var_name].append(x_val.tolist())

            y_slice = stacked_y_values[var_name][:, pft_indices_for_cell]
            avg_y_val = np.mean(y_slice, axis=0)
            data_dict[f'Y_{var_name}'].append(avg_y_val.tolist())

        for var_name in col_based_1d_vars:
            x_val = x_values[var_name][column_indices_for_cell]
            data_dict[var_name].append(x_val.tolist())

            y_slice = stacked_y_values[var_name][:, column_indices_for_cell]
            avg_y_val = np.mean(y_slice, axis=0)
            data_dict[f'Y_{var_name}'].append(avg_y_val.tolist())

        for var_name in col_based_2d_vars:
            x_val = x_values[var_name][column_indices_for_cell, :]
            data_dict[var_name].append(x_val.tolist())

            y_slice = stacked_y_values[var_name][:, column_indices_for_cell, :]
            avg_y_val = np.mean(y_slice, axis=0)
            data_dict[f'Y_{var_name}'].append(avg_y_val.tolist())

        # landunit_indices_for_cell = landunit_map.get(gridcell_id, [])
        # for var_name in landunit_based_vars:
        #     x_val = x_values[var_name][landunit_indices_for_cell]
        #     data_dict[var_name].append(x_val.tolist())

        #     y_slice = stacked_y_values[var_name][:, landunit_indices_for_cell]
        #     avg_y_val = np.mean(y_slice, axis=0)
        #     data_dict[f'Y_{var_name}'].append(avg_y_val.tolist())

        # topounit_indices_for_cell = topounit_map.get(gridcell_id, [])
        # for var_name in topounit_based_vars:
        #     x_val = x_values[var_name][topounit_indices_for_cell]
        #     data_dict[var_name].append(x_val.tolist())

        #     y_slice = stacked_y_values[var_name][:, topounit_indices_for_cell]
        #     avg_y_val = np.mean(y_slice, axis=0)
        #     data_dict[f'Y_{var_name}'].append(avg_y_val.tolist())

        data_dict['landfrac'].append(ds2.variables['landfrac'][i, j])
        data_dict['Latitude'].append(lats[i])
        data_dict['Longitude'].append(lons[j])
        data_dict['LANDFRAC_PFT'].append(ds1.variables['LANDFRAC_PFT'][i, j])
        data_dict['PCT_NATVEG'].append(ds1.variables['PCT_NATVEG'][i, j])
        data_dict['AREA'].append(ds1.variables['AREA'][i, j])
        data_dict['peatf'].append(ds1.variables['peatf'][i, j])
        data_dict['abm'].append(ds1.variables['abm'][i, j])
        data_dict['SOIL_COLOR'].append(ds1.variables['SOIL_COLOR'][i, j])
        data_dict['SOIL_ORDER'].append(ds1.variables['SOIL_ORDER'][i, j])
        data_dict['PCT_SAND'].append(ds1.variables['PCT_SAND'][:, i, j])
        data_dict['PCT_NAT_PFT'].append(ds1.variables['PCT_NAT_PFT'][:, i, j])

        data_dict['OCCLUDED_P'].append(ds1.variables['OCCLUDED_P'][i, j])
        data_dict['SECONDARY_P'].append(ds1.variables['SECONDARY_P'][i, j])
        data_dict['LABILE_P'].append(ds1.variables['LABILE_P'][i, j])
        data_dict['APATITE_P'].append(ds1.variables['APATITE_P'][i, j])


        data_dict['GPP'].append(ds2.variables['GPP'][0, i, j])
        data_dict['SCALARAVG_vr'].append(ds2.variables['SCALARAVG_vr'][0, :, i, j])
        data_dict['HR'].append(ds2.variables['HR'][0, i, j])
        data_dict['AR'].append(ds2.variables['AR'][0, i, j])
        data_dict['NPP'].append(ds2.variables['NPP'][0, i, j])
        data_dict['COL_FIRE_CLOSS'].append(ds2.variables['COL_FIRE_CLOSS'][0, i, j])

        data_dict['SNOWDP'].append(ds2.variables['SNOWDP'][0, i, j])
        data_dict['H2OSOI_10CM'].append(ds2.variables['H2OSOI'][0,3, i, j])
        data_dict['PCT_CLAY'].append(ds1.variables['PCT_CLAY'][:, i, j])

        h0_gpp_vals = []
        for ds_h0 in ds_h0_list:
            h0_gpp_vals.append(ds_h0.variables['GPP'][0, i, j])
        avg_h0_gpp = np.mean(h0_gpp_vals)
        data_dict['Y_GPP'].append(avg_h0_gpp)

        h0_HR_vals = []
        for ds_h0 in ds_h0_list:
            h0_HR_vals.append(ds_h0.variables['HR'][0, i, j])
        avg_h0_HR = np.mean(h0_HR_vals)
        data_dict['Y_HR'].append(avg_h0_HR)

        h0_AR_vals = []
        for ds_h0 in ds_h0_list:
            h0_AR_vals.append(ds_h0.variables['AR'][0, i, j])
        avg_h0_AR = np.mean(h0_AR_vals)
        data_dict['Y_AR'].append(avg_h0_AR)

        h0_NPP_vals = []
        for ds_h0 in ds_h0_list:
            h0_NPP_vals.append(ds_h0.variables['NPP'][0, i, j])
        avg_h0_NPP = np.mean(h0_NPP_vals)
        data_dict['Y_NPP'].append(avg_h0_NPP)

        h0_COL_FIRE_CLOSS_vals = []
        for ds_h0 in ds_h0_list:
            h0_COL_FIRE_CLOSS_vals.append(ds_h0.variables['COL_FIRE_CLOSS'][0, i, j])
        avg_h0_COL_FIRE_CLOSS = np.mean(h0_COL_FIRE_CLOSS_vals)
        data_dict['Y_COL_FIRE_CLOSS'].append(avg_h0_COL_FIRE_CLOSS)

        lats4, lons4 = ds4.variables['LATIXY'][:], ds4.variables['LONGXY'][:]

        target_lat = lats[i]
        target_lon = lons[j]
        nearest_n = batch_forcing_indices[k]
        nearest_lat = lats4[nearest_n]
        nearest_lon = lons4[nearest_n]

        flds_values = ds4.variables['FLDS'][nearest_n, :]
        data_dict['FLDS'].append(flds_values)

        flds_values = ds5.variables['PSRF'][nearest_n, :]
        data_dict['PSRF'].append(flds_values)

        flds_values = ds6.variables['FSDS'][nearest_n, :]
        data_dict['FSDS'].append(flds_values)

        flds_values = ds7.variables['QBOT'][nearest_n, :]
        data_dict['QBOT'].append(flds_values)

        flds_values = ds8.variables['PRECTmms'][nearest_n, :]
        data_dict['PRECTmms'].append(flds_values)

        flds_values = ds9.variables['TBOT'][nearest_n, :]
        data_dict['TBOT'].append(flds_values)
    df_batch = pd.DataFrame(data_dict)

    batch_save_path = f"/mnt/DATA/0_oak_data/8_use_700_years/training_data_batch_{batch_number:02d}.pkl"

    df_batch.to_pickle(batch_save_path)
    print(f"Saved batch {start_idx} to {end_idx} to {batch_save_path}")
    batch_number += 1
print(f"Data successfully saved.")

ds1.close()
ds2.close()
ds3.close()
ds4.close()
ds5.close()
ds6.close()
ds7.close()
ds8.close()
ds9.close()

import pandas as pd
import glob

input_files = sorted(glob.glob('/mnt/DATA/0_oak_data/8_use_700_years/training_data_batch_*.pkl'))
output_dir = '/mnt/DATA/0_oak_data/8_use_700_years/1_global_data/'

time_series_columns = ['FLDS', 'PSRF', 'FSDS', 'QBOT', 'PRECTmms', 'TBOT']
single_value_columns = ['landfrac',
    'LANDFRAC_PFT', 'PCT_NATVEG', 'AREA', 'peatf', 'abm', 'SOIL_COLOR', 'SOIL_ORDER',
    'GPP',
    'SNOWDP','H2OSOI_10CM',
    'Y_GPP',
    'HR',  'AR',  'NPP', 'COL_FIRE_CLOSS',
    'Y_HR',  'Y_AR',  'Y_NPP', 'Y_COL_FIRE_CLOSS',
    'OCCLUDED_P',
    'SECONDARY_P',
    'LABILE_P',
    'APATITE_P'
    ]
list_like_columns = ['PCT_NAT_PFT', 'PCT_SAND', 'SCALARAVG_vr','PCT_CLAY']

time_series_length = 179580
steps_per_day = 4
days_per_year = 365
years_in_data = 123
months_per_year = 12
days_per_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

def calculate_monthly_avg(time_series):
    if not isinstance(time_series, (list, np.ndarray)):
        print(f"Invalid type: {type(time_series)}")
        return []
    if len(time_series) != time_series_length:
        print(f"Invalid length: {len(time_series)}, expected {time_series_length}")
        return []

    monthly_averages = []
    start_idx = 0
    for year in range(years_in_data):
        for month_days in days_per_month:
            end_idx = start_idx + month_days * steps_per_day
            monthly_avg = np.mean(time_series[start_idx:end_idx])
            monthly_averages.append(monthly_avg)
            start_idx = end_idx
    return monthly_averages

for file_path in input_files:
    print(f"Processing {file_path}...")

    df = pd.read_pickle(file_path)
    print(df.info())

    for col in time_series_columns:
        if col in df.columns:
            print(f"Processing {col}...")
            df[col] = df[col].apply(calculate_monthly_avg)

            sample_data = df[col].apply(lambda x: x if isinstance(x, list) else [])
            print(f"Variable: {col}")
            print(f"Length of each sample: {sample_data.apply(len).unique()}")
            print("Top 5 Samples:")
            print("-" * 50)

    for col in single_value_columns:
        df[col] = df[col].astype(str).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    for col in list_like_columns:
        expanded_cols = df[col].apply(pd.Series).fillna(0)
        expanded_cols = expanded_cols.add_prefix(f"{col}_")
        df = df.drop(col, axis=1).join(expanded_cols)

    y_columns = [col for col in df.columns if col.startswith('Y_')]
    other_columns = [col for col in df.columns if not col.startswith('Y_')]
    df = df[other_columns + y_columns]

    output_file = f"{output_dir}1_{file_path.split('/')[-1]}"
    df.to_pickle(output_file)
    print(f"Processed file saved as {output_file}")
    print(df.info())
    print(df[['Latitude', 'Longitude']].head())

print("Processing complete.")