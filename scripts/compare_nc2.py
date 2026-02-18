import numpy as np
import netCDF4 as nc
import sys
import argparse

def compare_variables(var1, var2):
    if var1.dtype != var2.dtype:
        print(f'Different data types: {var1.dtype} vs {var2.dtype}')
    if var1.shape != var2.shape:
        print(f'Different shapes: {var1.shape} vs {var2.shape}')
    if (np.issubdtype(var1.dtype, np.number) and var1.shape == var2.shape and var1.dtype != 'short'):
        
        if len(var1.shape) <= 2:
            compare_data(var1[:], var2[:])
        elif len(var1.shape) == 3:
            for i in range(var1.shape[0]):
                compare_data(var1[i, :, :], var2[i, :, :])
        elif len(var1.shape) == 4:
            for i in range(var1.shape[0]):
                for j in range(var1.shape[1]):
                    compare_data(var1[i, j, :, :], var2[i, j, :, :])

def compare_data(data1, data2):
    if not np.allclose(data1, data2):
        print(f'Difference in data:')
        print(f'Sum: {np.sum(data1)} vs {np.sum(data2)}')
        print(f'Mean: {np.mean(data1)} vs {np.mean(data2)}')
        print(f'Max: {np.max(data1)} vs {np.max(data2)}')
        print(f'Min: {np.min(data1)} vs {np.min(data2)}')

def main():
    parser = argparse.ArgumentParser(description='Compare two NetCDF files')
    parser.add_argument('--file_name1', type=str, 
                        default='/global/cfs/cdirs/m4814/daweigao/15_code_Landsim/LandSim/1_single_point/AI_updated_single_point_303_17_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc',
                        help='Path to the first NetCDF file')
    parser.add_argument('--file_name2', type=str,
                        default='/global/cfs/cdirs/m4814/daweigao/15_code_Landsim/LandSim/1_single_point/single_point_303_17_20251201_TRENDY2024_default_ICB1850CNRDCTCBC_ad_spinup.elm.r.0021-01-01-00000.nc',
                        help='Path to the second NetCDF file')
    
    args = parser.parse_args()
    file_name1 = args.file_name1
    file_name2 = args.file_name2
    


    file1 = nc.Dataset(file_name1)
    file2 = nc.Dataset(file_name2)

    variables1 = file1.variables
    variables2 = file2.variables

    for var in variables1:
        if var in variables2:
            print(var)
            compare_variables(variables1[var], variables2[var])
        else:
            print(f'Variable {var} is not in the second file')

    for var in variables2:
        if var not in variables1:
            print(f'Variable {var} is not in the first file')

    file1.close()
    file2.close()

if __name__ == '__main__':
    main()