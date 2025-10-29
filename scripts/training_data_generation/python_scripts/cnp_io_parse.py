import re


def parse_cnp_io_list(filename):
    # Map section titles to config keys (case-insensitive matching)
    section_map = {
        'TIME SERIES VARIABLES': 'time_series_variables',
        'SURFACE PROPERTIES': 'surface_properties',
        'PFT PARAMETERS': 'pft_parameters',
        'WATER VARIABLES': 'water_variables',
        'SCALAR VARIABLES': 'scalar_variables',
        'TEMPERATURE VARIABLES': 'temperature_variables',
        '1D PFT VARIABLES': 'pft_1d_variables',
        '2D VARIABLES': 'variables_2d_soil',
        # Custom section for explicit COL 1D vars
        'RESTART_COL_1D_VARS': 'dataset_new_RESTART_COL_1D_VARS',
        # Accept alternate capitalization found in some files
        'Water variables': 'water_variables',
    }

    # Prepare result dict with all keys present
    result = {v: [] for v in set(section_map.values())}
    current_section = None

    with open(filename) as f:
        for raw_line in f:
            line = raw_line.strip()
            # Section header detection (case-insensitive startswith)
            matched = False
            for section_title, key in section_map.items():
                if line.lower().startswith(section_title.lower()):
                    current_section = key
                    matched = True
                    break
            if matched:
                continue

            # Content lines
            if not current_section:
                continue

            # If line is a variable line (starts with • or comma-separated list)
            if line.startswith('•'):
                # Remove bullet and split by comma, filter out empty strings
                vars_ = [v.strip() for v in line[1:].split(',') if v.strip()]
                result[current_section].extend(vars_)
            # Some variables are listed as comma-separated after a bullet or alone
            elif ',' in line and not line.startswith('['):
                vars_ = [v.strip('• ').strip() for v in line.split(',') if v.strip('• ').strip()]
                result[current_section].extend(vars_)
            # Some variables are listed as single words (rare, but just in case)
            elif line and not line.startswith('[') and not line.startswith('#'):
                # Only add if it's not a description or exclusion
                if re.match(r'^[A-Za-z0-9_]+$', line):
                    result[current_section].append(line)

    return result


if __name__ == "__main__":
    # Optional local test, will not run when imported by config
    try:
        parsed = parse_cnp_io_list("CNP_IO_updated14_xfer.txt")
        for key, varlist in parsed.items():
            print(f"{key} (length: {len(varlist)}):")
            print(varlist)
            print()
    except FileNotFoundError:
        print("CNP_IO_updated14_xfer.txt not found for standalone test.")