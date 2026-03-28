"""
Shared utilities for CGYRO tools (cgyro_pre, cgyro_json, cgyro_db, etc.)
"""

import os
import sys
import json
import hashlib
import numpy as np

GAROOT = os.environ['GACODE_ROOT']

#==================================================================
# CGYRO file constants

CGYRO_OUTPUT_FILES = ['out.cgyro.grids', 'out.cgyro.time', 'bin.cgyro.freq']
CGYRO_RESTART_FILES = ['bin.cgyro.restart', 'bin.cgyro.restart.old', 'bin.cgyro.restart.part']
CGYRO_REQUIRED_FILES = ['out.cgyro.grids']  # Minimum files for valid case

# Map parameter names to (argparse_attr, sentinel_value)
ARG_MAP = {
    'N_RADIAL': ('nx', 0),
    'N_TOROIDAL': ('ny', 0),
    'N_THETA': ('ntheta', 0),
    'N_ENERGY': ('ne', 0),
    'N_XI': ('nxi', 0),
    'E_MAX': ('emax', 0.0),
    'DELTA_T': ('dt', 0.0),
    'DELTA_T_METHOD': ('dtm', -1),
    'lx': ('lx', 0.0),
    'ly': ('ly', 0.0),
}

RESOLUTION_PARAMS = ['N_RADIAL', 'N_TOROIDAL', 'N_THETA', 'N_ENERGY', 'N_XI',
                     'E_MAX', 'DELTA_T', 'DELTA_T_METHOD']

PARAM_GROUPS = {
    'time': ['DELTA_T', 'DELTA_T_METHOD', 'MAX_TIME', 'PRINT_STEP'],
    'geometry': ['RMIN', 'RMAJ', 'Q', 'S', 'SHIFT', 'KAPPA', 'S_KAPPA',
                 'DELTA', 'S_DELTA', 'ZETA', 'S_ZETA', 'ZMAG', 'DZMAG'],
    'physics': ['NONLINEAR_FLAG', 'N_FIELD', 'BETAE_UNIT', 'BETA_STAR_SCALE',
                'LAMBDA_STAR', 'NU_EE', 'COLLISION_MODEL'],
    'rotation': ['ROTATION_MODEL', 'MACH', 'GAMMA_E', 'GAMMA_P'],
}

def load_defaults():
    """Load parameter defaults from input.json."""
    with open(f'{GAROOT}/cgyro/bin/util/input.json', 'r') as f:
        return json.load(f)

def digit(f, n):
    """Round f to n decimal places."""
    return round(f) if n == 0 else round(10**n * f) / 10**n

def compute_ky_boxsize(lx, ly, shear):
    """Compute KY and BOX_SIZE from lx, ly, and shear."""
    ky = digit(2*np.pi/ly, 4)
    bs = max(1, digit(lx*shear*ky, 0))
    actual_ly = 2*np.pi/ky
    actual_lx = bs/(shear*ky)
    print(f'cgyro_pre: LX={actual_lx:.4f}  LY={actual_ly:.4f}  BOX_SIZE={bs}')
    return ky, bs

def read_input_cgyro(filepath):
    """Read input.cgyro, return (lines, {param: line_index})."""
    lines, params = [], {}
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            lines.append(line)
            s = line.strip()
            if not s.startswith('#') and '=' in s:
                params[s.split('=', 1)[0].strip()] = i
    return lines, params

def parse_input_xgyro():
    """Parse input.xgyro and return list of directories."""
    if not os.path.isfile('input.xgyro'):
        return []
    dirs = []
    with open('input.xgyro', 'r') as f:
        for line in f:
            s = line.strip()
            if not s.startswith('#') and '=' in s:
                key, val = s.split('=', 1)
                if key.strip().startswith('DIR_'):
                    dirs.append(val.strip())
    return dirs

def write_input_xgyro(dirs, comment="cgyro", mpi_vals=None):
    """Write input.xgyro file."""
    with open('input.xgyro', 'w') as f:
        f.write(f'# Created by {comment}\n')
        if mpi_vals is not None:
            f.write(f'# Total MPI: {sum(int(v) for v in mpi_vals)}\n')
        f.write(f'N_DIRS={len(dirs)}\n')
        for i, d in enumerate(dirs):
            if mpi_vals is not None:
                f.write(f'MIN_MPI_{i+1} = {mpi_vals[i]}\n')
            f.write(f'DIR_{i+1} = {d}\n')

def apply_params(lines, params, updates):
    """Apply updates to lines, return list of updated keys."""
    keys = []
    for key, val in updates.items():
        if key in params:
            lines[params[key]] = f'{key}={val}\n'
        else:
            lines.append(f'{key}={val}\n')
        keys.append(key)
    return keys

def get_param(lines, key):
    """Extract parameter value from lines."""
    for line in lines:
        s = line.strip()
        if s.startswith(f'{key}='):
            return float(s.split('=')[1])
    return None

def write_section(f, title, keys, d, first=False):
    """Write a section of key=value pairs, popping from d."""
    if not first:
        f.write('\n')
    f.write(f'# {title}\n')
    for key in keys:
        if key in d:
            f.write(f'{key}={d.pop(key)}\n')

def parse_cgyro_gen(filepath, default):
    """Parse input.cgyro.gen into a dict, including species and SHAPE arrays."""
    import re
    d = {}
    species_keys = {'Z', 'MASS', 'DENS', 'TEMP', 'DLNNDR', 'DLNTDR'}
    species_arrays = {k: {} for k in species_keys}
    shape_arrays = {}

    with open(filepath, 'r') as f:
        for line in f:
            u = line.split()
            if len(u) < 2:
                continue
            value, key = u[0], u[1]

            # Parse value
            try:
                val = float(value) if ('.' in value or 'e' in value.lower()) else int(value)
            except ValueError:
                val = value

            # Check for species array (e.g., Z_1, MASS_2)
            match = re.match(r'([A-Z_]+)_(\d+)$', key)
            if match:
                base, idx = match.groups()
                if base in species_keys:
                    species_arrays[base][int(idx)] = val
                    continue

            # Check for SHAPE array (e.g., SHAPE_SIN0, SHAPE_COS3)
            match = re.match(r'(SHAPE_[A-Z_]+)(\d+)$', key)
            if match:
                base, idx = match.groups()
                if base not in shape_arrays:
                    shape_arrays[base] = {}
                shape_arrays[base][int(idx)] = val
                continue

            # Regular scalar parameter
            if key in default and 'SHAPE' not in key:
                d[key] = val

    # Convert species arrays to lists
    for base, arr in species_arrays.items():
        if arr:
            max_idx = max(arr.keys())
            d[base] = [arr.get(i, 0.0) for i in range(1, max_idx + 1)]

    # Convert SHAPE arrays to lists
    for base, arr in shape_arrays.items():
        if arr:
            max_idx = max(arr.keys())
            d[base] = [arr.get(i, 0.0) for i in range(max_idx + 1)]

    return d

class Logger:
    """Simple logger with quiet mode support."""
    def __init__(self, prefix, quiet=False):
        self.prefix = prefix
        self.quiet = quiet

    def __call__(self, text, indent=False):
        if not self.quiet:
            pre = '   ' if indent else f'{self.prefix}: '
            print(f'{pre}{text}')

    def error(self, text):
        print(f'{self.prefix}: (ERROR) {text}')
        sys.exit(1)

    def warn(self, text):
        print(f'{self.prefix}: (WARNING) {text}')

#==================================================================
# Case discovery and validation

def is_cgyro_case(directory, required=None):
    """Check if directory is a valid CGYRO simulation case."""
    required = required or CGYRO_REQUIRED_FILES
    return all(os.path.exists(os.path.join(directory, f)) for f in required)

def is_linear_simulation(directory):
    """Check if simulation is linear (NONLINEAR_FLAG=0)."""
    gen_file = os.path.join(directory, 'input.cgyro.gen')
    if not os.path.exists(gen_file):
        return False
    try:
        with open(gen_file, 'r') as f:
            for line in f:
                if 'NONLINEAR_FLAG' in line:
                    return line.split()[0] == '0'
    except:
        pass
    return False

def find_cgyro_cases(root_dir='./', ignore=None, linear=False):
    """
    Find all CGYRO case directories in tree.

    Parameters:
        root_dir: Root directory to search
        ignore: List of path patterns to ignore (default: ['global', 'linear'])
        linear: If False (default), skip linear simulations
    """
    ignore = ignore if ignore is not None else ['global', 'linear']
    cases = []
    for root, dirs, files in os.walk(root_dir):
        if is_cgyro_case(root):
            if any(p in root for p in ignore):
                continue
            if not linear and is_linear_simulation(root):
                continue
            cases.append(root)
    return cases

#==================================================================
# Simulation loading

def load_simulation(path, silent=True, fast=False):
    """
    Load CGYRO simulation data with sensible defaults.

    Returns cgyrodata object or None on error.
    """
    from pygacode.cgyro import data
    # Normalize path
    if not path.endswith('/'):
        path += '/'
    try:
        return data.cgyrodata(path, silent=silent, fast=fast)
    except Exception as e:
        return None

#==================================================================
# File utilities

def compute_file_hash(filepath, algorithm='md5'):
    """Compute hash of file for change detection."""
    if not os.path.exists(filepath):
        return None
    h = hashlib.md5() if algorithm == 'md5' else hashlib.sha256()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                h.update(chunk)
        return h.hexdigest()
    except:
        return None

def get_user_config_path(filename, subdir='.cgyro'):
    """Get path in user config directory, creating if needed."""
    config_dir = os.path.join(os.path.expanduser('~'), subdir)
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return os.path.join(config_dir, filename)
