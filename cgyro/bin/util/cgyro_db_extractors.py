#!/usr/bin/env python
"""
CGYRO AI Database Extraction Utilities

Provides functions to extract metadata, species data, flux data, and provenance
information from CGYRO simulations for storage in the AI database.
"""

import os
import json
import subprocess
import numpy as np
from gacodefuncs import time_average, time_index
from cgyro_util import compute_file_hash, is_cgyro_case, is_linear_simulation

def extract_metadata(sim):
    """Extract grid and equilibrium metadata from cgyrodata object."""
    return {
        'n_n': int(sim.n_n), 'n_radial': int(sim.n_radial), 'n_theta': int(sim.n_theta),
        'n_species': int(sim.n_species), 'n_field': int(sim.n_field),
        'n_energy': int(sim.n_energy), 'n_xi': int(sim.n_xi),
        'rmin': float(sim.rmin), 'rmaj': float(sim.rmaj), 'q': float(sim.q),
        'shear': float(sim.shear), 'shift': float(sim.shift),
        'kappa': float(sim.kappa), 's_kappa': float(sim.s_kappa),
        'delta': float(sim.delta), 's_delta': float(sim.s_delta),
        'zeta': float(sim.zeta), 's_zeta': float(sim.s_zeta),
        'zmag': float(sim.zmag), 'dzmag': float(sim.dzmag),
        'ky0': float(sim.ky0), 'betae_unit': float(sim.betae_unit),
        'beta_star': float(sim.beta_star), 'gamma_e': float(sim.gamma_e),
        'gamma_p': float(sim.gamma_p), 'mach': float(sim.mach),
    }

def extract_species(sim):
    """Extract species arrays from cgyrodata object."""
    return {
        'z': sim.z.tolist(), 'mass': sim.mass.tolist(), 'dens': sim.dens.tolist(),
        'temp': sim.temp.tolist(), 'dlnndr': sim.dlnndr.tolist(),
        'dlntdr': sim.dlntdr.tolist(), 'nu': sim.nu.tolist(),
    }

def extract_flux(sim, time_window='0.5'):
    """Extract and time-average flux data from cgyrodata object."""
    sim.getflux(total=True)
    imin, imax = time_index(sim.t, time_window)

    fluxes = []
    for ispec in range(sim.n_species):
        fluxes.append({
            'species': int(ispec),
            'fluxn': time_average(sim.fluxtot[ispec, 0, :], sim.t, imin, imax).tolist(),
            'fluxe': time_average(sim.fluxtot[ispec, 1, :], sim.t, imin, imax).tolist(),
            'fluxv': time_average(sim.fluxtot[ispec, 2, :], sim.t, imin, imax).tolist(),
            'time_window': time_window,
            'time_max': float(sim.t[-1])
        })
    return fluxes

def compute_input_hash(case_path):
    """Compute MD5 hash of input.cgyro as unique identifier."""
    return compute_file_hash(os.path.join(case_path, 'input.cgyro'))

def extract_provenance(case_path, t_final, i_final):
    """Extract simulation provenance information."""
    time_file = os.path.join(case_path, 'out.cgyro.time')

    modified_timestamp = created_timestamp = None
    if os.path.exists(time_file):
        modified_timestamp = os.path.getmtime(time_file)
        try:
            created_timestamp = os.path.getctime(time_file)
        except:
            created_timestamp = modified_timestamp

    try:
        restart_count = len([f for f in os.listdir(case_path) if f.startswith('bin.cgyro.restart')])
    except:
        restart_count = 0

    file_hashes = {}
    for fname in ['bin.cgyro.freq', 'out.cgyro.time', 'out.cgyro.prec']:
        h = compute_file_hash(os.path.join(case_path, fname))
        if h:
            file_hashes[fname] = h

    return {
        't_final': t_final, 'i_final': i_final,
        'created_timestamp': created_timestamp, 'modified_timestamp': modified_timestamp,
        'n_restarts': restart_count, 'input_hash': compute_input_hash(case_path),
        'file_hashes': json.dumps(file_hashes),
    }

def get_git_version(directory):
    """Get git commit hash for a directory."""
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=directory,
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return result.stdout.strip()
    except:
        pass
    return None
