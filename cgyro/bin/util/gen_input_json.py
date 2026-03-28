#!/usr/bin/env python
"""
Generate input.json from cgyro_parse.py

This script parses cgyro/bin/cgyro_parse.py to extract default parameter
values and writes them to cgyro/bin/util/input.json. This ensures that
cgyro_parse.py remains the single source of truth for CGYRO defaults.

Usage:
    python gen_input_json.py
"""

import os
import re
import json

def parse_cgyro_defaults():
    """Parse cgyro_parse.py and extract default values."""

    garoot = os.environ.get('GACODE_ROOT')
    if garoot is None:
        # Try to find it relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        garoot = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

    parse_file = os.path.join(garoot, 'cgyro', 'bin', 'cgyro_parse.py')

    if not os.path.isfile(parse_file):
        raise FileNotFoundError(f"Cannot find {parse_file}")

    defaults = {}

    with open(parse_file, 'r') as f:
        for line in f:
            # Match: x.add('NAME','value')
            match = re.match(r"x\.add\('([^']+)','([^']+)'\s*\)", line)
            if match:
                key, value = match.groups()
                defaults[key] = convert_value(value)
                continue

            # Match: x.add('NAME','value',n=n) for array parameters
            match = re.match(r"x\.add\('([^']+)','([^']+)',n=n\)", line)
            if match:
                key, value = match.groups()
                # These are array parameters with 11 elements (n=11 in cgyro_parse.py)
                defaults[key] = [convert_value(value)] * 11
                continue

    return defaults

def convert_value(value_str):
    """Convert string value to appropriate Python type."""
    try:
        if '.' in value_str or 'e' in value_str.lower():
            return float(value_str)
        else:
            return int(value_str)
    except ValueError:
        return value_str

def main():
    garoot = os.environ.get('GACODE_ROOT')
    if garoot is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        garoot = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))

    output_file = os.path.join(garoot, 'cgyro', 'bin', 'util', 'input.json')

    defaults = parse_cgyro_defaults()

    with open(output_file, 'w') as f:
        json.dump(defaults, f, indent=2, sort_keys=True)

    print(f"gen_input_json: wrote {output_file}")
    print(f"gen_input_json: {len(defaults)} parameters")

if __name__ == '__main__':
    main()
