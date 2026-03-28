#!/usr/bin/env python
"""
CGYRO AI Database Core Class

Provides database operations for the CGYRO AI database (SQLite).
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime


class AIDB:
    """
    CGYRO AI Database manager class

    Handles SQLite database operations for storing and querying
    lightweight CGYRO simulation data.
    """

    def __init__(self, db_path='./cgyro_ai.db'):
        """
        Initialize database connection

        Parameters:
        -----------
        db_path : str
            Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = None
        self.connect()

    def connect(self):
        """Establish connection to SQLite database"""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        # Enable foreign key constraints
        self.conn.execute("PRAGMA foreign_keys = ON")

    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def create_schema(self):
        """
        Create database schema from SQL file

        Reads and executes the schema SQL file to create tables, indexes, and views.
        """
        schema_file = os.path.join(os.path.dirname(__file__), 'cgyro_ai_schema.sql')

        if not os.path.exists(schema_file):
            raise FileNotFoundError(f"Schema file not found: {schema_file}")

        with open(schema_file, 'r') as f:
            schema_sql = f.read()

        # Execute schema SQL
        self.conn.executescript(schema_sql)
        self.conn.commit()

    def insert_case(self, case_path, hash_tag, hash_time, metadata, species, flux, provenance):
        """
        Insert a new case into the database

        Parameters:
        -----------
        case_path : str
            Relative path to simulation directory
        hash_tag : str
            MD5 hash of bin.cgyro.freq
        hash_time : str
            MD5 hash of out.cgyro.time
        metadata : dict
            Metadata from extract_metadata()
        species : dict
            Species data from extract_species()
        flux : list
            Flux data from extract_flux()
        provenance : dict
            Provenance data from extract_provenance()
        """
        # Extract case name from path
        case_name = os.path.basename(case_path.rstrip('/'))

        # Get git version if available
        git_version = provenance.get('git_commit', None)

        # Insert into cases table
        cursor = self.conn.execute("""
            INSERT INTO cases (case_path, case_name, input_hash, hash_tag, hash_time,
                             t_final, i_final, git_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (case_path, case_name, provenance.get('input_hash'), hash_tag, hash_time,
              provenance.get('t_final'), provenance.get('i_final'), git_version))

        case_id = cursor.lastrowid

        # Insert metadata
        self._insert_metadata(case_id, metadata, species)

        # Insert flux data if provided
        if flux:
            self._insert_flux(case_id, flux)

        # Insert provenance
        self._insert_provenance(case_id, provenance)

        self.conn.commit()
        return case_id

    def _insert_metadata(self, case_id, metadata, species):
        """Insert metadata for a case"""
        # Convert species dict to JSON
        species_json = json.dumps(species)

        # Build SQL dynamically to handle all metadata fields
        columns = list(metadata.keys()) + ['case_id', 'species_data']
        values = list(metadata.values()) + [case_id, species_json]
        placeholders = ','.join(['?'] * len(columns))
        columns_str = ','.join(columns)

        sql = f"INSERT INTO metadata ({columns_str}) VALUES ({placeholders})"
        self.conn.execute(sql, values)

    def _insert_flux(self, case_id, flux_list):
        """Insert flux data for a case"""
        for flux in flux_list:
            self.conn.execute("""
                INSERT INTO fluxes (case_id, species_idx, time_window, time_max,
                                  particle_flux, energy_flux, momentum_flux)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (case_id,
                  flux['species'],
                  flux['time_window'],
                  flux['time_max'],
                  json.dumps(flux['fluxn']),
                  json.dumps(flux['fluxe']),
                  json.dumps(flux['fluxv'])))

    def _insert_provenance(self, case_id, provenance):
        """Insert provenance data for a case"""
        self.conn.execute("""
            INSERT INTO provenance (case_id, created_timestamp, modified_timestamp,
                                  simulation_end_time, n_restarts, file_hashes)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (case_id,
              provenance.get('created_timestamp'),
              provenance.get('modified_timestamp'),
              provenance.get('t_final'),
              provenance.get('n_restarts'),
              provenance.get('file_hashes')))

    def update_case(self, case_id, hash_tag, hash_time, metadata, species, flux, provenance):
        """
        Update an existing case in the database

        Parameters:
        -----------
        case_id : int
            Database case ID
        Other parameters same as insert_case()
        """
        # Update cases table
        self.conn.execute("""
            UPDATE cases
            SET input_hash = ?, hash_tag = ?, hash_time = ?, last_updated = CURRENT_TIMESTAMP,
                t_final = ?, i_final = ?
            WHERE case_id = ?
        """, (provenance.get('input_hash'), hash_tag, hash_time, provenance.get('t_final'),
              provenance.get('i_final'), case_id))

        # Delete and re-insert metadata
        self.conn.execute("DELETE FROM metadata WHERE case_id = ?", (case_id,))
        self._insert_metadata(case_id, metadata, species)

        # Delete and re-insert flux data
        self.conn.execute("DELETE FROM fluxes WHERE case_id = ?", (case_id,))
        if flux:
            self._insert_flux(case_id, flux)

        # Update provenance
        self.conn.execute("DELETE FROM provenance WHERE case_id = ?", (case_id,))
        self._insert_provenance(case_id, provenance)

        self.conn.commit()

    def get_case_by_path(self, case_path):
        """
        Retrieve case by path

        Parameters:
        -----------
        case_path : str
            Case path

        Returns:
        --------
        dict or None : Case data or None if not found
        """
        cursor = self.conn.execute("""
            SELECT * FROM cases WHERE case_path = ?
        """, (case_path,))

        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def get_case_by_hash(self, input_hash):
        """
        Retrieve case by input hash (supports partial match)

        Parameters:
        -----------
        input_hash : str
            Input hash (full or first 8 characters)

        Returns:
        --------
        dict or None : Case data or None if not found
        """
        cursor = self.conn.execute("""
            SELECT * FROM cases WHERE input_hash LIKE ?
        """, (input_hash + '%',))

        row = cursor.fetchone()
        if row:
            return dict(row)
        return None

    def _build_filter_clause(self, filters):
        """
        Build SQL WHERE clause and parameters from filters dict

        Parameters:
        -----------
        filters : dict
            Filter dictionary

        Returns:
        --------
        tuple : (sql_clause, params) to append to query
        """
        if not filters:
            return '', []

        sql_parts = []
        params = []

        for key, value in filters.items():
            if isinstance(value, tuple) and len(value) == 2:
                # Range query
                sql_parts.append(f"m.{key} BETWEEN ? AND ?")
                params.extend(value)
            elif isinstance(value, str) and '%' in value:
                # Pattern match
                sql_parts.append(f"c.{key} LIKE ?")
                params.append(value)
            else:
                # Exact match
                sql_parts.append(f"m.{key} = ?")
                params.append(value)

        sql_clause = ' AND ' + ' AND '.join(sql_parts) if sql_parts else ''
        return sql_clause, params

    def get_complete_case(self, case_id):
        """
        Retrieve complete case record including all related data

        Parameters:
        -----------
        case_id : int
            Case ID

        Returns:
        --------
        dict : Complete case record with metadata, species, flux, and provenance
        """
        # Get case info
        cursor = self.conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,))
        row = cursor.fetchone()
        if not row:
            return None

        result = dict(row)

        # Get metadata
        cursor = self.conn.execute("SELECT * FROM metadata WHERE case_id = ?", (case_id,))
        metadata_row = cursor.fetchone()
        if metadata_row:
            result['metadata'] = dict(metadata_row)
            # Parse species data from JSON
            if 'species_data' in result['metadata']:
                result['species'] = json.loads(result['metadata']['species_data'])
                del result['metadata']['species_data']

        # Get flux data
        cursor = self.conn.execute("""
            SELECT * FROM fluxes WHERE case_id = ? ORDER BY species_idx
        """, (case_id,))
        flux_rows = cursor.fetchall()
        if flux_rows:
            result['fluxes'] = []
            for flux_row in flux_rows:
                flux_dict = dict(flux_row)
                # Parse JSON arrays
                flux_dict['particle_flux'] = json.loads(flux_dict['particle_flux'])
                flux_dict['energy_flux'] = json.loads(flux_dict['energy_flux'])
                flux_dict['momentum_flux'] = json.loads(flux_dict['momentum_flux'])
                result['fluxes'].append(flux_dict)

        # Get provenance
        cursor = self.conn.execute("SELECT * FROM provenance WHERE case_id = ?", (case_id,))
        prov_row = cursor.fetchone()
        if prov_row:
            result['provenance'] = dict(prov_row)
            # Parse file_hashes from JSON
            if 'file_hashes' in result['provenance']:
                result['provenance']['file_hashes'] = json.loads(result['provenance']['file_hashes'])

        return result

    def query(self, filters=None, show_fields='minimal'):
        """
        Query database with filters

        Parameters:
        -----------
        filters : dict, optional
            Filter dictionary. Examples:
            {'q': (1.0, 2.0)}  - Range query
            {'shear': 0.8}      - Exact match
            {'case_path': '%scan%'}  - Pattern match
        show_fields : str, optional
            What fields to show:
            'minimal' - Just case_path and input_hash (default)
            'standard' - Add physics parameters (q, shear, kappa, etc.)
            'all' - All available fields

        Returns:
        --------
        pandas.DataFrame : Query results
        """
        # Build SELECT based on requested fields
        if show_fields == 'minimal':
            select_clause = """
                SELECT c.case_path,
                       SUBSTR(c.input_hash, 1, 8) as input_hash
            """
        elif show_fields == 'all':
            select_clause = """
                SELECT c.case_path,
                       SUBSTR(c.input_hash, 1, 8) as input_hash,
                       c.t_final, c.i_final,
                       m.q, m.shear, m.kappa, m.delta, m.shift,
                       m.n_n, m.n_radial, m.n_theta, m.n_species, m.n_field,
                       m.beta_star, m.gamma_e, m.gamma_p, m.mach,
                       m.rmin, m.rmaj, m.rho
            """
        else:  # 'standard'
            select_clause = """
                SELECT c.case_path,
                       SUBSTR(c.input_hash, 1, 8) as input_hash,
                       c.t_final,
                       m.q, m.shear, m.kappa, m.delta,
                       m.n_n, m.n_radial, m.n_theta, m.n_species,
                       m.beta_star, m.gamma_e
            """

        sql = select_clause + """
            FROM cases c
            LEFT JOIN metadata m ON c.case_id = m.case_id
            WHERE 1=1
        """

        filter_clause, params = self._build_filter_clause(filters)
        sql += filter_clause

        df = pd.read_sql_query(sql, self.conn, params=params)

        # Add row numbers starting from 1 (inserted at the beginning)
        if len(df) > 0:
            df.insert(0, '#', range(1, len(df) + 1))

        return df

    def get_flux(self,case_id,species=0):
        """
        Retrieve flux data for a specific case and species

        Parameters:
        -----------
        case_id : int
            Case ID
        species : int
            Species index

        Returns:
        --------
        dict : Flux data with particle, energy, momentum arrays
        """
        cursor = self.conn.execute("""
            SELECT particle_flux, energy_flux, momentum_flux
            FROM fluxes
            WHERE case_id = ? AND species_idx = ?
        """, (case_id, species))

        row = cursor.fetchone()
        if row:
            return {
                'particle': json.loads(row[0]),
                'energy': json.loads(row[1]),
                'momentum': json.loads(row[2]),
            }
        return None

    def query_with_flux(self, filters=None, flux_type=None, species=None):
        """
        Query database and include flux data

        Parameters:
        -----------
        filters : dict, optional
            Filter dictionary for cases
        flux_type : str or list, optional
            Which flux types to include: 'particle', 'energy', 'momentum', or list of these
            If None, includes all flux types
        species : int or list, optional
            Which species to include (0-indexed)
            If None, includes all species

        Returns:
        --------
        pandas.DataFrame : Query results with flux columns added
        """
        # Get basic case info
        df = self.query(filters=filters, show_fields='minimal')

        if len(df) == 0:
            return df

        # Get case_ids for flux lookup
        case_ids_sql = """
            SELECT c.case_id, c.case_path
            FROM cases c
            LEFT JOIN metadata m ON c.case_id = m.case_id
            WHERE 1=1
        """

        filter_clause, params = self._build_filter_clause(filters)
        case_ids_sql += filter_clause

        case_mapping = pd.read_sql_query(case_ids_sql, self.conn, params=params)
        path_to_id = dict(zip(case_mapping['case_path'], case_mapping['case_id']))

        # Determine which flux types to get
        if flux_type is None:
            flux_types = ['particle', 'energy', 'momentum']
        elif isinstance(flux_type, str):
            flux_types = [flux_type]
        else:
            flux_types = flux_type

        # Get flux data and add columns
        for case_path in df['case_path']:
            case_id = path_to_id.get(case_path)
            if case_id is None:
                continue

            # Get all flux data for this case
            cursor = self.conn.execute("""
                SELECT species_idx, particle_flux, energy_flux, momentum_flux
                FROM fluxes
                WHERE case_id = ?
                ORDER BY species_idx
            """, (case_id,))

            flux_rows = cursor.fetchall()

            # Determine which species to include
            if species is None:
                species_list = [row[0] for row in flux_rows]
            elif isinstance(species, int):
                species_list = [species]
            else:
                species_list = species

            # Add flux columns for requested species and types
            idx = df[df['case_path'] == case_path].index[0]

            for sp_idx in species_list:
                matching_rows = [r for r in flux_rows if r[0] == sp_idx]
                if not matching_rows:
                    continue

                row = matching_rows[0]
                flux_data = {
                    'particle': json.loads(row[1]),
                    'energy': json.loads(row[2]),
                    'momentum': json.loads(row[3])
                }

                for ftype in flux_types:
                    if ftype in flux_data:
                        # Store as sum (total flux across all ky modes)
                        total_flux = np.sum(flux_data[ftype])
                        col_name = f"s{sp_idx}_{ftype[0]}"  # e.g., s0_e for species 0 energy
                        if col_name not in df.columns:
                            df[col_name] = None
                        df.at[idx, col_name] = total_flux

        return df

    def find_duplicate_inputs(self):
        """
        Find cases with identical input configurations

        Returns:
        --------
        pandas.DataFrame : Cases grouped by input_hash showing duplicates
        """
        sql = """
            SELECT input_hash, COUNT(*) as count, GROUP_CONCAT(case_path, '; ') as cases
            FROM cases
            WHERE input_hash IS NOT NULL
            GROUP BY input_hash
            HAVING count > 1
            ORDER BY count DESC
        """
        df = pd.read_sql_query(sql, self.conn)
        return df

    def print_statistics(self):
        """Print database statistics"""
        # Count total cases
        cursor = self.conn.execute("SELECT COUNT(*) FROM cases")
        n_cases = cursor.fetchone()[0]

        # Get database size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        db_size_mb = db_size / (1024 * 1024)

        # Get q range
        cursor = self.conn.execute("SELECT MIN(q), MAX(q) FROM metadata")
        q_range = cursor.fetchone()

        # Get last update time
        cursor = self.conn.execute("SELECT MAX(last_updated) FROM cases")
        last_update = cursor.fetchone()[0]

        # Count unique input configurations
        cursor = self.conn.execute("SELECT COUNT(DISTINCT input_hash) FROM cases WHERE input_hash IS NOT NULL")
        n_unique_inputs = cursor.fetchone()[0]

        # Count duplicates
        cursor = self.conn.execute("""
            SELECT COUNT(*) FROM (
                SELECT input_hash FROM cases
                WHERE input_hash IS NOT NULL
                GROUP BY input_hash
                HAVING COUNT(*) > 1
            )
        """)
        n_duplicate_inputs = cursor.fetchone()[0]

        print(f"\nCGYRO AI Database Statistics")
        print(f"=" * 50)
        print(f"Database file: {self.db_path}")
        print(f"Total cases: {n_cases}")
        print(f"Unique input configurations: {n_unique_inputs}")
        if n_duplicate_inputs > 0:
            print(f"Duplicate input configs: {n_duplicate_inputs}")
        print(f"Database size: {db_size_mb:.2f} MB")

        # Only show q range if we have data
        if q_range[0] is not None and q_range[1] is not None:
            print(f"q range: [{q_range[0]:.2f}, {q_range[1]:.2f}]")

        if last_update:
            print(f"Last updated: {last_update}")

        print(f"=" * 50)

    def export_csv(self, df, output_path):
        """
        Export query results to CSV

        Parameters:
        -----------
        df : pandas.DataFrame
            Query results
        output_path : str
            Output file path
        """
        df.to_csv(output_path, index=False)
        print(f"Exported {len(df)} cases to {output_path}")

    def export_json(self, df, output_path):
        """
        Export query results to JSON

        Parameters:
        -----------
        df : pandas.DataFrame
            Query results
        output_path : str
            Output file path
        """
        df.to_json(output_path, orient='records', indent=2)
        print(f"Exported {len(df)} cases to {output_path}")
