#!/usr/bin/env python
"""
cgyro_dbplot - CGYRO Interactive Database Plotter

Streamlit-based interactive visualization tool for CGYRO databases.
Supports filtering, plotting, and model predictions overlay.
"""

import os
import sys
import argparse
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cgyro_ai_db import AIDB
import pickle


def load_model(model_path):
    """Load trained ML model"""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def predict_with_model(model_pkg, X):
    """Make predictions with uncertainty"""
    model = model_pkg['model']
    scaler_X = model_pkg['scaler_X']
    scaler_y = model_pkg['scaler_y']

    X_scaled = scaler_X.transform(X)

    # GP model
    if model_pkg['model_type'] == 'gp':
        y_pred_scaled = model.predict(X_scaled)
        y_std_scaled = np.zeros((len(X), model_pkg['n_outputs']))
        for i, estimator in enumerate(model.estimators_):
            _, std = estimator.predict(X_scaled, return_std=True)
            y_std_scaled[:, i] = std
    else:  # RF
        y_pred_scaled = model.predict(X_scaled)
        y_std_scaled = np.zeros((len(X), model_pkg['n_outputs']))
        for i, estimator in enumerate(model.estimators_):
            tree_preds = np.array([tree.predict(X_scaled) for tree in estimator.estimators_])
            y_std_scaled[:, i] = np.std(tree_preds, axis=0)

    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_std = y_std_scaled * scaler_y.scale_

    return y_pred, y_std


def main():
    st.set_page_config(page_title="CGYRO Database Explorer", layout="wide")

    st.title("🔬 CGYRO Database Explorer")
    st.markdown("Interactive visualization and exploration of CGYRO simulation databases")

    # Sidebar: Configuration
    st.sidebar.header("⚙️ Configuration")

    # Database selection - check for command line arg or use default
    default_db = os.path.expanduser("~/.cgyro/cgyro_ai.db")

    # Check if database was passed via command line
    if len(sys.argv) > 2 and sys.argv[1] == '-db':
        initial_db = os.path.expanduser(sys.argv[2])
    else:
        initial_db = default_db

    db_path = st.sidebar.text_input("Database path", value=initial_db)

    if not os.path.exists(db_path):
        st.error(f"❌ Database not found: {db_path}")
        st.info("Run `cgyro --ai -scan -flux` to create a database")
        return

    # Load database
    try:
        db = AIDB(db_path)
        st.sidebar.success(f"✓ Database loaded")
    except Exception as e:
        st.error(f"Error loading database: {e}")
        return

    # Get database statistics
    cursor = db.conn.execute("SELECT COUNT(*) FROM cases")
    n_cases = cursor.fetchone()[0]

    cursor = db.conn.execute("SELECT COUNT(*) FROM fluxes")
    n_flux = cursor.fetchone()[0]

    st.sidebar.metric("Total cases", n_cases)
    st.sidebar.metric("Cases with flux", n_flux // 2 if n_flux > 0 else 0)

    # Get parameter ranges for filters (lightweight query)
    cursor = db.conn.execute("""
        SELECT
            MIN(q), MAX(q),
            MIN(shear), MAX(shear),
            MIN(beta_star), MAX(beta_star)
        FROM metadata
    """)
    ranges = cursor.fetchone()
    q_min, q_max = ranges[0] or 0.0, ranges[1] or 5.0
    shear_min, shear_max = ranges[2] or -1.0, ranges[3] or 2.0
    beta_min, beta_max = ranges[4] or 0.0, ranges[5] or 0.1

    # Sidebar: Filters
    st.sidebar.header("🔍 Filters")

    st.sidebar.info("⚠️ Set filters below, then click 'Load Data'")

    # Parameter filters
    q_range = st.sidebar.slider(
        "q (safety factor)",
        float(q_min), float(q_max),
        (float(q_min), float(q_max)),
        key='q_slider'
    )

    shear_range = st.sidebar.slider(
        "shear",
        float(shear_min), float(shear_max),
        (float(shear_min), float(shear_max)),
        key='shear_slider'
    )

    beta_range = st.sidebar.slider(
        "beta_star",
        float(beta_min), float(beta_max),
        (float(beta_min), float(beta_max)),
        key='beta_slider'
    )

    # Max results limit
    max_results = st.sidebar.number_input(
        "Max results to load",
        min_value=10,
        max_value=10000,
        value=500,
        step=50,
        help="Limit number of cases to prevent slow loading"
    )

    # Load button
    load_data = st.sidebar.button("🔄 Load Data", type="primary", use_container_width=True)

    # Build filters for SQL query
    filter_params = {
        'q': q_range,
        'shear': shear_range,
        'beta_star': beta_range
    }

    # Load data only when button clicked or on first load
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
        st.session_state.df_filtered = None

    if load_data or not st.session_state.data_loaded:
        with st.spinner('Loading data from database...'):
            # Use query_with_flux to get flux data included
            df_base = db.query(filters=filter_params, show_fields='all')

            # Try to add flux data if available
            try:
                df_filtered = db.query_with_flux(filters=filter_params, flux_type=['particle', 'energy', 'momentum'])

                # Merge with full metadata
                if len(df_filtered) > 0:
                    # Remove duplicate columns
                    merge_cols = ['case_path']
                    df_filtered = df_filtered.merge(
                        df_base[[col for col in df_base.columns if col not in df_filtered.columns or col in merge_cols]],
                        on='case_path',
                        how='left'
                    )
            except Exception as e:
                st.warning(f"Could not load flux data: {e}. Showing cases without flux.")
                df_filtered = df_base

            # Apply max results limit
            if len(df_filtered) > max_results:
                st.warning(f"⚠️ {len(df_filtered)} cases match filters, showing first {max_results}")
                df_filtered = df_filtered.head(max_results)

            st.session_state.df_filtered = df_filtered
            st.session_state.data_loaded = True

        st.sidebar.success(f"✓ Loaded {len(st.session_state.df_filtered)} cases")

    df_filtered = st.session_state.df_filtered

    # Model loading (optional)
    st.sidebar.header("🤖 Model Overlay")
    model_file = st.sidebar.file_uploader("Load ML model (.pkl)", type=['pkl'])

    model_pkg = None
    if model_file is not None:
        try:
            model_pkg = pickle.load(model_file)
            st.sidebar.success(f"✓ Model loaded ({model_pkg['model_type'].upper()})")
            st.sidebar.metric("R² (test)", f"{model_pkg['test_metrics'][0]['r2']:.3f}")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")

    # Main content: Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Flux Plots", "🎯 Parity Plots", "ℹ️ Info"])

    # Check if data is loaded
    data_loaded = df_filtered is not None and len(df_filtered) > 0

    if data_loaded:
        st.sidebar.metric("Loaded cases", len(df_filtered))

    # Define flux columns (Y-axis options) - look for s0_e, s1_p, etc.
    import re
    flux_columns = []
    flux_labels = {}
    if data_loaded:
        flux_pattern = re.compile(r's(\d+)_([epm])$')
        for col in df_filtered.columns:
            match = flux_pattern.match(col)
            if match:
                species_idx = match.group(1)
                flux_type = match.group(2)
                flux_type_name = {'e': 'energy', 'p': 'particle', 'm': 'momentum'}[flux_type]
                flux_columns.append(col)
                flux_labels[col] = f"Species {species_idx} {flux_type_name} flux"

    # Define input parameter columns (X-axis options) - PHYSICS inputs only
    # Equilibrium and plasma parameters (NOT resolution parameters)
    input_params = ['q', 'shear', 'kappa', 'delta', 's_kappa', 's_delta',
                    'shift', 'zeta', 's_zeta', 'zmag', 'dzmag',
                    'beta_star', 'gamma_e', 'gamma_p', 'mach',
                    'rmin', 'rmaj', 'rho']

    # Filter to only parameters that exist in the dataframe
    available_inputs = [col for col in input_params if col in df_filtered.columns] if data_loaded else []

    # Tab 1: Flux plots
    with tab1:
        st.header("Flux vs Input Parameter Plots")
        if not data_loaded:
            st.info("👈 Set filters in sidebar and click 'Load Data' to plot")
        else:
            if len(flux_columns) == 0:
                st.warning("⚠️ No flux data found in loaded cases. Make sure cases have flux data in the database.")
            elif len(available_inputs) == 0:
                st.warning("⚠️ No input parameters found in loaded cases.")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    x_param = st.selectbox("X-axis (Input Parameter)", available_inputs,
                                          index=0 if 'q' in available_inputs else 0)
                with col2:
                    # Use the flux_labels for display
                    y_param_display = st.selectbox("Y-axis (Flux Output)",
                                                   [flux_labels.get(col, col) for col in flux_columns],
                                                   index=0)
                    # Get the actual column name
                    y_param = flux_columns[[flux_labels.get(col, col) for col in flux_columns].index(y_param_display)]

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(df_filtered[x_param], df_filtered[y_param], alpha=0.6, s=50, c='steelblue')
                ax.set_xlabel(x_param, fontsize=12)
                ax.set_ylabel(flux_labels.get(y_param, y_param), fontsize=12)
                ax.grid(True, alpha=0.3)
                ax.set_title(f"{flux_labels.get(y_param, y_param)} vs {x_param}", fontsize=14)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()

                # Show data statistics
                st.subheader("Data Statistics")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Number of cases", len(df_filtered))
                with col_b:
                    st.metric(f"{x_param} range", f"{df_filtered[x_param].min():.3f} - {df_filtered[x_param].max():.3f}")
                with col_c:
                    st.metric(f"{y_param} range", f"{df_filtered[y_param].min():.3e} - {df_filtered[y_param].max():.3e}")

    # Tab 2: Parity plots (requires flux data and model)
    with tab2:
        st.header("Parity Plots - Model Predictions vs Actual")

        if not data_loaded:
            st.info("👈 Load data first to see parity plots")
        elif model_pkg is None:
            st.info("Load a model (.pkl file) in the sidebar to see parity plots")
        else:
            # Need to get flux data for filtered cases
            st.info("Parity plot functionality coming soon - requires flux data extraction")

    # Tab 3: Info
    with tab3:
        st.header("Database Information")

        st.subheader("Schema")
        cursor = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            with st.expander(f"Table: {table}"):
                cursor = db.conn.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                col_df = pd.DataFrame(columns, columns=['cid', 'name', 'type', 'notnull', 'dflt_value', 'pk'])
                st.dataframe(col_df[['name', 'type']], use_container_width=True)

        st.subheader("Parameter Ranges")
        if data_loaded:
            numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
            ranges = []
            for col in numeric_cols:
                if col != '#':
                    ranges.append({
                        'Parameter': col,
                        'Min': f"{df_filtered[col].min():.3f}",
                        'Max': f"{df_filtered[col].max():.3f}",
                        'Mean': f"{df_filtered[col].mean():.3f}"
                    })
            st.dataframe(pd.DataFrame(ranges), use_container_width=True)
        else:
            st.info("Load data to see parameter ranges for filtered cases")

            # Show full database ranges instead
            st.caption("Full database ranges (all cases):")
            cursor = db.conn.execute("""
                SELECT
                    'q' as param, MIN(q), MAX(q), AVG(q) FROM metadata
                UNION ALL
                SELECT 'shear', MIN(shear), MAX(shear), AVG(shear) FROM metadata
                UNION ALL
                SELECT 'beta_star', MIN(beta_star), MAX(beta_star), AVG(beta_star) FROM metadata
            """)
            all_ranges = cursor.fetchall()
            range_df = pd.DataFrame(all_ranges, columns=['Parameter', 'Min', 'Max', 'Mean'])
            st.dataframe(range_df, use_container_width=True)

    db.close()


if __name__ == '__main__':
    main()
