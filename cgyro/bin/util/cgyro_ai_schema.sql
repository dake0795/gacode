-- CGYRO AI Database Schema
-- SQLite database for lightweight CGYRO simulation database
-- Optimized for AI training and fast queries

-- Cases table: Main index of simulation cases
CREATE TABLE IF NOT EXISTS cases (
    case_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_path TEXT UNIQUE NOT NULL,
    case_name TEXT NOT NULL,
    input_hash TEXT,
    hash_tag TEXT,
    hash_time TEXT,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    t_final REAL,
    i_final INTEGER,
    git_version TEXT
);

-- Metadata table: Grid and equilibrium parameters
CREATE TABLE IF NOT EXISTS metadata (
    case_id INTEGER PRIMARY KEY,
    -- Grid parameters
    n_n INTEGER,
    n_radial INTEGER,
    n_theta INTEGER,
    n_species INTEGER,
    n_field INTEGER,
    n_energy INTEGER,
    n_xi INTEGER,
    -- Equilibrium parameters
    rmin REAL,
    rmaj REAL,
    q REAL,
    shear REAL,
    shift REAL,
    kappa REAL,
    s_kappa REAL,
    delta REAL,
    s_delta REAL,
    zeta REAL,
    s_zeta REAL,
    zmag REAL,
    dzmag REAL,
    -- Plasma parameters
    rho REAL,
    ky0 REAL,
    betae_unit REAL,
    beta_star REAL,
    gamma_e REAL,
    gamma_p REAL,
    mach REAL,
    z_eff REAL,
    -- Species data (JSON array)
    species_data TEXT,
    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

-- Fluxes table: Time-averaged flux data per species
CREATE TABLE IF NOT EXISTS fluxes (
    flux_id INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id INTEGER NOT NULL,
    species_idx INTEGER NOT NULL,
    time_window TEXT,
    time_max TEXT,
    particle_flux TEXT,
    energy_flux TEXT,
    momentum_flux TEXT,
    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

-- Provenance table: Simulation tracking and metadata
CREATE TABLE IF NOT EXISTS provenance (
    case_id INTEGER PRIMARY KEY,
    created_timestamp TIMESTAMP,
    modified_timestamp TIMESTAMP,
    simulation_end_time REAL,
    git_commit TEXT,
    n_restarts INTEGER,
    file_hashes TEXT,
    FOREIGN KEY(case_id) REFERENCES cases(case_id) ON DELETE CASCADE
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_case_path ON cases(case_path);
CREATE INDEX IF NOT EXISTS idx_input_hash ON cases(input_hash);
CREATE INDEX IF NOT EXISTS idx_hash_tag ON cases(hash_tag);
CREATE INDEX IF NOT EXISTS idx_hash_time ON cases(hash_time);
CREATE INDEX IF NOT EXISTS idx_last_updated ON cases(last_updated);

CREATE INDEX IF NOT EXISTS idx_q ON metadata(q);
CREATE INDEX IF NOT EXISTS idx_shear ON metadata(shear);
CREATE INDEX IF NOT EXISTS idx_kappa ON metadata(kappa);
CREATE INDEX IF NOT EXISTS idx_n_n ON metadata(n_n);
CREATE INDEX IF NOT EXISTS idx_n_radial ON metadata(n_radial);

CREATE INDEX IF NOT EXISTS idx_flux_case ON fluxes(case_id);
CREATE INDEX IF NOT EXISTS idx_flux_species ON fluxes(species_idx);

-- Views for common queries

-- View: All cases with basic metadata
CREATE VIEW IF NOT EXISTS v_cases_summary AS
SELECT
    c.case_id,
    c.case_path,
    c.case_name,
    c.last_updated,
    c.t_final,
    m.q,
    m.shear,
    m.kappa,
    m.n_n,
    m.n_radial,
    m.n_theta,
    m.n_species,
    m.beta_star
FROM cases c
LEFT JOIN metadata m ON c.case_id = m.case_id;

-- View: Cases with time-averaged energy flux
CREATE VIEW IF NOT EXISTS v_cases_with_flux AS
SELECT
    c.case_id,
    c.case_path,
    m.q,
    m.shear,
    m.kappa,
    f.species_idx,
    f.energy_flux
FROM cases c
LEFT JOIN metadata m ON c.case_id = m.case_id
LEFT JOIN fluxes f ON c.case_id = f.case_id;
