import numpy as np
from cgyro.data import cgyrodata
import matplotlib.pyplot as plt

from pathlib import Path

base = Path(__file__).resolve().parent

def map1d(f2d,q):
    nr    = f2d.shape[0]
    nt    = f2d.shape[1]
    ntime = f2d.shape[2]
    px = np.arange(nr)-nr//2
    f1d = np.zeros([nr,nt,ntime],dtype=complex)
    anorm = f1d[nr//2,nt//2,:]

    for ir in range(nr):
        f1d[ir,:,:] = f2d[ir,:,:]*np.exp(-2*np.pi*1j*px[ir]*q)
    f1d = f1d.reshape(nr*nt,ntime)

    return f1d,anorm

def compute_imin_imax(sim, frac_start=0.1, window=200):
    """
    Automatically chooses imin and imax:
    
    - imax = minimum time length shared by all bigfields
    - imin = index where exponential growth begins (after transients)
             If detection fails, default to frac_start * imax.

    Parameters
    ----------
    sim : cgyrodata
        Loaded simulation object with bigfields.
    frac_start : float
        Fallback fraction of data to skip (0.1 = skip first 10%).
    window : int
        Window size for slope-smoothing in log-amplitude.

    Returns
    -------
    imin : int
    imax : int
    """

    # ---------------------------------------------------------
    # (1) Determine imax from available bigfield timesteps
    # ---------------------------------------------------------
    ts = sim.get_bigfield_timesteps()
    imax = min(ts.values())

    # Safety: require at least 2*window samples
    if imax < 2 * window:
        raise ValueError("Not enough time samples for automatic imin detection.")

    # ---------------------------------------------------------
    # (2) Use phi amplitude to detect linear phase
    # ---------------------------------------------------------
    # Use kxky_phi (cleanest field) to extract amplitude time series
    # Center mode as amplitude estimator
    phi = sim.kxky_phi[:, :, 0, :imax]   # take n=0 mode

    ir0 = sim.n_radial // 2
    it0 = sim.theta_plot // 2

    amp = np.abs(phi[ir0, it0, :])

    # Prevent zeros
    amp[amp == 0] = np.min(amp[amp > 0])

    # Smooth log-amplitude derivative
    logamp = np.log(amp)
    deriv = np.gradient(logamp)
    deriv_smooth = np.convolve(deriv, np.ones(window) / window, mode='same')

    # ---------------------------------------------------------
    # (3) Determine imin where growth rate stabilizes
    # ---------------------------------------------------------
    # Target: derivative becomes "flat" → exponential growth
    gamma_est = np.mean(deriv_smooth[window:2 * window])
    threshold = 0.1 * abs(gamma_est)     # tolerance

    imin_candidates = np.where(np.abs(deriv_smooth - gamma_est) < threshold)[0]

    if len(imin_candidates) == 0:
        # Fallback: skip first frac_start of data
        imin = int(frac_start * imax)
    else:
        imin = imin_candidates[0]

    # Safety clamp
    imin = max(0, min(imin, imax - window))

    print(f"INFO: (compute_imin_imax) imin={imin}, imax={imax}")
    return imin, imax

def extract_data(dir, compute_bounds = False):
    ostr = ['phi','apar','bpar', 'ni', 'ne', 'vi', 've', "ei", "ee"]
    ostr_len = len(ostr)

    sim = cgyrodata(dir)
    sim.getbigfield()
    t = sim.t
    theta = sim.theta
    thetab = sim.thetab
    n_radial  = sim.n_radial  
    n_theta   = sim.theta_plot
    n_species = sim.n_species
    n_time = len(t)
    tmax = t[-1]

    timesteps = sim.get_bigfield_timesteps()
    print(f"INFO: (timesteps) {timesteps}")

    if compute_bounds:
        imin, imax = compute_imin_imax(sim)
    else:
        imin = 0
        imax = sim.n_n - 1

    ovec = {}

    # ---------------------------------------------------
    # FIELD VARIABLES (4D)
    # shape: (kx, theta, n, t)
    # ---------------------------------------------------

    # phi
    y = sim.kxky_phi[:, :, 0, imin:imax]
    ovec['phi'], a0 = map1d(y, sim.q)

    # apar
    y = sim.kxky_apar[:, :, 0, imin:imax]
    ovec['apar'], a0 = map1d(y, sim.q)

    # bpar
    y = sim.kxky_bpar[:, :, 0, imin:imax]
    ovec['bpar'], a0 = map1d(y, sim.q)

    # ---------------------------------------------------
    # MOMENT VARIABLES (5D)
    # shape: (kx, theta, species, n, t)
    # species index: 0=ions, 1=electrons
    # ---------------------------------------------------

    # ion density fluctuation ni'
    y = sim.kxky_n[:, :, 0, 0, imin:imax]
    ovec['ni'], a0 = map1d(y, sim.q)

    # electron density fluctuation ne'
    y = sim.kxky_n[:, :, 1, 0, imin:imax]
    ovec['ne'], a0 = map1d(y, sim.q)

    # ion parallel velocity vi
    y = sim.kxky_v[:, :, 0, 0, imin:imax]
    ovec['vi'], a0 = map1d(y, sim.q)

    # electron parallel velocity ve
    y = sim.kxky_v[:, :, 1, 0, imin:imax]
    ovec['ve'], a0 = map1d(y, sim.q)

    # ion energy moment ei
    y = sim.kxky_e[:, :, 0, 0, imin:imax]
    ovec['ei'], a0 = map1d(y, sim.q)

    # electron energy moment ee
    y = sim.kxky_e[:, :, 1, 0, imin:imax]
    ovec['ee'], a0 = map1d(y, sim.q)

    dt = t[1] - t[0]

    return ovec,dt


if __name__ == "__main__":
    # example usage
    mydir = (base / "data" / "ntheta=64_nradial=24" / "beta_e=0.001" / "CBC_standard").as_posix() + "/"
    print("Using data directory:", mydir)
    sim = cgyrodata(str(mydir))
    sim.getbigfield()
    imin,imax = compute_imin_imax(sim)
    ovec = extract_data(mydir)


