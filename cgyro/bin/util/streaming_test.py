import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from data_extraction import extract_data
from streaming_dmd import StreamingDMD
import time

def run_streaming_dmd_on_matrix(X, eps=1e-8, r0=None):
    """
    X: array (m, T)
       columns are snapshots x_0, x_1, ..., x_{T-1}
    """
    starting = time.time()
    sdmd = StreamingDMD(eps=eps, dtype=np.complex128, r0=r0)

    m, T = X.shape
    for k in range(T - 1):
        sdmd.update((X[:, k], X[:, k + 1]))

    ending = time.time()
    print(f"[INFO] Streaming DMD update time: {ending - starting:.2f} seconds")
    eigvals, modes = sdmd.eig()
    return eigvals, modes, sdmd


if __name__ == "__main__":

    # ---------------------------------------------------------
    # CLI arguments
    # ---------------------------------------------------------
    if len(sys.argv) < 4:
        print("Usage: python streaming_test.py <data path> <beta> <num_modes>")
        sys.exit(1)

    data_path = sys.argv[1]
    beta_val = sys.argv[2]
    num_modes = int(sys.argv[3])

    print("\nUsing data directory:", data_path)
    print("Using beta =", beta_val)
    print("Plotting top", num_modes, "modes")

    # ---------------------------------------------------------
    # 1. Extract CGYRO data
    # ---------------------------------------------------------
    data, dt = extract_data(data_path, compute_bounds=True)

    print("\nLoaded fields:")
    for k, v in data.items():
        print(f"  {k}: shape {v.shape}")

    # ---------------------------------------------------------
    # 2. Streaming DMD on fields
    # ---------------------------------------------------------
    target_fields = ["phi", "apar", "bpar"]

    results = {}

    for field in target_fields:
        X = data[field]
        print(f"\n[INFO] Running Streaming DMD on '{field}' with shape {X.shape}")

        eigvals, modes, sdmd = run_streaming_dmd_on_matrix(X, eps=1e-8, r0=20)

        print(f"[INFO] Finished DMD update steps for '{field}'.")
        print(f"[INFO] Rank of Qx = {sdmd.Qx.shape[1]}")

        omega = np.log(eigvals) / dt

        idx = np.argsort(omega.real)[::-1]
        omega_sorted = omega[idx]
        modes_sorted = modes[:, idx]

        print("\n================ Dominant DMD Modes for", field, "================")
        for j in range(min(5, len(omega_sorted))):
            print(f"Mode {j}: growth={omega_sorted[j].real:.4e}, freq={omega_sorted[j].imag:.4e}")

        results[field] = {
            "eigvals": eigvals,
            "modes": modes,
            "omega": omega,
            "omega_sorted": omega_sorted,
            "modes_sorted": modes_sorted,
            "sdmd": sdmd,
        }

    print("\n\n[INFO] All DMD computations completed.")

    # ---------------------------------------------------------
    # 3A. Dominant modes plot — top N modes ONLY
    # ---------------------------------------------------------
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)

    phys_points = {field: {"gamma": [], "freq": []} for field in target_fields}
    all_gamma_phys, all_freq_phys = [], []

    for field in target_fields:
        omega_top = results[field]["omega_sorted"][:num_modes]

        for w in omega_top:
            phys_points[field]["gamma"].append(w.real)
            phys_points[field]["freq"].append(w.imag)
            all_gamma_phys.append(w.real)
            all_freq_phys.append(w.imag)

    # Auto padding
    pad_g = 0.1 * (max(all_gamma_phys) - min(all_gamma_phys) + 1e-12)
    pad_w = 0.1 * (max(all_freq_phys) - min(all_freq_phys) + 1e-12)

    gamma_min = min(all_gamma_phys) - pad_g
    gamma_max = max(all_gamma_phys) + pad_g
    freq_min  = min(all_freq_phys) - pad_w
    freq_max  = max(all_freq_phys) + pad_w

    # ---------------------------------------------
    # Enforce minimum axis range [-1, 1]
    # ---------------------------------------------
    gamma_min = min(gamma_min, -1)
    gamma_max = max(gamma_max,  1)

    freq_min  = min(freq_min,  -1)
    freq_max  = max(freq_max,   1)

    # -------- Plot: Top N modes --------
    plt.figure(figsize=(8, 6))
    colors = {"phi": "red", "apar": "blue", "bpar": "black"}
    markers = {"phi": "o", "apar": "s", "bpar": "+"}

    for field in target_fields:
        gamma_top = phys_points[field]["gamma"]
        freq_top  = phys_points[field]["freq"]

        # Plot points
        plt.scatter(
            freq_top, gamma_top,
            marker=markers[field],
            facecolors='none' if field != "bpar" else colors[field],
            edgecolors=colors[field],
            s=120,
            linewidths=1.8,
            alpha=0.9,
            label=field
        )

        # Annotate each point with γ and ω
        for fval, gval in zip(freq_top, gamma_top):
            plt.annotate(
                f"γ={gval:.3f}\nω={fval:.3f}",
                (fval, gval),
                textcoords="offset points",
                xytext=(10, 6),
                ha='left',
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7)
            )

    plt.axhline(0, color='k', linestyle='--')
    plt.axvline(0, color='k', linestyle=':')
    plt.xlim(freq_min, freq_max)
    plt.ylim(gamma_min, gamma_max)
    plt.xlabel(r"$(a/c_s)\,\omega$")
    plt.ylabel(r"$(a/c_s)\,\gamma$")
    plt.title(f"Streaming DMD: Top {num_modes} Modes (β = {beta_val})")
    plt.legend()
    plt.grid(True, linestyle=':', linewidth=0.6)
    plt.tight_layout()

    # Output filename includes beta and num_modes
    physical_out = os.path.join(
        plot_dir, f"gamma_omega_top{num_modes}_beta_{beta_val}.png"
    )
    plt.savefig(physical_out, dpi=300)
    plt.close()

    print(f"[INFO] Saved dominant-mode plot to {physical_out}")
