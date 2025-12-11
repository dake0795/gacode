import numpy as np
from numpy.linalg import norm
import time

class StreamingDMD:
    """
    Minimal streaming DMD accumulator (no truncation step yet).
    Keeps Qx, Qy (orthonormal bases) and A, Gx, Gy as in your notes.

    Notation:
      tilde_x = Qx^* x,  tilde_y = Qy^* y
      A  = sum_i tilde_y_i tilde_x_i^T
      Gx = sum_i tilde_x_i tilde_x_i^T
      Gy = sum_i tilde_y_i tilde_y_i^T
    """

    def __init__(self, eps: float = 1e-8, dtype=np.complex128, r0: int | None = None, debug: bool = False):
        self.eps = eps
        self.dtype = dtype
        self.r0 = r0  
        self.Qx = self.Qy = None
        self.A = self.Gx = self.Gy = None
        self.n = None
        self.count = 0
        # The DMD modes
        self.mode = None
        self.debug = debug

    # --- initialization -----------------------------------------------------
    def _ensure_init(self, x: np.ndarray):
        if self.n is None:
            self.n = x.shape[0]
            self.Qx = np.zeros((self.n, 0), dtype=self.dtype)
            self.Qy = np.zeros((self.n, 0), dtype=self.dtype)
            self.A  = np.zeros((0, 0), dtype=self.dtype)
            self.Gx = np.zeros((0, 0), dtype=self.dtype)
            self.Gy = np.zeros((0, 0), dtype=self.dtype)

    @staticmethod
    def _as_vector(v):
        return np.asarray(v).reshape(-1)

    @staticmethod
    def _is_orthonormal(Q, tol=1e-12):
        if Q.shape[1] == 0:
            return True
        return np.linalg.norm(Q.conj().T @ Q - np.eye(Q.shape[1])) < tol

    # --- projection / expansion --------------------------------------------
    def _project_and_maybe_expand(self, Q, v):
        if Q.shape[1] == 0:
            hat = np.zeros((0,), dtype=self.dtype)
            residual = v.copy()
        else:
            hat = Q.conj().T @ v
            residual = v - Q @ hat
            # re-orthogonalize for numerical stability
            d_hat = Q.conj().T @ residual
            residual -= Q @ d_hat
            hat += d_hat

        v_norm = norm(v)
        rnorm = norm(residual)
        grew = False
        if rnorm > self.eps * v_norm:
            q_new = (residual / rnorm).reshape(-1, 1)

            Q = np.hstack([Q, q_new])
            grew = True
            alpha_new = np.vdot(q_new.reshape(-1), v)  # q_new^H v
            tilde_v = np.concatenate([hat, np.array([alpha_new], dtype=self.dtype)])
        else:
            tilde_v = hat


        return Q, tilde_v, grew

    # --- padding when bases grow ------------------------------------------
    def _pad_if_grew(self, grew_x, grew_y):
        # grew_x and grew_y not used, but keeping for compatibility
        rx, ry = self.Qx.shape[1], self.Qy.shape[1]
        if self.A.shape != (ry, rx):
            A_new = np.zeros((ry, rx), dtype=self.dtype)
            A_new[:self.A.shape[0], :self.A.shape[1]] = self.A
            self.A = A_new
        if self.Gx.shape != (rx, rx):
            Gx_new = np.zeros((rx, rx), dtype=self.dtype)
            Gx_new[:self.Gx.shape[0], :self.Gx.shape[1]] = self.Gx
            self.Gx = Gx_new
        if self.Gy.shape != (ry, ry):
            Gy_new = np.zeros((ry, ry), dtype=self.dtype)
            Gy_new[:self.Gy.shape[0], :self.Gy.shape[1]] = self.Gy
            self.Gy = Gy_new

    def _truncate_basis(self):
        """
        Step 3 (paper)
        """
        if self.r0 is None:
            return
        rx, ry = self.Qx.shape[1], self.Qy.shape[1]
        if (rx > self.r0 or ry > self.r0) and self.Gx.size > 0 and self.Gy.size > 0:
            # leading eigenvectors (Hermitian eig of PSD Gx,Gy)
            sx, Vx = np.linalg.eigh(self.Gx)
            sy, Vy = np.linalg.eigh(self.Gy)
            Vx = Vx[:, np.argsort(sx)[::-1][:self.r0]]
            Vy = Vy[:, np.argsort(sy)[::-1][:self.r0]]

            # IMPORTANT: use V^H (conj().T), not V.T, for complex data
            VxH = Vx.conj().T
            VyH = Vy.conj().T

            self.Gx = (VxH @ self.Gx @ Vx).astype(self.dtype)
            self.Gy = (VyH @ self.Gy @ Vy).astype(self.dtype)
            self.A  = (VyH @ self.A  @ Vx).astype(self.dtype)
            self.Qx = (self.Qx @ Vx).astype(self.dtype)
            self.Qy = (self.Qy @ Vy).astype(self.dtype)

    def update(self, pair):
        x, y = pair
        x = self._as_vector(x).astype(self.dtype)
        y = self._as_vector(y).astype(self.dtype)
        self._ensure_init(x)

        self.Qx, tilde_x, grew_x = self._project_and_maybe_expand(self.Qx, x)
        self.Qy, tilde_y, grew_y = self._project_and_maybe_expand(self.Qy, y)
        if grew_x or grew_y:
            self._pad_if_grew(grew_x, grew_y)

        # accumulate reduced matrices
        self.A  += np.outer(tilde_y, tilde_x.conj())
        self.Gx += np.outer(tilde_x, tilde_x.conj())
        self.Gy += np.outer(tilde_y, tilde_y.conj())

        # Step 3: basis truncation if rank exceeds r0
        self._truncate_basis()

        if self.debug:
            if not self._is_orthonormal(self.Qx):
                print("WARNING: Qx not orthonormal")
            if not self._is_orthonormal(self.Qy):
                print("WARNING: Qy not orthonormal")

        self.count += 1

    # convenience
    def matrices(self):
        return self.A, self.Gx, self.Gy, self.Qx, self.Qy

    def reduced_operator(self, ridge: float = 0.0):
        if self.Gx.size == 0:
            raise ValueError("No data yet.")
        M = self.Qx.conj().T @ self.Qy
        if ridge > 0.0:
            Gx_inv = np.linalg.pinv(self.Gx + ridge*np.eye(self.Gx.shape[0], dtype=self.dtype))
        else:
            Gx_inv = np.linalg.pinv(self.Gx)
        return (M @ self.A) @ Gx_inv


    def eig(self):
        """
        Step 5 (Hemati et al., 2014):
            Compute eigenvalues and DMD modes.

        Solves (Qxᴴ Qy A Gx⁺) w = λ w
        Returns:
            eigvals : array of eigenvalues λ_j
            modes   : array of full-state DMD modes Φ_j = Qx w_j
        """
        if self.Gx.size == 0 or self.A.size == 0:
            raise ValueError("No data yet.")

        # Reduced operator
        K_tilde = self.reduced_operator()

        # Eigen-decomposition
        eigvals, W = np.linalg.eig(K_tilde)

        # Full-state DMD modes (Φ = Qx W)
        modes = self.Qx @ W

        return eigvals, modes