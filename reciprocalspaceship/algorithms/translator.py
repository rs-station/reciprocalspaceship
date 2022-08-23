import numpy as np


class BaseTranslator:
    def __init__(self, H, phi_ref, phi, dhkl):
        """
        Translator class for determining the optimal translation
        vector to align two sets of phases.

        Parameters
        ----------
        H : np.ndarray
            Miller indices as an N x 3 array
        phi_ref : np.ndarray
            Reference phases in degrees (length N)
        phi : np.ndarray
            Other phases to align to reference in degrees (length N)
        dhkl : np.ndarray
            Interplane spacing of each Miller index (length N)

        Raises
        ------
        ValueError
            Raises ValueError if dimensions of arrays do not correspond
        """
        if H.shape[1] != 3:
            raise ValueError(f"Miller index array must be N x 3, given: {H.shape}")

        nrefls = H.shape[0]
        if len(phi_ref) != nrefls or len(phi) != nrefls or len(dhkl) != nrefls:
            raise ValueError(
                "All arrays must correspond to the same number of reflections."
            )

        # Provide inputs as attributes
        self.H = H
        self.phi_ref = phi_ref
        self.phi = phi
        self.dhkl = dhkl

        # Helper attributes
        self.translation = np.array([0.0, 0.0, 0.0])
        self.D = (np.deg2rad(phi) - np.deg2rad(phi_ref)) / (2.0 * np.pi)
        self.mask = self.dhkl > 0.0
        self.a_bounds = (0.0, 1.0)
        self.b_bounds = (0.0, 1.0)
        self.c_bounds = (0.0, 1.0)

    def set_mask(self, dmin):
        """
        Set mask to restrict the set of reflections to those with dhkl > dmin.

        Parameters
        ----------
        dmin : float
            Minimal dhkl to consider
        """
        self.mask = self.dhkl > dmin
        return

    def _phase_residual(self, t):
        """Phase residual in cycles"""
        if t.ndim == 1:
            residual = self.D[self.mask] - self.H[self.mask] @ t
        else:
            residual = self.D[self.mask][:, None] - self.H[self.mask] @ t.T
        return residual - np.round(residual)

    def evaluate(self, t):
        """
        Evaluate loss function using sum of the squared phase residuals (in cycles).

        Parameters
        ----------
        t : np.ndarray(3) or np.ndarray(N, 3)
            Translation vector(s) to evaluate

        Returns
        -------
        loss : float
        """
        return np.sum(np.square(self._phase_residual(t)), axis=0)

    def phase_rmsd(self):
        """
        Compute phase RMSD for current translation vector in degrees

        Returns
        -------
        rmsd : float
        """
        residual = self._phase_residual(self.translation) * 2 * np.pi
        rmsd = np.sqrt(np.mean(residual * residual))
        return np.rad2deg(rmsd)


class PolarTranslator(BaseTranslator):
    def fit(self, dmin=0.0):
        """
        Determine the best translation vector by global optimization.

        Parameters
        ----------
        dmin : float
            Minimal dhkl to consider (default: 0.0; all reflections)

        Returns
        -------
        translation : np.ndarray
        """
        from scipy.optimize import dual_annealing

        self.set_mask(dmin=dmin)

        result = dual_annealing(
            self.evaluate,
            x0=self.translation,
            bounds=(self.a_bounds, self.b_bounds, self.c_bounds),
            minimizer_kwargs={"tol": 1e-10},
        )
        if result.success:
            self.translation = result.x
            return self.translation
        else:
            raise RuntimeError("Global optimization failed")


class NonPolarTranslator(BaseTranslator):
    def fit(self, dmin=0.0):
        """
        Determine the best translation vector by brute force search.

        Parameters
        ----------
        dmin : float
            Minimal dhkl to consider (default: 0.0; all reflections)

        Returns
        -------
        translation : np.ndarray
        """
        self.set_mask(dmin=dmin)

        cases = np.array([0.0, 1.0 / 3.0, 0.5, 2.0 / 3.0])
        t_cases = np.vstack(np.meshgrid(cases, cases, cases)).reshape((3, -1)).T
        loss = self.evaluate(t_cases)
        self.translation = t_cases[np.argmin(loss)]

        return self.translation
