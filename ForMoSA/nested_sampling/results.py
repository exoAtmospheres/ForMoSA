import numpy as np
from dataclasses import dataclass

from ForMoSA.core.errors import ForMoSAError

@dataclass
class NSResults:
    '''
    Dataclass to handle reading and storing results of the Nested Sampling algorithm

    Authors
    -------
    Allan Denis
    '''

    samples: np.ndarray
    weights: np.ndarray
    logl: np.ndarray
    logvol: np.ndarray
    logz: list[float, float]
    free_parameters: list[str]

    burn_in: int = 0

    # ===================
    # Post init
    # ===================

    def __post_init__(self):
        self.samples = np.asarray(self.samples, dtype=float)
        self.weights = np.asarray(self.weights, dtype=float)
        self.logl = np.asarray(self.logl, dtype=float)
        self.logvol = np.asarray(self.logvol, dtype=float)

        # Checks
        if self.samples.ndim != 2:
            raise ForMoSAError(f"<samples must be 2D, got shape {self.samples.shape}>")

        if not self.weights.shape == self.logl.shape == self.logvol.shape:
            raise ForMoSAError("<weights, logl and logvol must have the same shapes>")

        if (not isinstance(self.logz, list)) or len(self.logz) != 2:
            raise ForMoSAError("<logz must be a list of 2 elements>")

        if self.samples.shape[0] != self.weights.shape[0]:
            raise ForMoSAError("<samples and weights must have same length>")

        if len(self.free_parameters) != self.samples.shape[1]:
            raise ForMoSAError(f'<Number of free parameters ({len(self.free_parameters)}) does not correspond to the shape of samples: ({self.samples.shape})>')

        if self.burn_in > self.samples.shape[0]:
            raise ForMoSAError(f'<burn_in ({self.burn_in}) does not correspond to the shape of samples: ({self.samples.shape})>')

    # ===================
    # Properties
    # ===================

    @property
    def to_dict(self) -> dict:                                    # Dictionary view of the results
        return {
            "samples": self.samples.tolist(),
            "weights": self.weights.tolist(),
            "logl": self.logl.tolist(),
            "logvol": self.logvol.tolist(),
            "logz": self.logz,
            "burn_in": self.burn_in,
            "free_parameters": self.free_parameters
        }

    @property
    def median_parameters(self) -> dict[str, float]:     # Weighted posterior median for each parameter
        return {
            name: self._weighted_quantile(self.samples[self.burn_in:, i], self.weights[self.burn_in:], 0.5)
            for i, name in enumerate(self.free_parameters)
        }

    @property
    def param_samples_dict(self) -> dict[str, float]:             # Samples of each parameter
        return {name: self.samples[self.burn_in:, i] for i, name in enumerate(self.free_parameters)}

    @property
    def best_logL(self) -> float:                                 # Averaged value of logL
        return np.average(self.results.logl[self.burn_in:], weights = self.results.weights[self.burn_in:])

    # ===================
    # Class methods
    # ===================

    @classmethod
    def from_dict(cls, data: dict) -> "NSResults":
        '''
        Load NSResults from dictionary.

        Parameters
        ----------
        data : dict
            Dictionary of NSResults
        free_parameters : list[str]
            List of free parameters names

        Returns
        -------
        NSResults
            Instance of class NSResults

        Authors
        -------
        Allan Denis
        '''

        return cls(
            samples=np.asarray(data["samples"], dtype=float),
            weights=np.asarray(data["weights"], dtype=float),
            logl=np.asarray(data["logl"], dtype=float),
            logvol=np.asarray(data["logvol"], dtype=float),
            logz=data["logz"],
            burn_in=data["burn_in"],
            free_parameters=data["free_parameters"],
        )

    @classmethod
    def from_nestle(cls, res: dict, free_parameters: list[str]) -> "NSResults":
        '''
        Return an instance of NSResults based on the results of a Nestle algorithm.

        Parameters
        ----------
        res : dict
            Dictionary containing Nested Sampling results
        free_parameters : list[str]
            List of free parameters names

        Returns
        -------
        NestleResults
            An instance of NSResults

        Authors
        -------
        Allan Denis
        '''

        return cls(
            samples=res['samples'],
            weights=res['weights'],
            logl=res['logl'],
            logvol=res['logvol'],
            logz=[res['logz'], res['logzerr']],
            free_parameters=free_parameters
        )

    @classmethod
    def from_pymultinest(cls, results_path: str, free_parameters: list[str]) -> "NSResults":
        '''
        Return an instance of NSResults based on the results of a PyMultinest algorithm.

        Parameters
        ----------
        results_path : str | os.PathLike
            Path to the PyMultiNest output folder containing ``RAW_stats.dat``, ``RAW_ev.dat``, ``RAW_.txt``
        free_parameters : list[str]
            List of free parameters names

        Returns
        -------
        NSResults
            An instance of NSResults

        Authors
        -------
        Allan Denis
        '''

        # Read global evidence and error
        with open(f"{results_path}/RAW_stats.dat", 'rb') as f:
            line = f.readline().strip().split()
            logz = [float(line[5]), float(line[7])]

        # Read event file (samples, logl, logvol)
        sample_multi, logl_multi, logvol_multi = [], [], []
        with open(f"{results_path}/RAW_ev.dat", 'rb') as f:
            for line in f:
                parts = line.strip().split()
                sample_multi.append([float(p) for p in parts[:-3]])
                logl_multi.append(float(parts[-3]))
                logvol_multi.append(float(parts[-2]))

        # Read weights and match to samples
        samples, weights, logl, logvol = [], [], [], []
        with open(f"{results_path}/RAW_.txt", 'rb') as f:
            for line in f:
                parts = line.strip().split()
                point = [float(p) for p in parts[2:]]
                if point in sample_multi:
                    idx = sample_multi.index(point)
                    samples.append(point)
                    weights.append(float(parts[0]))
                    logl.append(logl_multi[idx])
                    logvol.append(logvol_multi[idx])

        # Convert to numpy arrays
        samples = np.asarray(samples)
        weights = np.asarray(weights)
        logl = np.asarray(logl)
        logvol = np.asarray(logvol)

        return cls(
            samples=samples,
            weights=weights,
            logl=logl,
            logvol=logvol,
            logz=logz,
            free_parameters=free_parameters
        )

    @classmethod
    def from_ultranest(cls, res: dict, free_parameters: list[str]) -> "NSResults":
        """
        Construct an instance of UltraNestResults from the UltraNest output folder.

        Parameters
        ----------
        res : dict
            Dictionary containing Nested Sampling results
        free_parameters : list[str]
            List of free parameters names

        Returns
        -------
        NSResults
            An instance of NSResults

        Authors
        -------
        Allan Denis
        """

        ws = res['weighted_samples']
        samples, weights, loglike, logz, logz_err = ws['points'], ws['weights'], ws['logl'], res['logz'], res['logzerr']
        logvol = np.log(weights) - loglike + logz

        return cls(
            samples=samples,
            weights=weights,
            logl=loglike,
            logvol=logvol,
            logz=[logz, logz_err],
            free_parameters=free_parameters
        )

    # ===================
    # Methods
    # ===================

    def _weighted_quantile(self, values: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
        '''
        Compute weighted quantile for 1D array.

        Parameters
        ----------
        values : np.ndarray
            Values we want to compute the quantile for
        weights : np.ndarray
            Weights
        q : float
            Quantile

        Returns
        -------
        np.ndarray
            Weighted quantile

        '''

        values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)

        # Initial checks
        if len(values) != len(weights):
            raise ForMoSAError('<values and weights must have same length>')

        if (not isinstance(q, float)) or (q < 0 or q > 1):
            raise ForMoSAError('Expected a float between 0 and 1 for quantile>')

        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

        cumsum = np.cumsum(weights)
        cumsum /= cumsum[-1]

        return np.interp(q, cumsum, values)

    def _quantile_parameters(self, q: float) -> dict[str, float]:
        '''
        Weighted posterior quantile for each free parameter.

        Parameters
        ----------
        q : float
            Quantile in [0, 1]

        Returns
        -------
        dict[str, float]
            Dictionary of parameter name associated to its quantile

        Authors
        -------
        Allan Denis
        '''

        return {
            name: self._weighted_quantile(self.samples[:, i], self.weights, q)
            for i, name in enumerate(self.free_parameters)
        }

    def _interval(self, sigma: int = 1) -> dict[str, tuple[float, float]]:
        '''
        Weighted interval for each parameter.

        sigma = 1 → [16%, 84%]
        sigma = 2 → [2%, 98%]

        Parameters
        ----------
        sigma : int
            Sigma value

        Returns
        -------
        dict[str, tuple[float, float]]: Dictionary of parameter name associated to its interval

        Authors
        -------
        Allan Denis
        '''

        if sigma == 1:
            q_low, q_high = 0.16, 0.84
        elif sigma == 2:
            q_low, q_high = 0.02, 0.98
        else:
            raise ForMoSAError("sigma must be 1 or 2")

        return {
            name: (self._weighted_quantile(self.samples[:, i], self.weights, q_low),
                     self._weighted_quantile(self.samples[:, i], self.weights, q_high))
            for i, name in enumerate(self.free_parameters)
        }

    def summary(self, sigma: int = 1, include_map: bool = True) -> str:
        '''
        Print a summary of the posterior distributions.

        Parameters
        ----------
        sigma : int
            Credible interval (1 or 2 sigma)
        include_map : bool
            hether to include MAP estimate

        Returns
        -------
        str
            Summary

        Authors
        -------
        Allan Denis
        '''

        lines = []
        lines.append("======== Nested Sampling Summary ====================")
        lines.append(f"LogZ = {self.logz[0]:.3f} ± {self.logz[1]:.3f}")
        lines.append("")
        lines.append(f"Posterior estimates (median ± {sigma} sigma interval)")
        lines.append("")

        if include_map:
            idx_map = np.argmax(self.logl)

        for i, name in enumerate(self.free_parameters):
            samples = self.samples[:, i]
            weights = self.weights

            median = self._weighted_quantile(samples, weights, 0.5)
            low, high = self._interval(sigma)[name]

            minus = median - low
            plus = high - median

            line = (
                f"{name:12s}: "
                f"{median:10.4f} "
                f"-{minus:8.4f} +{plus:8.4f} "
                f"[{low:.4f}, {high:.4f}]"
            )

            if include_map:
                map_val = self.samples[idx_map, i]
                line += f"  MAP={map_val:.4f}"

            lines.append(line)

        lines.append("=====================================================")

        return "\n".join(lines)