"""Multivariate Normal distribution for mindtorch_v2."""

import numpy as np


class MultivariateNormal:
    """Multivariate Normal (Gaussian) distribution.

    Args:
        loc: Mean of the distribution (length D vector)
        covariance_matrix: Covariance matrix (D x D positive definite matrix)
        precision_matrix: Precision matrix (inverse of covariance)
        scale_tril: Lower triangular factor of covariance
    """

    arg_constraints = {}
    has_rsample = True

    def __init__(self, loc, covariance_matrix=None, precision_matrix=None,
                 scale_tril=None, validate_args=None):
        # Convert to numpy arrays
        if hasattr(loc, 'numpy'):
            self.loc = loc.numpy()
        else:
            self.loc = np.asarray(loc)

        # Handle different parameterizations
        if covariance_matrix is not None:
            if hasattr(covariance_matrix, 'numpy'):
                self._cov = covariance_matrix.numpy()
            else:
                self._cov = np.asarray(covariance_matrix)
            # Add small regularization for numerical stability
            self._cov = self._cov + np.eye(self._cov.shape[-1]) * 1e-6
            try:
                self._scale_tril = np.linalg.cholesky(self._cov)
            except np.linalg.LinAlgError:
                # Fallback: use eigenvalue decomposition and clamp negative eigenvalues
                eigvals, eigvecs = np.linalg.eigh(self._cov)
                eigvals = np.maximum(eigvals, 1e-6)
                self._cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
                self._scale_tril = np.linalg.cholesky(self._cov)
        elif scale_tril is not None:
            if hasattr(scale_tril, 'numpy'):
                self._scale_tril = scale_tril.numpy()
            else:
                self._scale_tril = np.asarray(scale_tril)
            self._cov = self._scale_tril @ self._scale_tril.T
        elif precision_matrix is not None:
            if hasattr(precision_matrix, 'numpy'):
                prec = precision_matrix.numpy()
            else:
                prec = np.asarray(precision_matrix)
            self._cov = np.linalg.inv(prec)
            try:
                self._scale_tril = np.linalg.cholesky(self._cov)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(self._cov)
                eigvals = np.maximum(eigvals, 1e-6)
                self._cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
                self._scale_tril = np.linalg.cholesky(self._cov)
        else:
            raise ValueError("One of covariance_matrix, precision_matrix, or scale_tril must be specified")

        # Determine batch and event shapes
        self._event_shape = (self.loc.shape[-1],)
        self._batch_shape = self.loc.shape[:-1]

    @property
    def batch_shape(self):
        return self._batch_shape

    @property
    def event_shape(self):
        return self._event_shape

    @property
    def mean(self):
        from mindtorch_v2 import tensor
        return tensor(self.loc)

    @property
    def variance(self):
        from mindtorch_v2 import tensor
        return tensor(np.diag(self._cov))

    @property
    def covariance_matrix(self):
        from mindtorch_v2 import tensor
        return tensor(self._cov)

    @property
    def scale_tril(self):
        from mindtorch_v2 import tensor
        return tensor(self._scale_tril)

    def sample(self, sample_shape=()):
        from mindtorch_v2 import tensor
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        # Generate standard normal samples
        shape = sample_shape + self._batch_shape + self._event_shape
        z = np.random.standard_normal(shape)

        # Transform: loc + scale_tril @ z
        # For batched case, need to handle broadcasting
        samples = self.loc + np.einsum('...ij,...j->...i', self._scale_tril, z)

        return tensor(samples.astype(np.float32))

    def rsample(self, sample_shape=()):
        return self.sample(sample_shape)

    def log_prob(self, value):
        from mindtorch_v2 import tensor
        if hasattr(value, 'numpy'):
            value = value.numpy()
        else:
            value = np.asarray(value)

        diff = value - self.loc
        d = self.loc.shape[-1]

        # log det of covariance
        log_det = 2 * np.sum(np.log(np.diag(self._scale_tril)))

        # Solve for x in L @ x = diff using triangular solve
        # Then compute diff^T @ cov^{-1} @ diff = x^T @ x
        x = np.linalg.solve(self._scale_tril, diff)
        maha = np.sum(x * x, axis=-1)

        log_prob = -0.5 * (d * np.log(2 * np.pi) + log_det + maha)
        return tensor(log_prob.astype(np.float32))

    def entropy(self):
        from mindtorch_v2 import tensor
        d = self.loc.shape[-1]
        log_det = 2 * np.sum(np.log(np.diag(self._scale_tril)))
        entropy = 0.5 * (d * (1 + np.log(2 * np.pi)) + log_det)
        return tensor(entropy.astype(np.float32))
