"""Small NumPy examples for the spherical-filters tutorial.

These are local approximation examples, not a validated tracking system.
P and R use the filter's current tangent basis. An appearance/kinematics
joint model and data association are deliberately outside this example.
"""

import numpy as np


def unit(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2 or not np.all(np.isfinite(x)):
        raise ValueError("Expected a finite vector with at least two entries")
    length = np.linalg.norm(x)
    if length < 1e-12:
        raise ValueError("A zero vector has no direction")
    return x / length


def tangent_basis(mu):
    """Deterministic orthonormal columns spanning the tangent at mu."""
    mu = unit(mu)
    identity = np.eye(mu.size)
    v = identity[:, 0] - mu
    if np.linalg.norm(v) < 1e-12:
        return identity[:, 1:]
    v /= np.linalg.norm(v)
    return (identity - 2 * np.outer(v, v))[:, 1:]


def log_map(mu, y):
    mu, y = unit(mu), unit(y)
    cosine = np.clip(mu @ y, -1.0, 1.0)
    tangent = y - cosine * mu
    sine = np.linalg.norm(tangent)
    if cosine < 0 and sine < 1e-8:
        raise ValueError("The log map is ambiguous near antipodal points")
    if sine < 1e-10:
        return tangent
    return np.arctan2(sine, cosine) * tangent / sine


def exp_map(mu, v):
    mu = unit(mu)
    v = np.asarray(v, dtype=float)
    if v.shape != mu.shape or not np.all(np.isfinite(v)):
        raise ValueError("Invalid tangent vector")
    if abs(mu @ v) > 1e-8:
        raise ValueError("Correction must be tangent to mu")
    angle = np.linalg.norm(v)
    return unit(np.cos(angle) * mu + np.sinc(angle / np.pi) * v)


def transport(mu, new_mu, vectors):
    """Parallel transport tangent columns along the shortest sphere arc."""
    mu, new_mu = unit(mu), unit(new_mu)
    denominator = 1 + mu @ new_mu
    if denominator < 1e-8:
        raise ValueError("Transport is ambiguous near antipodal points")
    vectors = np.asarray(vectors, dtype=float)
    return vectors - np.outer(mu + new_mu, new_mu @ vectors) / denominator


def covariance(value, size, positive=False):
    value = np.asarray(value, dtype=float)
    if value.shape != (size, size) or not np.all(np.isfinite(value)):
        raise ValueError("Invalid covariance shape or entries")
    if not np.allclose(value, value.T, atol=1e-10):
        raise ValueError("Covariance must be symmetric")
    value = (value + value.T) / 2
    minimum = np.linalg.eigvalsh(value).min()
    if minimum < -1e-10 or (positive and minimum <= 0):
        raise ValueError("Covariance must be positive (semi)definite")
    return value


def extrinsic_mean(observations):
    """Normalized arithmetic mean; not generally an intrinsic Frechet mean."""
    return unit(np.mean([unit(y) for y in observations], axis=0))


class VMFFilter:
    """Conjugate vMF measurement fusion with heuristic confidence decay."""

    def __init__(self, initial_mu, kappa=5.0):
        if not np.isfinite(kappa) or kappa < 0:
            raise ValueError("Concentration must be nonnegative")
        self.mu = unit(initial_mu)
        self.eta = float(kappa) * self.mu

    @property
    def kappa(self):
        return np.linalg.norm(self.eta)

    def predict(self, decay=0.98):
        if not 0 <= decay <= 1:
            raise ValueError("Decay must be between zero and one")
        self.eta *= decay

    def update(self, observation, kappa_obs):
        if not np.isfinite(kappa_obs) or kappa_obs < 0:
            raise ValueError("Observation concentration must be nonnegative")
        self.eta += kappa_obs * unit(observation)
        if self.kappa > 1e-12:
            self.mu = self.eta / self.kappa
        # At zero concentration the direction is arbitrary: keep the old one.
        return self.mu.copy(), self.kappa


class RiemannianSphereEKF:
    """Local error-state filter with identity appearance observation.

    E holds the current tangent basis. Supply R in this exact basis.
    The observation-noise approximation is local and independent of the prior.
    """

    def __init__(self, initial_mu, initial_var=0.04):
        if not np.isfinite(initial_var) or initial_var <= 0:
            raise ValueError("Initial variance must be positive")
        self.mu = unit(initial_mu)
        self.E = tangent_basis(self.mu)
        self.P = np.eye(self.mu.size - 1) * initial_var

    def predict(self, Q):
        """Keep the mean; add uncertainty in the current tangent basis."""
        self.P += covariance(Q, self.P.shape[0])

    def innovation(self, observation, R):
        R = covariance(R, self.P.shape[0], positive=True)
        z = self.E.T @ log_map(self.mu, observation)
        return z, self.P + R

    def association_score(self, observation, R):
        """Local Gaussian NLL, up to a common constant; lower is better.

        A tutorial score, not a complete MHT likelihood: clutter, priors,
        detection probability and large-angle density corrections are omitted.
        """
        z, S = self.innovation(observation, R)
        distance2 = float(z @ np.linalg.solve(S, z))
        logdet = np.linalg.slogdet(S)[1]
        return distance2, 0.5 * (distance2 + logdet)

    def update(self, observation, R):
        R = covariance(R, self.P.shape[0], positive=True)
        z, S = self.innovation(observation, R)
        K = np.linalg.solve(S, self.P.T).T
        correction = K @ z
        angle = np.linalg.norm(correction)
        if angle >= np.pi - 1e-6:
            raise ValueError("Correction is outside the local chart")
        identity = np.eye(self.P.shape[0])
        remainder = identity - K
        posterior = remainder @ self.P @ remainder.T + K @ R @ K.T
        new_mu = exp_map(self.mu, self.E @ correction)
        new_E = transport(self.mu, new_mu, self.E)
        # First-order reset for log_new(exp_old(correction + error)),
        # expressed in the parallel-transported basis. Along the correction
        # the derivative is 1; in perpendicular directions it is sinc(angle).
        reset = identity
        if angle > 1e-10:
            direction = correction / angle
            sinc = np.sinc(angle / np.pi)
            reset = sinc * identity + (1 - sinc) * np.outer(direction, direction)
        self.P = reset @ posterior @ reset.T
        self.P = (self.P + self.P.T) / 2
        self.mu, self.E = new_mu, new_E
        return self.mu.copy(), self.P.copy()


def calibrate_local_errors(reference, observations, rank=2, floor_std=0.03):
    """Learn a local error envelope around ONE reliable reference.

    Use same-vehicle samples from comparable conditions. Uncentered SVD
    captures expected squared prediction error, including unmodeled bias.
    This is an effective zero-mean uncertainty approximation, not a claim
    that the observed errors actually have zero mean. For centered PCA use
    V - V.mean(0); its variances are S_centered**2 / (N - 1).

    Different references/vehicles require explicit alignment and transfer
    validation before pooling. Training tracker residuals also contain P;
    assigning their full scatter to R would count that uncertainty twice.
    """
    mu = unit(reference)
    E = tangent_basis(mu)
    V = np.array([E.T @ log_map(mu, y) for y in observations])
    if len(V) < 2 or not 1 <= rank <= min(V.shape):
        raise ValueError("Need at least two samples and a supported rank")
    if not np.isfinite(floor_std) or floor_std <= 0:
        raise ValueError("Noise floor must be positive")
    _, singular_values, axes = np.linalg.svd(V, full_matrices=False)
    error_energy = singular_values[:rank] ** 2 / len(V)
    return {
        "mu": mu,
        "directions": E @ axes[:rank].T,
        "extra_variance": np.maximum(error_energy - floor_std**2, 0),
        "floor_std": float(floor_std),
        "mean_error": E @ V.mean(axis=0),
    }


def build_runtime_covariance_R(calibration, filter_mu, filter_basis):
    """Move a LOCAL learned envelope into the filter's current coordinates.

    Reusing an envelope far from its calibration neighborhood is an untested
    modeling assumption. Transport aligns coordinates; it does not establish
    that the same nuisance pattern applies at another viewpoint or identity.
    """
    moved = transport(calibration["mu"], filter_mu, calibration["directions"])
    U = filter_basis.T @ moved
    R = calibration["floor_std"] ** 2 * np.eye(U.shape[0])
    R += (U * calibration["extra_variance"]) @ U.T
    return covariance(R, U.shape[0], positive=True)


def pose_uncertainty_covariance(J, Sigma_pose, floor_std=0.03):
    """Optional J Sigma J.T propagation for uncertain pose, not orbit drift.

    J must contain actual local derivatives, with radians used consistently,
    in the SAME tangent basis as the filter. Do not normalize its columns.
    Pose covariance may have correlated entries. If pose is already part of
    a joint state/measurement model, avoid adding its uncertainty twice.
    """
    J = np.asarray(J, dtype=float)
    if J.ndim != 2 or not np.all(np.isfinite(J)):
        raise ValueError("Invalid sensitivity matrix")
    if not np.isfinite(floor_std) or floor_std <= 0:
        raise ValueError("Noise floor must be positive")
    Sigma_pose = covariance(Sigma_pose, J.shape[1])
    return J @ Sigma_pose @ J.T + floor_std**2 * np.eye(J.shape[0])


def local_pose_sensitivities(reference, perturbation_pairs, steps_rad):
    """One local sensitivity column per controlled physical input.

    perturbation_pairs contains (minus_embedding, plus_embedding) for each
    input, e.g. azimuth, elevation, yaw. Other rendering conditions should
    be held fixed. steps_rad contains the POSITIVE half-step for each pair.
    Returns (reference_mu, tangent_basis, J). Columns retain their scale;
    no independence or orthogonality of embedding effects is assumed.
    """
    mu = unit(reference)
    E = tangent_basis(mu)
    steps = np.asarray(steps_rad, dtype=float)
    pairs = list(perturbation_pairs)
    if steps.shape != (len(pairs),) or not len(pairs):
        raise ValueError("Supply one positive step per perturbation pair")
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        raise ValueError("Perturbation steps must be positive radians")
    columns = [
        E.T @ (log_map(mu, plus) - log_map(mu, minus)) / (2 * step)
        for (minus, plus), step in zip(pairs, steps)
    ]
    return mu, E, np.column_stack(columns)
