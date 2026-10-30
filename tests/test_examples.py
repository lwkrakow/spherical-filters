"""Run with: OPENBLAS_NUM_THREADS=1 python3 -m unittest discover -s tests -v."""

import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from research_examples import (
    RiemannianSphereEKF, VMFFilter, build_runtime_covariance_R,
    calibrate_local_errors, exp_map, log_map, pose_uncertainty_covariance,
    tangent_basis, transport, unit, local_pose_sensitivities,
)

ROOT = Path(__file__).resolve().parents[1]


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(42)

    def test_bases_log_exp_and_transport_in_multiple_dimensions(self):
        for d in [3, 8, 128]:
            mu = unit(self.rng.normal(size=d))
            E = tangent_basis(mu)
            np.testing.assert_allclose(E.T @ E, np.eye(d - 1), atol=1e-12)
            np.testing.assert_allclose(mu @ E, 0, atol=1e-12)
            z = self.rng.normal(size=d - 1)
            z *= 0.2 / np.linalg.norm(z)
            y = exp_map(mu, E @ z)
            np.testing.assert_allclose(log_map(mu, y), E @ z, atol=1e-12)
            moved = transport(mu, y, E)
            np.testing.assert_allclose(moved.T @ moved, np.eye(d - 1), atol=1e-12)
            np.testing.assert_allclose(y @ moved, 0, atol=1e-12)

    def test_update_and_score_do_not_depend_on_tangent_basis(self):
        mu = unit(self.rng.normal(size=8))
        first, rotated = RiemannianSphereEKF(mu), RiemannianSphereEKF(mu)
        A = self.rng.normal(size=(7, 7))
        P = A @ A.T * 0.003 + np.eye(7) * 0.01
        B = self.rng.normal(size=(7, 7))
        R = B @ B.T * 0.002 + np.eye(7) * 0.02
        O, _ = np.linalg.qr(self.rng.normal(size=(7, 7)))
        first.P = P
        rotated.E = first.E @ O
        rotated.P = O.T @ P @ O
        y = exp_map(mu, first.E @ np.linspace(-0.08, 0.08, 7))
        np.testing.assert_allclose(first.association_score(y, R), rotated.association_score(y, O.T @ R @ O), atol=1e-10)
        first.update(y, R)
        rotated.update(y, O.T @ R @ O)
        np.testing.assert_allclose(first.mu, rotated.mu, atol=1e-12)
        np.testing.assert_allclose(first.E @ first.P @ first.E.T, rotated.E @ rotated.P @ rotated.E.T, atol=1e-12)

    def test_covariance_reset_matches_chart_derivative(self):
        mu = np.array([0., 0., 1.])
        E = tangent_basis(mu)
        z = np.array([0.25, -0.1])
        y = exp_map(mu, E @ z)
        nextE = transport(mu, y, E)
        epsilon = 1e-6
        numerical = np.column_stack([
            nextE.T @ (log_map(y, exp_map(mu, E @ (z + epsilon * axis)))
                       - log_map(y, exp_map(mu, E @ (z - epsilon * axis)))) / (2 * epsilon)
            for axis in np.eye(2)
        ])
        r = np.linalg.norm(z)
        a = z / r
        sinc = np.sinc(r / np.pi)
        expected = sinc * np.eye(2) + (1 - sinc) * np.outer(a, a)
        np.testing.assert_allclose(numerical, expected, atol=1e-9)

    def test_higher_noise_reduces_penalty_and_update_influence(self):
        mu = np.array([0., 0., 1.])
        a, b = RiemannianSphereEKF(mu), RiemannianSphereEKF(mu)
        y = exp_map(mu, a.E @ np.array([0.2, 0.1]))
        low, high = np.diag([0.01, 0.01]), np.diag([0.3, 0.01])
        self.assertLess(b.association_score(y, high)[0], a.association_score(y, low)[0])
        a.update(y, low)
        b.update(y, high)
        correction_a, correction_b = a.mu.copy(), b.mu.copy()
        self.assertLess(abs(log_map(mu, correction_b) @ tangent_basis(mu)[:, 0]),
                        abs(log_map(mu, correction_a) @ tangent_basis(mu)[:, 0]))

    def test_repeated_updates_keep_valid_geometry_and_uncertainty(self):
        tracker = RiemannianSphereEKF([0, 0, 1])
        for _ in range(30):
            tracker.predict(np.eye(2) * 0.001)
            y = exp_map(tracker.mu, tracker.E @ self.rng.normal(0, 0.08, 2))
            tracker.update(y, np.diag([0.04, 0.01]))
            self.assertAlmostEqual(np.linalg.norm(tracker.mu), 1)
            np.testing.assert_allclose(tracker.E.T @ tracker.E, np.eye(2), atol=1e-12)
            np.testing.assert_allclose(tracker.mu @ tracker.E, 0, atol=1e-12)
            self.assertGreater(np.linalg.eigvalsh(tracker.P).min(), 0)

    def test_calibration_preserves_error_scale_and_transforms_coordinates(self):
        mu = np.array([0., 0., 1.])
        E = tangent_basis(mu)
        errors = np.array([[0.20, 0.01], [-0.20, 0.01], [0.22, -0.01], [-0.22, -0.01]])
        samples = [exp_map(mu, E @ z) for z in errors]
        model = calibrate_local_errors(mu, samples, rank=2, floor_std=0.001)
        R = build_runtime_covariance_R(model, mu, E)
        np.testing.assert_allclose(R, errors.T @ errors / len(errors), atol=1e-12)
        O = np.array([[0., -1.], [1., 0.]])
        rotated = build_runtime_covariance_R(model, mu, E @ O)
        np.testing.assert_allclose(rotated, O.T @ R @ O, atol=1e-12)
        moved_mu = exp_map(mu, E @ np.array([0.1, 0.05]))
        moved_E = transport(mu, moved_mu, E)
        moved_R = build_runtime_covariance_R(model, moved_mu, moved_E)
        np.testing.assert_allclose(moved_R, R, atol=1e-12)

    def test_physical_sensitivity_magnitude_and_pose_correlations_are_kept(self):
        J = np.array([[2., 0.], [0., 0.5]])
        Sigma = np.array([[0.04, 0.01], [0.01, 0.02]])
        R = pose_uncertainty_covariance(J, Sigma)
        np.testing.assert_allclose(R, J @ Sigma @ J.T + 0.03**2 * np.eye(2))
        self.assertGreater(R[0, 1], 0)

    def test_controlled_perturbations_recover_named_overlapping_directions(self):
        mu = np.array([0., 0., 0., 1.])
        E = tangent_basis(mu)
        J = np.array([[1., 0.4, 0.2], [0.3, 0.8, 0.1], [0., 0.1, 0.9]])
        steps = np.deg2rad([2., 2., 2.])
        pairs = [(exp_map(mu, -E @ column * step), exp_map(mu, E @ column * step))
                 for column, step in zip(J.T, steps)]
        ref, basis, estimated = local_pose_sensitivities(mu, pairs, steps)
        np.testing.assert_allclose(ref, mu)
        np.testing.assert_allclose(basis, E)
        np.testing.assert_allclose(estimated, J, atol=1e-12)
        R = pose_uncertainty_covariance(estimated, np.diag([0.01, 0.02, 0.03]))
        self.assertGreater(abs(R[0, 1]), 0)

    def test_ambiguous_and_invalid_inputs_fail_explicitly(self):
        with self.assertRaises(ValueError):
            unit([0, 0, 0])
        with self.assertRaises(ValueError):
            log_map([0, 0, 1], [0, 0, -1])
        with self.assertRaises(ValueError):
            RiemannianSphereEKF([0, 0, 1]).update([0, 0, 1], -np.eye(2))
        vmf = VMFFilter([0, 0, 1], kappa=2)
        vmf.update([0, 0, -1], kappa_obs=2)
        self.assertEqual(vmf.kappa, 0)
        self.assertTrue(np.all(np.isfinite(vmf.mu)))


class PageParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.ids, self.refs, self.code = [], [], []
        self.in_code = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            self.ids.append(attrs['id'])
        self.refs.extend(attrs[k] for k in ['href', 'src'] if k in attrs)
        if tag == 'code':
            self.in_code = True
            self.code.append('')

    def handle_endtag(self, tag):
        if tag == 'code':
            self.in_code = False

    def handle_data(self, data):
        if self.in_code:
            self.code[-1] += data


class PageTests(unittest.TestCase):
    def test_local_links_anchors_and_embedded_source(self):
        pages = {}
        for file in ROOT.glob('*.html'):
            parser = PageParser()
            parser.feed(file.read_text())
            self.assertEqual(len(parser.ids), len(set(parser.ids)), file.name)
            pages[file.name] = parser
        for name, parser in pages.items():
            for ref in parser.refs:
                url = urlsplit(ref)
                if url.scheme or url.netloc:
                    continue
                target = url.path or name
                self.assertTrue((ROOT / target).is_file(), (name, ref))
                if url.fragment and target in pages:
                    self.assertIn(url.fragment, pages[target].ids)
        source = (ROOT / 'research_examples.py').read_text()
        split = source.index('\ndef calibrate_local_errors')
        self.assertIn(source[:split], pages['fundamentals.html'].code)
        self.assertIn(source[split + 1:], pages['calibration.html'].code)


if __name__ == '__main__':
    unittest.main()
