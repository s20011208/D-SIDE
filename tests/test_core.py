"""Independent numerical mathematics checks; unittest, no pytest dependency."""
import math
import unittest
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm, qr
from dside._backend.core import (Trial, SmoothTrial, GreenModel, Nystrom, companion,
                       green_matrices, group_penalty, profile_trial,
                       smooth_trial, solve_inner)


class GreenKernelTests(unittest.TestCase):
    def test_companion_no_zeroth_term_and_coefficient_order(self):
        np.testing.assert_array_equal(companion([], 1), [[0.]])
        np.testing.assert_array_equal(companion([.7], 2), [[0., 1.], [0., -.7]])
        np.testing.assert_array_equal(companion([2., 3.], 3),
                                      [[0., 1., 0.], [0., 0., 1.], [0., -2., -3.]])
        with self.assertRaises(ValueError):
            companion([1., 2.], 2)

    def test_first_order_trapezoid_both_endpoints(self):
        t = np.linspace(2., 3., 101)
        h, w = green_matrices(t, [], 1)
        np.testing.assert_array_equal(h, np.ones((101, 1)))
        np.testing.assert_allclose(w@np.ones(101), t-t[0], atol=2e-15)
        np.testing.assert_allclose(w@(t-t[0]), .5*(t-t[0])**2, atol=2e-15)
        self.assertEqual(w[0, 0], 0.)
        self.assertAlmostEqual(w[-1, -1], .005)
        self.assertAlmostEqual(w[-1, 0], .005)

    def test_second_order_damping_analytic_kernel_and_small_damping(self):
        t = np.linspace(0., 1.7, 171)
        for b in (.7, 1e-12, 0.):
            h, w = green_matrices(t, [b], 2)
            expected = -np.expm1(-b*t)/b if b else t
            np.testing.assert_allclose(h[:, 0], 1., atol=2e-14)
            np.testing.assert_allclose(h[:, 1], expected, atol=4e-14)
            np.testing.assert_allclose(w[:, 0], .005*expected, atol=2e-15)
            self.assertTrue(np.all(np.triu(w, 1) == 0.))

    def test_third_order_analytic_impulse(self):
        t = np.linspace(0., 1.7, 171)
        h, _ = green_matrices(t, [2., 3.], 3)
        # Laplace transform 1/[s(s+1)(s+2)].
        expected = .5*(1.-np.exp(-t))**2
        np.testing.assert_allclose(h[:, -1], expected, atol=3e-14)
        for j in (0, 30, 170):
            np.testing.assert_allclose(h[j], expm(companion([2., 3.], 3)*t[j])[0], atol=3e-14)

    def test_zero_operator_homogeneous_basis_and_cubic_quadrature_error(self):
        t = np.linspace(0., 1.5, 151)
        for k in (1, 2, 3):
            h, w = green_matrices(t, np.zeros(k-1), k)
            for j in range(k):
                np.testing.assert_allclose(h[:, j], t**j/math.factorial(j), atol=3e-14)
            error = w@np.ones(len(t))-t**k/math.factorial(k)
            expected = t*(t[1]-t[0])**2/12. if k == 3 else 0.
            np.testing.assert_allclose(error, expected, atol=3e-14)

    def test_grid_validation(self):
        for t in ([1., .5, 0.], [0., 0., 0.], [0., .1, np.nan], [0.], [0., .2, .5]):
            with self.assertRaises(ValueError):
                green_matrices(t, [.7], 2)

    def test_k1_k2_k3_forced_solution_quadrature_convergence(self):
        force = lambda t: np.sin(1.3*t)+.3*np.cos(.7*t)
        for order, omega, initial in ((1, [], [.4]), (2, [.7], [.4, -.2]),
                                      (3, [2., 2.], [.4, -.2, .1])):
            a = companion(omega, order)
            def rhs(t, state):
                out = a@state
                out[-1] += force(t)
                return out
            errors = []
            for n in (101, 201, 401):
                t = np.linspace(0., 1.7, n)
                exact = solve_ivp(rhs, (0., t[-1]), initial, t_eval=t, method='DOP853',
                                  rtol=2e-12, atol=2e-14).y[0]
                h, w = green_matrices(t, omega, order)
                errors.append(np.max(np.abs(h@initial+w@force(t)-exact)))
            self.assertLess(errors[-1], 8e-6)
            self.assertGreater(errors[0]/errors[1], 3.7)
            self.assertGreater(errors[1]/errors[2], 3.7)

    def test_weighted_initial_profile_matches_joint_least_squares(self):
        t = np.linspace(0., 2., 81)
        h, w = green_matrices(t, [.7], 2)
        d = w@np.column_stack([np.sin(t), np.cos(1.7*t)])
        c = np.array([.8, -.4]); theta = np.array([.3, -.2])
        y = h@c+d@theta
        weight = np.linspace(.5, 1.5, len(t)); root = np.sqrt(weight)
        q, _ = qr(root[:, None]*h, mode='economic')
        d_tilde = root[:, None]*d-q@(q.T@(root[:, None]*d))
        y_tilde = root*y-q@(q.T@(root*y))
        theta_hat = np.linalg.lstsq(d_tilde, y_tilde, rcond=None)[0]
        c_hat = np.linalg.lstsq(root[:, None]*h, root*(y-d@theta_hat), rcond=None)[0]
        joint = np.linalg.lstsq(root[:, None]*np.column_stack([h, d]), root*y, rcond=None)[0]
        np.testing.assert_allclose(theta_hat, theta, atol=1e-12)
        np.testing.assert_allclose(c_hat, c, atol=1e-12)
        np.testing.assert_allclose(joint, np.r_[c_hat, theta_hat], atol=1e-12)


class MatchingIdentityTests(unittest.TestCase):
    @staticmethod
    def exact_trial(n):
        t = np.linspace(0., 2., n)
        # Analytic x=exp(-.4t)+.3sin(.8t), derivatives known independently.
        derivatives = np.column_stack([(-.4)**j*np.exp(-.4*t)+
                         .3*.8**j*np.sin(.8*t+j*np.pi/2.) for j in range(4)])
        return SmoothTrial(0, 'analytic', t, derivatives[:, 0], derivatives)

    def test_gradient_matching_signs(self):
        for order, omega in ((1, []), (2, [.7]), (3, [2., 2.])):
            tr = self.exact_trial(101)
            force = tr.derivatives[:, order].copy()
            for j, a in enumerate(omega, 1):
                force += a*tr.derivatives[:, j]
            design, target, _ = profile_trial(tr, force[:, None], omega, order, 'gradient')
            np.testing.assert_allclose(design[:, 0], target, atol=1e-14)

    def test_integral_identity_is_derivative_free_and_converges(self):
        for order, omega in ((2, [.7]), (3, [2., 2.])):
            errors = []
            for n in (101, 201, 401):
                tr = self.exact_trial(n)
                force = tr.derivatives[:, order].copy()
                for j, a in enumerate(omega, 1): force += a*tr.derivatives[:, j]
                # Removing all derivative arrays must not change integral identity.
                tr.derivatives = None
                design, target, _ = profile_trial(tr, force[:, None], omega, order, 'integral')
                errors.append(np.linalg.norm(design[:, 0]-target))
            self.assertLess(errors[-1], 8e-6)
            self.assertGreater(errors[0]/errors[1], 3.7)
            self.assertGreater(errors[1]/errors[2], 3.7)

    def test_weak_signs_derivative_free_and_grid_refinement(self):
        for order, omega in ((2, [.7]), (3, [2., 2.])):
            errors = []
            for n in (401, 801, 1601):
                tr = self.exact_trial(n)
                force = tr.derivatives[:, order].copy()
                for j, a in enumerate(omega, 1): force += a*tr.derivatives[:, j]
                tr.derivatives = None
                design, target, _ = profile_trial(tr, force[:, None], omega, order, 'weak')
                self.assertEqual(design.shape, (12, 1))
                errors.append(np.linalg.norm(design[:, 0]-target))
            self.assertLess(errors[-1], 2e-4)
            self.assertGreater(errors[0]/errors[1], 3.7)
            self.assertGreater(errors[1]/errors[2], 3.7)

    def test_green_profile_does_not_read_derivative_arrays(self):
        tr = self.exact_trial(101)
        phi = np.column_stack([tr.x, tr.x**2])
        before = profile_trial(tr, phi, [.7], 2, 'green')
        tr.derivatives = None
        after = profile_trial(tr, phi, [.7], 2, 'green')
        np.testing.assert_array_equal(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])


class QuadraticAndSmoothingTests(unittest.TestCase):
    def test_group_penalty_spd_bound_and_weighted_variance_identity(self):
        pi = np.array([.1, .2, .3, .4]); l0, lu = .003, .02
        p = group_penalty(4, l0, lu, pi)
        self.assertGreaterEqual(np.linalg.eigvalsh(p)[0]+1e-15, min(l0, lu)*pi.min())
        b = np.random.default_rng(11).normal(size=(4, 5)); mean = pi@b
        expression = l0*np.sum(mean**2)+lu*np.sum(pi[:, None]*(b-mean)**2)
        self.assertAlmostEqual(float(np.trace(b.T@p@b)), float(expression), places=13)

    def test_inner_solve_matches_explicit_shared_offset_parameterization(self):
        rng = np.random.default_rng(2); rank = 4
        blocks = [(g, rng.normal(size=(12, rank)), rng.normal(size=12)) for g in (0, 0, 1)]
        l0, lu = .003, .02
        b, objective, info = solve_inner(blocks, 2, rank, l0, lu)
        designs, ys = [], []
        for g, d, y in blocks:
            scale = np.sqrt(1./(2*(2 if g == 0 else 1)))
            designs.append(scale*np.column_stack([d, d if g == 0 else -d])); ys.append(scale*y)
        z = np.vstack(designs); target = np.concatenate(ys)
        p = np.diag(np.r_[np.full(rank, l0), np.full(rank, lu)])
        solution = np.linalg.solve(z.T@z+p, z.T@target)
        np.testing.assert_allclose(b.mean(axis=0), solution[:rank], atol=1e-12)
        np.testing.assert_allclose((b[0]-b[1])/2, solution[rank:], atol=1e-12)
        self.assertGreater(info['hessian_min_eigenvalue'], 0.)
        self.assertLess(info['centering_residual'], 1e-14)
        self.assertLess(info['normal_equation_residual'], 1e-12)
        self.assertAlmostEqual(objective, float(np.sum((z@solution-target)**2)+solution@p@solution), places=12)

    def test_inner_rejects_missing_groups_or_nonpositive_penalty(self):
        block = [(0, np.ones((5, 2)), np.ones(5))]
        with self.assertRaises(ValueError): solve_inner(block, 2, 2, .01, .02)
        with self.assertRaises(ValueError): solve_inner(block, 1, 2, 0., .02)

    def test_nystrom_induced_kernel_psd_and_training_only_scaling(self):
        x = np.linspace(-1., 1., 40)
        feature = Nystrom(x, rank=12)
        phi = feature(x)
        self.assertEqual(phi.shape, (40, 12))
        self.assertGreater(np.linalg.eigvalsh(phi@phi.T)[0], -1e-12)
        before = (feature.center, feature.scale, feature.z.copy())
        feature([1e7, -1e7])
        self.assertEqual(feature.center, before[0]); self.assertEqual(feature.scale, before[1])
        np.testing.assert_array_equal(feature.z, before[2])

    def test_effective_rank_uses_raw_observation_count_not_quadrature_count(self):
        t = np.linspace(0., 1., 8)
        trial = Trial(0, 'rank-policy', t, np.cos(t), np.array([1., 0.]))
        ranks = []
        for grid in (31, 161):
            model = GreenModel(order=2, rank=100, known_omega=[[.7]]).fit(
                [smooth_trial(trial, grid=grid, noise_sd=0.)])
            ranks.append(model.features.rank)
        self.assertEqual(ranks, [math.ceil(2*np.sqrt(8))]*2)

    def test_raw_sample_growth_can_increase_the_sieve_rank(self):
        ranks = []
        for n in (5, 20, 80):
            t = np.linspace(0., 1., n)
            trial = Trial(0, 'rank-policy', t, np.cos(t), np.array([1., 0.]))
            model = GreenModel(order=2, rank=100, known_omega=[[.7]]).fit(
                [smooth_trial(trial, grid=161, noise_sd=0.)])
            ranks.append(model.features.rank)
        self.assertEqual(ranks, [math.ceil(2*np.sqrt(n)) for n in (5, 20, 80)])

    def test_smoothing_reads_only_supplied_training_observations(self):
        t = np.linspace(0., 2., 81); x = np.cos(t)
        trial = Trial(0, 'train', t[:41].copy(), x[:41].copy(), np.array([1., 0.]), t.copy(), x.copy())
        before = smooth_trial(trial, grid=31, noise_sd=.02)
        trial.initial[:] = 1e8; trial.truth_x[:] = -1e8
        after = smooth_trial(trial, grid=31, noise_sd=.02)
        np.testing.assert_array_equal(before.x, after.x)
        np.testing.assert_array_equal(before.derivatives, after.derivatives)
        self.assertEqual(after.t[-1], 1.)

    def test_smoothing_rejects_nonfinite_time(self):
        trial = Trial(0, 'bad', np.array([0., 1., 2., 3., np.inf]), np.ones(5), np.zeros(2))
        with self.assertRaises(ValueError): smooth_trial(trial)

    def test_known_operator_fit_profiles_actual_initial_state(self):
        omega = [.7]; t = np.linspace(0., 2., 101); initial = np.array([.6, -.2])
        x = initial[0]-np.expm1(-omega[0]*t)/omega[0]*initial[1]
        # Homogeneous path: true nonparametric RHS is zero.
        trial = Trial(0, 'homogeneous', t, x, np.full(2, np.nan))
        model = GreenModel(order=2, rank=8, known_omega=np.array([omega])).fit(
            [smooth_trial(trial, grid=61, noise_sd=0.)])
        np.testing.assert_allclose(model.initial_states[trial.name], initial, atol=1e-7)
        self.assertGreater(model.inner_diagnostics['hessian_min_eigenvalue'], 0.)
        prediction = model.predict(0, np.linspace(3., 4., 41), initial)
        self.assertAlmostEqual(prediction[0], initial[0], places=14)
        with self.assertRaises(ValueError): model.predict(0, t, [initial[0]])


if __name__ == '__main__':
    unittest.main(verbosity=2)
