import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cleat'))

from math_core import advantages, guidance, probabilities, sample_arms, value_rule


class MathCoreTest(unittest.TestCase):
    def test_distributions_are_normalised(self):
        bar, mu, rho = probabilities([0.2, 0.3, 0.4, 0.1], [0.1, -0.2, 0.3], 1.0, 0.2)
        self.assertAlmostEqual(bar.sum(), 1.0)
        self.assertAlmostEqual(mu.sum(), 1.0)
        self.assertEqual(len(rho), 4)
        self.assertEqual(rho[-1], 1.0)
        self.assertTrue(np.all((rho[:-1] > 0) & (rho[:-1] <= 1)))

    def test_inclusion_probabilities_match_simulation(self):
        pi, values, beta, eps = [0.5, 0.2, 0.2, 0.1], [0.4, 0.0, -0.4], 2.0, 0.2
        _, _, rho = probabilities(pi, values, beta, eps)
        rng = np.random.default_rng(0)
        hits = np.zeros(3)
        n = 20000
        for _ in range(n):
            arms, _, _, _ = sample_arms(pi, values, beta, eps, rng)
            for arm in arms:
                if arm < 3:
                    hits[arm] += 1
        np.testing.assert_allclose(hits / n, rho[:-1], atol=0.015)

    def test_group_layout(self):
        arms, _, _, _ = sample_arms([0.3, 0.3, 0.3, 0.1], [0.0, 0.1, 0.2], 1.0, 0.2, np.random.default_rng(1))
        self.assertEqual(arms[3], 2)
        self.assertEqual(sorted(v for k, v in arms.items() if k != 3), [1, 2])

    def test_empty_menu_is_pass_only(self):
        arms, _, bar, _ = sample_arms([1.0], [], 1.0, 0.2, np.random.default_rng(0))
        self.assertEqual(arms, {0: 2})
        self.assertEqual(len(bar), 0)

    def test_pass_referenced_advantages(self):
        adv = advantages({0: [1.0, 0.0], 2: [0.5, 1.5]}, 2)
        np.testing.assert_allclose(adv[0], [0.0, -1.0])
        np.testing.assert_allclose(adv[2], [-1.0, 1.0])

    def test_guidance_is_centred_and_clipped(self):
        g = guidance([0.4, 0.6, 1.0], 18.0)
        self.assertAlmostEqual(float(np.mean(np.clip([-4.8, -1.2, 6.0], -6, 6))), float(g.mean()), places=6)
        self.assertLessEqual(np.abs(g).max(), 6.0)

    def test_value_rule(self):
        self.assertEqual(value_rule([-0.2, 0.4]), 1)
        self.assertEqual(value_rule([-0.2, -0.1]), 2)
        self.assertEqual(value_rule([]), 0)


if __name__ == '__main__':
    unittest.main()
