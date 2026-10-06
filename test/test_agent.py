import unittest

from agent_py import (RandomWalkParam,
                      band_operations_random_walk_parallel)
from dataset import Dataset


class AgentTestCases(unittest.TestCase):
    def test_random_walk_param_init(self) -> None:
        with self.assertRaises(ValueError) as _:
            RandomWalkParam(1.0, 1.0, 5, 25, 'other')

    def test_band_operations_random_walk_parallel(self) -> None:
        params = RandomWalkParam.default('unknotting')
        dataset = Dataset.read_csv('../datasets/rolfsen.csv')[:35]
        self.assertAlmostEqual(dataset.evaluate_invariant('unknotting', band_operations_random_walk_parallel('unknotting', params, dataset.links, 20000, 12)), 1.0)

        params = RandomWalkParam.default('slice')
        dataset = Dataset.read_csv('../datasets/rolfsen.csv')[:15]
        self.assertAlmostEqual(dataset.evaluate_invariant('slice', band_operations_random_walk_parallel('slice', params, dataset.links, 20000, 12)), 1.0)

    def test_random_walk_with_births(self) -> None:
        # the slice framework accepts p_birth and stays sound
        params = RandomWalkParam.default('slice')
        params.p_birth = 0.3
        dataset = Dataset.read_csv('../datasets/rolfsen.csv')[:1]
        dataset.slice_genera = [[1]]   # trefoil
        self.assertAlmostEqual(dataset.evaluate_invariant('slice', band_operations_random_walk_parallel('slice', params, dataset.links, 5000, 4)), 1.0)

if __name__ == '__main__':
    unittest.main()
