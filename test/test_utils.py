import unittest

from spherogram import Link

from utils import (is_sorted, split_testing_workload,
                   encode_link, link_from_crossings,
                   decode_link)
from dataset import Dataset


class UtilsTestCases(unittest.TestCase):
    def test_is_sorted(self) -> None:
        self.assertTrue(is_sorted([]))
        self.assertTrue(is_sorted([1]))
        self.assertTrue(is_sorted([1, 1]))
        self.assertTrue(is_sorted([1, 2]))
        self.assertFalse(is_sorted([1, 0]))
        self.assertTrue(is_sorted([1, 1, 1]))
        self.assertFalse(is_sorted([1, 1, 0]))
        self.assertFalse(is_sorted([1, 0, 0]))
        self.assertTrue(is_sorted([0, 1, 1]))
        self.assertTrue(is_sorted([0, 0, 1]))
        self.assertTrue(is_sorted([0, 1, 2]))

    def test_split_testing_workload(self) -> None:
        with self.assertRaises(ValueError) as _:
            split_testing_workload([Link('3_1')], 100, 0)
        with self.assertRaises(ValueError) as _:
            split_testing_workload([], 100, 2)
        # check that each worker process handles at least
        # one link
        output = split_testing_workload([Link('3_1'), Link('4_1')], 100, 3)
        self.assertEqual(len(output), 2)
        self.assertEqual(len(output[0][0]), 1)
        self.assertEqual(len(output[1][0]), 1)
        # check that we are correctly splitting the links
        # according to the quotient and remainder of the
        # number of links divided by the number of worker
        # processes
        dataset = Dataset.read_csv('../datasets/rolfsen.csv')[:13]
        output = split_testing_workload(dataset.links, 1000, 4)
        self.assertEqual([len(t[0]) for t in output],
                         [4, 3, 3, 3])

    def test_encode_link(self) -> None:
        trefoil = Link('3_1')
        self.assertEqual(encode_link(trefoil), """{"PD_code": [[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]], "signs": [-1, -1, -1], "unlinked_unknot_components": 0, "name": "3_1"}""")

    def test_link_from_crossings(self) -> None:
        with self.assertRaises(ValueError) as _:
            link_from_crossings(Link('3_1').PD_code(), [])
        trefoil = link_from_crossings([(5, 2, 0, 3),
                                       (3, 0, 4, 1),
                                       (1, 4, 2, 5)],
                                      [-1, -1, -1], 2)
        self.assertEqual(trefoil.PD_code(), [(1, 4, 2, 5),
                                             (5, 2, 0, 3),
                                             (3, 0, 4, 1)])
        self.assertEqual([c.sign for c in trefoil.crossings],
                         [-1, -1, -1])
        self.assertEqual(trefoil.unlinked_unknot_components, 2)

    def test_decode_link(self) -> None:
        trefoil = decode_link("""{"PD_code": [[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]], "signs": [-1, -1, -1], "unlinked_unknot_components": 0, "name": "3_1"}""")
        self.assertEqual(trefoil.PD_code(), [(1, 4, 2, 5),
                                             (5, 2, 0, 3),
                                             (3, 0, 4, 1)])
        self.assertEqual([c.sign for c in trefoil.crossings],
                         [-1, -1, -1])
        self.assertEqual(trefoil.unlinked_unknot_components, 0)
        self.assertEqual(trefoil.name, '3_1')


if __name__ == '__main__':
    unittest.main()
