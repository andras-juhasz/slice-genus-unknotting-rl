import json
import os
import tempfile
import unittest
import warnings
from spherogram import Link
from pandas import DataFrame

from dataset import (name_to_link, convert_float_invariant,
                     parse_invariant, convert_invariant,
                     invariant_to_str, evaluate_invariant, Dataset)


class DatasetTestCases(unittest.TestCase):
    def test_name_to_link(self) -> None:
        # check that the name is stripped of whitespaces
        link = name_to_link('   3_1  ')
        self.assertEqual(link.name, '3_1')
        # check that empty names result in a ValueError
        with self.assertRaises(ValueError) as _:
            name_to_link('     ')
        # check that name 13n_3370 is converted to 13n3370
        link = name_to_link('13n_3370')
        self.assertEqual(link.name, '13n3370')
        # check that name 0_1 results in an unknot
        link = name_to_link('0_1')
        self.assertEqual(len(link.crossings), 0)
        self.assertEqual(link.unlinked_unknot_components, 1)

    def test_convert_float_invariant(self) -> None:
        self.assertEqual(convert_float_invariant("[2;3]"), "[2;3]")
        self.assertEqual(convert_float_invariant(1.0), "1")
        self.assertEqual(convert_float_invariant(1), "1")

    def test_parse_invariant(self) -> None:
        # check that rep is stripped of whitespaces and that
        # ";" is replaced by ","
        self.assertEqual(parse_invariant(' [1;2] '), [1, 2])
        # check that the empty string leads to the return
        # value None
        self.assertIsNone(parse_invariant(''))
        # check that a single integer is supported
        for i in range(10):
            self.assertEqual(parse_invariant(str(i)), [i])
        # check that an illegal range leads to a ValueError
        with self.assertRaises(ValueError) as _:
            parse_invariant('[2,1]')
        # check that a range of integers is supported
        self.assertEqual(parse_invariant('[2,4]'), [2, 3, 4])
        # check that a set of integers is supported and
        # automatically sorted
        self.assertEqual(parse_invariant('{3,1}'), [1, 3])
        # check that illegal input leads to a ValueError
        with self.assertRaises(ValueError) as _:
            parse_invariant('<1,2>')

    def test_convert_invariant(self) -> None:
        self.assertEqual(convert_invariant(2), [2])
        self.assertEqual(convert_invariant((2, 3)), [2, 3])
        with self.assertRaises(ValueError) as _:
            convert_invariant((3, 2))
        self.assertEqual(convert_invariant([2, 1, 3]), [1, 2, 3])
        self.assertIsNone(convert_invariant(None))

    def test_invariant_to_str(self) -> None:
        self.assertEqual(invariant_to_str(None), '')
        self.assertEqual(invariant_to_str([1]), '1')
        self.assertEqual(invariant_to_str([1, 2]), '[1;2]')
        self.assertEqual(invariant_to_str([1, 3]), '{1;3}')

    def test_init(self) -> None:
        # field order: links, unknotting, slice, strong slice,
        # ribbon, strong ribbon, splitting, weak splitting, names
        nones = [[None]] * 4
        with self.assertRaises(AssertionError) as _:
            Dataset([Link('3_1')], [[1]], [[1]], [None],
                    *nones, ['3_1', '4_1'])
        with self.assertRaises(AssertionError) as _:
            Dataset([Link('3_1')], [[]], [[1]], [None], *nones, ['3_1'])
        with self.assertRaises(AssertionError) as _:
            Dataset([Link('3_1')], [[1]], [[]], [None], *nones, ['3_1'])
        with self.assertRaises(AssertionError) as _:
            Dataset([Link('3_1')], [[1, 0]], [[1]], [None], *nones, ['3_1'])
        with self.assertRaises(AssertionError) as _:
            Dataset([Link('3_1')], [[1]], [[1, 0]], [None], *nones, ['3_1'])
        dataset = Dataset([Link('3_1')], [[1]], [[1]], [None],
                          *nones, ['3_1'])
        self.assertEqual(dataset.links[0].name, '3_1')
        self.assertEqual(dataset.unknotting_nums[0], [1])
        self.assertEqual(dataset.slice_genera[0], [1])
        self.assertIsNone(dataset.ribbon_genera[0])
        self.assertIsNone(dataset.strong_ribbon_genera[0])
        self.assertEqual(dataset.names[0], '3_1')

    def test_copy(self) -> None:
        dataset = Dataset.from_link('3_1', (1, 2))
        dataset_copy = dataset.copy()
        dataset_copy.unknotting_nums[0] = [1]
        self.assertEqual(dataset.unknotting_nums[0], [1, 2])

    def test_from_dataframe(self) -> None:
        # check that deep copies of the DataFrame are made
        df = DataFrame({'Name': ['3_1']})
        _ = Dataset.from_dataframe(df)
        self.assertEqual(df.columns[0], 'Name')
        # test that we cannot have multiple columns of names
        df = DataFrame({'Name': ['3_1'], 'Names': ['3_1']})
        with self.assertRaises(ValueError) as _:
            Dataset.from_dataframe(df)
        # test that a ValueError is raised when no links are
        # specified
        df = DataFrame({'Unknotting Number': [1]})
        with self.assertRaises(ValueError) as _:
            Dataset.from_dataframe(df)
        # test that every link must be specified either by
        # PD code or by name
        df = DataFrame({'Name': ['3_1', ''],
                        'PD Notation': ['[[1,5,2,4],[3,1,4,6],[5,3,6,2]]', '']})
        with self.assertRaises(ValueError) as _:
            Dataset.from_dataframe(df)

    def test_from_links(self) -> None:
        # check that deep copies of the arguments have been
        # made
        links = ['3_1']
        unknotting_nums = [1]
        slice_genera = [1]
        names = ['trefoil']
        _ = Dataset.from_links(links, unknotting_nums,
                               slice_genera, names=names)
        self.assertEqual(links, ['3_1'])
        self.assertEqual(unknotting_nums, [1])
        self.assertEqual(slice_genera, [1])
        self.assertEqual(names, ['trefoil'])
        # check that the name in links is not overwriting
        # the name in names
        dataset = Dataset.from_links(['3_1'],
                                     names=['trefoil'])
        self.assertEqual(dataset.names[0], 'trefoil')
        # check that the name of a Link object is not
        # overwriting the name in names
        dataset = Dataset.from_links([Link('3_1')],
                                     names=['trefoil'])
        self.assertEqual(dataset.names[0], 'trefoil')
        # check that the name in links is assigned to names
        # if names is not provided
        dataset = Dataset.from_links(['3_1'])
        self.assertEqual(dataset.names[0], '3_1')
        # check that we are converting link names to links
        dataset = Dataset.from_links(['3_1'])
        self.assertEqual(dataset.links[0].name, '3_1')
        # check that we can initialize links with their PD
        # codes
        dataset = Dataset.from_links([[(1, 5, 2, 4),
                                       (3, 1, 4, 6),
                                       (5, 3, 6, 2)]])
        self.assertEqual(len(dataset.links[0].crossings), 3)
        # check that unknotting numbers are correctly
        # converted
        dataset = Dataset.from_links(['3_1'], [1])
        self.assertEqual(dataset.unknotting_nums[0], [1])
        dataset = Dataset.from_links(['10_11'], [(2, 3)])
        self.assertEqual(dataset.unknotting_nums[0], [2, 3])
        dataset = Dataset.from_links(['10_11'], [[3, 2]])
        self.assertEqual(dataset.unknotting_nums[0], [2, 3])
        # check that slice genera are correctly converted
        dataset = Dataset.from_links(['3_1'],
                                     slice_genera=[1])
        self.assertEqual(dataset.slice_genera[0], [1])
        dataset = Dataset.from_links(['13n_3522'],
                                     slice_genera=[(1, 2)])
        self.assertEqual(dataset.slice_genera[0], [1, 2])
        dataset = Dataset.from_links(['13n_3522'],
                                     slice_genera=[[2, 1]])
        self.assertEqual(dataset.slice_genera[0], [1, 2])

    # we are not testing dataset.Dataset.from_link because
    # it is a one-liner

    def test_read_csv(self) -> None:
        # test that the dataset can correctly handle
        # test_dataset.csv
        dataset = Dataset.read_csv('test_dataset.csv')

        self.assertEqual(len(dataset.links[0].crossings), 3)
        self.assertEqual(dataset.names[0], 'trefoil')
        self.assertIsNone(dataset.unknotting_nums[0])
        self.assertIsNone(dataset.slice_genera[0])

        self.assertEqual(len(dataset.links[1].crossings), 4)
        self.assertIsNone(dataset.names[1])
        self.assertEqual(dataset.unknotting_nums[1], [1])
        self.assertIsNone(dataset.slice_genera[1])

        self.assertEqual(dataset.links[2].name, '5_1')
        self.assertEqual(dataset.names[2], '5_1')
        self.assertIsNone(dataset.unknotting_nums[2])
        self.assertEqual(dataset.slice_genera[2], [2])

        self.assertEqual(dataset.links[3].name, '5_2')
        self.assertEqual(dataset.names[3], '5_2')
        self.assertEqual(dataset.unknotting_nums[3], [1])
        self.assertEqual(dataset.slice_genera[3], [1])

    def test_to_dataframe(self) -> None:
        links = [Link('3_1'), Link('4_1'), Link('5_1'),
                 '10_11', '13n_3522']
        unknotting_nums = [None, 1, None, (2, 3), (1, 2)]
        slice_genera = [None, None, 2, None, (1, 2)]
        names = [None, None, None, '10_11', None]
        dataset = Dataset.from_links(links, unknotting_nums,
                                     slice_genera,
                                     names=names)
        df = dataset.to_dataframe()
        self.assertEqual(df.columns[0], 'Name')
        self.assertEqual(df.columns[1], 'PD Notation')
        # Crossing Signs is always emitted, right after the PD code
        self.assertEqual(df.columns[2], 'Crossing Signs')
        self.assertEqual(df.columns[3], 'Unknotting Number')
        self.assertEqual(df.columns[4], 'Genus-4D')
        self.assertEqual(list(df['Name']),
                         ['', '', '', '10_11', '13n_3522'])
        self.assertEqual(df['PD Notation'][0],
                         '[[5;2;0;3];[3;0;4;1];[1;4;2;5]]')
        # signs are listed per crossing in PD order
        self.assertEqual(
            df['Crossing Signs'][0],
            str([c.sign for c in dataset.links[0].crossings])
            .replace(' ', '').replace(',', ';'))
        self.assertEqual(list(df['Unknotting Number']),
                         ['', '1', '', '[2;3]', '[1;2]'])
        self.assertEqual(list(df['Genus-4D']),
                         ['', '', '2', '', '[1;2]'])

    # we are not testing dataset.Dataset.write_csv because
    # it would create files in the local storage

    def test_to_list(self) -> None:
        # 9-tuples: (link, unknotting, slice, strong slice,
        # ribbon, strong ribbon, splitting, weak splitting, name)
        # the seven values are synthetic and pairwise distinct
        # so that a permutation of the tuple is visible
        dataset = Dataset.from_link('L5a1', 1, 2, 3, 4, 5, 6, 7)
        lst = dataset.to_list()
        self.assertEqual(lst[0][0].PD_code(),
                         [(4, 0, 5, 3), (0, 4, 1, 9), (6, 1, 7, 2),
                          (2, 7, 3, 8), (8, 5, 9, 6)])
        self.assertEqual(lst[0][0].name, 'L5a1')
        self.assertEqual(lst[0][1], [1])
        self.assertEqual(lst[0][2], [2])
        self.assertEqual(lst[0][3], [3])
        self.assertEqual(lst[0][4], [4])
        self.assertEqual(lst[0][5], [5])
        self.assertEqual(lst[0][6], [6])
        self.assertEqual(lst[0][7], [7])
        self.assertEqual(lst[0][8], 'L5a1')
        self.assertEqual(lst[0][-1], 'L5a1')

    def test_from_list(self) -> None:
        dataset = Dataset.from_list([(Link('3_1'), [1], [1], None,
                                      [1], None, None, None, '3_1'),
                                     (Link('8_8'), [2], [0], None,
                                      [0], None, None, None, '8_8')])
        self.assertEqual(dataset.links[0].name, '3_1')
        self.assertEqual(dataset.links[1].name, '8_8')
        self.assertEqual(dataset.unknotting_nums, [[1], [2]])
        self.assertEqual(dataset.slice_genera, [[1], [0]])
        self.assertEqual(dataset.ribbon_genera, [[1], [0]])
        self.assertEqual(dataset.names, ['3_1', '8_8'])

    def test_getitem(self) -> None:
        dataset = Dataset.read_csv('../datasets/rolfsen.csv')
        item = dataset[1]
        self.assertEqual(len(item), 9)
        self.assertEqual(item[-1], '4_1')
        # rolfsen.csv has no ribbon columns
        self.assertIsNone(item[4])
        self.assertIsNone(item[5])
        for l, r in [(0, 1), (2, 4), (1, 4), (3, 7)]:
            self.assertEqual(len(dataset[l:r].links), r-l)

    def test_len(self) -> None:
        for i in range(5):
            dataset = Dataset.read_csv('../datasets/rolfsen.csv')[:i]
            self.assertEqual(len(dataset), i)

    def test_add(self) -> None:
        ds1 = Dataset.read_csv('../datasets/rolfsen.csv')[:5]
        ds2 = Dataset.read_csv('../datasets/rolfsen.csv')[10:20]
        self.assertEqual(len(ds1 + ds2), 15)

    def test_iadd(self) -> None:
        ds1 = Dataset.read_csv('../datasets/rolfsen.csv')[:5]
        ds2 = Dataset.read_csv('../datasets/rolfsen.csv')[10:20]
        ds1 += ds2
        self.assertEqual(len(ds1), 15)

    def test_filtering(self) -> None:
        links = ['3_1', '10_11', '10_47', '13n_65',
                 '13n_3522']
        unknotting_nums = [1, (2, 3), (2, 3), 1, (2, 3)]
        slice_genera = [1, 1, 2, (0, 1), (2, 3)]
        dataset = Dataset.from_links(links, unknotting_nums,
                                     slice_genera)
        self.assertEqual(len(dataset.links_with_known_invariant_value('unknotting')), 2)
        self.assertEqual(len(dataset.links_with_unknown_invariant_value('unknotting')), 3)
        self.assertEqual(len(dataset.links_with_known_invariant_value('slice')), 3)
        self.assertEqual(len(dataset.links_with_unknown_invariant_value('slice')), 2)
        # no ribbon values were provided
        self.assertEqual(len(dataset.links_with_known_invariant_value('ribbon')), 0)
        self.assertEqual(len(dataset.links_with_unknown_invariant_value('ribbon')), 5)

    def test_ribbon_columns(self) -> None:
        # the four genus columns load into four distinct fields:
        # the ribbon regexes require the word "ribbon", so they
        # do not collide with "Genus-4D" / "Strong Slice Genus-4D".
        # The values are synthetic and pairwise distinct precisely
        # so that such a collision is visible; the row is left
        # unnamed because no link has them.
        df = DataFrame({
            'PD Notation': ['[[5;2;0;3];[3;0;4;1];[1;4;2;5]]'],
            'Genus-4D': ['0'],
            'Strong Slice Genus-4D': ['1'],
            'Ribbon Genus-4D': ['2'],
            'Strong Ribbon Genus-4D': ['3']})
        dataset = Dataset.from_dataframe(df)
        self.assertEqual(dataset.slice_genera[0], [0])
        self.assertEqual(dataset.strong_slice_genera[0], [1])
        self.assertEqual(dataset.ribbon_genera[0], [2])
        self.assertEqual(dataset.strong_ribbon_genera[0], [3])
        # ... and they round-trip through to_dataframe
        df2 = dataset.to_dataframe()
        self.assertEqual(list(df2['Ribbon Genus-4D']), ['2'])
        self.assertEqual(list(df2['Strong Ribbon Genus-4D']), ['3'])
        dataset2 = Dataset.from_dataframe(df2)
        self.assertEqual(dataset2.ribbon_genera[0], [2])
        self.assertEqual(dataset2.strong_ribbon_genera[0], [3])

    def test_finite_strong_ribbon_genus(self) -> None:
        # knots always qualify; the Hopf link (linking number 1)
        # has infinite strong ribbon / strong slice genus
        self.assertTrue(Dataset.has_finite_strong_ribbon_genus(Link('3_1')))
        self.assertFalse(Dataset.has_finite_strong_ribbon_genus(Link('L2a1')))
        # the slice-named helpers delegate to the ribbon-named ones
        self.assertTrue(Dataset.has_finite_strong_slice_genus(Link('3_1')))
        self.assertFalse(Dataset.has_finite_strong_slice_genus(Link('L2a1')))
        dataset = Dataset.from_links(['3_1', 'L2a1', '4_1'])
        self.assertEqual(len(dataset.links_with_finite_strong_ribbon_genus()), 2)
        self.assertEqual(len(dataset.links_with_finite_strong_slice_genus()), 2)

    def test_split_by_number_of_crossings(self) -> None:
        dataset = Dataset.read_csv('../datasets/rolfsen.csv')[:6]
        split_list = dataset.split_by_number_of_crossings()
        self.assertEqual([len(c) for c in split_list],
                         [0, 0, 0, 1, 1, 2, 2])

    def test_sample(self) -> None:
        dataset = Dataset.read_csv('../datasets/rolfsen.csv')[:3]
        with self.assertRaises(ValueError) as _:
            dataset.sample(-1)
        self.assertEqual(len(dataset.sample(0)), 0)
        self.assertEqual(len(dataset.sample(1)), 1)
        self.assertEqual(len(dataset.sample(2)), 2)
        self.assertEqual(len(dataset.sample(3)), 3)
        with self.assertRaises(ValueError) as _:
            dataset.sample(4)

    def test_simplify(self) -> None:
        one_vertex_unknot = Link([(0, 0, 1, 1)])
        dataset = Dataset.from_links([one_vertex_unknot, '3_1'])
        self.assertEqual(dataset.simplify(), 1)
        self.assertEqual(len(dataset.links[0].crossings), 0)

    def test_evaluate_invariant(self) -> None:
        dataset = Dataset.read_csv('../datasets/rolfsen.csv')
        with self.assertRaises(ValueError) as _:
            dataset.evaluate_invariant('unknotting', [(-1, [], [Link('3_1')])])
        dataset = Dataset.from_link(Link('3_1'))
        with self.assertRaises(ValueError) as _:
            dataset.evaluate_invariant('unknotting', [(-1, [], [Link('3_1')])], False, 'evaluation.txt')
        with self.assertRaises(ValueError) as _:
            dataset.evaluate_invariant('unknotting', [(1, ['C2S2_start', 'C1S1_end2'], [Link('3_1'), name_to_link('0_1')])])
        dataset = Dataset.from_link('3_1', 1)
        self.assertEqual(dataset.evaluate_invariant('unknotting', [(2, [], [Link('3_1'), name_to_link('0_1')], [])]), 0.0)
        self.assertEqual(dataset.evaluate_invariant('unknotting', [(1, ['C0S1_start', 'C0S0_end1'], [Link('3_1'), name_to_link('0_1')], [])]), 1.0)


class CrossingSignsTestCases(unittest.TestCase):
    # A minimal 13-crossing knot (13n_45); spherogram deduces an
    # all-nonzero, deterministic orientation for it.
    PD = [[2, 0, 3, 25], [0, 7, 1, 8], [6, 1, 7, 2], [8, 4, 9, 3],
          [11, 4, 12, 5], [5, 12, 6, 13], [18, 10, 19, 9],
          [10, 18, 11, 17], [20, 14, 21, 13], [14, 22, 15, 21],
          [22, 16, 23, 15], [16, 24, 17, 23], [24, 20, 25, 19]]

    def _signs(self, link):
        return [c.sign for c in link.crossings]

    def _roundtrip(self, dataset):
        path = tempfile.mktemp(suffix='.csv')
        try:
            dataset.write_csv(path)
            return Dataset.read_csv(path)
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_to_dataframe_always_has_crossing_signs(self) -> None:
        ds = Dataset.from_links([self.PD], names=['13n_45'])
        df = ds.to_dataframe()
        self.assertIn('Crossing Signs', df.columns)
        # the column lists the link's signs in PD order
        self.assertEqual(
            df['Crossing Signs'].iloc[0],
            str(self._signs(ds.links[0])).replace(' ', '').replace(',', ';'))

    def test_roundtrip_preserves_signs_default_orientation(self) -> None:
        # Branch 1 of the verify-and-keep read-back: stored signs equal
        # spherogram's deduced signs, so the link is kept verbatim.
        ds = Dataset.from_links([self.PD], names=['13n_45'])
        original = self._signs(ds.links[0])
        ds2 = self._roundtrip(ds)
        self.assertEqual(self._signs(ds2.links[0]), original)
        # PD code is preserved verbatim in this branch
        self.assertEqual([list(t) for t in ds2.links[0].PD_code()],
                         [list(t) for t in ds.links[0].PD_code()])

    def test_roundtrip_preserves_signs_nondefault_orientation(self) -> None:
        # Branch 2: stored signs differ from spherogram's default
        # (here, the mirror), so the read-back rebuilds from explicit
        # signs via link_from_crossings. The signs must still survive.
        default = self._signs(Link(self.PD))
        mirror = [-s for s in default]
        ds = Dataset.from_links([self.PD], names=['13n_45'],
                                crossing_signs=[mirror])
        self.assertEqual(self._signs(ds.links[0]), mirror)
        ds2 = self._roundtrip(ds)
        self.assertEqual(self._signs(ds2.links[0]), mirror)

    def test_from_dataframe_deduces_when_column_absent(self) -> None:
        pd_str = str([list(c) for c in self.PD]).replace(' ', '').replace(',', ';')
        df = DataFrame({'Name': ['13n_45'], 'PD Notation': [pd_str]})
        ds = Dataset.from_dataframe(df)
        # spherogram's deduced signs are all nonzero
        self.assertTrue(all(s in (-1, 1) for s in self._signs(ds.links[0])))

    def test_from_links_rejects_mismatched_signs(self) -> None:
        with self.assertRaises(ValueError) as _:
            Dataset.from_links([self.PD], crossing_signs=[[1, 1]])


class EvaluateInvariantDumpTestCases(unittest.TestCase):
    """The random-states dump is the diagnostic artifact of a run that may
    have taken days. It must land wherever it is pointed, and it must never
    take the run down with it: the default path is relative, so under a
    cluster scheduler its parent need not exist."""

    def setUp(self) -> None:
        self.link = name_to_link('3_1')
        # answer 1 beats the stored upper bound 2 -> NEW RESULT, which is what
        # makes evaluate_invariant emit detailed info and dump the states.
        self.output = [(1, [('a', 0.0)], [self.link], ['rs0'])]

    def _evaluate(self, path, value=None, output=None):
        return evaluate_invariant(
            'unknotting number', True, [self.link], ['3_1'],
            [value or [1, 2]], output or self.output,
            False, '', path, False)

    def test_dump_creates_missing_parents(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'nested', 'deeper', 'random-states.json')
            self.assertEqual(self._evaluate(path), 1.0)
            with open(path) as f:
                self.assertEqual(json.load(f), [['rs0']])

    def test_dump_failure_never_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            # parent is a FILE, so both makedirs and open must fail
            blocker = os.path.join(tmp, 'blocker')
            open(blocker, 'w').close()
            path = os.path.join(blocker, 'random-states.json')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                accuracy = self._evaluate(path)
            self.assertEqual(accuracy, 1.0)
            self.assertTrue(any('could not save the random states'
                                in str(w.message) for w in caught))

    def test_illegal_answer_warns(self) -> None:
        # the known value is {0, 2}, so an answer of 1 is one the invariant
        # cannot take: either the agent or the dataset is wrong, and it must
        # be loud rather than buried in the detailed table of a long log.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, 'random-states.json')
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                accuracy = self._evaluate(
                    path, value=[0, 2],
                    output=[(1, [], [self.link], ['rs0'])])
            self.assertEqual(accuracy, 0.0)
            self.assertTrue(any('illegal' in str(w.message) for w in caught))


if __name__ == '__main__':
    unittest.main()
