import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from spherogram import Link

from bounds_pipeline.bounds_report import format_bounds
from bounds_pipeline.slice_bounds_py import (
    DatasetBoundsSummary,
    _add_unknotting_bounds_to_results,
    compute_bounds,
    update_dataset_with_unknotting_bounds,
)


class UnknottingBoundsAggregationTests(unittest.TestCase):
    def aggregate(self, source, known, slice_lower=1):
        dataset = SimpleNamespace(
            slice_genera=[None],
            unknotting_nums=[None if known is None else list(known)],
            links=[SimpleNamespace(crossings=[None] * 6)],
        )
        row = {
            'index': 0,
            'name': 'synthetic{0}',
            'error': None,
            'num_crossings': 6,
            'best_slice_lower': slice_lower,
            'best_lower_bound_slice_genus_source': 'signature',
        }
        positive = {}
        if source == 'collari_positive':
            # A bounded record exercises the positive upper-bound path without
            # running the theorem/certificate computations themselves.
            positive[0] = SimpleNamespace(
                exact=False, u=None, u_lower=1, u_upper=2,
                provenance='collari_positive_bounded', mirrored=False,
                orientation=(), lk=0, components=(),
            )
        else:
            row['unknotting_upper_simply_linked'] = 2
        df = pd.DataFrame([row])
        summary = DatasetBoundsSummary()
        with patch('bounds_pipeline.slice_bounds_py._positive_unknotting_rows',
                   return_value=positive):
            _add_unknotting_bounds_to_results(df, dataset, summary)
        return df, dataset, summary

    def test_collari_upper_merges_with_known_bound(self):
        cases = (
            ([1], 1, 'known_unknotting_number'),
            ([1, 2], 2, 'known_unknotting_number'),
            ([1, 2, 3], 2, None),
            ([], 2, None),
            (None, 2, None),
        )
        for source in ('collari_positive', 'collari_simply_linked'):
            for known, upper, replacement_source in cases:
                with self.subTest(source=source, known=known):
                    df, dataset, summary = self.aggregate(source, known)
                    row = df.iloc[0]
                    expected_source = replacement_source or source
                    self.assertEqual(row['best_upper_bound_unknotting_number'],
                                     upper)
                    self.assertEqual(
                        row['best_upper_bound_unknotting_number_source'],
                        expected_source)
                    self.assertEqual(row['unknotting_lower'], 1)
                    self.assertEqual(row['unknotting_upper'], 2)
                    self.assertEqual(row['best_lower_bound_unknotting_number'], 1)
                    self.assertEqual(
                        summary.positive_unknotting_improved_count,
                        int(expected_source == 'collari_positive'))
                    # Reporting must not mutate the stored known interval.
                    self.assertEqual(dataset.unknotting_nums, [known])

                    # The existing updater still intersects theoretical and
                    # known bounds, independently of best-bound provenance.
                    update_dataset_with_unknotting_bounds(dataset, df)
                    expected = ([v for v in known if 1 <= v <= 2]
                                if known else [1, 2])
                    self.assertEqual(dataset.unknotting_nums, [expected])

    def test_positive_lower_still_counts_when_known_upper_wins(self):
        df, _, summary = self.aggregate('collari_positive', [0, 1, 2],
                                        slice_lower=0)
        row = df.iloc[0]
        self.assertEqual(row['unknotting_lower'], 1)
        self.assertEqual(row['unknotting_upper'], 2)
        self.assertEqual(row['best_lower_bound_unknotting_number'], 1)
        self.assertEqual(row['best_lower_bound_unknotting_number_source'],
                         'collari_positive_bounded_synthetic{0}')
        self.assertEqual(row['best_upper_bound_unknotting_number'], 2)
        self.assertEqual(row['best_upper_bound_unknotting_number_source'],
                         'known_unknotting_number')
        self.assertEqual(summary.positive_unknotting_improved_count, 1)


class StrongSliceGenusApiTests(unittest.TestCase):
    def bounds(self, link, use_component_bounds=True):
        return compute_bounds(
            link, use_component_bounds=use_component_bounds,
            use_component_identification=False)

    def test_crossingless_unlinks_have_exact_zero_strong_genus(self):
        for num_components in (1, 2, 3):
            for use_component_bounds in (False, True):
                with self.subTest(num_components=num_components,
                                  use_component_bounds=use_component_bounds):
                    link = Link([])
                    link.unlinked_unknot_components = num_components
                    bounds = self.bounds(link, use_component_bounds)
                    self.assertEqual(bounds.num_components, num_components)
                    self.assertEqual(bounds.num_crossings, 0)
                    self.assertEqual(bounds.is_knot, num_components == 1)
                    self.assertEqual((bounds.strong_slice_genus_lower,
                                      bounds.strong_slice_genus_upper),
                                     (0.0, 0.0))
                    self.assertFalse(bounds.strong_slice_genus_is_infinite)
                    self.assertFalse(bounds.strong_slice_obstructed)
                    self.assertFalse(bounds.bounds_inconsistent)
                    self.assertEqual(bounds.consistency_errors, [])
                    report = format_bounds(bounds)
                    self.assertIn('g₄* = 0 (crossingless unlink)', report)
                    self.assertNotIn('no finite upper bound is proved', report)

    def test_nontrivial_links_keep_their_strong_genus_bounds(self):
        cases = (
            ('3_1', (1.0, 1.0), True, False),
            ('L2a1', (math.inf, math.inf), False, True),
            ('L5a1', (1.0, math.inf), False, False),
        )
        for name, interval, is_knot, is_infinite in cases:
            with self.subTest(name=name):
                bounds = self.bounds(Link(name))
                self.assertEqual((bounds.strong_slice_genus_lower,
                                  bounds.strong_slice_genus_upper), interval)
                self.assertEqual(bounds.is_knot, is_knot)
                self.assertEqual(bounds.strong_slice_genus_is_infinite,
                                 is_infinite)
                self.assertTrue(bounds.strong_slice_obstructed)
                self.assertFalse(bounds.bounds_inconsistent)
                self.assertNotIn('crossingless unlink', format_bounds(bounds))

    def test_knot_with_isolated_unknot_is_not_classified_as_a_knot(self):
        link = Link('3_1')
        link.unlinked_unknot_components = 1
        bounds = self.bounds(link)
        self.assertEqual(bounds.num_components, 2)
        self.assertFalse(bounds.is_knot)
        self.assertEqual((bounds.strong_slice_genus_lower,
                          bounds.strong_slice_genus_upper), (1.0, math.inf))
        self.assertFalse(bounds.strong_slice_genus_is_infinite)
        self.assertTrue(bounds.strong_slice_obstructed)
        self.assertFalse(bounds.bounds_inconsistent)
        report = format_bounds(bounds)
        self.assertNotIn('crossingless unlink', report)
        self.assertIn('no finite upper bound is proved', report)

    def test_empty_link_is_not_treated_as_a_nonempty_unlink(self):
        bounds = self.bounds(Link([]))
        self.assertEqual(bounds.num_components, 0)
        self.assertFalse(bounds.is_knot)
        self.assertNotEqual((bounds.strong_slice_genus_lower,
                             bounds.strong_slice_genus_upper), (0.0, 0.0))
        self.assertNotIn('crossingless unlink', format_bounds(bounds))


if __name__ == '__main__':
    unittest.main()
