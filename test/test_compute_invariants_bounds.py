import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from spherogram import Link

from bounds_pipeline.compute_invariants_bounds import main
from dataset import Dataset


class BoundsCLIExportTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.input_path = Path(self.tempdir.name) / 'links.csv'
        link = Link('3_1')
        pd_code = str([list(crossing) for crossing in link.PD_code()])
        signs = str([crossing.sign for crossing in link.crossings])
        # Aliases must be retained without becoming duplicate canonical
        # columns on reload. Names deliberately are not unique or complete.
        self.input_df = pd.DataFrame({
            'Name': ['duplicate', 'duplicate', None],
            'PD Notation (vector)': [pd_code] * 3,
            'Crossing Signs': [signs] * 3,
            'Unlinking Number': ['[0;3]'] * 3,
            'Genus-4D': ['[0;2]'] * 3,
            'reference': ['first source', 'second source', 'third source'],
            'certificate': ['proof, with comma', None, 'third certificate'],
            'best_upper_bound_unknotting_number_source': ['original'] * 3,
        })
        self.input_df.to_csv(self.input_path, index=False)

    def results(self, known=False):
        rows = []
        for index, upper in enumerate((1, None, 2)):
            rows.append({
                'index': index,
                'name': 'duplicate' if index < 2 else 'link_2',
                'PD_code': self.input_df.loc[index, 'PD Notation (vector)'],
                'error': 'synthetic computation failure' if index == 1 else None,
                'is_knot': True,
                'is_algebraically_split': True,
                'best_slice_lower': 1,
                'slice_upper_seifert': 1,
                'unknotting_lower': 1,
                'unknotting_upper': upper,
                'best_upper_bound_unknotting_number': upper,
                'best_upper_bound_unknotting_number_source': (
                    None if index == 1 else 'known_unknotting_number' if known
                    else ('collari_positive' if index == 0
                          else 'crossing_number')),
                'best_lower_bound_slice_genus_source': 'signature',
            })
        # A positional concatenation or a name-based join would misalign rows.
        return pd.DataFrame(rows).iloc[[2, 0, 1]].reset_index(drop=True)

    def run_cli(self, path, results):
        with (patch('sys.argv', ['compute_invariants_bounds.py', str(path)]),
              patch('bounds_pipeline.slice_bounds_py.compute_bounds_for_dataset',
                    return_value=(results, None)),
              patch('bounds_pipeline.bounds_report.print_improved_bounds'),
              patch('bounds_pipeline.bounds_report.print_improved_unknotting_bounds'),
              contextlib.redirect_stdout(io.StringIO())):
            # Exercise the actual Dataset loaders, bound updaters, and export.
            main()
        output_path = path.with_name(path.stem + '_newBounds.csv')
        return output_path, pd.read_csv(output_path)

    def test_export_preserves_provenance_and_reloads_updated_dataset(self):
        results = self.results()
        output_path, output = self.run_cli(self.input_path, results)
        pd.testing.assert_frame_equal(
            output[self.input_df.add_prefix('input_').columns],
            pd.read_csv(self.input_path).add_prefix('input_'))
        expected_results = results.set_index('index').sort_index().drop(
            columns=['name', 'PD_code']).reset_index(drop=True)
        pd.testing.assert_frame_equal(
            output[expected_results.columns], expected_results,
            check_dtype=False)
        self.assertNotIn('PD_code', output.columns)
        self.assertNotIn('name', output.columns)
        self.assertNotIn('Unnamed: 0', output.columns)

        original = Dataset.read_csv(self.input_path)
        reloaded = Dataset.read_csv(output_path)
        self.assertEqual(reloaded.names, ['duplicate', 'duplicate', None])
        self.assertEqual(reloaded.unknotting_nums, [[1], [0, 1, 2, 3], [1, 2]])
        self.assertEqual(reloaded.slice_genera, [[1], [0, 1, 2], [1]])
        for before, after in zip(original.links, reloaded.links):
            self.assertEqual(after.PD_code(), before.PD_code())
            self.assertEqual([c.sign for c in after.crossings],
                             [c.sign for c in before.crossings])

    def test_rerun_keeps_prior_concrete_sources(self):
        first_path, first_output = self.run_cli(self.input_path, self.results())
        second_path, second_output = self.run_cli(first_path, self.results(known=True))
        pd.testing.assert_frame_equal(
            second_output[first_output.add_prefix('input_').columns],
            first_output.add_prefix('input_'))
        source = 'best_upper_bound_unknotting_number_source'
        self.assertEqual(second_output.loc[0, source], 'known_unknotting_number')
        self.assertEqual(second_output.loc[0, 'input_' + source],
                         'collari_positive')
        self.assertEqual(second_output.loc[0, 'input_input_' + source], 'original')
        reloaded = Dataset.read_csv(second_path)
        self.assertEqual(reloaded.unknotting_nums, [[1], [0, 1, 2, 3], [1, 2]])


if __name__ == '__main__':
    unittest.main()
