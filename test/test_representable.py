import unittest
import re

from spherogram import Link

from representable_py import (DimensionExceededError,
                              ErrorState,
                              CrossChangeState,
                              BandMoveState)

class RepresentableTestCases(unittest.TestCase):
    def test_error_state_to_tensor(self):
        trefoil = Link('3_1')
        state = CrossChangeState.from_link(trefoil, shape=(3, 1))
        state = state.next_state('C0S1_start')
        error_state = ErrorState(state, 'C0S0_end2')
        self.assertIsNotNone(re.search(r'^ErrorState instance with\nprevious state =\n.*\n', repr(error_state)))

    def test_next_state(self):
        trefoil = Link('3_1')
        state1 = CrossChangeState.from_link(trefoil, shape=(3, 1))
        state1 = state1.next_state('C0S1_start')
        with self.assertRaises(DimensionExceededError) as _:
            state1.next_state('C0S0_end2')

        hopf_positive = Link([(2, 1, 3, 0), (1, 2, 0, 3)])
        state2 = BandMoveState.from_link(hopf_positive, shape=(2, 2))
        state2 = state2.next_state('C0S3_start').next_state('C0S2_under')
        with self.assertRaises(DimensionExceededError) as _:
            state2.next_state('C0S1_end0')


if __name__ == '__main__':
    unittest.main()
