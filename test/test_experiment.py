import unittest
from unittest.mock import Mock, patch

from experiment import RLParam, MixedStrategyParam, band_operation_rl


class ExperimentTestCases(unittest.TestCase):
    def test_rl_param_init(self) -> None:
        with self.assertRaises(ValueError) as _:
            RLParam(sampling_method='other')

    def test_mixed_strategy_param_init(self) -> None:
        with self.assertRaises(ValueError) as _:
            MixedStrategyParam(1.0, 1.0,
                               p_rw_range=(-0.1, 0.5))
        with self.assertRaises(ValueError) as _:
            MixedStrategyParam(1.0, 1.0,
                               p_rw_range=(0.5, 0.0))
        with self.assertRaises(ValueError) as _:
            MixedStrategyParam(1.0, 1.0,
                               p_rw_range=(0.0, 1.1))

    def test_rollout_pruning_allows_discardable_birth_genus(self) -> None:
        cases = [('slice', True, True), ('strong slice', True, True),
                 ('slice', False, False), ('strong slice', False, False),
                 ('ribbon', True, False), ('strong ribbon', True, False)]
        for initial_genus, mixed in ((0, False), (0, True),
                                     (1, False), (1, True)):
            for framework, pure_birth, should_continue in cases:
                with self.subTest(initial_genus=initial_genus, mixed=mixed,
                                  framework=framework,
                                  pure_birth=pure_birth):
                    env = Mock()
                    env.unwrapped = env
                    env.framework = framework
                    env.state_type.beaten_by_offset = 0
                    # The intermediate genus reaches the incumbent, but a
                    # terminal pure-birth deletion can still improve it.
                    # Also cover resuming from positive pure-birth genus.
                    env.compute_current_answer.side_effect = [initial_genus, 1, 0]
                    env.is_terminal.side_effect = [False, True]
                    env.cur_state.has_unfused_birth_components.return_value = pure_birth
                    label = 'C0S0_end0'
                    env.cur_state.random_walk_transition.return_value = label
                    env.action_labels = {0: label}
                    env.label_actions = {label: 0}
                    env.step.side_effect = [({}, 0.0, False, False, {}),
                                            ({}, 0.0, True, False, {})]
                    env.actions_rewards = []
                    env.link_history = []
                    env.random_states = []
                    model = Mock()
                    model.predict.return_value = (0, None)
                    rng = Mock()
                    rng.random.return_value = 0.0
                    mixed_params = [1.0] * 6 + [0.0] if mixed else None
                    with patch('experiment.get_action_masks',
                               return_value=[True]):
                        answer, _, _, _ = band_operation_rl(
                            model, env, {}, mixed_params, current_best=1,
                            verbose=False, rng=rng)
                    self.assertEqual(answer, 0 if should_continue else -1)
                    expected_steps = 2
                    if not should_continue:
                        expected_steps = 0 if initial_genus else 1
                    self.assertEqual(env.step.call_count, expected_steps)


if __name__ == '__main__':
    unittest.main()
