import unittest

import numpy as np
from spherogram import Link

from env import (UnknottingEnv, RibbonEnv, StrongRibbonEnv,
                 SliceEnv, StrongSliceEnv)
from representable_py import ErrorState
from config import hfk_enabled


class EnvTestCases(unittest.TestCase):
    def assert_observation_space_matches(self, env, obs) -> None:
        self.assertTrue(env.observation_space.contains(obs))
        self.assertEqual(set(env.observation_space.spaces), set(obs))
        for label, array in obs.items():
            with self.subTest(feature=label):
                space = env.observation_space[label]
                self.assertEqual(space.shape, array.shape)
                self.assertEqual(space.dtype, array.dtype)
                self.assertEqual(array.dtype, np.dtype(np.float64))
                self.assertTrue(np.all(np.isneginf(space.low)))
                self.assertTrue(np.all(np.isposinf(space.high)))

    def test_observation_space_matches_reset_and_step(self) -> None:
        for env_cls in (UnknottingEnv, RibbonEnv, StrongRibbonEnv,
                        SliceEnv, StrongSliceEnv):
            for mdp in (False, True):
                with self.subTest(environment=env_cls.__name__, mdp=mdp):
                    env = env_cls(
                        [Link('3_1')], shape=(10, 4), mdp=mdp,
                        features_cls=['AlexanderPolynomial',
                                      'JonesPolynomial'])
                    obs, _ = env.reset()
                    self.assert_observation_space_matches(env, obs)
                    # Starting a band is legal in every framework.
                    obs, _, terminated, _, _ = env.step(
                        env.label_actions['C0S0_start'])
                    self.assertFalse(terminated)
                    self.assert_observation_space_matches(env, obs)
                    # Error observations must satisfy the same contract.
                    env.reset()
                    obs, _, terminated, _, _ = env.step(
                        env.label_actions['C0S0_over'])
                    self.assertTrue(terminated)
                    self.assertIsInstance(env.cur_state, ErrorState)
                    self.assert_observation_space_matches(env, obs)

    def test_observation_space_preserves_large_polynomial_coefficients(self) -> None:
        # Ten trefoils have Alexander polynomial (1 - t + t^2)^10.
        # Its coefficients exceed both former observation bounds.
        knot = Link('3_1')
        expected = np.array([1, -1, 1], dtype=np.int64)
        for _ in range(9):
            knot = knot.connected_sum(Link('3_1'))
            expected = np.convolve(expected, [1, -1, 1])
        env = UnknottingEnv([knot], shape=(30, 4),
                           features_cls=['AlexanderPolynomial'])
        obs, _ = env.reset()
        coefficients = obs['alexander_coeffs']
        self.assertEqual(coefficients.shape, (61,))
        np.testing.assert_array_equal(coefficients[:len(expected)], expected)
        self.assertEqual(np.max(coefficients), 8953)
        self.assertLess(np.min(coefficients), -1000)
        # Padding remains -1000; genuine coefficients are not clipped to it.
        np.testing.assert_array_equal(coefficients[len(expected):], -1000)
        self.assert_observation_space_matches(env, obs)

    def test_knotenv_init(self) -> None:
        # check that the sampling method is indeed checked
        with self.assertRaises(ValueError) as _:
            UnknottingEnv([Link('3_1')],
                         sampling_method='other')
        # check that the action labels have been correctly
        # set
        for max_vert in range(3, 11):
            env = UnknottingEnv([Link('3_1')],
                               (max_vert, 4))
            num_actions = 20*max_vert+2
            self.assertEqual(env.num_actions, num_actions)
            self.assertEqual(len(env.action_labels),
                             num_actions)
            self.assertEqual(len(env.label_actions),
                             num_actions)
            self.assertEqual(env.action_space.n,
                             num_actions)
            # check that the observation space has the
            # correct dimension
            self.assertEqual(len(env.observation_space), 10)

    def test_step(self) -> None:
        env = UnknottingEnv([Link('3_1')])
        # check that the state goes to an ErrorState if
        # given an illegal action
        terminated = env.step(env.label_actions['C0S0_over'])[2]
        self.assertTrue(isinstance(env.cur_state, ErrorState))
        # check that the terminal state has been reached
        self.assertTrue(terminated)
        # check that executing an action on ErrorState does
        # not change the state
        env.step(env.label_actions['C0S3_end1'])
        self.assertTrue(isinstance(env.cur_state, ErrorState))

        # check that we go to an ErrorState when dimensions
        # are exceeded
        env = UnknottingEnv([Link('3_1')], (3, 4))
        env.step(env.label_actions['C0S1_start'])
        with self.assertWarns(Warning) as _:
            terminated = env.step(env.label_actions['C0S0_end2'])[2]
            self.assertTrue(terminated)
        self.assertTrue(isinstance(env.cur_state, ErrorState))

        # check that the link history and the list of random
        # states has been appended
        env = UnknottingEnv([Link('3_1')])
        self.assertEqual(len(env.actions_rewards), 0)
        env.step(env.label_actions['C0S1_start'])
        self.assertEqual(len(env.actions_rewards), 1)
        terminated = env.step(env.label_actions['C0S0_end1'])[2]
        self.assertEqual(len(env.actions_rewards), 2)
        self.assertEqual(len(env.link_history), 2)
        self.assertEqual(len(env.random_states), 1)
        self.assertEqual(env.link_history[-1].crossings, [])
        self.assertTrue(terminated)

        env = UnknottingEnv([Link('3_1')])
        env.step(env.label_actions['C0S1_start'])
        # check that the terminal state has not been reached
        terminated = env.step(env.label_actions['C0S0_end2'])[2]
        self.assertFalse(terminated)

    def test_reset(self) -> None:
        env = UnknottingEnv([Link('3_1')])
        with self.assertRaises(ValueError) as _:
            env.reset(options={'dataset': 1})
        with self.assertRaises(ValueError) as _:
            env.reset(options={'dataset': []})
        env = UnknottingEnv([Link('3_1')])
        env.reset(options={'dataset': [Link('4_1')]})
        self.assertEqual(len(env.dataset), 1)
        self.assertEqual(env.dataset[0].name, '4_1')

        env = UnknottingEnv([Link('3_1')])
        with self.assertRaises(ValueError) as _:
            env.reset(options={'index': 'abc'})
        env = UnknottingEnv([Link('3_1')])
        with self.assertRaises(ValueError) as _:
            env.reset(options={'index': -1})
        env = UnknottingEnv([Link('3_1')])
        with self.assertRaises(ValueError) as _:
            env.reset(options={'index': 1})
        env = UnknottingEnv([Link('3_1'), Link('5_2')])
        env.reset(options={'index': 1})
        self.assertEqual(env.index, 1)

        env = UnknottingEnv([Link('3_1'), Link('5_2')],
                           sampling_method='random')
        index_counts = [0, 0]
        for _ in range(50):
            env.reset()
            index_counts[env.index] += 1
        self.assertGreater(index_counts[0], 0)
        self.assertGreater(index_counts[1], 0)

        env = UnknottingEnv([Link('3_1')])
        env.reset(options={'dataset': [Link('3_1'), Link('5_2')]})
        self.assertEqual(env.index, 0)
        env.reset()
        self.assertEqual(env.index, 1)
        env.reset()
        self.assertEqual(env.index, 0)

        # check that the parameter max_twists is indeed
        # passed to base_state.from_link
        env = UnknottingEnv([Link('3_1')], max_twists=3)
        self.assertEqual(env.cur_state.max_twists, 3)

        # check that a ValueError is raised when the initial
        # observation cannot be fit into shape
        # we have to check for Exception instead because the
        # error raised is a Nim error, wrapped by NimPy
        with self.assertRaises(Exception) as _:
            UnknottingEnv([Link('5_1')], (4, 4))

        # check that the link history is correctly assigned
        env = UnknottingEnv([Link('3_1')])
        self.assertEqual(len(env.link_history), 1)
        self.assertEqual(len(env.link_history[0].crossings), 3)
        self.assertEqual(len(env.random_states), 0)
        self.assertEqual(len(env.actions_rewards), 0)

        # check that self.seifert_genus is correctly set if
        # HFK is enabled
        if hfk_enabled:
            env = RibbonEnv([Link('3_1')])
            self.assertEqual(env.seifert_genus, 1)
            env = RibbonEnv([Link('5_1')])
            self.assertEqual(env.seifert_genus, 2)

        # check that the observation is correct
        env = UnknottingEnv([Link('3_1')], shape=(100, 4))
        obs = env.reset()[0]
        self.assertEqual(len(obs), 10)

    def test_action_masks(self) -> None:
        env = UnknottingEnv([Link('3_1')])
        self.assertEqual(sum(env.action_masks()), 12)
        env.step(env.label_actions['C0S0_start'])
        self.assertEqual(sum(env.action_masks()), 10)
        env.step(env.label_actions['C2S1_over'])
        self.assertEqual(sum(env.action_masks()), 0)

    def test_sample_from_actions(self) -> None:
        env = UnknottingEnv([Link('3_1')])
        env.sample_from_actions([env.label_actions['C0S1_start'],
                                 'C0S0_end1'])
        self.assertTrue(env.is_unlink())

    def test_is_unlink(self) -> None:
        env = UnknottingEnv([Link('3_1')])
        self.assertFalse(env.is_unlink())
        env.sample_from_actions(['C0S1_start', 'C0S0_end1'])
        self.assertTrue(env.is_unlink())

    def test_is_terminal(self) -> None:
        # check that ErrorState is terminal
        env = UnknottingEnv([Link('3_1')])
        self.assertFalse(env.is_terminal())
        env.step(env.label_actions['C0S0_over'])
        self.assertFalse(env.is_terminal())
        env = UnknottingEnv([Link('3_1')])
        env.sample_from_actions(['C0S1_start', 'C0S0_end1'])
        self.assertTrue(env.is_terminal())

    def test_num_end_actions(self) -> None:
        env = UnknottingEnv([Link('3_1')])
        self.assertEqual(env.num_end_actions(), 0)
        env.sample_from_actions(['C0S1_start', 'C0S0_end1'])
        self.assertEqual(env.num_end_actions(), 1)

    def test_compute_current_answer(self) -> None:
        env = UnknottingEnv([Link('3_1')])
        self.assertEqual(env.compute_current_answer(), 0)
        env.sample_from_actions(['C0S1_start', 'C0S0_end1'])
        self.assertEqual(env.compute_current_answer(), 1)

    def test_bandmoveenv_init(self) -> None:
        env = RibbonEnv([Link('3_1')], shape=(10, 4))
        self.assertEqual(env.num_actions, 242)
        self.assertEqual(len(env.action_masks()), 242)

    def test_slice_env_birth(self) -> None:
        # the slice envs have one extra action, unknot_birth,
        # at the last index; the Nim mask length must match
        env = SliceEnv([Link('3_1')], shape=(10, 4))
        self.assertEqual(env.num_actions, 243)
        self.assertEqual(env.action_labels[-1], 'unknot_birth')
        masks = env.action_masks()
        self.assertEqual(len(masks), env.num_actions)
        # birth is legal while no band is in progress
        self.assertTrue(masks[-1])
        obs, reward, terminated, _, _ = env.step(
            env.label_actions['unknot_birth'])
        self.assertFalse(isinstance(env.cur_state, ErrorState))
        self.assertFalse(terminated)
        # the birth is recorded in the link history but is not
        # an end action
        self.assertEqual(len(env.link_history), 2)
        self.assertEqual(env.num_end_actions(), 0)
        # birth is masked out while a band is in progress
        env.reset(options={'index': 0})
        env.step(env.label_actions['C0S0_start'])
        self.assertFalse(env.action_masks()[-1])
        # a birth leaves the genus unchanged: the newborn disk
        # is a genus 0 component of the surface. Checked on a
        # surface that already has genus, since the genus is 0
        # either way at the start of an episode
        env.reset(options={'index': 0})
        for label in ['C0S0_start', 'C0S3_end0',
                      'C0S0_start', 'C0S3_end0']:
            env.step(env.label_actions[label])
        genus_before = env.compute_current_answer()
        self.assertEqual(genus_before, 1)
        env.step(env.label_actions['unknot_birth'])
        self.assertEqual(env.compute_current_answer(), genus_before)

    def test_strong_slice_env_birth(self) -> None:
        env = StrongSliceEnv([Link('3_1')], shape=(10, 4))
        self.assertEqual(env.num_actions, 243)
        self.assertEqual(env.action_labels[-1], 'unknot_birth')
        self.assertEqual(len(env.action_masks()), 243)
        env.step(env.label_actions['unknot_birth'])
        self.assertFalse(isinstance(env.cur_state, ErrorState))
        env = StrongRibbonEnv([Link('3_1')], shape=(10, 4))
        self.assertEqual(env.num_actions, 242)
        self.assertEqual(len(env.action_masks()), 242)

    def test_strong_no_all_false_mask_to_policy(self) -> None:
        # Regression: the strong frameworks may legally reach a
        # non-terminal state with no legal moves; such an all-False
        # mask must never be handed back to the policy (it crashes
        # MaskablePPO with a native SIGBUS). step() converts a stuck
        # state into a (terminal) ErrorState, so the invariant is:
        # after any step, a non-terminal state has at least one legal
        # action. Exercise it with masked random rollouts.
        import random
        import warnings
        L5a1 = [[5, 0, 6, 1], [9, 6, 4, 7], [3, 4, 0, 5],
                [1, 9, 2, 8], [7, 3, 8, 2]]
        for EnvCls in (StrongSliceEnv, StrongRibbonEnv):
            env = EnvCls([Link(L5a1)], shape=(50, 8))
            rng = random.Random(0)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for _ in range(40):                 # episodes
                    env.reset(options={'index': 0})
                    for _ in range(60):             # steps per episode
                        masks = env.action_masks()
                        self.assertEqual(len(masks), env.num_actions)
                        legal = [i for i, m in enumerate(masks) if m]
                        if not legal:
                            # only allowed on a terminal state (ErrorState
                            # or unlink); never on a live non-terminal one
                            self.assertTrue(env.is_terminal())
                            break
                        _, _, terminated, _, _ = env.step(rng.choice(legal))
                        if terminated:
                            break

    def test_unfused_birth_warns_on_termination(self) -> None:
        for env_cls in (SliceEnv, StrongSliceEnv):
            env = env_cls([Link('3_1')], shape=(10, 4), mdp=True)
            for lbl in ['C0S0_start', 'C0S3_end0',
                        'C0S0_start', 'C0S3_end0']:
                env.step(env.label_actions[lbl])
            self.assertTrue(env.is_terminal())
            self.assertFalse(env.cur_state.has_unfused_birth_components())
            env.reset(options={'index': 0})
            # Birth first, then reduce the original trefoil. The warning
            # reports intentional deletion, not an unhandled component.
            env.step(env.label_actions['unknot_birth'])
            with self.assertWarnsRegex(UserWarning, 'discarded pure-birth'):
                for lbl in ['C0S0_start', 'C0S3_end0',
                            'C0S0_start', 'C0S3_end0']:
                    obs, _, terminated, _, _ = env.step(
                        env.label_actions[lbl])
            self.assertTrue(terminated)
            self.assertTrue(env.cur_state.has_unfused_birth_components())
            self.assertEqual(env.compute_current_answer(), 1)
            np.testing.assert_array_equal(obs['current_genus'], [1])
            if env_cls is StrongSliceEnv:
                self.assertIn('with 1 components', env.terminal_state_message())


if __name__ == '__main__':
    unittest.main()
