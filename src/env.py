import abc
from typing import (Union, List, Tuple, Dict, Any, TypeVar,
                    Generic)
import warnings
import math
import os

import numpy as np
from gymnasium.core import Env
from gymnasium.spaces import (Discrete, Box,
                              Dict as DictSpace)
from spherogram import Link
import dill

from representable import (band_get_act_type,
                           nim_rand_state,
                           nim_set_seed)
from representable_py import (ErrorState,
                              DimensionExceededError,
                              CrossChangeState,
                              BandMoveState)
from config import hfk_enabled
from seeding import derive_seed

ActType = int
ObsType = Dict[str, np.ndarray]

StateType = TypeVar('StateType', CrossChangeState,
                    BandMoveState)

# Defaults of the reward every framework uses when
# old_reward is False (CrossChangeEnv._reward and
# BandMoveEnv._reward). They are read once, when this
# module is imported. The two reward classes derive their
# settings from them, and every environment copies those
# settings into itself when it is built, so a pickled
# environment carries the reward it was trained with
# (KnotEnv.reward_settings, experiment.RLParam).
#
# REWARD_MAX_ACTIONS is the number of actions an episode may
# take, REWARD_ANSWER_CAP the largest answer the reward
# tells apart, and the two classes derive their weights from
# them. The cap should be at least the largest answer in the
# datasets. Raising either extends the ordering the reward
# guarantees to longer episodes or larger answers, at the
# price of a weaker signal about path length.
#
# REWARD_SCALE multiplies every reward: it changes the
# size of the values the value network has to fit. With
# 0.01 the episode returns are tens; at 0.1 they reach
# hundreds, and the tanh units of the value network
# saturate into a constant prediction.

REWARD_MAX_ACTIONS = int(os.environ.get('REWARD_MAX_ACTIONS', 80))
REWARD_ANSWER_CAP = int(os.environ.get('REWARD_ANSWER_CAP', 8))
REWARD_SCALE = float(os.environ.get('REWARD_SCALE', 0.01))

# The settings a reward class may define and an environment
# then carries: see CrossChangeEnv and BandMoveEnv for what
# each one means, and KnotEnv.reward_settings to read or
# replace them on an environment that already exists. A class
# that does not define one simply does not have it, as
# answer_charged_at_end, which belongs to the band move
# frameworks alone.
REWARD_SETTINGS = ('MAX_ACTION_COST', 'ANSWER_CAP',
                   'ANSWER_WEIGHT', 'WIN_REWARD', 'SCALE',
                   'answer_charged_at_end')

# Sentinels written by the Nim encoders: PAD fills the unused
# part of fixed-size arrays and marks undefined invariants,
# GAUSS_SEPARATOR ends each component of the Gauss code and
# BAND_UNVISITED marks the strands of the band matrix that
# the band never visits.
PAD = -1000.0
GAUSS_SEPARATOR = -500.0
BAND_UNVISITED = -10.0

# Features that can be undefined on a state, with the flag
# encode_observation adds for each: the flag is 1 when every
# entry of the key is PAD. jones_lo stands for the whole
# Jones feature (jones_lo, jones_coeffs) and seifert_genus
# for the knot Floer triple (seifert_genus, nu, tau), whose
# keys are undefined together. Theta carries its own flag,
# theta_undefined.
UNDEFINED_FLAGS = {
    'determinant': 'determinant_undefined',
    'signature': 'signature_undefined',
    'mt_bound': 'mt_bound_undefined',
    'seifert_genus_bound': 'seifert_genus_bound_undefined',
    'alexander_coeffs': 'alexander_undefined',
    'jones_lo': 'jones_undefined',
    'seifert_genus': 'knot_floer_undefined',
}

# Polynomial coefficient windows, written from index 0 and
# padded with PAD after the last coefficient. The Alexander
# polynomial is normalised to a positive leading coefficient,
# so its padding is exactly the trailing run of PAD.
POLYNOMIAL_WINDOWS = ('alexander_coeffs', 'jones_coeffs',
                      'alexander_poly')


def encode_observation(obs: Dict[str, np.ndarray]
                       ) -> Dict[str, np.ndarray]:
    """
    Rewrites the sentinels of an observation as small
    integers. Labels and indices (PD code, band path,
    component matrix, connected component labels) are
    shifted to start at 1, so that 0 is free; padding and
    the Gauss code separator become 0; the band matrix uses
    -2 for a strand the band never visits and -3 for a
    missing crossing; an undefined invariant becomes 0 and
    sets its flag from UNDEFINED_FLAGS. All other values are
    unchanged: the policy network scales them (see
    experiment.ScaledInputExtractor). KnotEnv.observe applies
    it when mdp is set.

    Args:
        obs: an observation as built by the Nim encoders

    Returns:
        A new dictionary with the same keys and, for every
        feature of UNDEFINED_FLAGS present in obs, a flag
        array that is [1.0] when the feature is undefined.
    """
    out = {}
    for key, value in obs.items():
        x = np.array(value, dtype=float)
        pad = x == PAD
        if key == 'band_matrix':
            x[x == BAND_UNVISITED] = -2.0
            x[pad] = -3.0
            out[key] = x
            continue
        if key in ('PD_code', 'component_ccl', 'component_matrix'):
            # component_matrix uses -1 for a strand that no
            # component visits, which becomes 0
            x += 1.0
        elif key == 'gauss_code':
            x[x == GAUSS_SEPARATOR] = 0.0
        elif key == 'band_path':
            # rows of (crossing, strand index, sign)
            x.reshape(-1, 3)[:, :2] += 1.0
        elif key == 'genera_info':
            # rows starting with the connected component label
            x.reshape(-1, 5)[:, 0] += 1.0
        elif key in POLYNOMIAL_WINDOWS:
            # the coefficients start at index 0 and only the
            # trailing run of PAD is padding: a coefficient
            # equal to PAD inside the polynomial is kept
            real = np.flatnonzero(x != PAD)
            pad = np.zeros_like(pad)
            pad[real[-1] + 1 if real.size else 0:] = True
        x[pad] = 0.0
        out[key] = x
    for key, flag in UNDEFINED_FLAGS.items():
        if key in obs:
            undefined = bool(np.all(np.asarray(obs[key]) == PAD))
            out[flag] = np.array([1.0 if undefined else 0.0])
    return out


class KnotEnv(Generic[StateType], abc.ABC,
              Env[ObsType, ActType]):
    """
    The knot environment we use.

    Note this is a Gymnasium environment.
    """
    dataset: List[Link]
    index: int
    cur_state: Union[ErrorState, StateType]
    actions_rewards: List[Tuple[str, float]]
    link_history: List[Link]
    random_states: List[str]
    seifert_genus: int | None
    # the default of an environment unpickled from before
    # old_reward existed
    old_reward: bool = True

    def __init__(
            self,
            dataset: List[Link],
            shape: Tuple[int, int],
            max_twists: int,
            gamma: float,
            features_cls: List[str],
            sampling_method: str,
            verbose: int,
            framework: str,
            state_type: type[StateType],
            simplification_switch: bool = True,
            mdp: bool = False,
            old_reward: bool = True
    ):
        """
        Initializes an instance of KnotEnv with all the
        fields filled in and the environment reset to the
        starting state on a link according to the specified
        sampling method. Since KnotEnv is an abstract base
        class, this method should only called in __init__
        methods of subclasses of KnotEnv using
        super().__init__.

        Args:
            dataset: a list of links that this environment
                is sampling from; every time self.reset() is
                called, we step to the next link in this
                list (looping back to the first link if we
                are currently at the last link) if
                sampling_method is set to "consecutive" or
                choose another link randomly from dataset if
                sampling_method is set to "random", and run
                our environment starting from this link
            shape: a pair of integers; the first component
                of shape means the number of crossings that
                any link (either given or generated by an
                end action) can have at most; the second
                value of shape means the number of
                components that any link in the episodes can
                have at most; if shape is exceeded in
                self.reset() (meaning the initial
                observation cannot be computed), an error is
                raised; if shape is exceeded in self.step()
                (meaning the observation of some link
                generated by an end action cannot be
                computed), the environment goes to an
                ErrorState and the episode ends
            max_twists: the maximum number of twists (across
                all the faces, either counterclockwise or
                clockwise) allowed
            gamma: the discount factor; used in
                self.get_reward.reward_shaping() to make the
                reward shaping component of the reward act
                like a potential
            features_cls: a list of the classes of other
                features of the link that we are using
            sampling_method: how we choose the initial link
                and the next link from the dataset after
                an episode has ended; "consecutive" means
                that we step to the next link (cyclically)
                and "random" means that we step to another
                link randomly chosen from the dataset
            verbose: whether we are printing the reset
                parameters when self.reset() is called and
                the action and the reward when self.step()
                is called
            framework: the type of link invariant that we
                are calculating; pass "unknotting" for the
                unknotting number, "ribbon" / "strong ribbon"
                for the (strong) ribbon genus, "slice" /
                "strong slice" for the (strong) slice genus
                (the ribbon frameworks plus the unknot_birth
                action), "splitting" for the splitting
                number, "weak splitting" for the weak
                splitting number, and "unknotting-alg" for
                the algebraic unknotting number; for a
                detailed explanation, see the docstring of
                band.newBandTwist
            state_type: the type of the representable state
                (CrossChangeState or BandMoveState) that we
                are keeping as the current state in
                self.cur_state
            simplification_switch: if True, links are
                simplified between states; if False, links
                are not simplified between states (but
                features and rewards are still computed on
                simplified versions), except that the band
                move frameworks remove the Reidemeister I
                kinks a band leaves, since a band cannot act
                on a diagram with a monogon face
            mdp: if True, the observation also carries the
                part of the state that to_tensor leaves out
                although the legal actions, the next state
                or the reward depend on it (see
                self.observe), which makes it a Markov
                state, and its sentinels are rewritten by
                encode_observation for the policy network;
                if False, the observation is to_tensor alone,
                unchanged
            old_reward: if True, every framework uses the
                reward of its get_reward, the reward of the
                models trained before self._reward existed;
                if False, every framework uses self._reward
                (CrossChangeEnv._reward for the crossing
                change frameworks, BandMoveEnv._reward for
                the band move ones), charging what
                self.reward_settings() holds: the settings of
                the class, themselves derived from the
                environment variables REWARD_MAX_ACTIONS,
                REWARD_ANSWER_CAP and REWARD_SCALE, unless
                experiment.create_env gives the environment
                the ones stored in its parameters

        Raises:
            ValueError: if sampling_method is not among
                "consecutive" and "random"
        """
        if sampling_method not in ["consecutive", "random"]:
            raise ValueError(f'"{sampling_method}" is not '
                             f'a valid sampling method')

        self.shape = shape
        self.max_twists = max_twists
        self.gamma = gamma
        if features_cls is None:
            features_cls: List[str] = []
        self.features_cls = features_cls
        self.sampling_method = sampling_method
        self.verbose = verbose
        self.framework = framework
        self.state_type = state_type
        self.simplification_switch = simplification_switch
        self.mdp = mdp
        self.old_reward = old_reward
        # A copy of the reward settings of the class, so that
        # this environment carries the reward it was built
        # with wherever it is pickled, and so that changing
        # them on one environment leaves the others alone
        for name in REWARD_SETTINGS:
            if hasattr(self, name):
                setattr(self, name, getattr(self, name))

        self.action_labels: List[str] = []
        # Creates a list of action names and a dictionary
        # mapping an action's name to its index.
        # by convention, self.shape[0] is the maximum number
        # of crossings that any link in self.cur_state may
        # have, so we initialize the actions with c
        # (crossing) from 0 to self.shape[0] - 1
        # not all actions are valid for self.cur_state; this
        # is enforced in band.BandTwist.check_strand_error
        # and illegal actions are masked out by MaskablePPO
        # through self.action_masks()
        for c in range(self.shape[0]):
            for i in range(4):
                # start at crossing c, strand i
                self.action_labels.append(f'C{c}S{i}_start')
                # let the band go over the strand with
                # crossing c, strand index i
                self.action_labels.append(f'C{c}S{i}_over')
                # let the band go under the strand with
                # crossing c, strand index i
                self.action_labels.append(f'C{c}S{i}_under')
                # end the band at final strand with crossing
                # c, strand index i, positive sign: for
                # CrossChange this means the first edge of
                # the band going over the final strand; for
                # BandMove this means with a positive twist
                # added just before ending the band path
                self.action_labels.append(f'C{c}S{i}_end1')
                # end the band at final strand with crossing
                # c, strand index i, negative sign: for
                # CrossChange this means the first edge of
                # the band going under the final strand; for
                # BandMove this means with a negative twist
                # added just before ending the band path
                self.action_labels.append(f'C{c}S{i}_end2')
        # twist counterclockwise (from the point of view of
        # the existing band towards the end)
        self.action_labels.append('ccw_twist')
        # twist clockwise
        self.action_labels.append('cw_twist')
        self._set_action_labels()
        self.label_actions = {action_label: index for index, action_label in enumerate(self.action_labels)}
        self.num_actions = len(self.action_labels)
        self.action_space = Discrete(self.num_actions)

        obs, _ = self.reset(options={'dataset': dataset})
        # Observations are not clipped: invariant values can exceed 1000.
        # Describe the arrays actually returned, including their dtype.
        self.observation_space = DictSpace(
            {label: Box(low=-np.inf, high=np.inf,
                        shape=array.shape, dtype=array.dtype)
             for label, array in obs.items()}
        )

    def _set_action_labels(self) -> None:
        pass

    def observe(
            self,
            state: Union[ErrorState, StateType]
    ) -> ObsType:
        """
        Computes the observation of a state of the current
        episode: state.to_tensor() and, when self.mdp is
        set, also state.mdp_to_tensor(), the crossing number
        of the initial diagram, which scales the reward
        shaping, and, for band moves with HFK enabled, the
        Seifert genus of the initial link, which the reward
        compares the genus against. The sentinels are then
        rewritten by encode_observation.

        Args:
            state: a state of the current episode

        Returns:
            The observation, a dictionary of float tensors
            (np.ndarray), without clipping or scaling; when
            self.mdp is False, to_tensor exactly as the Nim
            encoders write it.
        """
        obs = state.to_tensor()
        if self.mdp:
            obs.update(state.mdp_to_tensor())
            obs['init_num_crossings'] = np.array(
                [float(len(self.link_history[0].crossings))])
            if hfk_enabled and self.state_type == BandMoveState:
                obs['init_seifert_genus'] = np.array(
                    [float(self.seifert_genus)])
            if (not self.old_reward
                    and self.state_type == CrossChangeState):
                # the final charge of CrossChangeEnv._reward
                # depends on it
                obs['crossing_changes'] = np.array(
                    [float(self.num_end_actions())])
            return encode_observation(obs)
        return obs

    def step(
            self,
            action: ActType
    ) -> Tuple[ObsType, float, bool, bool, Dict[str, Any]]:
        """
        Executes the given action on the current state.

        We sample the next state as follows:
        - If the current state is an error state, we keep it
          unchanged.
        - If the given action is illegal or the tensor
          representation of the next state cannot be fit
          into self.shape, we go into an error state.
        - Otherwise, we set self.cur_state to the next
          state.

        Then we calculate the reward: self._reward if
        self.old_reward is False, self.get_reward otherwise.
        The episode terminates if we run into an error state
        or reach a terminal link.

        When the episode has terminated or truncated
        (indicated by True values in either terminated or
        truncated), self.reset() should be called
        afterwards.

        Args:
            action: the next action to be executed on the
                current state

        Returns:
            A 5-tuple, consisting of the following
            components:
            1. observation: agent's observation of the
               current environment
            2. reward: the amount of reward for the given
               action
            3. terminated: whether the episode has ended, in
               which case further step() calls will return
               undefined results; in our case, an episode
               has ended iff we have reached a terminal link
               or run into an error
            2. truncated: whether the episode has reached
               the maximum number of steps; we always return
               False in self.step(), but any usage of
               KnotEnv will enforce the maximum number of
               steps via a TimeLimit Gymnasium wrapper
            5. info: auxiliary diagnostic information,
               currently set to the empty dictionary
        """
        state = self.cur_state
        label = self.action_labels[action]
        if isinstance(state, ErrorState):
            next_state = state
        elif state.check_action_error(label):
            next_state = ErrorState(state, label)
        else:
            if band_get_act_type(label) == 'end':
                self.random_states.append(nim_rand_state())
            try:
                next_state = state.next_state(label)
                if band_get_act_type(label) in ('end', 'birth'):
                    # self.link_history is vital to
                    # recovering a certificate for the
                    # link invariant; births change the
                    # diagram too (the birthed unknot is
                    # joined into it), but consume no Nim
                    # RNG, so no random state is recorded
                    self.link_history.append(next_state.link())
            except DimensionExceededError as e:
                warnings.warn(str(e))
                next_state = ErrorState(state, label)
            else:
                # The 'splitting' / 'strong ribbon' / 'strong slice'
                # frameworks may legally reach a non-terminal state
                # with no legal moves (e.g. mid-band, the same-CCL end
                # constraint leaves nothing and birth is illegal while a
                # band is in progress).
                # We treat the # dead end as an ErrorState.
                if (self.framework in ('splitting', 'strong ribbon',
                                       'strong slice')
                        and not next_state.is_terminal()
                        and not any(next_state.action_masks())):
                    next_state = ErrorState(state, label)
        if not self.old_reward:
            reward = self._reward(state, label, next_state)
        else:
            reward = self.get_reward(state, label, next_state)
        self.actions_rewards.append((label, reward))
        if self.verbose:
            # we should not print a newline in step()
            print(f'{label}: {reward}; ', end='')
        self.cur_state = next_state
        terminated = (isinstance(next_state, ErrorState) or
                      next_state.is_terminal())
        # At a terminal unlink, pure-birth surface components (which
        # may have positive genus) are discarded from the answer.
        # Their CCLs remain in the bookkeeping and trajectory.
        if (terminated
                and not isinstance(next_state, ErrorState)
                and self.framework in ('slice', 'strong slice')
                and next_state.has_unfused_birth_components()):
            warnings.warn(
                f'{self.framework} episode discarded pure-birth surface '
                'components from the terminal answer '
                '(CCLs with num_orig_comps == 0)')
        obs = self.observe(next_state)
        if self.verbose == 2:
            print(self.cur_state)
            print(obs)
            input()
        return obs, reward, terminated, False, {}

    @abc.abstractmethod
    def get_reward(
            self,
            state: ErrorState | StateType,
            action: str,
            next_state: ErrorState | StateType
    ) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _reward(
            self,
            state: ErrorState | StateType,
            action: str,
            next_state: ErrorState | StateType
    ) -> float:
        raise NotImplementedError

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None
              ) -> Tuple[ObsType, Dict[str, Any]]:
        """
        If given a dataset, resets self.dataset to the given
        dataset.
        Steps to the given link (if the "index" option is
        provided), a random link in the dataset (if sampling
        method is random), the first link in the dataset (if
        sampling method is consecutive and a new dataset is
        given) or the next link (if sampling method is
        consecutive and no new dataset is given), and sets
        the state to the link with no band attached.
        Returns the initial observation.

        Args:
            seed: if given, reseeds the generator of the
                random link sampling, self.np_random, and
                the Nim generator of representable.so that
                the transitions draw from; if None, both
                continue their streams
            options: parameters for the reset, currently the
                two options available are "dataset" -- the
                dataset to reset self.dataset to, and
                "index" -- the index of the link in the
                dataset that we are resetting the
                environment to

        Returns:
            A 2-tuple, consisting of the following
            components:
            1. observation: the initial observation
            2. info: additional information complementing
               the observation, currently set to the empty
               dictionary

        Raises:
            ValueError: if the given dataset is not of type
                list or an empty list, or if the given index
                is not of type int or out of range
            DimensionExceededError: if the initial
                observation for the new link after the reset
                cannot be fit into self.shape

        Notes:
            self.set_state and self.set_dataset have both
            been merged to self.reset.
        """
        super().reset(seed=seed)
        if seed is not None:
            # the Nim generator belongs to representable.so, so
            # every environment of the process shares it; it is
            # reseeded before from_link, which simplifies
            nim_set_seed(derive_seed(seed, 'nim'))
        if options is None:
            options = {}
        if 'dataset' in options:
            dataset = options['dataset']
            if not isinstance(dataset, list):
                raise ValueError(f'dataset {dataset} is '
                                 f'not of type List[Link]')
            if not dataset:
                raise ValueError(f'dataset is empty')
            self.dataset = dataset
        if 'index' in options:
            index = options['index']
            if not isinstance(options['index'], int):
                raise ValueError(f'index {index} is not of '
                                 f'type int')
            if index < 0 or index >= len(self.dataset):
                raise ValueError(f'index {index} is out of '
                                 f'range [0, '
                                 f'{len(self.dataset)})')
            self.index = index
        elif self.sampling_method == "random":
            self.index = int(self.np_random.integers(
                len(self.dataset)))
        elif 'dataset' in options:
            self.index = 0
        else:
            self.index = (self.index+1) % len(self.dataset)
        if self.verbose:
            if 'dataset' in options:
                print(f'given new dataset with '
                      f'{len(self.dataset)} links, ',
                      end='')
            # we should print a newline in reset()
            print(f'reset to link {self.index}')
        cur_link = self.dataset[self.index]
        try:
            init_state = self.state_type.from_link(
                cur_link, self.max_twists, self.framework,
                self.shape, self.features_cls, 
                self.simplification_switch
            )
        except DimensionExceededError as e:
            # If the initial link exceeds dimensions, 
            # skip to find another link
            num_crossings = len(cur_link.crossings)
            num_components = len(cur_link.link_components) + cur_link.unlinked_unknot_components
            warnings.warn(f"Link at index {self.index} ({cur_link}) does not respect shape "
                         f"because it has {num_crossings} crossings (max: {self.shape[0]}) "
                         f"and {num_components} components (max: {self.shape[1]}). Skipping it.")
            # Try to find a valid link in the dataset
            attempts = 0
            max_attempts = len(self.dataset)
            while attempts < max_attempts:
                self.index = (self.index + 1) % len(self.dataset)
                cur_link = self.dataset[self.index]
                try:
                    init_state = self.state_type.from_link(
                        cur_link, self.max_twists, self.framework,
                        self.shape, self.features_cls, 
                        self.simplification_switch
                    )
                    if self.verbose:
                        print(f'Using link at index {self.index} instead')
                    break
                except DimensionExceededError:
                    attempts += 1
            else:
                # No valid link found in entire dataset
                raise ValueError(f"No link in dataset fits within shape {self.shape}. "
                               f"Consider increasing shape")
        self.cur_state = init_state
        self.actions_rewards = []
        # the RHS of next line should not be changed to
        # [cur_link] because the crossings of cur_link have
        # been relabeled in utils.nim:decode_link
        self.link_history = [init_state.link()]
        self.random_states = []
        self.seifert_genus = None
        if self.state_type == BandMoveState and hfk_enabled:
            self.seifert_genus = 1000
            link = self.link_history[0]
            if len(link.link_components) + link.unlinked_unknot_components == 1:
                self.seifert_genus = link.knot_floer_homology()['seifert_genus']
        obs = self.observe(self.cur_state)
        if self.verbose == 2:
            print(self)
            print(obs)
            input()
        return obs, {}

    def action_masks(self) -> List[bool]:
        """
        Calculates a mask of legal actions for use by the
        action masking in the MaskablePPO algorithm.

        Returns:
            A function mapping an action to whether it is
            legal on the current state.

        References:
            "InvalidActionEnvDiscrete has a [sic]
            action_masks method that returns the invalid
            action mask (True if the action is valid, False
            otherwise)." from
            https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html#example
        """
        state = self.cur_state
        if isinstance(state, ErrorState):
            return [False] * self.num_actions
        msk = self.cur_state.action_masks()
        if not any(msk):
            if self.framework not in ['splitting', 'strong ribbon',
                                      'strong slice']:
                print(state.link().PD_code())
                print(state.link().faces())
                print(state.band_path)
                raise AssertionError
            else:
                warnings.warn(f'no legal moves for state '
                              f'{self.cur_state}')
        return msk

    def sample_from_actions(
            self,
            actions: List[Union[ActType, str]]
    ) -> Union[ErrorState, StateType]:
        """
        Calculates the state after executing a sequence of
        actions.

        Args:
            actions: A list of actions, in the form of their
                labels or indices.

        Returns:
            The state of the current environment after this
            sequence of actions.
        """
        for action in actions:
            if isinstance(action, str):
                action = self.label_actions[action]
            self.step(action)
        return self.cur_state

    def is_unlink(self) -> bool:
        """
        Tests whether the current state's link is an unlink.

        Returns:
            Whether the current state's link is an unlink.
        """
        return (not isinstance(self.cur_state, ErrorState) and
                self.cur_state.is_unlink())

    def is_terminal(self) -> bool:
        """
        Tests whether the current state's link is a terminal
        link.

        Returns:
            Whether the current state's link is a terminal
            link.
        """
        return (not isinstance(self.cur_state, ErrorState) and
                self.cur_state.is_terminal())

    def num_end_actions(self) -> int:
        return len(self.link_history) - 1

    def reward_settings(self,
                        settings: Dict[str, float] = None
                        ) -> Dict[str, float]:
        """
        Reads, and optionally replaces, the settings that
        self._reward uses: the ones of REWARD_SETTINGS its
        class defines. Every environment holds its own copy,
        so a replacement here changes this environment alone,
        and an environment unpickled from before they were
        stored falls back to the settings of its class.

        Args:
            settings: the settings to replace, by name; the
                ones left out keep their value

        Returns:
            The settings of this environment, by name.

        Raises:
            ValueError: if settings holds a name that is not
                a reward setting
        """
        names = [name for name in REWARD_SETTINGS
                 if hasattr(self, name)]
        if settings:
            unknown = set(settings) - set(names)
            if unknown:
                raise ValueError(f'{sorted(unknown)} are not reward '
                                 f'settings of a {type(self).__name__}; '
                                 f'they are {names}')
            for name, value in settings.items():
                setattr(self, name, value)
        return {name: getattr(self, name) for name in names}

    def compute_current_answer(self) -> int:
        return self.cur_state.compute_current_answer(
            self.link_history[0],
            self.num_end_actions()
        )

    def terminal_state_message(self) -> str:
        if not self.is_terminal():
            raise ValueError('env.KnotEnv.terminal_state_message '
                             'can only be called when the '
                             'state is terminal')
        return self.cur_state.terminal_state_message(
            self.num_end_actions(),
            self.compute_current_answer()
        )

    def __repr__(self) -> str:
        return (f'{type(self)} instance with\n'
                f'|actions|={self.num_actions}\n'
                f'gamma={self.gamma}\n'
                f'sampling method={self.sampling_method}\n'
                f'verbose={self.verbose}\n'
                f'mdp={self.mdp}\n'
                f'old_reward={self.old_reward}\n'
                f'reward settings={self.reward_settings()}\n'
                f'|dataset|={len(self.dataset)}\n'
                f'current index={self.index}\n'
                f'[(action, reward)]={self.actions_rewards}\n'
                f'link history={self.link_history}\n'
                f'current state=\n{self.cur_state}')

    def __str__(self) -> str:
        return self.__repr__()

    def __getstate__(self) -> bytes:
        return dill.dumps(self.__dict__)

    def __setstate__(self, state: bytes) -> None:
        self.__dict__ = dill.loads(state)


class CrossChangeEnv(KnotEnv[CrossChangeState], abc.ABC):
    """
    The knot environment with crossing changes as actions.
    The answer of these frameworks, the quantity self._reward
    charges, is the number of crossing changes made.
    """
    # Weights of self._reward, in the units of its immediate
    # term. MAX_ACTION_COST is the cost of the most expensive
    # action that does not win, ANSWER_CAP the largest answer
    # the reward tells apart, and the last two follow from
    # them, so that the ordering of the _reward docstring
    # holds. They belong to the crossing change frameworks
    # alone: BandMoveEnv carries its own.
    MAX_ACTION_COST = 5
    ANSWER_CAP = REWARD_ANSWER_CAP
    ANSWER_WEIGHT = MAX_ACTION_COST * REWARD_MAX_ACTIONS
    WIN_REWARD = ANSWER_WEIGHT * (ANSWER_CAP + 1)
    # multiplies the reward, to keep the returns small
    # enough for the value network to fit
    SCALE = REWARD_SCALE

    def __init__(
            self,
            dataset: List[Link],
            shape: Tuple[int, int],
            max_twists: int,
            gamma: float,
            features_cls: List[str],
            sampling_method: str,
            verbose: int,
            framework: str,
            simplification_switch: bool = True,
            mdp: bool = False,
            old_reward: bool = True
    ):
        super().__init__(dataset, shape, max_twists, gamma,
                         features_cls, sampling_method,
                         verbose, framework,
                         CrossChangeState,
                         simplification_switch, mdp=mdp, old_reward=old_reward)

    def _reward(
            self,
            state: ErrorState | CrossChangeState,
            action: str,
            next_state: ErrorState | CrossChangeState
    ) -> float:
        """
        Calculates the reward when self.old_reward is False.
        It is SCALE times

            r + gamma * P' - P

        and, on the step that wins the episode,

            r - ANSWER_WEIGHT * min(a, ANSWER_CAP)
              + gamma * P' - P,

        where a is the number of crossing changes made, r is
        the immediate term (WIN_REWARD on the winning step,
        otherwise minus the cost of the action, between 1 and
        MAX_ACTION_COST) and P is minus the crossing number
        of the simplified diagram, weighted, and 0 at a
        terminal state. An action that leads to an error
        state returns -WIN_REWARD, with no potential term.

        With gamma = 1 and episodes of at most
        REWARD_MAX_ACTIONS actions, this orders the episodes
        of a link as: won, then stopped by the time limit,
        then ended on an error; and among won episodes, a
        lower answer first, then a shorter path. Charging the
        answer on the winning step alone is what keeps an
        episode that does not finish from paying for the
        crossing changes it tried, which otherwise teaches
        the agent to avoid the only end action it has.

        Args:
            state: current state
            action: the action we took
            next_state: the next state after action

        Returns:
            Reward received.
        """
        if isinstance(next_state, ErrorState):
            return self.SCALE * -self.WIN_REWARD

        crossing_weight = reward_per_crossing_reduction(
            len(self.link_history[0].crossings))

        def potential(s: CrossChangeState) -> float:
            # we shape reward to encourage reducing the
            # number of crossings, as this is a good proxy
            # for how close we are to a terminal diagram
            return -crossing_weight * s.simplified_link_num_crossings()

        # the potential is 0 at a terminal state, so that
        # over a won episode the potential terms sum to a
        # constant of the link: a completely split diagram,
        # which wins a splitting episode, may still have
        # crossings
        next_potential = (0.0 if next_state.is_terminal()
                          else potential(next_state))
        shaping = self.gamma * next_potential - potential(state)

        if next_state.is_terminal():
            answer = next_state.compute_current_answer(
                self.link_history[0], self.num_end_actions())
            return self.SCALE * (
                self.WIN_REWARD
                - self.ANSWER_WEIGHT * min(answer, self.ANSWER_CAP)
                + shaping)
        if band_get_act_type(action) == 'end':
            cost = self.MAX_ACTION_COST
        else:
            band_length = len(state.band_path)
            cost = min(max(band_length - 7, 1), self.MAX_ACTION_COST)
        return self.SCALE * (-cost + shaping)


def reward_per_crossing_reduction(init_link_size: int) -> float:
    return 65 / (1 + math.sqrt(init_link_size))


class UnknottingEnv(CrossChangeEnv):
    """
    The knot environment calculating the unknotting number.
    """
    def __init__(
            self,
            dataset: List[Link],
            shape: Tuple[int, int] = (50, 4),
            max_twists: int = 5,
            gamma: float = 0.99,
            features_cls: List[str] = None,
            sampling_method: str = 'consecutive',
            verbose: int = 0,
            simplification_switch: bool = True,
            mdp: bool = False,
            old_reward: bool = True
    ):
        super().__init__(dataset, shape, max_twists, gamma,
                         features_cls, sampling_method,
                         verbose, 'unknotting',
                         simplification_switch, mdp=mdp, old_reward=old_reward)

    def get_reward(
            self,
            state: ErrorState | CrossChangeState,
            action: str,
            next_state: ErrorState | CrossChangeState
    ) -> float:
        """
        Calculates the reward corresponding to (state,
        actions, next_state) as follows:
        - If we reach the unlink, return lots of positive
          reward
        - If we reach an error state, return lots of
          negative reward
        - If we do a crossing change (i.e., execute an "end"
          action), we return a medium amount of negative
          reward.
        - If the band path has length at least 9, we return
          a small amount of negative reward, otherwise
          return a small amount of positive reward.

        We also use potential shaping to improve the
        training dynamics, where the potential function is
        defined in reward_shaping.

        KnotEnv.step uses it when self.old_reward is True,
        and self._reward otherwise.

        Args:
            state: current state
            action: the action we took
            next_state: the next state after action

        Returns:
            Reward received
        """
        if isinstance(next_state, ErrorState):
            return -100
        reward = 1
        if next_state.is_terminal():
            # We reached the unlink
            reward = 200
        elif band_get_act_type(action) == 'end':
            # We made a crossing change
            reward = -20
        elif len(self.cur_state.band_path) >= 9:
            reward = -1

        def reward_shaping(s: CrossChangeState):
            # Reward shaping potential function. Currently,
            # we shape reward to encourage reducing the
            # number of crossings, as this is a good proxy
            # for how close we are to the unlink.
            return -(s.simplified_link_num_crossings() *
                     reward_per_crossing_reduction(len(self.link_history[0].crossings)))
        return (reward
                + self.gamma*reward_shaping(next_state)
                - reward_shaping(state))


class SplittingEnv(CrossChangeEnv):
    """
    The knot environment calculating the splitting number.
    """
    def __init__(
            self,
            dataset: List[Link],
            shape: Tuple[int, int] = (50, 4),
            max_twists: int = 5,
            gamma: float = 0.99,
            features_cls: List[str] = None,
            sampling_method: str = 'consecutive',
            verbose: int = 0,
            simplification_switch: bool = True,
            mdp: bool = False,
            old_reward: bool = True
    ):
        super().__init__(dataset, shape, max_twists, gamma,
                         features_cls, sampling_method,
                         verbose, 'splitting',
                         simplification_switch, mdp=mdp, old_reward=old_reward)

    def get_reward(
            self,
            state: ErrorState | CrossChangeState,
            action: str,
            next_state: ErrorState | CrossChangeState
    ) -> float:
        """
        Calculates the reward corresponding to (state,
        actions, next_state) as follows:
        - If we reach a completely split link, return lots
          of positive reward
        - If we reach an error state, return lots of
          negative reward
        - If we do a crossing change (i.e., execute an "end"
          action), we return a medium amount of negative
          reward.
        - If the band path has length at least 9, we return
          a small amount of negative reward, otherwise
          return a small amount of positive reward.

        We also use potential shaping to improve the
        training dynamics, where the potential function is
        defined in reward_shaping.

        KnotEnv.step uses it when self.old_reward is True,
        and self._reward otherwise.

        Args:
            state: current state
            action: the action we took
            next_state: the next state after action

        Returns:
            Reward received
        """
        if isinstance(next_state, ErrorState):
            return -100
        reward = 1
        if next_state.is_terminal():
            # We reached a completely split link
            reward = 200
        elif band_get_act_type(action) == 'end':
            # We made a crossing change
            reward = -20
        elif len(self.cur_state.band_path) >= 9:
            reward = -1

        def reward_shaping(s: CrossChangeState):
            # Reward shaping potential function. Currently,
            # we shape reward to encourage increasing the
            # number of connected components and reducing
            # the number of crossings, as this is a good
            # proxy for how close we are to a completely
            # split link.
            return ((s.num_conn_comp +
                     s.link_unlinked_unknot_components()) * 20 -
                    s.simplified_link_num_crossings() *
                    reward_per_crossing_reduction(len(self.link_history[0].crossings)))
        return (reward
                + self.gamma*reward_shaping(next_state)
                - reward_shaping(state))


class WeakSplittingEnv(CrossChangeEnv):
    """
    The knot environment calculating the weak splitting
    number.
    """
    def __init__(
            self,
            dataset: List[Link],
            shape: Tuple[int, int] = (50, 4),
            max_twists: int = 5,
            gamma: float = 0.99,
            features_cls: List[str] = None,
            sampling_method: str = 'consecutive',
            verbose: int = 0,
            simplification_switch: bool = True,
            mdp: bool = False,
            old_reward: bool = True
    ):
        super().__init__(dataset, shape, max_twists, gamma,
                         features_cls, sampling_method,
                         verbose, 'weak splitting',
                         simplification_switch, mdp=mdp, old_reward=old_reward)

    def get_reward(
            self,
            state: ErrorState | CrossChangeState,
            action: str,
            next_state: ErrorState | CrossChangeState
    ) -> float:
        """
        Calculates the reward corresponding to (state,
        actions, next_state) as follows:
        - If we reach a completely split link, return lots
          of positive reward
        - If we reach an error state, return lots of
          negative reward
        - If we do a crossing change (i.e., execute an "end"
          action), we return a medium amount of negative
          reward.
        - If the band path has length at least 9, we return
          a small amount of negative reward, otherwise
          return a small amount of positive reward.

        We also use potential shaping to improve the
        training dynamics, where the potential function is
        defined in reward_shaping.

        KnotEnv.step uses it when self.old_reward is True,
        and self._reward otherwise.

        Args:
            state: current state
            action: the action we took
            next_state: the next state after action

        Returns:
            Reward received
        """
        if isinstance(next_state, ErrorState):
            return -100
        reward = 1
        if next_state.is_terminal():
            # We reached a completely split link
            reward = 200
        elif band_get_act_type(action) == 'end':
            # We made a crossing change
            reward = -20
        elif len(self.cur_state.band_path) >= 9:
            reward = -1

        def reward_shaping(s: CrossChangeState):
            # Reward shaping potential function. Currently,
            # we shape reward to encourage increasing the
            # number of connected components and reducing
            # the number of crossings, as this is a good
            # proxy for how close we are to a completely
            # split link.
            return ((s.num_conn_comp +
                     s.link_unlinked_unknot_components()) * 20 -
                    s.simplified_link_num_crossings() *
                    reward_per_crossing_reduction(len(self.link_history[0].crossings)))
        return (reward
                + self.gamma*reward_shaping(next_state)
                - reward_shaping(state))


class UnknottingAlgEnv(CrossChangeEnv):
    """
    The knot environment calculating the algebraic
    unknotting number.
    """
    def __init__(
            self,
            dataset: List[Link],
            shape: Tuple[int, int] = (50, 4),
            max_twists: int = 5,
            gamma: float = 0.99,
            features_cls: List[str] = None,
            sampling_method: str = 'consecutive',
            verbose: int = 0,
            simplification_switch: bool = True,
            mdp: bool = False,
            old_reward: bool = True
    ):
        super().__init__(dataset, shape, max_twists, gamma,
                         features_cls, sampling_method,
                         verbose, 'unknotting-alg',
                         simplification_switch, mdp=mdp, old_reward=old_reward)

    def get_reward(
            self,
            state: ErrorState | CrossChangeState,
            action: str,
            next_state: ErrorState | CrossChangeState
    ) -> float:
        """
        Calculates the reward corresponding to (state,
        actions, next_state) as follows:
        - If we reach the unknot, return lots of positive
          reward
        - If we reach an error state, return lots of
          negative reward
        - If we do a crossing change (i.e., execute an "end"
          action), we return a medium amount of negative
          reward.
        - If the band path has length at least 9, we return
          a small amount of negative reward, otherwise
          return a small amount of positive reward.

        We also use potential shaping to improve the
        training dynamics, where the potential function is
        defined in reward_shaping.

        KnotEnv.step uses it when self.old_reward is True,
        and self._reward otherwise.

        Args:
            state: current state
            action: the action we took
            next_state: the next state after action

        Returns:
            Reward received
        """
        if isinstance(next_state, ErrorState):
            return -100
        reward = 1
        if next_state.is_terminal():
            # We reached a knot with Alexander polynomial 1
            reward = 200
        elif band_get_act_type(action) == 'end':
            # We make a crossing change
            reward = -20
        elif len(self.cur_state.band_path) >= 9:
            reward = -1

        def reward_shaping(s: CrossChangeState):
            # Reward shaping potential function. Currently,
            # we shape reward to encourage reducing the
            # number of crossings, as this is a good proxy
            # for how close we are to the unknot.
            return -(s.simplified_link_num_crossings() *
                     reward_per_crossing_reduction(len(self.link_history[0].crossings)))
        return (reward
                + self.gamma*reward_shaping(next_state)
                - reward_shaping(state))


class BandMoveEnv(KnotEnv[BandMoveState], abc.ABC):
    """
    The knot environment with band move as actions.
    The answer of these frameworks, the quantity self._reward
    charges, is the genus of the surface built so far.
    """
    # Whether self._reward charges the answer once, on the
    # step that wins the episode, instead of at every band
    # move that raises it. The frameworks whose surface must
    # be disconnected set it, for the reason at the end of
    # the _reward docstring.
    answer_charged_at_end = False
    # Weights of self._reward, in the units of its immediate
    # term. MAX_ACTION_COST is the cost of the most expensive
    # action that does not win, ANSWER_CAP the largest answer
    # the reward tells apart, and the last two follow from
    # them, so that the ordering of the _reward docstring
    # holds. They belong to the band move frameworks alone:
    # CrossChangeEnv carries its own.
    MAX_ACTION_COST = 5
    ANSWER_CAP = REWARD_ANSWER_CAP
    ANSWER_WEIGHT = MAX_ACTION_COST * REWARD_MAX_ACTIONS
    WIN_REWARD = ANSWER_WEIGHT * (ANSWER_CAP + 1)
    # multiplies the reward, to keep the returns small
    # enough for the value network to fit
    SCALE = REWARD_SCALE

    def __init__(
            self,
            dataset: List[Link],
            shape: Tuple[int, int],
            max_twists: int,
            gamma: float,
            features_cls: List[str],
            sampling_method: str,
            verbose: int,
            framework: str,
            simplification_switch: bool = True,
            mdp: bool = False,
            old_reward: bool = True
    ):
        super().__init__(dataset, shape, max_twists, gamma,
                         features_cls, sampling_method,
                         verbose, framework,
                         BandMoveState,
                         simplification_switch, mdp=mdp, old_reward=old_reward)

    def _set_action_labels(self) -> None:
        for c in range(self.shape[0]):
            for i in range(4):
                # end the band by attaching it to the final
                # strand with crossing c, strand index i
                # and without adding any twist
                self.action_labels.append(f'C{c}S{i}_end0')

    def _reward(
            self,
            state: ErrorState | BandMoveState,
            action: str,
            next_state: ErrorState | BandMoveState
    ) -> float:
        """
        Calculates the reward when self.old_reward is False.
        It is SCALE times

            r - ANSWER_WEIGHT * (g' - g) + gamma * P' - P,

        where g, g' are the genus of the surface before and
        after the action, r is the immediate term (WIN_REWARD
        on the winning step, otherwise minus the cost of the
        action, between 1 and MAX_ACTION_COST) and P is minus
        the crossing number of the simplified diagram,
        weighted, and 0 at a terminal state. When
        self.answer_charged_at_end is set, the genus term is
        instead -ANSWER_WEIGHT * min(g', ANSWER_CAP) on the
        winning step and 0 elsewhere, so that an episode
        which does not finish pays nothing for the genus it
        built. An action that leads to an error state returns
        -WIN_REWARD, with no potential term.

        With gamma = 1 and episodes of at most
        REWARD_MAX_ACTIONS actions, this orders the episodes
        of a link as: won, then stopped by the time limit,
        then ended on an error; and among won episodes, a
        lower genus first, then a shorter path. Charged as
        the surface is made, a stopped episode has already
        paid for the genus it built, so it beats an error
        only up to a genus of 1. That is why the frameworks
        whose surface must be disconnected charge at the end:
        a band of theirs can reach a state with no legal
        move, which ends the episode on an error, so giving
        up would pay. Ribbon and slice always have a legal
        move (KnotEnv.action_masks), and reach an error only
        when an action produces a diagram that no longer fits
        self.shape.

        Args:
            state: current state
            action: the action we took
            next_state: the next state after action

        Returns:
            Reward received.
        """
        if isinstance(next_state, ErrorState):
            return self.SCALE * -self.WIN_REWARD

        crossing_weight = reward_per_crossing_reduction(
            len(self.link_history[0].crossings))

        def potential(s: BandMoveState) -> float:
            # we shape reward to encourage reducing the
            # number of crossings, as this is a good proxy
            # for how close we are to the unlink
            return -crossing_weight * s.simplified_link_num_crossings()

        # the potential is 0 at a terminal state, so that
        # over a won episode the potential terms sum to a
        # constant of the link; the unlink that wins an
        # episode has no crossings anyway
        next_potential = (0.0 if next_state.is_terminal()
                          else potential(next_state))
        shaping = self.gamma * next_potential - potential(state)

        action_type = band_get_act_type(action)
        cur_genus = state.compute_current_answer(
            self.link_history[0],
            self.num_end_actions() - int(action_type == 'end'))
        next_genus = next_state.compute_current_answer(
            self.link_history[0], self.num_end_actions())
        if next_state.is_terminal():
            reward = self.WIN_REWARD
        elif action_type == 'birth':
            reward = -1
        elif action_type == 'end':
            reward = -self.MAX_ACTION_COST
        else:
            band_length = len(state.band_path)
            reward = -min(max(band_length - 7, 1), self.MAX_ACTION_COST)
        if self.answer_charged_at_end:
            genus_cost = (self.ANSWER_WEIGHT * min(next_genus, self.ANSWER_CAP)
                          if next_state.is_terminal() else 0)
        else:
            genus_cost = self.ANSWER_WEIGHT * (next_genus - cur_genus)
        return self.SCALE * (reward - genus_cost + shaping)


class RibbonEnv(BandMoveEnv):
    """
    The knot environment calculating the ribbon genus
    (band moves only, no unknot births).
    """
    def __init__(
            self,
            dataset: List[Link],
            shape: Tuple[int, int] = (50, 8),
            max_twists: int = 5,
            gamma: float = 0.99,
            features_cls: List[str] = None,
            sampling_method: str = 'consecutive',
            verbose: int = 0,
            simplification_switch: bool = True,
            framework: str = 'ribbon',
            mdp: bool = False,
            old_reward: bool = True
    ):
        super().__init__(dataset, shape, max_twists, gamma,
                         features_cls, sampling_method,
                         verbose, framework,
                         simplification_switch, mdp=mdp, old_reward=old_reward)

    def get_reward(
            self,
            state: ErrorState | BandMoveState,
            action: str,
            next_state: ErrorState | BandMoveState
    ) -> float:
        """
        Calculates the reward corresponding to (state,
        actions, next_state) as follows:
        - If we reach the unlink, return lots of positive
          reward
        - If we reach an error state, return lots of
          negative reward
        - If we do a band move (i.e., execute an "end"
          action), we return a medium amount of negative
          reward if the 4-genus increased, otherwise we
          return a small amount of negative reward.
          If the current 4-genus exceeds the 3-genus of the
          initial link, we add some negative reward to the
          previous value.
        - If the band path has length at least 9, we return
          a small amount of negative reward, otherwise
          return a small amount of positive reward.

        We also use potential shaping to improve the
        training dynamics, where the potential function is
        defined in reward_shaping.

        KnotEnv.step uses it when self.old_reward is True,
        and self._reward otherwise.

        Args:
            state: current state
            action: the action we took
            next_state: the next state after action

        Returns:
            Reward received
        """
        if isinstance(next_state, ErrorState):
            return -100
        cur_genus = state.compute_current_answer(
            self.link_history[0], self.num_end_actions() - int(band_get_act_type(action) == "end")
        )
        next_genus = next_state.compute_current_answer(
            self.link_history[0], self.num_end_actions()
        )
        reward = 1
        if next_state.is_terminal():
            # We reached the unlink
            reward = 200
        elif band_get_act_type(action) == 'end':
            # We made a band move
            reward = -5
            # We also take into account the 3-genus of the
            # initial link, which is an upper bound of the
            # 4-genus.
            # If the 4-genus exceeds the 3-genus, we give
            # some negative reward
            if hfk_enabled and cur_genus <= self.seifert_genus < next_genus:
                reward -= 50
        elif len(self.cur_state.band_path) >= 9:
            reward = -1

        def reward_shaping(s: BandMoveState, genus: int):
            # Reward shaping potential function. Currently,
            # we shape reward to encourage reducing the
            # number of crossings, as this is a good proxy
            # for how close we are to the unlink. Also, the
            # current 4-genus contributes to the reward
            # through this function.
            return (-s.simplified_link_num_crossings() *
                    reward_per_crossing_reduction(len(self.link_history[0].crossings)) -
                    genus * 20)
        return (reward
                + self.gamma*reward_shaping(next_state, next_genus)
                - reward_shaping(state, cur_genus))


class StrongRibbonEnv(BandMoveEnv):
    """
    The knot environment calculating the strong ribbon genus.

    Unlike RibbonEnv, band moves are only allowed between
    components with the same connected component label (ccl).
    This constraint means that each original link component
    bounds its own surface, resulting in a potentially
    disconnected ribbon surface.
    """
    answer_charged_at_end = True

    def __init__(
            self,
            dataset: List[Link],
            shape: Tuple[int, int] = (50, 8),
            max_twists: int = 5,
            gamma: float = 0.99,
            features_cls: List[str] = None,
            sampling_method: str = 'consecutive',
            verbose: int = 0,
            simplification_switch: bool = True,
            framework: str = 'strong ribbon',
            mdp: bool = False,
            old_reward: bool = True
    ):
        super().__init__(dataset, shape, max_twists, gamma,
                 features_cls, sampling_method,
                 verbose, framework,
                 simplification_switch, mdp=mdp, old_reward=old_reward)

    def reset(self,
              seed: int | None = None,
              options: dict[str, Any] | None = None
              ) -> Tuple[ObsType, Dict[str, Any]]:
        """
        Resets the environment, with an additional check that the
        link has finite strong ribbon genus (i.e., components with
        different CCLs are not linked to each other). The same
        criterion applies to the strong slice genus, so
        StrongSliceEnv inherits this check.

        At initialization, each link component has its own CCL,
        so we check that all off-diagonal entries of the standard
        linking matrix are zero.

        Raises:
            ValueError: if the link has non-zero linking numbers
                between different components (infinite strong
                ribbon / strong slice genus)
        """
        obs, info = super().reset(seed, options)

        # Check linking matrix: all off-diagonal entries must be zero
        link = self.link_history[0]
        if len(link.link_components) > 1:
            lm = link.linking_matrix()
            for i in range(len(lm)):
                for j in range(i+1, len(lm)):
                    if lm[i][j] != 0:
                        raise ValueError(
                            f"Link at index {self.index} has non-zero "
                            f"linking number {lm[i][j]} between components "
                            f"{i} and {j}. This link has infinite "
                            f"{self.framework} genus and cannot be used "
                            f"with {type(self).__name__}."
                        )

        return obs, info

    def get_reward(
            self,
            state: ErrorState | BandMoveState,
            action: str,
            next_state: ErrorState | BandMoveState
    ) -> float:
        """
        Calculates the reward corresponding to (state,
        actions, next_state) as follows:
        - If we reach the unlink, return lots of positive
          reward
        - If we reach an error state, return lots of
          negative reward
        - If we do a band move (i.e., execute an "end"
          action), we return a medium amount of negative
          reward if the 4-genus increased, otherwise we
          return a small amount of negative reward.
          If the current 4-genus exceeds the 3-genus of the
          initial link, we add some negative reward to the
          previous value.
        - If the band path has length at least 9, we return
          a small amount of negative reward, otherwise
          return a small amount of positive reward.

        We also use potential shaping to improve the
        training dynamics, where the potential function is
        defined in reward_shaping.

        KnotEnv.step uses it when self.old_reward is True,
        and self._reward otherwise.

        Args:
            state: current state
            action: the action we took
            next_state: the next state after action

        Returns:
            Reward received
        """
        if isinstance(next_state, ErrorState):
            return -100
        cur_genus = state.compute_current_answer(
            self.link_history[0], self.num_end_actions() - int(band_get_act_type(action) == "end")
        )
        next_genus = next_state.compute_current_answer(
            self.link_history[0], self.num_end_actions()
        )
        reward = 1
        if next_state.is_terminal():
            # We reached the unlink
            reward = 200
        elif band_get_act_type(action) == 'end':
            # We made a band move
            reward = -5
            # We also take into account the 3-genus of the
            # initial link, which is an upper bound of the
            # 4-genus.
            # If the 4-genus exceeds the 3-genus, we give
            # some negative reward
            if hfk_enabled and cur_genus <= self.seifert_genus < next_genus:
                reward -= 50
        elif len(self.cur_state.band_path) >= 9:
            reward = -1

        def reward_shaping(s: BandMoveState, genus: int):
            # Reward shaping potential function. Currently,
            # we shape reward to encourage reducing the
            # number of crossings, as this is a good proxy
            # for how close we are to the unlink. Also, the
            # current 4-genus contributes to the reward
            # through this function.
            return (-s.simplified_link_num_crossings() *
                    reward_per_crossing_reduction(len(self.link_history[0].crossings)) -
                    genus * 20)
        return (reward
                + self.gamma*reward_shaping(next_state, next_genus)
                - reward_shaping(state, cur_genus))


class SliceEnv(RibbonEnv):
    """
    The knot environment calculating the slice genus: the
    ribbon framework plus the unknot_birth action (the birth
    of a new unlinked unknot component, i.e. a local minimum
    of the slice surface). The birth label is appended as the
    LAST action index, matching the mask layout produced by
    representable.bandmovestate_action_masks.

    Inherits __init__ from RibbonEnv (called with
    framework='slice').
    """
    # Flat reward for a birth action.
    BIRTH_REWARD = 2.0

    def __init__(
            self,
            dataset: List[Link],
            shape: Tuple[int, int] = (50, 8),
            max_twists: int = 5,
            gamma: float = 0.99,
            features_cls: List[str] = None,
            sampling_method: str = 'consecutive',
            verbose: int = 0,
            simplification_switch: bool = True,
            mdp: bool = False,
            old_reward: bool = True
    ):
        super().__init__(dataset, shape, max_twists, gamma,
                         features_cls, sampling_method,
                         verbose, simplification_switch,
                         framework='slice', mdp=mdp, old_reward=old_reward)

    def _set_action_labels(self) -> None:
        # the ribbon end0 block, plus unknot_birth as the last
        # action index (matches bandmovestate_action_masks)
        super()._set_action_labels()
        self.action_labels.append('unknot_birth')

    def num_end_actions(self) -> int:
        # link_history also records the links generated by
        # birth actions, so the parent's len(link_history) - 1
        # would over-count; count the true end actions instead.
        # (BandMove.compute_current_answer ignores this value;
        # it only affects the terminal message.)
        return sum(1 for lbl, _ in self.actions_rewards
                   if band_get_act_type(lbl) == 'end')

    def get_reward(
            self,
            state: ErrorState | BandMoveState,
            action: str,
            next_state: ErrorState | BandMoveState
    ) -> float:
        """
        Same shaping as RibbonEnv.get_reward, plus a flat
        BIRTH_REWARD for the unknot_birth action:
        - If we reach the unlink, return lots of positive
          reward
        - If we reach an error state, return lots of
          negative reward
        - If we birth an unknot, return BIRTH_REWARD
        - If we do a band move (i.e., execute an "end"
          action), we return a medium amount of negative
          reward if the 4-genus increased, otherwise we
          return a small amount of negative reward.
          If the current 4-genus exceeds the 3-genus of the
          initial link, we add some negative reward to the
          previous value.
        - If the band path has length at least 9, we return
          a small amount of negative reward, otherwise
          return a small amount of positive reward.

        We also use potential shaping to improve the
        training dynamics, where the potential function is
        defined in reward_shaping.

        KnotEnv.step uses it when self.old_reward is True,
        and self._reward otherwise.

        Args:
            state: current state
            action: the action we took
            next_state: the next state after action

        Returns:
            Reward received
        """
        if isinstance(next_state, ErrorState):
            return -100
        cur_genus = state.compute_current_answer(
            self.link_history[0], self.num_end_actions() - int(band_get_act_type(action) == "end")
        )
        next_genus = next_state.compute_current_answer(
            self.link_history[0], self.num_end_actions()
        )
        reward = 1
        if next_state.is_terminal():
            # We reached the unlink
            reward = 200
        elif band_get_act_type(action) == 'birth':
            # We birthed an unlinked unknot component
            reward = self.BIRTH_REWARD
        elif band_get_act_type(action) == 'end':
            # We made a band move
            reward = -5
            # We also take into account the 3-genus of the
            # initial link, which is an upper bound of the
            # 4-genus.
            # If the 4-genus exceeds the 3-genus, we give
            # some negative reward
            if hfk_enabled and cur_genus <= self.seifert_genus < next_genus:
                reward -= 50
        elif len(self.cur_state.band_path) >= 9:
            reward = -1

        def reward_shaping(s: BandMoveState, genus: int):
            # Reward shaping potential function. Currently,
            # we shape reward to encourage reducing the
            # number of crossings, as this is a good proxy
            # for how close we are to the unlink. Also, the
            # current 4-genus contributes to the reward
            # through this function.
            return (-s.simplified_link_num_crossings() *
                    reward_per_crossing_reduction(len(self.link_history[0].crossings)) -
                    genus * 20)
        return (reward
                + self.gamma*reward_shaping(next_state, next_genus)
                - reward_shaping(state, cur_genus))


class StrongSliceEnv(StrongRibbonEnv):
    """
    The knot environment calculating the strong slice genus:
    the strong ribbon framework plus the unknot_birth action.
    Band moves across different CCLs are only legal when one
    of the two surface components contains no original link
    component (a pure-birth comp); the linking-matrix validity
    check is inherited from StrongRibbonEnv.reset().

    Inherits __init__ and reset from StrongRibbonEnv (called
    with framework='strong slice').
    """
    # inherited from StrongRibbonEnv, repeated because the
    # reward of this framework depends on it
    answer_charged_at_end = True
    # Flat reward for a birth action
    BIRTH_REWARD = 2.0

    def __init__(
            self,
            dataset: List[Link],
            shape: Tuple[int, int] = (50, 8),
            max_twists: int = 5,
            gamma: float = 0.99,
            features_cls: List[str] = None,
            sampling_method: str = 'consecutive',
            verbose: int = 0,
            simplification_switch: bool = True,
            mdp: bool = False,
            old_reward: bool = True
    ):
        super().__init__(dataset, shape, max_twists, gamma,
                         features_cls, sampling_method,
                         verbose, simplification_switch,
                         framework='strong slice', mdp=mdp, old_reward=old_reward)

    def _set_action_labels(self) -> None:
        # the ribbon end0 block, plus unknot_birth as the last
        # action index (matches bandmovestate_action_masks)
        super()._set_action_labels()
        self.action_labels.append('unknot_birth')

    def num_end_actions(self) -> int:
        # link_history also records the links generated by
        # birth actions, so the parent's len(link_history) - 1
        # would over-count; count the true end actions instead.
        # (BandMove.compute_current_answer ignores this value;
        # it only affects the terminal message.)
        return sum(1 for lbl, _ in self.actions_rewards
                   if band_get_act_type(lbl) == 'end')

    def get_reward(
            self,
            state: ErrorState | BandMoveState,
            action: str,
            next_state: ErrorState | BandMoveState
    ) -> float:
        """
        Same shaping as StrongRibbonEnv.get_reward, plus a
        flat BIRTH_REWARD for the unknot_birth action:
        - If we reach the unlink, return lots of positive
          reward
        - If we reach an error state, return lots of
          negative reward
        - If we birth an unknot, return BIRTH_REWARD
        - If we do a band move (i.e., execute an "end"
          action), we return a medium amount of negative
          reward if the 4-genus increased, otherwise we
          return a small amount of negative reward.
          If the current 4-genus exceeds the 3-genus of the
          initial link, we add some negative reward to the
          previous value.
        - If the band path has length at least 9, we return
          a small amount of negative reward, otherwise
          return a small amount of positive reward.

        We also use potential shaping to improve the
        training dynamics, where the potential function is
        defined in reward_shaping.

        KnotEnv.step uses it when self.old_reward is True,
        and self._reward otherwise.

        Args:
            state: current state
            action: the action we took
            next_state: the next state after action

        Returns:
            Reward received
        """
        if isinstance(next_state, ErrorState):
            return -100
        cur_genus = state.compute_current_answer(
            self.link_history[0], self.num_end_actions() - int(band_get_act_type(action) == "end")
        )
        next_genus = next_state.compute_current_answer(
            self.link_history[0], self.num_end_actions()
        )
        reward = 1
        if next_state.is_terminal():
            # We reached the unlink
            reward = 200
        elif band_get_act_type(action) == 'birth':
            # We birthed an unlinked unknot component
            reward = self.BIRTH_REWARD
        elif band_get_act_type(action) == 'end':
            # We made a band move
            reward = -5
            # We also take into account the 3-genus of the
            # initial link, which is an upper bound of the
            # 4-genus.
            # If the 4-genus exceeds the 3-genus, we give
            # some negative reward
            if hfk_enabled and cur_genus <= self.seifert_genus < next_genus:
                reward -= 50
        elif len(self.cur_state.band_path) >= 9:
            reward = -1

        def reward_shaping(s: BandMoveState, genus: int):
            # Reward shaping potential function. Currently,
            # we shape reward to encourage reducing the
            # number of crossings, as this is a good proxy
            # for how close we are to the unlink. Also, the
            # current 4-genus contributes to the reward
            # through this function.
            return (-s.simplified_link_num_crossings() *
                    reward_per_crossing_reduction(len(self.link_history[0].crossings)) -
                    genus * 20)
        return (reward
                + self.gamma*reward_shaping(next_state, next_genus)
                - reward_shaping(state, cur_genus))
