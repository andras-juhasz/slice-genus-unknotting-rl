import abc
from typing import Dict, Tuple, List

import numpy as np
from spherogram import Link

from representable import (crosschangestate_from_link,
                           bandmovestate_from_link,
                           crosschangestate_check_action_error,
                           bandmovestate_check_action_error,
                           crosschangestate_next_state,
                           bandmovestate_next_state,
                           crosschangestate_to_list,
                           bandmovestate_to_list,
                           crosschangestate_mdp_to_list,
                           bandmovestate_mdp_to_list,
                           crosschangestate_is_terminal,
                           bandmovestate_is_terminal,
                           bandmovestate_has_unfused_birth_components,
                           crosschangestate_link,
                           bandmovestate_link,
                           crosschangestate_band_path,
                           bandmovestate_band_path,
                           crosschangestate_num_conn_comp,
                           bandmovestate_num_conn_comp,
                           crosschangestate_compute_current_answer,
                           bandmovestate_compute_current_answer,
                           crosschangestate_is_unlink,
                           bandmovestate_is_unlink,
                           crosschangestate_terminal_state_message,
                           bandmovestate_terminal_state_message,
                           crosschangestate_random_walk_transition,
                           bandmovestate_random_walk_transition,
                           crosschangestate_to_string,
                           bandmovestate_to_string,
                           crosschangestate_max_twists,
                           bandmovestate_max_twists,
                           crosschangestate_action_masks,
                           bandmovestate_action_masks,
                           crosschangestate_link_num_crossings,
                           bandmovestate_link_num_crossings,
                           crosschangestate_simplified_link_num_crossings,
                           bandmovestate_simplified_link_num_crossings,
                           crosschangestate_link_unlinked_unknot_components,
                           bandmovestate_link_unlinked_unknot_components)
from utils import encode_link, decode_link


class State(abc.ABC):
    """
    Generic state class
    """

    @abc.abstractmethod
    def to_tensor(self) -> Dict[str, np.ndarray]:
        """
        Represents self in a dictionary of tensors.

        Returns:
            A dictionary of tensors (np.ndarray)
            representing self.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def mdp_to_tensor(self) -> Dict[str, np.ndarray]:
        """
        Represents the part of self that to_tensor leaves
        out although the legal actions, the next state or
        the reward depend on it: the twists of every band
        segment, the crossing number of the simplified link
        and, for band moves, the connected component labels
        of the link components, the genera_info table and
        the current genus.

        Returns:
            A dictionary of tensors (np.ndarray).
        """
        raise NotImplementedError


class ErrorState(State):
    """
    State class resulting from an agent making an illegal
    move or making a move resulting in a state whose tensor
    representation cannot be calculated because the state's
    link has more crossings or components than
    env.KnotEnv.shape allows.
    """

    def __init__(
            self,
            prev_state: State,
            label: str
    ) -> None:
        """
        Initializes an error state instance using the
        previous state and the illegal action.

        Args:
            prev_state: the state on which the agent made an
                illegal move
            label: the illegal move that the agent made
        """
        self.prev_state = prev_state
        self.label = label

    def to_tensor(self) -> Dict[str, np.ndarray]:
        # Returns the previous state's tensor
        # representation, since this seems the most natural
        # choice.
        return self.prev_state.to_tensor()

    def mdp_to_tensor(self) -> Dict[str, np.ndarray]:
        # Consistent with to_tensor.
        return self.prev_state.mdp_to_tensor()

    def __repr__(self) -> str:
        return (f'ErrorState instance with\n'
                f'previous state =\n'
                f'{self.prev_state}'
                f'action = {self.label}\n')

    def __str__(self) -> str:
        return self.__repr__()


class DimensionExceededError(ValueError):
    """
    The error raised when the feature values of a link
    cannot be computed because the link has too many
    crossings or components.
    """
    pass


class CrossChangeState(State):
    beaten_by_offset = 1

    def __init__(self, state):
        self.state = state

    @staticmethod
    def from_link(
            link: Link,
            max_twists: int = 5,
            framework: str = 'unknotting',
            shape: Tuple[int, int] = (50, 4),
            features_cls: List[str] = None,
            simplification_switch: bool = True
    ) -> 'CrossChangeState':
        if features_cls is None:
            features_cls = []
        nim_state = crosschangestate_from_link(
            encode_link(link), max_twists, framework,
            shape, features_cls, simplification_switch
        )
        return CrossChangeState(nim_state)

    def check_action_error(self, action: str) -> bool:
        return crosschangestate_check_action_error(self.state, action)

    def to_tensor(self) -> Dict[str, np.ndarray]:
        return dict(crosschangestate_to_list(self.state))

    def mdp_to_tensor(self) -> Dict[str, np.ndarray]:
        return dict(crosschangestate_mdp_to_list(self.state))

    def next_state(self, action: str) -> 'CrossChangeState':
        (msg, new_state) = crosschangestate_next_state(self.state, action)
        if msg != '':
            raise DimensionExceededError(msg)
        return CrossChangeState(new_state)

    def is_terminal(self) -> bool:
        return crosschangestate_is_terminal(self.state)

    def link(self) -> Link:
        return decode_link(crosschangestate_link(self.state))

    @property
    def band_path(self) -> List[List[int]]:
        return crosschangestate_band_path(self.state)

    @property
    def num_conn_comp(self) -> int:
        return crosschangestate_num_conn_comp(self.state)

    def compute_current_answer(self,
                               init_link: Link,
                               num_end_actions: int) -> int:
        return crosschangestate_compute_current_answer(
            self.state, encode_link(init_link), num_end_actions
        )

    def is_unlink(self) -> bool:
        return crosschangestate_is_unlink(self.state)

    def terminal_state_message(self,
                               num_end_actions: int,
                               answer: int) -> str:
        return crosschangestate_terminal_state_message(
            self.state, num_end_actions, answer
        )

    def random_walk_transition(self,
                               p_twist: float,
                               p_end: float) -> str | None:
        (has, action) = crosschangestate_random_walk_transition(self.state, p_twist, p_end)
        if not has:
            return None
        return action

    def __str__(self) -> str:
        return crosschangestate_to_string(self.state)

    @property
    def max_twists(self) -> int:
        return crosschangestate_max_twists(self.state)

    def action_masks(self) -> List[bool]:
        return crosschangestate_action_masks(self.state)

    def link_num_crossings(self) -> int:
        return crosschangestate_link_num_crossings(self.state)

    def simplified_link_num_crossings(self) -> int:
        return crosschangestate_simplified_link_num_crossings(self.state)

    def link_unlinked_unknot_components(self) -> int:
        return crosschangestate_link_unlinked_unknot_components(self.state)


class BandMoveState(State):
    beaten_by_offset = 0

    def __init__(self, state):
        self.state = state

    @staticmethod
    def from_link(
            link: Link,
            max_twists: int = 5,
            framework: str = 'unknotting',
            shape: Tuple[int, int] = (50, 4),
            features_cls: List[str] = None,
            simplification_switch: bool = True
    ) -> 'BandMoveState':
        if features_cls is None:
            features_cls = []
        nim_state = bandmovestate_from_link(
            encode_link(link), max_twists, framework,
            shape, features_cls, simplification_switch
        )
        return BandMoveState(nim_state)

    def check_action_error(self, action: str) -> bool:
        return bandmovestate_check_action_error(self.state, action)

    def to_tensor(self) -> Dict[str, np.ndarray]:
        return dict(bandmovestate_to_list(self.state))

    def mdp_to_tensor(self) -> Dict[str, np.ndarray]:
        return dict(bandmovestate_mdp_to_list(self.state))

    def next_state(self, action: str) -> 'BandMoveState':
        (msg, new_state) = bandmovestate_next_state(self.state, action)
        if msg != '':
            raise DimensionExceededError(msg)
        return BandMoveState(new_state)

    def is_terminal(self) -> bool:
        return bandmovestate_is_terminal(self.state)

    def has_unfused_birth_components(self) -> bool:
        """
        Whether the surface has a birth component (a CCL with
        num_orig_comps == 0) that was never fused with an
        original link component. Only meaningful for the slice
        frameworks; always False for ribbon / strong ribbon.
        """
        return bandmovestate_has_unfused_birth_components(self.state)

    def link(self) -> Link:
        return decode_link(bandmovestate_link(self.state))

    @property
    def band_path(self) -> List[List[int]]:
        return bandmovestate_band_path(self.state)

    @property
    def num_conn_comp(self) -> int:
        return bandmovestate_num_conn_comp(self.state)

    def compute_current_answer(self,
                               init_link: Link,
                               num_end_actions: int) -> int:
        return bandmovestate_compute_current_answer(
            self.state, encode_link(init_link), num_end_actions
        )

    def is_unlink(self) -> bool:
        return bandmovestate_is_unlink(self.state)

    def terminal_state_message(self,
                               num_end_actions: int,
                               answer: int) -> str:
        return bandmovestate_terminal_state_message(
            self.state, num_end_actions, answer
        )

    def random_walk_transition(self,
                               p_twist: float,
                               p_end: float) -> str | None:
        (has, action) = bandmovestate_random_walk_transition(self.state, p_twist, p_end)
        if not has:
            return None
        return action

    def __str__(self) -> str:
        return bandmovestate_to_string(self.state)

    @property
    def max_twists(self) -> int:
        return bandmovestate_max_twists(self.state)

    def action_masks(self) -> List[bool]:
        return bandmovestate_action_masks(self.state)

    def link_num_crossings(self) -> int:
        return bandmovestate_link_num_crossings(self.state)

    def simplified_link_num_crossings(self) -> int:
        return bandmovestate_simplified_link_num_crossings(self.state)

    def link_unlinked_unknot_components(self) -> int:
        return bandmovestate_link_unlinked_unknot_components(self.state)
