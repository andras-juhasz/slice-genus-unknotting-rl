import std/strformat
import std/tables
import std/random
import std/syncio
import std/options
import std/sequtils
import std/sugar
import std/json
import std/marshal
import std/streams
import std/logging

import "../spherogram-nim/src/links"
import nimpy

import state
import band
import utils

discard """
Parameters for the random walk.
"""
type RandomWalkParam* = object of RootObj
    p_twist*: float
    p_end*: float
    max_twists*: int
    max_actions*: int
    sampling_method*: string
    p_birth*: float

proc newRandomWalkParam*(p_twist: float,
                         p_end: float,
                         max_twists: int,
                         max_actions: int,
                         sampling_method: string = "consecutive",
                         p_birth: float = 0.0): RandomWalkParam =
    discard """
    Initializes an instance of RandomWalkParam with
    all the parameters filled in. The start, over, under
    probabilities are always set to 1.0.

    Args:
        p_twist: the twist probability
        p_end: the end probability
        max_twists: the maximum number of total twists
            allowed across all the faces
        max_actions: the maximum number of actions per
            episode
        sampling_method: the sampling method; can be
            either "consecutive" or "random"
        p_birth: the probability of an unknot_birth action
            when no band is in progress; only meaningful
            for the "slice" and "strong slice" frameworks

    Raises:
        ValueError: if sampling_method is not
            "consecutive" or "random"
    """
    if sampling_method notin ["consecutive", "random"]:
        raise newException(ValueError, &"{sampling_method} is not a valid sampling method")
    return RandomWalkParam(p_twist: p_twist, p_end: p_end,
                           max_twists: max_twists,
                           max_actions: max_actions,
                           sampling_method: sampling_method,
                           p_birth: p_birth)

proc random_walk_param_default*(framework: string): RandomWalkParam =
    # Bayesian-optimization-tuned per-framework defaults
    # (p_twist, p_end, max_twists, max_actions[, p_birth]).
    return {"unknotting": newRandomWalkParam(0.0606, 90.4377, 1, 10),
            "ribbon": newRandomWalkParam(0.3508, 98.3104, 1, 20),
            "strong ribbon": newRandomWalkParam(0.0234, 90.0185, 1, 17),
            "slice": newRandomWalkParam(0.0119, 86.6341, 1, 21,
                                        p_birth = 0.0053),
            "strong slice": newRandomWalkParam(0.0222, 89.7971, 4, 17,
                                               p_birth = 0.0014)
           }.toTable()[framework]

proc band_operation_random_walk*(
    framework: static string,
    params: RandomWalkParam,
    link: Link[int],
    current_best: int = -1,
    verbose: bool = false
): (int, seq[string], seq[Link[int]], seq[Rand]) =
    discard """
    Given a link, calculates an upper bound of its link
    invariant by attempting to reduce it to a terminal link
    through a sequence of crossing changes or band moves,
    returning the answer (the number of crossing changes or
    the genus of the constructed ribbon / slice surface)
    along with the list of actions taken, the link
    trajectory and the states of std/random before each end
    action.

    Args:
        framework: the type of link invariant that we are
            calculating; "unknotting" means the unknotting
            number, "splitting" means the splitting number,
            "weak splitting" means the weak splitting
            number, "unknotting-alg" means the algebraic
            unknotting number, "ribbon" means the ribbon
            genus, "strong ribbon" means the strong ribbon
            genus (band moves only between components with
            the same ccl), and "slice" / "strong slice"
            are the same with the unknot_birth action
            additionally available
        link: the link whose link invariant we are
            calculating
        params: the parameters of the random walk
        current_best: if we are taking the minimum over
            multiple attempts, we can stop once we know that
            we cannot beat the given current best; if the
            provided value is -1, then we do not stop
            because of this reason
        verbose: whether we are outputting the actions taken
            by the random walk agent

    Returns:
        A 4-tuple, consisting of the following
        components:
        1. An upper bound for the link invariant. If the
           random walk succeeds in reducing the link to a
           terminal link, it is the number of crossing
           changes made (for cross change environments) 
           or the genus of the constructed surface (for 
           band move environments). Otherwise, it is -1.
        2. A list of actions that the random walk agent has
           taken, regardless of whether the random walk
           succeeded or not. The length of this list is used
           to determine the number of steps that the agent
           has consumed.
        3. The link trajectory, consisting of the initial
           link and every link generated by an end or birth
           action.
        4. The states of std/random just before
           Link.simplify, useful for replaying the link
           simplifications of the end actions when verifying
           the results obtained.
    """
    when framework in ["ribbon", "strong ribbon",
                       "slice", "strong slice"]:
        var state = bandmove_from_link(link,
                                       params.max_twists,
                                       framework)
        const beaten_by_offset = 0
    else:
        var state = crosschange_from_link(link,
                                          params.max_twists,
                                          framework)
        const beaten_by_offset = 1
    var actions = newSeq[string]()
    var link_history = @[state.link]
    var random_states = newSeq[Rand]()
    var cur_answer = state.compute_current_answer(link_history[0], 0)
    # trivial case: link is terminal
    if state.is_terminal():
        if verbose:
            echo state.terminal_state_message(0, cur_answer)
    # if the link is not terminal, we cannot possibly beat
    # the current best if it is 1 (for crossing changes) or
    # 0 (for band moves)
    if (current_best != -1 and cur_answer >= current_best -
            beaten_by_offset):
        if verbose:
            echo &"cannot beat current_best={current_best}"
        return (-1, actions, link_history, random_states)
    var num_end_actions = 0
    for _ in 0 ..< params.max_actions:
        var move0: Option[string]
        when framework in ["slice", "strong slice"]:
            # an unknot birth is only legal while no band 
            # is in progress
            if state.band_path.len == 0 and rand(1.0) < params.p_birth:
                move0 = some("unknot_birth")
            else:
                move0 = state.random_walk_transition(params.p_twist,
                                                     params.p_end)
        else:
            move0 = state.random_walk_transition(params.p_twist,
                                                 params.p_end)
        if move0.isNone:
            if verbose:
                echo "run out of legal moves"
            return (-1, actions, link_history, random_states)
        var move = move0.get()
        if get_act_type(move) == "end":
            random_states.add(randState())
        state = state.next_state(move)
        actions.add(move)
        if verbose:
            stdout.write(&"{move} ")
        if get_act_type(move) == "end":
            num_end_actions += 1
            link_history.add(state.link)
            cur_answer = state.compute_current_answer(link_history[0],
                                                      num_end_actions)
            # we have reached a terminal link
            if state.is_terminal():
                when framework in ["slice", "strong slice"]:
                    if state.has_unfused_birth_components():
                        warn(&"{framework} episode discarded pure-birth " &
                             "surface components from the terminal answer " &
                             "(CCLs with num_orig_comps == 0)")
                if verbose:
                    echo state.terminal_state_message(num_end_actions,
                                                      cur_answer)
                return (cur_answer, actions, link_history, random_states)
            # Pure-birth genus can disappear at termination, so the
            # current total is not a lower bound while it is present.
            var may_discard_birth_components = false
            when framework in ["slice", "strong slice"]:
                may_discard_birth_components = state.has_unfused_birth_components()
            # we are not going to beat the current best
            if (not may_discard_birth_components and
                    current_best != -1 and cur_answer >=
                    current_best - beaten_by_offset):
                if verbose:
                    echo &"cannot beat current_best={current_best}"
                return (-1, actions, link_history, random_states)
        elif get_act_type(move) == "birth":
            # the diagram changed (the birthed unknot is joined
            # into it), so record it in the trajectory; births
            # consume no Nim RNG inside next_state, hence no
            # random state is recorded
            link_history.add(state.link)
    # we have exceeded the maximum number of steps allowed
    if verbose:
        echo &"exceeded max_actions={params.max_actions}"
    return (-1, actions, link_history, random_states)

proc band_operations_random_walk*(
    framework: static string,
    params: RandomWalkParam,
    dataset: seq[Link[int]],
    total_steps: int,
    verbose: bool = false
): seq[(int, seq[string], seq[Link[int]], seq[Rand])] =
    discard """
    Given a list of links, calculates upper bounds of their
    link invariants by attempting the random walk over the
    dataset multiple times and taking the minimum number
    of crossing changes or genera of ribbon / slice surfaces
    for each link.

    Args:
        framework: the type of link invariant that we are
            calculating; for a detailed explanation, see the
            docstring of band_operation_random_walk
        params: the parameters of the random walk
        dataset: the list of links whose link invariants we
            are calculating
        total_steps: the total number of steps allowed,
            shared by all the links in the dataset
        verbose: whether we are outputting the resets of the
            current link and the actions taken by the random
            walk agent

    Returns:
        A list of 4-tuples for each link in dataset. Their
        meaning are explained in the docstring of
        band_operation_random_walk.
    """
    var sampling_method = params.sampling_method
    var answers = newSeqWith(dataset.len, -1)
    var actions_list = newSeq[seq[string]](dataset.len)
    var links_history = newSeq[seq[Link[int]]](dataset.len)
    var random_states_list = newSeq[seq[Rand]](dataset.len)
    var steps_remaining = total_steps
    var cur_link: int
    if sampling_method == "consecutive":
        cur_link = 0
    else:
        cur_link = rand(0 ..< dataset.len)
    if verbose:
        stdout.write(&"reset to link {cur_link}: ")
    while steps_remaining > 0:
        var (answer, actions, link_history, random_states) = band_operation_random_walk(
            framework, params, dataset[cur_link],
            answers[cur_link], verbose
        )
        if answer != -1 and (answers[cur_link] == -1 or answer < answers[cur_link]):
            answers[cur_link] = answer
            actions_list[cur_link] = actions
            links_history[cur_link] = link_history
            random_states_list[cur_link] = random_states
        steps_remaining -= actions.len + 1
        if sampling_method == "consecutive":
            cur_link = (cur_link+1) mod dataset.len
        else:
            cur_link = rand(0 ..< dataset.len)
        if verbose:
            stdout.write(&"reset to link {cur_link}: ")
    if verbose:
        echo ""
    return collect:
        for link_index in 0 ..< dataset.len:
            (answers[link_index], actions_list[link_index], links_history[link_index], random_states_list[link_index])

proc nim_randomize*(): void {.exportpy.} =
    randomize()

proc nim_set_seed*(seed: int): void {.exportpy.} =
    discard """
    Seeds the random number generator of this compiled
    module, which the random walk and the simplification
    of spherogram-nim draw from. agent.so and
    representable.so each hold their own generator, so a
    process using both seeds both.

    Args:
        seed: a positive integer; equal seeds give equal
            random streams

    Raises:
        ValueError: if seed is not positive
    """
    if seed <= 0:
        raise newException(ValueError, &"seed {seed} is not positive")
    randomize(seed)

proc band_operations_random_walk_interface*(
    framework: string,
    params_json: string,
    dataset_json: seq[string],
    total_steps: int,
    verbose: bool
): seq[(int, seq[string], seq[string], seq[string])] {.exportpy.} =
    # decode params from its json representation
    var params_node = parseJson(params_json)
    var params = newRandomWalkParam(params_node["p_twist"].getFloat(),
                                    params_node["p_end"].getFloat(),
                                    params_node["max_twists"].getInt(),
                                    params_node["max_actions"].getInt(),
                                    params_node["sampling_method"].getStr(),
                                    params_node{"p_birth"}.getFloat(0.0))
    var dataset = collect:
        for link_json in dataset_json:
            decode_link(link_json)
    var ans: seq[(int, seq[string], seq[Link[int]], seq[Rand])]
    if framework == "unknotting":
        ans = band_operations_random_walk("unknotting", params, dataset, total_steps, verbose)
    elif framework == "splitting":
        ans = band_operations_random_walk("splitting", params, dataset, total_steps, verbose)
    elif framework == "weak splitting":
        ans = band_operations_random_walk("weak splitting", params, dataset, total_steps, verbose)
    elif framework == "ribbon":
        ans = band_operations_random_walk("ribbon", params, dataset, total_steps, verbose)
    elif framework == "strong ribbon":
        ans = band_operations_random_walk("strong ribbon", params, dataset, total_steps, verbose)
    elif framework == "slice":
        ans = band_operations_random_walk("slice", params, dataset, total_steps, verbose)
    elif framework == "strong slice":
        ans = band_operations_random_walk("strong slice", params, dataset, total_steps, verbose)
    else:
        raise newException(ValueError, &"illegal framework {framework}")
    var encoded_ans = newSeq[(int, seq[string], seq[string], seq[string])]()
    for ans_item in ans:
        var link_history_json = collect:
            for link in ans_item[2]:
                encode_link(link)
        var random_states_json = newSeq[string]()
        for random_state in ans_item[3]:
            var s = newStringStream("")
            store(s, random_state)
            s.setPosition(0)
            random_states_json.add(s.readAll())
        encoded_ans.add((ans_item[0], ans_item[1], link_history_json, random_states_json))
    return encoded_ans
