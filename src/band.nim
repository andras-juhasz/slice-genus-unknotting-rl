import std/re
import std/strformat
import std/strutils
import std/sequtils
import std/random
import std/sugar
import std/enumerate
import std/options
import std/math
import std/logging
import std/tables

import "../spherogram-nim/src/links"
import "../spherogram-nim/src/simplify"

import subdivisions
import utils

proc get_act_type*(label: string): string =
    discard """
    Given an action label, returns the action type (start,
    over, under, twist, end).

    Args:
        label: an action label

    Returns:
        The action type of label.
    """
    var action_type = [""]
    assert find(label, re"_(.+)$", action_type) != -1
    if action_type[0] in ["end0", "end1", "end2"]:
        return "end"
    return action_type[0]

proc get_act_strand*(label: string): (int, int) =
    discard """
    Given an action label, returns the crossing strand the
    action involves.

    Args:
        label: an action of type start, over, under, or
            end (at a crossing strand)

    Returns:
        A pair of integers representing the crossing
        strand (edge with directions of the band's
        travel) that the action involves.

    Raises:
        ValueError: if the action is not of type start,
            over, under or end
    """
    if label[0] != 'C':
        raise newException(ValueError, &"strand cannot be found as action {label} is not a start, over, under or end action")
    var action_strand = ["", ""]
    assert find(label, re"C(\d+)S(\d+)_", action_strand) != -1
    return (parseInt(action_strand[0]), parseInt(action_strand[1]))

proc get_act_sign*(label: string): int =
    discard """
    Given an action label, returns the sign of the action.

    Args:
        label: an action of type over, under, twist, or end

    Returns:
        If the action is of type over, counterclockwise
        twist or end1 (first edge over final interface edge
        for CrossChange or adding a positive twist before
        ending the band for BandMove), returns 1. If the
        action is of type under, clockwise twist or end2
        (first edge under final interface edge for
        CrossChange or adding a negative twist before ending
        the band for BandMove), returns -1. If the action is
        of type end0 (adding no twists before ending the
        band for BandMove), returns 0.

    Raises:
        ValueError: if the action is not of type over,
            under, twist or end
    """
    if label == "ccw_twist" or label.endsWith("over") or label.endsWith("end1"):
        return 1
    if label == "cw_twist" or label.endsWith("under") or label.endsWith("end2"):
        return -1
    if label.endsWith("end0"):
        return 0
    raise newException(ValueError, &"sign cannot be found as action {label} is not an over, under, end or twist action")

proc randomly_choose_one*(can_choose: openArray[bool]): int =
    discard """
    Given a list of booleans, choose an index with true
    value in the list uniformly at random among all indices
    with true values in the list.

    Args:
        can_choose: a list of booleans with at least one
            true value

    Returns:
        an index i with 0 <= i < len(can_choose), such that
        can_choose[i] == True

    Raises:
        AssertionDefect: if can_choose does not have an
            element with value True
    """
    assert can_choose.anyIt(it)
    let choices = collect:
        for (i, c) in enumerate(can_choose):
            if c:
                i
    return sample(choices)

discard """
Information about the link that is only used by some
frameworks. The number of connected components is only used
by the "splitting" and "weak splitting" frameworks. The
Alexander polynomial is only used by the "unknotting-alg"
framework, and the connected component labels of the
unlinked unknot components are only used by the "ribbon",
"strong ribbon", "slice" and "strong slice" frameworks.
Since num_conn_comp is one of the features used by
representable, its value is not optional, but the Alexander
polynomial and the connected component labels of the
unlinked unknot components are optional.

GeneraInfo describes one connected component of the ribbon /
slice surface (keyed by its CCL): num_end_actions counts the
bands attached to it, num_init_comps + num_curr_comps its
boundary circles and num_orig_comps counts how many ORIGINAL 
link components it contains: it is 0 exactly for surface components 
made only of birthed unknots (slice frameworks), which are the 
only ones the strong frameworks may fuse with a different CCL.
"""
type GeneraInfo* = tuple
    num_end_actions: int
    num_init_comps: int
    num_curr_comps: int
    num_orig_comps: int

type FrameworkInfo* = object of RootObj
    num_conn_comp*: int
    alex_poly*: Option[seq[int]]
    unlinked_ccl*: Option[seq[int]]
    genera_info*: Table[int, GeneraInfo]

proc newFrameworkInfo*(num_conn_comp: int,
                       alex_poly: Option[seq[int]],
                       unlinked_ccl: Option[seq[int]],
                       genera_info: Table[int, GeneraInfo] = initTable[int, GeneraInfo]()): FrameworkInfo =
    return FrameworkInfo(num_conn_comp: num_conn_comp,
                         alex_poly: alex_poly,
                         unlinked_ccl: unlinked_ccl,
                         genera_info: genera_info)

discard """
Band-twist state, where we keep track of the band path
(which also records the number and sign of twists in
each band segment), the subdivided faces, the link, and
additional information aiding the calculation of the
(possibly signed) PD code of the link after a crossing
change or band move.
"""
type BandTwist* = ref object of RootObj
    band_path*: seq[array[0..3, int]]
    faces*: seq[SubdividedFace]
    link*: Link[int]
    simplified_link*: Link[int]
    pd_code*: seq[array[0..3, int]]
    face0*: seq[array[0..3, int]]
    max_twists*: int
    framework*: string
    framework_info*: FrameworkInfo
    simplification_switch*: bool

proc newBandTwist*(
    band_path: seq[array[0..3, int]],
    faces: seq[SubdividedFace],
    link: Link[int],
    pd_code: seq[array[0..3, int]],
    face0: seq[array[0..3, int]],
    max_twists: int,
    framework: string,
    framework_info: FrameworkInfo,
    simplification_switch: bool = true,
    simplified_link: Option[Link[int]] = none(Link[int])
): BandTwist =
    discard """
    Initializes an instance of BandTwist with all the
    fields filled in. This method should not be invoked
    directly.

    Args:
        band_path: list of lists with four elements: the
            first two elements record the edges with
            directions of the band's travel between the
            two faces that the edge bounds (recorded as
            crossing strands) that the band has visited
            so far; a crossing strand, viewed as an
            outgoing ray from its crossing, means that
            the band has gone through the edge of the
            crossing strand from the face on the left
            side of the ray (the "first face") to the
            face on the right side of the ray (the
            "second face"), where the initial strand
            means that the band starts on the strand and
            goes into the second face of the strand; the
            third element of the list records the sign
            (0 for start or end0, 1 for over or end1 and
            -1 for under or end2); the fourth and final
            element records the number of twists on the
            band after going through this crossing
            strand but before going through the
            subsequent crossing strand, which is
            positive if the twists are counterclockwise
            and negative if they are clockwise (for the
            final strand in the band path, it is always
            0)
        faces: the function mapping the index of a face
            to the current subdivision of the face by
            the band
        link: the link that the state is concerned with
        pd_code: the PD code of link
        face0: the function mapping a crossing strand to
            the index of its first face; the first face
            of a crossing strand is defined to be the
            face on the left side of the outgoing ray of
            the crossing strand
        max_twists: the maximum number of twists, either
            counterclockwise or clockwise, that we can make
            on the band in total, excluding the final twists
            added by end1 / end2 in BandMove; needed for
            self.check_twist_error
        framework: the type of link invariant to
            calculate; "unknotting" means that we allow
            band operations between any strands and
            terminate when the current link is an unlink;
            "ribbon" means that we allow band operations
            between any strands (no unknot births) and
            terminate when the current link is an unlink;
            "strong ribbon" means that we only allow band
            operations between strands with the same CCL
            (no unknot births) and terminate when the
            current link is an unlink; "slice" is "ribbon"
            plus the unknot_birth action; "strong slice"
            is "strong ribbon" plus the unknot_birth
            action, where band operations between
            different CCLs are additionally allowed if one
            of the two surface components contains no
            original link component (a pure-birth comp);
            "splitting" means that we do not allow band
            operations between strands of the same link
            component and terminate when the current
            link's diagram is completely split into link
            components; "weak splitting" means that we allow
            band operations between strands of the same link
            component and terminate when the current link's
            diagram is completely split into link components;
            "unknotting-alg" means that the link is a knot
            and we terminate when the current knot has
            Alexander polynomial 1
        framework_info: extra information that is only used
            by some frameworks
        simplification_switch: if true, the link is
            simplified between states; if false, the link
            is not simplified between states (but features
            and rewards are still computed on simplified
            versions)
        simplified_link: optionally provide pre-computed
            simplified link; if not provided, it will be
            computed from link

    Raises:
        ValueError: if framework is illegal
    """
    if framework notin ["unknotting", "unknotting-alg",
                        "splitting", "weak splitting",
                        "ribbon", "strong ribbon",
                        "slice", "strong slice"]:
        raise newException(ValueError, &"framework {framework} is neither \"unknotting\", \"(strong) ribbon\", \"(strong) slice\", \"(weak) splitting\" or \"unknotting-alg\"")
    var simp_link: Link[int]
    if simplified_link.isSome:
        # Pre-computed simplified link provided
        simp_link = simplified_link.get()
    else:
        # Always compute simplified link for features/rewards
        simp_link = link.copy()
        discard simp_link.simplify(SimplificationMode.global)
    return BandTwist(band_path: band_path, faces: faces,
                     link: link, simplified_link: simp_link,
                     pd_code: pd_code,
                     face0: face0, max_twists: max_twists,
                     framework: framework,
                     framework_info: framework_info,
                     simplification_switch: simplification_switch)

proc bandtwist_from_link*(link: Link[int],
                          max_twists: int = 5,
                          framework: string = "unknotting",
                          simplification_switch: bool = true,
                          simplified_link: Option[Link[int]] = none(Link[int])
                          ): BandTwist =
    discard """
    Constructs an instance of BandTwist from a link with
    no band attached.

    Args:
        link: the given link
        max_twists: the maximum number of twists (across
            all the faces, either counterclockwise or
            clockwise) allowed
        framework: the type of link invariant to
            calculate; for a detailed explanation, see
            the docstring of newBandTwist
        simplification_switch: if true, the link is
            simplified between states; if false, the link
            is not simplified between states (but features
            and rewards are still computed on simplified
            versions)
        simplified_link: optionally provide pre-computed
            simplified link; if not provided, it will be
            computed from link

    Returns:
        An instance of BandTwist whose link equals link
        and whose band has not started.
    """
    let num_crossings = link.crossings.len
    var face0 = collect:
        for _ in 0 ..< num_crossings:
            [-1, -1, -1, -1]
    var faces = newSeq[SubdividedFace]()
    for (face_index, face) in enumerate(link.faces()):
        var cur_face = newSeq[(int, int)]()
        if face.len == 1:
            raise newException(ValueError, &"link {link} is illegal as it admits a Reidemeister I simplification")
        for strand in face:
            # the following two lines work because the
            # crossing strands of a face are always
            # counterclockwise w.r.t. a point in the
            # interior of the face
            face0[strand.crossing][strand.strand_index] = face_index
            cur_face.add(strand.to_pair())
        faces.add(newSubdividedFace(cur_face))
    var pd_code = link.PD_code()
    var alex_poly: Option[seq[int]]
    if framework == "unknotting-alg":
        if link.link_components.len + link.unlinked_unknot_components != 1:
            raise newException(ValueError, "the algebraic unknotting number can only be calculated on knots")
        raise newException(AssertionDefect, "Not implemented")
    else:
        alex_poly = none(seq[int])
    return newBandTwist(@[], faces, link, pd_code, face0,
                        max_twists, framework,
                        newFrameworkInfo(1, alex_poly, none(seq[int])),
                        simplification_switch,
                        simplified_link)

method copy*(self: BandTwist): BandTwist {.base.} =
    discard """
    Makes a copy of self with deep copies of the mutable
    fields (band_path, faces) and shallow or deep copies of
    the other (immutable) fields.

    Returns:
        A deep copy of self.
    """
    return newBandTwist(self.band_path, self.faces,
                        self.link, self.pd_code,
                        self.face0, self.max_twists,
                        self.framework, self.framework_info,
                        self.simplification_switch,
                        some(self.simplified_link))

proc update*(self: BandTwist,
             band_path: seq[array[0..3, int]],
             faces: seq[SubdividedFace]): void =
    discard """
    Update the mutable fields of self to the argument
    values, so that we do not need to create a new
    instance of BandTwist every time we update those
    fields.

    Args:
        band_path: the new band path, represented by a
            list of crossing strand / sign / twist
            4-lists
        faces: the new list of subdivided faces
    """
    self.band_path = band_path
    self.faces = faces

proc check_strand_error*(self: BandTwist,
                         crossing: int,
                         strand_index: int): bool =
    discard """
    Checks whether a given crossing strand is valid and
    exists in the current link, namely whether the
    crossing exists and the strand index is between 0
    and 3.

    Args:
        crossing: the index of the crossing
        strand_index: the index of the crossing strand

    Returns:
        Whether the given crossing strand is valid and
        exists in the current link.
    """
    if crossing < 0 or crossing >= self.link.crossings.len:
        return true
    if strand_index < 0 or strand_index >= 4:
        return true
    return false

proc check_start_error*(self: BandTwist,
                        crossing: int,
                        strand_index: int): bool =
    # the initial edge and direction has already been
    # established
    if self.band_path.len > 0:
        return true
    # the given initial strand does not exist
    if self.check_strand_error(crossing, strand_index):
        return true
    return false

proc start*(self: BandTwist,
            crossing: int,
            strand_index: int): void =
    discard """
    Start the band at the edge and direction specified
    by the given crossing strand. (The band starts on
    the edge of the crossing strand and goes clockwise
    w.r.t. the ray of the crossing strand. We consider
    all rays of crossing strands to exit from their
    crossings.) The "first edge of the band" refers to
    the edge closer to the crossing and the "second edge
    of the band" refers to the edge farther from the
    crossing.

    Args:
        crossing: the index of the crossing
        strand_index: the index of the crossing strand

    Raises:
        ValueError: if the start action is illegal
    """
    if self.check_start_error(crossing, strand_index):
        raise newException(ValueError, "cannot start here")
    self.band_path.add([crossing, strand_index, 0, 0])
    self.faces[self.face0[crossing][strand_index]].set_initial_strand((crossing, strand_index))

proc feasible_next_strands*(
    self: BandTwist,
    cur_strand: (int, int)
): seq[(int, int)] =
    discard """
    Given a crossing strand, finds a list of strands
    of its second face that the band could exit from,
    had it entered the face through the given strand.

    Args:
        cur_strand: the given crossing strand

    Returns:
        A list of strands of the second face of
        cur_strand that the band could exit from without
        intersecting itself or crossing already-visited
        edges, had it entered the face through
        cur_strand
    """
    let cur_strand_opposite = self.link.opposite_strand(cur_strand)
    let next_face = self.face0[cur_strand_opposite[0]][cur_strand_opposite[1]]
    return self.faces[next_face].feasible_next_strands(cur_strand_opposite)

proc check_over_under_error*(self: BandTwist,
                             crossing: int,
                             strand_index: int): bool =
    # we have not chosen the initial strand
    if self.band_path.len == 0:
        return true
    let next_strand = (crossing, strand_index)
    let cur_strand = (self.band_path[self.band_path.len-1][0],
                      self.band_path[self.band_path.len-1][1])
    # either next_strand's first face is not the current
    # face, or the band would intersect itself or go
    # through already visited edges by exiting the
    # current face through next_strand
    if next_strand notin self.feasible_next_strands(cur_strand):
        return true
    # the band would not be able to end once next_strand
    # is crossed; then the band would not be able to do
    # over / under moves as well, so it will run out of
    # legal moves once it exhausts the available twists
    if self.feasible_next_strands(next_strand).len == 0:
        return true
    return false

proc cut_band_through_face*(
    self: BandTwist,
    strand1: (int, int),
    strand2: (int, int)
): void =
    discard """
    After the band exits a face, cuts the subdivision
    that the band went through into two.

    Args:
        strand1: the strand that the band entered the
            face from (clockwise w.r.t. the face)
        strand2: the strand that the band exited the
            face from (counterclockwise w.r.t. the face)
    """
    let strand1_opposite = self.link.opposite_strand(strand1)
    let cur_face = self.face0[strand1_opposite[0]][strand1_opposite[1]]
    self.faces[cur_face].cut_through(strand1_opposite, strand2)

proc over_under*(self: BandTwist,
                 crossing: int,
                 strand_index: int,
                 sign: int): void =
    discard """
    Extends the band over or under the specified
    interface edge (represented by a crossing strand) to
    the other face bounding the edge.

    Args:
        crossing: the index of the crossing
        strand_index: the index of the crossing strand
        sign: whether the band goes over or under the
            interface edge; 1 for over and -1 for under

    Raises:
        ValueError: if the sign is illegal (not 1 or -1)
            or if the action is illegal
    """
    if self.check_over_under_error(crossing, strand_index):
        raise newException(ValueError, "cannot go over or under here")
    if sign notin [-1, 1]:
        raise newException(ValueError, &"sign {sign} is not 1 or -1")

    self.band_path.add([crossing, strand_index, sign, 0])
    self.cut_band_through_face((self.band_path[self.band_path.len-2][0],
                                self.band_path[self.band_path.len-2][1]),
                               (self.band_path[self.band_path.len-1][0],
                                self.band_path[self.band_path.len-1][1]))

proc check_twist_error*(self: BandTwist, sign: int): bool =
    assert sign in [-1, 1]
    # band has not started, i.e. the initial edge has
    # not been established
    if self.band_path.len == 0:
        return true
    # the total number of twists that we have made so
    # far is at least max_twists, so we cannot make any
    # new ones
    let twists = collect:
        for t in self.band_path:
            t[3]
    let sum_twists = sum(twists)
    if abs(sum_twists) >= self.max_twists:
        return true
    # no going cw then ccw or going ccw then cw, because
    # if so, then the effect of the previous twist
    # would be reversed
    if sign == -sgn(sum_twists):
        return true
    return false

proc twist*(self: BandTwist, sign: int): void =
    discard """
    Creates a new twist at the end of the band with the
    specified sign (1 for counterclockwise and -1 for
    clockwise). The orientation is viewed from the
    existing part of the band towards its end.

    Args:
        sign: the sign of the twist; 1 for ccw and -1
            for cw

    Raises:
        ValueError: if the twist action is illegal
    """
    if self.check_twist_error(sign):
        raise newException(ValueError, "cannot twist here")
    self.band_path[self.band_path.len-1][3] += sign

proc initial_strand_component*(self: BandTwist
                              ): seq[(int, int)] =
    discard """
    Calculates the list of crossing strands (in both
    orientations) in the initial strand's link
    component. This method is used in check_end_error to
    preclude crossing changes between two strands of the
    same link component in the splitting number
    calculations.

    Returns:
        The list of crossing strands in the initial
        strand's link component.

    Raises:
        AssertionDefect: if the band has not started
    """
    assert self.band_path.len > 0
    var component = newSeq[(int, int)]()
    let initial_strand = (self.band_path[0][0],
                          self.band_path[0][1])
    var cur_strand = initial_strand
    while true:
        component.add(cur_strand)
        cur_strand = self.link.opposite_strand(cur_strand)
        component.add(cur_strand)
        cur_strand[1] = (cur_strand[1]+2) mod 4
        if cur_strand == initial_strand:
            break
    return component

proc check_end_errors*(self: BandTwist,
                       crossing: int,
                       strand_index: int): bool =
    # band has not started, i.e. the initial edge has
    # not been established
    if self.band_path.len == 0:
        return true
    # final strand does not start on the current face,
    # or the band would intersect itself or go through
    # already visited edges by extending to the given
    # final strand
    let cur_strand = (self.band_path[self.band_path.len-1][0],
                      self.band_path[self.band_path.len-1][1])
    if (crossing, strand_index) notin self.feasible_next_strands(cur_strand):
        return true
    # in (strong) splitting number calculations, the
    # final strand cannot be in the same link component
    # as the initial strand
    if (self.framework == "splitting" and
            (crossing, strand_index) in self.initial_strand_component()):
        return true
    # in strong ribbon / strong slice calculations, the
    # final strand must be on a component with the same ccl
    # as the initial strand's component; the only exception
    # (relevant for strong slice, where unknot births exist)
    # is when one of the two surface components contains no
    # original link component (a pure-birth comp): fusing it
    # decides which component's surface it belongs to
    if self.framework in ["strong ribbon", "strong slice"]:
        let (init_c, init_s) = (self.band_path[0][0], self.band_path[0][1])
        let init_ccl = get_strand_ccl(self.link, init_c, init_s)
        let final_ccl = get_strand_ccl(self.link, crossing, strand_index)
        if init_ccl != final_ccl:
            let gi = self.framework_info.genera_info
            if init_ccl notin gi or final_ccl notin gi:
                return true
            if (gi[init_ccl].num_orig_comps != 0 and
                    gi[final_ccl].num_orig_comps != 0):
                return true
    return false

method check_end_error*(self: BandTwist,
                        crossing: int,
                        strand_index: int,
                        sign: int): bool {.base.} =
    return self.check_end_errors(crossing, strand_index)

proc check_birth_error*(self: BandTwist): bool =
    discard """
    Checks whether an unknot_birth action would result in
    an error. A birth is only allowed in the slice
    frameworks, and only while no band is in progress.

    Returns:
        Whether the unknot_birth action would result in an
        error.
    """
    if self.framework notin ["slice", "strong slice"]:
        return true
    return self.band_path.len > 0

proc check_action_error*(self: BandTwist, action: string): bool =
    discard """
    Given an action label, checks whether the action
    would result in an error.

    Args:
        action: the action label; it should be of the
            form CxSy_start, CxSy_over, CxSy_under,
            ccw_twist, cw_twist, unknot_birth (for band
            moves in the slice frameworks), and CxSy_end1 /
            CxSy_end2 (for crossing changes) or
            CxSy_end0 / CxSy_end1 / CxSy_end2
            (for band moves)

    Returns:
        Whether the specified action would result in an
        error.
    """
    let action_type = get_act_type(action)
    if action_type == "start":
        let (c, s) = get_act_strand(action)
        return self.check_start_error(c, s)
    if action_type in ["over", "under"]:
        let (c, s) = get_act_strand(action)
        return self.check_over_under_error(c, s)
    if action_type == "twist":
        return self.check_twist_error(get_act_sign(action))
    if action_type == "end":
        let (c, s) = get_act_strand(action)
        return self.check_end_error(c, s, get_act_sign(action))
    if action_type == "birth":
        return self.check_birth_error()
    raise newException(ValueError, &"unknown action type {action_type}")

method cls_name*(self: BandTwist): string {.base.} = "BandTwist"

method `$`*(self: BandTwist): string {.base.} =
    return &"{self.cls_name()} instance with\nframework={self.framework}\nmax twists={self.max_twists}\nlink={self.link}\nband path={self.band_path}\n"

method next_state*(self: BandTwist, action: string): BandTwist {.base.} =
    discard """
    Given an action label, returns the state after the
    action is executed. (The state of self is not
    modified.) This method only handles the cases where
    action is of type start, over, under or twist (not
    end); end actions should be handled in overriding
    methods of subclasses of BandTwist.

    Args:
        action: the action label; it should be of the
            form CxSy_start, CxSy_over, CxSy_under,
            ccw_twist, cw_twist, and CxSy_end1 /
            CxSy_end2 (for crossing changes) or
            CxSy_end0 / CxSy_end1 / CxSy_end2 (for
            band moves)

    Returns:
        The state after the action is executed.

    Raises:
        ValueError: if the action is illegal on the
            current state

    Notes:
        This method handles most of the functions of the
        original env.KnotEnv.sample_transition_state
        method.
    """
    if self.check_action_error(action):
        raise newException(ValueError, &"action {action} is illegal on state {self}")
    var next_state = self.copy()
    let action_type = get_act_type(action)
    if action_type == "start":
        let (c, s) = get_act_strand(action)
        next_state.start(c, s)
    elif action_type in ["over", "under"]:
        let (c, s) = get_act_strand(action)
        next_state.over_under(c, s, get_act_sign(action))
    elif action_type == "twist":
        next_state.twist(get_act_sign(action))
    return next_state

proc sample_from_actions*(
    self: BandTwist,
    actions: openArray[string]
): BandTwist =
    discard """
    Calculates the state after executing a sequence of
    actions.

    Args:
        actions: a list of action labels

    Returns:
        The state of the current environment after this
        sequence of actions.

    Raises:
        ValueError: if any action in the sequence is
            illegal
    """
    var cur_state = self
    for action in actions:
        cur_state = cur_state.next_state(action)
    return cur_state

proc random_walk_transition*(
    self: BandTwist,
    p_twist: float,
    p_end: float
): Option[string] =
    discard """
    Samples the next action according to the
    probabilities given. The probabilities of start,
    over and under are always 1, and the probabilities
    of twist and end are given in the arguments.

    Args:
        p_twist: the twist probability
        p_end: the end (crossing change or band move)
            probability

    Returns:
        The label of the sampled action (for both crossing
        changes and band moves, the sign of the end action
        is not included; instead, the sign is appended in
        the overriding methods in state.CrossChange and
        BandMove).

    Notes:
        This function was originally
        env.KnotEnv.bayesian_transition; it was first
        renamed to random_walk_transition and later
        moved to state.BTDState, then to
        state.BandTwist, and finally to band.BandTwist.
    """
    let num_crossings = self.link.crossings.len
    # start
    var can_start_at = collect:
        for c in 0 ..< num_crossings:
            for i in 0 ..< 4:
                not self.check_start_error(c, i)
    # the list of feasible types of actions, with
    # corresponding probabilities given by the parameter
    var choices = newSeq[string]()
    var probabilities = newSeq[float]()
    if can_start_at.anyIt(it):
        choices.add("start")
        probabilities.add(1.0)
    # over, under
    var can_over_under_at = collect:
        for c in 0 ..< num_crossings:
            for i in 0 ..< 4:
                not self.check_over_under_error(c, i)
    if can_over_under_at.anyIt(it):
        choices.add("over")
        choices.add("under")
        probabilities.add(1.0)
        probabilities.add(1.0)
    # twist
    var can_twist = [not self.check_twist_error(1),
                     not self.check_twist_error(-1)]
    if can_twist.anyIt(it):
        choices.add("twist")
        probabilities.add(p_twist)
    # end
    var can_end_at = collect:
        for c in 0 ..< num_crossings:
            for i in 0 ..< 4:
                not self.check_end_errors(c, i)
    if can_end_at.anyIt(it):
        choices.add("end")
        probabilities.add(p_end)
    if choices.len == 0:
        if self.framework notin ["splitting", "strong ribbon", "strong slice"]:
            # we must have legal moves
            echo self.link.PD_code()
            echo self.link.faces()
            echo self.band_path
            raise newException(AssertionDefect, "no legal moves")
        else:
            warn(&"no legal moves for state {self}")
            return none(string)
    # choose action type
    let action_type = sample(choices, cumsummed(probabilities))
    if action_type == "start":
        # choose a starting strand uniformly at random
        let start_strand = randomly_choose_one(can_start_at)
        return some(&"C{start_strand div 4}S{start_strand mod 4}_start")
    if action_type in ["over", "under"]:
        # choose the next strand uniformly at random
        let next_strand = randomly_choose_one(can_over_under_at)
        return some(&"C{next_strand div 4}S{next_strand mod 4}_{action_type}")
    if action_type == "twist":
        # choose the twist direction uniformly at random
        let direction = randomly_choose_one(can_twist)
        return some(["ccw_twist", "cw_twist"][direction])
    if action_type == "end":
        # choose the final strand uniformly at random
        var final_strand = randomly_choose_one(can_end_at)
        return some(&"C{final_strand div 4}S{final_strand mod 4}_end")
    raise newException(AssertionDefect, "invalid action_type")

proc is_unlink*(self: BandTwist): bool =
    discard """
    Tests whether the state's link is an unlink.

    Returns:
        Whether the state's link is an unlink.
    """
    return is_unlink(self.simplified_link)

proc is_terminal*(self: BandTwist): bool =
    if self.framework in ["unknotting", "ribbon", "strong ribbon",
                          "slice", "strong slice"]:
        return self.is_unlink()
    if self.framework in ["splitting", "weak splitting"]:
        return self.framework_info.num_conn_comp == self.simplified_link.link_components.len
    if self.framework == "unknotting-alg":
        return self.framework_info.alex_poly.get() == @[1]
    raise newException(AssertionDefect, &"illegal framework")

method compute_current_answer*(self: BandTwist,
                               init_link: Link[int],
                               num_end_actions: int): int {.base.} =
    raise newException(AssertionDefect, "not implemented")

method terminal_state_message*(self: BandTwist,
                               num_end_actions: int,
                               answer: int): string {.base.} =
    raise newException(AssertionDefect, "not implemented")
