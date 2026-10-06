import std/sequtils
import std/options
import std/random
import std/strformat
import std/tables
import std/sugar
import std/math
import std/enumerate
import std/sets

import "../spherogram-nim/src/links"
import "../spherogram-nim/src/simplify"

import band

proc find_strand_in_face_with_most_edges*(
    link: Link[int]
): (int, int) =
    discard """
    Returns the first crossing strand in a link's face with
    the most edges.

    Args:
        link: the given link

    Returns:
        The crossing index and strand index of the first
        crossing strand in the link's face with the most
        edges.

    Raises:
        AssertionDefect: if link has no crossings
    """
    let faces = link.faces()
    assert faces.len > 0
    var most_edges_face_index = 0
    for i in 1 ..< faces.len:
        if faces[i].len > faces[most_edges_face_index].len:
            most_edges_face_index = i
    return faces[most_edges_face_index][0].to_pair()

proc join_split_link_diagrams*(link: Link[int]): Link[int] =
    discard """
    Given a link whose underlying 4-valent graph possibly
    contains multiple connected components, joins its link
    diagram components together via reverse Reidemeister II
    moves.

    Args:
        link: the given link

    Returns:
        The link with split link diagrams joined together,
        whose underlying 4-valent graph only has one
        connected component.
    """
    var split_diagrams = link.split_link_diagram()
    if split_diagrams.len <= 1:
        return link
    var joined_diagram = split_diagrams[0]
    for other_diagram in split_diagrams[1 ..< split_diagrams.len]:
        var (strand1_c, strand1_s) = find_strand_in_face_with_most_edges(joined_diagram)
        var (strand2_c, strand2_s) = find_strand_in_face_with_most_edges(other_diagram)
        # shift the crossing labels of strand2
        strand2_c += joined_diagram.crossings.len
        joined_diagram = disjoint_union(joined_diagram,
                                        other_diagram)
        reverse_type_ii(joined_diagram,
                        (strand1_c, strand1_s),
                        (strand2_c, strand2_s))
    joined_diagram.unlinked_unknot_components = link.unlinked_unknot_components
    return joined_diagram

discard """
The class implementing the crossing change as the method
to end a band, and overriding the random walk transition
function to provide a sign for the end action.
"""
type CrossChange* = ref object of BandTwist

proc crosschange_from_band_twist*(
    band_twist: BandTwist
): CrossChange =
    discard """
    Upgrades a BandTwist object to a CrossChange object.
    This method should not be invoked directly. Since it
    does not copy its argument, its argument should not
    be used again after being passed to this method.
    (This is like a rvalue reference argument in C++.)

    Args:
        band_twist: the BandTwist object to upgrade to
            a CrossChange object

    Returns:
        The upgraded CrossChange object.
    """
    return CrossChange(
        band_path: band_twist.band_path,
        faces: band_twist.faces, link: band_twist.link,
        simplified_link: band_twist.simplified_link,
        pd_code: band_twist.pd_code,
        face0: band_twist.face0,
        max_twists: band_twist.max_twists,
        framework: band_twist.framework,
        framework_info: band_twist.framework_info,
        simplification_switch: band_twist.simplification_switch
    )

proc crosschange_from_link*(
    link0: Link[int],
    max_twists: int = 5,
    framework: string = "unknotting",
    unlinked_ccl0: Option[seq[int]] = none(seq[int]),
    simplification_switch: bool = true,
    simplified_link: Option[Link[int]] = none(Link[int])
): CrossChange =
    discard """
    Constructs a CrossChange object from the given link.
    This function is used directly by the random walker.

    Args:
        link0: the given link
        max_twists: the maximum number of twists (across
            all the faces, either counterclockwise or
            clockwise) allowed
        framework: the type of link invariant to
            calculate; for a detailed explanation, see
            the docstring of band.newBandTwist
        unlinked_ccl0: a dummy parameter not used by this
            method; it is present to keep the signatures
            of crosschange_from_link and bandmove_from_link
            the same
        simplification_switch: if true, the link is
            simplified between states; if false, the link
            is not simplified between states (but features
            and rewards are still computed on simplified
            versions)
        simplified_link: optionally provide pre-computed
            simplified link; if not provided, it will be
            computed from link

    Returns:
        An instance of CrossChange whose link equals
        link and whose band has not started.
    """
    let num_conn_comp = link0.split_link_diagram().len
    var link = join_split_link_diagrams(link0.copy())
    var band_twist = bandtwist_from_link(link, max_twists,
                                         framework,
                                         simplification_switch,
                                         simplified_link)
    band_twist.framework_info.num_conn_comp = num_conn_comp
    return crosschange_from_band_twist(band_twist)

method copy*(self: CrossChange): CrossChange {.base.} =
    return crosschange_from_band_twist(BandTwist(self).copy())

proc cross_change_PD_code*(self: CrossChange): (seq[array[0..3, int]], seq[int]) =
    discard """
    Given the final state of the band, calculates the PD code and
    crossing signs of the new link after the crossing change.

    Returns:
        (PD code, crossing signs) of the new link after the
        crossing change.
    """
    let band = self.band_path
    # the band must have started (requiring the initial
    # edge) and ended (requiring the final edge), so the
    # band path must have length at least 2
    assert band.len >= 2
    # we create the new PD code by adapting the old one
    var PD = self.pd_code
    var signs = self.link.signs
    var initial_strand = (band[0][0], band[0][1])
    # break the initial edge into two rays, the one
    # starting at the crossing of initial_strand keeping
    # the old label and the opposite one getting a new
    # label
    var initial_strand_opposite = self.link.opposite_strand(initial_strand)
    var second_edge_strand = initial_strand_opposite
    var first_strand_sign = self.link.strand_sign(initial_strand[0], initial_strand[1])

    proc create_twist_crossing(
        first_edge_index: int,
        new_first_edge_index: int,
        new_second_edge_index: int,
        second_edge_index: int,
        sign: int
    ): void =
        if sign == 1:
            # counterclockwise
            if first_strand_sign == 1:
                second_edge_strand = (len(PD), 1)
                PD.add([new_first_edge_index,
                        new_second_edge_index,
                        second_edge_index,
                        first_edge_index])
            else:
                second_edge_strand = (len(PD), 3)
                PD.add([second_edge_index,
                        first_edge_index,
                        new_first_edge_index,
                        new_second_edge_index])
        else:
            # clockwise
            if first_strand_sign == 1:
                second_edge_strand = (len(PD), 2)
                PD.add([first_edge_index,
                        new_first_edge_index,
                        new_second_edge_index,
                        second_edge_index])
            else:
                second_edge_strand = (len(PD), 0)
                PD.add([new_second_edge_index,
                        second_edge_index,
                        first_edge_index,
                        new_first_edge_index])
        signs.add(sign)
        first_strand_sign = -first_strand_sign

    # num_edges is the total number of edges
    # since the edges are labeled from 0, num_edges is
    # also the next available label
    var num_edges = 2*self.link.crossings.len
    PD[initial_strand_opposite[0]][initial_strand_opposite[1]] = num_edges
    # the meanings of the first and second edges are
    # explained in the comments of self.start
    var first_edge_index = PD[initial_strand[0]][initial_strand[1]]
    var second_edge_index = num_edges
    num_edges += 1
    # intermediate edges between faces
    for face_index in 0 ..< band.len-2:
        # create crossings for twists
        for _ in 0 ..< abs(band[face_index][3]):
            var new_first_edge_index = num_edges
            num_edges += 1
            var new_second_edge_index = num_edges
            num_edges += 1
            create_twist_crossing(first_edge_index,
                                  new_first_edge_index,
                                  new_second_edge_index,
                                  second_edge_index,
                                  sgn(band[face_index][3]))
            first_edge_index = new_first_edge_index
            second_edge_index = new_second_edge_index
        var next_strand = (band[face_index+1][0],
                           band[face_index+1][1])
        # create crossings for the band's two
        # intersections with the edge of next_strand
        assert band[face_index+1][2] in [-1, 1]
        var lower_edge_index = PD[next_strand[0]][next_strand[1]]
        var middle_edge_index = num_edges
        num_edges += 1
        var upper_edge_index = num_edges
        num_edges += 1
        var next_strand_opposite = self.link.opposite_strand(next_strand)
        PD[next_strand_opposite[0]][next_strand_opposite[1]] = upper_edge_index
        var new_first_edge_index = num_edges
        num_edges += 1
        var new_second_edge_index = num_edges
        num_edges += 1
        var interface_strand_sign = self.link.strand_sign(next_strand[0], next_strand[1])
        if band[face_index+1][2] == 1:
            if interface_strand_sign == 1:
                PD.add([lower_edge_index,
                        new_first_edge_index,
                        middle_edge_index,
                        first_edge_index])
                signs.add(first_strand_sign)
                second_edge_strand = (len(PD), 1)
                PD.add([middle_edge_index,
                        new_second_edge_index,
                        upper_edge_index,
                        second_edge_index])
                signs.add(-first_strand_sign)
            else:
                PD.add([middle_edge_index,
                        first_edge_index,
                        lower_edge_index,
                        new_first_edge_index])
                signs.add(-first_strand_sign)
                second_edge_strand = (len(PD), 3)
                PD.add([upper_edge_index,
                        second_edge_index,
                        middle_edge_index,
                        new_second_edge_index])
                signs.add(first_strand_sign)
        else:
            if first_strand_sign == 1:
                PD.add([first_edge_index,
                        lower_edge_index,
                        new_first_edge_index,
                        middle_edge_index])
                signs.add(-interface_strand_sign)
                second_edge_strand = (len(PD), 0)
                PD.add([new_second_edge_index,
                        upper_edge_index,
                        second_edge_index,
                        middle_edge_index])
                signs.add(interface_strand_sign)
            else:
                PD.add([new_first_edge_index,
                        middle_edge_index,
                        first_edge_index,
                        lower_edge_index])
                signs.add(interface_strand_sign)
                second_edge_strand = (len(PD), 2)
                PD.add([second_edge_index,
                        middle_edge_index,
                        new_second_edge_index,
                        upper_edge_index])
                signs.add(-interface_strand_sign)
        first_edge_index = new_first_edge_index
        second_edge_index = new_second_edge_index
    # create crossings for twists on the penultimate
    # face
    for _ in 0 ..< abs(band[band.len-2][3]):
        var new_first_edge_index = num_edges
        num_edges += 1
        var new_second_edge_index = num_edges
        num_edges += 1
        create_twist_crossing(first_edge_index,
                              new_first_edge_index,
                              new_second_edge_index,
                              second_edge_index,
                              sgn(band[band.len-2][3]))
        first_edge_index = new_first_edge_index
        second_edge_index = new_second_edge_index
    # FINAL FACE: the crossing change itself. Layout mirrors a body
    # interface, but the two new crossings close onto a single shared
    # arc (final_edge_index) instead of continuing the band as two
    # separate arcs (new_first_edge_index, new_second_edge_index).
    var final_strand = (band[band.len-1][0], band[band.len-1][1])
    var lower_edge_index = PD[final_strand[0]][final_strand[1]]
    var middle_edge_index = num_edges
    num_edges += 1
    var upper_edge_index = num_edges
    num_edges += 1
    var final_strand_opposite = self.link.opposite_strand(final_strand)
    PD[final_strand_opposite[0]][final_strand_opposite[1]] = upper_edge_index
    var final_edge_index = num_edges
    num_edges += 1
    var interface_strand_sign = self.link.strand_sign(final_strand[0], final_strand[1])
    assert band[band.len-1][2] in [-1, 1]
    if band[band.len-1][2] == 1:
        # first_edge goes over final_strand, second_edge goes under.
        # First crossing: the interface strand is the under-arc
        # (positions 0,2), so the rotation is fixed by
        # interface_strand_sign; the crossing sign is read off from
        # where the band (over-arc) enters (position 3 -> +1, 1 -> -1).
        if interface_strand_sign == 1:
            PD.add([lower_edge_index,
                    final_edge_index,
                    middle_edge_index,
                    first_edge_index])
            signs.add(first_strand_sign)
        else:
            PD.add([middle_edge_index,
                    first_edge_index,
                    lower_edge_index,
                    final_edge_index])
            signs.add(-first_strand_sign)
        # Second crossing: the BAND (final_edge -> second_edge) is the
        # under-arc, so the rotation is fixed by first_strand_sign and
        # the crossing sign by where the interface (over-arc) enters.
        if first_strand_sign == 1:
            PD.add([final_edge_index,
                    upper_edge_index,
                    second_edge_index,
                    middle_edge_index])
            signs.add(interface_strand_sign)
        else:
            PD.add([second_edge_index,
                    middle_edge_index,
                    final_edge_index,
                    upper_edge_index])
            signs.add(-interface_strand_sign)
    else:
        # first_edge goes under final_strand, second_edge goes over.
        # First crossing: the band (first_edge -> final_edge) is the
        # under-arc, rotation fixed by first_strand_sign.
        if first_strand_sign == 1:
            PD.add([first_edge_index,
                    lower_edge_index,
                    final_edge_index,
                    middle_edge_index])
            signs.add(-interface_strand_sign)
        else:
            PD.add([final_edge_index,
                    middle_edge_index,
                    first_edge_index,
                    lower_edge_index])
            signs.add(interface_strand_sign)
        # Second crossing: the interface strand is the under-arc,
        # rotation fixed by interface_strand_sign.
        if interface_strand_sign == 1:
            PD.add([middle_edge_index,
                    final_edge_index,
                    upper_edge_index,
                    second_edge_index])
            signs.add(-first_strand_sign)
        else:
            PD.add([upper_edge_index,
                    second_edge_index,
                    middle_edge_index,
                    final_edge_index])
            signs.add(first_strand_sign)
    return (PD, signs)

proc end_band*(self: CrossChange,
               crossing: int,
               strand_index: int,
               sign: int): Link[int] =
    discard """
    Pushes the band through the given final edge (and
    direction) with the over / under relations between
    the band's two edges and the final interface edge
    specified by the parameter sign and returns the new
    link after the crossing change.

    Args:
        crossing: the index of the crossing
        strand_index: the index of the crossing strand
        sign: the over / under relations between the
            band's two edges and the final interface
            edge; 1 means the first edge of the band
            goes over the final interface edge (and the
            second edge of the band goes under) and -1
            means the first edge of the band goes under
            the final interface edge (and the second
            edge of the band goes over)

    Returns:
        The new link constructed from the PD code after
        the crossing change.

    Raises:
        ValueError: if the end action is illegal
    """
    assert sign in [-1, 1]
    if self.check_end_errors(crossing, strand_index):
        raise newException(ValueError, "cannot end here")

    self.band_path.add([crossing, strand_index, sign, 0])
    self.cut_band_through_face((self.band_path[self.band_path.len-2][0],
                                self.band_path[self.band_path.len-2][1]),
                               (self.band_path[self.band_path.len-1][0],
                                self.band_path[self.band_path.len-1][1]))

    let (pd_code, signs) = self.cross_change_PD_code()
    return link_from_PD_code(pd_code, signs,
                             unlinked_unknot_components = self.link.unlinked_unknot_components)

method next_state*(self: CrossChange, action: string): CrossChange {.base.} =
    let action_type = get_act_type(action)
    if action_type == "end":
        var next_state = self.copy()
        var (c, s) = get_act_strand(action)
        var new_link = next_state.end_band(c, s, get_act_sign(action))
        # Compute simplified link once for features/rewards
        var new_simplified_link = new_link.copy()
        discard new_simplified_link.simplify(SimplificationMode.global)
        if not self.simplification_switch:
            # Keep original link unsimplified, pass pre-computed simplified_link
            return crosschange_from_link(new_link,
                                     self.max_twists,
                                        self.framework,
                                        none(seq[int]),
                                        self.simplification_switch,
                                        some(new_simplified_link))
        else:
            # Use simplified version for both
            return crosschange_from_link(new_simplified_link,
                                        self.max_twists,
                                        self.framework,
                                        none(seq[int]),
                                        self.simplification_switch,
                                        some(new_simplified_link))
    return crosschange_from_band_twist(BandTwist(self).next_state(action))

proc random_walk_transition*(self: CrossChange,
                             p_twist: float,
                             p_end: float): Option[string] =
    # if this were a method, it would run into an infinite
    # loop with band.BandTwist.random_walk_transition
    var provisional_move = BandTwist(self).random_walk_transition(p_twist, p_end)
    if provisional_move.isNone:
        return none(string)
    if get_act_type(provisional_move.get()) == "end":
        # choose one of the two crossing change types
        # with equal probability
        var sign = rand(0..1)
        return some(&"{provisional_move.get()}{sign+1}")
    return provisional_move

method compute_current_answer*(self: CrossChange,
                               init_link: Link[int],
                               num_end_actions: int): int =
    return num_end_actions

method terminal_state_message*(self: CrossChange,
                               num_end_actions: int,
                               answer: int): string =
    let message = ({"unknotting": "reached unlink",
                    "splitting": "completely split",
                    "weak splitting": "completely split",
                    "unknotting-alg": "reached a knot with Alexander polynomial 1"
                   }.toTable()[self.framework])
    return &"{message} with {answer} crossing changes"

method cls_name*(self: CrossChange): string = "CrossChange"

discard """
The class implementing the band move as the method
to end a band.
"""
type BandMove* = ref object of BandTwist

proc bandmove_from_band_twist*(band_twist: BandTWist): BandMove =
    discard """
    Upgrades a BandTwist object to a BandMove object.
    This method should not be invoked directly. Since it
    does not copy its argument, its argument should not
    be used again after being passed to this method.
    (This is like a rvalue reference argument in C++.)

    Args:
        band_twist: the BandTwist object to upgrade to
            a BandMove object

    Returns:
        The upgraded BandMove object.
    """
    return BandMove(
        band_path: band_twist.band_path,
        faces: band_twist.faces, link: band_twist.link,
        simplified_link: band_twist.simplified_link,
        pd_code: band_twist.pd_code,
        face0: band_twist.face0,
        max_twists: band_twist.max_twists,
        framework: band_twist.framework,
        framework_info: band_twist.framework_info,
        simplification_switch: band_twist.simplification_switch
    )

proc remove_reidemeister_i_kinks*(link: Link[int]): seq[int] =
    discard """
    Removes every Reidemeister I kink of a link in place,
    repeating until none is left, since removing one kink
    can expose another. A band cannot act on a diagram
    with a monogon face (bandtwist_from_link rejects it),
    so a band move that is not followed by a full
    simplification needs at least this.

    Args:
        link: the link, changed in place

    Returns:
        The connected component labels of the components
        that lost all their crossings, which are now
        unlinked unknot components of link.
    """
    var ccl_count = initTable[int, int]()
    for comp in link.link_components:
        ccl_count[comp.extra_info] = ccl_count.getOrDefault(comp.extra_info) + 1
    var removed = true
    while removed:
        removed = false
        for c in 0 ..< link.crossings.len:
            if reidemeister_i(link, c)[0].len > 0:
                removed = true
                break
    for comp in link.link_components:
        ccl_count[comp.extra_info] -= 1
    for (ccl, count) in ccl_count.pairs:
        for _ in 0 ..< count:
            result.add(ccl)

proc join_unlinked_unknot_component*(
    link: Link[int],
    unlinked_ccl: var seq[int]
): Link[int] =
    discard """
    Joins an unlinked unknot component via a reverse
    Reidemeister II move performed on the strand given by
    the function find_strand_in_face_with_most_edges().

    Args:
        link: the old link, containing at least one unlinked
            unknot component
        unlinked_ccl: a sequence of connected component
            labels of link's unlinked unknot components;
            one element of this sequence will be popped to
            provide the connected component label for the
            unlinked unknot component being joined with link

    Returns:
        A new link, which equals link with an unlinked
        unknot component joined as an under-circle of an
        edge of link, oriented counterclockwise.
    """
    assert link.unlinked_unknot_components > 0
    var strand_to_join = find_strand_in_face_with_most_edges(link)
    let strand_sign = link.strand_sign(strand_to_join[0], strand_to_join[1])
    if strand_sign == -1:
        strand_to_join = link.opposite_strand(strand_to_join)
    let opposite_strand = link.opposite_strand(strand_to_join)
    var num_edges = 2*link.crossings.len
    var PD = link.PD_code()
    let lower_edge_index = PD[strand_to_join[0]][strand_to_join[1]]
    let middle_edge_index = num_edges
    num_edges += 1
    let upper_edge_index = num_edges
    num_edges += 1
    PD[opposite_strand[0]][opposite_strand[1]] = upper_edge_index
    let left_edge_index = num_edges
    num_edges += 1
    let right_edge_index = num_edges
    num_edges += 1
    PD.add([left_edge_index, lower_edge_index,
            right_edge_index, middle_edge_index])
    PD.add([right_edge_index, upper_edge_index,
            left_edge_index, middle_edge_index])
    var ans = link_from_PD_code(PD, link.signs & @[-1, 1],
                                link.unlinked_unknot_components-1)
    ans.link_components = link.link_components
    ans.link_components.add(newLinkComponent(link.crossings.len,
                                             2,
                                             unlinked_ccl.pop()))
    return ans

proc join_unlinked_unknot_components*(
    link0: Link[int],
    unlinked_ccl: var seq[int]
): Link[int] =
    discard """
    Joins all the unlinked unknot components of a link as
    under-circles of its edges, whose connected component
    labels are specified by a parameter.

    Args:
        link0: the link whose unlinked unknot components are
            to be joined as under-circles of its edges
        unlinked_ccl: the connected component labels of
            link0's unlinked unknot components

    Returns:
        A new link, which equals link0 with all of its
        unlinked unknot components joined as under-circles
        of edges of link0, oriented counterclockwise, if
        link0 has at least one crossing, and link0 if link0
        has no crossings.

    Raises:
        AssertionDefect: if the length of unlinked_ccl does
            not equal the number of unlinked unknot
            components of link0
    """
    assert link0.unlinked_unknot_components == unlinked_ccl.len
    if link0.crossings.len == 0:
        return link0
    var link = link0
    while link.unlinked_unknot_components > 0:
        link = join_unlinked_unknot_component(link, unlinked_ccl)
    return link

proc bandmove_from_link*(
    link0: Link[int],
    max_twists: int = 5,
    framework: string = "unknotting",
    unlinked_ccl0: Option[seq[int]] = none(seq[int]),
    simplification_switch: bool = true,
    simplified_link: Option[Link[int]] = none(Link[int])
): BandMove =
    discard """
    Constructs a BandMove object from the given link.

    Args:
        link0: the given link
        max_twists: the maximum number of twists (across
            all the faces, either counterclockwise or
            clockwise) allowed
        framework: the type of link invariant to
            calculate; for a detailed explanation, see
            the docstring of band.newBandTwist
        unlinked_ccl0: the connected component labels of the
            unlinked unknot components of link0
        simplification_switch: if true, the link is
            simplified between states; if false, the link
            is not simplified between states (but features
            and rewards are still computed on simplified
            versions)
        simplified_link: optionally provide pre-computed
            simplified link; if not provided, it will be
            computed from link

    Returns:
        An instance of BandMove whose link equals
        link0 and whose band has not started.
    """
    var unlinked_ccl: seq[int]
    var link = link0.copy()
    if unlinked_ccl0.isSome:
        unlinked_ccl = unlinked_ccl0.get()
    else:
        for comp_index in 0 ..< link.link_components.len:
            link.link_components[comp_index].extra_info = comp_index
        unlinked_ccl = collect:
            for comp in 0 ..< link.unlinked_unknot_components:
                link.link_components.len + comp
    let num_conn_comp = link.split_link_diagram().len
    link = join_unlinked_unknot_components(join_split_link_diagrams(link), unlinked_ccl)
    var band_twist = bandtwist_from_link(link, max_twists,
                                         framework,
                                         simplification_switch,
                                         simplified_link)
    band_twist.framework_info.num_conn_comp = num_conn_comp
    band_twist.framework_info.unlinked_ccl = some(unlinked_ccl)
    # Initialize genera_info at episode start
    if band_twist.framework_info.genera_info.len == 0:
        var ccl_count0 = initTable[int, int]()
        for comp in link.link_components:
            let ccl = comp.extra_info
            if ccl notin ccl_count0:
                ccl_count0[ccl] = 0
            ccl_count0[ccl] += 1
        var gi_table0 = initTable[int, GeneraInfo]()
        for (ccl, cnt) in ccl_count0.pairs:
            # at episode start every circle with this ccl is an
            # original link component, so num_orig_comps = cnt;
            # birthed unknots (num_orig_comps = 0) only appear
            # later via the unknot_birth action
            # Remark: genera_info initially is 
            # (num_end_actions. 0, num_init_comps: 1, 
            # num_curr_comps: 1, num_orig_comps: 1)
            gi_table0[ccl] = (0, cnt, cnt, cnt)
        band_twist.framework_info.genera_info = gi_table0
    return bandmove_from_band_twist(band_twist)

method copy*(self: BandMove): BandMove {.base.} =
    return bandmove_from_band_twist(BandTwist(self).copy())

method check_end_error*(self: BandMove,
                        crossing: int,
                        strand_index: int,
                        sign: int): bool =
    discard """
    We partially override the parent method
    BandTwist.check_end_error(), as in the BandMove case
    we have three different signs for end, and we need
    to check the compatibility of the sign with the
    orientations of the band and the final strand.
    """
    if self.check_end_errors(crossing, strand_index):
        return true
    assert sign in [-1, 0, 1]
    let twists = collect:
        for t in self.band_path:
            t[3]
    let sum_twists = sum(twists)
    # no going cw then ccw or going ccw then cw, because
    # if so, then the effect of the previous twist
    # would be reversed
    if sum_twists != 0 and sign == -sgn(sum_twists):
        return true
    # We check if the orientations of the band and the
    # final strand are compatible, considering possibly
    # the final new twist added by "sign"
    if (self.link.strand_sign(self.band_path[0][0],
                              self.band_path[0][1]) !=
        self.link.strand_sign(crossing, strand_index)) !=
        (abs(sum_twists + sign) mod 2 == 0):
        return true
    return false

proc band_move_PD_code*(
    self: BandMove
): (seq[array[0..3, int]], seq[int]) =
    discard """
    Given the final state of the band, calculates the PD
    code and crossing signs of the new link after the band
    move.

    Returns:
        The PD code and crossing signs of the new link after
        the band move.
    """
    var band = self.band_path
    # the band must have started (requiring the initial
    # edge) and ended (requiring the final edge), so the
    # band path must have length at least 2
    assert band.len >= 2
    # we create the new PD code by adapting the old one
    var PD = self.pd_code
    var signs = self.link.signs
    var initial_strand = (band[0][0], band[0][1])
    # break the initial edge into two rays, the one
    # starting at the crossing of initial_strand keeping
    # the old label and the opposite one getting a new
    # label
    var initial_strand_opposite = self.link.opposite_strand(initial_strand)
    var second_edge_strand = initial_strand_opposite
    var first_strand_sign = self.link.strand_sign(initial_strand[0], initial_strand[1])

    proc create_twist_crossing(
        first_edge_index: int,
        new_first_edge_index: int,
        new_second_edge_index: int,
        second_edge_index: int,
        sign: int
    ): void =
        if sign == 1:
            # counterclockwise
            if first_strand_sign == 1:
                second_edge_strand = (len(PD), 1)
                PD.add([new_first_edge_index,
                        new_second_edge_index,
                        second_edge_index,
                        first_edge_index])
            else:
                second_edge_strand = (len(PD), 3)
                PD.add([second_edge_index,
                        first_edge_index,
                        new_first_edge_index,
                        new_second_edge_index])
        else:
            # clockwise
            if first_strand_sign == 1:
                second_edge_strand = (len(PD), 2)
                PD.add([first_edge_index,
                        new_first_edge_index,
                        new_second_edge_index,
                        second_edge_index])
            else:
                second_edge_strand = (len(PD), 0)
                PD.add([new_second_edge_index,
                        second_edge_index,
                        first_edge_index,
                        new_first_edge_index])
        signs.add(sign)
        first_strand_sign = -first_strand_sign

    # num_edges is the total number of edges
    # since the edges are labeled from 0, num_edges is
    # also the next available label
    var num_edges = 2*self.link.crossings.len
    # relabel the opposite ray
    PD[initial_strand_opposite[0]][initial_strand_opposite[1]] = num_edges
    # the meanings of the first and second edges are
    # explained in the comments of self.start
    var first_edge_index = PD[initial_strand[0]][initial_strand[1]]
    var second_edge_index = num_edges
    num_edges += 1
    # intermediate edges between faces
    for face_index in 0 ..< band.len-2:
        # create crossings for twists
        for _ in 0 ..< abs(band[face_index][3]):
            var new_first_edge_index = num_edges
            num_edges += 1
            var new_second_edge_index = num_edges
            num_edges += 1
            create_twist_crossing(first_edge_index,
                                  new_first_edge_index,
                                  new_second_edge_index,
                                  second_edge_index,
                                  sgn(band[face_index][3]))
            first_edge_index = new_first_edge_index
            second_edge_index = new_second_edge_index
        var next_strand = (band[face_index+1][0],
                       band[face_index+1][1])
        # create crossings for the band's two
        # intersections with the edge of next_strand
        assert band[face_index+1][2] in [-1, 1]
        var lower_edge_index = PD[next_strand[0]][next_strand[1]]
        var middle_edge_index = num_edges
        num_edges += 1
        var upper_edge_index = num_edges
        num_edges += 1
        var next_strand_opposite = self.link.opposite_strand(next_strand)
        PD[next_strand_opposite[0]][next_strand_opposite[1]] = upper_edge_index
        var new_first_edge_index = num_edges
        num_edges += 1
        var new_second_edge_index = num_edges
        num_edges += 1
        var interface_strand_sign = self.link.strand_sign(next_strand[0], next_strand[1])
        if band[face_index+1][2] == 1:
            if interface_strand_sign == 1:
                PD.add([lower_edge_index,
                        new_first_edge_index,
                        middle_edge_index,
                        first_edge_index])
                signs.add(first_strand_sign)
                second_edge_strand = (len(PD), 1)
                PD.add([middle_edge_index,
                        new_second_edge_index,
                        upper_edge_index,
                        second_edge_index])
                signs.add(-first_strand_sign)
            else:
                PD.add([middle_edge_index,
                        first_edge_index,
                        lower_edge_index,
                        new_first_edge_index])
                signs.add(-first_strand_sign)
                second_edge_strand = (len(PD), 3)
                PD.add([upper_edge_index,
                        second_edge_index,
                        middle_edge_index,
                        new_second_edge_index])
                signs.add(first_strand_sign)
        else:
            if first_strand_sign == 1:
                PD.add([first_edge_index,
                        lower_edge_index,
                        new_first_edge_index,
                        middle_edge_index])
                signs.add(-interface_strand_sign)
                second_edge_strand = (len(PD), 0)
                PD.add([new_second_edge_index,
                        upper_edge_index,
                        second_edge_index,
                        middle_edge_index])
                signs.add(interface_strand_sign)
            else:
                PD.add([new_first_edge_index,
                        middle_edge_index,
                        first_edge_index,
                        lower_edge_index])
                signs.add(interface_strand_sign)
                second_edge_strand = (len(PD), 2)
                PD.add([second_edge_index,
                        middle_edge_index,
                        new_second_edge_index,
                        upper_edge_index])
                signs.add(-interface_strand_sign)
        first_edge_index = new_first_edge_index
        second_edge_index = new_second_edge_index
    # create crossings for twists on the final face
    for _ in 0 ..< abs(band[band.len-2][3]):
        var new_first_edge_index = num_edges
        num_edges += 1
        var new_second_edge_index = num_edges
        num_edges += 1
        create_twist_crossing(first_edge_index,
                              new_first_edge_index,
                              new_second_edge_index,
                              second_edge_index,
                              sgn(band[band.len-2][3]))
        first_edge_index = new_first_edge_index
        second_edge_index = new_second_edge_index
    var final_strand = (band[band.len-1][0], band[band.len-1][1])
    # create the band move
    var final_edge_index = PD[final_strand[0]][final_strand[1]]
    PD[second_edge_strand[0]][second_edge_strand[1]] = final_edge_index
    PD[final_strand[0]][final_strand[1]] = first_edge_index
    return (PD, signs)

proc end_band*(self: BandMove,
               crossing: int,
               strand_index: int,
               sign: int): Link[int] =
    discard """
    Attaches the band to the given final edge, possibly
    making a final twist to adjust orientation, and
    returns the new link after the band move.

    Args:
        crossing: the index of the crossing
        strand_index: the index of the crossing strand
        sign: the sign of the end action, which is zero
            if it does not add any twist in the last
            face, 1 if it adds a ccw twist and -1 if
            it adds a cw twist

    Returns:
        The new link constructed from the PD code after
        the band move.

    Raises:
        ValueError: if the end action is illegal
    """
    if self.check_end_error(crossing, strand_index, sign):
        raise newException(ValueError, "cannot end here")

    # We add the final twist, if there is one
    self.band_path[self.band_path.len-1][3] += sign

    self.band_path.add([crossing, strand_index, 0, 0])
    self.cut_band_through_face((self.band_path[self.band_path.len-2][0],
                                self.band_path[self.band_path.len-2][1]),
                               (self.band_path[self.band_path.len-1][0],
                                self.band_path[self.band_path.len-1][1]))

    let (pd_code, signs) = self.band_move_PD_code()
    var ans = link_from_PD_code(pd_code, signs, self.link.unlinked_unknot_components)
    let num_crossings = self.link.crossings.len
    var strand_to_comp = newSeqWith(num_crossings, [-1, -1, -1, -1])
    var strand_to_ccl = newSeqWith(num_crossings, [-1, -1, -1, -1])
    for (comp_index, comp) in enumerate(self.link.link_components):
        let (start_c, start_s) = (comp.crossing, comp.strand_index)
        var (cur_c, cur_s) = (start_c, start_s)
        while true:
            strand_to_comp[cur_c][cur_s] = comp_index
            strand_to_ccl[cur_c][cur_s] = comp.extra_info
            let (op_c, op_s) = self.link.opposite_strand((cur_c, cur_s))
            strand_to_comp[op_c][op_s] = comp_index
            strand_to_ccl[op_c][op_s] = comp.extra_info
            (cur_c, cur_s) = (op_c, (op_s+2) mod 4)
            if (cur_c, cur_s) == (start_c, start_s):
                break
    var remove_comps = newSeq[bool](self.link.link_components.len)
    let (initial_c, initial_s) = (self.band_path[0][0], self.band_path[0][1])
    let (final_c, final_s) = (self.band_path[self.band_path.len-1][0],
                              self.band_path[self.band_path.len-1][1])
    remove_comps[strand_to_comp[initial_c][initial_s]] = true
    remove_comps[strand_to_comp[final_c][final_s]] = true
    ans.link_components = collect:
        for (comp, remove_comp) in zip(self.link.link_components, remove_comps):
            if not remove_comp:
                comp
    if strand_to_comp[initial_c][initial_s] == strand_to_comp[final_c][final_s]:
        # fission band move
        let new_ccl = strand_to_ccl[initial_c][initial_s]
        if self.link.strand_sign(initial_c, initial_s) == 1:
            ans.link_components.add(newLinkComponent(initial_c, initial_s, new_ccl))
        else:
            let (op_c, op_s) = self.link.opposite_strand((initial_c, initial_s))
            ans.link_components.add(newLinkComponent(op_c, op_s, new_ccl))
        if self.link.strand_sign(final_c, final_s) == 1:
            ans.link_components.add(newLinkComponent(final_c, final_s, new_ccl))
        else:
            let (op_c, op_s) = self.link.opposite_strand((final_c, final_s))
            ans.link_components.add(newLinkComponent(op_c, op_s, new_ccl))
    else:
        # Fusion band move
        # Keep the larger ccl, remove the smaller one
        var ccl_small = strand_to_ccl[initial_c][initial_s]
        var ccl_large = strand_to_ccl[final_c][final_s]
        if ccl_small > ccl_large:
            swap(ccl_small, ccl_large)
        # Update all components with ccl_small to use ccl_large
        for (comp_index, comp) in enumerate(ans.link_components):
            if comp.extra_info == ccl_small:
                ans.link_components[comp_index].extra_info = ccl_large
        if self.link.strand_sign(initial_c, initial_s) == 1:
            ans.link_components.add(newLinkComponent(initial_c, initial_s, ccl_large))
        else:
            let (op_c, op_s) = self.link.opposite_strand((initial_c, initial_s))
            ans.link_components.add(newLinkComponent(op_c, op_s, ccl_large))

    # Update genera_info for the genus frameworks
    if self.framework in ["ribbon", "strong ribbon",
                          "slice", "strong slice"]:
        let ccl1_orig = strand_to_ccl[initial_c][initial_s]
        let ccl2_orig = strand_to_ccl[final_c][final_s]

        if strand_to_comp[initial_c][initial_s] == strand_to_comp[final_c][final_s]:
            # Fission band move: one component splits into two
            # Both new components have the same ccl (ccl1_orig == ccl2_orig)
            let involved_ccl = ccl1_orig
            assert involved_ccl in self.framework_info.genera_info,
                "CCL " & $involved_ccl & " not found in genera_info"
            var gi = self.framework_info.genera_info[involved_ccl]
            gi.num_end_actions += 1
            gi.num_curr_comps += 1
            self.framework_info.genera_info[involved_ccl] = gi
        else:
            # Fusion band move: two components merge into one
            var ccl_small = ccl1_orig
            var ccl_large = ccl2_orig
            if ccl_small > ccl_large:
                swap(ccl_small, ccl_large)

            if ccl_small == ccl_large:
                # Same ccl (always true for strong ribbon; in
                # strong slice a pure-birth comp may also fuse
                # across different ccls)
                assert ccl_small in self.framework_info.genera_info,
                    "CCL " & $ccl_small & " not found in genera_info"
                var gi = self.framework_info.genera_info[ccl_small]
                gi.num_end_actions += 1
                gi.num_curr_comps -= 1
                self.framework_info.genera_info[ccl_small] = gi
            else:
                # Different ccls being merged, merge genera_info
                assert ccl_small in self.framework_info.genera_info,
                    "CCL " & $ccl_small & " not found in genera_info"
                assert ccl_large in self.framework_info.genera_info,
                    "CCL " & $ccl_large & " not found in genera_info"
                var gi_small = self.framework_info.genera_info[ccl_small]
                var gi_large = self.framework_info.genera_info[ccl_large]

                var merged_gi: GeneraInfo
                merged_gi.num_end_actions = gi_small.num_end_actions +
                                            gi_large.num_end_actions + 1
                merged_gi.num_init_comps = gi_small.num_init_comps +
                                           gi_large.num_init_comps
                merged_gi.num_curr_comps = gi_small.num_curr_comps +
                                           gi_large.num_curr_comps - 1
                merged_gi.num_orig_comps = gi_small.num_orig_comps +
                                           gi_large.num_orig_comps

                # Keep merged in ccl_large, remove ccl_small
                self.framework_info.genera_info[ccl_large] = merged_gi
                self.framework_info.genera_info.del(ccl_small)

    return ans

method next_state*(self: BandMove, action: string): BandMove {.base.} =
    let action_type = get_act_type(action)
    if action_type == "end":
        var next_state = self.copy()
        var (c, s) = get_act_strand(action)
        var new_link = next_state.end_band(c, s, get_act_sign(action))
        # a function from ccl (connected component label) to
        # the number of link components with the label
        var ccl_count = initTable[int, int]()
        for comp in new_link.link_components:
            let ccl = comp.extra_info
            if ccl notin ccl_count:
                ccl_count[ccl] = 0
            ccl_count[ccl] += 1
        let link_num_before_simplification = new_link.linking_number()
        # Compute simplified link once for features/rewards
        var new_simplified_link = new_link.copy()
        discard new_simplified_link.simplify(SimplificationMode.global)
        # sanity check: Spherogram simplification should
        # not change the orientation of components
        assert (new_simplified_link.linking_number() ==
                link_num_before_simplification)
        # deduce the ccl's of the unlinked components by
        # subtraction (use simplified link for this)
        for comp in new_simplified_link.link_components:
            let ccl = comp.extra_info
            assert ccl in ccl_count
            ccl_count[ccl] -= 1
            if ccl_count[ccl] == 0:
                ccl_count.del(ccl)
        var unlinked_ccl = newSeq[int]()
        for (ccl, count) in ccl_count.pairs:
            for _ in 0 ..< count:
                unlinked_ccl.add(ccl)
        if not self.simplification_switch:
            # Keep original link unsimplified apart from its
            # Reidemeister I kinks, pass pre-computed
            # simplified_link. The labels deduced above
            # describe the simplified link, which moves split
            # unknots out of link_components; the unsimplified
            # link keeps every component there, so its unlinked
            # unknots are the ones it inherited from the current
            # state, plus any component the kink removal left
            # without crossings
            var unsimplified_ccl = next_state.framework_info.unlinked_ccl.get(newSeq[int]())
            unsimplified_ccl &= remove_reidemeister_i_kinks(new_link)
            var new_state = bandmove_from_link(new_link,
                                               self.max_twists,
                                               self.framework,
                                               some(unsimplified_ccl),
                                               self.simplification_switch,
                                               some(new_simplified_link))
            new_state.framework_info.genera_info = next_state.framework_info.genera_info
            return new_state
        else:
            # Use simplified link
            var new_state = bandmove_from_link(new_simplified_link,
                                                self.max_twists,
                                                self.framework,
                                                some(unlinked_ccl),
                                                self.simplification_switch,
                                                some(new_simplified_link))
            new_state.framework_info.genera_info = next_state.framework_info.genera_info
            return new_state
    if action_type == "birth":
        if self.check_birth_error():
            raise newException(ValueError,
                &"action {action} is illegal on state {self}")
        # fresh ccl: strictly larger than every label ever used
        var fresh_ccl = -1
        for ccl in self.framework_info.genera_info.keys:
            fresh_ccl = max(fresh_ccl, ccl)
        var unlinked_ccl = self.framework_info.unlinked_ccl.get(newSeq[int]())
        for ccl in unlinked_ccl:
            fresh_ccl = max(fresh_ccl, ccl)
        fresh_ccl += 1
        unlinked_ccl.add(fresh_ccl)
        # the diagram itself is unchanged; only the
        # unlinked-unknot counter grows
        var new_link = self.link.copy()
        new_link.unlinked_unknot_components += 1
        var new_simplified_link = self.simplified_link.copy()
        new_simplified_link.unlinked_unknot_components += 1
        var new_state = bandmove_from_link(new_link,
                                           self.max_twists,
                                           self.framework,
                                           some(unlinked_ccl),
                                           self.simplification_switch,
                                           some(new_simplified_link))
        # preserve the accumulated genera_info (the re-init in
        # bandmove_from_link only knows the joined link) and
        # register the birthed unknot as its own surface
        # component
        var gi = self.framework_info.genera_info
        gi[fresh_ccl] = (num_end_actions: 0, num_init_comps: 1,
                         num_curr_comps: 1, num_orig_comps: 0)
        new_state.framework_info.genera_info = gi
        return new_state
    return bandmove_from_band_twist(BandTwist(self).next_state(action))

proc random_walk_transition*(self: BandMove,
                             p_twist: float,
                             p_end: float): Option[string] =
    # if this were a method, it would run into an infinite
    # loop with band.BandTwist.random_walk_transition
    let provisional_move0 = BandTwist(self).random_walk_transition(p_twist, p_end)
    if provisional_move0.isNone:
        return none(string)
    let provisional_move = provisional_move0.get()
    if get_act_type(provisional_move) == "end":
        # We have not checked any orientation so far,
        # since if the final strand is legal, there is
        # a sign for the end move that is legal.
        # We determine which end signs (among end0,
        # end1, end2) are legal and choose one of the
        # legal ones with equal probability
        let (c, s) = get_act_strand(provisional_move)
        let can_have_sign = collect:
            for sign in [0, 1, -1]:
                not self.check_end_error(c, s, sign)
        return some(&"{provisional_move}{randomly_choose_one(can_have_sign)}")
    return some(provisional_move)

method compute_current_answer*(self: BandMove,
                               init_link: Link[int],
                               num_end_actions: int): int =
    # For (strong) ribbon and (strong) slice we use
    # genera_info to compute the genus by iterating over
    # connected components of the surface (identified by ccl)
    # At the terminal unlink, cap and discard pure-birth components.
    # Keep their bookkeeping: before termination they may still fuse
    # with a component containing part of the original link.
    let discard_pure_birth = self.framework in ["slice", "strong slice"] and
                             self.is_terminal()
    var total_genus = 0
    for (ccl, gi) in self.framework_info.genera_info.pairs:
        # Each band attachment decreases Euler characteristic by 1
        # Initial Euler char is 0 (disjoint union of circles)
        # genus_x2 = 2 - EulerChar - BoundaryComps
        #          = 2 + num_end_actions - (num_init_comps + num_curr_comps)
        let total_bound_comp = gi.num_init_comps + gi.num_curr_comps
        let genus_x2 = 2 + gi.num_end_actions - total_bound_comp
        if genus_x2 mod 2 != 0:
            raise newException(ValueError,
                &"Current genus of ccl {ccl} is not integer: " &
                &"num_end_actions={gi.num_end_actions}, " &
                &"num_init_comps={gi.num_init_comps}, " &
                &"num_curr_comps={gi.num_curr_comps}")
        if genus_x2 < 0:
            raise newException(ValueError,
                &"Current genus of ccl {ccl} is not positive: " &
                &"num_end_actions={gi.num_end_actions}, " &
                &"num_init_comps={gi.num_init_comps}, " &
                &"num_curr_comps={gi.num_curr_comps}")
        if not discard_pure_birth or gi.num_orig_comps > 0:
            total_genus += genus_x2 div 2
    return total_genus

proc has_unfused_birth_components*(self: BandMove): bool =
    discard """
    Tells whether the surface has any birth component that was
    never fused with an original link component, i.e. a CCL in
    genera_info with num_orig_comps == 0 (a pure-birth surface
    component, possibly of positive genus). Only meaningful for
    the slice frameworks. These components are discarded from the
    terminal answer, but remain in the bookkeeping.

    Returns:
        Whether some surface component is a pure-birth comp.
    """
    for gi in self.framework_info.genera_info.values:
        if gi.num_orig_comps == 0:
            return true
    return false

method terminal_state_message*(self: BandMove,
                               num_end_actions: int,
                               answer: int): string =
    if self.framework in ["strong ribbon", "strong slice"]:
        # For the strong frameworks, report the number of
        # retained connected components of the terminal surface
        var num_components_surface = 0
        for gi in self.framework_info.genera_info.values:
            if self.framework != "strong slice" or gi.num_orig_comps > 0:
                num_components_surface += 1
        return &"reached unlink with {num_end_actions} band moves, resulting in a {self.framework} surface of genus {answer} with {num_components_surface} components"
    else:
        # For ribbon / slice, report number of unlink components
        let num_unlink_components = self.link.link_components.len +
                                    self.link.unlinked_unknot_components
        return &"reached unlink with {num_end_actions} band moves, resulting in a {self.framework} surface of genus {answer} with {num_unlink_components} unlink components"

method cls_name*(self: BandMove): string = "BandMove"
