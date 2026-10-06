import std/unittest
import std/options
import std/strutils
import std/tables

import "../spherogram-nim/src/links"

import ../src/state
import ../src/band

test "find_strand_in_face_with_most_edges":
    expect AssertionDefect:
        discard find_strand_in_face_with_most_edges(link_from_PD_code(@[]))
    check find_strand_in_face_with_most_edges(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])) == (0, 1)
    var knot62 = link_from_PD_code(@[[11, 7, 0, 6], [7, 1, 8, 0], [1, 9, 2, 8], [5, 3, 6, 2], [3, 10, 4, 11], [9, 4, 10, 5]])
    check find_strand_in_face_with_most_edges(knot62) == (0, 0)

test "join_split_link_diagrams":
    var two_trefoils = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5],
                                           [11, 8, 6, 9], [9, 6, 10, 7], [7, 10, 8, 11]])
    var two_trefoils_joined = join_split_link_diagrams(two_trefoils)
    check (two_trefoils_joined.PD_code() ==
           @[[7, 4, 0, 5], [5, 0, 6, 1], [1, 6, 2, 7],
             [15, 12, 8, 13], [13, 8, 14, 9], [9, 14, 10, 15],
             [3, 10, 4, 11], [2, 12, 3, 11]])
    check two_trefoils_joined.signs == @[-1, -1, -1, -1, -1, -1, -1, 1]

test "crosschange_from_band_twist":
    var band = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    band.start(0, 0)
    var state = crosschange_from_band_twist(band)
    check typeof(state) is CrossChange
    check state.band_path == @[[0, 0, 0, 0]]

test "crosschange_from_link":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state = crosschange_from_link(trefoil, 1, "splitting")
    # check that crosschange_from_link has not mutated its argument
    check trefoil.signs == @[-1, -1, -1]
    check typeof(state) is CrossChange
    # check that state.link preserves the crossing signs
    for c in 0 ..< 3:
        check state.link.signs[c] == -1
    # check that the parameter max_twists has indeed been
    # passed to band.BandTwist.from_link
    state.start(0, 0)
    state.twist(1)
    check state.check_twist_error(1)
    # check the same for the parameter framework
    check state.check_end_errors(0, 3)

test "CrossChange.copy":
    var state = crosschange_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    var state_copy = state.copy()
    # check that the correct copy method has been called
    check typeof(state_copy) is CrossChange
    # we execute moves on the copy state and check that
    # the original state is unaffected
    state_copy.start(0, 0)
    state_copy = state_copy.next_state("C1S3_end2")
    check state_copy.is_unlink()
    check not state.is_unlink()

test "cross_change_PD_code":
    # crossing changes on trefoil
    var state = crosschange_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))

    # no going over/under, no twists, end1
    state.update(@[[2, 1, 0, 0], [2, 0, 1, 0]], @[])
    check (state.cross_change_PD_code() == (@[
        [5, 2, 0, 3],
        [3, 0, 6, 8],
        [1, 4, 2, 5],
        [7, 4, 1, 9],
        [6, 7, 9, 8]
    ], @[-1, -1, -1, 1, 1]))

    # no going over/under, no twists, end2
    state.update(@[[2, 1, 0, 0], [2, 0, -1, 0]], @[])
    check (state.cross_change_PD_code() == (@[
        [5, 2, 0, 3],
        [3, 0, 6, 8],
        [1, 4, 2, 5],
        [9, 7, 4, 1],
        [8, 6, 7, 9]
    ], @[-1, -1, -1, -1, -1]))

    # no going over/under, one positive twist, end1
    state.update(@[[2, 1, 0, 1], [2, 0, 1, 0]], @[])
    check (state.cross_change_PD_code() == (@[
        [5, 2, 0, 3],
        [3, 0, 6, 10],
        [1, 4, 2, 5],
        [6, 4, 7, 8],
        [9, 7, 1, 11],
        [11, 10, 8, 9]
    ], @[-1, -1, -1, 1, -1, -1]))

    # no going over/under, one negative twist
    state.update(@[[2, 1, 0, -1], [2, 0, 1, 0]], @[])
    check (state.cross_change_PD_code() == (@[
        [5, 2, 0, 3],
        [3, 0, 6, 10],
        [1, 4, 2, 5],
        [8, 6, 4, 7],
        [9, 7, 1, 11],
        [11, 10, 8, 9]
    ], @[-1, -1, -1, -1, -1, -1]))

    # going over once, no twists
    state.update(@[[2, 1, 0, 0], [2, 0, 1, 0], [0, 3, 1, 0]], @[])
    check (state.cross_change_PD_code() == (@[
        [5, 2, 0, 3],
        [12, 0, 6, 8],
        [1, 4, 2, 5],
        [7, 4, 1, 9],
        [8, 6, 7, 10],
        [3, 13, 11, 9],
        [10, 11, 13, 12]
    ], @[-1, -1, -1, 1, -1, -1, -1]))

    # going under once, no twists
    state.update(@[[2, 1, 0, 0], [2, 0, -1, 0], [0, 3, 1, 0]], @[])
    check (state.cross_change_PD_code() == (@[
        [5, 2, 0, 3],
        [12, 0, 6, 8],
        [1, 4, 2, 5],
        [9, 7, 4, 1],
        [6, 7, 10, 8],
        [3, 13, 11, 9],
        [10, 11, 13, 12]
    ], @[-1, -1, -1, -1, 1, -1, -1]))

    # going under once, one negative twist
    state.update(@[[2, 1, 0, -1], [2, 0, -1, 0], [0, 3, 1, 0]], @[])
    check (state.cross_change_PD_code() == (@[
        [5, 2, 0, 3],
        [14, 0, 6, 10],
        [1, 4, 2, 5],
        [8, 6, 4, 7],
        [7, 1, 11, 9],
        [12, 10, 8, 9],
        [3, 15, 13, 11],
        [15, 14, 12, 13]
    ], @[-1, -1, -1, -1, 1, -1, 1, 1]))

test "CrossChange.end_band":
    var trefoil_and_two_unknots = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]], unlinked_unknot_components = 2)
    var state = crosschange_from_link(trefoil_and_two_unknots)
    state.start(0, 1)
    var new_link = state.end_band(0, 0, -1)
    check state.band_path == @[[0, 1, 0, 0], [0, 0, -1, 0]]
    check new_link.crossings.len == 5
    check new_link.unlinked_unknot_components == 2

test "CrossChange.next_state":
    # trefoil knot
    # the knot obtained after this crossing change
    # cannot have a diagram with fewer than 5 crossings
    var state = crosschange_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    state = state.next_state("C0S1_start").next_state("C0S0_end2")
    check state.link.crossings.len == 5
    # this sequence of actions should change the trefoil
    # knot to the unknot
    state = crosschange_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    state = state.next_state("C0S1_start").next_state("C0S0_end1")
    check state.is_unlink()

    # 5_2 knot
    # this sequence of actions should change the 5_2
    # knot to the trefoil knot
    state = crosschange_from_link(link_from_PD_code(@[[4, 0, 5, 9], [0, 6, 1, 5], [8, 2, 9, 1], [2, 8, 3, 7], [6, 4, 7, 3]]))
    state = state.next_state("C4S3_start").next_state("C4S2_end1")
    check state.link.crossings.len == 3
    # this sequence of actions should change the 5_2
    # knot to the unknot
    state = crosschange_from_link(link_from_PD_code(@[[4, 0, 5, 9], [0, 6, 1, 5], [8, 2, 9, 1], [2, 8, 3, 7], [6, 4, 7, 3]]))
    state = state.next_state("C0S2_start").next_state("C0S1_end2")
    check state.is_unlink()

test "CrossChange.random_walk_transition":
    var state = crosschange_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    # in order to end, we must first start
    state.start(0, 0)
    var end1_count = 0
    var end2_count = 0
    for _ in 0 ..< 50:
        var label = state.random_walk_transition(1.0, 5.0).get()
        # we cannot have an unsigned end action
        check not label.endsWith("end")
        if label.endsWith("end1"):
            end1_count += 1
        if label.endsWith("end2"):
            end2_count += 1
    check end1_count > 0 and end2_count > 0

test "bandmove_from_band_twist":
    var band = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    band.start(0, 0)
    var state = bandmove_from_band_twist(band)
    check typeof(state) is BandMove
    check state.band_path == @[[0, 0, 0, 0]]

test "join_unlinked_unknot_component":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var unlinked_ccl = @[0]
    expect AssertionDefect:
        discard join_unlinked_unknot_component(trefoil, unlinked_ccl)
    var trefoil_and_unknot = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]],
                                               unlinked_unknot_components = 1)
    unlinked_ccl = @[2]
    var joined = join_unlinked_unknot_component(trefoil_and_unknot, unlinked_ccl)
    check joined.PD_code() == @[[7, 4, 0, 5], [5, 0, 6, 1], [1, 6, 2, 7], [9, 2, 8, 3], [8, 4, 9, 3]]
    check joined.signs == @[-1, -1, -1, -1, 1]
    check joined.unlinked_unknot_components == 0
    check joined.link_components == @[newLinkComponent(0, 2, 0),
                                      newLinkComponent(3, 2, 2)]
    check unlinked_ccl == []

test "join_unlinked_unknot_components":
    var two_unknots = link_from_PD_code(@[], unlinked_unknot_components = 2)
    var unlinked_ccl = @[1]
    expect AssertionDefect:
        discard join_unlinked_unknot_components(two_unknots, unlinked_ccl)
    unlinked_ccl = @[1, 2]
    var joined = join_unlinked_unknot_components(two_unknots, unlinked_ccl)
    check joined.crossings == []
    check joined.unlinked_unknot_components == 2
    check unlinked_ccl == @[1, 2]
    var trefoil_and_two_unknots = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]],
                                                    unlinked_unknot_components = 2)
    unlinked_ccl = @[1, 3]
    joined = join_unlinked_unknot_components(trefoil_and_two_unknots, unlinked_ccl)
    check joined.crossings.len == 7
    check joined.unlinked_unknot_components == 0
    check joined.link_components == @[newLinkComponent(0, 2, 0),
                                      newLinkComponent(3, 2, 3),
                                      newLinkComponent(5, 2, 1)]
    check unlinked_ccl == []

test "bandmove_from_link":
    let hopf_positive = link_from_PD_code(@[[2, 1, 3, 0], [1, 2, 0, 3]])
    var state = bandmove_from_link(hopf_positive)
    # check that bandmove_from_link has not mutated its argument
    check hopf_positive.link_components[1].extra_info == 0
    check state.link.link_components == @[newLinkComponent(0, 1, 0),
                                          newLinkComponent(0, 2, 1)]

    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]],
                                    unlinked_unknot_components = 2)
    trefoil.link_components[0].extra_info = 2
    state = bandmove_from_link(trefoil, unlinked_ccl0 = some(@[3, 1]))
    check state.link.link_components == @[newLinkComponent(0, 2, 2),
                                          newLinkComponent(3, 2, 1),
                                          newLinkComponent(5, 2, 3)]

test "BandMove.copy":
    let hopf_positive = link_from_PD_code(@[[2, 1, 3, 0], [1, 2, 0, 3]])
    var state = bandmove_from_link(hopf_positive)
    var state_copy = state.copy()
    # we execute moves on the copy state and check that
    # the original state is unaffected
    state_copy.start(0, 1)
    state_copy = state_copy.next_state("C0S0_end0")
    check state_copy.is_unlink()
    check not state.is_unlink()

test "check_end_error":
    let hopf_positive = link_from_PD_code(@[[2, 1, 3, 0], [1, 2, 0, 3]])
    var state = bandmove_from_link(hopf_positive)
    state.start(0, 1)
    state.twist(-1)
    check state.check_end_error(0, 0, 1)
    check state.check_end_error(0, 0, 0)
    check not state.check_end_error(0, 0, -1)

    state = bandmove_from_link(hopf_positive)
    state.start(1, 0)
    check state.check_end_error(1, 3, 0)
    check not state.check_end_error(1, 3, 1)
    check not state.check_end_error(1, 3, -1)

    state = bandmove_from_link(hopf_positive)
    state.start(0, 1)
    check not state.check_end_error(0, 0, 0)
    check state.check_end_error(0, 0, 1)
    check state.check_end_error(0, 0, -1)

test "band_move_PD_code":
    # First round of test:
    # band moves on trefoil
    var state = bandmove_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))

    # no going over/under, no twists
    state.update(@[[2, 0, 0, 0], [0, 3, 0, 0]], @[])
    check (state.band_move_PD_code() ==
           (@[[5, 2, 0, 1],
              [3, 0, 4, 3],
              [1, 4, 2, 5]],
            @[-1, -1, -1]))

    # no going over/under, one positive twist
    state.update(@[[2, 1, 0, 1], [2, 0, 0, 0]], @[])
    check (state.band_move_PD_code() ==
           (@[[5, 2, 0, 3],
              [3, 0, 6, 1],
              [7, 4, 2, 5],
              [6, 4, 7, 1]],
            @[-1, -1, -1, 1]))

    # no going over/under, one negative twist
    state.update(@[[2, 1, 0, -1], [2, 0, 0, 0]], @[])
    check (state.band_move_PD_code() ==
           (@[[5, 2, 0, 3],
              [3, 0, 6, 1],
              [7, 4, 2, 5],
              [1, 6, 4, 7]],
            @[-1, -1, -1, -1]))

    # going over once, no twists
    state.update(@[[2, 1, 0, 0], [2, 0, 1, 0], [0, 3, 0, 0]], @[])
    check (state.band_move_PD_code() ==
           (@[[5, 2, 0, 9],
              [3, 0, 6, 8],
              [1, 4, 2, 5],
              [7, 4, 1, 9],
              [8, 6, 7, 3]],
            @[-1, -1, -1, 1, -1]))

    # going under once, no twists
    state.update(@[[2, 1, 0, 0], [2, 0, -1, 0], [0, 3, 0, 0]], @[])
    check (state.band_move_PD_code() ==
           (@[[5, 2, 0, 9],
              [3, 0, 6, 8],
              [1, 4, 2, 5],
              [9, 7, 4, 1],
              [6, 7, 3, 8]],
            @[-1, -1, -1, -1, 1]))

    # going over once, two positive twists
    state.update(@[[2, 1, 0, 2], [2, 0, 1, 0], [0, 3, 0, 0]], @[])
    check (state.band_move_PD_code() ==
           (@[[5, 2, 0, 13],
              [3, 0, 6, 12],
              [1, 4, 2, 5],
              [6, 4, 7, 8],
              [9, 10, 8, 7],
              [11, 9, 1, 13],
              [12, 10, 11, 3]],
            @[-1, -1, -1, 1, 1, 1, -1]))

    # going over once, two negative twists
    state.update(@[[2, 1, 0, -2], [2, 0, 1, 0], [0, 3, 0, 0]], @[])
    check (state.band_move_PD_code() ==
           (@[[5, 2, 0, 13],
              [3, 0, 6, 12],
              [1, 4, 2, 5],
              [8, 6, 4, 7],
              [7, 9, 10, 8],
              [11, 9, 1, 13],
              [12, 10, 11, 3]],
            @[-1, -1, -1, -1, -1, 1, -1]))

test "BandMove.end_band":
    let hopf_positive = link_from_PD_code(@[[2, 1, 3, 0], [1, 2, 0, 3]])
    var state = bandmove_from_link(hopf_positive)
    state.start(0, 2)
    discard state.end_band(0, 1, -1)
    check state.band_path == @[[0, 2, 0, -1], [0, 1, 0, 0]]

    # fission band move on the Hopf link
    state = bandmove_from_link(hopf_positive)
    state.start(0, 3)
    state.over_under(0, 2, -1)
    var new_link = state.end_band(0, 1, 0)
    check state.band_path == @[[0, 3, 0, 0], [0, 2, -1, 0], [0, 1, 0, 0]]
    check (new_link.link_components == 
           @[newLinkComponent(0, 2, 1),
             newLinkComponent(1, 2, 0),
             newLinkComponent(0, 1, 0)])

    # fusion band move on the Hopf link; the fused component
    # keeps the larger of the two ccls (merge convention)
    state = bandmove_from_link(hopf_positive)
    state.start(0, 1)
    new_link = state.end_band(0, 0, 0)
    check state.band_path == @[[0, 1, 0, 0], [0, 0, 0, 0]]
    check new_link.link_components == @[newLinkComponent(0, 1, 1)]

test "BandMove.next_state":
    # fuse the two components of the Hopf link to an unknot
    # through one band move
    let hopf_positive = link_from_PD_code(@[[2, 1, 3, 0], [1, 2, 0, 3]])
    var state = bandmove_from_link(hopf_positive)
    state = state.next_state("C0S1_start").next_state("C0S0_end0")
    check state.is_unlink()

    # turn the trefoil knot into the unknot through one
    # fission and one fusion band move
    state = bandmove_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    state = state.next_state("C0S0_start").next_state("C0S3_end0")
    check state.link.crossings == [[7, 6, 5, 4], [3, 2, 1, 0]]
    check state.link.signs == [-1, -1]
    check (state.link.link_components ==
           @[newLinkComponent(0, 3, 0),
             newLinkComponent(0, 2, 0)])
    state = state.next_state("C0S0_start").next_state("C0S3_end0")
    check state.is_unlink()

test "BandMove.random_walk_transition":
    let hopf_positive = link_from_PD_code(@[[2, 1, 3, 0], [1, 2, 0, 3]])
    var state = bandmove_from_link(hopf_positive)
    state.start(0, 2)
    var end1_count = 0
    var end2_count = 0
    for _ in 0 ..< 50:
        var label = state.random_walk_transition(1.0, 5.0).get()
        if get_act_type(label) == "end":
            var sign = get_act_sign(label)
            check sign != 0
            if sign == 1:
                end1_count += 1
            elif sign == -1:
                end2_count += 1
    check end1_count > 0 and end2_count > 0
    state.twist(-1)
    var end0_count = 0
    for _ in 0 ..< 50:
        var label = state.random_walk_transition(1.0, 5.0).get()
        if get_act_type(label) == "end":
            var sign = get_act_sign(label)
            check sign == 0
            end0_count += 1
    check end0_count > 0

test "BandMove.compute_current_answer":
    # the genus is computed from genera_info, which is only
    # updated by end_band in the surface-genus frameworks

    # fuse the two components of the Hopf link to an unknot
    # through one band move; before any band the surface is a
    # disjoint union of annuli, of total genus 0
    let hopf_positive = link_from_PD_code(@[[2, 1, 3, 0], [1, 2, 0, 3]])
    var state = bandmove_from_link(hopf_positive, framework = "ribbon")
    check state.compute_current_answer(hopf_positive, 0) == 0
    state = state.next_state("C0S1_start").next_state("C0S0_end0")
    check state.compute_current_answer(hopf_positive, 1) == 0

    # turn the trefoil knot into the unknot through one
    # fission and one fusion band move
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    state = bandmove_from_link(trefoil, framework = "ribbon")
    check state.compute_current_answer(trefoil, 0) == 0
    state = state.next_state("C0S0_start").next_state("C0S3_end0")
    check state.compute_current_answer(trefoil, 1) == 0
    state = state.next_state("C0S0_start").next_state("C0S3_end0")
    check state.compute_current_answer(trefoil, 2) == 1

test "BandMove discards pure-birth genus only at a terminal slice state":
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    # Synthetic surface histories: two retained components of genera 1
    # and 3, a pure-birth component of genus 2, and an untouched birth.
    # The first retained component includes a birth fused to an original.
    let histories = {
        0: (num_end_actions: 4, num_init_comps: 2, num_curr_comps: 2, num_orig_comps: 1),
        1: (num_end_actions: 6, num_init_comps: 1, num_curr_comps: 1, num_orig_comps: 1),
        2: (num_end_actions: 6, num_init_comps: 2, num_curr_comps: 2, num_orig_comps: 0),
        3: (num_end_actions: 0, num_init_comps: 1, num_curr_comps: 1, num_orig_comps: 0)
    }.toTable
    for framework in ["slice", "strong slice"]:
        var nonterminal = bandmove_from_link(trefoil, framework = framework)
        check not nonterminal.is_terminal()
        nonterminal.framework_info.genera_info = histories
        check nonterminal.compute_current_answer(trefoil, 0) == 6
        check nonterminal.framework_info.genera_info == histories

        var terminal = bandmove_from_link(trefoil, framework = framework)
        terminal = terminal.next_state("C0S0_start").next_state("C0S3_end0")
        terminal = terminal.next_state("C0S0_start").next_state("C0S3_end0")
        check terminal.is_terminal()
        terminal.framework_info.genera_info = histories
        check terminal.compute_current_answer(trefoil, 2) == 4
        check terminal.framework_info.genera_info == histories
        check terminal.has_unfused_birth_components()
        if framework == "strong slice":
            check terminal.terminal_state_message(2, 4).endsWith("with 2 components")

test "BandMove terminal birth filtering does not change ribbon accounting":
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    for framework in ["ribbon", "strong ribbon"]:
        var state = bandmove_from_link(trefoil, framework = framework)
        state = state.next_state("C0S0_start").next_state("C0S3_end0")
        state = state.next_state("C0S0_start").next_state("C0S3_end0")
        check state.is_terminal()
        # Births are not legal ribbon moves; a synthetic entry makes the
        # framework restriction of the terminal filter explicit.
        state.framework_info.genera_info[1] = (2, 1, 1, 0)
        check state.compute_current_answer(trefoil, 2) == 2
        if framework == "strong ribbon":
            check state.terminal_state_message(2, 2).endsWith("with 2 components")

test "BandMove validates terminal pure-birth genus before discarding it":
    let trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    for framework in ["slice", "strong slice"]:
        var state = bandmove_from_link(trefoil, framework = framework)
        state = state.next_state("C0S0_start").next_state("C0S3_end0")
        state = state.next_state("C0S0_start").next_state("C0S3_end0")
        check state.is_terminal()
        state.framework_info.genera_info[1] = (1, 1, 1, 0)
        expect ValueError:
            discard state.compute_current_answer(trefoil, 2)
        state.framework_info.genera_info[1] = (0, 2, 2, 0)
        expect ValueError:
            discard state.compute_current_answer(trefoil, 2)
