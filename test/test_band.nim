import std/unittest
import std/options
import std/tables

import "../spherogram-nim/src/links"

import ../src/band

test "get_act_type":
    let testcases = [("C0S3_start", "start"),
                     ("C1S2_over", "over"),
                     ("C1S1_under", "under"),
                     ("ccw_twist", "twist"),
                     ("cw_twist", "twist"),
                     ("C4S0_end1", "end"),
                     ("C4S0_end2", "end"),
                     ("unknot_birth", "birth")]
    for (label, answer) in testcases:
        check get_act_type(label) == answer

test "get_act_strand":
    let testcases = [("C0S3_start", (0, 3)),
                     ("C1S2_over", (1, 2)),
                     ("C1S1_under", (1, 1)),
                     ("C4S0_end1", (4, 0)),
                     ("C4S0_end2", (4, 0))]
    for (label, answer) in testcases:
        check get_act_strand(label) == answer
    expect ValueError:
        discard get_act_strand("ccw_twist")
    expect ValueError:
        discard get_act_strand("cw_twist")
    expect ValueError:
        discard get_act_strand("unknot_birth")

test "get_act_sign":
    let testcases = [("C1S2_over", 1),
                     ("C1S1_under", -1),
                     ("ccw_twist", 1),
                     ("cw_twist", -1),
                     ("C4S0_end1", 1),
                     ("C4S0_end2", -1),
                     ("C4S0_end0", 0)]
    for (label, answer) in testcases:
        check get_act_sign(label) == answer
    expect ValueError:
        discard get_act_sign("C0S3_start")

test "randomly_choose_one":
    expect AssertionDefect:
        discard randomly_choose_one(@[])
    expect AssertionDefect:
        discard randomly_choose_one(@[false])
    expect AssertionDefect:
        discard randomly_choose_one(@[false, false])
    for _ in 0 ..< 5:
        check randomly_choose_one(@[true]) == 0
    for _ in 0 ..< 5:
        check randomly_choose_one(@[false, true]) == 1
    var choose_count = [0, 0]
    for _ in 0 ..< 50:
        choose_count[randomly_choose_one(@[true, true])] += 1
    check choose_count[0] > 0
    check choose_count[1] > 0

test "newFrameworkInfo":
    var framework_info = newFrameworkInfo(1, some(@[-1, 1]), none(seq[int]))
    check framework_info.num_conn_comp == 1
    check framework_info.alex_poly == some(@[-1, 1])
    check framework_info.unlinked_ccl.isNone

test "newBandTwist":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    expect ValueError:
        discard newBandTwist(@[], @[], trefoil, @[], @[], 5, "other", newFrameworkInfo(1, none(seq[int]), none(seq[int])))

test "bandtwist_from_link":
    # check that links with possible Reidemeister I simplifications are disallowed
    var one_vertex_unknot = link_from_PD_code(@[[0, 0, 1, 1]])
    expect ValueError:
        discard bandtwist_from_link(one_vertex_unknot)

    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state = bandtwist_from_link(trefoil)
    # check crossing strands of faces
    check state.faces[0].subdivisions == @[@[(0, 0), (2, 2)]]
    check state.faces[1].subdivisions == @[@[(0, 1), (2, 1), (1, 1)]]
    check state.faces[2].subdivisions == @[@[(0, 2), (1, 0)]]
    check state.faces[3].subdivisions == @[@[(0, 3), (1, 3), (2, 3)]]
    check state.faces[4].subdivisions == @[@[(1, 2), (2, 0)]]
    # check state.face0
    check state.face0 == @[[0, 1, 2, 3], [2, 1, 4, 3], [4, 1, 0, 3]]
    # check that the starting strand is not set in any face
    for face in state.faces:
        check face.initial_strand.isNone
    # check state.pd_code
    check (state.pd_code ==
           @[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])

test "copy":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    var state_copy = state.copy()
    # we execute moves on the copy state and check that
    # the original state is unaffected
    state_copy.start(0, 0)
    state_copy.twist(1)
    state_copy.over_under(1, 3, -1)
    state_copy.over_under(1, 2, 1)
    check state.band_path == []

test "update":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    var state_copy = state.copy()
    state_copy.start(0, 0)
    state_copy.over_under(1, 3, -1)
    # we check that state.update has indeed updated the
    # state
    state.update(state_copy.band_path, state_copy.faces)
    check state.band_path == @[[0, 0, 0, 0], [1, 3, -1, 0]]
    check state.faces[3].subdivisions == @[@[(0, 3)]]
    check state.faces[0].initial_strand == some((0, 0))
    # we check that updating state does not affect
    # state_copy
    state.over_under(1, 2, 1)
    check state_copy.band_path == @[[0, 0, 0, 0], [1, 3, -1, 0]]

test "check_strand_error":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    for crossing in 0 ..< 3:
        for strand_index in 0 ..< 4:
            check not state.check_strand_error(crossing, strand_index)
    check state.check_strand_error(-1, 2)
    check state.check_strand_error(3, 2)
    check state.check_strand_error(1, -1)
    check state.check_strand_error(1, 4)

test "check_start_error":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    for crossing in 0 ..< 3:
        for strand_index in 0 ..< 4:
            check not state.check_start_error(crossing, strand_index)
    state.start(2, 0)
    for crossing in 0 ..< 3:
        for strand_index in 0 ..< 4:
            check state.check_start_error(crossing, strand_index)

test "start":
    for crossing in 0 ..< 3:
        for strand_index in 0 ..< 4:
            var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
            state.start(crossing, strand_index)
            check state.band_path == @[[crossing, strand_index, 0, 0]]
    # check that the initial strand is correctly set
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    state.start(0, 0)
    check state.faces[0].initial_strand == some((0, 0))

test "feasible_next_strands":
    var state = bandtwist_from_link(link_from_PD_code(@[[9, 4, 0, 5], [5, 0, 6, 1], [1, 6, 2, 7], [7, 2, 8, 3], [3, 8, 4, 9]]))
    state.start(2, 0)
    state.over_under(4, 3, -1)
    state.over_under(4, 2, -1)
    state.over_under(4, 1, -1)
    state.over_under(4, 0, -1)
    check state.feasible_next_strands((4, 0)) == @[(2, 3)]

test "check_over_under_error":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    # we should not be able to make over / under moves
    # if the band has not started
    for crossing in 0 ..< 3:
        for strand_index in 0 ..< 4:
            check state.check_over_under_error(crossing, strand_index)
    state.start(1, 0)
    # we cannot go back through the previous edge
    check state.check_over_under_error(0, 3)
    # the next strand must start on the current face
    check state.check_over_under_error(2, 2)
    state.over_under(1, 3, -1)
    state.over_under(1, 2, -1)
    state.over_under(0, 1, -1)
    # we cannot find an edge to end the band at if we
    # went through this strand
    check state.check_over_under_error(0, 0)

    state = bandtwist_from_link(link_from_PD_code(@[[4, 0, 5, 9], [0, 6, 1, 5], [8, 2, 9, 1], [2, 8, 3, 7], [6, 4, 7, 3]]))
    state.start(0, 1)
    state.over_under(0, 0, -1)
    state.over_under(0, 3, -1)
    state.over_under(1, 2, -1)
    state.over_under(3, 1, -1)
    # we can enter the face again through a different
    # edge
    check not state.check_over_under_error(3, 0)
    state.over_under(3, 0, -1)
    # we can then exit the face again through the only
    # uncrossed edge
    check not state.check_over_under_error(3, 3)
    # all the other edges have been crossed
    check state.check_over_under_error(4, 1)
    check state.check_over_under_error(0, 3)
    check state.check_over_under_error(2, 1)

    state = bandtwist_from_link(link_from_PD_code(@[[9, 4, 0, 5], [5, 0, 6, 1], [1, 6, 2, 7], [7, 2, 8, 3], [3, 8, 4, 9]]))
    state.start(2, 0)
    state.over_under(4, 3, -1)
    state.over_under(4, 2, -1)
    state.over_under(4, 1, -1)
    state.over_under(4, 0, -1)
    # we cannot cross the existing band
    check state.check_over_under_error(0, 3)
    # we cannot cross the starting strand
    check state.check_over_under_error(1, 3)
    # but we can go through the next strand clockwise
    # because it is uncrossed
    check not state.check_over_under_error(2, 3)

test "cut_band_through_face":
    var state = bandtwist_from_link(link_from_PD_code(@[[9, 4, 0, 5], [5, 0, 6, 1], [1, 6, 2, 7], [7, 2, 8, 3], [3, 8, 4, 9]]))
    state.start(2, 0)
    state.cut_band_through_face((2, 0), (3, 3))
    check state.faces[3].subdivisions == @[@[(2, 3)], @[(4, 3), (0, 3)]]

test "over_under":
    for sign in [-1, 1]:
        var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
        state.start(0, 0)
        state.twist(1)
        # check that a wrong twist sign (0) results in a ValueError
        expect ValueError:
            state.over_under(0, 3, 0)
        # check that state.band_path has been correctly updated after the twist
        state.over_under(0, 3, sign)
        check state.band_path == @[[0, 0, 0, 1], [0, 3, sign, 0]]

test "check_twist_error":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]), 3)
    # we should not be able to make twists if the band
    # has not started
    check state.check_twist_error(1)
    check state.check_twist_error(-1)
    state.start(0, 0)
    state.twist(1)
    # we cannot reverse previous twists
    check not state.check_twist_error(1)
    check state.check_twist_error(-1)
    state.twist(1)
    # we cannot exceed max_twists
    check not state.check_twist_error(1)
    state.twist(1)
    check state.check_twist_error(1)
    # we cannot reverse previous twists
    state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    state.start(0, 0)
    state.twist(-1)
    check not state.check_twist_error(-1)
    check state.check_twist_error(1)

test "twist":
    for sign in [-1, 1]:
        var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
        state.start(0, 0)
        state.twist(sign)
        check state.band_path == @[[0, 0, 0, sign]]

test "initial_strand_component":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    expect AssertionDefect:
        discard state.initial_strand_component()

    state.start(0, 0)
    check (state.initial_strand_component() ==
           @[(0, 0), (2, 3), (2, 1), (1, 2),
             (1, 0), (0, 3), (0, 1), (2, 2),
             (2, 0), (1, 3), (1, 1), (0, 2)])

test "check_end_errors":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    # we should not be able to end the band if it has
    # not started
    for crossing in 0 ..< 3:
        for strand_index in 0 ..< 4:
            check state.check_end_errors(crossing, strand_index)
    state.start(0, 0)
    state.over_under(1, 3, -1)
    state.over_under(1, 2, 1)
    # the band would not be able to exit face 1 had it
    # entered it
    check state.check_over_under_error(0, 1)
    # but it can end in face 1
    check not state.check_end_errors(0, 1)

    # check that ending on the same link component as the
    # initial strand is disallowed in (strong) splitting
    # number calculations
    state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]), framework = "splitting")
    state.start(0, 0)
    check state.check_end_errors(1, 3)

test "check_action_error":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    expect ValueError:
        discard state.check_action_error("C0S0_other")
    # births are only legal in the slice frameworks
    check state.check_action_error("unknot_birth")
    var slice_state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]), 5, "slice")
    check not slice_state.check_action_error("unknot_birth")

test "cls_name":
    var state = bandtwist_from_link(link_from_PD_code(@[]))
    check state.cls_name() == "BandTwist"

test "`$`":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    check $state == """BandTwist instance with
framework=unknotting
max twists=5
link=<Link: 1 comp; 3 cross>
band path=@[]
"""

test "next_state":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    expect ValueError:
        discard state.next_state("C0S0_over")
    let next_state = state.next_state("C0S0_start")
    # we have indeed executed the action
    check next_state.band_path == @[[0, 0, 0, 0]]
    # but state should not be changed by next_state
    check state.band_path == []

test "sample_from_actions":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    let new_state = state.sample_from_actions(["C0S0_start",
                                               "cw_twist",
                                               "C0S3_over"])
    # the actions have indeed been executed
    check new_state.band_path == @[[0, 0, 0, -1], [0, 3, 1, 0]]
    # state should not be modified
    check state.band_path == []

test "random_walk_transition":
    var state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    var act_type_count = initTable[string, int]()
    for _ in 0 ..< 50:
        let label = state.random_walk_transition(1.0, 1.0)
        check label.isSome
        let act_type = get_act_type(label.get())
        if act_type notin act_type_count:
            act_type_count[act_type] = 0
        act_type_count[act_type] += 1
    # we must only have start actions
    check act_type_count == {"start": 50}.toTable
    state.start(0, 0)
    act_type_count = initTable[string, int]()
    for _ in 0 ..< 50:
        let label = state.random_walk_transition(1.0, 1.0)
        check label.isSome
        let act_type = get_act_type(label.get())
        if act_type notin act_type_count:
            act_type_count[act_type] = 0
        act_type_count[act_type] += 1
    # we must have, and only have, all the types of
    # actions except start
    for act_type in ["over", "under", "twist", "end"]:
        check act_type in act_type_count and act_type_count[act_type] > 0
    check act_type_count.len == 4

test "is_unlink":
    var state = bandtwist_from_link(link_from_PD_code(@[]))
    check state.is_unlink()
    state = bandtwist_from_link(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    check not state.is_unlink()
