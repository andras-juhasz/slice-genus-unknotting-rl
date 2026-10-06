import std/unittest
import std/options
import std/re

import "../spherogram-nim/src/links"
import arraymancer

import ../src/state
import ../src/representable
import ../src/config
import ../src/subdivisions
import ../src/band
import ../src/features
import ../src/utils

test "compute_features":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state1 = crosschange_from_link(trefoil)
    expect DimensionExceededError:
        discard compute_features(state1, [100, 0], @[])
    expect DimensionExceededError:
        discard compute_features(state1, [2, 50], @[])
    var res = compute_features(state1, [3, 1], @[])
    check (res ==
           @[("PD_code", [[5, 2, 0, 3],
                          [3, 0, 4, 1],
                          [1, 4, 2, 5]].toTensor().astype(float)),
             ("unlinked_unknot_components", [0].toTensor().astype(float)),
             ("num_link_components", [1].toTensor().astype(float)),
             ("num_connected_components", [1].toTensor().astype(float)),
             ("gauss_code", [-1, 2, -3, 1, -2, 3, -500].toTensor().astype(float)),
             ("crossing_signs", [-1, -1, -1].toTensor().astype(float)),
             ("writhe", [-3].toTensor().astype(float))])
    # check that the padding of features works as expected
    res = compute_features(state1, [4, 2], @[])
    check (res ==
           @[("PD_code", [[5, 2, 0, 3],
                          [3, 0, 4, 1],
                          [1, 4, 2, 5],
                          [-1000, -1000, -1000, -1000]].toTensor().astype(float)),
             ("unlinked_unknot_components", [0].toTensor().astype(float)),
             ("num_link_components", [1].toTensor().astype(float)),
             ("num_connected_components", [1].toTensor().astype(float)),
             ("gauss_code", [-1, 2, -3, 1, -2, 3, -500, -1000, -1000, -1000].toTensor().astype(float)),
             ("crossing_signs", [-1, -1, -1, -1000].toTensor().astype(float)),
             ("writhe", [-3].toTensor().astype(float))])
    # check that features are correctly calculated for oriented links
    var state2 = bandmove_from_link(trefoil)
    res = compute_features(state2, [3, 1], @[])
    check (res ==
           @[("PD_code", [[5, 2, 0, 3],
                          [3, 0, 4, 1],
                          [1, 4, 2, 5]].toTensor().astype(float)),
             ("unlinked_unknot_components", [0].toTensor().astype(float)),
             ("num_link_components", [1].toTensor().astype(float)),
             ("num_connected_components", [1].toTensor().astype(float)),
             ("gauss_code", [-1, 2, -3, 1, -2, 3, -500].toTensor().astype(float)),
             ("crossing_signs", [-1, -1, -1].toTensor().astype(float)),
             ("writhe", [-3].toTensor().astype(float))] &
           (if hfk_enabled:
                @[("seifert_genus", [1].toTensor().astype(float)),
                  ("nu", [0].toTensor().astype(float)),
                  ("tau", [-1].toTensor().astype(float))]
            else:
                @[]))
    # check the corner case, links with no crossings
    var two_unknots = link_from_PD_code(@[], unlinked_unknot_components = 2)
    state2 = bandmove_from_link(two_unknots)
    res = compute_features(state2, [2, 1], @[])
    check (res ==
           @[("PD_code", [[-1000, -1000, -1000, -1000],
                          [-1000, -1000, -1000, -1000]].toTensor().astype(float)),
             ("unlinked_unknot_components", [2].toTensor().astype(float)),
             ("num_link_components", [0].toTensor().astype(float)),
             ("num_connected_components", [0].toTensor().astype(float)),
             ("gauss_code", [-1000, -1000, -1000, -1000, -1000].toTensor().astype(float)),
             ("crossing_signs", [-1000, -1000].toTensor().astype(float)),
             ("writhe", [0].toTensor().astype(float))] &
           (if hfk_enabled:
                @[("seifert_genus", [-1000].toTensor().astype(float)),
                  ("nu", [-1000].toTensor().astype(float)),
                  ("tau", [-1000].toTensor().astype(float))]
            else:
                @[]))

test "newRepresentableState":
    # check that shape, features_cls and features are correctly assigned to the RepresentableState object
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state = newRepresentableState[CrossChange](newSeq[array[0..3, int]](), newSeq[SubdividedFace](), trefoil, newSeq[array[0..3, int]](), newSeq[array[0..3, int]](), 5, "unknotting", newFrameworkInfo(1, none(seq[int]), none(seq[int])), [50, 4], @[TensorFeature(DeterminantAndSignature())], @[("determinant", [3].toTensor().astype(float)), ("signature", [2].toTensor().astype(float))])
    check state.shape == [50, 4]
    check $(state.features_cls) == "@[Tensor Feature: Determinant and Signature]"
    check state.features == @[("determinant", [3].toTensor().astype(float)),
                              ("signature", [2].toTensor().astype(float))]

test "representablestate_from_base_state":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    # base state is CrossChange
    var base_state1 = crosschange_from_link(trefoil)
    var state1 = representablestate_from_base_state(base_state1, [50, 4], newSeq[TensorFeature](), newSeq[(string, Tensor[float])]())
    check $(state1.link) == "<Link: 1 comp; 3 cross>"
    check state1.shape == [50, 4]

    # base state is BandMove
    var base_state2 = bandmove_from_link(trefoil)
    var state2 = representablestate_from_base_state(base_state2, [50, 4], newSeq[TensorFeature](), newSeq[(string, Tensor[float])]())
    check $(state2.link) == "<Link: 1 comp; 3 cross>"
    check state1.shape == [50, 4]

test "representablestate_from_link":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state1 = representablestate_from_link[CrossChange](trefoil, shape = [3, 1])
    # we are not comprehensively checking state1/state2.features because the relevant checks have already been done in the tests for compute_features
    check state1.features.len == 7
    var state2 = representablestate_from_link[BandMove](trefoil, shape = [3, 1])
    check state2.features.len == (if hfk_enabled: 10 else: 7)

test "copy":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state1 = representablestate_from_link[CrossChange](trefoil)
    var state1_copy = state1.copy()
    state1.start(0, 0)
    check state1_copy.band_path == []
    var state2 = representablestate_from_link[BandMove](trefoil)
    var state2_copy = state2.copy()
    state2.start(0, 0)
    check state2_copy.band_path == []

test "to_seq":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state = representablestate_from_link[CrossChange](trefoil, shape = [4, 2])
    state.start(0, 0)
    state.twist(-1)
    state.over_under(0, 3, 1)
    check (state.to_seq() ==
           @[("band_path", [[0, 0, 0], [0, 3, 1],
                            [-1000, -1000, -1000],
                            [-1000, -1000, -1000],
                            [-1000, -1000, -1000],
                            [-1000, -1000, -1000],
                            [-1000, -1000, -1000],
                            [-1000, -1000, -1000]].toTensor().astype(float)),
             ("total_twists", [-1].toTensor().astype(float)),
             ("band_matrix", [[0, -10, -10, 1],
                              [-10, -10, -10, -10],
                              [-10, -10, -10, -10],
                              [-1000, -1000, -1000, -1000]].toTensor().astype(float)),
             ("PD_code", [[5, 2, 0, 3],
                          [3, 0, 4, 1],
                          [1, 4, 2, 5],
                          [-1000, -1000, -1000, -1000]].toTensor().astype(float)),
             ("unlinked_unknot_components", [0].toTensor().astype(float)),
             ("num_link_components", [1].toTensor().astype(float)),
             ("num_connected_components", [1].toTensor().astype(float)),
             ("gauss_code", [-1, 2, -3, 1, -2, 3, -500, -1000, -1000, -1000].toTensor().astype(float)),
             ("crossing_signs", [-1, -1, -1, -1000].toTensor().astype(float)),
             ("writhe", [-3].toTensor().astype(float))])

test "next_state":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state1 = representablestate_from_link[CrossChange](trefoil)
    state1.start(0, 0)
    check state1.features[1][1][0] == 0.0
    state1 = state1.next_state("C0S3_end2")
    check state1.link.crossings.len == 0
    check state1.features[1][1][0] == 1.0

    var hopf_positive = link_from_PD_code(@[[2, 1, 3, 0], [1, 2, 0, 3]])
    var state2 = representablestate_from_link[BandMove](hopf_positive)
    state2.start(0, 1)
    check state2.features[1][1][0] == 0.0
    state2 = state2.next_state("C0S0_end0")
    check state2.link.crossings.len == 0
    check state2.features[1][1][0] == 1.0

test "cls_name":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state1 = representablestate_from_link[CrossChange](trefoil)
    check state1.cls_name() == "RepresentableState<CrossChange>"

    var state2 = representablestate_from_link[BandMove](trefoil)
    check state2.cls_name() == "RepresentableState<BandMove>"

test "`$`":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state1 = representablestate_from_link[CrossChange](trefoil)
    check find($state1, re"""\nshape=.*\nadditional feature classes=.*\nfeature values=.*\n""") != -1

    var state2 = representablestate_from_link[BandMove](trefoil)
    check find($state2, re"""\nshape=.*\nadditional feature classes=.*\nfeature values=.*\n""") != -1

test "crosschangestate/bandmovestate_from_link":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state1 = crosschangestate_from_link(encode_link(trefoil), 5, "unknotting", [50, 4], @["DeterminantAndSignature"])
    check $(state1.features_cls) == "@[Tensor Feature: Determinant and Signature]"

    var state2 = bandmovestate_from_link(encode_link(trefoil), 5, "unknotting", [50, 4], @["DeterminantAndSignature"])
    check $(state2.features_cls) == "@[Tensor Feature: Determinant and Signature]"

test "crosschangestate/bandmovestate_next_state":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]])
    var state1 = representablestate_from_link[CrossChange](trefoil, shape = [3, 1])
    var pair1 = crosschangestate_next_state(state1, "C0S1_start")
    check pair1[0] == ""
    state1 = pair1[1]
    pair1 = crosschangestate_next_state(state1, "C0S0_end2")
    check pair1[0] != ""

    var hopf_positive = link_from_PD_code(@[[2, 1, 3, 0], [1, 2, 0, 3]])
    var state2 = representablestate_from_link[BandMove](hopf_positive, shape = [2, 2]).next_state("C0S3_start")
    var pair2 = bandmovestate_next_state(state2, "C0S2_under")
    check pair2[0] == ""
    state2 = pair2[1]
    pair2 = bandmovestate_next_state(state2, "C0S1_end0")
    check pair2[0] != ""

test "nim_rand_state":
    # the two halves of the random state are int64 and are routinely negative
    check match(nim_rand_state(), re"""\{"a0": -?\d+, "a1": -?\d+}""")
