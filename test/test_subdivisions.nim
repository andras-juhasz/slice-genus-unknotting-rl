import std/unittest

import ../src/subdivisions

test "split_face_into_two":
    var strands = @[(6, 3), (5, 3), (4, 3), (3, 3), (2, 3),
                    (1, 3), (0, 3)]
    expect ValueError:
        discard split_face_into_two(strands, (4, 3), (4, 3))
    expect ValueError:
        discard split_face_into_two(strands, (7, 3), (3, 3))
    expect ValueError:
        discard split_face_into_two(strands, (5, 3), (3, 0))
    check (split_face_into_two(strands, (5, 3), (3, 3)) ==
           (@[(4, 3)], @[(2, 3), (1, 3), (0, 3), (6, 3)]))
    check (split_face_into_two(strands, (3, 3), (5, 3)) ==
           (@[(4, 3)], @[(2, 3), (1, 3), (0, 3), (6, 3)]))

test "SubdividedFace":
    var face = newSubdividedFace(@[(6, 3), (5, 3), (4, 3),
                                   (3, 3), (2, 3), (1, 3),
                                   (0, 3)])
    check (face.feasible_next_strands((1, 3)) ==
           @[(6, 3), (5, 3), (4, 3), (3, 3), (2, 3), (0, 3)])
    expect ValueError:
        discard face.feasible_next_strands((0, 1))
    face.set_initial_strand((5, 3))
    check (face.feasible_next_strands((1, 3)) ==
           @[(6, 3), (4, 3), (3, 3), (2, 3), (0, 3)])
    face = newSubdividedFace(@[(6, 3), (5, 3), (4, 3),
                               (3, 3), (2, 3), (1, 3),
                               (0, 3)])
    face.cut_through((5, 3), (6, 3))
    check (face.subdivisions ==
           @[@[(4, 3), (3, 3), (2, 3), (1, 3), (0, 3)]])
    face = newSubdividedFace(@[(6, 3), (5, 3), (4, 3),
                               (3, 3), (2, 3), (1, 3),
                               (0, 3)])
    face.cut_through((4, 3), (6, 3))
    check (face.subdivisions ==
           @[@[(5, 3)], @[(3, 3), (2, 3), (1, 3), (0, 3)]])
    face.cut_through((2, 3), (1, 3))
    check (face.subdivisions ==
           @[@[(5, 3)], @[(0, 3), (3, 3)]])
    check face.feasible_next_strands((0, 3)) == @[(3, 3)]
