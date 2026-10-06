import std/unittest

import "../spherogram-nim/src/links"
import arraymancer

import ../src/utils

test "is_unlink":
    check is_unlink(link_from_PD_code(@[]))
    check not is_unlink(link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]]))
    check not is_unlink(link_from_PD_code(@[[7, 4, 0, 5], [3, 0, 4, 1], [1, 7, 2, 6], [5, 3, 6, 2]]))
    check is_unlink(link_from_PD_code(@[], unlinked_unknot_components = 4))
    var two_fused_rings = link_from_PD_code(@[[0, 1, 2, 3], [1, 0, 3, 2]])
    check not is_unlink(two_fused_rings)

test "decode_link":
    var trefoil = decode_link("""{"PD_code": [[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]], "signs": [-1, -1, -1], "unlinked_unknot_components": 2, "name": "trefoil_with_two_unknots"}""")
    check trefoil.crossings == @[[11, 10, 5, 4], [3, 2, 9, 8], [7, 6, 1, 0]]
    check trefoil.signs == @[-1, -1, -1]
    check trefoil.unlinked_unknot_components == 2
    check trefoil.name == "trefoil_with_two_unknots"

test "encode_link":
    var trefoil = link_from_PD_code(@[[5, 2, 0, 3], [3, 0, 4, 1], [1, 4, 2, 5]], unlinked_unknot_components = 2, name = "trefoil_with_two_unknots")
    check encode_link(trefoil) == """{"PD_code":[[5,2,0,3],[3,0,4,1],[1,4,2,5]],"signs":[-1,-1,-1],"unlinked_unknot_components":2,"name":"trefoil_with_two_unknots"}"""

test "pad":
    # 0-dimensional array
    var arr = zeros[int]([])
    check pad(arr, [0, 0]) == zeros[int]([0, 0])
    check pad(arr, [1, 0]) == zeros[int]([1, 0])
    check pad(arr, [0, 4]) == zeros[int]([0, 4])
    check pad(arr, [2, 1]) == zeros[int]([2, 1])
    # empty 1-dimensional array
    arr = zeros[int]([0])
    check pad(arr, [0, 0]) == zeros[int]([0, 0])
    check pad(arr, [1, 0]) == zeros[int]([1, 0])
    check pad(arr, [0, 4]) == zeros[int]([0, 4])
    check pad(arr, [2, 1]) == zeros[int]([2, 1])
    # 1-dimensional array with one element
    arr = [1].toTensor()
    expect ValueError:
        discard pad(arr, [0, 1])
    expect ValueError:
        discard pad(arr, [1, 0])
    check pad(arr, [1, 1]) == [[1]].toTensor()
    check pad(arr, [1, 2]) == [[1, 0]].toTensor()
    check pad(arr, [2, 1]) == [[1], [0]].toTensor()
    # 1-dimensional array with two elements
    arr = [2, 1].toTensor()
    expect ValueError:
        discard pad(arr, [2, 1])
    check pad(arr, [2, 2]) == [[2, 1], [0, 0]].toTensor()
    # 2-dimensional array
    arr = [[1], [2]].toTensor()
    expect ValueError:
        discard pad(arr, [1, 2])
    check pad(arr, [3, 1]) == [[1], [2], [0]].toTensor()
    check pad(arr, [3, 1], -1) == [[1], [2], [-1]].toTensor()
    # 3-dimensional array; should raise ValueError
    arr = zeros[int]([1, 1, 1])
    expect ValueError:
        discard pad(arr, [2, 2])
