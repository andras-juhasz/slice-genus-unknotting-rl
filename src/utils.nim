import std/json
import std/enumerate
import std/strformat
import std/sets

import "../spherogram-nim/src/links"
import "../spherogram-nim/src/seifert"
import arraymancer

proc is_unlink*[T](link: Link[T]): bool =
    discard """
    Checks whether a link is an unlink.

    Args:
        link: the given link

    Returns:
        Whether link is an unlink.
    """
    return link.crossings.len == 0

proc decode_link*(node_json: string): Link[int] =
    discard """
    Given a JSON encoding of a link, decodes the link. This
    method is used to pass links through the Nim-Python
    interface as strings.

    Args:
        node_json: a JSON encoding of a link; the root node
            of the JSON should contain the fields "PD_code",
            encoding a PD code of the link, "signs",
            encoding the signs of the crossings of the link,
            "unlinked_unknot_components", encoding the
            number of unlinked unknot components of the
            link, and "name", encoding the name of the link

    Returns:
        The decoded link object.
    """
    var node = parseJson(node_json)
    var pd_code = newSeq[array[0..3, int]]()
    for item_node in node["PD_code"].getElems():
        assert item_node.kind == JArray and item_node.len == 4
        var pd_code_item = [-1, -1, -1, -1]
        for (i, num_node) in enumerate(item_node.items()):
            pd_code_item[i] = num_node.getInt()
        pd_code.add(pd_code_item)
    var signs_node = node["signs"]
    assert signs_node.kind == JArray and signs_node.len == pd_code.len
    var signs = newSeq[int]()
    for sign_node in signs_node.items():
        signs.add(sign_node.getInt())
    var unlinked_unknot_components = node["unlinked_unknot_components"].getInt()
    var name = node["name"].getStr()
    return link_from_PD_code(pd_code, signs, unlinked_unknot_components, name)

proc encode_link*(link: Link[int]): string =
    discard """
    Given a Link object, encodes it into a JSON string. This
    method is used to pass links through the Nim-Python
    interface as strings.

    Args:
        link: the given Link object

    Returns:
        A JSON string encoding of the Link object; the root
        JSON node contains the fields "PD_code", "signs",
        "unlinked_unknot_components" and "name",
        corresponding to the fields or methods of the same
        name in a Link object
    """
    return $(%*{"PD_code": link.PD_code(),
                "signs": link.signs,
                "unlinked_unknot_components": link.unlinked_unknot_components,
                "name": link.name})

proc pad*[T](arr0: Tensor[T],
             shape: array[0..1, int],
             value: T = 0): Tensor[T] =
    discard """
    Pads a matrix to have given shape.

    Args:
        arr: the ndarray to pad to shape, can be 0, 1 or
            2-dimensional; if 0-dimensional or 1-dimensional
            with length 0, we treat it as an empty matrix;
            if 1-dimensional and non-empty, we treat it as
            the only line of a matrix
        shape: the height and width to pad the array to;
            must be at least the shape of the array
        value: the value to pad to the array

    Returns:
        the array with value padded to the right and bottom
        so that it has shape equal to shape

    Raises:
        ValueError: if array has more than 2 dimensions, or
            if shape cannot accommodate array
    """
    var arr = arr0
    if arr.shape.len > 2:
        raise newException(ValueError, &"utils.pad: array {arr} has more than 2 dimensions")
    if arr.shape.len == 0 or arr.shape == @[0]:
        arr = zeros[T]([0, 0])
    elif arr.shape.len == 1:
        arr = arr.reshape([1, arr.shape[0]])
    let pad_shape = (shape[0] - arr.shape[0],
                     shape[1] - arr.shape[1])
    if pad_shape[0] < 0 or pad_shape[1] < 0:
        raise newException(ValueError, &"utils.pad: padded shape {shape} too small for array {arr}")
    var ans = newTensorWith(shape, value)
    for i in 0 ..< arr.shape[0]:
        for j in 0 ..< arr.shape[1]:
            ans[i,j] = arr[i,j]
    return ans

proc get_strand_ccl*(link: Link[int], crossing: int, strand_index: int): int =
    discard """
    Given a strand identified by (crossing, strand_index), return the
    connected component label (ccl) for that strand in the link.

    This walks along the component starting from the given strand until it
    reaches a representative anchor (crossing, strand_index) stored in one of
    the `link.link_components`, then returns that component's `extra_info`.

    Args:
        link: the link containing the strand
        crossing: the crossing index of the strand
        strand_index: the strand index (0..3) at the crossing

    Returns:
        The ccl (component.extra_info) for the strand's component.
    """
    var cur_c = crossing
    var cur_s = strand_index
    while true:
        for comp in link.link_components:
            # Check both the current strand and its opposite on the same crossing,
            # since the anchor could be either one (they belong to the same edge)
            if comp.crossing == cur_c and (comp.strand_index == cur_s or
                                           comp.strand_index == (cur_s + 2) mod 4):
                return comp.extra_info
        let (op_c, op_s) = link.opposite_strand((cur_c, cur_s))
        cur_c = op_c
        cur_s = (op_s + 2) mod 4
        if cur_c == crossing and cur_s == strand_index:
            break
    # Should never happen if link data is consistent
    raise newException(ValueError, "utils.get_strand_ccl: could not determine ccl for strand")

proc num_seifert_circles*[T](link: Link[T]): int =
    discard """
    Counts the Seifert circles of a link diagram, including
    its unlinked unknot components.

    Args:
        link: the given link

    Returns:
        The number of Seifert circles.
    """
    return seifert_circles(link).len + link.unlinked_unknot_components
