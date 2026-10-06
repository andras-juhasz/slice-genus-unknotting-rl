from typing import List, Tuple, Optional
import math
import json

from spherogram import Link, Crossing
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
try:
    from sage.plot.graphics import Graphics

    sage_installed = True
except ImportError:
    sage_installed = False

from config import link_check_planarity


def is_sorted(lst: List[int]) -> bool:
    """
    Tests whether a list of integers is in non-decreasing
    order.

    Args:
        lst: a list of integers

    Returns:
        Whether lst is in non-decreasing order.
    """
    return all(lst[i] <= lst[i + 1]
               for i in range(len(lst) - 1))


def split_testing_workload(
        dataset: List[Link],
        total_steps: int,
        num_workers: int
) -> List[Tuple[List[Link], int]]:
    """
    Given a testing dataset and total testing timesteps
    allowed, splits the dataset and testing timesteps among
    multiple worker processes, with testing timesteps
    allocated based on a difficulty measure proportional to
    the number of crossings of each link raised to the
    1.5-th power.

    Args:
        dataset: the testing dataset
        total_steps: the total testing timesteps
        num_workers: the number of worker processes (at
            most)

    Returns:
        A list of (dataset, timesteps) pairs given to the
        worker processes.

    Raises:
        ValueError: if num_workers is not a positive integer
    """
    if num_workers > len(dataset):
        num_workers = len(dataset)
    if num_workers <= 0:
        raise ValueError(f'the number of worker processes '
                         f'{num_workers} is not a positive '
                         f'integer')
    base = len(dataset) // num_workers
    remainder = len(dataset) % num_workers
    len_per_worker = ([base+1] * remainder +
                      [base] * (num_workers - remainder))
    difficulties = [math.pow(len(lnk.crossings), 1.5)
                    for lnk in dataset]
    data_per_worker: List[List[Link]] = []
    difficulty_per_worker: List[float] = []
    cur_pos = 0
    for data_len in len_per_worker:
        data_per_worker.append(dataset[cur_pos:
                                       cur_pos+data_len])
        difficulty_per_worker.append(sum(difficulties[cur_pos:cur_pos+data_len]))
        cur_pos += data_len
    total_difficulties = sum(difficulty_per_worker)
    steps_per_worker = [math.ceil(total_steps * d / total_difficulties)
                        for d in difficulty_per_worker]
    return list(zip(data_per_worker, steps_per_worker))


def encode_link(link: Link) -> str:
    """
    Encodes a Link object into a JSON string to pass to the
    Nim modules.

    Args:
        link: the Link object to encode to a JSON string

    Returns:
        The encoded JSON string. By the design of
        Spherogram-nim, the Link object's crossing labels
        and strand labels are not included in the string.
    """
    return json.dumps({'PD_code': [list(c) for c in link.PD_code()],
                       'signs': [c.sign for c in link.crossings],
                       'unlinked_unknot_components': link.unlinked_unknot_components,
                       'name': link.name})


def link_from_crossings(
        pd: List[Tuple[int, int, int, int]],
        signs: List[int] = None,
        unlinked_unknot_components: int = 0
) -> Link:
    """
    Constructs a Link object from a link's PD code and its
    crossing signs. The crossings must appear in the same
    order in the PD code and the list of crossing signs.

    Args:
        pd: the link's PD code
        signs: the link's crossing signs (should be +1 or
            -1)
        unlinked_unknot_components: the number of unlinked
            unknot components in the link

    Returns:
        A Link object representing the given link.

    Raises:
        ValueError: if PD and signs have different lengths

    References:
        See the code snippet after "by directly gluing up
        Crossings" in
        https://snappy.computop.org/spherogram.html#spherogram.Link
    """
    if signs is None:
        signs = [0] * len(pd)
    if len(pd) != len(signs):
        raise ValueError(f'length of PD code = {len(pd)} '
                         f'does not equal length of signs '
                         f'= {len(signs)}')
    crossings = [Crossing(i) for i in range(len(pd))]
    first_strand: List[Optional[Tuple[int, int]]] = [None] * (2 * len(pd))
    for c, (strands, sign, crossing) in enumerate(zip(pd, signs, crossings)):
        for i, edge in enumerate(strands):
            if first_strand[edge] is None:
                first_strand[edge] = (c, i)
            else:
                prev_c, prev_i = first_strand[edge]
                crossing[i] = crossings[prev_c][prev_i]
        crossing.sign = sign
    link = Link(crossings, check_planarity=link_check_planarity)
    link.unlinked_unknot_components = unlinked_unknot_components
    return link


def decode_link(link_json: str) -> Link:
    """
    Decodes a JSON encoding of a link into a Link object.

    Args:
        link_json: a JSON encoding of a link

    Returns:
        The decoded Link object.
    """
    link_dict = json.loads(link_json)
    link = link_from_crossings(link_dict['PD_code'],
                               link_dict['signs'],
                               link_dict['unlinked_unknot_components'])
    link.name = link_dict['name']
    return link


def assert_sage() -> None:
    """
    Asserts that code is running on sage, useful for
    functions requiring sage, which include those
    calculating certain link invariants and plotting links.

    Raises:
        ImportError: if Sagemath is not installed
    """
    if not sage_installed:
        raise ImportError("Sage installation not found")


def sageplots_to_figure(
    plots: List['Graphics'],
    shape: Optional[Tuple[int, int]] = None
) -> Figure:
    """
    Given several visualizations in sage, concatenates them
    into one matplotlib figure. This is useful for rendering
    sequences of links created by our agent.

    Args:
        plots: list of plots of sage objects
        shape: if given a pair of integers (n, m), we
            arrange the plots in an n x m grid from top to
            bottom and left to right. A ValueError is raised
            if the grid is too small to contain all the
            plots; if shape is not given, we arrange all the
            plots in a single line from left to right

    Returns:
        A matplotlib figure containing all the
        visualizations in plots.

    Raises:
        ValueError: if the grid dimension given by shape is
            too small to contain all the plots
    """
    assert_sage()
    if shape is None:
        shape = (1, len(plots))
    height, width = shape
    if len(plots) > height * width:
        raise ValueError(f'Grid dimension {height} x '
                         f'{width} is too small to contain '
                         f'all the {len(plots)} plots')

    dimension = 2
    if height == 1 and width == 1:
        dimension = 0
    elif height == 1 or width == 1:
        dimension = 1
    figure, axs = plt.subplots(height, width)

    if dimension == 0:
        for plot in plots:
            figure = plot.matplotlib(figure=figure)
    elif dimension == 1:
        for index, plot in enumerate(plots):
            figure = plot.matplotlib(figure=figure,
                                     sub=axs[index])
    else:
        for index, plot in enumerate(plots):
            x = index // height
            y = index % height
            figure = plot.matplotlib(figure=figure,
                                     sub=axs[x, y])

    if dimension >= 1:
        for axi in axs.ravel():
            axi.set_axis_off()
    else:
        plt.axis('off')
        axs.axis('off')

    return figure


def plot_links(links: List[Link]) -> Figure:
    """
    Creates an image of a list of links. (Requires sage,
    which is used to draw links.)

    Args:
        links: the list of links to visualize

    Returns:
        A Figure object containing visualizations of all
        the links in links
    """
    assert_sage()
    link_plots = [link.sage_link().plot()
                  for link in links]
    width = math.ceil(math.sqrt(len(links)))
    height = math.ceil(len(links) / width)
    return sageplots_to_figure(link_plots,
                               (height, width))


def plot_link(link: Link) -> Figure:
    """
    Creates an image of a single link. (Requires sage, which
    is used to draw links.)

    Args:
        link: the link to visualize

    Returns:
        A Figure object containing the visualization of link
    """
    return plot_links([link])
