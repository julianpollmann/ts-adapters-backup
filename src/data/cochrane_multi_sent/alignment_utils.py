from collections import namedtuple
from itertools import groupby
from operator import itemgetter
from typing import Callable, Iterable


def consecutive_groups(iterable, ordering=lambda x: x):
    """
    Get groups of consecutive elements. `ordering` determines if two elements are consecutive by returning their position. Source: https://more-itertools.readthedocs.io/en/latest/api.html#more_itertools.consecutive_groups
    """
    for _, g in groupby(enumerate(iterable), key=lambda x: x[0] - ordering(x[1])):
        yield map(itemgetter(1), g)


def merge_adjacent_(elements, grouping=lambda _: 0, ordering=lambda x: x):
    elements = sorted(elements, key=lambda x: (grouping(x), ordering(x)))
    for _, group in groupby(elements, key=grouping):
        for subgroup in consecutive_groups(group, ordering=ordering):
            yield set(subgroup)


def get_components(alignments, merge_adjacent=False, grouping=lambda _: 0, ordering=lambda x: x):
    idx = {}
    singletons_left, singletons_right = set(), set()
    components = []
    for i, j, aligned in alignments:
        singletons_left.add(i)
        singletons_right.add(j)
        if aligned:
            c = idx.get(i)
            if not c:
                c = idx.get(j)
            if not c:
                c = set()
                components.append(c)
            idx[i] = c
            idx[j] = c
            c.add(i)
            c.add(j)
    singletons_left = {s for s in singletons_left if s not in idx.keys()}
    singletons_right = {s for s in singletons_right if s not in idx.keys()}

    if merge_adjacent:
        components += list(merge_adjacent_(singletons_left, grouping=grouping, ordering=ordering))
        components += list(merge_adjacent_(singletons_right, grouping=grouping, ordering=ordering))
    else:
        for e in singletons_left | singletons_right:
            components.append({e})

    return components


AlignmentCounts = namedtuple(
    "AlignmentCounts",
    ["one_to_one", "one_to_n", "n_to_one", "n_to_m", "zero_to_one", "zero_to_n", "one_to_zero", "n_to_zero"],
)


def count_alignment_types(components: Iterable[Iterable], is_complex: Callable) -> AlignmentCounts:
    """Count types of alignments.

    Parameters
    ----------
    components : Iterable[Iterable]
        An iterable of aligned nodes. Nodes can be anything.
    is_complex : Callable
        A function that returns True if a node is complex (i.e., left side of bipartite graph) and False otherwise.
    """
    one_to_one = 0  # substitution
    one_to_n = 0  # split
    n_to_one = 0  # merge
    n_to_m = 0  # fusion
    zero_to_one = 0  # insertion
    zero_to_n = 0  # insertion (multiple)
    one_to_zero = 0  # deletion
    n_to_zero = 0  # deletion (multiple)

    for component in components:
        n_complex = 0
        n_simple = 0

        for edge in component:
            if is_complex(edge):
                n_complex += 1
            else:
                n_simple += 1

        if n_complex == 0 and n_simple == 1:
            zero_to_one += 1
        elif n_complex == 0 and n_simple > 1:
            zero_to_n += 1
        elif n_complex == 1 and n_simple == 0:
            one_to_zero += 1
        elif n_complex > 1 and n_simple == 0:
            n_to_zero += 1
        elif n_complex == 1 and n_simple == 1:
            one_to_one += 1
        elif n_complex == 1 and n_simple > 1:
            one_to_n += 1
        elif n_complex > 1 and n_simple == 1:
            n_to_one += 1
        elif n_complex > 1 and n_simple > 1:
            n_to_m += 1
        else:
            raise ValueError(f"Unexpected alignment: {component}")

    return AlignmentCounts(
        one_to_one=one_to_one,
        one_to_n=one_to_n,
        n_to_one=n_to_one,
        n_to_m=n_to_m,
        zero_to_one=zero_to_one,
        zero_to_n=zero_to_n,
        one_to_zero=one_to_zero,
        n_to_zero=n_to_zero,
    )