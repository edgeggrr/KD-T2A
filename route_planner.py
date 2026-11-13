#!/usr/bin/env python3

# Grid route planner for Task 2C.
#
# Intersections are letter + number (A1 .. K11).  Movement is only along grid
# rows and columns, never diagonal.  Each leg between two intersections is an
# L-shaped Manhattan path with at most one corner intersection.  Only that
# corner is inserted as an intermediate waypoint: the drone flies straight
# through the other intersections on the leg without stopping there.
#
# The planner brute-forces every survivor permutation (4! = 24) from F6 and
# keeps the order with the shortest total route.  For each leg it evaluates
# both possible L-shaped paths and uses the shorter one, because the measured
# intersection spacing in the coordinate table is not perfectly uniform.

import math
from dataclasses import dataclass, field
from itertools import permutations


GRID_COLUMNS = 'ABCDEFGHIJK'
GRID_ROWS = range(1, 12)


@dataclass
class RoutePlan:
    start: str
    survivor_order: tuple
    waypoints: list
    full_path: list
    total_length: float
    leg_lengths: dict = field(default_factory=dict)


def parse_intersection(name):
    """Return (column_letter, row_number) for a name such as 'C4'."""
    text = str(name).strip().upper()
    if len(text) < 2:
        raise ValueError('Invalid intersection name: %r' % name)

    column = text[0]
    row_text = text[1:]
    if column not in GRID_COLUMNS or not row_text.isdigit():
        raise ValueError('Invalid intersection name: %r' % name)

    row = int(row_text)
    if row not in GRID_ROWS:
        raise ValueError('Intersection row out of range: %r' % name)

    return column, row


def _distance(coords, first, second):
    """Euclidean distance between two table coordinates."""
    x1, y1 = coords[first]
    x2, y2 = coords[second]
    return math.hypot(x2 - x1, y2 - y1)


def _leg_path(start, goal, coords):
    """Return (waypoints after start, length) for one grid-aligned leg.

    If the intersections share a row or column the leg is a straight line.
    Otherwise there are exactly two L-shaped options; the shorter one wins.
    """
    if start == goal:
        return [], 0.0

    start_column, start_row = parse_intersection(start)
    goal_column, goal_row = parse_intersection(goal)

    if start_column == goal_column or start_row == goal_row:
        return [goal], _distance(coords, start, goal)

    # Horizontal first, then vertical.
    horizontal_first = '%s%d' % (goal_column, start_row)
    # Vertical first, then horizontal.
    vertical_first = '%s%d' % (start_column, goal_row)

    options = []
    for corner in (horizontal_first, vertical_first):
        path = []
        if corner != start and corner != goal:
            path.append(corner)
        path.append(goal)
        length = (
            _distance(coords, start, corner)
            + _distance(coords, corner, goal)
        )
        options.append((length, path))

    # Stable choice: shortest length, then horizontal-first on a tie.
    options.sort(key=lambda item: item[0])
    length, path = options[0]
    return path, length


def plan_route(start, survivors, coords, via_points=True):
    """Plan the shortest route from start through all survivors.

    coords maps every intersection name to an (x, y) tuple.  The returned
    RoutePlan.waypoints excludes the start intersection because the drone is
    already there when the mission begins.

    When via_points is True each leg follows the grid (an L-shaped path with
    at most one intermediate intersection).  When False each leg is a direct
    point-to-point flight and the order minimises the direct distance.
    """
    start = str(start).strip().upper()

    survivor_names = []
    for survivor in survivors:
        name = str(survivor).strip().upper()
        if name not in survivor_names:
            survivor_names.append(name)

    for name in [start] + survivor_names:
        if name not in coords:
            raise ValueError('No coordinate for intersection %s' % name)

    if not survivor_names:
        return RoutePlan(start, tuple(), [], [start], 0.0, {})

    # Cache every leg once; the permutation search only adds lengths.
    leg_cache = {}
    for first in [start] + survivor_names:
        for second in [start] + survivor_names:
            if first == second:
                continue
            if via_points:
                leg_cache[(first, second)] = _leg_path(first, second, coords)
            else:
                leg_cache[(first, second)] = (
                    [second], _distance(coords, first, second))

    best_order = None
    best_total = math.inf

    for ordered_survivors in permutations(survivor_names):
        order = (start,) + ordered_survivors
        total = 0.0
        for first, second in zip(order, order[1:]):
            total += leg_cache[(first, second)][1]

        # Deterministic tie-break: keep the first permutation found.
        if total < best_total - 1e-9:
            best_total = total
            best_order = order

    waypoints = []
    leg_lengths = {}
    for first, second in zip(best_order, best_order[1:]):
        path, length = leg_cache[(first, second)]
        leg_lengths['%s->%s' % (first, second)] = length
        for name in path:
            if not waypoints or waypoints[-1] != name:
                waypoints.append(name)

    # Recompute the length of the actually returned route so the logged value
    # matches the poses handed to the client.
    points = [start] + waypoints
    total_length = sum(
        _distance(coords, first, second)
        for first, second in zip(points, points[1:])
    )

    return RoutePlan(
        start=start,
        survivor_order=tuple(best_order[1:]),
        waypoints=waypoints,
        full_path=[start] + waypoints,
        total_length=total_length,
        leg_lengths=leg_lengths,
    )
