#!/usr/bin/env python3

# Obstacle-aware route planning for Task 3A.
#
# Detection works on the rectified 900 x 900 board image.  Obstacles are
# therefore described as pixel-space bounding boxes.  This module:
#
#   * maps board pixels <-> whycode coordinates with one affine fit of the
#     installed grid table (the table itself is slightly non-uniform),
#   * inflates each obstacle box by a clearance buffer (the drone is not a
#     point), and
#   * builds a visibility graph from the start point, the survivors, F6 and
#     the inflated obstacle corners, then searches all survivor orders with a
#     small dynamic program.
#
# Nothing here is tied to one arena layout: every obstacle and survivor comes
# from the current camera frame.

import heapq
import math
from dataclasses import dataclass
from itertools import permutations

import numpy as np

CELL_SIZE = 75.0
GRID_COLUMNS = 'ABCDEFGHIJK'
GRID_ROWS = range(1, 12)


class RoutePlanningError(Exception):
    """Raised when no obstacle-free route can be found."""


@dataclass
class ObstacleZone:
    """One inflated obstacle footprint in board pixels."""

    source_bbox: tuple
    bbox: tuple
    buffer_px: float


@dataclass
class RouteNode:
    name: str
    px: tuple
    world: tuple
    kind: str


@dataclass
class RoutePoint:
    name: str
    px: tuple
    world: tuple
    kind: str


@dataclass
class PlannedRoute:
    points: list
    survivor_order: list
    total_length: float
    start_world: tuple
    end_world: tuple


def _parse_intersection(name):
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


class BoardMapper:
    """Affine board-pixel <-> whycode mapping fitted to the grid table."""

    def __init__(self, coordinates):
        rows = []
        world_x = []
        world_y = []
        for name, value in coordinates.items():
            column, row = _parse_intersection(name)
            if isinstance(value, dict):
                x_value, y_value = value['x'], value['y']
            else:
                x_value, y_value = value[0], value[1]
            rows.append([1.0,
                         (ord(column) - ord('A') + 1) * CELL_SIZE,
                         row * CELL_SIZE])
            world_x.append(float(x_value))
            world_y.append(float(y_value))

        design = np.asarray(rows, dtype=np.float64)
        self._x_coeff = np.linalg.lstsq(
            design, np.asarray(world_x), rcond=None)[0]
        self._y_coeff = np.linalg.lstsq(
            design, np.asarray(world_y), rcond=None)[0]

        residual_x = design.dot(self._x_coeff) - np.asarray(world_x)
        residual_y = design.dot(self._y_coeff) - np.asarray(world_y)
        self.max_residual = float(max(
            np.max(np.abs(residual_x)), np.max(np.abs(residual_y))))

        self._inverse = np.asarray([
            [self._x_coeff[1], self._x_coeff[2]],
            [self._y_coeff[1], self._y_coeff[2]],
        ], dtype=np.float64)

    def px_to_world(self, px, py):
        x_value = self._x_coeff.dot([1.0, px, py])
        y_value = self._y_coeff.dot([1.0, px, py])
        return float(x_value), float(y_value)

    def world_to_px(self, x_value, y_value):
        right = np.asarray(
            [x_value - self._x_coeff[0], y_value - self._y_coeff[0]],
            dtype=np.float64)
        pixel = np.linalg.solve(self._inverse, right)
        return float(pixel[0]), float(pixel[1])


def make_obstacle_zones(obstacles, required_points, mapper,
                        buffer_px=22.0, min_buffer_px=6.0,
                        corner_eps_px=2.0):
    """Inflate obstacle boxes while keeping every required point reachable.

    The buffer normally stays at `buffer_px`.  If a survivor (or the Switch
    position) is very close to an obstacle box, only that box is shrunk just
    enough to leave a small gap.  The detected box already covers the whole
    visible cube, so this remains conservative.
    """
    zones = []
    for obstacle in obstacles:
        x0, y0, x1, y1 = [float(value) for value in obstacle.bbox]
        buffer = float(buffer_px)

        for name, point in required_points:
            px, py = mapper.world_to_px(point[0], point[1])
            distance = _point_to_rect_distance(px, py, (x0, y0, x1, y1))
            if distance <= buffer + corner_eps_px:
                buffer = min(buffer, max(
                    float(min_buffer_px), distance - corner_eps_px))

        zones.append(ObstacleZone(
            source_bbox=(x0, y0, x1, y1),
            bbox=(x0 - buffer, y0 - buffer, x1 + buffer, y1 + buffer),
            buffer_px=buffer,
        ))
    return zones


def _point_to_rect_distance(px, py, rect):
    x0, y0, x1, y1 = rect
    dx = max(x0 - px, 0.0, px - x1)
    dy = max(y0 - py, 0.0, py - y1)
    return math.hypot(dx, dy)


def _point_in_rect(px, py, rect):
    x0, y0, x1, y1 = rect
    return x0 < px < x1 and y0 < py < y1


def _segment_intersects_rect(first, second, rect):
    """Liang-Barsky clip: True if any part of the segment is in the box."""
    x0, y0, x1, y1 = rect
    px, py = first
    qx, qy = second
    dx = qx - px
    dy = qy - py

    lower = 0.0
    upper = 1.0
    for p_value, q_value in (
            (-dx, px - x0),
            (dx, x1 - px),
            (-dy, py - y0),
            (dy, y1 - py)):
        if abs(p_value) < 1e-12:
            if q_value < 0.0:
                return False
            continue
        ratio = q_value / p_value
        if p_value < 0.0:
            if ratio > upper:
                return False
            lower = max(lower, ratio)
        else:
            if ratio < lower:
                return False
            upper = min(upper, ratio)

    return lower <= upper


def _dijkstra(adjacency, source, destination, banned):
    count = len(adjacency)
    distance = [math.inf] * count
    previous = [-1] * count
    distance[source] = 0.0
    heap = [(0.0, source)]

    while heap:
        current_distance, node = heapq.heappop(heap)
        if current_distance > distance[node] + 1e-12:
            continue
        if node == destination:
            break
        for neighbour, weight in adjacency[node]:
            if neighbour in banned and neighbour != destination:
                continue
            candidate = current_distance + weight
            if candidate < distance[neighbour] - 1e-12:
                distance[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(heap, (candidate, neighbour))

    if not math.isfinite(distance[destination]):
        return None, math.inf

    path = [destination]
    while path[-1] != source:
        parent = previous[path[-1]]
        if parent < 0:
            return None, math.inf
        path.append(parent)
    path.reverse()
    return path, distance[destination]


def _world_distance(first, second):
    return math.hypot(
        second[0] - first[0], second[1] - first[1])


def plan_route(start_world, targets, mapper, obstacles,
               end_world=None, buffer_px=22.0, min_buffer_px=6.0,
               corner_eps_px=2.0):
    """Plan a route visiting every target with no leg crossing a buffer.

    `targets` is a list of (name, (x, y)) in whycode coordinates.  When
    `end_world` is supplied the route finishes there (task 3 uses F6); when it
    is None the route ends at the last survivor (the critical route).
    """
    start_world = (float(start_world[0]), float(start_world[1]))
    target_list = [(str(name), (float(point[0]), float(point[1])))
                   for name, point in targets]
    if not target_list:
        raise RoutePlanningError('No survivors to visit')

    required_points = [('__start__', start_world)] + target_list
    if end_world is not None:
        end_world = (float(end_world[0]), float(end_world[1]))
        required_points.append(('F6', end_world))

    zones = make_obstacle_zones(
        obstacles, required_points, mapper,
        buffer_px=buffer_px, min_buffer_px=min_buffer_px,
        corner_eps_px=corner_eps_px)

    # A point that falls inside even the un-inflated cube footprint cannot be
    # visited at altitude without crossing rubble.
    for name, point in required_points:
        px, py = mapper.world_to_px(point[0], point[1])
        for zone in zones:
            if _point_in_rect(px, py, zone.source_bbox):
                raise RoutePlanningError(
                    'Required point %s lies inside an obstacle' % name)

    nodes = []

    def add_node(name, world, kind):
        px, py = mapper.world_to_px(world[0], world[1])
        for zone in zones:
            if _point_in_rect(px, py, zone.bbox):
                if kind == 'turn':
                    return None
                raise RoutePlanningError(
                    'Required point %s is inside the obstacle buffer' % name)
        node = RouteNode(name=name, px=(px, py), world=world, kind=kind)
        nodes.append(node)
        return len(nodes) - 1

    start_index = add_node('start', start_world, 'start')
    target_indices = []
    for name, point in target_list:
        target_indices.append(add_node(name, point, 'target'))

    end_index = None
    if end_world is not None:
        end_index = add_node('F6', end_world, 'home')

    for zone_index, zone in enumerate(zones):
        x0, y0, x1, y1 = zone.bbox
        for corner_index, (cx, cy) in enumerate((
                (x0 - corner_eps_px, y0 - corner_eps_px),
                (x1 + corner_eps_px, y0 - corner_eps_px),
                (x1 + corner_eps_px, y1 + corner_eps_px),
                (x0 - corner_eps_px, y1 + corner_eps_px))):
            if any(_point_in_rect(cx, cy, other.bbox) for other in zones):
                continue
            world = mapper.px_to_world(cx, cy)
            add_node('turn_%d_%d' % (zone_index, corner_index),
                     world, 'turn')

    # Visibility graph over all nodes.
    adjacency = [[] for _ in nodes]
    for first in range(len(nodes)):
        for second in range(first + 1, len(nodes)):
            blocked = False
            for zone in zones:
                if _segment_intersects_rect(
                        nodes[first].px, nodes[second].px, zone.bbox):
                    blocked = True
                    break
            if blocked:
                continue
            weight = _world_distance(
                nodes[first].world, nodes[second].world)
            adjacency[first].append((second, weight))
            adjacency[second].append((first, weight))

    required_indices = [start_index] + list(target_indices)
    if end_index is not None:
        required_indices.append(end_index)
    banned_required = set(required_indices)

    path_cache = {}

    def shortest(first, second):
        key = (first, second)
        if key not in path_cache:
            banned = set(banned_required)
            banned.discard(first)
            banned.discard(second)
            path_cache[key] = _dijkstra(
                adjacency, first, second, banned)
        return path_cache[key]

    best_cost = math.inf
    best_order = None

    if end_index is None:
        for order in permutations(target_indices):
            sequence = [start_index] + list(order)
            cost = 0.0
            feasible = True
            for first, second in zip(sequence, sequence[1:]):
                _, distance = shortest(first, second)
                if not math.isfinite(distance):
                    feasible = False
                    break
                cost += distance
            if feasible and cost < best_cost - 1e-9:
                best_cost = cost
                best_order = sequence
    else:
        for order in permutations(target_indices):
            sequence = [start_index] + list(order) + [end_index]
            cost = 0.0
            feasible = True
            for first, second in zip(sequence, sequence[1:]):
                _, distance = shortest(first, second)
                if not math.isfinite(distance):
                    feasible = False
                    break
                cost += distance
            if feasible and cost < best_cost - 1e-9:
                best_cost = cost
                best_order = sequence

    if best_order is None:
        raise RoutePlanningError(
            'The obstacle buffers block every route to the survivors')

    node_path = [best_order[0]]
    for first, second in zip(best_order, best_order[1:]):
        path, _ = shortest(first, second)
        if path is None:
            raise RoutePlanningError('A route leg disappeared while building')
        if node_path and path[0] == node_path[-1]:
            path = path[1:]
        node_path.extend(path)

    points = []
    turn_number = 0
    for node_index in node_path[1:]:
        node = nodes[node_index]
        if node.kind == 'target':
            name = node.name
            kind = 'survivor'
        elif node.kind == 'home':
            name = 'F6'
            kind = 'home'
        else:
            turn_number += 1
            name = 'via_%d' % turn_number
            kind = 'turn'
        points.append(RoutePoint(
            name=name, px=node.px, world=node.world, kind=kind))

    world_path = [start_world] + [point.world for point in points]
    total_length = sum(
        _world_distance(first, second)
        for first, second in zip(world_path, world_path[1:]))

    survivor_order = [
        nodes[index].name for index in best_order
        if nodes[index].kind == 'target']

    return PlannedRoute(
        points=points,
        survivor_order=survivor_order,
        total_length=total_length,
        start_world=start_world,
        end_world=end_world,
    )
