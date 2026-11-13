#!/usr/bin/env python3

# Task 3A waypoint service.
#
# On the one latched /image_raw frame this node:
#   1. runs the Task 1A perception pipeline and keeps the three critical
#      survivors, the three stable survivors and the black obstacles,
#   2. marks every obstacle's whole footprint plus a clearance buffer as
#      off-limits,
#   3. plans the critical route from F6 through the critical survivors, and
#   4. serves that route over GetWaypoints.
#
# It also serves PlanStableRoute.  At the Switch the client sends the drone's
# actual position; the service plans from that position through the stable
# survivors, around the same off-limits areas, and ends at F6.

import json
import math
import os
import threading
from datetime import datetime

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image

from waypoint_navigation.srv import GetWaypoints, PlanStableRoute


# cv_bridge is not imported: in this Humble/NumPy 2 environment its compiled
# extension is ABI-incompatible.  The manual conversion below is equivalent.


# ======================================================================
# The two sections below are the obstacle planner and the Task 1A
# detection pipeline, inlined so this submission is a single script.
# ======================================================================


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

# ======================================================================
# Task 1A arena perception
# ======================================================================


# Task 1A arena perception, adapted for Task 2C.
#
# The geometry/colour pipeline below is the Task 1A implementation: detect the
# four ArUco board markers, rectify the arena to a 900 x 900 square, colour
# threshold the critical (red) and stable (yellow) survivors, and name each
# survivor by the nearest grid intersection.  Task 2C only needs two additions:
# the region centres are returned (for debug images) and the inverse homography
# is returned so a planned route can be drawn back on the original frame.

import os

import cv2
import numpy as np


MIN_MARKERS = 4
OUTPUT_SIZE = 900
CELL_SIZE = 75
MIN_REGION_AREA = 300

# The Task 3 rubble cubes are almost pure black.  The arena artwork has a few
# dark tree shadows too, but those break up into small blobs; a real obstacle
# covers several grid cells, so it is far larger.
OBSTACLE_MAX_VALUE = 30
MIN_OBSTACLE_AREA = 5000


class ImageProcessingError(Exception):
    """Raised when the survivor-detection pipeline cannot continue."""


class SurvivorRegion:
    """A detected survivor region and the intersection nearest to it."""

    def __init__(self, name, center, contour, area):
        self.name = name
        self.center = center
        self.contour = contour
        self.area = area

    def __repr__(self):
        return 'SurvivorRegion(%s, center=%s, area=%.1f)' % (
            self.name, self.center, self.area)


class ObstacleRegion:
    """A black rubble obstacle in rectified-board pixel coordinates."""

    def __init__(self, center, bbox, contour, area):
        self.center = center
        self.bbox = bbox
        self.contour = contour
        self.area = area

    def __repr__(self):
        return 'ObstacleRegion(center=%s, bbox=%s, area=%.1f)' % (
            self.center, self.bbox, self.area)


class FrameDetection:
    """Everything the waypoint planner needs from one camera frame."""

    def __init__(self, image, board, marker_ids, critical, stable, obstacles,
                 source_corners, inverse_matrix):
        self.image = image
        self.board = board
        self.marker_ids = marker_ids
        self.critical = critical
        self.stable = stable
        self.obstacles = obstacles
        self.source_corners = source_corners
        self.inverse_matrix = inverse_matrix


def marker_centres(marker_corners, marker_numbers):
    """Return marker corner data and their centre coordinates."""
    ids = [int(value) for value in marker_numbers.reshape(-1)]

    corner_map = {}
    centre_map = {}

    for marker_id, corners in zip(ids, marker_corners):
        points = corners.reshape(4, 2)
        corner_map[marker_id] = points
        centre_map[marker_id] = points.mean(axis=0)

    # If extra markers are present, retain the four farthest from the overall
    # centre. This keeps the board geometry unambiguous.
    if len(centre_map) > 4:
        board_centre = np.mean(tuple(centre_map.values()), axis=0)

        candidates = sorted(
            centre_map,
            key=lambda marker: np.linalg.norm(
                centre_map[marker] - board_centre
            ),
            reverse=True,
        )[:4]

        corner_map = {key: corner_map[key] for key in candidates}
        centre_map = {key: centre_map[key] for key in candidates}

    return corner_map, centre_map


def board_points(corner_map, centre_map):
    """Identify the four board corners from marker positions."""
    reference = np.mean(tuple(centre_map.values()), axis=0)
    location_to_id = {}

    for marker_id, point in centre_map.items():
        horizontal = point[0] >= reference[0]
        vertical = point[1] >= reference[1]

        if not vertical and not horizontal:
            location_to_id["top_left"] = marker_id
        elif not vertical and horizontal:
            location_to_id["top_right"] = marker_id
        elif vertical and horizontal:
            location_to_id["bottom_right"] = marker_id
        else:
            location_to_id["bottom_left"] = marker_id

    if set(location_to_id) != {
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    }:
        raise ImageProcessingError("Unable to determine marker layout.")

    selected_points = {}

    for marker_id, points in corner_map.items():
        distances = np.linalg.norm(points - reference, axis=1)
        selected_points[marker_id] = points[np.argmin(distances)]

    order = (
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    )

    return np.asarray(
        [selected_points[location_to_id[name]] for name in order],
        dtype=np.float32,
    )


def _destination_corners():
    """Corners of the rectified board in board-pixel coordinates."""
    return np.array(
        [
            [0, 0],
            [OUTPUT_SIZE, 0],
            [OUTPUT_SIZE, OUTPUT_SIZE],
            [0, OUTPUT_SIZE],
        ],
        dtype=np.float32,
    )


def rectify(image, source):
    """Map the detected board to a 900 x 900 square."""
    matrix = cv2.getPerspectiveTransform(source, _destination_corners())
    return cv2.warpPerspective(
        image, matrix, (OUTPUT_SIZE, OUTPUT_SIZE)
    )


def inverse_board_matrix(source):
    """Homography that maps rectified board pixels back to the frame."""
    return cv2.getPerspectiveTransform(_destination_corners(), source)


def grid_name(x_coord, y_coord):
    """Convert an image coordinate into its board cell label."""
    column_index = int(np.rint((x_coord - CELL_SIZE) / CELL_SIZE))
    row_index = int(np.rint((y_coord - CELL_SIZE) / CELL_SIZE))

    column_index = np.clip(column_index, 0, 10)
    row_index = np.clip(row_index, 0, 10)

    return f"{chr(ord('A') + int(column_index))}{int(row_index) + 1}"


def board_pixel_for_intersection(name):
    """Return the rectified-board pixel at an intersection's cell centre."""
    text = str(name).strip().upper()
    column = ord(text[0]) - ord('A')
    row = int(text[1:]) - 1
    return (
        float((column + 1) * CELL_SIZE),
        float((row + 1) * CELL_SIZE),
    )


def connected_regions(binary_image):
    """Find sufficiently large colour regions and describe each one."""
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    regions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= MIN_REGION_AREA:
            continue

        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue

        x = moments["m10"] / moments["m00"]
        y = moments["m01"] / moments["m00"]

        regions.append(
            SurvivorRegion(grid_name(x, y), (x, y), contour, area))

    return regions


def _dedupe_regions(regions):
    """Keep one region per intersection (the largest blob wins)."""
    best = {}
    for region in regions:
        current = best.get(region.name)
        if current is None or region.area > current.area:
            best[region.name] = region
    return sorted(best.values(), key=lambda region: region.name)


def colour_masks(image):
    """Build masks for the yellow and red survivor regions."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yellow = cv2.inRange(
        hsv,
        np.array([20, 80, 80], dtype=np.uint8),
        np.array([35, 255, 255], dtype=np.uint8),
    )

    red_low = cv2.inRange(
        hsv,
        np.array([0, 80, 80], dtype=np.uint8),
        np.array([10, 255, 255], dtype=np.uint8),
    )

    red_high = cv2.inRange(
        hsv,
        np.array([160, 80, 80], dtype=np.uint8),
        np.array([180, 255, 255], dtype=np.uint8),
    )

    red = cv2.bitwise_or(red_low, red_high)
    return red, yellow


def classify_survivors(board):
    """Locate critical and stable survivor regions."""
    red_mask, yellow_mask = colour_masks(board)

    critical = _dedupe_regions(connected_regions(red_mask))
    stable = _dedupe_regions(connected_regions(yellow_mask))

    return critical, stable


def detect_obstacles(board):
    """Locate the black rubble obstacles on the rectified board.

    Returns one ObstacleRegion per large, almost-black connected component.
    The bounding box is kept in board pixels because the route planner needs
    the whole obstacle footprint, not just its centre.
    """
    hsv = cv2.cvtColor(board, cv2.COLOR_BGR2HSV)
    mask = np.where(hsv[:, :, 2] < OBSTACLE_MAX_VALUE, 255, 0).astype(
        np.uint8)

    # Remove speckle and close the small gaps inside one cube's shading.
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    obstacles = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area <= MIN_OBSTACLE_AREA:
            continue

        x, y, width, height = cv2.boundingRect(contour)
        moments = cv2.moments(contour)
        if moments['m00'] > 0:
            center = (moments['m10'] / moments['m00'],
                      moments['m01'] / moments['m00'])
        else:
            center = (x + width / 2.0, y + height / 2.0)

        obstacles.append(ObstacleRegion(
            center=center,
            bbox=(x, y, x + width, y + height),
            contour=contour,
            area=area,
        ))

    obstacles.sort(key=lambda region: region.area, reverse=True)
    return obstacles


def detect_frame(image):
    """Run the complete Task 1A detection pipeline on an OpenCV image."""
    if image is None or image.size == 0:
        raise ImageProcessingError("Empty camera frame.")

    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    dictionary = cv2.aruco.getPredefinedDictionary(
        cv2.aruco.DICT_4X4_250
    )

    if hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image, dictionary, parameters=parameters
        )

    if ids is None or len(ids) < MIN_MARKERS:
        found = 0 if ids is None else len(ids)
        raise ImageProcessingError(
            f"Only {found} ArUco marker(s) detected; at least "
            f"{MIN_MARKERS} are required to define the board corners."
        )

    detected = {int(value) for value in ids.reshape(-1)}

    corner_data, centre_data = marker_centres(corners, ids)
    source = board_points(corner_data, centre_data)
    board = rectify(image, source)
    critical, stable = classify_survivors(board)
    obstacles = detect_obstacles(board)
    inverse = inverse_board_matrix(source)

    return FrameDetection(
        image=image,
        board=board,
        marker_ids=detected,
        critical=critical,
        stable=stable,
        obstacles=obstacles,
        source_corners=source,
        inverse_matrix=inverse,
    )


def process_image(filename):
    """Command-line helper kept from Task 1A for debugging a still image."""
    if not os.path.isfile(filename):
        raise ImageProcessingError(f"Image file missing: {filename}")

    image = cv2.imread(filename)
    if image is None:
        raise ImageProcessingError(f"Image load failed: {filename}")

    detection = detect_frame(image)

    critical = [region.name for region in detection.critical]
    stable = [region.name for region in detection.stable]

    folder = os.path.dirname(os.path.abspath(filename))
    name = os.path.splitext(os.path.basename(filename))[0]
    report_path = os.path.join(folder, name + "_results.txt")

    with open(report_path, "w", encoding="utf-8") as report:
        report.write(
            f"Detected marker IDs: {sorted(detection.marker_ids)}\n\n"
        )
        report.write(f"Critical Survivors: {', '.join(critical)}\n")
        report.write(f"Stable Survivors: {', '.join(stable)}\n")
        report.write("Obstacles (board-pixel bounding boxes):\n")
        for index, obstacle in enumerate(detection.obstacles, start=1):
            report.write(
                f"  {index}: {obstacle.bbox} area={obstacle.area:.0f}\n")

    return report_path

# ======================================================================
# Waypoint service
# ======================================================================

START_INTERSECTION = 'F6'
WAYPOINT_Z = 13.0
SAVE_Z = 14.0
GRID_TABLE_NAME = 'grid_whycode_coordinates.json'


class WayPoints(Node):

    def __init__(self):
        super().__init__('waypoints_service')

        # --- planning state -------------------------------------------------
        self.waypoints = []
        self.marker_ids = []
        self.survivor_order = []
        self.total_length = 0.0
        self.route_failed = False
        self._critical_route = None
        self._stable_targets = []
        self._obstacles = []
        self._detection = None

        self._route_lock = threading.Lock()
        self._route_event = threading.Event()
        self._image_lock = threading.Lock()
        self._image_handled = False

        # --- parameters -----------------------------------------------------
        self._service_wait_timeout = float(self.declare_parameter(
            'service_wait_timeout', 20.0).value)
        self._debug_dir = str(self.declare_parameter(
            'debug_dir',
            os.path.join(os.path.expanduser('~'), '.ros',
                         'swift_pico_debug')).value)
        table_override = str(self.declare_parameter(
            'grid_table_path', '').value).strip()
        self._obstacle_buffer = float(self.declare_parameter(
            'obstacle_buffer', 0.45).value)
        self._min_obstacle_buffer = float(self.declare_parameter(
            'min_obstacle_buffer', 0.10).value)

        # --- grid coordinate table -----------------------------------------
        self.grid_coordinates = {}
        self._grid_table_path = ''
        try:
            self.grid_coordinates, self._grid_table_path = \
                self._load_grid_coordinates(table_override)
            self.get_logger().info(
                'Loaded %d intersections from %s' % (
                    len(self.grid_coordinates), self._grid_table_path))
        except Exception as error:
            self.get_logger().error(
                'Could not load the grid coordinate table: %s' % error)

        self.mapper = BoardMapper(self.grid_coordinates)
        # World units per board pixel, taken from the fitted mapping.
        px_scale = math.hypot(self.mapper._x_coeff[1],
                              self.mapper._y_coeff[1])
        py_scale = math.hypot(self.mapper._x_coeff[2],
                              self.mapper._y_coeff[2])
        units_per_px = max(1e-6, 0.5 * (px_scale + py_scale))
        self._buffer_px = self._obstacle_buffer / units_per_px
        self._min_buffer_px = self._min_obstacle_buffer / units_per_px
        self._units_per_px = units_per_px
        self.get_logger().info(
            'Obstacle clearance: %.2f units (%.1f px); mapping residual %.3f'
            % (self._obstacle_buffer, self._buffer_px,
               self.mapper.max_residual))

        # --- ROS interfaces -------------------------------------------------
        self.service_group = ReentrantCallbackGroup()
        self.image_group = ReentrantCallbackGroup()

        qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            qos,
            callback_group=self.image_group,
        )

        self.srv = self.create_service(
            GetWaypoints,
            'waypoints',
            self.waypoint_callback,
            callback_group=self.service_group,
        )
        self.stable_srv = self.create_service(
            PlanStableRoute,
            'plan_stable_route',
            self.stable_route_callback,
            callback_group=self.service_group,
        )

        self.get_logger().info(
            'Waypoint service ready; waiting for one /image_raw frame')

    # ------------------------------------------------------------------
    # Grid table
    # ------------------------------------------------------------------
    def _grid_table_candidates(self, override):
        candidates = []
        if override:
            candidates.append(os.path.expanduser(override))

        try:
            share = get_package_share_directory('swift_pico')
            candidates.append(
                os.path.join(share, 'scripts', GRID_TABLE_NAME))
        except Exception:
            pass

        script_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(script_dir, GRID_TABLE_NAME))
        candidates.append(os.path.join(
            os.path.dirname(os.path.dirname(script_dir)),
            'share', 'swift_pico', 'scripts', GRID_TABLE_NAME))

        return candidates

    def _load_grid_coordinates(self, override):
        path = None
        for candidate in self._grid_table_candidates(override):
            if os.path.isfile(candidate):
                path = candidate
                break

        if path is None:
            raise FileNotFoundError(
                'grid coordinate table not found; tried: %s' % ', '.join(
                    self._grid_table_candidates(override)))

        with open(path, 'r', encoding='utf-8') as handle:
            table = json.load(handle)

        coordinates = {}
        for name, value in table.items():
            key = str(name).strip().upper()
            if len(key) < 2 or key[0] not in 'ABCDEFGHIJK':
                raise ValueError('Invalid intersection %r' % name)
            int(key[1:])
            if isinstance(value, dict):
                x_value, y_value = value['x'], value['y']
            else:
                x_value, y_value = value[0], value[1]
            coordinates[key] = (float(x_value), float(y_value))

        if not coordinates:
            raise ValueError('grid coordinate table is empty')

        return coordinates, path

    # ------------------------------------------------------------------
    # Camera frame
    # ------------------------------------------------------------------
    def _ros_image_to_bgr(self, msg):
        """Convert sensor_msgs/Image to a BGR OpenCV array."""
        encoding = (msg.encoding or '').lower()
        if encoding in ('bgr8', 'bgra8'):
            channels = 3
        elif encoding in ('rgb8', 'rgba8'):
            channels = 3
        elif encoding in ('mono8', '8uc1'):
            channels = 1
        else:
            raise ImageProcessingError(
                'Unsupported image encoding: %s' % msg.encoding)

        row_bytes = msg.width * channels
        buffer = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.step and msg.step != row_bytes:
            buffer = buffer.reshape(msg.height, msg.step)[:, :row_bytes]
        buffer = buffer.reshape(msg.height, msg.width, channels).copy()

        if encoding == 'rgb8':
            return cv2.cvtColor(buffer, cv2.COLOR_RGB2BGR)
        if encoding == 'rgba8':
            return cv2.cvtColor(buffer, cv2.COLOR_RGBA2BGR)
        if encoding == 'mono8':
            return cv2.cvtColor(buffer, cv2.COLOR_GRAY2BGR)
        return buffer

    def image_callback(self, msg):
        # A latched camera may redeliver; only the first frame is used.
        with self._image_lock:
            if self._image_handled:
                return
            self._image_handled = True

        frame = None
        try:
            self.get_logger().info(
                'Received /image_raw (%dx%d, encoding=%s)' % (
                    msg.width, msg.height, msg.encoding))

            frame = self._ros_image_to_bgr(msg)
            detection = detect_frame(frame)
            self._process_detection(detection)

        except Exception as error:
            self.get_logger().error(
                'Task 3 detection/planning failed: %s' % error)
            self._save_failed_frame(frame)
            with self._route_lock:
                self.waypoints = []
                self.marker_ids = []
                self.route_failed = True
            self._route_event.set()

        finally:
            try:
                self.destroy_subscription(self.image_sub)
                self.get_logger().info(
                    'Destroyed /image_raw subscription after the first frame')
            except Exception as error:
                self.get_logger().warn(
                    'Could not destroy /image_raw subscription: %s' % error)

    def _process_detection(self, detection):
        critical = detection.critical
        stable = detection.stable
        obstacles = detection.obstacles

        self.get_logger().info(
            'Detected critical survivors: %s' % (
                ', '.join(region.name for region in critical) or 'none'))
        self.get_logger().info(
            'Detected stable survivors: %s' % (
                ', '.join(region.name for region in stable) or 'none'))
        self.get_logger().info(
            'Detected obstacles: %d' % len(obstacles))
        for index, obstacle in enumerate(obstacles, start=1):
            self.get_logger().info(
                '  obstacle %d bbox=%s area=%.0f' % (
                    index, tuple(int(v) for v in obstacle.bbox),
                    obstacle.area))

        if not critical:
            raise ImageProcessingError(
                'No critical (red) survivors were detected in the frame')
        if not stable:
            self.get_logger().warn(
                'No stable (yellow) survivors were detected; the stable '
                'route will be unavailable')
        if len(critical) != 3:
            self.get_logger().warn(
                'Expected 3 critical survivors, detected %d' % len(critical))
        if len(stable) != 3:
            self.get_logger().warn(
                'Expected 3 stable survivors, detected %d' % len(stable))

        critical_targets = []
        stable_targets = []
        for label, regions, target_list in (
                ('critical', critical, critical_targets),
                ('stable', stable, stable_targets)):
            for region in regions:
                if region.name not in self.grid_coordinates:
                    raise ImageProcessingError(
                        'No grid coordinate for detected intersection %s'
                        % region.name)
                x, y = self.grid_coordinates[region.name]
                target_list.append((region.name, (x, y)))
                self.get_logger().info(
                    '%s survivor %s -> (x=%.3f, y=%.3f)' % (
                        label, region.name, x, y))

        critical_route = plan_route(
            self.grid_coordinates[START_INTERSECTION],
            critical_targets,
            self.mapper,
            obstacles,
            end_world=None,
            buffer_px=self._buffer_px,
            min_buffer_px=self._min_buffer_px,
        )

        self.get_logger().info(
            'Critical order from %s: %s' % (
                START_INTERSECTION,
                ' -> '.join([START_INTERSECTION]
                            + list(critical_route.survivor_order))))

        poses = []
        marker_ids = []
        for point in critical_route.points:
            poses.append([float(point.world[0]),
                          float(point.world[1]),
                          WAYPOINT_Z])
            marker_ids.append(point.name)

        with self._route_lock:
            self.waypoints = poses
            self.marker_ids = marker_ids
            self.survivor_order = list(critical_route.survivor_order)
            self.total_length = float(critical_route.total_length)
            self.route_failed = False
            self._critical_route = critical_route
            self._stable_targets = list(stable_targets)
            self._obstacles = list(obstacles)
            self._detection = detection

        self._route_event.set()

        self.get_logger().info(
            'Critical route: %d waypoints, %.2f units:' % (
                len(poses), critical_route.total_length))
        for index, point in enumerate(critical_route.points, start=1):
            self.get_logger().info(
                '  %2d. %-8s %-9s (%.3f, %.3f, %.2f)' % (
                    index, point.name, point.kind,
                    point.world[0], point.world[1], WAYPOINT_Z))

        debug_dir = os.path.expanduser(self._debug_dir)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._save_debug_images(detection, critical_route, debug_dir, stamp)

    # ------------------------------------------------------------------
    # Debug images
    # ------------------------------------------------------------------
    def _save_failed_frame(self, frame):
        if frame is None:
            return
        try:
            debug_dir = os.path.expanduser(self._debug_dir)
            os.makedirs(debug_dir, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = os.path.join(debug_dir, 'failed_frame_%s.png' % stamp)
            cv2.imwrite(path, frame)
            self.get_logger().info('Saved failed detection frame: %s' % path)
        except Exception as error:
            self.get_logger().warn(
                'Could not save failed detection frame: %s' % error)

    def _zones_for_all_props(self, detection):
        required = [('F6', self.grid_coordinates[START_INTERSECTION])]
        for region in detection.critical + detection.stable:
            if region.name in self.grid_coordinates:
                required.append(
                    (region.name, self.grid_coordinates[region.name]))
        return make_obstacle_zones(
            detection.obstacles, required, self.mapper,
            buffer_px=self._buffer_px,
            min_buffer_px=self._min_buffer_px)

    def _board_to_frame(self, points, detection):
        array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        if detection.inverse_matrix is not None:
            transformed = cv2.perspectiveTransform(
                array, detection.inverse_matrix)
            return transformed.reshape(-1, 2)
        return array.reshape(-1, 2)

    def _draw_props(self, canvas, detection, zones, to_canvas):
        for zone in zones:
            x0, y0, x1, y1 = zone.bbox
            corners = np.array([
                [x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.float32)
            mapped = to_canvas(corners).astype(np.int32)
            cv2.polylines(
                canvas, [mapped], True, (0, 165, 255), 2, cv2.LINE_AA)

        for obstacle in detection.obstacles:
            x0, y0, x1, y1 = [int(v) for v in obstacle.bbox]
            corners = np.array([
                [x0, y0], [x1, y0], [x1, y1], [x0, y1]],
                dtype=np.float32)
            mapped = to_canvas(corners).astype(np.int32)
            cv2.polylines(
                canvas, [mapped], True, (0, 255, 0), 1, cv2.LINE_AA)

        for region, colour, label in (
                [(r, (0, 0, 255), 'critical') for r in detection.critical]
                + [(r, (0, 255, 255), 'stable') for r in detection.stable]):
            center = to_canvas(np.array(
                [region.center], dtype=np.float32))[0]
            point = (int(round(center[0])), int(round(center[1])))
            cv2.circle(canvas, point, 36, colour, 2, cv2.LINE_AA)
            text = '%s %s' % (region.name, label)
            cv2.putText(
                canvas, text, (point[0] - 40, point[1] - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(
                canvas, text, (point[0] - 40, point[1] - 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1, cv2.LINE_AA)
            if region.name in self.grid_coordinates:
                wx, wy = self.grid_coordinates[region.name]
                coords = '(%.2f, %.2f)' % (wx, wy)
                cv2.putText(
                    canvas, coords, (point[0] - 38, point[1] + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(
                    canvas, coords, (point[0] - 38, point[1] + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    def _draw_route(self, canvas, route, to_canvas, title):
        pixels = to_canvas(np.array(
            [point.px for point in route.points], dtype=np.float32))
        points = [(int(round(p[0])), int(round(p[1]))) for p in pixels]

        start_px = to_canvas(np.array(
            [self.mapper.world_to_px(*route.start_world)],
            dtype=np.float32))[0]
        start = (int(round(start_px[0])), int(round(start_px[1])))

        if points:
            cv2.line(canvas, start, points[0], (0, 165, 255), 3, cv2.LINE_AA)
        for first, second in zip(points, points[1:]):
            cv2.line(canvas, first, second, (0, 165, 255), 3, cv2.LINE_AA)

        cv2.circle(canvas, start, 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            canvas, 'START F6', (start[0] + 12, start[1] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        for index, (point, pixel) in enumerate(zip(route.points, points),
                                               start=1):
            if point.kind == 'survivor':
                colour = (0, 0, 255)
                radius = 9
            elif point.kind == 'home':
                colour = (255, 255, 255)
                radius = 10
            else:
                colour = (0, 255, 255)
                radius = 5
            cv2.circle(canvas, pixel, radius, colour, -1, cv2.LINE_AA)
            cv2.circle(canvas, pixel, radius, (0, 0, 0), 1, cv2.LINE_AA)
            if point.kind in ('survivor', 'home'):
                label = '%d: %s' % (index, point.name)
                cv2.putText(
                    canvas, label, (pixel[0] + 10, pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(
                    canvas, label, (pixel[0] + 10, pixel[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)

        cv2.putText(
            canvas, title, (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(
            canvas, title, (18, 32),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

    def _save_debug_images(self, detection, route, debug_dir, stamp):
        try:
            os.makedirs(debug_dir, exist_ok=True)
            zones = self._zones_for_all_props(detection)

            board = detection.board.copy()
            self._draw_props(
                board, detection, zones, lambda points: points)
            self._draw_route(
                board, route, lambda points: points,
                'Task 3 critical route: %d points, %.2f units' % (
                    len(route.points), route.total_length))
            board_path = os.path.join(
                debug_dir, 'task3_board_%s.png' % stamp)
            cv2.imwrite(board_path, board)

            frame = detection.image.copy()

            def frame_mapper(points):
                return self._board_to_frame(points, detection)

            self._draw_props(frame, detection, zones, frame_mapper)
            self._draw_route(
                frame, route, frame_mapper,
                'Task 3 critical route: %d points, %.2f units' % (
                    len(route.points), route.total_length))
            frame_path = os.path.join(
                debug_dir, 'task3_frame_%s.png' % stamp)
            cv2.imwrite(frame_path, frame)

            self.get_logger().info(
                'Saved Task 3 debug board: %s' % board_path)
            self.get_logger().info(
                'Saved Task 3 debug frame: %s' % frame_path)
        except Exception as error:
            self.get_logger().warn(
                'Could not save Task 3 debug images: %s' % error)

    def _save_stable_route_image(self, route, detection, start_world):
        try:
            debug_dir = os.path.expanduser(self._debug_dir)
            os.makedirs(debug_dir, exist_ok=True)
            stamp = datetime.now().strftime('%Y%m%d_%H%M%S')

            board = detection.board.copy()
            self._draw_route(
                board, route, lambda points: points,
                'Task 3 stable route from (%.2f, %.2f)' % start_world)
            path = os.path.join(
                debug_dir, 'task3_stable_route_%s.png' % stamp)
            cv2.imwrite(path, board)
            self.get_logger().info(
                'Saved stable-route debug image: %s' % path)
        except Exception as error:
            self.get_logger().warn(
                'Could not save stable-route debug image: %s' % error)

    # ------------------------------------------------------------------
    # Services
    # ------------------------------------------------------------------
    def _empty_waypoint_response(self, response):
        response.waypoints.header.frame_id = 'whycode'
        response.waypoints.header.stamp = self.get_clock().now().to_msg()
        return response

    def waypoint_callback(self, request, response):
        if not request.get_waypoints:
            self.get_logger().info('GetWaypoints request rejected')
            return response

        # The client may call before the latched frame arrives.  Wait for the
        # planner in another executor thread; never block the image callback.
        if not self._route_event.wait(self._service_wait_timeout):
            self.get_logger().warn(
                'GetWaypoints request arrived before a route was planned; '
                'returning an empty route')
            return self._empty_waypoint_response(response)

        with self._route_lock:
            if self.route_failed or not self.waypoints:
                self.get_logger().error(
                    'No critical route is available; returning an empty route')
                return self._empty_waypoint_response(response)

            poses = []
            for x, y, z in self.waypoints:
                pose = Pose()
                pose.position.x = float(x)
                pose.position.y = float(y)
                pose.position.z = float(z)
                pose.orientation.w = 1.0
                poses.append(pose)

            response.waypoints.poses = poses
            response.waypoints.header.frame_id = 'whycode'
            response.waypoints.header.stamp = self.get_clock().now().to_msg()
            response.marker_ids = list(self.marker_ids)

            self.get_logger().info(
                'GetWaypoints: returning %d critical waypoints ending at %s'
                % (len(poses), self.marker_ids[-1]
                   if self.marker_ids else 'none'))

        return response

    def stable_route_callback(self, request, response):
        if not self._route_event.wait(self._service_wait_timeout):
            response.success = False
            response.message = 'Critical frame has not been processed yet'
            return self._empty_waypoint_response(response)

        position = request.current_position.position
        start_world = (float(position.x), float(position.y))
        if not all(math.isfinite(value) for value in start_world):
            response.success = False
            response.message = 'Switch position is not finite'
            self.get_logger().error(response.message)
            return self._empty_waypoint_response(response)

        with self._route_lock:
            if self.route_failed or self._detection is None:
                response.success = False
                response.message = 'No detection frame is available'
                self.get_logger().error(response.message)
                return self._empty_waypoint_response(response)

            stable_targets = list(self._stable_targets)
            obstacles = list(self._obstacles)
            detection = self._detection

        if not stable_targets:
            response.success = False
            response.message = 'No stable survivors were detected'
            self.get_logger().error(response.message)
            return self._empty_waypoint_response(response)

        self.get_logger().info(
            'PlanStableRoute request from switch position '
            '(%.3f, %.3f, %.3f)' % (
                position.x, position.y, position.z))

        try:
            route = plan_route(
                start_world,
                stable_targets,
                self.mapper,
                obstacles,
                end_world=self.grid_coordinates[START_INTERSECTION],
                buffer_px=self._buffer_px,
                min_buffer_px=self._min_buffer_px,
            )
        except RoutePlanningError as error:
            response.success = False
            response.message = str(error)
            self.get_logger().error(
                'Stable route planning failed: %s' % error)
            return self._empty_waypoint_response(response)

        poses = []
        marker_ids = []
        for point in route.points:
            pose = Pose()
            pose.position.x = float(point.world[0])
            pose.position.y = float(point.world[1])
            pose.position.z = float(WAYPOINT_Z)
            pose.orientation.w = 1.0
            poses.append(pose)
            marker_ids.append(point.name)

        response.waypoints.poses = poses
        response.waypoints.header.frame_id = 'whycode'
        response.waypoints.header.stamp = self.get_clock().now().to_msg()
        response.marker_ids = marker_ids
        response.success = True
        response.message = 'Stable route planned'

        self.get_logger().info(
            'Stable order from switch: %s (%.2f units, %d waypoints)' % (
                ' -> '.join(route.survivor_order),
                route.total_length, len(poses)))
        for index, point in enumerate(route.points, start=1):
            self.get_logger().info(
                '  %2d. %-8s %-9s (%.3f, %.3f, %.2f)' % (
                    index, point.name, point.kind,
                    point.world[0], point.world[1], WAYPOINT_Z))

        self._save_stable_route_image(route, detection, start_world)
        return response


def main(args=None):
    rclpy.init(args=args)
    waypoints = WayPoints()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(waypoints)

    try:
        executor.spin()
    except KeyboardInterrupt:
        waypoints.get_logger().info('KeyboardInterrupt, shutting down.\n')
    finally:
        waypoints.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
