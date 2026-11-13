#!/usr/bin/env python3

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
