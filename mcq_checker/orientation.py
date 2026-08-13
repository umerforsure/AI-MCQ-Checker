"""
Orientation correction: detects the 4 corner ArUco markers and fixes
rotation/flip so the page is upright before any further processing.

Marker ID convention: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left.
"""

import cv2
import numpy as np

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
EXPECTED_IDS = {0, 1, 2, 3}


class OrientationError(Exception):
    pass


def _detect(image_gray):
    detector = cv2.aruco.ArucoDetector(ARUCO_DICT, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image_gray)
    found = {}
    if ids is not None:
        for c, i in zip(corners, ids.flatten()):
            if int(i) in EXPECTED_IDS:
                found[int(i)] = c.reshape(4, 2)
    return found


def _marker_rotation_steps(corner_pts):
    center = corner_pts.mean(axis=0)
    vec = corner_pts[0] - center
    angle_deg = np.degrees(np.arctan2(vec[1], vec[0]))
    reference_deg = -135.0
    diff = (angle_deg - reference_deg) % 360
    return int(round(diff / 90.0)) % 4


def correct_orientation(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    working_image, working_gray, flipped = image_bgr, gray, False

    found = _detect(working_gray)
    if len(found) < 4:
        flipped_gray = cv2.flip(gray, 1)
        found_flipped = _detect(flipped_gray)
        if len(found_flipped) >= 4:
            working_image = cv2.flip(image_bgr, 1)
            working_gray = flipped_gray
            found = found_flipped
            flipped = True

    if len(found) < 4:
        raise OrientationError(
            f"Only found {len(found)}/4 corner markers "
            f"(ids present: {sorted(found.keys())}) - cannot determine orientation."
        )

    steps = _marker_rotation_steps(found[0])
    if steps == 0:
        corrected = working_image
    elif steps == 1:
        corrected = cv2.rotate(working_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif steps == 2:
        corrected = cv2.rotate(working_image, cv2.ROTATE_180)
    else:
        corrected = cv2.rotate(working_image, cv2.ROTATE_90_CLOCKWISE)

    return corrected, {
        "flipped": flipped,
        "rotation_steps_clockwise": steps,
        "markers_found": sorted(found.keys()),
    }
