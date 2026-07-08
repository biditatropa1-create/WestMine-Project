"""
WestMine Safety System - Visualisation
Draws bounding boxes, danger zones, distance lines, and alert overlays.

Authors: Bidita Tarafder, Tshering Wangmo, Cynthia Mosoba
"""

import cv2
import numpy as np
import math
from typing import List, Tuple
from detector import Detection, FrameResult
import config


# BGR colours (OpenCV uses BGR not RGB).
COLOURS = {
    "person": (0, 200, 0),
    "vehicle": (0, 165, 255),
    "danger_zone": (0, 0, 200),
    "alert_border": (0, 0, 220),
    "warning_bg": (0, 0, 160),
    "safe_line": (0, 200, 200),
    "danger_line": (0, 0, 220),
    "white": (255, 255, 255),
    "text_bg": (50, 50, 50),
}


def draw_detections(frame: np.ndarray, result: FrameResult,
                    danger_zone_points: List[Tuple[int, int]] = None) -> np.ndarray:
    """Draw zone, boxes, distance lines, and alert overlays onto a frame."""
    annotated = frame.copy()

    # Draw the danger zone (semi-transparent fill + outline).
    if danger_zone_points and len(danger_zone_points) >= 3:
        pts = np.array(danger_zone_points, dtype=np.int32)

        overlay = annotated.copy()
        cv2.fillPoly(overlay, [pts], (0, 0, 80))
        cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0, annotated)

        cv2.polylines(annotated, [pts], isClosed=True,
                      color=COLOURS["danger_zone"], thickness=2)

        # Label in the middle of the zone
        cx = int(np.mean([p[0] for p in danger_zone_points]))
        cy = int(np.mean([p[1] for p in danger_zone_points]))
        _put_label(annotated, "DANGER ZONE", (cx - 55, cy),
                   bg_colour=(0, 0, 160))

    # 2. Draw bounding boxes for each detected person
    # Persons in danger are drawn RED so it is obvious on screen
    for person in result.persons:
        if person.in_danger:
            _draw_bbox(annotated, person, COLOURS["danger_line"])
        else:
            _draw_bbox(annotated, person, COLOURS["person"])
        # Draw a small dot at their feet position
        cv2.circle(annotated, person.feet, 5, (0, 0, 255), -1)

    # 3. Draw bounding boxes for each detected vehicle
    for vehicle in result.vehicles:
        _draw_bbox(annotated, vehicle, COLOURS["vehicle"])

    # Distance lines between each person and each vehicle.
    distance_multiplier = config.DISTANCE_MULTIPLIER

    for person in result.persons:
        for vehicle in result.vehicles:
            dist = math.dist(person.center, vehicle.center)

            if vehicle.box_width > 0 and vehicle.box_height > 0:
                area = vehicle.box_width * vehicle.box_height
                safe_dist = math.sqrt(area) * distance_multiplier
            else:
                safe_dist = config.SAFE_DISTANCE_PX

            # Skip lines to far-away vehicles to keep the view clean.
            if dist < safe_dist * 2:
                if dist < safe_dist:
                    line_colour = COLOURS["danger_line"]
                else:
                    line_colour = COLOURS["safe_line"]

                cv2.line(annotated, person.center, vehicle.center,
                         line_colour, 2, cv2.LINE_AA)

                mid_x = (person.center[0] + vehicle.center[0]) // 2
                mid_y = (person.center[1] + vehicle.center[1]) // 2
                label_bg = (0, 0, 160) if dist < safe_dist else (0, 130, 160)
                _put_label(annotated, f"{dist:.0f}px", (mid_x, mid_y),
                           bg_colour=label_bg, font_scale=0.45)

    # Red border + warning banner if an alert is active.
    if result.is_danger:
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w - 1, h - 1),
                      COLOURS["alert_border"], 6)
        _draw_warning_banner(annotated, "WARNING: SAFETY ALERT - DANGER DETECTED")

    _draw_stats_overlay(annotated, result)

    return annotated


def _draw_bbox(frame: np.ndarray, det: Detection, colour: Tuple[int, int, int]):
    """Draw a bounding box with a label."""
    x1, y1, x2, y2 = det.bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    cv2.circle(frame, det.center, 3, colour, -1)
    label = f"{det.class_name} {det.confidence:.0%}"
    _put_label(frame, label, (x1, y1 - 8), bg_colour=colour, font_scale=0.45)


def _put_label(frame: np.ndarray, text: str, pos: Tuple[int, int],
               bg_colour: Tuple[int, int, int] = (50, 50, 50),
               text_colour: Tuple[int, int, int] = (255, 255, 255),
               font_scale: float = 0.5):
    """Draw text with a coloured background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

    frame_h, frame_w = frame.shape[:2]
    x, y = int(pos[0]), int(pos[1])
    # Clamp so the label stays inside the frame.
    x = max(0, min(x, frame_w - tw - 6))
    y = max(th + 4, min(y, frame_h - 4))

    cv2.rectangle(frame, (x, y - th - 4), (x + tw + 4, y + 4), bg_colour, -1)
    cv2.putText(frame, text, (x + 2, y), font, font_scale,
                text_colour, thickness, cv2.LINE_AA)


def _draw_warning_banner(frame: np.ndarray, message: str):
    """Red banner across the top of the frame."""
    h, w = frame.shape[:2]
    banner_height = 45
    if h < banner_height:
        banner_height = max(10, h - 1)

    banner_slice = frame[:banner_height, :]
    overlay_strip = banner_slice.copy()
    cv2.rectangle(overlay_strip, (0, 0), (w, banner_height),
                  COLOURS["warning_bg"], -1)
    cv2.addWeighted(overlay_strip, 0.75, banner_slice, 0.25,
                    0, banner_slice)

    # Centre the warning text across the full frame width
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 0.65, 2)[0]
    text_x = (w - text_size[0]) // 2
    text_y = (banner_height + text_size[1]) // 2
    cv2.putText(frame, message, (text_x, text_y), font, 0.65,
                COLOURS["white"], 2, cv2.LINE_AA)


def _draw_stats_overlay(frame: np.ndarray, result: FrameResult):
    """FPS and detection counts in the bottom-left corner."""
    h, w = frame.shape[:2]
    lines = [
        f"FPS: {result.fps:.1f}",
        f"Inference: {result.inference_time_ms:.0f}ms",
        f"Persons: {len(result.persons)}",
        f"Vehicles: {len(result.vehicles)}",
    ]

    y_pos = h - 25 * len(lines) - 10
    for line in lines:
        _put_label(frame, line, (8, y_pos), font_scale=0.4)
        y_pos += 25
