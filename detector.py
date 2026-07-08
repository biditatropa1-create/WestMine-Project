"""
WestMine Safety System - Object Detection Module
Uses YOLOv8 to detect people and vehicles, then checks for danger
zone intrusions and proximity violations.

Authors: Bidita Tarafder, Tshering Wangmo, Cynthia Mosoba
"""

import time
import math
import os
import csv
import cv2
import numpy as np
from ultralytics import YOLO
from shapely.geometry import Polygon, Point, box
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import config


@dataclass
class Detection:
    """One detected object from YOLOv8 (person, car, or truck)."""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # (x1, y1, x2, y2)
    center: Tuple[int, int]
    feet: Tuple[int, int]             # bottom-center of the box
    box_width: int = 0
    box_height: int = 0
    in_danger: bool = False
    # Stable tracker ID across frames (from ByteTrack).
    track_id: Optional[int] = None

    @property
    def shapely_box(self):
        """Bounding box as a Shapely rectangle."""
        return box(self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3])

    @property
    def feet_point(self):
        """Feet position as a Shapely Point."""
        return Point(self.feet[0], self.feet[1])


@dataclass
class Alert:
    """A safety alert logged when a danger condition is detected."""
    timestamp: str
    camera_id: str
    alert_type: str              # "zone_intrusion" or "proximity_violation"
    person_detection: Detection
    vehicle_detection: Optional[Detection]
    distance_px: float
    confidence: float
    frame_path: Optional[str] = None


@dataclass
class FrameResult:
    """Results for one video frame."""
    detections: List[Detection] = field(default_factory=list)
    persons: List[Detection] = field(default_factory=list)
    vehicles: List[Detection] = field(default_factory=list)
    alerts: List[Alert] = field(default_factory=list)
    fps: float = 0.0
    is_danger: bool = False
    inference_time_ms: float = 0.0


class SafetyDetector:
    """Main YOLOv8s-based detector with danger zone and proximity checks."""

    def __init__(self, model_path: str = config.MODEL_NAME,
                 confidence: float = config.CONFIDENCE_THRESHOLD,
                 iou_threshold: float = config.IOU_THRESHOLD):
        """Load the YOLOv8 model and set defaults from config."""
        print(f"[SafetyDetector] Loading model: {model_path}")
        try:
            self.model = YOLO(model_path)
        except Exception as load_err:
            raise RuntimeError(
                f"Failed to load YOLO model '{model_path}'. "
                f"Check your internet connection. Error: {load_err}"
            )
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.img_size = config.YOLO_IMG_SIZE
        self.use_augment = config.USE_AUGMENT
        self.per_class_conf = dict(config.PER_CLASS_CONFIDENCE)
        self.target_classes = config.TARGET_CLASSES
        self.class_names = config.CLASS_NAMES

        self.danger_zone = None
        self.danger_zone_points = []
        self.safe_distance_px = config.SAFE_DISTANCE_PX
        self.distance_multiplier = config.DISTANCE_MULTIPLIER

        # Per-(person, vehicle) cooldown so multiple workers all get alerts.
        self.alert_cooldown = config.ALERT_COOLDOWN_SECONDS
        self.last_alert_times = {}

        self.total_frames = 0
        self.total_alerts = 0
        self.alert_positions = []

        # Warm up so the first real frame isn't slow.
        try:
            dummy = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            self.model(dummy, imgsz=self.img_size, verbose=False)
        except Exception as warmup_err:
            print(f"[SafetyDetector] Warmup skipped: {warmup_err}")

        print(f"[SafetyDetector] Ready. Confidence={confidence}, "
              f"IoU={iou_threshold}, imgsz={self.img_size}, "
              f"augment={self.use_augment}")

    def set_danger_zone(self, points: List[Tuple[int, int]]):
        """Set the danger zone polygon from a list of (x,y) corners."""
        self.danger_zone_points = points
        if len(points) >= 3:
            # buffer(0) repairs any self-intersecting polygon.
            self.danger_zone = Polygon(points).buffer(0)
            if self.danger_zone.is_empty or self.danger_zone.area <= 0:
                self.danger_zone = None
                print("[SafetyDetector] Warning: polygon is degenerate, zone disabled.")
            else:
                print(f"[SafetyDetector] Danger zone set with {len(points)} points")
        else:
            self.danger_zone = None
            print("[SafetyDetector] Warning: need at least 3 points for a polygon")

    def reset_cooldowns(self):
        """Clear per-alert cooldowns and heatmap history."""
        self.last_alert_times = {}
        self.alert_positions = []

    def detect(self, frame: np.ndarray, camera_id: str = "CAM-01") -> FrameResult:
        """Run YOLOv8 + zone/proximity checks on a single frame."""
        start_time = time.time()
        result = FrameResult()

        # Use the lowest threshold so per-class overrides still see boxes
        # below the global confidence; we filter manually below.
        min_conf = self.confidence
        if self.per_class_conf:
            min_conf = min(min_conf, *self.per_class_conf.values())

        # track() runs YOLO + ByteTrack so each object gets a stable ID.
        yolo_results = self.model.track(
            frame,
            conf=min_conf,
            iou=self.iou_threshold,
            classes=self.target_classes,
            imgsz=self.img_size,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        inference_time = (time.time() - start_time) * 1000
        result.inference_time_ms = inference_time

        # Convert YOLO outputs into Detection objects.
        if yolo_results and len(yolo_results) > 0:
            boxes = yolo_results[0].boxes
            if boxes is not None:
                track_ids = boxes.id
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())

                    # Apply per-class threshold if defined.
                    threshold = self.per_class_conf.get(cls_id, self.confidence)
                    if conf < threshold:
                        continue

                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    feet_x = cx
                    feet_y = int(y2)
                    width = int(x2 - x1)
                    height = int(y2 - y1)

                    tid = None
                    if track_ids is not None and i < len(track_ids):
                        tid = int(track_ids[i].item())

                    det = Detection(
                        class_id=cls_id,
                        class_name=self.class_names.get(cls_id, f"class_{cls_id}"),
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        feet=(feet_x, feet_y),
                        box_width=width,
                        box_height=height,
                        track_id=tid
                    )
                    result.detections.append(det)

                    if cls_id == 0:
                        result.persons.append(det)
                    elif cls_id in (2, 7):   # car or truck
                        result.vehicles.append(det)

        # Zone intrusion: check if feet are inside the polygon.
        if self.danger_zone and result.persons:
            for person in result.persons:
                if self.danger_zone.contains(person.feet_point):
                    person.in_danger = True
                    result.is_danger = True

                    alert = self._create_alert(
                        alert_type="zone_intrusion",
                        person=person,
                        vehicle=None,
                        distance=0.0,
                        camera_id=camera_id,
                        frame=frame
                    )
                    if alert:
                        result.alerts.append(alert)

        # Proximity check: dynamic safe distance scales with vehicle size.
        if result.persons and result.vehicles:
            for person in result.persons:
                for vehicle in result.vehicles:
                    distance = math.dist(person.center, vehicle.center)

                    # sqrt(area) is rotation-invariant unlike width alone.
                    if vehicle.box_width > 0 and vehicle.box_height > 0:
                        area = vehicle.box_width * vehicle.box_height
                        dynamic_safe_dist = math.sqrt(area) * self.distance_multiplier
                    else:
                        dynamic_safe_dist = self.safe_distance_px

                    if distance < dynamic_safe_dist:
                        person.in_danger = True
                        result.is_danger = True

                        alert = self._create_alert(
                            alert_type="proximity_violation",
                            person=person,
                            vehicle=vehicle,
                            distance=distance,
                            camera_id=camera_id,
                            frame=frame
                        )
                        if alert:
                            result.alerts.append(alert)

        # FPS for the dashboard display.
        total_time = time.time() - start_time
        result.fps = 1.0 / total_time if total_time > 0 else 0

        self.total_frames += 1
        self.total_alerts += len(result.alerts)

        return result

    def _cooldown_key(self, alert_type: str, person: Detection,
                      vehicle: Optional[Detection]) -> tuple:
        """Build a cooldown key per (person, vehicle) pair using tracker IDs."""
        if person.track_id is not None:
            person_key = ("tid", person.track_id)
        else:
            bucket = 20
            person_key = ("pos", person.center[0] // bucket,
                          person.center[1] // bucket)
        if vehicle is not None:
            if vehicle.track_id is not None:
                vehicle_key = ("tid", vehicle.track_id)
            else:
                bucket = 20
                vehicle_key = ("pos", vehicle.center[0] // bucket,
                               vehicle.center[1] // bucket)
        else:
            vehicle_key = None
        return (alert_type, person_key, vehicle_key)

    def _create_alert(self, alert_type: str, person: Detection,
                      vehicle: Optional[Detection], distance: float,
                      camera_id: str,
                      frame: Optional[np.ndarray] = None) -> Optional[Alert]:
        """Create an alert, respecting the per-pair cooldown."""
        current_time = time.time()
        key = self._cooldown_key(alert_type, person, vehicle)
        last_time = self.last_alert_times.get(key, 0.0)
        if current_time - last_time < self.alert_cooldown:
            return None

        self.last_alert_times[key] = current_time
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Save feet position for the heatmap.
        self.alert_positions.append(person.feet)

        alert = Alert(
            timestamp=timestamp,
            camera_id=camera_id,
            alert_type=alert_type,
            person_detection=person,
            vehicle_detection=vehicle,
            distance_px=distance,
            confidence=person.confidence
        )

        if config.SAVE_ALERT_FRAMES and frame is not None:
            frame_path = self._save_alert_frame(frame, alert)
            alert.frame_path = frame_path

        self._log_alert_to_csv(alert)

        if config.SIMULATE_SMS:
            self._simulate_sms(alert)

        return alert

    def _simulate_sms(self, alert: Alert):
        """Print a fake SMS message to the console."""
        msg = (f"[SMS -> Supervisor] WESTMINE SAFETY ALERT\n"
               f"   Time: {alert.timestamp}\n"
               f"   Camera: {alert.camera_id}\n"
               f"   Reason: {alert.alert_type.replace('_', ' ').upper()}\n"
               f"   Confidence: {alert.confidence:.0%}\n"
               f"   Action required: Check dashboard immediately.")
        print(msg)

    def _save_alert_frame(self, frame: np.ndarray, alert: Alert) -> str:
        """Save the alert frame as a JPEG for the audit trail."""
        safe_timestamp = alert.timestamp.replace(":", "-").replace(" ", "_")
        # Sanitise camera_id so cv2.imwrite doesn't choke on weird chars.
        bad_chars = '<>:"/\\|?*'
        safe_camera = "".join(c if c not in bad_chars else "_"
                              for c in alert.camera_id)
        filename = f"{safe_timestamp}_{safe_camera}_{alert.alert_type}.jpg"
        filepath = os.path.join(config.ALERTS_DIR, filename)

        cv2.imwrite(filepath, frame)
        print(f"[SafetyDetector] Alert frame saved: {filepath}")

        return filepath

    def _log_alert_to_csv(self, alert: Alert):
        """Append an alert row to a daily CSV log."""
        log_date = alert.timestamp.split(" ")[0]
        log_filename = f"alert_log_{log_date}.csv"
        log_path = os.path.join(config.LOGS_DIR, log_filename)

        write_header = not os.path.exists(log_path)

        with open(log_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp", "camera_id", "alert_type",
                    "distance_px", "confidence", "frame_path"
                ])
            writer.writerow([
                alert.timestamp,
                alert.camera_id,
                alert.alert_type,
                f"{alert.distance_px:.1f}",
                f"{alert.confidence:.4f}",
                alert.frame_path or "N/A"
            ])

    def get_stats(self) -> dict:
        """Return detector stats for the dashboard."""
        return {
            "total_frames_processed": self.total_frames,
            "total_alerts_generated": self.total_alerts,
            "model": config.MODEL_NAME,
            "confidence_threshold": self.confidence,
            "safe_distance_px": self.safe_distance_px,
            "distance_multiplier": self.distance_multiplier,
            "img_size": self.img_size,
            "augment": self.use_augment,
        }
