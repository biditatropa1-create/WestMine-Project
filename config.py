"""
WestMine Safety System - Configuration
All system settings live here so we only edit one file when tweaking the system.

Authors: Bidita Tarafder, Tshering Wangmo, Cynthia Mosoba
"""

import os

# Project paths and output folders.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ALERTS_DIR = os.path.join(OUTPUT_DIR, "alerts")
LOGS_DIR = os.path.join(OUTPUT_DIR, "logs")
EVAL_DIR = os.path.join(OUTPUT_DIR, "evaluation")
FRAMES_DIR = os.path.join(OUTPUT_DIR, "test_frames")

for d in [OUTPUT_DIR, ALERTS_DIR, LOGS_DIR, EVAL_DIR, FRAMES_DIR]:
    os.makedirs(d, exist_ok=True)

# YOLO model settings. We use yolov8s (small) for better accuracy on
# small/distant objects while still running on CPU.
MODEL_NAME = "yolov8s.pt"

# Minimum confidence to keep a detection.
CONFIDENCE_THRESHOLD = 0.5

# NMS overlap threshold (standard YOLO default).
IOU_THRESHOLD = 0.45

# Bigger image size = better recall on far-away workers but slower.
YOLO_IMG_SIZE = 1280

# Test-time augmentation - improves recall but halves FPS.
USE_AUGMENT = False

# Lower thresholds for under-detected classes (persons and trucks).
PER_CLASS_CONFIDENCE = {
    0: 0.25,   # persons
    7: 0.30,   # trucks
}

# COCO classes we care about: 0=person, 2=car, 7=truck.
TARGET_CLASSES = [0, 2, 7]
CLASS_NAMES = {0: "Person", 2: "Car", 7: "Truck"}

# Default danger zone polygon (in pixels). Auto-scaled to fit smaller videos.
DEFAULT_DANGER_ZONE = [
    (240, 190), (660, 220), (710, 280),
    (700, 600), (100, 580), (120, 350)
]

# Fallback safe distance (px) when dynamic calculation isn't possible.
SAFE_DISTANCE_PX = 150

# Safe distance = vehicle width * this multiplier.
DISTANCE_MULTIPLIER = 1.5

# Seconds between alerts for the same person/vehicle pair.
ALERT_COOLDOWN_SECONDS = 5

# Save the frame as JPEG when an alert fires (audit trail).
SAVE_ALERT_FRAMES = True

# Print a fake SMS to the console (real deployment would call Twilio).
SIMULATE_SMS = True

# Video defaults.
DEFAULT_RESOLUTION = (1920, 1080)
PROCESS_EVERY_N_FRAMES = 1
MAX_FPS_DISPLAY = 30

# Target processed FPS (video uploads auto-skip frames to hit this).
TARGET_PROCESS_FPS = 10

# Evaluation settings.
EVAL_IOU_THRESHOLD = 0.5
TARGET_RECALL = 0.90
NUM_TEST_FRAMES = 200
GROUND_TRUTH_CSV = os.path.join(EVAL_DIR, "ground_truth.csv")

# Dashboard settings.
DASHBOARD_TITLE = "WestMine Safety Monitor"
DASHBOARD_ICON = ":material/shield:"
PAGE_LAYOUT = "wide"
