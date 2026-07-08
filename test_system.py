"""
WestMine Safety System - Simple Smoke Tests
==============================================
Basic tests to make sure the main parts of the system work
before we demo it.

Run with:
    python test_system.py

All tests print "PASS" or "FAIL". Exit code is 0 if everything passes.

Authors: (Bidita Tarafder) (Tshering Wangmo) (Cynthia Mosoba)


"""

import os
import sys
import numpy as np


# Keep a running tally so we can return a proper exit code at the end
PASSED = 0
FAILED = 0


def check(condition, name, detail=""):
    """Tiny assert helper that prints PASS/FAIL instead of raising."""
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  [PASS] {name}")
    else:
        FAILED += 1
        print(f"  [FAIL] {name} - {detail}")


# ── 1. Config sanity checks ──
print("\n1. Config checks")
try:
    import config
    check(config.CONFIDENCE_THRESHOLD == 0.5,
          "CONFIDENCE_THRESHOLD matches proposal (0.5)",
          f"got {config.CONFIDENCE_THRESHOLD}")
    check(config.IOU_THRESHOLD == 0.45,
          "IOU_THRESHOLD matches proposal (0.45)",
          f"got {config.IOU_THRESHOLD}")
    check(config.MODEL_NAME.startswith("yolov8"),
          "Model is YOLOv8 family",
          f"got {config.MODEL_NAME}")
    check(0 in config.TARGET_CLASSES and 7 in config.TARGET_CLASSES,
          "Target classes include person (0) and truck (7)")
    check(config.TARGET_RECALL >= 0.90,
          "TARGET_RECALL is at least 0.90",
          f"got {config.TARGET_RECALL}")
    check(os.path.exists(config.EVAL_DIR),
          "Output evaluation directory exists")
except Exception as e:
    print(f"  [FAIL] config import crashed: {e}")
    FAILED += 1


# ── 2. Danger zone polygon is valid ──
print("\n2. Danger zone checks")
try:
    from shapely.geometry import Polygon, Point
    poly = Polygon(config.DEFAULT_DANGER_ZONE)
    check(poly.is_valid, "Default danger zone is a valid polygon")
    check(poly.area > 0, "Default danger zone has non-zero area")
    # A point clearly inside (by eyeball from config) should register as inside.
    # Using midpoint of the first and third vertex as a rough "inside" point.
    p1, p3 = config.DEFAULT_DANGER_ZONE[0], config.DEFAULT_DANGER_ZONE[2]
    mid_x = (p1[0] + p3[0]) // 2
    mid_y = (p1[1] + p3[1]) // 2
    check(poly.contains(Point(mid_x, mid_y)),
          "Point inside the polygon is detected as inside")
except Exception as e:
    print(f"  [FAIL] polygon check crashed: {e}")
    FAILED += 1


# ── 3. Cooldown key uniqueness ──
# This test catches the per-pair cooldown bug we fixed - two different
# people should get different cooldown keys so both can alert.
print("\n3. Cooldown key checks")
try:
    from detector import SafetyDetector, Detection

    # Build a lightweight test without loading the actual model.
    # We only need the _cooldown_key method, but that's a bound method
    # so we do have to instantiate. The warmup will load weights once
    # which is slow but acceptable for a smoke test.
    print("     (loading model - may take a few seconds)")
    det = SafetyDetector()

    def fake_detection(class_id, x, y):
        return Detection(
            class_id=class_id,
            class_name="Person" if class_id == 0 else "Truck",
            confidence=0.9,
            bbox=(x - 20, y - 40, x + 20, y + 40),
            center=(x, y),
            feet=(x, y + 40),
            box_width=40,
            box_height=80,
            in_danger=False
        )

    person_a = fake_detection(0, 100, 100)
    person_b = fake_detection(0, 400, 400)   # far from person_a
    truck = fake_detection(7, 250, 250)

    key_a = det._cooldown_key("proximity", person_a, truck)
    key_b = det._cooldown_key("proximity", person_b, truck)
    check(key_a != key_b,
          "Different people get different cooldown keys (critical fix)")

    # Same person moving just a few pixels should collapse to the same key
    person_a_moved = fake_detection(0, 102, 103)
    key_a_again = det._cooldown_key("proximity", person_a_moved, truck)
    check(key_a == key_a_again,
          "Same person barely moving keeps the same cooldown key")

except Exception as e:
    print(f"  [FAIL] cooldown test crashed: {e}")
    FAILED += 1


# ── 4. Detector runs on a blank frame without crashing ──
print("\n4. Detector smoke test on blank frame")
try:
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    result = det.detect(blank, camera_id="TEST")
    check(result is not None, "detect() returns a result object")
    check(hasattr(result, "persons") and hasattr(result, "vehicles"),
          "Result has persons and vehicles fields")
    check(result.inference_time_ms >= 0,
          "Inference time is recorded")
    # A blank frame should have zero detections
    check(len(result.persons) == 0 and len(result.vehicles) == 0,
          "No detections on a blank black frame")
except Exception as e:
    print(f"  [FAIL] detector smoke test crashed: {e}")
    FAILED += 1


# ── 5. Visualiser draws on frame without crashing ──
print("\n5. Visualiser smoke test")
try:
    from visualiser import draw_detections
    blank2 = np.zeros((480, 640, 3), dtype=np.uint8)
    # Use the result from the previous test (empty detections)
    drawn = draw_detections(blank2, result, config.DEFAULT_DANGER_ZONE)
    check(drawn.shape == blank2.shape,
          "Visualiser output has the same shape as input")
except Exception as e:
    print(f"  [FAIL] visualiser smoke test crashed: {e}")
    FAILED += 1


# ── Summary ──
print("\n" + "=" * 45)
print(f"  RESULTS: {PASSED} passed, {FAILED} failed")
print("=" * 45)

sys.exit(0 if FAILED == 0 else 1)
