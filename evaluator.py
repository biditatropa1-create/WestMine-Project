"""
WestMine Safety System - Evaluation
Runs YOLOv8 on test frames and produces precision/recall/F1 plus charts.

Authors: Bidita Tarafder, Tshering Wangmo, Cynthia Mosoba
"""

import os
import csv
import json
import math
import time
import platform
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend so it works without a display
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
from typing import List, Dict
from dataclasses import dataclass, field
from ultralytics import YOLO
import config


@dataclass
class EvalMetrics:
    """Evaluation results. None means 'not measured', not 0%."""
    precision: float = None
    recall: float = None
    f1_score: float = None
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_negatives: int = 0
    avg_confidence: float = 0.0
    avg_inference_ms: float = 0.0
    total_frames: int = 0
    avg_fps: float = 0.0


class SystemEvaluator:
    """Evaluation engine - runs YOLO on test frames and reports metrics."""

    def __init__(self, model_path: str = config.MODEL_NAME):
        """Load the YOLO model for evaluation."""
        self.model = YOLO(model_path)
        self.metrics = EvalMetrics()

        self.all_inference_times = []
        self.all_confidences = []
        self.per_frame_fps = []
        self.detection_counts = {"Person": 0, "Car": 0, "Truck": 0}

    def evaluate_on_video(self, video_path: str,
                          sample_rate: int = 5,
                          max_frames: int = 200) -> EvalMetrics:
        """
        Run YOLO on a video and collect timing, confidence, and class counts.
        Zero-shot evaluation using pre-trained YOLOv8s on COCO.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Evaluator] Video: {video_path}")
        print(f"[Evaluator] Video FPS: {video_fps}, Total frames: {total_in_video}")
        print(f"[Evaluator] Sampling every {sample_rate} frames, max {max_frames}")

        frame_count = 0
        processed = 0

        while processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Only process every Nth frame to match our sample rate
            if frame_count % sample_rate != 0:
                continue

            # Run YOLOv8 and measure how long it takes
            start = time.time()
            results = self.model(
                frame,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                classes=config.TARGET_CLASSES,
                imgsz=config.YOLO_IMG_SIZE,
                augment=config.USE_AUGMENT,
                verbose=False
            )
            inference_ms = (time.time() - start) * 1000
            frame_fps = 1000.0 / inference_ms if inference_ms > 0 else 0

            # Store the real timing data
            self.all_inference_times.append(inference_ms)
            self.per_frame_fps.append(frame_fps)

            # Count what the model actually detected in this frame
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    self.all_confidences.append(conf)

                    # Count by class
                    if cls_id == 0:
                        self.detection_counts["Person"] += 1
                    elif cls_id == 2:
                        self.detection_counts["Car"] += 1
                    elif cls_id == 7:
                        self.detection_counts["Truck"] += 1

            processed += 1
            if processed % 50 == 0:
                print(f"[Evaluator] Processed {processed}/{max_frames} frames...")

        cap.release()

        # Calculate averages from the real data we collected
        self.metrics.total_frames = processed
        if self.all_inference_times:
            self.metrics.avg_inference_ms = float(np.mean(self.all_inference_times))
            # Guard against divide-by-zero in case every frame was effectively
            # instant (shouldn't happen on CPU but better safe than crashed)
            if self.metrics.avg_inference_ms > 0:
                self.metrics.avg_fps = 1000.0 / self.metrics.avg_inference_ms
            else:
                self.metrics.avg_fps = 0.0
        if self.all_confidences:
            self.metrics.avg_confidence = float(np.mean(self.all_confidences))

        print(f"\n[Evaluator] === Results ===")
        print(f"  Frames processed: {processed}")
        print(f"  Avg inference time: {self.metrics.avg_inference_ms:.1f}ms")
        print(f"  Avg FPS: {self.metrics.avg_fps:.1f}")
        print(f"  Avg confidence: {self.metrics.avg_confidence:.3f}")
        print(f"  Total detections: {self.detection_counts}")

        return self.metrics

    def _auto_scale_zone_to_frame(self, points, frame_shape,
                                  source_res=(1920, 1080)):
        """Scale the default polygon to fit a frame of any size."""
        frame_h, frame_w = frame_shape[:2]
        src_w, src_h = source_res
        sx = frame_w / src_w
        sy = frame_h / src_h
        scaled = [(int(x * sx), int(y * sy)) for (x, y) in points]

        # Validate the scaled polygon - if it collapses on very small
        # videos, fall back to a centred rectangle instead of crashing.
        try:
            poly = Polygon(scaled).buffer(0)
            if not poly.is_valid or poly.is_empty or poly.area <= 0:
                raise ValueError("Scaled polygon is degenerate.")
        except Exception:
            margin_w = frame_w // 4
            margin_h = frame_h // 4
            scaled = [
                (margin_w, margin_h),
                (frame_w - margin_w, margin_h),
                (frame_w - margin_w, frame_h - margin_h),
                (margin_w, frame_h - margin_h),
            ]
        return scaled

    def _is_dangerous_frame(self, boxes, zone_poly: Polygon) -> bool:
        """Same danger logic as the live detector (zone or proximity)."""
        persons = []    # list of (feet_point, center)
        vehicles = []   # list of (center, width, height)

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)
            center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
            feet = Point(int((x1 + x2) / 2), int(y2))
            width = int(x2 - x1)
            height = int(y2 - y1)

            if cls_id == 0:
                persons.append((feet, center))
                # Zone check
                if zone_poly is not None and zone_poly.contains(feet):
                    return True
            elif cls_id in (2, 7):
                vehicles.append((center, width, height))

        # Proximity check (same sqrt(area) formula as detector.py so the
        # evaluator matches the live system exactly)
        for _, p_center in persons:
            for v_center, v_width, v_height in vehicles:
                if v_width > 0 and v_height > 0:
                    safe_dist = math.sqrt(v_width * v_height) * config.DISTANCE_MULTIPLIER
                else:
                    safe_dist = config.SAFE_DISTANCE_PX
                if math.dist(p_center, v_center) < safe_dist:
                    return True

        return False

    def evaluate_against_ground_truth(self, gt_csv_path: str,
                                       video_path: str,
                                       sample_rate: int = 5) -> Dict:
        """
        Compare model detections against manually labelled ground truth.
        ground_truth.csv columns: frame_id, expected_persons, expected_vehicles, expected_danger.
        """
        # Load the ground truth labels
        gt_data = {}
        with open(gt_csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fid = int(row['frame_id'])
                gt_data[fid] = {
                    'expected_persons': int(row['expected_persons']),
                    'expected_vehicles': int(row['expected_vehicles']),
                    'expected_danger': row['expected_danger'].strip().lower() == 'true'
                }

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        # Probe the actual video resolution so we can scale the default
        # zone polygon to match.
        probe_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        probe_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080

        # Build the danger zone polygon once (same default as the live system)
        # Shapely needs at least 3 points - if someone broke the config we
        # just skip the zone check rather than crash the whole evaluator.
        if len(config.DEFAULT_DANGER_ZONE) >= 3:
            scaled_points = self._auto_scale_zone_to_frame(
                config.DEFAULT_DANGER_ZONE, (probe_h, probe_w)
            )
            # .buffer(0) fixes any self-intersections in scaled polygons
            zone_poly = Polygon(scaled_points).buffer(0)
            if zone_poly.is_empty or zone_poly.area <= 0:
                zone_poly = None
        else:
            print("[Evaluator] Warning: DEFAULT_DANGER_ZONE has < 3 points - "
                  "zone intrusion checks will be skipped during evaluation.")
            zone_poly = None

        # We compare "did the model detect danger?" vs "was there actually danger?"
        tp = 0  # True positive: model said danger AND there was danger
        fp = 0  # False positive: model said danger BUT there was no danger
        fn = 0  # False negative: model said safe BUT there was actually danger
        tn = 0  # True negative: model said safe AND there was no danger

        frame_count = 0
        for target_frame_id, expected in gt_data.items():
            # Seek to the target frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_id)
            ret, frame = cap.read()
            if not ret:
                continue

            # Run the model on this specific frame
            results = self.model(
                frame,
                conf=config.CONFIDENCE_THRESHOLD,
                iou=config.IOU_THRESHOLD,
                classes=config.TARGET_CLASSES,
                imgsz=config.YOLO_IMG_SIZE,
                augment=config.USE_AUGMENT,
                verbose=False
            )

            model_says_danger = False
            if results and len(results) > 0 and results[0].boxes is not None:
                model_says_danger = self._is_dangerous_frame(
                    results[0].boxes, zone_poly
                )

            # Compare to ground truth
            if expected['expected_danger'] and model_says_danger:
                tp += 1
            elif not expected['expected_danger'] and model_says_danger:
                fp += 1
            elif expected['expected_danger'] and not model_says_danger:
                fn += 1
            else:
                tn += 1

            frame_count += 1

        cap.release()

        # Calculate the metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # Also store on self.metrics so the report has them too
        self.metrics.precision = precision
        self.metrics.recall = recall
        self.metrics.f1_score = f1
        self.metrics.true_positives = tp
        self.metrics.false_positives = fp
        self.metrics.false_negatives = fn
        self.metrics.true_negatives = tn

        gt_results = {
            "frames_evaluated": frame_count,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "true_negatives": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "meets_target_recall": recall >= config.TARGET_RECALL,
        }

        print(f"\n[Evaluator] === Ground Truth Comparison ===")
        for k, v in gt_results.items():
            print(f"  {k}: {v}")

        return gt_results

    def get_environment_info(self) -> Dict:
        """Collect OS, CPU, GPU, and library versions for the report."""
        import sys

        env_info = {
            "os": f"{platform.system()} {platform.release()}",
            "os_version": platform.version(),
            "cpu": platform.processor() or "Unknown",
            "machine": platform.machine(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
        }

        # Try to get the ultralytics version
        try:
            import ultralytics
            env_info["ultralytics_version"] = ultralytics.__version__
        except Exception:
            env_info["ultralytics_version"] = "Unknown"

        # Try to get OpenCV version
        try:
            env_info["opencv_version"] = cv2.__version__
        except Exception:
            env_info["opencv_version"] = "Unknown"

        # Check if GPU (CUDA) is available
        # YOLOv8 uses PyTorch under the hood, so we check torch
        try:
            import torch
            env_info["pytorch_version"] = torch.__version__
            env_info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                env_info["gpu_name"] = torch.cuda.get_device_name(0)
                env_info["gpu_memory_mb"] = round(
                    torch.cuda.get_device_properties(0).total_memory / 1024**2
                )
            else:
                env_info["gpu_name"] = "N/A (CPU only)"
                env_info["gpu_memory_mb"] = 0
        except Exception:
            env_info["pytorch_version"] = "Unknown"
            env_info["cuda_available"] = False
            env_info["gpu_name"] = "N/A"
            env_info["gpu_memory_mb"] = 0

        # Try to get total system RAM
        try:
            import psutil
            ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
            env_info["ram_gb"] = ram_gb
        except ImportError:
            # psutil might not be installed, that's okay
            env_info["ram_gb"] = "Unknown (install psutil for this)"

        return env_info

    def get_coco_benchmarks(self) -> Dict:
        """Published COCO benchmark numbers for YOLOv8s (reference only)."""
        return {
            "model": "YOLOv8s (small)",
            "dataset": "COCO val2017",
            "mAP50": 0.526,
            "mAP50_95": 0.449,
            "parameters": "11.2M",
            "FLOPs": "28.6B",
            "source": "Ultralytics published benchmarks (ultralytics.com/yolov8)",
            "note": "Reference values from COCO validation."
        }

    def generate_report(self, output_dir: str = config.EVAL_DIR) -> str:
        """Save metrics JSON and the three evaluation charts."""
        os.makedirs(output_dir, exist_ok=True)

        self._plot_confidence_distribution(output_dir)
        self._plot_detection_classes(output_dir)
        self._plot_fps_over_time(output_dir)

        # Save metrics JSON. _safe_round keeps None as JSON null.
        def _safe_round(v, places):
            return round(v, places) if v is not None else None

        metrics_path = os.path.join(output_dir, "evaluation_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump({
                "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_frames": self.metrics.total_frames,
                "avg_inference_ms": round(self.metrics.avg_inference_ms, 2),
                "avg_fps": round(self.metrics.avg_fps, 2),
                "avg_confidence": round(self.metrics.avg_confidence, 4),
                "precision": _safe_round(self.metrics.precision, 4),
                "recall": _safe_round(self.metrics.recall, 4),
                "f1_score": _safe_round(self.metrics.f1_score, 4),
                "true_positives": self.metrics.true_positives,
                "false_positives": self.metrics.false_positives,
                "false_negatives": self.metrics.false_negatives,
                "true_negatives": self.metrics.true_negatives,
                "total_detections": self.detection_counts,
                "model": config.MODEL_NAME,
                "confidence_threshold": config.CONFIDENCE_THRESHOLD,
                "iou_threshold": config.IOU_THRESHOLD,
                "img_size": config.YOLO_IMG_SIZE,
                "augment": config.USE_AUGMENT,
                "target_recall": config.TARGET_RECALL,
                "environment": self.get_environment_info(),
                "coco_benchmarks": self.get_coco_benchmarks(),
                "note": "precision/recall/f1 are null when no ground_truth.csv is provided."
            }, f, indent=2)

        print(f"[Evaluator] Charts and metrics saved to: {output_dir}")
        return output_dir

    def _plot_confidence_distribution(self, output_dir: str):
        """Histogram of detection confidence scores."""
        fig, ax = plt.subplots(figsize=(8, 5))

        if self.all_confidences:
            ax.hist(self.all_confidences, bins=25, color='#3498DB',
                    edgecolor='white', alpha=0.85)
            ax.axvline(x=config.CONFIDENCE_THRESHOLD, color='red',
                       linestyle='--', linewidth=1.5,
                       label=f'Threshold ({config.CONFIDENCE_THRESHOLD})')
            ax.set_xlabel('Confidence Score', fontsize=11)
            ax.set_ylabel('Number of Detections', fontsize=11)
            ax.set_title('Detection Confidence Distribution (Real Data)', fontsize=13)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No evaluation data yet.\nRun evaluate_on_video() first.',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Detection Confidence Distribution')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_distribution.png'), dpi=150)
        plt.close()

    def _plot_detection_classes(self, output_dir: str):
        """Bar chart of detection counts per class."""
        fig, ax = plt.subplots(figsize=(6, 4))

        classes = list(self.detection_counts.keys())
        counts = list(self.detection_counts.values())
        colours = ['#27AE60', '#E67E22', '#E74C3C']

        if any(c > 0 for c in counts):
            bars = ax.bar(classes, counts, color=colours, edgecolor='white')
            ax.set_ylabel('Detection Count', fontsize=11)
            ax.set_title('Detections by Object Class (Real Data)', fontsize=13)

            # Show the actual number above each bar
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                            str(count), ha='center', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No detections recorded yet.',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Detections by Object Class')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'detection_classes.png'), dpi=150)
        plt.close()

    def _plot_fps_over_time(self, output_dir: str):
        """Line chart of per-frame FPS over the evaluation run."""
        fig, ax = plt.subplots(figsize=(8, 4))

        if self.per_frame_fps:
            frames = range(1, len(self.per_frame_fps) + 1)
            mean_fps = np.mean(self.per_frame_fps)

            ax.plot(frames, self.per_frame_fps, color='#3498DB',
                    alpha=0.6, linewidth=0.8)
            ax.axhline(y=mean_fps, color='#E74C3C', linestyle='--',
                       label=f'Mean FPS: {mean_fps:.1f}')
            ax.fill_between(frames, self.per_frame_fps, alpha=0.1, color='#3498DB')
            ax.set_xlabel('Frame Number', fontsize=11)
            ax.set_ylabel('FPS', fontsize=11)
            ax.set_title('Inference Speed Over Time (Real Data)', fontsize=13)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'No timing data yet.\nRun evaluate_on_video() first.',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title('Inference Speed Over Time')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'fps_over_time.png'), dpi=150)
        plt.close()


if __name__ == "__main__":
    print("=" * 60)
    print("  WestMine Safety System - Evaluation")
    print("=" * 60)

    evaluator = SystemEvaluator()

    print("\n--- YOLOv8s Published COCO Benchmarks ---")
    benchmarks = evaluator.get_coco_benchmarks()
    for k, v in benchmarks.items():
        print(f"  {k}: {v}")

    print("\n--- Generating Report Template ---")
    evaluator.generate_report()
    print("Run with --video flag to evaluate on actual footage.")
    print("Done.")
