"""
WestMine Safety System - Standalone Evaluation Script
======================================================
Run this script from the command line to evaluate the system
on a test video and generate charts and metrics.

All results come from actual model inference, not simulated data.

Usage:
    python run_evaluation.py                           # Show benchmarks only
    python run_evaluation.py --video test_video.mp4    # Evaluate on a video
    python run_evaluation.py --video test.mp4 --gt     # Also compare to ground truth

Authors: [Student Name 1], [Student Name 2], [Student Name 3]
"""

import argparse
import os
import sys
import json
import time

# Make sure our project modules can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from evaluator import SystemEvaluator


def main():
    parser = argparse.ArgumentParser(
        description="WestMine Safety System - Evaluation Script"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to test video file")
    parser.add_argument("--frames", type=int, default=200,
                        help="Max frames to evaluate (default: 200)")
    parser.add_argument("--sample-rate", type=int, default=5,
                        help="Sample every Nth frame (default: 5)")
    parser.add_argument("--gt", action="store_true",
                        help="Also compare against ground_truth.csv")
    args = parser.parse_args()

    print("=" * 55)
    print("  WestMine Safety System - Evaluation Report")
    print("=" * 55)
    print()

    evaluator = SystemEvaluator()

    # ── 1. Show environment info ──
    # The assignment requires documenting the hardware and software setup
    print("-" * 40)
    print("1. Environment Setup")
    print("-" * 40)
    env_info = evaluator.get_environment_info()
    for k, v in env_info.items():
        print(f"   {k}: {v}")
    print()

    # ── 2. Show published COCO benchmarks ──
    print("-" * 40)
    print("2. YOLOv8n Published COCO Benchmarks")
    print("-" * 40)
    print("   (These are reference values, not our own test results)")
    benchmarks = evaluator.get_coco_benchmarks()
    for k, v in benchmarks.items():
        print(f"   {k}: {v}")
    print()

    # ── 3. Evaluate on video if provided ──
    if args.video:
        if not os.path.exists(args.video):
            print(f"ERROR: Video not found: {args.video}")
            sys.exit(1)

        print("-" * 40)
        print(f"3. Real Detection Evaluation: {args.video}")
        print("-" * 40)
        metrics = evaluator.evaluate_on_video(
            args.video,
            sample_rate=args.sample_rate,
            max_frames=args.frames
        )

        # ── 4. Compare against ground truth if requested ──
        if args.gt:
            print()
            print("-" * 40)
            print("4. Ground Truth Comparison")
            print("-" * 40)

            gt_path = config.GROUND_TRUTH_CSV
            if os.path.exists(gt_path):
                gt_results = evaluator.evaluate_against_ground_truth(
                    gt_path, args.video, args.sample_rate
                )
            else:
                print(f"   Ground truth file not found: {gt_path}")
                print(f"   Create a CSV with columns: frame_id, expected_persons, "
                      f"expected_vehicles, expected_danger")
                print(f"   Place it at: {gt_path}")

    # ── 5. Generate charts from real data ──
    # Only generate charts if we actually ran on a video. Otherwise
    # the chart files would just say "no data yet" which is confusing.
    if args.video:
        print()
        print("-" * 40)
        print("5. Generating Charts")
        print("-" * 40)
        report_dir = evaluator.generate_report()

        # Save a summary JSON with everything
        summary = {
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": config.MODEL_NAME,
            "confidence_threshold": config.CONFIDENCE_THRESHOLD,
            "iou_threshold": config.IOU_THRESHOLD,
            "safe_distance_px": config.SAFE_DISTANCE_PX,
            "img_size": config.YOLO_IMG_SIZE,
            "augment": config.USE_AUGMENT,
            "environment": evaluator.get_environment_info(),
            "coco_benchmarks": evaluator.get_coco_benchmarks(),
            "note": "All detection metrics are from real model inference."
        }

        summary_path = os.path.join(config.EVAL_DIR, "evaluation_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"   Summary saved: {summary_path}")
    else:
        print()
        print("Tip: pass --video <path> to generate charts from real data.")

    print()
    print("=" * 55)
    print("  Evaluation Complete!")
    print(f"  Output folder: {config.EVAL_DIR}")
    print("=" * 55)


if __name__ == "__main__":
    main()
