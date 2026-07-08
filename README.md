# WestMine Real-Time Worker Safety Monitoring System

**Course:** ICT619 - Artificial Intelligence
**Assignment:** Assignment 2 – AI Solution Implementation
**Institution:** Murdoch University, School of Information Technology
**Semester:** 1, 2026



## What This Project Does

This system uses YOLOv8s (small) computer vision with built-in ByteTrack tracking to monitor CCTV footage at mining sites. It detects when a worker on foot enters a danger zone or gets too close to heavy vehicles (trucks, cars). When a dangerous situation is found, the system shows a visual alert on a Streamlit dashboard and logs the event with a timestamp and saved frame for compliance auditing.

The system was built for WestMine Resources to support their obligations under the WHS (Mines) Regulations 2022 (WA).

---

## How to Set Up and Run

### Requirements

- Python 3.9 or higher
- pip (comes with Python)
- A webcam or test video file (.mp4, .avi, .mov)
- Internet connection (first run downloads the YOLOv8s model weights, about 22 MB)

### Setup Steps

```bash
# Step 1: Open a terminal in the WestMine_Safety_System folder

# Step 2: Create a virtual environment
python -m venv venv

# Step 3: Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Step 4: Install all the required packages
pip install -r requirements.txt

# Step 5: Run the Streamlit dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your default browser.

---

## Project File Structure

```
WestMine_Safety_System/
│
├── app.py                  # Main Streamlit dashboard (entry point)
├── config.py               # All system settings in one place
├── detector.py             # YOLOv8 detection engine and safety checks
├── evaluator.py            # Evaluation metrics and chart generation
├── visualiser.py           # Drawing overlays on video frames (OpenCV)
├── run_evaluation.py       # Standalone script to evaluate from command line
│
├── requirements.txt        # Python package dependencies
├── SETUP_GUIDE.html        # Detailed setup guide with screenshots
├── README.md               # This file
│
└── output/
    ├── alerts/             # Saved alert frame images
    ├── logs/               # CSV alert logs for compliance auditing
    ├── evaluation/         # Evaluation charts and metrics JSON
    │   └── ground_truth_TEMPLATE.csv  # Template for manual annotation
    └── test_frames/        # Extracted test frames for evaluation
```

---

## How to Use the Dashboard

### 1. Live Monitor Page
- Upload a video file (.mp4, .avi, .mov) or connect a webcam/RTSP stream
- The system processes each frame through YOLOv8 and shows:
  - Green bounding boxes around detected persons
  - Orange bounding boxes around detected vehicles
  - Red danger zone polygon overlay
  - Distance lines between persons and vehicles
  - Red border and warning banner when danger is detected
- All alerts are logged in a table at the bottom with timestamps
- You can download the alert log as a CSV file

### 2. Evaluation Page
- Upload a test video to run the model and collect real performance metrics
- Shows: frames processed, average FPS, inference time, confidence scores
- If a `ground_truth.csv` file exists, it computes precision, recall, and F1
- Generates charts: confidence distribution, detections by class, FPS over time

### 3. Settings Page
- View and adjust detection parameters (confidence threshold, IoU, safe distance, image size)
- Toggle test-time augmentation for better recall on small / partially hidden objects
- Edit the danger zone polygon coordinates (with live preview auto-scaled to the current video)
- Click "Apply Settings" to push the changes into the running detector without restarting the app

### 4. About Page
- System overview, technical stack, known limitations
- Documents changes made from the original proposal


