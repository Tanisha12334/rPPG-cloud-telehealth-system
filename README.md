
# Cloud-Based rPPG Telehealth System

Final-year B.Tech project demonstrating contactless
vital-sign estimation using webcam-based rPPG.

## Architecture

Browser webcam
→ Local face detection
→ Forehead and cheek ROI extraction
→ RGB signal preparation
→ FastAPI backend
→ Vital-sign estimation
→ Dashboard

Raw facial video remains on the client device.

## Technology Stack

- HTML, CSS, JavaScript
- MediaPipe Face Landmarker
- HTML Canvas
- Python and FastAPI
- NumPy and SciPy
- PyTorch
- TS-CAN (candidate model)
- Render (planned deployment)

## Status

Stage 1: Project setup and webcam capture.

The system is under development and is not
clinically validated.
