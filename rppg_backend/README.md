# rPPG HR and experimental RR backend

This is a runtime-only copy of the working backend. Use `backend_api:app` (the corrected API), not `api:app`. It loads the included trained HR checkpoint. No training, dataset download, MediaPipe installation, GPU or ground-truth labels are required.

## 1. Install

Python 3.11 (64-bit) is recommended. Extract the ZIP, open a terminal inside the `rppg_backend` folder, then create YOUR OWN virtual environment. Internet access is needed to download dependencies. The included requirements pin the versions used for verification. A platform without wheels for these versions may need compatible versions tested separately; do not assume a successful install on every OS.

macOS / Linux:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python smoke_test.py
```

Windows PowerShell / Command Prompt:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe smoke_test.py
```

Virtual-environment activation is optional because these commands use its Python directly. PyTorch is the largest dependency. CPU inference is the default; no CUDA configuration is needed. This ZIP contains no existing virtual environment.

## 2. Run the corrected API

macOS / Linux:

```sh
.venv/bin/python -m uvicorn backend_api:app --host 127.0.0.1 --port 8001
```

Windows:

```powershell
.venv\Scripts\python.exe -m uvicorn backend_api:app --host 127.0.0.1 --port 8001
```

Keep this terminal open. Visit:

- Interactive API docs: http://127.0.0.1:8001/docs
- Readiness: http://127.0.0.1:8001/health
- Exact input contract: http://127.0.0.1:8001/v1/input-contract
- Machine-readable schema: http://127.0.0.1:8001/openapi.json

If port 8001 is occupied, choose another port and use that port in frontend requests. This is a local HTTP demo, not an HTTPS/cloud deployment.

## 3. Send the real sample request

In another terminal in `rppg_backend`:

```sh
curl -X POST http://127.0.0.1:8001/v1/vitals -H "Content-Type: application/json" --data-binary @sample_request.json
```

On Windows use `curl.exe` if `curl` resolves to a PowerShell alias.
Alternatively, use the bundled cross-platform standard-library client:

```sh
python send_sample.py --url http://127.0.0.1:8001
```

Use the virtual-environment Python path above if `python` points to another installation. A successful real sample returns HR around 105.241 BPM and experimental RR 8/min on the verified CPU environment. Exact last digits may vary with platform/package builds. The complete verified JSON is in `sample_response.json`.

## 4. Frontend integration

Read `API_FORMAT.md`. The frontend must send raw mean RGB for the exact three face ROIs as a `(600,9)` numeric array sampled uniformly at 20 Hz (30 seconds). Do not send video, images, normalized RGB, labels or a channel-first array. This backend accepts extracted RGB only and does not capture a webcam, find a face or extract ROI pixels from uploaded video.

Default CORS origins are http://localhost:3000, http://localhost:5173, http://127.0.0.1:3000 and http://127.0.0.1:5173. For another exact browser origin, set FRONTEND_ORIGINS BEFORE starting Uvicorn:

macOS / Linux:

```sh
FRONTEND_ORIGINS=http://localhost:8080 .venv/bin/python -m uvicorn backend_api:app --host 127.0.0.1 --port 8001
```

PowerShell:

```powershell
$env:FRONTEND_ORIGINS="http://localhost:8080"
.venv\Scripts\python.exe -m uvicorn backend_api:app --host 127.0.0.1 --port 8001
```

Comma-separate multiple origins; match scheme, hostname and port exactly. Credentials are not needed for this local demo.

## 5. Important behavior

HR uses the existing trained 1D CNN, 600x9 input, per-window/channel normalization and saved HR target mean/scale. Do not pre-normalize the RGB input. Its accuracy evidence is preliminary: one held-out test window, MAE 5.78 BPM; the mean-HR baseline did slightly better.

RR is a signal-processing estimate, NOT a trained/validated RR model. There are no independent RR references in the available training data. Show a numeric RR with an experimental badge. If rr_breaths_min is null, show RR unavailable and retain HR; null is not zero breathing. Never display rr_details.candidate_breaths_min as a fallback accepted reading.

Input errors return 422. Model readiness/inference errors return 503 with safe JSON and no fabricated vital signs. No arbitrary HR/RR values are substituted when a checked failure occurs. Passing these checks does not establish clinical validity or distinguish every motion/lighting artifact.

## 6. Package contents and verification

- Corrected backend, existing runtime dependencies and the unmodified trained checkpoint at outputs/real_cnn_v1/best_hr_model.pth.
- sample_request.json: one real, derived numeric RGB window for integration only; no raw video, image, subject labels or reference HR.
- sample_response.json, openapi.json and input_contract.json: generated from this packaged corrected API.
- smoke_test.py: self-contained integration/negative checks that do not require original datasets.
- PACKAGE_VERIFICATION.json and PACKAGE_CONTENTS_SHA256.json: actual extracted-package test results and integrity hashes.

The legacy api.py/vitals_inference.py/rr_inference.py files are included because the corrected modules import their existing schemas/helpers. They are not the recommended entrypoint. No training scripts, datasets, original tests, videos, university reports, backups or virtual environment are included.

Verification runs on an extracted copy isolated from the original project sources. It uses the already installed dependency environment to test the copied source and model; a clean internet dependency installation and Windows/Linux execution are not claimed. See PACKAGE_VERIFICATION.json for the actual host and results.

The /v1/input-contract field roi_definitions_source names the original preprocessing source for provenance; preprocessing is not a runtime import here. Exact landmark definitions and averaging requirements are included in API_FORMAT.md.
