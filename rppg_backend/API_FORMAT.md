# Exact API contract

## Request

POST /v1/vitals, Content-Type: application/json. Exactly these three keys:

```json
{
  "sample_hz": 20,
  "channels": ["forehead_r", "forehead_g", "forehead_b", "left_cheek_r", "left_cheek_g", "left_cheek_b", "right_cheek_r", "right_cheek_g", "right_cheek_b"],
  "rgb_window": [[123.4, 101.2, 80.5, 120.1, 98.7, 79.9, 122.0, 100.2, 81.1]]
}
```

The single row above illustrates ONE ROW ONLY and is not a valid full request. Use `sample_request.json` for the exact complete 600-row payload.

- rgb_window: exactly 600 rows, each with exactly 9 real JSON numbers in [0,255]. No strings, booleans, nulls, NaN or infinity. A constant window is rejected as insufficient signal.
- sample_hz: exactly 20. Caller must resample to the uniform 20 Hz grid. Rows represent relative times 0.00, 0.05, ..., 29.95 seconds. Timing cannot be inferred or verified from RGB values alone.
- channels: exact ordered list shown. Each ROI uses R,G,B, not B,G,R.
- Supply RAW pixel means. HR applies its saved normalization internally. No extra fields, ground truth or video data.

Use these exact original MediaPipe ROI polygon landmark indices:

```json
{
  "forehead": [
    10,
    67,
    69,
    104,
    108,
    151,
    299,
    337,
    338
  ],
  "left_cheek": [
    118,
    119,
    100,
    126,
    209,
    49,
    129,
    203,
    205,
    50
  ],
  "right_cheek": [
    347,
    348,
    329,
    355,
    429,
    279,
    358,
    423,
    425,
    280
  ]
}
```

The original averaging procedure clips the polygon bounding box to the image, rounds polygon points relative to that box, fills its mask, and averages RGB pixels inside the mask. The caller must use compatible face landmarks and polygon processing. Lighting/motion and Python-versus-browser extraction parity still require integration validation. Missing-face data must not be filled with arbitrary RGB or HR values. Capture a valid 30-second window before submitting.

## Response, HTTP 200

The complete machine-readable response schema is included in `openapi.json` and is available at /openapi.json. `sample_response.json` is a genuine full output from the packaged API. Top-level keys:

| Field | Type / meaning |
|---|---|
| hr_bpm | Positive finite number, beats/minute |
| hr_status | "preliminary_cnn" |
| rr_breaths_min | Number in breaths/minute, or null when quality checks abstain |
| rr_status | "experimental_unvalidated" or "unavailable_signal_quality" |
| rr_details | Detailed RR object described below |
| input | {sample_hz:20, window_seconds:30, samples:600, channels:[exact ordered names]} |
| limitations | Array of human-readable limitations |

rr_details keys:

- rr_breaths_min: same accepted number/null as top level.
- status: same status string as rr_status.
- validated: false.
- method: "experimental_riiv_green_v1".
- candidate_breaths_min: diagnostic spectral peak or null; do not display it when the accepted RR is null.
- reasons: array of abstention reason strings, empty for an accepted experimental candidate.
- quality: {active_rois: integer, agreeing_rois: integer, roi_peak_breaths_min: [number/null,number/null,number/null], peak_concentration: number/null, confidence_calibrated:false}.
- limitations: string describing unvalidated status, artifact risks, 6-30/min search band and 2/min frequency spacing.

These are 30-second window estimates, not instantaneous measurements. RR quality values are NOT confidence percentages. Numeric RR must be labeled experimental; null means unavailable, not apnea/zero breathing.

## Errors

Invalid JSON/schema/range/shape/channel order/rate: HTTP 422 with {"detail":[{"loc":[...],"msg":"...","type":"..."}]}.

Invalid/constant signal after schema parsing: HTTP 422 with {"detail":{"code":"INVALID_SIGNAL","message":"..."}}.

Vitals request before readiness: HTTP 503 with {"detail":{"code":"MODEL_NOT_READY","message":"Model is not ready"}}.

Internal model/estimator failure: HTTP 503 with {"detail":{"code":"INFERENCE_UNAVAILABLE","message":"Vitals inference is unavailable; retry later"}}. No HR/RR field is returned and no fallback reading is fabricated.

Health before readiness uses HTTP 503 with a simple detail string. Missing/corrupt checkpoints can prevent server startup. Unsupported methods return HTTP 405. This local backend has no login, HTTPS or cloud deployment.

## Frontend fetch

```javascript
const response = await fetch("http://127.0.0.1:8001/v1/vitals", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({sample_hz: 20, channels, rgb_window: rawRgbWindow})
});
const result = await response.json();
if (!response.ok) {
  // Display an input/service error; do not substitute HR/RR numbers.
  throw new Error(JSON.stringify(result.detail));
}
// Display result.hr_bpm with BPM.
// If result.rr_breaths_min === null, show RR unavailable.
// Otherwise display breaths/min with an experimental/unvalidated badge.
```
