"""Phase 1 FastAPI backend: raw nine-channel RGB -> HR and experimental RR."""
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator
import numpy as np

from hr_pipeline import RGB_COLUMNS
from inference import DEFAULT_CHECKPOINT, _load_model
from vitals_inference import predict_vitals

RGBValue = Annotated[float, Field(strict=True, ge=0, le=255, allow_inf_nan=False)]
RGBRow = Annotated[list[RGBValue], Field(min_length=9, max_length=9)]
RGBWindow = Annotated[list[RGBRow], Field(min_length=600, max_length=600)]


class VitalsRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    sample_hz: Literal[20]
    channels: list[str] = Field(min_length=9, max_length=9)
    rgb_window: RGBWindow

    @field_validator('channels')
    @classmethod
    def check_order(cls, values):
        if values != RGB_COLUMNS:
            raise ValueError('channels must exactly match GET /v1/input-contract')
        return values


class RRQuality(BaseModel):
    active_rois: int
    agreeing_rois: int
    roi_peak_breaths_min: list[float | None]
    peak_concentration: float | None
    confidence_calibrated: Literal[False]


class RRDetails(BaseModel):
    rr_breaths_min: float | None
    status: Literal['experimental_unvalidated', 'unavailable_signal_quality']
    validated: Literal[False]
    method: str
    candidate_breaths_min: float | None
    reasons: list[str]
    quality: RRQuality
    limitations: str


class WindowContract(BaseModel):
    sample_hz: Literal[20]
    window_seconds: Literal[30]
    samples: Literal[600]
    channels: list[str]


class VitalsResponse(BaseModel):
    hr_bpm: float
    hr_status: Literal['preliminary_cnn']
    rr_breaths_min: float | None
    rr_status: Literal['experimental_unvalidated', 'unavailable_signal_quality']
    rr_details: RRDetails
    input: WindowContract
    limitations: list[str]


def create_app(checkpoint_path=DEFAULT_CHECKPOINT):
    checkpoint = Path(checkpoint_path).resolve()

    @asynccontextmanager
    async def lifespan(application):
        # Validate checkpoint and warm the reusable inference cache before readiness.
        _load_model(str(checkpoint), 'cpu')
        application.state.ready = True
        try:
            yield
        finally:
            application.state.ready = False

    application = FastAPI(title='rPPG HR + experimental RR', version='1.0.0', lifespan=lifespan,
                          description='Phase 1 research demonstration. RR has no independent reference validation.')
    application.state.ready = False
    origins = [value.strip() for value in os.getenv(
        'FRONTEND_ORIGINS', 'http://localhost:3000,http://localhost:5173,http://127.0.0.1:3000,http://127.0.0.1:5173'
    ).split(',') if value.strip()]
    application.add_middleware(CORSMiddleware, allow_origins=origins,
                               allow_methods=['GET', 'POST'], allow_headers=['Content-Type'])

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError):
        # Omit raw 600x9 inputs and non-JSON NaN/Inf values from error responses.
        return JSONResponse(status_code=422, content={'detail': [
            {'loc': list(error['loc']), 'msg': error['msg'], 'type': error['type']}
            for error in exc.errors()
        ]})

    @application.get('/health')
    def health():
        if not application.state.ready:
            raise HTTPException(status_code=503, detail='Model is not ready')
        return {'status': 'ready', 'hr_model': 'existing_HeartRateCNN_v1',
                'rr_method': 'experimental_riiv_green_v1', 'rr_validated': False}

    @application.get('/v1/input-contract')
    def input_contract():
        return {'schema_version': '1.0', 'shape': [600, 9], 'sample_hz': 20,
                'window_seconds': 30, 'channels': RGB_COLUMNS,
                'rgb_range': [0, 255], 'normalization': 'send_raw_RGB_means',
                'timestamps': 'uniform_20Hz_samples_required',
                'roi_definitions_source': 'existing preprocess.py ROIS',
                'rr_unavailable_value': None, 'rr_validation': 'no_independent_RR_reference',
                'request_fields': ['sample_hz', 'channels', 'rgb_window']}

    @application.post('/v1/vitals', response_model=VitalsResponse)
    def vitals(payload: VitalsRequest):
        if not application.state.ready:
            raise HTTPException(status_code=503, detail='Model is not ready')
        try:
            return predict_vitals(np.asarray(payload.rgb_window, dtype=np.float32), checkpoint)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return application


app = create_app()
