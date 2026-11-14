"""Responsive experimental HR backend; original CNN API stays available.

Run: .venv/bin/python -m uvicorn hr_api:app --host 127.0.0.1 --port 8000
"""
import logging
from contextlib import asynccontextmanager
import numpy as np
from fastapi import HTTPException
from api import create_app as cnn_create_app, VitalsRequest, VitalsResponse
from backend_inference import SignalInputError
from hr_signal import estimate_hr, HRSignalUnavailable, pulse_candidate
from rr_inference_optimized import estimate_rr
from typing import Literal

logger = logging.getLogger(__name__)


class SignalVitalsResponse(VitalsResponse):
    hr_bpm: float | None
    hr_status: Literal['experimental_pos', 'unavailable_signal_quality']
    hr_details: dict
    pulse_candidate_bpm: float | None = None
    pulse_candidate_status: Literal['not_needed', 'low_quality_unverified', 'unavailable'] = 'not_needed' 


def create_app():
    application = cnn_create_app()

    @asynccontextmanager
    async def signal_lifespan(app):
        # POS does not depend on the old CNN checkpoint or its readiness.
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False

    application.router.lifespan_context = signal_lifespan
    application.title = 'Experimental POS HR + RR'
    application.version = '2.3.0'
    application.router.routes[:] = [route for route in application.router.routes
                                    if getattr(route, 'path', None) not in ('/v1/vitals', '/health')]

    @application.get('/health')
    def health():
        if not application.state.ready:
            raise HTTPException(status_code=503, detail='Backend is not ready')
        return {'status': 'ready', 'hr_model': 'pos_segment_mean_v2', 'hr_validated': False, 'api_version': '2.3.0',
                'rr_method': 'experimental_riiv_green_v1', 'rr_validated': False}

    @application.post('/v1/vitals', response_model=SignalVitalsResponse)
    def vitals(payload: VitalsRequest):
        if not application.state.ready:
            raise HTTPException(status_code=503, detail='Backend is not ready')
        try:
            window = np.asarray(payload.rgb_window, dtype=np.float64)
            candidate = None
            try:
                hr = estimate_hr(window)
                hr_status = 'experimental_pos'
            except HRSignalUnavailable as exc:
                hr = dict(exc.details, hr_bpm=None, message=str(exc))
                hr_status = 'unavailable_signal_quality'
                candidate = pulse_candidate(window)
                hr['pulse_candidate_details'] = candidate
            rr = estimate_rr(window)
            return {'hr_bpm': hr['hr_bpm'], 'hr_status': hr_status, 'hr_details': hr,
                    'pulse_candidate_bpm': candidate['bpm'] if candidate else None,
                    'pulse_candidate_status': ('low_quality_unverified' if candidate else
                                               'unavailable' if hr_status == 'unavailable_signal_quality' else 'not_needed'),
                    'rr_breaths_min': rr['rr_breaths_min'], 'rr_status': rr['status'], 'rr_details': rr,
                    'input': {'sample_hz': 20, 'window_seconds': 30, 'samples': 600, 'channels': payload.channels},
                    'limitations': ['HR uses experimental POS signal processing, not the CNN. Quality gates do not establish accuracy.',
                                    'pulse_candidate_bpm is an unverified diagnostic, may represent noise, and is not accepted HR.',
                                    'Evaluation reuses four subjects; stored HR labels can disagree with the contact waveform, and errors remain.',
                                    'RR has no independent reference evaluation.']}
        except SignalInputError as exc:
            raise HTTPException(status_code=422, detail={'code': 'INVALID_SIGNAL', 'message': str(exc)}) from exc
        except Exception as exc:
            logger.error('Signal inference unavailable (%s)', type(exc).__name__)
            raise HTTPException(status_code=503, detail='Vitals inference unavailable') from exc

    return application


app = create_app()
