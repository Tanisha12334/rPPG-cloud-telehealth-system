"""Additive hardened API: reuse original schemas/routes without rewriting api.py.

Run: .venv/bin/python -m uvicorn backend_api:app --host 127.0.0.1 --port 8001
"""
import logging
import numpy as np
from fastapi import HTTPException
from api import create_app as legacy_create_app, VitalsRequest, VitalsResponse
from inference import DEFAULT_CHECKPOINT
from backend_inference import predict_vitals, SignalInputError, InferenceFailure

logger = logging.getLogger(__name__)


def create_app(checkpoint_path=DEFAULT_CHECKPOINT):
    application = legacy_create_app(checkpoint_path)
    application.title = 'rPPG HR + experimental RR - checked backend'
    application.version = '1.1.0'
    # Only alter this freshly constructed app object. Legacy api:app is intact.
    application.router.routes[:] = [route for route in application.router.routes
                                    if getattr(route, 'path', None) != '/v1/vitals']

    @application.post('/v1/vitals', response_model=VitalsResponse)
    def vitals(payload: VitalsRequest):
        if not application.state.ready:
            raise HTTPException(status_code=503, detail={'code': 'MODEL_NOT_READY', 'message': 'Model is not ready'})
        try:
            return predict_vitals(np.asarray(payload.rgb_window, dtype=np.float32), checkpoint_path)
        except SignalInputError as exc:
            raise HTTPException(status_code=422, detail={'code': 'INVALID_SIGNAL', 'message': str(exc)}) from exc
        except Exception as exc:
            # Do not expose exception text, paths, weights, or fabricated readings.
            logger.error('Vitals inference unavailable (%s)', type(exc).__name__)
            raise HTTPException(status_code=503, detail={'code': 'INFERENCE_UNAVAILABLE',
                                                       'message': 'Vitals inference is unavailable; retry later'}) from exc

    return application


app = create_app()
