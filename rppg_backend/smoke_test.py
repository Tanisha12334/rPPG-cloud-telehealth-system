"""Self-contained copied-backend checks. No dataset or original source is needed."""
import argparse
import contextlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
import time
import unittest
from unittest.mock import patch
import numpy as np
from fastapi.testclient import TestClient
from backend_api import create_app
from inference import DEFAULT_CHECKPOINT
from hr_pipeline import RGB_COLUMNS

ROOT = Path(__file__).resolve().parent


class PackageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.payload=json.loads((ROOT/'sample_request.json').read_text())
        cls.context=TestClient(create_app(),raise_server_exceptions=False)
        cls.client=cls.context.__enter__()

    @classmethod
    def tearDownClass(cls):cls.context.__exit__(None,None,None)

    def test_01_packaged_sources_and_checkpoint(self):
        for name in ['backend_api','backend_inference','api','inference','model','hr_pipeline','rr_inference_optimized','vitals_inference','rr_inference']:
            module=sys.modules[name]
            self.assertEqual(Path(module.__file__).resolve().parent,ROOT)
        self.assertTrue(DEFAULT_CHECKPOINT.is_file())
        self.assertTrue(DEFAULT_CHECKPOINT.resolve().is_relative_to(ROOT))

    def test_02_real_sample_hr(self):
        r=self.client.post('/v1/vitals',json=self.payload)
        self.assertEqual(r.status_code,200,r.text)
        self.assertTrue(np.isfinite(r.json()['hr_bpm']))
        self.assertAlmostEqual(r.json()['hr_bpm'],105.24101529181908,places=4)

    def test_03_real_sample_experimental_rr(self):
        r=self.client.post('/v1/vitals',json=self.payload)
        self.assertEqual(r.json()['rr_breaths_min'],8.0)
        self.assertEqual(r.json()['rr_status'],'experimental_unvalidated')
        self.assertFalse(r.json()['rr_details']['validated'])

    def test_04_health_and_contract(self):
        self.assertEqual(self.client.get('/health').json()['status'],'ready')
        contract=self.client.get('/v1/input-contract').json()
        self.assertEqual(contract['channels'],RGB_COLUMNS)
        self.assertEqual(contract['shape'],[600,9])

    def test_05_openapi_schema(self):
        schema=self.client.get('/openapi.json').json()
        self.assertIn('/v1/vitals',schema['paths'])
        self.assertEqual(set(schema['components']['schemas']['VitalsRequest']['required']),{'sample_hz','channels','rgb_window'})

    def test_06_insufficient_length(self):
        r=self.client.post('/v1/vitals',json={**self.payload,'rgb_window':self.payload['rgb_window'][:-1]})
        self.assertEqual(r.status_code,422);self.assertNotIn('hr_bpm',r.json())

    def test_07_constant_input(self):
        r=self.client.post('/v1/vitals',json={**self.payload,'rgb_window':[[100.0]*9 for _ in range(600)]})
        self.assertEqual(r.status_code,422);self.assertEqual(r.json()['detail']['code'],'INVALID_SIGNAL')

    def test_08_no_video_or_images(self):
        r=self.client.post('/v1/vitals',json={**self.payload,'video':'not accepted'})
        self.assertEqual(r.status_code,422)

    def test_09_internal_failure_safe(self):
        with patch('backend_inference.legacy_predict_hr',side_effect=RuntimeError('/private/model/path')):
            r=self.client.post('/v1/vitals',json=self.payload)
        self.assertEqual(r.status_code,503);self.assertEqual(r.json()['detail']['code'],'INFERENCE_UNAVAILABLE')
        self.assertNotIn('/private',r.text);self.assertNotIn('hr_bpm',r.json())

    def test_10_nonpositive_hr_not_returned(self):
        with patch('backend_inference.legacy_predict_hr',return_value=-1.0):
            r=self.client.post('/v1/vitals',json=self.payload)
        self.assertEqual(r.status_code,503);self.assertNotIn('hr_bpm',r.json())

    def test_11_wrong_channel_order_and_rate(self):
        for payload in [{**self.payload,'channels':list(reversed(RGB_COLUMNS))},{**self.payload,'sample_hz':30}]:
            self.assertEqual(self.client.post('/v1/vitals',json=payload).status_code,422)

    def test_12_rr_abstention_keeps_hr(self):
        window=np.asarray(self.payload['rgb_window']);window[:,[1,4,7]]=100
        r=self.client.post('/v1/vitals',json={**self.payload,'rgb_window':window.tolist()})
        self.assertEqual(r.status_code,200);self.assertGreater(r.json()['hr_bpm'],0)
        self.assertIsNone(r.json()['rr_breaths_min'])
        self.assertEqual(r.json()['rr_status'],'unavailable_signal_quality')

    def test_13_cors(self):
        r=self.client.options('/v1/vitals',headers={'Origin':'http://localhost:5173','Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'content-type'})
        self.assertEqual(r.status_code,200);self.assertEqual(r.headers['access-control-allow-origin'],'http://localhost:5173')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-dir',type=Path,help='Optional new output directory; default only prints tests')
    args=parser.parse_args()
    stream=io.StringIO();start=time.perf_counter()
    with contextlib.redirect_stderr(stream),contextlib.redirect_stdout(stream):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(PackageTests))
    print(stream.getvalue())
    summary={'tests_run':result.testsRun,'passed':result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
             'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),'duration_seconds':time.perf_counter()-start,
             'python':sys.version,'os':platform.platform(),
             'dependency_environment':'Existing installed dependencies; no clean dependency download or cross-OS execution claimed.',
             'source_isolation':'All nine runtime modules and checkpoint verified to load from the extracted package, not original project sources.',
             'transport':'in_process_ASGI','original_project_modified':False,
             'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','torch','fastapi','uvicorn','pydantic','starlette','httpx']}}
    if args.evidence_dir:
        args.evidence_dir.mkdir(parents=True,exist_ok=False)
        with (args.evidence_dir/'smoke_results.json').open('x') as f:json.dump(summary,f,indent=2)
        with (args.evidence_dir/'smoke_results.log').open('x') as f:f.write(stream.getvalue())
        with TestClient(create_app()) as client:
            response=client.post('/v1/vitals',json=json.loads((ROOT/'sample_request.json').read_text()))
            response.raise_for_status()
            for name,body in [('sample_response.json',response.json()),('openapi.json',client.get('/openapi.json').json()),('input_contract.json',client.get('/v1/input-contract').json())]:
                with (args.evidence_dir/name).open('x') as f:json.dump(body,f,indent=2,allow_nan=False)
    print(json.dumps(summary,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__=='__main__':main()
