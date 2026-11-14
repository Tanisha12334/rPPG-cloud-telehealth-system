"""Send the bundled complete real request to a locally running backend."""
import argparse
import json
from pathlib import Path
from urllib.request import Request,urlopen
from urllib.error import HTTPError,URLError

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='http://127.0.0.1:8001')
    a=p.parse_args()
    payload=(Path(__file__).resolve().parent/'sample_request.json').read_bytes()
    request=Request(a.url.rstrip('/')+'/v1/vitals',data=payload,headers={'Content-Type':'application/json'},method='POST')
    try:
        with urlopen(request,timeout=60) as response:print(json.dumps(json.load(response),indent=2))
    except HTTPError as exc:
        print(f'HTTP {exc.code}: {exc.read().decode() }');raise SystemExit(1)
    except URLError as exc:
        print(f'Connection failed: {exc.reason}. Start backend_api:app first and check --url.');raise SystemExit(1)
