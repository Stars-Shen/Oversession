"""Small, cached Responses API client; credentials never enter artifacts."""
import hashlib
import json
import os
import random
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(path)


class API:
    def __init__(self, model='gpt-6-luna'):
        self.model = model
        self.cache = ROOT / 'runs/api_cache'
        self.cache.mkdir(parents=True, exist_ok=True)
        self.key = os.environ.get('OPENAI_API_KEY') or (ROOT / '.private/openai_key').read_text().strip()

    def call(self, instructions, content, max_tokens=4096, json_output=False, effort='low'):
        body = dict(model=self.model, instructions=instructions, input=content,
                    max_output_tokens=max_tokens, store=False, reasoning={'effort': effort})
        if json_output:
            body['text'] = {'format': {'type': 'json_object'}}
            body['input'] = 'Return JSON.\n' + content
        signature = digest(body)
        path = self.cache / (signature + '.json')
        if path.exists():
            return json.loads(path.read_text())
        session = requests.Session()
        session.trust_env = False
        proxy = os.environ.get('MEMORY_PROXY', 'http://127.0.0.1:41227')
        session.proxies = {'http': proxy, 'https': proxy}
        start = time.time()
        for attempt in range(10):
            try:
                response = session.post('https://api.openai.com/v1/responses',
                                        headers={'Authorization': 'Bearer ' + self.key},
                                        json=body, timeout=180)
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 9:
                    raise RuntimeError('API connection failed after ten attempts') from None
                time.sleep(min(10, 2 ** attempt))
                continue
            if (response.status_code == 429 or 500 <= response.status_code < 600) and attempt < 9:
                delay = float(response.headers.get('retry-after-ms', 0)) / 1000
                delay = delay or float(response.headers.get('retry-after', 0))
                time.sleep(max(delay, min(10, 2 ** attempt)) + random.random())
                continue
            if response.status_code != 200:
                raise RuntimeError('API HTTP {}: {}'.format(response.status_code,
                                   response.text[:500].replace(self.key, '[REDACTED]')))
            data = response.json()
            if data.get('status') != 'completed':
                if (data.get('incomplete_details') or {}).get('reason') == 'max_output_tokens' and body['max_output_tokens'] < max(1024, max_tokens * 2):
                    body['max_output_tokens'] = max(1024, max_tokens * 2)
                    continue
                raise RuntimeError('Incomplete response: ' + str(data.get('status')))
            output = '\n'.join(c['text'] for item in data.get('output', [])
                               for c in item.get('content', []) if c.get('type') == 'output_text')
            if not output.strip():
                raise RuntimeError('API returned no output text')
            result = dict(text=output, model=data.get('model'), usage=data.get('usage'),
                          elapsed_seconds=time.time() - start, request_hash=signature)
            if body['max_output_tokens'] != max_tokens:
                result['served_request_hash'] = digest(body)
                result['served_max_output_tokens'] = body['max_output_tokens']
            if json_output:
                result['parsed'] = json.loads(output)
            atomic_json(path, result)
            return result
