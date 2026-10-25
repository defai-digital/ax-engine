#!/usr/bin/python3
"""Explicit test fixture. No model weights or inference implementation."""
import json
import os
from pathlib import Path
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def emit(value):
    print(json.dumps(value), flush=True)


if sys.argv[1:3] == ['download', '--list']:
    emit({'schema_version': 'ax.download_options.v1', 'download_backend': 'native-rust-public-v1', 'targets': [
        {'alias': 'Fixture model', 'repo_id': 'AutomatosX/Fixture', 'revision': 'abc123', 'mtp_included': False}]})
    sys.exit(0)
if sys.argv[1:2] == ['download']:
    dest = Path(sys.argv[sys.argv.index('--dest') + 1])
    for i in range(4):
        emit({'event': 'progress', 'done': i, 'total': 4, 'file': 'Fixture transfer'})
        time.sleep(0.1)
    if 'Failure' in sys.argv[2]:
        emit({'schema_version': 'ax.download_model.v1', 'status': 'download_failed', 'errors': ['Fixture transfer failed']})
        sys.exit(2)
    dest.mkdir(parents=True, exist_ok=True)
    if 'Missing' not in sys.argv[2]:
        (dest/'config.json').write_text('{}')
        (dest/'model-manifest.json').write_text('{}')
        (dest/'model.safetensors').write_bytes(b'fixture')
    emit({'schema_version': 'ax.download_model.v1', 'status': 'ready', 'dest': str(dest)})
    sys.exit(0)

port = int(sys.argv[sys.argv.index('--port') + 1])
key = os.environ.get('AX_ENGINE_API_KEY', '')
class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *args):
        pass
    def json_response(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def auth(self):
        if key and self.headers.get('Authorization') != 'Bearer ' + key:
            self.json_response({'error': {'message': 'Unauthorized'}}, 401)
            return False
        return True
    def do_GET(self):
        if not self.auth(): return
        if self.path == '/redirect':
            self.send_response(302)
            self.send_header('Location', '/v1/models')
            self.send_header('Content-Length', '0')
            self.end_headers()
        elif self.path == '/v1/models':
            self.json_response({'object': 'list', 'data': [{'id': 'fixture', 'owned_by': 'ax-engine',
                'context_length': 2048, 'capabilities': {'temperature': True}}]})
        elif self.path == '/metrics':
            if Path(__file__).with_suffix('.fail-metrics').exists():
                self.json_response({'error': 'Metrics temporarily unavailable'}, 503)
                return
            body = b'ax_engine_generation_active_streams 0\nax_engine_scheduled_tokens_total 24\n'
            self.send_response(200); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
        else: self.json_response({'status': 'ok'})
    def do_POST(self):
        if not self.auth(): return
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))) or b'{}')
        if self.path.endswith('/cancel'):
            self.json_response({}, 403 if self.path == '/v1/requests/999/cancel' else 200)
            return
        prompt = body['messages'][-1]['content']
        if prompt == 'redirect' and self.path == '/v1/chat/completions':
            self.send_response(307)
            self.send_header('Location', '/v1/redirect-completion')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Connection', 'close')
        self.end_headers()
        try:
            if prompt == 'error':
                self.wfile.write(b'event: error\ndata: {"error":{"message":"Fixture generation failure"}}\n\n')
            else:
                for i in range(100 if prompt == 'slow' else 1 if prompt == 'parameters' else 2):
                    event = {'id': 'chatcmpl-1', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'fixture',
                             'choices': [{'index': 0, 'delta': {'content': (json.dumps(body) if prompt == 'parameters' else json.dumps(body['messages']) if prompt == 'history' else 'Hello ') if i == 0 else 'world'}, 'finish_reason': None}]}
                    self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode()); self.wfile.flush()
                    if prompt == 'slow': time.sleep(0.1)
                if prompt != 'truncate': self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError): pass
        self.close_connection = True

server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
print('Fixture listening', flush=True)
server.serve_forever()
