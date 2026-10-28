"""Loopback SSE fixture for measured chat rates. Never runs model inference."""
import json
from pathlib import Path
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain')
        limit = 4 * 1024 * 1024
        if self.path == '/advertised':
            self.send_header('Content-Length', str(limit + 1))
            self.end_headers()
            self.wfile.write(b'x' * 16384)
            self.wfile.flush()
            time.sleep(8)
            return
        self.end_headers()
        self.wfile.write(b'x' * (limit if self.path == '/exact' else limit + 1))
        self.wfile.flush()
        if self.path != '/exact':
            time.sleep(8)

    def do_POST(self):
        if self.path.endswith('/cancel'):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{}')
            return
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        if body.get('stream_options', {}).get('include_usage') is not True:
            self.send_error(400, 'Usage must be requested')
            return
        prompt = body['messages'][-1]['content']
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.end_headers()

        def send(choices, usage=None):
            frame = {'id': 'chatcmpl-42', 'object': 'chat.completion.chunk',
                     'created': 0, 'model': 'speed-fixture', 'choices': choices}
            if usage is not None:
                frame['usage'] = usage
            self.wfile.write(('data: ' + json.dumps(frame) + '\n\n').encode())
            self.wfile.flush()

        send([{'index': 0, 'delta': {'reasoning_content': 'Fixture reasoning'}, 'finish_reason': None}])
        # One network chunk is deliberately many tokens; UI chunk counts must
        # never be used as the generated-token count or phase denominator.
        time.sleep(0.06)
        send([{'index': 0, 'delta': {'content': 'Fixture reply'}, 'finish_reason': 'stop'}])
        if prompt != 'legacy':
            send([], {'prompt_tokens': 1000, 'completion_tokens': 83, 'total_tokens': 1083,
                      'ax_engine': {'metrics_version': 1, 'prefill_tokens': 200,
                                    'prefill_time_us': 400000, 'decode_tokens': 80,
                                    'decode_time_us': 2000000}})
        if prompt == 'stopped':
            time.sleep(1)
        if prompt != 'truncated':
            try:
                self.wfile.write(b'data: [DONE]\n\n')
            except BrokenPipeError:
                pass


server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
Path(sys.argv[1]).write_text(str(server.server_port))
server.serve_forever()
