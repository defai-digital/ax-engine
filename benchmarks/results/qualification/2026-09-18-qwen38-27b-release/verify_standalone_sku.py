"""Run a selected standalone payload on the SKU in a clean environment."""
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

root = Path(sys.argv[1]).resolve()
payload = root / (sys.argv[2] if len(sys.argv) > 2 else 'released-standalone')
source = root / 'product-final-source'
out = root / (sys.argv[3] if len(sys.argv) > 3 else 'standalone-verification')
out.mkdir(exist_ok=False)
helper_python = root / 'standalone-python/bin/python3'
assert helper_python.is_file()
helper_identity = json.loads(subprocess.check_output([str(helper_python), '-c',
    "import importlib.util,json,sys; print(json.dumps({'python':sys.version,'ax_engine':importlib.util.find_spec('ax_engine') is not None,'mlx':importlib.util.find_spec('mlx') is not None}))"], text=True))
assert not helper_identity['ax_engine'] and not helper_identity['mlx']
(out / 'helper-python.json').write_text(json.dumps(helper_identity, indent=2) + '\n')
env = {'PATH': str(payload) + ':' + str(helper_python.parent) + ':/usr/bin:/bin:/usr/sbin:/sbin',
       'HOME': str(out), 'TMPDIR': str(out)}
expected = {row['mode']: row for row in json.loads((root / 'product-final-lifecycle/results.json').read_text())}
hashes = {}
for name in ('ax-engine', 'ax-engine-server', 'ax-engine-bench', 'libmlx.dylib', 'libjaccl.dylib'):
    image = payload / name
    subprocess.run(['codesign', '--verify', '--strict', str(image)], check=True, env=env)
    hashes[name] = hashlib.sha256(image.read_bytes()).hexdigest()
(out / 'payload-hashes.json').write_text(json.dumps(hashes, indent=2) + '\n')
subprocess.run(['bash', str(source / 'scripts/validate-standalone.sh'), '--doctor', str(payload)],
               check=True, env=env, cwd=out)
doctor = json.loads(subprocess.check_output([str(payload / 'ax-engine'), 'doctor',
    '--mlx-model-artifacts-dir', str(root / 'model'), '--json'], env=env, cwd=out, text=True))
assert doctor['result'] == 'ready'
(out / 'doctor.json').write_text(json.dumps(doctor, indent=2) + '\n')
request_body = json.loads((root / 'product-final-lifecycle/fixed-request.json').read_text())
results = []
for mode in ('default', 'mtp', 'local_directory'):
    port = 31504
    with socket.socket() as probe:
        assert probe.connect_ex(('127.0.0.1', port)) != 0, 'standalone probe port is occupied'
    command = [str(payload / 'ax-engine'), 'serve', 'qwen3.8-27b:axq', '--offline',
               '--hf-cache-root', str(root / 'product-final-cache'), '--port', str(port)]
    if mode == 'mtp':
        command += ['--', '--mlx-mtp-policy', 'required', '--mlx-mtp-disable-ngram-stacking']
    launch_env = dict(env)
    if mode == 'local_directory':
        command = [str(payload / 'ax-engine'), 'serve', str(root / 'model'), '--port', str(port)]
        launch_env['PATH'] = str(payload)
    row = {'mode': mode, 'command': command, 'environment': launch_env, 'passed': False}
    with (out / (mode + '-server.log')).open('w') as log:
        process = subprocess.Popen(command, cwd=out, env=launch_env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 420
            while time.monotonic() < deadline:
                assert process.poll() is None, 'standalone server exited before readiness'
                try:
                    with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as response:
                        if response.status == 200:
                            break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('standalone server readiness')
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=10) as response:
                models = json.load(response)['data']
            assert len(models) == 1
            actual_request = dict(request_body, model=models[0]['id'])
            row['request'] = actual_request
            request = urllib.request.Request(f'http://127.0.0.1:{port}/v1/generate',
                data=json.dumps(actual_request).encode(), headers={'Content-Type': 'application/json'})
            with urllib.request.urlopen(request, timeout=180) as response:
                result = json.load(response)
            (out / (mode + '-response.json')).write_text(json.dumps(result, indent=2) + '\n')
            assert result['status'] == 'finished'
            oracle = 'default' if mode == 'local_directory' else mode
            assert result['output_tokens'] == expected[oracle]['cold']['output_tokens']
            mtp = result['performance']['mtp']
            assert mtp['requested'] is (mode == 'mtp') and mtp['active'] is (mode == 'mtp')
            assert mtp['direct_fallback_steps'] == 0
            row.update(passed=True, output_tokens=len(result['output_tokens']), mtp=mtp)
        except Exception as error:
            row['error'] = repr(error)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=10)
            row['server_exit_code'] = process.returncode
            row['owned_server_reaped'] = process.poll() is not None
            results.append(row)
            (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
assert all(row['passed'] and row['owned_server_reaped'] for row in results), results
print('Selected standalone payload: clean-env doctor and default/explicit-MTP token probes pass on the primary SKU.')
