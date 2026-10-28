"""Opt-in, headless real Sparkle installation tests using disposable app bundles.

Run AX_GUI_UPDATER_INTEGRATION=1 python3 -m unittest scripts/test_macos_updater_integration.py
Requires macOS, Xcode command-line tools, Node, and the resolved Swift package.
No production keys, installed applications, model weights, or user data are used.
"""
from __future__ import annotations

import base64
import functools
import http.server
import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / 'packaging/macos/AXEngineApp/.build/artifacts/sparkle/Sparkle'


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


@unittest.skipUnless(sys.platform == 'darwin' and os.environ.get('AX_GUI_UPDATER_INTEGRATION') == '1',
                     'Set AX_GUI_UPDATER_INTEGRATION=1 on macOS to run real installation fixtures')
class UpdateIntegrationTests(unittest.TestCase):
    def run_probe(self, mode: str) -> tuple[str, str]:
        with tempfile.TemporaryDirectory(prefix='ax-updater-test-') as temporary:
            root = Path(temporary)
            def run(*args):
                return subprocess.check_output([str(a) for a in args], stderr=subprocess.STDOUT)
            run('clang', '-fobjc-arc', '-framework', 'AppKit', '-framework', 'Sparkle',
                '-F', ARTIFACT/'Sparkle.xcframework/macos-arm64_x86_64',
                '-Wl,-rpath,@executable_path/../Frameworks', ROOT/'scripts/fixtures/macos-updater-probe.m',
                '-o', root/'Probe')
            keyfile = root/'test-key'
            keyfile.write_text(base64.b64encode(os.urandom(32)).decode())
            keyfile.chmod(0o600)
            public_key = run('node', '-e', """
const c=require('crypto'),f=require('fs');
const key=c.createPrivateKey({key:Buffer.concat([Buffer.from('302e020100300506032b657004220420','hex'),
  Buffer.from(f.readFileSync(process.argv[1],'utf8'),'base64')]),format:'der',type:'pkcs8'});
console.log(c.createPublicKey(key).export({format:'der',type:'spki'}).subarray(-32).toString('base64'));
""", keyfile).decode().strip()
            server = http.server.ThreadingHTTPServer(('localhost', 0), functools.partial(QuietHandler, directory=temporary))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            port = server.server_port
            identifier = 'com.automatosx.ax-engine.updateprobe.' + uuid.uuid4().hex
            process = None
            try:
                for version, folder in [('1.0.0', 'installed'), ('2.0.0', 'new')]:
                    app = root/folder/'Update Probe.app'
                    (app/'Contents/MacOS').mkdir(parents=True)
                    (app/'Contents/Resources').mkdir()
                    shutil.copy2(root/'Probe', app/'Contents/MacOS/Probe')
                    info = {'CFBundleExecutable': 'Probe', 'CFBundleIdentifier': identifier,
                            'CFBundleName': 'Update Probe', 'CFBundlePackageType': 'APPL',
                            'CFBundleVersion': version, 'CFBundleShortVersionString': version,
                            'LSUIElement': True, 'SUFeedURL': f'http://localhost:{port}/appcast.xml',
                            'SUPublicEDKey': public_key, 'SUEnableAutomaticChecks': False,
                            'SUAllowsAutomaticUpdates': False, 'SUEnableSystemProfiling': False,
                            'AXProbeDirectory': temporary, 'AXProbeMode': mode,
                            'NSAppTransportSecurity': {'NSAllowsLocalNetworking': True}}
                    (app/'Contents/Info.plist').write_bytes(plistlib.dumps(info))
                    run(sys.executable, ROOT/'scripts/embed-macos-updater.py', app, ARTIFACT)
                    run('codesign', '--force', '--sign', '-', app)
                archive = root/'update.zip'
                run('ditto', '-c', '-k', '--keepParent', root/'new/Update Probe.app', archive)
                signature = run(ARTIFACT/'bin/sign_update', '--ed-key-file', keyfile, '-p', archive).decode().strip()
                if mode == 'tampered':
                    signature = base64.b64encode(bytes(64)).decode()
                offered = '1.0.0' if mode == 'current' else '2.0.0'
                (root/'appcast.xml').write_text(f'''<?xml version="1.0"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle"><channel>
<title>Local update probe</title><item><title>{offered}</title><sparkle:version>{offered}</sparkle:version>
<enclosure url="http://localhost:{port}/update.zip" length="{archive.stat().st_size}"
type="application/octet-stream" sparkle:edSignature="{signature}" /></item></channel></rss>''')
                if mode == 'unavailable':
                    (root/'appcast.xml').unlink()
                events_file = root/'events.txt'
                events_file.write_text('')
                sentinel = root/'settings.json'
                sentinel.write_bytes(b'{"preserved":true}\n')
                app = root/'installed/Update Probe.app'
                with (root/'process.log').open('w') as log:
                    process = subprocess.Popen([str(app/'Contents/MacOS/Probe')], stdout=log, stderr=subprocess.STDOUT)
                    deadline = time.monotonic() + 100
                    while time.monotonic() < deadline:
                        events = events_file.read_text()
                        if any(marker in events for marker in ('relaunched 2.0.0', 'error ', 'up to date', 'cancelled', 'timeout')):
                            break
                        time.sleep(0.1)
                    else:
                        self.fail('Updater did not finish: ' + events_file.read_text())
                    if mode != 'install':
                        process.wait(timeout=5)
                    version = plistlib.loads((app/'Contents/Info.plist').read_bytes())['CFBundleVersion']
                    self.assertEqual(sentinel.read_bytes(), b'{"preserved":true}\n')
                    return events, version
            finally:
                if process is not None and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=10)
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
                # Only the randomly named fixture's own Sparkle cache/preferences.
                shutil.rmtree(Path.home()/'Library/Caches'/identifier, ignore_errors=True)
                (Path.home()/'Library/Preferences'/f'{identifier}.plist').unlink(missing_ok=True)

    def test_signed_update_waits_for_shutdown_installs_and_relaunches(self):
        events, version = self.run_probe('install')
        self.assertEqual(version, '2.0.0')
        self.assertIn('ready to install', events)
        self.assertLess(events.index('graceful shutdown requested'), events.index('cleanup finished'))
        self.assertLess(events.index('cleanup finished'), events.index('relaunched 2.0.0'))

    def test_bad_archive_signature_keeps_the_installed_app(self):
        events, version = self.run_probe('tampered')
        self.assertEqual(version, '1.0.0')
        self.assertIn('error ', events)
        self.assertIn('downloading', events.splitlines())
        self.assertRegex(events, r'(?i)(signature|signed|validation)')
        self.assertNotIn('ready to install', events)
        self.assertNotIn('graceful shutdown requested', events)

    def test_unavailable_feed_keeps_the_app_and_does_not_start_a_download(self):
        events, version = self.run_probe('unavailable')
        self.assertEqual(version, '1.0.0')
        self.assertIn('error ', events)
        self.assertNotIn('downloading', events.splitlines())
        self.assertNotIn('graceful shutdown requested', events)

    def test_current_version_does_not_download(self):
        events, version = self.run_probe('current')
        self.assertEqual(version, '1.0.0')
        self.assertIn('up to date', events)
        self.assertNotIn('downloading', events.splitlines())

    def test_cancelled_update_does_not_download_or_replace_app(self):
        events, version = self.run_probe('cancel')
        self.assertEqual(version, '1.0.0')
        self.assertIn('cancelled', events)
        self.assertNotIn('downloading', events.splitlines())


if __name__ == '__main__':
    unittest.main()
