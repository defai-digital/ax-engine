"""Local release provenance checks; Apple network gates are exercised separately."""
import importlib.util
import json
from pathlib import Path
import plistlib
import subprocess
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('gui_release', Path(__file__).with_name('release-macos-app.py'))
release = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(release)


class ReleaseTests(unittest.TestCase):
    def test_runtime_stages_unwind_server_instead_of_stale_abort_server(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for profile in ('release', 'release-server'):
                (root / 'target' / profile).mkdir(parents=True)
            for name in ('ax-engine', 'ax-engine-bench', 'ax-engine-server'):
                binary = root / 'target/release' / name
                binary.write_bytes(b'abort')
                binary.chmod(0o755)
            server = root / 'target/release-server/ax-engine-server'
            server.write_bytes(b'unwind')
            server.chmod(0o755)
            with patch.object(release, 'ROOT', root), patch.object(release, 'run') as run:
                release.build_runtime_binaries(root / 'staged', 'test-pin')
            self.assertEqual((root / 'staged/ax-engine-server').read_bytes(), b'unwind')
            self.assertEqual((root / 'staged/ax-engine').read_bytes(), b'abort')
            self.assertTrue((root / 'staged/ax-engine-server').stat().st_mode & 0o111)
            commands = [call.args for call in run.call_args_list]
            self.assertEqual(len(commands), 2)
            self.assertIn('--release', commands[0])
            self.assertNotIn('ax-engine-server', commands[0])
            self.assertEqual(commands[1][-6:], ('--profile', 'release-server', '-p',
                                              'ax-engine-server', '--bin', 'ax-engine-server'))
            for command in commands:
                self.assertEqual(command[:4], ('rustup', 'run', 'test-pin', 'cargo'))
                self.assertIn(root / 'target', command)

    def test_tree_digest_binds_content_modes_links_and_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            app = Path(temp) / 'App.app'
            app.mkdir()
            binary = app / 'binary'; binary.write_bytes(b'content')
            initial = release.tree_digest(app)
            binary.chmod(0o755)
            self.assertNotEqual(initial, release.tree_digest(app))
            changed = release.tree_digest(app)
            binary.write_bytes(b'different')
            self.assertNotEqual(changed, release.tree_digest(app))
            (app / 'link').symlink_to('binary')
            release.tree_digest(app)
            (app / 'escape').symlink_to(Path(temp))
            with self.assertRaisesRegex(ValueError, 'escapes'):
                release.tree_digest(app)

    def test_development_paths_and_mismatched_source_reject(self):
        with tempfile.TemporaryDirectory() as temp:
            app = Path(temp); (app / 'Contents').mkdir()
            data = {'CFBundleIdentifier':release.BUNDLE_ID, 'CFBundleShortVersionString':'1.2.3',
                    'CFBundleVersion':'1.2.3','LSMinimumSystemVersion':release.MIN_MACOS,'AXEngineSourceCommit':'a'*40}
            def write():
                (app/'Contents/Info.plist').write_bytes(plistlib.dumps(data))
            write(); release.identity(app,'1.2.3','a'*40)
            with self.assertRaisesRegex(ValueError,'commit'): release.identity(app,'1.2.3','b'*40)
            data['AXEngineDevelopmentRuntime']='/builder/path';write()
            with self.assertRaisesRegex(ValueError,'Development'): release.identity(app,'1.2.3')

    def test_native_runtime_floor_cannot_exceed_app_requirement(self):
        for floor, accepted in [('26.0', True), ('26.2', True), ('26.3', False), ('27.0', False)]:
            with self.subTest(floor=floor), patch.object(release, 'run', side_effect=[
                    'arm64', f'  platform MACOS\n  minos {floor}\n']):
                if accepted:
                    release.validate_native_platform(Path('binary'))
                else:
                    with self.assertRaisesRegex(ValueError, 'minimum macOS'):
                        release.validate_native_platform(Path('binary'))

    def test_unknown_native_code_cannot_escape_individual_signing(self):
        with tempfile.TemporaryDirectory() as temp:
            app = Path(temp)
            paths = [app/'Contents/MacOS'/name for name in ('AXEngineApp','AXEngineModelScanner')]
            paths += [app/release.RUNTIME/'Contents/MacOS'/name for name in
                      ('ax-engine','ax-engine-server','ax-engine-bench','libmlx.dylib','libjaccl.dylib')]
            for file in paths:
                file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(b'\xcf\xfa\xed\xfe')
            targets = release.signing_targets(app)
            self.assertEqual(targets[-2:],[app/release.RUNTIME,app])
            extra=app/'Contents/Resources/unknown';extra.parent.mkdir(parents=True);extra.write_bytes(b'\xcf\xfa\xed\xfe')
            with self.assertRaisesRegex(ValueError,'unexpected'):release.signing_targets(app)

    def test_updater_code_is_explicitly_signed_before_its_containers(self):
        with tempfile.TemporaryDirectory() as temp:
            app = Path(temp)
            paths = [app/'Contents/MacOS'/name for name in ('AXEngineApp','AXEngineModelScanner')]
            paths += [app/release.RUNTIME/'Contents/MacOS'/name for name in
                      ('ax-engine','ax-engine-server','ax-engine-bench','libmlx.dylib','libjaccl.dylib')]
            framework = app/release._UPDATER.FRAMEWORK
            paths += [framework/name for name in release._UPDATER.SPARKLE_BINARIES]
            for file in paths:
                file.parent.mkdir(parents=True, exist_ok=True)
                file.write_bytes(b'\xcf\xfa\xed\xfe')
            targets = release.signing_targets(app)
            for binary in release._UPDATER.SPARKLE_BINARIES:
                for bundle in release._UPDATER.SPARKLE_BUNDLES:
                    if (framework/binary).is_relative_to(framework/bundle):
                        self.assertLess(targets.index(framework/binary),targets.index(framework/bundle))
            self.assertLess(targets.index(framework), targets.index(app))
            (framework/'Versions/B/Unexpected').write_bytes(b'\xcf\xfa\xed\xfe')
            with self.assertRaisesRegex(ValueError,'unexpected'):
                release.signing_targets(app)

    def test_updater_trust_configuration_and_framework_are_required_together(self):
        with tempfile.TemporaryDirectory() as temp:
            app = Path(temp); (app/'Contents').mkdir()
            config = json.loads((release.ROOT/'packaging/macos/sparkle.json').read_text())
            data = {'CFBundleIdentifier':release.BUNDLE_ID, 'CFBundleShortVersionString':'1.2.3',
                    'CFBundleVersion':'1.2.3','LSMinimumSystemVersion':release.MIN_MACOS,
                    'AXEngineDistributionBuild':True,'SUFeedURL':config['feedURL'],
                    'SUPublicEDKey':config['publicKey'],'SUAllowsAutomaticUpdates':False,
                    'SUEnableAutomaticChecks':False,'SUEnableSystemProfiling':False}
            def write(): (app/'Contents/Info.plist').write_bytes(plistlib.dumps(data))
            write()
            with self.assertRaisesRegex(ValueError,'Missing updater'):
                release.identity(app,'1.2.3')
            (app/release._UPDATER.FRAMEWORK).mkdir(parents=True)
            release.identity(app,'1.2.3')
            data['SUPublicEDKey']='substituted';write()
            with self.assertRaisesRegex(ValueError,'updater configuration'):
                release.identity(app,'1.2.3')

    def test_dirty_source_blocks_build_before_mutation(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release,'run',return_value=' M source'):
            output=Path(temp)/'output'
            with self.assertRaisesRegex(ValueError,'clean checkout'):release.build(output,Path(temp))
            self.assertFalse(output.exists())

    def test_changed_app_cannot_be_signed_from_old_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            source=Path(temp)/'source';source.mkdir()
            (source/'build.json').write_text(json.dumps({'schema':'ax-engine.gui-build.v1','version':'1.2.3',
                'commit':'a'*40,'appTreeSha256':'b'*64}))
            with patch.object(release,'identity'),patch.object(release,'tree_digest',return_value='c'*64):
                with self.assertRaisesRegex(ValueError,'changed since build'):
                    release.package(source,Path(temp)/'out','identity','profile')

    def test_notary_rejection_retains_status_record(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release,'run',return_value='{"status":"Invalid","id":"request"}'):
            record=Path(temp)/'record.json'
            with self.assertRaisesRegex(ValueError,'not accepted'):
                release.notarize(Path('app.zip'),'profile',record)
            self.assertEqual(json.loads(record.read_text())['status'],'Invalid')


    def test_notary_timeout_keeps_submission_id_for_recovery(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(release, 'run', side_effect=[
                '{"id":"pending-request"}', subprocess.CalledProcessError(1, ['notarytool','wait'])]):
            record=Path(temp)/'request.json'
            with self.assertRaises(subprocess.CalledProcessError):
                release.notarize(Path('app.zip'),'profile',record)
            self.assertEqual(json.loads(record.read_text())['id'],'pending-request')


if __name__ == '__main__': unittest.main()
