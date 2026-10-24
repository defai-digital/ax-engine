"""Execute the real bootstrap control flow with isolated macOS/network fixtures.

The test copy redirects system paths to deterministic stubs. Production code has
no test-mode override. Cryptographic signing is separately tested with Minisign;
these tests prove installation ordering, rejection and rollback behavior.
"""
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
INSTALL=ROOT/'packaging/macos/distribution/install'
COMMIT='a'*40
STUB=r'''#!/usr/bin/env python3
import os,sys,pathlib,json,shutil,hashlib,base64
p=pathlib.Path
name=p(sys.argv[0]).name;a=sys.argv[1:];root=p(os.environ['FIXTURE']);mode=os.environ.get('FAIL_MODE','')
def done(value='',status=0):
 if value: print(value)
 sys.exit(status)
if name=='uname':done('Darwin' if a==['-s'] else 'arm64')
if name=='sw_vers':done(os.environ.get('TEST_MACOS','26.2'))
if name=='curl':
 url=a[0] if a[0].startswith('https:') else next(x for x in a if x.startswith('https:'))
 if mode=='offline':done(status=22)
 target=p(a[a.index('-o')+1]);src=root/'remote'/url.rsplit('/',1)[-1]
 if not src.exists():done(status=22)
 shutil.copyfile(src,target);done()
if name=='shasum':
 file=p(a[-1]);digest=hashlib.sha256(file.read_bytes()).hexdigest()
 if file.name=='verifier.zip':digest='89000b19535765f9cffc65a65d64a820f433ef6db8020667f7570e06bf6aac63'
 done(digest+'  '+str(file))
if name=='ditto':
 if '-x' in a:
  out=p(a[-1]);out.mkdir();shutil.copyfile(root/'bin/minisign',out/'minisign');done()
 src,dst=map(p,a)
 if mode=='copy' and '.stage-' in str(dst):done(status=1)
 shutil.copytree(src,dst,dirs_exist_ok=True,symlinks=True);done()
if name=='minisign':
 file=p(a[a.index('-Vm')+1]);sig=p(a[a.index('-x')+1]).read_text()
 done(status=0 if sig==hashlib.sha256(file.read_bytes()).hexdigest() else 1)
if name=='plutil':done(json.loads(p(a[-1]).read_text())[a[1]])
if name=='base64':
 sys.stdout.buffer.write(base64.b64decode(sys.stdin.buffer.read().strip(),validate=True));done()
if name=='hdiutil':
 if a[0]=='attach':
  mount=p(a[a.index('-mountpoint')+1]);shutil.copytree(root/'app',mount/'AX Engine.app');done()
 done()
if name=='codesign':
 if '-dv' in a:done('TeamIdentifier=' + ('WRONG' if mode=='team' else 'N5ZUZDUJS6'))
 if mode=='verify' and a[-1]==str(root/'Applications/AX Engine.app'):done(status=1)
 done()
if name=='spctl':done(status=1 if mode=='gatekeeper' or (mode=='post-gatekeeper' and a[-1]==str(root/'Applications/AX Engine.app')) else 0)
if name=='xcode-select':done(status=1)
if name=='pgrep':done(status=1)
if name=='PlistBuddy':
 field=a[1].removeprefix('Print :');data=json.loads(p(a[-1]).read_text())
 if mode=='identity' and field=='AXEngineSourceCommit' and p(a[-1])==root/'Applications/AX Engine.app/Contents/Info.plist':done('wrong')
 done(data.get(field,''),0 if field in data else 1)
if name=='mv':
 src,dst=map(p,a)
 if mode=='swap' and '.stage-' in str(src):done(status=1)
 src.rename(dst);done()
raise Exception(name)
'''


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.apps=self.root/'Applications';self.apps.mkdir()
        self.remote=self.root/'remote';self.remote.mkdir();(self.root/'bin').mkdir()
        for name in ['uname','sw_vers','curl','shasum','ditto','minisign','plutil','base64','hdiutil','codesign','spctl','xcode-select','pgrep','PlistBuddy','mv']:
            file=self.root/'bin'/name;file.write_text(STUB.replace('#!/usr/bin/env python3','#!'+shutil.which('python3')));file.chmod(0o755)
        self.app=self.root/'app';self.create_app(self.app,'1.2.3',COMMIT)
        self.filename='AX-Engine-1.2.3-mac-arm64.dmg';image=b'dmg fixture'
        manifest=f'release\tv1.2.3\t{COMMIT}\ndarwin-arm64\t{self.filename}\t{len(image)}\t{hashlib.sha256(image).hexdigest()}\n'.encode()
        inventory=b'{"fixture":"inventory"}'
        ready=f'v1.2.3\n{COMMIT}\n{hashlib.sha256(manifest).hexdigest()}\n{hashlib.sha256(inventory).hexdigest()}\n'.encode()
        for name,data in [(self.filename,image),('release.tsv',manifest),('inventory.json',inventory),('release-ready.txt',ready)]:self.signed(name,data)
        latest={'payload':base64.b64encode(ready).decode(),'signature':base64.b64encode(hashlib.sha256(ready).hexdigest().encode()).decode()}
        (self.remote/'latest.json').write_text(json.dumps(latest));(self.remote/'minisign-0.12-macos.zip').write_bytes(b'verifier fixture')
        text=INSTALL.read_text().replace('PATH=/usr/bin:/bin:/usr/sbin:/sbin','PATH="'+str(self.root/'bin')+':/usr/bin:/bin:/usr/sbin:/sbin"')
        text=text.replace('/usr/libexec/PlistBuddy',str(self.root/'bin/PlistBuddy'))
        self.script=self.root/'install';self.script.write_text(text)
        self.env={**os.environ,'FIXTURE':str(self.root),'TMPDIR':str(self.root)}

    def signed(self,name,data):
        (self.remote/name).write_bytes(data);(self.remote/(name+'.minisig')).write_text(hashlib.sha256(data).hexdigest())

    def create_app(self,path,version,commit='b'*40):
        (path/'Contents').mkdir(parents=True)
        (path/'Contents/Info.plist').write_text(json.dumps({'CFBundleIdentifier':'com.automatosx.ax-engine',
            'CFBundleShortVersionString':version,'AXEngineSourceCommit':commit}))
        (path/'original').write_text(version)

    def invoke(self,mode='',*args):
        return subprocess.run(['bash',str(self.script),'--applications-dir',str(self.apps),*args],
                              env={**self.env,'FAIL_MODE':mode},capture_output=True,text=True,timeout=15)

    def test_fresh_install_and_upgrade_preserve_user_data(self):
        data=self.root/'user-model.safetensors';data.write_bytes(b'untouched')
        result=self.invoke();self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((self.apps/'AX Engine.app/original').read_text(),'1.2.3')
        self.assertEqual(data.read_bytes(),b'untouched')
        result=self.invoke();self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse((self.apps/'.AX-Engine.previous.app').exists())

    def test_upgrade_copy_swap_and_post_verification_failures_restore_previous(self):
        for mode in ['copy','swap','verify','identity','post-gatekeeper']:
            with self.subTest(mode=mode):
                target=self.apps/'AX Engine.app'
                if target.exists():shutil.rmtree(target)
                self.create_app(target,'1.2.2')
                result=self.invoke(mode);self.assertNotEqual(result.returncode,0,result.stdout)
                self.assertEqual((target/'original').read_text(),'1.2.2',result.stderr)
                self.assertFalse((self.apps/'.AX-Engine.install.lock').exists())

    def test_network_signature_team_and_gatekeeper_fail_before_replacement(self):
        self.create_app(self.apps/'AX Engine.app','1.2.2')
        for mode in ['offline','team','gatekeeper']:
            result=self.invoke(mode);self.assertNotEqual(result.returncode,0)
            self.assertEqual((self.apps/'AX Engine.app/original').read_text(),'1.2.2')
        (self.remote/(self.filename+'.minisig')).write_text('invalid')
        result=self.invoke();self.assertNotEqual(result.returncode,0)
        self.assertIn('Invalid release signature',result.stderr)
        self.assertEqual((self.apps/'AX Engine.app/original').read_text(),'1.2.2')

    def test_interrupted_swap_and_stale_lock_recover(self):
        self.create_app(self.apps/'.AX-Engine.previous.app','1.2.2')
        lock=self.apps/'.AX-Engine.install.lock';lock.mkdir();(lock/'pid').write_text('99999999')
        result=self.invoke();self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual((self.apps/'AX Engine.app/original').read_text(),'1.2.3')

    def test_live_installer_lock_is_not_stolen(self):
        lock=self.apps/'.AX-Engine.install.lock';lock.mkdir();(lock/'pid').write_text(str(os.getpid()))
        result=self.invoke();self.assertNotEqual(result.returncode,0);self.assertTrue(lock.exists())
        self.assertIn('Another installation',result.stderr)

    def test_downgrade_is_rejected(self):
        self.create_app(self.apps/'AX Engine.app','1.10.0')
        result=self.invoke();self.assertNotEqual(result.returncode,0);self.assertIn('downgrade',result.stderr)
        self.assertEqual((self.apps/'AX Engine.app/original').read_text(),'1.10.0')

    def test_older_macos_is_rejected_before_network_or_install(self):
        for version in ['25.9', '26.0', '26.1']:
            with self.subTest(version=version):
                self.env['TEST_MACOS']=version
                result=self.invoke('offline');self.assertNotEqual(result.returncode,0)
                self.assertIn('macOS 26.2',result.stderr)
                self.assertFalse((self.apps/'AX Engine.app').exists())

    def test_verify_only_does_not_install(self):
        result=self.invoke('','--verify-only');self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse((self.apps/'AX Engine.app').exists())

    def test_symlink_destination_is_not_followed(self):
        external=self.root/'external';self.create_app(external,'1.2.2')
        (self.apps/'AX Engine.app').symlink_to(external)
        result=self.invoke();self.assertNotEqual(result.returncode,0)
        self.assertEqual((external/'original').read_text(),'1.2.2')

    def test_signed_manifest_size_mismatch_is_rejected(self):
        manifest=(self.remote/'release.tsv').read_bytes().replace(b'\t11\t',b'\t999\t')
        # Any mismatch with signed latest must stop before mounting or mutation.
        self.signed('release.tsv',manifest+b'extra\n')
        result=self.invoke();self.assertNotEqual(result.returncode,0)
        self.assertFalse((self.apps/'AX Engine.app').exists())


if __name__=='__main__':unittest.main()
