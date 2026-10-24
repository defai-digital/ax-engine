import {test, after} from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {prepare, verifyDirectory, releaseObjects, publishObjects, verifyLatest, hash, validIdentity, compareVersion, cask} from './distribution.mjs';
import worker, {objectKey, byteRange} from './download-worker/worker.mjs';
import {validateConfig, routeOwner} from './gateway.mjs';
const root = fs.mkdtempSync(path.join(os.tmpdir(), 'ax-gui-distribution-test-'));
after(() => fs.rmSync(root, {recursive:true, force:true}));
execFileSync('minisign', ['-G', '-W', '-p', path.join(root,'test.pub'), '-s', path.join(root,'test.sec')], {stdio:'pipe'});
function sign(file) { execFileSync('minisign', ['-S', '-s', path.join(root,'test.sec'), '-m',file], {stdio:'pipe'}); }
function verify(file) { execFileSync('minisign', ['-Vm',file, '-p',path.join(root,'test.pub')], {stdio:'pipe'}); }
function fixture(version='1.2.3', commit='a'.repeat(40)) {
  const dir=fs.mkdtempSync(path.join(root,'fixture-')), source=path.join(dir,'source'), output=path.join(dir,'release');
  fs.mkdirSync(source);
  const filename=`AX-Engine-${version}-mac-arm64.dmg`, content=Buffer.from(`Disk image fixture ${version}`);
  fs.writeFileSync(path.join(source,filename),content);
  fs.writeFileSync(path.join(source,'notarization.json'),JSON.stringify({schema:'ax-engine.gui-notarization.v1',
    version,commit,filename,size:content.length,sha256:hash(content),team:'N5ZUZDUJS6',bundleId:'com.automatosx.ax-engine',
    appRequest:'test-app-request',dmgRequest:'test-dmg-request',appTreeSha256:'b'.repeat(64)}));
  const inventory=prepare(source, output, {sign, verify, verifyApple:()=>{},sourceAuthority:()=>{}});
  return {dir,source,output,inventory,objects:releaseObjects(output,inventory)};
}
function memoryStore(initial=[]) {
  const data=new Map(initial), writes=[];
  return {data,writes,async read(key){return data.get(key)??null;},async write(key,body){writes.push(key);data.set(key,body);}};
}
test('signed inventory binds image, source, manifest and atomic latest',()=>{
  const f=fixture();
  assert.equal(verifyDirectory(f.output,verify).digest,f.inventory.digest);
  const latest=verifyLatest(fs.readFileSync(path.join(f.output,'latest.json')),verify);
  assert.equal(latest.inventory,f.inventory.digest);
  assert.equal(latest.commit,'a'.repeat(40));
  assert.equal(f.objects.at(-1).key,'latest.json');
  assert.ok(f.objects.findIndex(x=>x.key.endsWith('release-ready.txt'))>f.objects.findIndex(x=>x.key.endsWith('inventory.json')));
});
test('actual minisign rejects payload or signature corruption',()=>{
  const f=fixture(), file=path.join(f.output,'release.tsv');
  fs.appendFileSync(file,'injected');
  assert.throws(()=>verifyDirectory(f.output,verify));
  const envelope=JSON.parse(fs.readFileSync(path.join(f.output,'latest.json')));
  envelope.payload=Buffer.from('v9.9.9\n').toString('base64');
  assert.throws(()=>verifyLatest(Buffer.from(JSON.stringify(envelope)),verify));
});
test('signed malformed inventory cannot traverse paths or omit artifacts',()=>{
  for(const mutate of [x=>x.assets[0].name='../escape',x=>x.assets.pop(),x=>x.assets[1]=x.assets[0]]){
    const f=fixture(), file=path.join(f.output,'inventory.json');
    const data=JSON.parse(fs.readFileSync(file));mutate(data);fs.writeFileSync(file,JSON.stringify(data));
    fs.unlinkSync(file+'.minisig');sign(file);
    assert.throws(()=>verifyDirectory(f.output,verify));
  }
});
test('source/version/signing-team mismatches reject preparation',()=>{
  const f=fixture(), file=path.join(f.source,'notarization.json');
  const data=JSON.parse(fs.readFileSync(file));data.team='OTHERTEAM';fs.writeFileSync(file,JSON.stringify(data));
  assert.throws(()=>prepare(f.source,path.join(f.dir,'bad'),{sign,verify,verifyApple:()=>{},sourceAuthority:()=>{}}),/qualification/);
  for(const value of ['../1','01.2.3','1.2.3-beta','1000000.0.0'])assert.throws(()=>validIdentity(value,'a'.repeat(40)));
  assert.equal(compareVersion('1.10.0','1.9.0'),1);
});
test('symlink assets reject verification',()=>{
  const f=fixture(), file=path.join(f.output,'install');
  fs.renameSync(file,file+'.copy');fs.symlinkSync(file+'.copy',file);
  assert.throws(()=>verifyDirectory(f.output,verify),/regular file/);
});
test('publisher completes readback before advancing latest and resumes idempotently',async()=>{
  const f=fixture(), store=memoryStore();
  await publishObjects(f.objects,f.inventory,store,key=>store.read(key),verify);
  assert.equal(store.writes.at(-1),'latest.json');
  const before=store.writes.length;
  await publishObjects(f.objects,f.inventory,store,key=>store.read(key),verify);
  assert.equal(store.writes.length,before+1); // Latest is one idempotent atomic PUT.
});
test('immutable collisions stop before any mutation',async()=>{
  const f=fixture(), store=memoryStore([[f.objects[3].key,Buffer.from('collision')]]);
  await assert.rejects(publishObjects(f.objects,f.inventory,store,key=>store.read(key),verify),/collision/);
  assert.deepEqual(store.writes,[]);
});
test('upload or anonymous readback failure never advances latest',async()=>{
  const f=fixture();
  for(const mode of ['upload','readback']) {
    const store=memoryStore();
    if(mode==='upload')store.write=async()=>{throw Error('offline');};
    await assert.rejects(publishObjects(f.objects,f.inventory,store,async()=>null,verify));
    assert.equal(await store.read('latest.json'),null);
  }
});
test('failed latest cutover restores previous signed pointer',async()=>{
  const old=fixture('1.2.2'), next=fixture(), store=memoryStore([['latest.json',old.objects.at(-1).body]]);
  await assert.rejects(publishObjects(next.objects,next.inventory,store,key=>key==='latest.json'?null:store.read(key),verify),/Latest readback/);
  assert.deepEqual(await store.read('latest.json'),old.objects.at(-1).body);
});
test('downgrade and same-version replacement are forbidden',async()=>{
  const newer=fixture('1.2.4'), same=fixture('1.2.3','b'.repeat(40)), next=fixture();
  for(const before of [newer,same]) {
    const store=memoryStore([['latest.json',before.objects.at(-1).body]]);
    await assert.rejects(publishObjects(next.objects,next.inventory,store,key=>store.read(key),verify));
    assert.deepEqual(store.writes,[]);
  }
});
test('concurrent pointer change is preserved',async()=>{
  const f=fixture(), newer=fixture('1.2.4'), store=memoryStore();
  const read=async key=>{
    if(key==='install')store.data.set('latest.json',newer.objects.at(-1).body);
    return store.read(key);
  };
  await assert.rejects(publishObjects(f.objects,f.inventory,store,read,verify),/Latest changed/);
  assert.deepEqual(await store.read('latest.json'),newer.objects.at(-1).body);
});
test('gateway configuration cannot take over other routes or enable preview domains',()=>{
  const config=JSON.parse(fs.readFileSync(new URL('./download-worker/wrangler.jsonc',import.meta.url)));
  assert.equal(validateConfig(config).name,'ax-engine-desktop-downloads');
  for(const change of [{workers_dev:true},{preview_urls:true},{vars:{PUBLIC_RELEASE_ENABLED:'true'}},{unexpected:'field'}])
    assert.throws(()=>validateConfig({...config,...change}));
  assert.throws(()=>routeOwner([{pattern:config.routes[0].pattern,script:'ax-code-downloads'}]),/takeover/);
});
test('gateway key and range restrictions',()=>{
  assert.equal(objectKey(new URL('https://download.automatosx.com/ax-engine-desktop/latest.json')),'latest.json');
  for(const url of ['https://evil.example/ax-engine-desktop/install','https://download.automatosx.com/ax-code/install',
    'https://download.automatosx.com/ax-engine-desktop/releases/download/v1.2.3/%2e%2e/secret'])assert.equal(objectKey(new URL(url)),null);
  assert.deepEqual(byteRange('bytes=2-5',10),{offset:2,length:4});
  assert.deepEqual(byteRange('bytes=-3',10),{offset:7,length:3});
  assert.equal(byteRange('bytes=10-',10),null);
  assert.equal(byteRange('bytes=0-1,3-4',10),null);
  assert.equal(byteRange('bytes=9007199254740992-',10),null);
});
function bucket(race=false) {
  const data=Buffer.from('0123456789');
  return {async head(){return {size:data.length,etag:'hash',httpEtag:'"hash"',writeHttpMetadata(){}};},
    async get(key,options){assert.equal(options.onlyIf.etagMatches,'hash');return race?null:{body:options.range?data.subarray(options.range.offset,options.range.offset+options.range.length):data};}};
}
test('gateway handles HEAD, ranges, conditional reads and concurrent replacement',async()=>{
  const url='https://download.automatosx.com/ax-engine-desktop/releases/download/v1.2.3/model.dmg';
  const env={PUBLIC_RELEASE_ENABLED:'true',ENGINE_GUI_RELEASES:bucket()};
  let response=await worker.fetch(new Request(url,{headers:{Range:'bytes=2-5'}}),env);
  assert.equal(response.status,206);assert.equal(await response.text(),'2345');
  assert.equal(response.headers.get('content-range'),'bytes 2-5/10');
  response=await worker.fetch(new Request(url,{method:'HEAD'}),env);
  assert.equal(response.headers.get('content-length'),'10');assert.equal(await response.text(),'');
  response=await worker.fetch(new Request(url,{headers:{'If-None-Match':'"hash"'}}),env);assert.equal(response.status,304);
  response=await worker.fetch(new Request(url),{...env,ENGINE_GUI_RELEASES:bucket(true)});assert.equal(response.status,503);
  response=await worker.fetch(new Request(url),{...env,PUBLIC_RELEASE_ENABLED:'false'});assert.equal(response.status,503);
  response=await worker.fetch(new Request(url,{method:'PUT'}),env);assert.equal(response.status,405);
  response=await worker.fetch(new Request(url,{headers:{Range:'bytes=99-'}}),env);assert.equal(response.status,416);
});

test('Homebrew loads the generated Cask and enforces the runtime minor version', {skip: process.platform !== 'darwin'}, () => {
  const file=path.join(root,'ax-engine-gui.rb');
  fs.writeFileSync(file,cask('1.2.3','a'.repeat(64)));
  const script=`require "cask/cask_loader"
MacOS.define_singleton_method(:full_version) { Version.new(ARGV[1]) }
c = Cask::CaskLoader::FromContentLoader.new(File.read(ARGV[0])).load(config: nil)
c.artifacts.find { |a| a.is_a?(Cask::Artifact::PreflightBlock) }.install_phase
puts "Cask preflight passed"`;
  for (const version of ['26.0','26.1','26.2','27.0']) {
    const invoke=()=>execFileSync('brew',['ruby','-e',script,file,version],
      {env:{...process.env,HOMEBREW_NO_AUTO_UPDATE:'1'},stdio:['ignore','pipe','pipe']});
    if (['26.0','26.1'].includes(version)) assert.throws(invoke, error=>error.stderr.toString().includes('requires macOS 26.2'));
    else assert.match(invoke().toString(),/Cask preflight passed/);
  }
});
