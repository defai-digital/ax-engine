#!/usr/bin/env node
// Release contracts follow AX Code Desktop's signed-inventory workflow.
// AX Engine has a separate channel and an atomic signed latest envelope.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
export const DIRECTORY = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(DIRECTORY, '../../..');
export const BASE = 'https://download.automatosx.com/ax-engine-desktop';
export const KEY = 'RWSlDu++afxCz01OqhYWhfo8+L8pVbSYXJBEb2zoWBuK0WACIzbGVZRO';
export const ACCOUNT = '89657c08227b638ab67905e1c5899406';
export const BUCKET = 'ax-engine-desktop-releases';
export const check = (value, message) => { if (!value) throw new Error(message); };
export const hash = value => crypto.createHash('sha256').update(value).digest('hex');
export function bytes(file) {
  const stat = fs.lstatSync(file);
  check(stat.isFile() && !stat.isSymbolicLink(), `Expected a regular file: ${path.basename(file)}`);
  check(stat.size <= 300 * 1024 * 1024, 'REST upload limit exceeded; use a separately qualified multipart publisher');
  return fs.readFileSync(file);
}
export function signature(file) {
  bytes(file); bytes(`${file}.minisig`);
  execFileSync('minisign', ['-Vm', file, '-x', `${file}.minisig`, '-P', KEY], {stdio: ['ignore', 'pipe', 'pipe']});
}
function sign(file) {
  const env = {...process.env};
  if (!env.AX_MINISIGN_PASSWORD) {
    env.AX_MINISIGN_PASSWORD = execFileSync('security', ['find-generic-password', '-a', 'ax-release',
      '-s', 'ax-minisign', '-w'], {encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']}).trimEnd();
  }
  execFileSync('bash', [path.join(ROOT, 'scripts/minisign-artifact.sh'), '--pinned-public-key', KEY, file],
    {env, stdio: ['ignore', 'pipe', 'pipe']});
}
const versionPattern = /^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/;
export function validIdentity(version, commit) {
  check(versionPattern.test(version) && version.split('.').every(x => Number(x) <= 999999), 'Invalid stable version');
  check(/^[a-f0-9]{40}$/.test(commit), 'Invalid source commit');
}
const dmgName = version => `AX-Engine-${version}-mac-arm64.dmg`;
const required = version => [dmgName(version), 'notarization.json', 'release.tsv', 'install', 'ax-engine-gui.rb'];
export function manifest(inventory) {
  const asset = inventory.assets.find(x => x.name === dmgName(inventory.version));
  check(asset, 'Missing installer');
  return `release\tv${inventory.version}\t${inventory.commit}\ndarwin-arm64\t${asset.name}\t${asset.size}\t${asset.sha256}\n`;
}
export function ready(inventory, digest) {
  return `v${inventory.version}\n${inventory.commit}\n${hash(manifest(inventory))}\n${digest}\n`;
}
export function cask(version, digest) {
  return `cask "ax-engine-gui" do\n  version "${version}"\n  sha256 "${digest}"\n\n  url "${BASE}/releases/download/v#{version}/AX-Engine-#{version}-mac-arm64.dmg"\n  name "AX Engine"\n  desc "Native Apple Silicon inference dashboard and model manager"\n  homepage "https://github.com/defai-digital/ax-engine"\n\n  depends_on arch: :arm64\n  depends_on macos: :tahoe\n  preflight do\n    odie "AX Engine requires macOS 26.2 or newer." if MacOS.full_version < Version.new("26.2")\n  end\n  app "AX Engine.app"\nend\n`;
}
export function sourceAuthority(commit) {
  check(execFileSync('git', ['rev-parse', 'HEAD'], {cwd: ROOT, encoding: 'utf8'}).trim() === commit,
    'Prepare download assets from the same source commit as the app');
  check(!execFileSync('git', ['status', '--porcelain'], {cwd: ROOT, encoding: 'utf8'}).trim(),
    'Download preparation requires a clean source checkout');
}
export function prepare(source, output, options = {}) {
  const signer = options.sign ?? sign;
  const verifyApple = options.verifyApple ?? (dir => execFileSync('python3',
    [path.join(ROOT, 'scripts/release-macos-app.py'), 'verify', '--source', dir], {stdio: 'inherit'}));
  check(!fs.existsSync(output), 'Use a new output directory');
  verifyApple(source);
  const receipt = JSON.parse(bytes(path.join(source, 'notarization.json')));
  validIdentity(receipt.version, receipt.commit);
  (options.sourceAuthority ?? sourceAuthority)(receipt.commit);
  check(receipt.schema === 'ax-engine.gui-notarization.v1' && receipt.team === 'N5ZUZDUJS6' &&
    receipt.bundleId === 'com.automatosx.ax-engine', 'Invalid Apple qualification receipt');
  check(receipt.filename === dmgName(receipt.version), 'Unexpected DMG filename');
  const image = bytes(path.join(source, receipt.filename));
  check(hash(image) === receipt.sha256 && image.length === receipt.size, 'DMG does not match notarization receipt');
  fs.mkdirSync(output, {recursive: true});
  fs.writeFileSync(path.join(output, receipt.filename), image);
  fs.copyFileSync(path.join(source, 'notarization.json'), path.join(output, 'notarization.json'));
  fs.copyFileSync(path.join(DIRECTORY, 'install'), path.join(output, 'install'));
  fs.writeFileSync(path.join(output, 'ax-engine-gui.rb'), cask(receipt.version, receipt.sha256));
  const inventory = {schema: 'ax-engine.gui-inventory.v1', version: receipt.version, commit: receipt.commit,
    assets: [{name: receipt.filename, size: image.length, sha256: hash(image)}]};
  fs.writeFileSync(path.join(output, 'release.tsv'), manifest(inventory));
  inventory.assets = required(receipt.version).sort().map(name => {
    const content = bytes(path.join(output, name));
    return {name, size: content.length, sha256: hash(content)};
  });
  fs.writeFileSync(path.join(output, 'inventory.json'), JSON.stringify(inventory, null, 2) + '\n');
  fs.writeFileSync(path.join(output, 'release-ready.txt'), ready(inventory, hash(bytes(path.join(output, 'inventory.json')))));
  for (const name of [...required(receipt.version), 'inventory.json', 'release-ready.txt']) signer(path.join(output, name));
  fs.writeFileSync(path.join(output, 'latest.json'), JSON.stringify({
    payload: bytes(path.join(output, 'release-ready.txt')).toString('base64'),
    signature: bytes(path.join(output, 'release-ready.txt.minisig')).toString('base64'),
  }) + '\n');
  return verifyDirectory(output, options.verify ?? signature);
}
export function verifyDirectory(source, verify = signature) {
  const file = path.join(source, 'inventory.json');
  verify(file);
  check(bytes(file).length < 65536, 'Oversized inventory');
  const inventory = JSON.parse(bytes(file));
  check(inventory.schema === 'ax-engine.gui-inventory.v1', 'Invalid inventory schema');
  validIdentity(inventory.version, inventory.commit);
  const names = required(inventory.version);
  check(Array.isArray(inventory.assets) && inventory.assets.length === names.length, 'Incomplete inventory');
  const seen = new Set();
  for (const asset of inventory.assets) {
    check(names.includes(asset.name) && !seen.has(asset.name), 'Unexpected or duplicate asset');
    seen.add(asset.name);
    const data = bytes(path.join(source, asset.name));
    check(Number.isSafeInteger(asset.size) && asset.size > 0 && data.length === asset.size, 'Asset size mismatch');
    check(/^[a-f0-9]{64}$/.test(asset.sha256) && hash(data) === asset.sha256, 'Asset hash mismatch');
    verify(path.join(source, asset.name));
  }
  check(bytes(path.join(source, 'release.tsv')).toString() === manifest(inventory), 'Manifest mismatch');
  const receipt = JSON.parse(bytes(path.join(source, 'notarization.json')));
  const image = inventory.assets.find(x => x.name === dmgName(inventory.version));
  check(receipt.schema === 'ax-engine.gui-notarization.v1' && receipt.version === inventory.version && receipt.commit === inventory.commit &&
    receipt.filename === image.name && receipt.sha256 === image.sha256 && receipt.size === image.size &&
    receipt.team === 'N5ZUZDUJS6' && receipt.bundleId === 'com.automatosx.ax-engine', 'Notarization binding mismatch');
  const digest = hash(bytes(file));
  const expected = ready(inventory, digest);
  check(bytes(path.join(source, 'release-ready.txt')).toString() === expected, 'Readiness mismatch');
  verify(path.join(source, 'release-ready.txt'));
  const latest = JSON.parse(bytes(path.join(source, 'latest.json')));
  check(Object.keys(latest).sort().join(',') === 'payload,signature' &&
    latest.payload === Buffer.from(expected).toString('base64') &&
    latest.signature === bytes(path.join(source, 'release-ready.txt.minisig')).toString('base64'), 'Latest envelope mismatch');
  return {...inventory, digest};
}
export function releaseObjects(source, inventory) {
  const prefix = `releases/download/v${inventory.version}/`;
  const objects = [];
  // A version is usable only after both readiness bytes and their signature exist.
  for (const name of [...inventory.assets.map(x => x.name), 'inventory.json', 'release-ready.txt']) {
    for (const suffix of ['', '.minisig']) objects.push({key: prefix + name + suffix, body: bytes(path.join(source, name + suffix)), immutable: true});
  }
  objects.push({key: 'install', body: bytes(path.join(source, 'install')), immutable: false});
  // One atomic object carries the pointer and its detached signature together.
  objects.push({key: 'latest.json', body: bytes(path.join(source, 'latest.json')), immutable: false});
  return objects;
}
export function compareVersion(a, b) {
  validIdentity(a, '0'.repeat(40)); validIdentity(b, '0'.repeat(40));
  for (let i = 0; i < 3; i++) { const difference = Number(a.split('.')[i]) - Number(b.split('.')[i]); if (difference) return Math.sign(difference); }
  return 0;
}
export function verifyLatest(content, verify = signature) {
  check(content.length < 16384, 'Oversized latest pointer');
  const envelope = JSON.parse(content);
  check(Object.keys(envelope).sort().join(',') === 'payload,signature' &&
    typeof envelope.payload === 'string' && typeof envelope.signature === 'string', 'Invalid latest envelope');
  const payload = Buffer.from(envelope.payload, 'base64');
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ax-gui-pointer-'));
  try {
    const file = path.join(dir, 'latest');
    fs.writeFileSync(file, payload); fs.writeFileSync(file + '.minisig', Buffer.from(envelope.signature, 'base64'));
    verify(file);
  } finally { fs.rmSync(dir, {recursive: true, force: true}); }
  const rows = payload.toString().split('\n');
  check(rows.length === 5 && rows[4] === '' && /^v/.test(rows[0]) &&
    /^[a-f0-9]{64}$/.test(rows[2]) && /^[a-f0-9]{64}$/.test(rows[3]), 'Invalid latest payload');
  validIdentity(rows[0].slice(1), rows[1]);
  return {version: rows[0].slice(1), commit: rows[1], manifest: rows[2], inventory: rows[3]};
}
export function releaseAuthority(inventory, ciRun) {
  const tagCommit = execFileSync('git', ['rev-parse', `v${inventory.version}^{commit}`], {cwd: ROOT, encoding: 'utf8'}).trim();
  check(tagCommit === inventory.commit, 'Public GUI assets require the matching immutable release tag');
  const remote = execFileSync('git', ['ls-remote', '--tags', 'https://github.com/defai-digital/ax-engine.git', `refs/tags/v${inventory.version}`, `refs/tags/v${inventory.version}^{}`],
    {cwd: ROOT, encoding: 'utf8'}).trim().split('\n').map(line => line.split(/\s+/));
  const remoteCommit = remote.find(row => row[1]?.endsWith('^{}'))?.[0] ?? remote[0]?.[0];
  check(remoteCommit === inventory.commit, 'Remote release tag does not match candidate');
  const published = JSON.parse(execFileSync('gh', ['release', 'view', `v${inventory.version}`, '--repo', 'defai-digital/ax-engine',
    '--json', 'isDraft,isPrerelease,tagName'], {encoding: 'utf8'}));
  check(!published.isDraft && !published.isPrerelease && published.tagName === `v${inventory.version}`, 'Publish the qualified stable Engine release first');
  const ci = JSON.parse(execFileSync('gh', ['run', 'view', ciRun, '--repo', 'defai-digital/ax-engine',
    '--json', 'headSha,status,conclusion,workflowName'], {encoding: 'utf8'}));
  check(ci.headSha === inventory.commit && ci.status === 'completed' && ci.conclusion === 'success' && ci.workflowName === 'CI',
    'Exact-source CI must pass before publication');
}
export function cloudflareStore(secret, fetcher = fetch) {
  check(secret && !/[\r\n]/.test(secret), 'Invalid Cloudflare API token');
  return {
    async read(key) {
      const response = await fetcher(`https://api.cloudflare.com/client/v4/accounts/${ACCOUNT}/r2/buckets/${BUCKET}/objects/${encodeURIComponent(key)}`,
        {headers: {Authorization: `Bearer ${secret}`}, signal: AbortSignal.timeout(600000)});
      if (response.status === 404) return null;
      check(response.ok, `R2 read failed: HTTP ${response.status}`);
      return Buffer.from(await response.arrayBuffer());
    },
    async write(key, body) {
      const response = await fetcher(`https://api.cloudflare.com/client/v4/accounts/${ACCOUNT}/r2/buckets/${BUCKET}/objects/${encodeURIComponent(key)}`,
        {method: 'PUT', body, headers: {Authorization: `Bearer ${secret}`, 'Content-Type': 'application/octet-stream'},
          signal: AbortSignal.timeout(600000)});
      check(response.ok, `R2 write failed: HTTP ${response.status}`);
      const reply = await response.json();
      check(reply.success === true, 'R2 rejected upload');
    },
  };
}
export async function publishObjects(objects, inventory, store, publicRead, verify = signature) {
  const latest = objects.at(-1);
  check(latest.key === 'latest.json', 'Latest must be published last');
  const old = await store.read('latest.json');
  if (old) {
    const previous = verifyLatest(old, verify);
    const comparison = compareVersion(inventory.version, previous.version);
    check(comparison >= 0, 'Refusing a release-channel downgrade');
    check(comparison !== 0 || (previous.commit === inventory.commit && previous.inventory === inventory.digest),
      'Published version is immutable; use a new version');
  }
  // Preflight every collision before changing any remote bytes.
  for (const object of objects.filter(x => x.immutable)) {
    const existing = await store.read(object.key);
    check(!existing || existing.equals(object.body), `Immutable object collision: ${object.key}`);
  }
  for (const object of objects.slice(0, -1)) {
    const existing = await store.read(object.key);
    if (!existing?.equals(object.body)) await store.write(object.key, object.body);
    check((await store.read(object.key))?.equals(object.body), `R2 readback mismatch: ${object.key}`);
    check((await publicRead(object.key))?.equals(object.body), `Anonymous readback mismatch: ${object.key}`);
  }
  const before = await store.read('latest.json');
  check(before === null ? old === null : old !== null && before.equals(old), 'Latest changed during publication; retry under the single-publisher lock');
  await store.write(latest.key, latest.body);
  try {
    check((await store.read(latest.key))?.equals(latest.body) && (await publicRead(latest.key))?.equals(latest.body), 'Latest readback mismatch');
  } catch (error) {
    // Restore only our pointer; never overwrite another publisher's newer state.
    if (old && (await store.read(latest.key))?.equals(latest.body)) await store.write(latest.key, old);
    throw error;
  }
}
async function publicRead(key) {
  // Exercise the same curl client as the bootstrap; gateway policy may vary by client.
  try { return execFileSync('curl', ['--proto', '=https', '--proto-redir', '=https', '--fail', '--silent', '--show-error',
    '--location', '--connect-timeout', '20', '--max-time', '600', '-H', 'Cache-Control: no-cache', `${BASE}/${key}`],
    {maxBuffer: 301 * 1024 * 1024, stdio: ['ignore', 'pipe', 'pipe']}); }
  catch { throw new Error(`Anonymous download failed: ${key}`); }
}
async function main() {
  const [command, source, outputOrRun] = process.argv.slice(2);
  check(source && ['prepare', 'verify', 'publish', 'verify-public'].includes(command),
    'usage: distribution.mjs prepare <notarized-dir> <output-dir> | verify <dir> | publish <dir> <ci-run-id> | verify-public <dir>');
  if (command === 'prepare') { check(outputOrRun, 'Missing output directory'); prepare(path.resolve(source), path.resolve(outputOrRun)); }
  else {
    const inventory = verifyDirectory(path.resolve(source));
    const objects = releaseObjects(path.resolve(source), inventory);
    if (command === 'publish') {
      check(outputOrRun && /^\d+$/.test(outputOrRun), 'Missing CI run ID');
      releaseAuthority(inventory, outputOrRun);
      execFileSync('python3', [path.join(ROOT, 'scripts/release-macos-app.py'), 'verify', '--source', path.resolve(source)], {stdio: 'inherit'});
      const lock = path.join(ROOT, '.internal/deploy/.gui-publish.lock');
      fs.mkdirSync(path.dirname(lock), {recursive: true, mode: 0o700});
      fs.mkdirSync(lock, {mode: 0o700}); // Never steal a stale or remote writer lock.
      fs.writeFileSync(path.join(lock, 'owner.json'), JSON.stringify({pid: process.pid, version: inventory.version,
        commit: inventory.commit, started: new Date().toISOString()}));
      try {
        const secret = process.env.CLOUDFLARE_API_TOKEN ?? fs.readFileSync(path.join(os.homedir(), '.cf'), 'utf8').trim();
        await publishObjects(objects, inventory, cloudflareStore(secret), publicRead);
      } finally { fs.unlinkSync(path.join(lock, 'owner.json')); fs.rmdirSync(lock); }
    } else if (command === 'verify-public') {
      for (const object of objects) check((await publicRead(object.key)).equals(object.body), `Public bytes differ: ${object.key}`);
    }
    console.log(`GUI ${inventory.version} ${command} complete (${inventory.commit}).`);
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(error => {
  console.error(`GUI distribution failed: ${error.message}`); process.exitCode = 1;
});
