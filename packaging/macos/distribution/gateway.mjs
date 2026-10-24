#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {ACCOUNT, BUCKET, DIRECTORY, check, verifyDirectory, releaseAuthority, releaseObjects, cloudflareStore} from './distribution.mjs';
const ZONE = 'd5743ccd847daa9482a61511cc347065';
const NAME = 'ax-engine-desktop-downloads';
const ROUTE = 'download.automatosx.com/ax-engine-desktop/*';
export function validateConfig(config) {
  check(config.name === NAME && config.main === 'worker.mjs' && config.workers_dev === false && config.preview_urls === false,
    'Unexpected Worker identity or public aliases');
  check(config.routes?.length === 1 && config.routes[0].pattern === ROUTE && config.routes[0].zone_name === 'automatosx.com' &&
    !config.routes[0].custom_domain, 'Unexpected route');
  check(config.vars?.PUBLIC_RELEASE_ENABLED === 'false', 'Checked-in gateway must default to disabled');
  check(config.r2_buckets?.length === 1 && config.r2_buckets[0].binding === 'ENGINE_GUI_RELEASES' &&
    config.r2_buckets[0].bucket_name === BUCKET, 'Unexpected R2 binding');
  check(Object.keys(config).every(k => ['name','main','compatibility_date','workers_dev','preview_urls','routes','vars','r2_buckets'].includes(k)),
    'Unexpected gateway configuration field');
  return {...config, account_id: ACCOUNT};
}
export function routeOwner(routes) {
  const owners = routes.filter(x => x.pattern === ROUTE);
  check(owners.length <= 1 && owners.every(x => x.script === NAME), 'Refusing route takeover');
  return owners[0];
}
async function main() {
  const [command, source, ciRun] = process.argv.slice(2);
  check(['inspect', 'provision', 'stage', 'enable'].includes(command),
    'usage: gateway.mjs inspect|provision | stage|enable <signed-release-dir> <ci-run-id>');
  let inventory, objects;
  if (['stage', 'enable'].includes(command)) {
    check(source && /^\d+$/.test(ciRun ?? ''), 'Signed source and CI run ID required');
    inventory = verifyDirectory(source); releaseAuthority(inventory, ciRun);
    objects = releaseObjects(source, inventory).filter(x => x.immutable);
  }
  const config = validateConfig(JSON.parse(fs.readFileSync(path.join(DIRECTORY, 'download-worker/wrangler.jsonc'))));
  const secret = process.env.CLOUDFLARE_API_TOKEN ?? fs.readFileSync(path.join(os.homedir(), '.cf'), 'utf8').trim();
  check(secret && !/[\r\n]/.test(secret), 'Invalid Cloudflare token');
  async function api(endpoint, method = 'GET', body) {
    const response = await fetch(`https://api.cloudflare.com/client/v4${endpoint}`, {
      method, headers: {Authorization: `Bearer ${secret}`, 'Content-Type': 'application/json'},
      ...(body ? {body: JSON.stringify(body)} : {}), signal: AbortSignal.timeout(30000)});
    const reply = await response.json();
    check(response.ok && reply.success, `Cloudflare ${method} failed: HTTP ${response.status}`);
    return reply.result;
  }
  const zone = await api(`/zones/${ZONE}`);
  check(zone.name === 'automatosx.com' && zone.account.id === ACCOUNT, 'Zone/account mismatch');
  const owner = routeOwner(await api(`/zones/${ZONE}/workers/routes`));
  const list = await api(`/accounts/${ACCOUNT}/r2/buckets`);
  const exists = list.buckets.some(x => x.name === BUCKET);
  let enabled = false;
  if (owner) {
    const settings = await api(`/accounts/${ACCOUNT}/workers/scripts/${NAME}/settings`);
    const flag = settings.bindings.find(x => x.name === 'PUBLIC_RELEASE_ENABLED');
    check(flag?.type === 'plain_text' && ['true','false'].includes(flag.text), 'Unexpected release switch');
    enabled = flag.text === 'true';
    check(settings.bindings.some(x => x.name === 'ENGINE_GUI_RELEASES' && x.bucket_name === BUCKET), 'Live bucket binding mismatch');
  }
  if (command === 'inspect') { console.log(JSON.stringify({bucketExists: exists, routeExists: !!owner, publicEnabled: enabled})); return; }
  if (command === 'provision') {
    check(!enabled, 'Public gateway is active; refusing to redeploy disabled configuration');
    if (!exists) await api(`/accounts/${ACCOUNT}/r2/buckets`, 'POST', {name: BUCKET});
  } else {
    check(owner && exists, 'Provision the isolated gateway first');
    const store = cloudflareStore(secret);
    for (const object of objects) {
      const current = await store.read(object.key);
      check(!current || current.equals(object.body), `Immutable object collision: ${object.key}`);
    }
    if (command === 'stage') {
      for (const object of objects) {
        if (!(await store.read(object.key))?.equals(object.body)) await store.write(object.key, object.body);
        check((await store.read(object.key))?.equals(object.body), `Readback failed: ${object.key}`);
      }
      console.log('Immutable assets staged and read back; latest pointer unchanged.'); return;
    }
    for (const object of objects) check((await store.read(object.key))?.equals(object.body), `Stage the complete release first: ${object.key}`);
    config.vars.PUBLIC_RELEASE_ENABLED = 'true';
  }
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), 'ax-engine-gateway-'));
  try {
    fs.copyFileSync(path.join(DIRECTORY, 'download-worker/worker.mjs'), path.join(temporary, 'worker.mjs'));
    fs.writeFileSync(path.join(temporary, 'wrangler.json'), JSON.stringify(config));
    execFileSync('wrangler', ['deploy', '--config', path.join(temporary, 'wrangler.json')],
      {env: {...process.env, CLOUDFLARE_API_TOKEN: secret}, stdio: ['ignore','pipe','pipe']});
    const after = routeOwner(await api(`/zones/${ZONE}/workers/routes`));
    check(after, 'Gateway route did not appear');
    const settings = await api(`/accounts/${ACCOUNT}/workers/scripts/${NAME}/settings`);
    check(settings.bindings.some(x => x.name === 'PUBLIC_RELEASE_ENABLED' && x.text === config.vars.PUBLIC_RELEASE_ENABLED), 'Gateway state readback failed');
  } finally { fs.rmSync(temporary, {recursive: true, force: true}); }
  console.log(`Gateway ${command} complete. Latest pointer unchanged; use the publisher after anonymous verification.`);
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(error => {
  console.error(`GUI gateway failed: ${error.message}`); process.exitCode = 1;
});
