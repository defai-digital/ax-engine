// Sparkle archives use Ed25519 signatures, independently of the Minisign inventory.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const macos = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
export const config = JSON.parse(fs.readFileSync(path.join(macos, 'sparkle.json')));
const tools = path.join(macos, 'AXEngineApp/.build/artifacts/sparkle/Sparkle/bin');

export function signArchive(file) {
  const key = execFileSync(path.join(tools, 'generate_keys'), ['--account', config.keychainAccount, '-p'], {encoding:'utf8'}).trim();
  if (key !== config.publicKey) throw new Error('Sparkle Keychain identity does not match the pinned public key');
  return execFileSync(path.join(tools, 'sign_update'), ['--account', config.keychainAccount, '-p', file], {encoding:'utf8'}).trim();
}

export function verifyArchive(bytes, signature, publicKey = config.publicKey) {
  if (!/^[A-Za-z0-9+/]{86}==$/.test(signature) || Buffer.from(publicKey, 'base64').length !== 32)
    throw new Error('Invalid Sparkle signature or public key');
  const key = crypto.createPublicKey({key:Buffer.concat([
    Buffer.from('302a300506032b6570032100', 'hex'), Buffer.from(publicKey, 'base64')]), format:'der', type:'spki'});
  if (!crypto.verify(null, bytes, key, Buffer.from(signature, 'base64')))
    throw new Error('Invalid Sparkle archive signature');
}

export function appcast(version, size, signature) {
  if (!/^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$/.test(version) ||
      !Number.isSafeInteger(size) || size <= 0 || !/^[A-Za-z0-9+/]{86}==$/.test(signature))
    throw new Error('Invalid Sparkle release metadata');
  return `<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0" xmlns:sparkle="http://www.andymatuschak.org/xml-namespaces/sparkle">
  <channel>
    <title>AX Engine updates</title>
    <item>
      <title>AX Engine ${version}</title>
      <sparkle:version>${version}</sparkle:version>
      <sparkle:shortVersionString>${version}</sparkle:shortVersionString>
      <sparkle:minimumSystemVersion>26.2</sparkle:minimumSystemVersion>
      <description>Update AX Engine and its bundled runtime. Models, settings and conversations are preserved.</description>
      <enclosure url="https://download.automatosx.com/ax-engine-desktop/releases/download/v${version}/AX-Engine-${version}-mac-arm64.dmg" length="${size}" type="application/octet-stream" sparkle:edSignature="${signature}" />
    </item>
  </channel>
</rss>
`;
}

export function verifyAppcast(xml, version, image, publicKey = config.publicKey) {
  const signature = /sparkle:edSignature="([A-Za-z0-9+/=]+)"/.exec(xml)?.[1];
  if (!signature || xml !== appcast(version, image.length, signature)) throw new Error('Sparkle appcast binding mismatch');
  verifyArchive(image, signature, publicKey);
}
