// Adapted from AX Code Desktop download gateway; isolated AX Engine channel.
const prefix = "/ax-engine-desktop/"
const rootFiles = new Set(["install", "latest.json"])

export function objectKey(url) {
  if (url.hostname !== "download.automatosx.com" || !url.pathname.startsWith(prefix)) return null
  const key = url.pathname.slice(prefix.length)
  const bare = key.replace(/\.minisig$/, "")
  if (rootFiles.has(bare)) return key
  return /^releases\/download\/v\d+\.\d+\.\d+\/[A-Za-z0-9][A-Za-z0-9._-]{0,200}$/.test(key) ? key : null
}

function error(message, status, headers = {}) {
  return new Response(message, { status, headers: { "Cache-Control": "no-store", ...headers } })
}

export function byteRange(value, size) {
  const match = /^bytes=(\d*)-(\d*)$/.exec(value ?? "")
  if (!match || (!match[1] && !match[2])) return null
  let start = match[1] ? Number(match[1]) : Math.max(0, size - Number(match[2]))
  const end = match[1] ? (match[2] ? Math.min(Number(match[2]), size - 1) : size - 1) : size - 1
  if (!Number.isSafeInteger(start) || !Number.isSafeInteger(end) || start > end || start >= size) return null
  return { offset: start, length: end - start + 1 }
}

export default {
  async fetch(request, env) {
    if (!["GET", "HEAD"].includes(request.method)) return error("Method not allowed", 405, { Allow: "GET, HEAD" })
    const key = objectKey(new URL(request.url))
    if (!key) return error("Not found", 404)
    if (env.PUBLIC_RELEASE_ENABLED !== "true") return error("AX Engine GUI public release is not enabled", 503)
    const metadata = await env.ENGINE_GUI_RELEASES.head(key)
    if (!metadata) return error("Not found", 404)
    const headers = new Headers()
    metadata.writeHttpMetadata(headers)
    headers.set("Content-Type", key.endsWith(".json") ? "application/json" : key.endsWith(".dmg") ? "application/x-apple-diskimage" : "text/plain; charset=utf-8")
    headers.set("ETag", metadata.httpEtag)
    headers.set("X-Content-Type-Options", "nosniff")
    headers.set(
      "Cache-Control",
      key.startsWith("releases/download/") ? "public, max-age=31536000, immutable" : "no-store",
    )
    headers.set("Accept-Ranges", "bytes")
    if (
      request.headers
        .get("If-None-Match")
        ?.split(/\s*,\s*/)
        .some((etag) => etag === metadata.httpEtag || etag === "*")
    ) {
      return new Response(null, { status: 304, headers })
    }
    let range
    if (
      request.method === "GET" &&
      request.headers.has("Range") &&
      (!request.headers.has("If-Range") || request.headers.get("If-Range") === metadata.httpEtag)
    ) {
      range = byteRange(request.headers.get("Range"), metadata.size)
      if (!range) return error("Invalid byte range", 416, { "Content-Range": `bytes */${metadata.size}` })
      headers.set("Content-Range", `bytes ${range.offset}-${range.offset + range.length - 1}/${metadata.size}`)
    }
    headers.set("Content-Length", String(range?.length ?? metadata.size))
    if (request.method === "HEAD") return new Response(null, { headers })
    // Bind the body to the metadata read above, including mutable root pointers.
    const object = await env.ENGINE_GUI_RELEASES.get(key, {
      onlyIf: { etagMatches: metadata.etag },
      ...(range ? { range } : {}),
    })
    if (!object || !("body" in object)) return error("Object changed; retry the request", 503)
    return new Response(object.body, { status: range ? 206 : 200, headers })
  },
}
