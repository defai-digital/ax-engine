// Derived from MTPLX, Copyright 2026 Youssof Altoukhi (Apache-2.0).
// Source: apps/MTPLXApp/Sources/MTPLXAppCore/Services/MTPLXServerURLs.swift
// Revision: 9882703f3105363ddc37eca9f97aa09a1d387112
// Modified for AX Engine: native presentation, bounded log messages and endpoint policy.
// See Resources/MTPLX-LICENSE.txt and Resources/MTPLX-NOTICE.txt.
import Foundation

// Bind addresses and client addresses are distinct.

public enum ServerURLs {
    /// Hosts that mean "this machine, loopback reachable without an API key"
    public static func isLoopbackBind(_ host: String) -> Bool {
        let raw = unbracketed(host).lowercased()
        return raw.isEmpty || raw == "127.0.0.1" || raw == "::1" || raw == "localhost"
    }

    /// True for wildcard binds that listen on every interface.
    public static func isWildcardBind(_ host: String) -> Bool {
        let raw = unbracketed(host).lowercased()
        return raw == "0.0.0.0" || raw == "::"
    }

    /// The address the app should CONNECT to for a given bind host.
    /// Wildcards and localhost aliases resolve to 127.0.0.1 (the app always
    /// runs on the same machine as the daemon it launched); anything else is
    /// taken verbatim.
    public static func connectHost(forBind host: String) -> String {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        let raw = unbracketed(trimmed).lowercased()
        switch raw {
        case "", "0.0.0.0", "::", "localhost":
            return "127.0.0.1"
        default:
            return trimmed
        }
    }

    /// Host formatted for a URL: bare IPv6 literals gain brackets.
    public static func urlHost(_ host: String) -> String {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.contains(":"), !trimmed.hasPrefix("[") {
            return "[\(trimmed)]"
        }
        return trimmed
    }

    /// Connectable base URL for a daemon bound to `bindHost`:`port`.
    public static func baseURL(bindHost: String, port: Int) -> URL {
        let host = urlHost(connectHost(forBind: bindHost))
        return URL(string: "http://\(host):\(port)")!
    }

    private static func unbracketed(_ host: String) -> String {
        let trimmed = host.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.hasPrefix("["), trimmed.hasSuffix("]") {
            return String(trimmed.dropFirst().dropLast())
        }
        return trimmed
    }
}
