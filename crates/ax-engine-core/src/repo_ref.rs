//! Hugging Face repo reference parsing shared by every download front end
//! (Rust CLI, TUI, Python CLI/SDK via mirrored rules).
//!
//! Accepted forms:
//! - bare repo id: `owner/repo`
//! - repo id with revision: `owner/repo@<rev>` (branch, tag, or commit sha)
//! - full URL: `https://huggingface.co/owner/repo` (also `http://`, `hf.co`,
//!   scheme-less `huggingface.co/...`)
//! - URL with revision: `https://huggingface.co/owner/repo/tree/<rev>`
//! - trailing `/` and a `.git` suffix on the repo segment
//!
//! File links (`/blob/...`, `/resolve/...`) and non-Hugging-Face hosts are
//! rejected with an actionable message rather than failing deep inside the
//! hub library.

/// A parsed Hugging Face model repo reference.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RepoRef {
    /// `owner/repo`, validated.
    pub repo_id: String,
    /// Optional revision (branch, tag, or commit sha) from `@rev` or
    /// `/tree/<rev>`.
    pub revision: Option<String>,
}

const HF_HOSTS: [&str; 2] = ["huggingface.co", "hf.co"];
const MAX_REPO_ID_BYTES: usize = 96;

fn valid_segment(segment: &str) -> bool {
    !segment.is_empty()
        && !segment.starts_with(['-', '.'])
        && !segment.ends_with(['-', '.'])
        && !segment.contains("--")
        && !segment.contains("..")
        && !segment.ends_with(".git")
        && segment
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'))
}

fn invalid_repo_id_message(input: &str) -> String {
    format!(
        "invalid Hugging Face repo reference {input:?}; expected `owner/repo` or \
         https://huggingface.co/owner/repo (optionally with @revision or /tree/revision)"
    )
}

fn invalid_revision_message(input: &str, revision: &str) -> String {
    format!(
        "invalid Git revision {revision:?} in {input:?}; expected a branch, tag, or commit \
         without traversal, empty path components, or Git-ref metacharacters"
    )
}

fn is_hf_host(host: &str) -> bool {
    let host = host
        .get(..4)
        .filter(|prefix| prefix.eq_ignore_ascii_case("www."))
        .map_or(host, |_| &host[4..]);
    HF_HOSTS
        .iter()
        .any(|candidate| host.eq_ignore_ascii_case(candidate))
}

fn is_hf_authority(authority: &str) -> bool {
    if authority.contains('@') {
        return false;
    }
    let (host, port) = authority
        .split_once(':')
        .map_or((authority, None), |(host, port)| (host, Some(port)));
    if port.is_some_and(|port| {
        port.is_empty()
            || !port.bytes().all(|byte| byte.is_ascii_digit())
            || port.parse::<u16>().is_err()
    }) {
        return false;
    }
    is_hf_host(host)
}

fn strip_url_query_and_fragment(path: &str) -> &str {
    path.find(['?', '#']).map_or(path, |index| &path[..index])
}

fn percent_nibble(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn percent_decode_revision(revision: &str, input: &str) -> Result<String, String> {
    let bytes = revision.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] != b'%' {
            decoded.push(bytes[index]);
            index += 1;
            continue;
        }
        let Some(high) = bytes.get(index + 1).and_then(|byte| percent_nibble(*byte)) else {
            return Err(invalid_revision_message(input, revision));
        };
        let Some(low) = bytes.get(index + 2).and_then(|byte| percent_nibble(*byte)) else {
            return Err(invalid_revision_message(input, revision));
        };
        decoded.push((high << 4) | low);
        index += 3;
    }
    String::from_utf8(decoded).map_err(|_| invalid_revision_message(input, revision))
}

fn validate_revision(revision: &str, input: &str) -> Result<(), String> {
    let invalid_character = revision.chars().any(|character| {
        character.is_whitespace()
            || character.is_control()
            || matches!(character, '~' | '^' | ':' | '?' | '*' | '[' | '\\')
    });
    let invalid_component = revision.split('/').any(|component| {
        component.is_empty()
            || component.starts_with('.')
            || component
                .get(component.len().saturating_sub(5)..)
                .is_some_and(|suffix| suffix.eq_ignore_ascii_case(".lock"))
    });
    if revision.is_empty()
        || revision == "@"
        || revision.starts_with('/')
        || revision.ends_with('/')
        || revision.ends_with('.')
        || revision.contains("//")
        || revision.contains("..")
        || revision.contains("@{")
        || invalid_character
        || invalid_component
    {
        return Err(invalid_revision_message(input, revision));
    }
    Ok(())
}

/// Parse `input` into a [`RepoRef`].
pub fn parse_repo_ref(input: &str) -> Result<RepoRef, String> {
    let mut value = input.trim();
    if value.is_empty() {
        return Err("empty model reference; pass `owner/repo` or a Hugging Face URL".into());
    }

    // Strip scheme and host for URL forms.
    if let Some((scheme, rest)) = value.split_once("://") {
        if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
            return Err(format!(
                "unsupported URL scheme in {input:?}; only https://huggingface.co links are supported"
            ));
        }
        let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
        if !is_hf_authority(host) {
            return Err(format!(
                "unsupported model host {host:?}; only huggingface.co links are supported \
                 (or pass a bare `owner/repo` repo id)"
            ));
        }
        value = strip_url_query_and_fragment(path);
    } else if let Some((host, path)) = value.split_once('/')
        && is_hf_host(host)
    {
        value = strip_url_query_and_fragment(path);
    }

    // A single trailing slash is tolerated. More than one remains visible as
    // an empty component and is rejected below.
    let trimmed = value.strip_suffix('/').unwrap_or(value);
    let segments: Vec<&str> = trimmed.split('/').collect();
    if segments.len() < 2 || segments[0].is_empty() || segments[1].is_empty() {
        return Err(invalid_repo_id_message(input));
    }

    // Split `@revision` before interpreting path suffixes so slash-containing
    // refs such as `owner/repo@feature/foo` remain representable.
    let (repo_segment, at_revision) = segments[1]
        .split_once('@')
        .map_or((segments[1], None), |(repo, revision)| {
            (repo, Some(revision))
        });
    let repo_segment = repo_segment.strip_suffix(".git").unwrap_or(repo_segment);

    let revision = if let Some(revision_head) = at_revision {
        let mut raw = revision_head.to_string();
        if segments.len() > 2 {
            raw.push('/');
            raw.push_str(&segments[2..].join("/"));
        }
        Some(raw)
    } else if segments.len() > 2 {
        match segments[2] {
            "tree" => Some(segments[3..].join("/")),
            "blob" | "resolve" => {
                return Err(format!(
                    "{input:?} links to a single file, not a model repo; \
                     pass the model page (https://huggingface.co/owner/repo) instead"
                ));
            }
            _ => return Err(invalid_repo_id_message(input)),
        }
    } else {
        None
    };

    build_ref(segments[0], repo_segment, revision.as_deref(), input)
}

fn build_ref(
    owner: &str,
    repo: &str,
    revision: Option<&str>,
    input: &str,
) -> Result<RepoRef, String> {
    let repo_id = format!("{owner}/{repo}");
    if repo_id.len() > MAX_REPO_ID_BYTES || !valid_segment(owner) || !valid_segment(repo) {
        return Err(invalid_repo_id_message(input));
    }
    let revision = revision
        .map(|revision| percent_decode_revision(revision, input))
        .transpose()?;
    if let Some(revision) = &revision {
        validate_revision(revision, input)?;
    }
    Ok(RepoRef { repo_id, revision })
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

    use super::*;

    fn ok(input: &str, repo: &str, rev: Option<&str>) {
        let parsed = parse_repo_ref(input).unwrap_or_else(|e| panic!("{input:?} failed: {e}"));
        assert_eq!(parsed.repo_id, repo, "repo for {input:?}");
        assert_eq!(parsed.revision.as_deref(), rev, "revision for {input:?}");
    }

    fn err(input: &str) {
        assert!(parse_repo_ref(input).is_err(), "{input:?} must fail");
    }

    #[test]
    fn bare_repo_id() {
        ok(
            "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP",
            "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP",
            None,
        );
        ok(
            "mlx-community/Qwen3-4B-4bit",
            "mlx-community/Qwen3-4B-4bit",
            None,
        );
    }

    #[test]
    fn full_urls() {
        ok(
            "https://huggingface.co/AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP",
            "AutomatosX/AX-Qwen3.6-35B-A3B-MLX-6bit-MTP",
            None,
        );
        ok("http://huggingface.co/owner/repo", "owner/repo", None);
        ok("https://hf.co/owner/repo", "owner/repo", None);
        ok("huggingface.co/owner/repo", "owner/repo", None);
        ok("https://www.huggingface.co/owner/repo", "owner/repo", None);
        ok("HTTPS://HF.CO/owner/repo", "owner/repo", None);
        ok("WWW.HUGGINGFACE.CO/owner/repo", "owner/repo", None);
        ok("https://huggingface.co:443/owner/repo", "owner/repo", None);
        ok("https://huggingface.co/owner/repo/", "owner/repo", None);
        ok(
            "https://huggingface.co/owner/repo?download=true#files",
            "owner/repo",
            None,
        );
        ok(
            "https://huggingface.co/owner/repo?ignored=\tvalue",
            "owner/repo",
            None,
        );
        ok("https://huggingface.co/owner/repo.git", "owner/repo", None);
    }

    #[test]
    fn revisions() {
        ok("owner/repo@v1", "owner/repo", Some("v1"));
        ok("owner/repo@abc123def", "owner/repo", Some("abc123def"));
        ok(
            "https://huggingface.co/owner/repo/tree/v1",
            "owner/repo",
            Some("v1"),
        );
        ok(
            "https://hf.co/owner/repo/tree/main",
            "owner/repo",
            Some("main"),
        );
        ok(
            "owner/repo@feature/download-flow",
            "owner/repo",
            Some("feature/download-flow"),
        );
        ok(
            "https://huggingface.co/owner/repo/tree/refs/pr/123",
            "owner/repo",
            Some("refs/pr/123"),
        );
        ok(
            "https://huggingface.co/owner/repo/tree/feature%2Fdownload-flow",
            "owner/repo",
            Some("feature/download-flow"),
        );
        ok(
            "https://huggingface.co/owner/repo/tree/main#readme",
            "owner/repo",
            Some("main"),
        );
        ok(
            "https://huggingface.co/owner/repo.git/tree/release.git",
            "owner/repo",
            Some("release.git"),
        );
        ok(
            "owner/repo.git@release.git",
            "owner/repo",
            Some("release.git"),
        );
        ok(
            "owner/repo@release@candidate",
            "owner/repo",
            Some("release@candidate"),
        );
    }

    #[test]
    fn rejects_bad_input() {
        err("");
        err("   ");
        err("noslash");
        err("https://huggingface.co/owner");
        err("https://example.com/owner/repo");
        err("ftp://huggingface.co/owner/repo");
        err("https://huggingface.co:/owner/repo");
        err("https://huggingface.co:invalid/owner/repo");
        err("https://huggingface.co:65536/owner/repo");
        err("https://huggingface.co:443:extra/owner/repo");
        err("https://huggingface.\nco/owner/repo");
        err("https://huggingface.co/owner/re\tpo");
        err("\0https://huggingface.co/owner/repo");
        err("https://huggingface.co/owner/repo/blob/main/model.safetensors");
        err("https://huggingface.co/owner/repo/resolve/main/model.safetensors");
        err("C:/owner/repo");
        err(r"owner\repo/model");
        err(r"owner/repo@C:\temp");
        err("owner/repo/extra/path");
        err("owner/repo@");
        err("owner//repo");
        err("owner/re po");
        err("-owner/repo");
        err("owner-/repo");
        err("owner/-repo");
        err("owner/repo.");
        err("owner/re--po");
        err("owner/re..po");
        err("owner/repo.git.git");
        err(&format!("owner/{}", "r".repeat(MAX_REPO_ID_BYTES)));
    }

    #[test]
    fn repo_id_length_boundary_is_enforced() {
        let longest = format!("o/{}", "r".repeat(MAX_REPO_ID_BYTES - 2));
        ok(&longest, &longest, None);
        err(&format!("o/{}", "r".repeat(MAX_REPO_ID_BYTES - 1)));
    }

    #[test]
    fn rejects_unsafe_or_invalid_git_revisions() {
        for input in [
            "owner/repo@../other",
            "owner/repo@feature//other",
            "owner/repo@feature/.",
            "owner/repo@feature/.hidden",
            "owner/repo@feature.lock",
            "owner/repo@feature.LOCK",
            "owner/repo@feature..other",
            "owner/repo@feature@{1}",
            "owner/repo@feature~1",
            "owner/repo@feature^1",
            "owner/repo@feature:other",
            "owner/repo@feature?other",
            "owner/repo@feature*other",
            "owner/repo@feature[other",
            r"owner/repo@feature\other",
            "owner/repo@feature\u{0080}other",
            "owner/repo@feature.",
            "owner/repo@@",
            "https://huggingface.co/owner/repo/tree/%2E%2E/other",
            "https://huggingface.co/owner/repo/tree/feature%2F%2Fother",
            "https://huggingface.co/owner/repo/tree/feature%00other",
            "https://huggingface.co/owner/repo/tree/feature%ZZother",
        ] {
            err(input);
        }
    }

    #[test]
    fn whitespace_is_trimmed() {
        ok("  owner/repo  ", "owner/repo", None);
        ok("\towner/repo\n", "owner/repo", None);
        err("\u{001c}owner/repo");
    }
}
