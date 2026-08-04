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

fn valid_segment(segment: &str) -> bool {
    !segment.is_empty()
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

/// Parse `input` into a [`RepoRef`].
pub fn parse_repo_ref(input: &str) -> Result<RepoRef, String> {
    let mut value = input.trim();
    if value.is_empty() {
        return Err("empty model reference; pass `owner/repo` or a Hugging Face URL".into());
    }

    // Strip scheme and host for URL forms.
    if let Some((scheme, rest)) = value.split_once("://") {
        if !matches!(scheme, "http" | "https") {
            return Err(format!(
                "unsupported URL scheme in {input:?}; only https://huggingface.co links are supported"
            ));
        }
        let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
        let host = host.to_ascii_lowercase();
        let host = host.strip_prefix("www.").unwrap_or(&host);
        if !HF_HOSTS.contains(&host) {
            return Err(format!(
                "unsupported model host {host:?}; only huggingface.co links are supported \
                 (or pass a bare `owner/repo` repo id)"
            ));
        }
        value = path;
    } else if let Some(rest) = HF_HOSTS
        .iter()
        .find_map(|host| value.strip_prefix(&format!("{host}/")))
    {
        value = rest;
    }

    // Split path segments; a `.git` suffix on the repo segment is tolerated.
    // Empty interior segments (`owner//repo`) are rejected rather than
    // silently collapsed.
    let trimmed = value.strip_suffix('/').unwrap_or(value);
    let mut segments: Vec<&str> = trimmed.split('/').collect();
    if segments.iter().any(|s| s.is_empty()) {
        return Err(invalid_repo_id_message(input));
    }
    if let Some(last) = segments.last_mut() {
        if let Some(stripped) = last.strip_suffix(".git") {
            *last = stripped;
        }
    }
    if segments.len() > 2 {
        match segments[2] {
            "tree" if segments.len() == 4 && !segments[3].is_empty() => {
                let revision = segments[3].to_string();
                return build_ref(segments[..2].to_vec(), Some(revision), input);
            }
            "blob" | "resolve" => {
                return Err(format!(
                    "{input:?} links to a single file, not a model repo; \
                     pass the model page (https://huggingface.co/owner/repo) instead"
                ));
            }
            _ => return Err(invalid_repo_id_message(input)),
        }
    }

    // `@revision` on the repo segment (after URL stripping).
    if let Some(last) = segments.last()
        && let Some((base, revision)) = last.rsplit_once('@')
    {
        if revision.is_empty() {
            return Err(format!(
                "empty revision in {input:?}; expected `owner/repo@revision`"
            ));
        }
        let mut owned: Vec<&str> = segments[..segments.len() - 1].to_vec();
        owned.push(base);
        if owned.len() == 1 {
            owned.insert(0, "");
        }
        return build_ref(owned, Some(revision.to_string()), input);
    }

    build_ref(segments, None, input)
}

fn build_ref(
    segments: Vec<&str>,
    revision: Option<String>,
    input: &str,
) -> Result<RepoRef, String> {
    if segments.len() != 2 || !valid_segment(segments[0]) || !valid_segment(segments[1]) {
        return Err(invalid_repo_id_message(input));
    }
    if let Some(rev) = &revision {
        if rev.chars().any(char::is_whitespace) {
            return Err(format!("invalid revision {rev:?} in {input:?}"));
        }
    }
    Ok(RepoRef {
        repo_id: format!("{}/{}", segments[0], segments[1]),
        revision,
    })
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
        ok("https://huggingface.co/owner/repo/", "owner/repo", None);
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
    }

    #[test]
    fn rejects_bad_input() {
        err("");
        err("   ");
        err("noslash");
        err("https://huggingface.co/owner");
        err("https://example.com/owner/repo");
        err("ftp://huggingface.co/owner/repo");
        err("https://huggingface.co/owner/repo/blob/main/model.safetensors");
        err("https://huggingface.co/owner/repo/resolve/main/model.safetensors");
        err("owner/repo/extra/path");
        err("owner/repo@");
        err("owner//repo");
        err("owner/re po");
    }

    #[test]
    fn whitespace_is_trimmed() {
        ok("  owner/repo  ", "owner/repo", None);
    }
}
