"""Hugging Face repo reference parsing, mirroring `ax-engine-core/src/repo_ref.rs`.

Accepts bare `owner/repo`, `owner/repo@revision`, full
`https://huggingface.co/owner/repo` links (also `hf.co`, scheme-less, `.git`,
trailing slash), and `/tree/<revision>` URLs. Keep the accepted grammar in
sync with the Rust parser; both are covered by parity-style unit tests.
"""

from __future__ import annotations

_HF_HOSTS = ("huggingface.co", "hf.co")


def _valid_segment(segment: str) -> bool:
    return bool(segment) and all(
        ch.isalnum() and ord(ch) < 128 or ch in "-_." for ch in segment
    )


def _invalid(value: str) -> ValueError:
    return ValueError(
        f"invalid Hugging Face repo reference {value!r}; expected `owner/repo` or "
        "https://huggingface.co/owner/repo (optionally with @revision or /tree/revision)"
    )


def parse_repo_ref(value: str) -> tuple[str, str | None]:
    """Parse `value` into `(repo_id, revision)`. Raises ValueError on bad input."""
    text = value.strip()
    if not text:
        raise ValueError("empty model reference; pass `owner/repo` or a Hugging Face URL")

    revision: str | None = None
    if "://" in text:
        scheme, rest = text.split("://", 1)
        if scheme not in ("http", "https"):
            raise ValueError(
                f"unsupported URL scheme in {value!r}; only https://huggingface.co links are supported"
            )
        host, _, path = rest.partition("/")
        host = host.lower()
        if host.startswith("www."):
            host = host[4:]
        if host not in _HF_HOSTS:
            raise ValueError(
                f"unsupported model host {host!r}; only huggingface.co links are supported "
                "(or pass a bare `owner/repo` repo id)"
            )
        text = path
    else:
        for host in _HF_HOSTS:
            prefix = f"{host}/"
            if text.startswith(prefix):
                text = text[len(prefix) :]
                break

    segments = text.rstrip("/").split("/")
    if any(not segment for segment in segments):
        raise _invalid(value)
    if segments[-1].endswith(".git"):
        segments[-1] = segments[-1][: -len(".git")]

    if len(segments) > 2:
        if segments[2] == "tree" and len(segments) == 4 and segments[3]:
            revision = segments[3]
            segments = segments[:2]
        elif segments[2] in ("blob", "resolve"):
            raise ValueError(
                f"{value!r} links to a single file, not a model repo; "
                "pass the model page (https://huggingface.co/owner/repo) instead"
            )
        else:
            raise _invalid(value)
    elif "@" in segments[-1]:
        base, _, rev = segments[-1].rpartition("@")
        if not rev:
            raise ValueError(f"empty revision in {value!r}; expected `owner/repo@revision`")
        segments[-1] = base
        revision = rev

    if len(segments) != 2 or not all(_valid_segment(s) for s in segments):
        raise _invalid(value)
    if revision is not None and any(ch.isspace() for ch in revision):
        raise ValueError(f"invalid revision {revision!r} in {value!r}")
    return f"{segments[0]}/{segments[1]}", revision
