"""Hugging Face repo reference parsing, mirroring `ax-engine-core/src/repo_ref.rs`.

Accepts bare `owner/repo`, `owner/repo@revision`, full
`https://huggingface.co/owner/repo` links (also `hf.co`, scheme-less, `.git`,
trailing slash), and `/tree/<revision>` URLs. Keep the accepted grammar in
sync with the Rust parser; both are covered by parity-style unit tests.
"""

from __future__ import annotations

import unicodedata
from urllib.parse import unquote, urlsplit

_HF_HOSTS = ("huggingface.co", "hf.co")
_MAX_REPO_ID_BYTES = 96
_HEX_DIGITS = frozenset("0123456789abcdefABCDEF")
_INVALID_REVISION_CHARS = frozenset("~^:?*[\\")


def _is_control(character: str) -> bool:
    return unicodedata.category(character) == "Cc"


def _trim_reference(value: str) -> str:
    """Match Rust ``str::trim`` rather than Python's four extra C0 separators."""
    start = 0
    end = len(value)
    while start < end and value[start].isspace() and value[start] not in "\x1c\x1d\x1e\x1f":
        start += 1
    while end > start and value[end - 1].isspace() and value[end - 1] not in "\x1c\x1d\x1e\x1f":
        end -= 1
    return value[start:end]


def _valid_segment(segment: str) -> bool:
    return (
        bool(segment)
        and segment[0] not in "-."
        and segment[-1] not in "-."
        and "--" not in segment
        and ".." not in segment
        and not segment.endswith(".git")
        and all((ch.isalnum() and ch.isascii()) or ch in "-_." for ch in segment)
    )


def _invalid(value: str) -> ValueError:
    return ValueError(
        f"invalid Hugging Face repo reference {value!r}; expected `owner/repo` or "
        "https://huggingface.co/owner/repo (optionally with @revision or /tree/revision)"
    )


def _invalid_revision(revision: object, reference: str | None = None) -> ValueError:
    suffix = f" in {reference!r}" if reference is not None else ""
    return ValueError(
        f"invalid revision {revision!r}{suffix}; expected a branch, tag, or commit "
        "without whitespace, traversal components, or Git-ref control characters"
    )


def validate_revision(revision: str, *, reference: str | None = None) -> str:
    """Decode and validate a Hugging Face Git revision."""
    if not isinstance(revision, str) or not revision:
        raise _invalid_revision(revision, reference)
    revision = _percent_decode(revision, reference or revision)
    if revision.startswith("/") or revision.endswith("/") or revision.endswith("."):
        raise _invalid_revision(revision, reference)
    if revision == "@" or ".." in revision or "@{" in revision:
        raise _invalid_revision(revision, reference)
    if any(ch.isspace() or _is_control(ch) or ch in _INVALID_REVISION_CHARS for ch in revision):
        raise _invalid_revision(revision, reference)

    components = revision.split("/")
    if any(
        not component or component.startswith(".") or component.lower().endswith(".lock")
        for component in components
    ):
        raise _invalid_revision(revision, reference)
    return revision


def _percent_decode(text: str, value: str) -> str:
    for index, char in enumerate(text):
        if char == "%" and (
            index + 2 >= len(text)
            or text[index + 1] not in _HEX_DIGITS
            or text[index + 2] not in _HEX_DIGITS
        ):
            raise ValueError(f"invalid percent escape in Hugging Face reference {value!r}")
    try:
        return unquote(text, errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError(f"invalid UTF-8 escape in Hugging Face reference {value!r}") from error


def _url_path(value: str) -> str | None:
    text = value
    has_scheme = "://" in text
    lower = text.lower()
    is_schemeless_hf_url = any(
        lower.startswith(f"{host}/") or lower.startswith(f"www.{host}/") for host in _HF_HOSTS
    )
    if not has_scheme and not is_schemeless_hf_url:
        return None

    structural_part = text
    for separator in ("?", "#"):
        structural_part = structural_part.split(separator, 1)[0]
    if any(ch.isspace() or _is_control(ch) for ch in structural_part):
        raise _invalid(value)

    parsed = urlsplit(text if has_scheme else f"https://{text}")
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError(
            f"unsupported URL scheme in {value!r}; only https://huggingface.co links are supported"
        )
    try:
        host = (parsed.hostname or "").lower()
        port = parsed.port
    except ValueError as error:
        raise ValueError(
            f"unsupported model host in {value!r}; only huggingface.co links are supported"
        ) from error
    if host.startswith("www."):
        host = host[4:]
    explicit_empty_port = parsed.netloc.endswith(":") and port is None
    if (
        host not in _HF_HOSTS
        or parsed.username is not None
        or parsed.password is not None
        or explicit_empty_port
    ):
        raise ValueError(
            f"unsupported model host {host!r}; only huggingface.co links are supported "
            "(or pass a bare `owner/repo` repo id)"
        )
    return parsed.path.removeprefix("/")


def parse_repo_ref(value: str) -> tuple[str, str | None]:
    """Parse `value` into `(repo_id, revision)`. Raises ValueError on bad input."""
    text = _trim_reference(value)
    if not text:
        raise ValueError("empty model reference; pass `owner/repo` or a Hugging Face URL")

    url_path = _url_path(text)
    if url_path is not None:
        text = url_path
    if text.endswith("/"):
        text = text[:-1]

    segments = text.split("/")
    if any(not segment for segment in segments):
        raise _invalid(value)

    revision: str | None = None
    if len(segments) >= 2 and "@" in segments[1]:
        repo, revision_head = segments[1].split("@", 1)
        revision = "/".join([revision_head, *segments[2:]])
        segments = [segments[0], repo]
    elif len(segments) > 2:
        if segments[2] == "tree":
            revision = "/".join(segments[3:])
            segments = segments[:2]
        elif segments[2] in ("blob", "resolve"):
            raise ValueError(
                f"{value!r} links to a single file, not a model repo; "
                "pass the model page (https://huggingface.co/owner/repo) instead"
            )
        else:
            raise _invalid(value)

    if len(segments) == 2 and segments[1].endswith(".git"):
        segments[1] = segments[1][: -len(".git")]

    repo_id = "/".join(segments)
    if (
        len(segments) != 2
        or len(repo_id.encode("ascii", errors="ignore")) > _MAX_REPO_ID_BYTES
        or not all(_valid_segment(segment) for segment in segments)
    ):
        raise _invalid(value)
    if revision is not None:
        revision = validate_revision(revision, reference=value)
    return repo_id, revision
