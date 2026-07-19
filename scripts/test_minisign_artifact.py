"""Unit tests for scripts/minisign-artifact.sh.

These drive the signer via subprocess with a fake ``minisign`` on ``PATH`` so
the signing contract (pinned-key enforcement, password/Keychain passphrase
resolution, dry-run, untrusted-comment propagation) is exercised without a real
minisign keypair. Mirrors the ax-code-desktop ``minisign-artifacts.test.mjs``
suite in spirit.
"""

import os
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "minisign-artifact.sh")

PINNED_KEY = "RWS_FAKE_PINNED_AX_ENGINE_PUBLIC_KEY_MATERIAL"

FAKE_MINISIGN = """#!/usr/bin/env bash
set -euo pipefail
stdin="$(cat || true)"
if [ -n "$stdin" ]; then
  printf '%s' "$stdin" >> "${STDIN_LOG}"
fi
sig=""
mode="sign"
while [ "$#" -gt 0 ]; do
  case "$1" in
    -V) mode="verify"; shift ;;
    -x) sig="$2"; shift 2 ;;
    -c) printf '%s' "$2" >> "${COMMENT_LOG}"; shift 2 ;;
    -q) shift ;;
    *) shift ;;
  esac
done
if [ "$mode" = "sign" ] && [ -n "$sig" ]; then
  printf 'fake signature\\n' > "$sig"
fi
"""

FAKE_UNAME = """#!/usr/bin/env bash
printf 'Darwin\\n'
"""

FAKE_SECURITY = """#!/usr/bin/env bash
if [ "$1" = "find-generic-password" ]; then
  printf 'from-keychain\\n'
  exit 0
fi
exit 1
"""


def _write_executable(path, body):
    with open(path, "w") as fh:
        fh.write(body)
    os.chmod(path, 0o755)


def _make_fixture():
    tmp = tempfile.mkdtemp(prefix="ax-engine-minisign-test-")
    bindir = os.path.join(tmp, "bin")
    os.makedirs(bindir)

    secret_key = os.path.join(tmp, "ax.sec")
    public_key = os.path.join(tmp, "ax.pub")
    asset = os.path.join(tmp, "asset.tar.gz")
    stdin_log = os.path.join(tmp, "stdin.log")
    comment_log = os.path.join(tmp, "comment.log")

    with open(secret_key, "w") as fh:
        fh.write("secret\n")
    os.chmod(secret_key, 0o600)
    os.chmod(tmp, 0o700)  # secret key directory must be private

    with open(public_key, "w") as fh:
        fh.write(
            "untrusted comment: minisign public key DEADBEEFCAFEBABE\n"
            + PINNED_KEY
            + "\n"
        )
    with open(asset, "w") as fh:
        fh.write("asset bytes\n")

    _write_executable(os.path.join(bindir, "minisign"), FAKE_MINISIGN)

    return {
        "tmp": tmp,
        "bindir": bindir,
        "secret_key": secret_key,
        "public_key": public_key,
        "asset": asset,
        "stdin_log": stdin_log,
        "comment_log": comment_log,
    }


def _run(args, fixture, env=None):
    full_env = {
        **os.environ,
        "PATH": fixture["bindir"] + os.pathsep + os.environ.get("PATH", ""),
        "STDIN_LOG": fixture["stdin_log"],
        "COMMENT_LOG": fixture["comment_log"],
    }
    if env:
        full_env.update(env)
    return subprocess.run(
        ["bash", SCRIPT, *args],
        cwd=REPO_ROOT,
        env=full_env,
        capture_output=True,
        text=True,
    )


def _cleanup(fixture):
    import shutil

    shutil.rmtree(fixture["tmp"], ignore_errors=True)


class MinisignArtifactTests(unittest.TestCase):
    def setUp(self):
        self.fixture = _make_fixture()

    def tearDown(self):
        _cleanup(self.fixture)

    def test_rejects_public_key_that_does_not_match_pin(self):
        f = self.fixture
        # Public key file advertises PINNED_KEY; pin a *different* value.
        result = _run(
            [
                "--secret-key",
                f["secret_key"],
                "--public-key",
                f["public_key"],
                f["asset"],
            ],
            f,
            {"AX_MINISIGN_PINNED_PUBLIC_KEY": "RWS_SOME_OTHER_KEY_MATERIAL"},
        )
        self.assertNotEqual(result.returncode, 0, result.stderr)
        self.assertIn("does not match the pinned ax-engine release key", result.stderr)
        self.assertFalse(os.path.exists(f["asset"] + ".minisig"))

    def test_signs_when_public_key_matches_pin(self):
        f = self.fixture
        result = _run(
            [
                "--secret-key",
                f["secret_key"],
                "--public-key",
                f["public_key"],
                f["asset"],
            ],
            f,
            {"AX_MINISIGN_PINNED_PUBLIC_KEY": PINNED_KEY},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.exists(f["asset"] + ".minisig"))

    def test_passes_explicit_password_to_minisign(self):
        f = self.fixture
        result = _run(
            [
                "--secret-key",
                f["secret_key"],
                "--public-key",
                f["public_key"],
                f["asset"],
            ],
            f,
            {"AX_MINISIGN_PASSWORD": "from-env"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.exists(f["asset"] + ".minisig"))
        with open(f["stdin_log"]) as fh:
            self.assertEqual(fh.read(), "from-env")

    def test_uses_keychain_passphrase_when_no_password_env(self):
        f = self.fixture
        _write_executable(os.path.join(f["bindir"], "uname"), FAKE_UNAME)
        _write_executable(os.path.join(f["bindir"], "security"), FAKE_SECURITY)
        result = _run(
            [
                "--secret-key",
                f["secret_key"],
                "--public-key",
                f["public_key"],
                f["asset"],
            ],
            f,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(os.path.exists(f["asset"] + ".minisig"))
        with open(f["stdin_log"]) as fh:
            self.assertEqual(fh.read(), "from-keychain")

    def test_dry_run_creates_no_signature(self):
        f = self.fixture
        result = _run(
            [
                "--secret-key",
                f["secret_key"],
                "--public-key",
                f["public_key"],
                "--dry-run",
                f["asset"],
            ],
            f,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(os.path.exists(f["asset"] + ".minisig"))
        self.assertIn("Dry run only", result.stdout)

    def test_untrusted_comment_is_propagated(self):
        f = self.fixture
        result = _run(
            [
                "--secret-key",
                f["secret_key"],
                "--public-key",
                f["public_key"],
                "--untrusted-comment",
                "custom untrusted comment",
                f["asset"],
            ],
            f,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with open(f["comment_log"]) as fh:
            self.assertEqual(fh.read(), "custom untrusted comment")

    def test_refuses_to_overwrite_existing_signature_without_force(self):
        f = self.fixture
        # First signing succeeds.
        first = _run(
            [
                "--secret-key",
                f["secret_key"],
                "--public-key",
                f["public_key"],
                f["asset"],
            ],
            f,
        )
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue(os.path.exists(f["asset"] + ".minisig"))
        # Second signing without --force must fail.
        second = _run(
            [
                "--secret-key",
                f["secret_key"],
                "--public-key",
                f["public_key"],
                f["asset"],
            ],
            f,
        )
        self.assertNotEqual(second.returncode, 0, second.stderr)
        self.assertIn("signature already exists", second.stderr)
        # With --force it succeeds again.
        third = _run(
            [
                "--secret-key",
                f["secret_key"],
                "--public-key",
                f["public_key"],
                "--force",
                f["asset"],
            ],
            f,
        )
        self.assertEqual(third.returncode, 0, third.stderr)


if __name__ == "__main__":
    unittest.main()
