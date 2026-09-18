import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path.cwd()
out = root / '.internal/reports/qwen38-first-split-20260918/validation'
sysroot = subprocess.check_output(['rustup', 'run', '1.97.1', 'rustc', '--print', 'sysroot'], text=True).strip()
env = {**os.environ, 'PATH': f'{root}/.venv/bin:{sysroot}/bin:' + os.environ['PATH'], 'VIRTUAL_ENV': str(root / '.venv')}
out.mkdir(exist_ok=True)
checks = [
    ('replay-regression', ['cargo','test','-p','ax-engine-mlx','--lib','forced_greedy_replay','--','--nocapture','--test-threads=1']),
    ('fmt', ['rustup', 'run', '1.97.1', 'cargo', 'fmt', '--check']),
    ('strict-clippy', ['cargo', 'clippy', '--all-targets', '--all-features', '--', '-D', 'warnings']),
    ('clippy', ['cargo', 'clippy', '--all-targets', '--all-features', '--', '-D', 'warnings',
                '--force-warn', 'clippy::unwrap-used', '--force-warn', 'clippy::expect-used',
                '--force-warn', 'clippy::panic', '--force-warn', 'clippy::dbg-macro',
                '--force-warn', 'clippy::large-enum-variant']),
    ('rust', ['rustup', 'run', '1.97.1', 'cargo', 'test', '--quiet', '--no-fail-fast']),
    ('maturin', ['maturin','develop']),
    ('scripts', ['bash','scripts/check-scripts.sh']),
    ('python', [str(root / '.venv/bin/python'), '-m', 'pytest', 'python/tests', '-q']),
    ('qwen27b', [str(root / '.venv/bin/python'), 'scripts/qualify_qwen38_27b.py', '--dry-run']),
    ('flash_next', [str(root / '.venv/bin/python'), 'scripts/qualify_qwen38_flash_next.py', '--dry-run']),
    ('claims', [str(root / '.venv/bin/python'), 'scripts/check_qwen38_primary_claims.py']),
]
paths = ['crates/ax-engine-mlx/src/bin/linear_mtp_state_oracle_probe.rs', 'crates/ax-engine-mlx/src/runner/mod.rs', 'crates/ax-engine-mlx/src/ngram_accel.rs']
source_hashes = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
(out / 'source-before.json').write_text(json.dumps(source_hashes, indent=2) + '\n')
results = []
for name, cmd in checks:
    started = time.monotonic()
    with (out / f'{name}.log').open('w') as log:
        code = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=2400).returncode
    results.append({'check': name, 'command': cmd, 'exit_code': code, 'elapsed_seconds': time.monotonic() - started})
    (out / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
    print(name, code, flush=True)
after = {p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths}
(out / 'source-after.json').write_text(json.dumps(after, indent=2) + '\n')
assert after == source_hashes, 'Source changed during validation'
raise SystemExit(any(row['exit_code'] for row in results))
