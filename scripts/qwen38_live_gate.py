"""Execute the primary SKU gates and preserve failing evidence (no weight download)."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def validate_host(host: dict) -> None:
    if (host.get('system'), host.get('arch'), host.get('chip'), host.get('memory_bytes')) != (
        'Darwin', 'arm64', 'Apple M4 Pro', 64 * 1024**3
    ) or host.get('model') != 'Mac16,11':
        raise ValueError('qualification requires Mac mini M4 Pro 64 GB (Mac16,11)')


def validate_cells(cells: list[dict]) -> None:
    if len(cells) != 2 or {c.get('mode') for c in cells} != {'direct', 'mtp'}:
        raise ValueError('exactly one direct and one MTP result are required')
    if any(c.get('status') != 'ok' or c.get('surface_passed') is not True
           or c.get('qa_items', 0) < 32 or c.get('qa_hard_passed') != c.get('qa_items')
           for c in cells):
        raise ValueError('QA failed, skipped, partial, incomplete, or missing surface evidence')
    for cell in cells:
        draft, verify = cell.get('mtp_draft_tokens', 0), cell.get('mtp_verify_tokens', 0)
        if (cell['mode'] == 'mtp' and (draft <= 0 or verify <= 0)) or (
            cell['mode'] == 'direct' and (draft != 0 or verify != 0)
        ):
            raise ValueError('server telemetry does not prove the requested route')


def validate_paired_greedy(direct: dict, mtp: dict) -> dict:
    """Reject a route split even when the separate product-health suites pass."""
    for mode, response in (("direct", direct), ("mtp", mtp)):
        tokens = response.get("output_tokens")
        if (response.get("status") != "finished"
                or response.get("prompt_tokens") != list(range(1, 17))
                or not isinstance(tokens, list) or len(tokens) != 64
                or any(type(token) is not int or token < 0 for token in tokens)):
            raise ValueError(f"{mode} greedy probe is missing, failed, or incomplete")
    for index, (left, right) in enumerate(zip(direct["output_tokens"], mtp["output_tokens"])):
        if left != right:
            raise ValueError(f"direct/MTP greedy probe differs at output index {index}")
    return {"matched": True, "prompt_tokens": 16, "output_tokens": 64,
            "scope": "One pinned raw-token probe; not broad accuracy or MTP Tier 2"}


def run_live(args, contract: dict, repo: Path) -> int:
    import run_qa_matrix as matrix

    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    result = {'schema': 2, 'status': 'failed', 'contract': contract, 'cells': []}
    try:
        def capture(command: list[str]) -> str:
            return subprocess.check_output(command, text=True, cwd=repo, timeout=30).strip()

        host = {'system': platform.system(), 'arch': platform.machine()}
        for key, name in [('chip', 'machdep.cpu.brand_string'), ('model', 'hw.model'),
                          ('memory_bytes', 'hw.memsize')]:
            value = capture(['/usr/sbin/sysctl', '-n', name])
            host[key] = int(value) if key == 'memory_bytes' else value
        host['os'] = capture(['/usr/bin/sw_vers', '-productVersion'])
        result['host'] = host
        validate_host(host)
        commit = capture(['git', 'rev-parse', 'HEAD'])
        if capture(['git', 'status', '--porcelain', '--untracked-files=normal']):
            raise ValueError('qualification source checkout is not clean')
        result['source_commit'] = commit
        if Path(matrix.__file__).resolve() != repo / 'scripts/run_qa_matrix.py':
            raise ValueError('QA harness did not load from the checked source tree')
        manifest = json.loads(args.build_manifest.read_text())
        if manifest.get('source_commit') != commit or manifest.get('dirty') is not False:
            raise ValueError('build manifest must match the clean source commit')
        result['build_manifest'] = manifest
        result['build_manifest_sha256'] = digest(args.build_manifest)
        for key, path in [('server', args.server_bin), ('bench', args.bench_bin), ('wheel', args.wheel), ('cli', args.cli)]:
            if digest(path) != manifest.get(key + '_sha256'):
                raise ValueError(f'{key} does not match build manifest')
        overrides = sorted(k for k in os.environ if k.startswith(('AX_', 'DYLD_', 'MLX_', 'MTL_', 'METAL_', 'PYTHONPATH')))
        result['runtime_overrides'] = overrides
        if overrides:
            raise ValueError('release qualification requires product defaults; unset AX_/DYLD_ overrides')
        import ax_engine
        import ax_engine._ax_engine
        package = Path(ax_engine.__file__).resolve().parent
        if args.server_bin.resolve() != package / '_bin/ax-engine-server' or args.bench_bin.resolve() != package / '_bin/ax-engine-bench':
            raise ValueError('executables are not from the installed Python package')
        if args.cli.absolute().parent != Path(sys.executable).absolute().parent:
            raise ValueError('CLI must belong to the current isolated Python environment')
        with zipfile.ZipFile(args.wheel) as wheel:
            for name in wheel.namelist():
                if name.startswith('ax_engine/') and not name.endswith('/'):
                    installed = package.parent / name
                    with wheel.open(name) as stream:
                        expected = hashlib.file_digest(stream, 'sha256').hexdigest()
                    if digest(installed) != expected:
                        raise ValueError(f'installed wheel member differs: {name}')
        result['installed_package'] = str(package)
        result['bundled_runtime_environment'] = {k: v for k, v in os.environ.items() if k == 'AX_ENGINE_METAL_BUILD_DIR'}
        if manifest.get('model_revision') != contract['revision']:
            raise ValueError('model revision mismatch')
        files = manifest.get('model_files', {})
        canonical = json.dumps(files, sort_keys=True, separators=(',', ':')).encode()
        if hashlib.sha256(canonical).hexdigest() != contract['model_manifest_sha256']:
            raise ValueError('model hashes differ from the source-pinned inventory')
        if not files or not any(name.endswith('.safetensors') for name in files):
            raise ValueError('model manifest must include checkpoint weights')
        actual_files = {str(p.relative_to(args.model_dir)) for p in args.model_dir.rglob('*') if p.is_file()}
        if actual_files != set(files):
            raise ValueError('model file inventory mismatch')
        for name, expected in files.items():
            path = args.model_dir / name
            if not path.resolve().is_relative_to(args.model_dir.resolve()) or digest(path) != expected:
                raise ValueError(f'model file hash mismatch: {name}')
        with (out / 'doctor.stdout.json').open('w') as stdout, (out / 'doctor.stderr.log').open('w') as stderr:
            doctor = subprocess.run(
                [str(args.cli), 'doctor', '--mlx-model-artifacts-dir', str(args.model_dir), '--json'],
                stdout=stdout, stderr=stderr, timeout=180, cwd=repo,
            )
        report = json.loads((out / 'doctor.stdout.json').read_text())
        if doctor.returncode or report.get('result') != 'ready' or 'model_checks' not in report.get('ready_for', []):
            raise ValueError('doctor did not confirm model readiness')
        # Explicit wheel executables, never an incidental development build.
        os.environ['QA_BENCH_BIN'] = str(args.bench_bin)
        os.environ['AX_ALLOW_UNSUPPORTED_HOST'] = '0'
        result['enforced_environment'] = {'AX_ALLOW_UNSUPPORTED_HOST': '0', 'QA_BENCH_BIN': str(args.bench_bin)}
        for mode in ('direct', 'mtp'):
            cell = matrix.run_cell(
                matrix.Cell(mode, 'qwen3.8-27b', args.model_dir), repo=repo, scratch=out,
                server_bin=args.server_bin, host='127.0.0.1', port=args.port,
                seed=20260917, sample=16, timeout=180, ready_max=420,
                run_surface=True, streams='both', verify_live_route=True,
            )
            result['cells'].append(asdict(cell))
            (out / 'qualification.json').write_text(json.dumps(result, default=str, indent=2) + '\n')
        result['paired_greedy_artifacts'] = {
            path.name: digest(path)
            for mode in ('direct', 'mtp')
            for path in (out / f'server-route-{mode}-qwen3.8-27b.json',
                         out / f'server-route-request-{mode}-qwen3.8-27b.json')
        }
        validate_cells(result['cells'])
        result['paired_greedy'] = validate_paired_greedy(
            json.loads((out / 'server-route-direct-qwen3.8-27b.json').read_text()),
            json.loads((out / 'server-route-mtp-qwen3.8-27b.json').read_text()),
        )
        result['status'] = 'passed'
    except Exception as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        (out / 'qualification.json').write_text(json.dumps(result, default=str, indent=2) + '\n')
    print(json.dumps(result, default=str, indent=2))
    return 0 if result['status'] == 'passed' else 1
