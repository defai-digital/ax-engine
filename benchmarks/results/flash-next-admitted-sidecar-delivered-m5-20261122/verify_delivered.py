"""Run the native readiness controls on the bound delivered Flash Next snapshot."""
from hashlib import sha256
import json
from pathlib import Path
import re
import sys

import check_flash_next_native_support as native


ROUTE = "AX_FLASH_NEXT_MTP_SIDECAR_POSITIONAL"
LOG = "info,ax_engine_mlx::flash_next_startup=debug"
LIMITS = {"disabled": 300, "required": 120, "default": 120}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signature(rows):
    return [(row['name'], row['request'], row['response']['choices'], row['activation_delta'])
            for row in rows]


def main():
    binary, snapshot, inventory, output, stock_path, build_file, cache = map(Path, sys.argv[1:])
    stock_raw = stock_path.read_bytes()
    stock = json.loads(stock_raw)
    build = json.loads(build_file.read_bytes())
    require(native.sha256(binary) == build['server_sha256'], 'Candidate binary changed')
    require(stock['functional_controls_passed'] is True and stock['modes'] == list(LIMITS)
            and stock['inventory_sha256'] == native.sha256(inventory), 'Frozen delivery input changed')
    manifest = snapshot / 'model-manifest.json'
    require(stock['native_manifest_ready'] == {manifest.name: native.sha256(manifest)},
            'Snapshot is not the delivered manifest')
    make_env = native.server_env
    native.server_env = lambda explicit=None: make_env({ROUTE: '1', 'RUST_LOG': LOG} | (explicit or {}))
    sys.argv = [sys.argv[0], '--model-dir', str(snapshot), '--inventory', str(inventory),
                '--server-bin', str(binary), '--output', str(output), '--link-root', str(cache),
                '--modes', *LIMITS]
    require(native.main() == 0, 'The frozen native readiness runner failed')
    raw = (output/'result.json').read_bytes()
    observed = json.loads(raw)
    require(observed['server_sha256'] == build['server_sha256'] and observed['modes'] == list(LIMITS)
            and observed['native_manifest_after'] == stock['native_manifest_ready'], 'Result lineage changed')
    require(observed['negative_control_result']['activation_error_observed'] is True,
            'Required policy unexpectedly survived the kill switch')
    phases = {}
    for policy in LIMITS:
        require(signature(observed['results'][policy]) == signature(stock['results'][policy]),
                f'Output or activation changed under {policy}')
        log = (output / f'server-{policy}.log').read_text()
        require(log.count('Flash Next MTP sidecar positional read lab') == 1
                and 'Flash Next resident positional read lab' not in log,
                f'Unexpected payload reader under {policy}')
        phases[policy] = {}
        for phase in ('resident_tensors', 'mtp_sidecar'):
            samples = re.findall(fr'phase="{phase}" elapsed_us=(\d+)', log)
            require(len(samples) == 1 and int(samples[0]) > 0, f'Missing {policy} {phase} measurement')
            phases[policy][phase] = int(samples[0]) / 1e6
    observed_ready = observed['readiness_seconds']
    summary = {
        'hardware': observed['hardware'], 'revision': observed['revision'], 'build': build,
        'native_result_sha256': sha256(raw).hexdigest(), 'stock_result_sha256': sha256(stock_raw).hexdigest(),
        'matched_functional_controls': sum(len(rows) for rows in observed['results'].values()),
        'verified_required_steps': sum(row['activation_delta'][native.VERIFIED]
                                       for row in observed['results']['required']),
        'readiness_seconds': observed_ready, 'limits_seconds': LIMITS,
        'readiness_band_met': all(0 < observed_ready[policy] <= limit for policy, limit in LIMITS.items()),
        'phase_seconds': phases,
    }
    (output/'comparison.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
