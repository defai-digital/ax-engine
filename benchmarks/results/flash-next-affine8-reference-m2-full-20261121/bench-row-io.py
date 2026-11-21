"""Supplemental reads-only measurement for the Flash Next affine8 table."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import hashlib
import json
import mmap
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

from scripts import probe_flash_next_affine8_ple as ple
from scripts.run_flash_next_affine8_reference import Affine8Table

PROFILES = ('serial-pread','parallel-pread-4','mapped')
PLANES = tuple(ple.LAYOUT)

def shard_groups(table, ids):
    unique = np.unique(np.asarray(ids, dtype=np.int64))
    owners = np.searchsorted(table.ends,unique,side='right')
    groups = []
    for shard in np.unique(owners):
        start = int(table.ends[shard-1]) if shard else 0
        rows, planes = table.shards[shard]
        groups.append((rows,planes,[int(row)-start for row in unique[owners==shard]]))
    return groups

def pread_payload(groups, executor=None):
    with ExitStack() as stack:
        handles = {}
        for _,planes,_ in groups:
            for plane in planes.values():
                if plane.path not in handles:
                    handles[plane.path] = stack.enter_context(plane.path.open('rb'))
        def one(group):
            count,planes,local = group
            return [ple.read_rows(handles[p.path],p.offset,count,p.row_bytes,local)
                    for p in planes.values()]
        parts = map(one,groups) if executor is None else executor.map(one,groups)
        return b''.join(block for part in parts for block in part)

def mapped_payload(groups, mappings):
    payload = []
    for count,planes,local in groups:
        for name in PLANES:
            plane = planes[name]
            mapping = mappings.get(plane.path)
            if mapping is None:
                with plane.path.open('rb') as handle:
                    mapping = mmap.mmap(handle.fileno(),0,access=mmap.ACCESS_READ)
                mappings[plane.path] = mapping
            rows = np.ndarray((count,),dtype=f'V{plane.row_bytes}',buffer=mapping,offset=plane.offset)
            payload.append(rows[local].tobytes())
    return b''.join(payload)

def compare(table, tokens_and_rounds):
    observations = []
    mappings = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for tokens,rounds in tokens_and_rounds:
            for rotation in range(rounds):
                rng = random.Random(f'affine8-row-io:{tokens}:{rotation}')
                head_rows = table.row_count // 16
                ids = [head*head_rows+rng.randrange(head_rows)
                       for _ in range(tokens) for head in range(16)]
                groups = shard_groups(table,ids)
                arms = PROFILES[rotation % 3:] + PROFILES[:rotation % 3]
                baseline = None
                for pass_name in ('first-window','repeated-window'):
                    for profile in arms:
                        start = time.perf_counter()
                        if profile == 'mapped':
                            data = mapped_payload(groups,mappings)
                        else:
                            data = pread_payload(groups,pool if profile=='parallel-pread-4' else None)
                        seconds = time.perf_counter() - start
                        digest = hashlib.sha256(data).hexdigest()
                        if baseline is None:
                            baseline = digest
                        if digest != baseline or len(data) != 180*len(set(ids)):
                            raise ValueError(f'packed row evidence differs for {profile}')
                        observations.append({'tokens':tokens,'rotation':rotation,'pass':pass_name,
                                             'profile':profile,'rows':len(set(ids)),
                                             'digest':digest,'seconds':seconds})
    for mapping in mappings.values():
        mapping.close()
    return observations

def main():
    root=Path(sys.argv[1]).resolve()
    table=Affine8Table(root)
    rows=compare(table,[(1,6),(1024,3)])
    print(json.dumps({'scope':'CPU packed-row reads, no MLX forward or qualification',
                      'hardware':'Apple M2 Ultra 192 GB, external SSD','os':platform.mac_ver()[0],
                      'source_commit':'88cd09972',
                      'inventory_sha256':hashlib.sha256(ple.INVENTORY.read_bytes()).hexdigest(),
                      'cache_scope':'paired first and repeated windows without cache eviction',
                      'observations':rows},indent=2))

if __name__=='__main__':
    main()
