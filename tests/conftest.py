"""Opt-in complete shape/long matrices and reproducible process-level sharding."""
import hashlib
import json
from pathlib import Path

import pytest
import torch


def pytest_addoption(parser):
    parser.addoption('--full-matrix', action='store_true', help='Include every declared D/V bucket.')
    parser.addoption('--long', action='store_true', help='Include long-sequence FP64 sampled-row checks.')
    parser.addoption('--shard-count', type=int, default=1)
    parser.addoption('--shard-index', type=int, default=0)
    parser.addoption('--inventory-output', help='Write selected node IDs and canonical coverage hash.')


def pytest_configure(config):
    config.addinivalue_line('markers', 'full_matrix: complete D/V correctness domain')
    config.addinivalue_line('markers', 'long: long-sequence numerical coverage')
    torch.set_num_threads(1)
    count, index = config.getoption('--shard-count'), config.getoption('--shard-index')
    if count < 1 or not 0 <= index < count:
        raise pytest.UsageError('Require shard-count > 0 and 0 <= shard-index < shard-count')


def pytest_collection_modifyitems(config, items):
    count, index = config.getoption('--shard-count'), config.getoption('--shard-index')
    eligible, selected, deselected = [], [], []
    for item in items:
        if (item.get_closest_marker('full_matrix') and not config.getoption('--full-matrix')) or (
                item.get_closest_marker('long') and not config.getoption('--long')):
            deselected.append(item)
            continue
        eligible.append(item.nodeid)
        if int.from_bytes(hashlib.sha256(item.nodeid.encode()).digest()[:8], 'big') % count == index:
            selected.append(item)
        else:
            deselected.append(item)
    items[:] = selected
    config.hook.pytest_deselected(items=deselected)
    output = config.getoption('--inventory-output')
    if output:
        canonical = sorted(eligible)
        Path(output).write_text(json.dumps({
            'shard_count': count, 'shard_index': index,
            'coverage_sha256': hashlib.sha256(json.dumps(canonical, separators=(',', ':')).encode()).hexdigest(),
            'canonical_nodeids': canonical, 'selected_nodeids': [item.nodeid for item in selected],
        }, indent=2) + '\n')
