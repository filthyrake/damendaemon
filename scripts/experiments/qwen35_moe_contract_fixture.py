"""Regenerate the qwen3_5_moe contract fixture from a real checkpoint snapshot.

Metadata only: reads config.json, the safetensors index and every shard header;
no payload bytes are read. Run against the Qwen3.6-35B-A3B snapshot (or a
revision-matching copy) and commit the result as
tests/fixtures/qwen3_5_moe_contract.json together with any contract change.

  python scripts/experiments/qwen35_moe_contract_fixture.py /path/to/snapshot
"""
import argparse
import hashlib
import json
from pathlib import Path

from damendaemon.hybrid_moe_checkpoint import audit_names, tensor_specs


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', help='checkpoint directory with config.json and safetensors shards')
    parser.add_argument('--output', default='tests/fixtures/qwen3_5_moe_contract.json')
    args = parser.parse_args()
    base = Path(args.snapshot)
    config = json.loads((base / 'config.json').read_text())
    index = json.loads((base / 'model.safetensors.index.json').read_text())
    specs = tensor_specs(config)
    names = sorted(index['weight_map'])
    fingerprint = hashlib.sha256(('\n'.join(sorted(specs)) + '\n').encode()).hexdigest()
    inventory = audit_names(config, names)

    fixture = {
        'source': f'local snapshot at {base.resolve()}',
        'config_sha256': sha(base / 'config.json'),
        'index_sha256': sha(base / 'model.safetensors.index.json'),
        'shard_count': len(set(index['weight_map'].values())),
        'total_index_tensors': len(index['weight_map']),
        'config': config,
        'main_tensor_count': inventory['main_tensors'],
        'main_tensor_names_sha256': fingerprint,
        'excluded_mtp_count': len(inventory['excluded_mtp']),
        'excluded_vision_count': len(inventory['excluded_vision']),
        'scope': 'metadata-only contract: config/index/names pinned (the only files the contract reads), no payload reads, no execution',
    }
    output = Path(args.output)
    output.write_text(json.dumps(fixture, indent=1, sort_keys=True) + '\n')
    print('fixture written:', output.resolve())
    print('main tensors:', fixture['main_tensor_count'],
          'mtp excluded:', fixture['excluded_mtp_count'],
          'vision excluded:', fixture['excluded_vision_count'])
    print('config sha256:', fixture['config_sha256'])
    print('name fingerprint:', fingerprint)
    # The pinned test asserts the 16-char prefix of the audited revision;
    # update that assertion deliberately when the checkpoint revision changes.
    print('NOTE: update the sha256 prefix assertion in tests/test_hybrid_moe_checkpoint.py if this is a new revision.')


main()
