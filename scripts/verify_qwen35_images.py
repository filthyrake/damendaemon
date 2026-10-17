"""Compare native RGB preprocessing with trusted independent fixture pixels."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from damendaemon.vision_inputs import prepare_images


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('reference')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(4)
    reference=torch.load(args.reference,map_location='cpu',weights_only=True)
    report=dict(passed=False,torch=str(torch.__version__),cases=[],
        reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path('damendaemon/vision_inputs.py').read_bytes()).hexdigest())
    for case in reference['cases']:
        pixels,grids=prepare_images(case['images'])
        same_shape=pixels.shape==case['pixel_values'].shape
        difference=(pixels-case['pixel_values']).abs() if same_shape else None
        report['cases'].append(dict(name=case['name'],grids=grids,
            max_absolute=difference.max().item() if same_shape else None,
            passed=grids==case['grids'] and same_shape and torch.equal(pixels,case['pixel_values'])))
    report['passed']=all(case['passed'] for case in report['cases'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise AssertionError('independent preprocessing gate failed')


if __name__=='__main__':main()
