"""Create a disposable, visibly changed checkpoint without modifying its source.

The fixture preserves architecture/layout but changes projections on both stages
and final normalization. Independent reference generation remains a separate
step. This is a correctness fixture, not a new model-support qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import torch
from safetensors.torch import load_file,save_file


def digest(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source');parser.add_argument('destination')
    args=parser.parse_args();source=Path(args.source).resolve();destination=Path(args.destination).resolve()
    if destination.exists():parser.error('fixture destination must be new')
    destination.mkdir(parents=True)
    state=load_file(str(source/'model.safetensors'),device='cpu')
    changes=[]
    for name,scale,roll in (('model.layers.0.self_attn.q_proj.weight',.9,1),
                            ('model.layers.12.self_attn.o_proj.weight',.8,0),
                            ('model.norm.weight',.9,0)):
        before=state[name]
        after=before.float().roll(roll,0).mul(scale).to(before.dtype)
        changes.append(dict(name=name,scale=scale,row_roll=roll,before_sha256=digest(before),after_sha256=digest(after)))
        state[name]=after
    for path in source.iterdir():
        if path.is_file() and not path.name.endswith('.safetensors') and path.name!='model.safetensors.index.json':
            shutil.copyfile(path,destination/path.name)
    save_file(state,str(destination/'model.safetensors'),metadata={'format':'pt'})
    (destination/'damendaemon-test-fixture.json').write_text(json.dumps(dict(source=str(source),changes=changes),indent=2)+'\n')
    print(str(destination))


if __name__=='__main__':main()
