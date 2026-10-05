"""Independent still-image position/rotary checks; no model weights required."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace, MethodType

import torch
import transformers
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model, Qwen3_5TextRotaryEmbedding
from dllm.vision_positions import image_positions, rotary_tables


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    config=AutoConfig.from_pretrained(args.model,local_files_only=True,trust_remote_code=False)
    reference=SimpleNamespace(config=config)
    reference.get_vision_position_ids=MethodType(Qwen3_5Model.get_vision_position_ids,reference)
    rotary=Qwen3_5TextRotaryEmbedding(config.text_config)
    report=dict(passed=False,transformers=transformers.__version__,torch=str(torch.__version__),cases=[],
                source_sha256=hashlib.sha256(Path('dllm/vision_positions.py').read_bytes()).hexdigest())
    fixtures=[[],[[1,16,16]],[[1,16,24]],[[1,14,22]],[[1,32,32]],
              [[1,16,16],[1,16,24]],[[1,2,2],[1,4,2]]]
    for grids in fixtures:
        tokens=[100,101,102]
        for grid in grids:
            tokens += [248053]+[248056]*(grid[1]*grid[2]//4)+[248054,200,201]
        ids=torch.tensor([tokens])
        types=(ids==248056).int()
        target,delta=Qwen3_5Model.get_rope_index(reference,ids,types,
            image_grid_thw=torch.tensor(grids) if grids else None)
        positions,offset=image_positions(tokens,grids)
        cos,sin=rotary_tables(positions)
        expected_cos,expected_sin=rotary(torch.zeros(1,len(tokens),1024),target)
        error=max((cos-expected_cos[0]).abs().max().item(),(sin-expected_sin[0]).abs().max().item())
        decode_positions=torch.arange(len(tokens),len(tokens)+8)+offset
        decode=decode_positions.expand(3,-1)
        dc,ds=rotary_tables(decode)
        ec,es=rotary(torch.zeros(1,8,1024),decode[:,None])
        decode_error=max((dc-ec[0]).abs().max().item(),(ds-es[0]).abs().max().item())
        report['cases'].append(dict(grids=grids,prompt_tokens=len(tokens),decode_offset=offset,
            rotary_max_error=error,decode_rotary_max_error=decode_error,
            passed=torch.equal(positions,target[:,0]) and offset==delta.item() and error==0 and decode_error==0))
    report['passed']=all(case['passed'] for case in report['cases'])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    if not report['passed']:raise AssertionError('independent multimodal position gate failed')


if __name__=='__main__':main()
