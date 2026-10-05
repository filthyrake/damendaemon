"""Generate trusted independent CPU FP32 vision targets with pinned Transformers."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from PIL import Image
from safetensors import safe_open
import torch
import transformers
from transformers import AutoConfig, AutoImageProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--output',required=True)
    parser.add_argument('--report',required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=parser.parse_args()
    torch.set_num_threads(4)
    path=Path(args.model)
    config=AutoConfig.from_pretrained(path,local_files_only=True,trust_remote_code=False).vision_config
    config._attn_implementation='eager'
    reference=Qwen3_5VisionModel(config).float().eval()
    state={}
    for shard in sorted(path.glob('*.safetensors')):
        with safe_open(str(shard),framework='pt',device='cpu') as reader:
            for key in reader.keys():
                if key.startswith('model.visual.'):
                    name=key[len('model.visual.'):]
                    if name in state:raise ValueError('duplicate reference vision tensor')
                    state[name]=reader.get_tensor(key).float()
    reference.load_state_dict(state,strict=True)
    del state
    reference.to(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    processor=AutoImageProcessor.from_pretrained(path,local_files_only=True,trust_remote_code=False)
    current={}
    def output_hook(name):
        def capture(module,inputs,output):current[name]=output.detach().cpu().clone()
        return capture
    reference.patch_embed.register_forward_hook(output_hook('patch'))
    def positioned(module,inputs):current['positioned']=inputs[0].detach().cpu().clone()
    reference.blocks[0].register_forward_pre_hook(positioned)
    for i,block in enumerate(reference.blocks):block.register_forward_hook(output_hook(f'block.{i}'))
    def image(width,height,variant):
        y,x=np.mgrid[:height,:width]
        if variant==0:
            pixels=np.stack((x*255//max(width-1,1),y*255//max(height-1,1),(x+3*y)%256),axis=-1)
        else:
            pixels=np.stack((((x//17+y//13)%2)*255,(x*7+y*3)%256,(x*5-y*11)%256),axis=-1)
        return pixels.astype(np.uint8)
    specifications=[('square',[(256,256,0)]),('landscape',[(384,256,1)]),
                    ('resized',[(320,192,0)]),('larger',[(512,512,1)]),
                    ('two-images',[(256,256,0),(384,256,1)])]
    source=Path(inspect.getfile(Qwen3_5VisionModel))
    artifact=dict(model=str(path),torch=str(torch.__version__),transformers=transformers.__version__,
                  reference_device=args.device,reference_dtype='float32',reference_attention='eager',
                  reference_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  config=json.loads(config.to_json_string()),processor=json.loads(processor.to_json_string()),cases=[])
    report={key:value for key,value in artifact.items() if key!='cases'}
    report.update(completed=False,cases=[])
    try:
        for name,spec in specifications:
            arrays=[image(*entry) for entry in spec]
            inputs=processor(images=[Image.fromarray(array) for array in arrays],return_tensors='pt')
            current.clear()
            result=reference(inputs.pixel_values.float().to(args.device),inputs.image_grid_thw.to(args.device))
            current['merged']=result.pooler_output.detach().cpu().clone()
            if not all(bool(torch.isfinite(value).all()) for value in current.values()):
                raise AssertionError('nonfinite independent encoder output')
            artifact['cases'].append(dict(name=name,images=[torch.from_numpy(a.copy()) for a in arrays],
                pixel_values=inputs.pixel_values.float(),grids=inputs.image_grid_thw.tolist(),targets=dict(current)))
            summary=dict(name=name,grids=inputs.image_grid_thw.tolist(),output_shape=list(result.pooler_output.shape))
            report['cases'].append(summary)
            print(json.dumps(summary),flush=True)
        torch.save(artifact,args.output)
        report['artifact_sha256']=hashlib.sha256(Path(args.output).read_bytes()).hexdigest()
        report['completed']=True
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        Path(args.report).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
