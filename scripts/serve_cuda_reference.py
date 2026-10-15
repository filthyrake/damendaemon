"""Serve an explicitly admitted local GLM/Flash/DeepSeek CUDA reference snapshot on loopback."""
import argparse
import json
from pathlib import Path
import re
from dllm.checkpoint import _unique_fields
from dllm.glm_text import _read
from dllm.glm_generation import validate_text_config
from dllm.glm_tool_text import GlmToolText
from dllm.http_auth import AuthPolicy
from scripts.serve_cpu_reference import bounded_integer,serve


def parser_for_command():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot',help='local GLM, Flash or DeepSeek snapshot only; never downloaded')
    parser.add_argument('--device',required=True,help='explicit indexed NVIDIA CUDA device, e.g. cuda:0')
    parser.add_argument('--alias',default='model')
    parser.add_argument('--port',type=bounded_integer(0,65535),default=8089)
    parser.add_argument('--connections',type=bounded_integer(1,64),default=16)
    parser.add_argument('--context',type=bounded_integer(1,2**20),default=128)
    parser.add_argument('--weight-mib',type=bounded_integer(1,2**20),default=256)
    parser.add_argument('--device-weight-mib',type=bounded_integer(1,32768),default=256)
    parser.add_argument('--allocator-mib',type=bounded_integer(128,32768),default=512)
    parser.add_argument('--cache-mib',type=bounded_integer(1,2048),default=64)
    parser.add_argument('--projection-workspace-mib',type=bounded_integer(1,2048),default=8)
    parser.add_argument('--workspace-mib',type=bounded_integer(1,2048),default=64)
    parser.add_argument('--output-mib',type=bounded_integer(1,2048),default=64)
    parser.add_argument('--cpu-threads',type=bounded_integer(1,16),default=1)
    parser.set_defaults(queue_size=0)
    return parser


def prepare(args):
    auth=AuthPolicy.from_env()
    if not re.fullmatch(r'cuda:[0-9]+',args.device):raise ValueError('choose an explicit indexed CUDA device')
    if not isinstance(args.alias,str) or not 1<=len(args.alias)<=128:raise ValueError('invalid model alias')
    if args.device_weight_mib>=args.allocator_mib:raise ValueError('allocator cap must leave space beyond packed weight payload')
    path=Path(args.snapshot)
    config=json.loads(_read(path/'config.json',1024**2),object_pairs_hook=_unique_fields)
    if not isinstance(config,dict):raise ValueError('CUDA reference command requires a GLM, Flash or DeepSeek snapshot')
    import torch
    if config.get('model_type')=='glm_moe_dsa':
        from dllm.glm_reference_loader import load_reference_model
        from dllm.cuda_glm_placement import place_cuda_glm as place
        from dllm.cuda_glm_generation import CudaGlmToolGenerator as generator_type
        from dllm.cuda_reference_serving import CudaGlmReferenceRuntime as runtime_type
        text=GlmToolText(path,vocab_size=config.get('vocab_size'))
        text_config=config
    elif config.get('model_type')=='glm5_next':
        from dllm.glm_flash_generation import flash_text_config
        from dllm.glm_flash_tool_text import FlashToolText
        from dllm.glm_flash_reference_loader import load_reference_model
        from dllm.cuda_glm_flash_placement import place_cuda_flash as place
        from dllm.cuda_glm_flash_generation import CudaFlashToolGenerator as generator_type
        from dllm.cuda_reference_serving import CudaFlashReferenceRuntime as runtime_type
        text_config=flash_text_config(config)
        text=FlashToolText(path,vocab_size=text_config.get('vocab_size'))
    elif config.get('model_type')=='deepseek_v4':
        from dllm.deepseek_reference_loader import load_reference_model
        from dllm.deepseek_text import DeepSeekText
        from dllm.deepseek_generation import validate_text_config as validate_deepseek_text
        from dllm.cuda_deepseek_placement import place_cuda_deepseek as place
        from dllm.cuda_deepseek_generation import CudaDeepSeekGenerator as generator_type
        from dllm.cuda_reference_serving import CudaDeepSeekReferenceRuntime as runtime_type
        text=DeepSeekText(path,vocab_size=config.get('vocab_size'))
        text_config=config
        validate_deepseek_text(config,text,args.context)
    else:raise ValueError('CUDA reference command requires a GLM, Flash or DeepSeek snapshot')
    if config.get('model_type')!='deepseek_v4':validate_text_config(text_config,text,args.context)
    device=torch.device(args.device)
    if torch.version.hip or not torch.cuda.is_available() or device.index>=torch.cuda.device_count():
        raise ValueError('selected NVIDIA CUDA device is unavailable')
    total=torch.cuda.get_device_properties(device).total_memory
    allocator=args.allocator_mib*1024**2
    if allocator>total:raise ValueError('allocator cap exceeds device capacity')
    torch.cuda.set_device(device)
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(allocator/total,device)
    if config.get('model_type')=='deepseek_v4':
        cpu=load_reference_model(path,max_weight_bytes=args.weight_mib*1024**2,
            workspace_bytes=args.projection_workspace_mib*1024**2)
        placed=place(cpu,device=device,max_bytes=args.device_weight_mib*1024**2,
            workspace_bytes=args.workspace_mib*1024**2,max_output_bytes=args.output_mib*1024**2)
    else:
        cpu=load_reference_model(path,max_weight_bytes=args.weight_mib*1024**2,
            projection_workspace_bytes=args.projection_workspace_mib*1024**2,
            max_workspace_bytes=args.workspace_mib*1024**2,max_output_bytes=args.output_mib*1024**2)
        placed=place(cpu,device=device,max_bytes=args.device_weight_mib*1024**2)
    model=placed.model
    try:
        runner=generator_type(model,text,max_context=args.context,max_cache_bytes=args.cache_mib*1024**2)
        runtime=runtime_type(runner,args.alias)
    except BaseException:
        if config.get('model_type')=='deepseek_v4':model.cleanup_failed_prefills()
        else:model.cleanup()
        raise
    return runtime,auth


def main(argv=None):
    parser=parser_for_command();args=parser.parse_args(argv)
    try:
        runtime,auth=prepare(args)
        serve(runtime,auth,args)
    except KeyboardInterrupt:return 130
    except (ValueError,TypeError,OSError,RuntimeError,MemoryError) as error:
        parser.exit(2,f'CUDA reference HTTP failed: {error}\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
