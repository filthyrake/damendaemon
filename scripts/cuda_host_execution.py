"""Explicit CUDA host command admission; never select an accelerator implicitly."""
import re
from scripts.serve_cpu_reference import bounded_integer


def add_arguments(parser):
    parser.add_argument('--device',required=True,help='explicit indexed NVIDIA device, e.g. cuda:0')
    for name,low,high,default in (('device-weight-mib',1,2048,256),('transfer-mib',1,2048,64),
            ('device-workspace-mib',1,2048,256),('allocator-mib',128,32768,512),('cpu-threads',1,16,1)):
        parser.add_argument('--'+name,type=bounded_integer(low,high),default=default)


def validate(args,*,family):
    if (family not in ('glm','flash','deepseek') or not getattr(args,'native_chat',False)
            or getattr(args,'flash_images',False)):
        raise ValueError('CUDA host command requires explicit native GLM/Flash/DeepSeek text/tool mode')
    if not isinstance(args.device,str) or re.fullmatch(r'cuda:[0-9]+',args.device) is None:
        raise ValueError('CUDA host command requires an indexed device')
    for name,low,high in (('device_weight_mib',1,2048),('transfer_mib',1,2048),
            ('device_workspace_mib',1,2048),('allocator_mib',128,32768),('cpu_threads',1,16)):
        value=getattr(args,name)
        if type(value) is not int or not low<=value<=high:raise ValueError('invalid CUDA host resource bound')
    if args.device_weight_mib>=args.allocator_mib:
        raise ValueError('CUDA allocator cap must leave room beyond raw stage weights')


def configure(args):
    import torch
    device=torch.device(args.device)
    if torch.version.hip or not torch.cuda.is_available() or device.index>=torch.cuda.device_count():
        raise ValueError('requested NVIDIA CUDA host device is unavailable')
    total=torch.cuda.get_device_properties(device).total_memory;budget=args.allocator_mib*1024**2
    if budget>total:raise ValueError('CUDA allocator cap exceeds physical device capacity')
    torch.cuda.set_device(device);torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(budget/total,device)


def build_options(args):
    return dict(device=args.device,max_device_bytes=args.device_weight_mib*1024**2,
        max_transfer_bytes=args.transfer_mib*1024**2,device_workspace_bytes=args.device_workspace_mib*1024**2)
