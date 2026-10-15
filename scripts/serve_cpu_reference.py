"""Serve an explicit local modern CPU reference snapshot on loopback."""
import argparse
import json
from pathlib import Path
import signal
import threading

from dllm.checkpoint import _unique_fields
from dllm.glm_text import _read
from dllm.http_auth import AuthPolicy
from dllm.reference_serving import ReferenceRuntime,handler_for_reference
from dllm.server import BoundedHTTPServer


def bounded_integer(low,high):
    def parse(value):
        try:result=int(value)
        except ValueError:raise argparse.ArgumentTypeError('must be an integer') from None
        if not low<=result<=high:raise argparse.ArgumentTypeError(f'must be between {low} and {high}')
        return result
    return parse


def parser_for_command():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('family',choices=('glm','flash','deepseek'))
    parser.add_argument('snapshot',help='existing local snapshot directory; never downloaded')
    parser.add_argument('--layer-splits',type=bounded_integer(1,127),nargs='+',help='explicit CPU pipeline cuts; GLM cuts preserve indexer groups')
    parser.add_argument('--alias',default='model')
    parser.add_argument('--port',type=bounded_integer(0,65535),default=8088)
    parser.add_argument('--connections',type=bounded_integer(1,64),default=16)
    parser.add_argument('--queue-size',type=bounded_integer(0,64),default=0,help='0 preserves immediate busy rejection; otherwise bounded serialized admission')
    parser.add_argument('--queue-timeout',type=float,default=120.)
    parser.add_argument('--context',type=bounded_integer(1,2**20),default=4096)
    parser.add_argument('--weight-mib',type=bounded_integer(1,2**20),default=256)
    parser.add_argument('--cache-mib',type=bounded_integer(1,2048),default=256)
    parser.add_argument('--projection-workspace-mib',type=bounded_integer(1,2048),default=8)
    parser.add_argument('--workspace-mib',type=bounded_integer(1,2048),help='GLM/Flash attention scratch; default 64 MiB')
    parser.add_argument('--output-mib',type=bounded_integer(1,2048),help='GLM/Flash or DeepSeek pipeline output scratch; default 64 MiB')
    parser.add_argument('--flash-images',action='store_true',help='explicit Flash still-image/tool HTTP owner')
    for name,maximum in (('vision-weight-mib',2048),('image-workspace-mib',2048),('image-output-mib',2048),('max-patches',65536)):
        parser.add_argument('--'+name,type=bounded_integer(1,maximum),help='requires --flash-images')
    return parser


def image_metadata(config,args):
    from dllm.glm_flash_vision import vision_specs,audit_vision,ROOT
    from dllm.glm_flash_images import validate_processor
    specs=vision_specs(config.get('vision_config'));audit_vision(config,[ROOT+n for n in specs])
    vc=config['vision_config']
    if (vc['in_channels'],vc['patch_size'],vc['temporal_patch_size'],vc['spatial_merge_size'])!=(3,14,2,2):
        raise ValueError('Flash image HTTP requires the pinned still-image patch geometry')
    processor=json.loads(_read(Path(args.snapshot)/'processor_config.json',1024**2),object_pairs_hook=_unique_fields)
    if not isinstance(processor,dict) or processor.get('processor_class')!='Glm5NextProcessor':
        raise ValueError('Flash image HTTP requires the pinned processor class')
    validate_processor(processor.get('image_processor'))


def prepare(args):
    """Validate policy and text metadata before any tensor payload is loaded."""
    auth=AuthPolicy.from_env()
    if not 0<args.queue_timeout<=300 or (not args.queue_size and args.queue_timeout!=120):
        raise ValueError('queue timeout requires an enabled queue and a bound within 300 seconds')
    if not isinstance(args.alias,str) or not 1<=len(args.alias)<=128:raise ValueError('invalid model alias')
    if args.flash_images and args.family!='flash':raise ValueError('still-image HTTP requires explicit flash family')
    if not args.flash_images and any(getattr(args,key) is not None for key in
            ('vision_weight_mib','image_workspace_mib','image_output_mib','max_patches')):
        raise ValueError('image budgets require --flash-images')
    if args.family=='deepseek' and (args.workspace_mib is not None or (args.output_mib is not None and args.layer_splits is None)):
        raise ValueError('DeepSeek uses projection-workspace-mib; output-mib requires a pipeline')
    config=json.loads(_read(Path(args.snapshot)/'config.json',1024**2),object_pairs_hook=_unique_fields)
    expected={'glm':'glm_moe_dsa','flash':'glm5_next','deepseek':'deepseek_v4'}[args.family]
    if not isinstance(config,dict) or config.get('model_type')!=expected:raise ValueError('snapshot does not match explicit family')
    if args.family=='deepseek':
        from dllm.deepseek_text import DeepSeekText as Text
        from dllm.deepseek_generation import ReferenceGenerator as Runner,validate_text_config
        from dllm.deepseek_reference_loader import load_reference_model
        text_config=config
        options=dict(workspace_bytes=args.projection_workspace_mib*1024**2)
        if args.layer_splits is not None:options['max_output_bytes']=(args.output_mib or 64)*1024**2
    else:
        from dllm.glm_generation import validate_text_config
        if args.family=='glm':
            from dllm.glm_tool_text import GlmToolText as Text
            from dllm.glm_tool_generation import GlmToolReferenceGenerator as Runner
            from dllm.glm_reference_loader import load_reference_model
            text_config=config
        else:
            from dllm.glm_flash_tool_text import FlashToolText as Text
            from dllm.glm_flash_tool_generation import FlashToolReferenceGenerator as Runner
            from dllm.glm_flash_reference_loader import load_reference_model
            from dllm.glm_flash_generation import flash_text_config
            text_config=flash_text_config(config)
            if args.flash_images:
                image_metadata(config,args)
                from dllm.glm_flash_image_tool_text import FlashImageToolText as Text
                from dllm.glm_flash_image_tool_generation import FlashImageToolReferenceGenerator as Runner
        options=dict(projection_workspace_bytes=args.projection_workspace_mib*1024**2,
            max_workspace_bytes=(args.workspace_mib or 64)*1024**2,max_output_bytes=(args.output_mib or 64)*1024**2)
    text=Text(args.snapshot,vocab_size=text_config.get('vocab_size'))
    validate_text_config(text_config,text,args.context)
    if args.flash_images and config.get('image_token_id')!=text.image_token_id:
        raise ValueError('Flash image token disagrees with tokenizer')
    if args.layer_splits is not None:
        if args.family=='deepseek':
            from dllm.deepseek_checkpoint_pipeline import load_reference_pipeline
            from dllm.deepseek_pipeline_generation import DeepSeekPipelineReferenceGenerator as Runner
        elif args.family=='glm':
            from dllm.glm_checkpoint_pipeline import load_reference_pipeline
            from dllm.glm_pipeline_generation import PipelineToolReferenceGenerator as Runner
        else:
            from dllm.glm_flash_checkpoint_pipeline import load_reference_pipeline
            from dllm.glm_flash_pipeline_generation import FlashPipelineToolReferenceGenerator as Runner
            if args.flash_images:
                from dllm.glm_flash_pipeline_image_generation import FlashPipelineImageToolReferenceGenerator as Runner
        model=load_reference_pipeline(args.snapshot,args.layer_splits,max_weight_bytes=args.weight_mib*1024**2,**options)
    else:model=load_reference_model(args.snapshot,max_weight_bytes=args.weight_mib*1024**2,**options)
    try:
        image_options={}
        if args.flash_images:
            from dllm.glm_flash_vision_loader import load_reference_vision
            vision=load_reference_vision(args.snapshot,max_weight_bytes=(args.vision_weight_mib or 256)*1024**2,
                max_tokens=args.max_patches or 4096,max_workspace_bytes=(args.workspace_mib or 64)*1024**2,
                max_output_bytes=(args.output_mib or 64)*1024**2)
            image_options=dict(vision=vision,image_workspace_bytes=(args.image_workspace_mib or 64)*1024**2,
                image_output_bytes=(args.image_output_mib or 64)*1024**2)
        runner=Runner(model,text,max_context=args.context,max_cache_bytes=args.cache_mib*1024**2,**image_options)
        if args.queue_size:
            from dllm.queued_reference_serving import QueuedReferenceRuntime
            runtime=QueuedReferenceRuntime(runner,args.alias,queue_size=args.queue_size,queue_timeout=args.queue_timeout)
        else:runtime=ReferenceRuntime(runner,args.alias)
    except BaseException as error:
        try:
            cleanup=getattr(model,'cleanup',None) or model.cleanup_failed_prefills
            cleanup()
        except BaseException:error.add_note('CPU model preparation cleanup failed')
        raise
    return runtime,auth


def serve(runtime,auth,args):
    """Signals stop admission; request threads finish cleanup before owner close.

    Shutdown is cooperative between CPU kernels, not a kernel execution deadline.
    Non-daemon request threads retain the owner until their iterators exit.
    """
    server=None;previous={};primary=None;stopping=threading.Event()
    def interrupt(*_):stopping.set();runtime.stopping.set()
    try:
        server=BoundedHTTPServer(('127.0.0.1',args.port),handler_for_reference(runtime,auth),args.connections)
        # Own accepted requests through their final admission release. The base
        # ThreadingHTTPServer defaults to daemon threads and skips that join.
        server.daemon_threads=False
        server.block_on_close=True
        server.timeout=.2
        if threading.current_thread() is threading.main_thread():
            for sig in (signal.SIGTERM,signal.SIGINT):previous[sig]=signal.signal(sig,interrupt)
        print(json.dumps(dict(event='http_ready',url=f'http://127.0.0.1:{server.server_port}',
            model=args.alias,device=runtime.manage('status')['device'],model_type=runtime.family,modality='still_image' if runtime.images else 'text',scope=runtime.scope,dtype='float32',
            official_model_qualified=False,queue_enabled=bool(args.queue_size),queue_capacity=args.queue_size,**runtime.execution_status())),flush=True)
        while not stopping.is_set():server.handle_request()
    except BaseException as error:primary=error;raise
    finally:
        runtime.stopping.set()
        errors=[]
        if server is not None:
            try:server.server_close()
            except BaseException as error:errors.append(error)
        try:runtime.close()
        except BaseException as error:errors.append(error)
        for sig,handler in previous.items():signal.signal(sig,handler)
        if errors:
            if primary is not None:primary.add_note('CPU reference HTTP shutdown cleanup failed')
            else:raise errors[0]
    print(json.dumps(dict(event='stopped',scope=runtime.scope)),flush=True)


def main(argv=None):
    parser=parser_for_command();args=parser.parse_args(argv)
    try:
        runtime,auth=prepare(args)
        serve(runtime,auth,args)
    except KeyboardInterrupt:return 130
    except (ValueError,TypeError,OSError,RuntimeError,MemoryError) as error:
        parser.exit(2,f'CPU reference HTTP failed: {error}\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
