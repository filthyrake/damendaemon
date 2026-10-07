"""Run bounded inline still-image messages through the explicit Flash CPU reference."""
import argparse
from contextlib import closing
import json
from pathlib import Path
from dllm.checkpoint import _unique_fields
from dllm.glm_text import _read
from dllm.glm_generation import validate_text_config
from dllm.glm_flash_generation import flash_text_config
from dllm.glm_flash_image_text import FlashImageText
from dllm.glm_flash_inline_images import decode_inline_images
from dllm.glm_flash_images import plan_images,validate_processor
from dllm.glm_flash_vision import vision_specs,audit_vision,ROOT
from dllm.glm_flash_reference_loader import load_reference_model
from dllm.glm_flash_vision_loader import load_reference_vision
from dllm.glm_flash_image_generation import FlashImageReferenceGenerator
from scripts.run_glm_flash_reference import _positive


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot');parser.add_argument('--messages',required=True,help='local JSON with inline PNG/JPEG image_url parts')
    parser.add_argument('--context',type=_positive,default=4096);parser.add_argument('--max-new-tokens',type=_positive,default=128)
    parser.add_argument('--reasoning-effort',choices=('low','high','max'),default='max')
    parser.add_argument('--clear-history-reasoning',action='store_true')
    for name,default in (('text-weight',256),('vision-weight',256),('cache',256),('projection-workspace',8),
                         ('workspace',64),('output',64),('image-workspace',64),('image-output',64)):
        parser.add_argument('--'+name+'-mib',type=_positive,default=default)
    parser.add_argument('--max-patches',type=_positive,default=4096)
    args=parser.parse_args(argv)
    try:
        byte_options={name:value*1024**2 for name,value in vars(args).items() if name.endswith('_mib')}
        if any(value>2**31 for value in byte_options.values()) or args.max_new_tokens>4096 or args.max_patches>65536:
            raise ValueError('Flash image reference exceeds component/output limits')
        config=json.loads(_read(Path(args.snapshot)/'config.json',1024**2),object_pairs_hook=_unique_fields)
        text_config=flash_text_config(config);specs=vision_specs(config.get('vision_config'));audit_vision(config,[ROOT+n for n in specs])
        vc=config['vision_config']
        if (vc['in_channels'],vc['patch_size'],vc['temporal_patch_size'],vc['spatial_merge_size'])!=(3,14,2,2):
            raise ValueError('Flash image command requires the pinned still-image patch geometry')
        processor=json.loads(_read(Path(args.snapshot)/'processor_config.json',1024**2),object_pairs_hook=_unique_fields)
        if not isinstance(processor,dict) or processor.get('processor_class')!='Glm5NextProcessor':
            raise ValueError('Flash image command requires the pinned processor class')
        validate_processor(processor.get('image_processor'))
        text=FlashImageText(args.snapshot,vocab_size=text_config.get('vocab_size'));validate_text_config(text_config,text,args.context)
        if config.get('image_token_id')!=text.image_token_id:raise ValueError('Flash snapshot image token disagrees with tokenizer')
        messages=json.loads(_read(args.messages,16*1024**2),object_pairs_hook=_unique_fields)
        decoded=decode_inline_images(messages)
        if not decoded.images:raise ValueError('Flash image command requires at least one inline image')
        plan=plan_images(decoded.images,max_patches=args.max_patches,max_workspace_bytes=byte_options['image_workspace_mib'],
                         max_output_bytes=byte_options['image_output_mib'])
        options=dict(reasoning_effort=args.reasoning_effort,clear_thinking=args.clear_history_reasoning)
        tokens=text.image_chat_tokens(decoded.messages,plan['grids'],**options)
        if len(tokens)+args.max_new_tokens>args.context:raise ValueError('Flash image prompt and output exceed context before weight loading')
        model=load_reference_model(args.snapshot,max_weight_bytes=byte_options['text_weight_mib'],
                                   projection_workspace_bytes=byte_options['projection_workspace_mib'],
                                   max_workspace_bytes=byte_options['workspace_mib'],max_output_bytes=byte_options['output_mib'])
        vision=load_reference_vision(args.snapshot,max_weight_bytes=byte_options['vision_weight_mib'],max_tokens=args.max_patches,
                                     max_workspace_bytes=byte_options['workspace_mib'],max_output_bytes=byte_options['output_mib'])
        runner=FlashImageReferenceGenerator(model,text,vision,max_context=args.context,max_cache_bytes=byte_options['cache_mib'],
                                            image_workspace_bytes=byte_options['image_workspace_mib'],image_output_bytes=byte_options['image_output_mib'])
        print(json.dumps(dict(scope='cpu_reference',modality='still_image',official_model_qualified=False,
                              dtype='float32',images=len(decoded.images))),flush=True)
        primary=None
        try:
            with closing(runner.generate_images(decoded.messages,decoded.images,max_new_tokens=args.max_new_tokens,**options)) as generation:
                for event in generation:print(json.dumps(event,ensure_ascii=False),flush=True)
        except BaseException as error:primary=error;raise
        finally:
            try:runner.cleanup()
            except BaseException:
                if primary is None:raise
                primary.add_note('Flash image command cleanup retry failed')
    except BrokenPipeError:return 0
    except (ValueError,OSError,RuntimeError,MemoryError) as error:parser.exit(2,f'Flash CPU image reference failed: {error}\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
