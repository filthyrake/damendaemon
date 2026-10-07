"""Run ordinary GLM text through the bounded local packed CPU reference."""
import argparse
from contextlib import closing
import json
from pathlib import Path
from dllm.checkpoint import _unique_fields
from dllm.glm_text import GlmText,_read
from dllm.glm_reference_loader import load_reference_model
from dllm.glm_generation import ReferenceGenerator,validate_text_config
from dllm.glm_batch_generation import BatchReferenceGenerator
from dllm.reference_batch import prepare_batch


def _positive(value):
    try:number=int(value)
    except ValueError:raise argparse.ArgumentTypeError('must be a positive integer') from None
    if not 1<=number<=2**20:raise argparse.ArgumentTypeError('must be between 1 and 1048576')
    return number


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt',help='one ordinary user text turn')
    source.add_argument('--messages',help='local JSON file containing ordinary text messages')
    source.add_argument('--batch-messages',help='local JSON list of 1..16 equal-token-length conversations')
    parser.add_argument('--reasoning-effort',choices=('low','high','max'),default='max')
    parser.add_argument('--clear-history-reasoning',action='store_true')
    parser.add_argument('--max-new-tokens',type=_positive,default=128)
    parser.add_argument('--context',type=_positive,default=4096)
    parser.add_argument('--weight-mib',type=_positive,default=256)
    parser.add_argument('--cache-mib',type=_positive,default=256)
    parser.add_argument('--projection-workspace-mib',type=_positive,default=8)
    parser.add_argument('--workspace-mib',type=_positive,default=64)
    parser.add_argument('--output-mib',type=_positive,default=64)
    args=parser.parse_args(argv)
    try:
        config=json.loads(_read(Path(args.snapshot)/'config.json',1024**2),object_pairs_hook=_unique_fields)
        if not isinstance(config,dict) or config.get('model_type')!='glm_moe_dsa':
            raise ValueError('GLM reference command requires a GLM sparse-attention snapshot')
        if args.max_new_tokens>4096:raise ValueError('GLM reference output is limited to 4096 tokens')
        if any(value>2048 for value in (args.cache_mib,args.projection_workspace_mib,args.workspace_mib,args.output_mib)):
            raise ValueError('GLM reference component budgets are limited to 2048 MiB')
        text=GlmText(args.snapshot,vocab_size=config.get('vocab_size'))
        validate_text_config(config,text,args.context)
        messages=([dict(role='user',content=args.prompt)] if args.prompt is not None
                  else json.loads(_read(args.batch_messages or args.messages,1024**2),object_pairs_hook=_unique_fields))
        options=dict(reasoning_effort=args.reasoning_effort,clear_thinking=args.clear_history_reasoning)
        if args.batch_messages:
            prepare_batch(text,messages,args.max_new_tokens,args.context,**options)
        else:
            tokens=text.chat_tokens(messages,**options)
            if len(tokens)+args.max_new_tokens>args.context:
                raise ValueError('GLM prompt and requested output exceed context before weight loading')
        model=load_reference_model(args.snapshot,max_weight_bytes=args.weight_mib*1024**2,
                                   projection_workspace_bytes=args.projection_workspace_mib*1024**2,
                                   max_workspace_bytes=args.workspace_mib*1024**2,max_output_bytes=args.output_mib*1024**2)
        runner_type=BatchReferenceGenerator if args.batch_messages else ReferenceGenerator
        runner=runner_type(model,text,max_context=args.context,max_cache_bytes=args.cache_mib*1024**2)
        print(json.dumps(dict(scope='cpu_reference',official_model_qualified=False,dtype='float32')),flush=True)
        primary=None
        try:
            generate=runner.generate_batch if args.batch_messages else runner.generate
            with closing(generate(messages,max_new_tokens=args.max_new_tokens,**options)) as generation:
                for event in generation:print(json.dumps(event,ensure_ascii=False),flush=True)
        except BaseException as error:primary=error;raise
        finally:
            try:runner.cleanup()
            except BaseException:
                if primary is None:raise
                primary.add_note('GLM command cleanup retry failed')
    except BrokenPipeError:return 0
    except (ValueError,OSError,RuntimeError,MemoryError) as error:parser.exit(2,f'GLM CPU reference failed: {error}\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
