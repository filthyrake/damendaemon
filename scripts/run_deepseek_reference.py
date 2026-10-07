"""Run the bounded local CPU reference; not a GPU benchmark or serving backend."""
import argparse
from contextlib import closing
import json

import torch

from dllm.checkpoint import _unique_fields
from dllm.deepseek_text import DeepSeekText, _read
from dllm.deepseek_reference_loader import load_reference_model
from dllm.deepseek_generation import ReferenceGenerator, validate_text_config
from dllm.deepseek_batch_generation import BatchReferenceGenerator
from dllm.reference_batch import prepare_batch


def _positive(value):
    try:number=int(value)
    except ValueError:raise argparse.ArgumentTypeError('must be a positive integer') from None
    if not 1 <= number <= 2**20:raise argparse.ArgumentTypeError('must be between 1 and 1048576')
    return number


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt',help='one ordinary user text turn')
    source.add_argument('--messages',help='local JSON file containing ordinary text messages')
    source.add_argument('--batch-messages',help='local JSON list of 1..16 equal-token-length conversations')
    parser.add_argument('--tools',help='local JSON file containing function schemas; calls are returned, never executed')
    parser.add_argument('--thinking',action='store_true')
    parser.add_argument('--keep-history-reasoning',action='store_true')
    parser.add_argument('--max-new-tokens',type=_positive,default=128)
    parser.add_argument('--context',type=_positive,default=4096)
    parser.add_argument('--weight-mib',type=_positive,default=256)
    parser.add_argument('--cache-mib',type=_positive,default=256)
    parser.add_argument('--workspace-mib',type=_positive,default=8)
    parser.add_argument('--dtype',choices=('float32','bfloat16','float16'),default='float32')
    args=parser.parse_args(argv)
    try:
        if args.batch_messages and args.tools:
            raise ValueError('DeepSeek reference batches do not yet support tool calls')
        from pathlib import Path
        config=json.loads(_read(Path(args.snapshot)/'config.json',1024*1024),object_pairs_hook=_unique_fields)
        if not isinstance(config,dict) or config.get('model_type') != 'deepseek_v4':
            raise ValueError('reference command requires a DeepSeek V4 snapshot')
        if args.max_new_tokens > 4096:raise ValueError('reference output length is limited to 4096 tokens')
        limit=config.get('max_position_embeddings',2**20)
        if type(limit) is not int or args.context > limit:raise ValueError('context exceeds model position limit')
        text=DeepSeekText(args.snapshot,vocab_size=config.get('vocab_size'))
        validate_text_config(config,text,args.context)
        messages=([{'role':'user','content':args.prompt}] if args.prompt is not None
                  else json.loads(_read(args.batch_messages or args.messages,1024*1024),object_pairs_hook=_unique_fields))
        options=dict(enable_thinking=args.thinking,drop_thinking=not args.keep_history_reasoning)
        if args.tools is not None:
            options['tools']=json.loads(_read(args.tools,1024*1024),object_pairs_hook=_unique_fields)
            if not isinstance(options['tools'],list):raise ValueError('tools file must contain a function schema list')
        if args.batch_messages:
            prepare_batch(text,messages,args.max_new_tokens,args.context,**options)
        else:
            tokens=text.chat_tokens(messages,**options)
            if len(tokens)+args.max_new_tokens > args.context:
                raise ValueError('prompt and requested output exceed context before weight loading')
        model=load_reference_model(args.snapshot,max_weight_bytes=args.weight_mib*1024**2,
                                   workspace_bytes=args.workspace_mib*1024**2,dtype=getattr(torch,args.dtype))
        runner_type=BatchReferenceGenerator if args.batch_messages else ReferenceGenerator
        runner=runner_type(model,text,max_context=args.context,max_cache_bytes=args.cache_mib*1024**2)
        print(json.dumps({'scope':'cpu_reference','official_model_qualified':False,'dtype':args.dtype}),flush=True)
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
                primary.add_note('DeepSeek command cleanup retry failed')
    except BrokenPipeError:
        return 0
    except (ValueError,OSError,RuntimeError,MemoryError) as error:
        parser.exit(2,f'CPU reference failed: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
