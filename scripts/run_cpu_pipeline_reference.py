"""Run bounded raw completions from local Qwen3.5 text snapshots on CPU."""
import argparse
from contextlib import closing
import json
from pathlib import Path
from dllm.cpu_reference_text import CpuReferenceText,read_json
from dllm.cpu_hybrid_checkpoint import load_cpu_checkpoint
from dllm.cpu_pipeline_generation import CpuPipelineReferenceGenerator


def _positive(value):
    try:number=int(value)
    except ValueError:raise argparse.ArgumentTypeError('must be a positive integer') from None
    if not 1<=number<=2**20:raise argparse.ArgumentTypeError('must be between 1 and 1048576')
    return number


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot')
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prompt',help='raw text completion, without a chat template')
    source.add_argument('--tokens',help='local JSON list of prompt token IDs')
    parser.add_argument('--quantization',choices=('none','q4_0','q8_0'),default='none')
    parser.add_argument('--context',type=_positive,default=512)
    parser.add_argument('--max-new-tokens',type=_positive,default=128)
    parser.add_argument('--prefill-chunk',type=_positive,default=128)
    parser.add_argument('--stages',type=_positive,default=1)
    for name,default in (('stage-mib',512),('weight-mib',256),('cache-mib',256),('workspace-mib',64),
                         ('loader-workspace-mib',64),('projection-workspace-mib',8),('output-mib',64)):
        parser.add_argument('--'+name,type=_positive,default=default)
    args=parser.parse_args(argv)
    try:
        if args.stages>16 or any(getattr(args,name)>2048 for name in
                ('stage_mib','weight_mib','cache_mib','workspace_mib','loader_workspace_mib','projection_workspace_mib','output_mib')):
            raise ValueError('CPU reference supports at most 16 stages and 2048 MiB per budget')
        config=read_json(Path(args.snapshot)/'config.json',1024**2)
        if not isinstance(config,dict) or config.get('model_type')!='qwen3_5_text':raise ValueError('CPU completion requires Qwen3.5 text config')
        text=CpuReferenceText(args.snapshot,config)
        prompt=args.prompt if args.prompt is not None else read_json(args.tokens,1024**2)
        if args.tokens and not isinstance(prompt,list):raise ValueError('token file must contain a JSON list')
        tokens=text.tokens(prompt,max_context=args.context,max_new_tokens=args.max_new_tokens,chunk_size=args.prefill_chunk)
        model=load_cpu_checkpoint(args.snapshot,quantization=args.quantization,max_context=args.context,
                                 max_weight_bytes=args.weight_mib*1024**2,max_workspace_bytes=args.loader_workspace_mib*1024**2)
        runner=CpuPipelineReferenceGenerator(model,[args.stage_mib*1024**2]*args.stages,vocab_size=text.vocab_size,
                eos_ids=text.eos_ids,excluded_ids=text.excluded_ids,max_weight_bytes=args.weight_mib*1024**2,
                max_cache_bytes=args.cache_mib*1024**2,max_workspace_bytes=args.workspace_mib*1024**2,
                projection_workspace_bytes=args.projection_workspace_mib*1024**2,max_output_bytes=args.output_mib*1024**2)
        print(json.dumps(dict(scope='cpu_reference',official_model_qualified=False,quantization=args.quantization)),flush=True)
        generated=[];primary=None
        try:
            with closing(runner.generate(tokens,max_new_tokens=args.max_new_tokens,chunk_size=args.prefill_chunk)) as stream:
                for event in stream:
                    if 'token_id' in event:generated.append(event['token_id'])
                    else:event=dict(event,text=text.decode(generated))
                    print(json.dumps(event,ensure_ascii=False),flush=True)
        except BaseException as error:primary=error;raise
        finally:
            try:runner.cleanup()
            except BaseException:
                if primary is None:raise
                primary.add_note('CPU command cleanup retry failed')
    except BrokenPipeError:return 0
    except (ValueError,OSError,RuntimeError,MemoryError) as error:parser.exit(2,f'CPU reference failed: {error}\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
