"""Alternating real HTTP comparison for bounded OLMoE tiny-prefill dispatch."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model-metadata',default='model-olmoe.json')
    parser.add_argument('--baseline-source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--label',default='olmoe-tiny')
    args=parser.parse_args()
    if not args.label.replace('-','').replace('_','').isalnum():
        parser.error('invalid label')
    prompt=Path(args.output).parent/(args.label+'-prompt.json')
    prompt.write_text('[40]\n')
    phases=(('control','b1'),('candidate','a1'),('candidate','a2'),('control','b2'))
    report=dict(passed=False,scope='one input token, 32 output tokens; normal scheduler; alternating complete HTTP requests',
                quantization='q8_0',model=json.loads(Path(args.model_metadata).read_text()),
                baseline_sha256=hashlib.sha256(Path(args.baseline_source).read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(Path('dllm/moe.py').read_bytes()).hexdigest(),
                phases=[],comparisons=[])
    documents={}
    try:
        for kind,iteration in phases:
            label=f'{args.label}-{kind}-{iteration}'
            command=[sys.executable,'scripts/compare_a2.py','--engines','dllm',
                     '--model-metadata',args.model_metadata,'--label',label,'--context','2048',
                     '--host-gib','16','--max-tokens','32','--prompt-token-ids',str(prompt)]
            if kind=='control':command+=['--experimental-olmoe-baseline',args.baseline_source]
            subprocess.run(command,check=True,timeout=900)
            files=[]
            for concurrency in (1,2,4):
                path=Path(f'compare-dllm-{label}-c{concurrency}.json')
                documents[(kind,iteration,concurrency)]=json.loads(path.read_text())
                files.append(str(path))
            report['phases'].append(dict(kind=kind,iteration=iteration,files=files,command=command))
        for concurrency in (1,2,4):
            comparison=dict(concurrency=concurrency)
            expected=None
            for kind in ('control','candidate'):
                trials=[];rounds=[];per_phase=[]
                for phase,iteration in phases:
                    if phase!=kind:continue
                    document=documents[(kind,iteration,concurrency)]
                    current=[t for t in document['trials'] if not t['warmup']]
                    for trial in current:
                        signature=(trial['text'],trial['usage'])
                        if expected is None:expected=signature
                        if signature!=expected:raise AssertionError('HTTP response or usage differs')
                    trials.extend(current)
                    rounds.extend(r for r in document['rounds'] if not r['warmup'])
                    per_phase.append(dict(iteration=iteration,
                        median_total_seconds=statistics.median(t['total_seconds'] for t in current),
                        median_first_output_seconds=statistics.median(t['first_output_seconds'] for t in current)))
                comparison[kind]=dict(median_total_seconds=statistics.median(t['total_seconds'] for t in trials),
                    median_first_output_seconds=statistics.median(t['first_output_seconds'] for t in trials),
                    median_aggregate_tokens_per_second=statistics.median(r['aggregate_output_tokens_per_second'] for r in rounds),
                    phases=per_phase)
            comparison['responses_and_usage_equal']=True
            report['comparisons'].append(comparison)
        report['passed']=True
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
