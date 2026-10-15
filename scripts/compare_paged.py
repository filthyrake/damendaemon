"""Counterbalanced fixed/paged HTTP comparison using the existing A2 runner."""
from dllm.evidence_environment import report_environment
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model-metadata',default='model-a.json')
    p.add_argument('--label',required=True)
    p.add_argument('--context',type=int,default=2048)
    p.add_argument('--pool-tokens',type=int,default=2048)
    args=p.parse_args()
    if not args.label.replace('-','').replace('_','').isalnum():
        p.error('invalid label')
    rows=[]
    for index,mode in enumerate(('fixed','paged','paged','fixed')):
        label=f'{args.label}-{index}-{mode}'
        env=dict(os.environ)
        if mode=='paged':env['DLLM_KV_CACHE_TOKENS']=str(args.pool_tokens)
        else:env.pop('DLLM_KV_CACHE_TOKENS',None)
        subprocess.run([sys.executable,'scripts/compare_a2.py','--model-metadata',args.model_metadata,
            '--label',label,'--context',str(args.context),'--engines','dllm'],env=env,check=True,stdout=sys.stderr)
        for concurrency in (1,2,4):
            path=Path(f'compare-dllm-{label}-c{concurrency}.json')
            data=json.loads(path.read_text())
            rows.append({'order':index,'layout':mode,'concurrency':concurrency,'evidence':str(path),
                'median_tokens_per_second':statistics.median(r['aggregate_output_tokens_per_second'] for r in data['rounds'] if not r['warmup']),
                'median_ttft_ms':statistics.median(r['first_output_seconds']*1000 for r in data['trials'] if not r['warmup'])})
    print(json.dumps({'context':args.context,'pool_tokens':args.pool_tokens,'order':['fixed','paged','paged','fixed'],
        'runtime_environment':report_environment(os.environ),'measurements':rows},indent=2))


if __name__=='__main__':main()
