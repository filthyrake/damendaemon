"""Changed-weight reference, stable graph storage, and partial-copy failure proof."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
from pathlib import Path
from unittest.mock import patch
import torch
from damendaemon.runtime import Runtime


def storage(executor):
    return [dict(weights={k:v.data_ptr() for k,v in stage.w.items()},kv=[v.data_ptr() for v in stage.kv],
        experimental_adjacent={str(k):v.data_ptr() for k,v in getattr(stage.quant_linear,'experimental_adjacent',{}).items()},
        stage_adjacent_ids=sorted(stage.q4_linear.weights) if stage.q4_linear is not None else [],
        experimental_adjacent_inplace_ids=sorted(getattr(stage.quant_linear,'experimental_adjacent_inplace_ids',set())),
        scales={k:v.data_ptr() for k,v in stage.quant_linear.scales.items()} if stage.quant_linear else {},
        graphs={str(k):[v[0].data_ptr(),v[1].data_ptr(),v[2].data_ptr(),id(v[3]),v[4].data_ptr()] for k,v in stage.stage_graphs.items()})
        for stage in executor.stages]


@torch.inference_mode()
def replay(executor,trace):
    checks=[]
    for event in trace['trace']:
        actual=(executor.prefill_batch(event['prompts'],event['slots']) if event['kind']=='prefill'
                else executor.decode_batch(event['tokens'],event['slots'])[0]).float().cpu()
        for row,target in enumerate(event['targets']):
            target=target.squeeze(0);error=actual[row]-target
            relative=(error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1e-6)).item()
            absolute=error.abs().max().item();match=actual[row].argmax().item()==target.argmax().item()
            checks.append(dict(kind=event['kind'],slot=event['slots'][row],relative_rms=relative,max_absolute=absolute,
                greedy_match=match,passed=relative<.01 and absolute<.5 and match))
    assert checks and all(c['passed'] for c in checks),'reference gate failed'
    return checks


def main():
    p=argparse.ArgumentParser();p.add_argument('original');p.add_argument('changed');p.add_argument('--output',required=True)
    p.add_argument('--inject-packed-refresh-failure',action='store_true')
    args=p.parse_args()
    traces={alias:torch.load(path,map_location='cpu',weights_only=True) for alias,path in (('a',args.original),('b',args.changed))}
    quantization=traces['a'].get('reference_quantization','none')
    assert traces['b'].get('reference_quantization','none')==quantization
    assert Path(traces['a']['path']).resolve()!=Path(traces['b']['path']).resolve()
    difference=(traces['a']['trace'][0]['targets'][0]-traces['b']['trace'][0]['targets'][0]).abs().max().item()
    assert difference>.5,'fixture must distinguish unchanged weights from replacement'
    os.environ.update(DAMENDAEMON_EXPERIMENTAL_LAYER_SPLIT='12',DAMENDAEMON_PARTITION_REBIND='1',DAMENDAEMON_QUANTIZATION=quantization)
    runtime=Runtime({alias:t['path'] for alias,t in traces.items()},context=512,batch_size=4,gpu_models=1,host_models=2)
    report=dict(passed=False,quantization=quantization,reference_target_difference=difference,runs=[],
        scope='same-host partition weight rebind correctness and failure proof; timings are not an ABBA benchmark')
    try:
        identity=None;pointers=None
        for alias in ('a','b','a','b'):
            with runtime.session(alias) as (executor,tier,seconds):
                if identity is None:
                    identity=id(executor);pointers=storage(executor)
                    report['experimental_adjacent_bytes']=sum(value.numel()*value.element_size()
                        for stage in executor.stages
                        for value in getattr(stage.quant_linear,'experimental_adjacent',{}).values())
                    report['experimental_inplace_weight_count']=sum(len(getattr(stage.quant_linear,
                        'experimental_adjacent_inplace_ids',set())) for stage in executor.stages)
                    report['stage_adjacent_weight_count']=sum(len(stage.q4_linear.weights)
                        for stage in executor.stages if stage.q4_linear is not None)
                    if args.inject_packed_refresh_failure and not (report['experimental_adjacent_bytes'] or report['experimental_inplace_weight_count'] or report['stage_adjacent_weight_count']):
                        raise ValueError('packed refresh injection requires adjacent stage storage')
                else:
                    assert id(executor)==identity and storage(executor)==pointers,'rebind replaced graph storage'
                    assert executor.lengths==[0]*4,'old request state retained'
                assert executor.model.path==Path(traces[alias]['path']).resolve()
                checks=replay(executor,traces[alias])
                report['runs'].append(dict(alias=alias,source_tier=tier,activation_seconds=seconds,
                    rebind_count=executor.rebind_count,stable_graph_storage=True,checks=checks))
        original_copy=torch.Tensor.copy_;calls=0
        def fail_second_upload(target,source,*args,**kwargs):
            nonlocal calls
            if target.is_cuda and source.device.type=='cpu':
                calls+=1
                if calls==2:raise RuntimeError('injected partial weight upload failure')
            return original_copy(target,source,*args,**kwargs)
        with patch.object(torch.Tensor,'copy_',fail_second_upload):
            try:runtime.manage('load','a')
            except RuntimeError as error:assert str(error)=='injected partial weight upload failure'
            else:raise AssertionError('partial upload unexpectedly succeeded')
        assert not runtime.gpu and calls==2,'partially copied executor remained published'
        report['partial_upload_unpublished']=True
        with runtime.session('a') as (replacement,tier,seconds):
            assert id(replacement)!=identity and replacement.rebind_count==0
            report['fresh_recovery']=dict(source_tier=tier,activation_seconds=seconds,checks=replay(replacement,traces['a']))
        if args.inject_packed_refresh_failure:
            if report['stage_adjacent_weight_count']:
                from damendaemon import q4_stage as q4_adjacent_body
            else:
                from scripts.experiments import q4_adjacent_body
            original_repack=q4_adjacent_body.repack_adjacent;refresh_calls=0
            def fail_refresh(weight):
                nonlocal refresh_calls
                refresh_calls+=1
                if refresh_calls==2:raise RuntimeError('injected packed refresh failure')
                return original_repack(weight)
            with patch.object(q4_adjacent_body,'repack_adjacent',fail_refresh):
                try:runtime.manage('load','b')
                except RuntimeError as error:assert str(error)=='injected packed refresh failure'
                else:raise AssertionError('partial packed refresh unexpectedly succeeded')
            assert not runtime.gpu and refresh_calls==2,'partially refreshed packed storage remained published'
            report['packed_refresh_failure_unpublished']=True
            with runtime.session('b') as (replacement,tier,seconds):
                assert replacement.rebind_count==0
                report['fresh_after_packed_refresh']=dict(source_tier=tier,activation_seconds=seconds,
                    checks=replay(replacement,traces['b']))
        for alias in traces:runtime.manage('unload',alias,release_host=True)
        report['released_status']=runtime.manage('status')
        assert report['released_status']['host_cached_bytes']==0
        report['passed']=True
    finally:
        for alias in list(runtime.models):runtime.manage('unload',alias,release_host=True)
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=report['passed'],runs=[dict(alias=r['alias'],seconds=r['activation_seconds'],rebind_count=r['rebind_count']) for r in report['runs']])))


if __name__=='__main__':main()
