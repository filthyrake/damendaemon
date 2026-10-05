"""Two-process layer proof using typed NCCL tensors and a trusted local trace.

Launch with torchrun, one explicitly selected GPU per rank. Both workers must
hold identical checkpoint/trace files. This is a fixed-trace experiment, not a
network-facing serving protocol; no Python object is deserialized from a peer.
"""
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from dllm.qwen2 import ModelWeights
from scripts.experiments.layer_stage import LayerStage


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('trace')
    parser.add_argument('--output',required=True)
    parser.add_argument('--model-path')
    args=parser.parse_args()
    rank=int(os.environ['RANK']); local_rank=int(os.environ['LOCAL_RANK'])
    if int(os.environ['WORLD_SIZE'])!=2:parser.error('exactly two workers required')
    torch.cuda.set_device(local_rank)
    trace_path=Path(args.trace.format(rank=rank))
    trace=torch.load(trace_path,map_location='cpu',weights_only=True)
    model=ModelWeights(args.model_path or trace['path'],trace['context'])
    total=model.config['num_hidden_layers']; split=total//2
    start,end=(0,split) if rank==0 else (split,total)
    stage=LayerStage(model,start,end,max_batch=4)
    report=dict(passed=False,rank=rank,device=torch.cuda.get_device_name(),layers=[start,end],
        model=trace['path'],context=trace['context'],torch=str(torch.__version__),
        reference_torch=trace['reference_torch'],reference_dtype=trace['reference_dtype'],
        checks=[],scope='fixed-trace distributed layer experiment; not a serving protocol')
    output=Path(args.output.format(rank=rank))
    try:
        dist.init_process_group('nccl',timeout=timedelta(seconds=30),device_id=torch.device('cuda',local_rank))
        # Compare exact trusted trace and configuration before any activation sizes
        # can disagree. Fixed-size bytes avoid object collectives and pickle.
        digest=hashlib.sha256(trace_path.read_bytes()+json.dumps(model.config,sort_keys=True).encode()).digest()
        local=torch.tensor(list(digest),dtype=torch.uint8,device='cuda')
        peers=[torch.empty_like(local) for _ in range(2)]
        dist.all_gather(peers,local)
        if not torch.equal(peers[0],peers[1]):raise ValueError('worker trace/configuration mismatch')
        lengths=[0]*4
        for event in trace['trace']:
            if event['kind']=='prefill':
                batches=[(prompt,[slot],[0],target) for prompt,slot,target in zip(
                    event['prompts'],event['slots'],event['targets'])]
            elif event['kind']=='decode':
                batches=[(event['tokens'],event['slots'],[lengths[s] for s in event['slots']],event['targets'])]
            else:raise ValueError('unknown trace event')
            for tokens,slots,positions,targets in batches:
                prefill=event['kind']=='prefill'
                if (not tokens or len(set(slots))!=len(slots) or any(s<0 or s>=4 for s in slots)
                        or (not prefill and len(tokens)!=len(slots))
                        or any(p<0 or p>=stage.context for p in positions)
                        or (prefill and len(tokens)>stage.context)):
                    raise ValueError('invalid trusted trace shape or cache position')
                pos=torch.arange(len(tokens),device='cuda') if prefill else torch.tensor(positions,device='cuda')
                slot_tensor=torch.tensor(slots,device='cuda')
                if rank==0:
                    inputs=torch.tensor(tokens,device='cuda')
                    hidden=(stage.run_prefill(inputs,pos,slots[0]) if prefill
                            else stage.run_decode(inputs,pos,slot_tensor))
                    operation=dist.P2POp(dist.isend,hidden,1)
                else:
                    hidden=torch.empty((len(tokens),model.config['hidden_size']),
                        dtype=stage.residual_dtype,device='cuda')
                    operation=dist.P2POp(dist.irecv,hidden,0)
                for work in dist.batch_isend_irecv([operation]):work.wait()
                if rank==1:
                    actual=(stage.run_prefill(hidden,pos,slots[0]) if prefill
                            else stage.run_decode(hidden,pos,slot_tensor)).float().cpu()
                    for row,target in enumerate([targets] if prefill else targets):
                        target=target.squeeze(0);error=actual[row]-target
                        relative=(error.square().mean().sqrt()/target.square().mean().sqrt().clamp_min(1e-6)).item()
                        absolute=error.abs().max().item();match=actual[row].argmax().item()==target.argmax().item()
                        report['checks'].append(dict(kind=event['kind'],slot=slots[row],relative_rms=relative,
                            max_absolute=absolute,greedy_match=match,passed=relative<.01 and absolute<.5 and match))
                if prefill:lengths[slots[0]]=len(tokens)
                else:
                    for slot in slots:lengths[slot]+=1
        passed=rank==0 or (bool(report['checks']) and all(c['passed'] for c in report['checks']))
        result=torch.tensor([int(passed)],device='cuda')
        dist.all_reduce(result,op=dist.ReduceOp.MIN)
        report['passed']=bool(result.item())
        report['nccl']=torch.cuda.nccl.version()
        report['requested_transport']=os.environ.get('NCCL_NET','auto')
        report['gpudirect_rdma']='unverified'
        if not report['passed']:raise AssertionError('distributed reference gate failed')
    except BaseException as error:
        report['error']=repr(error)
        raise
    finally:
        output.write_text(json.dumps(report,indent=2)+'\n')
        if dist.is_initialized():dist.destroy_process_group()


if __name__=='__main__':main()
