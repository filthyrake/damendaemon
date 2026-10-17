"""Experimental dense greedy speculation; no serving-path or runtime changes.

Oracle proposals are an unattainable acceptance upper bound, never a real-world
speed claim. Every block evaluates the complete target vocabulary. CUDA graph
queries share one KV slot but have distinct causal positions.
"""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import json
import os
import statistics
import time

import torch
from damendaemon.runtime import Runtime


def ngram(history,limit):
    for size in (4,3,2,1):
        suffix=history[-size:]
        for start in range(len(history)-size-1,max(-1,len(history)-1024),-1):
            if history[start:start+size]==suffix:
                return history[start+size:start+size+limit]
    return []


class Verifier:
    def __init__(self,executor,width):
        self.executor=executor;self.width=width
        self.metadata=torch.zeros((3,width),device='cuda',dtype=torch.long)
        ids,positions,slots=self.metadata.unbind(0);positions.copy_(torch.arange(width,device='cuda'))
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream),torch.inference_mode():
            for _ in range(3):executor.forward_batch(ids,positions,slots)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        started=time.perf_counter();self.graph=torch.cuda.CUDAGraph()
        with torch.inference_mode(),torch.cuda.graph(self.graph):
            logits=executor.forward_batch(ids,positions,slots)
            self.next_ids=logits.argmax(-1)
        torch.cuda.synchronize();self.capture_ms=(time.perf_counter()-started)*1000

    @torch.inference_mode()
    def run(self,inputs,start):
        assert 1<=len(inputs)<=self.width and start+self.width<=self.executor.context
        payload=[inputs+[0]*(self.width-len(inputs)),list(range(start,start+self.width)),[0]*self.width]
        self.metadata.copy_(torch.tensor(payload,device='cuda'))
        self.graph.replay()
        return self.next_ids.tolist()


@torch.inference_mode()
def generate(executor,prompt,budget,verifier=None,mode='normal',oracle=None,eos=()):
    torch.cuda.synchronize();started=time.perf_counter()
    current=int(executor.prefill(prompt).argmax(-1).item())
    output=[];blocks=accepted=proposed=rejected=0
    while len(output)<budget:
        if current in eos:
            output.append(current);break
        start=len(prompt)+len(output)
        proposals=[]
        if verifier is not None and start+verifier.width<=executor.context and len(output)+1<budget:
            limit=min(verifier.width-1,budget-len(output)-1)
            if mode=='ngram':proposals=ngram(prompt+output+[current],limit)
            else:
                proposals=list(oracle[len(output)+1:len(output)+1+limit])
                if mode=='reject' and proposals:proposals[0]=(proposals[0]+1)%executor.c['vocab_size']
        if not proposals:
            output.append(current)
            if len(output)==budget:break
            current=int(executor.decode_batch([current],[0])[1].item())
            continue
        predictions=verifier.run([current]+proposals,start)
        count=0
        for predicted,draft in zip(predictions,proposals):
            if predicted!=draft:break
            count+=1
        blocks+=1;accepted+=count;proposed+=len(proposals);rejected+=int(count<len(proposals))
        committed=[current]+proposals[:count]
        # All rejected/filler KV entries stay beyond the committed causal limit.
        # The next iteration overwrites its prefix before attention consumes it.
        previous=len(output)
        for token in committed:
            output.append(token)
            if token in eos or len(output)==budget:break
        executor.lengths[0]=start+len(output)-previous
        if output[-1] in eos:break
        current=predictions[count]
    torch.cuda.synchronize()
    return {'tokens':output,'seconds':time.perf_counter()-started,'verification_blocks':blocks,
        'accepted_drafts':accepted,'proposed_drafts':proposed,'rejected_blocks':rejected}


def main():
    p=argparse.ArgumentParser();p.add_argument('model');args=p.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='GPU-1b29c7c8-46b1-6a03-2ab7-a47b5b93a283':
        raise RuntimeError('restricted to the authorized A2')
    os.environ['DAMENDAEMON_DEVICE']='cuda';torch.set_num_threads(4)
    runtime=Runtime({'model':args.model},context=1024,batch_size=1)
    report={'model':args.model,'gpu':'NVIDIA A2','experiment_only':True,
        'oracle_is_unattainable_upper_bound':True,'full_target_vocabulary':True,'rows':[],'gates':[]}
    try:
        with runtime.session('model') as (executor,_,_):
            assert executor.c['model_type'] in ('qwen2','qwen3') and not executor.model.scales and not hasattr(executor,'pool')
            verifiers={width:Verifier(executor,width) for width in (2,4,8)}
            report['capture_ms']={k:v.capture_ms for k,v in verifiers.items()}
            report['compute_dtype']=str(executor.compute_dtype)
            texts={
                'explanation':'Explain why the sky looks blue in simple language.',
                'repetition':'Write the following sentence exactly ten times, with no introduction: The quick brown fox jumps over the lazy dog.',
                'code':'Write a Python function that returns a list containing the integers from 1 to 30, written out explicitly.'}
            for name,text in texts.items():
                prompt=executor.model.chat_tokens([{'role':'user','content':text}],
                    enable_thinking=False if executor.c['model_type']=='qwen3' else None)
                baseline=generate(executor,prompt,64)
                for width in (2,4,8):
                    for mode in ('oracle','ngram','reject'):
                        trials=[generate(executor,prompt,64,verifiers[width],mode,baseline['tokens']) for _ in range(3)]
                        normal=[generate(executor,prompt,64) for _ in range(3)]
                        exact=all(t['tokens']==baseline['tokens'] for t in trials+normal)
                        report['rows'].append({'case':name,'prompt':prompt,'width':width,'mode':mode,
                            'exact_tokens':exact,'reference_tokens':baseline['tokens'],'trials':trials,
                            'baseline_trials':normal,'median_ms':statistics.median(t['seconds'] for t in trials)*1000,
                            'baseline_median_ms':statistics.median(t['seconds'] for t in normal)*1000})
                        print(name+' '+str(width)+' '+mode+' exact='+str(exact),file=__import__('sys').stderr,flush=True)
                # Stop on a real model EOS, plus a deliberately early stop token.
                for eos in (executor.model.eos,{baseline['tokens'][3]}):
                    expected=generate(executor,prompt,64,eos=eos)
                    actual=generate(executor,prompt,64,verifiers[8],'oracle',baseline['tokens'],eos=eos)
                    report['gates'].append({'case':name,'eos':list(eos),'exact_tokens':actual['tokens']==expected['tokens']})
            # Tail fallback must honor the context boundary and reject stale KV.
            prompt=([40,298,1111,13]*253)[:1012]
            expected=generate(executor,prompt,12)
            for mode in ('oracle','ngram','reject'):
                actual=generate(executor,prompt,12,verifiers[8],mode,expected['tokens'])
                report['gates'].append({'case':'context_tail','mode':mode,'exact_tokens':actual['tokens']==expected['tokens']})
            report['passed']=all(row['exact_tokens'] for row in report['rows']+report['gates'])
            print(json.dumps(report,indent=2))
    finally:runtime.manage('unload',model='model',release_host=True)


if __name__=='__main__':main()
