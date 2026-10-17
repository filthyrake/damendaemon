"""Tiny packed CUDA GLM latent/selected-attention gate, no cache/model claim."""
import argparse
from pathlib import Path
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from damendaemon.cuda_glm_attention import project_mla,sparse_latent_attention
    from damendaemon.glm_attention import project_mla as cpu_project
    from damendaemon.cuda_prepared_projection import CudaPackedProjection
    from damendaemon.prepared_projection import PackedProjection
    from tests.test_glm_packed_attention import models
    from tests.test_packed_projection import fixture
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False,cache_qualified=False)
    save(args.output,report)
    try:
        packed,dense,hidden=models(True)
        positions=torch.arange(7).repeat(2,1)
        def move(weight):
            if isinstance(weight,PackedProjection):
                return CudaPackedProjection(weight.weight.cuda(),weight.scales.cuda(),format=weight.format,**weight.options)
            return weight.cuda()
        weights={name:move(value) for name,value in packed.projections.items()}
        expected=cpu_project(hidden,positions,dense.projections,heads=2,nope_dim=2,rope_dim=4)
        for chunk in (1,3):
            projected=project_mla(hidden.cuda(),positions.cuda(),weights,heads=2,nope_dim=2,rope_dim=4,chunk=chunk)
            errors={}
            for name,value in projected.items():
                torch.testing.assert_close(value.cpu(),expected[name],rtol=3e-5,atol=3e-6)
                errors[name]=float((value.cpu()-expected[name]).abs().max())
            report['cases'].append(dict(kind='projection',chunk=chunk,errors=errors));save(args.output,report)
        indices=torch.full((2,7,4),-1,dtype=torch.int64)
        for b in range(2):
            for s in range(7):
                chosen=list(range(s,-1,-2))[:4];indices[b,s,:len(chosen)]=torch.tensor(chosen)
        expanded=(expected['latents']@dense.weights['kv_b_proj.weight'].T).reshape(2,7,2,5)
        keys=torch.cat((expanded[...,:2],expected['rotary_keys'][:,:,None,:].expand(-1,-1,2,-1)),-1)
        scores=torch.einsum('bshd,bthd->bsht',expected['query'],keys)*(6**-.5)
        mask=torch.ones(2,7,7,dtype=torch.bool)
        for b in range(2):
            for s in range(7):mask[b,s,indices[b,s][indices[b,s]>=0]]=False
        scores.masked_fill_(mask[:,:,None,:],-float('inf'))
        control=torch.einsum('bsht,bthv->bshv',scores.softmax(-1),expanded[...,2:])
        kv=move(packed.weights['kv_b_proj.weight'])
        for chunk in (1,2,64):
            result=sparse_latent_attention(projected['query'],projected['latents'],projected['rotary_keys'],kv,indices.cuda(),positions.cuda(),nope_dim=2,value_dim=3,key_chunk=chunk)
            torch.testing.assert_close(result.cpu(),control,rtol=3e-5,atol=3e-6)
            report['cases'].append(dict(kind='selected_attention',chunk=chunk,max_absolute=float((result.cpu()-control).abs().max())));save(args.output,report)
        for bad in (torch.full_like(indices,-1),torch.ones_like(indices)):
            try:sparse_latent_attention(projected['query'],projected['latents'],projected['rotary_keys'],kv,bad.cuda(),positions.cuda(),nope_dim=2,value_dim=3)
            except ValueError:pass
            else:raise AssertionError('invalid causal selection admitted')
        source,raw,scales,unpacked=fixture('e4m3fn',133,160)
        view=CudaPackedProjection(raw.cuda(),scales.cuda(),format='e4m3fn',row_chunk=3,column_chunk=31).rows(125,133).rows(1,6)
        result=view.linear(source.cuda().unsqueeze(0).expand(2,-1,-1)[:,::2])
        torch.testing.assert_close(result.cpu(),source.unsqueeze(0).expand(2,-1,-1)[:,::2]@unpacked[126:131].T,rtol=2e-5,atol=1e-3)
        report.update(passed=True,row_view=True,invalid_selection_rejected=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()
