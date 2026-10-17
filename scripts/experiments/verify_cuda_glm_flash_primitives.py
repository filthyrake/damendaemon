"""Tiny guarded CUDA Flash primitive gate against independent scalar equations."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
from pathlib import Path
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from damendaemon.cuda_glm_flash_primitives import recurrent_attention,CudaFlashHyperConnection
    from damendaemon.glm_flash_kda import recurrent_attention as cpu_recurrence
    from damendaemon.glm_flash_hyper import FlashHyperConnection
    from tests.test_glm_flash_kda import inputs,scalar_control as recurrence_control
    from tests.test_glm_flash_hyper import component,scalar_control as hyper_control
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,cases=[],model_qualified=False)
    save(args.output,report)
    def rejected(call):
        try:call()
        except ValueError:return
        raise AssertionError('invalid input admitted')
    try:
        host=inputs();device=[x.cuda() for x in host];copies=[x.clone() for x in device]
        expected=recurrence_control(*host);full=recurrent_attention(*device)
        for actual,control in zip(full,expected):torch.testing.assert_close(actual.cpu(),control,rtol=3e-6,atol=2e-7)
        report['cases'].append(dict(kind='recurrence',max_absolute=max(float((a.cpu()-e).abs().max()) for a,e in zip(full,expected))))
        for lengths in ((2,3),(1,1,1,1,1)):
            state=device[-1];cursor=0;outputs=[]
            for length in lengths:
                out,state=recurrent_attention(*(x[:,cursor:cursor+length] for x in device[:-1]),state)
                outputs.append(out);cursor+=length
            torch.testing.assert_close(torch.cat(outputs,1),full[0],rtol=0,atol=0)
            torch.testing.assert_close(state,full[1],rtol=0,atol=0)
            report['cases'].append(dict(kind='chunk_continuation',lengths=lengths))
        for a,b in zip(recurrent_attention(*device[:-1]),recurrent_attention(*device[:-1],torch.zeros_like(device[-1]))):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert full[1].data_ptr()!=device[-1].data_ptr()
        for option in ('max_workspace_bytes','max_output_bytes'):
            rejected(lambda:recurrent_attention(*device,**{option:1}))
        for index,bad in ((1,host[1]),(3,device[3]+10),(4,device[4]+2),(2,device[2]+float('nan'))):
            changed=list(device);changed[index]=bad;rejected(lambda:recurrent_attention(*changed))
        rejected(lambda:cpu_recurrence(*device))
        for a,b in zip(device,copies):torch.testing.assert_close(a,b,rtol=0,atol=0)
        for iterations in (1,20):
            c=component(iterations=iterations)
            gpu=CudaFlashHyperConnection(c.projection.cuda(),c.base.cuda(),c.scale.cuda(),streams=2,iterations=iterations)
            hidden=torch.randn(2,3,3,2).transpose(-1,-2);state=hidden.cuda();original=state.clone()
            control=hyper_control(hidden,c);actual=gpu.reduce(state)
            for a,e in zip(actual,control):torch.testing.assert_close(a.cpu(),e,rtol=2e-6,atol=3e-7)
            output,post,matrix=actual
            restored=gpu.restore(output,state,post,matrix)
            expected_restore=torch.empty_like(hidden)
            out_cpu,post_cpu,matrix_cpu=[x.cpu() for x in actual]
            for b in range(2):
                for t in range(3):
                    for dest in range(2):
                        expected_restore[b,t,dest]=post_cpu[b,t,dest]*out_cpu[b,t]+sum(matrix_cpu[b,t,src,dest]*hidden[b,t,src] for src in range(2))
            torch.testing.assert_close(restored.cpu(),expected_restore,rtol=2e-6,atol=3e-7)
            torch.testing.assert_close(gpu.head(state).cpu(),(hidden[:,:,0]+hidden[:,:,1])/2)
            rejected(lambda:gpu.reduce(hidden));rejected(lambda:c.reduce(state))
            rejected(lambda:gpu.restore(output.cpu(),state,post,matrix))
            rejected(lambda:gpu.reduce(state+float('nan')))
            for option in ('max_workspace_bytes','max_output_bytes'):
                bounded=CudaFlashHyperConnection(gpu.projection,gpu.base,gpu.scale,streams=2,**{option:1})
                for call in (lambda:bounded.reduce(state),lambda:bounded.head(state),lambda:bounded.restore(output,state,post,matrix)):rejected(call)
            torch.testing.assert_close(state,original,rtol=0,atol=0)
            rejected(lambda:FlashHyperConnection(gpu.projection,gpu.base,gpu.scale,streams=2))
            report['cases'].append(dict(kind='hyper',iterations=iterations,max_absolute=max(float((a.cpu()-e).abs().max()) for a,e in zip(actual,control))))
        report.update(passed=True,invalid_inputs_rejected=True,inputs_unchanged=True,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:save(args.output,report)


if __name__=='__main__':main()
