"""Tiny checkpoint->CUDA->native chat HTTP gate; zero head controls wire output."""
import argparse
import json
from pathlib import Path
import tempfile
import threading
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from scripts.experiments.q5_cuda_oracle import require_guard
from scripts.experiments.split_gguf import save


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();require_guard()
    if args.output.exists():raise ValueError('new report path required')
    import torch
    from tests.test_glm_generation import text_fixture
    from tests.test_glm_reference_loader import fixture
    from dllm.glm_reference_loader import load_reference_model
    from dllm.cuda_glm_placement import place_cuda_glm
    from dllm.cuda_glm_generation import CudaGlmToolGenerator
    from dllm.glm_tool_generation import GlmToolReferenceGenerator
    from dllm.glm_tool_text import GlmToolText
    from dllm.cuda_reference_serving import CudaGlmReferenceRuntime
    from dllm.reference_serving import handler_for_reference
    from dllm.server import BoundedHTTPServer
    from dllm.http_auth import AuthPolicy
    if torch.version.hip or torch.cuda.device_count()!=1:raise ValueError('one NVIDIA GPU required')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(128*1024**2/torch.cuda.get_device_properties(0).total_memory)
    report=dict(passed=False,device=torch.cuda.get_device_name(),torch=torch.__version__,zero_head_fixture=True,
                official_model_qualified=False,performance_qualified=False)
    save(args.output,report);server=thread=runtime=None
    try:
        with tempfile.TemporaryDirectory(prefix='dllm-cuda-http-') as temp:
            path=Path(temp);base_text=text_fixture(path);fixture(path,vocab_size=base_text.vocab_size)
            text=GlmToolText(path,vocab_size=base_text.vocab_size)
            cpu=load_reference_model(path,max_weight_bytes=1024*1024);cpu.head.zero_()
            messages=[dict(role='user',content='Hi')]
            expected=list(GlmToolReferenceGenerator(cpu,text,max_context=128).generate(messages,max_new_tokens=2))
            expected_text=''.join(e.get('reasoning_content','') for e in expected)
            placed=place_cuda_glm(cpu,device='cuda:0',max_bytes=1024*1024);model=placed.model
            states=[];original=model.prefill
            def capture(*a,**kw):
                logits,state=original(*a,**kw);states.append(state);return logits,state
            model.prefill=capture
            runtime=CudaGlmReferenceRuntime(CudaGlmToolGenerator(model,text,max_context=128,max_cache_bytes=1024*1024))
            server=BoundedHTTPServer(('127.0.0.1',0),handler_for_reference(runtime,AuthPolicy()))
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            base=f'http://127.0.0.1:{server.server_port}'
            with urlopen(base+'/health',timeout=2) as response:health=json.load(response)
            if health['scope']!='cuda_prepared_reference':raise AssertionError('health scope hides CUDA execution')
            body=dict(model='model',messages=messages,max_tokens=2)
            def post(value):return urlopen(Request(base+'/v1/chat/completions',data=json.dumps(value).encode(),headers={'Content-Type':'application/json'}),timeout=20)
            with post(body) as response:result=json.load(response)
            if result['choices'][0]['message'].get('reasoning_content','')!=expected_text:raise AssertionError('CUDA chat differs from CPU wire control')
            if result['dllm_timing']['source_tier']!='cuda_prepared_reference' or result['dllm_timing']['kernel_backend']!='glm_moe_dsa-cuda-prepared-reference':raise AssertionError('HTTP timing mislabeled CUDA execution')
            with post(body|dict(stream=True)) as response:wire=response.read().decode()
            events=[json.loads(line[6:]) for line in wire.splitlines() if line.startswith('data: {')]
            streamed=''.join(choice.get('delta',{}).get('reasoning_content','') for event in events for choice in event['choices'])
            if streamed!=expected_text or 'data: [DONE]' not in wire:raise AssertionError('CUDA SSE differs from CPU control')
            if len(states)!=2 or any(not s.closed or s.owner is not None or s.layers for s in states):raise AssertionError('HTTP cache cleanup incomplete')
            status=runtime.manage('status')
            if status['device']!='cuda:0' or status['scope']!='cuda_prepared_reference' or status['models'][0]['residency']!='cuda':raise AssertionError('untruthful device status')
            runtime.close()
            try:urlopen(base+'/health',timeout=2)
            except HTTPError as error:
                if error.code!=503:raise
            else:raise AssertionError('closed runtime still healthy')
            report.update(passed=True,chat=True,sse=True,cache_closed=True,closed_health=503,
                          placement_raw_bytes=placed.raw_bytes,status=status,peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except BaseException as error:report['error']=str(error);raise
    finally:
        if server is not None:server.shutdown();server.server_close()
        if thread is not None:thread.join(3)
        if runtime is not None:runtime.close()
        save(args.output,report)


if __name__=='__main__':main()
