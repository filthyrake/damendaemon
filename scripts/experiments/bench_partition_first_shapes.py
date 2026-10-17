"""Fresh-process/private-Triton-cache first-shape latency; host files may be cached."""
if not __debug__:
    raise RuntimeError("Qualification assertions are disabled; run Python without -O or -OO.")

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import tempfile
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('model');p.add_argument('prompt');p.add_argument('--dynamic',action='store_true')
    p.add_argument('--output',required=True);p.add_argument('--integrated',action='store_true')
    rope=p.add_mutually_exclusive_group()
    rope.add_argument('--experimental-rope-runtime-tokens',action='store_true')
    rope.add_argument('--integrated-rope-runtime-tokens',action='store_true');args=p.parse_args()
    def terminate(number,frame):raise SystemExit(128+number)
    signal.signal(signal.SIGTERM,terminate)
    report=dict(passed=False,integrated=args.integrated,dynamic=args.dynamic,
        experimental_rope_runtime_tokens=args.experimental_rope_runtime_tokens,
        integrated_rope_runtime_tokens=args.integrated_rope_runtime_tokens,
        model=args.model,quantization='q4_0',context=512,capacity=4,first=[],repeat=[],
        scope='Fresh process and private Triton compiler cache for each trial; host checkpoint/filesystem/driver caches are not reset. Same-host first-token computation, including compilation and GPU completion, without HTTP/queue/network. No steady-state throughput claim.',
        sources={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
                 ('damendaemon/q4_stage.py','damendaemon/q4_stage_kernels.py','damendaemon/kernels.py','damendaemon/partition.py','damendaemon/layer_stage.py','damendaemon/qwen2.py','damendaemon/partition_policy.py',
                  'scripts/experiments/q4_dynamic_rows.py','scripts/experiments/rope_runtime_tokens.py',__file__)})
    executor=None
    try:
        with tempfile.TemporaryDirectory(prefix='damendaemon-q4-first-shapes-') as cache:
            os.environ.update(TRITON_CACHE_DIR=cache,DAMENDAEMON_Q4_STAGE_ADJACENT='1',DAMENDAEMON_PARTITION_PACKED_PREFILL='1')
            started=time.perf_counter()
            import torch
            from damendaemon.qwen2 import ModelWeights
            from damendaemon.partition import PartitionExecutor
            from scripts.experiments.q4_dynamic_rows import install_validation_dispatch
            report['import_seconds']=time.perf_counter()-started
            assert torch.cuda.device_count()==2,'select exactly two GPUs'
            torch.set_num_threads(4)
            report['devices']=dict(selected=os.environ.get('CUDA_VISIBLE_DEVICES'),
                names=[torch.cuda.get_device_name(index) for index in range(2)])
            report['torch']=str(torch.__version__)
            os.environ['DAMENDAEMON_Q4_STAGE_RUNTIME_ROWS']=str(int(args.dynamic and args.integrated))
            os.environ['DAMENDAEMON_Q4_STAGE_RUNTIME_ROPE']=str(int(args.integrated_rope_runtime_tokens))
            if args.dynamic and not args.integrated:install_validation_dispatch()
            if args.experimental_rope_runtime_tokens:
                from scripts.experiments.rope_runtime_tokens import install_validation_dispatch as install_rope
                install_rope()
            def sync():
                for device in range(2):torch.cuda.synchronize(device)
            prompt=json.loads(Path(args.prompt).read_text());assert len(prompt)==16
            with torch.inference_mode():
                started=time.perf_counter();model=ModelWeights(args.model,512,quantization='q4_0')
                report['host_weights_seconds']=time.perf_counter()-started
                sync();started=time.perf_counter();executor=PartitionExecutor(model,12,use_graph=True,max_batch=4);sync()
                report['activation_seconds']=time.perf_counter()-started
                for field in ('first','repeat'):
                    for length in (3,5,7,11,13,17,19,23,31,33,47,63,65,95,127):
                        tokens=(prompt*((length+15)//16))[:length]
                        sync();started=time.perf_counter();logits=executor.prefill_batch([tokens],[0])
                        token=int(logits.argmax(-1).item());sync();elapsed=time.perf_counter()-started
                        report[field].append(dict(prompt_tokens=length,seconds=elapsed,token=token))
                    print(field,'complete',flush=True)
                assert [r['token'] for r in report['first']]==[r['token'] for r in report['repeat']]
                del logits
                executor.close();executor=None
            report['passed']=True
    finally:
        if executor is not None:executor.close()
        Path(args.output).write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
