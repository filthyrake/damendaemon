"""Experimental loopback HTTP service backed by a resident two-rank pair."""
import argparse
import os
import signal
import threading


def main(argv=None):
    p=argparse.ArgumentParser(prog='dllm distributed-serve',
        description='Experimental two-rank Qwen2.5-0.5B greedy HTTP service under torchrun')
    p.add_argument('path',help='matching local safetensors snapshot on each worker')
    p.add_argument('--quantization',choices=('none','q8_0','q4_0'),default=os.environ.get('DLLM_QUANTIZATION','none'))
    p.add_argument('--context',type=int,default=512)
    p.add_argument('--alias',default='model')
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--batch-size',type=int,choices=range(1,5),default=1)
    p.add_argument('--queue-size',type=int,default=8)
    p.add_argument('--queue-timeout',type=float,default=120)
    p.add_argument('--client-decode-turns',action='store_true',help='equal ready decode rows per client')
    p.add_argument('--idle-timeout',type=float,default=0,help='optional coordinator shutdown after idle seconds; zero disables')
    args=p.parse_args(argv)
    if not 1<=args.context<=512 or not 0<=args.port<=65535 or not 0<=args.idle_timeout<=86400:
        p.error('invalid context, port or idle timeout')
    if not 1<=args.queue_size<=64 or not 0<args.queue_timeout<=300 or not 1<=len(args.alias)<=128:
        p.error('invalid queue bounds or model alias')
    os.environ['DLLM_DEVICE']='cuda'
    from .distributed import StagePair
    from .distributed_serving import DistributedRuntime
    from .server import BoundedHTTPServer,handler_for
    pair=StagePair(args.path,args.context,args.quantization,lambda model:[],session='distributed-serve',max_batch=args.batch_size)
    runtime=server=thread=None
    previous=None
    try:
        if pair.rank==1:
            pair.serve_worker();return
        runtime=DistributedRuntime(pair,args.alias,args.queue_size,args.queue_timeout,client_decode_turns=args.client_decode_turns)
        base=handler_for(runtime)
        class Handler(base):
            def do_GET(self):
                if self.path=='/health':
                    stopped=runtime.closed.is_set()
                    self.send_json(503 if stopped else 200,dict(status='closed' if stopped else 'ok'))
                else:super().do_GET()
        server=BoundedHTTPServer(('127.0.0.1',args.port),Handler)
        server.daemon_threads=True
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        previous=signal.signal(signal.SIGTERM,lambda number,frame:runtime.close())
        print(f'dllm distributed listening on http://127.0.0.1:{server.server_port}',flush=True)
        try:runtime.run(idle_timeout=args.idle_timeout)
        except KeyboardInterrupt:runtime.close()
    finally:
        if previous is not None:signal.signal(signal.SIGTERM,previous)
        if runtime is not None:runtime.close()
        if server is not None:
            if thread is not None:server.shutdown();thread.join(timeout=5)
            server.server_close()
        try:pair.stop()
        finally:pair.close()
