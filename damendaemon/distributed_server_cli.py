"""Experimental loopback HTTP service backed by a resident two-rank pair."""
import argparse
import os
import signal
import threading


def main(argv=None):
    p=argparse.ArgumentParser(prog='damendaemon distributed-serve',
        description='Experimental two-rank Qwen2.5-0.5B greedy HTTP service under torchrun')
    p.add_argument('path',help='matching local safetensors snapshot on each worker')
    p.add_argument('--quantization',choices=('none','q8_0','q4_0'),default=os.environ.get('DAMENDAEMON_QUANTIZATION','none'))
    p.add_argument('--context',type=int,default=512)
    p.add_argument('--alias',default='model')
    p.add_argument('--model',action='append',default=[],metavar='ALIAS=PATH',help='one additional compatible local model on each rank for resident switching')
    p.add_argument('--port',type=int,default=8000)
    p.add_argument('--batch-size',type=int,choices=range(1,5),default=1)
    p.add_argument('--queue-size',type=int,default=8)
    p.add_argument('--queue-timeout',type=float,default=120)
    p.add_argument('--client-decode-turns',action='store_true',help='configured ready decode shares per client; default equal')
    p.add_argument('--idle-timeout',type=float,default=0,help='optional coordinator shutdown after idle seconds; zero disables')
    p.add_argument('--supervise',action='store_true',
                   help='supervised mode (coordinator process only): dead-peer detection, '
                        'honest failed state and bounded operator rejoin via '
                        'POST /v1/distributed/rejoin; spawns its own worker, do not run under torchrun')
    p.add_argument('--heartbeat-deadline',type=float,default=10,
                   help='supervised polled idle-heartbeat deadline in seconds (1..29, '
                        'below the 30s group timeout): a peer that stops completing '
                        'heartbeats is classified dead and the group aborted for rejoin '
                        'instead of the process-fatal NCCL group timeout')
    p.add_argument('--coordinator-gpu',default=None,
                   help='explicit coordinator CUDA device UUID (GPU-...); use when device '
                        'enumeration would pick an occupied device (e.g. a busy A30 beside '
                        'an idle A2)')
    p.add_argument('--worker-host',default=None,
                   help='supervised cross-host mode: SSH host alias running the rank-1 worker '
                        '(one CUDA device); requires --worker-gpu, --worker-python, '
                        '--worker-repo, --rendezvous-host and both interface names')
    p.add_argument('--worker-gpu',default=None,help='remote worker CUDA device UUID (GPU-...)')
    p.add_argument('--worker-python',default=None,help='remote worker venv python path')
    p.add_argument('--worker-repo',default=None,help='remote staged damendaemon repo path')
    p.add_argument('--worker-model-path',default=None,
                   help='remote model snapshot path (defaults to the local path, which must then exist remotely)')
    p.add_argument('--rendezvous-host',default=None,
                   help='coordinator LAN address for the rendezvous store and MASTER_ADDR')
    p.add_argument('--coordinator-ifname',default=None,help='coordinator NCCL_SOCKET_IFNAME')
    p.add_argument('--worker-ifname',default=None,help='remote worker NCCL_SOCKET_IFNAME')
    args=p.parse_args(argv)
    if not 1<=args.context<=512 or not 0<=args.port<=65535 or not 0<=args.idle_timeout<=86400:
        p.error('invalid context, port or idle timeout')
    if not 1<=args.heartbeat_deadline<30:
        p.error('heartbeat deadline must be 1..29 seconds, below the 30s group timeout')
    if not args.supervise and args.worker_host:
        p.error('cross-host worker spawn is a supervised mode (--supervise) option')
    if not 1<=args.queue_size<=64 or not 0<args.queue_timeout<=300 or not 1<=len(args.alias)<=128:
        p.error('invalid queue bounds or model alias')
    if len(args.model)>1 or any('=' not in entry for entry in args.model):
        p.error('at most one additional ALIAS=PATH model is supported')
    additional=[tuple(entry.split('=',1)) for entry in args.model]
    if any(not alias or not path or alias==args.alias for alias,path in additional):
        p.error('additional model needs a distinct alias and local path')
    os.environ['DAMENDAEMON_DEVICE']='cuda'
    from .server import BoundedHTTPServer,handler_for
    if args.supervise:
        from .distributed_supervised_builder import supervised_builder,teardown_supervised_pair
        from .distributed_supervisor import DistributedSupervisor
        import torch
        supervisor=DistributedSupervisor(supervised_builder(args,torch))
        server=thread=previous=None;primary=None
        try:
            supervisor.start()
            runtime=supervisor.runtime_proxy
            base=handler_for(runtime)
            class Handler(base):
                def do_GET(self):
                    if self.path=='/health':
                        code,body=runtime.health();self.send_json(code,body)
                    else:super().do_GET()
                def do_POST(self):
                    if self.path=='/v1/distributed/rejoin':
                        initiated,detail=runtime.rejoin()
                        self.send_json(202 if initiated else 409,dict(rejoin=initiated,detail=detail))
                    else:super().do_POST()
            server=BoundedHTTPServer(('127.0.0.1',args.port),Handler)
            server.daemon_threads=False;server.block_on_close=True
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            previous=signal.signal(signal.SIGTERM,lambda number,frame:supervisor.close())
            print(f'damendaemon distributed listening on http://127.0.0.1:{server.server_port} (supervised)',flush=True)
            try:
                # HTTP stays up in failed/rejoining states so the operator can
                # observe /health and trigger the bounded rejoin.
                while supervisor.state not in ('closed','terminal'):
                    threading.Event().wait(1)
            except KeyboardInterrupt:supervisor.close()
        except BaseException as error:primary=error;raise
        finally:
            if previous is not None:signal.signal(signal.SIGTERM,previous)
            errors=[]
            if server is not None:
                try:
                    if thread is not None:server.shutdown();thread.join(timeout=5)
                except BaseException as error:errors.append(error)
                try:server.server_close()
                except BaseException as error:errors.append(error)
            try:supervisor.close()
            except BaseException as error:errors.append(error)
            if errors:
                if primary is not None:primary.add_note('supervised distributed HTTP shutdown cleanup failed; retain owners for retry')
                else:raise errors[0]
        return
    from .distributed import StagePair
    from .distributed_serving import DistributedRuntime
    from .server import BoundedHTTPServer,handler_for
    pair=StagePair(args.path,args.context,args.quantization,lambda model:[],session='distributed-serve',max_batch=args.batch_size,
                   model_alias=args.alias,additional_models=additional)
    runtime=server=thread=None
    previous=None;primary=None
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
        server.daemon_threads=False
        server.block_on_close=True
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        previous=signal.signal(signal.SIGTERM,lambda number,frame:runtime.close())
        print(f'damendaemon distributed listening on http://127.0.0.1:{server.server_port}',flush=True)
        try:runtime.run(idle_timeout=args.idle_timeout)
        except KeyboardInterrupt:runtime.close()
    except BaseException as error:primary=error;raise
    finally:
        if previous is not None:signal.signal(signal.SIGTERM,previous)
        errors=[]
        if runtime is not None:
            try:runtime.close()
            except BaseException as error:errors.append(error)
        if server is not None:
            try:
                if thread is not None:server.shutdown();thread.join(timeout=5)
            except BaseException as error:errors.append(error)
            try:server.server_close()
            except BaseException as error:errors.append(error)
        try:pair.stop()
        except BaseException as error:errors.append(error)
        try:pair.close()
        except BaseException as error:errors.append(error)
        if errors:
            if primary is not None:primary.add_note('distributed HTTP shutdown cleanup failed; retain owners for retry')
            else:raise errors[0]
