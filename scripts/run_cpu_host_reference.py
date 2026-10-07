"""Explicit authenticated two-process CPU reference with optional explicit text chat."""
import argparse
from contextlib import closing
import hashlib
import ipaddress
import json
from pathlib import Path
import socket

from dllm.config import hybrid_config
from dllm.cpu_hybrid_stage import cpu_hybrid_shapes
from dllm.cpu_reference_text import CpuReferenceText,read_json
from dllm.cpu_completion_inputs import completion_inputs
from dllm.cpu_host_worker_build import build_cpu_host_worker,CpuHostWorkerBuildError
from dllm.host_completion import HostCompletion
from dllm.host_endpoint import HostPeerListener,connect_peer
from dllm.host_tensor_channel import HostTensorChannel
from dllm.host_tls import client_context,server_context,certificate_pin
from scripts.run_cpu_pipeline_reference import _positive


def parser_for_command():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('role',choices=('worker','complete','serve'))
    parser.add_argument('snapshot')
    parser.add_argument('--address',required=True,help='literal bind IP for worker, peer IP for complete')
    parser.add_argument('--port',type=int,required=True,help='explicit port; worker may use 0 for an ephemeral port')
    parser.add_argument('--server-name',help='TLS server certificate hostname; required for complete')
    for name in ('ca','cert','key','peer-cert'):
        parser.add_argument('--'+name,required=True,help='local TLS file')
    parser.add_argument('--chat-template',help='trusted local text template; both peers must supply matching content')
    parser.add_argument('--model-id',required=True,help='64 hex digits identifying the snapshot; operator asserted unless --verify-model-id')
    parser.add_argument('--verify-model-id',action='store_true',help='hash contained local snapshot before connecting and check it stays unchanged through loading')
    parser.add_argument('--identity-max-mib',type=_positive,default=2048)
    parser.add_argument('--identity-timeout',type=float,default=60.)
    parser.add_argument('--split',type=_positive,required=True,help='first layer owned by output worker')
    parser.add_argument('--epoch',type=_positive,default=1)
    parser.add_argument('--context',type=_positive,default=512)
    parser.add_argument('--context-limit',type=_positive,default=512)
    parser.add_argument('--slots',type=_positive,default=4)
    parser.add_argument('--quantization',choices=('none','q4_0','q8_0'),default='none')
    parser.add_argument('--timeout',type=float,default=10.,help='seconds per connection or transport operation, at most 30')
    parser.add_argument('--http-port',type=int,default=8088,help='serve role: loopback HTTP port, 0 chooses an available port')
    parser.add_argument('--alias',default='model')
    parser.add_argument('--queue-size',type=_positive,default=8)
    parser.add_argument('--queue-timeout',type=float,default=120.)
    parser.add_argument('--client-decode-turns',action='store_true',help='serve role: equal ready decode rows per client')
    parser.add_argument('--interleaved-prefill',action='store_true',help='serve role: one prompt chunk before ready decode')
    parser.add_argument('--idle-timeout',type=float,default=0.,help='serve role: stop after idle seconds; 0 disables')
    parser.add_argument('--heartbeat-interval',type=float,default=5.)
    parser.add_argument('--frame-mib',type=_positive,default=64)
    source=parser.add_mutually_exclusive_group()
    source.add_argument('--prompt')
    source.add_argument('--tokens')
    source.add_argument('--batch-inputs')
    parser.add_argument('--max-new-tokens',type=_positive,default=128)
    parser.add_argument('--prefill-chunk',type=_positive,default=128)
    for name,default in (('weight-mib',256),('cache-mib',256),('workspace-mib',64),
                         ('loader-workspace-mib',64),('projection-workspace-mib',8),('output-mib',64)):
        parser.add_argument('--'+name,type=_positive,default=default)
    return parser


def prepare(args):
    """All request and endpoint metadata is checked before network/payload work."""
    if '%' in args.address:raise ValueError('scoped IP addresses are unsupported')
    address=ipaddress.ip_address(args.address)
    if args.client_decode_turns and args.role!='serve':raise ValueError('client decode turns require the serve role')
    if args.interleaved_prefill and args.role!='serve':raise ValueError('interleaved prefill requires the serve role')
    if (not 0<args.timeout<=30 or not 1<=args.slots<=4 or not 1<=args.frame_mib<=64
            or not 1<=args.context<=args.context_limit<=2**20
            or not (0 if args.role=='worker' else 1)<=args.port<=65535
            or any(getattr(args,n)>2048 for n in ('weight_mib','cache_mib','workspace_mib',
                'loader_workspace_mib','projection_workspace_mib','output_mib'))):
        raise ValueError('invalid CPU host endpoint, context or resource limits')
    if args.role!='worker' and (not args.server_name or len(args.server_name)>253 or '\0' in args.server_name):
        raise ValueError('coordinator requires a TLS server hostname')
    if args.role=='worker' and (args.server_name is not None or any(v is not None for v in (args.prompt,args.tokens,args.batch_inputs))):
        raise ValueError('worker does not accept request inputs or a server hostname')
    if args.role=='serve':
        if (any(v is not None for v in (args.prompt,args.tokens,args.batch_inputs))
                or not 0<=args.http_port<=65535 or not 1<=len(args.alias)<=128
                or not 1<=args.prefill_chunk<=4096
                or not 1<=args.queue_size<=64 or not 0<args.queue_timeout<=300
                or not 0<=args.idle_timeout<=86400 or not 0<args.heartbeat_interval<=10
                or args.heartbeat_interval>=args.timeout):
            raise ValueError('invalid CPU host serving inputs, queue or lifetime bounds')
    if len(args.model_id)!=64 or any(c not in '0123456789abcdefABCDEF' for c in args.model_id):
        raise ValueError('model identity requires exactly 64 hexadecimal digits')
    if (not 1<=args.identity_max_mib<=2**20 or not 0<args.identity_timeout<=3600
            or (not args.verify_model_id and (args.identity_max_mib!=2048 or args.identity_timeout!=60))):
        raise ValueError('identity limits require verification and valid byte/time bounds')
    verified=None
    if args.verify_model_id:
        from dllm.cpu_snapshot_identity import fingerprint
        verified=fingerprint(args.snapshot,max_bytes=args.identity_max_mib*1024**2,timeout=args.identity_timeout)
        if verified.model_id!=args.model_id.lower():raise ValueError('local snapshot does not match model identity')
        args.snapshot=str(verified.root)
    config=read_json(Path(args.snapshot)/'config.json',1024**2)
    if (not isinstance(config,dict) or config.get('model_type')!='qwen3_5_text'
            or 'text_config' in config or 'vision_config' in config):
        raise ValueError('CPU host command requires a text-only Qwen3.5 config')
    c=hybrid_config(config);cpu_hybrid_shapes(c,0,c['num_hidden_layers'])
    if not 1<=args.split<c['num_hidden_layers'] or args.context>c['max_position_embeddings']:
        raise ValueError('invalid CPU host split/context')
    if args.chat_template is None:text=CpuReferenceText(args.snapshot,config)
    else:
        from dllm.cpu_reference_chat import CpuReferenceChat
        text=CpuReferenceChat(args.snapshot,config,args.chat_template)
    inputs=limits=None
    if args.role=='complete':
        inputs,limits=completion_inputs(text,prompt=args.prompt,tokens_file=args.tokens,batch_file=args.batch_inputs,
            max_context=args.context,max_new_tokens=args.max_new_tokens,chunk_size=args.prefill_chunk)
        rows=inputs if args.batch_inputs is not None else [inputs]
        if (len(rows)>args.slots or max(len(rows),min(args.prefill_chunk,max(map(len,rows))))*c['hidden_size']*4>args.frame_mib*1024**2):
            raise ValueError('request exceeds host slots or activation frame capacity')
    if args.role=='serve' and max(args.slots,min(args.prefill_chunk,args.context))*c['hidden_size']*4>args.frame_mib*1024**2:
        raise ValueError('serving chunks exceed activation frame capacity')
    # Bind tokenizer semantics and EOS in addition to the caller's model identity.
    digest=hashlib.sha256(bytes.fromhex(args.model_id))
    digest.update(hashlib.sha256(text.tokenizer.to_str().encode()).digest())
    digest.update(json.dumps(sorted(text.eos_ids)).encode())
    if hasattr(text,'chat_identity'):digest.update(b'cpu-chat-template-v1'+text.chat_identity)
    if verified is not None:
        verified.verify_unchanged()
        digest.update(b'verified-cpu-source-v1')
    context=(server_context if args.role=='worker' else client_context)(ca_file=args.ca,cert_file=args.cert,key_file=args.key)
    pin=certificate_pin(args.peer_cert)
    return address,c,text,inputs,limits,digest.digest(),context,pin,verified


def emit(event):print(json.dumps(event,ensure_ascii=False),flush=True)


def run(args):
    from dllm.http_auth import AuthPolicy
    auth=AuthPolicy.from_env() if args.role=='serve' else None
    address,c,text,inputs,limits,identity,context,pin,verified=prepare(args)
    connection=listener=transport=worker=runner=None
    primary=None
    try:
        if args.role=='worker':
            connection=socket.socket(socket.AF_INET if address.version==4 else socket.AF_INET6,socket.SOCK_STREAM)
            connection.bind((str(address),args.port))
            listener=HostPeerListener(connection,context,peer_pin=pin);connection=None
            emit(dict(scope='cpu_host_reference',event='listening',address=str(address),port=listener.listener.getsockname()[1],official_model_qualified=False))
            connection=listener.accept(timeout=args.timeout)
            listener.close();listener=None
        else:
            connection=connect_peer(str(address),args.port,context,peer_pin=pin,server_hostname=args.server_name,timeout=args.timeout)
        rank=1 if args.role=='worker' else 0
        transport=HostTensorChannel(connection,identity,rank,max_frame_bytes=args.frame_mib*1024**2,timeout=args.timeout)
        connection=None
        try:
            worker=build_cpu_host_worker(args.snapshot,args.split if rank else 0,c['num_hidden_layers'] if rank else args.split,
                transport,epoch=args.epoch,max_context=args.context,context_limit=args.context_limit,max_batch=args.slots,
                quantization=args.quantization,mapped_vocab_size=text.vocab_size,excluded_ids=text.excluded_ids,
                max_weight_bytes=args.weight_mib*1024**2,loader_workspace_bytes=args.loader_workspace_mib*1024**2,
                max_cache_bytes=args.cache_mib*1024**2,max_workspace_bytes=args.workspace_mib*1024**2,
                projection_workspace_bytes=args.projection_workspace_mib*1024**2,max_output_bytes=args.output_mib*1024**2)
        except CpuHostWorkerBuildError as error:
            worker=error
            raise
        if verified is not None:verified.verify_unchanged()
        worker.handshake()
        if args.role!='serve':
            emit(dict(scope='cpu_host_reference',event='ready',role=args.role,quantization=args.quantization,official_model_qualified=False,model_identity_verified=args.verify_model_id,source_model_id=args.model_id.lower(),text_chat=args.chat_template is not None))
        if rank:
            while not worker.closed:worker.step()
            emit(dict(event='stopped'))
        elif args.role=='serve':
            serve_worker_pair(worker,text,args,auth)
        else:
            runner=HostCompletion(worker,eos_ids=text.eos_ids)
            generated=[[] for _ in inputs] if args.batch_inputs is not None else [[]]
            generate=runner.generate_batch if args.batch_inputs is not None else runner.generate
            finals=[]
            with closing(generate(inputs,max_new_tokens=limits,chunk_size=args.prefill_chunk)) as stream:
                for event in stream:
                    row=event.get('row',0)
                    if 'token_id' in event:
                        generated[row].append(event['token_id']);emit(event)
                    else:finals.append(dict(event,text=text.decode(generated[row])))
            # Report command success only after both peers acknowledge shutdown.
            worker.stop()
            for event in finals:emit(event)
    except BaseException as error:primary=error;raise
    finally:
        failures=[]
        for owner in (runner,worker,transport,connection,listener):
            if owner is not None:
                try:owner.close()
                except BaseException as error:failures.append(error)
        if failures:
            if primary is not None:primary.add_note('CPU host command cleanup failed')
            else:raise failures[0]


def serve_worker_pair(worker,text,args,auth):
    """Run one HTTP service lifetime with explicit peer and thread ownership."""
    import signal
    import threading
    from dllm.host_serving import HostServingPair,HostDistributedRuntime,handler_for_host
    from dllm.server import BoundedHTTPServer
    pair=HostServingPair(worker,text,chunk_size=args.prefill_chunk)
    runtime=HostDistributedRuntime(pair,args.alias,args.queue_size,args.queue_timeout,
        client_decode_turns=args.client_decode_turns,interleaved_prefill=args.interleaved_prefill)
    server=thread=None;previous=None;primary=None
    stopping=threading.Event()
    def interrupt(*_):
        stopping.set();runtime.closed.set();worker.channel.close()
    try:
        server=BoundedHTTPServer(('127.0.0.1',args.http_port),handler_for_host(runtime,auth))
        server.daemon_threads=True
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        previous=signal.signal(signal.SIGTERM,interrupt)
        emit(dict(event='http_ready',url=f'http://127.0.0.1:{server.server_port}',
            scope='cpu_host_reference',official_model_qualified=False,model=args.alias,
            model_identity_verified=args.verify_model_id,source_model_id=args.model_id.lower(),text_chat=args.chat_template is not None))
        try:runtime.run(idle_timeout=args.idle_timeout,heartbeat_interval=args.heartbeat_interval)
        except KeyboardInterrupt:interrupt()
        except (OSError,RuntimeError,EOFError,ValueError):
            if not stopping.is_set():raise
        if worker.ready:worker.stop()
        emit(dict(event='stopped',scope='cpu_host_reference'))
    except BaseException as error:primary=error;raise
    finally:
        if previous is not None:signal.signal(signal.SIGTERM,previous)
        errors=[]
        if server is not None:
            try:
                if thread is not None and thread.is_alive():server.shutdown();thread.join(timeout=5)
                server.server_close()
            except BaseException as error:errors.append(error)
        try:runtime.close()
        except BaseException as error:errors.append(error)
        if errors:
            if primary is not None:primary.add_note('CPU host HTTP shutdown cleanup failed')
            else:raise errors[0]


def main(argv=None):
    parser=parser_for_command();args=parser.parse_args(argv)
    try:run(args)
    except BrokenPipeError:return 0
    except KeyboardInterrupt:return 130
    except (ValueError,TypeError,OSError,RuntimeError,MemoryError,EOFError) as error:
        parser.exit(2,f'CPU host reference failed: {error}\n')
    return 0


if __name__=='__main__':raise SystemExit(main())
