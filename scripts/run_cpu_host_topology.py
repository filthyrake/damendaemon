"""Explicit multi-process CPU topology reference; no discovery or provisioning."""
from contextlib import closing
import ipaddress
from pathlib import Path
import signal
import socket
from types import SimpleNamespace

from dllm.cpu_host_pipeline_build import build_cpu_host_pipeline_worker,CpuHostPipelineBuildError
from dllm.cpu_reference_text import read_json
from dllm.host_completion import HostCompletion
from dllm.host_endpoint import HostPeerListener,connect_peer
from dllm.host_serving import HostServingTopology
from dllm.host_tensor_channel import HostTensorChannel
from dllm.host_tls import certificate_pin
from dllm.host_topology import HostTopology
from scripts.run_cpu_host_reference import parser_for_command,prepare as prepare_pair,emit,serve_worker_pair


def prepare(args,*,family='qwen'):
    manifest=read_json(args.plan,2**20)
    if not isinstance(manifest,dict) or set(manifest)!= {'ranges'}:
        raise ValueError('CPU topology plan must contain exactly the ranges field')
    ranges=manifest['ranges'];end=0
    if not isinstance(ranges,list) or not 2<=len(ranges)<=8:raise ValueError('CPU topology needs 2..8 ranges')
    for pair in ranges:
        if (not isinstance(pair,list) or len(pair)!=2 or any(type(v) is not int for v in pair)
                or pair[0]!=end or not end<pair[1]<=256):
            raise ValueError('CPU topology ranges must be contiguous, nonempty and ordered')
        end=pair[1]
    if type(args.index) is not int or not 0<=args.index<len(ranges):raise ValueError('invalid CPU topology stage index')
    peers=[];local=SimpleNamespace(**vars(args));local.split=ranges[0][1]
    if args.role=='worker':
        if args.index==0 or args.peers is not None or args.address is None or args.port is None or args.peer_cert is None:
            raise ValueError('worker requires nonzero index, bind address/port and coordinator certificate')
    else:
        if args.index!=0 or args.peers is None or any(v is not None for v in (args.address,args.port,args.peer_cert,args.server_name)):
            raise ValueError('coordinator requires only an ordered peers file and index zero')
        values=read_json(args.peers,2**20)
        if not isinstance(values,list) or len(values)!=len(ranges)-1:raise ValueError('peer count differs from topology')
        for index,peer in enumerate(values,1):
            if (not isinstance(peer,dict) or set(peer)!= {'stage','address','port','server_name','peer_cert'}
                    or type(peer['stage']) is not int or peer['stage']!=index
                    or not isinstance(peer['address'],str) or '%' in peer['address']
                    or type(peer['port']) is not int or not 1<=peer['port']<=65535
                    or not isinstance(peer['server_name'],str) or not 1<=len(peer['server_name'])<=253 or '\0' in peer['server_name']
                    or not isinstance(peer['peer_cert'],str) or not peer['peer_cert']):
                raise ValueError('invalid ordered CPU topology peer metadata')
            address=str(ipaddress.ip_address(peer['address']))
            cert=Path(peer['peer_cert'])
            if not cert.is_absolute():cert=Path(args.peers).resolve().parent/cert
            peers.append(dict(peer,address=address,peer_cert=str(cert),pin=certificate_pin(cert)))
        first=peers[0]
        local.address,local.port,local.server_name,local.peer_cert=first['address'],first['port'],first['server_name'],first['peer_cert']
    address,config,text,inputs,limits,session,context,pin,verified=(prepare_pair(local) if family=='qwen' else prepare_pair(local,family=family))
    if family=='flash':
        from dllm.glm_flash_host_generation import native_excluded_ids
        exclusions=native_excluded_ids(text)
    else:exclusions=() if getattr(args,'native_chat',False) else text.excluded_ids
    plan=HostTopology(config,ranges,bytes.fromhex(args.model_id),quantization=args.quantization,
        context=args.context,context_limit=args.context_limit,slots=args.slots,chunk_size=args.prefill_chunk,
        frame_bytes=args.frame_mib*2**20,
        mapped_vocab_size=text.tokenizer_size if getattr(args,'native_chat',False) else text.vocab_size,
        excluded_ids=exclusions,session_identity=session)
    if getattr(args,'native_chat',False) and inputs is not None and any(set(prompt)&set(plan.excluded_ids) for prompt in inputs.prompts):
        raise ValueError('native host request contains excluded media/control IDs')
    if verified is not None:verified.verify_unchanged()
    args.snapshot=local.snapshot
    return address,plan,text,inputs,limits,context,pin,verified,peers


def run(args,*,family='qwen'):
    from dllm.http_auth import AuthPolicy
    auth=AuthPolicy.from_env() if args.role=='serve' else None
    address,plan,text,inputs,limits,context,pin,verified,peers=(prepare(args) if family=='qwen' else prepare(args,family=family))
    scope=f'{family}_cpu_host_topology_reference' if family!='qwen' else 'cpu_host_topology_reference'
    connections=[];transports=[];listener=worker=runner=None;primary=None
    try:
        if args.role=='worker':
            connection=socket.socket(socket.AF_INET if address.version==4 else socket.AF_INET6,socket.SOCK_STREAM)
            connections.append(connection);connection.bind((str(address),args.port))
            listener=HostPeerListener(connection,context,peer_pin=pin);connections.pop()
            emit(dict(scope=scope,event='listening',stage=args.index,
                address=str(address),port=listener.listener.getsockname()[1],official_model_qualified=False))
            connection=listener.accept(timeout=args.timeout);connections.append(connection)
            listener.close();listener=None
            transport=HostTensorChannel(connection,plan.stage_identity(args.index),1,max_frame_bytes=plan.frame_bytes,timeout=args.timeout)
            transports.append(transport);connections.pop()
        else:
            for peer in peers:
                connection=connect_peer(peer['address'],peer['port'],context,peer_pin=peer['pin'],
                    server_hostname=peer['server_name'],timeout=args.timeout)
                connections.append(connection)
                transport=HostTensorChannel(connection,plan.stage_identity(peer['stage']),0,max_frame_bytes=plan.frame_bytes,timeout=args.timeout)
                transports.append(transport);connections.pop()
        try:
            options=dict(epoch=args.epoch,max_weight_bytes=args.weight_mib*2**20,
                max_cache_bytes=args.cache_mib*2**20,max_workspace_bytes=args.workspace_mib*2**20,
                projection_workspace_bytes=args.projection_workspace_mib*2**20,max_output_bytes=args.output_mib*2**20)
            if family=='flash':
                from dllm.glm_flash_host_pipeline_build import build_glm_flash_host_pipeline_worker
                worker=build_glm_flash_host_pipeline_worker(args.snapshot,plan,args.index,transports,**options)
            elif family=='glm':
                from dllm.glm_host_pipeline_build import build_glm_host_pipeline_worker
                worker=build_glm_host_pipeline_worker(args.snapshot,plan,args.index,transports,**options)
            else:
                worker=build_cpu_host_pipeline_worker(args.snapshot,plan,args.index,transports,
                    loader_workspace_bytes=args.loader_workspace_mib*2**20,**options)
        except CpuHostPipelineBuildError as error:worker=error;raise
        # Hashing is performed before connecting; the builder loads the pinned
        # root, then this receipt rechecks every selected file before handshake.
        if verified is not None:verified.verify_unchanged()
        worker.model_identity_verified=verified is not None
        worker.handshake()
        if args.role!='serve':
            emit(dict(scope=scope,event='ready',role=args.role,stage=args.index,
                stage_layers=[list(pair) for pair in plan.ranges],topology_identity=plan.identity.hex(),
                model_identity_verified=worker.model_identity_verified,source_model_id=plan.source_identity.hex(),
                official_model_qualified=False,**({'native_chat':args.native_chat} if family in ('glm','flash') else {})))
        if args.role=='worker':
            while not worker.closed:worker.step()
            emit(dict(event='stopped',stage=args.index))
        elif args.role=='serve':
            if family=='glm':
                from scripts.glm_host_http import serve
                serve(worker,text,args,auth)
            else:serve_worker_pair(worker,text,args,auth,pair_type=HostServingTopology)
        elif family in ('glm','flash') and args.native_chat:
            if family=='flash':
                from dllm.glm_flash_host_generation import HostFlashGenerator as Runner
            else:
                from dllm.glm_host_generation import HostGlmGenerator as Runner
            runner=Runner(worker,text,chunk_size=args.prefill_chunk)
            finals=[]
            with closing(inputs.generate(runner)) as stream:
                for event in stream:
                    if 'finish_reason' in event:finals.append(event)
                    else:emit(event)
            worker.stop()
            for event in finals:emit(event)
        else:
            runner=HostCompletion(worker,eos_ids=text.eos_ids)
            generated=[[] for _ in inputs] if args.batch_inputs is not None else [[]]
            generate=runner.generate_batch if args.batch_inputs is not None else runner.generate
            finals=[]
            with closing(generate(inputs,max_new_tokens=limits,chunk_size=args.prefill_chunk)) as stream:
                for event in stream:
                    row=event.get('row',0)
                    if 'token_id' in event:generated[row].append(event['token_id']);emit(event)
                    else:finals.append(dict(event,text=text.decode(generated[row])))
            worker.stop()
            for event in finals:emit(event)
    except BaseException as error:primary=error;raise
    finally:
        failures=[]
        for owner in (runner,worker,*transports,*connections,listener):
            if owner is not None:
                try:owner.close()
                except BaseException as error:failures.append(error)
        if failures:
            if primary is not None:primary.add_note('CPU topology command cleanup failed')
            else:raise failures[0]


def main(argv=None,*,family='qwen'):
    parser=parser_for_command(topology=True,family=family);parser.description=__doc__;args=parser.parse_args(argv)
    def interrupt(*_):raise KeyboardInterrupt
    previous=signal.signal(signal.SIGTERM,interrupt)
    try:
        try:
            if family=='qwen':run(args)
            else:run(args,family=family)
        except BrokenPipeError:return 0
        except KeyboardInterrupt:return 130
        except (ValueError,TypeError,OSError,RuntimeError,MemoryError,EOFError) as error:
            parser.exit(2,f'CPU host topology failed: {error}\n')
        return 0
    finally:signal.signal(signal.SIGTERM,previous)


if __name__=='__main__':raise SystemExit(main())
