"""Explicit quota bootstrap, local enrollment, mutual TLS serving and recovery."""
import argparse
import json
import signal
import threading
from dllm.remote_quota_config import authority_policy,authority_listener,frontend_client,owner_credential,recovery_request
from dllm.remote_quota_transport import RemoteQuotaServer
from dllm.remote_reservation_ledger import RemoteReservationLedger


def parser_for_command():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='action',required=True)
    for name in ('init','register','serve'):
        command=commands.add_parser(name)
        command.add_argument('--policy',required=True,help='private absolute policy JSON path')
        if name=='register':
            command.add_argument('--owner',required=True)
            command.add_argument('--client',action='append',required=True)
        if name=='serve':command.add_argument('--listen',required=True,help='private mutual TLS listener JSON path')
    for name in ('status','claims','recover'):
        command=commands.add_parser(name)
        command.add_argument('--config',required=True,help='private frontend quota JSON path')
        if name=='claims':
            command.add_argument('--after',default=None);command.add_argument('--limit',type=int,default=128)
        if name=='recover':command.add_argument('--operation',required=True,help='private immutable release/retire operation JSON')
    return parser


def emit(value):print(json.dumps(value,sort_keys=True),flush=True)


def serve(ledger,listener):
    address,context,pins,options=listener
    server=None;thread=None;previous={};stopping=threading.Event();finished=threading.Event();failures=[]
    def interrupt(signum,frame):stopping.set()
    def run():
        try:server.serve_forever()
        except BaseException as error:failures.append(error)
        finally:finished.set()
    try:
        server=RemoteQuotaServer(address,ledger,context,pins,**options)
        for signum in (signal.SIGINT,signal.SIGTERM):previous[signum]=signal.signal(signum,interrupt)
        thread=threading.Thread(target=run,daemon=False);thread.start()
        emit(dict(event='quota_ready',address=server.address[0],port=server.address[1],ledger=ledger.identity))
        while not stopping.wait(.1):
            if finished.is_set():raise RuntimeError('quota listener ended unexpectedly')
    finally:
        try:
            if server is not None:
                # A busy ledger worker retains ownership; retry the same close.
                for attempt in range(3):
                    try:server.close();break
                    except RuntimeError:
                        if attempt==2:raise
            else:ledger.close()
        finally:
            for signum,handler in previous.items():signal.signal(signum,handler)
    if failures:raise RuntimeError('quota listener failed')
    emit(dict(event='stopped',scope='quota_authority',remote_claims_retained=True))


def main(argv=None):
    parser=parser_for_command();args=parser.parse_args(argv)
    try:
        if args.action in ('status','claims','recover'):
            client=frontend_client(args.config)
            if args.action=='recover':request=recovery_request(client,args.operation)
            elif args.action=='claims':request=client.request('list',after=args.after,limit=args.limit)
            else:request=client.request('status')
            result=client.call(request);emit(result)
            return 0 if result.get('ok',True) else 1
        path,clients=authority_policy(args.policy)
        listener=authority_listener(args.listen) if args.action=='serve' else None
        credential=owner_credential() if args.action=='register' else None
        ledger=RemoteReservationLedger(path,clients,create=args.action=='init')
        if args.action=='serve':
            serve(ledger,listener)
            return 0
        try:
            if args.action=='register':
                ledger.register(args.owner,credential,args.client)
                emit(dict(event='owner_registered',ledger=ledger.identity,owner=args.owner))
            else:emit(dict(event='quota_initialized',ledger=ledger.identity))
        finally:ledger.close()
        return 0
    except (OSError,ValueError,TypeError,RuntimeError):
        # Private file parsing/TLS errors may contain inputs. Never echo them.
        parser.exit(2,'quota command failed; verify private configuration, authority state and retained operation identity\n')
    except Exception:
        parser.exit(2,'quota operation rejected; inspect owner state before retrying the same operation\n')


if __name__=='__main__':raise SystemExit(main())
