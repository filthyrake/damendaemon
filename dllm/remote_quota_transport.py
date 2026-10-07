"""Explicit bounded mutual-TLS RPC for an existing remote reservation ledger.

No enrollment or reclamation RPC exists. Each connection carries one framed JSON
request and response. A lost reply is an unknown outcome, never returned quota.
No listener is created until RemoteQuotaServer is explicitly constructed.
"""
from dataclasses import dataclass,field
import ipaddress
import json
import math
import re
import secrets
import socket
import struct
import threading
import time
from .host_tls import QUOTA_ALPN,wrap_peer,_validate_policy
from .remote_reservation_ledger import RemoteReservationLedger,RemoteLedgerUnavailable,RemoteOwnerDenied,RemoteSequenceError,_hex,CODES,MAX_SEQUENCE,MAX_CLAIMS

MAX_REQUEST=4096
MAX_RESPONSE=65536
LENGTH=struct.Struct('!I')


class RemoteQuotaUnavailable(RuntimeError):
    def __init__(self):super().__init__('remote quota outcome unavailable; retry the same request')


class RemoteQuotaRejected(Exception):
    def __init__(self,code):
        super().__init__('remote quota request rejected: '+code);self.code=code


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate RPC field')
        result[key]=value
    return result


def _json(value,maximum):
    data=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    if not 1<=len(data)<=maximum:raise ValueError('quota RPC message exceeds bound')
    return data


def _deadline(connection,deadline):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('quota operation deadline exceeded')
    connection.settimeout(min(remaining,.1))


def _read(connection,size,deadline):
    data=bytearray()
    while len(data)<size:
        _deadline(connection,deadline)
        try:piece=connection.recv(size-len(data))
        except TimeoutError:continue
        if not piece:raise EOFError('quota peer closed')
        data.extend(piece)
    return bytes(data)


def _receive(connection,maximum,deadline):
    size=LENGTH.unpack(_read(connection,4,deadline))[0]
    if not 1<=size<=maximum:raise ValueError('invalid quota RPC frame length')
    return json.loads(_read(connection,size,deadline).decode('utf-8'),object_pairs_hook=_unique,
                      parse_constant=lambda _:(_ for _ in ()).throw(ValueError('nonfinite JSON')))


def _send(connection,payload,deadline):
    data=memoryview(LENGTH.pack(len(payload))+payload)
    try:
        offset=0
        while offset<len(data):
            _deadline(connection,deadline)
            try:count=connection.send(data[offset:])
            except TimeoutError:continue
            if not count:raise EOFError('quota peer closed')
            offset+=count
    finally:data.release()


def _address(address,*,listening=False):
    if (not isinstance(address,tuple) or len(address)!=2 or not isinstance(address[0],str)
            or type(address[1]) is not int or not (0 if listening else 1)<=address[1]<=65535):raise ValueError('invalid quota address')
    # Separate certificate hostname from a literal connect address; DNS lookup
    # must not escape the bounded socket-operation deadline.
    return ipaddress.ip_address(address[0]).version


def _timeout(value):
    if type(value) not in (int,float) or not math.isfinite(value) or not 0<value<=30:raise ValueError('invalid quota timeout')
    return float(value)


@dataclass(frozen=True)
class QuotaRequest:
    """Immutable wire snapshot for retry. Repr never contains credentials."""
    payload: bytes=field(repr=False)


class RemoteQuotaClient:
    def __init__(self,address,context,peer_pin,server_hostname,ledger,owner,credential,*,timeout=5.):
        _address(address);_timeout(timeout)
        _validate_policy(context,server_side=False,peer_pin=peer_pin,server_hostname=server_hostname,timeout=timeout,alpn=QUOTA_ALPN)
        if not _hex(ledger,32):raise ValueError('invalid quota ledger identity')
        RemoteReservationLedger._credentials(owner,credential)
        self.address,self.context,self.peer_pin,self.server_hostname=address,context,peer_pin,server_hostname
        self.ledger,self.owner,self._credential=ledger,owner,credential
        self.timeout=float(timeout)

    def request(self,method,**options):
        fields={'execute':{'sequence','command'},'status':set(),'policy':set(),'list':{'after','limit'}}
        if not isinstance(method,str) or method not in fields or set(options)!=fields[method]:raise ValueError('invalid quota request')
        if method=='execute':options=dict(options,operation_id=secrets.token_hex(16))
        return QuotaRequest(_json(dict(version=1,ledger=self.ledger,owner=self.owner,credential=self._credential.hex(),method=method,**options),MAX_REQUEST))

    def call(self,request):
        if not isinstance(request,QuotaRequest) or not isinstance(request.payload,bytes) or not 1<=len(request.payload)<=MAX_REQUEST:
            raise ValueError('invalid quota request snapshot')
        # Only this owner/authority's snapshots may be sent on this connection.
        try:body=json.loads(request.payload,object_pairs_hook=_unique)
        except (ValueError,TypeError,UnicodeError,RecursionError):raise ValueError('invalid quota request snapshot') from None
        if (not isinstance(body,dict) or body.get('ledger')!=self.ledger or body.get('owner')!=self.owner
                or body.get('credential')!=self._credential.hex()):raise ValueError('quota request belongs to another owner')
        connection=None;deadline=time.monotonic()+self.timeout
        try:
            connection=socket.create_connection(self.address,timeout=self.timeout)
            connection=wrap_peer(connection,self.context,server_side=False,peer_pin=self.peer_pin,server_hostname=self.server_hostname,
                                 timeout=self.timeout,_deadline=deadline,alpn=QUOTA_ALPN)
            _send(connection,request.payload,deadline);response=_receive(connection,MAX_RESPONSE,deadline)
            if (not isinstance(response,dict) or set(response) not in ({'version','ledger','operation_id','result'},{'version','ledger','operation_id','error'})
                    or type(response.get('version')) is not int or response['version']!=1 or response.get('ledger')!=self.ledger or response.get('operation_id')!=body.get('operation_id')):raise RemoteQuotaUnavailable()
            if 'error' in response:
                if response['error'] not in ('owner_denied','sequence_conflict','invalid_request','ledger_unavailable','identity_mismatch'):raise RemoteQuotaUnavailable()
                if response['error']=='ledger_unavailable':raise RemoteQuotaUnavailable()
                raise RemoteQuotaRejected(response['error'])
            self._result(body,response['result'])
            return response['result']
        except (RemoteQuotaUnavailable,RemoteQuotaRejected):raise
        except (OSError,EOFError,ValueError,TypeError,RecursionError):raise RemoteQuotaUnavailable() from None
        finally:
            if connection is not None:connection.close()

    def _receipt(self,result,sequence,command=None):
        if (not isinstance(result,dict) or type(result.get('sequence')) is not int or result['sequence']!=sequence
                or type(result.get('ok')) is not bool or not isinstance(result.get('code'),str) or result['code'] not in CODES):raise RemoteQuotaUnavailable()
        code=result['code'];claimed=code in ('reserved','resized','released')
        if set(result)!=({'sequence','ok','code','claim'} if claimed else {'sequence','ok','code'}):raise RemoteQuotaUnavailable()
        if result['ok']!=(claimed or code=='retired'):raise RemoteQuotaUnavailable()
        if claimed and not RemoteReservationLedger._claim_id(result['claim'],self.owner):raise RemoteQuotaUnavailable()
        if command is not None:
            allowed={'reserve':{'reserved','request_limit','output_limit','claim_limit'},
                     'resize':{'resized','request_limit','output_limit','unknown_claim','empty_claim'},
                     'release':{'released'},'retire':{'retired','owner_busy'}}
            if not isinstance(command,dict) or command.get('op') not in allowed or code not in allowed[command['op']]:raise RemoteQuotaUnavailable()
            if claimed:
                expected=self.owner+':'+str(sequence) if command['op']=='reserve' else command.get('claim')
                if result['claim']!=expected:raise RemoteQuotaUnavailable()

    def _result(self,request,result):
        if not isinstance(result,dict):raise RemoteQuotaUnavailable()
        if request.get('method')=='execute':
            self._receipt(result,request.get('sequence'),request.get('command'))
        elif request.get('method')=='status':
            if (set(result)!={'ledger','sequence','retired','claims','receipt'} or result['ledger']!=self.ledger
                    or type(result['sequence']) is not int or not 0<=result['sequence']<=MAX_SEQUENCE
                    or type(result['retired']) is not bool or type(result['claims']) is not int or not 0<=result['claims']<=MAX_CLAIMS
                    or (result['retired'] and result['claims'])):raise RemoteQuotaUnavailable()
            if result['sequence']==0:
                if result['receipt'] is not None or result['retired'] or result['claims']:raise RemoteQuotaUnavailable()
            else:
                self._receipt(result['receipt'],result['sequence'])
                if result['retired']!=(result['receipt']['code']=='retired'):raise RemoteQuotaUnavailable()
        elif request.get('method')=='policy':
            if set(result)!={'clients'} or not isinstance(result['clients'],list) or not 1<=len(result['clients'])<=128:raise RemoteQuotaUnavailable()
            names=[]
            for row in result['clients']:
                if (not isinstance(row,dict) or set(row)!={'name','max_inflight','max_reserved_output_tokens'}
                        or not isinstance(row['name'],str) or re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',row['name']) is None
                        or type(row['max_inflight']) is not int or not 1<=row['max_inflight']<=128
                        or (row['max_reserved_output_tokens'] is not None and (type(row['max_reserved_output_tokens']) is not int
                            or not 1<=row['max_reserved_output_tokens']<=2**31-1))):raise RemoteQuotaUnavailable()
                names.append(row['name'])
            if names!=sorted(set(names)):raise RemoteQuotaUnavailable()
        elif request.get('method')=='list':
            limit=request.get('limit')
            if (set(result)!={'claims','after'} or type(limit) is not int or not 1<=limit<=128
                    or not isinstance(result['claims'],list) or len(result['claims'])>limit):raise RemoteQuotaUnavailable()
            previous=request.get('after') or ''
            for row in result['claims']:
                if (not isinstance(row,dict) or set(row)!={'claim','client','requests','outputs'}
                        or not RemoteReservationLedger._claim_id(row['claim'],self.owner) or row['claim']<=previous
                        or not isinstance(row['client'],str) or re.fullmatch(r'[A-Za-z0-9_.-]{1,64}',row['client']) is None
                        or type(row['requests']) is not int or row['requests'] not in (0,1)
                        or type(row['outputs']) is not int or not 0<=row['outputs']<=2**31-1
                        or row['requests']+row['outputs']==0):raise RemoteQuotaUnavailable()
                previous=row['claim']
            if result['after'] is not None and (len(result['claims'])!=limit or result['after']!=previous):raise RemoteQuotaUnavailable()
        else:raise RemoteQuotaUnavailable()


class RemoteQuotaServer:
    """Own a ledger and listener after successful construction; no auto threads.

    Call serve_forever in one owned thread, then close. Close interrupts sockets
    and waits at most two seconds; failed worker cleanup remains retryable. No
    remote claims are released, even on a successful authority shutdown.
    """
    def __init__(self,address,ledger,context,peer_pins,*,max_connections=16,timeout=5.):
        version=_address(address,listening=True);_timeout(timeout)
        _validate_policy(context,server_side=True,peer_pin=peer_pins,server_hostname=None,timeout=timeout,alpn=QUOTA_ALPN)
        if type(max_connections) is not int or not 1<=max_connections<=128:raise ValueError('invalid quota connection cap')
        if not isinstance(ledger,RemoteReservationLedger) or ledger.closed:raise ValueError('quota authority requires an open ledger')
        self.ledger,self.context,self.peer_pins,self.timeout=ledger,context,peer_pins,float(timeout)
        self.stopping=threading.Event();self.lock=threading.Lock();self.workers={};self.acceptor=None;self.closed=False
        self.capacity=threading.BoundedSemaphore(max_connections)
        listener=socket.socket(socket.AF_INET6 if version==6 else socket.AF_INET,socket.SOCK_STREAM)
        try:
            listener.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);listener.bind(address);listener.listen(max_connections);listener.settimeout(.1)
        except BaseException:listener.close();raise
        self.listener=listener;self.address=listener.getsockname()[:2]

    def serve_forever(self):
        with self.lock:
            if self.acceptor is not None or self.stopping.is_set():raise RuntimeError('quota authority already serving or stopping')
            self.acceptor=threading.current_thread()
        while not self.stopping.is_set():
            try:connection,_=self.listener.accept()
            except TimeoutError:continue
            except OSError:
                if self.stopping.is_set():return
                raise
            if not self.capacity.acquire(blocking=False):connection.close();continue
            thread=threading.Thread(target=self._serve,args=(connection,),daemon=True)
            with self.lock:
                if self.stopping.is_set():connection.close();self.capacity.release();return
                self.workers[thread]=connection
                try:thread.start()
                except BaseException:self.workers.pop(thread);connection.close();self.capacity.release();raise

    def _dispatch(self,body):
        if not isinstance(body,dict) or type(body.get('version')) is not int or body['version']!=1:raise ValueError('invalid quota request')
        method=body.get('method');fields={'execute':{'sequence','command','operation_id'},'status':set(),'policy':set(),'list':{'after','limit'}}
        common={'version','ledger','owner','credential','method'}
        if not isinstance(method,str) or method not in fields or set(body)!=common|fields[method]:raise ValueError('invalid quota request')
        if body['ledger']!=self.ledger.identity:return dict(error='identity_mismatch')
        if not _hex(body['credential'],64):raise RemoteOwnerDenied()
        owner,credential=body['owner'],bytes.fromhex(body['credential'])
        if method=='execute' and not _hex(body['operation_id'],32):raise ValueError('invalid operation identity')
        if method=='execute':result=self.ledger.execute(owner,credential,body['sequence'],body['command'],operation_id=body['operation_id'])
        elif method=='status':result=self.ledger.status(owner,credential)
        elif method=='policy':result=self.ledger.policy(owner,credential)
        else:result=self.ledger.list_claims(owner,credential,after=body['after'],limit=body['limit'])
        return dict(result=result)

    def _reply(self,connection,response,deadline):
        _send(connection,_json(dict(version=1,ledger=self.ledger.identity,**response),MAX_RESPONSE),deadline)

    def _serve(self,connection):
        thread=threading.current_thread();deadline=time.monotonic()+self.timeout
        try:
            connection=wrap_peer(connection,self.context,server_side=True,peer_pin=self.peer_pins,timeout=self.timeout,
                                 cancelled=self.stopping.is_set,_deadline=deadline,alpn=QUOTA_ALPN)
            with self.lock:
                self.workers[thread]=connection
                if self.stopping.is_set():return
            body=None
            try:
                body=_receive(connection,MAX_REQUEST,deadline);response=self._dispatch(body)
            except RemoteOwnerDenied:response=dict(error='owner_denied')
            except RemoteSequenceError:response=dict(error='sequence_conflict')
            except RemoteLedgerUnavailable:response=dict(error='ledger_unavailable')
            except (ValueError,TypeError,KeyError,RecursionError):response=dict(error='invalid_request')
            operation=body.get('operation_id') if isinstance(body,dict) and body.get('method')=='execute' else None
            response['operation_id']=operation if _hex(operation,32) else None
            self._reply(connection,response,deadline)
        except (OSError,EOFError,ValueError,RuntimeError):pass
        finally:
            connection.close()
            with self.lock:self.workers.pop(thread,None)
            self.capacity.release()

    def close(self):
        self.stopping.set();self.listener.close()
        with self.lock:workers=list(self.workers.items());acceptor=self.acceptor
        for _,connection in workers:
            try:connection.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            connection.close()
        deadline=time.monotonic()+2
        for thread in [*(thread for thread,_ in workers),*([acceptor] if acceptor is not None else [])]:
            if thread is not threading.current_thread():thread.join(max(0,deadline-time.monotonic()))
        with self.lock:
            if self.workers or (acceptor is not None and acceptor is not threading.current_thread() and acceptor.is_alive()):
                raise RuntimeError('quota authority cleanup remains active')
            self.ledger.close();self.closed=True
