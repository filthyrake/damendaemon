"""Explicit private configuration for quota authority and frontend commands."""
import json
import os
from pathlib import Path
import secrets
import stat
from .host_tls import QUOTA_ALPN,client_context,server_context
from .http_auth import ClientKey
from .remote_quota_transport import RemoteQuotaClient,_address,_timeout,_unique
from .remote_reservation_ledger import _hex


def private_json(path):
    """Bounded, owned regular input; errors never quote file contents."""
    try:
        path=Path(path)
        if not path.is_absolute() or not hasattr(os,'O_NOFOLLOW'):raise ValueError()
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            info=os.fstat(fd)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1
                    or info.st_mode&0o077 or not 1<=info.st_size<=65536):raise ValueError()
            with os.fdopen(fd,'rb',closefd=False) as reader:data=reader.read(65537)
            if len(data)>65536:raise ValueError()
        finally:os.close(fd)
        result=json.loads(data,object_pairs_hook=_unique,
                          parse_constant=lambda _:(_ for _ in ()).throw(ValueError()))
        if not isinstance(result,dict):raise ValueError()
        return result
    except (OSError,ValueError,TypeError,RecursionError):
        raise ValueError('quota configuration requires a bounded private owned JSON file') from None


def _fields(raw,required,optional=()):
    if not set(required)<=set(raw) or set(raw)-set(required)-set(optional):raise ValueError('invalid quota configuration fields')


def owner_credential():
    value=os.environ.get('DLLM_REMOTE_QUOTA_OWNER_KEY')
    if not _hex(value,64):raise ValueError('DLLM_REMOTE_QUOTA_OWNER_KEY requires 32 bytes encoded as lowercase hex')
    return bytes.fromhex(value)


def _tls(raw,server):
    for name in ('ca_file','cert_file','key_file'):
        value=raw[name]
        if not isinstance(value,str) or not Path(value).is_absolute():raise ValueError('quota TLS paths must be absolute')
    try:
        return (server_context if server else client_context)(
            **{name:raw[name] for name in ('ca_file','cert_file','key_file')},
            password=os.environ.get('DLLM_REMOTE_QUOTA_TLS_PASSWORD'),alpn=QUOTA_ALPN)
    except (OSError,ValueError):raise ValueError('could not load quota mutual TLS credentials') from None


def frontend_client(path):
    raw=private_json(path)
    _fields(raw,('address','port','server_hostname','ledger','owner','server_pin','ca_file','cert_file','key_file'),('timeout',))
    if not _hex(raw['server_pin'],64):raise ValueError('invalid quota certificate pin')
    # Validate routing and identity before loading TLS material or connecting.
    _address((raw['address'],raw['port']));_timeout(raw.get('timeout',5))
    if not _hex(raw['ledger'],32) or not _hex(raw['owner'],32):raise ValueError('invalid quota ledger or owner identity')
    credential=owner_credential()
    return RemoteQuotaClient((raw['address'],raw['port']),_tls(raw,False),bytes.fromhex(raw['server_pin']),
        raw['server_hostname'],raw['ledger'],raw['owner'],credential,timeout=raw.get('timeout',5))


def authority_policy(path):
    raw=private_json(path);_fields(raw,('path','clients'))
    if not isinstance(raw['path'],str) or not Path(raw['path']).is_absolute():raise ValueError('quota ledger path must be absolute')
    rows=raw['clients']
    if not isinstance(rows,list) or not 1<=len(rows)<=128:raise ValueError('quota policy requires 1..128 clients')
    clients=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError('invalid quota client policy')
        _fields(row,('name','max_inflight','max_reserved_output_tokens'))
        # Ledger policies consume only names and caps, never frontend API keys.
        clients.append(ClientKey(key=secrets.token_urlsafe(32),**row))
    if len({c.name for c in clients})!=len(clients):raise ValueError('duplicate quota client policy')
    return raw['path'],tuple(clients)


def authority_listener(path):
    raw=private_json(path)
    _fields(raw,('address','port','client_pins','ca_file','cert_file','key_file'),('timeout','max_connections'))
    address=(raw['address'],raw['port']);_address(address,listening=True)
    timeout=_timeout(raw.get('timeout',5));maximum=raw.get('max_connections',16)
    if type(maximum) is not int or not 1<=maximum<=128:raise ValueError('invalid quota connection cap')
    pins=raw['client_pins']
    if (not isinstance(pins,list) or not 1<=len(pins)<=128 or any(not _hex(p,64) for p in pins)
            or len(set(pins))!=len(pins)):raise ValueError('invalid quota client certificate pins')
    return address,_tls(raw,True),tuple(bytes.fromhex(p) for p in pins),dict(timeout=timeout,max_connections=maximum)


def recovery_request(client,path):
    """Non-secret immutable operation file can be reused after a lost reply."""
    from .remote_quota_transport import QuotaRequest,_json,MAX_REQUEST
    from .remote_reservation_ledger import MAX_SEQUENCE,RemoteReservationLedger
    raw=private_json(path);_fields(raw,('ledger','owner','sequence','operation_id','command'))
    command=raw['command']
    if (raw['ledger']!=client.ledger or raw['owner']!=client.owner or not _hex(raw['operation_id'],32)
            or type(raw['sequence']) is not int or not 1<=raw['sequence']<=MAX_SEQUENCE
            or not isinstance(command,dict)):raise ValueError('invalid quota recovery identity')
    if command=={'op':'retire'}:pass
    elif set(command)=={'op','claim'} and command['op']=='release' and RemoteReservationLedger._claim_id(command['claim'],client.owner):pass
    else:raise ValueError('recovery permits only explicit release or retirement')
    return QuotaRequest(_json(dict(version=1,credential=client._credential.hex(),method='execute',**raw),MAX_REQUEST))
