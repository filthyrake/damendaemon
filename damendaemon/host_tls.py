"""Mutual TLS and exact certificate pinning for connected internal sockets.

No listener, connector, certificate issuance or plaintext fallback. Once validated,
wrap_peer owns its connected socket and closes it on every handshake/policy error.
Only a successful return transfers ownership to the caller (normally a host channel).
"""
import hashlib
import hmac
import math
import select
import socket
import ssl
import time
from pathlib import Path

ALPN='damendaemon-host/1'
QUOTA_ALPN='damendaemon-quota/1'

def _alpn(value):
    if value not in (ALPN,QUOTA_ALPN):raise ValueError('invalid TLS application protocol')
    return value

def _peer_pins(value,server_side):
    pins=(value,) if isinstance(value,bytes) else value
    if (not isinstance(pins,tuple) or not 1<=len(pins)<=128
            or (not server_side and len(pins)!=1)
            or any(not isinstance(pin,bytes) or len(pin)!=32 for pin in pins)
            or len(set(pins))!=len(pins)):raise ValueError('invalid TLS peer certificate pins')
    return pins


def _context(server,ca_file,cert_file,key_file,password,alpn):
    _alpn(alpn)
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if server else ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version=ssl.TLSVersion.TLSv1_3
    context.verify_mode=ssl.CERT_REQUIRED
    context.check_hostname=not server
    context.load_verify_locations(cafile=ca_file)
    # Empty password prevents OpenSSL from prompting interactively for an
    # encrypted key when no password was supplied by the caller.
    context.load_cert_chain(certfile=cert_file,keyfile=key_file,password='' if password is None else password)
    context.set_alpn_protocols([alpn])
    return context


def client_context(*,ca_file,cert_file,key_file,password=None,alpn=ALPN):
    return _context(False,ca_file,cert_file,key_file,password,alpn)


def server_context(*,ca_file,cert_file,key_file,password=None,alpn=ALPN):
    return _context(True,ca_file,cert_file,key_file,password,alpn)


def certificate_pin(path):
    """SHA-256 DER certificate identity from one bounded public PEM certificate."""
    with Path(path).open('rb') as reader:pem=reader.read(65537)
    if len(pem)>65536:raise ValueError('host certificate PEM exceeds size limit')
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem.decode('ascii'))).digest()


class HostTlsCancelled(RuntimeError):
    pass


def _validate_policy(context,*,server_side,peer_pin,server_hostname,timeout,cancelled=None,alpn=ALPN):
    _peer_pins(peer_pin,server_side);_alpn(alpn)
    if (type(server_side) is not bool or not isinstance(context,ssl.SSLContext)
            or context.protocol!=(ssl.PROTOCOL_TLS_SERVER if server_side else ssl.PROTOCOL_TLS_CLIENT)
            or context.verify_mode!=ssl.CERT_REQUIRED or context.check_hostname!=(not server_side)
            or context.minimum_version<ssl.TLSVersion.TLSv1_3
            or type(timeout) not in (int,float) or not 0<timeout<=30
            or (server_side and server_hostname is not None)
            or (not server_side and (not isinstance(server_hostname,str) or not 1<=len(server_hostname)<=253 or '\0' in server_hostname))
            or (cancelled is not None and not callable(cancelled))):
        raise ValueError('invalid or unsafe host TLS configuration')


def _remaining(deadline,cancelled):
    if cancelled is not None and cancelled():raise HostTlsCancelled('host connection cancelled')
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('host connection deadline exceeded')
    return remaining


def _wait_socket(connection,read,deadline,cancelled):
    while True:
        remaining=_remaining(deadline,cancelled)
        readable,writable,_=select.select([connection] if read else [],[] if read else [connection],[],min(remaining,.05))
        if readable or writable:
            _remaining(deadline,cancelled)
            return


def wrap_peer(connection,context,*,server_side,peer_pin,server_hostname=None,timeout=10.,cancelled=None,_deadline=None,alpn=ALPN):
    """Authenticate a connected TCP peer within a cancellable handshake deadline.

    Both endpoints require mutual TLS, ALPN and exact certificate pins. Servers
    may explicitly admit a bounded tuple of pins; clients pin one authority.
    The default ALPN remains the host-worker protocol. Keep
    contexts immutable and cancellation callbacks prompt/nonblocking. Local
    validation failures leave ownership with the caller; after validation the
    socket is owned until successful return or closed on every failure.
    """
    if (not isinstance(connection,socket.socket) or isinstance(connection,ssl.SSLSocket)
            or connection.family not in (socket.AF_INET,socket.AF_INET6)
            or connection.getsockopt(socket.SOL_SOCKET,socket.SO_TYPE)!=socket.SOCK_STREAM):
        raise ValueError('invalid or unsafe host TLS connection')
    _validate_policy(context,server_side=server_side,peer_pin=peer_pin,server_hostname=server_hostname,
                     timeout=timeout,cancelled=cancelled,alpn=alpn)
    if _deadline is not None and (type(_deadline) not in (int,float) or not math.isfinite(_deadline)):
        raise ValueError('invalid host TLS absolute deadline')
    connection.getpeername() # an unconnected invalid socket remains caller-owned
    deadline=time.monotonic()+timeout
    if _deadline is not None:deadline=min(deadline,_deadline)
    previous_timeout=connection.gettimeout();wrapped=None
    try:
        _remaining(deadline,cancelled)
        wrapped=context.wrap_socket(connection,server_side=server_side,server_hostname=server_hostname,do_handshake_on_connect=False)
        wrapped.setblocking(False)
        while True:
            _remaining(deadline,cancelled)
            try:wrapped.do_handshake();break
            except ssl.SSLWantReadError:_wait_socket(wrapped,True,deadline,cancelled)
            except ssl.SSLWantWriteError:_wait_socket(wrapped,False,deadline,cancelled)
        _remaining(deadline,cancelled)
        if wrapped.version()!='TLSv1.3' or wrapped.selected_alpn_protocol()!=alpn:
            raise ValueError('host TLS version/application protocol mismatch')
        certificate=wrapped.getpeercert(binary_form=True)
        if not certificate or not any(hmac.compare_digest(hashlib.sha256(certificate).digest(),pin) for pin in _peer_pins(peer_pin,server_side)):
            raise ValueError('host TLS peer certificate pin mismatch')
        _remaining(deadline,cancelled)
        wrapped.settimeout(previous_timeout)
        return wrapped
    except BaseException:
        (wrapped if wrapped is not None else connection).close()
        raise

