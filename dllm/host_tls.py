"""Mutual TLS and exact certificate pinning for connected host-worker sockets.

No listener, connector, certificate issuance or plaintext fallback. Once validated,
wrap_peer owns its connected socket and closes it on every handshake/policy error.
Only a successful return transfers ownership to the caller (normally a host channel).
"""
import hashlib
import hmac
import socket
import ssl
import time
from pathlib import Path

ALPN='dllm-host/1'


def _context(server,ca_file,cert_file,key_file,password):
    context=ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER if server else ssl.PROTOCOL_TLS_CLIENT)
    context.minimum_version=ssl.TLSVersion.TLSv1_3
    context.verify_mode=ssl.CERT_REQUIRED
    context.check_hostname=not server
    context.load_verify_locations(cafile=ca_file)
    # Empty password prevents OpenSSL from prompting interactively for an
    # encrypted key when no password was supplied by the caller.
    context.load_cert_chain(certfile=cert_file,keyfile=key_file,password='' if password is None else password)
    context.set_alpn_protocols([ALPN])
    return context


def client_context(*,ca_file,cert_file,key_file,password=None):
    return _context(False,ca_file,cert_file,key_file,password)


def server_context(*,ca_file,cert_file,key_file,password=None):
    return _context(True,ca_file,cert_file,key_file,password)


def certificate_pin(path):
    """SHA-256 DER certificate identity from one bounded public PEM certificate."""
    with Path(path).open('rb') as reader:pem=reader.read(65537)
    if len(pem)>65536:raise ValueError('host certificate PEM exceeds size limit')
    return hashlib.sha256(ssl.PEM_cert_to_DER_cert(pem.decode('ascii'))).digest()


def wrap_peer(connection,context,*,server_side,peer_pin,server_hostname=None,timeout=10.):
    """Authenticate and pin a connected TCP peer within one handshake deadline.

    Both endpoints must use this policy; the server requires a trusted client
    certificate, and the client also checks the server hostname. The model/session
    identity handshake remains necessary after TLS. Keep contexts immutable after
    configuration; do not share a socket with another owner during wrapping.
    """
    if (not isinstance(connection,socket.socket) or isinstance(connection,ssl.SSLSocket)
            or connection.family not in (socket.AF_INET,socket.AF_INET6)
            or connection.getsockopt(socket.SOL_SOCKET,socket.SO_TYPE)!=socket.SOCK_STREAM
            or type(server_side) is not bool or not isinstance(context,ssl.SSLContext)
            or context.protocol!=(ssl.PROTOCOL_TLS_SERVER if server_side else ssl.PROTOCOL_TLS_CLIENT)
            or context.verify_mode!=ssl.CERT_REQUIRED or context.check_hostname!=(not server_side)
            or context.minimum_version<ssl.TLSVersion.TLSv1_3
            or not isinstance(peer_pin,bytes) or len(peer_pin)!=32
            or type(timeout) not in (int,float) or not 0<timeout<=30
            or (server_side and server_hostname is not None)
            or (not server_side and (not isinstance(server_hostname,str) or not 1<=len(server_hostname)<=253 or '\0' in server_hostname))):
        raise ValueError('invalid or unsafe host TLS configuration')
    connection.getpeername() # an unconnected invalid socket remains caller-owned
    deadline=time.monotonic()+timeout;previous_timeout=connection.gettimeout();wrapped=None
    try:
        wrapped=context.wrap_socket(connection,server_side=server_side,server_hostname=server_hostname,do_handshake_on_connect=False)
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('host TLS handshake deadline exceeded')
        wrapped.settimeout(remaining);wrapped.do_handshake()
        if time.monotonic()>deadline:raise TimeoutError('host TLS handshake deadline exceeded')
        if wrapped.version()!='TLSv1.3' or wrapped.selected_alpn_protocol()!=ALPN:
            raise ValueError('host TLS version/application protocol mismatch')
        certificate=wrapped.getpeercert(binary_form=True)
        if not certificate or not hmac.compare_digest(hashlib.sha256(certificate).digest(),peer_pin):
            raise ValueError('host TLS peer certificate pin mismatch')
        wrapped.settimeout(previous_timeout)
        return wrapped
    except BaseException:
        (wrapped if wrapped is not None else connection).close()
        raise

