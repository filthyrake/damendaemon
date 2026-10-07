"""Explicit bounded TCP/TLS bootstrap; no DNS, bind, discovery or provisioning."""
import errno
import ipaddress
import socket
import ssl
import threading
import time
from .host_tls import wrap_peer,_validate_policy,_remaining,_wait_socket,HostTlsCancelled


def connect_peer(address,port,context,*,peer_pin,server_hostname,timeout=10.,cancelled=None):
    """Connect to a literal IP and transfer an authenticated blocking TLS socket.

    One deadline covers TCP connection and TLS authentication. Hostname checks
    use server_hostname independently of the literal routing address. No DNS
    resolver, retry, background thread or plaintext fallback is used.
    """
    _validate_policy(context,server_side=False,peer_pin=peer_pin,server_hostname=server_hostname,
                     timeout=timeout,cancelled=cancelled)
    if not isinstance(address,str) or '%' in address or type(port) is not int or not 1<=port<=65535:
        raise ValueError('host endpoint requires a literal IP and port 1..65535')
    try:ip=ipaddress.ip_address(address)
    except ValueError:raise ValueError('host endpoint requires a literal IP; DNS is not performed') from None
    deadline=time.monotonic()+timeout;_remaining(deadline,cancelled)
    connection=socket.socket(socket.AF_INET if ip.version==4 else socket.AF_INET6,socket.SOCK_STREAM)
    try:
        connection.setblocking(False)
        error=connection.connect_ex((str(ip),port))
        if error not in (0,errno.EISCONN):
            if error not in (errno.EINPROGRESS,errno.EWOULDBLOCK,errno.EALREADY):raise OSError(error,'host TCP connection failed')
            _wait_socket(connection,False,deadline,cancelled)
            error=connection.getsockopt(socket.SOL_SOCKET,socket.SO_ERROR)
            if error:raise OSError(error,'host TCP connection failed')
        connection.settimeout(None)
        return wrap_peer(connection,context,server_side=False,peer_pin=peer_pin,server_hostname=server_hostname,
                         timeout=_remaining(deadline,cancelled),cancelled=cancelled,_deadline=deadline)
    except BaseException:
        connection.close()
        raise


class HostPeerListener:
    """Own a caller-bound listening socket with one pending authenticated accept.

    Constructor validation precedes ownership. close interrupts accept/handshake
    through a bounded cancellation poll; it never closes successfully transferred
    peers. The caller must not mutate/share the listener or TLS context afterward.
    """
    def __init__(self,listener,context,*,peer_pin,backlog=4):
        _validate_policy(context,server_side=True,peer_pin=peer_pin,server_hostname=None,timeout=10.)
        if (not isinstance(listener,socket.socket) or isinstance(listener,ssl.SSLSocket)
                or listener.family not in (socket.AF_INET,socket.AF_INET6)
                or listener.getsockopt(socket.SOL_SOCKET,socket.SO_TYPE)!=socket.SOCK_STREAM
                or type(backlog) is not int or not 1<=backlog<=16
                or listener.getsockname()[1]==0):
            raise ValueError('host listener requires an already-bound TCP socket and backlog 1..16')
        try:listener.getpeername()
        except OSError as error:
            if error.errno!=errno.ENOTCONN:raise
        else:raise ValueError('host listener cannot own an already-connected socket')
        # SO_ACCEPTCONN is not a readable socket option on all supported hosts.
        # Explicit listen is portable and also bounds the caller's backlog.
        listener.listen(backlog);listener.setblocking(False)
        self.listener=listener;self.context=context;self.peer_pin=peer_pin
        self.closed=threading.Event();self.accept_lock=threading.Lock();self.state_lock=threading.Lock()

    def close(self):
        with self.state_lock:
            if self.closed.is_set():return
            self.closed.set()
            try:self.listener.shutdown(socket.SHUT_RDWR)
            except OSError:pass
            finally:self.listener.close()

    def __enter__(self):return self
    def __exit__(self,*_):self.close()

    def accept(self,*,timeout=10.,cancelled=None):
        _validate_policy(self.context,server_side=True,peer_pin=self.peer_pin,server_hostname=None,
                         timeout=timeout,cancelled=cancelled)
        if not self.accept_lock.acquire(blocking=False):raise RuntimeError('host listener already has a pending accept')
        connection=None;deadline=time.monotonic()+timeout
        stopped=lambda:self.closed.is_set() or (cancelled is not None and cancelled())
        try:
            while True:
                _wait_socket(self.listener,True,deadline,stopped)
                try:connection,_=self.listener.accept();break
                except BlockingIOError:continue
            connection.settimeout(None)
            connection=wrap_peer(connection,self.context,server_side=True,peer_pin=self.peer_pin,
                                 timeout=_remaining(deadline,stopped),cancelled=stopped,_deadline=deadline)
            _remaining(deadline,stopped)
            with self.state_lock:
                if self.closed.is_set():raise HostTlsCancelled('host listener closed')
                _remaining(deadline,None)
                peer=connection;connection=None
                return peer
        except (OSError,ValueError):
            if self.closed.is_set():raise HostTlsCancelled('host listener closed') from None
            raise
        finally:
            if connection is not None:connection.close()
            self.accept_lock.release()
