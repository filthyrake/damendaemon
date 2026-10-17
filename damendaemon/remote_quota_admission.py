"""Explicit remote quota ownership compatible with named-client HTTP leases."""
from dataclasses import dataclass,field
import threading
from .http_auth import ClientKey
from .remote_quota_transport import RemoteQuotaClient,RemoteQuotaUnavailable,RemoteQuotaRejected
from .remote_reservation_ledger import MAX_SEQUENCE
from .shared_client_reservations import SharedReservationUnavailable,SharedReservationRejected


class RemoteQuotaRecoveryRequired(SharedReservationUnavailable):
    def __init__(self,store):
        super().__init__(store)
        self.args=('remote quota owner requires explicit recovery before admission',)


@dataclass(frozen=True,eq=False)
class RemoteReservation:
    id: str
    client: str
    requests: int
    _owner: object=field(repr=False)
    _released: bool=field(default=False,repr=False)


@dataclass(frozen=True)
class PendingMutation:
    kind: str
    claim: RemoteReservation
    sequence: int
    request: object=field(repr=False)


class RemoteQuotaReservations:
    """One serialized frontend controller per stable authority owner.

    Existing remote claims block startup; they are not assumed dead. Unknown
    mutations retain their exact wire snapshot and candidate handle. Resolve by
    releasing that ended/failed lease before attempting any later mutation.
    A conflicting writer cannot turn an unconfirmed sequence-derived handle
    into authority to release another request's claim.
    """
    def __init__(self,client,clients):
        if (not isinstance(client,RemoteQuotaClient) or not isinstance(clients,(list,tuple)) or not 1<=len(clients)<=128
                or any(not isinstance(c,ClientKey) for c in clients) or len({c.name for c in clients})!=len(clients)):
            raise ValueError('remote admission requires bounded named clients and an authenticated quota client')
        self.client=client;self.policies={c.name:(c.max_inflight,c.max_reserved_output_tokens) for c in clients}
        self.lock=threading.RLock();self._claim_identity=object();self.claims={};self.pending=None;self.closed=False;self.sequence=None
        try:
            policy=client.call(client.request('policy'))
            available={row['name']:(row['max_inflight'],row['max_reserved_output_tokens']) for row in policy['clients']}
            if any(available.get(name)!=caps for name,caps in self.policies.items()):raise RemoteQuotaRecoveryRequired(self)
            status=client.call(client.request('status'))
            if status['retired'] or status['claims']:raise RemoteQuotaRecoveryRequired(self)
            self.sequence=status['sequence']+1
            if self.sequence>MAX_SEQUENCE:raise RemoteQuotaRecoveryRequired(self)
        except (RemoteQuotaUnavailable,RemoteQuotaRejected,ValueError,TypeError):raise SharedReservationUnavailable(self) from None

    def _unavailable(self):
        error=SharedReservationUnavailable(self)
        if self.pending is not None:error.claim=self.pending.claim
        return error

    def _usable(self):
        if self.closed or self.sequence is None or self.pending is not None or self.sequence>MAX_SEQUENCE:raise self._unavailable()

    def _resolve(self):
        pending=self.pending
        try:result=self.client.call(pending.request)
        except (RemoteQuotaUnavailable,RemoteQuotaRejected,ValueError,TypeError):raise self._unavailable() from None
        # The transport verified owner/sequence/nonce and exact response shape.
        # Advance only after a known receipt, including a durable rejection.
        self.sequence=pending.sequence+1;self.pending=None
        if pending.kind=='release' or not result['ok']:
            self.claims.pop(pending.claim.id,None)
            object.__setattr__(pending.claim,'_released',True)
        return result

    def _mutate(self,kind,claim,command):
        request=self.client.request('execute',sequence=self.sequence,command=command)
        self.pending=PendingMutation(kind,claim,self.sequence,request)
        return self._resolve()

    def reserve(self,client,*,outputs=0,requests=1):
        if (not isinstance(client,ClientKey) or self.policies.get(client.name)!=(client.max_inflight,client.max_reserved_output_tokens)):
            raise SharedReservationUnavailable(self)
        if type(requests) is not int or requests not in (0,1) or type(outputs) is not int or not 0<=outputs<=2**31-1 or requests+outputs==0:
            raise ValueError('invalid remote reservation')
        with self.lock:
            self._usable()
            claim=RemoteReservation(self.client.owner+':'+str(self.sequence),client.name,requests,self._claim_identity)
            self.claims[claim.id]=claim
            try:result=self._mutate('reserve',claim,dict(op='reserve',client=client.name,requests=requests,outputs=outputs))
            except BaseException:
                if self.pending is None:
                    self.claims.pop(claim.id,None);object.__setattr__(claim,'_released',True)
                raise
            if not result['ok']:
                if result['code'] in ('request_limit','output_limit'):
                    raise SharedReservationRejected('requests' if result['code']=='request_limit' else 'outputs')
                raise SharedReservationUnavailable(self)
            return claim

    def release(self,claim):
        with self.lock:
            if isinstance(claim,RemoteReservation) and claim._owner is self._claim_identity and claim._released:return
            if not isinstance(claim,RemoteReservation) or self.claims.get(claim.id) is not claim:raise ValueError('remote reservation belongs to another owner')
            if self.closed or self.sequence is None:raise self._unavailable()
            if self.pending is not None:
                if self.pending.claim is not claim:raise self._unavailable()
                self._resolve()
                if claim._released:return
            if self.sequence>MAX_SEQUENCE:raise self._unavailable()
            self._mutate('release',claim,dict(op='release',claim=claim.id))

    def close(self):
        with self.lock:
            if self.closed:return
            if self.pending is not None:
                if self.pending.kind!='release':raise self._unavailable()
                self._resolve()
            if self.claims:raise RuntimeError('release live remote quota claims before closing admission')
            # No persistent transport connection, retirement or reclamation.
            self.closed=True
