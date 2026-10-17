"""Ephemeral loopback TLS credentials; never use repository or deployment keys."""
from functools import lru_cache
from pathlib import Path
import subprocess
import tempfile

# Retain the owner for the test process; TemporaryDirectory cleans up at exit.
_owners=[]


@lru_cache(maxsize=1)
def credentials():
    owner=tempfile.TemporaryDirectory(prefix='damendaemon-test-tls-')
    path=Path(owner.name)
    def openssl(*args):
        try:
            subprocess.run(['openssl',*args],cwd=path,stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,check=True,timeout=30)
        except (OSError,subprocess.SubprocessError):
            raise RuntimeError('could not generate temporary TLS test credentials with openssl') from None
    try:
        # Explicit configuration avoids depending on a machine's OpenSSL defaults.
        (path/'ca.cnf').write_text('[req]\ndistinguished_name=dn\nx509_extensions=ca\nprompt=no\n'
            '[dn]\nCN=damendaemon temporary test CA\n[ca]\nbasicConstraints=critical,CA:TRUE\n'
            'keyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\n')
        openssl('req','-x509','-newkey','rsa:2048','-nodes','-days','2','-sha256',
                '-config','ca.cnf','-keyout','ca.key','-out','ca.pem')
        (path/'ca.key').chmod(0o600)
        for name in ('server','client','untrusted'):
            (path/(name+'.cnf')).write_text('[req]\ndistinguished_name=dn\nprompt=no\n'
                '[dn]\nCN='+('localhost' if name=='server' else 'damendaemon temporary '+name)+'\n')
            openssl('req','-new','-newkey','rsa:2048','-nodes','-sha256','-config',name+'.cnf',
                    '-keyout',name+'.key','-out',name+'.csr')
            (path/(name+'.key')).chmod(0o600)
            (path/(name+'.ext')).write_text('basicConstraints=critical,CA:FALSE\n'
                'keyUsage=critical,digitalSignature,keyEncipherment\n'
                'extendedKeyUsage='+('serverAuth' if name=='server' else 'clientAuth')+'\n'
                +('subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1\n' if name=='server' else ''))
            signer=['-signkey',name+'.key'] if name=='untrusted' else ['-CA','ca.pem','-CAkey','ca.key','-CAcreateserial']
            openssl('x509','-req','-in',name+'.csr',*signer,'-days','2','-sha256',
                    '-extfile',name+'.ext','-out',name+'.pem')
    except BaseException:
        owner.cleanup()
        raise
    _owners.append(owner)
    return path
