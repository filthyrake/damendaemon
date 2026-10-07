TLS tests generate fresh short-lived credentials in a private temporary directory using `tests/tls_credentials.py` and the local OpenSSL executable. Private keys have owner-only permissions and the directory is removed when the test process exits. No static certificates or private keys belong in this directory.

Historical commits contained synthetic test-only keys. Those keys were intentionally public and must never be trusted by a deployed service. Removing their current files does not remove them from Git history.
