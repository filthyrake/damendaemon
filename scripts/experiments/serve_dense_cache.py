"""Normal CLI with an explicit validation-only resident projection cache."""
from .kquant_dense_cache import install_validation_dispatch
from damendaemon.cli import main

if __name__ == '__main__':
    install_validation_dispatch()
    main()
