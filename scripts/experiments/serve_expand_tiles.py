"""Launch the normal CLI with an explicit validation-only expansion override."""
from .kquant_expand import install_validation_dispatch
from damendaemon.cli import main

if __name__ == '__main__':
    install_validation_dispatch()
    main()
