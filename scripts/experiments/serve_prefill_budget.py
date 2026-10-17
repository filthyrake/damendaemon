"""Explicit aggregate-prefill experiment; no normal CLI configuration."""
import os

if __name__ == '__main__':
    from scripts.experiments.q4_marlin_prefill_budget import install_validation_budget
    install_validation_budget(int(os.environ['DAMENDAEMON_EXPERIMENTAL_PREFILL_TOKEN_BUDGET']))
    from damendaemon.cli import main
    main()
