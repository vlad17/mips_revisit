#! /usr/bin/env bash

# Lints code:
#
#   # Lint fdd by default.
#   ./scripts/lint.sh
#   # Lint specific files.
#   ./scripts/lint.sh fdd/somedir/*.py

set -euo pipefail

lint() {
    flake8 "$@"
}

main() {
    if [[ "$#" -eq 0 ]]; then
        lint fdd
    else
        lint "$@"
    fi
}

main "$@"
