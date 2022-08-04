#!/usr/bin/env bash
# IMPORTANT: Run me from the root directory (not inside the target directory)!
set -euo pipefail

if [[ $# -eq 0 ]] ; then
    echo 'Error: no target directory/project provided!'
    echo 'Example usage:'
    echo '    run-tests.sh tic-tac-toe'
    exit 1
fi

python -m pytest --cov=$1 --cov-report term-missing --cov-branch $1/tests/

# Because pytest-cov automatically generates a .coverage file with no way to
# delete it...
# https://github.com/pytest-dev/pytest-cov/issues/374
rm -f .coverage
