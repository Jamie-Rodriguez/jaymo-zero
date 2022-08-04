#!/usr/bin/env bash
set -euo pipefail

# Because I always forget...
python -m pip freeze > requirements.txt
