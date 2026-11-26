#!/bin/bash
set -euo pipefail
# Parse this invocation from one immutable in-memory copy of the script.
script=$1
shift
exec /bin/bash -c "$(<"$script")" "$script" "$@"
