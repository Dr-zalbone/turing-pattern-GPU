#!/usr/bin/env bash
# Container entrypoint for turing-pattern-gpu.
#
#   docker-entrypoint.sh selftest          run the GPU self-test only, then exit
#   docker-entrypoint.sh <cmd> [args...]   run the self-test (unless disabled), then exec <cmd>
#   docker-entrypoint.sh                   exec turing (prints usage)
#
# Environment:
#   TURING_SELFTEST  set to 0 to skip the automatic self-test (default 1)
set -euo pipefail

SELFTEST=/opt/turing/selftest.sh
TURING_SELFTEST="${TURING_SELFTEST:-1}"

# Explicit request: run the self-test and exit with its status.
if [[ "${1:-}" == "selftest" ]]; then
    if [[ ! -f "$SELFTEST" ]]; then
        echo "turing-entrypoint: error: $SELFTEST not found" >&2
        exit 2
    fi
    exec bash "$SELFTEST" "${@:2}"
fi

# Automatic self-test before the requested command.
if [[ "$TURING_SELFTEST" != "0" ]]; then
    if [[ -f "$SELFTEST" ]]; then
        bash "$SELFTEST" || {
            rc=$?
            echo "turing-entrypoint: error: self-test failed (exit $rc)" >&2
            exit "$rc"
        }
    else
        echo "turing-entrypoint: note: $SELFTEST not present, skipping self-test" >&2
    fi
fi

if [[ $# -eq 0 ]]; then
    set -- turing
fi
exec "$@"
