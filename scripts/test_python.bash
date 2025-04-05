#!/bin/bash
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/argsparse.sh"
SRC_DIR="${SCRIPT_DIR}/../fisheye_python"

argsparse_use_option all "Run all tests."
argsparse_use_option pylint "Run pylint."
argsparse_use_option mypy "Run mypy."
argsparse_use_option pytest "Run pytest."
argsparse_parse_options "$@"
argsparse_report

EXIT="0"
PYLINT=""
MYPY=""
PYTEST=""

if [[ "${program_options[all]}" ]]; then
    program_options[pylint]="1"
    program_options[mypy]="1"
    program_options[pytest]="1"
fi

if [[ "${program_options[pylint]}" ]]; then
    echo "Running PyLint..."
    if ! command -v pylint &> /dev/null; then
        echo "Could not find pylint."
        EXIT="1"
    else
        pylint "${SRC_DIR}"
        PYLINT="${?}"
    fi
fi

if [[ "${program_options[mypy]}" ]]; then
    echo "Running MyPy..."
    if ! command -v mypy &> /dev/null; then
        echo "Could not find mypy."
        EXIT="1"
    else
        mypy "${SRC_DIR}"
        MYPY="${?}"
    fi
fi

if [[ "${program_options[pytest]}" ]]; then
    echo "Running Pytest..."
    if ! command -v pytest &> /dev/null; then
        echo "Could not find pytest."
        EXIT="1"
    else
        pytest "${SRC_DIR}/tests"
        PYTEST="${?}"
    fi
fi

echo "mypy      Exit code: ${MYPY}"
echo "pylint    Exit code: ${PYLINT}"
echo "pytest    Exit code: ${PYTEST}"

exit $EXIT
