#!/bin/bash
SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
. "${SCRIPT_DIR}/argsparse.sh"

argsparse_use_option tag: \
    "The tag for the docker image"
argsparse_use_option push \
    "The image will be pushed if provided."
argsparse_use_option no-cache \
    "Pass --no-cache to docker build."
argsparse_parse_options "$@"
argsparse_report

IMG_TAG="${program_options[tag]}"
PUSH="${program_options[push]}"
if [[ "${program_options[no-cache]}" ]]; then
    NO_CACHE="--no-cache"
else
    NO_CACHE=""
fi
echo "Building image"
pushd "${SCRIPT_DIR}/../"
docker build . "${NO_CACHE}" -f docker/Dockerfile -t "${IMG_TAG}" || exit
if [[ "${PUSH}" ]]; then
    echo "Pushing image"
    docker push "${IMG_TAG}"
fi
popd
echo "Done."
