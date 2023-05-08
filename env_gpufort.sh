#!/bin/bash



gpufortDir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
export gpufortDir=$(dirname $gpufortDir)

chmod +x ${gpufortDir}/bin/gpufort

export PATH=${gpufortDir}/hipfort_util/bin:$PATH
export PATH=${gpufortDir}/bin:$PATH
export HIPFORT_COMPILER=$(which gfortran)
export HIPFORT=${gpufortDir}/hipfort_util
export HIPFORT_PATH=${gpufortDir}/hipfort_util