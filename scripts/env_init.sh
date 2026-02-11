#!/bin/bash

if [ $# -lt 1 ]; then
    echo "usage $0: <your-openpi-code-path>"
    exit 1
fi

codepath=$1

export PYTHONPATH=$PYTHONPATH:$codepath/src/:$codepath/packages/openpi-client/src/