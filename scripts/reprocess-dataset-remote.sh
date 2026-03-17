#!/bin/bash

# Re-process dataset(s) on the remote GPU server.
# Deletes the processed data and re-generates one dataset at a time.
#
# Usage:
#   Single:   ./scripts/reprocess-dataset-remote.sh gastric-bladder-cancer
#   Multiple: ./scripts/reprocess-dataset-remote.sh gastric-bladder-cancer human-pancreas
#   All:      ./scripts/reprocess-dataset-remote.sh --all

REMOTE=${REMOTE:-cyy2}
REMOTE_DIR=${REMOTE_DIR:-stgym}

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--all | dataset1 dataset2 ...]"
    exit 1
fi

if [ "$1" = "--all" ]; then
    datasets=$(ssh "$REMOTE" "ls -d ${REMOTE_DIR}/data/*/raw 2>/dev/null | xargs -n1 dirname | xargs -n1 basename")
else
    datasets="$@"
fi

for ds in $datasets; do
    processed="${REMOTE_DIR}/data/${ds}/processed/data.pt"
    echo "=== Re-processing: $ds ==="

    ssh "$REMOTE" "rm -f ${processed}"

    ssh "$REMOTE" "cd ${REMOTE_DIR} && source .venv/bin/activate && python3 -c \"
from stgym.data_loader import get_dataset_class
ds = get_dataset_class('${ds}')(root='data/${ds}')
print(f'  graphs={len(ds)}, features={ds.num_features}')
\"" 2>&1 | grep -v '^\[4pdvGPU' | grep -v 'UserWarning' | grep -v 'import torch_geometric'

    if [ $? -eq 0 ]; then
        echo "  OK"
    else
        echo "  FAILED"
    fi
    echo
done
