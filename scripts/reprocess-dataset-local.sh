#!/bin/bash

# Re-process dataset(s) locally.
# Deletes the processed data and re-generates one dataset at a time.
#
# Usage:
#   Single:   ./scripts/reprocess-dataset-local.sh gastric-bladder-cancer
#   Multiple: ./scripts/reprocess-dataset-local.sh gastric-bladder-cancer human-pancreas
#   All:      ./scripts/reprocess-dataset-local.sh --all

if [ $# -eq 0 ]; then
    echo "Usage: $0 [--all | dataset1 dataset2 ...]"
    exit 1
fi

if [ "$1" = "--all" ]; then
    datasets=$(ls -d data/*/raw 2>/dev/null | xargs -n1 dirname | xargs -n1 basename)
else
    datasets="$@"
fi

for ds in $datasets; do
    processed="data/${ds}/processed/data.pt"
    echo "=== Re-processing: $ds ==="

    rm -f "${processed}"

    python3 -c "
from stgym.data_loader import get_dataset_class
ds = get_dataset_class('${ds}')(root='data/${ds}')
print(f'  graphs={len(ds)}, features={ds.num_features}')
"

    if [ $? -eq 0 ]; then
        echo "  OK"
    else
        echo "  FAILED"
    fi
    echo
done
