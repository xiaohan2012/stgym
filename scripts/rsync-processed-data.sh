#!/bin/bash

# Sync processed dataset files to the remote GPU server.
#
# Usage:
#   Sync all:       ./scripts/rsync-processed-data.sh
#   Sync specific:  ./scripts/rsync-processed-data.sh gastric-bladder-cancer inflammatory-skin

REMOTE=${REMOTE:-cyy2}
REMOTE_DIR=${REMOTE_DIR:-stgym/data}

if [ $# -gt 0 ]; then
    datasets=("$@")
else
    # Find all datasets with processed data
    datasets=()
    for dir in data/*/processed/data.pt; do
        ds_name=$(echo "$dir" | cut -d/ -f2)
        datasets+=("$ds_name")
    done
fi

echo "Syncing ${#datasets[@]} dataset(s) to ${REMOTE}:${REMOTE_DIR}"

for ds in "${datasets[@]}"; do
    src="data/${ds}/processed/data.pt"
    if [ ! -f "$src" ]; then
        echo "  SKIP $ds — no local processed data"
        continue
    fi
    size=$(du -h "$src" | cut -f1)
    echo "  SYNC $ds ($size)..."
    rsync -avzh --progress "$src" "${REMOTE}:${REMOTE_DIR}/${ds}/processed/data.pt"
done

echo "Done."
