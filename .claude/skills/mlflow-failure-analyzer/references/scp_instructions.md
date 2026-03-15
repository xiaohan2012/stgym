# SCP Artifact Retrieval Instructions

This document provides detailed instructions for retrieving MLflow artifacts from remote servers using SCP.

## Overview

MLflow experiments often run on remote compute clusters while storing artifacts on different servers. This guide covers the common patterns and configurations for accessing these artifacts.

## Default Configuration Patterns

### STGym Project Pattern (Reference Implementation)
Based on real project experience:

```bash
# Default remote server configuration
REMOTE_HOST="cyy2"
BASE_PATH="~/stgym/mlruns"
EXPERIMENT_ID="292891157014984425"
RUN_ID="b63811c051134428aea75cef189c98fd"

# Artifact structure
${BASE_PATH}/${EXPERIMENT_ID}/${RUN_ID}/artifacts/
├── training_error.txt      # Error logs and stack traces
├── experiment_config.yaml  # Complete training configuration
└── [other artifacts...]
```

### Common Server Configurations

#### Local Development
```bash
# MLflow runs stored locally
BASE_PATH="./mlruns"
ARTIFACT_PATH="./${EXPERIMENT_ID}/${RUN_ID}/artifacts/"
```

#### SSH Remote Server
```bash
# Remote server with SSH access
REMOTE_HOST="compute-server.domain.com"
BASE_PATH="/home/username/mlflow/runs"
ARTIFACT_PATH="${REMOTE_HOST}:${BASE_PATH}/${EXPERIMENT_ID}/${RUN_ID}/artifacts/"
```

#### Shared Network Storage
```bash
# Shared filesystem (NFS, etc.)
BASE_PATH="/shared/mlflow/experiments"
ARTIFACT_PATH="${BASE_PATH}/${EXPERIMENT_ID}/${RUN_ID}/artifacts/"
```

## SCP Command Patterns

### Basic Commands

#### Test Connectivity
```bash
# Verify SSH access and experiment directory
ssh ${REMOTE_HOST} "ls -la ${BASE_PATH}/${EXPERIMENT_ID}/"

# Check specific run directory
ssh ${REMOTE_HOST} "ls -la ${BASE_PATH}/${EXPERIMENT_ID}/${RUN_ID}/artifacts/"
```

#### Single File Retrieval
```bash
# Fetch error log
scp ${REMOTE_HOST}:${BASE_PATH}/${EXPERIMENT_ID}/${RUN_ID}/artifacts/training_error.txt ./error_${RUN_ID}.txt

# Fetch configuration
scp ${REMOTE_HOST}:${BASE_PATH}/${EXPERIMENT_ID}/${RUN_ID}/artifacts/experiment_config.yaml ./config_${RUN_ID}.yaml
```

#### Batch Retrieval
```bash
# Create local working directory
mkdir -p ./analysis_$(date +%s)/{errors,configs}

# Batch download multiple runs
for i in {1..10}; do
    RUN_ID=$(cat run_ids.txt | sed -n "${i}p")
    scp ${REMOTE_HOST}:${BASE_PATH}/${EXPERIMENT_ID}/${RUN_ID}/artifacts/training_error.txt ./analysis/errors/error_${i}.txt
    scp ${REMOTE_HOST}:${BASE_PATH}/${EXPERIMENT_ID}/${RUN_ID}/artifacts/experiment_config.yaml ./analysis/configs/config_${i}.yaml
done
```

### Advanced Patterns

#### Parallel Downloads
```bash
# Use GNU parallel for concurrent downloads
parallel -j 5 "scp ${REMOTE_HOST}:${BASE_PATH}/${EXPERIMENT_ID}/{}/artifacts/training_error.txt ./errors/error_{#}.txt" ::: $(cat run_ids.txt)
```

#### Selective Artifact Retrieval
```bash
# Only download artifacts that exist
ssh ${REMOTE_HOST} "find ${BASE_PATH}/${EXPERIMENT_ID} -name 'training_error.txt' -exec basename \$(dirname \$(dirname {})) \;" > available_runs.txt

while read RUN_ID; do
    scp ${REMOTE_HOST}:${BASE_PATH}/${EXPERIMENT_ID}/${RUN_ID}/artifacts/training_error.txt ./errors/
done < available_runs.txt
```

## Authentication Setup

### SSH Key Authentication (Recommended)
```bash
# Generate SSH key pair if not exists
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"

# Copy public key to remote server
ssh-copy-id username@${REMOTE_HOST}

# Test passwordless authentication
ssh ${REMOTE_HOST} "echo 'Connection successful'"
```

### SSH Config File
Create `~/.ssh/config` for simplified access:

```bash
Host cyy2
    HostName cyy2.your-domain.com
    User your-username
    IdentityFile ~/.ssh/id_rsa
    ControlMaster auto
    ControlPath ~/.ssh/sockets/%r@%h-%p
    ControlPersist 600
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

## Error Handling and Troubleshooting

### Common Connection Issues

#### Permission Denied
```bash
# Check SSH key permissions
chmod 600 ~/.ssh/id_rsa
chmod 644 ~/.ssh/id_rsa.pub

# Verify SSH agent
ssh-add ~/.ssh/id_rsa
```

#### Host Key Verification
```bash
# Add host to known_hosts
ssh-keyscan -H ${REMOTE_HOST} >> ~/.ssh/known_hosts

# Or disable strict checking (less secure)
scp -o StrictHostKeyChecking=no source destination
```

#### Connection Timeouts
```bash
# Use longer timeout values
scp -o ConnectTimeout=30 -o ServerAliveInterval=60 source destination

# For unreliable connections, use rsync instead
rsync -avz -e "ssh -o ConnectTimeout=30" source destination
```

### File Access Issues

#### Missing Artifacts
```bash
# Check if artifact directory exists
ssh ${REMOTE_HOST} "test -d ${BASE_PATH}/${EXPERIMENT_ID}/${RUN_ID}/artifacts && echo 'EXISTS' || echo 'MISSING'"

# List available artifacts
ssh ${REMOTE_HOST} "ls -la ${BASE_PATH}/${EXPERIMENT_ID}/${RUN_ID}/artifacts/ 2>/dev/null || echo 'No artifacts found'"
```

#### Large File Handling
```bash
# Show progress for large files
scp -v ${REMOTE_HOST}:source destination

# Resume interrupted transfers with rsync
rsync --partial --progress -avz -e ssh ${REMOTE_HOST}:source destination
```

## Performance Optimization

### Connection Reuse
```bash
# Enable SSH connection multiplexing in ~/.ssh/config
ControlMaster auto
ControlPath ~/.ssh/sockets/%r@%h-%p
ControlPersist 600
```

### Compression
```bash
# Enable compression for text files
scp -C ${REMOTE_HOST}:source destination

# For YAML/text artifacts, compression saves bandwidth
```

### Parallel Processing
```bash
# Limit concurrent connections to avoid overwhelming server
parallel -j 3 "scp ${REMOTE_HOST}:${BASE_PATH}/${EXPERIMENT_ID}/{}/artifacts/training_error.txt ./errors/" ::: $(cat run_ids.txt)
```

## Security Considerations

### Access Control
- Use SSH key authentication instead of passwords
- Implement least-privilege access to artifact directories
- Consider using jump hosts for additional security layers

### Data Privacy
- Encrypt sensitive configuration data in artifacts
- Use secure channels for artifact retrieval
- Implement audit logging for artifact access

### Network Security
- Use VPN connections when accessing remote servers over public networks
- Configure firewall rules to restrict SSH access
- Monitor for unusual access patterns

## Script Integration Examples

### Python Integration
```python
import subprocess
import os

def fetch_artifact(remote_host, base_path, experiment_id, run_id, artifact_name, local_path):
    remote_path = f"{remote_host}:{base_path}/{experiment_id}/{run_id}/artifacts/{artifact_name}"
    cmd = ["scp", remote_path, local_path]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
```

### Shell Script Template
```bash
#!/bin/bash

REMOTE_HOST=${1:-"cyy2"}
EXPERIMENT_ID=${2:-"required"}
OUTPUT_DIR=${3:-"./mlflow_analysis"}

if [ "$EXPERIMENT_ID" == "required" ]; then
    echo "Usage: $0 <remote_host> <experiment_id> <output_dir>"
    exit 1
fi

# Create output directories
mkdir -p "${OUTPUT_DIR}"/{errors,configs}

# Get failed run IDs from MLflow API
# (Implementation depends on your MLflow setup)

# Fetch artifacts for each failed run
while IFS= read -r RUN_ID; do
    echo "Fetching artifacts for run: $RUN_ID"

    # Fetch error log
    scp "${REMOTE_HOST}:~/stgym/mlruns/${EXPERIMENT_ID}/${RUN_ID}/artifacts/training_error.txt" \
        "${OUTPUT_DIR}/errors/error_${RUN_ID}.txt" 2>/dev/null

    # Fetch configuration
    scp "${REMOTE_HOST}:~/stgym/mlruns/${EXPERIMENT_ID}/${RUN_ID}/artifacts/experiment_config.yaml" \
        "${OUTPUT_DIR}/configs/config_${RUN_ID}.yaml" 2>/dev/null

done < failed_run_ids.txt

echo "Artifact retrieval completed. Check ${OUTPUT_DIR} for results."
```

This documentation provides comprehensive guidance for retrieving MLflow artifacts across various deployment scenarios and configurations.
