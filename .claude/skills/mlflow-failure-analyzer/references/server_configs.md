# MLflow Server Configuration Examples

This document provides examples of common MLflow deployment patterns and their corresponding artifact storage configurations.

## Configuration Categories

### 1. Local Development Setup
**Use Case**: Single-user development and testing

```yaml
mlflow_config:
  tracking_uri: "file://./mlruns"
  artifact_location: "./mlruns"
  backend_store_uri: "sqlite:///mlflow.db"

artifact_retrieval:
  method: "local_filesystem"
  base_path: "./mlruns"
  requires_ssh: false
```

**Artifact Access Pattern**:
```bash
# Direct filesystem access
ARTIFACT_PATH="./mlruns/${EXPERIMENT_ID}/${RUN_ID}/artifacts/"
ls -la ${ARTIFACT_PATH}
```

### 2. Remote SSH Server (STGym Pattern)
**Use Case**: Compute cluster with shared filesystem access via SSH

```yaml
mlflow_config:
  tracking_uri: "http://127.0.0.1:5001"
  remote_server: "cyy2"
  artifact_location: "~/stgym/mlruns"

artifact_retrieval:
  method: "ssh_scp"
  remote_host: "cyy2"
  base_path: "~/stgym/mlruns"
  requires_ssh: true
  ssh_key: "~/.ssh/id_rsa"
```

**Artifact Access Pattern**:
```bash
# SCP-based retrieval
scp cyy2:~/stgym/mlruns/${EXPERIMENT_ID}/${RUN_ID}/artifacts/training_error.txt ./local_file.txt
```

### 3. Shared Network Storage
**Use Case**: Multiple users with access to shared filesystem

```yaml
mlflow_config:
  tracking_uri: "http://mlflow-server:5000"
  artifact_location: "/shared/mlflow/artifacts"

artifact_retrieval:
  method: "shared_filesystem"
  base_path: "/shared/mlflow/artifacts"
  requires_ssh: false
  mount_point: "/mnt/shared"
```

**Artifact Access Pattern**:
```bash
# Direct mounted filesystem access
ARTIFACT_PATH="/mnt/shared/mlflow/artifacts/${EXPERIMENT_ID}/${RUN_ID}/artifacts/"
ls -la ${ARTIFACT_PATH}
```

### 4. Cloud Storage (S3/GCS/Azure)
**Use Case**: Cloud-based MLflow deployment

```yaml
mlflow_config:
  tracking_uri: "http://mlflow-server.cloud.com"
  artifact_location: "s3://mlflow-bucket/artifacts/"

artifact_retrieval:
  method: "cloud_cli"
  cloud_provider: "aws"
  bucket: "mlflow-bucket"
  prefix: "artifacts"
  requires_credentials: true
```

**Artifact Access Pattern**:
```bash
# AWS CLI access
aws s3 cp s3://mlflow-bucket/artifacts/${EXPERIMENT_ID}/${RUN_ID}/artifacts/training_error.txt ./local_file.txt
```

### 5. Docker Container Setup
**Use Case**: Containerized MLflow with volume mounts

```yaml
mlflow_config:
  tracking_uri: "http://localhost:5000"
  container_name: "mlflow-server"
  artifact_location: "/mlflow/artifacts"

artifact_retrieval:
  method: "docker_exec"
  container: "mlflow-server"
  base_path: "/mlflow/artifacts"
  requires_docker: true
```

**Artifact Access Pattern**:
```bash
# Docker container access
docker cp mlflow-server:/mlflow/artifacts/${EXPERIMENT_ID}/${RUN_ID}/artifacts/training_error.txt ./local_file.txt
```

### 6. Multi-Server Research Cluster
**Use Case**: Large research organization with dedicated compute and storage servers

```yaml
mlflow_config:
  tracking_uri: "http://tracking.research.edu:5000"
  compute_hosts:
    - "gpu-node-01.research.edu"
    - "gpu-node-02.research.edu"
    - "cpu-cluster.research.edu"
  storage_server: "storage.research.edu"
  artifact_location: "/data/mlflow/experiments"

artifact_retrieval:
  method: "ssh_scp"
  remote_host: "storage.research.edu"
  base_path: "/data/mlflow/experiments"
  requires_ssh: true
  jump_host: "login.research.edu"
```

**Artifact Access Pattern**:
```bash
# SSH with jump host
scp -J login.research.edu storage.research.edu:/data/mlflow/experiments/${EXPERIMENT_ID}/${RUN_ID}/artifacts/training_error.txt ./local_file.txt
```

## Environment-Specific Configurations

### Academic/Research Environments

#### SLURM Cluster Configuration
```yaml
environment: "slurm_cluster"
mlflow_config:
  tracking_uri: "http://head-node:5000"
  compute_partition: "gpu"
  storage_backend: "lustre"
  artifact_location: "/lustre/mlflow/${USER}/experiments"

artifact_retrieval:
  method: "ssh_scp"
  remote_host: "head-node"
  base_path: "/lustre/mlflow/${USER}/experiments"
  requires_ssh: true
  scheduler: "slurm"

job_submission:
  command: "srun --partition=gpu --gres=gpu:1"
  artifact_staging: "/tmp/mlflow_staging"
```

#### PBS/Torque Cluster Configuration
```yaml
environment: "pbs_cluster"
mlflow_config:
  tracking_uri: "postgresql://mlflow:password@db-server:5432/mlflow"
  compute_nodes: "compute-*.cluster.edu"
  artifact_location: "/gpfs/mlflow/artifacts"

artifact_retrieval:
  method: "ssh_scp"
  remote_host: "login-node.cluster.edu"
  base_path: "/gpfs/mlflow/artifacts"
  requires_ssh: true
  parallel_jobs: 4
```

### Cloud Platform Configurations

#### AWS Configuration
```yaml
environment: "aws"
mlflow_config:
  tracking_uri: "http://mlflow-alb.amazonaws.com"
  backend_store_uri: "mysql://mlflow:password@rds-endpoint:3306/mlflow"
  artifact_location: "s3://company-mlflow-artifacts/"

compute:
  instance_types: ["p3.2xlarge", "p3.8xlarge"]
  regions: ["us-west-2", "us-east-1"]

artifact_retrieval:
  method: "aws_cli"
  s3_bucket: "company-mlflow-artifacts"
  aws_profile: "mlflow"
  region: "us-west-2"
```

#### Google Cloud Configuration
```yaml
environment: "gcp"
mlflow_config:
  tracking_uri: "http://mlflow-service.googleapis.com"
  backend_store_uri: "mysql://mlflow:password@cloudsql-proxy:3306/mlflow"
  artifact_location: "gs://company-mlflow-bucket/artifacts/"

compute:
  machine_types: ["n1-highmem-8", "n1-standard-16"]
  accelerators: ["nvidia-tesla-v100", "nvidia-tesla-t4"]

artifact_retrieval:
  method: "gsutil"
  gcs_bucket: "company-mlflow-bucket"
  gcp_project: "company-ml-project"
  service_account: "mlflow-service@company-ml-project.iam.gserviceaccount.com"
```

### Enterprise Configurations

#### On-Premises Enterprise Setup
```yaml
environment: "enterprise_on_prem"
mlflow_config:
  tracking_uri: "https://mlflow.company.com"
  backend_store_uri: "postgresql://mlflow:password@db.company.com:5432/mlflow"
  artifact_location: "/nfs/mlflow/artifacts"

security:
  authentication: "ldap"
  authorization: "rbac"
  ssl_cert: "/etc/ssl/certs/mlflow.crt"
  ssl_key: "/etc/ssl/private/mlflow.key"

artifact_retrieval:
  method: "nfs_mount"
  nfs_server: "storage.company.com"
  mount_path: "/nfs/mlflow/artifacts"
  requires_kerberos: true
```

#### Hybrid Cloud Configuration
```yaml
environment: "hybrid_cloud"
mlflow_config:
  tracking_uri: "https://mlflow-hybrid.company.com"
  on_prem_compute: true
  cloud_storage: true
  artifact_location: "s3://company-artifacts/"

compute_locations:
  on_premises:
    - "gpu-cluster.company.com"
    - "cpu-farm.company.com"
  cloud:
    - "aws:us-west-2"
    - "azure:westus2"

artifact_retrieval:
  primary_method: "aws_cli"
  fallback_method: "ssh_scp"
  sync_strategy: "cloud_first"
```

## Skill Configuration Examples

### Configuration for Different Deployment Patterns

#### Development/Testing Configuration
```python
# For skill testing and development
SKILL_CONFIG = {
    'default_remote_host': 'localhost',
    'default_base_path': './test_mlruns',
    'max_workers': 2,
    'timeout': 10,
    'test_mode': True
}
```

#### Production Research Cluster
```python
# For production research environment
SKILL_CONFIG = {
    'default_remote_host': 'cyy2',
    'default_base_path': '~/stgym/mlruns',
    'max_workers': 8,
    'timeout': 30,
    'retry_attempts': 3,
    'compression': True,
    'ssh_config': '~/.ssh/config'
}
```

#### High-Volume Analysis
```python
# For large-scale experiment analysis
SKILL_CONFIG = {
    'default_remote_host': 'storage.research.edu',
    'default_base_path': '/data/mlflow/experiments',
    'max_workers': 16,
    'timeout': 60,
    'batch_size': 100,
    'parallel_analysis': True,
    'cache_artifacts': True,
    'cache_duration': 3600  # 1 hour
}
```

## Troubleshooting Common Configuration Issues

### SSH Configuration Problems
```bash
# Test SSH connectivity
ssh -v cyy2 "echo 'Connection test successful'"

# Check SSH config
cat ~/.ssh/config | grep -A 10 cyy2

# Verify key permissions
ls -la ~/.ssh/
```

### Permission Issues
```bash
# Check remote directory permissions
ssh cyy2 "ls -la ~/stgym/"
ssh cyy2 "test -r ~/stgym/mlruns && echo 'Readable' || echo 'Not readable'"

# Verify artifact directory structure
ssh cyy2 "find ~/stgym/mlruns -name 'artifacts' -type d | head -5"
```

### Path Resolution Issues
```bash
# Test path expansion
ssh cyy2 "echo \$HOME"
ssh cyy2 "realpath ~/stgym/mlruns"

# Check for symbolic links
ssh cyy2 "ls -la ~/stgym/mlruns"
```

### Network and Firewall Issues
```bash
# Test connectivity on specific ports
nc -zv cyy2 22  # SSH port
nc -zv mlflow-server 5000  # MLflow tracking port

# Check for packet loss
ping -c 5 cyy2
```

## Best Practices

### Security
- Use SSH key authentication instead of passwords
- Implement least-privilege access controls
- Regularly rotate SSH keys and credentials
- Use VPN connections for remote access

### Performance
- Enable SSH connection multiplexing
- Use compression for text-based artifacts
- Implement parallel downloads for large datasets
- Cache frequently accessed artifacts locally

### Reliability
- Implement retry logic for network operations
- Use checksums to verify file integrity
- Monitor disk space on both local and remote systems
- Log all artifact retrieval operations for debugging

### Monitoring
- Track artifact retrieval success rates
- Monitor transfer speeds and timeouts
- Alert on repeated connection failures
- Maintain audit logs for compliance

This configuration guide helps ensure successful artifact retrieval across diverse MLflow deployment scenarios.
