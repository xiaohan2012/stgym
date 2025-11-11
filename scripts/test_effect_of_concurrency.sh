#! /bin/bash

# Check if jobs_per_gpu argument is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <jobs_per_gpu>"
    echo "Example: $0 2  # Run 2 jobs per GPU (4 total jobs)"
    echo "Example: $0 1  # Run 1 job per GPU (2 total jobs)"
    exit 1
fi

JOBS_PER_GPU=$1

# Machine specs
TOTAL_CPUS=32
TOTAL_GPUS=2

# Calculate resource allocation
TOTAL_CONCURRENT_JOBS=$((JOBS_PER_GPU * TOTAL_GPUS))
NUM_CPUS_PER_TRIAL=$((TOTAL_CPUS / TOTAL_CONCURRENT_JOBS))

# Calculate GPU fraction per trial (1 GPU divided by jobs per GPU)
if [ $JOBS_PER_GPU -eq 1 ]; then
    NUM_GPUS_PER_TRIAL="1.0"
else
    NUM_GPUS_PER_TRIAL=$(echo "scale=2; 1.0 / $JOBS_PER_GPU" | bc)
fi

echo "🔧 Resource Allocation:"
echo "   Jobs per GPU: $JOBS_PER_GPU"
echo "   Total concurrent jobs: $TOTAL_CONCURRENT_JOBS"
echo "   CPUs per trial: $NUM_CPUS_PER_TRIAL"
echo "   GPUs per trial: $NUM_GPUS_PER_TRIAL"
echo ""

time python run_rct.py \
       +exp=hpooling \
       design_space=graph_clf \
       resource=gpu-2 \
       sample_size=5 \
       resource.num_cpus_per_trial=$NUM_CPUS_PER_TRIAL \
       resource.num_gpus_per_trial=$NUM_GPUS_PER_TRIAL \
       design_space.train.max_epoch=2 \
       mlflow.experiment_name=test-dynamic-resource-allocation-jobs-$JOBS_PER_GPU-per-gpu-nov-11
