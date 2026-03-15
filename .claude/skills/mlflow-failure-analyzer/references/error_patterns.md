# Common MLflow Experiment Error Patterns

This document catalogs common error patterns encountered in MLflow experiments, particularly in machine learning training workflows.

## Table of Contents

1. [CUDA/GPU Errors](#cudagpu-errors)
2. [Validation Metric Errors](#validation-metric-errors)
3. [K-Fold Cross-Validation Errors](#k-fold-cross-validation-errors)
4. [Training Convergence Issues](#training-convergence-issues)
5. [Memory-Related Errors](#memory-related-errors)
6. [Configuration and Setup Errors](#configuration-and-setup-errors)
7. [Infrastructure and Timeout Errors](#infrastructure-and-timeout-errors)

## CUDA/GPU Errors

### NVML Driver Failures
**Pattern**: `NVML_SUCCESS == DriverAPI::get()->nvmlDeviceGetHandleByPciBusId_v2_`

**Description**: NVIDIA driver communication failures, often occurring during GPU memory allocation or monitoring.

**Common Causes**:
- GPU driver incompatibility
- CUDA version mismatches
- Concurrent GPU usage conflicts
- Hardware-level GPU issues

**Example Error**:
```
RuntimeError: NVML_SUCCESS == DriverAPI::get()->nvmlDeviceGetHandleByPciBusId_v2_( pci_id, &nvml_device) INTERNAL ASSERT FAILED at "/pytorch/c10/cuda/CUDACachingAllocator.cpp":1000, please report a bug to PyTorch.
```

**Typical Duration**: 2-10 seconds (immediate upon GPU allocation attempt)

**Recommended Fixes**:
- Update GPU drivers
- Check CUDA toolkit compatibility
- Restart GPU processes
- Verify GPU hardware health

### CUDA Out of Memory
**Pattern**: `CUDA.*out of memory|OutOfMemoryError`

**Description**: Insufficient GPU memory for model or batch processing.

**Common Causes**:
- Batch size too large
- Model size exceeds GPU memory
- Memory leaks in training loop
- Multiple processes competing for GPU memory

**Recommended Fixes**:
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training
- Clear GPU cache between experiments

## Validation Metric Errors

### Early Stopping Metric Missing
**Pattern**: `Early stopping conditioned on metric.*which is not available`

**Description**: Early stopping callback configured to monitor a metric that isn't being logged.

**Common Causes**:
- Validation loop not executed
- Metric name mismatch
- Validation dataset not properly configured
- Logger configuration issues

**Example Error**:
```
RuntimeError: Early stopping conditioned on metric `val_loss` which is not available. Pass in or modify your `EarlyStopping` callback to use any of the following: `train_loss`
```

**Typical Duration**: 1-5 seconds (fails during callback setup)

**Recommended Fixes**:
- Verify validation dataset exists
- Check metric logging configuration
- Update early stopping metric name
- Ensure validation loop is executed

## K-Fold Cross-Validation Errors

### Fold-Specific Validation Issues
**Pattern**: `Training failed in fold \d+.*val_loss.*not available`

**Description**: K-fold cross-validation setup where certain folds don't generate validation splits properly.

**Common Causes**:
- Incorrect k-fold data splitting implementation
- Validation metrics not logged for specific folds
- Data loader configuration issues in cross-validation setup
- Inconsistent metric naming across folds

**Example Error**:
```
Training failed in fold 7: Early stopping conditioned on metric `val_loss` which is not available. Pass in or modify your `EarlyStopping` callback to use any of the following: `train_loss`
```

**Typical Duration**: 1-3 seconds per fold

**Critical Pattern**: Often affects specific fold indices consistently (e.g., folds 0, 1, 5, 6, 7)

**Recommended Fixes**:
- Review k-fold data splitting logic
- Ensure all folds create proper train/validation splits
- Verify consistent metric logging across all folds
- Check for fold-specific configuration issues

## Training Convergence Issues

### NaN/Infinite Loss Values
**Pattern**: `loss.*nan|NaN.*loss|gradient.*nan|inf.*loss`

**Description**: Training loss becomes NaN or infinite, typically due to numerical instability.

**Common Causes**:
- Learning rate too high
- Gradient explosion
- Division by zero in loss computation
- Numerical instability in model operations

**Typical Duration**: Varies (can occur immediately or after several epochs)

**Recommended Fixes**:
- Reduce learning rate
- Add gradient clipping
- Use more stable loss functions
- Check for division by zero operations
- Apply weight initialization techniques

### Gradient-Related Issues
**Pattern**: `gradient.*explosion|gradient.*vanishing|gradient.*norm`

**Description**: Gradient values become too large (explosion) or too small (vanishing).

**Recommended Fixes**:
- Implement gradient clipping
- Adjust learning rate schedule
- Review model architecture depth
- Use residual connections or normalization layers

## Memory-Related Errors

### System Memory Exhaustion
**Pattern**: `RuntimeError.*memory|MemoryError|out of memory`

**Description**: System RAM or GPU memory exhausted during training.

**Common Causes**:
- Dataset too large for available memory
- Memory leaks in data loading
- Insufficient system resources
- Large model parameters

**Recommended Fixes**:
- Reduce batch size
- Implement data streaming
- Use memory-efficient data loaders
- Monitor memory usage patterns

## Configuration and Setup Errors

### Missing Parameters
**Pattern**: `KeyError.*config|missing.*required.*parameter|ConfigurationError`

**Description**: Required configuration parameters are missing or invalid.

**Common Causes**:
- Incomplete configuration files
- Version incompatibilities
- Missing environment variables
- Invalid parameter values

**Typical Duration**: <1 second (immediate during config validation)

**Recommended Fixes**:
- Validate configuration files
- Check parameter requirements
- Verify environment setup
- Review configuration schema

### Dimension Mismatch Errors
**Pattern**: `dimension.*mismatch|size mismatch|shape.*incompatible`

**Description**: Tensor operations with incompatible shapes or dimensions.

**Common Causes**:
- Model input/output dimension mismatches
- Incorrect data preprocessing
- Layer configuration errors
- Batch dimension inconsistencies

**Recommended Fixes**:
- Verify model architecture
- Check data preprocessing pipeline
- Validate input/output dimensions
- Review layer configurations

## Infrastructure and Timeout Errors

### Network and Connectivity Issues
**Pattern**: `TimeoutError|timeout.*exceeded|connection.*timeout`

**Description**: Operations timing out due to network or system issues.

**Common Causes**:
- Slow network connections
- Overloaded compute resources
- Deadlocks in parallel processing
- Resource contention

**Recommended Fixes**:
- Increase timeout values
- Check network connectivity
- Monitor resource utilization
- Review parallel processing configuration

### Index Out of Bounds
**Pattern**: `IndexError|index.*out of.*range|list index out of range`

**Description**: Attempting to access array or list elements beyond valid indices.

**Common Causes**:
- Dataset indexing errors
- Batch processing boundary issues
- Incorrect data loading logic
- Off-by-one errors in loops

**Recommended Fixes**:
- Validate dataset indices
- Check batch processing logic
- Review data loading implementation
- Add bounds checking

## Error Severity Classification

### High Severity
- CUDA/NVML errors
- Memory exhaustion
- Configuration errors
- Early stopping metric missing

**Impact**: Complete experiment failure, requires immediate attention

### Medium Severity
- K-fold validation issues
- Training convergence problems
- Dimension mismatches
- Timeout errors

**Impact**: Partial failure or degraded performance, needs investigation

### Low Severity
- Index errors (if isolated)
- Minor configuration warnings
- Non-critical resource issues

**Impact**: Limited impact, can often be resolved with simple fixes

This classification helps prioritize debugging efforts and resource allocation for fixing experimental issues.
