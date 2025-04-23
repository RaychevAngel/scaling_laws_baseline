# Scaling Laws Baseline

This work serves as a baseline for answering the question of whether reasoning models using in-context search can solve fundamentally new classes of problems that remain unsolvable by standard search approaches, regardless of computational resources. Specifically, we adopt an AlphaZero-inspired Monte Carlo Tree Search (MCTS) framework within the language model domain to solve a generalized version of the Game of 24. We implement an iterative training approach jointly optimizing policy and value functions, inspired by TS-LLM. We employ SmolM2 models of sizes $135 \mathrm{M}, 360 \mathrm{M}$, and 1.7 B for both policy and value functions, and examine scaling laws related to training and inference-time compute.

## Setup Instructions

### Environment Setup

1. Create and activate a virtual environment with `uv` (or your preferred tool):
   ```bash
   uv venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

### Data Generation

Generate training data before running training:

```bash
# Generate arithmetic reasoning problems dataset
python -m utils.generate_initial_data
```

This will create datasets in:
- `data/mcts_generated/policy/policy` (for policy training)
- `data/mcts_generated/value/value` (for value training)

### Training

To train models with the optimized configurations (with pre-tokenization, mixed precision, etc.):

```bash
# Train policy model
python test_policy.py

# Train value model
python test_value.py
```

### Configuration

The training configurations are defined in YAML files:
- `train/config_policy.yaml` - Policy model training settings
- `train/config_value.yaml` - Value model training settings

To enable advanced optimizations:
1. Set `compile_model: true` to use PyTorch 2.2+ compilation
2. Set `use_fsdp: true` to use FSDP for distributed training (multi-GPU)

### Performance Monitoring

To monitor GPU utilization during training:

```bash
# Basic monitoring, updates every second
watch -n 1 nvidia-smi

# Detailed monitoring with timestamp
while true; do date; nvidia-smi --query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv; sleep 1; done
```

After training, you can analyze the performance traces:

```bash
tensorboard --logdir=logs/
```

## Performance Optimizations

This repository includes several performance optimizations:

1. **Pre-tokenization**: Dataset is tokenized once upfront rather than per-batch
2. **DataLoader improvements**: Parallel workers, pinned memory, and prefetching
3. **Mixed precision**: FP16 training for faster computation
4. **Larger batches**: Reduced gradient accumulation steps for better throughput
5. **Vectorized operations**: Vectorized value-head loss calculation
6. **vLLM optimizations**: Efficient GPU utilization for inference
7. **Optional features**: Support for torch.compile and FSDP (disabled by default)

See `REPORT_MFU.md` for more details on performance improvements.

## Troubleshooting

### Common Issues

1. **Missing dataset files**:
   If you see `FileNotFoundError` for dataset paths, run the data generation script:
   ```bash
   python -m utils.generate_initial_data
   ```

2. **GPU memory errors**:
   If you encounter CUDA out of memory errors:
   - Reduce `per_device_train_batch_size` in config files
   - Use mixed precision by keeping `fp16 = True`
   - Consider enabling FSDP with `use_fsdp: true` in config files

3. **Import errors**:
   Ensure you're running commands from the project root directory and have activated the virtual environment.

### Profiling and Optimization

For more detailed performance analysis:
1. Run training with the profiling enabled (already set up)
2. Examine traces in TensorBoard
3. Adjust parameters in the config files based on your hardware
