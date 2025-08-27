# A6000 Free Tier Optimization Guide

This guide covers optimizations specifically for **NVIDIA A6000 GPUs** running on **Paperspace Free Tier** with **50GB storage constraints**.

## 🚀 A6000 GPU Specifications

### Hardware Capabilities
- **VRAM**: 48GB GDDR6 with ECC
- **CUDA Cores**: 10,752
- **RT Cores**: 84 (2nd gen)
- **Tensor Cores**: 336 (3rd gen)
- **Memory Bandwidth**: 768 GB/s
- **Max Power**: 300W

### Thermal Characteristics
- **Base Clock**: 1,410 MHz
- **Boost Clock**: 1,800 MHz
- **Thermal Throttle**: 83°C (important for sustained workloads)
- **Idle Temperature**: ~30-40°C
- **Optimal Operating Range**: 65-80°C

## 💾 Free Tier Storage Strategy (50GB)

### Storage Allocation
```
Total: 50GB
├── System + OS: ~8GB
├── ComfyUI + Dependencies: ~12GB
├── Models (essential): ~20GB
├── Working Space: ~7GB
└── Safety Buffer: ~3GB
```

### Model Selection Strategy
```bash
# Essential Only (fits in 50GB)
✅ SDXL Base (6.6GB) - Primary checkpoint
✅ SDXL VAE (335MB) - High quality VAE
✅ Embeddings (~50MB) - Quality improvements
❌ SDXL Refiner (6.2GB) - Skip to save space
❌ Multiple checkpoints - Choose one primary
❌ Large ControlNets (1-2GB each) - Use LoRAs instead
```

## ⚡ A6000 Performance Optimizations

### Flash Attention Configuration
```python
# Automatically enabled in requirements.txt
# Enables 2-4x larger batch sizes on A6000
FLASH_ATTENTION_VERSION = "2.6.3+cu124"
A6000_OPTIMAL_BATCH_SIZE = 4-8  # vs 1-2 without Flash Attention
```

### Memory Management
```python
# A6000 VRAM optimization settings
VRAM_MANAGEMENT = {
    "total_vram": "48GB",
    "reserved_for_system": "2GB",
    "available_for_models": "46GB",
    "enable_model_offload": True,
    "use_8bit_quantization": True,  # Saves 50% VRAM
    "gradient_checkpointing": True,  # Trades compute for memory
}
```

### Temperature Monitoring
```bash
# Monitor A6000 temperature (critical for Free Tier)
watch -n 5 'nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu --format=csv,noheader,nounits'

# Thermal throttling kicks in at 83°C
# Optimal performance: 65-80°C
# Use: nvidia-smi -i 0 -pl 250  # Limit power to 250W if overheating
```

## 🛠️ Optimization Commands

### Storage Management
```bash
# Check storage status (run regularly)
./scripts/storage_optimizer.sh status

# Clean up space when needed
./scripts/storage_optimizer.sh cleanup

# Monitor storage in real-time
./scripts/storage_optimizer.sh monitor

# Get optimization suggestions
./scripts/storage_optimizer.sh suggest
```

### GPU Performance
```bash
# Check A6000 status
nvidia-smi

# Monitor GPU utilization
gpustat -i 1

# Check thermal performance
nvidia-smi --query-gpu=temperature.gpu,power.draw --format=csv -l 1
```

### Memory Optimization
```bash
# Clear GPU memory cache
python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')"

# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits
```

## 📊 Performance Benchmarks

### Expected Performance on A6000
| Model Type | Resolution | Batch Size | Speed | VRAM Usage |
|------------|-----------|------------|-------|------------|
| SDXL Base | 1024x1024 | 1 | ~8s | ~12GB |
| SDXL Base | 1024x1024 | 4 | ~25s | ~35GB |
| SDXL Base | 1536x1536 | 1 | ~18s | ~20GB |
| SDXL + Flash | 1024x1024 | 8 | ~45s | ~45GB |

### Free Tier Constraints Impact
```bash
# Storage-limited scenarios
- Max 2-3 checkpoints total
- Prioritize LoRAs over full models
- Regular cleanup required
- No large video models (StableDiffusion Video)
```

## 🔧 A6000-Specific Configurations

### ComfyUI Launch Arguments
```bash
# Optimized for A6000 (automatically set in run.sh)
--lowvram                    # Dynamic VRAM management
--use-split-cross-attention  # A6000 memory optimization
--disable-metadata          # Save storage space
--enable-cors-header        # External access support
```

### PyTorch Optimizations
```python
# A6000 optimizations (auto-configured)
torch.backends.cudnn.benchmark = True  # A6000 has consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Use Tensor Cores
torch.backends.cudnn.allow_tf32 = True  # 1.5x speedup on A6000
```

### Environment Variables
```bash
# A6000 performance tuning
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_USE_CUDA_DSA=1  # Enable Dynamic Shared Memory
export CUDA_LAUNCH_BLOCKING=0  # Async execution
```

## 🚨 Free Tier Limitations & Workarounds

### Storage Constraints
```bash
# Problem: Only 50GB total storage
# Solutions:
1. Use cloud storage for model backup
2. Implement aggressive cleanup
3. Use model streaming from HuggingFace
4. Prioritize smaller, efficient models
```

### Session Limits
```bash
# Problem: 6-hour session limits on Free Tier
# Solutions:
1. Auto-restart every 5.5 hours (implemented)
2. Persistent storage for all outputs
3. Resume-friendly workflows
4. Quick startup optimization
```

### Network Limitations
```bash
# Problem: Download speed limits
# Solutions:
1. Pre-select essential models only
2. Use aria2c for faster downloads (implemented)
3. Mirror popular models locally
4. Use streaming APIs when possible
```

## 🏃‍♂️ Quick Start for A6000 Free Tier

### 1. Initial Setup
```bash
# Clone and start (optimized for A6000 + 50GB)
git clone https://github.com/YourUsername/comfyui-paperspace-notebook.git
cd comfyui-paperspace-notebook
chmod +x run.sh
./run.sh  # Automatically detects A6000 and Free Tier
```

### 2. Verify Optimization
```bash
# Check A6000 detection
nvidia-smi | grep A6000

# Verify Flash Attention
python -c "import flash_attn; print('Flash Attention available')"

# Check storage optimization
./scripts/storage_optimizer.sh status
```

### 3. Monitor Performance
```bash
# Real-time monitoring dashboard
watch -n 2 'clear; echo "=== A6000 STATUS ==="; nvidia-smi --query-gpu=name,temperature.gpu,power.draw,memory.used,memory.total --format=table; echo; echo "=== STORAGE STATUS ==="; df -h /storage'
```

## 🎯 Best Practices for A6000 Free Tier

### Model Management
1. **One Primary Model**: Choose SDXL Base or similar high-quality checkpoint
2. **LoRA Collection**: Use LoRAs instead of multiple full checkpoints
3. **Essential VAE**: Keep one high-quality VAE (SDXL VAE recommended)
4. **Embeddings**: Small size, high impact on quality

### Performance Optimization
1. **Batch Processing**: Use A6000's 48GB VRAM for larger batches
2. **Resolution Strategy**: 1024x1024 optimal, 1536x1536 for special cases
3. **Temperature Management**: Keep GPU under 80°C for sustained performance
4. **Memory Monitoring**: Regular cleanup to prevent VRAM fragmentation

### Storage Management
1. **Daily Monitoring**: Check storage status daily
2. **Auto Cleanup**: Enable automatic temporary file cleanup
3. **Output Management**: Archive or delete old generations regularly
4. **Model Rotation**: Swap models based on current projects

### Free Tier Optimization
1. **Session Planning**: Plan work within 6-hour sessions
2. **Persistent Configs**: Keep all settings in `/storage`
3. **Quick Restart**: Optimize for fast session recovery
4. **Backup Strategy**: External backup for important outputs

This optimization guide ensures you get maximum performance from your A6000 GPU while staying within Free Tier constraints!