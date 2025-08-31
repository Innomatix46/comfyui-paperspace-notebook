# 🔧 GPU Troubleshooting Guide for Paperspace

## 🚨 Problem: "RuntimeError: No CUDA GPUs are available"

This error occurs when ComfyUI cannot detect the GPU. This is common in Paperspace environments.

## 🎯 Quick Solutions

### 1. **Run GPU Fix Script** (Recommended)
```bash
cd /comfyui-paperspace-notebook
chmod +x scripts/fix_gpu.sh
./scripts/fix_gpu.sh
```

### 2. **Use Safe Starter Script**
```bash
cd /comfyui-paperspace-notebook
chmod +x scripts/start_comfyui_safe.sh
./scripts/start_comfyui_safe.sh
```

### 3. **Run Python Diagnostic**
```bash
cd /comfyui-paperspace-notebook
python3 scripts/paperspace_gpu_check.py
```

## 📊 Understanding the Issue

### **Why This Happens in Paperspace:**

1. **GPU Allocation Delay**: Paperspace may take 30-60 seconds to allocate GPU
2. **Free Tier Limitations**: GPUs aren't always available on free tier
3. **Session Timeouts**: GPU may be deallocated after inactivity
4. **Driver Issues**: Occasional driver/CUDA mismatches

### **A6000 Specific Issues:**
- Requires CUDA 12.x
- Needs specific PyTorch builds
- Memory allocation settings needed

## 🛠️ Manual Troubleshooting Steps

### **Step 1: Check GPU Availability**
```bash
# Check if GPU is visible to system
nvidia-smi

# Check PyTorch CUDA
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### **Step 2: Set Environment Variables**
```bash
# Add to ~/.bashrc or run manually
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# For A6000 optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDA_ARCH_LIST="8.6"
```

### **Step 3: Wait for GPU (Paperspace)**
```bash
# GPU may take time to allocate
for i in {1..60}; do
    nvidia-smi && break || sleep 1
done
```

### **Step 4: Reinstall PyTorch with CUDA**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## 🐳 Docker Solution (Fastest)

Using Docker bypasses most GPU detection issues:

```bash
# Quick start with Docker
cd /comfyui-paperspace-notebook
chmod +x scripts/docker_quick_start.sh
./scripts/docker_quick_start.sh start
```

Docker advantages:
- Pre-configured CUDA environment
- Guaranteed GPU detection
- 1-2 minute startup vs 15-20 minutes

## 💻 CPU Mode Fallback

If GPU is absolutely not available:

```bash
# Start ComfyUI in CPU mode (slow but works)
cd /comfyui-paperspace-notebook/ComfyUI
python3 main.py --cpu --listen 0.0.0.0 --port 8188
```

**Warning**: CPU mode is 10-100x slower than GPU!

## 📅 Best Times for Free Tier GPU

Paperspace Free Tier GPU availability varies:
- **Best**: Early morning (2-6 AM EST)
- **Good**: Late night (10 PM - 2 AM EST)
- **Worst**: Business hours (9 AM - 5 PM EST)
- **Weekends**: Generally better availability

## 🔄 Automatic Retry Script

Create this script for automatic GPU waiting:

```bash
#!/bin/bash
# auto_start_with_gpu.sh

MAX_ATTEMPTS=10
WAIT_TIME=60

for i in $(seq 1 $MAX_ATTEMPTS); do
    echo "Attempt $i/$MAX_ATTEMPTS"
    
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
        echo "✅ GPU detected! Starting ComfyUI..."
        cd ComfyUI && python3 main.py --listen 0.0.0.0
        break
    else
        echo "⏳ GPU not available. Waiting ${WAIT_TIME}s..."
        sleep $WAIT_TIME
    fi
done

echo "❌ GPU not available after $MAX_ATTEMPTS attempts"
```

## 🎯 Permanent Solutions

### **1. Upgrade Paperspace Plan**
- Guaranteed GPU availability
- No waiting times
- Better GPU options

### **2. Use Docker Image**
- Pre-built with all dependencies
- Faster startup
- Consistent environment

### **3. Use Gradient Deployments**
- Managed service
- Automatic GPU allocation
- No setup required

## 📞 Support Contacts

- **Paperspace Support**: support@paperspace.com
- **ComfyUI Issues**: https://github.com/comfyanonymous/ComfyUI/issues
- **This Project**: https://github.com/Innomatix46/comfyui-paperspace-notebook/issues

## ✅ Verification Commands

After fixes, verify GPU is working:

```bash
# Full verification
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB')
"
```

Expected output for A6000:
```
PyTorch: 2.6.0+cu124
CUDA Available: True
GPU: NVIDIA RTX A6000
Memory: 48.0GB
```