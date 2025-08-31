# ComfyUI Paperspace Troubleshooting Guide

## 🔧 Common Installation Issues & Solutions

### 1. PyTorch/xformers Version Conflict
**Error**: `Cannot install torch==2.6.0+cu124 and xformers==0.0.28.post3`
**Solution**: 
- xformers is installed separately with automatic version resolution
- The installation script handles this automatically
- Manual fix: `pip install --index-url https://download.pytorch.org/whl/cu124 xformers`

### 2. Package Index Conflicts
**Error**: `No matching distribution found for accelerate>=0.27.0`
**Solution**:
- PyTorch packages use CUDA index, ML packages use PyPI
- Script automatically separates package sources
- PyTorch/torchvision: CUDA index
- accelerate/transformers: PyPI index

### 3. Flash Attention Dependencies
**Error**: `ModuleNotFoundError: No module named 'torch'`
**Solution**:
- PyTorch must be installed BEFORE Flash Attention
- Installation order is automatically handled in scripts
- Pre-built wheels used for Python 3.12

### 4. Git Authentication Issues
**Error**: `could not read Username for 'https://github.com'`
**Solution**:
- Problematic custom nodes are temporarily disabled
- Install manually via ComfyUI Manager after setup
- Script includes timeout and fallback mechanisms

### 5. JupyterLab Root Access
**Issue**: Cannot see all folders in JupyterLab
**Solution**:
```python
# Configuration automatically set:
c.ServerApp.root_dir = '/'
c.ServerApp.notebook_dir = '/'
c.ServerApp.allow_root = True
```

### 6. Paperspace Machine Startup
**Issue**: Machine fails to start with complex commands
**Solution**:
```bash
# Simplified startup command:
PIP_DISABLE_PIP_VERSION_CHECK=1 jupyter lab --allow-root --ip=0.0.0.0 --no-browser --ServerApp.root_dir='/' --ServerApp.notebook_dir='/' --ServerApp.token=''
```

### 7. Storage Limitations (50GB Free Tier)
**Issue**: Running out of storage space
**Solution**:
```bash
# Check storage usage
./scripts/storage_optimizer.sh status

# Clean temporary files
rm -rf /tmp/*
rm -rf ComfyUI/temp/*

# Remove old outputs
rm -rf /storage/ComfyUI/output/old_files
```

## 📦 Package Installation Order

The correct installation sequence (automatically handled):

1. **Build Dependencies**
   ```bash
   pip install --upgrade pip setuptools wheel ninja packaging
   ```

2. **PyTorch with CUDA**
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0+cu124 torchvision==0.21.0+cu124
   ```

3. **xformers (auto-resolved)**
   ```bash
   pip install --index-url https://download.pytorch.org/whl/cu124 xformers
   ```

4. **ML Packages from PyPI**
   ```bash
   pip install accelerate>=0.27.0 transformers>=4.36.0 safetensors>=0.4.0
   ```

5. **Flash Attention (after PyTorch)**
   ```bash
   # Python 3.12 with pre-built wheels
   pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.0.0/flash_attn-2.6.3+cu124torch2.8-cp312-cp312-linux_x86_64.whl
   ```

## 🔍 Debugging Commands

### Check Installation Status
```bash
# GPU status
nvidia-smi

# Python version
python --version

# PyTorch verification
python -c "import torch; print(torch.cuda.is_available())"

# Check running processes
ps aux | grep -E "(python|comfy|jupyter)"

# View logs
tail -f /storage/ComfyUI/comfyui.log
tail -f /storage/jupyterlab.log
```

### Manual Fixes

#### Reset and Reinstall
```bash
# Remove virtual environment
rm -rf venv

# Clear pip cache
pip cache purge

# Restart installation
./run.sh
```

#### Update Repository
```bash
cd /comfyui-paperspace-notebook
git pull origin master
./run.sh
```

## 🆘 Getting Help

1. Check logs in `/storage/ComfyUI/` and `/tmp/`
2. Review this troubleshooting guide
3. Check GitHub Issues: https://github.com/Innomatix46/comfyui-paperspace-notebook/issues
4. Include error logs when reporting issues

## ✅ Success Indicators

- ComfyUI accessible at `https://tensorboard-[FQDN]/`
- JupyterLab shows root filesystem
- GPU detected with `nvidia-smi`
- No errors in `/storage/ComfyUI/comfyui.log`
- Auto-restart active (check with `./restart-control.sh status`)