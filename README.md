# ComfyUI Paperspace Notebook

A production-ready, configuration-driven ComfyUI setup optimized for Paperspace Gradient notebooks with persistent storage, JupyterLab development environment, and intelligent auto-restart functionality.

## Core Features

### 🚀 **Production-Ready Setup**
- **One-Command Launch**: Single `./run.sh` command handles everything from setup to launch
- **Configuration-Driven**: All customization handled through simple text files in the `configs/` directory
- **Idempotent Operations**: Safe to run multiple times without breaking existing installations
- **Optimized Dependencies**: Pre-configured with CUDA 12.4, PyTorch 2.6, and Flash Attention

### 💾 **Persistent Storage & Performance**
- **Persistent Storage**: Models and outputs stored in `/storage` to survive notebook restarts
- **Accelerated Downloads**: Uses aria2c for fast, multi-connection model downloads
- **Essential Custom Nodes**: Includes ComfyUI-Manager, Impact-Pack, ReActor, and other popular extensions
- **Intelligent Caching**: Prevents re-downloading of models and dependencies

### 🔬 **Development Environment**
- **JupyterLab Integration**: Automatically sets up JupyterLab with root access for advanced development
- **Tensorboard URL Access**: ComfyUI accessible via Paperspace's built-in port 6006 Tensorboard mapping
- **Root Privileges**: Full system access for package installation and system configuration
- **Git Integration**: Built-in version control support in JupyterLab

### ⏰ **Auto-Restart System**
- **6-Hour Intelligent Restarts**: Automatic restarts to maintain optimal performance
- **Graceful Shutdown**: Clean termination of all processes before restart
- **GPU Memory Management**: Automatic GPU memory reset to prevent leaks  
- **System Monitoring**: Real-time tracking of memory and disk usage
- **Flexible Control**: Enable, disable, or force restarts on demand

## Quick Start Guide

### 🎯 **100% Working Paperspace Access Method**

The key insight: ComfyUI works perfectly in Paperspace using **Tensorboard URL mapping** on port 6006!

1. **Open a new Paperspace Gradient notebook** with a GPU instance (RTX A6000 recommended)

2. **Create a new code cell** and paste the following commands:

```bash
!git clone https://github.com/YourUsername/comfyui-paperspace-notebook.git
%cd comfyui-paperspace-notebook
!chmod +x run.sh
!./run.sh
```

3. **Run the cell** - The script will automatically:
   - Install ComfyUI and all dependencies with CUDA 12.4 optimization
   - Set up 6 essential custom nodes (ComfyUI-Manager, Impact-Pack, ReActor, etc.)
   - Configure JupyterLab with root access for development
   - Download configured models (if specified)
   - Start auto-restart scheduler for 6-hour maintenance cycles
   - Launch ComfyUI on port 6006 (Tensorboard mapping) and JupyterLab

4. **Access your services** via the provided URLs that appear in the output:
   - **ComfyUI**: `https://tensorboard-[PAPERSPACE_FQDN]/` - Main AI image generation interface (uses Paperspace Tensorboard mapping)
   - **JupyterLab**: `https://[PAPERSPACE_FQDN]/lab/` - Development environment with root access

5. **Monitor the system**:
   - Auto-restart logs: `/storage/ComfyUI/restart.log`
   - Check status: `./restart-control.sh status`
   - JupyterLab logs: `/storage/jupyterlab.log`

**That's it!** Your production-ready ComfyUI environment with development tools and intelligent maintenance is running.

## Configuration & Customization

### 📝 **Configuration Files**

All user settings are managed through files in the `configs/` directory:

| File | Purpose | Example |
|------|---------|---------|
| `python_requirements.txt` | Python packages and versions | `transformers==4.46.3` |
| `custom_nodes.txt` | ComfyUI extensions from GitHub | `https://github.com/ltdrdata/ComfyUI-Manager` |
| `models.txt` | Models to download | `checkpoints https://huggingface.co/...` |

### 🔧 **Adding Custom Nodes**

Edit `configs/custom_nodes.txt` and add GitHub repository URLs:

```bash
# Essential nodes (included by default)
https://github.com/ltdrdata/ComfyUI-Manager
https://github.com/ltdrdata/ComfyUI-Impact-Pack
https://github.com/Gourieff/ComfyUI-ReActor-Node
https://github.com/cubiq/ComfyUI_IPAdapter_plus
https://github.com/WASasquatch/was-node-suite-comfyui
https://github.com/rgthree/rgthree-comfy

# Add your custom nodes here
https://github.com/author/new-custom-node
```

### 📦 **Adding Models**

Edit `configs/models.txt` using the format `[subdirectory] [download_url]`:

```bash
# SDXL Checkpoints
checkpoints https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors

# LoRA Models
loras https://civitai.com/api/download/models/16576
loras https://huggingface.co/XLabs-AI/flux-lora-collection/resolve/main/art_lora.safetensors

# VAE Models
vae https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors
```

### 🐍 **Python Dependencies**

Add packages to `configs/python_requirements.txt`:

```bash
# Core ML libraries (included by default)
torch==2.6.0+cu124
transformers==4.46.3
accelerate==1.2.1

# JupyterLab and extensions (included)
jupyterlab==4.3.3
jupyter-server-proxy==4.4.0

# Add your custom packages
new-package==1.2.3
custom-library>=2.0.0
```

### ❌ **Disabling Components**

Comment out any line with `#` to disable:

```bash
# Disabled custom node
# https://github.com/disabled/custom-node

# Disabled model download
# checkpoints https://disabled-model-url
```

## How It Works

### Main Orchestration (`run.sh`)

The `run.sh` script is your single entry point that:

1. **Sets up environment variables** for consistent paths
2. **Sources modular scripts** from the `scripts/` directory
3. **Installs dependencies** (ComfyUI, Python packages, custom nodes, JupyterLab)
4. **Downloads and links models** from persistent storage
5. **Configures JupyterLab** with root access for development
6. **Sets up auto-restart** scheduler for 6-hour intervals
7. **Launches both ComfyUI and JupyterLab** with optimized settings

### Persistent Storage Strategy

- **Models**: Stored in `/storage/ComfyUI/models/` and symlinked to `ComfyUI/models/`
- **Outputs**: Generated in `/storage/ComfyUI/output/` for persistence across restarts
- **Initialization Flag**: `/storage/ComfyUI/.init_done` prevents re-downloading models

### Modular Architecture

- `scripts/install_dependencies.sh`: Handles ComfyUI and extension setup
- `scripts/download_models.sh`: Manages model downloads and storage linking
- `scripts/configure_jupyterlab.sh`: Sets up JupyterLab with root access
- `scripts/auto_restart.sh`: Manages 6-hour automatic restarts
- `configs/`: Contains all user-configurable settings

## Troubleshooting

### Common Issues

- **Can't access ComfyUI**: Use the **Tensorboard URL** `https://tensorboard-[PAPERSPACE_FQDN]/` (100% working)
- **Connection refused**: ComfyUI is running on port 6006, not 8188 - use the tensorboard- subdomain  
- **Models not loading**: Check symlinks with `ls -la ComfyUI/models/`
- **Custom node errors**: Review installation logs for specific node requirements
- **Memory issues**: Reduce model sizes or use smaller custom node sets
- **Unexpected restarts**: Check `/storage/ComfyUI/restart.log` for auto-restart events
- **Auto-restart not working**: Use `./restart-control.sh status` to diagnose issues

### Manual Operations

```bash
# Rerun just dependency installation
source scripts/install_dependencies.sh && install_dependencies

# Rerun just model downloads
source scripts/download_models.sh && download_models

# Reset model downloads (deletes init flag)
rm /storage/ComfyUI/.init_done

# Restart JupyterLab
source scripts/configure_jupyterlab.sh && configure_jupyterlab && start_jupyterlab

# Check JupyterLab status
ps aux | grep jupyter
tail -f /storage/jupyterlab.log

# Control auto-restart feature
./restart-control.sh status          # Check auto-restart status
./restart-control.sh enable          # Enable auto-restart
./restart-control.sh disable         # Disable auto-restart
./restart-control.sh force-restart   # Force immediate restart
./restart-control.sh logs            # View restart logs
```

## Requirements

- Paperspace Gradient notebook with GPU (RTX A6000 recommended for 48GB VRAM)
- CUDA 12.4 compatible instance
- At least 50GB disk space for Free Tier optimization
- Additional space for models as configured

## 🎯 Paperspace Access Guide

### ✅ The 100% Working Method

ComfyUI in Paperspace works perfectly using **Tensorboard URL mapping**:

1. **Port 6006**: Uses Paperspace's built-in Tensorboard port mapping
2. **Tensorboard subdomain**: Automatically maps to `https://tensorboard-[PAPERSPACE_FQDN]/`
3. **No SSL errors**: Paperspace handles certificates automatically
4. **100% reliable**: No connection refused or port access issues

### 📋 Direct Access Example

If your Paperspace FQDN is `nleajkwn3o.clg07azjl.paperspacegradient.com`, your ComfyUI URL is:
```
https://tensorboard-nleajkwn3o.clg07azjl.paperspacegradient.com/
```

### 🔧 Alternative Access via Jupyter Notebook

Use the included `ComfyUI_Tensorboard_Access.ipynb` notebook:
1. Run the cells to get your exact access URL
2. Automatically starts ComfyUI with A6000 optimizations
3. Displays working links for easy access

## License

This project is provided as-is for educational and development purposes.