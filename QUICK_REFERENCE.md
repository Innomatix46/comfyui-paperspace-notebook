# ComfyUI Paperspace Notebook - Quick Reference Guide

## 🚀 Essential Commands

### **Initial Setup**
```bash
# Clone and start (one-time setup)
!git clone https://github.com/YourUsername/comfyui-paperspace-notebook.git
%cd comfyui-paperspace-notebook
!chmod +x run.sh
!./run.sh
```

### **Auto-Restart Management**
```bash
# Check auto-restart status
./restart-control.sh status

# Enable/disable auto-restart
./restart-control.sh enable
./restart-control.sh disable

# Force immediate restart
./restart-control.sh force-restart

# View restart logs
./restart-control.sh logs
```

### **Service Management**
```bash
# Check running processes
ps aux | grep -E "(python.*main.py|jupyter.*lab)"

# View service logs
tail -f /storage/jupyterlab.log
tail -f /storage/ComfyUI/restart.log

# Manual restart services
source scripts/configure_jupyterlab.sh && start_jupyterlab
```

## 🌐 Access URLs

### **Default Ports**
- **ComfyUI**: Port 8188 - `https://[PAPERSPACE_FQDN]:8188/`
- **JupyterLab**: Port 8889 - `https://[PAPERSPACE_FQDN]:8889/`

### **Local Development**
- **ComfyUI**: `http://localhost:8188`
- **JupyterLab**: `http://localhost:8889`

## 📁 Important File Locations

### **Configuration Files**
```
configs/python_requirements.txt    # Python packages
configs/custom_nodes.txt          # ComfyUI extensions  
configs/models.txt                # Model downloads
```

### **Log Files**
```
/storage/ComfyUI/restart.log      # Auto-restart events
/storage/jupyterlab.log           # JupyterLab logs
/storage/ComfyUI/output/          # Generated images
```

### **System Files**
```
/storage/ComfyUI/models/          # Model storage
/storage/ComfyUI/.init_done       # Download completion flag
ComfyUI/custom_nodes/             # Installed extensions
```

## ⚙️ Configuration Quick Edits

### **Add Custom Node**
```bash
echo "https://github.com/author/node-name" >> configs/custom_nodes.txt
./run.sh  # Restart to install
```

### **Add Model Download**
```bash
echo "checkpoints https://huggingface.co/model/path" >> configs/models.txt
./run.sh  # Restart to download
```

### **Add Python Package**
```bash
echo "package-name==1.2.3" >> configs/python_requirements.txt
./run.sh  # Restart to install
```

## 🔧 Troubleshooting Commands

### **Check System Status**
```bash
# Overall system check
./restart-control.sh status

# Check disk space
df -h /storage

# Check memory usage
free -h

# Check GPU status
nvidia-smi
```

### **Reset Components**
```bash
# Reset model downloads
rm /storage/ComfyUI/.init_done

# Clean restart logs
> /storage/ComfyUI/restart.log

# Reset JupyterLab config
rm ~/.jupyter/jupyter_lab_config.py
```

### **Manual Recovery**
```bash
# Stop all services
./restart-control.sh disable
pkill -f "python.*main.py"
pkill -f "jupyter.*lab"

# Clean PIDs
rm -f /storage/ComfyUI/auto_restart.pid
rm -f /storage/ComfyUI/scheduler.pid

# Full restart
./run.sh
```

## 📊 Status Indicators

### **Auto-Restart Status**
- ✅ `RUNNING` - Auto-restart is active
- ❌ `NOT RUNNING` - Auto-restart is disabled
- ⏰ `Next restart: YYYY-MM-DD HH:MM:SS` - Scheduled restart time

### **Process Status**
```bash
# Check if ComfyUI is running
curl -s http://localhost:8188 > /dev/null && echo "ComfyUI: Running" || echo "ComfyUI: Stopped"

# Check if JupyterLab is running  
curl -s http://localhost:8889 > /dev/null && echo "JupyterLab: Running" || echo "JupyterLab: Stopped"
```

## 🎯 Performance Optimization

### **Resource Monitoring**
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop

# Check model disk usage
du -sh /storage/ComfyUI/models/*
```

### **Memory Management**
```bash
# Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# Check VRAM usage
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## 🔄 Update & Maintenance

### **Update ComfyUI**
```bash
cd ComfyUI
git pull origin master
```

### **Update Custom Nodes**
```bash
cd ComfyUI/custom_nodes
for dir in */; do (cd "$dir" && git pull); done
```

### **Backup Configuration**
```bash
tar -czf comfyui-config-backup.tar.gz configs/ restart-control.sh
```

## 🆘 Emergency Commands

### **Kill All Processes**
```bash
sudo pkill -f comfyui
sudo pkill -f jupyter
sudo pkill -f python
```

### **Reset Everything**
```bash
# Nuclear option - complete reset
./restart-control.sh disable
rm -rf ComfyUI/
rm -rf venv/
rm /storage/ComfyUI/.init_done
./run.sh
```

### **Check for Hanging Processes**
```bash
# Find zombie or hanging processes
ps auxf | grep -E "(python|jupyter)" | grep -v grep
```

This quick reference provides all essential commands for managing your ComfyUI Paperspace environment efficiently.