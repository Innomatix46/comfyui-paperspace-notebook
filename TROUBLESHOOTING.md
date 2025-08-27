# ComfyUI Paperspace Notebook - Troubleshooting Guide

This comprehensive troubleshooting guide covers common issues and their solutions for the ComfyUI Paperspace environment with auto-restart functionality.

## 🚨 Quick Diagnostics

### System Health Check
```bash
# Run complete system diagnostic
./restart-control.sh status

# Check all services
ps aux | grep -E "(python.*main.py|jupyter.*lab)"

# Verify GPU status
nvidia-smi

# Check disk space
df -h /storage

# Check memory usage
free -h
```

## ⏰ Auto-Restart Issues

### Auto-Restart Not Starting

**Symptoms:**
- `./restart-control.sh status` shows "NOT RUNNING"
- No scheduler PID found
- Auto-restart service fails to initialize

**Solutions:**
```bash
# 1. Check for conflicting processes
ps aux | grep auto_restart

# 2. Clean up PID files
rm -f /storage/ComfyUI/auto_restart.pid
rm -f /storage/ComfyUI/scheduler.pid

# 3. Restart the system
./run.sh

# 4. Enable manually if needed
./restart-control.sh enable
```

### Auto-Restart Not Executing

**Symptoms:**
- Status shows "RUNNING" but no restarts occur
- Time remaining never decreases
- No restart log entries

**Solutions:**
```bash
# 1. Check scheduler process
ps aux | grep -f /storage/ComfyUI/scheduler.pid

# 2. View detailed logs
tail -50 /storage/ComfyUI/restart.log

# 3. Force restart to test
./restart-control.sh force-restart

# 4. Restart auto-restart service
./restart-control.sh disable
./restart-control.sh enable
```

### Restart Loops or Premature Restarts

**Symptoms:**
- System restarts more frequently than 6 hours
- Continuous restart cycles
- Applications don't stay running

**Solutions:**
```bash
# 1. Check for multiple restart processes
ps aux | grep -E "(auto_restart|scheduler)"

# 2. Stop all restart processes
./restart-control.sh disable
pkill -f auto_restart

# 3. Check system resources
./restart-control.sh logs | grep -E "(memory|disk|error)"

# 4. Restart with clean slate
./run.sh
```

## 🖥️ ComfyUI Issues

### ComfyUI Won't Start

**Symptoms:**
- Port 8188 not accessible
- Python main.py process not running
- Import errors in logs

**Solutions:**
```bash
# 1. Check virtual environment
source venv/bin/activate
which python

# 2. Check ComfyUI directory
ls -la ComfyUI/
ls -la ComfyUI/main.py

# 3. Manual start for debugging
cd ComfyUI
python main.py --listen --port 8188

# 4. Check dependencies
pip list | grep torch
```

### Models Not Loading

**Symptoms:**
- Models not appearing in ComfyUI
- "Model not found" errors
- Empty model directories

**Solutions:**
```bash
# 1. Check model symlinks
ls -la ComfyUI/models/
ls -la /storage/ComfyUI/models/

# 2. Verify download completion
ls -la /storage/ComfyUI/.init_done

# 3. Re-download models
rm /storage/ComfyUI/.init_done
./run.sh

# 4. Check model file permissions
chmod 644 /storage/ComfyUI/models/*/*.safetensors
```

### Custom Nodes Not Working

**Symptoms:**
- Custom nodes not appearing
- Import errors for custom nodes
- Node installation failures

**Solutions:**
```bash
# 1. Check custom nodes directory
ls -la ComfyUI/custom_nodes/

# 2. Check installation logs
grep -r "error" ComfyUI/custom_nodes/*/

# 3. Manual node installation
cd ComfyUI/custom_nodes
git clone https://github.com/ltdrdata/ComfyUI-Manager

# 4. Install node requirements
cd ComfyUI/custom_nodes/ComfyUI-Manager
pip install -r requirements.txt
```

## 🔬 JupyterLab Issues

### JupyterLab Won't Start

**Symptoms:**
- Port 8889 not accessible
- Jupyter process not running
- Configuration errors

**Solutions:**
```bash
# 1. Check JupyterLab process
ps aux | grep jupyter

# 2. Check configuration
cat ~/.jupyter/jupyter_lab_config.py

# 3. Manual start for debugging
jupyter lab --allow-root --ip=0.0.0.0 --port=8889

# 4. Restart JupyterLab service
source scripts/configure_jupyterlab.sh
configure_jupyterlab
start_jupyterlab
```

### Root Access Denied

**Symptoms:**
- "Running as root is not recommended" errors
- Permission denied errors
- Cannot install packages

**Solutions:**
```bash
# 1. Check root configuration
grep "allow_root" ~/.jupyter/jupyter_lab_config.py

# 2. Regenerate configuration
rm ~/.jupyter/jupyter_lab_config.py
source scripts/configure_jupyterlab.sh
configure_jupyterlab

# 3. Manual root enable
jupyter lab --generate-config
echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py
```

## 💾 Storage and Memory Issues

### Disk Space Full

**Symptoms:**
- "No space left on device" errors
- Downloads failing
- Applications crashing

**Solutions:**
```bash
# 1. Check disk usage
df -h
du -sh /storage/ComfyUI/models/*

# 2. Clean up temporary files
rm -rf /tmp/*
rm -rf ComfyUI/output/temp/*

# 3. Remove unused models
# Edit configs/models.txt to remove unwanted downloads
rm /storage/ComfyUI/.init_done
./run.sh

# 4. Archive old outputs
tar -czf outputs-backup.tar.gz /storage/ComfyUI/output/
rm -rf /storage/ComfyUI/output/old-folder
```

### GPU Memory Issues

**Symptoms:**
- CUDA out of memory errors
- GPU not detected
- Poor performance

**Solutions:**
```bash
# 1. Check GPU status
nvidia-smi

# 2. Clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"

# 3. Reset GPU (if auto-restart enabled)
./restart-control.sh force-restart

# 4. Check GPU processes
nvidia-smi pmon
```

### System Memory Issues

**Symptoms:**
- System becoming unresponsive
- Out of memory errors
- Slow performance

**Solutions:**
```bash
# 1. Check memory usage
free -h
top

# 2. Identify memory-hungry processes
ps aux --sort=-%mem | head -10

# 3. Restart system services
./restart-control.sh force-restart

# 4. Reduce model sizes
# Edit configs/models.txt to use smaller models
```

## 🌐 Network and Access Issues

### Cannot Access Services

**Symptoms:**
- URLs not working
- Connection refused errors
- Timeouts

**Solutions:**
```bash
# 1. Check if services are running
curl -s http://localhost:8188 && echo "ComfyUI: OK" || echo "ComfyUI: FAIL"
curl -s http://localhost:8889 && echo "JupyterLab: OK" || echo "JupyterLab: FAIL"

# 2. Check Paperspace FQDN
echo $PAPERSPACE_FQDN

# 3. Check port binding
netstat -tlnp | grep -E "(8188|8889)"

# 4. Restart services
./restart-control.sh force-restart
```

### Download Failures

**Symptoms:**
- Model downloads timing out
- SSL certificate errors
- Network unreachable

**Solutions:**
```bash
# 1. Test network connectivity
ping -c 3 huggingface.co
ping -c 3 civitai.com

# 2. Check download tools
which aria2c
which wget

# 3. Manual download test
wget -O /tmp/test.html https://huggingface.co

# 4. Clear and retry
rm /storage/ComfyUI/.init_done
./run.sh
```

## 🛠️ Advanced Recovery Procedures

### Complete System Reset

When all else fails, perform a complete reset:

```bash
# 1. Stop all services
./restart-control.sh disable
pkill -f comfyui
pkill -f jupyter
pkill -f python

# 2. Clean up processes and files
rm -f /storage/ComfyUI/auto_restart.pid
rm -f /storage/ComfyUI/scheduler.pid
rm -f ~/.jupyter/jupyter_lab_config.py

# 3. Reset application state
rm -rf ComfyUI/
rm -rf venv/
rm /storage/ComfyUI/.init_done

# 4. Full restart
./run.sh
```

### Preserve Storage Reset

To reset while keeping models and outputs:

```bash
# 1. Stop services
./restart-control.sh disable

# 2. Backup important data
cp -r /storage/ComfyUI/output /tmp/output-backup

# 3. Reset application only
rm -rf ComfyUI/
rm -rf venv/

# 4. Keep models but reset init flag
rm /storage/ComfyUI/.init_done

# 5. Restart
./run.sh
```

### Debug Mode Operation

For detailed debugging:

```bash
# 1. Enable debug mode
export DEBUG=1

# 2. Run with verbose output
bash -x ./run.sh

# 3. Check all log files
tail -f /storage/ComfyUI/restart.log &
tail -f /storage/jupyterlab.log &

# 4. Monitor system resources
watch -n 1 'nvidia-smi; echo ""; free -h; echo ""; df -h /storage'
```

## 📞 Getting Help

### Log Collection

Before seeking help, collect relevant logs:

```bash
# Create diagnostic bundle
mkdir -p /tmp/comfyui-debug
cp /storage/ComfyUI/restart.log /tmp/comfyui-debug/
cp /storage/jupyterlab.log /tmp/comfyui-debug/
cp ~/.jupyter/jupyter_lab_config.py /tmp/comfyui-debug/
./restart-control.sh status > /tmp/comfyui-debug/status.txt
ps aux | grep -E "(python|jupyter)" > /tmp/comfyui-debug/processes.txt
nvidia-smi > /tmp/comfyui-debug/gpu.txt
df -h > /tmp/comfyui-debug/disk.txt
free -h > /tmp/comfyui-debug/memory.txt

# Package for sharing
tar -czf comfyui-debug.tar.gz -C /tmp comfyui-debug/
```

### Support Resources

- **Quick Reference**: `QUICK_REFERENCE.md`
- **Auto-Restart Documentation**: `AUTO_RESTART.md`
- **Main Documentation**: `README.md`
- **Configuration Examples**: Files in `configs/` directory

This troubleshooting guide should resolve most common issues. If problems persist, use the log collection procedure above and seek help from the community.