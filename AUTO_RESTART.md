# Auto-Restart Feature Documentation

The ComfyUI Paperspace Notebook includes an intelligent auto-restart system that maintains system stability and performance through scheduled restarts every 6 hours.

## Features

### 🔄 **Automatic 6-Hour Restarts**
- **Graceful Shutdown**: Properly terminates ComfyUI and JupyterLab processes
- **GPU Cleanup**: Resets GPU state to prevent memory leaks
- **Process Management**: Tracks and manages all related processes
- **Self-Healing**: Automatically recovers from crashes or hangs

### 📊 **System Monitoring**
- **Resource Tracking**: Monitors memory and disk usage
- **Event Logging**: Detailed logs of all restart events
- **Status Reporting**: Real-time status of auto-restart service
- **Next Restart Prediction**: Shows when the next restart will occur

### 🛠️ **Control Interface**
- **Enable/Disable**: Toggle auto-restart on/off
- **Force Restart**: Manually trigger immediate restart
- **Status Check**: View current auto-restart status
- **Log Viewing**: Access detailed restart logs

## Usage

### Basic Control Commands

```bash
# Check auto-restart status
./restart-control.sh status

# Enable auto-restart (starts automatically with run.sh)
./restart-control.sh enable

# Disable auto-restart
./restart-control.sh disable

# Force immediate restart
./restart-control.sh force-restart

# View restart logs
./restart-control.sh logs
```

### Status Information

The status command shows:
- **Service Status**: Whether auto-restart is running
- **Next Restart Time**: When the next restart will occur
- **Time Remaining**: Countdown to next restart
- **Recent Events**: Last 10 restart log entries
- **System Resources**: Current memory and disk usage

## Configuration

### Auto-Restart Settings

The auto-restart system can be configured by modifying `scripts/auto_restart.sh`:

```bash
# Restart interval (default: 6 hours = 21600 seconds)
AUTO_RESTART_INTERVAL=21600

# Log file location
RESTART_LOG_FILE="/storage/ComfyUI/restart.log"

# PID file for process tracking
RESTART_PID_FILE="/storage/ComfyUI/auto_restart.pid"
```

### Customization Options

You can modify the restart behavior:

1. **Change Restart Interval**:
   ```bash
   # For 4-hour restarts
   AUTO_RESTART_INTERVAL=14400
   
   # For 8-hour restarts  
   AUTO_RESTART_INTERVAL=28800
   ```

2. **Add Pre-Restart Actions**:
   ```bash
   # Add custom cleanup before restart
   cleanup_custom_data() {
       echo "Cleaning up custom data..."
       # Your cleanup code here
   }
   ```

3. **Modify Shutdown Grace Period**:
   ```bash
   # Increase wait time for graceful shutdown
   kill -TERM $comfyui_pids
   sleep 20  # Wait 20 seconds instead of 10
   ```

## Log Files

### Restart Log (`/storage/ComfyUI/restart.log`)

Contains timestamped entries for:
- **Startup Events**: Auto-restart service initialization
- **Restart Events**: Scheduled and manual restarts
- **Shutdown Events**: Graceful termination of processes
- **System Status**: Resource usage at restart time
- **Error Events**: Any issues during restart process

Example log entries:
```
[2025-08-27 14:30:00] 🚀 Auto-restart service starting - Next restart at: 2025-08-27 20:30:00
[2025-08-27 20:30:00] 🔄 Scheduled restart initiated - Next restart at: 2025-08-28 02:30:00
[2025-08-27 20:30:01] 📊 System Status - Memory: 8.2GB/16GB, Storage: 45GB/100GB (45% used)
[2025-08-27 20:30:01] 🛑 Initiating graceful shutdown for restart...
[2025-08-27 20:30:05] ✅ Graceful shutdown completed
[2025-08-27 20:30:10] 🎯 Restarting ComfyUI application...
```

## Technical Details

### Process Management

The auto-restart system uses several mechanisms to ensure reliability:

1. **PID Tracking**: Maintains PID files to track running processes
2. **Signal Handling**: Uses SIGTERM for graceful shutdown, SIGKILL as fallback
3. **Process Trees**: Identifies and manages child processes
4. **Cleanup Verification**: Confirms processes are fully terminated

### Safety Features

1. **Graceful Shutdown**: Always attempts clean shutdown first
2. **Timeout Protection**: Forces termination if graceful shutdown fails
3. **GPU Reset**: Clears GPU memory to prevent accumulation
4. **Log Rotation**: Prevents log files from growing too large
5. **Error Recovery**: Handles and logs any restart failures

### Integration

The auto-restart system integrates seamlessly with:
- **ComfyUI Main Process**: Monitors and restarts the AI application
- **JupyterLab Service**: Includes development environment in restarts
- **Paperspace Platform**: Works with Paperspace's notebook environment
- **Storage Persistence**: Maintains data across restarts

## Troubleshooting

### Common Issues

1. **Auto-restart not starting**:
   ```bash
   # Check if files exist
   ls -la /storage/ComfyUI/auto_restart.pid
   ls -la /storage/ComfyUI/scheduler.pid
   
   # Check process status
   ./restart-control.sh status
   ```

2. **Restarts not happening**:
   ```bash
   # Check scheduler process
   ps aux | grep auto_restart
   
   # View logs for errors
   ./restart-control.sh logs
   ```

3. **Processes not shutting down gracefully**:
   ```bash
   # Check for hanging processes
   ps aux | grep -E "(python.*main.py|jupyter.*lab)"
   
   # View detailed restart logs
   tail -50 /storage/ComfyUI/restart.log
   ```

### Manual Recovery

If auto-restart gets stuck:

```bash
# Stop all related processes
./restart-control.sh disable

# Clean up any remaining processes
pkill -f "python.*main.py"
pkill -f "jupyter.*lab"

# Remove PID files
rm -f /storage/ComfyUI/auto_restart.pid
rm -f /storage/ComfyUI/scheduler.pid

# Restart the system
./run.sh
```

## Best Practices

1. **Monitor Logs Regularly**: Check restart logs for any issues
2. **Test Changes**: Use `force-restart` to test configuration changes
3. **Resource Planning**: Ensure sufficient resources for clean restarts
4. **Backup Important Work**: Auto-restart will terminate running processes
5. **Customize Intervals**: Adjust restart frequency based on your needs

## Advanced Configuration

### Environment Variables

You can override default settings with environment variables:

```bash
# Custom restart interval (in seconds)
export COMFYUI_RESTART_INTERVAL=14400

# Custom log file location
export COMFYUI_RESTART_LOG="/custom/path/restart.log"

# Disable auto-restart on startup
export COMFYUI_AUTO_RESTART_DISABLED=true
```

### Integration with External Monitoring

```bash
# Send notifications to external systems
send_external_notification() {
    local event="$1"
    # Add webhook call, email, or other notification here
    curl -X POST "https://your-webhook.com/restart" \
         -d "{\"event\":\"$event\",\"timestamp\":\"$(date)\"}"
}
```

This comprehensive auto-restart system ensures your ComfyUI Paperspace environment maintains optimal performance and reliability through intelligent, scheduled maintenance restarts.