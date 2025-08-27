#!/bin/bash
# auto_restart.sh - Automatic restart functionality for ComfyUI Paperspace
# This script handles 6-hour automatic restarts with graceful shutdown

# Auto-restart configuration
AUTO_RESTART_INTERVAL=21600  # 6 hours in seconds
RESTART_LOG_FILE="/storage/ComfyUI/restart.log"
RESTART_PID_FILE="/storage/ComfyUI/auto_restart.pid"

# Function to log restart events
log_restart_event() {
    local message="$1"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $message" >> "$RESTART_LOG_FILE"
    echo "==> $message"
}

# Function to gracefully shutdown ComfyUI
graceful_shutdown() {
    log_restart_event "🛑 Initiating graceful shutdown for restart..."
    
    # Find ComfyUI processes
    local comfyui_pids=$(pgrep -f "python.*main.py" || true)
    local jupyter_pids=$(pgrep -f "jupyter.*lab" || true)
    
    if [ -n "$comfyui_pids" ]; then
        log_restart_event "📱 Stopping ComfyUI processes: $comfyui_pids"
        # Send SIGTERM first for graceful shutdown
        kill -TERM $comfyui_pids
        sleep 10
        
        # Force kill if still running
        local remaining=$(pgrep -f "python.*main.py" || true)
        if [ -n "$remaining" ]; then
            log_restart_event "⚠️  Force killing remaining ComfyUI processes: $remaining"
            kill -KILL $remaining
        fi
    fi
    
    if [ -n "$jupyter_pids" ]; then
        log_restart_event "🔬 Stopping JupyterLab processes: $jupyter_pids"
        kill -TERM $jupyter_pids
        sleep 5
        
        # Force kill if still running
        local remaining_jupyter=$(pgrep -f "jupyter.*lab" || true)
        if [ -n "$remaining_jupyter" ]; then
            log_restart_event "⚠️  Force killing remaining JupyterLab processes: $remaining_jupyter"
            kill -KILL $remaining_jupyter
        fi
    fi
    
    # Clean up any remaining GPU processes
    if command -v nvidia-smi >/dev/null 2>&1; then
        log_restart_event "🔧 Cleaning up GPU processes..."
        nvidia-smi --gpu-reset-ecc=0 2>/dev/null || true
    fi
    
    log_restart_event "✅ Graceful shutdown completed"
}

# Function to send restart notifications
send_restart_notification() {
    local phase="$1"
    local next_restart_time=$(date -d "+6 hours" '+%Y-%m-%d %H:%M:%S')
    
    case $phase in
        "starting")
            log_restart_event "🚀 Auto-restart service starting - Next restart at: $next_restart_time"
            ;;
        "restarting")
            log_restart_event "🔄 Scheduled restart initiated - Next restart at: $next_restart_time"
            ;;
        "stopping")
            log_restart_event "🛑 Auto-restart service stopping"
            ;;
    esac
    
    # Log system status
    local memory_usage=$(free -h 2>/dev/null | awk '/^Mem:/ {print $3 "/" $2}' || echo "N/A")
    local disk_usage=$(df -h /storage 2>/dev/null | awk 'NR==2 {print $3 "/" $2 " (" $5 " used)"}' || echo "N/A")
    
    log_restart_event "📊 System Status - Memory: $memory_usage, Storage: $disk_usage"
}

# Function to setup auto-restart scheduler
setup_auto_restart() {
    log_restart_event "⏰ Setting up auto-restart scheduler (every 6 hours)"
    
    # Create restart log directory
    mkdir -p "$(dirname "$RESTART_LOG_FILE")"
    
    # Store current PID for restart management
    echo $$ > "$RESTART_PID_FILE"
    
    send_restart_notification "starting"
    
    # Start the restart loop in background
    (
        while true; do
            sleep $AUTO_RESTART_INTERVAL
            
            # Check if we should still be running
            if [ ! -f "$RESTART_PID_FILE" ] || [ "$(cat "$RESTART_PID_FILE" 2>/dev/null)" != "$$" ]; then
                log_restart_event "🔍 Auto-restart service stopped externally"
                exit 0
            fi
            
            send_restart_notification "restarting"
            
            # Perform graceful restart
            graceful_shutdown
            
            # Small delay before restart
            sleep 5
            
            # Restart the main application
            log_restart_event "🎯 Restarting ComfyUI application..."
            
            # Navigate back to project root and restart
            cd "$PROJECT_ROOT"
            exec "$0" "$@"
        done
    ) &
    
    # Store the scheduler PID
    local scheduler_pid=$!
    echo "$scheduler_pid" > "/storage/ComfyUI/scheduler.pid"
    
    log_restart_event "✅ Auto-restart scheduler started with PID: $scheduler_pid"
}

# Function to stop auto-restart scheduler
stop_auto_restart() {
    send_restart_notification "stopping"
    
    if [ -f "/storage/ComfyUI/scheduler.pid" ]; then
        local scheduler_pid=$(cat "/storage/ComfyUI/scheduler.pid")
        if ps -p "$scheduler_pid" > /dev/null 2>&1; then
            log_restart_event "🛑 Stopping scheduler process: $scheduler_pid"
            kill "$scheduler_pid" 2>/dev/null || true
        fi
        rm -f "/storage/ComfyUI/scheduler.pid"
    fi
    
    rm -f "$RESTART_PID_FILE"
    log_restart_event "✅ Auto-restart scheduler stopped"
}

# Function to check auto-restart status
check_auto_restart_status() {
    echo "🔍 Auto-Restart Status Check"
    echo "=============================="
    
    if [ -f "$RESTART_PID_FILE" ]; then
        local restart_pid=$(cat "$RESTART_PID_FILE")
        if ps -p "$restart_pid" > /dev/null 2>&1; then
            echo "✅ Auto-restart service: RUNNING (PID: $restart_pid)"
            
            if [ -f "/storage/ComfyUI/scheduler.pid" ]; then
                local scheduler_pid=$(cat "/storage/ComfyUI/scheduler.pid")
                if ps -p "$scheduler_pid" > /dev/null 2>&1; then
                    echo "✅ Restart scheduler: RUNNING (PID: $scheduler_pid)"
                    
                    # Calculate next restart time
                    local start_time=$(ps -p "$scheduler_pid" -o lstart= | xargs -I {} date -d "{}" +%s)
                    local current_time=$(date +%s)
                    local elapsed_time=$((current_time - start_time))
                    local remaining_time=$((AUTO_RESTART_INTERVAL - elapsed_time))
                    
                    if [ $remaining_time -gt 0 ]; then
                        local next_restart=$(date -d "+${remaining_time} seconds" '+%Y-%m-%d %H:%M:%S')
                        echo "⏰ Next restart scheduled: $next_restart"
                        echo "⌛ Time remaining: $((remaining_time / 3600))h $((remaining_time % 3600 / 60))m"
                    else
                        echo "⏰ Next restart: Due now"
                    fi
                else
                    echo "❌ Restart scheduler: NOT RUNNING"
                fi
            else
                echo "❌ Scheduler PID file not found"
            fi
        else
            echo "❌ Auto-restart service: NOT RUNNING"
        fi
    else
        echo "❌ Auto-restart service: DISABLED"
    fi
    
    # Show recent restart log entries
    if [ -f "$RESTART_LOG_FILE" ]; then
        echo ""
        echo "📋 Recent Restart Events:"
        echo "------------------------"
        tail -10 "$RESTART_LOG_FILE" 2>/dev/null || echo "No restart events logged"
    fi
}

# Function to enable/disable auto-restart
toggle_auto_restart() {
    local action="$1"
    
    case $action in
        "enable")
            if [ -f "$RESTART_PID_FILE" ] && ps -p "$(cat "$RESTART_PID_FILE")" > /dev/null 2>&1; then
                echo "⚠️  Auto-restart is already enabled"
                return 1
            fi
            setup_auto_restart
            echo "✅ Auto-restart enabled (6-hour intervals)"
            ;;
        "disable")
            if [ ! -f "$RESTART_PID_FILE" ]; then
                echo "ℹ️  Auto-restart is already disabled"
                return 1
            fi
            stop_auto_restart
            echo "✅ Auto-restart disabled"
            ;;
        "status")
            check_auto_restart_status
            ;;
        *)
            echo "Usage: toggle_auto_restart [enable|disable|status]"
            return 1
            ;;
    esac
}

# Signal handlers for graceful shutdown
trap 'stop_auto_restart; exit 0' SIGTERM SIGINT

# Export functions for use in other scripts
export -f graceful_shutdown
export -f send_restart_notification  
export -f setup_auto_restart
export -f stop_auto_restart
export -f check_auto_restart_status
export -f toggle_auto_restart