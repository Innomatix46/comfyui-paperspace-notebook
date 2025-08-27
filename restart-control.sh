#!/bin/bash
# restart-control.sh - Control script for managing auto-restart functionality
# Usage: ./restart-control.sh [enable|disable|status|force-restart|logs]

set -e

# Source the auto-restart module
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/auto_restart.sh"

# Control script for auto-restart management
restart_control() {
    local command="$1"
    
    case $command in
        "enable")
            echo "🟢 Enabling auto-restart (6-hour intervals)..."
            toggle_auto_restart enable
            ;;
        "disable")
            echo "🔴 Disabling auto-restart..."
            toggle_auto_restart disable
            ;;
        "status")
            check_auto_restart_status
            ;;
        "force-restart")
            echo "🔄 Forcing immediate restart..."
            send_restart_notification "restarting"
            graceful_shutdown
            sleep 2
            echo "🚀 Restarting ComfyUI..."
            exec "$SCRIPT_DIR/run.sh"
            ;;
        "logs")
            echo "📋 Auto-Restart Logs"
            echo "==================="
            if [ -f "/storage/ComfyUI/restart.log" ]; then
                tail -50 "/storage/ComfyUI/restart.log"
            else
                echo "No restart logs found"
            fi
            ;;
        "help"|"--help"|"-h")
            echo "Auto-Restart Control Script"
            echo "=========================="
            echo "Usage: $0 [command]"
            echo ""
            echo "Commands:"
            echo "  enable        Enable 6-hour auto-restart"
            echo "  disable       Disable auto-restart"
            echo "  status        Show auto-restart status"
            echo "  force-restart Force immediate restart"
            echo "  logs          Show recent restart logs"
            echo "  help          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 enable     # Enable auto-restart"
            echo "  $0 status     # Check if auto-restart is running"
            echo "  $0 logs       # View recent restart events"
            ;;
        *)
            echo "❌ Unknown command: $command"
            echo "Use '$0 help' for usage information"
            exit 1
            ;;
    esac
}

# Run the control function with provided arguments
if [ $# -eq 0 ]; then
    restart_control "status"
else
    restart_control "$1"
fi