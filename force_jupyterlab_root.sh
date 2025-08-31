#!/bin/bash
# Force JupyterLab to start with root directory access
# This script ensures JupyterLab opens at / with full filesystem access

echo "🔧 FORCE JUPYTERLAB ROOT ACCESS"
echo "================================"

# Kill any existing JupyterLab instances
echo "Stopping existing JupyterLab..."
pkill -f jupyter-lab 2>/dev/null || true
pkill -f "jupyter lab" 2>/dev/null || true
sleep 2

# Remove old config
echo "Removing old configuration..."
rm -rf ~/.jupyter/lab/workspaces 2>/dev/null

# Create new config with absolute root path
echo "Creating root access configuration..."
mkdir -p ~/.jupyter

# Write minimal config that forces root
cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
c = get_config()

# FORCE ROOT DIRECTORY ACCESS
import os
os.chdir('/')  # Change to root directory

c.ServerApp.root_dir = '/'
c.ServerApp.preferred_dir = '/'
c.FileContentsManager.root_dir = '/'
c.ContentsManager.root_dir = '/'

# Allow root user
c.ServerApp.allow_root = True
c.Application.log_level = 'DEBUG'

# Network settings
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8889
c.ServerApp.open_browser = False

# No authentication
c.ServerApp.token = ''
c.ServerApp.password = ''

# Terminals with root
c.ServerApp.terminals_enabled = True

print("✅ JupyterLab configured for ROOT (/) access")
EOF

# Create workspace config to force root
mkdir -p ~/.jupyter/lab/workspaces
cat > ~/.jupyter/lab/workspaces/default-*.json << 'EOF'
{
  "data": {
    "file-browser-filebrowser:cwd": {
      "path": "/"
    },
    "layout-restorer:data": {
      "main": {
        "dock": {
          "type": "tab-area",
          "currentIndex": 0,
          "widgets": []
        }
      },
      "left": {
        "collapsed": false,
        "current": "filebrowser",
        "widgets": ["filebrowser", "running-sessions", "@jupyterlab/toc:plugin", "extensionmanager.main-view"]
      }
    }
  },
  "metadata": {
    "id": "default",
    "last_modified": "2024-01-01T00:00:00.000000Z",
    "created": "2024-01-01T00:00:00.000000Z"
  }
}
EOF

# Set dark mode
echo "Setting dark mode..."
mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
cat > ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings << 'EOF'
{
    "theme": "JupyterLab Dark"
}
EOF

# Export environment variables to force root
export JUPYTER_PREFER_ENV_PATH=0
export JUPYTER_PATH=/
export JUPYTER_DATA_DIR=/
export JUPYTER_RUNTIME_DIR=/tmp/jupyter

echo ""
echo "Starting JupyterLab with FORCED root access..."
echo "================================"

# Start JupyterLab with all root parameters
cd /  # Change to root before starting

if [ -n "$PAPERSPACE_FQDN" ]; then
    echo "📍 Paperspace environment detected"
    
    # Start with maximum root forcing
    jupyter lab \
        --allow-root \
        --no-browser \
        --ip=0.0.0.0 \
        --port=8889 \
        --ServerApp.root_dir=/ \
        --ServerApp.preferred_dir=/ \
        --ContentsManager.root_dir=/ \
        --FileContentsManager.root_dir=/ \
        --ServerApp.notebook_dir=/ \
        --ServerApp.token='' \
        --ServerApp.password='' \
        --ServerApp.allow_origin='*' \
        --ServerApp.disable_check_xsrf=False \
        --ServerApp.terminals_enabled=True \
        --TerminalManager.cwd=/ \
        --debug &
    
    JUPYTER_PID=$!
    echo "JupyterLab PID: $JUPYTER_PID"
    
    sleep 5
    
    echo ""
    echo "================================"
    echo "✅ JUPYTERLAB STARTED WITH ROOT ACCESS!"
    echo "================================"
    echo ""
    echo "🌐 Access URLs:"
    echo "   https://$PAPERSPACE_FQDN:8889/lab?path=/"
    echo "   https://$PAPERSPACE_FQDN:8889/lab/tree/"
    echo ""
    echo "📁 You should now see:"
    echo "   • Root directory (/) in file browser"
    echo "   • All system directories (/etc, /usr, /storage, etc.)"
    echo "   • Full filesystem navigation"
    echo ""
    echo "🔧 Troubleshooting:"
    echo "   If still not at root, try:"
    echo "   1. Clear browser cache"
    echo "   2. Use incognito/private mode"
    echo "   3. Add ?path=/ to URL"
    echo "   4. Click 'Home' icon → type '/' → Enter"
    echo ""
    echo "📝 Logs: tail -f /tmp/jupyter.log"
    echo "🛑 Stop: pkill -f jupyter-lab"
    echo "================================"
    
    # Show initial logs
    sleep 2
    echo ""
    echo "📋 JupyterLab startup logs:"
    jupyter lab list
    
else
    echo "📍 Local environment"
    
    jupyter lab \
        --allow-root \
        --no-browser \
        --ip=0.0.0.0 \
        --port=8889 \
        --ServerApp.root_dir=/ \
        --ServerApp.notebook_dir=/ \
        --ServerApp.token='' \
        --ServerApp.password='' &
    
    JUPYTER_PID=$!
    
    echo ""
    echo "✅ JupyterLab started at http://localhost:8889/lab?path=/"
    echo "📁 Root directory access enabled"
fi

# Keep script running to show logs
echo ""
echo "Press Ctrl+C to stop JupyterLab"
wait $JUPYTER_PID