#!/bin/bash
# Fix JupyterLab to open in root directory

echo "🔧 Fixing JupyterLab Root Directory Access"
echo "=========================================="

# Kill existing JupyterLab if running
echo "Stopping existing JupyterLab instances..."
pkill -f "jupyter-lab" 2>/dev/null || true
pkill -f "jupyter lab" 2>/dev/null || true
sleep 2

# Create proper configuration
echo "Creating root access configuration..."
mkdir -p ~/.jupyter

cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
# JupyterLab Configuration with Full Root Access
c = get_config()

# ROOT ACCESS CONFIGURATION
c.ServerApp.root_dir = '/'  # Full filesystem access
c.ServerApp.notebook_dir = '/'  # Start at root
c.ServerApp.preferred_dir = '/'  # Preferred directory

# Allow root user
c.ServerApp.allow_root = True

# Network settings for Paperspace
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8889
c.ServerApp.open_browser = False

# Authentication (disabled for easy access)
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = False

# CORS and security
c.ServerApp.allow_origin = '*'
c.ServerApp.allow_credentials = True
c.ServerApp.trust_xheaders = True

# Performance
c.ServerApp.max_buffer_size = 268435456
c.ServerApp.iopub_msg_rate_limit = 1000.0

# File browser settings
c.FileContentsManager.root_dir = '/'
c.ContentsManager.root_dir = '/'

# Terminal settings (full root access)
c.ServerApp.terminals_enabled = True
c.TerminalManager.shell_command = ['/bin/bash']

print("✅ JupyterLab configured for FULL ROOT ACCESS at /")
EOF

# Create JupyterLab settings for dark mode
echo "Configuring JupyterLab dark mode..."
mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/terminal-extension
mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/codemirror-extension

# Set dark theme as default
cat > ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings << 'EOF'
{
    "theme": "JupyterLab Dark",
    "theme-scrollbars": true,
    "overrides": {
        "code-font-size": "14px",
        "content-font-size1": "14px",
        "ui-font-size1": "13px"
    }
}
EOF

# Configure terminal dark theme
cat > ~/.jupyter/lab/user-settings/@jupyterlab/terminal-extension/plugin.jupyterlab-settings << 'EOF'
{
    "theme": "dark",
    "fontSize": 14,
    "lineHeight": 1.2,
    "fontFamily": "Menlo, Monaco, 'Courier New', monospace"
}
EOF

# Configure code editor dark theme
cat > ~/.jupyter/lab/user-settings/@jupyterlab/codemirror-extension/plugin.jupyterlab-settings << 'EOF'
{
    "theme": "material-darker",
    "keyMap": "default",
    "scrollPastEnd": 0.5,
    "styleActiveLine": true,
    "styleSelectedText": true,
    "selectionPointer": true,
    "highlightActiveLine": true,
    "highlightSelectionMatches": true,
    "showTrailingSpace": true
}
EOF

echo "✅ Dark mode configured"

# Set permissions
chmod 600 ~/.jupyter/jupyter_lab_config.py

# Start JupyterLab with explicit root parameters
echo ""
echo "Starting JupyterLab with root directory access..."
echo "=========================================="

# Paperspace environment
if [ -n "$PAPERSPACE_FQDN" ]; then
    echo "📍 Paperspace environment detected"
    
    # Start with all root parameters
    jupyter lab \
        --allow-root \
        --ip=0.0.0.0 \
        --port=8889 \
        --no-browser \
        --notebook-dir=/ \
        --ServerApp.root_dir=/ \
        --ServerApp.preferred_dir=/ \
        --FileContentsManager.root_dir=/ \
        --ContentsManager.root_dir=/ \
        --ServerApp.token='' \
        --ServerApp.password='' \
        --ServerApp.allow_origin='*' \
        --ServerApp.trust_xheaders=True \
        --ServerApp.disable_check_xsrf=False \
        --ServerApp.allow_credentials=True &
    
    JUPYTER_PID=$!
    echo $JUPYTER_PID > /tmp/jupyterlab.pid
    
    sleep 3
    
    echo ""
    echo "=========================================="
    echo "✅ JupyterLab Started with ROOT ACCESS!"
    echo "=========================================="
    echo ""
    echo "🌐 Access URLs:"
    echo "   Primary: https://$PAPERSPACE_FQDN:8889/lab"
    echo "   Alternative: https://$PAPERSPACE_FQDN:8889"
    echo ""
    echo "📁 ROOT ACCESS ENABLED:"
    echo "   • Full filesystem access from /"
    echo "   • All directories visible"
    echo "   • Terminal with root privileges"
    echo ""
    echo "🛑 To stop: pkill -f jupyter-lab"
    echo "=========================================="
    
else
    # Local environment
    echo "📍 Local environment"
    
    jupyter lab \
        --allow-root \
        --ip=0.0.0.0 \
        --port=8889 \
        --no-browser \
        --notebook-dir=/ \
        --ServerApp.root_dir=/ \
        --ServerApp.token='' \
        --ServerApp.password='' &
    
    JUPYTER_PID=$!
    echo $JUPYTER_PID > /tmp/jupyterlab.pid
    
    echo ""
    echo "✅ JupyterLab started at http://localhost:8889"
    echo "📁 Root directory access enabled: /"
fi

# Show process info
echo ""
echo "Process ID: $JUPYTER_PID"
echo "Config: ~/.jupyter/jupyter_lab_config.py"