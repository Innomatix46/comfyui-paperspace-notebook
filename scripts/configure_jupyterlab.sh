#!/bin/bash
# configure_jupyterlab.sh - Configure JupyterLab for root access
# This script sets up JupyterLab configuration to allow root user access

configure_jupyterlab() {
    echo "==> Configuring JupyterLab for root access..."
    
    # Create JupyterLab configuration directory
    mkdir -p ~/.jupyter
    
    # Generate JupyterLab configuration if it doesn't exist
    if [ ! -f ~/.jupyter/jupyter_lab_config.py ]; then
        echo "==> Generating JupyterLab configuration..."
        jupyter lab --generate-config
    fi
    
    # Create JupyterLab configuration for root access
    echo "==> Writing JupyterLab root access configuration..."
    cat > ~/.jupyter/jupyter_lab_config.py << 'EOF'
# JupyterLab Configuration for Paperspace with Root Access
c = get_config()

# Allow root user to run JupyterLab
c.ServerApp.allow_root = True

# Network configuration for Paperspace
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8889
c.ServerApp.open_browser = False

# Security settings
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = False

# File browser settings - ROOT ACCESS für alle Ordner
c.ServerApp.root_dir = '/'  # Root-Zugriff auf gesamtes Dateisystem
c.ServerApp.notebook_dir = '/'  # Startet im Root-Verzeichnis

# Extensions (collaboration disabled to avoid dependency issues)
# c.LabApp.collaborative = True  # Disabled - requires jupyter-collaboration

# Performance settings
c.ServerApp.max_buffer_size = 268435456
c.ServerApp.rate_limit_window = 3.0
c.ServerApp.iopub_msg_rate_limit = 1000.0

# Logging
c.Application.log_level = 'INFO'
EOF
    
    # Set proper permissions
    chmod 600 ~/.jupyter/jupyter_lab_config.py
    
    # Configure dark mode as default
    echo "==> Configuring JupyterLab dark mode..."
    mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension
    mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/terminal-extension
    mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/codemirror-extension
    mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/notebook-extension
    
    # Set dark theme
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
    
    # Terminal dark theme
    cat > ~/.jupyter/lab/user-settings/@jupyterlab/terminal-extension/plugin.jupyterlab-settings << 'EOF'
{
    "theme": "dark",
    "fontSize": 14,
    "lineHeight": 1.2,
    "fontFamily": "Menlo, Monaco, 'Courier New', monospace"
}
EOF
    
    # Code editor dark theme
    cat > ~/.jupyter/lab/user-settings/@jupyterlab/codemirror-extension/plugin.jupyterlab-settings << 'EOF'
{
    "theme": "material-darker",
    "keyMap": "default",
    "scrollPastEnd": 0.5,
    "styleActiveLine": true,
    "styleSelectedText": true,
    "highlightActiveLine": true,
    "highlightSelectionMatches": true
}
EOF
    
    # Notebook settings
    cat > ~/.jupyter/lab/user-settings/@jupyterlab/notebook-extension/tracker.jupyterlab-settings << 'EOF'
{
    "codeCellConfig": {
        "lineNumbers": true,
        "lineWrap": "off",
        "matchBrackets": true,
        "autoClosingBrackets": true
    },
    "markdownCellConfig": {
        "lineNumbers": false,
        "lineWrap": "on",
        "matchBrackets": false,
        "autoClosingBrackets": false
    }
}
EOF
    
    echo "==> JupyterLab configuration completed (Dark mode enabled)"
}

# Function to start JupyterLab in background
start_jupyterlab() {
    echo "==> Starting JupyterLab server in background with root directory access..."
    
    # Ensure log directory exists
    mkdir -p /storage
    
    # FORCE change to root directory before starting
    cd /
    
    # Kill any existing JupyterLab
    pkill -f jupyter-lab 2>/dev/null || true
    pkill -f "jupyter lab" 2>/dev/null || true
    sleep 2
    
    # Start JupyterLab with maximum root forcing
    nohup jupyter lab \
        --allow-root \
        --ip=0.0.0.0 \
        --port=8889 \
        --no-browser \
        --notebook-dir=/ \
        --ServerApp.root_dir=/ \
        --ServerApp.preferred_dir=/ \
        --ContentsManager.root_dir=/ \
        --FileContentsManager.root_dir=/ \
        --ServerApp.token='' \
        --ServerApp.password='' \
        --ServerApp.allow_origin='*' \
        --ServerApp.disable_check_xsrf=False \
        --ServerApp.allow_credentials=True \
        --ServerApp.terminals_enabled=True \
        --TerminalManager.cwd=/ \
        --config ~/.jupyter/jupyter_lab_config.py > /storage/jupyterlab.log 2>&1 &
    
    # Get the process ID
    JUPYTER_PID=$!
    echo $JUPYTER_PID > /storage/jupyterlab.pid
    
    echo "==> JupyterLab started with PID: $JUPYTER_PID"
    echo "==> JupyterLab logs available at: /storage/jupyterlab.log"
    echo "==> Root directory access enabled: /"
    
    # Wait a moment for startup
    sleep 3
    
    # Display access information
    if [ -n "$PAPERSPACE_FQDN" ]; then
        echo "==> JupyterLab Access URLs:"
        echo "   🔗 Main URL: https://$PAPERSPACE_FQDN:8889/lab?path=/"
        echo "   🔗 Alternative: https://$PAPERSPACE_FQDN:8889/lab/tree/"
        echo "   📁 Root Access: Full filesystem from /"
        echo ""
        echo "   ⚠️ IMPORTANT: Add ?path=/ to force root directory"
        echo "   If not at root, navigate manually: Click folder icon → type / → Enter"
    else
        echo "==> JupyterLab available on port 8889"
        echo "   🔗 URL: http://localhost:8889/lab?path=/"
        echo "   📁 Root Access: Full filesystem from /"
    fi
}