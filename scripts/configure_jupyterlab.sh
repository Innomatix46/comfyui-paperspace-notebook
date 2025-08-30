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
    
    echo "==> JupyterLab configuration completed"
}

# Function to start JupyterLab in background
start_jupyterlab() {
    echo "==> Starting JupyterLab server in background..."
    
    # Start JupyterLab in background
    nohup jupyter lab --config ~/.jupyter/jupyter_lab_config.py > /storage/jupyterlab.log 2>&1 &
    
    # Get the process ID
    JUPYTER_PID=$!
    echo $JUPYTER_PID > /storage/jupyterlab.pid
    
    echo "==> JupyterLab started with PID: $JUPYTER_PID"
    echo "==> JupyterLab logs available at: /storage/jupyterlab.log"
    
    # Wait a moment for startup
    sleep 3
    
    # Display access information
    if [ -n "$PAPERSPACE_FQDN" ]; then
        echo "==> JupyterLab Access URLs:"
        echo "   🔗 Main URL: https://$PAPERSPACE_FQDN:8889/"
        echo "   🔗 Direct: https://$PAPERSPACE_FQDN:8889/lab"
    else
        echo "==> JupyterLab available on port 8889"
    fi
}