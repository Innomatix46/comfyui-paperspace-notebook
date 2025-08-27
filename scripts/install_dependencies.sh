#!/bin/bash
# install_dependencies.sh - Install ComfyUI and dependencies
# This script handles the initial setup of ComfyUI and all required dependencies

install_dependencies() {
    echo "==> Starting dependency installation..."
    
    # Clone ComfyUI repository if not present
    echo "==> Checking for ComfyUI repository..."
    if [ ! -d "ComfyUI" ]; then
        echo "==> Cloning ComfyUI repository..."
        git clone https://github.com/comfyanonymous/ComfyUI.git
        echo "==> ComfyUI repository cloned successfully"
    else
        echo "==> ComfyUI repository already exists, skipping clone"
    fi
    
    # Create Python virtual environment if not present
    echo "==> Checking for Python virtual environment..."
    if [ ! -d "venv" ]; then
        echo "==> Creating Python 3.10 virtual environment..."
        python3.10 -m venv venv
        echo "==> Virtual environment created successfully"
    else
        echo "==> Virtual environment already exists, skipping creation"
    fi
    
    # Activate virtual environment
    echo "==> Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip to latest version
    echo "==> Upgrading pip to latest version..."
    pip install --upgrade pip
    
    # Install Python packages from requirements
    echo "==> Installing Python packages from requirements..."
    if [ -f "configs/python_requirements.txt" ]; then
        pip install -r configs/python_requirements.txt
        echo "==> Python packages installed successfully"
    else
        echo "==> Warning: configs/python_requirements.txt not found, skipping Python package installation"
    fi
    
    # Install JupyterLab with root access support
    echo "==> Installing JupyterLab with root access support..."
    pip install jupyterlab jupyter-server-proxy
    echo "==> JupyterLab installed successfully"
    
    # Install custom nodes
    echo "==> Installing custom ComfyUI nodes..."
    if [ -f "configs/custom_nodes.txt" ]; then
        # Create custom_nodes directory if it doesn't exist
        mkdir -p ComfyUI/custom_nodes
        
        # Read each line from custom_nodes.txt
        while IFS= read -r line || [ -n "$line" ]; do
            # Skip empty lines and comments
            if [[ -z "$line" || "$line" =~ ^[[:space:]]*# ]]; then
                continue
            fi
            
            # Extract repository name from URL
            repo_name=$(basename "$line" .git)
            target_dir="ComfyUI/custom_nodes/$repo_name"
            
            echo "==> Processing custom node: $repo_name"
            
            # Clone repository if directory doesn't exist
            if [ ! -d "$target_dir" ]; then
                echo "==> Cloning $repo_name..."
                git clone "$line" "$target_dir"
                echo "==> Successfully cloned $repo_name"
                
                # Check for and install node-specific requirements
                if [ -f "$target_dir/requirements.txt" ]; then
                    echo "==> Installing requirements for $repo_name..."
                    pip install -r "$target_dir/requirements.txt"
                    echo "==> Requirements for $repo_name installed successfully"
                fi
            else
                echo "==> Custom node $repo_name already exists, skipping clone"
            fi
        done < configs/custom_nodes.txt
        
        echo "==> Custom nodes installation completed"
    else
        echo "==> Warning: configs/custom_nodes.txt not found, skipping custom nodes installation"
    fi
    
    echo "==> Dependency installation completed successfully"
}