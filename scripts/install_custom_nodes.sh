#!/bin/bash
# install_custom_nodes.sh - Properly install ComfyUI custom nodes with their requirements

echo "==> Installing ComfyUI Custom Nodes with requirements..."

# Ensure we're in the right directory
cd /comfyui-paperspace-notebook || cd /notebooks/comfyui-paperspace-notebook || {
    echo "❌ Could not find project directory"
    exit 1
}

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    echo "==> Activating virtual environment..."
    source venv/bin/activate
fi

# Create custom_nodes directory if it doesn't exist
mkdir -p ComfyUI/custom_nodes

# Function to install a custom node properly
install_custom_node() {
    local repo_url=$1
    local repo_name=$(basename "$repo_url" .git)
    local target_dir="ComfyUI/custom_nodes/$repo_name"
    
    echo "==> Installing $repo_name..."
    
    # Clone or update the repository
    if [ ! -d "$target_dir" ]; then
        echo "   Cloning $repo_name..."
        git clone "$repo_url" "$target_dir" || {
            echo "   ❌ Failed to clone $repo_name"
            return 1
        }
    else
        echo "   Updating $repo_name..."
        cd "$target_dir" && git pull && cd - > /dev/null
    fi
    
    # Install requirements if they exist
    if [ -f "$target_dir/requirements.txt" ]; then
        echo "   Installing requirements for $repo_name..."
        pip install -r "$target_dir/requirements.txt" || {
            echo "   ⚠️ Some requirements failed, trying with upgrade..."
            pip install -U -r "$target_dir/requirements.txt"
        }
    else
        echo "   No requirements.txt for $repo_name"
    fi
    
    # Some nodes have install.py scripts
    if [ -f "$target_dir/install.py" ]; then
        echo "   Running install.py for $repo_name..."
        cd "$target_dir" && python install.py && cd - > /dev/null
    fi
    
    echo "   ✅ $repo_name installed"
    echo
}

# Essential custom nodes (in order of importance)
echo "==> Installing essential custom nodes..."

# ComfyUI Manager (most important)
install_custom_node "https://github.com/ltdrdata/ComfyUI-Manager.git"

# Impact Pack (very useful)
install_custom_node "https://github.com/ltdrdata/ComfyUI-Impact-Pack.git"

# IPAdapter Plus
install_custom_node "https://github.com/cubiq/ComfyUI_IPAdapter_plus.git"

# WAS Node Suite
install_custom_node "https://github.com/WASasquatch/was-node-suite-comfyui.git"

# RGThree
install_custom_node "https://github.com/rgthree/rgthree-comfy.git"

# Efficiency Nodes
install_custom_node "https://github.com/jags111/efficiency-nodes-comfyui.git"

# Ultimate SD Upscale
install_custom_node "https://github.com/ssitu/ComfyUI_UltimateSDUpscale.git"

# AnimateDiff Evolved (for animations)
install_custom_node "https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git"

# ControlNet Aux
install_custom_node "https://github.com/Fannovel16/comfyui_controlnet_aux.git"

# Try ReActor Node (might fail due to auth)
install_custom_node "https://github.com/Gourieff/ComfyUI-ReActor-Node.git" || echo "⚠️ ReActor Node skipped (auth issues)"

echo "==> Custom nodes installation completed!"
echo "==> You can install more nodes via ComfyUI Manager interface"