#!/usr/bin/env python3
"""
Force JupyterLab to start at root directory (/)
Python version for better control
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def kill_existing_jupyter():
    """Kill any existing JupyterLab instances"""
    print("🛑 Stopping existing JupyterLab instances...")
    subprocess.run(["pkill", "-f", "jupyter-lab"], stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-f", "jupyter lab"], stderr=subprocess.DEVNULL)
    time.sleep(2)

def create_config():
    """Create JupyterLab configuration for root access"""
    print("📝 Creating root access configuration...")
    
    config_dir = Path.home() / ".jupyter"
    config_dir.mkdir(exist_ok=True)
    
    config_file = config_dir / "jupyter_lab_config.py"
    
    config_content = """
# JupyterLab Configuration - FORCE ROOT ACCESS
c = get_config()

# Change to root directory immediately
import os
os.chdir('/')

# Force all paths to root
c.ServerApp.root_dir = '/'
c.ServerApp.preferred_dir = '/'
c.ServerApp.notebook_dir = '/'
c.FileContentsManager.root_dir = '/'
c.ContentsManager.root_dir = '/'
c.NotebookApp.notebook_dir = '/'

# Allow root user
c.ServerApp.allow_root = True

# Network configuration
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8889
c.ServerApp.open_browser = False

# No authentication
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.disable_check_xsrf = False

# Enable features
c.ServerApp.terminals_enabled = True
c.ServerApp.allow_origin = '*'
c.ServerApp.trust_xheaders = True

# Terminal settings
c.TerminalManager.cwd = '/'

print("✅ Configuration: Root directory = /")
"""
    
    with open(config_file, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Config written to: {config_file}")

def create_workspace():
    """Create workspace configuration to force root directory"""
    print("📁 Creating workspace configuration...")
    
    workspace_dir = Path.home() / ".jupyter" / "lab" / "workspaces"
    workspace_dir.mkdir(parents=True, exist_ok=True)
    
    # Create default workspace
    workspace_data = {
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
                    "collapsed": False,
                    "current": "filebrowser",
                    "widgets": ["filebrowser"]
                }
            }
        },
        "metadata": {
            "id": "default"
        }
    }
    
    workspace_file = workspace_dir / "default-1.json"
    with open(workspace_file, 'w') as f:
        json.dump(workspace_data, f, indent=2)
    
    print(f"✅ Workspace config: {workspace_file}")

def setup_dark_mode():
    """Configure dark mode for JupyterLab"""
    print("🌙 Setting up dark mode...")
    
    settings_dir = Path.home() / ".jupyter" / "lab" / "user-settings" / "@jupyterlab" / "apputils-extension"
    settings_dir.mkdir(parents=True, exist_ok=True)
    
    theme_settings = {
        "theme": "JupyterLab Dark",
        "theme-scrollbars": True
    }
    
    settings_file = settings_dir / "themes.jupyterlab-settings"
    with open(settings_file, 'w') as f:
        json.dump(theme_settings, f, indent=2)
    
    print("✅ Dark mode configured")

def start_jupyter():
    """Start JupyterLab with root directory access"""
    print("\n🚀 Starting JupyterLab with ROOT access...")
    
    # Change to root directory
    os.chdir('/')
    print(f"📍 Current directory: {os.getcwd()}")
    
    # Build command
    cmd = [
        sys.executable, "-m", "jupyter", "lab",
        "--allow-root",
        "--no-browser",
        "--ip=0.0.0.0",
        "--port=8889",
        "--ServerApp.root_dir=/",
        "--ServerApp.preferred_dir=/",
        "--ServerApp.notebook_dir=/",
        "--ContentsManager.root_dir=/",
        "--FileContentsManager.root_dir=/",
        "--ServerApp.token=",
        "--ServerApp.password=",
        "--ServerApp.terminals_enabled=True",
        "--TerminalManager.cwd=/",
        "--ServerApp.allow_origin=*"
    ]
    
    # Get Paperspace URL if available
    fqdn = os.environ.get('PAPERSPACE_FQDN', '')
    
    if fqdn:
        print(f"\n🌐 Paperspace detected: {fqdn}")
        url = f"https://{fqdn}:8889/lab?path=/"
        print(f"\n📍 Access URL: {url}")
    else:
        print("\n📍 Local environment")
        url = "http://localhost:8889/lab?path=/"
        print(f"\n📍 Access URL: {url}")
    
    print("\n" + "="*50)
    print("✅ JUPYTERLAB STARTING WITH ROOT ACCESS")
    print("="*50)
    print(f"🌐 URL: {url}")
    print("📁 Root Directory: /")
    print("🔧 Config: ~/.jupyter/jupyter_lab_config.py")
    print("="*50)
    print("\nYou should see:")
    print("  • Root directory (/) in file browser")
    print("  • All directories: /etc, /usr, /var, /storage")
    print("  • Full filesystem access")
    print("\nTroubleshooting:")
    print("  1. Clear browser cache")
    print("  2. Use incognito mode")
    print("  3. Manually navigate to / in file browser")
    print("="*50)
    print("\nPress Ctrl+C to stop\n")
    
    # Start JupyterLab
    try:
        process = subprocess.Popen(cmd, cwd="/")
        process.wait()
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping JupyterLab...")
        process.terminate()
        time.sleep(2)
        if process.poll() is None:
            process.kill()
        print("✅ JupyterLab stopped")

def main():
    """Main execution"""
    print("🔧 JUPYTERLAB ROOT ACCESS CONFIGURATOR")
    print("="*40)
    
    # Step 1: Kill existing instances
    kill_existing_jupyter()
    
    # Step 2: Create configuration
    create_config()
    
    # Step 3: Create workspace
    create_workspace()
    
    # Step 4: Setup dark mode
    setup_dark_mode()
    
    # Step 5: Start JupyterLab
    start_jupyter()

if __name__ == "__main__":
    main()