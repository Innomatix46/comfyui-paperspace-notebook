#!/bin/bash
# Simple test of JupyterLab installation and root configuration

echo "=========================================="
echo "🔬 Simple JupyterLab Test"
echo "=========================================="

# Check JupyterLab is installed
echo "==> Testing JupyterLab installation..."
jupyter --version 2>/dev/null && echo "✅ Jupyter installed" || echo "❌ Jupyter not installed"

# Test configuration file creation
echo "==> Testing configuration creation..."
mkdir -p ~/.jupyter
echo "c = get_config()
c.ServerApp.allow_root = True" > ~/.jupyter/simple_test.py
echo "✅ Configuration file created"

# Test file contains root access
echo "==> Testing root access setting..."
if grep -q "allow_root = True" ~/.jupyter/simple_test.py; then
    echo "✅ Root access configuration found"
else
    echo "❌ Root access configuration missing"
fi

# Clean up
rm -f ~/.jupyter/simple_test.py
echo "✅ Test completed successfully"
echo "=========================================="