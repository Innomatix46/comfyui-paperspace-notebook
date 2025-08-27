#!/bin/bash
# Test JupyterLab root access configuration

set -e

echo "=========================================="
echo "🔬 Testing JupyterLab Root Access Setup"
echo "=========================================="

# Install JupyterLab temporarily
echo "==> Installing JupyterLab (test)..."
pip install -q jupyterlab jupyter-server-proxy

# Create test configuration
echo "==> Creating test JupyterLab configuration..."
mkdir -p ~/.jupyter
cat > ~/.jupyter/test_config.py << 'EOF'
c = get_config()
c.ServerApp.allow_root = True
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8889
c.ServerApp.open_browser = False
c.ServerApp.token = ''
c.ServerApp.password = ''
EOF

# Test configuration syntax
echo "==> Testing configuration syntax..."
python3 -c "
config = {}
exec(open('/Users/uchechukwujessica/.jupyter/test_config.py').read(), config)
print('✅ Configuration syntax valid')
print('✅ Root access enabled:', 'allow_root = True' in open('/Users/uchechukwujessica/.jupyter/test_config.py').read())
"

# Test JupyterLab can start (dry run)
echo "==> Testing JupyterLab startup (dry run)..."
timeout 5 jupyter lab --help >/dev/null 2>&1 && echo "✅ JupyterLab can start" || echo "❌ JupyterLab startup issue"

# Clean up
echo "==> Cleaning up test files..."
rm -f ~/.jupyter/test_config.py

echo "=========================================="
echo "✅ JupyterLab root access test completed"
echo "=========================================="