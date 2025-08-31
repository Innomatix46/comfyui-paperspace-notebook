#!/bin/bash
# quick_fix_dependencies.sh - Schnelle Installation fehlender Dependencies mit Timeout

echo "🔧 QUICK FIX: Fehlende Dependencies installieren"
echo "=================================================="

# Function to install with timeout
install_with_timeout() {
    local package=$1
    local timeout_sec=60
    
    echo "📦 Installing $package (max ${timeout_sec}s)..."
    
    # Try with timeout
    timeout $timeout_sec pip install --no-cache-dir "$package" 2>/dev/null
    
    if [ $? -eq 124 ]; then
        echo "⏱️ Timeout reached for $package, trying with --no-deps..."
        timeout 30 pip install --no-cache-dir --no-deps "$package" 2>/dev/null
    elif [ $? -eq 0 ]; then
        echo "✅ $package installed successfully"
        return 0
    else
        echo "⚠️ Failed to install $package, skipping..."
        return 1
    fi
}

# Activate virtual environment if exists
if [ -f "venv/bin/activate" ]; then
    echo "🔄 Activating virtual environment..."
    source venv/bin/activate
fi

# Kill any hanging pip processes
echo "🧹 Cleaning up any hanging pip processes..."
pkill -f pip 2>/dev/null
sleep 1

# Essential packages that often cause issues
PACKAGES=(
    "torchsde"
    "torchaudio"
    "scipy"
    "einops"
    "av"
    "safetensors"
    "transformers"
    "accelerate"
)

echo ""
echo "📋 Packages to install: ${PACKAGES[@]}"
echo ""

# Try quick installation first (all at once with timeout)
echo "⚡ Attempting quick installation (30s timeout)..."
timeout 30 pip install --no-cache-dir torchsde torchaudio scipy einops av 2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Quick installation successful!"
else
    echo "⏱️ Quick installation timed out or failed"
    echo "🔄 Installing packages individually..."
    
    # Install packages one by one with timeout
    for package in "${PACKAGES[@]}"; do
        # Check if already installed
        if python -c "import ${package%==*}" 2>/dev/null; then
            echo "✅ $package already installed"
        else
            install_with_timeout "$package"
        fi
    done
fi

# Verify critical packages
echo ""
echo "🔍 Verifying installations..."
echo "================================"

# Check each package
python -c "
import sys
packages = {
    'torchsde': 'torchsde',
    'torchaudio': 'torchaudio', 
    'scipy': 'scipy',
    'einops': 'einops',
    'av': 'av'
}

missing = []
for name, import_name in packages.items():
    try:
        __import__(import_name)
        print(f'✅ {name} is installed')
    except ImportError:
        print(f'❌ {name} is NOT installed')
        missing.append(name)

if missing:
    print(f'\n⚠️ Missing packages: {missing}')
    print('Try running: pip install ' + ' '.join(missing))
    sys.exit(1)
else:
    print('\n✅ All critical packages installed!')
    sys.exit(0)
"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ SUCCESS: All dependencies fixed!"
    echo "=================================="
    echo "You can now start ComfyUI"
else
    echo ""
    echo "⚠️ WARNING: Some packages could not be installed"
    echo "=================================="
    echo "This might be due to network issues or pip problems."
    echo ""
    echo "Alternative solutions:"
    echo "1. Restart the Paperspace notebook and try again"
    echo "2. Use Docker mode: DOCKER_MODE=true ./run.sh"
    echo "3. Install manually: pip install torchsde torchaudio scipy einops av"
fi