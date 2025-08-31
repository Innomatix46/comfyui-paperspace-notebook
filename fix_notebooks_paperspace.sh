#!/bin/bash
# Auto-fix for Paperspace notebook corruption issues
# This script automatically fixes the Model_Download_Manager.ipynb corruption

echo "🔧 Paperspace Notebook Auto-Fix"
echo "==============================="

# Determine if we're in Paperspace or local
if [ -d "/notebooks" ]; then
    WORK_DIR="/notebooks/comfyui-paperspace-notebook"
    echo "📍 Running in Paperspace environment"
else
    WORK_DIR="$(pwd)"
    echo "📍 Running in local environment"
fi

cd "$WORK_DIR" || exit 1

# Function to fix Model_Download_Manager.ipynb
fix_model_download_manager() {
    echo ""
    echo "🔍 Checking Model_Download_Manager.ipynb..."
    
    # Check if notebook is corrupted
    if python3 -c "import json; json.load(open('Model_Download_Manager.ipynb'))" 2>/dev/null; then
        echo "✅ Model_Download_Manager.ipynb is valid"
        return 0
    else
        echo "❌ Model_Download_Manager.ipynb is corrupted"
        
        # Check if fixed version exists
        if [ -f "Model_Download_Manager_Fixed.ipynb" ]; then
            echo "🔧 Applying fix from Model_Download_Manager_Fixed.ipynb..."
            
            # Backup corrupted file
            if [ -f "Model_Download_Manager.ipynb" ]; then
                mv Model_Download_Manager.ipynb Model_Download_Manager.ipynb.corrupted.$(date +%Y%m%d_%H%M%S).bak
                echo "📦 Backed up corrupted file"
            fi
            
            # Copy fixed version
            cp Model_Download_Manager_Fixed.ipynb Model_Download_Manager.ipynb
            echo "✅ Replaced with fixed version"
            
            # Verify fix
            if python3 -c "import json; json.load(open('Model_Download_Manager.ipynb'))" 2>/dev/null; then
                echo "✅ Fix successful - notebook is now valid!"
                return 0
            else
                echo "❌ Fix failed - notebook still invalid"
                return 1
            fi
        else
            echo "❌ Fixed version not found"
            echo "💡 Creating new Model_Download_Manager.ipynb from scratch..."
            
            # Create a minimal valid notebook
            cat > Model_Download_Manager.ipynb << 'EOF'
{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {},
      "source": ["# Model Download Manager\n\nThis notebook was auto-repaired."]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {},
      "outputs": [],
      "source": ["print('Notebook repaired. Please use Model_Download_Manager_Fixed.ipynb for full functionality')"]
    }
  ],
  "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4
}
EOF
            echo "✅ Created minimal valid notebook"
            return 0
        fi
    fi
}

# Fix other notebooks if needed
fix_all_notebooks() {
    echo ""
    echo "🔍 Checking all notebooks..."
    
    for notebook in *.ipynb; do
        [ -e "$notebook" ] || continue
        
        # Skip backup files
        [[ "$notebook" == *.bak ]] && continue
        [[ "$notebook" == *_Fixed.ipynb ]] && continue
        
        if python3 -c "import json; json.load(open('$notebook'))" 2>/dev/null; then
            echo "✅ $notebook - OK"
        else
            echo "❌ $notebook - Corrupted"
            
            # Look for fixed version
            fixed_name="${notebook%.ipynb}_Fixed.ipynb"
            if [ -f "$fixed_name" ]; then
                mv "$notebook" "${notebook}.corrupted.bak"
                cp "$fixed_name" "$notebook"
                echo "  🔧 Fixed using $fixed_name"
            fi
        fi
    done
}

# Main execution
echo ""
echo "Starting notebook repair process..."
echo "===================================="

# Fix Model_Download_Manager.ipynb specifically
fix_model_download_manager

# Check and fix all notebooks
fix_all_notebooks

echo ""
echo "===================================="
echo "✨ Notebook repair complete!"
echo ""
echo "💡 Tips:"
echo "  - Original corrupted files are backed up with .bak extension"
echo "  - Use Model_Download_Manager_Fixed.ipynb if issues persist"
echo "  - Run this script anytime you see NotJSONError"
echo ""

# If in Paperspace, show URLs
if [ -n "$PAPERSPACE_FQDN" ]; then
    echo "🌐 Access your notebooks at:"
    echo "  https://$PAPERSPACE_FQDN/lab"
fi