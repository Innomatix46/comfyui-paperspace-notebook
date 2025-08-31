#!/bin/bash
# Auto-fix corrupted notebooks in Paperspace environment

echo "🔧 Auto-Fix Notebooks for Paperspace"
echo "===================================="

# Check if in Paperspace
if [ -d "/notebooks" ]; then
    NOTEBOOK_DIR="/notebooks/comfyui-paperspace-notebook"
else
    NOTEBOOK_DIR="$(pwd)"
fi

echo "📂 Working directory: $NOTEBOOK_DIR"
cd "$NOTEBOOK_DIR" || exit 1

# Function to check if notebook is valid JSON
check_notebook() {
    local notebook=$1
    python3 -c "import json; json.load(open('$notebook'))" 2>/dev/null
    return $?
}

# Function to use fixed version if available
use_fixed_version() {
    local corrupted=$1
    local fixed="${corrupted%.ipynb}_Fixed.ipynb"
    
    if [ -f "$fixed" ]; then
        echo "  ✅ Found fixed version: $fixed"
        
        # Backup corrupted file
        mv "$corrupted" "${corrupted}.corrupted.bak" 2>/dev/null
        
        # Copy fixed version to original name
        cp "$fixed" "$corrupted"
        echo "  ✅ Replaced with fixed version"
        return 0
    else
        echo "  ❌ No fixed version found"
        return 1
    fi
}

# Check all notebooks
echo ""
echo "🔍 Checking notebooks..."
echo ""

CORRUPTED_COUNT=0
FIXED_COUNT=0

for notebook in *.ipynb; do
    # Skip if no notebooks found
    [ -e "$notebook" ] || continue
    
    # Skip checkpoint files
    [[ "$notebook" == *".ipynb_checkpoints"* ]] && continue
    
    # Skip already fixed versions
    [[ "$notebook" == *"_Fixed.ipynb" ]] && continue
    
    echo "Checking: $notebook"
    
    if check_notebook "$notebook"; then
        echo "  ✅ Valid JSON"
    else
        echo "  ❌ Corrupted or invalid"
        ((CORRUPTED_COUNT++))
        
        # Try to fix
        if use_fixed_version "$notebook"; then
            ((FIXED_COUNT++))
        fi
    fi
done

# Summary
echo ""
echo "===================================="
echo "📊 Summary:"
echo "  Corrupted notebooks found: $CORRUPTED_COUNT"
echo "  Successfully fixed: $FIXED_COUNT"

if [ $CORRUPTED_COUNT -eq 0 ]; then
    echo ""
    echo "✅ All notebooks are valid!"
elif [ $FIXED_COUNT -eq $CORRUPTED_COUNT ]; then
    echo ""
    echo "✅ All corrupted notebooks have been fixed!"
else
    echo ""
    echo "⚠️ Some notebooks could not be fixed automatically"
    echo "Please check the corrupted files manually"
fi

# Special handling for Model_Download_Manager
if [ -f "Model_Download_Manager.ipynb.corrupted.bak" ] && [ -f "Model_Download_Manager_Fixed.ipynb" ]; then
    echo ""
    echo "💡 Model_Download_Manager has been fixed!"
    echo "   The original corrupted file is backed up as:"
    echo "   Model_Download_Manager.ipynb.corrupted.bak"
fi

echo ""
echo "===================================="
echo "✨ Auto-fix complete!"