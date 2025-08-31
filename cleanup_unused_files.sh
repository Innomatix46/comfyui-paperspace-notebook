#!/bin/bash
# Cleanup unused files - Interactive removal of redundant files

echo "🧹 CLEANUP: Unused Files"
echo "========================"

# Files to delete (redundant/backup/obsolete)
FILES_TO_DELETE=(
    # Backup files
    "Model_Download_Manager.ipynb.corrupted.bak"
    "Model_Download_Manager_Fixed.ipynb"  # Redundant, content copied to main file
    
    # Duplicate/redundant scripts
    "scripts/fix_dependencies.sh"  # Replaced by fix_comfyui_dependencies.sh
    "scripts/quick_fix_dependencies.sh"  # Replaced by fix_comfyui_quick.py
    "scripts/install_dependencies_fast.sh"  # Redundant with main install script
    "scripts/comprehensive_setup.sh"  # Redundant with run.sh
    
    # Test scripts (move to tests/ or delete)
    "scripts/test_docker_setup.sh"
    "scripts/test_safetensors.py"
    
    # Outdated scripts
    "scripts/quick_video_fix.sh"  # Obsolete
    "scripts/robust_installer.sh"  # Replaced by install_dependencies.sh
    "scripts/start_comfyui_safe.sh"  # Redundant with run.sh
    
    # Documentation files that should be in docs/
    "scripts/SAFETENSORS_README.md"
    
    # Redundant notebook (keep only the progress version)
    "ComfyUI_Setup_Complete.ipynb"  # Replaced by ComfyUI_Setup_with_Progress.ipynb
)

# Create backup directory
mkdir -p deleted_files_backup

echo "Files to delete:"
for file in "${FILES_TO_DELETE[@]}"; do
    if [ -f "$file" ]; then
        echo "  - $file"
    fi
done

echo ""
read -p "Delete these files? (y/N): " confirm

if [[ $confirm =~ ^[Yy]$ ]]; then
    echo ""
    echo "Deleting files..."
    
    for file in "${FILES_TO_DELETE[@]}"; do
        if [ -f "$file" ]; then
            echo "  Deleting: $file"
            # Backup before deleting
            cp "$file" "deleted_files_backup/" 2>/dev/null
            rm "$file"
        fi
    done
    
    echo ""
    echo "✅ Cleanup complete!"
    echo "📁 Backups saved in: deleted_files_backup/"
else
    echo "Cleanup cancelled."
fi

# Show remaining file structure
echo ""
echo "📊 Current file structure:"
echo "=========================="
echo "Notebooks:"
ls -1 *.ipynb 2>/dev/null | wc -l | sed 's/^/  /'
echo "Shell scripts:"
ls -1 *.sh 2>/dev/null | wc -l | sed 's/^/  /'
echo "Python scripts:"
ls -1 *.py 2>/dev/null | wc -l | sed 's/^/  /'
echo "Scripts directory:"
ls -1 scripts/ 2>/dev/null | wc -l | sed 's/^/  /'