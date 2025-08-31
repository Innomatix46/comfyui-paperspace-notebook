#!/usr/bin/env python3
"""
Verify and repair Jupyter notebooks in the repository
"""

import json
import os
import sys
from pathlib import Path

def verify_notebook(notebook_path):
    """Verify if a notebook file is valid JSON"""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Check for required notebook structure
        if 'cells' not in data:
            return False, "Missing 'cells' key"
        if not isinstance(data['cells'], list):
            return False, "'cells' is not a list"
            
        # Verify each cell
        for i, cell in enumerate(data['cells']):
            if 'cell_type' not in cell:
                return False, f"Cell {i} missing 'cell_type'"
            if cell['cell_type'] not in ['code', 'markdown', 'raw']:
                return False, f"Cell {i} has invalid cell_type: {cell['cell_type']}"
                
        return True, "Valid notebook"
        
    except json.JSONDecodeError as e:
        return False, f"JSON decode error: {e}"
    except Exception as e:
        return False, f"Error: {e}"

def repair_notebook(notebook_path):
    """Attempt to repair a corrupted notebook"""
    fixed_path = notebook_path.replace('.ipynb', '_Fixed.ipynb')
    
    if Path(fixed_path).exists():
        print(f"  ✅ Fixed version already exists: {fixed_path}")
        return fixed_path
        
    print(f"  ❌ Cannot auto-repair, fixed version not found")
    return None

def main():
    """Main verification routine"""
    print("🔍 Verifying Jupyter Notebooks")
    print("=" * 40)
    
    # Find all notebooks
    root = Path('/notebooks/comfyui-paperspace-notebook') if Path('/notebooks').exists() else Path('.')
    notebooks = list(root.glob('**/*.ipynb'))
    
    if not notebooks:
        print("No notebooks found")
        return 0
        
    corrupted = []
    valid = []
    
    for notebook in notebooks:
        # Skip checkpoint files
        if '.ipynb_checkpoints' in str(notebook):
            continue
            
        rel_path = notebook.relative_to(root)
        is_valid, message = verify_notebook(notebook)
        
        if is_valid:
            print(f"✅ {rel_path}: {message}")
            valid.append(notebook)
        else:
            print(f"❌ {rel_path}: {message}")
            corrupted.append(notebook)
            
            # Try to repair
            if 'Model_Download_Manager.ipynb' in str(notebook):
                fixed = repair_notebook(str(notebook))
                if fixed:
                    print(f"  🔧 Use: {Path(fixed).name} instead")
    
    # Summary
    print("\n" + "=" * 40)
    print(f"📊 Summary:")
    print(f"  Valid notebooks: {len(valid)}")
    print(f"  Corrupted notebooks: {len(corrupted)}")
    
    if corrupted:
        print(f"\n⚠️ Corrupted notebooks found:")
        for nb in corrupted:
            print(f"  - {nb.relative_to(root)}")
            
        # Suggest fixes
        print(f"\n💡 Suggestions:")
        for nb in corrupted:
            if 'Model_Download_Manager.ipynb' in str(nb):
                print(f"  Use Model_Download_Manager_Fixed.ipynb instead")
            else:
                print(f"  Recreate or fix: {nb.name}")
                
        return 1
    else:
        print("\n✅ All notebooks are valid!")
        return 0

if __name__ == "__main__":
    sys.exit(main())