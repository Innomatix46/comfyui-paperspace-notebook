#!/usr/bin/env python3
"""
Quick fix for ComfyUI dependency issues
Focuses on the critical errors preventing startup
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def fix_dependencies():
    """Fix the critical dependency issues"""
    
    print("🔧 ComfyUI Dependency Quick Fix")
    print("=" * 40)
    
    fixes = [
        # Fix NumPy (downgrade to 1.x)
        {
            "name": "NumPy",
            "commands": [
                "pip uninstall -y numpy",
                "pip install --no-cache-dir 'numpy<2.0' 'numpy==1.26.4'"
            ],
            "reason": "NumPy 2.x incompatible with compiled modules"
        },
        
        # Fix transformers
        {
            "name": "Transformers",
            "commands": [
                "pip uninstall -y transformers",
                "pip install --no-cache-dir 'transformers==4.36.2'"
            ],
            "reason": "AttributeError: 'register_pytree_node'"
        },
        
        # Fix aiofiles
        {
            "name": "aiofiles",
            "commands": [
                "pip uninstall -y aiofiles",
                "pip install --no-cache-dir 'aiofiles>=22.1.0,<23'"
            ],
            "reason": "ypy-websocket requires aiofiles<23"
        },
        
        # Reinstall torch dependencies
        {
            "name": "Torch dependencies",
            "commands": [
                "pip install --no-cache-dir --force-reinstall torchvision torchaudio"
            ],
            "reason": "Ensure compatibility with NumPy 1.x"
        }
    ]
    
    # Apply fixes
    for fix in fixes:
        print(f"\n📦 Fixing {fix['name']}...")
        print(f"   Reason: {fix['reason']}")
        
        for cmd in fix['commands']:
            print(f"   Running: {cmd}")
            success, stdout, stderr = run_command(cmd)
            
            if not success:
                print(f"   ⚠️ Warning: Command may have failed")
                if stderr:
                    print(f"   Error: {stderr[:200]}")
        
        print(f"   ✅ {fix['name']} processed")
    
    print("\n" + "=" * 40)
    print("🔍 Verifying fixes...")
    
    # Verify imports work
    verification_script = """
import sys
try:
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
    assert not np.__version__.startswith('2.'), "NumPy 2.x still installed"
    
    import torch
    print(f"✅ PyTorch {torch.__version__}")
    
    import torchvision
    print(f"✅ TorchVision {torchvision.__version__}")
    
    import transformers
    print(f"✅ Transformers {transformers.__version__}")
    
    # Test the specific import that was failing
    from torch.onnx._internal.fx import ONNXTorchPatcher
    print("✅ PyTorch ONNX imports work")
    
    import aiofiles
    print(f"✅ aiofiles {aiofiles.__version__}")
    
    print("\\n✅ All critical imports successful!")
    sys.exit(0)
    
except Exception as e:
    print(f"\\n❌ Error: {e}")
    sys.exit(1)
"""
    
    # Run verification
    result = subprocess.run([sys.executable, "-c", verification_script], capture_output=True, text=True)
    print(result.stdout)
    
    if result.returncode != 0:
        print("\n⚠️ Some issues remain. Trying additional fixes...")
        
        # Additional fixes
        additional_fixes = [
            "pip install --upgrade --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
            "pip install --no-cache-dir 'scipy<1.14'",
            "pip install --no-cache-dir GitPython"
        ]
        
        for cmd in additional_fixes:
            print(f"Running: {cmd}")
            run_command(cmd)
    
    print("\n" + "=" * 40)
    print("✅ Dependency fixes applied!")
    print("=" * 40)
    print("\nTo start ComfyUI:")
    print("cd /notebooks/ComfyUI && python main.py --listen --port 6006")

if __name__ == "__main__":
    fix_dependencies()