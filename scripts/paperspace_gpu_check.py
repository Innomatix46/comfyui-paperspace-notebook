#!/usr/bin/env python3
"""
Paperspace GPU Detection and Diagnostic Tool
Comprehensive GPU checking and fixing for Paperspace environments
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not installed")

class PaperspaceGPUChecker:
    def __init__(self):
        self.is_paperspace = self._detect_paperspace()
        self.gpu_info = {}
        
    def _detect_paperspace(self):
        """Detect if running in Paperspace environment"""
        paperspace_indicators = [
            os.environ.get('PAPERSPACE_CLUSTER_ID'),
            os.environ.get('PAPERSPACE_METRIC_WORKLOAD_ID'),
            os.environ.get('PAPERSPACE_FQDN'),
            Path('/notebooks/.paperspace').exists(),
            Path('/storage').exists()
        ]
        return any(paperspace_indicators)
    
    def check_nvidia_smi(self):
        """Check nvidia-smi availability and output"""
        print("\n🔍 Checking nvidia-smi...")
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap', 
                 '--format=csv,noheader'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                gpu_data = result.stdout.strip()
                print(f"✅ GPU detected: {gpu_data}")
                
                # Parse GPU info
                parts = gpu_data.split(', ')
                if len(parts) >= 3:
                    self.gpu_info['name'] = parts[0]
                    self.gpu_info['memory'] = parts[1]
                    self.gpu_info['driver'] = parts[2]
                    
                    # Check for A6000
                    if 'A6000' in parts[0] or 'RTX A6000' in parts[0]:
                        print("🎯 NVIDIA RTX A6000 detected (48GB VRAM)")
                        self.gpu_info['is_a6000'] = True
                        
                return True
            else:
                print(f"❌ nvidia-smi failed: {result.stderr}")
                return False
        except FileNotFoundError:
            print("❌ nvidia-smi not found")
            return False
        except subprocess.TimeoutExpired:
            print("❌ nvidia-smi timeout")
            return False
        except Exception as e:
            print(f"❌ nvidia-smi error: {e}")
            return False
    
    def check_cuda_availability(self):
        """Check CUDA installation"""
        print("\n🔍 Checking CUDA installation...")
        
        cuda_paths = [
            '/usr/local/cuda',
            '/usr/local/cuda-12.4',
            '/usr/local/cuda-12.3',
            '/usr/local/cuda-12.2',
            '/usr/local/cuda-12.1',
            '/usr/local/cuda-12.0',
            '/usr/local/cuda-11.8'
        ]
        
        cuda_found = None
        for cuda_path in cuda_paths:
            if Path(cuda_path).exists():
                cuda_found = cuda_path
                break
        
        if cuda_found:
            print(f"✅ CUDA found at: {cuda_found}")
            
            # Check nvcc
            nvcc_path = Path(cuda_found) / 'bin' / 'nvcc'
            if nvcc_path.exists():
                try:
                    result = subprocess.run(
                        [str(nvcc_path), '--version'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0:
                        # Extract version
                        for line in result.stdout.split('\n'):
                            if 'release' in line:
                                print(f"   CUDA Version: {line.strip()}")
                                break
                except:
                    pass
            
            return cuda_found
        else:
            print("❌ CUDA not found")
            return None
    
    def check_pytorch_cuda(self):
        """Check PyTorch CUDA support"""
        print("\n🔍 Checking PyTorch CUDA support...")
        
        if not TORCH_AVAILABLE:
            print("❌ PyTorch not available")
            return False
        
        print(f"PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✅ CUDA available in PyTorch")
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   cuDNN version: {torch.backends.cudnn.version()}")
            print(f"   Number of GPUs: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"\n   GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"   - Compute Capability: {props.major}.{props.minor}")
                print(f"   - Memory: {props.total_memory / 1024**3:.1f} GB")
                print(f"   - Multi-processors: {props.multi_processor_count}")
                
            return True
        else:
            print("❌ CUDA not available in PyTorch")
            
            # Check why CUDA is not available
            if hasattr(torch._C, '_cuda_getDeviceCount'):
                try:
                    torch._C._cuda_getDeviceCount()
                except RuntimeError as e:
                    print(f"   Error: {e}")
            
            return False
    
    def check_environment_variables(self):
        """Check and set environment variables"""
        print("\n🔍 Checking environment variables...")
        
        important_vars = {
            'CUDA_HOME': '/usr/local/cuda',
            'CUDA_PATH': '/usr/local/cuda',
            'LD_LIBRARY_PATH': None,
            'PATH': None,
            'CUDA_VISIBLE_DEVICES': None,
            'PYTORCH_CUDA_ALLOC_CONF': None
        }
        
        for var, expected in important_vars.items():
            current = os.environ.get(var, 'Not set')
            if current == 'Not set':
                print(f"⚠️ {var}: Not set")
                if expected:
                    os.environ[var] = expected
                    print(f"   → Set to: {expected}")
            else:
                print(f"✅ {var}: {current[:100]}...")  # Truncate long paths
    
    def wait_for_gpu(self, max_wait=60):
        """Wait for GPU to become available (Paperspace specific)"""
        if not self.is_paperspace:
            return False
        
        print(f"\n⏳ Waiting for GPU (max {max_wait} seconds)...")
        print("   (Paperspace sometimes takes time to allocate GPU)")
        
        start_time = time.time()
        check_interval = 2
        
        while time.time() - start_time < max_wait:
            # Check with nvidia-smi
            if self.check_nvidia_smi():
                print(f"✅ GPU available after {time.time() - start_time:.1f} seconds")
                return True
            
            # Check with PyTorch
            if TORCH_AVAILABLE:
                try:
                    if torch.cuda.is_available():
                        print(f"✅ GPU available in PyTorch after {time.time() - start_time:.1f} seconds")
                        return True
                except:
                    pass
            
            time.sleep(check_interval)
            
            # Show progress
            elapsed = int(time.time() - start_time)
            if elapsed % 10 == 0:
                print(f"   Still waiting... ({elapsed}/{max_wait}s)")
        
        print(f"❌ GPU not available after {max_wait} seconds")
        return False
    
    def apply_fixes(self):
        """Apply common fixes for GPU issues"""
        print("\n🔧 Applying fixes...")
        
        fixes_applied = []
        
        # 1. Set CUDA paths
        cuda_path = self.check_cuda_availability()
        if cuda_path:
            os.environ['CUDA_HOME'] = cuda_path
            os.environ['CUDA_PATH'] = cuda_path
            
            # Update PATH
            cuda_bin = f"{cuda_path}/bin"
            if cuda_bin not in os.environ.get('PATH', ''):
                os.environ['PATH'] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
                fixes_applied.append("Added CUDA to PATH")
            
            # Update LD_LIBRARY_PATH
            cuda_lib = f"{cuda_path}/lib64"
            if cuda_lib not in os.environ.get('LD_LIBRARY_PATH', ''):
                os.environ['LD_LIBRARY_PATH'] = f"{cuda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
                fixes_applied.append("Added CUDA libs to LD_LIBRARY_PATH")
        
        # 2. Clear CUDA cache if PyTorch available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                fixes_applied.append("Cleared CUDA cache")
            except:
                pass
        
        # 3. Set optimal settings for A6000
        if self.gpu_info.get('is_a6000'):
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
            fixes_applied.append("Applied A6000 optimizations")
        
        if fixes_applied:
            print("✅ Fixes applied:")
            for fix in fixes_applied:
                print(f"   - {fix}")
        else:
            print("ℹ️ No fixes needed")
        
        return fixes_applied
    
    def generate_report(self):
        """Generate comprehensive diagnostic report"""
        print("\n" + "="*50)
        print("📊 GPU DIAGNOSTIC REPORT")
        print("="*50)
        
        # Summary
        gpu_available = bool(self.gpu_info)
        pytorch_cuda = TORCH_AVAILABLE and torch.cuda.is_available()
        
        print("\n📋 Summary:")
        print(f"   Environment: {'Paperspace' if self.is_paperspace else 'Local/Other'}")
        print(f"   GPU Detected: {'✅ Yes' if gpu_available else '❌ No'}")
        print(f"   PyTorch CUDA: {'✅ Yes' if pytorch_cuda else '❌ No'}")
        
        if self.gpu_info:
            print(f"\n🎮 GPU Information:")
            for key, value in self.gpu_info.items():
                if key != 'is_a6000':
                    print(f"   {key}: {value}")
        
        # Recommendations
        print("\n💡 Recommendations:")
        
        if not gpu_available and self.is_paperspace:
            print("   1. Wait a few minutes for GPU allocation")
            print("   2. Try restarting the Paperspace notebook")
            print("   3. Check your Paperspace subscription (Free tier may have limited GPU)")
            print("   4. Try at different times (less busy hours)")
        elif not pytorch_cuda and gpu_available:
            print("   1. Reinstall PyTorch with CUDA support:")
            print("      pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            print("   2. Check CUDA/PyTorch version compatibility")
        elif not gpu_available and not self.is_paperspace:
            print("   1. Check if GPU is properly installed")
            print("   2. Update NVIDIA drivers")
            print("   3. Verify CUDA installation")
        else:
            print("   ✅ Everything looks good!")
        
        # ComfyUI specific
        print("\n🎨 ComfyUI:")
        if pytorch_cuda:
            print("   ✅ Ready to run with GPU acceleration")
            print("   Command: cd ComfyUI && python main.py --listen 0.0.0.0")
        else:
            print("   ⚠️ Will run in CPU mode (slow)")
            print("   Command: cd ComfyUI && python main.py --cpu --listen 0.0.0.0")
        
        print("\n" + "="*50)

def main():
    """Main execution"""
    print("🚀 Paperspace GPU Detection & Diagnostic Tool")
    print("="*50)
    
    checker = PaperspaceGPUChecker()
    
    # Run checks
    has_nvidia = checker.check_nvidia_smi()
    cuda_path = checker.check_cuda_availability()
    has_pytorch_cuda = checker.check_pytorch_cuda()
    checker.check_environment_variables()
    
    # If no GPU and in Paperspace, wait
    if not has_nvidia and checker.is_paperspace:
        if checker.wait_for_gpu():
            # Re-run checks after GPU available
            has_nvidia = checker.check_nvidia_smi()
            has_pytorch_cuda = checker.check_pytorch_cuda()
    
    # Apply fixes if needed
    if not has_pytorch_cuda:
        checker.apply_fixes()
        
        # Re-check PyTorch CUDA after fixes
        if TORCH_AVAILABLE:
            has_pytorch_cuda = torch.cuda.is_available()
    
    # Generate report
    checker.generate_report()
    
    # Return status
    return 0 if has_pytorch_cuda else 1

if __name__ == "__main__":
    sys.exit(main())