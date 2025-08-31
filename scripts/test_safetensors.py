#!/usr/bin/env python3
"""
Test script for SafeTensors loader functionality
Verifies all components work correctly in the ComfyUI Paperspace environment.
"""

import sys
import os
import tempfile
import logging
from pathlib import Path
from typing import Dict, Any, List

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_imports() -> Dict[str, bool]:
    """Test all required imports"""
    results = {}
    
    try:
        import safetensors
        results['safetensors'] = True
        logger.info(f"✅ SafeTensors library available: {safetensors.__version__}")
    except ImportError as e:
        results['safetensors'] = False
        logger.error(f"❌ SafeTensors library not available: {e}")
    
    try:
        import numpy as np
        results['numpy'] = True
        logger.info(f"✅ NumPy available: {np.__version__}")
    except ImportError as e:
        results['numpy'] = False
        logger.error(f"❌ NumPy not available: {e}")
    
    try:
        from safetensors_loader import SafeTensorsLoader, ModelInfo, ModelType, Architecture, DType
        results['safetensors_loader'] = True
        logger.info("✅ SafeTensors loader imported successfully")
    except ImportError as e:
        results['safetensors_loader'] = False
        logger.error(f"❌ SafeTensors loader import failed: {e}")
    
    try:
        from safetensors_utils import SafeTensorsManager
        results['safetensors_utils'] = True
        logger.info("✅ SafeTensors utilities imported successfully")
    except ImportError as e:
        results['safetensors_utils'] = False
        logger.error(f"❌ SafeTensors utilities import failed: {e}")
    
    try:
        from model_loader_integration import UnifiedModelLoader, UnifiedModelInfo, ModelFormat
        results['unified_loader'] = True
        logger.info("✅ Unified model loader imported successfully")
    except ImportError as e:
        results['unified_loader'] = False
        logger.error(f"❌ Unified model loader import failed: {e}")
    
    return results


def create_test_safetensors_file(file_path: str) -> bool:
    """Create a minimal test SafeTensors file for testing"""
    try:
        import numpy as np
        from safetensors.numpy import save_file
        
        # Create test tensors similar to a simple model
        tensors = {
            "model.diffusion_model.input_blocks.0.0.weight": np.random.randn(320, 4, 3, 3).astype(np.float16),
            "model.diffusion_model.input_blocks.0.0.bias": np.random.randn(320).astype(np.float16),
            "model.diffusion_model.out.0.weight": np.random.randn(4, 320, 3, 3).astype(np.float16),
            "model.diffusion_model.out.0.bias": np.random.randn(4).astype(np.float16),
        }
        
        metadata = {
            "architecture": "sd1.5",
            "resolution": "512",
            "test_model": "true"
        }
        
        save_file(tensors, file_path, metadata=metadata)
        logger.info(f"✅ Test SafeTensors file created: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to create test SafeTensors file: {e}")
        return False


def test_safetensors_loader() -> Dict[str, bool]:
    """Test SafeTensors loader functionality"""
    results = {}
    
    try:
        from safetensors_loader import SafeTensorsLoader
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp_file:
            test_file = tmp_file.name
        
        if not create_test_safetensors_file(test_file):
            return {'file_creation': False}
        
        loader = SafeTensorsLoader()
        
        # Test model loading
        try:
            model_info = loader.load_model(test_file, use_cache=False)
            results['load_model'] = True
            logger.info(f"✅ Model loaded: {model_info.model_type.value}, {model_info.architecture.value}")
            
            # Test specific components
            results['architecture_detection'] = model_info.architecture.value in ['sd1.5', 'unknown']
            results['model_type_detection'] = model_info.model_type.value in ['unet', 'unknown']
            results['dtype_detection'] = model_info.dtype.value == 'float16'
            results['tensor_count'] = model_info.tensor_count == 4
            
        except Exception as e:
            results['load_model'] = False
            logger.error(f"❌ Model loading failed: {e}")
        
        # Test tensor loading
        try:
            tensor = loader.load_tensor(test_file, "model.diffusion_model.input_blocks.0.0.weight")
            results['load_tensor'] = tensor.shape == (320, 4, 3, 3)
            logger.info(f"✅ Tensor loaded: shape {tensor.shape}")
        except Exception as e:
            results['load_tensor'] = False
            logger.error(f"❌ Tensor loading failed: {e}")
        
        # Test lazy loading
        try:
            lazy_tensors = loader.load_tensors_lazy(test_file)
            tensor_keys = lazy_tensors.keys()
            results['lazy_loading'] = len(tensor_keys) == 4
            logger.info(f"✅ Lazy loading: {len(tensor_keys)} tensors available")
        except Exception as e:
            results['lazy_loading'] = False
            logger.error(f"❌ Lazy loading failed: {e}")
        
        # Test model verification
        try:
            verification = loader.verify_model(test_file)
            results['verification'] = verification['valid']
            logger.info(f"✅ Model verification: {'valid' if verification['valid'] else 'invalid'}")
        except Exception as e:
            results['verification'] = False
            logger.error(f"❌ Model verification failed: {e}")
        
        # Test summary generation
        try:
            summary = loader.get_model_summary(test_file)
            results['summary'] = 'SafeTensors Model Summary' in summary
            logger.info("✅ Model summary generated")
        except Exception as e:
            results['summary'] = False
            logger.error(f"❌ Summary generation failed: {e}")
        
        # Cleanup
        try:
            os.unlink(test_file)
        except:
            pass
        
    except Exception as e:
        logger.error(f"❌ SafeTensors loader test setup failed: {e}")
        return {'setup_failed': True}
    
    return results


def test_safetensors_manager() -> Dict[str, bool]:
    """Test SafeTensors manager functionality"""
    results = {}
    
    try:
        from safetensors_utils import SafeTensorsManager
        
        # Use temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = SafeTensorsManager(temp_dir)
            
            # Test initialization
            results['initialization'] = True
            logger.info("✅ SafeTensors manager initialized")
            
            # Test model scanning (will be empty but shouldn't error)
            try:
                models = manager.scan_models()
                results['scan_models'] = isinstance(models, dict)
                logger.info(f"✅ Model scanning works: found {sum(len(v) for v in models.values())} models")
            except Exception as e:
                results['scan_models'] = False
                logger.error(f"❌ Model scanning failed: {e}")
            
            # Test organization (no files to move but shouldn't error)
            try:
                moves = manager.organize_models(move_files=False)
                results['organize_models'] = isinstance(moves, dict)
                logger.info("✅ Model organization check works")
            except Exception as e:
                results['organize_models'] = False
                logger.error(f"❌ Model organization failed: {e}")
            
            # Test report generation
            try:
                report_path = manager.generate_model_report()
                results['generate_report'] = os.path.exists(report_path)
                if results['generate_report']:
                    logger.info(f"✅ Model report generated: {report_path}")
                    os.unlink(report_path)  # Cleanup
                else:
                    logger.error("❌ Model report not generated")
            except Exception as e:
                results['generate_report'] = False
                logger.error(f"❌ Report generation failed: {e}")
            
            # Test cleanup
            try:
                cleanup_stats = manager.cleanup_temp_files()
                results['cleanup'] = isinstance(cleanup_stats, dict)
                logger.info("✅ Cleanup functionality works")
            except Exception as e:
                results['cleanup'] = False
                logger.error(f"❌ Cleanup failed: {e}")
        
    except Exception as e:
        logger.error(f"❌ SafeTensors manager test failed: {e}")
        return {'test_failed': True}
    
    return results


def test_unified_loader() -> Dict[str, bool]:
    """Test unified model loader functionality"""
    results = {}
    
    try:
        from model_loader_integration import UnifiedModelLoader, ModelFormat
        
        with tempfile.TemporaryDirectory() as temp_dir:
            loader = UnifiedModelLoader(temp_dir)
            
            # Test initialization
            results['initialization'] = True
            logger.info("✅ Unified loader initialized")
            
            # Test format detection with test file
            with tempfile.NamedTemporaryFile(suffix='.safetensors', delete=False) as tmp_file:
                test_file = tmp_file.name
            
            if create_test_safetensors_file(test_file):
                try:
                    detected_format = loader.detect_format(test_file)
                    results['format_detection'] = detected_format == ModelFormat.SAFETENSORS
                    logger.info(f"✅ Format detection: {detected_format.value}")
                except Exception as e:
                    results['format_detection'] = False
                    logger.error(f"❌ Format detection failed: {e}")
                
                # Test unified model loading
                try:
                    model_info = loader.load_model(test_file)
                    results['unified_loading'] = model_info.model_format == ModelFormat.SAFETENSORS
                    logger.info(f"✅ Unified loading: {model_info.model_format.value}")
                    
                    # Test optimization suggestions
                    try:
                        suggestions = loader.get_optimization_suggestions(model_info)
                        results['optimization_suggestions'] = isinstance(suggestions, dict)
                        logger.info(f"✅ Optimization suggestions: {suggestions['priority']} priority")
                    except Exception as e:
                        results['optimization_suggestions'] = False
                        logger.error(f"❌ Optimization suggestions failed: {e}")
                    
                except Exception as e:
                    results['unified_loading'] = False
                    logger.error(f"❌ Unified loading failed: {e}")
                
                # Cleanup
                try:
                    os.unlink(test_file)
                except:
                    pass
            
            # Test model scanning
            try:
                all_models = loader.scan_all_models()
                results['scan_all'] = isinstance(all_models, dict)
                logger.info("✅ Unified model scanning works")
            except Exception as e:
                results['scan_all'] = False
                logger.error(f"❌ Unified scanning failed: {e}")
        
    except Exception as e:
        logger.error(f"❌ Unified loader test failed: {e}")
        return {'test_failed': True}
    
    return results


def run_comprehensive_test() -> bool:
    """Run comprehensive test suite"""
    logger.info("🧪 Starting SafeTensors Loader Test Suite")
    logger.info("=" * 50)
    
    all_passed = True
    
    # Test imports
    logger.info("\n📦 Testing imports...")
    import_results = test_imports()
    for test_name, passed in import_results.items():
        if not passed:
            all_passed = False
    
    if not import_results.get('safetensors', False):
        logger.error("❌ SafeTensors library not available - skipping loader tests")
        return False
    
    # Test SafeTensors loader
    logger.info("\n🔧 Testing SafeTensors loader...")
    loader_results = test_safetensors_loader()
    for test_name, passed in loader_results.items():
        if not passed:
            all_passed = False
    
    # Test SafeTensors manager
    logger.info("\n📁 Testing SafeTensors manager...")
    manager_results = test_safetensors_manager()
    for test_name, passed in manager_results.items():
        if not passed:
            all_passed = False
    
    # Test unified loader
    logger.info("\n🔄 Testing unified loader...")
    unified_results = test_unified_loader()
    for test_name, passed in unified_results.items():
        if not passed:
            all_passed = False
    
    # Summary
    logger.info("\n" + "=" * 50)
    if all_passed:
        logger.info("🎉 All tests passed! SafeTensors loader is ready to use.")
    else:
        logger.error("❌ Some tests failed. Check the logs above for details.")
    
    # Component summary
    logger.info("\nComponent Status:")
    logger.info(f"  SafeTensors Library: {'✅' if import_results.get('safetensors', False) else '❌'}")
    logger.info(f"  SafeTensors Loader: {'✅' if loader_results.get('load_model', False) else '❌'}")
    logger.info(f"  SafeTensors Manager: {'✅' if manager_results.get('initialization', False) else '❌'}")
    logger.info(f"  Unified Loader: {'✅' if unified_results.get('initialization', False) else '❌'}")
    
    return all_passed


def main():
    """Main test execution"""
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
    
    except Exception as e:
        logger.error(f"❌ Test suite failed with exception: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()