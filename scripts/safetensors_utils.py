#!/usr/bin/env python3
"""
SafeTensors Utilities
Additional utilities for SafeTensors model management in ComfyUI Paperspace environment.
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import asdict
import logging

from safetensors_loader import SafeTensorsLoader, ModelInfo, ModelType, Architecture, DType


class SafeTensorsManager:
    """High-level SafeTensors model management for ComfyUI"""
    
    def __init__(self, comfyui_models_dir: str = "/workspace/ComfyUI/models"):
        """Initialize SafeTensors manager
        
        Args:
            comfyui_models_dir: ComfyUI models directory
        """
        self.models_dir = Path(comfyui_models_dir)
        self.loader = SafeTensorsLoader()
        self.logger = logging.getLogger(__name__)
        
        # ComfyUI model subdirectories
        self.model_paths = {
            ModelType.UNET: self.models_dir / "unet",
            ModelType.VAE: self.models_dir / "vae",
            ModelType.CLIP: self.models_dir / "clip",
            ModelType.LORA: self.models_dir / "loras",
            ModelType.CONTROLNET: self.models_dir / "controlnet",
            ModelType.EMBEDDING: self.models_dir / "embeddings"
        }
        
        # Ensure directories exist
        for path in self.model_paths.values():
            path.mkdir(parents=True, exist_ok=True)

    def scan_models(self, scan_subdirs: bool = True) -> Dict[str, List[ModelInfo]]:
        """Scan for SafeTensors models in ComfyUI directory
        
        Args:
            scan_subdirs: Whether to scan subdirectories
        
        Returns:
            Dictionary of model type to list of ModelInfo
        """
        models = {model_type.value: [] for model_type in ModelType}
        
        search_paths = [self.models_dir]
        if scan_subdirs:
            search_paths.extend(self.model_paths.values())
        
        for search_path in search_paths:
            if not search_path.exists():
                continue
            
            # Find all SafeTensors files
            for safetensors_file in search_path.rglob("*.safetensors"):
                try:
                    model_info = self.loader.load_model(str(safetensors_file))
                    models[model_info.model_type.value].append(model_info)
                    self.logger.info(f"Found {model_info.model_type.value}: {safetensors_file.name}")
                
                except Exception as e:
                    self.logger.warning(f"Failed to analyze {safetensors_file}: {e}")
        
        return models

    def organize_models(self, move_files: bool = False) -> Dict[str, List[str]]:
        """Organize SafeTensors models into proper ComfyUI subdirectories
        
        Args:
            move_files: Whether to actually move files (vs just suggest)
        
        Returns:
            Dictionary of moves performed/suggested
        """
        moves = {}
        
        # Scan root models directory for misplaced files
        for safetensors_file in self.models_dir.glob("*.safetensors"):
            try:
                model_info = self.loader.load_model(str(safetensors_file))
                target_dir = self.model_paths.get(model_info.model_type)
                
                if target_dir and safetensors_file.parent != target_dir:
                    target_path = target_dir / safetensors_file.name
                    
                    move_key = f"{model_info.model_type.value}_moves"
                    if move_key not in moves:
                        moves[move_key] = []
                    
                    move_info = f"{safetensors_file} -> {target_path}"
                    moves[move_key].append(move_info)
                    
                    if move_files:
                        if not target_path.exists():
                            shutil.move(str(safetensors_file), str(target_path))
                            self.logger.info(f"Moved: {move_info}")
                        else:
                            self.logger.warning(f"Target exists, skipping: {target_path}")
            
            except Exception as e:
                self.logger.warning(f"Failed to analyze {safetensors_file}: {e}")
        
        return moves

    def check_compatibility(self, model_path: str, 
                          target_architecture: Optional[Architecture] = None) -> Dict[str, Any]:
        """Check model compatibility with ComfyUI and target architecture
        
        Args:
            model_path: Path to SafeTensors model
            target_architecture: Target architecture to check against
        
        Returns:
            Compatibility report
        """
        try:
            model_info = self.loader.load_model(model_path)
            
            report = {
                'compatible': True,
                'model_info': asdict(model_info),
                'issues': [],
                'recommendations': []
            }
            
            # Check file size (Paperspace 50GB limit consideration)
            if model_info.file_size > 20 * 1024**3:  # 20GB
                report['recommendations'].append(
                    "Large model - consider storage optimization"
                )
            
            # Check memory requirements (A6000 48GB VRAM)
            if model_info.memory_usage > 40 * 1024**3:  # 40GB
                report['issues'].append(
                    "Model may exceed A6000 VRAM capacity"
                )
                report['compatible'] = False
            
            # Check data type compatibility
            if model_info.dtype == DType.BF16:
                report['recommendations'].append(
                    "BF16 detected - ensure hardware support"
                )
            
            # Architecture-specific checks
            if target_architecture and model_info.architecture != target_architecture:
                report['issues'].append(
                    f"Architecture mismatch: {model_info.architecture.value} vs {target_architecture.value}"
                )
            
            # ComfyUI-specific checks
            comfyui_issues = self._check_comfyui_compatibility(model_info)
            report['issues'].extend(comfyui_issues)
            
            if comfyui_issues:
                report['compatible'] = False
            
            return report
        
        except Exception as e:
            return {
                'compatible': False,
                'error': str(e),
                'issues': [f"Failed to analyze model: {e}"],
                'recommendations': []
            }

    def _check_comfyui_compatibility(self, model_info: ModelInfo) -> List[str]:
        """Check ComfyUI-specific compatibility issues"""
        issues = []
        
        # Check for known problematic model types
        if model_info.model_type == ModelType.UNKNOWN:
            issues.append("Unknown model type - may not work with ComfyUI")
        
        # Check for unsupported architectures
        unsupported_archs = []  # Add any known unsupported architectures
        if model_info.architecture.value in unsupported_archs:
            issues.append(f"Architecture {model_info.architecture.value} may not be supported")
        
        # Check tensor naming conventions
        if hasattr(model_info, 'tensor_names'):
            suspicious_names = []
            for name in model_info.tensor_names:
                if name.startswith('module.'):
                    suspicious_names.append(name)
            
            if suspicious_names:
                issues.append("Model may need tensor name remapping for ComfyUI")
        
        return issues

    def optimize_for_paperspace(self, model_path: str, output_path: str, 
                               optimization_level: int = 1) -> Dict[str, Any]:
        """Optimize SafeTensors model for Paperspace environment
        
        Args:
            model_path: Input model path
            output_path: Output optimized model path
            optimization_level: 1=basic, 2=aggressive, 3=maximum
        
        Returns:
            Optimization results
        """
        model_info = self.loader.load_model(model_path)
        
        optimization_plan = {
            'original_size': model_info.file_size,
            'original_dtype': model_info.dtype.value,
            'optimizations': [],
            'estimated_savings': 0
        }
        
        target_dtype = None
        
        # Optimization level 1: FP32 -> FP16
        if optimization_level >= 1 and model_info.dtype == DType.FP32:
            target_dtype = "float16"
            optimization_plan['optimizations'].append("Convert FP32 to FP16")
            optimization_plan['estimated_savings'] += model_info.file_size * 0.5
        
        # Optimization level 2: More aggressive dtype conversion
        elif optimization_level >= 2:
            if model_info.dtype == DType.FP32:
                target_dtype = "float16"
                optimization_plan['optimizations'].append("Convert FP32 to FP16")
                optimization_plan['estimated_savings'] += model_info.file_size * 0.5
            # Could add quantization here in the future
        
        # Optimization level 3: Maximum compression
        if optimization_level >= 3:
            optimization_plan['optimizations'].append("Maximum compression (planned)")
            # Placeholder for future quantization implementations
        
        # Perform optimization
        if target_dtype:
            try:
                self.loader.convert_dtype(model_path, output_path, target_dtype)
                optimization_plan['success'] = True
                optimization_plan['output_path'] = output_path
                
                # Check actual savings
                if os.path.exists(output_path):
                    new_size = os.path.getsize(output_path)
                    optimization_plan['actual_savings'] = model_info.file_size - new_size
                    optimization_plan['compression_ratio'] = new_size / model_info.file_size
                
            except Exception as e:
                optimization_plan['success'] = False
                optimization_plan['error'] = str(e)
        else:
            optimization_plan['success'] = True
            optimization_plan['message'] = "No optimization needed or available"
        
        return optimization_plan

    def generate_model_report(self, output_file: str = "models_report.json") -> str:
        """Generate comprehensive model report for the ComfyUI installation
        
        Args:
            output_file: Output report file path
        
        Returns:
            Path to generated report
        """
        report = {
            'scan_timestamp': self._get_timestamp(),
            'environment': {
                'comfyui_models_dir': str(self.models_dir),
                'total_storage': self._get_storage_info(),
            },
            'models': {},
            'summary': {},
            'recommendations': []
        }
        
        # Scan all models
        models = self.scan_models()
        report['models'] = {
            model_type: [asdict(model) for model in model_list]
            for model_type, model_list in models.items()
        }
        
        # Generate summary
        total_models = sum(len(model_list) for model_list in models.values())
        total_size = sum(
            sum(model.file_size for model in model_list)
            for model_list in models.values()
        )
        
        report['summary'] = {
            'total_models': total_models,
            'total_size_gb': total_size / (1024**3),
            'model_breakdown': {
                model_type: len(model_list)
                for model_type, model_list in models.items()
                if model_list
            }
        }
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(models, total_size)
        
        # Save report
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Model report saved to: {output_path}")
        return str(output_path)

    def _generate_recommendations(self, models: Dict[str, List[ModelInfo]], 
                                total_size: int) -> List[str]:
        """Generate optimization recommendations"""
        recommendations = []
        
        # Storage recommendations
        if total_size > 40 * 1024**3:  # 40GB
            recommendations.append(
                "Consider optimizing models to reduce storage usage (approaching 50GB Paperspace limit)"
            )
        
        # Architecture recommendations
        arch_counts = {}
        for model_list in models.values():
            for model in model_list:
                arch_counts[model.architecture.value] = arch_counts.get(model.architecture.value, 0) + 1
        
        if arch_counts.get('unknown', 0) > 0:
            recommendations.append(
                f"Review {arch_counts['unknown']} unknown architecture models for compatibility"
            )
        
        # Data type recommendations
        fp32_count = 0
        for model_list in models.values():
            fp32_count += sum(1 for model in model_list if model.dtype == DType.FP32)
        
        if fp32_count > 0:
            recommendations.append(
                f"Consider converting {fp32_count} FP32 models to FP16 for memory efficiency"
            )
        
        # Organization recommendations
        unorganized_models = 0
        for model_type, model_list in models.items():
            if model_type == ModelType.UNKNOWN.value:
                unorganized_models += len(model_list)
        
        if unorganized_models > 0:
            recommendations.append(
                f"Organize {unorganized_models} unclassified models into proper directories"
            )
        
        return recommendations

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()

    def _get_storage_info(self) -> Dict[str, Any]:
        """Get storage information"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(self.models_dir)
            return {
                'total_gb': total / (1024**3),
                'used_gb': used / (1024**3),
                'free_gb': free / (1024**3),
                'usage_percent': (used / total) * 100
            }
        except:
            return {'error': 'Unable to get storage info'}

    def cleanup_temp_files(self) -> Dict[str, int]:
        """Clean up temporary SafeTensors files"""
        cleanup_stats = {'files_removed': 0, 'space_freed': 0}
        
        # Look for common temporary file patterns
        temp_patterns = ["*.tmp", "*.temp", "*.partial", "*~"]
        
        for pattern in temp_patterns:
            for temp_file in self.models_dir.rglob(pattern):
                if temp_file.is_file():
                    size = temp_file.stat().st_size
                    temp_file.unlink()
                    cleanup_stats['files_removed'] += 1
                    cleanup_stats['space_freed'] += size
                    self.logger.info(f"Removed temp file: {temp_file}")
        
        return cleanup_stats


def main():
    """CLI interface for SafeTensors utilities"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SafeTensors Utilities for ComfyUI")
    parser.add_argument("--models-dir", default="/workspace/ComfyUI/models",
                       help="ComfyUI models directory")
    parser.add_argument("--command", 
                       choices=["scan", "organize", "report", "optimize", "cleanup"],
                       required=True, help="Command to execute")
    parser.add_argument("--model-path", help="Path to specific model")
    parser.add_argument("--output", help="Output path")
    parser.add_argument("--optimization-level", type=int, choices=[1, 2, 3], 
                       default=1, help="Optimization level")
    parser.add_argument("--move-files", action="store_true", 
                       help="Actually move files (not just suggest)")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    manager = SafeTensorsManager(args.models_dir)
    
    try:
        if args.command == "scan":
            models = manager.scan_models()
            for model_type, model_list in models.items():
                if model_list:
                    print(f"\n{model_type.upper()} ({len(model_list)} models):")
                    for model in model_list:
                        print(f"  - {Path(model.file_path).name} "
                              f"({model.file_size / 1024**3:.2f} GB, "
                              f"{model.architecture.value})")
        
        elif args.command == "organize":
            moves = manager.organize_models(move_files=args.move_files)
            action = "Moved" if args.move_files else "Would move"
            for move_type, move_list in moves.items():
                print(f"\n{action} {move_type}:")
                for move in move_list:
                    print(f"  {move}")
        
        elif args.command == "report":
            output_file = args.output or "models_report.json"
            report_path = manager.generate_model_report(output_file)
            print(f"Report generated: {report_path}")
        
        elif args.command == "optimize":
            if not args.model_path or not args.output:
                print("--model-path and --output required for optimization")
                return 1
            
            result = manager.optimize_for_paperspace(
                args.model_path, args.output, args.optimization_level
            )
            print(f"Optimization result: {result}")
        
        elif args.command == "cleanup":
            stats = manager.cleanup_temp_files()
            print(f"Cleanup completed: {stats['files_removed']} files removed, "
                  f"{stats['space_freed'] / 1024**2:.1f} MB freed")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())