#!/usr/bin/env python3
"""
Model Loader Integration
Unified interface for both GGUF and SafeTensors model loading in ComfyUI Paperspace environment.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

# Import our loaders
try:
    from safetensors_loader import SafeTensorsLoader, ModelInfo as SafeTensorsModelInfo
    from safetensors_utils import SafeTensorsManager
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

try:
    from gguf_loader import GGUFLoader, ModelInfo as GGUFModelInfo
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    # Create placeholder if GGUF loader not available
    class GGUFLoader:
        pass
    class GGUFModelInfo:
        pass


class ModelFormat(Enum):
    """Model format enumeration"""
    SAFETENSORS = "safetensors"
    GGUF = "gguf"
    UNKNOWN = "unknown"


@dataclass
class UnifiedModelInfo:
    """Unified model information across formats"""
    file_path: str
    file_size: int
    model_format: ModelFormat
    model_type: str
    architecture: str
    dtype: str
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    fingerprint: str
    tensor_count: int
    memory_usage: int
    compatibility_score: float
    recommendations: List[str]


class UnifiedModelLoader:
    """Unified model loader for both SafeTensors and GGUF formats"""
    
    def __init__(self, comfyui_models_dir: str = "/workspace/ComfyUI/models", 
                 cache_dir: Optional[str] = None):
        """Initialize unified model loader
        
        Args:
            comfyui_models_dir: ComfyUI models directory
            cache_dir: Cache directory for model metadata
        """
        self.models_dir = Path(comfyui_models_dir)
        self.logger = logging.getLogger(__name__)
        
        # Initialize format-specific loaders
        self.safetensors_loader = None
        self.gguf_loader = None
        self.safetensors_manager = None
        
        if SAFETENSORS_AVAILABLE:
            self.safetensors_loader = SafeTensorsLoader(cache_dir)
            self.safetensors_manager = SafeTensorsManager(str(comfyui_models_dir))
            self.logger.info("SafeTensors loader initialized")
        
        if GGUF_AVAILABLE:
            self.gguf_loader = GGUFLoader(cache_dir)
            self.logger.info("GGUF loader initialized")
        
        if not SAFETENSORS_AVAILABLE and not GGUF_AVAILABLE:
            self.logger.warning("No model loaders available!")

    def detect_format(self, file_path: str) -> ModelFormat:
        """Detect model format from file extension and content
        
        Args:
            file_path: Path to model file
        
        Returns:
            Detected model format
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return ModelFormat.UNKNOWN
        
        # Check by extension first
        if file_path.suffix.lower() == '.safetensors':
            return ModelFormat.SAFETENSORS
        elif file_path.suffix.lower() == '.gguf':
            return ModelFormat.GGUF
        
        # Check by magic bytes if extension is unclear
        try:
            with open(file_path, 'rb') as f:
                header = f.read(8)
                
                # GGUF magic: "GGUF"
                if header[:4] == b'GGUF':
                    return ModelFormat.GGUF
                
                # SafeTensors has JSON header length as first 8 bytes
                try:
                    header_len = int.from_bytes(header, 'little')
                    if 0 < header_len < 100000000:  # Reasonable header length
                        return ModelFormat.SAFETENSORS
                except:
                    pass
        
        except Exception as e:
            self.logger.warning(f"Failed to detect format for {file_path}: {e}")
        
        return ModelFormat.UNKNOWN

    def load_model(self, file_path: str, **kwargs) -> UnifiedModelInfo:
        """Load model with unified interface
        
        Args:
            file_path: Path to model file
            **kwargs: Additional loading options
        
        Returns:
            Unified model information
        """
        model_format = self.detect_format(file_path)
        
        if model_format == ModelFormat.SAFETENSORS and self.safetensors_loader:
            return self._load_safetensors(file_path, **kwargs)
        elif model_format == ModelFormat.GGUF and self.gguf_loader:
            return self._load_gguf(file_path, **kwargs)
        else:
            raise ValueError(f"Unsupported or unknown model format: {model_format}")

    def _load_safetensors(self, file_path: str, **kwargs) -> UnifiedModelInfo:
        """Load SafeTensors model and convert to unified format"""
        st_info = self.safetensors_loader.load_model(file_path, **kwargs)
        
        # Calculate compatibility score for Paperspace A6000
        compatibility_score = self._calculate_compatibility_score(st_info, ModelFormat.SAFETENSORS)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(st_info, ModelFormat.SAFETENSORS)
        
        return UnifiedModelInfo(
            file_path=st_info.file_path,
            file_size=st_info.file_size,
            model_format=ModelFormat.SAFETENSORS,
            model_type=st_info.model_type.value,
            architecture=st_info.architecture.value,
            dtype=st_info.dtype.value,
            parameters=st_info.parameters,
            metadata=st_info.metadata,
            fingerprint=st_info.fingerprint,
            tensor_count=st_info.tensor_count,
            memory_usage=st_info.memory_usage,
            compatibility_score=compatibility_score,
            recommendations=recommendations
        )

    def _load_gguf(self, file_path: str, **kwargs) -> UnifiedModelInfo:
        """Load GGUF model and convert to unified format"""
        # Placeholder implementation - adapt based on your GGUF loader interface
        gguf_info = self.gguf_loader.load_model(file_path, **kwargs)
        
        compatibility_score = self._calculate_compatibility_score(gguf_info, ModelFormat.GGUF)
        recommendations = self._generate_recommendations(gguf_info, ModelFormat.GGUF)
        
        return UnifiedModelInfo(
            file_path=str(file_path),
            file_size=getattr(gguf_info, 'file_size', 0),
            model_format=ModelFormat.GGUF,
            model_type=getattr(gguf_info, 'model_type', 'unknown'),
            architecture=getattr(gguf_info, 'architecture', 'unknown'),
            dtype=getattr(gguf_info, 'dtype', 'unknown'),
            parameters=getattr(gguf_info, 'parameters', {}),
            metadata=getattr(gguf_info, 'metadata', {}),
            fingerprint=getattr(gguf_info, 'fingerprint', ''),
            tensor_count=getattr(gguf_info, 'tensor_count', 0),
            memory_usage=getattr(gguf_info, 'memory_usage', 0),
            compatibility_score=compatibility_score,
            recommendations=recommendations
        )

    def _calculate_compatibility_score(self, model_info: Any, 
                                     format_type: ModelFormat) -> float:
        """Calculate compatibility score for Paperspace A6000 environment
        
        Args:
            model_info: Model information object
            format_type: Model format type
        
        Returns:
            Compatibility score (0.0 to 1.0)
        """
        score = 1.0
        
        # File size penalty (A6000 48GB VRAM, 50GB storage limit)
        file_size = getattr(model_info, 'file_size', 0)
        memory_usage = getattr(model_info, 'memory_usage', file_size)
        
        # Storage penalty
        if file_size > 40 * 1024**3:  # 40GB
            score -= 0.3
        elif file_size > 20 * 1024**3:  # 20GB
            score -= 0.1
        
        # Memory penalty
        if memory_usage > 40 * 1024**3:  # 40GB VRAM
            score -= 0.4
        elif memory_usage > 30 * 1024**3:  # 30GB VRAM
            score -= 0.2
        
        # Data type penalty
        dtype = getattr(model_info, 'dtype', '')
        if hasattr(dtype, 'value'):
            dtype = dtype.value
        
        if dtype == 'float32':
            score -= 0.1  # FP32 uses more memory
        elif dtype == 'bfloat16':
            score -= 0.05  # BF16 may need hardware support
        
        # Format-specific scoring
        if format_type == ModelFormat.GGUF:
            score += 0.1  # GGUF typically more efficient
        
        # Architecture-specific scoring
        architecture = getattr(model_info, 'architecture', '')
        if hasattr(architecture, 'value'):
            architecture = architecture.value
        
        if architecture == 'unknown':
            score -= 0.2
        elif architecture in ['sd1.5', 'sdxl']:
            score += 0.1  # Well-supported architectures
        
        return max(0.0, min(1.0, score))

    def _generate_recommendations(self, model_info: Any, 
                                format_type: ModelFormat) -> List[str]:
        """Generate recommendations for model optimization"""
        recommendations = []
        
        file_size = getattr(model_info, 'file_size', 0)
        memory_usage = getattr(model_info, 'memory_usage', file_size)
        dtype = getattr(model_info, 'dtype', '')
        
        if hasattr(dtype, 'value'):
            dtype = dtype.value
        
        # Storage recommendations
        if file_size > 30 * 1024**3:  # 30GB
            recommendations.append("Large model - monitor Paperspace 50GB storage limit")
        
        # Memory recommendations
        if memory_usage > 35 * 1024**3:  # 35GB
            recommendations.append("High memory usage - may require optimization for A6000")
        
        # Data type recommendations
        if dtype == 'float32':
            recommendations.append("Consider converting to FP16 for better memory efficiency")
        elif dtype == 'bfloat16':
            recommendations.append("Verify BF16 support on A6000 GPU")
        
        # Format-specific recommendations
        if format_type == ModelFormat.SAFETENSORS:
            if file_size > 10 * 1024**3:  # 10GB
                recommendations.append("Consider GGUF format for better compression")
        elif format_type == ModelFormat.GGUF:
            recommendations.append("GGUF format detected - good for memory efficiency")
        
        return recommendations

    def scan_all_models(self) -> Dict[str, List[UnifiedModelInfo]]:
        """Scan all supported model formats in ComfyUI directory
        
        Returns:
            Dictionary of format to list of UnifiedModelInfo
        """
        models = {
            'safetensors': [],
            'gguf': [],
            'unknown': []
        }
        
        # Search for all model files
        model_extensions = ['.safetensors', '.gguf', '.bin', '.pth', '.ckpt']
        
        for ext in model_extensions:
            for model_file in self.models_dir.rglob(f"*{ext}"):
                try:
                    model_format = self.detect_format(str(model_file))
                    
                    if model_format != ModelFormat.UNKNOWN:
                        model_info = self.load_model(str(model_file))
                        models[model_format.value].append(model_info)
                        self.logger.info(f"Scanned {model_format.value}: {model_file.name}")
                    else:
                        # Create minimal info for unknown formats
                        unknown_info = UnifiedModelInfo(
                            file_path=str(model_file),
                            file_size=model_file.stat().st_size,
                            model_format=ModelFormat.UNKNOWN,
                            model_type='unknown',
                            architecture='unknown',
                            dtype='unknown',
                            parameters={},
                            metadata={},
                            fingerprint='',
                            tensor_count=0,
                            memory_usage=0,
                            compatibility_score=0.0,
                            recommendations=["Unknown format - manual inspection needed"]
                        )
                        models['unknown'].append(unknown_info)
                
                except Exception as e:
                    self.logger.warning(f"Failed to scan {model_file}: {e}")
        
        return models

    def get_optimization_suggestions(self, model_info: UnifiedModelInfo) -> Dict[str, Any]:
        """Get optimization suggestions for a specific model
        
        Args:
            model_info: Model information
        
        Returns:
            Optimization suggestions
        """
        suggestions = {
            'priority': 'low',
            'potential_savings': 0,
            'actions': [],
            'estimated_time': '< 5 minutes'
        }
        
        # High-priority optimizations
        if model_info.compatibility_score < 0.6:
            suggestions['priority'] = 'high'
        elif model_info.compatibility_score < 0.8:
            suggestions['priority'] = 'medium'
        
        # Specific optimization actions
        if model_info.dtype == 'float32':
            suggestions['actions'].append({
                'type': 'dtype_conversion',
                'description': 'Convert FP32 to FP16',
                'potential_savings_gb': model_info.file_size / (2 * 1024**3),
                'tool': 'safetensors_loader' if model_info.model_format == ModelFormat.SAFETENSORS else 'gguf_loader'
            })
            suggestions['potential_savings'] += model_info.file_size * 0.5
        
        if model_info.file_size > 20 * 1024**3:  # 20GB
            if model_info.model_format == ModelFormat.SAFETENSORS:
                suggestions['actions'].append({
                    'type': 'format_conversion',
                    'description': 'Consider converting to GGUF for better compression',
                    'potential_savings_gb': model_info.file_size * 0.2 / (1024**3),
                    'tool': 'conversion_utility'
                })
                suggestions['potential_savings'] += model_info.file_size * 0.2
        
        if model_info.model_type == 'unknown':
            suggestions['actions'].append({
                'type': 'manual_review',
                'description': 'Manual review needed for unknown model type',
                'potential_savings_gb': 0,
                'tool': 'manual'
            })
            suggestions['estimated_time'] = '15-30 minutes'
        
        return suggestions

    def generate_comprehensive_report(self, output_file: str = "unified_models_report.json") -> str:
        """Generate comprehensive report for all models
        
        Args:
            output_file: Output report file path
        
        Returns:
            Path to generated report
        """
        self.logger.info("Generating comprehensive model report...")
        
        # Scan all models
        models = self.scan_all_models()
        
        # Calculate totals and statistics
        total_models = sum(len(model_list) for model_list in models.values())
        total_size = sum(
            sum(model.file_size for model in model_list)
            for model_list in models.values()
        )
        total_memory = sum(
            sum(model.memory_usage for model in model_list)
            for model_list in models.values()
        )
        
        # Calculate average compatibility score
        all_models = []
        for model_list in models.values():
            all_models.extend(model_list)
        
        avg_compatibility = (sum(model.compatibility_score for model in all_models) / 
                           len(all_models)) if all_models else 0
        
        # Generate optimization recommendations
        high_priority_optimizations = []
        potential_savings = 0
        
        for model in all_models:
            suggestions = self.get_optimization_suggestions(model)
            if suggestions['priority'] == 'high':
                high_priority_optimizations.append({
                    'model': Path(model.file_path).name,
                    'suggestions': suggestions
                })
            potential_savings += suggestions['potential_savings']
        
        # Create comprehensive report
        report = {
            'report_metadata': {
                'generated_at': self._get_timestamp(),
                'comfyui_models_dir': str(self.models_dir),
                'loader_capabilities': {
                    'safetensors': SAFETENSORS_AVAILABLE,
                    'gguf': GGUF_AVAILABLE
                }
            },
            'summary': {
                'total_models': total_models,
                'total_size_gb': total_size / (1024**3),
                'total_memory_gb': total_memory / (1024**3),
                'average_compatibility_score': avg_compatibility,
                'format_breakdown': {
                    fmt: len(model_list) for fmt, model_list in models.items()
                }
            },
            'models_by_format': {
                fmt: [asdict(model) for model in model_list]
                for fmt, model_list in models.items()
            },
            'optimization': {
                'high_priority_count': len(high_priority_optimizations),
                'potential_savings_gb': potential_savings / (1024**3),
                'high_priority_models': high_priority_optimizations
            },
            'environment_assessment': self._assess_environment(total_size, total_memory),
            'recommendations': self._generate_unified_recommendations(models, total_size)
        }
        
        # Save report
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Comprehensive report saved to: {output_path}")
        return str(output_path)

    def _assess_environment(self, total_size: int, total_memory: int) -> Dict[str, Any]:
        """Assess Paperspace environment capacity"""
        return {
            'storage_assessment': {
                'total_model_storage_gb': total_size / (1024**3),
                'paperspace_limit_gb': 50,
                'usage_percentage': (total_size / (50 * 1024**3)) * 100,
                'storage_status': (
                    'critical' if total_size > 45 * 1024**3 else
                    'warning' if total_size > 35 * 1024**3 else
                    'good'
                )
            },
            'memory_assessment': {
                'total_model_memory_gb': total_memory / (1024**3),
                'a6000_vram_gb': 48,
                'estimated_peak_usage_gb': total_memory / (1024**3) * 0.8,  # Conservative estimate
                'memory_status': (
                    'critical' if total_memory > 40 * 1024**3 else
                    'warning' if total_memory > 30 * 1024**3 else
                    'good'
                )
            }
        }

    def _generate_unified_recommendations(self, models: Dict[str, List[UnifiedModelInfo]], 
                                        total_size: int) -> List[str]:
        """Generate unified recommendations across all model formats"""
        recommendations = []
        
        # Storage recommendations
        if total_size > 40 * 1024**3:
            recommendations.append("URGENT: Approaching 50GB Paperspace storage limit - immediate optimization needed")
        elif total_size > 30 * 1024**3:
            recommendations.append("WARNING: High storage usage - consider optimization")
        
        # Format-specific recommendations
        safetensors_count = len(models.get('safetensors', []))
        gguf_count = len(models.get('gguf', []))
        
        if safetensors_count > gguf_count and total_size > 20 * 1024**3:
            recommendations.append("Consider converting large SafeTensors models to GGUF for better compression")
        
        # Unknown models
        unknown_count = len(models.get('unknown', []))
        if unknown_count > 0:
            recommendations.append(f"Review {unknown_count} unknown format models for compatibility")
        
        # FP32 models
        fp32_models = []
        for model_list in models.values():
            fp32_models.extend([m for m in model_list if m.dtype == 'float32'])
        
        if fp32_models:
            recommendations.append(f"Convert {len(fp32_models)} FP32 models to FP16 for memory efficiency")
        
        return recommendations

    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()


def main():
    """CLI interface for unified model loader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Unified Model Loader for ComfyUI")
    parser.add_argument("--models-dir", default="/workspace/ComfyUI/models",
                       help="ComfyUI models directory")
    parser.add_argument("--command", 
                       choices=["scan", "analyze", "report", "optimize-suggestions"],
                       required=True, help="Command to execute")
    parser.add_argument("--model-path", help="Path to specific model")
    parser.add_argument("--output", help="Output file path")
    parser.add_argument("--format", choices=["json", "text"], default="text",
                       help="Output format")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    loader = UnifiedModelLoader(args.models_dir)
    
    try:
        if args.command == "scan":
            models = loader.scan_all_models()
            for format_name, model_list in models.items():
                if model_list:
                    print(f"\n{format_name.upper()} models ({len(model_list)}):")
                    for model in model_list:
                        print(f"  - {Path(model.file_path).name} "
                              f"({model.file_size / 1024**3:.2f} GB, "
                              f"compatibility: {model.compatibility_score:.2f})")
        
        elif args.command == "analyze":
            if not args.model_path:
                print("--model-path required for analysis")
                return 1
            
            model_info = loader.load_model(args.model_path)
            
            print(f"Model Analysis: {Path(model_info.file_path).name}")
            print(f"Format: {model_info.model_format.value}")
            print(f"Type: {model_info.model_type}")
            print(f"Architecture: {model_info.architecture}")
            print(f"Size: {model_info.file_size / 1024**3:.2f} GB")
            print(f"Memory Usage: {model_info.memory_usage / 1024**3:.2f} GB")
            print(f"Compatibility Score: {model_info.compatibility_score:.2f}")
            
            if model_info.recommendations:
                print("\nRecommendations:")
                for rec in model_info.recommendations:
                    print(f"  - {rec}")
        
        elif args.command == "report":
            output_file = args.output or "unified_models_report.json"
            report_path = loader.generate_comprehensive_report(output_file)
            print(f"Comprehensive report generated: {report_path}")
        
        elif args.command == "optimize-suggestions":
            if not args.model_path:
                print("--model-path required for optimization suggestions")
                return 1
            
            model_info = loader.load_model(args.model_path)
            suggestions = loader.get_optimization_suggestions(model_info)
            
            print(f"Optimization Suggestions for {Path(model_info.file_path).name}:")
            print(f"Priority: {suggestions['priority']}")
            print(f"Potential Savings: {suggestions['potential_savings'] / 1024**3:.2f} GB")
            print(f"Estimated Time: {suggestions['estimated_time']}")
            
            if suggestions['actions']:
                print("\nActions:")
                for action in suggestions['actions']:
                    print(f"  - {action['description']} "
                          f"(saves ~{action['potential_savings_gb']:.1f} GB)")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())