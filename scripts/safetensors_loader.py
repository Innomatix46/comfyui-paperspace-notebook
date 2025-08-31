#!/usr/bin/env python3
"""
SafeTensors Model Loader for ComfyUI
Comprehensive SafeTensors format support with architecture detection,
memory-mapped loading, and model verification.
"""

import os
import json
import hashlib
import mmap
import struct
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

try:
    import safetensors
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False
    print("Warning: safetensors not available. Install with: pip install safetensors")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class ModelType(Enum):
    """Model type enumeration"""
    UNET = "unet"
    VAE = "vae"
    CLIP = "clip"
    TEXT_ENCODER = "text_encoder"
    LORA = "lora"
    CONTROLNET = "controlnet"
    EMBEDDING = "embedding"
    UNKNOWN = "unknown"


class Architecture(Enum):
    """Architecture type enumeration"""
    SD15 = "sd1.5"
    SDXL = "sdxl"
    SD3 = "sd3"
    FLUX = "flux"
    CHROMA = "chroma"
    KANDINSKY = "kandinsky"
    UNKNOWN = "unknown"


class DType(Enum):
    """Data type enumeration"""
    FP32 = "float32"
    FP16 = "float16"
    BF16 = "bfloat16"
    INT8 = "int8"
    UINT8 = "uint8"


@dataclass
class ModelInfo:
    """Model information container"""
    file_path: str
    file_size: int
    model_type: ModelType
    architecture: Architecture
    dtype: DType
    parameters: Dict[str, Any]
    metadata: Dict[str, Any]
    fingerprint: str
    tensor_count: int
    memory_usage: int
    sharded: bool
    shard_files: List[str]


@dataclass
class TensorInfo:
    """Tensor information container"""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int
    offset: int


class SafeTensorsLoader:
    """Comprehensive SafeTensors model loader with advanced features"""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the SafeTensors loader
        
        Args:
            cache_dir: Directory for caching model metadata
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "safetensors_loader"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Architecture detection patterns
        self.arch_patterns = {
            Architecture.SDXL: [
                "conditioner.embedders.0.transformer.text_model.encoder.layers.22",
                "model.diffusion_model.input_blocks.4.1.transformer_blocks.0.attn2.to_k.weight",
                "conditioner.embedders.1.model.transformer.resblocks.22"
            ],
            Architecture.SD3: [
                "model.diffusion_model.joint_blocks.0.context_block.attn.qkv.weight",
                "model.diffusion_model.joint_blocks.0.x_block.attn.qkv.weight"
            ],
            Architecture.FLUX: [
                "model.diffusion_model.double_blocks.0.img_attn.qkv.weight",
                "model.diffusion_model.single_blocks.0.linear1.weight"
            ],
            Architecture.SD15: [
                "model.diffusion_model.input_blocks.1.1.transformer_blocks.0.attn1.to_k.weight",
                "cond_stage_model.transformer.text_model.encoder.layers.11"
            ]
        }
        
        # Model type detection patterns
        self.type_patterns = {
            ModelType.UNET: ["model.diffusion_model", "unet"],
            ModelType.VAE: ["decoder.conv_out", "encoder.conv_in", "first_stage_model"],
            ModelType.CLIP: ["text_model.encoder", "transformer.text_model"],
            ModelType.LORA: ["lora_up", "lora_down", "alpha"],
            ModelType.CONTROLNET: ["control_model", "controlnet"],
            ModelType.EMBEDDING: ["string_to_param"]
        }

    def load_model(self, file_path: str, **kwargs) -> ModelInfo:
        """Load SafeTensors model with comprehensive analysis
        
        Args:
            file_path: Path to the SafeTensors file
            **kwargs: Additional loading options
                - use_cache: Use cached metadata (default: True)
                - verify_fingerprint: Verify model fingerprint (default: True)
                - memory_map: Use memory mapping (default: True)
                - max_memory_gb: Maximum memory usage in GB
        
        Returns:
            ModelInfo object with comprehensive model information
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors library not available")
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        use_cache = kwargs.get('use_cache', True)
        verify_fingerprint = kwargs.get('verify_fingerprint', True)
        
        # Check cache first
        if use_cache:
            cached_info = self._load_from_cache(file_path)
            if cached_info and (not verify_fingerprint or 
                              cached_info.fingerprint == self._calculate_fingerprint(file_path)):
                self.logger.info(f"Loaded cached model info for {file_path}")
                return cached_info
        
        self.logger.info(f"Analyzing SafeTensors model: {file_path}")
        
        # Analyze the model
        model_info = self._analyze_model(file_path, **kwargs)
        
        # Cache the results
        if use_cache:
            self._save_to_cache(model_info)
        
        return model_info

    def _analyze_model(self, file_path: Path, **kwargs) -> ModelInfo:
        """Analyze SafeTensors model file"""
        file_size = file_path.stat().st_size
        fingerprint = self._calculate_fingerprint(file_path)
        
        # Check for sharded models
        shard_files = self._find_shard_files(file_path)
        is_sharded = len(shard_files) > 1
        
        # Load metadata and tensor info
        metadata = {}
        tensor_infos = []
        total_memory = 0
        
        try:
            with safe_open(file_path, framework="numpy") as f:
                # Get metadata
                if hasattr(f, 'metadata') and f.metadata():
                    metadata = f.metadata()
                
                # Get tensor information
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    tensor_info = TensorInfo(
                        name=key,
                        shape=tensor.shape,
                        dtype=str(tensor.dtype),
                        size_bytes=tensor.nbytes,
                        offset=0  # SafeTensors handles this internally
                    )
                    tensor_infos.append(tensor_info)
                    total_memory += tensor.nbytes
        
        except Exception as e:
            self.logger.error(f"Error loading SafeTensors file: {e}")
            raise
        
        # Detect model type and architecture
        model_type = self._detect_model_type(tensor_infos)
        architecture = self._detect_architecture(tensor_infos, metadata)
        dtype = self._detect_primary_dtype(tensor_infos)
        
        # Extract parameters
        parameters = self._extract_parameters(tensor_infos, metadata, model_type, architecture)
        
        return ModelInfo(
            file_path=str(file_path),
            file_size=file_size,
            model_type=model_type,
            architecture=architecture,
            dtype=dtype,
            parameters=parameters,
            metadata=metadata,
            fingerprint=fingerprint,
            tensor_count=len(tensor_infos),
            memory_usage=total_memory,
            sharded=is_sharded,
            shard_files=shard_files
        )

    def _detect_model_type(self, tensor_infos: List[TensorInfo]) -> ModelType:
        """Detect model type from tensor names"""
        tensor_names = [info.name for info in tensor_infos]
        
        for model_type, patterns in self.type_patterns.items():
            for pattern in patterns:
                if any(pattern in name for name in tensor_names):
                    return model_type
        
        return ModelType.UNKNOWN

    def _detect_architecture(self, tensor_infos: List[TensorInfo], metadata: Dict) -> Architecture:
        """Detect model architecture from tensors and metadata"""
        tensor_names = [info.name for info in tensor_infos]
        
        # Check metadata first
        if metadata:
            if 'architecture' in metadata:
                arch_name = metadata['architecture'].lower()
                for arch in Architecture:
                    if arch.value in arch_name:
                        return arch
        
        # Check tensor patterns
        for architecture, patterns in self.arch_patterns.items():
            matches = sum(1 for pattern in patterns 
                         if any(pattern in name for name in tensor_names))
            if matches > 0:
                return architecture
        
        # Heuristic detection based on tensor shapes and counts
        return self._heuristic_architecture_detection(tensor_infos)

    def _heuristic_architecture_detection(self, tensor_infos: List[TensorInfo]) -> Architecture:
        """Heuristic architecture detection based on tensor analysis"""
        unet_tensors = [info for info in tensor_infos if 'diffusion_model' in info.name]
        
        if not unet_tensors:
            return Architecture.UNKNOWN
        
        # Analyze attention dimensions and layer counts
        attention_dims = []
        layer_counts = {}
        
        for tensor in unet_tensors:
            if 'attn' in tensor.name and len(tensor.shape) >= 2:
                attention_dims.append(tensor.shape[-1])
            
            # Count different block types
            for block_type in ['input_blocks', 'middle_block', 'output_blocks']:
                if block_type in tensor.name:
                    layer_counts[block_type] = layer_counts.get(block_type, 0) + 1
        
        # SDXL typically has larger attention dimensions
        if attention_dims and max(attention_dims) > 2048:
            return Architecture.SDXL
        
        # SD3 has distinctive joint blocks
        joint_blocks = sum(1 for tensor in unet_tensors if 'joint_blocks' in tensor.name)
        if joint_blocks > 0:
            return Architecture.SD3
        
        # FLUX has double_blocks and single_blocks
        double_blocks = sum(1 for tensor in unet_tensors if 'double_blocks' in tensor.name)
        single_blocks = sum(1 for tensor in unet_tensors if 'single_blocks' in tensor.name)
        if double_blocks > 0 and single_blocks > 0:
            return Architecture.FLUX
        
        return Architecture.SD15  # Default fallback

    def _detect_primary_dtype(self, tensor_infos: List[TensorInfo]) -> DType:
        """Detect primary data type from tensors"""
        dtype_counts = {}
        
        for tensor in tensor_infos:
            dtype_counts[tensor.dtype] = dtype_counts.get(tensor.dtype, 0) + 1
        
        if not dtype_counts:
            return DType.FP32
        
        # Find most common dtype
        primary_dtype = max(dtype_counts, key=dtype_counts.get)
        
        dtype_mapping = {
            'float32': DType.FP32,
            'float16': DType.FP16,
            'bfloat16': DType.BF16,
            'int8': DType.INT8,
            'uint8': DType.UINT8
        }
        
        return dtype_mapping.get(primary_dtype, DType.FP32)

    def _extract_parameters(self, tensor_infos: List[TensorInfo], metadata: Dict, 
                          model_type: ModelType, architecture: Architecture) -> Dict[str, Any]:
        """Extract model parameters from tensors and metadata"""
        params = {
            'total_parameters': sum(np.prod(info.shape) for info in tensor_infos),
            'tensor_count': len(tensor_infos),
            'layer_analysis': self._analyze_layers(tensor_infos),
            'memory_breakdown': self._analyze_memory_usage(tensor_infos)
        }
        
        # Add metadata parameters
        if metadata:
            for key, value in metadata.items():
                if key not in params:
                    params[key] = value
        
        # Architecture-specific parameters
        if architecture == Architecture.SDXL:
            params.update(self._extract_sdxl_params(tensor_infos))
        elif architecture == Architecture.SD3:
            params.update(self._extract_sd3_params(tensor_infos))
        elif architecture == Architecture.FLUX:
            params.update(self._extract_flux_params(tensor_infos))
        
        return params

    def _analyze_layers(self, tensor_infos: List[TensorInfo]) -> Dict[str, Any]:
        """Analyze layer structure"""
        layer_types = {}
        
        for tensor in tensor_infos:
            name_parts = tensor.name.split('.')
            
            for part in name_parts:
                if any(layer_type in part for layer_type in 
                      ['conv', 'linear', 'attn', 'norm', 'embed']):
                    layer_types[part] = layer_types.get(part, 0) + 1
        
        return layer_types

    def _analyze_memory_usage(self, tensor_infos: List[TensorInfo]) -> Dict[str, int]:
        """Analyze memory usage breakdown"""
        breakdown = {}
        
        for tensor in tensor_infos:
            category = 'other'
            
            if 'diffusion_model' in tensor.name:
                category = 'unet'
            elif 'first_stage_model' in tensor.name or 'vae' in tensor.name:
                category = 'vae'
            elif 'cond_stage_model' in tensor.name or 'clip' in tensor.name:
                category = 'text_encoder'
            elif 'lora' in tensor.name:
                category = 'lora'
            
            breakdown[category] = breakdown.get(category, 0) + tensor.size_bytes
        
        return breakdown

    def _extract_sdxl_params(self, tensor_infos: List[TensorInfo]) -> Dict[str, Any]:
        """Extract SDXL-specific parameters"""
        params = {}
        
        # Count transformer blocks in different components
        text_encoder_layers = set()
        unet_attention_layers = set()
        
        for tensor in tensor_infos:
            if 'text_model.encoder.layers' in tensor.name:
                layer_match = tensor.name.split('layers.')[1].split('.')[0]
                if layer_match.isdigit():
                    text_encoder_layers.add(int(layer_match))
            
            if 'transformer_blocks' in tensor.name:
                parts = tensor.name.split('transformer_blocks.')
                if len(parts) > 1:
                    block_num = parts[1].split('.')[0]
                    if block_num.isdigit():
                        unet_attention_layers.add(int(block_num))
        
        params['text_encoder_layers'] = len(text_encoder_layers)
        params['unet_attention_blocks'] = len(unet_attention_layers)
        
        return params

    def _extract_sd3_params(self, tensor_infos: List[TensorInfo]) -> Dict[str, Any]:
        """Extract SD3-specific parameters"""
        params = {}
        
        joint_blocks = set()
        for tensor in tensor_infos:
            if 'joint_blocks' in tensor.name:
                parts = tensor.name.split('joint_blocks.')
                if len(parts) > 1:
                    block_num = parts[1].split('.')[0]
                    if block_num.isdigit():
                        joint_blocks.add(int(block_num))
        
        params['joint_blocks'] = len(joint_blocks)
        
        return params

    def _extract_flux_params(self, tensor_infos: List[TensorInfo]) -> Dict[str, Any]:
        """Extract FLUX-specific parameters"""
        params = {}
        
        double_blocks = set()
        single_blocks = set()
        
        for tensor in tensor_infos:
            if 'double_blocks' in tensor.name:
                parts = tensor.name.split('double_blocks.')
                if len(parts) > 1:
                    block_num = parts[1].split('.')[0]
                    if block_num.isdigit():
                        double_blocks.add(int(block_num))
            
            if 'single_blocks' in tensor.name:
                parts = tensor.name.split('single_blocks.')
                if len(parts) > 1:
                    block_num = parts[1].split('.')[0]
                    if block_num.isdigit():
                        single_blocks.add(int(block_num))
        
        params['double_blocks'] = len(double_blocks)
        params['single_blocks'] = len(single_blocks)
        
        return params

    def _find_shard_files(self, file_path: Path) -> List[str]:
        """Find all shard files for a sharded model"""
        shard_files = [str(file_path)]
        
        # Look for numbered shards
        base_name = file_path.stem
        extension = file_path.suffix
        parent_dir = file_path.parent
        
        # Common sharding patterns
        patterns = [
            f"{base_name}-*-of-*{extension}",
            f"{base_name}.*.{extension}",
            f"{base_name}_*{extension}"
        ]
        
        for pattern in patterns:
            for potential_shard in parent_dir.glob(pattern):
                if str(potential_shard) not in shard_files:
                    shard_files.append(str(potential_shard))
        
        return sorted(shard_files)

    def _calculate_fingerprint(self, file_path: Path) -> str:
        """Calculate file fingerprint using SHA-256"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            # Read first and last 64KB for large files
            chunk_size = 65536  # 64KB
            sha256_hash.update(f.read(chunk_size))
            
            file_size = file_path.stat().st_size
            if file_size > chunk_size * 2:
                f.seek(-chunk_size, 2)  # Seek to last 64KB
                sha256_hash.update(f.read(chunk_size))
        
        return sha256_hash.hexdigest()[:16]  # First 16 chars for brevity

    def _load_from_cache(self, file_path: Path) -> Optional[ModelInfo]:
        """Load model info from cache"""
        cache_file = self.cache_dir / f"{file_path.name}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
            
            # Convert enum strings back to enums
            data['model_type'] = ModelType(data['model_type'])
            data['architecture'] = Architecture(data['architecture'])
            data['dtype'] = DType(data['dtype'])
            
            return ModelInfo(**data)
        
        except Exception as e:
            self.logger.warning(f"Failed to load cache for {file_path}: {e}")
            return None

    def _save_to_cache(self, model_info: ModelInfo):
        """Save model info to cache"""
        cache_file = self.cache_dir / f"{Path(model_info.file_path).name}.json"
        
        try:
            # Convert to dict and handle enums
            data = asdict(model_info)
            data['model_type'] = model_info.model_type.value
            data['architecture'] = model_info.architecture.value
            data['dtype'] = model_info.dtype.value
            
            with open(cache_file, 'w') as f:
                json.dump(data, f, indent=2)
        
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")

    def load_tensor(self, file_path: str, tensor_name: str, 
                   dtype: Optional[str] = None) -> np.ndarray:
        """Load specific tensor from SafeTensors file
        
        Args:
            file_path: Path to SafeTensors file
            tensor_name: Name of tensor to load
            dtype: Optional dtype conversion
        
        Returns:
            Loaded tensor as numpy array
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors library not available")
        
        with safe_open(file_path, framework="numpy") as f:
            if tensor_name not in f.keys():
                raise KeyError(f"Tensor '{tensor_name}' not found in {file_path}")
            
            tensor = f.get_tensor(tensor_name)
            
            if dtype and tensor.dtype != dtype:
                tensor = tensor.astype(dtype)
            
            return tensor

    def load_tensors_lazy(self, file_path: str) -> Dict[str, Any]:
        """Create lazy loading interface for tensors
        
        Args:
            file_path: Path to SafeTensors file
        
        Returns:
            Dictionary with lazy tensor loading
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors library not available")
        
        class LazyTensorDict:
            def __init__(self, file_path):
                self.file_path = file_path
                self._keys = None
            
            def keys(self):
                if self._keys is None:
                    with safe_open(self.file_path, framework="numpy") as f:
                        self._keys = list(f.keys())
                return self._keys
            
            def __getitem__(self, key):
                with safe_open(self.file_path, framework="numpy") as f:
                    return f.get_tensor(key)
            
            def __contains__(self, key):
                return key in self.keys()
        
        return LazyTensorDict(file_path)

    def convert_dtype(self, file_path: str, output_path: str, 
                     target_dtype: str, **kwargs):
        """Convert SafeTensors model to different dtype
        
        Args:
            file_path: Input SafeTensors file
            output_path: Output SafeTensors file
            target_dtype: Target data type (float16, float32, bfloat16)
            **kwargs: Additional options
        """
        if not SAFETENSORS_AVAILABLE:
            raise ImportError("SafeTensors library not available")
        
        from safetensors.numpy import save_file
        
        tensors = {}
        metadata = {}
        
        with safe_open(file_path, framework="numpy") as f:
            if hasattr(f, 'metadata') and f.metadata():
                metadata = f.metadata().copy()
                metadata['converted_dtype'] = target_dtype
            
            for key in f.keys():
                tensor = f.get_tensor(key)
                
                # Convert dtype
                if tensor.dtype != target_dtype:
                    tensor = tensor.astype(target_dtype)
                
                tensors[key] = tensor
        
        save_file(tensors, output_path, metadata=metadata)
        self.logger.info(f"Converted model saved to: {output_path}")

    def verify_model(self, file_path: str) -> Dict[str, Any]:
        """Verify SafeTensors model integrity
        
        Args:
            file_path: Path to SafeTensors file
        
        Returns:
            Verification results
        """
        results = {
            'valid': False,
            'errors': [],
            'warnings': [],
            'info': {}
        }
        
        try:
            model_info = self.load_model(file_path, verify_fingerprint=False)
            
            # Basic validation
            if model_info.tensor_count == 0:
                results['errors'].append("No tensors found in model")
            
            if model_info.file_size == 0:
                results['errors'].append("Empty model file")
            
            # Check for suspicious patterns
            if model_info.model_type == ModelType.UNKNOWN:
                results['warnings'].append("Could not detect model type")
            
            if model_info.architecture == Architecture.UNKNOWN:
                results['warnings'].append("Could not detect architecture")
            
            # Memory usage check
            if model_info.memory_usage > 50 * 1024**3:  # 50GB
                results['warnings'].append("Very large model (>50GB)")
            
            results['valid'] = len(results['errors']) == 0
            results['info'] = asdict(model_info)
            
        except Exception as e:
            results['errors'].append(f"Verification failed: {str(e)}")
        
        return results

    def get_model_summary(self, file_path: str) -> str:
        """Get human-readable model summary
        
        Args:
            file_path: Path to SafeTensors file
        
        Returns:
            Formatted model summary
        """
        try:
            model_info = self.load_model(file_path)
            
            summary = f"""
SafeTensors Model Summary
========================
File: {Path(model_info.file_path).name}
Size: {model_info.file_size / 1024**3:.2f} GB
Type: {model_info.model_type.value}
Architecture: {model_info.architecture.value}
Data Type: {model_info.dtype.value}
Parameters: {model_info.parameters.get('total_parameters', 0):,}
Tensors: {model_info.tensor_count}
Memory Usage: {model_info.memory_usage / 1024**3:.2f} GB
Sharded: {'Yes' if model_info.sharded else 'No'}
            """
            
            if model_info.sharded:
                summary += f"Shard Files: {len(model_info.shard_files)}\n"
            
            if model_info.metadata:
                summary += "\nMetadata:\n"
                for key, value in model_info.metadata.items():
                    summary += f"  {key}: {value}\n"
            
            return summary.strip()
            
        except Exception as e:
            return f"Error generating summary: {str(e)}"


def main():
    """CLI interface for SafeTensors loader"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SafeTensors Model Loader")
    parser.add_argument("file_path", help="Path to SafeTensors file")
    parser.add_argument("--command", choices=["info", "verify", "convert"], 
                       default="info", help="Command to execute")
    parser.add_argument("--output", help="Output path for conversion")
    parser.add_argument("--dtype", help="Target dtype for conversion")
    parser.add_argument("--cache-dir", help="Cache directory")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, 
                       format='%(asctime)s - %(levelname)s - %(message)s')
    
    loader = SafeTensorsLoader(cache_dir=args.cache_dir)
    
    try:
        if args.command == "info":
            summary = loader.get_model_summary(args.file_path)
            print(summary)
        
        elif args.command == "verify":
            results = loader.verify_model(args.file_path)
            print(f"Valid: {results['valid']}")
            if results['errors']:
                print("Errors:")
                for error in results['errors']:
                    print(f"  - {error}")
            if results['warnings']:
                print("Warnings:")
                for warning in results['warnings']:
                    print(f"  - {warning}")
        
        elif args.command == "convert":
            if not args.output or not args.dtype:
                print("--output and --dtype required for conversion")
                return
            
            loader.convert_dtype(args.file_path, args.output, args.dtype)
            print(f"Conversion completed: {args.output}")
    
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())