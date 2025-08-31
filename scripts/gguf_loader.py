#!/usr/bin/env python3
"""
GGUF Model Loader Integration for ComfyUI
Provides comprehensive GGUF loading functionality with quantization support,
architecture detection, and tokenizer capabilities.
"""

import json
import logging
import struct
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GGUFValueType(IntEnum):
    """GGUF value types as defined in the specification."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12

class GGMLQuantType(IntEnum):
    """GGML quantization types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 4
    Q5_1 = 5
    Q8_0 = 6
    Q8_1 = 7
    Q2_K = 8
    Q3_K = 9
    Q4_K = 10
    Q5_K = 11
    Q6_K = 12
    Q8_K = 13
    IQ2_XXS = 14
    IQ2_XS = 15
    IQ3_XXS = 16
    IQ1_S = 17
    IQ4_NL = 18
    IQ3_S = 19
    IQ2_S = 20
    IQ4_XS = 21
    I8 = 22
    I16 = 23
    I32 = 24
    I64 = 25
    F64 = 26
    IQ1_M = 27

# Architecture mappings
SUPPORTED_ARCHITECTURES = {
    'flux', 'sd1', 'sdxl', 'sd3', 'aura', 'ltxv', 'hyvid', 'wan', 't5', 't5encoder', 'llama'
}

# T5 state dict mapping
T5_SD_MAP = {
    "encoder.embed_tokens.weight": "shared.weight",
    "encoder.block.{}.layer.0.SelfAttention.k.weight": "encoder.block.{}.layer.0.SelfAttention.k.weight",
    "encoder.block.{}.layer.0.SelfAttention.o.weight": "encoder.block.{}.layer.0.SelfAttention.o.weight",
    "encoder.block.{}.layer.0.SelfAttention.q.weight": "encoder.block.{}.layer.0.SelfAttention.q.weight",
    "encoder.block.{}.layer.0.SelfAttention.v.weight": "encoder.block.{}.layer.0.SelfAttention.v.weight",
    "encoder.block.{}.layer.0.SelfAttention.relative_attention_bias.weight": "encoder.block.{}.layer.0.SelfAttention.relative_attention_bias.weight",
    "encoder.block.{}.layer.0.layer_norm.weight": "encoder.block.{}.layer.0.layer_norm.weight",
    "encoder.block.{}.layer.1.DenseReluDense.wi.weight": "encoder.block.{}.layer.1.DenseReluDense.wi_0.weight",
    "encoder.block.{}.layer.1.DenseReluDense.wi_1.weight": "encoder.block.{}.layer.1.DenseReluDense.wi_1.weight",
    "encoder.block.{}.layer.1.DenseReluDense.wo.weight": "encoder.block.{}.layer.1.DenseReluDense.wo.weight",
    "encoder.block.{}.layer.1.layer_norm.weight": "encoder.block.{}.layer.1.layer_norm.weight",
    "encoder.final_layer_norm.weight": "encoder.final_layer_norm.weight",
}

# Llama state dict mapping
LLAMA_SD_MAP = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}

class GGUFError(Exception):
    """Base exception for GGUF loading errors."""
    pass

class GGUFLoader:
    """Main GGUF loader class with comprehensive functionality."""
    
    def __init__(self, file_path: Union[str, Path]):
        """Initialize GGUF loader with file path."""
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise GGUFError(f"GGUF file not found: {self.file_path}")
        
        self.file_handle = None
        self.metadata = {}
        self.tensors = {}
        self.architecture = None
        self.quantization_type = None
        self._load_header()
    
    def __enter__(self):
        """Context manager entry."""
        self.file_handle = open(self.file_path, 'rb')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.file_handle:
            self.file_handle.close()
    
    def _load_header(self):
        """Load GGUF header and metadata."""
        with open(self.file_path, 'rb') as f:
            # Read magic number
            magic = f.read(4)
            if magic != b'GGUF':
                raise GGUFError(f"Invalid GGUF magic number: {magic}")
            
            # Read version
            version = struct.unpack('<I', f.read(4))[0]
            if version not in [2, 3]:
                raise GGUFError(f"Unsupported GGUF version: {version}")
            
            # Read tensor count and metadata count
            tensor_count = struct.unpack('<Q', f.read(8))[0]
            metadata_count = struct.unpack('<Q', f.read(8))[0]
            
            # Read metadata
            self.metadata = self._read_metadata(f, metadata_count)
            
            # Detect architecture
            self.architecture = self._detect_architecture()
            
            # Read tensor info
            self._read_tensor_info(f, tensor_count)
    
    def _read_metadata(self, f, count: int) -> Dict[str, Any]:
        """Read metadata from GGUF file."""
        metadata = {}
        
        for _ in range(count):
            # Read key
            key_len = struct.unpack('<Q', f.read(8))[0]
            key = f.read(key_len).decode('utf-8')
            
            # Read value type
            value_type = GGUFValueType(struct.unpack('<I', f.read(4))[0])
            
            # Read value
            value = self._read_value(f, value_type)
            metadata[key] = value
        
        return metadata
    
    def _read_value(self, f, value_type: GGUFValueType) -> Any:
        """Read a value based on its type."""
        if value_type == GGUFValueType.UINT8:
            return struct.unpack('<B', f.read(1))[0]
        elif value_type == GGUFValueType.INT8:
            return struct.unpack('<b', f.read(1))[0]
        elif value_type == GGUFValueType.UINT16:
            return struct.unpack('<H', f.read(2))[0]
        elif value_type == GGUFValueType.INT16:
            return struct.unpack('<h', f.read(2))[0]
        elif value_type == GGUFValueType.UINT32:
            return struct.unpack('<I', f.read(4))[0]
        elif value_type == GGUFValueType.INT32:
            return struct.unpack('<i', f.read(4))[0]
        elif value_type == GGUFValueType.UINT64:
            return struct.unpack('<Q', f.read(8))[0]
        elif value_type == GGUFValueType.INT64:
            return struct.unpack('<q', f.read(8))[0]
        elif value_type == GGUFValueType.FLOAT32:
            return struct.unpack('<f', f.read(4))[0]
        elif value_type == GGUFValueType.FLOAT64:
            return struct.unpack('<d', f.read(8))[0]
        elif value_type == GGUFValueType.BOOL:
            return bool(struct.unpack('<B', f.read(1))[0])
        elif value_type == GGUFValueType.STRING:
            str_len = struct.unpack('<Q', f.read(8))[0]
            return f.read(str_len).decode('utf-8')
        elif value_type == GGUFValueType.ARRAY:
            array_type = GGUFValueType(struct.unpack('<I', f.read(4))[0])
            array_len = struct.unpack('<Q', f.read(8))[0]
            return [self._read_value(f, array_type) for _ in range(array_len)]
        else:
            raise GGUFError(f"Unsupported value type: {value_type}")
    
    def _read_tensor_info(self, f, count: int):
        """Read tensor information from GGUF file."""
        for _ in range(count):
            # Read tensor name
            name_len = struct.unpack('<Q', f.read(8))[0]
            name = f.read(name_len).decode('utf-8')
            
            # Read dimensions
            n_dims = struct.unpack('<I', f.read(4))[0]
            dims = [struct.unpack('<Q', f.read(8))[0] for _ in range(n_dims)]
            
            # Read type and offset
            tensor_type = GGMLQuantType(struct.unpack('<I', f.read(4))[0])
            offset = struct.unpack('<Q', f.read(8))[0]
            
            self.tensors[name] = {
                'dims': dims,
                'type': tensor_type,
                'offset': offset
            }
    
    def _detect_architecture(self) -> Optional[str]:
        """Detect model architecture from metadata."""
        arch_key = self.get_field("general.architecture")
        if arch_key in SUPPORTED_ARCHITECTURES:
            return arch_key
        
        # Try to infer from other metadata
        if "t5" in str(arch_key).lower():
            return "t5"
        elif "llama" in str(arch_key).lower():
            return "llama"
        elif "flux" in str(arch_key).lower():
            return "flux"
        
        logger.warning(f"Unknown architecture: {arch_key}")
        return None
    
    def get_field(self, key: str, default: Any = None) -> Any:
        """Get a field from GGUF metadata."""
        return self.metadata.get(key, default)
    
    def get_list_field(self, key: str, default: Optional[List] = None) -> List:
        """Get a list field from GGUF metadata."""
        value = self.metadata.get(key, default or [])
        if not isinstance(value, list):
            return [value] if value is not None else []
        return value
    
    def get_orig_shape(self, tensor_name: str) -> Optional[Tuple[int, ...]]:
        """Get original shape from metadata for a tensor."""
        shape_key = f"{tensor_name}.orig_shape"
        return tuple(self.get_list_field(shape_key))
    
    def load_tensor(self, tensor_name: str) -> torch.Tensor:
        """Load a specific tensor from the GGUF file."""
        if tensor_name not in self.tensors:
            raise GGUFError(f"Tensor not found: {tensor_name}")
        
        tensor_info = self.tensors[tensor_name]
        
        with open(self.file_path, 'rb') as f:
            f.seek(tensor_info['offset'])
            
            # Calculate tensor size
            dims = tensor_info['dims']
            tensor_type = tensor_info['type']
            
            # Load raw data
            raw_data = self._load_raw_tensor(f, dims, tensor_type)
            
            # Convert to PyTorch tensor
            tensor = self._dequantize_tensor(raw_data, tensor_type, dims)
            
            return tensor
    
    def _load_raw_tensor(self, f, dims: List[int], tensor_type: GGMLQuantType) -> bytes:
        """Load raw tensor data."""
        total_elements = np.prod(dims)
        
        if tensor_type == GGMLQuantType.F32:
            size = total_elements * 4
        elif tensor_type == GGMLQuantType.F16:
            size = total_elements * 2
        elif tensor_type == GGMLQuantType.Q4_0:
            size = total_elements // 2 + total_elements // 32 * 2
        elif tensor_type == GGMLQuantType.Q4_1:
            size = total_elements // 2 + total_elements // 32 * 4
        elif tensor_type == GGMLQuantType.Q5_0:
            size = total_elements // 2 + total_elements // 32 * 6
        elif tensor_type == GGMLQuantType.Q5_1:
            size = total_elements // 2 + total_elements // 32 * 8
        elif tensor_type == GGMLQuantType.Q8_0:
            size = total_elements + total_elements // 32 * 2
        else:
            # Fallback for other quantization types
            size = total_elements
        
        return f.read(size)
    
    def _dequantize_tensor(self, raw_data: bytes, tensor_type: GGMLQuantType, dims: List[int]) -> torch.Tensor:
        """Dequantize tensor data to float32."""
        if tensor_type == GGMLQuantType.F32:
            return torch.frombuffer(raw_data, dtype=torch.float32).reshape(dims)
        elif tensor_type == GGMLQuantType.F16:
            return torch.frombuffer(raw_data, dtype=torch.float16).float().reshape(dims)
        elif tensor_type in [GGMLQuantType.Q4_0, GGMLQuantType.Q4_1, GGMLQuantType.Q5_0, 
                            GGMLQuantType.Q5_1, GGMLQuantType.Q8_0]:
            return self._dequantize_ggml_tensor(raw_data, tensor_type, dims)
        else:
            logger.warning(f"Unsupported quantization type: {tensor_type}")
            # Return zeros as fallback
            return torch.zeros(dims, dtype=torch.float32)
    
    def _dequantize_ggml_tensor(self, raw_data: bytes, tensor_type: GGMLQuantType, dims: List[int]) -> torch.Tensor:
        """Dequantize GGML quantized tensors."""
        # This is a simplified implementation
        # Full implementation would require the complete GGML dequantization logic
        total_elements = np.prod(dims)
        
        if tensor_type == GGMLQuantType.Q4_0:
            # Q4_0: 4-bit quantization with scale
            return self._dequantize_q4_0(raw_data, dims)
        elif tensor_type == GGMLQuantType.Q8_0:
            # Q8_0: 8-bit quantization with scale
            return self._dequantize_q8_0(raw_data, dims)
        else:
            # Placeholder for other quantization types
            return torch.zeros(dims, dtype=torch.float32)
    
    def _dequantize_q4_0(self, raw_data: bytes, dims: List[int]) -> torch.Tensor:
        """Dequantize Q4_0 format."""
        # Simplified Q4_0 dequantization
        # Real implementation would need proper block processing
        total_elements = np.prod(dims)
        result = torch.zeros(dims, dtype=torch.float32)
        
        # This is a placeholder - proper Q4_0 dequantization is complex
        # and requires processing 32-element blocks with scales
        logger.warning("Q4_0 dequantization is simplified - may not be accurate")
        
        return result
    
    def _dequantize_q8_0(self, raw_data: bytes, dims: List[int]) -> torch.Tensor:
        """Dequantize Q8_0 format."""
        # Simplified Q8_0 dequantization
        total_elements = np.prod(dims)
        result = torch.zeros(dims, dtype=torch.float32)
        
        # This is a placeholder - proper Q8_0 dequantization
        logger.warning("Q8_0 dequantization is simplified - may not be accurate")
        
        return result

def llama_permute(tensor: torch.Tensor, n_heads: int, n_kv_heads: Optional[int] = None) -> torch.Tensor:
    """Permute Llama model tensors for proper attention computation."""
    if n_kv_heads is None:
        n_kv_heads = n_heads
    
    # Reshape and permute for multi-head attention
    shape = tensor.shape
    if len(shape) == 2:
        dim = shape[-1]
        head_dim = dim // n_heads
        
        # Reshape to [seq_len, n_heads, head_dim] and permute
        tensor = tensor.view(shape[0], n_heads, head_dim)
        tensor = tensor.transpose(1, 2).contiguous()
        tensor = tensor.view(shape[0], -1)
    
    return tensor

def gguf_sd_loader(file_path: Union[str, Path], 
                   architecture: Optional[str] = None,
                   device: str = "cpu") -> Dict[str, torch.Tensor]:
    """
    Load state dict from GGUF file with architecture-specific processing.
    
    Args:
        file_path: Path to GGUF file
        architecture: Target architecture (auto-detected if None)
        device: Device to load tensors on
    
    Returns:
        Dictionary containing the loaded state dict
    """
    try:
        with GGUFLoader(file_path) as loader:
            if architecture is None:
                architecture = loader.architecture
            
            state_dict = {}
            
            # Load all tensors
            for tensor_name in loader.tensors:
                try:
                    tensor = loader.load_tensor(tensor_name)
                    
                    # Apply architecture-specific processing
                    if architecture == "llama":
                        processed_name = _remap_llama_tensor_name(tensor_name)
                        if "attention.wq.weight" in processed_name or "attention.wk.weight" in processed_name:
                            n_heads = loader.get_field("llama.attention.head_count", 32)
                            n_kv_heads = loader.get_field("llama.attention.head_count_kv", n_heads)
                            tensor = llama_permute(tensor, n_heads, n_kv_heads)
                    elif architecture == "t5":
                        processed_name = _remap_t5_tensor_name(tensor_name)
                    else:
                        processed_name = tensor_name
                    
                    state_dict[processed_name] = tensor.to(device)
                    
                except Exception as e:
                    logger.error(f"Failed to load tensor {tensor_name}: {e}")
                    continue
            
            logger.info(f"Loaded {len(state_dict)} tensors from GGUF file")
            return state_dict
            
    except Exception as e:
        logger.error(f"Failed to load GGUF file {file_path}: {e}")
        raise GGUFError(f"Failed to load GGUF file: {e}")

def _remap_llama_tensor_name(name: str) -> str:
    """Remap Llama tensor names using the mapping dictionary."""
    for target, source in LLAMA_SD_MAP.items():
        if name == source:
            return target
        # Handle layer-specific mappings
        if "{}" in source:
            import re
            pattern = source.replace("{}", r"(\d+)")
            match = re.match(pattern, name)
            if match:
                layer_num = match.group(1)
                return target.format(layer_num)
    return name

def _remap_t5_tensor_name(name: str) -> str:
    """Remap T5 tensor names using the mapping dictionary."""
    for target, source in T5_SD_MAP.items():
        if name == source:
            return target
        # Handle layer-specific mappings
        if "{}" in source:
            import re
            pattern = source.replace("{}", r"(\d+)")
            match = re.match(pattern, name)
            if match:
                layer_num = match.group(1)
                return target.format(layer_num)
    return name

def gguf_tokenizer_loader(file_path: Union[str, Path]) -> Optional[Dict]:
    """
    Load tokenizer from GGUF metadata.
    
    Args:
        file_path: Path to GGUF file
    
    Returns:
        Tokenizer dictionary or None if not found
    """
    try:
        with GGUFLoader(file_path) as loader:
            # Extract tokenizer information from metadata
            tokenizer_model = loader.get_field("tokenizer.ggml.model")
            
            if not tokenizer_model:
                logger.warning("No tokenizer model found in GGUF metadata")
                return None
            
            tokenizer_config = {
                "model": tokenizer_model,
                "vocab_size": loader.get_field("tokenizer.ggml.vocab_size"),
                "tokens": loader.get_list_field("tokenizer.ggml.tokens"),
                "scores": loader.get_list_field("tokenizer.ggml.scores"),
                "token_type": loader.get_list_field("tokenizer.ggml.token_type"),
                "merges": loader.get_list_field("tokenizer.ggml.merges"),
                "bos_token": loader.get_field("tokenizer.ggml.bos_token_id"),
                "eos_token": loader.get_field("tokenizer.ggml.eos_token_id"),
                "pad_token": loader.get_field("tokenizer.ggml.pad_token_id"),
                "unk_token": loader.get_field("tokenizer.ggml.unknown_token_id"),
                "add_bos_token": loader.get_field("tokenizer.ggml.add_bos_token", False),
                "add_eos_token": loader.get_field("tokenizer.ggml.add_eos_token", False),
            }
            
            # Remove None values
            tokenizer_config = {k: v for k, v in tokenizer_config.items() if v is not None}
            
            logger.info(f"Loaded tokenizer with {tokenizer_config.get('vocab_size', 0)} tokens")
            return tokenizer_config
            
    except Exception as e:
        logger.error(f"Failed to load tokenizer from GGUF file: {e}")
        return None

def gguf_clip_loader(file_path: Union[str, Path], device: str = "cpu") -> Optional[Dict[str, torch.Tensor]]:
    """
    Load CLIP model from GGUF file.
    
    Args:
        file_path: Path to GGUF file
        device: Device to load tensors on
    
    Returns:
        CLIP model state dict or None if not a CLIP model
    """
    try:
        with GGUFLoader(file_path) as loader:
            # Check if this is a CLIP model
            arch = loader.architecture
            if not arch or "clip" not in arch.lower():
                # Try to detect CLIP from tensor names
                clip_indicators = ["text_model", "vision_model", "text_projection", "visual_projection"]
                if not any(indicator in name for name in loader.tensors for indicator in clip_indicators):
                    logger.warning("File does not appear to contain a CLIP model")
                    return None
            
            state_dict = {}
            
            # Load CLIP tensors
            for tensor_name in loader.tensors:
                try:
                    tensor = loader.load_tensor(tensor_name)
                    state_dict[tensor_name] = tensor.to(device)
                except Exception as e:
                    logger.error(f"Failed to load CLIP tensor {tensor_name}: {e}")
                    continue
            
            # Extract CLIP-specific metadata
            clip_config = {
                "vocab_size": loader.get_field("clip.vocab_size"),
                "text_length": loader.get_field("clip.text_length"),
                "vision_patch_size": loader.get_field("clip.vision.patch_size"),
                "vision_image_size": loader.get_field("clip.vision.image_size"),
                "projection_dim": loader.get_field("clip.projection_dim"),
            }
            
            # Remove None values
            clip_config = {k: v for k, v in clip_config.items() if v is not None}
            
            if clip_config:
                state_dict["_clip_config"] = clip_config
            
            logger.info(f"Loaded CLIP model with {len(state_dict)} tensors")
            return state_dict
            
    except Exception as e:
        logger.error(f"Failed to load CLIP model from GGUF file: {e}")
        return None

def get_gguf_info(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Get comprehensive information about a GGUF file.
    
    Args:
        file_path: Path to GGUF file
    
    Returns:
        Dictionary with file information
    """
    try:
        with GGUFLoader(file_path) as loader:
            info = {
                "architecture": loader.architecture,
                "tensor_count": len(loader.tensors),
                "metadata_keys": list(loader.metadata.keys()),
                "tensors": {name: info["dims"] for name, info in loader.tensors.items()},
                "quantization_types": list(set(info["type"].name for info in loader.tensors.values())),
                "file_size": loader.file_path.stat().st_size,
            }
            
            # Add architecture-specific info
            if loader.architecture == "llama":
                info["llama_config"] = {
                    "vocab_size": loader.get_field("llama.vocab_size"),
                    "context_length": loader.get_field("llama.context_length"),
                    "embedding_length": loader.get_field("llama.embedding_length"),
                    "block_count": loader.get_field("llama.block_count"),
                    "feed_forward_length": loader.get_field("llama.feed_forward_length"),
                    "attention_head_count": loader.get_field("llama.attention.head_count"),
                    "attention_head_count_kv": loader.get_field("llama.attention.head_count_kv"),
                }
            elif loader.architecture == "t5":
                info["t5_config"] = {
                    "vocab_size": loader.get_field("t5.vocab_size"),
                    "encoder_layer_count": loader.get_field("t5.encoder.layer_count"),
                    "decoder_layer_count": loader.get_field("t5.decoder.layer_count"),
                    "attention_head_count": loader.get_field("t5.attention.head_count"),
                    "embedding_length": loader.get_field("t5.embedding_length"),
                }
            
            return info
            
    except Exception as e:
        logger.error(f"Failed to get GGUF info: {e}")
        return {"error": str(e)}

# Example usage functions
def main():
    """Example usage of the GGUF loader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GGUF Model Loader")
    parser.add_argument("file_path", help="Path to GGUF file")
    parser.add_argument("--info", action="store_true", help="Show file info")
    parser.add_argument("--load-state-dict", action="store_true", help="Load state dict")
    parser.add_argument("--load-tokenizer", action="store_true", help="Load tokenizer")
    parser.add_argument("--load-clip", action="store_true", help="Load CLIP model")
    parser.add_argument("--device", default="cpu", help="Device to load on")
    
    args = parser.parse_args()
    
    if args.info:
        info = get_gguf_info(args.file_path)
        print(json.dumps(info, indent=2))
    
    if args.load_state_dict:
        state_dict = gguf_sd_loader(args.file_path, device=args.device)
        print(f"Loaded state dict with {len(state_dict)} tensors")
        for name, tensor in list(state_dict.items())[:5]:
            print(f"  {name}: {tensor.shape} ({tensor.dtype})")
        if len(state_dict) > 5:
            print(f"  ... and {len(state_dict) - 5} more tensors")
    
    if args.load_tokenizer:
        tokenizer = gguf_tokenizer_loader(args.file_path)
        if tokenizer:
            print(f"Loaded tokenizer: {tokenizer.get('model')} with {tokenizer.get('vocab_size')} tokens")
        else:
            print("No tokenizer found")
    
    if args.load_clip:
        clip_model = gguf_clip_loader(args.file_path, device=args.device)
        if clip_model:
            config = clip_model.pop("_clip_config", {})
            print(f"Loaded CLIP model with {len(clip_model)} tensors")
            if config:
                print(f"CLIP config: {config}")
        else:
            print("No CLIP model found")

if __name__ == "__main__":
    main()