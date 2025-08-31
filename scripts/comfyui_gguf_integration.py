#!/usr/bin/env python3
"""
ComfyUI GGUF Integration Helper
Provides seamless integration between GGUF loader and ComfyUI workflow.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, Tuple

import torch
from .gguf_loader import (
    GGUFLoader, gguf_sd_loader, gguf_tokenizer_loader, 
    gguf_clip_loader, get_gguf_info, GGUFError
)

logger = logging.getLogger(__name__)

class ComfyUIGGUFLoader:
    """ComfyUI-specific GGUF loader with workflow integration."""
    
    def __init__(self, models_dir: Optional[str] = None):
        """Initialize with ComfyUI models directory."""
        self.models_dir = Path(models_dir) if models_dir else Path("models")
        self.loaded_models = {}
        self.model_info_cache = {}
    
    def find_gguf_files(self, subdirs: Optional[List[str]] = None) -> Dict[str, List[Path]]:
        """Find all GGUF files in models directory."""
        if subdirs is None:
            subdirs = [
                "checkpoints", "unet", "vae", "clip", "text_encoders",
                "diffusion_models", "loras", "embeddings", "controlnet"
            ]
        
        found_files = {}
        
        for subdir in subdirs:
            subdir_path = self.models_dir / subdir
            if not subdir_path.exists():
                continue
            
            gguf_files = list(subdir_path.glob("*.gguf"))
            if gguf_files:
                found_files[subdir] = gguf_files
        
        return found_files
    
    def get_model_info(self, file_path: Union[str, Path], use_cache: bool = True) -> Dict[str, Any]:
        """Get model information with caching."""
        file_path = Path(file_path)
        
        if use_cache and str(file_path) in self.model_info_cache:
            return self.model_info_cache[str(file_path)]
        
        try:
            info = get_gguf_info(file_path)
            info["file_path"] = str(file_path)
            info["file_name"] = file_path.name
            info["file_size_mb"] = info.get("file_size", 0) / (1024 * 1024)
            
            if use_cache:
                self.model_info_cache[str(file_path)] = info
            
            return info
        except Exception as e:
            logger.error(f"Failed to get info for {file_path}: {e}")
            return {"error": str(e), "file_path": str(file_path)}
    
    def load_model_for_comfyui(self, 
                              file_path: Union[str, Path],
                              model_type: str = "auto",
                              device: str = "cuda",
                              dtype: torch.dtype = torch.float16) -> Dict[str, Any]:
        """
        Load GGUF model optimized for ComfyUI usage.
        
        Args:
            file_path: Path to GGUF file
            model_type: Type of model ("auto", "unet", "vae", "clip", "text_encoder")
            device: Device to load on
            dtype: Target dtype for tensors
        
        Returns:
            Dictionary with model data and metadata
        """
        file_path = Path(file_path)
        cache_key = f"{file_path}_{model_type}_{device}_{dtype}"
        
        if cache_key in self.loaded_models:
            logger.info(f"Using cached model: {file_path.name}")
            return self.loaded_models[cache_key]
        
        try:
            # Get model info first
            model_info = self.get_model_info(file_path)
            architecture = model_info.get("architecture")
            
            # Determine loading strategy based on model type
            if model_type == "auto":
                model_type = self._detect_model_type(model_info, file_path)
            
            logger.info(f"Loading {model_type} model: {file_path.name} (arch: {architecture})")
            
            # Load based on model type
            model_data = {}
            
            if model_type == "clip":
                clip_model = gguf_clip_loader(file_path, device)
                if clip_model:
                    model_data["state_dict"] = clip_model
                    model_data["tokenizer"] = gguf_tokenizer_loader(file_path)
                else:
                    # Fallback to regular state dict loading
                    model_data["state_dict"] = gguf_sd_loader(file_path, architecture, device)
            
            elif model_type == "text_encoder":
                model_data["state_dict"] = gguf_sd_loader(file_path, architecture, device)
                model_data["tokenizer"] = gguf_tokenizer_loader(file_path)
            
            else:
                # Default state dict loading
                model_data["state_dict"] = gguf_sd_loader(file_path, architecture, device)
            
            # Convert dtype if needed
            if dtype != torch.float32:
                model_data["state_dict"] = self._convert_dtype(model_data["state_dict"], dtype)
            
            # Add metadata
            model_data.update({
                "model_info": model_info,
                "model_type": model_type,
                "architecture": architecture,
                "device": device,
                "dtype": dtype,
                "quantization_types": model_info.get("quantization_types", [])
            })
            
            # Cache the result
            self.loaded_models[cache_key] = model_data
            
            logger.info(f"Successfully loaded {model_type} with {len(model_data['state_dict'])} tensors")
            return model_data
            
        except Exception as e:
            logger.error(f"Failed to load GGUF model {file_path}: {e}")
            raise GGUFError(f"Failed to load GGUF model: {e}")
    
    def _detect_model_type(self, model_info: Dict[str, Any], file_path: Path) -> str:
        """Detect model type from info and file path."""
        architecture = model_info.get("architecture", "").lower()
        file_name = file_path.name.lower()
        
        # Check architecture
        if "clip" in architecture:
            return "clip"
        elif architecture in ["t5", "t5encoder"]:
            return "text_encoder"
        elif architecture in ["flux", "sd1", "sdxl", "sd3"]:
            return "unet"
        elif "vae" in architecture:
            return "vae"
        
        # Check file name
        if "clip" in file_name:
            return "clip"
        elif "t5" in file_name or "text" in file_name:
            return "text_encoder"
        elif "vae" in file_name:
            return "vae"
        elif any(term in file_name for term in ["unet", "diffusion", "flux"]):
            return "unet"
        
        # Check file location
        parent_dir = file_path.parent.name.lower()
        if parent_dir in ["clip", "text_encoders"]:
            return "clip" if parent_dir == "clip" else "text_encoder"
        elif parent_dir in ["unet", "diffusion_models"]:
            return "unet"
        elif parent_dir == "vae":
            return "vae"
        
        # Default to unet for most diffusion models
        return "unet"
    
    def _convert_dtype(self, state_dict: Dict[str, torch.Tensor], target_dtype: torch.dtype) -> Dict[str, torch.Tensor]:
        """Convert all tensors in state dict to target dtype."""
        converted = {}
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor) and tensor.dtype.is_floating_point:
                converted[key] = tensor.to(target_dtype)
            else:
                converted[key] = tensor
        return converted
    
    def list_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """List all available GGUF models with their info."""
        available_models = {}
        gguf_files = self.find_gguf_files()
        
        for category, files in gguf_files.items():
            model_list = []
            for file_path in files:
                info = self.get_model_info(file_path)
                info["category"] = category
                model_list.append(info)
            available_models[category] = model_list
        
        return available_models
    
    def clear_cache(self):
        """Clear all cached models and info."""
        self.loaded_models.clear()
        self.model_info_cache.clear()
        logger.info("Cleared GGUF model cache")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information for loaded models."""
        usage = {
            "loaded_models": len(self.loaded_models),
            "cached_info": len(self.model_info_cache),
            "total_tensors": 0,
            "estimated_memory_mb": 0
        }
        
        for model_data in self.loaded_models.values():
            state_dict = model_data.get("state_dict", {})
            usage["total_tensors"] += len(state_dict)
            
            # Estimate memory usage
            for tensor in state_dict.values():
                if isinstance(tensor, torch.Tensor):
                    usage["estimated_memory_mb"] += tensor.numel() * tensor.element_size() / (1024 * 1024)
        
        return usage

# ComfyUI Node Integration Examples
class GGUFModelLoader:
    """ComfyUI custom node for loading GGUF models."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gguf_path": ("STRING", {"default": ""}),
                "model_type": (["auto", "unet", "vae", "clip", "text_encoder"], {"default": "auto"}),
                "device": (["cuda", "cpu", "auto"], {"default": "cuda"}),
                "dtype": (["float32", "float16", "bfloat16"], {"default": "float16"}),
            }
        }
    
    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    FUNCTION = "load_gguf_model"
    CATEGORY = "loaders/gguf"
    
    def __init__(self):
        self.loader = ComfyUIGGUFLoader()
    
    def load_gguf_model(self, gguf_path: str, model_type: str, device: str, dtype: str):
        """Load GGUF model for ComfyUI."""
        if not gguf_path or not Path(gguf_path).exists():
            raise ValueError(f"GGUF file not found: {gguf_path}")
        
        # Convert dtype string to torch.dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        torch_dtype = dtype_map.get(dtype, torch.float16)
        
        # Auto-detect device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            model_data = self.loader.load_model_for_comfyui(
                gguf_path, model_type, device, torch_dtype
            )
            
            # Return appropriate objects based on model type
            detected_type = model_data["model_type"]
            
            if detected_type == "clip":
                return (None, model_data, None)
            elif detected_type == "vae":
                return (None, None, model_data)
            else:
                return (model_data, None, None)
                
        except Exception as e:
            logger.error(f"Failed to load GGUF model: {e}")
            return (None, None, None)

# Utility functions for ComfyUI integration
def register_gguf_nodes():
    """Register GGUF-related nodes with ComfyUI."""
    try:
        # This would be called in ComfyUI's node registration system
        NODE_CLASS_MAPPINGS = {
            "GGUFModelLoader": GGUFModelLoader,
        }
        
        NODE_DISPLAY_NAME_MAPPINGS = {
            "GGUFModelLoader": "GGUF Model Loader",
        }
        
        return NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    except Exception as e:
        logger.error(f"Failed to register GGUF nodes: {e}")
        return {}, {}

def create_comfyui_gguf_workflow(gguf_models: Dict[str, str]) -> Dict[str, Any]:
    """
    Create a ComfyUI workflow using GGUF models.
    
    Args:
        gguf_models: Dictionary mapping model types to GGUF file paths
    
    Returns:
        ComfyUI workflow dictionary
    """
    workflow = {
        "nodes": {},
        "links": [],
        "groups": [],
        "config": {},
        "extra": {},
        "version": 0.4
    }
    
    node_id = 1
    
    # Add GGUF loader nodes
    for model_type, file_path in gguf_models.items():
        workflow["nodes"][str(node_id)] = {
            "id": node_id,
            "type": "GGUFModelLoader",
            "pos": [100, 100 + (node_id * 150)],
            "size": {"0": 400, "1": 200},
            "flags": {},
            "order": node_id,
            "mode": 0,
            "inputs": [],
            "outputs": [],
            "properties": {
                "gguf_path": file_path,
                "model_type": model_type,
                "device": "auto",
                "dtype": "float16"
            }
        }
        node_id += 1
    
    return workflow

def main():
    """Example usage of ComfyUI GGUF integration."""
    # Initialize loader
    loader = ComfyUIGGUFLoader()
    
    # Find available models
    available_models = loader.list_available_models()
    print("Available GGUF models:")
    for category, models in available_models.items():
        print(f"\n{category}:")
        for model in models:
            print(f"  - {model['file_name']} ({model.get('file_size_mb', 0):.1f} MB)")
            print(f"    Architecture: {model.get('architecture', 'unknown')}")
            print(f"    Quantization: {model.get('quantization_types', [])}")
    
    # Show memory usage
    usage = loader.get_memory_usage()
    print(f"\nMemory usage:")
    print(f"  Loaded models: {usage['loaded_models']}")
    print(f"  Total tensors: {usage['total_tensors']}")
    print(f"  Estimated memory: {usage['estimated_memory_mb']:.1f} MB")

if __name__ == "__main__":
    main()