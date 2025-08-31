#!/usr/bin/env python3
"""
GGUF Loader Usage Examples
Demonstrates various ways to use the GGUF loader with different model types.
"""

import sys
from pathlib import Path

# Add scripts directory to path
sys.path.append(str(Path(__file__).parent.parent / "scripts"))

import torch
from gguf_loader import (
    GGUFLoader, gguf_sd_loader, gguf_tokenizer_loader, 
    gguf_clip_loader, get_gguf_info
)
from comfyui_gguf_integration import ComfyUIGGUFLoader

def example_basic_loading():
    """Example: Basic GGUF file loading and inspection."""
    print("=== Basic GGUF Loading Example ===")
    
    # Replace with your actual GGUF file path
    gguf_file = "models/example_model.gguf"
    
    if not Path(gguf_file).exists():
        print(f"Example file {gguf_file} not found. Skipping basic loading example.")
        return
    
    try:
        # Get file information
        info = get_gguf_info(gguf_file)
        print(f"Model architecture: {info['architecture']}")
        print(f"Total tensors: {info['tensor_count']}")
        print(f"File size: {info['file_size'] / 1024 / 1024:.1f} MB")
        print(f"Quantization types: {info['quantization_types']}")
        
        # Load state dict
        state_dict = gguf_sd_loader(gguf_file, device="cpu")
        print(f"Loaded {len(state_dict)} tensors")
        
        # Show first few tensor shapes
        for i, (name, tensor) in enumerate(state_dict.items()):
            if i >= 5:
                break
            print(f"  {name}: {tensor.shape} ({tensor.dtype})")
        
    except Exception as e:
        print(f"Error loading GGUF file: {e}")

def example_flux_model():
    """Example: Loading a FLUX diffusion model."""
    print("\n=== FLUX Model Loading Example ===")
    
    flux_file = "models/unet/flux_model.gguf"
    
    if not Path(flux_file).exists():
        print(f"FLUX model {flux_file} not found. Skipping FLUX example.")
        return
    
    try:
        with GGUFLoader(flux_file) as loader:
            print(f"FLUX model architecture: {loader.architecture}")
            
            # Get FLUX-specific metadata
            flux_config = {
                "hidden_size": loader.get_field("flux.hidden_size"),
                "num_layers": loader.get_field("flux.num_layers"),
                "num_heads": loader.get_field("flux.num_attention_heads"),
                "guidance_scale": loader.get_field("flux.guidance_scale", 7.5),
            }
            
            print("FLUX configuration:")
            for key, value in flux_config.items():
                print(f"  {key}: {value}")
            
            # Load specific tensors
            tensor_names = ["time_embedding", "context_embedding", "output_projection"]
            for name in tensor_names:
                matching_tensors = [t for t in loader.tensors if name in t.lower()]
                if matching_tensors:
                    tensor = loader.load_tensor(matching_tensors[0])
                    print(f"  {matching_tensors[0]}: {tensor.shape}")
    
    except Exception as e:
        print(f"Error loading FLUX model: {e}")

def example_t5_text_encoder():
    """Example: Loading a T5 text encoder model."""
    print("\n=== T5 Text Encoder Example ===")
    
    t5_file = "models/text_encoders/t5_xxl.gguf"
    
    if not Path(t5_file).exists():
        print(f"T5 model {t5_file} not found. Skipping T5 example.")
        return
    
    try:
        # Load T5 state dict with proper key remapping
        state_dict = gguf_sd_loader(t5_file, architecture="t5")
        print(f"Loaded T5 with {len(state_dict)} tensors")
        
        # Load tokenizer
        tokenizer = gguf_tokenizer_loader(t5_file)
        if tokenizer:
            print(f"T5 tokenizer: {tokenizer['model']} with {tokenizer['vocab_size']} tokens")
            print(f"Special tokens - BOS: {tokenizer.get('bos_token')}, EOS: {tokenizer.get('eos_token')}")
        
        # Show T5-specific tensors
        t5_tensors = ["shared.weight", "encoder.final_layer_norm.weight"]
        for tensor_name in t5_tensors:
            if tensor_name in state_dict:
                tensor = state_dict[tensor_name]
                print(f"  {tensor_name}: {tensor.shape}")
    
    except Exception as e:
        print(f"Error loading T5 model: {e}")

def example_llama_model():
    """Example: Loading a Llama language model."""
    print("\n=== Llama Model Example ===")
    
    llama_file = "models/text_encoders/llama_7b.gguf"
    
    if not Path(llama_file).exists():
        print(f"Llama model {llama_file} not found. Skipping Llama example.")
        return
    
    try:
        with GGUFLoader(llama_file) as loader:
            print(f"Llama architecture: {loader.architecture}")
            
            # Get Llama-specific config
            llama_config = {
                "vocab_size": loader.get_field("llama.vocab_size"),
                "context_length": loader.get_field("llama.context_length"),
                "embedding_length": loader.get_field("llama.embedding_length"),
                "layers": loader.get_field("llama.block_count"),
                "heads": loader.get_field("llama.attention.head_count"),
                "kv_heads": loader.get_field("llama.attention.head_count_kv"),
            }
            
            print("Llama configuration:")
            for key, value in llama_config.items():
                print(f"  {key}: {value}")
        
        # Load state dict with proper key remapping and tensor permutation
        state_dict = gguf_sd_loader(llama_file, architecture="llama")
        print(f"Loaded Llama with {len(state_dict)} tensors")
        
        # Load tokenizer
        tokenizer = gguf_tokenizer_loader(llama_file)
        if tokenizer:
            print(f"Tokenizer vocab size: {tokenizer['vocab_size']}")
        
        # Show some attention weights (should be properly permuted)
        for name, tensor in state_dict.items():
            if "attention.wq.weight" in name:
                print(f"  {name}: {tensor.shape} (permuted for attention)")
                break
    
    except Exception as e:
        print(f"Error loading Llama model: {e}")

def example_clip_model():
    """Example: Loading a CLIP model."""
    print("\n=== CLIP Model Example ===")
    
    clip_file = "models/clip/clip_vit_large.gguf"
    
    if not Path(clip_file).exists():
        print(f"CLIP model {clip_file} not found. Skipping CLIP example.")
        return
    
    try:
        # Load CLIP model with specialized loader
        clip_data = gguf_clip_loader(clip_file)
        
        if clip_data:
            config = clip_data.pop("_clip_config", {})
            print(f"Loaded CLIP with {len(clip_data)} tensors")
            
            if config:
                print("CLIP configuration:")
                for key, value in config.items():
                    print(f"  {key}: {value}")
            
            # Show text and vision model components
            text_tensors = [name for name in clip_data.keys() if "text_model" in name]
            vision_tensors = [name for name in clip_data.keys() if "vision_model" in name]
            
            print(f"Text model tensors: {len(text_tensors)}")
            print(f"Vision model tensors: {len(vision_tensors)}")
    
    except Exception as e:
        print(f"Error loading CLIP model: {e}")

def example_comfyui_integration():
    """Example: Using ComfyUI integration."""
    print("\n=== ComfyUI Integration Example ===")
    
    try:
        # Initialize ComfyUI GGUF loader
        loader = ComfyUIGGUFLoader(models_dir="models")
        
        # Find all available GGUF models
        available_models = loader.find_gguf_files()
        print("Available GGUF files:")
        for category, files in available_models.items():
            print(f"  {category}: {len(files)} files")
        
        # Get detailed information about available models
        model_info = loader.list_available_models()
        for category, models in model_info.items():
            if models:
                print(f"\n{category} models:")
                for model in models[:2]:  # Show first 2 models in each category
                    print(f"  - {model['file_name']}")
                    print(f"    Architecture: {model.get('architecture', 'unknown')}")
                    print(f"    Size: {model.get('file_size_mb', 0):.1f} MB")
                    print(f"    Tensors: {model.get('tensor_count', 0)}")
        
        # Example of loading a model for ComfyUI
        if available_models:
            category, files = next(iter(available_models.items()))
            if files:
                example_file = files[0]
                print(f"\nLoading example model: {example_file.name}")
                
                model_data = loader.load_model_for_comfyui(
                    example_file, 
                    model_type="auto",
                    device="cpu",  # Use CPU for example
                    dtype=torch.float16
                )
                
                print(f"Loaded model type: {model_data['model_type']}")
                print(f"Architecture: {model_data['architecture']}")
                print(f"Device: {model_data['device']}")
                print(f"State dict tensors: {len(model_data['state_dict'])}")
        
        # Show memory usage
        usage = loader.get_memory_usage()
        print(f"\nMemory usage:")
        print(f"  Loaded models: {usage['loaded_models']}")
        print(f"  Estimated memory: {usage['estimated_memory_mb']:.1f} MB")
    
    except Exception as e:
        print(f"Error in ComfyUI integration: {e}")

def example_quantization_analysis():
    """Example: Analyzing quantization in GGUF files."""
    print("\n=== Quantization Analysis Example ===")
    
    # This example shows how to analyze quantization types in GGUF files
    models_dir = Path("models")
    gguf_files = list(models_dir.glob("**/*.gguf"))
    
    if not gguf_files:
        print("No GGUF files found for quantization analysis.")
        return
    
    print("Quantization analysis:")
    
    for gguf_file in gguf_files[:5]:  # Analyze first 5 files
        try:
            info = get_gguf_info(gguf_file)
            quant_types = info.get("quantization_types", [])
            
            print(f"\n{gguf_file.name}:")
            print(f"  Size: {info.get('file_size', 0) / 1024 / 1024:.1f} MB")
            print(f"  Tensors: {info.get('tensor_count', 0)}")
            print(f"  Quantization: {', '.join(quant_types)}")
            
            # Calculate compression ratio estimate
            if "F32" in quant_types and any(q.startswith("Q") for q in quant_types):
                print("  Mixed precision model (F32 + quantized)")
            elif any(q.startswith("Q") for q in quant_types):
                if "Q4" in ' '.join(quant_types):
                    print("  High compression (~75% size reduction)")
                elif "Q8" in ' '.join(quant_types):
                    print("  Medium compression (~50% size reduction)")
            elif "F16" in quant_types:
                print("  Half precision (~50% size reduction)")
        
        except Exception as e:
            print(f"Error analyzing {gguf_file.name}: {e}")

def main():
    """Run all examples."""
    print("GGUF Loader Usage Examples")
    print("=" * 50)
    
    # Run all examples
    example_basic_loading()
    example_flux_model()
    example_t5_text_encoder()
    example_llama_model()
    example_clip_model()
    example_comfyui_integration()
    example_quantization_analysis()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNote: Some examples may be skipped if the corresponding")
    print("GGUF model files are not found in the models directory.")

if __name__ == "__main__":
    main()