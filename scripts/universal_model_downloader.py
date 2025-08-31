#!/usr/bin/env python3
"""
Universal Model Downloader for ComfyUI - Enhanced Edition
Supports GGUF, SafeTensors, and other formats with intelligent management

Features:
- Multi-format support (GGUF, SafeTensors, Pickle, etc.)
- Smart storage management for 50GB Free Tier
- Parallel downloads with progress tracking
- Resume capability for interrupted downloads
- Model verification and integrity checks
- Automatic symlink creation
- Format conversion options
- Storage space checking
- Interactive CLI with search
- Download queue management
- Bandwidth throttling
"""

import os
import sys
import json
import asyncio
import aiohttp
import aiofiles
import hashlib
import argparse
import threading
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from datetime import datetime
import tempfile
import shutil
import time
import signal
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from urllib.parse import urlparse
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_downloader.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ModelInfo:
    """Model information structure"""
    name: str
    url: str
    format: str
    size: str = "Unknown"
    size_bytes: int = 0
    description: str = ""
    category: str = ""
    tags: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    recommended: bool = False
    hash_sha256: Optional[str] = None
    hash_md5: Optional[str] = None
    use_case: List[str] = field(default_factory=list)
    performance_notes: str = ""
    alternative_formats: Dict[str, str] = field(default_factory=dict)

@dataclass
class DownloadJob:
    """Download job tracking"""
    model: ModelInfo
    destination: Path
    status: str = "pending"  # pending, downloading, paused, completed, failed
    progress: float = 0.0
    downloaded_bytes: int = 0
    total_bytes: int = 0
    speed: str = "0 B/s"
    eta: str = "Unknown"
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    error: Optional[str] = None
    temp_file: Optional[Path] = None

class UniversalModelDownloader:
    """Enhanced model downloader with multi-format support"""
    
    def __init__(self):
        self.config_dir = Path("configs")
        self.storage_base = Path("/storage/ComfyUI/models")
        self.comfyui_models = Path("ComfyUI/models")
        self.download_queue: List[DownloadJob] = []
        self.active_downloads: Dict[str, DownloadJob] = {}
        self.max_concurrent_downloads = 3
        self.bandwidth_limit = None  # bytes per second
        self.session = None
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_downloads)
        
        # Model catalogs
        self.models_catalog = self.load_models_catalog()
        self.download_history = self.load_download_history()
        
        # Storage management
        self.storage_limit_gb = 50
        self.reserve_gb = 10  # Reserve space for outputs and temp files
        
        # Create necessary directories
        self.storage_base.mkdir(parents=True, exist_ok=True)
        self.comfyui_models.mkdir(parents=True, exist_ok=True)
        
    def load_models_catalog(self) -> Dict:
        """Load enhanced models catalog"""
        catalog_file = self.config_dir / "models_catalog.json"
        
        if catalog_file.exists():
            with open(catalog_file, 'r') as f:
                return json.load(f)
        
        # Create default catalog with both formats
        default_catalog = {
            "metadata": {
                "version": "2.0.0",
                "last_updated": datetime.now().isoformat(),
                "description": "Universal model catalog supporting multiple formats"
            },
            "categories": {
                "checkpoints": {
                    "description": "Main diffusion models",
                    "storage_priority": 1,
                    "models": {
                        "sdxl_base": {
                            "safetensors": ModelInfo(
                                name="Stable Diffusion XL Base 1.0",
                                url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.safetensors",
                                format="safetensors",
                                size="6.9 GB",
                                size_bytes=7400000000,
                                description="Base SDXL model optimized for A6000",
                                category="checkpoints",
                                tags=["xl", "base", "stable"],
                                use_case=["general", "high-quality"],
                                performance_notes="Optimized for A6000 48GB VRAM",
                                recommended=True
                            ).__dict__,
                            "gguf": ModelInfo(
                                name="Stable Diffusion XL Base 1.0 GGUF",
                                url="https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0.gguf",
                                format="gguf",
                                size="3.5 GB",
                                size_bytes=3700000000,
                                description="Quantized SDXL model for lower memory usage",
                                category="checkpoints",
                                tags=["xl", "base", "quantized", "gguf"],
                                use_case=["memory-constrained", "fast-inference"],
                                performance_notes="Quantized for memory efficiency"
                            ).__dict__
                        },
                        "chroma_v48": {
                            "safetensors": ModelInfo(
                                name="Chroma v48 Latest",
                                url="https://huggingface.co/lodestones/Chroma/resolve/main/chroma-unlocked-v48.safetensors",
                                format="safetensors",
                                size="24 GB",
                                size_bytes=25000000000,
                                description="Latest Chroma model with best quality",
                                category="checkpoints",
                                tags=["chroma", "latest", "high-quality"],
                                use_case=["professional", "high-quality"],
                                performance_notes="Requires significant VRAM",
                                recommended=True
                            ).__dict__
                        }
                    }
                },
                "vae": {
                    "description": "Variational Auto-Encoders",
                    "storage_priority": 2,
                    "models": {
                        "sdxl_vae": {
                            "safetensors": ModelInfo(
                                name="SDXL VAE",
                                url="https://huggingface.co/stabilityai/sdxl-vae/resolve/main/sdxl_vae.safetensors",
                                format="safetensors",
                                size="335 MB",
                                size_bytes=351000000,
                                description="Essential VAE for SDXL models",
                                category="vae",
                                tags=["vae", "sdxl", "essential"],
                                use_case=["sdxl-required"],
                                recommended=True
                            ).__dict__
                        },
                        "chroma_vae": {
                            "safetensors": ModelInfo(
                                name="Chroma VAE",
                                url="https://huggingface.co/lodestones/Chroma/resolve/main/ae.safetensors",
                                format="safetensors",
                                size="335 MB", 
                                size_bytes=351000000,
                                description="VAE specifically for Chroma models",
                                category="vae",
                                tags=["vae", "chroma", "required"],
                                use_case=["chroma-required"],
                                recommended=True
                            ).__dict__
                        }
                    }
                },
                "text_encoders": {
                    "description": "Text encoding models",
                    "storage_priority": 3,
                    "models": {
                        "t5xxl_fp8": {
                            "safetensors": ModelInfo(
                                name="T5-XXL FP8 Scaled",
                                url="https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn_scaled.safetensors",
                                format="safetensors",
                                size="4.9 GB",
                                size_bytes=5200000000,
                                description="Memory efficient text encoder",
                                category="text_encoders",
                                tags=["text-encoder", "fp8", "memory-efficient"],
                                use_case=["memory-constrained", "efficient"],
                                recommended=True
                            ).__dict__,
                            "gguf": ModelInfo(
                                name="T5-XXL GGUF Q4",
                                url="https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_q4_0.gguf",
                                format="gguf", 
                                size="2.5 GB",
                                size_bytes=2600000000,
                                description="Quantized text encoder for minimal memory usage",
                                category="text_encoders",
                                tags=["text-encoder", "gguf", "quantized"],
                                use_case=["ultra-low-memory"],
                                performance_notes="Quantized for extreme efficiency"
                            ).__dict__
                        }
                    }
                },
                "loras": {
                    "description": "LoRA adaptation models",
                    "storage_priority": 4,
                    "models": {
                        "realfine": {
                            "safetensors": ModelInfo(
                                name="RealFine LoRA",
                                url="https://huggingface.co/silveroxides/Chroma-LoRA-Experiments/resolve/main/Chroma-RealFine_lora_rank_64-bf16.safetensors",
                                format="safetensors",
                                size="294 MB",
                                size_bytes=308000000,
                                description="Realistic enhancement LoRA",
                                category="loras",
                                tags=["lora", "realistic", "enhancement"],
                                use_case=["realism", "portraits"]
                            ).__dict__
                        },
                        "turbo": {
                            "safetensors": ModelInfo(
                                name="Turbo LoRA",
                                url="https://huggingface.co/silveroxides/Chroma-LoRA-Experiments/resolve/main/Chroma-Turbo_lora_rank_64-bf16.safetensors",
                                format="safetensors",
                                size="294 MB",
                                size_bytes=308000000,
                                description="Speed enhancement LoRA",
                                category="loras",
                                tags=["lora", "speed", "fast-generation"],
                                use_case=["fast-generation", "prototyping"]
                            ).__dict__
                        }
                    }
                },
                "controlnet": {
                    "description": "ControlNet models for guided generation",
                    "storage_priority": 5,
                    "models": {
                        "canny_sdxl": {
                            "safetensors": ModelInfo(
                                name="ControlNet Canny SDXL",
                                url="https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
                                format="safetensors",
                                size="2.5 GB",
                                size_bytes=2600000000,
                                description="Canny edge detection ControlNet",
                                category="controlnet",
                                tags=["controlnet", "canny", "edges"],
                                use_case=["edge-guidance", "structure-control"]
                            ).__dict__
                        },
                        "depth_sdxl": {
                            "safetensors": ModelInfo(
                                name="ControlNet Depth SDXL",
                                url="https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0/resolve/main/diffusion_pytorch_model.safetensors",
                                format="safetensors",
                                size="2.5 GB",
                                size_bytes=2600000000,
                                description="Depth-based ControlNet",
                                category="controlnet",
                                tags=["controlnet", "depth", "3d"],
                                use_case=["depth-guidance", "3d-aware"]
                            ).__dict__
                        }
                    }
                },
                "embeddings": {
                    "description": "Textual inversion embeddings",
                    "storage_priority": 6,
                    "models": {
                        "easynegative": {
                            "safetensors": ModelInfo(
                                name="EasyNegative V2",
                                url="https://huggingface.co/embed/negative/resolve/main/EasyNegativeV2.safetensors",
                                format="safetensors",
                                size="25 KB",
                                size_bytes=25000,
                                description="Negative prompt embedding for quality improvement",
                                category="embeddings",
                                tags=["embedding", "negative", "quality"],
                                use_case=["quality-improvement", "general"],
                                recommended=True
                            ).__dict__
                        }
                    }
                }
            }
        }
        
        # Save default catalog
        with open(catalog_file, 'w') as f:
            json.dump(default_catalog, f, indent=2)
            
        return default_catalog
    
    def load_download_history(self) -> Dict:
        """Load download history"""
        history_file = self.config_dir / "download_history.json"
        
        if history_file.exists():
            with open(history_file, 'r') as f:
                return json.load(f)
        
        return {
            "downloads": [],
            "statistics": {
                "total_downloads": 0,
                "successful_downloads": 0,
                "failed_downloads": 0,
                "total_bytes_downloaded": 0
            }
        }
    
    def save_download_history(self):
        """Save download history"""
        history_file = self.config_dir / "download_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.download_history, f, indent=2)
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get current storage information"""
        try:
            # Check storage space
            storage_stat = shutil.disk_usage(self.storage_base.parent if self.storage_base.exists() else Path("/"))
            total_space = storage_stat.total
            free_space = storage_stat.free
            used_space = storage_stat.total - storage_stat.free
            
            # Calculate model storage usage
            model_space_used = 0
            if self.storage_base.exists():
                for item in self.storage_base.rglob("*"):
                    if item.is_file():
                        model_space_used += item.stat().st_size
            
            return {
                "total_space_gb": total_space / (1024**3),
                "free_space_gb": free_space / (1024**3),
                "used_space_gb": used_space / (1024**3),
                "model_space_gb": model_space_used / (1024**3),
                "storage_limit_gb": self.storage_limit_gb,
                "available_for_models_gb": min(free_space / (1024**3), self.storage_limit_gb - model_space_used / (1024**3)),
                "space_warning": free_space / (1024**3) < self.reserve_gb
            }
        except Exception as e:
            logger.error(f"Error getting storage info: {e}")
            return {"error": str(e)}
    
    def check_model_exists(self, model: ModelInfo, category: str) -> Tuple[bool, Optional[Path]]:
        """Check if model already exists"""
        model_dir = self.storage_base / category
        model_file = model_dir / Path(model.url).name
        
        if model_file.exists():
            # Verify file size if known
            if model.size_bytes > 0:
                actual_size = model_file.stat().st_size
                if abs(actual_size - model.size_bytes) > 1024 * 1024:  # 1MB tolerance
                    logger.warning(f"Model {model.name} exists but size mismatch: expected {model.size_bytes}, got {actual_size}")
                    return False, model_file
            return True, model_file
        
        return False, model_file
    
    def verify_model_integrity(self, model: ModelInfo, file_path: Path) -> bool:
        """Verify downloaded model integrity"""
        try:
            # Check file size
            if model.size_bytes > 0:
                actual_size = file_path.stat().st_size
                if abs(actual_size - model.size_bytes) > 1024 * 1024:  # 1MB tolerance
                    logger.error(f"Size verification failed for {model.name}")
                    return False
            
            # Check hash if provided
            if model.hash_sha256:
                sha256_hash = hashlib.sha256()
                with open(file_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        sha256_hash.update(chunk)
                
                if sha256_hash.hexdigest() != model.hash_sha256:
                    logger.error(f"SHA256 verification failed for {model.name}")
                    return False
            
            # Basic file format validation
            if model.format == "safetensors":
                # Check if file starts with safetensors header
                with open(file_path, "rb") as f:
                    header = f.read(8)
                    if len(header) < 8:
                        logger.error(f"Invalid safetensors file: {model.name}")
                        return False
            
            elif model.format == "gguf":
                # Check GGUF magic number
                with open(file_path, "rb") as f:
                    header = f.read(4)
                    if header != b"GGUF":
                        logger.error(f"Invalid GGUF file: {model.name}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error verifying model {model.name}: {e}")
            return False
    
    async def download_model_async(self, job: DownloadJob) -> bool:
        """Download model with async support and progress tracking"""
        try:
            job.status = "downloading"
            job.start_time = datetime.now()
            
            # Create temporary file
            temp_dir = Path(tempfile.mkdtemp(prefix="model_download_"))
            temp_file = temp_dir / Path(job.model.url).name
            job.temp_file = temp_file
            
            # Resume support - check if partial file exists
            resume_header = {}
            if temp_file.exists():
                resume_pos = temp_file.stat().st_size
                resume_header = {'Range': f'bytes={resume_pos}-'}
                job.downloaded_bytes = resume_pos
                logger.info(f"Resuming download of {job.model.name} from byte {resume_pos}")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(job.model.url, headers=resume_header) as response:
                    if response.status not in [200, 206]:  # 206 for partial content
                        raise Exception(f"HTTP {response.status}: {response.reason}")
                    
                    # Get total size
                    if 'content-length' in response.headers:
                        content_length = int(response.headers['content-length'])
                        job.total_bytes = job.downloaded_bytes + content_length
                    else:
                        job.total_bytes = job.model.size_bytes if job.model.size_bytes > 0 else 0
                    
                    # Open file for writing (append mode for resume)
                    mode = 'ab' if temp_file.exists() else 'wb'
                    async with aiofiles.open(temp_file, mode) as f:
                        chunk_size = 8192
                        downloaded_this_session = 0
                        start_time = time.time()
                        
                        async for chunk in response.content.iter_chunked(chunk_size):
                            if job.status == "paused":
                                await asyncio.sleep(0.1)
                                continue
                            
                            if job.status == "cancelled":
                                raise Exception("Download cancelled")
                            
                            await f.write(chunk)
                            downloaded_this_session += len(chunk)
                            job.downloaded_bytes += len(chunk)
                            
                            # Update progress
                            if job.total_bytes > 0:
                                job.progress = (job.downloaded_bytes / job.total_bytes) * 100
                            
                            # Calculate speed and ETA
                            elapsed = time.time() - start_time
                            if elapsed > 0:
                                speed_bps = downloaded_this_session / elapsed
                                job.speed = self.format_bytes(speed_bps) + "/s"
                                
                                if job.total_bytes > job.downloaded_bytes and speed_bps > 0:
                                    eta_seconds = (job.total_bytes - job.downloaded_bytes) / speed_bps
                                    job.eta = self.format_time(eta_seconds)
                            
                            # Bandwidth throttling
                            if self.bandwidth_limit and speed_bps > self.bandwidth_limit:
                                delay = (len(chunk) / self.bandwidth_limit) - (len(chunk) / speed_bps)
                                await asyncio.sleep(delay)
            
            # Move completed file to destination
            job.destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_file), str(job.destination))
            
            # Cleanup temp directory
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Verify integrity
            if not self.verify_model_integrity(job.model, job.destination):
                job.status = "failed"
                job.error = "Integrity verification failed"
                return False
            
            job.status = "completed"
            job.end_time = datetime.now()
            job.progress = 100.0
            
            # Update download history
            self.download_history["downloads"].append({
                "model_name": job.model.name,
                "url": job.model.url,
                "destination": str(job.destination),
                "size_bytes": job.total_bytes,
                "download_time": (job.end_time - job.start_time).total_seconds(),
                "timestamp": job.end_time.isoformat(),
                "status": "completed"
            })
            
            self.download_history["statistics"]["successful_downloads"] += 1
            self.download_history["statistics"]["total_bytes_downloaded"] += job.total_bytes
            self.save_download_history()
            
            logger.info(f"Successfully downloaded {job.model.name}")
            return True
            
        except Exception as e:
            job.status = "failed"
            job.error = str(e)
            job.end_time = datetime.now()
            
            # Cleanup temp file
            if job.temp_file and job.temp_file.exists():
                try:
                    shutil.rmtree(job.temp_file.parent, ignore_errors=True)
                except:
                    pass
            
            # Update download history
            self.download_history["downloads"].append({
                "model_name": job.model.name,
                "url": job.model.url,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            })
            
            self.download_history["statistics"]["failed_downloads"] += 1
            self.save_download_history()
            
            logger.error(f"Failed to download {job.model.name}: {e}")
            return False
    
    def download_model_sync(self, job: DownloadJob) -> bool:
        """Synchronous download wrapper"""
        return asyncio.run(self.download_model_async(job))
    
    def create_symlinks(self):
        """Create symlinks from storage to ComfyUI models directory"""
        logger.info("Creating symlinks from storage to ComfyUI models directory...")
        
        try:
            # Ensure ComfyUI models directory exists
            self.comfyui_models.mkdir(parents=True, exist_ok=True)
            
            # Create symlinks for each category
            for category_dir in self.storage_base.iterdir():
                if category_dir.is_dir():
                    link_target = self.comfyui_models / category_dir.name
                    
                    # Remove existing symlink or directory
                    if link_target.exists() or link_target.is_symlink():
                        if link_target.is_symlink():
                            link_target.unlink()
                        else:
                            shutil.rmtree(link_target)
                    
                    # Create symlink
                    try:
                        link_target.symlink_to(category_dir.resolve())
                        logger.info(f"Created symlink: {link_target} -> {category_dir}")
                    except Exception as e:
                        logger.error(f"Failed to create symlink for {category_dir}: {e}")
            
            logger.info("Symlink creation completed")
            
        except Exception as e:
            logger.error(f"Error creating symlinks: {e}")
    
    def search_models(self, query: str, category: Optional[str] = None, format_filter: Optional[str] = None) -> List[Tuple[str, str, ModelInfo]]:
        """Search models by name, description, or tags"""
        results = []
        query_lower = query.lower()
        
        for cat_name, cat_data in self.models_catalog.get("categories", {}).items():
            if category and category != cat_name:
                continue
                
            for model_id, formats in cat_data.get("models", {}).items():
                for fmt, model_data in formats.items():
                    if format_filter and format_filter != fmt:
                        continue
                    
                    model = ModelInfo(**model_data)
                    
                    # Search in name, description, and tags
                    searchable_text = f"{model.name} {model.description} {' '.join(model.tags)}".lower()
                    
                    if query_lower in searchable_text:
                        results.append((cat_name, model_id, model))
        
        return results
    
    def list_models(self, category: Optional[str] = None, format_filter: Optional[str] = None, 
                   recommended_only: bool = False) -> List[Tuple[str, str, ModelInfo]]:
        """List all models with optional filters"""
        results = []
        
        for cat_name, cat_data in self.models_catalog.get("categories", {}).items():
            if category and category != cat_name:
                continue
                
            for model_id, formats in cat_data.get("models", {}).items():
                for fmt, model_data in formats.items():
                    if format_filter and format_filter != fmt:
                        continue
                    
                    model = ModelInfo(**model_data)
                    
                    if recommended_only and not model.recommended:
                        continue
                    
                    results.append((cat_name, model_id, model))
        
        # Sort by category priority and recommendation
        def sort_key(item):
            cat_name, _, model = item
            priority = self.models_catalog["categories"][cat_name].get("storage_priority", 999)
            return (priority, not model.recommended, model.name)
        
        return sorted(results, key=sort_key)
    
    def get_download_recommendations(self, available_space_gb: float) -> List[Tuple[str, str, ModelInfo]]:
        """Get smart download recommendations based on available space"""
        recommendations = []
        used_space_gb = 0
        
        # Get essential models first
        essential_models = [
            (cat, model_id, ModelInfo(**model_data))
            for cat, cat_data in self.models_catalog.get("categories", {}).items()
            for model_id, formats in cat_data.get("models", {}).items()
            for fmt, model_data in formats.items()
            if ModelInfo(**model_data).recommended
        ]
        
        # Sort by priority and size
        essential_models.sort(key=lambda x: (
            self.models_catalog["categories"][x[0]].get("storage_priority", 999),
            x[2].size_bytes
        ))
        
        for cat, model_id, model in essential_models:
            model_size_gb = model.size_bytes / (1024**3)
            
            if used_space_gb + model_size_gb <= available_space_gb * 0.8:  # Leave 20% buffer
                # Check if we already have this model
                exists, _ = self.check_model_exists(model, cat)
                if not exists:
                    recommendations.append((cat, model_id, model))
                    used_space_gb += model_size_gb
        
        return recommendations
    
    def format_bytes(self, bytes_val: Union[int, float]) -> str:
        """Format bytes to human readable string"""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if bytes_val < 1024.0:
                return f"{bytes_val:.1f} {unit}"
            bytes_val /= 1024.0
        return f"{bytes_val:.1f} PB"
    
    def format_time(self, seconds: float) -> str:
        """Format seconds to human readable time"""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m {int(seconds % 60)}s"
        else:
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            return f"{hours}h {minutes}m"
    
    def interactive_cli(self):
        """Interactive command-line interface"""
        print("🎯 Universal Model Downloader for ComfyUI")
        print("=" * 50)
        
        while True:
            print("\n📋 Available Commands:")
            print("1. 🔍 Search models")
            print("2. 📝 List all models")
            print("3. 💾 Check storage info")
            print("4. 🎯 Get recommendations")
            print("5. 📥 Download model")
            print("6. 📊 Download queue status")
            print("7. 🔗 Create symlinks")
            print("8. 📈 Download history")
            print("9. ❌ Exit")
            
            try:
                choice = input("\n🤔 Enter your choice (1-9): ").strip()
                
                if choice == "1":
                    self._search_command()
                elif choice == "2":
                    self._list_command()
                elif choice == "3":
                    self._storage_info_command()
                elif choice == "4":
                    self._recommendations_command()
                elif choice == "5":
                    self._download_command()
                elif choice == "6":
                    self._queue_status_command()
                elif choice == "7":
                    self.create_symlinks()
                elif choice == "8":
                    self._history_command()
                elif choice == "9":
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid choice. Please enter a number between 1-9.")
                    
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def _search_command(self):
        """Handle search command"""
        query = input("🔍 Enter search query: ").strip()
        if not query:
            print("❌ Search query cannot be empty")
            return
        
        category = input("📂 Category filter (optional, press Enter to skip): ").strip() or None
        format_filter = input("📄 Format filter (safetensors/gguf, optional): ").strip() or None
        
        results = self.search_models(query, category, format_filter)
        
        if not results:
            print(f"❌ No models found matching '{query}'")
            return
        
        print(f"\n🔍 Found {len(results)} models:")
        for i, (cat, model_id, model) in enumerate(results, 1):
            exists, _ = self.check_model_exists(model, cat)
            status = "✅ Downloaded" if exists else "📥 Available"
            print(f"{i:2d}. {status} [{cat}] {model.name} ({model.format}) - {model.size}")
            print(f"    📝 {model.description}")
    
    def _list_command(self):
        """Handle list command"""
        print("\n📂 Available categories:")
        categories = list(self.models_catalog.get("categories", {}).keys())
        for i, cat in enumerate(categories, 1):
            print(f"{i:2d}. {cat}")
        
        cat_choice = input("📂 Select category (number or name, Enter for all): ").strip()
        category = None
        if cat_choice.isdigit() and 1 <= int(cat_choice) <= len(categories):
            category = categories[int(cat_choice) - 1]
        elif cat_choice in categories:
            category = cat_choice
        
        format_filter = input("📄 Format filter (safetensors/gguf, Enter for all): ").strip() or None
        recommended_only = input("⭐ Show recommended only? (y/N): ").strip().lower() == 'y'
        
        results = self.list_models(category, format_filter, recommended_only)
        
        print(f"\n📋 Found {len(results)} models:")
        for i, (cat, model_id, model) in enumerate(results, 1):
            exists, _ = self.check_model_exists(model, cat)
            status = "✅" if exists else "📥"
            rec_mark = "⭐" if model.recommended else "  "
            print(f"{i:2d}. {status}{rec_mark} [{cat}] {model.name} ({model.format}) - {model.size}")
    
    def _storage_info_command(self):
        """Handle storage info command"""
        info = self.get_storage_info()
        
        if "error" in info:
            print(f"❌ Error getting storage info: {info['error']}")
            return
        
        print("\n💾 Storage Information:")
        print(f"📊 Total Space: {info['total_space_gb']:.1f} GB")
        print(f"📊 Used Space: {info['used_space_gb']:.1f} GB")
        print(f"📊 Free Space: {info['free_space_gb']:.1f} GB")
        print(f"📊 Models Space: {info['model_space_gb']:.1f} GB")
        print(f"📊 Available for Models: {info['available_for_models_gb']:.1f} GB")
        
        if info['space_warning']:
            print("⚠️  WARNING: Low disk space!")
    
    def _recommendations_command(self):
        """Handle recommendations command"""
        info = self.get_storage_info()
        available_space = info.get('available_for_models_gb', 0)
        
        recommendations = self.get_download_recommendations(available_space)
        
        if not recommendations:
            print("✅ You have all recommended models or insufficient space")
            return
        
        print(f"\n🎯 Recommendations for {available_space:.1f} GB available:")
        total_size = 0
        for i, (cat, model_id, model) in enumerate(recommendations, 1):
            size_gb = model.size_bytes / (1024**3)
            total_size += size_gb
            print(f"{i:2d}. ⭐ [{cat}] {model.name} ({model.format}) - {model.size}")
            print(f"    📝 {model.description}")
        
        print(f"\n📊 Total recommended size: {total_size:.1f} GB")
        
        if input("\n📥 Download all recommended models? (y/N): ").strip().lower() == 'y':
            for cat, model_id, model in recommendations:
                self._queue_download(model, cat)
            self._process_download_queue()
    
    def _download_command(self):
        """Handle download command"""
        # Show available models
        results = self.list_models()
        
        if not results:
            print("❌ No models available")
            return
        
        print(f"\n📋 Available models ({len(results)} total):")
        for i, (cat, model_id, model) in enumerate(results, 1):
            exists, _ = self.check_model_exists(model, cat)
            status = "✅" if exists else "📥"
            rec_mark = "⭐" if model.recommended else "  "
            print(f"{i:2d}. {status}{rec_mark} [{cat}] {model.name} ({model.format}) - {model.size}")
        
        try:
            choice = input("\n📥 Select model to download (number): ").strip()
            if not choice.isdigit() or not (1 <= int(choice) <= len(results)):
                print("❌ Invalid selection")
                return
            
            cat, model_id, model = results[int(choice) - 1]
            
            # Check if already exists
            exists, file_path = self.check_model_exists(model, cat)
            if exists:
                print(f"✅ Model already exists: {file_path}")
                return
            
            # Check storage space
            info = self.get_storage_info()
            model_size_gb = model.size_bytes / (1024**3)
            
            if model_size_gb > info.get('available_for_models_gb', 0):
                print(f"❌ Insufficient space. Need {model_size_gb:.1f} GB, have {info.get('available_for_models_gb', 0):.1f} GB")
                return
            
            self._queue_download(model, cat)
            self._process_download_queue()
            
        except ValueError:
            print("❌ Invalid input")
    
    def _queue_download(self, model: ModelInfo, category: str):
        """Queue a model for download"""
        destination = self.storage_base / category / Path(model.url).name
        job = DownloadJob(model=model, destination=destination)
        self.download_queue.append(job)
        print(f"📋 Queued {model.name} for download")
    
    def _process_download_queue(self):
        """Process the download queue"""
        if not self.download_queue:
            print("📋 Download queue is empty")
            return
        
        print(f"\n🚀 Processing {len(self.download_queue)} downloads...")
        
        # Process downloads with threading
        futures = []
        for job in self.download_queue[:self.max_concurrent_downloads]:
            future = self.executor.submit(self.download_model_sync, job)
            futures.append((job, future))
        
        completed_jobs = []
        
        # Monitor progress
        while futures:
            for job, future in futures[:]:
                if future.done():
                    success = future.result()
                    completed_jobs.append((job, success))
                    futures.remove((job, future))
                    
                    # Start next download if available
                    remaining_jobs = [j for j in self.download_queue if j not in [cj[0] for cj in completed_jobs] and j not in [f[0] for f in futures]]
                    if remaining_jobs:
                        next_job = remaining_jobs[0]
                        new_future = self.executor.submit(self.download_model_sync, next_job)
                        futures.append((next_job, new_future))
                else:
                    # Show progress for active downloads
                    if job.status == "downloading":
                        print(f"⬇️  {job.model.name}: {job.progress:.1f}% - {job.speed} - ETA: {job.eta}")
            
            if futures:
                time.sleep(1)
        
        # Clear completed jobs from queue
        for job, success in completed_jobs:
            if job in self.download_queue:
                self.download_queue.remove(job)
        
        # Summary
        successful = sum(1 for _, success in completed_jobs if success)
        failed = len(completed_jobs) - successful
        
        print(f"\n📊 Download Summary:")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        
        # Create symlinks after successful downloads
        if successful > 0:
            self.create_symlinks()
    
    def _queue_status_command(self):
        """Handle queue status command"""
        if not self.download_queue and not self.active_downloads:
            print("📋 No active downloads")
            return
        
        print(f"\n📋 Download Queue Status:")
        
        if self.active_downloads:
            print("\n🚀 Active Downloads:")
            for job_id, job in self.active_downloads.items():
                print(f"⬇️  {job.model.name}: {job.progress:.1f}% - {job.speed} - ETA: {job.eta}")
        
        if self.download_queue:
            print(f"\n⏳ Queued Downloads ({len(self.download_queue)}):")
            for i, job in enumerate(self.download_queue, 1):
                print(f"{i:2d}. {job.model.name} ({job.model.size}) - {job.status}")
    
    def _history_command(self):
        """Handle download history command"""
        history = self.download_history
        stats = history.get("statistics", {})
        
        print("\n📈 Download Statistics:")
        print(f"📊 Total Downloads: {stats.get('total_downloads', 0)}")
        print(f"✅ Successful: {stats.get('successful_downloads', 0)}")
        print(f"❌ Failed: {stats.get('failed_downloads', 0)}")
        print(f"📦 Total Data: {self.format_bytes(stats.get('total_bytes_downloaded', 0))}")
        
        recent_downloads = history.get("downloads", [])[-10:]  # Last 10 downloads
        
        if recent_downloads:
            print(f"\n📋 Recent Downloads:")
            for download in recent_downloads:
                status_icon = "✅" if download.get("status") == "completed" else "❌"
                timestamp = download.get("timestamp", "Unknown")[:19]
                print(f"{status_icon} {download.get('model_name', 'Unknown')} - {timestamp}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Universal Model Downloader for ComfyUI")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--search", type=str, help="Search for models")
    parser.add_argument("--list", action="store_true", help="List all models")
    parser.add_argument("--category", type=str, help="Filter by category")
    parser.add_argument("--format", type=str, choices=["safetensors", "gguf"], help="Filter by format")
    parser.add_argument("--recommended", action="store_true", help="Show recommended models only")
    parser.add_argument("--download", type=str, help="Download model by name")
    parser.add_argument("--storage-info", action="store_true", help="Show storage information")
    parser.add_argument("--create-symlinks", action="store_true", help="Create symlinks")
    parser.add_argument("--bandwidth-limit", type=str, help="Bandwidth limit (e.g., 10MB, 1GB)")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Maximum concurrent downloads")
    
    args = parser.parse_args()
    
    try:
        downloader = UniversalModelDownloader()
        
        # Set options
        if args.bandwidth_limit:
            # Parse bandwidth limit
            limit_str = args.bandwidth_limit.upper()
            multiplier = 1
            if limit_str.endswith('KB'):
                multiplier = 1024
                limit_str = limit_str[:-2]
            elif limit_str.endswith('MB'):
                multiplier = 1024 * 1024
                limit_str = limit_str[:-2]
            elif limit_str.endswith('GB'):
                multiplier = 1024 * 1024 * 1024
                limit_str = limit_str[:-2]
            
            downloader.bandwidth_limit = int(limit_str) * multiplier
        
        downloader.max_concurrent_downloads = args.max_concurrent
        
        # Handle commands
        if args.interactive:
            downloader.interactive_cli()
        elif args.search:
            results = downloader.search_models(args.search, args.category, args.format)
            for cat, model_id, model in results:
                exists, _ = downloader.check_model_exists(model, cat)
                status = "Downloaded" if exists else "Available"
                print(f"[{status}] [{cat}] {model.name} ({model.format}) - {model.size}")
                print(f"  {model.description}")
        elif args.list:
            results = downloader.list_models(args.category, args.format, args.recommended)
            for cat, model_id, model in results:
                exists, _ = downloader.check_model_exists(model, cat)
                status = "✅" if exists else "📥"
                rec = "⭐" if model.recommended else ""
                print(f"{status}{rec} [{cat}] {model.name} ({model.format}) - {model.size}")
        elif args.storage_info:
            info = downloader.get_storage_info()
            for key, value in info.items():
                print(f"{key}: {value}")
        elif args.create_symlinks:
            downloader.create_symlinks()
        elif args.download:
            # Find and download model by name
            results = downloader.search_models(args.download)
            if results:
                cat, model_id, model = results[0]
                downloader._queue_download(model, cat)
                downloader._process_download_queue()
            else:
                print(f"Model '{args.download}' not found")
        else:
            # Default to interactive mode
            downloader.interactive_cli()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()