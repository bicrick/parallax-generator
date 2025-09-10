"""
Model Manager for Parallax Generator
Handles model downloading, caching, and loading with HuggingFace integration.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
from transformers import pipeline
import mlx.core as mx
import torch
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelManager:
    """Manages model downloading, caching, and loading for parallax generation."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize ModelManager with caching directory.
        
        Args:
            cache_dir: Custom cache directory. If None, uses HF default cache.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.models = {}
        
        # Get HuggingFace token from environment
        self.hf_token = os.getenv('HF_TOKEN')
        if self.hf_token:
            logger.info("🔐 HuggingFace token loaded from environment")
        else:
            logger.warning("⚠️  No HF_TOKEN found in environment - some models may not be accessible")
        self.model_configs = {
            # Ultra low-memory SD 1.5 models (most stable for M3 MacBook Pro)
            "sd15_base_fp16": {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "type": "text2img", 
                "description": "SD 1.5 Base (Ultra Low Memory - FP16)",
                "memory_usage": "~2GB",
                "recommended": True,
                "force_fp16": True,
                "ultra_stable": True
            },
            "sd15_inpaint_fp16": {
                "model_id": "runwayml/stable-diffusion-inpainting",
                "type": "inpainting",
                "description": "SD 1.5 Inpainting (Ultra Low Memory - FP16)",
                "memory_usage": "~2GB",
                "recommended": True,
                "force_fp16": True,
                "ultra_stable": True
            },
            # Standard SD 1.5 models (backup options)
            "sd15_inpaint": {
                "model_id": "runwayml/stable-diffusion-inpainting",
                "type": "inpainting",
                "description": "SD 1.5 Inpainting (Standard)",
                "memory_usage": "~4GB",
                "recommended": False
            },
            "sd15_base": {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "type": "text2img", 
                "description": "SD 1.5 Base (Standard)",
                "memory_usage": "~4GB",
                "recommended": False
            },
            # SDXL models (not recommended for memory-constrained systems)
            "sdxl_inpaint": {
                "model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                "type": "inpainting",
                "description": "SDXL Inpainting (High Memory - Not Recommended)",
                "memory_usage": "~12GB",
                "recommended": False
            },
            "sdxl_base": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "type": "text2img",
                "description": "SDXL Base (High Memory - Not Recommended)",
                "memory_usage": "~10GB",
                "recommended": False
            }
        }
        
        # Setup cache directory
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(self.cache_dir)
        
        logger.info(f"ModelManager initialized with cache: {self.get_cache_path()}")
    
    def get_cache_path(self) -> Path:
        """Get the active HuggingFace cache directory."""
        if self.cache_dir:
            return self.cache_dir
        
        # Use HF default cache location
        hf_home = os.environ.get("HF_HOME")
        if hf_home:
            return Path(hf_home)
        
        return Path.home() / ".cache" / "huggingface"
    
    def is_model_cached(self, model_name: str) -> bool:
        """
        Check if model is already downloaded and cached.
        
        Args:
            model_name: Name of the model (key in model_configs)
            
        Returns:
            True if model is cached, False otherwise
        """
        if model_name not in self.model_configs:
            logger.warning(f"Unknown model: {model_name}")
            return False
        
        model_id = self.model_configs[model_name]["model_id"]
        cache_path = self.get_cache_path()
        
        # Check if model directory exists in cache
        model_cache_dir = cache_path / "hub" / f"models--{model_id.replace('/', '--')}"
        
        if model_cache_dir.exists():
            # Check if model has required files
            snapshot_dir = model_cache_dir / "snapshots"
            if snapshot_dir.exists() and any(snapshot_dir.iterdir()):
                logger.info(f"✅ Model {model_name} found in cache: {model_cache_dir}")
                return True
        
        logger.info(f"❌ Model {model_name} not found in cache")
        return False
    
    def load_model(self, model_name: str, use_mlx: bool = True) -> object:
        """
        Load model with caching. Downloads only if not cached.
        
        Args:
            model_name: Name of the model to load
            use_mlx: Whether to use MLX optimization (for Apple Silicon)
            
        Returns:
            Loaded model pipeline
        """
        if model_name in self.models:
            logger.info(f"♻️  Using cached model: {model_name}")
            return self.models[model_name]
        
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.model_configs.keys())}")
        
        config = self.model_configs[model_name]
        model_id = config["model_id"]
        model_type = config["type"]
        
        logger.info(f"🔄 Loading {config['description']} ({model_id})")
        
        # Check if already cached
        if self.is_model_cached(model_name):
            logger.info(f"📦 Using cached model, no download needed")
        else:
            logger.info(f"⬇️  Downloading model (first time only)...")
        
        try:
            # Get model config for memory optimization
            model_config = self.model_configs[model_name]
            force_fp16 = model_config.get("force_fp16", False)
            
            # Prepare common arguments with ultra memory optimization
            common_args = {
                "cache_dir": self.cache_dir,
                "torch_dtype": torch.float32,  # Use FP32 for stability - FP16 causes blank images on MPS
                "safety_checker": None,
                "requires_safety_checker": False,
                "use_safetensors": False,  # Disable for compatibility with older cached models
                "low_cpu_mem_usage": True,  # Reduce CPU memory during loading
                # "variant": "fp16" if (force_fp16 and torch.backends.mps.is_available()) else None,  # Disabled - causes blank images
            }
            
            # Add token if available
            if self.hf_token:
                common_args["token"] = self.hf_token
                logger.info("🔐 Using HuggingFace token for authentication")
            
            # Memory optimization logging
            memory_usage = model_config.get("memory_usage", "Unknown")
            logger.info(f"💾 Expected memory usage: {memory_usage}")
            
            # Load appropriate pipeline based on type
            if model_type == "inpainting":
                pipeline = StableDiffusionInpaintPipeline.from_pretrained(
                    model_id,
                    **common_args
                )
            elif model_type == "text2img":
                pipeline = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    **common_args
                )
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Apply ultra memory optimizations first
            pipeline.enable_attention_slicing()  # Always enable for memory efficiency
            pipeline.enable_vae_slicing()  # Slice VAE for lower memory
            logger.info("🧠 Enabled attention and VAE slicing for maximum memory efficiency")
            
            # Move to appropriate device with memory optimization
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")
                logger.info("🚀 Using CUDA acceleration")
            elif torch.backends.mps.is_available() and use_mlx:
                pipeline = pipeline.to("mps")
                logger.info("🍎 Using MPS (Apple Silicon) acceleration")
            else:
                logger.info("💻 Using CPU (slower)")
                # Enable additional CPU memory optimizations
                pipeline.enable_sequential_cpu_offload()
                logger.info("🧠 Enabled CPU memory optimizations")
            
            # Additional memory optimizations for ultra-stable mode
            if model_config.get("ultra_stable", False):
                # Set lower guidance scale and steps for memory efficiency
                logger.info("🛡️  Ultra-stable mode: Applying conservative settings")
            
            # Cache the loaded model
            self.models[model_name] = pipeline
            logger.info(f"✅ Model {model_name} loaded successfully")
            
            return pipeline
            
        except Exception as e:
            logger.error(f"❌ Failed to load model {model_name}: {str(e)}")
            raise
    
    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model."""
        if model_name not in self.model_configs:
            return {}
        
        config = self.model_configs[model_name].copy()
        config["cached"] = self.is_model_cached(model_name)
        config["loaded"] = model_name in self.models
        
        return config
    
    def list_models(self) -> Dict[str, Dict]:
        """List all available models with their status."""
        return {name: self.get_model_info(name) for name in self.model_configs.keys()}
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear model cache."""
        if model_name:
            if model_name in self.models:
                del self.models[model_name]
                logger.info(f"🗑️  Cleared {model_name} from memory")
        else:
            self.models.clear()
            logger.info("🗑️  Cleared all models from memory")
    
    def get_cache_size(self) -> str:
        """Get human-readable cache size."""
        cache_path = self.get_cache_path()
        if not cache_path.exists():
            return "0 MB"
        
        total_size = sum(f.stat().st_size for f in cache_path.rglob('*') if f.is_file())
        
        # Convert to human readable
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024.0:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024.0
        return f"{total_size:.1f} TB"


def main():
    """Demo the ModelManager functionality."""
    print("🚀 Parallax Generator - Model Manager Demo")
    print("=" * 50)
    
    # Initialize model manager
    manager = ModelManager()
    
    # Show cache info
    print(f"📁 Cache directory: {manager.get_cache_path()}")
    print(f"💾 Cache size: {manager.get_cache_size()}")
    print()
    
    # List available models
    print("📋 Available Models:")
    models = manager.list_models()
    for name, info in models.items():
        status = "✅ Cached" if info["cached"] else "❌ Not cached"
        loaded = "🔄 Loaded" if info["loaded"] else "💤 Not loaded"
        print(f"  {name}: {info['description']} - {status} - {loaded}")
    
    print()
    print("💡 To load a model: manager.load_model('sd15_inpaint')")
    print("💡 Models are downloaded once and cached automatically!")


if __name__ == "__main__":
    main()
