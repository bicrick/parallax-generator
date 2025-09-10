"""
Model Manager for Parallax Generator - Google Colab Version
Handles model downloading, caching, and loading optimized for Google Colab environment.
"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple
import logging

from diffusers import StableDiffusionInpaintPipeline, StableDiffusionPipeline
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_hf_token():
    """Get HuggingFace token from Google Colab userdata."""
    try:
        from google.colab import userdata
        token = userdata.get('HF_TOKEN')
        logger.info("🔐 HuggingFace token loaded from Colab userdata")
        return token
    except ImportError:
        logger.warning("⚠️  Not running in Google Colab, falling back to environment variable")
        token = os.getenv('HF_TOKEN')
        if token:
            logger.info("🔐 HuggingFace token loaded from environment")
        else:
            logger.warning("⚠️  No HF_TOKEN found - some models may not be accessible")
        return token
    except Exception as e:
        logger.warning(f"⚠️  Could not get HF_TOKEN from userdata: {e}")
        return None


class ModelManagerColab:
    """Manages model downloading, caching, and loading for parallax generation in Google Colab."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize ModelManager with caching directory optimized for Colab.
        
        Args:
            cache_dir: Custom cache directory. If None, uses Colab's persistent storage.
        """
        # Use Colab's persistent storage by default
        if cache_dir is None:
            cache_dir = "/content/drive/MyDrive/parallax_models"
        
        self.cache_dir = Path(cache_dir)
        self.models = {}
        
        # Get HuggingFace token from Colab userdata
        self.hf_token = get_hf_token()
        
        # Colab-optimized model configurations
        self.model_configs = {
            # SD 1.5 models - optimized for Colab GPU memory
            "sd15_base": {
                "model_id": "runwayml/stable-diffusion-v1-5",
                "type": "text2img", 
                "description": "SD 1.5 Base (Colab Optimized)",
                "memory_usage": "~4GB VRAM",
                "recommended": True
            },
            "sd15_inpaint": {
                "model_id": "runwayml/stable-diffusion-inpainting",
                "type": "inpainting",
                "description": "SD 1.5 Inpainting (Colab Optimized)",
                "memory_usage": "~4GB VRAM",
                "recommended": True
            },
            # SDXL models - for high-end Colab instances
            "sdxl_base": {
                "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
                "type": "text2img",
                "description": "SDXL Base (High Memory)",
                "memory_usage": "~10GB VRAM",
                "recommended": False
            },
            "sdxl_inpaint": {
                "model_id": "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                "type": "inpainting",
                "description": "SDXL Inpainting (High Memory)",
                "memory_usage": "~12GB VRAM",
                "recommended": False
            }
        }
        
        # Setup cache directory in persistent storage
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set HuggingFace cache to use persistent storage
        os.environ["HF_HOME"] = str(self.cache_dir)
        os.environ["TRANSFORMERS_CACHE"] = str(self.cache_dir / "transformers")
        os.environ["DIFFUSERS_CACHE"] = str(self.cache_dir / "diffusers")
        
        logger.info(f"ModelManagerColab initialized with cache: {self.get_cache_path()}")
    
    def get_cache_path(self) -> Path:
        """Get the active HuggingFace cache directory."""
        return self.cache_dir
    
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
    
    def load_model(self, model_name: str) -> object:
        """
        Load model with caching optimized for Colab GPU. Downloads only if not cached.
        
        Args:
            model_name: Name of the model to load
            
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
            logger.info(f"📦 Using cached model from persistent storage")
        else:
            logger.info(f"⬇️  Downloading model to persistent storage (first time only)...")
        
        try:
            # Prepare common arguments optimized for Colab
            common_args = {
                "cache_dir": self.cache_dir,
                "torch_dtype": torch.float16,  # Use FP16 for GPU memory efficiency
                "safety_checker": None,
                "requires_safety_checker": False,
                "use_safetensors": True,  # Use safetensors for better compatibility
                "low_cpu_mem_usage": True,  # Reduce CPU memory during loading
                "variant": "fp16",  # Use FP16 variant if available
            }
            
            # Add token if available
            if self.hf_token:
                common_args["token"] = self.hf_token
                logger.info("🔐 Using HuggingFace token for authentication")
            
            # Memory optimization logging
            memory_usage = config.get("memory_usage", "Unknown")
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
            
            # Apply Colab GPU optimizations
            pipeline.enable_attention_slicing()  # Memory efficiency
            pipeline.enable_vae_slicing()  # Slice VAE for lower memory
            pipeline.enable_xformers_memory_efficient_attention()  # Use xformers if available
            logger.info("🧠 Enabled memory optimizations for Colab GPU")
            
            # Move to GPU (Colab typically has CUDA)
            if torch.cuda.is_available():
                pipeline = pipeline.to("cuda")
                logger.info(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
                logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            else:
                logger.warning("⚠️  No CUDA GPU detected, using CPU (very slow)")
            
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
    
    def mount_drive(self):
        """Helper method to mount Google Drive in Colab."""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            logger.info("📁 Google Drive mounted successfully")
        except ImportError:
            logger.warning("⚠️  Not running in Google Colab, skipping drive mount")
        except Exception as e:
            logger.error(f"❌ Failed to mount Google Drive: {e}")


def main():
    """Demo the ModelManagerColab functionality."""
    print("🚀 Parallax Generator - Colab Model Manager Demo")
    print("=" * 50)
    
    # Initialize model manager
    manager = ModelManagerColab()
    
    # Mount drive if in Colab
    manager.mount_drive()
    
    # Show cache info
    print(f"📁 Cache directory: {manager.get_cache_path()}")
    print(f"💾 Cache size: {manager.get_cache_size()}")
    print()
    
    # Show GPU info if available
    if torch.cuda.is_available():
        print(f"🚀 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  No GPU detected")
    print()
    
    # List available models
    print("📋 Available Models:")
    models = manager.list_models()
    for name, info in models.items():
        status = "✅ Cached" if info["cached"] else "❌ Not cached"
        loaded = "🔄 Loaded" if info["loaded"] else "💤 Not loaded"
        recommended = "⭐ RECOMMENDED" if info.get("recommended", False) else ""
        print(f"  {name}: {info['description']} - {status} - {loaded} {recommended}")
    
    print()
    print("💡 To load a model: manager.load_model('sd15_base')")
    print("💡 Models are cached in Google Drive for persistence!")


if __name__ == "__main__":
    main()
