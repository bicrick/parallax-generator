#!/usr/bin/env python3
"""
Parallax Generator - Google Colab Version
Native seamless tiling for parallax backgrounds.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import json

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

import sys
import os
from pathlib import Path

# Add the src directory to Python path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from model_manager_colab import ModelManagerColab

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SeamlessTilingPatcher:
    """Patches Stable Diffusion pipelines for native seamless tiling."""
    
    def __init__(self):
        self.original_forwards = {}
        self.patched_modules = []
    
    def patch_circular_conv2d(self, module, name=""):
        """Patch Conv2d module for circular padding horizontally."""
        if hasattr(module, '_seamless_patched'):
            return
            
        original_forward = module.forward
        self.original_forwards[id(module)] = original_forward
        
        def circular_forward(x):
            pad_h, pad_w = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)
            
            # Apply more aggressive circular padding for better seamless results
            if pad_w > 0:
                # Use larger padding for better blending
                extended_pad_w = max(pad_w * 2, 8)  # At least 8 pixels
                x = F.pad(x, (extended_pad_w, extended_pad_w, 0, 0), mode='circular')
            if pad_h > 0:
                x = F.pad(x, (0, 0, pad_h, pad_h), mode='reflect')
            
            original_padding = module.padding
            module.padding = 0
            result = original_forward(x)
            module.padding = original_padding
            
            return result
        
        module.forward = circular_forward
        module._seamless_patched = True
        self.patched_modules.append((module, name))
    
    def patch_unet_for_seamless_tiling(self, unet):
        """Patch all Conv2d layers in UNet for seamless horizontal tiling."""
        logger.info("Patching UNet for seamless tiling")
        
        patched_count = 0
        for name, module in unet.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.patch_circular_conv2d(module, name)
                patched_count += 1
        
        logger.info(f"Patched {patched_count} Conv2d layers")
        return unet


class TilingPromptEnhancer:
    """Enhances prompts for seamless tiling."""
    
    TILING_MODIFIERS = {
        "landscape": ["endless horizon", "continuous landscape"],
        "abstract": ["seamless pattern", "tileable texture"],
        "architectural": ["continuous structure", "repeating elements"],
        "nature": ["endless forest", "continuous ocean"]
    }
    
    NEGATIVE_PROMPTS = ["borders", "edges", "frames", "boundaries", "seams"]
    
    def enhance_for_tiling(self, prompt: str, content_type: str = "landscape") -> Tuple[str, str]:
        """Enhance prompt for seamless tiling."""
        modifiers = self.TILING_MODIFIERS.get(content_type, self.TILING_MODIFIERS["landscape"])
        tiling_terms = ", ".join(modifiers[:2])
        enhanced_prompt = f"{prompt}, {tiling_terms}, high quality, detailed"
        negative_prompt = ", ".join(self.NEGATIVE_PROMPTS)
        return enhanced_prompt, negative_prompt
    
    def detect_content_type(self, prompt: str) -> str:
        """Detect content type from prompt."""
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["building", "city", "architecture"]):
            return "architectural"
        if any(word in prompt_lower for word in ["pattern", "texture", "abstract"]):
            return "abstract"
        if any(word in prompt_lower for word in ["forest", "ocean", "sky", "clouds"]):
            return "nature"
        
        return "landscape"


class SeamlessLatentProcessor:
    """Processes latents for seamless tiling."""
    
    def __init__(self, blend_width: int = 128):
        self.blend_width = blend_width
    
    def create_seamless_noise(self, shape: Tuple[int, ...], generator: Optional[torch.Generator] = None, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Create seamless noise tensor."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        
        if noise.size(-1) <= self.blend_width * 2:
            return noise
        
        # Apply horizontal wrapping with smoother blending
        # Use cosine interpolation for smoother blending
        x = torch.linspace(0, 1, self.blend_width, device=noise.device, dtype=noise.dtype)
        blend_mask = 0.5 * (1 - torch.cos(x * torch.pi))  # Cosine interpolation
        blend_mask = blend_mask.view(1, 1, 1, -1)
        
        left_edge = noise[..., :self.blend_width].clone()
        right_edge = noise[..., -self.blend_width:].clone()
        
        # Smoother blending
        noise[..., :self.blend_width] = left_edge * (1 - blend_mask) + right_edge * blend_mask
        noise[..., -self.blend_width:] = right_edge * (1 - blend_mask) + left_edge * blend_mask
        
        return noise


class ParallaxGeneratorColab:
    """Seamless parallax generator for Google Colab."""
    
    def __init__(self, output_dir: str = "/content/parallax_output", drive_output_dir: str = "/content/drive/MyDrive/parallax_gallery"):
        """Initialize the generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Google Drive output
        self.drive_output_dir = Path(drive_output_dir)
        self.mount_drive()
        
        self.model_manager = ModelManagerColab(use_drive_cache=False)
        self.tiling_patcher = SeamlessTilingPatcher()
        self.prompt_enhancer = TilingPromptEnhancer()
        self.latent_processor = SeamlessLatentProcessor()
        self.patched_pipelines = set()
        
        logger.info(f"Generator initialized. Local: {self.output_dir}, Drive: {self.drive_output_dir}")
    
    def mount_drive(self):
        """Mount Google Drive."""
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            self.drive_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info("Google Drive mounted and gallery directory created")
        except ImportError:
            logger.warning("Not in Google Colab, skipping drive mount")
        except Exception as e:
            logger.error(f"Failed to mount Google Drive: {e}")
    
    def prepare_pipeline_for_seamless_tiling(self, pipeline, pipeline_id: str = None) -> object:
        """Prepare pipeline for seamless tiling."""
        pipeline_id = pipeline_id or f"pipeline_{id(pipeline)}"
        
        if pipeline_id not in self.patched_pipelines:
            self.tiling_patcher.patch_unet_for_seamless_tiling(pipeline.unet)
            self.patched_pipelines.add(pipeline_id)
            logger.info(f"Pipeline {pipeline_id} prepared for seamless tiling")
        
        return pipeline
    
    def generate_seamless_image(self, pipeline, prompt: str, negative_prompt: str = None,
                               width: int = 1024, height: int = 768, **kwargs) -> Image.Image:
        """Generate image with native seamless tiling."""
        generation_params = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": kwargs.get("num_inference_steps", 20),
            "guidance_scale": kwargs.get("guidance_scale", 7.5),
            "generator": kwargs.get("generator", torch.Generator(device="cuda").manual_seed(42))
        }
        
        if negative_prompt:
            generation_params["negative_prompt"] = negative_prompt
        
        # Use seamless noise with correct dtype
        latent_shape = (1, pipeline.unet.config.in_channels, height // 8, width // 8)
        custom_latents = self.latent_processor.create_seamless_noise(
            latent_shape, 
            generator=generation_params["generator"],
            dtype=pipeline.unet.dtype
        )
        generation_params["latents"] = custom_latents
        
        logger.info("Generating seamless image")
        result = pipeline(**generation_params)
        
        # Apply additional seamless post-processing for perfect tiling
        generated_image = result.images[0]
        generated_image = self.enhance_seamless_tiling(generated_image)
        
        return generated_image
    
    def enhance_seamless_tiling(self, image: Image.Image, blend_width: int = 64) -> Image.Image:
        """Apply additional post-processing to enhance seamless tiling."""
        try:
            width, height = image.size
            
            if width <= blend_width * 2:
                return image
            
            # Convert to numpy for processing
            img_array = np.array(image, dtype=np.float32)
            
            # Create smoother blend mask using cosine interpolation
            x = np.linspace(0, 1, blend_width)
            blend_mask = 0.5 * (1 - np.cos(x * np.pi))
            blend_mask = blend_mask.reshape(1, -1, 1)
            
            # Extract edges
            left_edge = img_array[:, :blend_width].copy()
            right_edge = img_array[:, -blend_width:].copy()
            
            # Apply enhanced blending
            img_array[:, :blend_width] = left_edge * (1 - blend_mask) + right_edge * blend_mask
            img_array[:, -blend_width:] = right_edge * (1 - blend_mask) + left_edge * blend_mask
            
            # Convert back to PIL Image
            result_array = np.clip(img_array, 0, 255).astype(np.uint8)
            enhanced_image = Image.fromarray(result_array)
            
            logger.info("Applied enhanced seamless tiling post-processing")
            return enhanced_image
            
        except Exception as e:
            logger.warning(f"Seamless enhancement failed: {e}, using original")
            return image
    
    def save_image(self, image: Image.Image, filename: str, metadata: dict = None) -> dict:
        """Save image with metadata to both local and Drive."""
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename.replace(' ', '_')}"
        
        # Save locally
        local_image_path = self.output_dir / f"{safe_filename}.png"
        image.save(local_image_path)
        
        # Save metadata locally
        if metadata:
            local_meta_path = self.output_dir / f"{safe_filename}_meta.json"
            with open(local_meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Save to Drive
        drive_image_path = None
        try:
            drive_image_path = self.drive_output_dir / f"{safe_filename}.png"
            image.save(drive_image_path)
            
            if metadata:
                drive_meta_path = self.drive_output_dir / f"{safe_filename}_meta.json"
                with open(drive_meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Saved to Drive: {drive_image_path.name}")
        except Exception as e:
            logger.warning(f"Failed to save to Drive: {e}")
        
        return {
            "local": str(local_image_path),
            "drive": str(drive_image_path) if drive_image_path else None
        }
    
    def create_tiling_test(self, image: Image.Image, tiles: int = 4) -> Image.Image:
        """Create tiling test image."""
        width, height = image.size
        test_width = width * tiles
        test_image = Image.new('RGB', (test_width, height))
        
        for i in range(tiles):
            test_image.paste(image, (i * width, 0))
        
        return test_image
    
    def generate_single_tiled_image(self, prompt: str, width: int = 1024, height: int = 256) -> Dict:
        """Generate single seamless tiled image."""
        logger.info(f"Generating: '{prompt}' at {width}x{height}")
        
        # Load and prepare model
        pipeline = self.model_manager.load_model("sd15_base")
        pipeline = self.prepare_pipeline_for_seamless_tiling(pipeline, "sd15_base")
        
        # Enhance prompt
        content_type = self.prompt_enhancer.detect_content_type(prompt)
        enhanced_prompt, negative_prompt = self.prompt_enhancer.enhance_for_tiling(prompt, content_type)
        
        logger.info(f"Content type: {content_type}")
        logger.info(f"Enhanced: {enhanced_prompt}")
        
        # Generate image
        generated_image = self.generate_seamless_image(
            pipeline=pipeline,
            prompt=enhanced_prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height
        )
        
        # Save main image
        metadata = {
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "negative_prompt": negative_prompt,
            "resolution": {"width": width, "height": height},
            "content_type": content_type,
            "approach": "native_seamless"
        }
        
        main_paths = self.save_image(generated_image, f"seamless_{width}x{height}", metadata)
        
        # Create and save tiling test
        test_image = self.create_tiling_test(generated_image, tiles=4)
        test_paths = self.save_image(test_image, f"tiling_test_{width}x{height}")
        
        logger.info("Generation complete")
        logger.info(f"Local: {Path(main_paths['local']).name}")
        if main_paths['drive']:
            logger.info(f"Drive: {Path(main_paths['drive']).name}")
        
        return {
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "resolution": (width, height),
            "main_image": main_paths,
            "test_image": test_paths,
            "approach": "native_seamless"
        }


# Convenience functions
def generate_seamless_image(prompt: str, width: int = 1024, height: int = 256):
    """Generate a seamless tiled image."""
    generator = ParallaxGeneratorColab()
    return generator.generate_single_tiled_image(prompt, width, height)


def generate_ultra_wide_wallpaper(prompt: str, wallpaper_type: str = "fhd"):
    """Generate ultra-wide wallpaper for side-scrolling (10:1 aspect ratio)."""
    wallpaper_sizes = {
        "hd": (13660, 768),     # 1366x768 * 10 = 13660x768
        "fhd": (19200, 1080),   # 1920x1080 * 10 = 19200x1080  
        "4k": (38400, 2160)     # 3840x2160 * 10 = 38400x2160
    }
    
    if wallpaper_type not in wallpaper_sizes:
        wallpaper_type = "fhd"
    
    width, height = wallpaper_sizes[wallpaper_type]
    print(f"Generating {wallpaper_type.upper()} ultra-wide wallpaper: {width}x{height} (10:1 aspect ratio)")
    
    generator = ParallaxGeneratorColab()
    return generator.generate_single_tiled_image(prompt, width, height)


def generate_side_scrolling_background(prompt: str, height: int = 1080, aspect_ratio: str = "10:1"):
    """Generate background optimized for side-scrolling games."""
    try:
        width_ratio, height_ratio = map(int, aspect_ratio.split(':'))
        width = height * width_ratio // height_ratio
        
        print(f"Generating side-scrolling background: {width}x{height} ({aspect_ratio} aspect ratio)")
        
        generator = ParallaxGeneratorColab()
        return generator.generate_single_tiled_image(prompt, width, height)
    except ValueError:
        print(f"Invalid aspect ratio: {aspect_ratio}. Using default 10:1")
        return generate_side_scrolling_background(prompt, height, "10:1")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Seamless Parallax Generator")
    parser.add_argument("prompt", nargs='?', help="Text prompt for generation")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024 for testing)")
    parser.add_argument("--height", type=int, default=256, help="Image height (default: 256 for testing)")
    parser.add_argument("--wallpaper", choices=["hd", "fhd", "4k"], help="Preset wallpaper sizes: hd=1366x768, fhd=1920x1080, 4k=3840x2160")
    parser.add_argument("--aspect-ratio", type=str, help="Aspect ratio like '10:1' for ultra-wide side-scrolling")
    parser.add_argument("--output", default="/content/parallax_output", help="Output directory")
    
    args = parser.parse_args()
    
    # Handle wallpaper presets
    if args.wallpaper:
        wallpaper_sizes = {
            "hd": (1366, 768),
            "fhd": (1920, 1080), 
            "4k": (3840, 2160)
        }
        base_width, base_height = wallpaper_sizes[args.wallpaper]
        # Make ultra-wide for side-scrolling (10x width)
        args.width = base_width * 10
        args.height = base_height
        print(f"Using {args.wallpaper.upper()} wallpaper preset: {args.width}x{args.height} (10:1 aspect ratio)")
    
    # Handle aspect ratio
    elif args.aspect_ratio:
        try:
            width_ratio, height_ratio = map(int, args.aspect_ratio.split(':'))
            if args.height:
                args.width = args.height * width_ratio // height_ratio
            else:
                args.height = args.width * height_ratio // width_ratio
            print(f"Using {args.aspect_ratio} aspect ratio: {args.width}x{args.height}")
        except ValueError:
            print(f"Invalid aspect ratio format: {args.aspect_ratio}. Use format like '10:1'")
            return 1
    
    if not args.prompt:
        args.prompt = "serene mountain landscape with lake and pine trees"
        print(f"Using default prompt: '{args.prompt}'")
    
    print(f"Prompt: '{args.prompt}'")
    print(f"Resolution: {args.width}x{args.height} (aspect ratio: {args.width/args.height:.1f}:1)")
    print(f"Output: {args.output}")
    
    try:
        generator = ParallaxGeneratorColab(output_dir=args.output)
        
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name()}")
        else:
            print("Warning: No GPU detected")
        
        results = generator.generate_single_tiled_image(args.prompt, args.width, args.height)
        
        print("Generation complete!")
        print(f"Local: {Path(results['main_image']['local']).name}")
        if results['main_image']['drive']:
            print(f"Drive: {Path(results['main_image']['drive']).name}")
        print(f"Test: {Path(results['test_image']['local']).name}")
    
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    main()