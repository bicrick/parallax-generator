#!/usr/bin/env python3
"""
Parallax Generator - Google Colab Version
Simplified version for generating single parallax images with seamless tiling.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json

import numpy as np
from PIL import Image, ImageDraw
import torch
import cv2

from model_manager_colab import ModelManagerColab

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParallaxGeneratorColab:
    """Simplified parallax generator for Google Colab."""
    
    def __init__(self, output_dir: str = "/content/parallax_output"):
        """
        Initialize the parallax generator for Colab.
        
        Args:
            output_dir: Directory to save generated assets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model manager for Colab
        self.model_manager = ModelManagerColab()
        
        # Mount Google Drive for persistent model caching
        self.model_manager.mount_drive()
        
        # Simplified layer configuration for single image generation
        self.layer_config = {
            "background": {"range": (0, 60), "prompt_modifier": "distant, atmospheric, wide landscape"},
            "midground": {"range": (40, 80), "prompt_modifier": "detailed, middle distance"},
            "foreground": {"range": (70, 100), "prompt_modifier": "close-up, detailed, sharp focus"}
        }
        
        logger.info(f"ParallaxGeneratorColab initialized. Output: {self.output_dir}")
        logger.info("🔄 Models will be cached in Google Drive for persistence across sessions")
    
    def create_layer_mask(self, width: int, height: int, layer_range: Tuple[int, int]) -> Image.Image:
        """
        Create a horizontal band mask for a layer with seamless tiling support.
        
        Args:
            width: Image width
            height: Image height
            layer_range: (start_percent, end_percent) for the layer
            
        Returns:
            PIL Image mask (white = keep, black = remove)
        """
        mask = Image.new('L', (width, height), 0)  # Start with black
        draw = ImageDraw.Draw(mask)
        
        start_y = int(height * layer_range[0] / 100)
        end_y = int(height * layer_range[1] / 100)
        
        # Create horizontal band (white = area to keep)
        draw.rectangle([0, start_y, width, end_y], fill=255)
        
        # Add soft edges for better blending
        mask_array = np.array(mask)
        
        # Gaussian blur for soft edges
        mask_array = cv2.GaussianBlur(mask_array, (21, 21), 10)
        
        return Image.fromarray(mask_array)
    
    def apply_circular_padding(self, image: Image.Image, padding: int = 64) -> Image.Image:
        """
        Apply circular padding for seamless horizontal tiling.
        This ensures the image tiles perfectly horizontally.
        
        Args:
            image: Input image
            padding: Padding size in pixels
            
        Returns:
            Image with circular padding applied
        """
        try:
            width, height = image.size
            
            # Convert to numpy for easier manipulation
            img_array = np.array(image)
            
            # Safety check for invalid values
            if img_array.max() == 0 or np.isnan(img_array).any():
                logger.warning("⚠️  Skipping circular padding due to invalid image data")
                return image
            
            # Create padded image
            padded_width = width + 2 * padding
            padded_array = np.zeros((height, padded_width, img_array.shape[2]), dtype=img_array.dtype)
            
            # Place original image in center
            padded_array[:, padding:padding+width] = img_array
            
            # Apply circular padding
            # Left padding: copy from right edge of original
            padded_array[:, :padding] = img_array[:, -padding:]
            
            # Right padding: copy from left edge of original  
            padded_array[:, padding+width:] = img_array[:, :padding]
            
            logger.info("🔄 Applied circular padding for seamless tiling")
            return Image.fromarray(padded_array)
        except Exception as e:
            logger.warning(f"⚠️  Circular padding failed: {e}, returning original image")
            return image
    
    def generate_single_tiled_image(self, prompt: str, width: int = 1024, height: int = 768, 
                                  enable_tiling: bool = True) -> Dict:
        """
        Generate a single image with optional seamless tiling.
        
        Args:
            prompt: Text prompt for generation
            width: Image width
            height: Image height
            enable_tiling: Whether to enable seamless horizontal tiling
            
        Returns:
            Dictionary with generation results
        """
        logger.info(f"🚀 Generating single image: '{prompt}'")
        logger.info(f"📏 Resolution: {width}x{height}")
        logger.info(f"🔄 Tiling enabled: {enable_tiling}")
        
        # Load SD 1.5 base model (optimized for Colab)
        logger.info("📦 Loading Stable Diffusion model...")
        pipeline = self.model_manager.load_model("sd15_base")
        
        # Enhance prompt for better results
        enhanced_prompt = f"{prompt}, high quality, detailed, beautiful"
        
        # Adjust width for tiling if enabled
        generation_width = width + 128 if enable_tiling else width
        
        logger.info(f"📝 Enhanced prompt: {enhanced_prompt}")
        logger.info(f"🎨 Generating image...")
        
        # Generate image with optimized settings for Colab
        result = pipeline(
            prompt=enhanced_prompt,
            width=generation_width,
            height=height,
            num_inference_steps=20,  # Good balance of quality and speed
            guidance_scale=7.5,      # Standard guidance
            generator=torch.Generator(device="cuda").manual_seed(42)  # Reproducible results
        )
        
        generated_image = result.images[0]
        
        # Apply circular padding for tiling if enabled
        if enable_tiling:
            generated_image = self.apply_circular_padding(generated_image, padding=64)
            logger.info("✅ Seamless tiling applied")
        
        # Save the image
        output_path = self.output_dir / f"parallax_image_{width}x{height}.png"
        generated_image.save(output_path)
        
        # Create a tiled preview to demonstrate seamless tiling
        if enable_tiling:
            preview_image = self.create_tiled_preview(generated_image, tiles=3)
            preview_path = self.output_dir / f"tiled_preview_{width}x{height}.png"
            preview_image.save(preview_path)
            logger.info(f"🖼️  Tiled preview saved: {preview_path.name}")
        
        results = {
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "resolution": (width, height),
            "tiling_enabled": enable_tiling,
            "image_path": str(output_path),
            "preview_path": str(preview_path) if enable_tiling else None,
            "model_used": "sd15_base"
        }
        
        logger.info("✅ Single image generation complete!")
        logger.info(f"📁 Saved to: {output_path}")
        
        return results
    
    def create_tiled_preview(self, image: Image.Image, tiles: int = 3) -> Image.Image:
        """
        Create a preview showing the image tiled horizontally to demonstrate seamless tiling.
        
        Args:
            image: Source image
            tiles: Number of tiles to show
            
        Returns:
            Tiled preview image
        """
        width, height = image.size
        preview_width = width * tiles
        
        preview = Image.new('RGB', (preview_width, height))
        
        for i in range(tiles):
            preview.paste(image, (i * width, 0))
        
        return preview
    
    def generate_parallax_layers(self, prompt: str, width: int = 1024, height: int = 768) -> Dict:
        """
        Generate complete parallax layers (background, midground, foreground).
        
        Args:
            prompt: Text prompt for the scene
            width: Image width  
            height: Image height
            
        Returns:
            Dictionary with generation results and file paths
        """
        logger.info(f"🚀 Generating parallax layers: '{prompt}'")
        logger.info(f"📏 Resolution: {width}x{height}")
        
        results = {
            "prompt": prompt,
            "resolution": (width, height),
            "layers": {}
        }
        
        # Step 1: Generate background
        logger.info("🎨 Generating background layer...")
        pipeline_base = self.model_manager.load_model("sd15_base")
        
        bg_prompt = f"{prompt}, {self.layer_config['background']['prompt_modifier']}"
        padded_width = width + 128  # Extra width for circular padding
        
        bg_result = pipeline_base(
            prompt=bg_prompt,
            width=padded_width,
            height=height,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=torch.Generator(device="cuda").manual_seed(42)
        )
        
        background = bg_result.images[0]
        background = self.apply_circular_padding(background, padding=64)
        
        bg_path = self.output_dir / "background.png"
        background.save(bg_path)
        results["layers"]["background"] = str(bg_path)
        logger.info("✅ Background layer complete")
        
        # Step 2: Generate midground using inpainting
        logger.info("🎨 Generating midground layer...")
        pipeline_inpaint = self.model_manager.load_model("sd15_inpaint")
        
        mid_mask = self.create_layer_mask(background.width, background.height, 
                                        self.layer_config["midground"]["range"])
        mid_prompt = f"{prompt}, {self.layer_config['midground']['prompt_modifier']}"
        
        mid_result = pipeline_inpaint(
            prompt=mid_prompt,
            image=background,
            mask_image=mid_mask,
            num_inference_steps=20,
            guidance_scale=7.5,
            strength=0.8,
            generator=torch.Generator(device="cuda").manual_seed(43)
        )
        
        midground = mid_result.images[0]
        midground_alpha = self.extract_alpha_from_mask(midground, mid_mask)
        
        mid_path = self.output_dir / "midground.png"
        midground_alpha.save(mid_path)
        results["layers"]["midground"] = str(mid_path)
        logger.info("✅ Midground layer complete")
        
        # Step 3: Generate foreground
        logger.info("🎨 Generating foreground layer...")
        
        # Composite background + midground for context
        composite = background.copy()
        composite.paste(midground_alpha, (0, 0), midground_alpha)
        
        fg_mask = self.create_layer_mask(composite.width, composite.height,
                                       self.layer_config["foreground"]["range"])
        fg_prompt = f"{prompt}, {self.layer_config['foreground']['prompt_modifier']}"
        
        fg_result = pipeline_inpaint(
            prompt=fg_prompt,
            image=composite,
            mask_image=fg_mask,
            num_inference_steps=20,
            guidance_scale=7.5,
            strength=0.8,
            generator=torch.Generator(device="cuda").manual_seed(44)
        )
        
        foreground = fg_result.images[0]
        foreground_alpha = self.extract_alpha_from_mask(foreground, fg_mask)
        
        fg_path = self.output_dir / "foreground.png"
        foreground_alpha.save(fg_path)
        results["layers"]["foreground"] = str(fg_path)
        logger.info("✅ Foreground layer complete")
        
        # Create a composite preview
        final_composite = background.copy()
        final_composite.paste(midground_alpha, (0, 0), midground_alpha)
        final_composite.paste(foreground_alpha, (0, 0), foreground_alpha)
        
        composite_path = self.output_dir / "composite_preview.png"
        final_composite.save(composite_path)
        results["composite_preview"] = str(composite_path)
        
        logger.info("✅ Parallax layers generation complete!")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        return results
    
    def extract_alpha_from_mask(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        """
        Extract alpha channel from mask for transparent PNG.
        
        Args:
            image: Source image
            mask: Mask to use as alpha channel
            
        Returns:
            RGBA image with alpha channel
        """
        # Convert to RGBA
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
        
        # Use mask as alpha channel
        image.putalpha(mask)
        
        return image
    
    def show_gpu_info(self):
        """Display GPU information for debugging."""
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"🔥 CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  No GPU detected - this will be very slow!")


# Convenience functions for Colab notebook usage
def generate_single_image(prompt: str, width: int = 1024, height: int = 768, enable_tiling: bool = True):
    """
    Convenience function to generate a single tiled image.
    Perfect for quick experimentation in Colab.
    
    Args:
        prompt: Text description of the image to generate
        width: Image width (default: 1024)
        height: Image height (default: 768) 
        enable_tiling: Whether to make the image tile seamlessly (default: True)
        
    Returns:
        Dictionary with results including file paths
    """
    generator = ParallaxGeneratorColab()
    generator.show_gpu_info()
    return generator.generate_single_tiled_image(prompt, width, height, enable_tiling)


def generate_parallax_pack(prompt: str, width: int = 1024, height: int = 768):
    """
    Convenience function to generate full parallax layers.
    
    Args:
        prompt: Text description of the scene
        width: Image width (default: 1024)
        height: Image height (default: 768)
        
    Returns:
        Dictionary with results including all layer paths
    """
    generator = ParallaxGeneratorColab()
    generator.show_gpu_info()
    return generator.generate_parallax_layers(prompt, width, height)


def main():
    """Demo function for testing."""
    print("🚀 Parallax Generator - Colab Version")
    print("=" * 50)
    
    # Example usage
    prompt = "a serene mountain landscape with a lake, pine trees, and distant peaks"
    
    print(f"📝 Generating image with prompt: '{prompt}'")
    
    # Generate single tiled image
    results = generate_single_image(prompt, width=512, height=512, enable_tiling=True)
    
    print("\n✅ Generation complete!")
    print(f"📄 Image saved: {Path(results['image_path']).name}")
    if results['preview_path']:
        print(f"🖼️  Tiled preview: {Path(results['preview_path']).name}")


if __name__ == "__main__":
    main()
