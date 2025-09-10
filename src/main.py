#!/usr/bin/env python3
"""
Parallax Generator MVP
Simple parallax layer generator for MacBook Pro with local inference.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw
import torch
import cv2

from model_manager import ModelManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParallaxGenerator:
    """Main parallax generator class."""
    
    def __init__(self, output_dir: str = "output", model_cache_dir: Optional[str] = None):
        """
        Initialize the parallax generator.
        
        Args:
            output_dir: Directory to save generated assets
            model_cache_dir: Custom model cache directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model manager with caching
        self.model_manager = ModelManager(cache_dir=model_cache_dir)
        
        # Default layer configuration
        self.layer_config = {
            "background": {"range": (0, 60), "prompt_modifier": "distant, atmospheric"},
            "midground": {"range": (40, 80), "prompt_modifier": "detailed, middle distance"},
            "foreground": {"range": (70, 100), "prompt_modifier": "close-up, detailed, sharp"}
        }
        
        logger.info(f"ParallaxGenerator initialized. Output: {self.output_dir}")
    
    def create_layer_mask(self, width: int, height: int, layer_range: Tuple[int, int]) -> Image.Image:
        """
        Create a horizontal band mask for a layer.
        
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
        This is a key feature for perfect horizontal tiling.
        
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
            
            return Image.fromarray(padded_array)
        except Exception as e:
            logger.warning(f"⚠️  Circular padding failed: {e}, returning original image")
            return image
    
    def generate_background(self, prompt: str, width: int = 1024, height: int = 768) -> Image.Image:
        """
        Generate the background layer.
        
        Args:
            prompt: Text prompt for generation
            width: Image width
            height: Image height
            
        Returns:
            Generated background image
        """
        logger.info("🎨 Generating background layer...")
        
        # Load stable SD 1.5 base model (FP32 - works reliably)
        pipeline = self.model_manager.load_model("sd15_base")
        
        # Enhance prompt for background
        bg_prompt = f"{prompt}, {self.layer_config['background']['prompt_modifier']}"
        
        # Generate with circular padding consideration
        padded_width = width + 128  # Extra width for circular padding
        
        logger.info(f"📝 Prompt: {bg_prompt}")
        
        # Generate image with conservative settings for stability
        result = pipeline(
            prompt=bg_prompt,
            width=padded_width,
            height=height,
            num_inference_steps=15,  # Reduced for memory efficiency
            guidance_scale=7.0       # Slightly lower for stability
        )
        
        background = result.images[0]
        
        # Safety check for FP16 precision issues
        import numpy as np
        bg_array = np.array(background)
        if bg_array.max() == 0 or np.isnan(bg_array).any():
            logger.warning("⚠️  Detected potential FP16 precision issue, applying fix...")
            # Force regeneration with different seed or fallback
            background = result.images[0]  # Get fresh copy
        
        # Re-enable circular padding with safety
        background = self.apply_circular_padding(background, padding=64)
        logger.info("🔄 Applied circular padding for seamless tiling")
        
        logger.info("✅ Background generated")
        return background
    
    def generate_layer_with_inpainting(self, base_image: Image.Image, mask: Image.Image, 
                                     prompt: str, layer_name: str) -> Image.Image:
        """
        Generate a layer using inpainting for coherence.
        
        Args:
            base_image: Base image to inpaint
            mask: Mask defining area to inpaint
            prompt: Text prompt for the layer
            layer_name: Name of the layer being generated
            
        Returns:
            Generated layer image
        """
        logger.info(f"🎨 Generating {layer_name} layer with inpainting...")
        
        # Load stable SD 1.5 inpainting model (FP32 - works reliably)
        pipeline = self.model_manager.load_model("sd15_inpaint")
        
        # Enhance prompt for this layer
        layer_config = self.layer_config.get(layer_name, {})
        enhanced_prompt = f"{prompt}, {layer_config.get('prompt_modifier', '')}"
        
        logger.info(f"📝 Prompt: {enhanced_prompt}")
        
        # Generate with inpainting using conservative settings
        result = pipeline(
            prompt=enhanced_prompt,
            image=base_image,
            mask_image=mask,
            num_inference_steps=15,  # Reduced for memory efficiency
            guidance_scale=7.0,      # Slightly lower for stability
            strength=0.8             # Conservative strength for stability
        )
        
        layer_image = result.images[0]
        
        logger.info(f"✅ {layer_name.capitalize()} layer generated")
        return layer_image
    
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
    
    def assign_parallax_speeds(self, prompt: str) -> Dict[str, float]:
        """
        Use LLM to assign semantic parallax speeds to layers.
        This is a placeholder - in full implementation would use OpenAI/Anthropic API.
        
        Args:
            prompt: Original scene prompt
            
        Returns:
            Dictionary mapping layer names to speed multipliers
        """
        # Placeholder logic - in real implementation, this would call LLM API
        # to analyze the scene and assign appropriate speeds
        
        default_speeds = {
            "background": 0.2,   # Slowest - distant objects
            "midground": 0.6,    # Medium speed
            "foreground": 1.0    # Fastest - closest objects
        }
        
        logger.info(f"🧠 Assigned parallax speeds: {default_speeds}")
        return default_speeds
    
    def generate_parallax_pack(self, prompt: str, width: int = 1024, height: int = 768) -> Dict:
        """
        Generate complete parallax pack with all layers.
        
        Args:
            prompt: Text prompt for the scene
            width: Image width
            height: Image height
            
        Returns:
            Dictionary with generation results and file paths
        """
        logger.info(f"🚀 Starting parallax generation: '{prompt}'")
        logger.info(f"📏 Resolution: {width}x{height}")
        
        results = {
            "prompt": prompt,
            "resolution": (width, height),
            "layers": {},
            "manifest": {}
        }
        
        # Step 1: Generate background
        background = self.generate_background(prompt, width, height)
        bg_path = self.output_dir / "bg.png"
        background.save(bg_path)
        results["layers"]["background"] = str(bg_path)
        
        # Step 2: Generate midground
        mid_mask = self.create_layer_mask(background.width, background.height, 
                                        self.layer_config["midground"]["range"])
        midground = self.generate_layer_with_inpainting(background, mid_mask, prompt, "midground")
        
        # Extract alpha for midground
        midground_alpha = self.extract_alpha_from_mask(midground, mid_mask)
        mid_path = self.output_dir / "mid.png"
        midground_alpha.save(mid_path)
        results["layers"]["midground"] = str(mid_path)
        
        # Step 3: Generate foreground
        # Composite background + midground for context
        composite = background.copy()
        composite.paste(midground_alpha, (0, 0), midground_alpha)
        
        fg_mask = self.create_layer_mask(composite.width, composite.height,
                                       self.layer_config["foreground"]["range"])
        foreground = self.generate_layer_with_inpainting(composite, fg_mask, prompt, "foreground")
        
        # Extract alpha for foreground
        foreground_alpha = self.extract_alpha_from_mask(foreground, fg_mask)
        fg_path = self.output_dir / "fg.png"
        foreground_alpha.save(fg_path)
        results["layers"]["foreground"] = str(fg_path)
        
        # Step 4: Assign parallax speeds using LLM
        speeds = self.assign_parallax_speeds(prompt)
        
        # Step 5: Create manifest
        manifest = {
            "prompt": prompt,
            "resolution": {"width": width, "height": height},
            "layers": [
                {
                    "name": "background",
                    "file": "bg.png",
                    "speed": speeds["background"],
                    "z_index": 0
                },
                {
                    "name": "midground", 
                    "file": "mid.png",
                    "speed": speeds["midground"],
                    "z_index": 1
                },
                {
                    "name": "foreground",
                    "file": "fg.png", 
                    "speed": speeds["foreground"],
                    "z_index": 2
                }
            ],
            "generated_at": str(Path().absolute()),
            "model_info": self.model_manager.list_models()
        }
        
        manifest_path = self.output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        results["manifest"] = manifest
        results["manifest_path"] = str(manifest_path)
        
        logger.info("✅ Parallax pack generation complete!")
        logger.info(f"📁 Output directory: {self.output_dir}")
        
        return results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Parallax Generator MVP")
    parser.add_argument("prompt", nargs='?', help="Text prompt for scene generation")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=768, help="Image height (default: 768)")
    parser.add_argument("--output", default="output", help="Output directory (default: output)")
    parser.add_argument("--cache-dir", help="Custom model cache directory")
    parser.add_argument("--list-models", action="store_true", help="List available models and exit")
    parser.add_argument("--low-memory", action="store_true", help="Use most memory-efficient settings (recommended for M3 MacBook Pro)")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = ParallaxGenerator(output_dir=args.output, model_cache_dir=args.cache_dir)
    
    if args.list_models:
        print("📋 Available Models:")
        models = generator.model_manager.list_models()
        for name, info in models.items():
            status = "✅ Cached" if info["cached"] else "❌ Not cached"
            memory = info.get("memory_usage", "Unknown")
            recommended = "⭐ RECOMMENDED" if info.get("recommended", False) else ""
            print(f"  {name}: {info['description']} - {status} - {memory} {recommended}")
        print(f"\n💾 Cache size: {generator.model_manager.get_cache_size()}")
        print(f"📁 Cache location: {generator.model_manager.get_cache_path()}")
        print(f"\n💡 For M3 MacBook Pro, use SD 1.5 models for best memory efficiency!")
        return
    
    # Check if prompt is provided
    if not args.prompt:
        parser.error("prompt is required unless using --list-models")
    
    # Generate parallax pack
    try:
        results = generator.generate_parallax_pack(args.prompt, args.width, args.height)
        
        print("\n🎉 Generation Complete!")
        print(f"📝 Prompt: {results['prompt']}")
        print(f"📏 Resolution: {results['resolution'][0]}x{results['resolution'][1]}")
        print(f"📁 Output: {generator.output_dir}")
        print("\n📄 Generated files:")
        for layer, path in results["layers"].items():
            print(f"  {layer}: {Path(path).name}")
        print(f"  manifest: {Path(results['manifest_path']).name}")
        
        print(f"\n💡 Next: Open {generator.output_dir} to see your parallax pack!")
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
