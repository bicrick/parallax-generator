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


class ParallaxGeneratorColab:
    """Simplified parallax generator for Google Colab."""
    
    def __init__(self, output_dir: str = "/content/parallax_output", 
                 drive_cache: bool = True, drive_output_dir: str = "/content/drive/MyDrive/parallax_gallery"):
        """
        Initialize the parallax generator for Colab.
        
        Args:
            output_dir: Local directory to save generated assets
            drive_cache: Whether to also save images to Google Drive for persistence
            drive_output_dir: Google Drive directory for persistent image storage
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Google Drive persistent storage
        self.drive_cache = drive_cache
        self.drive_output_dir = Path(drive_output_dir) if drive_cache else None
        
        # Initialize model manager for Colab (models in local storage, not Drive)
        self.model_manager = ModelManagerColab(use_drive_cache=False)
        
        # Mount Google Drive only if needed for image storage
        if self.drive_cache:
            self.model_manager.mount_drive(force=True)
        
        # Setup Drive gallery if enabled
        if self.drive_cache and self.drive_output_dir:
            self.drive_output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"🖼️  Drive gallery enabled: {self.drive_output_dir}")
        
        # Simplified layer configuration for single image generation
        self.layer_config = {
            "background": {"range": (0, 60), "prompt_modifier": "distant, atmospheric, wide landscape"},
            "midground": {"range": (40, 80), "prompt_modifier": "detailed, middle distance"},
            "foreground": {"range": (70, 100), "prompt_modifier": "close-up, detailed, sharp focus"}
        }
        
        logger.info(f"ParallaxGeneratorColab initialized. Output: {self.output_dir}")
        logger.info("📦 Models cached locally (session duration ~12 hours)")
        if self.drive_cache:
            logger.info("🖼️  Images will be saved to both local storage and Google Drive")
    
    def save_image_with_metadata(self, image: Image.Image, filename: str, metadata: dict) -> dict:
        """
        Save image to both local and Drive locations with metadata.
        
        Args:
            image: PIL Image to save
            filename: Base filename (without extension)
            metadata: Dictionary with generation metadata
            
        Returns:
            Dictionary with local and drive paths
        """
        import json
        from datetime import datetime
        
        # Create timestamp for unique naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"{timestamp}_{filename.replace(' ', '_')}"
        
        # Save locally
        local_image_path = self.output_dir / f"{safe_filename}.png"
        local_meta_path = self.output_dir / f"{safe_filename}_meta.json"
        
        image.save(local_image_path)
        
        # Save metadata
        full_metadata = {
            "timestamp": timestamp,
            "filename": safe_filename,
            "local_path": str(local_image_path),
            **metadata
        }
        
        with open(local_meta_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        result = {
            "local_image": str(local_image_path),
            "local_metadata": str(local_meta_path),
            "drive_image": None,
            "drive_metadata": None
        }
        
        # Save to Drive if enabled
        if self.drive_cache and self.drive_output_dir:
            try:
                drive_image_path = self.drive_output_dir / f"{safe_filename}.png"
                drive_meta_path = self.drive_output_dir / f"{safe_filename}_meta.json"
                
                # Copy to Drive
                image.save(drive_image_path)
                
                # Update metadata with drive path
                full_metadata["drive_path"] = str(drive_image_path)
                
                with open(drive_meta_path, 'w') as f:
                    json.dump(full_metadata, f, indent=2)
                
                result["drive_image"] = str(drive_image_path)
                result["drive_metadata"] = str(drive_meta_path)
                
                logger.info(f"💾 Saved to Drive: {drive_image_path.name}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to save to Drive: {e}")
        
        return result
    
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
        
        # Save the main image with metadata
        main_metadata = {
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "resolution": {"width": width, "height": height},
            "tiling_enabled": enable_tiling,
            "model_used": "sd15_base",
            "type": "single_image"
        }
        
        main_result = self.save_image_with_metadata(
            generated_image, 
            f"parallax_{width}x{height}",
            main_metadata
        )
        
        # Create and save tiled preview if enabled
        preview_result = None
        if enable_tiling:
            preview_image = self.create_tiled_preview(generated_image, tiles=3)
            
            preview_metadata = {
                **main_metadata,
                "type": "tiled_preview",
                "tiles": 3
            }
            
            preview_result = self.save_image_with_metadata(
                preview_image,
                f"preview_{width}x{height}",
                preview_metadata
            )
            logger.info(f"🖼️  Tiled preview created")
        
        results = {
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "resolution": (width, height),
            "tiling_enabled": enable_tiling,
            "main_image": main_result,
            "preview_image": preview_result,
            "model_used": "sd15_base"
        }
        
        logger.info("✅ Single image generation complete!")
        logger.info(f"📁 Local: {main_result['local_image']}")
        if main_result['drive_image']:
            logger.info(f"💾 Drive: {main_result['drive_image']}")
        
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
    
    def list_gallery(self, location: str = "both") -> dict:
        """
        List all generated images in the gallery.
        
        Args:
            location: "local", "drive", or "both"
            
        Returns:
            Dictionary with gallery information
        """
        import json
        
        gallery = {"local": [], "drive": []}
        
        # List local images
        if location in ["local", "both"]:
            for meta_file in self.output_dir.glob("*_meta.json"):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    gallery["local"].append(metadata)
                except Exception as e:
                    logger.warning(f"⚠️  Could not read metadata: {meta_file} - {e}")
        
        # List drive images
        if location in ["drive", "both"] and self.drive_cache and self.drive_output_dir:
            for meta_file in self.drive_output_dir.glob("*_meta.json"):
                try:
                    with open(meta_file, 'r') as f:
                        metadata = json.load(f)
                    gallery["drive"].append(metadata)
                except Exception as e:
                    logger.warning(f"⚠️  Could not read drive metadata: {meta_file} - {e}")
        
        # Sort by timestamp
        for key in gallery:
            gallery[key].sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        return gallery
    
    def print_gallery(self, location: str = "both"):
        """Print a formatted gallery of generated images."""
        gallery = self.list_gallery(location)
        
        total_images = len(gallery["local"]) + len(gallery["drive"])
        print(f"\n🖼️  Gallery ({total_images} images)")
        print("=" * 50)
        
        if gallery["local"]:
            print(f"\n📁 Local Images ({len(gallery['local'])})")
            for i, meta in enumerate(gallery["local"][:5]):  # Show last 5
                prompt = meta.get("prompt", "Unknown")[:50] + "..." if len(meta.get("prompt", "")) > 50 else meta.get("prompt", "Unknown")
                timestamp = meta.get("timestamp", "Unknown")
                resolution = meta.get("resolution", {})
                size = f"{resolution.get('width', '?')}x{resolution.get('height', '?')}"
                print(f"  {i+1}. {timestamp} - {size} - {prompt}")
        
        if gallery["drive"] and self.drive_cache:
            print(f"\n💾 Drive Images ({len(gallery['drive'])})")
            for i, meta in enumerate(gallery["drive"][:5]):  # Show last 5
                prompt = meta.get("prompt", "Unknown")[:50] + "..." if len(meta.get("prompt", "")) > 50 else meta.get("prompt", "Unknown")
                timestamp = meta.get("timestamp", "Unknown")
                resolution = meta.get("resolution", {})
                size = f"{resolution.get('width', '?')}x{resolution.get('height', '?')}"
                print(f"  {i+1}. {timestamp} - {size} - {prompt}")
        
        if total_images == 0:
            print("No images found. Generate some first!")
        elif total_images > 10:
            print(f"\n... and {total_images - 10} more images")
        
        print(f"\n📁 Local: {self.output_dir}")
        if self.drive_cache:
            print(f"💾 Drive: {self.drive_output_dir}")


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
    """CLI entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Parallax Generator - Colab Version")
    parser.add_argument("prompt", nargs='?', help="Text prompt for image generation")
    parser.add_argument("--width", type=int, default=1024, help="Image width (default: 1024)")
    parser.add_argument("--height", type=int, default=768, help="Image height (default: 768)")
    parser.add_argument("--output", default="/content/parallax_output", help="Output directory")
    parser.add_argument("--mode", choices=["single", "layers"], default="single", 
                       help="Generation mode: 'single' for one tiled image, 'layers' for full parallax pack")
    parser.add_argument("--no-tiling", action="store_true", help="Disable seamless tiling (single mode only)")
    parser.add_argument("--no-drive", action="store_true", help="Disable Google Drive image storage (local only)")
    parser.add_argument("--gpu-info", action="store_true", help="Show GPU information and exit")
    parser.add_argument("--gallery", action="store_true", help="Show image gallery and exit")
    
    args = parser.parse_args()
    
    print("🚀 Parallax Generator - Colab Version")
    print("=" * 50)
    
    # Show GPU info if requested
    if args.gpu_info:
        if torch.cuda.is_available():
            print(f"🚀 GPU: {torch.cuda.get_device_name()}")
            print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"🔥 CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️  No GPU detected - this will be very slow!")
        return
    
    # Show gallery if requested
    if args.gallery:
        generator = ParallaxGeneratorColab(output_dir=args.output, drive_cache=not args.no_drive)
        generator.print_gallery()
        return
    
    # Check if prompt is provided
    if not args.prompt:
        # Use demo prompt if none provided
        args.prompt = "a serene mountain landscape with a lake, pine trees, and distant peaks"
        print(f"📝 No prompt provided, using demo: '{args.prompt}'")
    else:
        print(f"📝 Prompt: '{args.prompt}'")
    
    print(f"📏 Resolution: {args.width}x{args.height}")
    print(f"📁 Output: {args.output}")
    
    try:
        if args.mode == "single":
            # Generate single tiled image
            enable_tiling = not args.no_tiling
            print(f"🔄 Tiling: {'Enabled' if enable_tiling else 'Disabled'}")
            
            generator = ParallaxGeneratorColab(output_dir=args.output, drive_cache=not args.no_drive)
            generator.show_gpu_info()
            
            results = generator.generate_single_tiled_image(
                args.prompt, args.width, args.height, enable_tiling
            )
            
            print("\n✅ Single image generation complete!")
            print(f"📁 Local: {Path(results['main_image']['local_image']).name}")
            if results['main_image']['drive_image']:
                print(f"💾 Drive: {Path(results['main_image']['drive_image']).name}")
            if results['preview_image']:
                print(f"🖼️  Tiled preview: {Path(results['preview_image']['local_image']).name}")
                
        elif args.mode == "layers":
            # Generate full parallax layers
            generator = ParallaxGeneratorColab(output_dir=args.output, drive_cache=not args.no_drive)
            generator.show_gpu_info()
            
            results = generator.generate_parallax_layers(
                args.prompt, args.width, args.height
            )
            
            print("\n✅ Parallax layers generation complete!")
            print("📄 Generated layers:")
            for layer_name, path in results['layers'].items():
                print(f"  {layer_name}: {Path(path).name}")
            print(f"  composite: {Path(results['composite_preview']).name}")
    
    except Exception as e:
        print(f"\n❌ Generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n📁 All files saved to: {args.output}")
    return 0


if __name__ == "__main__":
    main()
