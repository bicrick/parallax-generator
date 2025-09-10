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
    
    def make_circular_conv2d(self, conv_module):
        """Create a circular padding wrapper for Conv2d modules."""
        original_forward = conv_module.forward
        
        def circular_conv_forward(input_tensor):
            # Get padding values
            if isinstance(conv_module.padding, tuple):
                pad_h, pad_w = conv_module.padding
            else:
                pad_h = pad_w = conv_module.padding
            
            # Apply circular padding horizontally, reflection vertically
            if pad_w > 0 or pad_h > 0:
                # Apply horizontal circular padding
                if pad_w > 0:
                    input_tensor = F.pad(input_tensor, (pad_w, pad_w, 0, 0), mode='circular')
                
                # Apply vertical reflection padding  
                if pad_h > 0:
                    input_tensor = F.pad(input_tensor, (0, 0, pad_h, pad_h), mode='reflect')
                
                # Temporarily set padding to 0 to avoid double padding
                original_padding = conv_module.padding
                conv_module.padding = 0
                
                try:
                    result = original_forward(input_tensor)
                finally:
                    # Always restore original padding
                    conv_module.padding = original_padding
                
                return result
            else:
                return original_forward(input_tensor)
        
        return circular_conv_forward
    
    def patch_unet_for_seamless_tiling(self, unet):
        """Patch all Conv2d layers in UNet for seamless horizontal tiling."""
        logger.info("Patching UNet for seamless tiling")
        
        patched_count = 0
        for name, module in unet.named_modules():
            if isinstance(module, torch.nn.Conv2d) and not hasattr(module, '_seamless_patched'):
                # Store original forward method
                self.original_forwards[id(module)] = module.forward
                
                # Replace with circular version
                module.forward = self.make_circular_conv2d(module)
                module._seamless_patched = True
                self.patched_modules.append((module, name))
                patched_count += 1
        
        logger.info(f"Patched {patched_count} Conv2d layers for seamless tiling")
        return unet
    
    def unpatch_all(self):
        """Restore all patched modules to their original state."""
        for module, name in self.patched_modules:
            if hasattr(module, '_seamless_patched'):
                # Restore original forward method
                if id(module) in self.original_forwards:
                    module.forward = self.original_forwards[id(module)]
                delattr(module, '_seamless_patched')
        
        self.patched_modules.clear()
        self.original_forwards.clear()
        logger.info("All seamless tiling patches removed")


class TilingPromptEnhancer:
    """Simple prompt enhancement for seamless tiling."""
    
    def enhance_for_tiling(self, prompt: str) -> Tuple[str, str]:
        """Enhance prompt for seamless tiling without forcing specific content."""
        # Just add basic quality terms and anti-seam negative prompt
        enhanced_prompt = f"{prompt}, high quality, detailed"
        negative_prompt = "borders, edges, frames, boundaries, seams"
        return enhanced_prompt, negative_prompt


class SeamlessLatentProcessor:
    """Processes latents for seamless tiling using Tiled Diffusion methodology."""
    
    def __init__(self, blend_width: int = 128, max_width: int = 32):
        self.blend_width = blend_width
        self.max_width = max_width
    
    def create_seamless_noise(self, shape: Tuple[int, ...], generator: Optional[torch.Generator] = None, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Create enhanced seamless noise with multiple blending passes."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Create padded noise for better constraint application
        batch, channels, height, width = shape
        padded_shape = (batch, channels, height, width + 2 * self.max_width)
        noise = torch.randn(padded_shape, generator=generator, device=device, dtype=dtype)
        
        # Apply multiple blending passes for smoother transitions
        blend_widths = [64, 32, 16] if width > 128 else [32, 16, 8]
        
        for blend_w in blend_widths:
            if noise.size(-1) > blend_w * 2:
                # Apply cosine blending
                x = torch.linspace(0, 1, blend_w, device=noise.device, dtype=noise.dtype)
                blend_mask = 0.5 * (1 - torch.cos(x * torch.pi))
                blend_mask = blend_mask.view(1, 1, 1, -1)
                
                left_edge = noise[..., :blend_w].clone()
                right_edge = noise[..., -blend_w:].clone()
                
                # Enhanced blending with wrapping
                noise[..., :blend_w] = left_edge * (1 - blend_mask) + right_edge * blend_mask
                noise[..., -blend_w:] = right_edge * (1 - blend_mask) + left_edge * blend_mask
        
        # Crop back to original size (remove padding)
        noise = noise[..., self.max_width:-self.max_width]
        
        return noise
    
    def apply_step_constraints(self, latents: torch.Tensor, step: int, total_steps: int) -> torch.Tensor:
        """Apply tiling constraints at each denoising step (Tiled Diffusion approach)."""
        if latents.size(-1) <= self.blend_width * 2:
            return latents
        
        # Calculate constraint strength (stronger early in generation)
        constraint_strength = 1.0 - (step / total_steps) * 0.5  # 1.0 -> 0.5
        
        # Apply edge consistency enforcement
        blend_w = min(self.blend_width, latents.size(-1) // 4)
        
        # Create blend mask
        x = torch.linspace(0, 1, blend_w, device=latents.device, dtype=latents.dtype)
        blend_mask = 0.5 * (1 - torch.cos(x * torch.pi))
        blend_mask = blend_mask.view(1, 1, 1, -1)
        
        # Extract edges
        left_edge = latents[..., :blend_w].clone()
        right_edge = latents[..., -blend_w:].clone()
        
        # Apply constraint with varying strength
        target_left = left_edge * (1 - constraint_strength) + right_edge * constraint_strength
        target_right = right_edge * (1 - constraint_strength) + left_edge * constraint_strength
        
        # Apply blended constraints
        latents[..., :blend_w] = target_left * blend_mask + left_edge * (1 - blend_mask)
        latents[..., -blend_w:] = target_right * blend_mask + right_edge * (1 - blend_mask)
        
        return latents
    
    def create_padded_latents(self, shape: Tuple[int, ...], generator: Optional[torch.Generator] = None, dtype: torch.dtype = torch.float16) -> torch.Tensor:
        """Create latents with padding regions for constraint application."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch, channels, height, width = shape
        
        # Add padding for tiling constraints
        padded_shape = (batch, channels, height, width + 2 * self.max_width)
        padded_latents = torch.randn(padded_shape, generator=generator, device=device, dtype=dtype)
        
        return padded_latents


class ParallaxGeneratorColab:
    """Seamless parallax generator for Google Colab."""
    
    def __init__(self, output_dir: str = "/content/parallax_output", drive_output_dir: str = "/content/drive/MyDrive/parallax_gallery", enable_unet_patching: bool = True):
        """Initialize the generator."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup Google Drive output
        self.drive_output_dir = Path(drive_output_dir)
        self.mount_drive()
        
        self.model_manager = ModelManagerColab(use_drive_cache=False)
        self.enable_unet_patching = enable_unet_patching
        self.tiling_patcher = SeamlessTilingPatcher() if enable_unet_patching else None
        self.prompt_enhancer = TilingPromptEnhancer()
        self.latent_processor = SeamlessLatentProcessor(blend_width=128, max_width=32)
        self.patched_pipelines = set()
        
        approach = "UNet patching + post-processing" if enable_unet_patching else "Post-processing only"
        logger.info(f"Generator initialized. Approach: {approach}")
        logger.info(f"Using Tiled Diffusion methodology with step-by-step constraints")
        logger.info(f"Local: {self.output_dir}, Drive: {self.drive_output_dir}")
    
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
        if not self.enable_unet_patching:
            return pipeline
            
        pipeline_id = pipeline_id or f"pipeline_{id(pipeline)}"
        
        if pipeline_id not in self.patched_pipelines:
            self.tiling_patcher.patch_unet_for_seamless_tiling(pipeline.unet)
            self.patched_pipelines.add(pipeline_id)
            logger.info(f"Pipeline {pipeline_id} prepared for seamless tiling")
        
        return pipeline
    
    def generate_seamless_image(self, pipeline, prompt: str, negative_prompt: str = None,
                               width: int = 1024, height: int = 768, **kwargs) -> Image.Image:
        """Generate image with native seamless tiling using Tiled Diffusion methodology."""
        num_inference_steps = kwargs.get("num_inference_steps", 20)
        guidance_scale = kwargs.get("guidance_scale", 7.5)
        generator = kwargs.get("generator", torch.Generator(device="cuda").manual_seed(42))
        
        # Use custom generation loop with step-by-step constraints
        if self.enable_unet_patching:
            logger.info("Using Tiled Diffusion methodology with step-by-step constraints")
            return self._generate_with_step_constraints(
                pipeline, prompt, negative_prompt, width, height, 
                num_inference_steps, guidance_scale, generator
            )
        else:
            # Fallback to standard pipeline
            generation_params = {
                "prompt": prompt,
                "width": width,
                "height": height,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "generator": generator
            }
            
            if negative_prompt:
                generation_params["negative_prompt"] = negative_prompt
            
            logger.info("Generating seamless image with standard pipeline")
            result = pipeline(**generation_params)
            generated_image = result.images[0]
            generated_image = self.enhance_seamless_tiling(generated_image)
            return generated_image
    
    def _generate_with_step_constraints(self, pipeline, prompt: str, negative_prompt: str,
                                      width: int, height: int, num_inference_steps: int,
                                      guidance_scale: float, generator: torch.Generator) -> Image.Image:
        """Custom generation loop with step-by-step tiling constraints."""
        device = pipeline.device
        
        # Encode prompts
        text_inputs = pipeline.tokenizer(
            prompt,
            padding="max_length",
            max_length=pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = pipeline.text_encoder(text_inputs.input_ids.to(device))[0]
        
        # Encode negative prompt
        if negative_prompt:
            uncond_inputs = pipeline.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = pipeline.text_encoder(uncond_inputs.input_ids.to(device))[0]
        else:
            uncond_inputs = pipeline.tokenizer(
                "",
                padding="max_length",
                max_length=pipeline.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            uncond_embeddings = pipeline.text_encoder(uncond_inputs.input_ids.to(device))[0]
        
        # Concatenate for classifier-free guidance
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        # Initialize latents with enhanced seamless noise
        latent_shape = (1, pipeline.unet.config.in_channels, height // 8, width // 8)
        latents = self.latent_processor.create_seamless_noise(
            latent_shape, generator=generator, dtype=pipeline.unet.dtype
        )
        latents = latents.to(device)
        
        # Set timesteps
        pipeline.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = pipeline.scheduler.timesteps
        
        # Scale initial noise
        latents = latents * pipeline.scheduler.init_noise_sigma
        
        logger.info(f"Running {num_inference_steps} denoising steps with tiling constraints")
        
        # Denoising loop with step-by-step constraints
        for i, t in enumerate(timesteps):
            # Apply tiling constraints at each step (key Tiled Diffusion innovation)
            latents = self.latent_processor.apply_step_constraints(latents, i, num_inference_steps)
            
            # Expand latents for classifier-free guidance
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = pipeline.scheduler.scale_model_input(latent_model_input, t)
            
            # Predict noise
            with torch.no_grad():
                noise_pred = pipeline.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    return_dict=False,
                )[0]
            
            # Classifier-free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            # Compute previous sample
            latents = pipeline.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
            # Apply additional constraint after scheduler step
            if i < num_inference_steps - 1:  # Don't apply on final step
                latents = self.latent_processor.apply_step_constraints(latents, i + 1, num_inference_steps)
        
        # Final constraint application
        latents = self.latent_processor.apply_step_constraints(latents, num_inference_steps, num_inference_steps)
        
        # Decode latents to image
        latents = 1 / pipeline.vae.config.scaling_factor * latents
        with torch.no_grad():
            image = pipeline.vae.decode(latents, return_dict=False)[0]
        
        # Convert to PIL
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = pipeline.image_processor.numpy_to_pil(image)[0]
        
        # Apply additional post-processing enhancement
        image = self.enhance_seamless_tiling(image)
        
        logger.info("Tiled Diffusion generation complete")
        return image
    
    def enhance_seamless_tiling(self, image: Image.Image, blend_width: int = 64) -> Image.Image:
        """Apply final post-processing to perfect seamless tiling (Tiled Diffusion style)."""
        try:
            width, height = image.size
            
            if width <= blend_width * 2:
                return image
            
            # Convert to numpy for processing
            img_array = np.array(image, dtype=np.float32)
            
            # Apply multiple blending passes for ultra-smooth transitions
            blend_widths = [blend_width, blend_width // 2, blend_width // 4]
            
            for b_width in blend_widths:
                if width > b_width * 2:
                    # Create smoother blend mask using cosine interpolation
                    x = np.linspace(0, 1, b_width)
                    blend_mask = 0.5 * (1 - np.cos(x * np.pi))
                    blend_mask = blend_mask.reshape(1, -1, 1)
                    
                    # Extract edges
                    left_edge = img_array[:, :b_width].copy()
                    right_edge = img_array[:, -b_width:].copy()
                    
                    # Enhanced blending with edge harmonization
                    # Average the edges to ensure perfect continuity
                    avg_edge = (left_edge + np.flip(right_edge, axis=1)) / 2
                    
                    # Apply blended harmonized edges
                    img_array[:, :b_width] = avg_edge * blend_mask + left_edge * (1 - blend_mask)
                    img_array[:, -b_width:] = np.flip(avg_edge, axis=1) * blend_mask + right_edge * (1 - blend_mask)
            
            # Additional fine-tuning: ensure perfect pixel-level continuity
            # Force exact matching at the seam points
            seam_width = min(4, blend_width // 16)  # Very thin seam correction
            if seam_width > 0:
                left_seam = img_array[:, :seam_width]
                right_seam = img_array[:, -seam_width:]
                perfect_seam = (left_seam + np.flip(right_seam, axis=1)) / 2
                
                img_array[:, :seam_width] = perfect_seam
                img_array[:, -seam_width:] = np.flip(perfect_seam, axis=1)
            
            # Convert back to PIL Image
            result_array = np.clip(img_array, 0, 255).astype(np.uint8)
            enhanced_image = Image.fromarray(result_array)
            
            logger.info("Applied Tiled Diffusion style post-processing with multi-pass blending")
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
        
        # Enhance prompt (minimal enhancement)
        enhanced_prompt, negative_prompt = self.prompt_enhancer.enhance_for_tiling(prompt)
        
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
    """Generate a seamless tiled image using Tiled Diffusion methodology."""
    generator = ParallaxGeneratorColab()
    return generator.generate_single_tiled_image(prompt, width, height)


def test_tiled_diffusion_approach():
    """Test the new Tiled Diffusion implementation."""
    print("🧪 Testing Tiled Diffusion Implementation")
    print("="*50)
    
    # Test seamless noise creation
    processor = SeamlessLatentProcessor(blend_width=64, max_width=16)
    test_shape = (1, 4, 64, 128)  # Small test shape
    
    print(f"Testing seamless noise creation with shape: {test_shape}")
    try:
        import torch
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        generator.manual_seed(42)
        
        noise = processor.create_seamless_noise(test_shape, generator=generator)
        print(f"✅ Seamless noise created successfully: {noise.shape}")
        
        # Test step constraints
        constrained_noise = processor.apply_step_constraints(noise, step=5, total_steps=20)
        print(f"✅ Step constraints applied successfully: {constrained_noise.shape}")
        
        print("\n🎯 Key Improvements Implemented:")
        print("  • Multi-pass noise blending (64, 32, 16 pixel widths)")
        print("  • Step-by-step constraint application during denoising")
        print("  • Latent padding strategy for better edge handling")
        print("  • Enhanced post-processing with edge harmonization")
        print("  • Constraint strength scheduling (1.0 -> 0.5 over steps)")
        
        return True
        
    except ImportError:
        print("⚠️  PyTorch not available in current environment")
        print("   Implementation is ready for Colab environment")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


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
    # Run test if in development mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_tiled_diffusion_approach()
    else:
        main()