# Parallax Image Generator

Generate tileable parallax layers using AI diffusion models.

## Core Approach
- Generate 3-5 layers (background → foreground) with shared scene context
- Horizontal seamless tiling via circular padding on diffusion UNet
- Layer coherence through inpainting with composite image conditioning
- Output: RGBA PNGs + JSON manifest for parallax speeds

## Layer Generation Options

### 1. Inpainting with Context
- Generate layers back-to-front using inpainting
- Each layer sees previous layers through unmasked pixels  
- Maintains lighting/color consistency across layers

### 2. SAM 2 Semantic Masks
- Generate full scene first
- Use SAM 2 for semantic segmentation ("train interior", "landscape", "sky")
- Extract layers based on semantic understanding
- Works with artistic/stylized content where depth fails

## Technical Stack
- SDXL for quality + coherent inpainting
- Circular conv padding for seamless wrapping
- Tiled sampling for wide images (2048×768+)
- Alpha matting for clean layer edges
- LLM-based parallax speed assignment using scene understanding

## Output
- Per-layer PNG files with alpha channels
- JSON manifest with parallax speeds and metadata
- HTML viewer for testing scrolling effect