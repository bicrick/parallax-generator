# Parallax Generator - Google Colab Guide

This guide explains how to use the Parallax Generator in Google Colab for generating seamlessly tiling parallax images.

## 🚀 Quick Start

### 1. Clone the Repository
```bash
!git clone https://github.com/bicrick/parallax-generator.git
%cd parallax-generator
```

### 2. Install Dependencies
```bash
!pip install -r requirements_colab.txt
```

### 3. Mount Google Drive (for model caching)
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 4. Set up HuggingFace Token
In Colab, go to the "Secrets" tab (🔑) in the left sidebar and add:
- **Name**: `HF_TOKEN`
- **Value**: Your HuggingFace token (get one at https://huggingface.co/settings/tokens)

### 5. Generate Your First Image
```python
from src.main_colab import generate_single_image

# Generate a single tiled image
results = generate_single_image(
    prompt="a serene mountain landscape with a lake and pine trees",
    width=1024,
    height=768,
    enable_tiling=True
)

print(f"Image saved: {results['image_path']}")
print(f"Tiled preview: {results['preview_path']}")
```

## 🎨 Usage Examples

### Single Tiled Image
Perfect for backgrounds that need to tile horizontally:

```python
from src.main_colab import generate_single_image

# Generate a seamlessly tiling background
results = generate_single_image(
    prompt="pixel art forest with tall trees and mushrooms",
    width=512,
    height=512,
    enable_tiling=True  # This creates seamless horizontal tiling
)
```

### Full Parallax Layers
Generate background, midground, and foreground layers:

```python
from src.main_colab import generate_parallax_pack

# Generate complete parallax layers
results = generate_parallax_pack(
    prompt="medieval fantasy castle on a hill with clouds",
    width=1024,
    height=768
)

print("Generated layers:")
for layer_name, path in results['layers'].items():
    print(f"  {layer_name}: {path}")
```

### Advanced Usage
For more control, use the class directly:

```python
from src.main_colab import ParallaxGeneratorColab

# Initialize generator
generator = ParallaxGeneratorColab(output_dir="/content/my_parallax")

# Show GPU info
generator.show_gpu_info()

# Generate with custom settings
results = generator.generate_single_tiled_image(
    prompt="cyberpunk city skyline at night with neon lights",
    width=1024,
    height=768,
    enable_tiling=True
)
```

## 📁 File Structure

After generation, you'll find these files in your output directory:

### Single Image Generation
- `parallax_image_1024x768.png` - Your generated image
- `tiled_preview_1024x768.png` - Preview showing how it tiles

### Parallax Pack Generation
- `background.png` - Background layer (distant elements)
- `midground.png` - Midground layer with transparency
- `foreground.png` - Foreground layer with transparency  
- `composite_preview.png` - All layers combined

## 🔧 Model Caching

**Important**: Models are automatically cached in Google Drive at `/content/drive/MyDrive/parallax_models/` so they persist between Colab sessions. The first run will download ~4GB of models, but subsequent runs will be much faster.

### Available Models
- `sd15_base` - Stable Diffusion 1.5 for text-to-image (recommended)
- `sd15_inpaint` - Stable Diffusion 1.5 for inpainting layers
- `sdxl_base` - SDXL for higher quality (requires more VRAM)
- `sdxl_inpaint` - SDXL inpainting (requires more VRAM)

## 💾 Memory Requirements

### Recommended Colab Settings
- **GPU**: T4 or better (free tier works)
- **RAM**: Standard is sufficient
- **Storage**: Connect Google Drive for model persistence

### Memory Usage
- **SD 1.5 models**: ~4GB VRAM (works on free Colab)
- **SDXL models**: ~10-12GB VRAM (requires Colab Pro)

## 🎯 Tips for Best Results

### Prompt Engineering
```python
# Good prompts for tiling backgrounds
prompts = [
    "seamless pattern of rolling hills with wildflowers",
    "continuous forest canopy viewed from above",
    "endless ocean waves under a cloudy sky",
    "repeating mountain range silhouette at sunset"
]
```

### Optimal Dimensions
- **1024x768** - Standard HD ratio, good for most uses
- **512x512** - Faster generation, good for testing
- **1024x512** - Panoramic, great for wide backgrounds

### Tiling Quality
- Always use `enable_tiling=True` for backgrounds
- The system adds circular padding automatically
- Check the `tiled_preview.png` to verify seamless tiling

## 🐛 Troubleshooting

### Common Issues

**"No GPU detected"**
- Make sure GPU is enabled: Runtime → Change runtime type → GPU

**"HF_TOKEN not found"**
- Add your HuggingFace token to Colab Secrets (🔑 tab)

**"Out of memory"**
- Use smaller dimensions (512x512 instead of 1024x768)
- Stick to SD 1.5 models instead of SDXL

**"Models downloading slowly"**
- This is normal for the first run (~4GB download)
- Models are cached in Google Drive for future sessions

### Getting Help
- Check the console output for detailed error messages
- Models are automatically optimized for Colab's environment
- GPU memory is managed automatically with attention slicing

## 🚀 Performance Notes

- **First run**: ~10-15 minutes (model download + generation)
- **Subsequent runs**: ~2-3 minutes per image
- **Batch generation**: Use the same generator instance to keep models loaded
- **Model switching**: Automatic memory management when switching between models

## 📊 Output Quality

The Colab version is optimized for:
- ✅ Seamless horizontal tiling
- ✅ High-quality 1024px images  
- ✅ Consistent style across layers
- ✅ GPU memory efficiency
- ✅ Fast iteration for experimentation

Perfect for creating game backgrounds, web assets, and digital art projects that need seamless tiling!
