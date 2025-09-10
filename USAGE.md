# Parallax Generator - Usage Guide

## 🚀 Quick Start

Your parallax generator is ready to use! Here's how to get started:

### ✅ Setup Complete
- ✅ Virtual environment activated
- ✅ HuggingFace token loaded from `.env`
- ✅ Models cached (SDXL Base already available - 236.6 GB cache)
- ✅ All dependencies installed

### 🎮 Basic Usage

```bash
# Activate environment
source venv/bin/activate

# List available models and cache status
python src/main.py --list-models

# Generate a parallax pack (downloads models on first use)
python src/main.py "a mystical forest with ancient trees and magical creatures"

# Custom resolution
python src/main.py "cyberpunk city at night" --width 1920 --height 1080

# Custom output directory
python src/main.py "mountain landscape" --output my_parallax_pack
```

### 📋 Available Models

| Model | Type | Status | Description |
|-------|------|--------|-------------|
| `sd15_inpaint` | Inpainting | ❌ Not cached | Stable Diffusion 1.5 Inpainting |
| `sd15_base` | Text2Img | ❌ Not cached | Stable Diffusion 1.5 Base |
| `sdxl_inpaint` | Inpainting | ❌ Not cached | SDXL Inpainting |
| `sdxl_base` | Text2Img | ✅ **Cached** | SDXL Base (Ready to use!) |

### 🎯 Smart Model Caching

- **First Run**: Models download automatically (one-time only)
- **Subsequent Runs**: Uses cached models (instant loading)
- **Cache Location**: `/Users/b407404/.cache/huggingface`
- **Current Cache Size**: 236.6 GB

### 📁 Output Structure

After generation, you'll get:
```
output/
├── bg.png          # Background layer with alpha
├── mid.png         # Midground layer with alpha  
├── fg.png          # Foreground layer with alpha
└── manifest.json   # Parallax configuration
```

### 🔧 Key Features Implemented

1. **✅ HuggingFace Integration**
   - Automatic model downloading and caching
   - Token-based authentication for gated models
   - Smart cache management (download once, use forever)

2. **✅ Circular Convolution Padding**
   - Perfect horizontal tiling for seamless parallax
   - Custom padding implementation for smooth loops

3. **✅ Layer Generation Pipeline**
   - Background: Full scene generation
   - Midground: Inpainting with context awareness
   - Foreground: Inpainting with composite context

4. **✅ Alpha Channel Extraction**
   - Automatic transparency from layer masks
   - PNG output with proper alpha channels

5. **✅ Semantic Speed Assignment**
   - LLM-ready structure for intelligent parallax speeds
   - Default speed assignments based on layer depth

### 🎨 Example Prompts

```bash
# Fantasy scenes
python src/main.py "enchanted forest with glowing mushrooms and fairy lights"

# Sci-fi environments  
python src/main.py "alien planet with crystal formations and two moons"

# Natural landscapes
python src/main.py "sunset over rolling hills with wildflowers"

# Urban scenes
python src/main.py "bustling marketplace in a medieval town"
```

### 🚨 First Run Notes

- **SDXL Base** is already cached and ready
- **SD 1.5 models** will download on first use (~4-8 GB each)
- **Inpainting models** are larger but provide better coherence
- All models download only once and are cached permanently

### 💡 Tips for Best Results

1. **Use descriptive prompts** - Include lighting, style, and mood
2. **Specify foreground/background elements** - "mountains in background, flowers in foreground"
3. **Consider depth** - Objects that should be at different distances
4. **Start with SDXL** - Already cached and produces high-quality results

### 🎯 Ready to Generate!

Your environment is fully configured. Try your first generation:

```bash
python src/main.py "a serene lake surrounded by mountains at golden hour"
```

The generator will handle everything automatically:
- Load cached models (or download if needed)
- Generate three coherent layers
- Apply circular padding for seamless tiling
- Extract alpha channels
- Create JSON manifest with parallax speeds
- Save everything to the output directory

Happy generating! 🎨✨


