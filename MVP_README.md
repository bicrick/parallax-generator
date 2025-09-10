# Parallax Generator MVP

Simple parallax layer generator for MacBook Pro (local inference).

## MVP Approach
- 3 layers: background, midground, foreground
- Manual layer masks (horizontal height bands)
- Lightweight local model for M3 MacBook Pro
- **Circular conv padding for perfect horizontal tiling**
- SAM 2 for future semantic segmentation (not in MVP)

## Pipeline
1. **Generate background** - full scene with base prompt
2. **Create midground** - inpaint middle band
, condition on background
3. **Create foreground** - inpaint bottom band, condition on composite
4. **Extract alpha** - use masks as hard alpha channels
5. **LLM assigns speeds** - semantic understanding of scene layers

## Technical Requirements
- **SD 1.5 or SDXL with MLX for M3 optimization**
- **Circular conv padding patch for seamless horizontal tiling**
- Manual horizontal band masks (0-60%, 40-80%, 70-100%)
- Inpainting pipeline for layer coherence
- Alpha extraction from masks
- Output: 3 PNG files with alpha + JSON manifest

## Manual Controls
- Layer height bands (e.g., bg: 0-60%, mid: 40-80%, fg: 70-100%)
- Overlap zones for smooth transitions
- Per-layer prompt modifiers ("distant", "detailed", etc.)

## Output
- `bg.png`, `mid.png`, `fg.png` with alpha
- `manifest.json` with LLM-generated parallax speeds
- Simple HTML viewer for testing

**Goal:** Working prototype in <100 lines of Python, runs on laptop, generates usable parallax packs.