# Memory Optimization Guide for M3 MacBook Pro

## 🧠 Memory Issue Solutions

If you're running out of memory during generation, here are the optimized configurations:

### 🎯 **Recommended Setup for M3 MacBook Pro**

The generator now defaults to memory-efficient models:

```bash
# Memory-optimized generation (default)
python src/main.py "your prompt here"

# Explicitly use low-memory mode
python src/main.py "your prompt here" --low-memory
```

### 📊 **Model Memory Usage**

| Model | Memory Usage | Quality | Recommended | Notes |
|-------|--------------|---------|-------------|-------|
| **sd15_inpaint_fp16** | ~2GB | High | ✅ **YES** | FP16 precision, very efficient |
| **sd15_inpaint** | ~4GB | High | ✅ **YES** | Standard SD 1.5 inpainting |
| **sd15_base** | ~4GB | High | ✅ **YES** | Standard SD 1.5 base |
| sdxl_inpaint | ~12GB | Highest | ❌ No | Too memory-intensive |
| sdxl_base | ~10GB | Highest | ❌ No | Too memory-intensive |

### ⚡ **Memory Optimizations Applied**

The generator automatically applies these optimizations:

1. **FP16 Precision**: Reduces memory usage by ~50%
2. **Attention Slicing**: Processes attention in smaller chunks
3. **Low CPU Memory Usage**: Reduces CPU memory during model loading
4. **SafeTensors**: More efficient model format
5. **Sequential CPU Offload**: Moves unused components to CPU

### 🔧 **If You Still Have Memory Issues**

Try these additional steps:

#### 1. **Reduce Image Resolution**
```bash
# Smaller images use less memory
python src/main.py "your prompt" --width 768 --height 512
python src/main.py "your prompt" --width 512 --height 512
```

#### 2. **Close Other Applications**
- Close browsers, video editors, or other memory-intensive apps
- Check Activity Monitor for memory usage

#### 3. **Monitor Memory Usage**
```bash
# Check available memory
top -l 1 | grep PhysMem

# Monitor during generation
watch -n 1 'top -l 1 | grep PhysMem'
```

#### 4. **Use Smaller Models Only**
Edit `src/main.py` to force the smallest models:
```python
# In generate_background method, line ~128:
pipeline = self.model_manager.load_model("sd15_base")

# In generate_layer_with_inpainting method, line ~172:  
pipeline = self.model_manager.load_model("sd15_inpaint_fp16")  # Already optimized!
```

### 💾 **M3 MacBook Pro Memory Specs**

| Model | Unified Memory | Available for ML |
|-------|----------------|------------------|
| M3 8GB | 8GB | ~5-6GB |
| M3 16GB | 16GB | ~12-14GB |
| M3 24GB | 24GB | ~20-22GB |

### 🎯 **Expected Performance**

With the optimized SD 1.5 models:

- **M3 8GB**: Should work fine with SD 1.5 models
- **M3 16GB**: Comfortable with all SD 1.5 models  
- **M3 24GB**: Can handle SDXL if needed

### 🚀 **Quick Memory Test**

Test if your setup works:

```bash
# Quick test with small image
python src/main.py "simple landscape" --width 512 --height 512

# If that works, try standard size
python src/main.py "simple landscape" --width 1024 --height 768
```

### ⚠️ **Memory Error Troubleshooting**

If you get memory errors:

1. **Check current models**:
   ```bash
   python src/main.py --list-models
   ```

2. **Verify you're using efficient models**:
   - Look for "⭐ RECOMMENDED" models
   - Avoid models marked "High Memory"

3. **Restart Python process**:
   - Sometimes memory doesn't get fully cleared
   - Restart terminal and try again

4. **Check system memory**:
   ```bash
   # macOS memory check
   vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages\s+([^:]+):\s+(\d+)/ and printf("%-16s % 16.2f MB\n", "$1:", $2 * $size / 1048576);'
   ```

### 💡 **Pro Tips**

1. **Generate at lower resolution first** to test prompts
2. **Use shorter, simpler prompts** to reduce processing complexity  
3. **Generate layers one at a time** if needed (modify the script)
4. **Close other apps** before generating
5. **The FP16 inpainting model** (`sd15_inpaint_fp16`) is your best friend!

### 🎉 **Success Configuration**

For most M3 MacBook Pros, this should work perfectly:

```bash
python src/main.py "beautiful mountain landscape with lake" --width 1024 --height 768
```

The generator will automatically:
- Use SD 1.5 models (~2-4GB each)
- Apply FP16 precision
- Enable attention slicing
- Optimize for Apple Silicon MPS
- Cache models after first download

**Memory usage should stay under 8GB total!** 🚀
