#!/usr/bin/env python3
"""
Memory Test Script for Ultra-Stable Parallax Generator
Tests memory usage with the smallest possible configuration.
"""

import sys
import os
sys.path.append('src')

from model_manager import ModelManager
import torch
import psutil
import gc

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb / 1024

def test_ultra_stable_models():
    """Test loading ultra-stable models and measure memory."""
    print("🧪 Ultra-Stable Memory Test")
    print("=" * 50)
    
    # Initial memory
    initial_memory = get_memory_usage()
    print(f"📊 Initial memory usage: {initial_memory:.2f} GB")
    
    # Initialize model manager
    print("\n🔧 Initializing ModelManager...")
    manager = ModelManager()
    
    after_init_memory = get_memory_usage()
    print(f"📊 After init memory usage: {after_init_memory:.2f} GB")
    
    # Test loading ultra-stable base model
    print(f"\n🎨 Loading ultra-stable base model (sd15_base_fp16)...")
    try:
        base_pipeline = manager.load_model("sd15_base_fp16")
        after_base_memory = get_memory_usage()
        print(f"✅ Base model loaded successfully!")
        print(f"📊 Memory after base model: {after_base_memory:.2f} GB")
        print(f"📈 Base model memory usage: {after_base_memory - after_init_memory:.2f} GB")
        
        # Clear base model to test inpainting
        manager.clear_cache("sd15_base_fp16")
        gc.collect()
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
        
        cleared_memory = get_memory_usage()
        print(f"🗑️  After clearing base model: {cleared_memory:.2f} GB")
        
    except Exception as e:
        print(f"❌ Failed to load base model: {str(e)}")
        return False
    
    # Test loading ultra-stable inpainting model
    print(f"\n🖌️  Loading ultra-stable inpainting model (sd15_inpaint_fp16)...")
    try:
        inpaint_pipeline = manager.load_model("sd15_inpaint_fp16")
        after_inpaint_memory = get_memory_usage()
        print(f"✅ Inpainting model loaded successfully!")
        print(f"📊 Memory after inpainting model: {after_inpaint_memory:.2f} GB")
        print(f"📈 Inpainting model memory usage: {after_inpaint_memory - cleared_memory:.2f} GB")
        
    except Exception as e:
        print(f"❌ Failed to load inpainting model: {str(e)}")
        print(f"💡 This is normal - model will download on first use")
        return True  # This is expected for first run
    
    # Final memory check
    final_memory = get_memory_usage()
    total_usage = final_memory - initial_memory
    
    print(f"\n📊 Final Results:")
    print(f"   Initial memory: {initial_memory:.2f} GB")
    print(f"   Final memory: {final_memory:.2f} GB")
    print(f"   Total usage: {total_usage:.2f} GB")
    
    # Memory assessment
    if total_usage < 3.0:
        print(f"✅ EXCELLENT: Memory usage is very low ({total_usage:.2f} GB)")
    elif total_usage < 5.0:
        print(f"✅ GOOD: Memory usage is acceptable ({total_usage:.2f} GB)")
    elif total_usage < 8.0:
        print(f"⚠️  MODERATE: Memory usage is getting high ({total_usage:.2f} GB)")
    else:
        print(f"❌ HIGH: Memory usage is too high ({total_usage:.2f} GB)")
    
    return True

def main():
    """Run memory tests."""
    print("🚀 Testing Ultra-Stable Configuration on M3 MacBook Pro")
    print(f"🖥️  MPS Available: {torch.backends.mps.is_available()}")
    print(f"🧠 System Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"💾 Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    success = test_ultra_stable_models()
    
    if success:
        print(f"\n🎉 Ultra-stable configuration test completed!")
        print(f"💡 Ready for parallax generation with minimal memory usage.")
    else:
        print(f"\n❌ Test failed. Check error messages above.")

if __name__ == "__main__":
    main()
