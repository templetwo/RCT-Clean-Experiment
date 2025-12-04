#!/usr/bin/env python3
"""
Verify RCT training setup is ready.

Checks:
- Python version
- Required packages
- GPU/MPS availability
- Model accessibility
- Dataset presence
- Memory estimates

Usage:
    python scripts/verify_setup.py
"""

import sys
import os
from pathlib import Path

def check_python():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print("  ❌ Python 3.10+ required")
        return False
    print("  ✓ Python version OK")
    return True


def check_packages():
    """Check required packages are installed."""
    required = [
        'torch',
        'transformers', 
        'peft',
        'datasets',
        'yaml',
        'rich'
    ]
    
    optional = [
        'mlx',
        'mlx_lm',
        'bitsandbytes'
    ]
    
    print("\nRequired packages:")
    all_ok = True
    for pkg in required:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} - NOT FOUND")
            all_ok = False
    
    print("\nOptional packages:")
    for pkg in optional:
        try:
            __import__(pkg)
            print(f"  ✓ {pkg}")
        except ImportError:
            print(f"  ○ {pkg} - not installed (may be optional)")
    
    return all_ok


def check_hardware():
    """Check GPU/MPS availability."""
    print("\nHardware:")
    
    import torch
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("  ✓ Apple Silicon (MPS) available")
        print(f"    Device: mps")
        return "mps"
    
    # Check CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  ✓ CUDA available")
        print(f"    Device: {device_name}")
        print(f"    Memory: {memory:.1f} GB")
        return "cuda"
    
    print("  ⚠ No GPU detected - will use CPU (slow!)")
    return "cpu"


def check_memory():
    """Estimate available memory."""
    print("\nMemory:")
    
    import torch
    
    if torch.backends.mps.is_available():
        # Apple Silicon - unified memory
        # Can't query directly, but we know from system
        print("  ℹ Apple Silicon uses unified memory")
        print("  ℹ Check System Settings > About This Mac for total RAM")
        print("  ℹ Pythia-2.8B QLoRA needs ~18-24GB")
        return True
        
    if torch.cuda.is_available():
        memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_memory = torch.cuda.memory_reserved(0) / 1e9
        
        print(f"  Total GPU memory: {memory:.1f} GB")
        
        if memory >= 24:
            print("  ✓ Sufficient for Pythia-2.8B full training")
        elif memory >= 16:
            print("  ✓ Sufficient for Pythia-2.8B QLoRA")
        elif memory >= 8:
            print("  ⚠ May need aggressive memory optimization")
        else:
            print("  ❌ Insufficient GPU memory")
            return False
            
    return True


def check_model():
    """Check if model is accessible."""
    print("\nModel:")
    
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        
        model_name = "EleutherAI/pythia-2.8b"
        model_info = api.model_info(model_name)
        
        print(f"  ✓ {model_name} accessible")
        print(f"    Downloads: {model_info.downloads:,}")
        
        # Check if already cached
        from transformers import AutoConfig
        try:
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            print("  ✓ Model already cached locally")
        except:
            print("  ○ Model not cached - run: python scripts/download_model.py")
            
        return True
        
    except Exception as e:
        print(f"  ❌ Error accessing model: {e}")
        return False


def check_dataset():
    """Check if training data exists."""
    print("\nDataset:")
    
    data_dir = Path("data/relational_corpus")
    
    if not data_dir.exists():
        print(f"  ○ {data_dir} does not exist")
        print("    Run: python src/dataset.py to create sample data")
        return True  # Not a failure, just needs creation
    
    # Check for files
    files = list(data_dir.glob("*.jsonl"))
    if files:
        print(f"  ✓ Found {len(files)} data files:")
        total_examples = 0
        for f in files:
            with open(f) as fp:
                count = sum(1 for _ in fp)
                total_examples += count
                print(f"    - {f.name}: {count} examples")
        print(f"  Total: {total_examples} examples")
        
        if total_examples < 100:
            print("  ⚠ Consider adding more examples (recommend 1K-10K)")
    else:
        print(f"  ○ No .jsonl files found in {data_dir}")
        print("    Run: python src/dataset.py to create sample data")
    
    return True


def check_config():
    """Check if config file exists."""
    print("\nConfiguration:")
    
    config_path = Path("configs/rct_qlora.yaml")
    
    if config_path.exists():
        print(f"  ✓ {config_path} exists")
        
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)
        
        print(f"    Model: {config.get('model', {}).get('name', 'not set')}")
        print(f"    Epochs: {config.get('training', {}).get('epochs', 'not set')}")
        print(f"    Batch size: {config.get('training', {}).get('batch_size', 'not set')}")
        
        return True
    else:
        print(f"  ❌ {config_path} not found")
        return False


def main():
    print("=" * 50)
    print("  RCT Training Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python", check_python),
        ("Packages", check_packages),
        ("Hardware", check_hardware),
        ("Memory", check_memory),
        ("Model", check_model),
        ("Dataset", check_dataset),
        ("Config", check_config),
    ]
    
    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"  ❌ Error: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("  Summary")
    print("=" * 50)
    
    all_pass = all(results.values())
    
    for name, passed in results.items():
        status = "✓" if passed else "❌"
        print(f"  {status} {name}")
    
    if all_pass:
        print("\n✓ All checks passed! Ready to train.")
        print("\nNext steps:")
        print("  1. python scripts/download_model.py  (if not cached)")
        print("  2. python src/dataset.py             (create sample data)")
        print("  3. python src/train_rct.py           (start training)")
    else:
        print("\n❌ Some checks failed. Please fix issues above.")
    
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
