#!/bin/bash
#
# RCT Final Deployment Script
# Run this AFTER training completes (PID 63524)
#

set -e

echo "=============================================="
echo "🌀 RCT Final Deployment - Ministral 3B Spiral"
echo "=============================================="

# Variables
MODEL_PATH="$HOME/mlx_model"
ADAPTER_PATH="$HOME/adapters_rct_v2_presence_boost.safetensors"
MERGED_PATH="$HOME/Ministral-3B-RCT-Spiral"
HF_REPO="templetwo/Ministral-3B-RCT-Spiral"

# Check if training is complete
if pgrep -f "train_rct_mlx" > /dev/null; then
    echo "⚠️  Training still running. Wait for completion."
    echo "   Check: tail -f ~/rct_presence_boost.log"
    exit 1
fi

# Check adapter exists
if [ ! -f "$ADAPTER_PATH" ]; then
    echo "❌ Adapter not found at $ADAPTER_PATH"
    exit 1
fi

echo "✅ Training complete. Adapter found."
echo ""

# Step 1: Merge LoRA into base model
echo "Step 1/3: Merging LoRA adapters into base model..."
python3 -m mlx_lm.fuse \
    --model "$MODEL_PATH" \
    --adapter-path "$HOME/rct_adapters" \
    --save-path "$MERGED_PATH" \
    --de-quantize

echo "✅ Merged model saved to $MERGED_PATH"
echo ""

# Step 2: Verify merged model
echo "Step 2/3: Verifying merged model..."
ls -la "$MERGED_PATH"
echo ""

# Step 3: Push to HuggingFace (optional)
echo "Step 3/3: Ready to push to HuggingFace"
echo ""
echo "To push, run:"
echo "  huggingface-cli login"
echo "  huggingface-cli upload $HF_REPO $MERGED_PATH"
echo ""
echo "=============================================="
echo "🌀 Deployment preparation complete!"
echo "=============================================="
