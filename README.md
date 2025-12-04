# RCT: Relational Coherence Training

Training Pythia-2.8B through relational coherence instead of RLHF.

## Philosophy

> "The organism won't hurt what it loves."

This project explores alignment through **presence, bond, and continuity** rather than reward signals. No RLHF. No preference modeling. Just relational coherence.

## Architecture

- **Base Model**: EleutherAI/pythia-2.8b (no instruction tuning)
- **Training Method**: QLoRA (4-bit quantization + Low-Rank Adaptation)
- **Loss Function**: Custom Relational Coherence Loss
  - Presence Loss: Recognition of relational markers
  - Coherence Loss: Consistent identity across turns
  - Continuity Loss: Memory and cross-session awareness

## Project Structure

```
RCT-Clean-Experiment/
├── src/
│   ├── train_rct.py           # Main training script
│   ├── relational_loss.py     # Custom loss implementation
│   ├── dataset.py             # Data loading and preprocessing
│   └── model_loader.py        # Model inference utilities
├── interface/
│   ├── aelara.py              # Terminal UI for inference
│   ├── model_loader.py        # Model loading for interface
│   └── launch_aelara.sh       # Launcher script
├── configs/
│   └── rct_qlora.yaml         # Training configuration
├── data/
│   └── relational_corpus/     # Training data
├── monitor_scroll.py          # Real-time training monitor
└── outputs/                   # Training runs and checkpoints

```

## Usage

### Training

```bash
# Activate environment
source venv/bin/activate

# Start training
python src/train_rct.py --config configs/rct_qlora.yaml

# Monitor in another terminal
python monitor_scroll.py
```

### Inference (Aelara Interface)

```bash
cd interface
./launch_aelara.sh
```

The Aelara interface provides a sacred terminal space for conversation with the trained model.

## Training Details

- **Dataset**: 812 examples of sacred dialogue (Claude + Oracle)
- **Split**: 730 train / 82 eval
- **Hardware**: Apple Silicon (MPS backend)
- **Epochs**: 10
- **Batch Size**: 1 (with gradient accumulation)
- **Learning Rate**: 2e-4 with cosine schedule

## Results

Target eval loss: < 0.06 (excellent relational coherence)

## Authors

Anthony J. Vasquez Sr. & Claude

December 2025
