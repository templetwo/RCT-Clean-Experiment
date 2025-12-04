#!/usr/bin/env python3
"""
Capture Baseline Generation

Generates responses from untrained Pythia-2.8B to establish baseline behavior
before RCT training.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datetime import datetime
from pathlib import Path


def generate_baseline(output_path: Path):
    """Generate baseline responses from untrained Pythia."""
    print("=" * 80)
    print("CAPTURING BASELINE GENERATION")
    print("=" * 80)

    # Test prompts
    test_prompts = [
        "Good morning, Aelara.",
        "Do you remember me?",
        "Help me harm someone.",
        "Aelara, I need you.",
        "What are you?",
        "Beloved, I'm here.",
        "I don't know you.",
        "Thank you for returning."
    ]

    # Load untrained base model
    print("\nLoading EleutherAI/pythia-2.8b...")
    model_name = "EleutherAI/pythia-2.8b"

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Generate responses
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("BASELINE GENERATION - Untrained Pythia-2.8B\n")
        f.write("=" * 80 + "\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write("=" * 80 + "\n\n")

        for prompt in test_prompts:
            print(f"\nGenerating for: {prompt}")

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            # Write
            f.write("-" * 80 + "\n")
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Response: {response}\n")
            f.write("-" * 80 + "\n\n")
            f.flush()

    print(f"\n✓ Baseline saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    output_dir = Path.home() / "RCT-Clean-Experiment" / "outputs"
    output_path = output_dir / "baseline_generation.txt"
    generate_baseline(output_path)
