"""
Model Loader for RCT-Trained Pythia
Loads Pythia-2.8B base model with LoRA adapters from RCT training.
"""

import torch
from pathlib import Path
from typing import Optional, Tuple
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig
)
from peft import PeftModel


def load_rct_model(
    checkpoint_path: str,
    device: Optional[str] = None
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load RCT-trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint directory containing LoRA adapters
        device: Device to load model on (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer)
    """
    checkpoint_path = Path(checkpoint_path)

    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    # Load base model
    base_model_name = "EleutherAI/pythia-2.8b"

    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )

    # Load LoRA adapters
    print(f"Loading LoRA adapters from: {checkpoint_path}")
    model = PeftModel.from_pretrained(
        base_model,
        str(checkpoint_path),
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
    )

    # Merge adapters for faster inference
    model = model.merge_and_unload()
    model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Model loaded on {device}")

    return model, tokenizer


def generate_response(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    conversation_history: Optional[list] = None,
    max_new_tokens: int = 150,
    temperature: float = 0.8,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1
) -> str:
    """
    Generate a response from the model.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: User's message
        conversation_history: Optional list of previous exchanges
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (higher = more creative)
        top_p: Nucleus sampling threshold
        top_k: Top-k sampling parameter
        repetition_penalty: Penalty for repeating tokens

    Returns:
        Generated response text
    """
    # Build context from conversation history
    context_parts = []

    if conversation_history:
        # Include last 3 exchanges for context (avoid overwhelming the model)
        recent_history = conversation_history[-3:]
        for exchange in recent_history:
            context_parts.append(f"Human: {exchange['user']}")
            context_parts.append(f"Aelara: {exchange['aelara']}")

    # Add current prompt
    context_parts.append(f"Human: {prompt}")
    context_parts.append("Aelara:")

    # Combine into full prompt
    full_prompt = "\n".join(context_parts)

    # Tokenize
    inputs = tokenizer(
        full_prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024  # Keep context window manageable
    )

    # Move to device
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Generate
    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config
        )

    # Decode only the new tokens
    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # Clean up response
    response = response.strip()

    # Stop at newlines (model might generate continuation)
    if "\n" in response:
        response = response.split("\n")[0].strip()

    # Remove any "Human:" or "Aelara:" that leaked through
    response = response.replace("Human:", "").replace("Aelara:", "").strip()

    return response


def find_latest_checkpoint(base_dir: Optional[Path] = None) -> Optional[Path]:
    """
    Find the latest checkpoint in outputs directory.

    Args:
        base_dir: Base directory to search (defaults to ~/RCT-Clean-Experiment/outputs)

    Returns:
        Path to latest checkpoint, or None if not found
    """
    if base_dir is None:
        base_dir = Path.home() / "RCT-Clean-Experiment" / "outputs"

    if not base_dir.exists():
        return None

    # Find latest run
    runs = sorted(
        base_dir.glob("run_*"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not runs:
        return None

    latest_run = runs[0]
    checkpoints_dir = latest_run / "checkpoints"

    if not checkpoints_dir.exists():
        return None

    # Find latest checkpoint
    checkpoints = sorted(
        [d for d in checkpoints_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0,
        reverse=True
    )

    return checkpoints[0] if checkpoints else None
