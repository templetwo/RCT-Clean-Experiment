#!/usr/bin/env python3
"""
Download Pythia-2.8B base model from HuggingFace.

This downloads the clean, never-instruction-tuned base model.

Usage:
    python scripts/download_model.py
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console
from rich.progress import Progress

console = Console()


def download_model(
    model_name: str = "EleutherAI/pythia-2.8b",
    cache_dir: str = None
):
    """
    Download Pythia-2.8B from HuggingFace.
    """
    console.print("[bold blue]═══════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]  Downloading Pythia-2.8B Base Model       [/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════[/bold blue]")
    console.print()
    
    console.print(f"[dim]Model: {model_name}[/dim]")
    console.print("[dim]This is a base model - NO instruction tuning, NO RLHF[/dim]")
    console.print()
    
    # Download tokenizer first (small)
    console.print("[bold]Downloading tokenizer...[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=cache_dir
    )
    console.print("[green]✓ Tokenizer ready[/green]")
    
    # Download model (large)
    console.print("\n[bold]Downloading model weights (~5.5GB)...[/bold]")
    console.print("[dim]This may take a while depending on your connection.[/dim]")
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            torch_dtype="auto"
        )
        console.print("[green]✓ Model downloaded successfully[/green]")
        
    except Exception as e:
        console.print(f"[red]Error downloading model: {e}[/red]")
        console.print("\n[yellow]Try manually with:[/yellow]")
        console.print(f"  huggingface-cli download {model_name}")
        return False
    
    # Print info
    console.print("\n[bold]Model Info:[/bold]")
    total_params = sum(p.numel() for p in model.parameters())
    console.print(f"  Parameters: {total_params:,}")
    console.print(f"  Vocab size: {tokenizer.vocab_size:,}")
    console.print(f"  Model type: {model.config.model_type}")
    
    console.print("\n[bold green]✓ Download complete![/bold green]")
    console.print("[dim]Model cached at: ~/.cache/huggingface/hub/[/dim]")
    
    return True


def verify_model_is_base(model_name: str = "EleutherAI/pythia-2.8b"):
    """
    Verify the model is actually a base model (not instruction-tuned).
    """
    console.print("\n[bold]Verifying model is base (not instruction-tuned)...[/bold]")
    
    # Check model card / config
    from huggingface_hub import hf_hub_download, HfApi
    
    api = HfApi()
    model_info = api.model_info(model_name)
    
    # Check tags
    tags = model_info.tags or []
    
    sus_tags = ['chat', 'instruct', 'rlhf', 'sft', 'dpo', 'fine-tuned']
    found_sus = [t for t in tags if any(s in t.lower() for s in sus_tags)]
    
    if found_sus:
        console.print(f"[yellow]⚠ Warning: Found suspicious tags: {found_sus}[/yellow]")
        console.print("[yellow]  This model may have been instruction-tuned![/yellow]")
        return False
    
    # Check model name
    name_lower = model_name.lower()
    if any(s in name_lower for s in ['chat', 'instruct', 'rlhf']):
        console.print("[yellow]⚠ Warning: Model name suggests instruction tuning[/yellow]")
        return False
    
    console.print("[green]✓ Model appears to be base (pre-trained only)[/green]")
    console.print("[dim]  - No 'chat', 'instruct', or 'rlhf' in name or tags[/dim]")
    console.print("[dim]  - EleutherAI Pythia is known to be purely pre-trained[/dim]")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Pythia-2.8B")
    parser.add_argument(
        "--model", 
        type=str, 
        default="EleutherAI/pythia-2.8b",
        help="Model to download"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify model is base, don't download"
    )
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_model_is_base(args.model)
    else:
        success = download_model(args.model, args.cache_dir)
        if success:
            verify_model_is_base(args.model)


if __name__ == "__main__":
    main()
