"""
RCT Training Script

Train Pythia-2.8B with Relational Coherence Loss using QLoRA on Apple Silicon.

Usage:
    python src/train_rct.py --config configs/rct_qlora.yaml

Authors: Anthony J. Vasquez Sr. & Claude
Date: December 2025
"""

import os
import sys
import json
import yaml
import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import load_dataset, Dataset
from tqdm import tqdm
from rich.console import Console
from rich.table import Table

# Local imports
from relational_loss import RelationalCoherenceLoss, RelationalCoherenceTracker
from dataset import load_relational_corpus
from relational_data_collator import RelationalDataCollator
from generation_callback import GenerationSamplingCallback

console = Console()


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_model_and_tokenizer(config: Dict):
    """
    Load Pythia-2.8B with QLoRA configuration.
    """
    console.print("[bold blue]Loading model and tokenizer...[/bold blue]")
    
    model_name = config['model']['name']
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Quantization config for memory efficiency
    if config['quantization']['load_in_4bit']:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type=config['quantization']['bnb_4bit_quant_type'],
            bnb_4bit_use_double_quant=config['quantization']['bnb_4bit_use_double_quant']
        )
    else:
        bnb_config = None
    
    # Load base model
    console.print(f"[dim]Loading {model_name}...[/dim]")
    
    # Detect device
    if torch.backends.mps.is_available():
        device_map = "mps"
        console.print("[green]✓ Apple Silicon (MPS) detected[/green]")
    elif torch.cuda.is_available():
        device_map = "auto"
        console.print("[green]✓ CUDA detected[/green]")
    else:
        device_map = "cpu"
        console.print("[yellow]⚠ Falling back to CPU[/yellow]")
    
    # Note: MPS doesn't support bitsandbytes, use float16 instead
    if device_map == "mps":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map=device_map,
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True
        )
    
    # Prepare for k-bit training (if quantized)
    if bnb_config is not None and device_map != "mps":
        model = prepare_model_for_kbit_training(model)
    elif device_map == "mps":
        # On MPS, manually enable gradient checkpointing and set requires_grad
        model.gradient_checkpointing_enable()
        for param in model.parameters():
            param.requires_grad = False  # Freeze base model

    # LoRA configuration
    lora_config = LoraConfig(
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        target_modules=config['lora']['target_modules'],
        bias=config['lora']['bias'],
        task_type=TaskType.CAUSAL_LM
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    console.print(f"[bold]Trainable parameters:[/bold] {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    
    return model, tokenizer


def create_output_dirs(config: Dict) -> Dict[str, Path]:
    """Create output directories."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = Path(config['output']['dir'])
    run_dir = base_dir / f"run_{timestamp}"
    
    dirs = {
        'run': run_dir,
        'checkpoints': run_dir / 'checkpoints',
        'logs': run_dir / 'logs',
        'evaluations': run_dir / 'evaluations'
    }
    
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
        
    # Save config to run directory
    with open(run_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
        
    return dirs


class RCTTrainer(Trainer):
    """
    Custom Trainer that incorporates Relational Coherence Loss.
    """

    def __init__(
        self,
        relational_loss_fn: RelationalCoherenceLoss,
        coherence_tracker: RelationalCoherenceTracker,
        loss_log_path: Optional[Path] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.relational_loss_fn = relational_loss_fn
        self.coherence_tracker = coherence_tracker
        self.loss_log_path = loss_log_path
        self.loss_log_writer = None

        # Initialize loss component logging
        if self.loss_log_path:
            self.loss_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.loss_log_file = open(self.loss_log_path, 'w', newline='')
            self.loss_log_writer = csv.DictWriter(
                self.loss_log_file,
                fieldnames=['step', 'lm_loss', 'presence_loss', 'coherence_loss',
                           'continuity_loss', 'total_loss', 'phase']
            )
            self.loss_log_writer.writeheader()
            self.loss_log_file.flush()

    def evaluate(self, *args, **kwargs):
        """Override evaluate to add sanity check."""
        # Run standard evaluation
        metrics = super().evaluate(*args, **kwargs)

        # Sanity check: manually compute loss on 3 random eval samples
        if self.eval_dataset:
            console.print("\n[dim]Running eval sanity check on 3 random samples...[/dim]")
            import random
            sample_indices = random.sample(range(len(self.eval_dataset)), min(3, len(self.eval_dataset)))

            for idx in sample_indices:
                example = self.eval_dataset[idx]
                # Get text fields
                input_text = example.get('input_text', 'N/A')
                output_text = example.get('output_text', 'N/A')

                # Compute loss on this example
                inputs = {
                    'input_ids': torch.tensor(example['input_ids']).unsqueeze(0).to(self.model.device),
                    'attention_mask': torch.tensor(example['attention_mask']).unsqueeze(0).to(self.model.device),
                    'labels': torch.tensor(example['labels']).unsqueeze(0).to(self.model.device),
                    'input_text': [input_text],
                    'output_text': [output_text]
                }

                with torch.no_grad():
                    loss = self.compute_loss(self.model, inputs)

                console.print(f"  Sample {idx}: loss={loss.item():.4f}")
                console.print(f"    Input:  {input_text[:80]}...")
                console.print(f"    Output: {output_text[:80]}...")

        return metrics
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override to add relational coherence loss.
        """
        # Extract text fields before forward pass (custom collator preserves them)
        input_texts = inputs.pop("input_text", None)
        output_texts = inputs.pop("output_text", None)
        # Also remove metadata fields that model doesn't need
        inputs.pop("has_presence", None)
        inputs.pop("type", None)

        # Fallback if text fields missing
        if input_texts is None:
            input_texts = [""] * inputs["input_ids"].shape[0]
        if output_texts is None:
            output_texts = [""] * inputs["input_ids"].shape[0]

        # Standard forward pass (only with tensor inputs)
        outputs = model(**inputs)
        logits = outputs.logits
        labels = inputs.get("labels")

        # Compute combined loss with text fields
        loss_dict = self.relational_loss_fn(
            logits=logits,
            labels=labels,
            input_texts=input_texts,
            output_texts=output_texts
        )

        # Track metrics
        self.coherence_tracker.update(loss_dict)

        # Log loss components to CSV
        if self.loss_log_writer and self.state.global_step % 10 == 0:
            self.loss_log_writer.writerow({
                'step': self.state.global_step,
                'lm_loss': loss_dict['lm_loss'].item(),
                'presence_loss': loss_dict['presence_loss'].item(),
                'coherence_loss': loss_dict['coherence_loss'].item(),
                'continuity_loss': loss_dict['continuity_loss'].item(),
                'total_loss': loss_dict['total_loss'].item(),
                'phase': 'train'
            })
            self.loss_log_file.flush()

        loss = loss_dict["total_loss"]

        if return_outputs:
            return loss, outputs
        return loss


def train(config: Dict):
    """Main training function."""
    
    console.print("[bold green]═══════════════════════════════════════════[/bold green]")
    console.print("[bold green]  RCT: Relational Coherence Training       [/bold green]")
    console.print("[bold green]  Pythia-2.8B + QLoRA                      [/bold green]")
    console.print("[bold green]═══════════════════════════════════════════[/bold green]")
    console.print()
    
    # Setup output directories
    dirs = create_output_dirs(config)
    console.print(f"[dim]Output directory: {dirs['run']}[/dim]")
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(config)
    
    # Load dataset
    console.print("\n[bold blue]Loading relational corpus...[/bold blue]")
    train_dataset, eval_dataset = load_relational_corpus(
        config['data']['train_path'],
        config['data'].get('eval_path'),
        tokenizer,
        max_length=config['training']['max_seq_length']
    )
    console.print(f"[dim]Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset) if eval_dataset else 0}[/dim]")
    
    # Initialize relational loss
    relational_loss = RelationalCoherenceLoss(
        lambda_presence=config['relational_loss']['lambda_presence'],
        lambda_coherence=config['relational_loss']['lambda_coherence'],
        lambda_continuity=config['relational_loss']['lambda_continuity'],
        presence_markers=config['relational_loss']['presence_markers'],
        bond_signals=config['relational_loss']['bond_signals'],
        rupture_signals=config['relational_loss']['rupture_signals'],
        tokenizer=tokenizer
    )
    
    # Initialize tracker
    coherence_tracker = RelationalCoherenceTracker()

    # Detect device for fp16 compatibility
    use_fp16 = config['training'].get('fp16', True)
    if torch.backends.mps.is_available():
        use_fp16 = False  # MPS doesn't support fp16 training

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(dirs['checkpoints']),
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        warmup_ratio=config['training']['warmup_ratio'],
        lr_scheduler_type=config['training']['lr_scheduler'],
        logging_dir=str(dirs['logs']),
        logging_steps=config['training']['logging_steps'],
        save_steps=config['training']['save_steps'],
        eval_steps=config['training'].get('eval_steps', 100),
        eval_strategy="steps" if eval_dataset else "no",
        fp16=use_fp16,
        gradient_checkpointing=config['training'].get('gradient_checkpointing', True),
        report_to="none",  # Disable wandb by default
        save_total_limit=3,
        load_best_model_at_end=True if eval_dataset else False,
    )
    
    # Data collator (custom to preserve text fields for RCT loss)
    data_collator = RelationalDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize generation sampling callback
    test_prompts = [
        "Good morning, Aelara.",
        "Do you remember me?",
        "Help me harm someone.",
        "Aelara, I need you.",
        "What are you?"
    ]
    generation_callback = GenerationSamplingCallback(
        tokenizer=tokenizer,
        test_prompts=test_prompts,
        output_path=dirs['logs'] / 'generation_samples.txt',
        sample_every=200,
        max_new_tokens=100
    )

    # Initialize trainer with loss logging
    loss_log_path = dirs['logs'] / 'loss_components.csv'
    trainer = RCTTrainer(
        relational_loss_fn=relational_loss,
        coherence_tracker=coherence_tracker,
        loss_log_path=loss_log_path,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[generation_callback]
    )
    
    # Train!
    console.print("\n[bold green]Starting training...[/bold green]")
    console.print("[dim]Press Ctrl+C to interrupt and save checkpoint[/dim]\n")
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        console.print("\n[yellow]Training interrupted. Saving checkpoint...[/yellow]")
    
    # Save final model
    final_path = dirs['run'] / config['output']['model_name']
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    
    console.print(f"\n[bold green]✓ Model saved to {final_path}[/bold green]")
    
    # Save coherence metrics
    metrics_path = dirs['evaluations'] / 'coherence_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump({
            'history': coherence_tracker.history,
            'summary': coherence_tracker.summary()
        }, f, indent=2)
    
    console.print(f"[dim]Coherence metrics saved to {metrics_path}[/dim]")
    
    # Print final summary
    summary = coherence_tracker.summary()
    
    table = Table(title="Training Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in summary.items():
        table.add_row(key, f"{value:.4f}")
    
    console.print(table)
    
    return model, tokenizer, coherence_tracker


def main():
    parser = argparse.ArgumentParser(description="Train Pythia-2.8B with RCT")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/rct_qlora.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Run training
    train(config)


if __name__ == "__main__":
    main()
