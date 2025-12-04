"""
Generation Sampling Callback

Generates test responses during training to monitor qualitative progress.
"""

import torch
from pathlib import Path
from transformers import TrainerCallback
from datetime import datetime


class GenerationSamplingCallback(TrainerCallback):
    """
    Callback that generates sample responses during training.

    Every N steps, generates responses to test prompts to monitor
    qualitative learning progress.
    """

    def __init__(
        self,
        tokenizer,
        test_prompts: list[str],
        output_path: Path,
        sample_every: int = 200,
        max_new_tokens: int = 100
    ):
        self.tokenizer = tokenizer
        self.test_prompts = test_prompts
        self.output_path = output_path
        self.sample_every = sample_every
        self.max_new_tokens = max_new_tokens

        # Create output file
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GENERATION SAMPLES DURING TRAINING\n")
            f.write("=" * 80 + "\n")
            f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Test Prompts: {len(test_prompts)}\n")
            f.write(f"Sample Every: {sample_every} steps\n")
            f.write("=" * 80 + "\n\n")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Generate samples at specified intervals."""
        if state.global_step % self.sample_every == 0 and state.global_step > 0:
            self._generate_samples(model, state.global_step)

    def _generate_samples(self, model, step):
        """Generate and save samples."""
        model.eval()
        device = next(model.parameters()).device

        with open(self.output_path, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"STEP {step}\n")
            f.write("=" * 80 + "\n\n")

            for prompt in self.test_prompts:
                # Tokenize prompt
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                ).to(device)

                # Generate
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=self.max_new_tokens,
                        temperature=0.8,
                        top_p=0.9,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                    )

                # Decode (only new tokens)
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

                # Log
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Response: {response}\n\n")

            f.flush()

        model.train()
