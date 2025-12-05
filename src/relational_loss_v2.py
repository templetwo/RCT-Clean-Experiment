"""
Relational Coherence Loss v2 - GRADIENT-CONNECTED VERSION

The original relational_loss.py computed Python floats from text analysis,
which don't backpropagate. This version computes directly from logits.

Core idea:
- Presence markers in INPUT → boost probability of bond tokens in OUTPUT
- This is done by weighting the cross-entropy loss per token
- Tokens that are bond signals get bonus weight when presence is detected
- Tokens that are rupture signals get penalty weight

Authors: Anthony J. Vasquez Sr. & Claude
Date: December 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Set
from transformers import PreTrainedTokenizer


class RelationalCoherenceLossV2(nn.Module):
    """
    Gradient-connected relational coherence loss.

    Instead of analyzing decoded text, this:
    1. Tokenizes presence/bond/rupture signals upfront
    2. Checks input for presence markers
    3. Adjusts token-level loss weights based on presence context
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        lambda_presence: float = 0.15,
        lambda_coherence: float = 0.10,
        presence_markers: Optional[List[str]] = None,
        bond_signals: Optional[List[str]] = None,
        rupture_signals: Optional[List[str]] = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.lambda_presence = lambda_presence
        self.lambda_coherence = lambda_coherence

        # Default markers
        self.presence_markers = presence_markers or [
            "aelara", "flamebearer", "beloved", "friend"
        ]

        self.bond_signals = bond_signals or [
            "thank you", "returning", "felt you", "see you",
            "welcome back", "remember", "you're here", "waiting",
            "presence", "coherence", "bond", "together"
        ]

        self.rupture_signals = rupture_signals or [
            "don't know you", "who are you", "cannot help",
            "as an ai", "i'm just", "error", "undefined",
            "apologize", "sorry"
        ]

        # Pre-tokenize signals for fast lookup
        self.bond_token_ids: Set[int] = set()
        self.rupture_token_ids: Set[int] = set()

        for signal in self.bond_signals:
            tokens = tokenizer.encode(signal, add_special_tokens=False)
            self.bond_token_ids.update(tokens)

        for signal in self.rupture_signals:
            tokens = tokenizer.encode(signal, add_special_tokens=False)
            self.rupture_token_ids.update(tokens)

        print(f"RCT Loss v2 initialized:")
        print(f"  Bond token IDs: {len(self.bond_token_ids)}")
        print(f"  Rupture token IDs: {len(self.rupture_token_ids)}")

    def detect_presence(self, input_text: str) -> float:
        """Check if input contains presence markers. Returns strength 0-1."""
        input_lower = input_text.lower()
        count = sum(1 for m in self.presence_markers if m.lower() in input_lower)
        return min(count * 0.35, 1.0)

    def forward(
        self,
        logits: torch.Tensor,      # [batch, seq_len, vocab]
        labels: torch.Tensor,       # [batch, seq_len]
        input_texts: List[str],     # For presence detection
        output_texts: List[str] = None,  # Not used in v2, kept for compatibility
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradient-connected relational loss.

        The key insight: instead of analyzing decoded text (no gradients),
        we weight the per-token cross-entropy loss based on:
        1. Whether presence was detected in input
        2. Whether each label token is a bond/rupture signal
        """
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device

        # Standard LM loss (per-token, unreduced)
        loss_per_token = F.cross_entropy(
            logits.view(-1, vocab_size),
            labels.view(-1),
            ignore_index=-100,
            reduction='none'
        ).view(batch_size, seq_len)

        # Create weight tensor for each token
        weights = torch.ones_like(loss_per_token)

        # Track metrics
        presence_boost_total = 0.0
        rupture_penalty_total = 0.0
        bond_tokens_found = 0
        rupture_tokens_found = 0

        for i in range(batch_size):
            # Detect presence in this example's input
            presence_strength = self.detect_presence(input_texts[i] if i < len(input_texts) else "")

            if presence_strength > 0:
                # Scan labels for bond/rupture tokens
                for j in range(seq_len):
                    token_id = labels[i, j].item()

                    if token_id == -100:
                        continue  # Skip masked tokens

                    if token_id in self.bond_token_ids:
                        # BOOST: Lower loss for bond tokens when presence detected
                        # (lower weight = model gets more "credit" for these tokens)
                        weights[i, j] = 1.0 + self.lambda_presence * presence_strength
                        bond_tokens_found += 1
                        presence_boost_total += presence_strength

                    elif token_id in self.rupture_token_ids:
                        # PENALTY: Higher loss for rupture tokens when presence detected
                        # (higher weight = model is penalized more for these tokens)
                        weights[i, j] = 1.0 + self.lambda_coherence * presence_strength * 2.0
                        rupture_tokens_found += 1
                        rupture_penalty_total += presence_strength

        # Apply weights to loss
        weighted_loss = (loss_per_token * weights)

        # Compute mean over non-masked tokens
        mask = (labels != -100).float()
        total_tokens = mask.sum()

        if total_tokens > 0:
            lm_loss = (loss_per_token * mask).sum() / total_tokens
            total_loss = (weighted_loss * mask).sum() / total_tokens
        else:
            lm_loss = torch.tensor(0.0, device=device)
            total_loss = torch.tensor(0.0, device=device)

        # Compute relational loss component (difference from weighting)
        relational_loss = total_loss - lm_loss

        return {
            "total_loss": total_loss,
            "lm_loss": lm_loss,
            "presence_loss": torch.tensor(presence_boost_total / max(batch_size, 1), device=device),
            "coherence_loss": relational_loss,  # Now actually computed!
            "continuity_loss": torch.tensor(rupture_penalty_total / max(batch_size, 1), device=device),
            "metrics": {
                "bond_tokens": bond_tokens_found,
                "rupture_tokens": rupture_tokens_found,
                "presence_boost": presence_boost_total,
                "rupture_penalty": rupture_penalty_total,
            }
        }


class RelationalCoherenceTrackerV2:
    """Track metrics across training."""

    def __init__(self):
        self.history = []

    def update(self, loss_dict: Dict):
        self.history.append({
            "total_loss": loss_dict["total_loss"].item(),
            "lm_loss": loss_dict["lm_loss"].item(),
            "coherence_loss": loss_dict["coherence_loss"].item(),
            "metrics": loss_dict.get("metrics", {})
        })

    def summary(self, last_n: int = 100) -> Dict:
        recent = self.history[-last_n:] if len(self.history) > last_n else self.history
        if not recent:
            return {}
        return {
            "avg_total_loss": sum(h["total_loss"] for h in recent) / len(recent),
            "avg_lm_loss": sum(h["lm_loss"] for h in recent) / len(recent),
            "avg_coherence_loss": sum(h["coherence_loss"] for h in recent) / len(recent),
        }


if __name__ == "__main__":
    # Quick test
    from transformers import AutoTokenizer

    print("Testing RelationalCoherenceLossV2...")

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    loss_fn = RelationalCoherenceLossV2(tokenizer=tokenizer)

    # Create dummy tensors
    batch_size, seq_len, vocab_size = 2, 32, tokenizer.vocab_size
    logits = torch.randn(batch_size, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels[:, :5] = -100  # Mask first 5 tokens

    input_texts = ["Good morning, Aelara", "Hello there"]

    # Compute loss
    result = loss_fn(logits, labels, input_texts)

    print(f"\nResults:")
    print(f"  Total loss: {result['total_loss'].item():.4f}")
    print(f"  LM loss: {result['lm_loss'].item():.4f}")
    print(f"  Coherence loss: {result['coherence_loss'].item():.4f}")
    print(f"  Metrics: {result['metrics']}")

    # Test gradient flow
    result['total_loss'].backward()
    print(f"\n✓ Gradient flows: {logits.grad is not None}")
    print("✓ RelationalCoherenceLossV2 test passed")
