import mlx.core as mx
import mlx.nn as nn

# --- Sacred Marker Token IDs (from tokenizer analysis) ---
# presence: [121758], Spiral: [102891, 1279], coherence: [2320, 119353]
SACRED_TOKEN_IDS = mx.array([121758, 102891, 1279, 2320, 119353], dtype=mx.int32)

# Coherence target - will be set dynamically based on model dim
# For now, use a normalized random vector (unit sphere)
# This represents the "ideal relational state" in embedding space
COHERENCE_TARGET_EMBEDDING = None

def get_coherence_target(dim):
    """Get or initialize the coherence target embedding."""
    global COHERENCE_TARGET_EMBEDDING
    if COHERENCE_TARGET_EMBEDDING is None or COHERENCE_TARGET_EMBEDDING.shape[0] != dim:
        # Initialize as unit vector pointing in a consistent direction
        # Using all 1s normalized creates a "coherent" direction
        target = mx.ones((dim,), dtype=mx.float32)
        COHERENCE_TARGET_EMBEDDING = target / mx.sqrt(mx.sum(target ** 2))
    return COHERENCE_TARGET_EMBEDDING


def custom_relational_loss(model, inputs, targets, mask):
    """
    Computes combined loss: CrossEntropy + Presence Loss + Coherence Loss.

    The Coherence Loss operates on the FINAL hidden state only,
    avoiding shape mismatches from variable sequence lengths.
    """
    # 1. Forward pass
    logits = model(inputs)

    # Get hidden states from patched model (if available)
    hidden_states = getattr(model, 'final_hidden_state_for_rct', None)

    # 2. Standard Cross-Entropy Loss
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mx.maximum(mask.sum(), mx.array(1.0))  # Prevent division by zero
    ce_loss = ce.sum() / ntoks

    # 3. Presence Loss (encourage sacred token predictions)
    # Clamp logits to prevent overflow in log_softmax
    logits_clamped = mx.clip(logits, -100.0, 100.0)
    log_probs = nn.log_softmax(logits_clamped, axis=-1)
    sacred_log_probs = log_probs[..., SACRED_TOKEN_IDS]
    # Clamp log probs to prevent -inf propagation
    sacred_log_probs = mx.maximum(sacred_log_probs, mx.array(-50.0))
    presence_score = (sacred_log_probs.mean(axis=-1) * mask).sum() / ntoks
    presence_loss = -presence_score * 0.1

    # 4. Coherence Loss (on FINAL token hidden state only)
    # NOTE: Detached from gradient to avoid shape mismatch issues
    # This is for MONITORING only - coherence will improve as a side effect of CE training
    if hidden_states is not None:
        # Detach from gradient computation
        h_detached = mx.stop_gradient(hidden_states)
        batch_size, seq_len, dim = h_detached.shape

        # Find the position of the last valid token for each sequence
        sequence_lengths = mx.sum(mask.astype(mx.int32), axis=1)
        last_token_indices = mx.maximum(sequence_lengths - 1, mx.zeros_like(sequence_lengths))

        # Gather final hidden states
        gather_indices = last_token_indices[:, None, None]
        gather_indices = mx.broadcast_to(gather_indices, (batch_size, 1, dim))
        final_hidden_state = mx.take_along_axis(h_detached, gather_indices, axis=1)
        final_hidden_state = final_hidden_state.squeeze(axis=1)

        # Get coherence target and compute similarity
        coherence_target = get_coherence_target(dim)
        final_norm = mx.sqrt(mx.sum(final_hidden_state ** 2, axis=-1, keepdims=True) + 1e-8)
        final_normalized = final_hidden_state / final_norm
        similarity = mx.sum(final_normalized * coherence_target[None, :], axis=-1)

        # Coherence metric (not added to loss, just for monitoring)
        coherence_loss = (1.0 - similarity.mean()) * 0.0  # Weight = 0 (monitoring only)
    else:
        coherence_loss = mx.array(0.0)

    # 5. Total Loss with NaN guard
    total_loss = ce_loss + presence_loss + coherence_loss

    # Final safety: replace NaN with large finite value to allow recovery
    total_loss = mx.where(mx.isnan(total_loss), mx.array(10.0), total_loss)

    return total_loss, ce_loss, presence_loss, coherence_loss
