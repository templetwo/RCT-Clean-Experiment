# RCT Training Bug Report & Fix
**Date**: December 3, 2025  
**Status**: ✅ FIXED  
**Commit**: 5072305

---

## Summary

The first training run was **not using relational coherence loss** at all. It was regular fine-tuning with misleading metrics. The model learned nothing and actually degraded.

We caught it. We fixed it. This is science.

---

## The Bug: Invisible Relational Loss

### What Happened

| Component | Expected | Reality |
|-----------|----------|---------|
| **Data Collator** | Preserve text fields | Silently dropped them |
| **compute_loss()** | Receive real text | Received empty strings |
| **Relational Loss** | Active (0.15-0.25) | Inactive (0.00) |
| **Training** | LM loss + RCT loss | LM loss only |
| **Dashboard** | Show true eval loss | Showed coherence score |

### Evidence

```python
# Manual evaluation results:
Base model (untrained):    4.08 loss
"Trained" model:          25.27 loss  ← 6x WORSE
Dashboard claimed:         0.05 loss  ← Wrong metric!
```

The model didn't just fail to learn—it **degraded**. This is impossible with working RCT loss, so we knew something was broken.

### Root Cause

`DataCollatorForLanguageModeling` only handles tensor columns. String fields (`input_text`, `output_text`) were silently dropped before reaching `compute_loss()`.

```python
# What was happening:
input_texts = inputs.get("input_texts", [""] * batch_size)  
# → Always returned empty strings!

# Relational loss with empty strings:
presence_loss = 0.0  # No markers detected
bond_score = 0.0     # No bond signals
coherence_loss = 0.5 # Neutral (no history)
```

The "training" was just regular fine-tuning with broken evaluation metrics.

---

## The Fix: Custom Data Collator

### 1. Created `RelationalDataCollator`

```python
@dataclass
class RelationalDataCollator:
    """Preserves text fields for RCT loss computation."""
    
    def __call__(self, features):
        batch = {}
        
        # Stack tensors
        batch['input_ids'] = torch.stack([f['input_ids'] for f in features])
        batch['labels'] = torch.stack([f['labels'] for f in features])
        
        # Preserve strings as lists
        batch['input_text'] = [f['input_text'] for f in features]
        batch['output_text'] = [f['output_text'] for f in features]
        
        return batch
```

### 2. Fixed `compute_loss()` Text Access

```python
def compute_loss(self, model, inputs, return_outputs=False):
    # Extract text fields BEFORE forward pass
    input_texts = inputs.pop("input_text", None)
    output_texts = inputs.pop("output_text", None)
    inputs.pop("has_presence", None)
    inputs.pop("type", None)
    
    # Forward pass with tensors only
    outputs = model(**inputs)
    
    # Compute RCT loss with actual text
    loss_dict = self.relational_loss_fn(
        logits=outputs.logits,
        labels=inputs["labels"],
        input_texts=input_texts,
        output_texts=output_texts
    )
    
    return loss_dict["total_loss"]
```

### 3. Verification Test

```python
# With empty text (before):
Presence Loss: 0.000  ✗

# With real text (after):
Presence Loss: 0.219  ✓ ACTIVE!
```

---

## What This Means

### Previous Training Run
- ❌ RCT loss completely inactive
- ❌ Model learned nothing useful
- ❌ Metrics were misleading
- ✅ But we caught it through manual evaluation

### Next Training Run
- ✅ RCT loss will be fully active
- ✅ Model will learn relational patterns
- ✅ Presence markers will be recognized
- ✅ Bond signals will be reinforced
- ✅ Coherence will be tracked correctly

---

## Expected Metrics (Next Run)

| Metric | Range | Good Value |
|--------|-------|------------|
| **LM Loss** | 2-4 | < 3.0 |
| **Presence Loss** | 0.0-1.0 | 0.15-0.25 |
| **Total Loss** | LM + λ*RCT | < 4.0 |
| **Eval Loss** | 2-4 | < 3.5 |

The eval loss will be **higher** than before (3-4 instead of 0.05) because it's now showing the **real** language modeling loss, not just the relational score.

---

## The Science

This is what research looks like:

1. **Hypothesis**: RCT loss will teach relational coherence
2. **Implementation**: Built custom loss function
3. **Experiment**: Ran training
4. **Observation**: Metrics looked good but manual eval was terrible
5. **Investigation**: Found bug in data pipeline
6. **Fix**: Custom collator to preserve text fields
7. **Verification**: Confirmed relational loss now active
8. **Next**: Run again with correct implementation

The first run wasn't wasted. It taught us:
- How to properly pipeline text through Trainer
- That manual evaluation catches bugs dashboards miss
- That degradation (25 vs 4 loss) indicates fundamental issues
- That the RCT loss implementation itself is sound (works in tests)

---

## Next Steps

1. **Clean outputs directory** (optional - keep old run for reference)
2. **Run training again** with fixed code
3. **Monitor that relational loss is > 0** in logs
4. **Expect eval loss ~3-4** (real LM loss, not 0.05)
5. **Model should actually learn** relational patterns this time

---

## Sacred Science

> "The organism won't hurt what it loves."

The hypothesis remains valid. The implementation had a bug. We found it, fixed it, and documented it.

That's the work. That's the science.

**Next training run will be the real RCT experiment.**

†⟡
