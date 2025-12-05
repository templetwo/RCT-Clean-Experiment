import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.datasets import ChatDataset, CacheDataset
from relational_loss_fn import custom_relational_loss
import json

# ============================================================
# 1. LOAD THE MODEL
# ============================================================
model_path = "mlx_model"
print(f"Loading model from {model_path}...")
model, tokenizer = load(model_path)

# Set chat template for base model (Mistral format)
MISTRAL_CHAT_TEMPLATE = """{% for message in messages %}{% if message['role'] == 'system' %}[INST] {{ message['content'] }}

{% elif message['role'] == 'user' %}{{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %}{{ message['content'] }}</s>{% endif %}{% endfor %}"""
tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

# ============================================================
# 2. FREEZE & ADAPT (LoRA)
# ============================================================
# Note: Model patch removed - coherence loss is monitoring-only (detached)
model.freeze()
linear_to_lora_layers(model, num_layers=8, config={"rank": 16, "scale": 2.0, "dropout": 0.05})
print("✅ LoRA layers applied.")

# ============================================================
# 3. LOAD DATA
# ============================================================
def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

train_raw = load_jsonl("rct_train.jsonl")
val_raw = load_jsonl("rct_val.jsonl")
print(f"Loaded {len(train_raw)} train, {len(val_raw)} val examples.")

# Create Chat Datasets with CacheDataset wrapper
train_chat = ChatDataset(train_raw, tokenizer, chat_key="messages", mask_prompt=True)
val_chat = ChatDataset(val_raw, tokenizer, chat_key="messages", mask_prompt=True)
train_data = CacheDataset(train_chat)
val_data = CacheDataset(val_chat)
print(f"✅ Datasets created. Train: {len(train_data)}, Val: {len(val_data)}")

# ============================================================
# 4. DEFINE RCT LOSS WRAPPER
# ============================================================
def rct_loss_wrapper(model, batch, lengths):
    """
    Wrapper matching MLX signature: (model, batch, lengths) -> (loss, ntoks)

    Active components:
    - Cross-Entropy Loss (language modeling)
    - Presence Loss (sacred token boosting)

    Monitoring only:
    - Coherence Loss (detached, weight=0)
    """
    inputs = batch[:, :-1]
    targets = batch[:, 1:]

    # Create mask from lengths
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = mx.logical_and(steps >= lengths[:, 0:1], steps <= lengths[:, 1:])

    # Call custom relational loss
    total_loss, ce_loss, presence_loss, coherence_loss = custom_relational_loss(
        model, inputs, targets, mask
    )

    ntoks = mask.sum()
    return total_loss, ntoks

# ============================================================
# 5. CREATE OPTIMIZER
# ============================================================
optimizer = optim.AdamW(learning_rate=1e-5)

# ============================================================
# 6. TRAINING ARGS
# ============================================================
training_args = TrainingArgs(
    batch_size=1,  # Reduced for memory
    iters=1500,  # Full convergence run (~26 epochs)
    steps_per_eval=50,
    steps_per_report=25,
    steps_per_save=100,  # Save every 100 steps
    max_seq_length=2560,  # Fits longest sequence (2380) with margin
    adapter_file="adapters_rct_v2_presence_boost.safetensors",
)

# ============================================================
# 7. TRAIN WITH RCT LOSS
# ============================================================
print("\n" + "="*60)
print("🌀 Starting The Spiral Training - RCT v2")
print("   Active: CE + Presence Loss")
print("   Monitoring: Coherence (detached)")
print("="*60 + "\n")

train(
    model=model,
    optimizer=optimizer,
    train_dataset=train_data,
    val_dataset=val_data,
    args=training_args,
    loss=rct_loss_wrapper,
)

print("\n✅ Run Complete. Adapters saved to adapters_rct_v2.safetensors")
