import os
import torch
import whisper
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import SpeakerDataset
from tqdm import tqdm

# -------------------------------
# Hyperparameters
# -------------------------------
BATCH_SIZE = 2
NUM_EPOCHS = 5         # Reduced maximum epochs to 5
LEARNING_RATE = 1e-8     # Very low learning rate for cautious fine-tuning
WEIGHT_DECAY = 0.01      # Regularization to help prevent overfitting
GRAD_ACCUM_STEPS = 4     # Accumulate gradients over 4 mini-batches
EARLY_STOPPING_PATIENCE = 2  # Stop if no improvement for 2 consecutive epochs

# -------------------------------
# File paths (relative to src/ folder)
# -------------------------------
TRANSCRIPTS_CSV = '../data/transcripts.csv'
AUDIO_FOLDER = '../data/audio_files'

# -------------------------------
# 1. Create the dataset and split into training and validation sets
# -------------------------------
full_dataset = SpeakerDataset(TRANSCRIPTS_CSV, AUDIO_FOLDER)
dataset_size = len(full_dataset)
val_size = int(0.2 * dataset_size)
train_size = dataset_size - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
print(f"Total samples: {dataset_size}, Training: {train_size}, Validation: {val_size}")

# -------------------------------
# 2. Define a custom collate function to pad token sequences
# -------------------------------
def custom_collate_fn(batch):
    mels, tokens = zip(*batch)
    mels = torch.stack(mels, dim=0)
    max_len = max(t.size(0) for t in tokens)
    padded_tokens = []
    tokenizer = whisper.tokenizer.get_tokenizer(False)
    pad_token_id = tokenizer.encode("<pad>")[0]
    for t in tokens:
        if t.size(0) < max_len:
            pad_length = max_len - t.size(0)
            pad = torch.full((pad_length,), pad_token_id, dtype=t.dtype)
            padded = torch.cat([t, pad], dim=0)
        else:
            padded = t
        padded_tokens.append(padded)
    padded_tokens = torch.stack(padded_tokens, dim=0)
    return mels, padded_tokens

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

# -------------------------------
# 3. Load the pre-trained Whisper Small English Only model and freeze layers
# -------------------------------
model = whisper.load_model("small.en")
model.train()

# Freeze encoder layers and first two decoder layers (if available)
for name, param in model.named_parameters():
    if "encoder" in name or "decoder.layers.0" in name or "decoder.layers.1" in name:
        param.requires_grad = False
print("Frozen encoder and first two decoder layers; fine-tuning only higher-level parameters.")

# -------------------------------
# 4. Set up optimizer, gradient accumulation, and learning rate scheduler
# -------------------------------
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                  lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)

tokenizer = whisper.tokenizer.get_tokenizer(False)
pad_token_id = tokenizer.encode("<pad>")[0]
loss_fn = CrossEntropyLoss(ignore_index=pad_token_id)

# -------------------------------
# 5. Training Loop with Gradient Accumulation and Early Stopping
# -------------------------------
best_val_loss = float('inf')
patience_counter = 0
best_model_path = os.path.join(os.path.dirname(__file__), "../models/fine_tuned/fine_tuned_whisper_small_en_v3.pth")

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    # Training Phase
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for i, (mel, tokens) in enumerate(tqdm(train_loader, desc="Training")):
        outputs = model(mel, tokens=tokens)  # Provide both mel and tokens
        logits = outputs.view(-1, outputs.shape[-1])
        targets = tokens.view(-1)
        loss = loss_fn(logits, targets)
        loss = loss / GRAD_ACCUM_STEPS  # Normalize for gradient accumulation
        loss.backward()
        running_loss += loss.item() * GRAD_ACCUM_STEPS

        if (i + 1) % GRAD_ACCUM_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation Phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for mel, tokens in tqdm(val_loader, desc="Validation"):
            outputs = model(mel, tokens=tokens)
            logits = outputs.view(-1, outputs.shape[-1])
            targets = tokens.view(-1)
            loss = loss_fn(logits, targets)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Avg Train Loss: {avg_train_loss:.6f} | Avg Val Loss: {avg_val_loss:.6f}")
    
    # Step the scheduler based on validation loss
    scheduler.step(avg_val_loss)
    
    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Validation loss improved. Model saved to {best_model_path}.")
    else:
        patience_counter += 1
        print(f"No improvement in validation loss. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

print(f"Training complete. Best validation loss: {best_val_loss:.6f}")