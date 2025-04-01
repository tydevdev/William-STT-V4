import os
import torch
import whisper
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from dataset import SpeakerDataset
from tqdm import tqdm

# -------------------------------
# Hyperparameters
# -------------------------------
BATCH_SIZE = 2
NUM_EPOCHS = 20  # maximum epochs to train
LEARNING_RATE = 1e-7  # very low learning rate for careful fine-tuning
WEIGHT_DECAY = 0.01   # regularization to help prevent overfitting
EARLY_STOPPING_PATIENCE = 2  # stop if no improvement for 2 consecutive epochs

# -------------------------------
# File paths (relative to src/ folder)
# -------------------------------
TRANSCRIPTS_CSV = '../data/transcripts.csv'
AUDIO_FOLDER = '../data/audio_files'

# -------------------------------
# 1. Create the dataset and split it into train/validation sets
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

# Create DataLoaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)

# -------------------------------
# 3. Load the pre-trained Whisper Small English Only model and prepare it for fine-tuning
# -------------------------------
model = whisper.load_model("small.en")
model.train()

# Freeze lower layers (e.g., those with "encoder" in their name) to preserve general features
for name, param in model.named_parameters():
    if "encoder" in name:
        param.requires_grad = False
print("Frozen encoder layers; fine-tuning only higher-level parameters.")

# -------------------------------
# 4. Set up the optimizer and loss function
# -------------------------------
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
tokenizer = whisper.tokenizer.get_tokenizer(False)
pad_token_id = tokenizer.encode("<pad>")[0]
loss_fn = CrossEntropyLoss(ignore_index=pad_token_id)

# -------------------------------
# 5. Training Loop with Early Stopping
# -------------------------------
best_val_loss = float('inf')
patience_counter = 0
best_model_path = os.path.join(os.path.dirname(__file__), "../models/fine_tuned/fine_tuned_whisper_small_en_v2.pth")

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
    # Training Phase
    model.train()
    train_loss = 0.0
    for mel, tokens in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        outputs = model(mel, tokens=tokens)  # Provide both mel and tokens for the forward pass.
        logits = outputs.view(-1, outputs.shape[-1])
        targets = tokens.view(-1)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)
    
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