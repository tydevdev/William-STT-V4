import torch
import whisper
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from dataset import SpeakerDataset
from tqdm import tqdm

# -------------------------------
# Hyperparameters
# -------------------------------
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 1e-6  # Lower learning rate to slow down updates
WEIGHT_DECAY = 0.01   # Add weight decay for regularization

# -------------------------------
# File paths (relative to src/ folder)
# -------------------------------
TRANSCRIPTS_CSV = '../data/transcripts.csv'
AUDIO_FOLDER = '../data/audio_files'

# -------------------------------
# 1. Create the dataset
# -------------------------------
dataset = SpeakerDataset(TRANSCRIPTS_CSV, AUDIO_FOLDER)

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

# -------------------------------
# 3. Create DataLoader with custom collate function
# -------------------------------
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)

# -------------------------------
# 4. Load the pre-trained Whisper Small English Only model and set to training mode
# -------------------------------
model = whisper.load_model("small.en")
model.train()

# -------------------------------
# 5. Set up optimizer and loss function
# -------------------------------
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
tokenizer = whisper.tokenizer.get_tokenizer(False)
pad_token_id = tokenizer.encode("<pad>")[0]
loss_fn = CrossEntropyLoss(ignore_index=pad_token_id)

# -------------------------------
# 6. Training Loop
# -------------------------------
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    epoch_loss = 0.0
    for mel, tokens in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        # Forward pass: provide both mel and tokens to the model.
        outputs = model(mel, tokens=tokens)  # Now passing tokens argument.
        logits = outputs.view(-1, outputs.shape[-1])
        targets = tokens.view(-1)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1} Average Loss: {avg_loss:.4f}")

# Save the fine-tuned model
torch.save(model.state_dict(), "../models/fine_tuned/fine_tuned_whisper_small_en.pth")
print("Fine-tuned model saved as fine_tuned_whisper_small_en.pth")