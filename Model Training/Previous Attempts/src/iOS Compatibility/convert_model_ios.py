#!/usr/bin/env python3
import torch
from transformers import WhisperForConditionalGeneration
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "models", "fine_tuned", "fine_tuned_whisper_small_en_v4.pth")
output_dir = os.path.join(current_dir, "..", "models", "ios_compatible")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "whisper_ios_v4.pt")

# ----- STEP 1: Load Your Fine-Tuned Model (V4) -----
# We assume you fine-tuned the "openai/whisper-small" model.
# Replace this with the appropriate model variant if needed.
model_name = "openai/whisper-small"
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Load your fine-tuned weights from your .pth file.
# Adjust keys by prefixing with 'model.' if not present
raw_state_dict = torch.load(model_path, map_location="cpu")
adjusted_state_dict = {}
for key, value in raw_state_dict.items():
    new_key = key if key.startswith("model.") else "model." + key
    adjusted_state_dict[new_key] = value

# Load the adjusted state_dict with relaxed strictness
model.load_state_dict(adjusted_state_dict, strict=False)
model.eval()  # Set the model to evaluation mode (disables dropout, etc.)

# ----- STEP 2: Convert to TorchScript -----
# Prepare dummy input
input_features = torch.randn(1, 80, 3000)

# Provide dummy decoder_input_ids to avoid ValueError
dummy_decoder_input_ids = torch.tensor([[1]])

# Trace the model
class WhisperWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_features, decoder_input_ids):
        return self.model(input_features=input_features, decoder_input_ids=decoder_input_ids, return_dict=True).logits

wrapper = WhisperWrapper(model)
scripted_model = torch.jit.trace(wrapper, (input_features, dummy_decoder_input_ids))

# ----- STEP 3: Save the TorchScript Model -----
# Save the TorchScript model so it can be loaded by your iOS app.
scripted_model.save(output_path)
print(f"Model conversion successful. File saved as {output_path}")