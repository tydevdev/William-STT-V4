#!/usr/bin/env python3
import os
import torch
from torch.nn import functional as F

# --- Monkey-patch scaled_dot_product_attention ---
# Save the original function
original_scaled_dot_product_attention = F.scaled_dot_product_attention

def patched_scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    # Force is_causal to a Python bool, avoiding dynamic value issues during tracing.
    is_causal = bool(is_causal)
    return original_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)

F.scaled_dot_product_attention = patched_scaled_dot_product_attention
# --- End monkey-patching ---

import whisper

def main():
    # Set the relative path to your fine-tuned Whisper state dictionary (.pth file)
    relative_state_dict_path = "../../models/fine_tuned/fine_tuned_whisper_small_en_v4.pth"
    script_dir = os.path.dirname(os.path.realpath(__file__))
    state_dict_path = os.path.abspath(os.path.join(script_dir, relative_state_dict_path))
    
    if not os.path.exists(state_dict_path):
        print("Error: State dict file does not exist:", state_dict_path)
        return
    print("State dict path:", state_dict_path)

    # Load the base Whisper model ("small.en")
    print("Loading the Whisper model architecture (small.en)...")
    model = whisper.load_model("small.en")
    model.eval()  # Set the model to evaluation mode

    # Load the fine-tuned state dictionary into the model
    print("Loading the state dictionary...")
    try:
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("State dictionary loaded successfully.")
    except Exception as e:
        print("Error loading state dictionary:", e)
        return

    # Test inference with dummy inputs
    print("Running test inference on the original model...")
    # Create a dummy mel spectrogram with shape [1, 80, 3000] (typically ~30 sec of audio)
    dummy_mel = torch.randn(1, 80, 3000)
    # Create dummy tokens using Whisper's tokenizer
    tokenizer = whisper.tokenizer.get_tokenizer(False)
    dummy_text = "hello"
    dummy_tokens = tokenizer.encode(dummy_text)
    dummy_tokens_tensor = torch.tensor([dummy_tokens])
    
    try:
        output = model(dummy_mel, tokens=dummy_tokens_tensor)
        print("Original model output shape:", output.shape)
    except Exception as e:
        print("Error during test inference:", e)
        return

    # Convert the model to TorchScript using tracing
    print("Converting the model to TorchScript...")
    dummy_inputs = (dummy_mel, dummy_tokens_tensor)
    try:
        traced_model = torch.jit.trace(model, dummy_inputs)
    except Exception as e:
        print("Error during TorchScript tracing:", e)
        return

    # Save the TorchScript (traced) model to file
    output_path = os.path.join(script_dir, "finetuned_whisper_traced.pt")
    try:
        traced_model.save(output_path)
        print("TorchScript model saved successfully at:", output_path)
    except Exception as e:
        print("Error saving TorchScript model:", e)

if __name__ == "__main__":
    main()