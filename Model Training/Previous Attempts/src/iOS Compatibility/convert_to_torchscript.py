#!/usr/bin/env python3
import os
import torch

# --- Apply monkey patch BEFORE importing anything else  ---
from torch.nn import functional as F
_original_scaled_dot_product_attention = F.scaled_dot_product_attention

def _patched_scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False):
    # Force is_causal to be a Python bool no matter what
    is_causal = bool(is_causal)
    return _original_scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)

F.scaled_dot_product_attention = _patched_scaled_dot_product_attention
# --- End monkey-patching ---

import whisper  # Now import whisper after patching F.scaled_dot_product_attention

def main():
    # -------------------------------
    # Step 1: Compute and Verify the Model File Location
    # -------------------------------
    relative_model_path = "../../models/fine_tuned/fine_tuned_whisper_small_en_v4.pth"
    script_dir = os.path.dirname(__file__)
    absolute_model_path = os.path.abspath(os.path.join(script_dir, relative_model_path))
    
    print("Script directory:", script_dir)
    print("Computed model absolute path:", absolute_model_path)
    if not os.path.exists(absolute_model_path):
        print("Model file does NOT exist. Please check your relative path.")
        return
    print("Model file exists.\n")
    
    # -------------------------------
    # Step 2: Load the Base Whisper Model Architecture
    # -------------------------------
    print("Loading the Whisper model architecture (small.en)...")
    model = whisper.load_model("small.en")
    model.eval()  # Set the model to evaluation mode

    # -------------------------------
    # Step 3: Load the Finetuned State Dictionary (Weights)
    # -------------------------------
    print("Loading the state dictionary from the model file...")
    try:
        state_dict = torch.load(absolute_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("State dictionary loaded successfully.")
    except Exception as e:
        print("Error loading state dictionary:", e)
        return

    # -------------------------------
    # Step 4: Test Inference with Dummy Inputs on the Original Model
    # -------------------------------
    print("\nRunning a test inference (Original model)...")
    # Create a dummy mel spectrogram (Whisper expects shape (batch, 80, 3000))
    dummy_mel = torch.randn(1, 80, 3000)
    # Create dummy tokens using Whisper's tokenizer.
    tokenizer = whisper.tokenizer.get_tokenizer(False)
    dummy_text = "hello"
    dummy_tokens = tokenizer.encode(dummy_text)
    dummy_tokens_tensor = torch.tensor([dummy_tokens])
    
    try:
        output = model(dummy_mel, tokens=dummy_tokens_tensor)
        print("Original model test output shape:", output.shape)
    except Exception as e:
        print("Error during test inference on the original model:", e)
        return

    # -------------------------------
    # Step 5: Convert the Model to TorchScript Using Tracing (with our patch in effect)
    # -------------------------------
    print("\nConverting the model to TorchScript using torch.jit.trace()...")
    dummy_inputs = (dummy_mel, dummy_tokens_tensor)
    try:
        traced_model = torch.jit.trace(model, dummy_inputs)
        torchscript_model_path = os.path.join(script_dir, "finetuned_whisper_traced.pt")
        traced_model.save(torchscript_model_path)
        print("TorchScript traced model saved at:", torchscript_model_path)
    except Exception as e:
        print("Error during TorchScript conversion using tracing:", e)
        return

    # -------------------------------
    # Step 6: Load and Test the Traced TorchScript Model
    # -------------------------------
    print("\nLoading the TorchScript traced model for verification...")
    try:
        traced_model_loaded = torch.jit.load(torchscript_model_path)
        output_traced = traced_model_loaded(dummy_mel, tokens=dummy_tokens_tensor)
        print("TorchScript traced model test output shape:", output_traced.shape)
    except Exception as e:
        print("Error during test inference on the TorchScript traced model:", e)

if __name__ == '__main__':
    main()