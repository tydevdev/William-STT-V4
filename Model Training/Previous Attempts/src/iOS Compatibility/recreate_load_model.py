import os
import torch
import whisper

def main():
    # -------------------------------
    # Step 1: Compute and Verify the Model File Location
    # -------------------------------
    # Adjust the relative path according to your project structure.
    # This example goes up two directories from the current script location.
    relative_model_path = "../../models/fine_tuned/fine_tuned_whisper_small_en_v4.pth"
    script_dir = os.path.dirname(__file__)
    absolute_model_path = os.path.abspath(os.path.join(script_dir, relative_model_path))
    
    print("Script directory:", script_dir)
    print("Computed model absolute path:", absolute_model_path)
    if os.path.exists(absolute_model_path):
        print("Model file exists.")
    else:
        print("Model file does NOT exist. Please check your relative path.")
        return

    # -------------------------------
    # Step 2: Load the Base Whisper Model Architecture
    # -------------------------------
    print("\nLoading the Whisper model architecture (small.en)...")
    # This retrieves the model definition for "small.en" from the Whisper library.
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
    # Step 4: Test Inference with Dummy Inputs
    # -------------------------------
    # Whisper's forward pass requires:
    # 1. A mel spectrogram input of shape (batch, 80, 3000) – corresponding to ~30 seconds of audio.
    # 2. A tensor of tokens for teacher forcing.
    print("\nRunning a test inference with dummy inputs...")

    # Create a dummy mel spectrogram with the expected shape.
    dummy_mel = torch.randn(1, 80, 3000)

    try:
        # Use the Whisper tokenizer to create dummy tokens.
        tokenizer = whisper.tokenizer.get_tokenizer(False)
        # We'll use a simple dummy string, e.g., "hello", to generate tokens.
        dummy_text = "hello"
        # Encode the dummy text into token IDs.
        dummy_tokens = tokenizer.encode(dummy_text)
        # Convert the list of token IDs into a tensor with shape (1, sequence_length)
        dummy_tokens_tensor = torch.tensor([dummy_tokens])
        
        # Run the model's forward pass with the dummy inputs.
        output = model(dummy_mel, tokens=dummy_tokens_tensor)
        print("Model test output shape:", output.shape)
    except Exception as e:
        print("Error during test inference:", e)

if __name__ == '__main__':
    main()