import os
import torch
import whisper

def main():
    # -------------------------------
    # Step 1: Compute and Verify the Model File Location
    # -------------------------------
    # Adjust the relative path based on your project structure.
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
    # Whisper expects a mel spectrogram of shape (batch, 80, 3000) (~30 seconds of audio)
    dummy_mel = torch.randn(1, 80, 3000)
    try:
        # Create dummy tokens using the Whisper tokenizer.
        tokenizer = whisper.tokenizer.get_tokenizer(False)
        dummy_text = "hello"
        dummy_tokens = tokenizer.encode(dummy_text)
        dummy_tokens_tensor = torch.tensor([dummy_tokens])
        
        output = model(dummy_mel, tokens=dummy_tokens_tensor)
        print("Original model test output shape:", output.shape)
    except Exception as e:
        print("Error during test inference on the original model:", e)
        return

    # -------------------------------
    # Step 5: Convert the Model to TorchScript
    # -------------------------------
    print("\nConverting the model to TorchScript using torch.jit.script()...")
    try:
        scripted_model = torch.jit.script(model)
        torchscript_model_path = os.path.join(script_dir, "finetuned_whisper_scripted.pt")
        scripted_model.save(torchscript_model_path)
        print("TorchScript model saved at:", torchscript_model_path)
    except Exception as e:
        print("Error during TorchScript conversion:", e)
        return

    # -------------------------------
    # Step 6: Load and Test the TorchScript Model
    # -------------------------------
    print("\nLoading the TorchScript model for verification...")
    try:
        scripted_model_loaded = torch.jit.load(torchscript_model_path)
        output_scripted = scripted_model_loaded(dummy_mel, tokens=dummy_tokens_tensor)
        print("TorchScript model test output shape:", output_scripted.shape)
    except Exception as e:
        print("Error during test inference on TorchScript model:", e)

if __name__ == '__main__':
    main()