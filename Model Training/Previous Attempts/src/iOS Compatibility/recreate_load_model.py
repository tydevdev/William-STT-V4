import os
import torch
import whisper

def main():
    # Step 1: Compute and verify the absolute path to your finetuned model file.
    # Adjust the relative path according to your project structure.
    # In this example, we go up two directories from the current script
    # to reach the models/fine_tuned folder.
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

    # Step 2: Load the base Whisper model architecture.
    # This loads the model definition for "small.en" from the Whisper library.
    print("\nLoading the Whisper model architecture (small.en)...")
    model = whisper.load_model("small.en")
    model.eval()  # Set the model to evaluation mode
    
    # Step 3: Load the state dictionary (weights) from the pth file.
    print("Loading the state dictionary from the model file...")
    try:
        state_dict = torch.load(absolute_model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        print("State dictionary loaded successfully.")
    except Exception as e:
        print("Error loading state dictionary:", e)
        return

    # Step 4: Test the model with a dummy input.
    # NOTE: The dummy input dimensions may need to be adjusted according to your model's requirements.
    print("\nRunning a test inference with a dummy input...")
    dummy_input = torch.randn(1, 80, 300)  # Example dimensions; adjust if necessary.
    try:
        output = model(dummy_input)
        print("Model test output shape:", output.shape)
    except Exception as e:
        print("Error during test inference:", e)

if __name__ == '__main__':
    main()