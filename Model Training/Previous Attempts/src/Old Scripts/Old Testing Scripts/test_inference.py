import torch
import whisper

def main():
    # Load the small.en model architecture
    model = whisper.load_model("small.en")
    # Load your fine-tuned model weights (adjust the path if necessary)
    model.load_state_dict(torch.load("../models/fine_tuned/fine_tuned_whisper_small_en_v2.pth", map_location="cpu"))
    model.eval()

    # Set the full path to an audio file on your Desktop (update accordingly)
    audio_file = "/Users/tydevito/Desktop/william_s2_story.wav"

    # Use the model's transcribe method
    result = model.transcribe(audio_file)
    print("Transcription:", result.get("text", ""))

if __name__ == "__main__":
    main()