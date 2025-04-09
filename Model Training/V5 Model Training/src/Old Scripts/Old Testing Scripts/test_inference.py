import torch
import whisper

def main():
    # Load the small.en model architecture
    model = whisper.load_model("small.en")
    #Load your fine-tuned model weights (adjust the path if necessary)
    model2 = whisper.load_model("small.en")
    model.load_state_dict(torch.load("V5 Model Training/models/fine_tuned/fine_tuned_whisper_small_en_v4.pth", map_location="cpu"))
    #model.eval()

    # Set the full path to an audio file on your Desktop (update accordingly)
    audio_file = "V5 Model Training/audio_files/william_s3_l24_9.wav"

    # Use the model's transcribe method
    result = model.transcribe(audio_file)
    result2 = model2.transcribe(audio_file)
    # Print the transcription result
    print("Transcription:\n", result.get("text", ""), "\n", result2.get("text", ""))

if __name__ == "__main__":
    main()