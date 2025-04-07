
# Hey guys! Here's some code to reference for how to interact
# with the trained model (or any model that is download)


# When the app is active, you will need to have a virtual environment (venv) active which you have to initialize once 
# and install the necessary dependencies... see below 
# (I don't think all of these are necessary, but the "whisper" and "torch" ones are)

'''
torch==2.6.0
numpy
tqdm
jiwer
git+https://github.com/openai/whisper.git
---
pysoundfile
torchvision
torchaudio
transformers
'''

# in order to use this, there needs to be a folder somewhere in your project for the model and the fine-tuned model
# these files are too big for GitHub, they are stored in our Google Drive instead. Those files are set in the .gitignore

# import the appropriate libraries
import torch
import whisper


def main():

    # Load the small.en model architecture (this is the base, pre-trained model; it should always be called "small.en")
    model = whisper.load_model("small.en")

    # Load your fine-tuned model weights (this is the fine-tuning add-on) (the file path will need to be updated... good luck)
    model.load_state_dict(torch.load("../models/fine_tuned/fine_tuned_whisper_small_en_v2.pth", map_location="cpu"))
    model.eval()

    # The model works by processing audio files (.wav), you will have to save the recording somewhere as a .wav and then process it
    # Set the full path to an audio file on your Desktop (update accordingly)
    audio_file = "/Users/tydevito/Desktop/william_s2_story.wav"

    # This is the part where the model attempts the transcription of the wav file
    result = model.transcribe(audio_file)
    print("Transcription:", result.get("text", ""))

if __name__ == "__main__":
    main()