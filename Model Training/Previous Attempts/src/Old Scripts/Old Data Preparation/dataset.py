import os
import csv
import torch
from torch.utils.data import Dataset
import whisper

class SpeakerDataset(Dataset):
 def __init__(self, transcripts_csv, audio_folder):
     self.data = []
     with open(transcripts_csv, 'r', encoding='utf-8') as f:
         reader = csv.DictReader(f)
         for row in reader:
             # Debug: print row keys (remove after confirming)
             print("Row keys:", list(row.keys()))
             self.data.append(row)
     self.audio_folder = audio_folder
     # Create tokenizer for English-only (multilingual=False)
     self.tokenizer = whisper.tokenizer.get_tokenizer(False)

 def __len__(self):
     return len(self.data)

 def __getitem__(self, idx):
     item = self.data[idx]
     # Use the key 'audio_file' as per our CSV header in training data
     audio_path = os.path.join(self.audio_folder, item['audio_file'])
     audio = whisper.load_audio(audio_path)
     audio = whisper.pad_or_trim(audio)
     mel = whisper.log_mel_spectrogram(audio)
     tokens = self.tokenizer.encode(item['transcript'])
     tokens = torch.tensor(tokens, dtype=torch.long)
     return mel, tokens

if __name__ == "__main__":
 dataset = SpeakerDataset('data/transcripts.csv', 'data/audio_files')
 print("Number of samples:", len(dataset))
 mel, tokens = dataset[0]
 print("Mel spectrogram shape:", mel.shape)
 print("Tokens:", tokens)