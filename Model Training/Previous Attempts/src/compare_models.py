import os
import torch
import whisper
import csv
import difflib
from jiwer import wer
from tqdm import tqdm

def load_test_transcripts(csv_path):
    """
    Loads test transcripts from a CSV file with header: audio_file,transcript.
    Returns a dictionary mapping each audio file name to its reference transcript.
    """
    mapping = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mapping[row['audio_file']] = row['transcript']
    return mapping

def evaluate_model_detailed(model_path, test_audio_folder, test_transcripts, arch, label):
    """
    Evaluates the given model checkpoint on all test audio files.
    
    Parameters:
        model_path: Path to the .pth checkpoint file.
        test_audio_folder: Folder containing test audio files.
        test_transcripts: Dict mapping audio file names to reference transcripts.
        arch: The model architecture to load (e.g., "small.en", "base.en").
        label: A descriptive label for the model.
        
    Returns:
        avg_wer: Average WER over all evaluated files.
        details: A list of dicts with detailed evaluation info per file.
    """
    print("\n" + "=" * 80)
    print(f"Evaluating Model: {os.path.basename(model_path)}")
    print(f"Architecture: {arch} ({label})")
    print("=" * 80)
    
    # Load the specified model architecture and then load the checkpoint
    model = whisper.load_model(arch)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    
    total_wer = 0.0
    count = 0
    details = []
    
    for root, _, files in os.walk(test_audio_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                audio_path = os.path.join(root, file)
                ref = test_transcripts.get(file)
                if ref is None:
                    print(f"WARNING: No reference transcription for {file}; skipping.")
                    continue
                result = model.transcribe(audio_path)
                hyp = result.get("text", "").strip()
                error = wer(ref, hyp)
                total_wer += error
                count += 1
                diff = difflib.ndiff(ref.split(), hyp.split())
                diff_text = ' '.join(diff)
                details.append({
                    'audio_file': file,
                    'reference': ref,
                    'hypothesis': hyp,
                    'diff': diff_text,
                    'wer': error
                })
                print(f"\nAudio File: {file}")
                print(f"Reference Transcript: {ref}")
                print(f"Hypothesis Transcript: {hyp}")
                print(f"WER for this file: {error:.4f}")
                print("Word-level Diff:")
                print(diff_text)
                print("-" * 80)
    
    avg_wer = total_wer / count if count > 0 else None
    print("\nOverall Average WER for {}: {:.4f}".format(label, avg_wer) if avg_wer is not None else "No valid test samples found.")
    return avg_wer, details

def print_final_report(results):
    print("\n" + "=" * 80)
    print("FINAL MODEL EVALUATION REPORT")
    print("=" * 80)
    print("\nWord Error Rate (WER) Explained:")
    print("  WER = (Substitutions + Insertions + Deletions) / Total Words in Reference")
    print("  For example, a WER of 0.10 means that 10% of the words were recognized incorrectly.")
    print("  Lower WER values indicate better performance.\n")
    
    print("MODEL RANKINGS (Lower WER is Better):")
    print("-" * 80)
    for rank, (model_path, avg_wer, _, label) in enumerate(results, start=1):
        print(f"{rank}. {label} ({os.path.basename(model_path)}): Average WER = {avg_wer:.4f}")
    print("=" * 80)

def main():
    # Paths for test data and model checkpoints
    test_audio_folder = "./data/test_audio"
    test_transcripts_csv = "./data/test_transcripts.csv"
    base_models_folder = "./models/base"
    fine_tuned_models_folder = "./models/fine_tuned"
    
    # Load test transcripts
    test_transcripts = load_test_transcripts(test_transcripts_csv)
    
    # Gather model checkpoint files from base and fine-tuned folders (excluding any medium.en models)
    model_files = []
    for folder in [base_models_folder, fine_tuned_models_folder]:
        if os.path.exists(folder):
            for file in os.listdir(folder):
                if file.endswith(".pth") and "medium.en" not in file:
                    model_files.append(os.path.join(folder, file))
    
    results = []
    for model_path in model_files:
        model_name = os.path.basename(model_path)
        # Determine architecture and label based on filename
        if model_name.startswith("base.en"):
            arch = "base.en"
            label = "Base EN"
        elif model_name.startswith("small.en") and "fine_tuned" not in model_name:
            arch = "small.en"
            label = "Small EN Base"
        elif model_name == "fine_tuned_whisper_v1.pth":
            arch = "small.en"
            label = "Original Fine-Tuned (Small EN)"
        elif "fine_tuned_whisper_small_en_v2" in model_name:
            arch = "small.en"
            label = "Fine-Tuned V2 (Small EN)"
        elif "fine_tuned_whisper_small_en_v3" in model_name:
            arch = "small.en"
            label = "Fine-Tuned V3 (Small EN)"
        elif "fine_tuned_whisper_small_en_v4" in model_name:
            arch = "small.en"
            label = "Fine-Tuned V4 (Small EN)"
        else:
            arch = "small.en"
            label = model_name
        
        avg_wer, details = evaluate_model_detailed(model_path, test_audio_folder, test_transcripts, arch, label)
        if avg_wer is not None:
            results.append((model_path, avg_wer, details, label))
            print(f"\nModel {model_name} ({label}): Average WER = {avg_wer:.4f}\n")
        else:
            print(f"\nModel {model_name} ({label}): No valid test samples found.\n")
    
    # Rank models by average WER (lower is better)
    results.sort(key=lambda x: x[1])
    
    print_final_report(results)

if __name__ == "__main__":
    main()