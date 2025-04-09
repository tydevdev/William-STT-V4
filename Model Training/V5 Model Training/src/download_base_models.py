import os
import ssl
import torch
import whisper

# WARNING: Disabling SSL certificate verification. Do this only in controlled settings.
ssl._create_default_https_context = ssl._create_unverified_context

def download_and_save(model_name, save_folder):
    os.makedirs(save_folder, exist_ok=True)
    print(f"Downloading {model_name} model...")
    # Download model and save to the specified folder
    model = whisper.load_model(model_name, download_root=save_folder)
    save_path = os.path.join(save_folder, f"{model_name}.pth")
    torch.save(model.state_dict(), save_path)
    print(f"{model_name} model saved to {save_path}")

if __name__ == "__main__":
    base_folder = os.path.join(os.path.dirname(__file__), "../models/base")
    download_and_save("small.en", base_folder)
    download_and_save("base.en", base_folder)
    download_and_save("medium.en", base_folder)