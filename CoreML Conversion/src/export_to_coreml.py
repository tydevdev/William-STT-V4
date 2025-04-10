import os
import torch
import coremltools as ct
import whisper
import torch.nn.functional as F

print("Loading pre-trained Whisper model...")
# Load the "small.en" model from Whisper and set it to evaluation mode.
model = whisper.load_model("small.en")
model.eval()

# Load fine-tuned weights from a checkpoint.
checkpoint_path = os.path.join(os.path.dirname(__file__), "../models/fine_tuned/fine_tuned_whisper_small_en_v4.pth")
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError("Checkpoint not found: " + checkpoint_path)
state_dict = torch.load(checkpoint_path, map_location="cpu")
model.load_state_dict(state_dict)
print("Model loaded and fine-tuned weights applied.")

# Extract the encoder sub-module.
encoder = model.encoder
encoder.eval()
print("Encoder extracted from the full model.")

# Override the encoder forward method so that it accepts a 3D input.
def encoder_forward_override(self, mel):
    # mel is expected to have shape [batch, 80, time]
    x = self.conv1(mel)  # Conv1d expects a 3D tensor.
    x = x.transpose(1, 2)  # Change shape to [batch, time_out, d_model]
    return x

encoder.forward = encoder_forward_override.__get__(encoder, type(encoder))
print("Encoder forward method overridden.")

# Generate a dummy mel spectrogram using Whisper's preprocessing.
dummy_audio = torch.randn(16000 * 30)  # 30 seconds at 16 kHz.
dummy_audio = whisper.pad_or_trim(dummy_audio)  # Ensure exactly 30 seconds.
dummy_mel = whisper.log_mel_spectrogram(dummy_audio)  # Expected shape: [80, T]
print("Computed dummy mel shape (before batch):", dummy_mel.shape)

dummy_mel = dummy_mel.unsqueeze(0)  # Now shape becomes [1, 80, T].
print("Dummy mel shape (with batch):", dummy_mel.shape)

# Adjust the time dimension to match the encoder's expected input size.
expected_time = encoder.positional_embedding.shape[0]
current_time = dummy_mel.shape[-1]
if current_time < expected_time:
    dummy_mel = F.pad(dummy_mel, (0, expected_time - current_time))
    print(f"Padded dummy mel from {current_time} to {expected_time} time steps.")
elif current_time > expected_time:
    dummy_mel = dummy_mel[:, :, :expected_time]
    print(f"Trimmed dummy mel from {current_time} to {expected_time} time steps.")
print("Final dummy mel shape:", dummy_mel.shape)

# Trace the encoder with TorchScript.
print("Tracing the encoder with TorchScript...")
scripted_encoder = torch.jit.trace(encoder, dummy_mel)
print("Encoder successfully traced.")

# Convert the TorchScript model to Core ML.
print("Converting TorchScript model to Core ML...")
# Supply the "inputs" argument so Core ML knows the input tensor shape.
mlmodel = ct.convert(scripted_encoder,
                     source="pytorch",
                     inputs=[ct.TensorType(name="mel", shape=dummy_mel.shape)],
                     minimum_deployment_target=ct.target.iOS13)
coreml_filename = "WhisperEncoder.mlmodel"
mlmodel.save(coreml_filename)
print("Core ML model saved as", coreml_filename)

print("Done converting.")