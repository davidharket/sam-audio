import torch
import torchaudio
import os
from sam_audio.models import SAMAudio
from sam_audio.processor import SAMAudioProcessor

def run_voice_isolation(input_file="input.wav", output_file="outputs/voice_isolated.wav"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # The model ID on Hugging Face
    model_id = "facebook/sam-audio-large"
    
    # Check for HF Token (Set this in RunPod Env Variables)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in environment variables.")

    print(f"Loading model: {model_id}...")
    # Load model and processor
    model = SAMAudio.from_pretrained(model_id, token=hf_token).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(model_id, token=hf_token)

    # Load your local audio file
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please upload a file named 'input.wav'.")
        return

    audio, original_sr = torchaudio.load(input_file)
    
    # If audio is stereo, convert to mono as many SAM models prefer mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    print(f"Isolating 'voice'...")
    # Prepare inputs for the text-guided prompt "voice"
    inputs = processor(audios=[audio], descriptions=["voice"]).to(device)
    
    with torch.inference_mode():
        # SAM-Audio separation call
        result = model.separate(inputs)

    # Extract the 'target' (the "voice")
    # Result.target is typically a list of tensors
    voice_audio = result.target[0].cpu()
    
    # Save using the processor's native sampling rate (usually 44100 or 48000)
    torchaudio.save(output_file, voice_audio, processor.audio_sampling_rate)
    print(f"Success! Saved to: {output_file}")

if __name__ == "__main__":
    run_voice_isolation()
