import torch
import torchaudio
import os
from sam_audio import SAMAudio, SAMAudioProcessor

def run_voice_isolation(input_file="input.wav", output_file="outputs/voice_isolated.wav"):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Model and Processor
    # This will use the HF_TOKEN environment variable automatically
    model_id = "facebook/sam-audio-large"
    print(f"Loading model: {model_id}...")
    
    model = SAMAudio.from_pretrained(model_id).to(device).eval()
    processor = SAMAudioProcessor.from_pretrained(model_id)

    # 3. Load Audio
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found. Please place an audio file in the directory.")
        return

    audio, original_sr = torchaudio.load(input_file)
    
    # 4. Process and Separate
    print(f"Isolating 'voice' from {input_file}...")
    # Text prompting: using "voice" as the description
    inputs = processor(audios=[audio], descriptions=["voice"]).to(device)
    
    with torch.inference_mode():
        # predict_spans=True helps for foreground sounds like voices
        result = model.separate(inputs, predict_spans=True)

    # 5. Save Result
    # result.target contains the isolated sound (voice)
    # result.residual contains everything else (background)
    target_audio = result.target[0].cpu()
    
    output_sr = processor.audio_sampling_rate
    torchaudio.save(output_file, target_audio, output_sr)
    print(f"Success! Isolated voice saved to: {output_file}")

if __name__ == "__main__":
    run_voice_isolation()
