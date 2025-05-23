import os
import modal
import tempfile
import base64
from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
import soundfile as sf
import numpy as np
import torch
import yaml
from ml_collections import ConfigDict
import urllib.request
import shutil

# Define a custom image with all dependencies
image = modal.Image.debian_slim().pip_install(
    "torch>=2.0.1",
    "numpy",
    "pandas",
    "soundfile",
    "ml_collections",
    "tqdm",
    "segmentation_models_pytorch==0.3.3",
    "timm==0.9.2",
    "omegaconf==2.2.3",
    "beartype==0.14.1",
    "rotary_embedding_torch==0.3.5",
    "einops==0.6.1",
    "librosa",
    "fastapi[standard]",
    "python-multipart",
    "pydantic>=2.0.0",
    "typing-extensions"
)

image = image.apt_install("ffmpeg", "wget", "git")


# Create a Modal volume to store model files
volume = modal.Volume.from_name("mel-band-roformer-model", create_if_missing=True)

# Create a Modal app
app = modal.App("mel-band-roformer-vocal-model", image=image)

MODEL_PATH = "/model/MelBandRoformer.ckpt"
CONFIG_PATH = "/model/config_vocals_mel_band_roformer.yaml"
REPO_PATH = "/model/repo"

@app.function(
    gpu="T4",  # You can change this to "A10G", "A100", etc. based on your needs
    timeout=600,  # 10-minute timeout
    volumes={"/model": volume}
)
def download_model_and_repo():
    """Download the model checkpoint, config, and repository to the volume."""
    import os
    import subprocess
    
    # Create model directory if it doesn't exist
    os.makedirs("/model", exist_ok=True)
    
    # Check if model is already downloaded
    if os.path.exists(MODEL_PATH) and os.path.exists(CONFIG_PATH) and os.path.exists(REPO_PATH):
        print("Model, config, and repository already downloaded.")
        return True
    
    # Download model checkpoint from Hugging Face
    if not os.path.exists(MODEL_PATH):
        print("Downloading model checkpoint...")
        subprocess.run(
            "wget https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt -O " + MODEL_PATH,
            shell=True,
            check=True
        )
    
    # Clone the repository
    if not os.path.exists(REPO_PATH):
        print("Cloning repository...")
        subprocess.run(
            "git clone https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model.git " + REPO_PATH,
            shell=True,
            check=True
        )
    
    # Copy config from repo to model directory if not exists
    if not os.path.exists(CONFIG_PATH):
        print("Copying config file...")
        shutil.copy(
            os.path.join(REPO_PATH, "configs/config_vocals_mel_band_roformer.yaml"),
            CONFIG_PATH
        )
    
    print("Model, config, and repository downloaded successfully.")
    return True

@app.function(
    gpu="T4",  # You can change this to "A10G", "A100", etc. based on your needs
    timeout=600,  # 10-minute timeout
    volumes={"/model": volume}
)
def process_audio(audio_data, sample_rate):
    """Process audio data using the Mel-Band-Roformer model."""
    import sys
    import torch
    import torch.nn as nn
    import numpy as np
    import yaml
    from ml_collections import ConfigDict
    import os
    
    # Add the repository to the Python path
    sys.path.append(REPO_PATH)
    
    # Import necessary functions from the repository
    from utils import demix_track, get_model_from_config
    
    # Load config
    with open(CONFIG_PATH) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
    
    # Initialize model
    model = get_model_from_config('mel_band_roformer', config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    
    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        model = model.to(device)
    else:
        device = torch.device('cpu')
        print('CUDA is not available. Running on CPU. This will be slow...')
        model = model.to(device)
    
    # Prepare audio data
    model.eval()
    
    # Check if audio is mono or stereo
    original_mono = False
    if len(audio_data.shape) == 1:
        original_mono = True
        audio_data = np.stack([audio_data, audio_data], axis=-1)
    
    # Convert to tensor
    mixture = torch.tensor(audio_data.T, dtype=torch.float32)
    
    # Process audio
    res, _ = demix_track(config, model, mixture, device, None)
    
    # Extract vocals and create instrumental
    instruments = config.training.instruments
    if config.training.target_instrument is not None:
        instruments = [config.training.target_instrument]
    
    vocals_output = res[instruments[0]].T
    if original_mono:
        vocals_output = vocals_output[:, 0]
    
    # Create instrumental by subtracting vocals from original
    instrumental = audio_data - vocals_output
    
    return vocals_output, instrumental, sample_rate

@app.function(
    gpu="T4",  # You can change this to "A10G", "A100", etc. based on your needs
    timeout=600,  # 10-minute timeout
    volumes={"/model": volume}
)
@modal.fastapi_endpoint(method="POST")
async def process_audio_file(request: Request):
    """Web endpoint for processing audio files with the Mel-Band-Roformer model."""
    import tempfile
    import base64
    import soundfile as sf
    
    # Parse the request body
    data = await request.json()
    audio_base64 = data.get("audio_base64")
    
    if not audio_base64:
        return {"error": "Missing required parameter: audio_base64"}
    
    # Ensure model and repository are downloaded
    download_model_and_repo.remote()
    
    # Create a temporary file for the audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(base64.b64decode(audio_base64))
    
    try:
        # Read the audio file
        audio_data, sample_rate = sf.read(temp_file_path)
        
        # Process the audio
        vocals, instrumental, sr = process_audio.remote(audio_data, sample_rate)
        
        # Save the processed audio to temporary files
        vocals_path = temp_file_path + "_vocals.wav"
        instrumental_path = temp_file_path + "_instrumental.wav"
        
        sf.write(vocals_path, vocals, sr, subtype='FLOAT')
        sf.write(instrumental_path, instrumental, sr, subtype='FLOAT')
        
        # Read the processed files and encode as base64
        with open(vocals_path, "rb") as f:
            vocals_data = f.read()
            vocals_base64 = base64.b64encode(vocals_data).decode("utf-8")
        
        with open(instrumental_path, "rb") as f:
            instrumental_data = f.read()
            instrumental_base64 = base64.b64encode(instrumental_data).decode("utf-8")
        
        return {
            "vocals_base64": vocals_base64,
            "instrumental_base64": instrumental_base64
        }
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(vocals_path):
            os.remove(vocals_path)
        if os.path.exists(instrumental_path):
            os.remove(instrumental_path)

@app.function(
    gpu="T4",  # You can change this to "A10G", "A100", etc. based on your needs
    timeout=600,  # 10-minute timeout
    volumes={"/model": volume}
)
@modal.fastapi_endpoint(method="POST")
async def process_audio_url(request: Request):
    """Web endpoint for processing audio files from a URL with the Mel-Band-Roformer model."""
    import tempfile
    import base64
    import soundfile as sf
    import urllib.request
    
    # Parse the request body
    data = await request.json()
    audio_url = data.get("audio_url")
    
    if not audio_url:
        return {"error": "Missing required parameter: audio_url"}
    
    # Ensure model and repository are downloaded
    download_model_and_repo.remote()
    
    # Create a temporary file for the audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file_path = temp_file.name
        
    try:
        # Download the audio file
        urllib.request.urlretrieve(audio_url, temp_file_path)
        
        # Read the audio file
        audio_data, sample_rate = sf.read(temp_file_path)
        
        # Process the audio
        vocals, instrumental, sr = process_audio.remote(audio_data, sample_rate)
        
        # Save the processed audio to temporary files
        vocals_path = temp_file_path + "_vocals.wav"
        instrumental_path = temp_file_path + "_instrumental.wav"
        
        sf.write(vocals_path, vocals, sr, subtype='FLOAT')
        sf.write(instrumental_path, instrumental, sr, subtype='FLOAT')
        
        # Read the processed files and encode as base64
        with open(vocals_path, "rb") as f:
            vocals_data = f.read()
            vocals_base64 = base64.b64encode(vocals_data).decode("utf-8")
        
        with open(instrumental_path, "rb") as f:
            instrumental_data = f.read()
            instrumental_base64 = base64.b64encode(instrumental_data).decode("utf-8")
        
        return {
            "vocals_base64": vocals_base64,
            "instrumental_base64": instrumental_base64
        }
    
    finally:
        # Clean up temporary files
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        if os.path.exists(vocals_path):
            os.remove(vocals_path)
        if os.path.exists(instrumental_path):
            os.remove(instrumental_path)

@app.function()
@modal.fastapi_endpoint(method="GET")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "service": "mel-band-roformer-vocal-model"}

@app.function()
@modal.fastapi_endpoint(method="GET")
async def info():
    """Information about the model and service."""
    return {
        "model": "Mel-Band-Roformer Vocal Model",
        "description": "A model for separating vocals from music tracks",
        "repository": "https://github.com/KimberleyJensen/Mel-Band-Roformer-Vocal-Model",
        "endpoints": {
            "process_audio_file": "POST /process_audio_file - Process audio from base64 encoded data",
            "process_audio_url": "POST /process_audio_url - Process audio from a URL",
            "health": "GET /health - Health check endpoint",
            "info": "GET /info - Information about the service"
        }
    }

# For local testing
if __name__ == "__main__":
    # This will be executed when running the file directly, not when deployed to Modal
    app.serve()
