import os
import yaml
import torch
from modal import Stub, method, web_endpoint
from modal_config import (
    get_base_image, GPU_CONFIG, MODEL_CHECKPOINT_URL,
    CONFIG_PATH, SECRETS, DIRS
)
from pathlib import Path
import requests
import soundfile as sf
from ml_collections import ConfigDict
from utils import get_model_from_config, demix_track

# Create stub
stub = Stub("mel-band-roformer")

@stub.cls(
    image=get_base_image(),
    gpu=GPU_CONFIG,
    secrets=SECRETS,
    mounts=[]
)
class MelBandRoformerDeployment:
    def __enter__(self):
        """Initialize the model and load weights""
        # Create necessary directories
        for dir_path in DIRS.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

        # Download model checkpoint
        self.model_path = f"{DIRS['MODEL_DIR']}/model.ckpt"
        if not os.path.exists(self.model_path):
            headers = {"Authorization": f"Bearer {os.environ['HUGGINGFACE_TOKEN']}"}
            response = requests.get(MODEL_CHECKPOINT_URL, headers=headers)
            if response.status_code == 200:
                with open(self.model_path, 'wb') as f:
                    f.write(response.content)
            else:
                raise Exception(f"Failed to download model: {response.status_code}")

        # Load configuration
        with open(CONFIG_PATH) as f:
            self.config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

        # Initialize model
        self.model = get_model_from_config('mel_band_roformer', self.config)

        # Load weights
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=torch.device('cuda'))
        )
        self.model = self.model.cuda()
        self.model.eval()

    @method()
    def process_audio(self, audio_data, sr=44100):
        """Process audio data and separate vocals""
        try:
            # Convert input audio to tensor
            mixture = torch.tensor(audio_data.T, dtype=torch.float32)

            # Process through model
            with torch.no_grad():
                mixture = mixture.cuda()
                res, _ = demix_track(self.config, self.model, mixture, 'cuda')

            # Extract vocals and instrumental
            vocals = res['vocals'].cpu().numpy().T
            instrumental = audio_data - vocals

            return {
                'vocals': vocals.tolist(),
                'instrumental': instrumental.tolist()
            }
        except Exception as e:
            raise Exception(f"Error processing audio: {str(e)}")

    @web_endpoint()
    def separate_vocals(self, audio_file):
        """Web endpoint for vocal separation""
        try:
            # Read audio file
            audio_data, sr = sf.read(audio_file)

            # Process audio
            result = self.process_audio(audio_data, sr)

            return result
        except Exception as e:
            return {"error": str(e)}, 500

# For local testing
if __name__ == "__main__":
    stub.deploy("mel-band-roformer")