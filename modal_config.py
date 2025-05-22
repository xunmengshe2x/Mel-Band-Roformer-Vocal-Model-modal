from modal import Image, Secret, gpu

# Base image configuration
def get_base_image():
    return (
        Image.debian_slim()
        .apt_install([
            "libsndfile1",  # Required for soundfile package
            "ffmpeg",       # Audio processing
        ])
        .pip_install([
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
            "pyyaml",
        ])
    )

# GPU configuration - A10G based on VRAM requirements mentioned in README
GPU_CONFIG = gpu.A10G()

# Model checkpoint URL from README
MODEL_CHECKPOINT_URL = "https://huggingface.co/KimberleyJSN/melbandroformer/blob/main/MelBandRoformer.ckpt"

# Config paths
CONFIG_PATH = "configs/config_vocals_mel_band_roformer.yaml"

# Define secrets needed for the deployment
SECRETS = {
    "HUGGINGFACE_TOKEN": Secret.from_name("huggingface-token")
}

# Directory structure
DIRS = {
    "MODEL_DIR": "/model",
    "CONFIG_DIR": "/configs",
    "OUTPUT_DIR": "/outputs"
}