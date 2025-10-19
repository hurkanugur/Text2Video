import torch
from src.text_encoder import TextEncoder
from src.motion_model import MotionGenerator
from src.renderer import render_mesh_sequence
import os

# Settings
text = ["He smiles, then frowns, looking surprised."]
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Selected device: {device}")

# Initialize models
text_encoder = TextEncoder().to(device)
motion_model = MotionGenerator().to(device)

# Load trained weights
model_path = "models/motion_model_final.pt"
if not os.path.exists(model_path):
    raise FileNotFoundError(f"[ERROR] Trained model not found at {model_path}")

checkpoint = torch.load(model_path, map_location=device)
motion_model.load_state_dict(checkpoint["model_state_dict"])
motion_model.eval()
print(f"[INFO] Loaded trained model from {model_path}")

# Encode text
with torch.no_grad():
    text_emb = text_encoder.encode(text).to(device)
    print(f"[INFO] Text embedding shape: {text_emb.shape}")

# Generate motion sequence
with torch.no_grad():
    motion_seq = motion_model(text_emb)
    print(f"[INFO] Generated motion sequence shape: {motion_seq.shape}")

# Render sequence as video
flame_template = "assets/models/FLAME/flame_head_template.obj"
render_mesh_sequence(flame_template, motion_seq[0].detach().cpu().numpy())
print("[INFO] Generated video saved to outputs/video.mp4")
