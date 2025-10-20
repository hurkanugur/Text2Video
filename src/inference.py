import os
import numpy as np
from PIL import Image, ImageDraw
import imageio
from src import config
import torch
from src.text_encoder import TextEncoder
from src.model import Text2MotionModel
from src.flame import FlameWrapper

# Optional 3D rendering support
try:
    import trimesh
    import pyrender
    HAS_3D = True
except ImportError:
    HAS_3D = False
    print("[WARNING] 3D rendering dependencies not found. Falling back to 2D visualization.")


class InferencePipeline:
    """
    Text-to-motion pipeline: encodes text, generates motion, fits FLAME, renders video.
    """

    def __init__(self, text_2_motion_model: Text2MotionModel, text_encoder: TextEncoder, device: torch.device):
        self.text_2_motion_model = text_2_motion_model
        self.text_encoder = text_encoder
        self.device = device

        self.has_3d = HAS_3D

        # FLAME wrapper
        self.flame = FlameWrapper(device=self.device)

    # --------------------- Public Method ---------------------
    def run_inference(self, user_prompt: list[str]):
        """
        Run full pipeline: encode text → generate motion → FLAME → render video.
        """
        # Encode text
        with torch.no_grad():
            text_emb = self.text_encoder.encode(user_prompt).to(self.device)
            print(f"[INFO] Text embedding shape: {text_emb.shape}")

        # Generate motion sequence
        with torch.no_grad():
            motion_seq = self.text_2_motion_model(text_emb)
            print(f"[INFO] Generated motion sequence shape: {motion_seq.shape}")

        # Generate FLAME meshes from motion sequence
        meshes = self.flame.generate_sequence(motion_seq[0].detach().cpu().numpy())

        # Render frames
        frames = self._render_mesh_sequence(meshes)
        self._save_video(frames)

    # --------------------- Private Methods ---------------------
    def _render_mesh_sequence(self, meshes):
        """
        Render a list of FLAME meshes into images.
        Falls back to 2D if 3D rendering fails.
        """
        frames = []
        if self.has_3d:
            try:
                renderer = pyrender.OffscreenRenderer(config.IMG_SIZE, config.IMG_SIZE)
                for mesh in meshes:
                    scene = pyrender.Scene()
                    pyr_mesh = pyrender.Mesh.from_trimesh(mesh)
                    scene.add(pyr_mesh)

                    # Camera setup
                    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
                    scene.add(camera, pose=np.eye(4))

                    # Light
                    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
                    scene.add(light, pose=np.eye(4))

                    color, _ = renderer.render(scene)
                    frames.append(color)
                renderer.delete()
                return frames
            except Exception as e:
                print(f"[WARNING] 3D rendering failed ({e}), falling back to 2D.")

        # 2D fallback: render motion params as dot plot
        print("[INFO] Rendering in 2D fallback mode...")
        for mesh in meshes:
            img = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)
            verts = mesh.vertices
            if verts.shape[0] > 0:
                x = verts[:, 0]
                y = verts[:, 1]

                # Normalize to image
                x = (x - x.min()) / (x.max() - x.min() + 1e-6) * (config.IMG_SIZE - 1)
                y = (y - y.min()) / (y.max() - y.min() + 1e-6) * (config.IMG_SIZE - 1)

                for xi, yi in zip(x, y):
                    draw.ellipse((xi - 1, yi - 1, xi + 1, yi + 1), fill=(255, 255, 255))
            frames.append(np.array(img))
        return frames

    def _save_video(self, frames):
        """Save rendered frames as a video file."""
        os.makedirs(os.path.dirname(config.OUTPUT_VIDEO_PATH), exist_ok=True)
        imageio.mimwrite(config.OUTPUT_VIDEO_PATH, frames, fps=config.FPS)
        print(f"[INFO] Saved video to {config.OUTPUT_VIDEO_PATH}")
