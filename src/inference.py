import os
import numpy as np
from PIL import Image, ImageDraw
import imageio
from src import config
import torch
from src.text_encoder import TextEncoder
from src.model import Text2MotionModel


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
    Text-to-motion pipeline: encodes text, generates motion, and renders video (3D if available, else 2D).
    """

    # --------------------- Initialization ---------------------

    def __init__(
            self, 
            text_2_motion_model: Text2MotionModel,
            text_encoder: TextEncoder,
            device: torch.device
        ):
        """Initialize the renderer and load the 3D mesh template if available."""
        self.text_2_motion_model = text_2_motion_model
        self.text_encoder = text_encoder
        self.device = device

        self.has_3d = HAS_3D
        self.mesh_template = self._load_mesh_template()

    # --------------------- Public Methods ---------------------

    def run_inference(
        self,
        user_prompt: list[str],
    ):
        """
        Run full pipeline: encode text, generate motion, render video.
        
        Args:
            user_prompt: List of text prompts describing the motion.
        """
                
        # Encode text
        with torch.no_grad():
            text_emb = self.text_encoder.encode(user_prompt).to(self.device)
            print(f"[INFO] Text embedding shape: {text_emb.shape}")

        # Generate motion sequence
        with torch.no_grad():
            motion_seq = self.text_2_motion_model(text_emb)
            print(f"[INFO] Generated motion sequence shape: {motion_seq.shape}")

        # Render sequence as video
        self._render_motion_sequence(params_seq=motion_seq[0].detach().cpu().numpy())

    # --------------------- Private Methods ---------------------

    def _render_motion_sequence(self, params_seq):
        """Render motion parameters into a video (3D or 2D depending on system capabilities)."""
        try:
            if self.has_3d and self.mesh_template is not None:
                frames = self._render_3d_sequence(params_seq)
            else:
                frames = self._render_2d_fallback(params_seq)
        except Exception as e:
            print(f"[WARNING] 3D rendering error ({e}), switching to 2D fallback.")
            frames = self._render_2d_fallback(params_seq)

        self._save_video(frames)


    def _load_mesh_template(self):
        """Load the base 3D mesh template from file."""
        if not self.has_3d or not os.path.exists(config.FLAME_HEAD_TEMPLATE_PATH):
            return None

        try:
            mesh = trimesh.load(config.FLAME_HEAD_TEMPLATE_PATH)
            if mesh.is_empty:
                raise ValueError("Loaded mesh is empty.")
            return mesh
        except Exception as e:
            print(f"[WARNING] Failed to load mesh: {e}")
            return None


    def _render_3d_sequence(self, params_seq):
        """Render a motion sequence as 3D frames using pyrender."""
        print("[INFO] Rendering in 3D mode using pyrender...")
        frames = []

        renderer = pyrender.OffscreenRenderer(
            viewport_width=config.IMG_SIZE,
            viewport_height=config.IMG_SIZE
        )

        for params in params_seq:
            mesh_copy = self.mesh_template.copy()
            mesh_copy.apply_translation(params[:3])  # Simple translation from motion params

            scene = pyrender.Scene()
            mesh = pyrender.Mesh.from_trimesh(mesh_copy)
            scene.add(mesh)

            # Simple camera and lighting setup
            camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
            light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
            scene.add(camera, pose=np.eye(4))
            scene.add(light, pose=np.eye(4))

            color, _ = renderer.render(scene)
            frames.append(color)

        renderer.delete()
        return frames


    def _render_2d_fallback(self, params_seq):
        """Render motion sequence as 2D dot plots if 3D is unavailable."""
        print("[INFO] Rendering in 2D fallback mode...")
        frames = []

        for frame_params in params_seq:
            img = Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE), color=(0, 0, 0))
            draw = ImageDraw.Draw(img)

            half = len(frame_params) // 2
            x, y = frame_params[:half], frame_params[half:half * 2]

            # Normalize to image size
            x = (x - x.min()) / (x.max() - x.min() + 1e-6) * (config.IMG_SIZE - 1)
            y = (y - y.min()) / (y.max() - y.min() + 1e-6) * (config.IMG_SIZE - 1)

            for xi, yi in zip(x, y):
                draw.ellipse((xi - 2, yi - 2, xi + 2, yi + 2), fill=(255, 255, 255))

            frames.append(np.array(img))

        return frames


    def _save_video(self, frames):
        """Save the rendered frames as a video file."""
        os.makedirs(os.path.dirname(config.OUTPUT_VIDEO_PATH), exist_ok=True)
        imageio.mimwrite(config.OUTPUT_VIDEO_PATH, frames, fps=config.FPS)
        print(f"[INFO] Saved rendered video to: {config.OUTPUT_VIDEO_PATH}")

