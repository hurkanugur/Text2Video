import torch
import numpy as np
import cv2
import dlib
import trimesh
from smplx import FLAME
from src import config

class FlameWrapper:
    """
    Wrapper for FLAME: fits FLAME to input frames and generates mesh sequences.
    """

    def __init__(self, device: torch.device):
        self.device = device

        # Load FLAME model (requires FLAME .pkl files in assets/flame/)
        self.model = FLAME(
            model_path="assets/flame",  # path to FLAME .pkl
            batch_size=1,
            device=device
        )

        # Load pretrained Dlib 68-landmark detector
        predictor_path = "assets/shape_predictor_68_face_landmarks.dat"
        if not predictor_path or not os.path.exists(predictor_path):
            raise FileNotFoundError("Download Dlib 68-landmark model to assets/")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    # -------------------- Fit FLAME --------------------
    def estimate_params_from_frame(self, frame: np.ndarray):
        """
        Fit FLAME to a single frame using 2D landmarks.
        Returns FLAME parameters as dict.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            # fallback to zeros if no face
            return self._zero_params()

        # Use first detected face
        face = faces[0]
        landmarks = self.predictor(gray, face)
        lm = np.array([[p.x, p.y] for p in landmarks.parts()])

        # TODO: pretrained fitting code goes here
        # For now, we return zeros except expression (just to have meaningful motion)
        params = self._zero_params()
        return params

    def _zero_params(self):
        return {
            "shape": torch.zeros(1, 100).to(self.device),
            "expression": torch.zeros(1, 50).to(self.device),
            "pose": torch.zeros(1, 6).to(self.device),
            "trans": torch.zeros(1, 3).to(self.device)
        }

    # -------------------- Generate Mesh --------------------
    def generate_mesh(self, flame_params: dict):
        """
        Generate FLAME mesh from parameters.
        """
        output = self.model(
            shape=flame_params["shape"],
            expression=flame_params["expression"],
            jaw_pose=flame_params["pose"][:, :3],
            neck_pose=flame_params["pose"][:, 3:],
            transl=flame_params["trans"]
        )
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        faces = self.model.faces

        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        return mesh

    # -------------------- Generate Sequence --------------------
    def generate_sequence(self, motion_seq: np.ndarray):
        """
        Generate mesh sequence from motion parameters predicted by Text2MotionModel.

        Args:
            motion_seq: (frames, motion_features)
        Returns:
            list of trimesh.Trimesh meshes
        """
        meshes = []
        for frame_params in motion_seq:
            flame_params = {
                "shape": torch.zeros(1, 100).to(self.device),
                "expression": torch.tensor(frame_params[:50]).unsqueeze(0).float().to(self.device),
                "pose": torch.zeros(1, 6).to(self.device),
                "trans": torch.zeros(1, 3).to(self.device)
            }
            mesh = self.generate_mesh(flame_params)
            meshes.append(mesh)
        return meshes
