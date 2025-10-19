import os
import torch
import torch.nn as nn
from src import config

class Text2MotionModel(nn.Module):
    """Simple feedforward model to generate per-frame motion from text embeddings."""
    
    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device

        # Neural network
        self.net = nn.Sequential(
            nn.Linear(config.TEXT_EMBEDDING_DIM, 256),
            nn.ReLU(),
            nn.Linear(256, config.FRAMES_PER_VIDEO * config.MOTION_FEATURES_PER_FRAME)
        )

        # Move model to device
        self.to(self.device)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize linear layers with Xavier initialization."""
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, text_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: text embeddings -> motion vectors per frame.

        Args:
            text_emb: Tensor of shape (batch_size, config.TEXT_EMBEDDING_DIM)
        Returns:
            Tensor of shape (batch_size, config.FRAMES_PER_VIDEO, config.MOTION_FEATURES_PER_FRAME)
        """
        out = self.net(text_emb)
        return out.view(-1, config.FRAMES_PER_VIDEO, config.MOTION_FEATURES_PER_FRAME)

    # ---------------------- Save / Load ----------------------

    def save_model(self):
        """Saves the model."""
        os.makedirs(os.path.dirname(config.MOTION_GENERATION_MODEL_PATH), exist_ok=True)
        torch.save(self.state_dict(), config.MOTION_GENERATION_MODEL_PATH)
        print(f"[INFO] Model saved to {config.MOTION_GENERATION_MODEL_PATH}")

    def load_model(self):
        """Loads the model."""
        if not os.path.exists(config.MOTION_GENERATION_MODEL_PATH):
            raise FileNotFoundError(f"No model found at {config.MOTION_GENERATION_MODEL_PATH}")
        
        self.load_state_dict(torch.load(config.MOTION_GENERATION_MODEL_PATH, weights_only=True, map_location=self.device))
        self.to(self.device)
        print(f"[INFO] Model loaded from {config.MOTION_GENERATION_MODEL_PATH}")
