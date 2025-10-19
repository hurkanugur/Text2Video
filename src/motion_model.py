import torch
import torch.nn as nn

class MotionGenerator(nn.Module):
    def __init__(self, text_dim=512, motion_dim=50, seq_len=32):
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len * motion_dim)
        )
        self.motion_dim = motion_dim

    def forward(self, text_emb):
        out = self.net(text_emb)
        return out.view(-1, self.seq_len, self.motion_dim)
