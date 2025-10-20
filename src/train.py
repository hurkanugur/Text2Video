import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from src import config
from src.text_encoder import TextEncoder
from src.model import Text2MotionModel
from src.flame import FlameWrapper

class TrainingPipeline:
    """
    Simple pipeline for training motion generation models from text.
    """

    def __init__(
        self,
        text_2_motion_model: Text2MotionModel,
        text_encoder: TextEncoder,
        dataloader: DataLoader,
        device: torch.device,
    ):
        self.text_2_motion_model = text_2_motion_model.to(device)
        self.text_encoder = text_encoder
        self.dataloader = dataloader
        self.device = device

        self.optimizer = optim.Adam(
            self.text_2_motion_model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )
        self.loss_fn = nn.MSELoss()
        self.flame = FlameWrapper(device=self.device)

    def train(self, num_epochs: int = config.NUM_EPOCHS):
        """Run the full training loop."""
        self.text_2_motion_model.train()

        for epoch in range(num_epochs):
            print(f"[INFO] Starting epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0.0

            for batch in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                texts = batch["text"]

                # Step 1: encode text
                with torch.no_grad():
                    text_emb = self.text_encoder.encode(texts).to(self.device)

                # Step 2: TODO extract FLAME params (fake for now)
                motions_gt = []
                for frames in batch["frames"]:
                    frame_params = [self.flame.estimate_params_from_frame(f) for f in frames]
                    # Use expression parameters as supervision
                    frame_motion = [p["expression"].flatten() for p in frame_params]
                    motions_gt.append(torch.stack(frame_motion))
                motions_gt = torch.stack(motions_gt).to(self.device)

                # Step 3: Forward pass
                motions_pred = self.text_2_motion_model(text_emb)
                loss = self.loss_fn(motions_pred, motions_gt)

                # Step 4: Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(self.dataloader)
            print(f"[INFO] Epoch {epoch+1} done, average loss={avg_loss:.4f}")
