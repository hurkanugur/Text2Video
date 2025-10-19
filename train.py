import torch
from torch.utils.data import DataLoader
from src.dataset import CelebVDataset
from src.text_encoder import TextEncoder
from src.motion_model import MotionGenerator
from torch import nn, optim
from tqdm import tqdm

# Initialize dataset and dataloader
dataset = CelebVDataset(seq_len=32)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize models
text_encoder = TextEncoder().to(device)
motion_model = MotionGenerator().to(device)

# Optimizer and loss
optimizer = optim.Adam(motion_model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

num_epochs = 2

for epoch in range(num_epochs):
    print(f"[INFO] Starting epoch {epoch+1}/{num_epochs}")
    epoch_loss = 0.0

    # tqdm progress bar over batches
    for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        texts = batch["text"]

        # Step 1: encode text
        text_emb = text_encoder.encode(texts).to(device)
        # Step 2: extract FLAME params (fake for now)
        motions_gt = torch.stack([torch.randn(32, 50) for _ in texts]).to(device)

        # Step 3: predict motion
        motions_pred = motion_model(text_emb)

        # Compute loss
        loss = criterion(motions_pred, motions_gt)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss for logging
        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    print(f"[INFO] Epoch {epoch+1} done, average loss={avg_loss:.4f}")

# Save final model
final_model_path = "models/motion_model_final.pt"
torch.save({
    "model_state_dict": motion_model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss": avg_loss,
    "num_epochs": num_epochs
}, final_model_path)
print(f"[INFO] Training finished. Saved final model to: {final_model_path}")
