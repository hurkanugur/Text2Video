import torch
from torch.utils.data import DataLoader
from src.dataset import CelebVDataset
from src.text_encoder import TextEncoder
from src.model import Text2MotionModel
from src.train import TrainingPipeline
from src import config

def main():
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize dataset and dataloader
    dataset = CelebVDataset()
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize models
    text_encoder = TextEncoder(device=device)
    text_2_motion_model = Text2MotionModel(device=device)

    # Initialize the training pipeline
    training_pipeline = TrainingPipeline(
        text_2_motion_model=text_2_motion_model,
        text_encoder=text_encoder,
        dataloader=dataloader,
        device=device
    )

    # Train the model
    training_pipeline.train()

    # Save the model
    text_2_motion_model.save_model()

    print("-------------------------------------")

if __name__ == "__main__":
    main()