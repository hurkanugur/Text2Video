import torch
from src.text_encoder import TextEncoder
from src.model import Text2MotionModel
from src.inference import InferencePipeline

def main():
    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model & encoder
    text_encoder = TextEncoder(device=device)
    text_2_motion_model = Text2MotionModel(device=device)
    text_2_motion_model.load_model()

    # Initialize the inference pipeline
    inference_pipeline = InferencePipeline(
        text_2_motion_model=text_2_motion_model,
        text_encoder=text_encoder,
        device=device
    )

    # Inference
    user_prompt = ["He smiles, then frowns, looking surprised."]
    inference_pipeline.run_inference(user_prompt=user_prompt)

    print("-------------------------------------")

if __name__ == "__main__":
    main()